import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';
import { Vec3 } from 'vec3';

function forceKillPid(pid: number | undefined): void {
  if (!pid || pid <= 0) return;
  try {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/PID', String(pid), '/T', '/F'], { detached: true, stdio: 'ignore' }).unref();
    } else {
      process.kill(pid, 'SIGKILL');
    }
  } catch {}
}

function trimJsonlFileIfTooLarge(filePath: string, maxBytes: number, keepLines: number): void {
  try {
    if (!fs.existsSync(filePath)) return;
    const st = fs.statSync(filePath);
    if (st.size <= maxBytes) return;
    const raw = fs.readFileSync(filePath, 'utf-8');
    const lines = raw.split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length <= keepLines) return;
    const tail = lines.slice(-keepLines).join('\n') + '\n';
    const tmp = `${filePath}.trim.tmp`;
    fs.writeFileSync(tmp, tail, 'utf-8');
    fs.renameSync(tmp, filePath);
    console.log(
      `[TrainLoop] Rotated ${path.basename(filePath)} (${(st.size / (1024 * 1024)).toFixed(1)} MB -> tail ${keepLines} lines).`
    );
  } catch (err: any) {
    console.warn(`[TrainLoop] Warning rotating ${path.basename(filePath)}: ${err?.message || err}`);
  }
}

async function runPythonScriptAsync(
  pythonExe: string,
  args: string[],
  timeoutMs: number
): Promise<string> {
  return await new Promise<string>((resolve, reject) => {
    const proc = spawn(pythonExe, args, { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] });
    const maxTailChars = 65536;
    let stdoutTail = '';
    let stderrTail = '';
    let settled = false;

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      forceKillPid(proc.pid);
      reject(new Error(`Python script timed out after ${timeoutMs}ms: ${path.basename(args[0] || 'script')}`));
    }, timeoutMs);

    proc.stdout?.on('data', (chunk: Buffer) => {
      stdoutTail += chunk.toString('utf-8');
      if (stdoutTail.length > maxTailChars) stdoutTail = stdoutTail.slice(-maxTailChars);
    });
    proc.stderr?.on('data', (chunk: Buffer) => {
      stderrTail += chunk.toString('utf-8');
      if (stderrTail.length > 16384) stderrTail = stderrTail.slice(-16384);
    });
    proc.on('error', (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(err);
    });
    proc.on('close', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (code === 0) {
        resolve(stdoutTail);
      } else {
        reject(new Error(`Python script exited ${code}: ${stderrTail.slice(-500) || stdoutTail.slice(-300)}`));
      }
    });
  });
}
import {
  MINEBENCH_TASKS,
  MULTIPLAYER_TASKS,
  MinebenchDomain,
  MinebenchTaskSpec,
} from './benchmarkSpec';
import { SparringPartnerBot } from './sparringPartner';
import { evaluateMinebenchEpisode, MinebenchEvaluationSnapshot } from './evaluators';
import { HeadlessMinecraftServerManager } from './headlessServer';
import { System1BotController, HOSTILE_MOBS, getEntityPriority, isBossOrCrystalEntity } from '../src/system1/botController';
import { DynamicMCPToolRegistry } from '../src/system2/mcp/toolRegistry';
import { NeuroCraftMCPServer } from '../src/system2/mcp/mcpServer';
import { registerBaseMCPTools } from '../src/system2/skills/baseSkills';
import { ExecutedStepRecord, UnifiedLLMClient } from '../src/system2/llmClient';
import { HierarchicalGoalPlanner } from '../src/system2/goalPlanner';
import { SubtaskStatus } from '../src/types';
import { generateTrainingVisuals, IterationRecord } from './generate_training_graphs';
import { runVanillaDragonSurvivalRun } from './dragon_survival_run';

export interface TrainingState {
  sessionCount: number;
  currentIteration: number;
  totalEpisodesPlayed: number;
  bestPassRate: number;
  bestMeanScore: number;
  bestIteration: number;
  latestMeanScore: number;
  latestPassRate: number;
  latestReward: number;
  lastSavedAt: string;
  latestCheckpoint: string;
  bestCheckpoint: string;
  history?: IterationRecord[];
}

export interface WorkerContext {
  id: number;
  botUsername: string;
  deltaX: number;
  system1: System1BotController;
  sparringPartner: SparringPartnerBot;
  registry: DynamicMCPToolRegistry;
  mcpServer: NeuroCraftMCPServer;
  llm: UnifiedLLMClient;
}

export class AsyncMutex {
  private queue: Array<() => void> = [];
  private locked = false;

  public async acquire(): Promise<() => void> {
    if (this.locked) {
      await new Promise<void>((resolve) => this.queue.push(resolve));
    }
    this.locked = true;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      if (this.queue.length > 0) {
        const next = this.queue.shift()!;
        next();
      } else {
        this.locked = false;
      }
    };
  }

  public async runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const release = await this.acquire();
    try {
      return await fn();
    } finally {
      release();
    }
  }
}

export class ConcurrentTaskQueue<T> {
  private items: T[];
  private currentIndex = 0;

  constructor(items: T[]) {
    this.items = [...items];
  }

  public dequeue(): { item: T; index: number } | null {
    if (this.currentIndex >= this.items.length) {
      return null;
    }
    const index = this.currentIndex++;
    return { item: this.items[index], index };
  }

  public get remaining(): number {
    return Math.max(0, this.items.length - this.currentIndex);
  }

  public get total(): number {
    return this.items.length;
  }
}

export async function runTaskQueuePool<T, R>(
  items: T[],
  concurrency: number,
  workerFn: (item: T, itemIndex: number, workerId: number) => Promise<R>
): Promise<R[]> {
  const queue = new ConcurrentTaskQueue(items);
  const results: R[] = [];
  const numWorkers = Math.max(1, Math.min(items.length || 1, concurrency));

  const workers = Array.from({ length: numWorkers }, async (_, workerId) => {
    while (true) {
      const next = queue.dequeue();
      if (!next) break;
      const res = await workerFn(next.item, next.index, workerId);
      results.push(res);
    }
  });

  await Promise.all(workers);
  return results;
}

export function parseConcurrencyArg(args: string[] = process.argv.slice(2), envVal?: string): number {
  let val: number | undefined;
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const lower = arg.toLowerCase();
    if (lower === '--concurrency' || lower === '--bots') {
      if (i + 1 < args.length && (!args[i + 1].startsWith('-') || /^-\d+/.test(args[i + 1]))) {
        const parsed = parseInt(args[i + 1], 10);
        if (!isNaN(parsed)) val = parsed;
      }
    } else if (lower.startsWith('--concurrency=')) {
      const parsed = parseInt(arg.split('=')[1], 10);
      if (!isNaN(parsed)) val = parsed;
    } else if (lower.startsWith('--bots=')) {
      const parsed = parseInt(arg.split('=')[1], 10);
      if (!isNaN(parsed)) val = parsed;
    }
  }

  if (val === undefined && envVal !== undefined && envVal.trim() !== '') {
    const parsed = parseInt(envVal.trim(), 10);
    if (!isNaN(parsed)) val = parsed;
  }

  if (val === undefined) {
    val = 3;
  }

  return Math.max(1, Math.min(5, Math.floor(val)));
}

export function getWorkerSpatialOffset(workerIndex: number): number {
  return Math.max(0, workerIndex) * 4000;
}

export function getWorkerBotName(workerIndex: number, concurrency: number): string {
  if (concurrency <= 1) return 'NeuroAgent';
  return `NeuroAgent_${workerIndex + 1}`;
}

export function shiftCommandCoordinates(cmd: string, deltaX: number): string {
  if (deltaX === 0) return cmd;

  let result = cmd.trim();

  // Shift coordinate selectors like x=100 anywhere in command
  if (/\bx=-?\d+/i.test(result)) {
    result = result.replace(/\bx=(-?\d+(?:\.\d+)?)\b/gi, (_, xVal) => {
      return `x=${parseFloat(xVal) + deltaX}`;
    });
  }

  // 1. /fill <x1> <y1> <z1> <x2> <y2> <z2> <block>
  const fillRegex = /^(\s*\/?fill\s+)(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(.*)$/i;
  const fillMatch = result.match(fillRegex);
  if (fillMatch) {
    const prefix = fillMatch[1];
    const x1 = parseFloat(fillMatch[2]) + deltaX;
    const y1 = fillMatch[3];
    const z1 = fillMatch[4];
    const x2 = parseFloat(fillMatch[5]) + deltaX;
    const y2 = fillMatch[6];
    const z2 = fillMatch[7];
    const rest = fillMatch[8];
    return `${prefix}${x1} ${y1} ${z1} ${x2} ${y2} ${z2}${rest}`;
  }

  // 2. /setblock <x> <y> <z> <block>
  const setblockRegex = /^(\s*\/?setblock\s+)(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(.*)$/i;
  const setblockMatch = result.match(setblockRegex);
  if (setblockMatch) {
    const prefix = setblockMatch[1];
    const x = parseFloat(setblockMatch[2]) + deltaX;
    const y = setblockMatch[3];
    const z = setblockMatch[4];
    const rest = setblockMatch[5];
    return `${prefix}${x} ${y} ${z}${rest}`;
  }

  // 3. /tp or /teleport [target] <x> <y> <z> [yaw pitch]
  const tpTargetRegex = /^(\s*\/?(?:tp|teleport)\s+)(?!-?\d+(?:\.\d+)?(?:\s+|$))([^\s~^]+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(.*)$/i;
  const tpTargetMatch = result.match(tpTargetRegex);
  if (tpTargetMatch) {
    const prefix = `${tpTargetMatch[1]}${tpTargetMatch[2]} `;
    const x = parseFloat(tpTargetMatch[3]) + deltaX;
    const y = tpTargetMatch[4];
    const z = tpTargetMatch[5];
    const rest = tpTargetMatch[6];
    return `${prefix}${x} ${y} ${z}${rest}`;
  }

  // 4. /tp or /teleport <x> <y> <z> [yaw pitch]
  const tpNoTargetRegex = /^(\s*\/?(?:tp|teleport)\s+)(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(.*)$/i;
  const tpNoTargetMatch = result.match(tpNoTargetRegex);
  if (tpNoTargetMatch) {
    const prefix = tpNoTargetMatch[1];
    const x = parseFloat(tpNoTargetMatch[2]) + deltaX;
    const y = tpNoTargetMatch[3];
    const z = tpNoTargetMatch[4];
    const rest = tpNoTargetMatch[5];
    return `${prefix}${x} ${y} ${z}${rest}`;
  }

  // 5. /summon <entity> <x> <y> <z> [nbt]
  const summonRegex = /^(\s*\/?summon\s+[\w:]+\s+)(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(.*)$/i;
  const summonMatch = result.match(summonRegex);
  if (summonMatch) {
    const prefix = summonMatch[1];
    const x = parseFloat(summonMatch[2]) + deltaX;
    const y = summonMatch[3];
    const z = summonMatch[4];
    const rest = summonMatch[5];
    return `${prefix}${x} ${y} ${z}${rest}`;
  }

  // 6. /clone <x1> <y1> <z1> <x2> <y2> <z2> <x3> <y3> <z3>
  const cloneRegex = /^(\s*\/?clone\s+)(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(.*)$/i;
  const cloneMatch = result.match(cloneRegex);
  if (cloneMatch) {
    const prefix = cloneMatch[1];
    const x1 = parseFloat(cloneMatch[2]) + deltaX;
    const y1 = cloneMatch[3];
    const z1 = cloneMatch[4];
    const x2 = parseFloat(cloneMatch[5]) + deltaX;
    const y2 = cloneMatch[6];
    const z2 = cloneMatch[7];
    const x3 = parseFloat(cloneMatch[8]) + deltaX;
    const y3 = cloneMatch[9];
    const z3 = cloneMatch[10];
    const rest = cloneMatch[11];
    return `${prefix}${x1} ${y1} ${z1} ${x2} ${y2} ${z2} ${x3} ${y3} ${z3}${rest}`;
  }

  // 7. /spawnpoint [target] <x> <y> <z>
  const spawnpointTargetRegex = /^(\s*\/?spawnpoint\s+)(?!-?\d+(?:\.\d+)?(?:\s+|$))([^\s~^]+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(.*)$/i;
  const spawnpointTargetMatch = result.match(spawnpointTargetRegex);
  if (spawnpointTargetMatch) {
    const prefix = `${spawnpointTargetMatch[1]}${spawnpointTargetMatch[2]} `;
    const x = parseFloat(spawnpointTargetMatch[3]) + deltaX;
    const y = spawnpointTargetMatch[4];
    const z = spawnpointTargetMatch[5];
    const rest = spawnpointTargetMatch[6];
    return `${prefix}${x} ${y} ${z}${rest}`;
  }

  // 8. /spawnpoint <x> <y> <z>
  const spawnpointNoTargetRegex = /^(\s*\/?spawnpoint\s+)(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(.*)$/i;
  const spawnpointNoTargetMatch = result.match(spawnpointNoTargetRegex);
  if (spawnpointNoTargetMatch) {
    const prefix = spawnpointNoTargetMatch[1];
    const x = parseFloat(spawnpointNoTargetMatch[2]) + deltaX;
    const y = spawnpointNoTargetMatch[3];
    const z = spawnpointNoTargetMatch[4];
    const rest = spawnpointNoTargetMatch[5];
    return `${prefix}${x} ${y} ${z}${rest}`;
  }

  // 9. /setworldspawn <x> <y> <z>
  const worldspawnRegex = /^(\s*\/?setworldspawn\s+)(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(.*)$/i;
  const worldspawnMatch = result.match(worldspawnRegex);
  if (worldspawnMatch) {
    const prefix = worldspawnMatch[1];
    const x = parseFloat(worldspawnMatch[2]) + deltaX;
    const y = worldspawnMatch[3];
    const z = worldspawnMatch[4];
    const rest = worldspawnMatch[5];
    return `${prefix}${x} ${y} ${z}${rest}`;
  }

  return result;
}

export function createShardedTaskSpec(
  task: MinebenchTaskSpec,
  workerIndex: number,
  concurrency: number
): MinebenchTaskSpec {
  const deltaX = getWorkerSpatialOffset(workerIndex);
  if (deltaX === 0 && concurrency === 1) {
    return task;
  }

  const botUsername = getWorkerBotName(workerIndex, concurrency);
  const shardedOrigin = {
    x: task.origin.x + deltaX,
    y: task.origin.y,
    z: task.origin.z,
  };

  return {
    ...task,
    origin: shardedOrigin,
    buildArenaCommands: (orig?: { x: number; y: number; z: number }) => {
      const targetOrigin = orig ?? shardedOrigin;
      const cmds = task.buildArenaCommands(targetOrigin);
      const baseCmds = task.buildArenaCommands(task.origin);
      return cmds.map((cmd, idx) => {
        let transformed = cmd;
        if (botUsername !== 'NeuroAgent') {
          transformed = transformed
            .replace(/\bNeuroAgent\b/g, botUsername)
            .replace(/@p\b/g, botUsername);
        }
        // If the task returned identical command despite different targetOrigin (static coords),
        // or if targetOrigin was unshifted task.origin, shift command coordinates by deltaX
        if (deltaX !== 0 && (targetOrigin.x === task.origin.x || cmd === baseCmds[idx])) {
          transformed = shiftCommandCoordinates(transformed, deltaX);
        }
        return transformed;
      });
    },
  };
}

const BLOCK_VOCAB: Record<string, number> = {
  air: 0,
  cave_air: 0,
  void_air: 0,
  stone: 1,
  cobblestone: 2,
  dirt: 3,
  grass_block: 4,
  sand: 5,
  oak_log: 7,
  oak_planks: 9,
  water: 10,
  lava: 11,
  coal_ore: 12,
  iron_ore: 13,
  gold_ore: 14,
  diamond_ore: 15,
  crafting_table: 17,
  furnace: 18,
  torch: 20,
  bedrock: 21,
  obsidian: 22,
  stone_bricks: 24,
  farmland: 26,
  wheat: 27,
  gold_block: 28,
  emerald_block: 29,
  end_stone: 30,
};

function mapBlockToVocabId(name?: string): number {
  if (!name) return 0;
  if (BLOCK_VOCAB[name] !== undefined) return BLOCK_VOCAB[name];
  if (name.includes('log')) return 7;
  if (name.includes('planks')) return 9;
  if (name.includes('water')) return 10;
  if (name.includes('lava')) return 11;
  if (name.includes('iron_ore')) return 13;
  if (name.includes('stone') || name.includes('brick') || name.includes('andesite')) return 1;
  return 24;
}

function sampleLiveRLTransitionState(system1: System1BotController): {
  voxels: number[];
  entities: number[][];
  botState: number[];
} | null {
  const bot = system1.bot;
  if (!bot || !bot.entity || !bot.entity.position) return null;

  const pos = bot.entity.position.floored();
  const voxels: number[] = new Array(15 * 15 * 15).fill(0);
  let idx = 0;
  for (let dx = -7; dx <= 7; dx++) {
    for (let dy = -3; dy <= 11; dy++) {
      for (let dz = -7; dz <= 7; dz++) {
        const b = bot.blockAt(new Vec3(pos.x + dx, pos.y + dy, pos.z + dz));
        voxels[idx++] = mapBlockToVocabId(b?.name);
      }
    }
  }

  const botPos = bot.entity.position;
  const rawEntities = Object.values(bot.entities || {})
    .filter((e: any) => {
      if (!e || e === bot.entity || !e.position || e.isValid === false) return false;
      const name = String(e.name || e.displayName || '').toLowerCase();
      const maxRange = isBossOrCrystalEntity(name) ? 128.0 : 48.0;
      return e.position.distanceTo(botPos) <= maxRange;
    })
    .sort((a: any, b: any) => {
      const nameA = String(a.name || a.displayName || '').toLowerCase();
      const nameB = String(b.name || b.displayName || '').toLowerCase();
      const pA = getEntityPriority(nameA, a.type);
      const pB = getEntityPriority(nameB, b.type);
      if (pA !== pB) return pA - pB;
      return a.position.distanceTo(botPos) - b.position.distanceTo(botPos);
    })
    .slice(0, 16);

  const entities: number[][] = [];
  for (let i = 0; i < 16; i++) {
    const e: any = rawEntities[i];
    if (!e) {
      entities.push([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      continue;
    }
    const name = String(e.name || e.displayName || '').toLowerCase();
    const dx = (e.position.x - botPos.x) / 32.0;
    const dy = (e.position.y - botPos.y) / 16.0;
    const dz = (e.position.z - botPos.z) / 32.0;
    const dist = Math.min(1.0, e.position.distanceTo(botPos) / 32.0);
    const vx = Number.isFinite(e.velocity?.x) ? e.velocity.x : 0;
    const vy = Number.isFinite(e.velocity?.y) ? e.velocity.y : 0;
    const vz = Number.isFinite(e.velocity?.z) ? e.velocity.z : 0;
    const isHostile = HOSTILE_MOBS.has(name) ? 1.0 : 0.0;
    const isCreeper = name === 'creeper' ? 1.0 : 0.0;
    const isSkeleton = name === 'skeleton' ? 1.0 : 0.0;
    const isDragonOrCrystal = name === 'ender_dragon' || name === 'end_crystal' ? 1.0 : 0.0;
    const fuseState = name === 'creeper' && Number(e.metadata?.[16]) > 0 ? 1.0 : 0.0;
    entities.push([
      Number(dx.toFixed(4)),
      Number(dy.toFixed(4)),
      Number(dz.toFixed(4)),
      Number(dist.toFixed(4)),
      Number(vx.toFixed(4)),
      Number(vy.toFixed(4)),
      Number(vz.toFixed(4)),
      isHostile,
      isCreeper,
      isSkeleton,
      isDragonOrCrystal,
      fuseState,
    ]);
  }

  const vel = bot.entity.velocity || { x: 0, y: 0, z: 0 };
  const yaw = bot.entity.yaw || 0;
  const pitch = bot.entity.pitch || 0;
  const botState = [
    Number(((bot.health ?? 20) / 20.0).toFixed(4)),
    Number(((bot.food ?? 20) / 20.0).toFixed(4)),
    Number((vel.x || 0).toFixed(4)),
    Number((vel.y || 0).toFixed(4)),
    Number((vel.z || 0).toFixed(4)),
    Number(Math.sin(yaw).toFixed(4)),
    Number(Math.cos(yaw).toFixed(4)),
    Number((pitch / Math.PI).toFixed(4)),
    bot.entity.onGround ? 1.0 : 0.0,
    bot.entity.isInWater ? 1.0 : 0.0,
  ];

  return { voxels, entities, botState };
}

function mapToolToDiscreteActionIndex(toolName: string): number {
  switch (toolName) {
    case 'engage_combat':
      return 3;
    case 'shoot_bow':
      return 4;
    case 'eat_food':
      return 8;
    case 'navigate_parkour_or_bridge':
    case 'pillar_up':
      return 2;
    case 'build_simple_house':
    case 'construct_modular_structure':
    case 'till_and_farm':
    case 'interact_block_or_puzzle':
      return 4;
    default:
      return 1;
  }
}

// 4-Layer Procedural RNG: Layer 3 Semantic Prompt Paraphrasing Clusters
const PROMPT_VARIANTS: Record<string, string[]> = {
  build_enclosed_shelter: [
    'Build a simple house right here with walls, a roof, windows, a doorway, and interior torch lighting.',
    'Construct an enclosed survival cabin with solid walls, a roof overhead, a front door, and a torch inside.',
    'Erect a sealed shelter before nightfall with walls, a roof, windows, and interior utilities.',
  ],
  build_creative_monument: [
    'Construct a creative smiley face monument right here in creative mode without worrying about roof seals!',
    'Sculpt an artistic face monument on this plot in creative mode (weather seal checks bypassed)!',
    'Build a creative pixel art display with blocks on this platform!',
  ],
  agility_sky_parkour: [
    'Complete the parkour course and reach the gold_block at the end without falling!',
    'Leap across the floating platforms to touch the gold_block at the finish line!',
    'Sprint-jump across the elevated parkour pillars to reach the gold_block goal!',
  ],
  agility_ravine_bridge: [
    'Bridge across the ravine chasm using cobblestone to touch the gold_block on the other side!',
    'Cross the deep ravine by bridging with blocks and reach the gold_block safely!',
  ],
  combat_creeper_kiting: [
    'Defeat the creeper without taking explosion damage by kiting and using your shield!',
    'Engage and slay the creeper safely; raise your shield during fuses!',
  ],
  combat_skeleton_duel: [
    'Defeat the armed skeleton using timed melee strikes and your shield to block arrows!',
    'Duel and slay the skeleton while deflecting incoming arrows with your shield!',
  ],
};

export function resolveMinecraftCoordinates(
  cmd: string,
  origin: { x: number; y: number; z: number },
  botName = 'NeuroAgent'
): string {
  return cmd
    .replace(/~([0-9\-\.]*)\s+~([0-9\-\.]*)\s+~([0-9\-\.]*)/g, (_, dx, dy, dz) => {
      const ox = dx ? parseFloat(dx) : 0;
      const oy = dy ? parseFloat(dy) : 0;
      const oz = dz ? parseFloat(dz) : 0;
      return `${origin.x + ox} ${origin.y + oy} ${origin.z + oz}`;
    })
    .replace(/~([0-9\-\.]*)/g, (_, dy) => String(origin.y + (dy ? parseFloat(dy) : 0)))
    .replace(/@p/g, botName);
}

export interface SurvivalCampaignGoal {
  id: string;
  title: string;
  prompt: string;
}

export const SURVIVAL_CAMPAIGN_GOALS: SurvivalCampaignGoal[] = [
  {
    id: 'ender_dragon',
    title: 'Slay the Ender Dragon',
    prompt:
      'Open-World Survival Campaign: Survive, progress through Wood, Stone, Iron, Diamond, and Nether tiers, craft gear, and defeat the Ender Dragon.',
  },
  {
    id: 'wither_boss',
    title: 'Summon & Defeat the Wither',
    prompt:
      'Open-World Survival Campaign: Progress through Stone and Iron gear, gather Nether resources and skulls, and defeat the Wither boss.',
  },
  {
    id: 'full_beacon',
    title: 'Activate a Full Beacon',
    prompt:
      'Open-World Survival Campaign: Mine ores and obsidian, smelt glass, craft a Beacon, and activate it atop a mineral pyramid.',
  },
  {
    id: 'ocean_sponges',
    title: 'Collect Ocean Monument Sponges',
    prompt:
      'Open-World Survival Campaign: Craft armor and weapons, hunt food, gear up for underwater exploration, and acquire Sponges.',
  },
  {
    id: 'diamond_armory',
    title: 'Craft & Equip Diamond Armory',
    prompt:
      'Open-World Survival Campaign: Gather wood and stone, smelt iron, excavate diamond ore underground, and craft and equip Diamond Armor and Tools.',
  },
  {
    id: 'homestead_automation',
    title: 'Build Self-Sustaining Homestead',
    prompt:
      'Open-World Survival Campaign: Gather wood and stone, build an enclosed survival shelter, craft chests and furnaces, till farmland, and breed livestock.',
  },
];

export const SURVIVAL_SECTOR_ORIGIN = { x: 20000, y: 100, z: 20000 };

export async function seedPersistentSurvivalSector(
  server: HeadlessMinecraftServerManager,
  botUsername = 'NeuroAgent_Surv'
): Promise<void> {
  const ox = SURVIVAL_SECTOR_ORIGIN.x;
  const oy = SURVIVAL_SECTOR_ORIGIN.y;
  const oz = SURVIVAL_SECTOR_ORIGIN.z;

  // Step 1: Load and force-lock survival chunks BEFORE placing terrain blocks
  server.sendCommand(`op ${botUsername}`);
  server.sendCommand(`gamemode creative ${botUsername}`);
  server.sendCommand(`forceload add ${ox - 24} ${oz - 24} ${ox + 24} ${oz + 24}`);
  server.sendCommand(`tp ${botUsername} ${ox} ${oy + 1} ${oz} 0 0`);
  server.sendCommand(`spawnpoint ${botUsername} ${ox} ${oy} ${oz}`);
  await new Promise((r) => setTimeout(r, 500));

  const cmds = [
    `fill ${ox - 24} ${oy - 9} ${oz - 24} ${ox + 24} ${oy - 9} ${oz + 24} minecraft:bedrock`,
    `fill ${ox - 24} ${oy - 8} ${oz - 24} ${ox + 24} ${oy - 2} ${oz + 24} minecraft:stone`,
    `fill ${ox - 24} ${oy - 1} ${oz - 24} ${ox + 24} ${oy - 1} ${oz + 24} minecraft:grass_block`,
    `fill ${ox - 24} ${oy} ${oz - 24} ${ox + 24} ${oy + 12} ${oz + 24} minecraft:air`,
    // Underground ore veins (Coal, Iron, Gold, Diamond strata)
    `fill ${ox - 6} ${oy - 3} ${oz - 6} ${ox - 2} ${oy - 2} ${oz - 2} minecraft:coal_ore`,
    `fill ${ox + 3} ${oy - 4} ${oz - 6} ${ox + 6} ${oy - 3} ${oz - 2} minecraft:iron_ore`,
    `fill ${ox - 5} ${oy - 6} ${oz + 3} ${ox - 2} ${oy - 5} ${oz + 6} minecraft:gold_ore`,
    `fill ${ox + 3} ${oy - 7} ${oz + 3} ${ox + 5} ${oy - 6} ${oz + 5} minecraft:diamond_ore`,
    // Exposed stone quarry outcrop & sand bank
    `fill ${ox - 9} ${oy} ${oz - 3} ${ox - 6} ${oy + 2} ${oz + 3} minecraft:stone`,
    `fill ${ox - 8} ${oy + 1} ${oz - 1} ${ox - 7} ${oy + 2} ${oz + 1} minecraft:coal_ore`,
    `fill ${ox - 12} ${oy - 1} ${oz + 8} ${ox - 6} ${oy - 1} ${oz + 14} minecraft:sand`,
    // Multi-tree oak & birch grove
    `fill ${ox + 5} ${oy} ${oz + 4} ${ox + 5} ${oy + 4} ${oz + 4} minecraft:oak_log`,
    `fill ${ox + 4} ${oy + 3} ${oz + 3} ${ox + 6} ${oy + 5} ${oz + 5} minecraft:oak_leaves`,
    `fill ${ox + 10} ${oy} ${oz - 6} ${ox + 10} ${oy + 4} ${oz - 6} minecraft:birch_log`,
    `fill ${ox + 9} ${oy + 3} ${oz - 7} ${ox + 11} ${oy + 5} ${oz - 5} minecraft:birch_leaves`,
    `fill ${ox - 4} ${oy} ${oz + 10} ${ox - 4} ${oy + 4} ${oz + 10} minecraft:oak_log`,
    `fill ${ox - 5} ${oy + 3} ${oz + 9} ${ox - 3} ${oy + 5} ${oz + 11} minecraft:oak_leaves`,
    // Water pond & isolated obsidian/lava hazard rim
    `fill ${ox + 6} ${oy - 1} ${oz + 10} ${ox + 10} ${oy - 1} ${oz + 14} minecraft:water`,
    `fill ${ox + 15} ${oy - 1} ${oz + 15} ${ox + 17} ${oy - 1} ${oz + 17} minecraft:lava`,
    // Natural passive livestock (world is persistent, never wiped; hostiles spawn naturally at night)
    `summon minecraft:cow ${ox + 8} ${oy} ${oz + 8}`,
    `summon minecraft:sheep ${ox + 9} ${oy} ${oz - 9}`,
    `tp ${botUsername} ${ox} ${oy} ${oz} 0 0`,
    `gamemode survival ${botUsername}`,
  ];

  for (const cmd of cmds) {
    server.sendCommand(cmd);
  }
  await new Promise((r) => setTimeout(r, 350));
}

export function evaluateSurvivalAdvancements(
  invNames: string[],
  blocksPlaced: number,
  mobsKilledTotal: number
): { unlocked: string[]; techTier: string; score: number } {
  const has = (sub: string) => invNames.some((n) => n.includes(sub));
  const unlocked: string[] = [];

  if (has('_log') || has('_stem') || has('_planks')) unlocked.push('Wood Age');
  if (has('cobblestone') || has('stone_pickaxe') || has('furnace')) unlocked.push('Stone Age');
  if (has('stone_pickaxe') || has('iron_pickaxe') || has('diamond_pickaxe')) unlocked.push('Getting an Upgrade');
  if (has('iron_ingot') || has('raw_iron') || has('iron_sword') || has('iron_pickaxe')) unlocked.push('Acquire Hardware');
  if (has('_helmet') || has('_chestplate') || has('_leggings') || has('_boots')) unlocked.push('Suit Up');
  if (has('bucket') || has('obsidian') || has('flint_and_steel')) unlocked.push('Hot Stuff');
  if (has('diamond')) unlocked.push('Diamonds!');
  if (has('blaze') || has('nether') || has('wither')) unlocked.push('We Need to Go Deeper');
  if (blocksPlaced >= 4 || has('bread') || has('wheat') || mobsKilledTotal > 0) unlocked.push('Homesteader & Hunter');

  let techTier = 'Raw';
  if (unlocked.includes('We Need to Go Deeper')) techTier = 'Nether';
  else if (unlocked.includes('Diamonds!')) techTier = 'Diamond';
  else if (unlocked.includes('Acquire Hardware') || unlocked.includes('Suit Up')) techTier = 'Iron';
  else if (unlocked.includes('Stone Age') || unlocked.includes('Getting an Upgrade')) techTier = 'Stone';
  else if (unlocked.includes('Wood Age')) techTier = 'Wood';

  const score = Math.min(100, unlocked.length * 14 + Math.min(16, blocksPlaced * 2 + mobsKilledTotal * 5));
  return { unlocked, techTier, score };
}

export function isHeldOutEvalTask(taskId: string): boolean {
  const id = String(taskId || '');
  if (!id.startsWith('omni_')) return false;
  // Deterministic 20% FNV-1a split so held-out tasks stay isolated across every training loop
  let hash = 2166136261;
  for (let i = 0; i < id.length; i++) {
    hash ^= id.charCodeAt(i);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash % 5 === 0;
}

export const WOOD_SPECIES_POOL = ['oak', 'birch', 'spruce', 'acacia', 'jungle', 'dark_oak'] as const;

export function applyRandomWoodSpecies(task: MinebenchTaskSpec, fixedSpecies?: string): MinebenchTaskSpec {
  const tid = String(task?.id || '');
  if (!tid.startsWith('omni_craft_') && tid !== 'prog_tier1_stone_age' && tid !== 'prog_tier3_iron_armor') {
    return task;
  }
  const species =
    fixedSpecies || WOOD_SPECIES_POOL[Math.floor(Math.random() * WOOD_SPECIES_POOL.length)];
  if (!species || species === 'oak') return task;

  const updatedPrompt = String(task.prompt || '')
    .replace(/\boak logs\b/gi, `${species} logs`)
    .replace(/\boak_log\b/g, `${species}_log`);

  const origBuild = task.buildArenaCommands;
  if (!origBuild) {
    return {
      ...task,
      prompt: updatedPrompt,
    };
  }
  return {
    ...task,
    prompt: updatedPrompt,
    buildArenaCommands: (orig) =>
      origBuild(orig).map((cmd) =>
        cmd
          .replace(/\boak_log\b/g, `${species}_log`)
          .replace(/\boak_leaves\b/g, `${species}_leaves`)
      ),
  };
}

function sampleDiverseOmniscienceTasks(tasksPath: string, count = 50): MinebenchTaskSpec[] {
  if (!fs.existsSync(tasksPath)) return [];
  try {
    const rawAll: any[] = JSON.parse(fs.readFileSync(tasksPath, 'utf-8'));
    const omniOnly = rawAll.filter((t) => t.taskId && t.taskId.startsWith('omni_') && t.taskId !== 'omni_mob_breeze');
    const trainByCat: Record<string, any[]> = {};
    const evalByCat: Record<string, any[]> = {};
    for (const t of omniOnly) {
      const cat = t.category || 'tool_affordance';
      const bucketMap = isHeldOutEvalTask(t.taskId) ? evalByCat : trainByCat;
      if (!bucketMap[cat]) bucketMap[cat] = [];
      bucketMap[cat].push(t);
    }

    for (const cat in trainByCat) trainByCat[cat].sort(() => Math.random() - 0.5);
    for (const cat in evalByCat) evalByCat[cat].sort(() => Math.random() - 0.5);

    const quotas: Record<string, number> = {
      tool_affordance: 8,
      nutrition: 7,
      equipment: 6,
      crafting: 7,
      block_mechanic: 4,
      mob_interaction: 15,
      structure_raid: 3,
    };

    const targetEvalCount = count >= 5 ? Math.max(1, Math.round(count * 0.2)) : 0;
    const selectedEval: any[] = [];
    const selectedTrain: any[] = [];

    for (const [cat, quota] of Object.entries(quotas)) {
      const evalQuota = targetEvalCount > 0 ? Math.max(1, Math.round(quota * 0.2)) : 0;
      const trainQuota = Math.max(0, quota - evalQuota);
      if (evalQuota > 0 && evalByCat[cat]) {
        selectedEval.push(...evalByCat[cat].slice(0, evalQuota));
      }
      if (trainByCat[cat]) {
        selectedTrain.push(...trainByCat[cat].slice(0, trainQuota));
      }
    }

    const selected: any[] = [...selectedTrain.slice(0, count - targetEvalCount), ...selectedEval.slice(0, targetEvalCount)];

    if (selected.length < count) {
      const selectedIds = new Set(selected.map((t) => t.taskId));
      const remaining = omniOnly.filter((t) => !selectedIds.has(t.taskId)).sort(() => Math.random() - 0.5);
      selected.push(...remaining.slice(0, count - selected.length));
    }

    const origin = { x: 500, y: 100, z: 500 };

    const domainMap: Record<string, MinebenchDomain> = {
      CRAFTING_RECIPES: 'HOMESTEADING_ECONOMY',
      TOOL_AFFORDANCES: 'HOMESTEADING_ECONOMY',
      MOB_INTERACTION: 'TACTICAL_COMBAT',
      STRUCTURE_RAID: 'TACTICAL_COMBAT',
      FARMING_HUSBANDRY: 'HOMESTEADING_ECONOMY',
      MINING_ORES: 'HOMESTEADING_ECONOMY',
      BLOCK_PERCEPTION: 'PUZZLE_ENGINEERING',
    };

    return selected.slice(0, count).map((t) => ({
      id: t.taskId,
      title: t.title || t.taskId,
      domain: domainMap[t.domain] || ('HOMESTEADING_ECONOMY' as MinebenchDomain),
      tier: t.tier || 1,
      maxDurationSeconds:
        t.domain === 'CRAFTING_RECIPES' || (t.taskId && t.taskId.startsWith('omni_craft_'))
          ? 65
          : t.taskId === 'omni_tool_lava_bucket_smelt_fuel_100'
          ? 45
          : t.domain === 'STRUCTURE_RAID'
          ? 40
          : 25,
      maxSteps:
        t.maxSteps ||
        (t.domain === 'CRAFTING_RECIPES' || (t.taskId && t.taskId.startsWith('omni_craft_')) ? 8 : 4),
      prompt: t.prompt,
      origin,
      targetObject: t.targetObject,
      initialLoadout: t.initialLoadout || [],
      expectedTool: t.expectedTool,
      expectedToolArgs: t.expectedToolArgs,
      evaluatorMilestones: t.evaluatorMilestones,
      buildArenaCommands: (orig: { x: number; y: number; z: number }) => {
        const rawCmds: string[] = t.setupCommands || [];
        let elevatedTpCmd: string | null = null;
        let maxTpDy = 0;
        for (const cmd of rawCmds) {
          const m = cmd.match(/^tp\s+@p\s+~\S*\s+~(\d+(?:\.\d+)?)\s+~/);
          if (m) {
            const dy = parseFloat(m[1]);
            if (dy >= 5) {
              elevatedTpCmd = cmd;
              maxTpDy = Math.max(maxTpDy, Math.ceil(dy));
            }
          }
        }
        const airTop = orig.y + Math.max(10, maxTpDy + 10);
        const finalTp = elevatedTpCmd
          ? resolveMinecraftCoordinates(elevatedTpCmd, orig)
          : `tp NeuroAgent ${orig.x} ${orig.y} ${orig.z} 0 0`;
        return [
          `gamerule commandModificationBlockLimit 100000`,
          `fill ${orig.x - 12} ${orig.y - 3} ${orig.z - 12} ${orig.x + 12} ${orig.y - 1} ${orig.z + 12} minecraft:stone`,
          `fill ${orig.x - 12} ${orig.y - 4} ${orig.z - 12} ${orig.x + 12} ${orig.y - 4} ${orig.z + 12} minecraft:bedrock`,
          `fill ${orig.x - 12} ${orig.y} ${orig.z - 12} ${orig.x + 12} ${airTop} ${orig.z + 12} minecraft:air`,
          `fill ${orig.x - 12} ${orig.y} ${orig.z - 12} ${orig.x + 12} ${orig.y + 2} ${orig.z + 12} minecraft:glass outline`,
          `fill ${orig.x - 11} ${orig.y} ${orig.z - 11} ${orig.x + 11} ${orig.y + 2} ${orig.z + 11} minecraft:air`,
          ...rawCmds
            .filter((cmd: string) => !cmd.startsWith('tp @p') && !cmd.startsWith('tp NeuroAgent'))
            .map((cmd: string) => resolveMinecraftCoordinates(cmd, orig)),
          finalTp,
        ];
      },
    }));
  } catch (err) {
    console.error('[TrainLoop] Error sampling Omniscience tasks:', err);
    return [];
  }
}

export async function runContinuousTrainingLoop(): Promise<void> {
  const resultsDir = path.resolve(process.cwd(), 'minebench', 'results');
  const checkpointsDir = path.resolve(process.cwd(), 'neural_lab', 'checkpoints');
  fs.mkdirSync(resultsDir, { recursive: true });
  fs.mkdirSync(checkpointsDir, { recursive: true });

  const cliArgs = process.argv.slice(2);
  const isEvalOnly = cliArgs.includes('--eval-only') || cliArgs.includes('--no-train');
  const iterIdx = cliArgs.indexOf('--iterations') !== -1 ? cliArgs.indexOf('--iterations') : cliArgs.indexOf('--loops');
  const maxCycles =
    iterIdx !== -1 && cliArgs[iterIdx + 1]
      ? parseInt(cliArgs[iterIdx + 1], 10)
      : isEvalOnly
      ? 1
      : Infinity;

  const maxHoursIdx = cliArgs.indexOf('--max-hours') !== -1 ? cliArgs.indexOf('--max-hours') : cliArgs.indexOf('--hours');
  const maxHours =
    maxHoursIdx !== -1 && cliArgs[maxHoursIdx + 1]
      ? parseFloat(cliArgs[maxHoursIdx + 1])
      : Infinity;
  const sessionStartTime = Date.now();

  const dragonIdx = cliArgs.indexOf('--full_dragon') !== -1 ? cliArgs.indexOf('--full_dragon') : cliArgs.indexOf('--full-dragon');
  const fullDragonInterval =
    dragonIdx !== -1 && cliArgs[dragonIdx + 1]
      ? parseInt(cliArgs[dragonIdx + 1], 10)
      : 10;
  const dragonFinalOnly = cliArgs.includes('--dragon-final') || cliArgs.includes('--dragon-at-end') || cliArgs.includes('--dragon');

  const omniIdx = cliArgs.indexOf('--omni-tasks') !== -1 ? cliArgs.indexOf('--omni-tasks') : cliArgs.indexOf('--omniscience-count');
  const omniTaskCount =
    omniIdx !== -1 && cliArgs[omniIdx + 1]
      ? parseInt(cliArgs[omniIdx + 1], 10)
      : 50;

  const concurrency = parseConcurrencyArg(cliArgs, process.env.CONCURRENCY);
  let cyclesCompleted = 0;

  const statePath = path.join(resultsDir, 'training_state.json');
  const replayPath = path.join(resultsDir, 'rl_replay_buffer.jsonl');
  const transitionsPath = path.join(resultsDir, 'real_rl_transitions.jsonl');
  const motorChunksPath = path.join(resultsDir, 'real_motor_chunks.jsonl');

  // Load existing persistent training state or initialize
  let state: TrainingState = {
    sessionCount: 1,
    currentIteration: 0,
    totalEpisodesPlayed: 0,
    bestPassRate: 0,
    bestMeanScore: 0,
    bestIteration: 0,
    latestMeanScore: 0,
    latestPassRate: 0,
    latestReward: 0,
    lastSavedAt: new Date().toISOString(),
    latestCheckpoint: 'voxel_reflex_policy_latest.pt',
    bestCheckpoint: 'voxel_reflex_policy_best.pt',
  };

  if (fs.existsSync(statePath)) {
    try {
      const parsed = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
      state = {
        ...state,
        ...parsed,
        sessionCount: (parsed.sessionCount || 0) + 1,
      };
    } catch {}
  }

  console.log(`\n======================================================================`);
  console.log(`  NEUROCRAFT: CONTINUOUS REINFORCEMENT LEARNING & MODEL TRAINER`);
  console.log(`  - Device: Local NVIDIA GeForce RTX 5060 (Blackwell sm_120)`);
  console.log(`  - Target: Official Mojang Minecraft 1.20.4 server.jar (Real Physics)`);
  console.log(`  - Mode: Resumable Iterative Loop (Ctrl+C safely pauses anytime)`);
  console.log(`  - Concurrency: ${concurrency} Parallel Bot Worker(s) (Spatial Arena Sharding ΔX=4000)`);
  console.log(`======================================================================`);
  if (state.currentIteration > 0) {
    console.log(
      `[Resume] Loaded Session #${state.sessionCount} | Resuming from Iteration ${state.currentIteration + 1}`
    );
    console.log(
      `[Resume] All-time Best Score: ${state.bestMeanScore}/100 (${state.bestPassRate}% pass rate at Iteration #${state.bestIteration})`
    );
  } else {
    console.log(`[Init] Starting fresh training session #1 at Iteration 1!`);
  }
  console.log(`======================================================================\n`);

  // Configure strict local SLM mode (zero cloud cost, genuine model reasoning)
  process.env.DISABLE_FAST_INTENT = 'true';
  process.env.STRICT_LLM_BENCHMARK = 'true';
  process.env.NEURAL_FORCE_LOCAL_SLM = 'true';
  process.env.LLM_PROVIDER = 'local_gpu';
  let spawnedDaemonProc: any = null;

  // Ensure RTX 5060 Neural Daemon is running on http://127.0.0.1:8765
  try {
    const hRes = await fetch('http://127.0.0.1:8765/health', { signal: AbortSignal.timeout(1500) });
    if (!hRes.ok) throw new Error();
    console.log('[TrainLoop] RTX 5060 Neural Daemon is live and responsive.');
  } catch {
    console.log('[TrainLoop] RTX 5060 Neural Daemon is down. Launching neural_daemon.py...');
    const venvPython = path.resolve(process.cwd(), 'neural_lab', '.venv', 'Scripts', 'python.exe');
    const daemonScript = path.resolve(process.cwd(), 'neural_lab', 'neural_daemon.py');
    const daemonProc = spawn(venvPython, [daemonScript], {
      detached: true,
      stdio: 'ignore',
      cwd: process.cwd(),
    });
    spawnedDaemonProc = daemonProc;
    daemonProc.unref();
    let isHealthy = false;
    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      try {
        const check = await fetch('http://127.0.0.1:8765/health', { signal: AbortSignal.timeout(1500) });
        if (check.ok) {
          isHealthy = true;
          break;
        }
      } catch {}
    }
    if (isHealthy) {
      console.log('[TrainLoop] RTX 5060 Neural Daemon successfully initialized in VRAM.');
    } else {
      console.warn('[TrainLoop] Warning: Neural Daemon took longer than expected to report health.');
    }
  }

  const serverPort = 25575;
  const server = new HeadlessMinecraftServerManager();
  const serverLifecycleMutex = new AsyncMutex();
  await server.startServer(serverPort);

  console.log(`[TrainLoop] Initializing ${concurrency} parallel worker bot(s) on port ${serverPort}...`);
  const workers: WorkerContext[] = [];
  for (let i = 0; i < concurrency; i++) {
    const botUsername = getWorkerBotName(i, concurrency);
    const deltaX = getWorkerSpatialOffset(i);
    const sparringUsername = concurrency === 1 ? 'DQN_Opponent' : `DQN_Opponent_${i + 1}`;

    const system1 = new System1BotController();
    let connected = false;
    for (let cAttempt = 1; cAttempt <= 3; cAttempt++) {
      try {
        console.log(`[TrainLoop] Connecting worker #${i} (${botUsername}, attempt #${cAttempt}) at X-offset +${deltaX}...`);
        await system1.connect('127.0.0.1', serverPort, botUsername);
        connected = true;
        break;
      } catch (cErr: any) {
        console.warn(`[TrainLoop] Warning: Worker #${i} connect attempt #${cAttempt} failed: ${cErr?.message || cErr}`);
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
    if (!connected) {
      throw new Error(`Failed to connect worker #${i} (${botUsername}) after 3 attempts`);
    }
    await new Promise((r) => setTimeout(r, 600));

    const sparringPartner = new SparringPartnerBot(sparringUsername, 'http://127.0.0.1:8765');
    try {
      await sparringPartner.connect('127.0.0.1', serverPort);
      console.log(`[TrainLoop] Connected sparring partner for worker #${i}: ${sparringUsername}`);
      await new Promise((r) => setTimeout(r, 300));
    } catch (err: any) {
      console.warn(`[TrainLoop] Could not connect sparring partner ${sparringUsername}: ${err?.message || err}`);
    }

    const registry = new DynamicMCPToolRegistry();
    registerBaseMCPTools(registry);
    const mcpServer = new NeuroCraftMCPServer(registry);
    const llm = new UnifiedLLMClient();

    workers.push({
      id: i,
      botUsername,
      deltaX,
      system1,
      sparringPartner,
      registry,
      mcpServer,
      llm,
    });
  }
  console.log(`[TrainLoop] All ${concurrency} worker(s) successfully connected and ready!`);

  // Connect dedicated Persistent Open-World Survival Worker ('NeuroAgent_Surv') at X=20000, Z=20000
  const enableSurvivalWorker = !cliArgs.includes('--no-survival');
  let survivalWorker: {
    botUsername: string;
    system1: System1BotController;
    registry: DynamicMCPToolRegistry;
    mcpServer: NeuroCraftMCPServer;
    llm: UnifiedLLMClient;
  } | null = null;
  let survivalSectorSeeded = false;
  const persistentUnlockedAdvancements = new Set<string>();

  if (enableSurvivalWorker) {
    try {
      const sSystem1 = new System1BotController();
      console.log(`[TrainLoop] Connecting Persistent Survival Worker (NeuroAgent_Surv) at (20000, 100, 20000)...`);
      await sSystem1.connect('127.0.0.1', serverPort, 'NeuroAgent_Surv');
      await new Promise((r) => setTimeout(r, 500));
      const sReg = new DynamicMCPToolRegistry();
      registerBaseMCPTools(sReg);
      survivalWorker = {
        botUsername: 'NeuroAgent_Surv',
        system1: sSystem1,
        registry: sReg,
        mcpServer: new NeuroCraftMCPServer(sReg),
        llm: new UnifiedLLMClient(),
      };
      await seedPersistentSurvivalSector(server, 'NeuroAgent_Surv');
      survivalSectorSeeded = true;
      console.log(`[TrainLoop] Persistent Open-World Survival Sector seeded at (20000, 100, 20000)!\n`);
    } catch (sErr: any) {
      console.warn(`[TrainLoop] Warning: Could not initialize Survival Worker: ${sErr?.message || sErr}\n`);
    }
  }

  let isShuttingDown = false;

  const saveTrainingState = (isBest: boolean) => {
    state.lastSavedAt = new Date().toISOString();
    fs.writeFileSync(statePath, JSON.stringify(state, null, 2), 'utf-8');
    try {
      generateTrainingVisuals();
    } catch {}
  };

  const handleInterrupt = async () => {
    if (isShuttingDown) return;
    isShuttingDown = true;
    console.log(`\n\n[TrainLoop] Ctrl+C detected! Safely pausing training and saving weights...`);
    saveTrainingState(false);
    for (const w of workers) {
      try {
        await w.sparringPartner.disconnect().catch(() => {});
      } catch {}
      try {
        w.system1.bot?.quit();
      } catch {}
    }
    try {
      survivalWorker?.system1.bot?.quit();
    } catch {}
    await server.stopServer();
    if (spawnedDaemonProc) {
      try {
        await fetch('http://127.0.0.1:8765/shutdown', { method: 'POST', signal: AbortSignal.timeout(1500) }).catch(() => {});
      } catch {}
      try {
        spawnedDaemonProc.kill();
      } catch {}
    }
    console.log(`======================================================================`);
    console.log(`  TRAINING PAUSED CLEANLY`);
    console.log(`  - Completed Iterations : ${state.currentIteration}`);
    console.log(`  - Total Real Episodes  : ${state.totalEpisodesPlayed}`);
    console.log(`  - All-Time Best Score  : ${state.bestMeanScore}/100 (Pass Rate: ${state.bestPassRate}%)`);
    console.log(`  - Latest Weights Saved : neural_lab/checkpoints/voxel_reflex_policy_latest.pt`);
    console.log(`  - Best Weights Saved   : neural_lab/checkpoints/voxel_reflex_policy_best.pt`);
    console.log(`  Run train_agent.bat anytime to pick up right where you left off!`);
    console.log(`======================================================================\n`);
    process.exit(0);
  };

  process.on('SIGINT', handleInterrupt);
  process.on('SIGTERM', handleInterrupt);

  try {
    while (!isShuttingDown) {
      state.currentIteration++;
      const iter = state.currentIteration;
      const iterStartTime = Date.now();
      const tasksPath = path.resolve(process.cwd(), 'minebench', 'results', 'omniscience_tasks.json');
      const sampledOmniTasks = sampleDiverseOmniscienceTasks(tasksPath, omniTaskCount).map((t) =>
        applyRandomWoodSpecies(t)
      );
      const canonicalTasks = MINEBENCH_TASKS.map((t) => applyRandomWoodSpecies(t));
      const loopTasks: MinebenchTaskSpec[] = [
        ...canonicalTasks,
        ...sampledOmniTasks,
        ...MULTIPLAYER_TASKS,
      ];
      const numTasks = loopTasks.length;

      console.log(`\n======================================================================`);
      console.log(`>>> [Iteration #${iter}] Starting Exhaustive Anti-Collapse Curriculum Pass:`);
      console.log(`    - 16 Canonical MINEbench Challenges`);
      console.log(`    - ${sampledOmniTasks.length} Stratified Omniscience Affordances across 7 categories (~20% held-out)`);
      console.log(`    - ${MULTIPLAYER_TASKS.length} Guaranteed Multiplayer Scenarios`);
      console.log(`    - Total In-Game Physical Scenarios: ${numTasks}`);
      console.log(`    - Active Bot Concurrency: ${concurrency} Worker(s) (Spatial Arena Sharding ΔX=4000)`);
      console.log(`======================================================================\n`);

      let iterPassed = 0;
      let iterScoreSum = 0;
      let iterRewardSum = 0;
      let heldOutPassed = 0;
      let heldOutTotal = 0;
      let heldOutScoreSum = 0;
      const domainStats: Record<string, { passed: number; total: number; scoreSum: number }> = {};

      const classifyDomainBucket = (t: MinebenchTaskSpec): string => {
        const tid = (t.id || '').toLowerCase();
        if (tid.startsWith('omni_craft_') || tid.startsWith('prog_')) return 'Crafting & Tech';
        if (tid.startsWith('omni_mob_') || t.domain === 'TACTICAL_COMBAT') return 'Tactical Combat';
        if (t.domain === 'AGILITY_PARKOUR') return 'Agility & Parkour';
        if (t.domain === 'ARCHITECTURE_BUILDING') return 'Building & 3D';
        if (t.domain === 'MULTIPLAYER_ARENA') return 'Multiplayer';
        if (t.domain === 'PUZZLE_ENGINEERING') return 'Puzzles & Tooling';
        return 'Survival & Affordances';
      };

      const arenaResetMutex = new AsyncMutex();
      const stateWriteMutex = new AsyncMutex();
      const taskQueue = new ConcurrentTaskQueue(loopTasks);
      let testbenchFinished = false;

      // Launch Persistent Open-World Survival Worker concurrently with Testbench workers (zero extra batch overhead)
      const campaignGoal = SURVIVAL_CAMPAIGN_GOALS[(iter - 1) % SURVIVAL_CAMPAIGN_GOALS.length];
      const survivalPromise = (async () => {
        if (!survivalWorker) return undefined;
        try {
          await serverLifecycleMutex.runExclusive(async () => {
            if (!server.isReady) {
              await server.startServer(serverPort);
            }
            if (
              !survivalWorker.system1.bot ||
              !survivalWorker.system1.bot.entity ||
              (survivalWorker.system1.bot as any)._client?.ended
            ) {
              await survivalWorker.system1.connect('127.0.0.1', serverPort, survivalWorker.botUsername);
              await new Promise((r) => setTimeout(r, 500));
            }
          });
          const sBot = survivalWorker.system1.bot;
          if (!sBot?.entity?.position) return undefined;

          if (!survivalSectorSeeded) {
            await seedPersistentSurvivalSector(server, survivalWorker.botUsername);
            survivalSectorSeeded = true;
          } else {
            const distFromOrigin = Math.hypot(
              sBot.entity.position.x - SURVIVAL_SECTOR_ORIGIN.x,
              sBot.entity.position.z - SURVIVAL_SECTOR_ORIGIN.z
            );
            if (distFromOrigin > 96 || sBot.entity.position.y < 60) {
              server.sendCommand(
                `tp ${survivalWorker.botUsername} ${SURVIVAL_SECTOR_ORIGIN.x} ${SURVIVAL_SECTOR_ORIGIN.y} ${SURVIVAL_SECTOR_ORIGIN.z} 0 0`
              );
              server.sendCommand(`gamemode survival ${survivalWorker.botUsername}`);
              await new Promise((r) => setTimeout(r, 200));
            }
          }

          let sDeaths = 0;
          let sBlocksPlaced = 0;
          let sKills = 0;
          const onSDeath = () => { sDeaths++; };
          const onSBlock = (o: any, n: any) => {
            if (o?.name === 'air' && n?.name && n.name !== 'air' && n.position?.distanceTo(sBot.entity.position) <= 6.0) {
              sBlocksPlaced++;
            }
          };
          const onSKill = (e: any) => {
            if (e && !String(e.name || '').toLowerCase().startsWith('neuroagent')) sKills++;
          };
          sBot.on('death', onSDeath);
          sBot.on('blockUpdate', onSBlock);
          sBot.on('entityDead', onSKill);

          const sStepHistory: ExecutedStepRecord[] = [];
          const sExecutedTools: any[] = [];
          const sGoalPlanner = new HierarchicalGoalPlanner();
          sGoalPlanner.setMacroGoal(campaignGoal.prompt);
          let bestAdvCount = persistentUnlockedAdvancements.size;
          let finalEval = evaluateSurvivalAdvancements(
            (sBot.inventory?.items() || []).map((i: any) => i.name.toLowerCase()),
            sBlocksPlaced,
            sKills
          );

          try {
            const maxSurvivalSteps = 10;
            for (let sIdx = 0; sIdx < maxSurvivalSteps; sIdx++) {
              if (isShuttingDown || sDeaths > 0 || (testbenchFinished && sIdx >= 3)) break;
              const preState = sampleLiveRLTransitionState(survivalWorker.system1);
              const worldState = survivalWorker.system1.getLatestWorldState();
              const schemas = survivalWorker.registry.listSchemas();
              const preStepSystemPrompt = survivalWorker.llm.buildGroundedTrainingSystemPrompt(
                schemas,
                worldState,
                sIdx === 0 ? undefined : sStepHistory
              );
              const goalHeader = sIdx === 0 ? campaignGoal.prompt : sGoalPlanner.formatGoalHorizonContext(worldState);
              const dynamicPrompt = `${goalHeader} [Tech Tier: ${finalEval.techTier} | Unlocked: ${
                finalEval.unlocked.join(', ') || 'none'
              }]`;

              const dec =
                sIdx === 0
                  ? await survivalWorker.llm.generateCognitiveDecision('SurvivalDirector', dynamicPrompt, worldState, schemas, [])
                  : await survivalWorker.llm.generateGoalContinuationDecision(dynamicPrompt, sStepHistory, worldState, schemas, []);

              if (sIdx === 0 && (dec.activeGoal || (Array.isArray(dec.planSteps) && dec.planSteps.length > 0))) {
                sGoalPlanner.setMacroGoal(dec.activeGoal || campaignGoal.prompt, dec.planSteps);
              }

              if (!dec.toolCall?.name) {
                if (dec.goalCompleted && sIdx > 0) break;
                continue;
              }

              const toolName = dec.toolCall.name;
              const toolArgs = dec.toolCall.arguments || {};
              const dur = toolName.includes('construct') || toolName.includes('build') ? 45000 : 22000;
              const abortCtrl = new AbortController();
              const timer = setTimeout(() => abortCtrl.abort(), dur);
              let ok = false;
              let summary = '';

              const sub: SubtaskStatus = {
                id: `survival_${iter}_${sIdx}`,
                toolName,
                description: `Survival ${toolName}`,
                args: toolArgs,
                startedAt: Date.now(),
                estimatedDurationMs: 5000,
                progress: 0,
                status: 'RUNNING',
              };

              let hardTimer: ReturnType<typeof setTimeout> | null = null;
              try {
                const rpcPromise = survivalWorker.mcpServer.handleRpcRequest(
                  { jsonrpc: '2.0', id: sub.id, method: 'tools/call', params: { name: toolName, arguments: toolArgs } },
                  () => survivalWorker!.system1.createToolExecutionContext(sub, async () => {}, abortCtrl.signal)
                );
                const hardTimeout = new Promise<any>((resolve) => {
                  hardTimer = setTimeout(() => {
                    try { abortCtrl.abort(); } catch {}
                    resolve({ error: { code: -32000, message: `Survival tool execution hard timeout exceeded (${dur}ms)` } });
                  }, dur + 4000);
                });
                const rpcRes = await Promise.race([rpcPromise, hardTimeout]);
                const payloadText = rpcRes.result?.content?.[0]?.text;
                if (payloadText) {
                  try {
                    const p = JSON.parse(payloadText);
                    ok = Boolean(p.success);
                    summary = String(p.summary || '');
                  } catch {
                    ok = !rpcRes.error;
                    summary = payloadText;
                  }
                } else if (rpcRes.error) {
                  ok = false;
                  summary = String(rpcRes.error.message || 'Survival step timeout');
                }
              } finally {
                if (hardTimer) clearTimeout(hardTimer);
                clearTimeout(timer);
                try {
                  sBot.deactivateItem?.();
                } catch {}
                try {
                  sBot.pathfinder?.setGoal(null);
                  sBot.clearControlStates?.();
                } catch {}
              }

              sStepHistory.push({ stepNumber: sIdx + 1, toolName, args: toolArgs, success: ok, summary });
              sGoalPlanner.advanceOrReplan(
                { toolName, args: toolArgs, success: ok, summary },
                null,
                survivalWorker.system1.getLatestWorldState()
              );

              const invNames = (sBot.inventory?.items() || []).map((i: any) => i.name.toLowerCase());
              const nextEval = evaluateSurvivalAdvancements(invNames, sBlocksPlaced, sKills);
              const newAdvs = nextEval.unlocked.filter((a) => !persistentUnlockedAdvancements.has(a));
              for (const a of newAdvs) persistentUnlockedAdvancements.add(a);
              const rewardDelta = newAdvs.length * 40 + (ok ? 5 : -2);
              finalEval = nextEval;

              sExecutedTools.push({
                name: toolName,
                arguments: toolArgs,
                args: toolArgs,
                thought: dec.thought || '',
                planSteps: Array.isArray(dec.planSteps) ? dec.planSteps : undefined,
                success: ok,
                summary,
                stepReward: rewardDelta,
                systemPrompt: preStepSystemPrompt,
              });

              if (preState) {
                await stateWriteMutex.runExclusive(async () => {
                  fs.appendFileSync(
                    transitionsPath,
                    JSON.stringify({
                      provider: 'local_gpu',
                      taskId: `campaign_${campaignGoal.id}`,
                      workerId: 99,
                      stepIdx: sIdx,
                      toolName,
                      voxels: preState.voxels,
                      entities: preState.entities,
                      botState: preState.botState,
                      actionIdx: mapToolToDiscreteActionIndex(toolName),
                      aimDelta: [0, 0],
                      reward: rewardDelta,
                    }) + '\n',
                    'utf-8'
                  );
                });
              }

              // Anchor trajectory into replay buffer whenever a new Advancement is unlocked
              if (newAdvs.length > 0 && sExecutedTools.some((t) => t.success)) {
                await stateWriteMutex.runExclusive(async () => {
                  fs.appendFileSync(
                    replayPath,
                    JSON.stringify({
                      timestamp: new Date().toISOString(),
                      iteration: iter,
                      provider: 'open_world_survival',
                      taskId: `campaign_${campaignGoal.id}`,
                      workerId: 99,
                      botUsername: survivalWorker!.botUsername,
                      domain: 'PROGRESSION_DRAGON',
                      tier: 4,
                      prompt: campaignGoal.prompt,
                      passed: true,
                      score: Math.max(75, finalEval.score),
                      rawReward: Math.max(75, finalEval.score),
                      health: sBot.health ?? 20,
                      food: sBot.food ?? 20,
                      deaths: sDeaths,
                      executedTools: sExecutedTools.filter((t) => t.success),
                    }) + '\n',
                    'utf-8'
                  );
                });
                console.log(
                  `  [Survival Campaign] 🌟 Unlocked Advancement(s): [${newAdvs.join(', ')}] | Goal: ${campaignGoal.title} | Tier: ${finalEval.techTier}`
                );
              }
            }
          } finally {
            sBot.removeListener('death', onSDeath);
            sBot.removeListener('blockUpdate', onSBlock);
            sBot.removeListener('entityDead', onSKill);
          }

          console.log(
            `  [Survival Worker] Campaign "${campaignGoal.title}" (${sStepHistory.length} steps) | Tier: ${finalEval.techTier} | Advancements: ${finalEval.unlocked.length} (+${Math.max(0, persistentUnlockedAdvancements.size - bestAdvCount)} new)`
          );
          return {
            goalId: campaignGoal.id,
            goalTitle: campaignGoal.title,
            techTierReached: finalEval.techTier,
            advancementsUnlocked: Array.from(persistentUnlockedAdvancements),
            survivalSteps: sStepHistory.length,
            score: finalEval.score,
          };
        } catch (survErr: any) {
          console.warn(`[Survival Worker] Campaign step warning: ${survErr?.message || survErr}`);
          return undefined;
        }
      })();

      const workerPromises = workers.map(async (worker) => {
        while (!isShuttingDown) {
          const elapsedHours = (Date.now() - sessionStartTime) / (1000 * 60 * 60);
          if (elapsedHours >= maxHours) {
            console.log(`[TrainLoop] Worker #${worker.id} reached session target runtime of ${maxHours} hour(s) (${elapsedHours.toFixed(2)}h elapsed). Stopping cleanly.`);
            break;
          }

          const next = taskQueue.dequeue();
          if (!next) break;

          const { item: task, index: taskIndex } = next;
          const taskT0 = Date.now();
          const shardedTask = createShardedTaskSpec(task, worker.id, concurrency);

          // Verify worker bot connection and server readiness under serverLifecycleMutex
          await serverLifecycleMutex.runExclusive(async () => {
            if (!server.isReady || !worker.system1.bot || !worker.system1.bot.entity || (worker.system1.bot as any)._client?.ended) {
              console.log(`[TrainLoop] Ensuring Minecraft server & ${worker.botUsername} are connected on port ${serverPort}...`);
              try { worker.system1.bot?.quit(); } catch {}
              if (!server.isReady) {
                console.log(`[TrainLoop] Minecraft server process offline. Restarting headless server on port ${serverPort}...`);
                try { await server.stopServer(); } catch {}
                await server.startServer(serverPort);
                for (const w of workers) {
                  try { w.system1.bot?.quit(); } catch {}
                  await w.system1.connect('127.0.0.1', serverPort, w.botUsername);
                  await new Promise((r) => setTimeout(r, 600));
                }
                if (survivalWorker) {
                  try {
                    survivalWorker.system1.bot?.quit();
                    await survivalWorker.system1.connect('127.0.0.1', serverPort, survivalWorker.botUsername);
                  } catch {}
                }
              } else {
                try {
                  await worker.system1.connect('127.0.0.1', serverPort, worker.botUsername);
                  await new Promise((r) => setTimeout(r, 800));
                } catch (cErr: any) {
                  console.warn(`[TrainLoop] Reconnect attempt failed (${cErr?.message || cErr}). Performing clean server restart...`);
                  try { await server.stopServer(); } catch {}
                  await server.startServer(serverPort);
                  for (const w of workers) {
                    try { w.system1.bot?.quit(); } catch {}
                    await w.system1.connect('127.0.0.1', serverPort, w.botUsername);
                    await new Promise((r) => setTimeout(r, 600));
                  }
                  if (survivalWorker) {
                    try {
                      survivalWorker.system1.bot?.quit();
                      await survivalWorker.system1.connect('127.0.0.1', serverPort, survivalWorker.botUsername);
                    } catch {}
                  }
                }
              }
            }
          });

          worker.sparringPartner.setMode({ mode: 'idle' });
          worker.sparringPartner.resetEpisodeCounters();
          if (!worker.sparringPartner.bot || !worker.sparringPartner.bot.entity || (worker.sparringPartner.bot as any)._client?.ended) {
            try {
              await worker.sparringPartner.disconnect();
              await worker.sparringPartner.connect('127.0.0.1', serverPort);
            } catch {}
          }

          // Atomically reset arena commands on live server for this worker's sharded coordinate box
          await arenaResetMutex.runExclusive(async () => {
            await server.resetArenaOnLiveServer(
              shardedTask,
              worker.botUsername,
              task.sparringConfig
                ? {
                    username: worker.sparringPartner.username,
                    mode: 'active',
                    spawnOffset: task.sparringConfig.spawnOffset,
                    loadout: task.sparringConfig.loadout,
                  }
                : {
                    username: worker.sparringPartner.username,
                    mode: 'parked',
                  },
              concurrency > 1
            );
          });

          // Synchronize Step-0 packets (teleport, terrain chunk, initial loadout, sparring entity) and invalidate stale scan cache
          const postResetBot = worker.system1.bot;
          const postOriginVec = new Vec3(shardedTask.origin.x, shardedTask.origin.y, shardedTask.origin.z);
          const syncStart = Date.now();
          while (Date.now() - syncStart < 1500) {
            const pNow = postResetBot?.entity?.position;
            const posOk = Boolean(pNow && pNow.distanceTo(postOriginVec) <= 4.0);
            const floorBlk = postResetBot?.blockAt?.(postOriginVec.offset(0, -1, 0));
            const floorOk = Boolean(floorBlk && floorBlk.name !== 'void_air');
            const needLoadout = Array.isArray(task.initialLoadout) && task.initialLoadout.length > 0;
            const invOk = !needLoadout || (postResetBot?.inventory?.items?.()?.length || 0) > 0;
            const oppOk =
              !task.sparringConfig ||
              Object.values(postResetBot?.entities || {}).some(
                (e: any) =>
                  e &&
                  e.username &&
                  e.username.toLowerCase() === worker.sparringPartner.username.toLowerCase()
              );
            if (posOk && floorOk && invOk && oppOk) break;
            await new Promise((r) => setTimeout(r, 50));
          }
          worker.system1.invalidatePerceptionCache();

          // Sparring partner mode is activated at step 0 so both agents engage simultaneously
          worker.sparringPartner.setMode({ mode: 'idle' });

          const bot = worker.system1.bot;
          const originVec = new Vec3(shardedTask.origin.x, shardedTask.origin.y, shardedTask.origin.z);
          const countEntitiesByName = (targetName: string): number => {
            const maxDist = targetName === 'ender_dragon' ? 512 : 48;
            return Object.values(bot.entities || {}).filter((e: any) => {
              if (!e || e === bot.entity || !e.position || e.isValid === false) return false;
              const n = String(e.name || e.displayName || '').toLowerCase();
              return n === targetName && e.position.distanceTo(originVec) <= maxDist;
            }).length;
          };

          const initialCreepers = Math.max(
            task.id === 'combat_creeper_kiting' ? 2 : 0,
            countEntitiesByName('creeper')
          );
          const initialSkeletons = Math.max(
            task.id === 'combat_skeleton_duel' ? 1 : 0,
            countEntitiesByName('skeleton')
          );
          const initialSpiders = Math.max(
            task.id === 'combat_spider_swarm' ? 5 : 0,
            countEntitiesByName('spider')
          );
          const initialEndermen = countEntitiesByName('enderman');
          const initialCrystals = countEntitiesByName('end_crystal');
          const initialDragons = countEntitiesByName('ender_dragon');

          let deaths = 0;
          let creeperBlasts = 0;
          let blocksPlaced = 0;
          let blocksMined = 0;
          let arrowsFired = 0;
          let leversFlipped = 0;
          let wolvesTamed = 0;
          let dragonsKilled = 0;
          let foodConsumed = 0;
          let affordanceActions = 0;
          let damageDealt = 0;
          let waterPlaced = 0;
          let ironPickaxeDurabilityUsedOnStone = 0;
          let activeStepAbortController: AbortController | null = null;
          const killedMobsMap: Record<string, number> = {};

          const targetEntityName = String((task as any).targetObject || '').toLowerCase();
          const initialTargetEntities = targetEntityName ? countEntitiesByName(targetEntityName) : 0;

          const onDeath = () => {
            deaths++;
            worker.sparringPartner.recordAgentDeath();
            try {
              activeStepAbortController?.abort();
            } catch {}
            try {
              server.sendCommand(`tp ${worker.botUsername} ${shardedTask.origin.x} ${shardedTask.origin.y} ${shardedTask.origin.z} 0 0`);
            } catch {}
          };
          const onEntityDead = (ent: any) => {
            const n = String(ent?.name || ent?.displayName || '').toLowerCase();
            if (n && !n.startsWith('neuroagent')) {
              killedMobsMap[n] = (killedMobsMap[n] || 0) + 1;
            }
            if (n === 'ender_dragon') dragonsKilled++;
          };
          const onEntityGone = (_ent: any) => {};
          const onExplosion = () => {
            if (task.id === 'combat_creeper_kiting') creeperBlasts++;
          };
          const onDiggingCompleted = (b: any) => {
            blocksMined++;
            if (b?.name === 'stone' && bot.heldItem?.name === 'iron_pickaxe') {
              ironPickaxeDurabilityUsedOnStone++;
            }
          };
          const isBlockPowered = (blk: any): boolean => {
            const props = typeof blk?.getProperties === 'function' ? blk.getProperties() : {};
            if (props && 'powered' in props) {
              return props.powered === true || props.powered === 'true';
            }
            return ((blk?.metadata ?? 0) & 8) !== 0;
          };
          const onBlockUpdate = (oldBlock: any, newBlock: any) => {
            if (!oldBlock || !newBlock || !bot.entity) return;
            if (
              (oldBlock.name === 'air' || oldBlock.name === 'lava' || oldBlock.name === 'water') &&
              newBlock.name !== 'air' &&
              newBlock.name !== 'lava' &&
              newBlock.name !== 'water' &&
              newBlock.position?.distanceTo(bot.entity.position) <= 6.0
            ) {
              blocksPlaced++;
            }
            if (newBlock.name === 'lever' && isBlockPowered(newBlock)) {
              leversFlipped++;
            }
            if (newBlock.name === 'water' && newBlock.position?.distanceTo(bot.entity.position) <= 8.0) {
              waterPlaced++;
            }
            if (newBlock.position?.distanceTo(bot.entity.position) <= 6.0) {
              affordanceActions++;
            }
          };
          const onEntityHurt = (ent: any) => {
            if (!ent || ent === bot.entity || !bot.entity) return;
            const recentAttack = Date.now() - ((bot as any)._lastAttackAt || 0) <= 650;
            if (recentAttack && ent.position?.distanceTo(bot.entity.position) <= 6.5) {
              damageDealt += 4;
            }
          };

          bot.on('death', onDeath);
          bot.on('entityDead', onEntityDead);
          bot.on('entityGone', onEntityGone);
          bot._client?.on('explosion', onExplosion);
          bot.on('diggingCompleted', onDiggingCompleted);
          bot.on('blockUpdate', onBlockUpdate);
          bot.on('entityHurt', onEntityHurt);

          const buildSnapshot = (): MinebenchEvaluationSnapshot => {
            const aliveCreepers = countEntitiesByName('creeper');
            const aliveSkeletons = countEntitiesByName('skeleton');
            const aliveSpiders = countEntitiesByName('spider');
            const aliveEndermen = countEntitiesByName('enderman');
            const aliveCrystals = countEntitiesByName('end_crystal');
            const aliveDragons = countEntitiesByName('ender_dragon');

            const leverPos = new Vec3(shardedTask.origin.x + 3, shardedTask.origin.y + 1, shardedTask.origin.z - 1);
            const leverBlock = bot.blockAt(leverPos);
            if (leverBlock && leverBlock.name === 'lever' && isBlockPowered(leverBlock)) {
              leversFlipped = Math.max(1, leversFlipped);
            }

            const wolves = Object.values(bot.entities || {}).filter(
              (e: any) => e && String(e.name || '').toLowerCase() === 'wolf'
            );
            for (const w of wolves as any[]) {
              const flags = Number(w.metadata?.[17] ?? 0);
              if ((flags & 0x04) !== 0) wolvesTamed = Math.max(1, wolvesTamed);
            }

            const invItems = (bot.inventory?.items() || []).map((it: any) => ({
              name: it.name,
              count: it.count,
            }));

            const getSlotName = (slotIdx: number): string | null => {
              const s = bot.inventory?.slots?.[slotIdx];
              return s ? s.name : null;
            };

            return {
              botPosition: {
                x: bot.entity?.position?.x ?? shardedTask.origin.x,
                y: bot.entity?.position?.y ?? shardedTask.origin.y,
                z: bot.entity?.position?.z ?? shardedTask.origin.z,
              },
              health: bot.health ?? 20,
              food: bot.food ?? 20,
              deaths,
              inventory: invItems,
              equippedArmor: {
                head: getSlotName(5),
                torso: getSlotName(6),
                legs: getSlotName(7),
                feet: getSlotName(8),
                offHand: getSlotName(45),
              },
              blockAt: (x: number, y: number, z: number) => {
                const b = bot.blockAt(new Vec3(x, y, z));
                if (!b) return null;
                return { name: b.name, metadata: b.metadata };
              },
              entities: Object.values(bot.entities || {})
                .filter((e: any) => e && e !== bot.entity && e.position)
                .map((e: any) => ({
                  id: e.id,
                  name: String(e.name || e.displayName || '').toLowerCase(),
                  type: String(e.type || ''),
                  position: { x: e.position.x, y: e.position.y, z: e.position.z },
                  isValid: e.isValid !== false,
                  metadata: e.metadata,
                })),
              stats: {
                blocksPlaced,
                blocksMined,
                mobsKilled: {
                  ...killedMobsMap,
                },
                creeperBlasts,
                arrowsFired,
                endCrystalsDestroyed: initialCrystals > 0 ? Math.max(0, initialCrystals - aliveCrystals) : 0,
                enderDragonSlain: dragonsKilled > 0 && deaths === 0,
                leversFlipped,
                wolvesTamed,
                foodConsumed,
                affordanceActions,
                damageDealt,
                waterPlaced,
                ironPickaxeDurabilityUsedOnStone,
              },
              multiplayer: task.sparringConfig ? worker.sparringPartner.getSnapshot() : undefined,
            };
          };

          const stepHistory: ExecutedStepRecord[] = [];
          const executedTools: any[] = [];
          const episodeTransitions: any[] = [];
          const episodeMotorChunks: any[] = [];
          const goalPlanner = new HierarchicalGoalPlanner();
          goalPlanner.setMacroGoal(task.prompt);
          let activeGoal = task.prompt;
          let lastEffectivePrompt = task.prompt;

          try {
            for (let stepIdx = 0; stepIdx < task.maxSteps; stepIdx++) {
              if (isShuttingDown || Date.now() - taskT0 > task.maxDurationSeconds * 1000) break;

              if (stepIdx === 0 && task.sparringConfig) {
                worker.sparringPartner.setMode({
                  mode: task.sparringConfig.mode,
                  targetUsername: worker.botUsername,
                  goalPos: task.sparringConfig.goalOffset
                    ? {
                        x: shardedTask.origin.x + task.sparringConfig.goalOffset.dx,
                        y: shardedTask.origin.y + task.sparringConfig.goalOffset.dy,
                        z: shardedTask.origin.z + task.sparringConfig.goalOffset.dz,
                      }
                    : undefined,
                  targetBlockName: task.sparringConfig.targetBlockName,
                });
              }

              const preRlState = sampleLiveRLTransitionState(worker.system1);
              const preSnap = buildSnapshot();
              const preGrading = evaluateMinebenchEpisode(shardedTask, preSnap);
              if (stepIdx > 0 && preGrading.passed) break;

              const worldState = worker.system1.getLatestWorldState();
              const schemas = worker.registry.listSchemas();
              const preStepSystemPrompt = worker.llm.buildGroundedTrainingSystemPrompt(
                schemas,
                worldState,
                stepIdx === 0 ? undefined : stepHistory
              );

              const variants = PROMPT_VARIANTS[task.id];
              let effectivePrompt =
                variants && variants.length > 0
                  ? variants[Math.floor(Math.random() * variants.length)]
                  : task.prompt;

              effectivePrompt = effectivePrompt
                .replace(/\bDQN_Opponent\b/g, worker.sparringPartner.username)
                .replace(/\bMirrorAgent\b/g, worker.sparringPartner.username)
                .replace(/\bCompanion_Player\b/g, worker.sparringPartner.username);
              lastEffectivePrompt = effectivePrompt;

              const continuationGoal = goalPlanner.formatGoalHorizonContext(worldState);
              const isCoop = task.domain === 'MULTIPLAYER_ARENA' && task.id.startsWith('coop_');
              const speaker = isCoop ? worker.sparringPartner.username : 'Trainer';
              const decision =
                stepIdx === 0
                  ? await worker.llm.generateCognitiveDecision(speaker, effectivePrompt, worldState, schemas, [])
                  : await worker.llm.generateGoalContinuationDecision(continuationGoal, stepHistory, worldState, schemas, []);

              if (decision.activeGoal) activeGoal = decision.activeGoal;
              if (stepIdx === 0 && (decision.activeGoal || (Array.isArray(decision.planSteps) && decision.planSteps.length > 0))) {
                goalPlanner.setMacroGoal(decision.activeGoal || effectivePrompt, decision.planSteps);
              }

              if (!decision.toolCall || !decision.toolCall.name) {
                if (decision.goalCompleted && stepIdx > 0) break;
                continue;
              }

              // Fetch real System 2 latent h (1536-dim) for passive System 1 20Hz motor chunk logging
              let stepLatentH: number[] | null = null;
              try {
                const latRes = await fetch(
                  `http://127.0.0.1:8765/vla/latest_latent?bot_id=${encodeURIComponent(worker.botUsername)}`,
                  { signal: AbortSignal.timeout(1000) }
                );
                if (latRes.ok) {
                  const latData: any = await latRes.json();
                  if (Array.isArray(latData?.latent_h) && latData.latent_h.length === 1536) {
                    stepLatentH = latData.latent_h;
                  }
                }
              } catch {}

              const toolName = decision.toolCall.name;
              const toolArgs = decision.toolCall.arguments || {};
              const isBuildTool = toolName.includes('construct') || toolName.includes('build') || toolName.includes('house');
              const isSmeltTool = toolName === 'smelt_item' || toolName === 'smelt_ores';
              const stepDurationMs = isBuildTool ? 55000 : isSmeltTool ? 40000 : 25000;
              const abortController = new AbortController();
              activeStepAbortController = abortController;
              const stepTimeout = setTimeout(() => abortController.abort(), stepDurationMs);

              const subtask: SubtaskStatus = {
                id: `${task.id}_w${worker.id}_${stepIdx}`,
                toolName,
                description: `Executing ${toolName}`,
                args: toolArgs,
                startedAt: Date.now(),
                estimatedDurationMs: 4000,
                progress: 0,
                status: 'RUNNING',
              };

              let stepSuccess = false;
              let stepSummary = '';
              let stepReward = 0;

              // Shadow 20Hz 8-tick physics sampler during live execution (up to 2 chunks/step, non-intrusive)
              const pendingStepChunks: any[] = [];
              let activeChunkTicks: any[] = [];
              let activeChunkPre: {
                voxels: number[];
                entities: number[][];
                proprio24: number[];
                baseYaw: number;
                basePitch: number;
              } | null = null;
              let motorSampleInterval: ReturnType<typeof setInterval> | null = null;

              const flushShadowChunk = () => {
                if (!activeChunkPre || activeChunkTicks.length < 4 || pendingStepChunks.length >= 2) {
                  activeChunkTicks = [];
                  activeChunkPre = null;
                  return;
                }
                while (activeChunkTicks.length < 8) {
                  activeChunkTicks.push({ ...activeChunkTicks[activeChunkTicks.length - 1] });
                }
                const ticks = activeChunkTicks.slice(0, 8);
                pendingStepChunks.push({
                  taskId: task.id,
                  toolName,
                  latent_h: stepLatentH,
                  proprio24: activeChunkPre.proprio24,
                  voxels: activeChunkPre.voxels,
                  entities: activeChunkPre.entities,
                  aimTargets: ticks.map((t) => [t.dyawDeg, t.dpitchDeg]),
                  moveTargets: ticks.map((t) => [t.vx, t.vz]),
                  discreteTargets: ticks.map((t) => [t.jump, t.sprint, t.sneak, t.attack, t.useItem, t.swapSlot]),
                  extendedTargets: ticks.map((t) => [t.digBlock, t.dropItem]),
                });
                activeChunkTicks = [];
                activeChunkPre = null;
              };

              if (stepLatentH && preRlState) {
                const startYaw = bot.entity?.yaw || 0;
                const startPitch = bot.entity?.pitch || 0;
                const initProprio24 =
                  worker.system1.vlaClient?.extractProprioception?.(bot) || new Array(24).fill(0);
                activeChunkPre = {
                  voxels: preRlState.voxels,
                  entities: preRlState.entities,
                  proprio24: initProprio24.map((v: number) => Number(Number(v || 0).toFixed(4))),
                  baseYaw: startYaw,
                  basePitch: startPitch,
                };
                const wrapDeg = (rad: number) => {
                  let a = rad % (2 * Math.PI);
                  if (a > Math.PI) a -= 2 * Math.PI;
                  if (a < -Math.PI) a += 2 * Math.PI;
                  return Number(Math.max(-180, Math.min(180, (a * 180) / Math.PI)).toFixed(2));
                };
                motorSampleInterval = setInterval(() => {
                  try {
                    if (!bot?.entity || pendingStepChunks.length >= 2) return;
                    if (!activeChunkPre) {
                      const s = sampleLiveRLTransitionState(worker.system1);
                      if (!s) return;
                      const p24 = worker.system1.vlaClient?.extractProprioception?.(bot) || new Array(24).fill(0);
                      activeChunkPre = {
                        voxels: s.voxels,
                        entities: s.entities,
                        proprio24: p24.map((v: number) => Number(Number(v || 0).toFixed(4))),
                        baseYaw: bot.entity.yaw || 0,
                        basePitch: bot.entity.pitch || 0,
                      };
                    }
                    const cs: any = bot.controlState || {};
                    const nowMs = Date.now();
                    const dyawDeg = wrapDeg((bot.entity.yaw || 0) - activeChunkPre.baseYaw);
                    const dpitchDeg = Number(
                      Math.max(-90, Math.min(90, (((bot.entity.pitch || 0) - activeChunkPre.basePitch) * 180) / Math.PI)).toFixed(2)
                    );
                    activeChunkTicks.push({
                      dyawDeg,
                      dpitchDeg,
                      vx: (cs.right ? 1 : 0) - (cs.left ? 1 : 0),
                      vz: (cs.forward ? 1 : 0) - (cs.back ? 1 : 0),
                      jump: cs.jump ? 1 : 0,
                      sprint: cs.sprint ? 1 : 0,
                      sneak: cs.sneak ? 1 : 0,
                      attack: nowMs - ((bot as any)._lastAttackAt || 0) <= 120 ? 1 : 0,
                      useItem: (bot as any).usingHeldItem ? 1 : 0,
                      swapSlot: 0,
                      digBlock: (bot as any).targetDigBlock ? 1 : 0,
                      dropItem: 0,
                    });
                    if (activeChunkTicks.length >= 8) {
                      flushShadowChunk();
                    }
                  } catch {}
                }, 50);
              }

              try {
                const prevYaw = bot.entity?.yaw || 0;
                const prevPitch = bot.entity?.pitch || 0;

                const rpcPromise = worker.mcpServer.handleRpcRequest(
                  {
                    jsonrpc: '2.0',
                    id: subtask.id,
                    method: 'tools/call',
                    params: { name: toolName, arguments: toolArgs },
                  },
                  () => {
                    const c = worker.system1.createToolExecutionContext(subtask, async () => {}, abortController.signal);
                    (c as any).taskOrigin = shardedTask.origin;
                    return c;
                  }
                );
                let hardStepTimer: ReturnType<typeof setTimeout> | null = null;
                const hardStepTimeoutPromise = new Promise<{ result?: any; error?: { code: number; message: string } }>((resolve) => {
                  hardStepTimer = setTimeout(() => {
                    try { abortController.abort(); } catch {}
                    resolve({ error: { code: -32000, message: `Tool execution hard timeout exceeded (${stepDurationMs}ms)` } });
                  }, stepDurationMs + 4000);
                });

                const rpcRes = await Promise.race([rpcPromise, hardStepTimeoutPromise]);
                if (hardStepTimer) {
                  clearTimeout(hardStepTimer);
                  hardStepTimer = null;
                }

                const payloadText = rpcRes.result?.content?.[0]?.text;
                if (payloadText) {
                  try {
                    const parsed = JSON.parse(payloadText);
                    stepSuccess = Boolean(parsed.success);
                    stepSummary = String(parsed.summary || '');
                    if (toolName === 'tame_animal' && stepSuccess) wolvesTamed = Math.max(1, wolvesTamed);
                    if (toolName === 'shoot_bow' && parsed.data?.fired) arrowsFired += Number(parsed.data.fired) || 0;
                    if ((toolName === 'eat_food' || toolName === 'consume_item') && stepSuccess) foodConsumed++;
                    if (toolName === 'engage_combat' && parsed.data?.totalStrikes) {
                      damageDealt += Number(parsed.data.totalStrikes) * 4;
                    }
                  } catch {
                    stepSuccess = !rpcRes.error;
                    stepSummary = payloadText;
                  }
                } else if (rpcRes.error) {
                  stepSuccess = false;
                  stepSummary = rpcRes.error.message;
                }

                const postSnap = buildSnapshot();
                const postGrading = evaluateMinebenchEpisode(shardedTask, postSnap);
                stepReward = Number((postGrading.rawReward - preGrading.rawReward).toFixed(3));

                if (preRlState) {
                  const dYaw = Math.max(-0.35, Math.min(0.35, (bot.entity?.yaw || 0) - prevYaw));
                  const dPitch = Math.max(-0.35, Math.min(0.35, (bot.entity?.pitch || 0) - prevPitch));
                  const transitionRecord = {
                    provider: 'local_gpu',
                    taskId: task.id,
                    workerId: worker.id,
                    stepIdx,
                    toolName,
                    voxels: preRlState.voxels,
                    entities: preRlState.entities,
                    botState: preRlState.botState,
                    actionIdx: mapToolToDiscreteActionIndex(toolName),
                    aimDelta: [Number(dYaw.toFixed(4)), Number(dPitch.toFixed(4))],
                    reward: stepReward,
                  };
                  episodeTransitions.push(transitionRecord);
                }
              } finally {
                if (motorSampleInterval) {
                  clearInterval(motorSampleInterval);
                  motorSampleInterval = null;
                }
                flushShadowChunk();
                for (const mc of pendingStepChunks) {
                  episodeMotorChunks.push({
                    ...mc,
                    stepSuccess,
                    reward: stepReward,
                  });
                }
                clearTimeout(stepTimeout);
                activeStepAbortController = null;
                try {
                  bot.deactivateItem?.();
                } catch {}
                try {
                  bot.pathfinder?.setGoal(null);
                  bot.clearControlStates?.();
                } catch {}
              }

              const AFFORDANCE_TOOLS = new Set([
                'shear_sheep',
                'breed_animals',
                'use_tool_on_block',
                'light_block_fire',
                'loot_chest',
                'interact_block_or_puzzle',
                'till_and_farm',
                'tame_animal',
                'give_item_to_player',
                'store_items_in_chest',
                'retrieve_items_from_chest',
                'equip_armor_or_item',
                'smelt_item',
                'smelt_ores',
                'eat_food',
              ]);
              if (stepSuccess && AFFORDANCE_TOOLS.has(toolName)) {
                affordanceActions++;
              }
              executedTools.push({
                name: toolName,
                arguments: toolArgs,
                args: toolArgs,
                thought: decision.thought || '',
                planSteps: Array.isArray(decision.planSteps) ? decision.planSteps : undefined,
                success: stepSuccess,
                summary: stepSummary,
                stepReward,
                systemPrompt: preStepSystemPrompt,
              });
              stepHistory.push({
                stepNumber: stepIdx + 1,
                toolName,
                args: toolArgs,
                success: stepSuccess,
                summary: stepSummary,
              });
              goalPlanner.advanceOrReplan(
                { toolName, args: toolArgs, success: stepSuccess, summary: stepSummary },
                null,
                worker.system1.getLatestWorldState()
              );

              if (decision.goalCompleted && stepIdx > 0) break;
              if (deaths > 0) {
                console.log(`[TrainLoop] [${worker.botUsername}] Bot died during task ${task.id} (step ${stepIdx + 1}). Aborting remaining steps.`);
                break;
              }
            }
          } finally {
            bot.removeListener('death', onDeath);
            bot.removeListener('entityDead', onEntityDead);
            bot.removeListener('entityGone', onEntityGone);
            bot._client?.removeListener('explosion', onExplosion);
            bot.removeListener('diggingCompleted', onDiggingCompleted);
            bot.removeListener('blockUpdate', onBlockUpdate);
            bot.removeListener('entityHurt', onEntityHurt);
            worker.sparringPartner.setMode({ mode: 'idle' });
          }

          const finalSnap = buildSnapshot();
          const grading = evaluateMinebenchEpisode(shardedTask, finalSnap);

          // Query vanilla scoreboard values to log objective ground truth telemetry for this worker bot
          server.sendCommand(`scoreboard players get ${worker.botUsername} deaths`);
          server.sendCommand(`scoreboard players get ${worker.botUsername} mob_kills`);
          server.sendCommand(`scoreboard players get ${worker.botUsername} dmg_dealt`);
          server.sendCommand(`scoreboard players get ${worker.botUsername} dmg_taken`);
          server.sendCommand(`scoreboard players get ${worker.botUsername} dmg_blocked`);
          server.sendCommand(`scoreboard players get ${worker.botUsername} sprint_cm`);
          server.sendCommand(`scoreboard players get ${worker.botUsername} jump_cnt`);
          server.sendCommand(`scoreboard players get ${worker.botUsername} mine_log`);
          server.sendCommand(`scoreboard players get ${worker.botUsername} mine_stone`);
          server.sendCommand(`scoreboard players get ${worker.botUsername} mine_iron`);
          server.sendCommand(`scoreboard players get ${worker.botUsername} craft_bench`);
          server.sendCommand(`scoreboard players get ${worker.botUsername} craft_pick`);

          const taskIsHeldOut = isHeldOutEvalTask(task.id);
          await stateWriteMutex.runExclusive(async () => {
            state.totalEpisodesPlayed++;
            if (grading.passed) iterPassed++;
            iterScoreSum += grading.score;
            iterRewardSum += grading.rawReward;
            if (taskIsHeldOut) {
              heldOutTotal++;
              if (grading.passed) heldOutPassed++;
              heldOutScoreSum += grading.score;
            }

            const bucket = classifyDomainBucket(task);
            if (!domainStats[bucket]) domainStats[bucket] = { passed: 0, total: 0, scoreSum: 0 };
            domainStats[bucket].total++;
            if (grading.passed) domainStats[bucket].passed++;
            domainStats[bucket].scoreSum += grading.score;

            for (const tr of episodeTransitions) {
              fs.appendFileSync(transitionsPath, JSON.stringify(tr) + '\n', 'utf-8');
            }

            if (episodeMotorChunks.length > 0) {
              for (const mc of episodeMotorChunks) {
                fs.appendFileSync(
                  motorChunksPath,
                  JSON.stringify({ ...mc, episodePassed: grading.passed, episodeReward: grading.rawReward }) + '\n',
                  'utf-8'
                );
              }
            }

            const replayRecord = {
              timestamp: new Date().toISOString(),
              iteration: iter,
              provider: 'local_gpu',
              taskId: task.id,
              isHeldOut: taskIsHeldOut,
              workerId: worker.id,
              botUsername: worker.botUsername,
              domain: task.domain,
              tier: task.tier,
              prompt: lastEffectivePrompt,
              passed: grading.passed,
              score: grading.score,
              rawReward: grading.rawReward,
              health: finalSnap.health,
              food: finalSnap.food,
              deaths: finalSnap.deaths,
              executedTools,
            };
            fs.appendFileSync(replayPath, JSON.stringify(replayRecord) + '\n', 'utf-8');
          });

          const badge = grading.passed ? 'PASS' : 'FAIL';
          const prefix = concurrency > 1 ? `[Worker #${worker.id}] ` : '';
          console.log(
            `  [${badge}] ${prefix}${task.id.padEnd(28)} | Score: ${String(grading.score).padStart(3)}/100 | R: ${String(
              grading.rawReward
            ).padStart(5)} | ${executedTools.map((t) => t.name).join(' -> ') || 'none'}`
          );
        }
      });

      await Promise.all(workerPromises);
      testbenchFinished = true;
      const survivalCampaignRes = await survivalPromise;

      if (isShuttingDown) break;

      // Compute domain-decomposed skill breakdown for this iteration
      const domainBreakdown: Record<
        string,
        { passed: number; total: number; passRate: number; meanScore: number }
      > = {};
      for (const [dom, st] of Object.entries(domainStats)) {
        domainBreakdown[dom] = {
          passed: st.passed,
          total: st.total,
          passRate: st.total > 0 ? Number(((st.passed / st.total) * 100).toFixed(1)) : 0,
          meanScore: st.total > 0 ? Number((st.scoreSum / st.total).toFixed(1)) : 0,
        };
      }

      // Iteration summary calculation across all sharded episodes
      const iterPassRate = Number(((iterPassed / numTasks) * 100).toFixed(1));
      const iterMeanScore = Number((iterScoreSum / numTasks).toFixed(1));
      const iterTotalReward = Number(iterRewardSum.toFixed(1));

      state.latestMeanScore = iterMeanScore;
      state.latestPassRate = iterPassRate;
      state.latestReward = iterTotalReward;

      const isNewBestScore = iterMeanScore > state.bestMeanScore;
      const isNewBestPass = iterPassRate > state.bestPassRate;
      if (isNewBestScore) {
        state.bestMeanScore = iterMeanScore;
        state.bestIteration = iter;
      }
      if (isNewBestPass) {
        state.bestPassRate = iterPassRate;
      }
      const isNewBest = isNewBestScore || isNewBestPass;

      // 1. Trigger RTX 5060 PPO + GRPO gradient update for System 1 Motor Reflex
      const venvPython = path.resolve(process.cwd(), 'neural_lab', '.venv', 'Scripts', 'python.exe');
      const rlScript = path.resolve(process.cwd(), 'minebench', 'rl_trainer.py');
      const slmScript = path.resolve(process.cwd(), 'minebench', 'slm_trainer.py');
      const actScript = path.resolve(process.cwd(), 'neural_lab', 'train_action_head.py');
      let rlLossDelta = '';
      let slmLossDelta = '';
      let actLossDelta = '';
      let actAimErrorDeg: number | undefined;
      let actDiscreteAccPct: number | undefined;

      const isFinalCyclePlanned = cyclesCompleted + 1 >= maxCycles;

      if (!isEvalOnly) {
        console.log(`[TrainLoop] Cleanly pausing bot connection(s) during gradient backpropagation...`);
        for (const w of workers) {
          try {
            await w.sparringPartner.disconnect().catch(() => {});
          } catch {}
          try {
            w.system1.bot?.quit();
          } catch {}
        }
        try {
          survivalWorker?.system1.bot?.quit();
        } catch {}
        await new Promise((r) => setTimeout(r, 600));

        // 1. Trigger RTX 5060 PPO + GRPO gradient update for System 1 Motor Reflex
        if (fs.existsSync(venvPython) && fs.existsSync(rlScript)) {
          try {
            const pyArgs = [rlScript, '--epochs', '8', '--loop', String(iter)];
            if (isNewBest) pyArgs.push('--is-best');
            const stdout = await runPythonScriptAsync(venvPython, pyArgs, 90000);
            const parsed = JSON.parse(stdout.trim().split('\n').pop() || '{}');
            rlLossDelta = `PPO: ${parsed.initialLoss} -> ${parsed.finalLoss} (${parsed.transitionsPerSecond} trans/s)`;
          } catch (err: any) {
            rlLossDelta = `RL note: ${err.message}`;
          }
        }

        try {
          // 2. Trigger RTX 5060 System 2 Cognitive SLM Hybrid DPO+NLL QLoRA Fine-Tuning ('Minecraft Savant')
          if (fs.existsSync(venvPython) && fs.existsSync(slmScript)) {
            try {
              const slmLr = iter <= 15 ? '1.5e-4' : iter <= 50 ? '1e-4' : '5e-5';
              const slmArgs = [
                slmScript,
                '--epochs',
                '1',
                '--lr',
                slmLr,
                '--dpo-beta',
                '0.1',
                '--nll-weight',
                '0.3',
                '--label-smoothing',
                '0.04',
                '--loop',
                String(iter),
                '--max-trajectories',
                '256',
                '--batch-size',
                '1',
                '--grad-accum',
                '4',
                '--max-length',
                '1280',
              ];
              if (isNewBest) slmArgs.push('--is-best');

              // Offload daemon's SLM to CPU to free VRAM for trainer
              try {
                const pauseRes = await fetch('http://127.0.0.1:8765/pause_slm', {
                  method: 'POST',
                  headers: { 'Connection': 'close' },
                  signal: AbortSignal.timeout(30000),
                });
                await pauseRes.json();
                await new Promise((r) => setTimeout(r, 2000));
              } catch {}

              const slmOut = await runPythonScriptAsync(venvPython, slmArgs, 360000);
              const slmMetricsPath = path.resolve(resultsDir, 'slm_training_metrics.json');
              if (fs.existsSync(slmMetricsPath)) {
                const parsedSlm = JSON.parse(fs.readFileSync(slmMetricsPath, 'utf-8'));
                slmLossDelta = `SLM LoRA: ${parsedSlm.initialLoss} -> ${parsedSlm.finalLoss} (${parsedSlm.samplesTrained} trajs in ${parsedSlm.trainingDurationSeconds}s)`;
              } else {
                const parsedSlm = JSON.parse(slmOut.trim().split('\n').pop() || '{}');
                slmLossDelta = `SLM LoRA: ${parsedSlm.initialLoss} -> ${parsedSlm.finalLoss} (${parsedSlm.samplesTrained} trajs in ${parsedSlm.trainingDurationSeconds}s)`;
              }
            } catch (err: any) {
              slmLossDelta = `SLM note: ${err.message}`;
            }
          }

          // 3. Trigger RTX 5060 80.8M Temporal Action Chunking Transformer (ACT) Fine-Tuning
          if (fs.existsSync(venvPython) && fs.existsSync(actScript)) {
            try {
              const actArgs = [actScript, '--epochs', '4', '--loop', String(iter)];
              if (isNewBest) actArgs.push('--is-best');
              const actOut = await runPythonScriptAsync(venvPython, actArgs, 180000);
              const parsedAct = JSON.parse(actOut.trim().split('\n').pop() || '{}');
              actAimErrorDeg = typeof parsedAct.aim_error_deg === 'number' ? parsedAct.aim_error_deg : undefined;
              actDiscreteAccPct =
                typeof parsedAct.discrete_accuracy_pct === 'number' ? parsedAct.discrete_accuracy_pct : undefined;
              actLossDelta = `ACT (80.8M): ${parsedAct.initial_loss} -> ${parsedAct.final_loss} (aim err: ${parsedAct.aim_error_deg}°, disc: ${parsedAct.discrete_accuracy_pct}%)`;
            } catch (err: any) {
              actLossDelta = `ACT note: ${err.message}`;
            }
          }
        } finally {
          // Hot-reload updated adapter, action head, and reflex policy in running neural daemon (restoring SLM to GPU)
          let reloaded = false;
          for (let attempt = 1; attempt <= 3; attempt++) {
            try {
              const reloadRes = await fetch('http://127.0.0.1:8765/reload_adapter', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Connection': 'close' },
                body: JSON.stringify({ adapter_name: 'slm_minecraft_savant_latest' }),
                signal: AbortSignal.timeout(90000),
              });
              if (reloadRes.ok) {
                const rData: any = await reloadRes.json();
                console.log(`[TrainLoop] Neural adapter & SLM GPU state successfully restored (attempt #${attempt}, device=${rData.slm_device || 'unknown'})`);
                reloaded = true;
                break;
              } else {
                console.warn(`[TrainLoop] Warning: reload_adapter returned HTTP ${reloadRes.status} (attempt #${attempt})`);
              }
            } catch (rErr: any) {
              console.warn(`[TrainLoop] Warning: reload_adapter fetch error on attempt #${attempt}: ${rErr?.message || rErr}`);
            }
            await new Promise((r) => setTimeout(r, 1000 * attempt));
          }
          if (!reloaded) {
            console.error(`[TrainLoop] ERROR: Could not confirm reload_adapter after 3 attempts! Respawning neural_daemon.py on a fresh CUDA context...`);
            try {
              await fetch('http://127.0.0.1:8765/shutdown', { method: 'POST', signal: AbortSignal.timeout(2000) }).catch(() => {});
              await new Promise((r) => setTimeout(r, 1500));
              forceKillPid(spawnedDaemonProc?.pid);
              await new Promise((r) => setTimeout(r, 500));
              const daemonScript = path.resolve(process.cwd(), 'neural_lab', 'neural_daemon.py');
              const freshDaemon = spawn(venvPython, [daemonScript], {
                detached: true,
                stdio: 'ignore',
                cwd: process.cwd(),
              });
              spawnedDaemonProc = freshDaemon;
              freshDaemon.unref();
              let respawnHealthy = false;
              for (let rWait = 0; rWait < 35; rWait++) {
                await new Promise((r) => setTimeout(r, 1000));
                try {
                  const hCheck = await fetch('http://127.0.0.1:8765/health', { signal: AbortSignal.timeout(1500) });
                  if (hCheck.ok) {
                    respawnHealthy = true;
                    break;
                  }
                } catch {}
              }
              if (respawnHealthy) {
                console.log(`[TrainLoop] Fresh neural_daemon.py respawned and healthy on CUDA!`);
              } else {
                console.error(`[TrainLoop] ERROR: Respawned neural_daemon.py did not report healthy within 35s.`);
              }
            } catch (respawnErr: any) {
              console.error(`[TrainLoop] ERROR during neural_daemon.py respawn: ${respawnErr?.message || respawnErr}`);
            }
          }
        }

        // Keep JSONL buffers bounded so 500+ iteration sustained runs never hit disk or parse-timeout ceilings
        trimJsonlFileIfTooLarge(motorChunksPath, 80 * 1024 * 1024, 5000);
        trimJsonlFileIfTooLarge(transitionsPath, 50 * 1024 * 1024, 6000);
        trimJsonlFileIfTooLarge(replayPath, 40 * 1024 * 1024, 3500);

        if (!isFinalCyclePlanned && !isShuttingDown) {
          // Reconnect worker bots and sparring partners cleanly to live server after backprop and adapter reload
          console.log(`[TrainLoop] Reconnecting ${concurrency} worker bot(s) cleanly to server on port ${serverPort}...`);
          for (const w of workers) {
            try {
              await w.system1.connect('127.0.0.1', serverPort, w.botUsername);
              await new Promise((r) => setTimeout(r, 600));
            } catch (err: any) {
              console.warn(`[TrainLoop] Bot reconnect warning for ${w.botUsername}: ${err?.message || err}`);
            }
            try {
              await w.sparringPartner.connect('127.0.0.1', serverPort);
              await new Promise((r) => setTimeout(r, 300));
            } catch (err: any) {
              console.warn(`[TrainLoop] Sparring partner reconnect warning for ${w.sparringPartner.username}: ${err?.message || err}`);
            }
          }
          if (survivalWorker) {
            try {
              await survivalWorker.system1.connect('127.0.0.1', serverPort, survivalWorker.botUsername);
              await new Promise((r) => setTimeout(r, 400));
            } catch {}
          }
        }
      } else {
        console.log(`  [EvalOnly Mode] Skipping gradient backpropagation & training updates.`);
      }

      // Check for Periodic or Session-End Full Unassisted Dragon Survival Run
      let dragonScore: number | undefined;
      let dragonMilestonesCount: number | undefined;

      const isFinalCycle = cyclesCompleted + 1 >= maxCycles;
      const shouldRunDragon =
        !isShuttingDown &&
        (dragonFinalOnly
          ? isFinalCycle
          : (fullDragonInterval > 0 && iter % fullDragonInterval === 0) ||
            (fullDragonInterval > 0 && maxCycles < Infinity && isFinalCycle && (cyclesCompleted + 1) % fullDragonInterval === 0));

      if (shouldRunDragon) {
        console.log(`\n======================================================================`);
        console.log(`  ⭐ [BENCHMARK] LAUNCHING UNASSISTED DRAGON SURVIVAL RUN (Iter #${iter})`);
        console.log(`  - Mode: Vanilla 8-Milestone Survival DAG (0 spawned gear)`);
        console.log(`======================================================================\n`);

        try {
          await serverLifecycleMutex.runExclusive(async () => {
            // Pause curriculum server and bots
            for (const w of workers) {
              try { await w.sparringPartner.disconnect().catch(() => {}); } catch {}
              try { w.system1.bot?.quit(); } catch {}
            }
            try { survivalWorker?.system1.bot?.quit(); } catch {}
            await server.stopServer();
            await new Promise((r) => setTimeout(r, 1500));

            // Run full survival run
            const dragonRes = await runVanillaDragonSurvivalRun(30);
            dragonScore = dragonRes.progressDepthScore;
            dragonMilestonesCount = dragonRes.milestonesAchievedCount;
            console.log(`[TrainLoop] Dragon Survival Run Finished! Score: ${dragonScore.toFixed(1)}% (${dragonMilestonesCount}/8 milestones).`);

            if (!isFinalCycle && !isShuttingDown) {
              // Restart curriculum server for next training cycle
              await server.startServer(serverPort);
              for (const w of workers) {
                try {
                  await w.system1.connect('127.0.0.1', serverPort, w.botUsername);
                  await new Promise((r) => setTimeout(r, 600));
                } catch {}
                try {
                  await w.sparringPartner.connect('127.0.0.1', serverPort);
                  await new Promise((r) => setTimeout(r, 300));
                } catch {}
              }
              if (survivalWorker) {
                try {
                  await survivalWorker.system1.connect('127.0.0.1', serverPort, survivalWorker.botUsername);
                  await seedPersistentSurvivalSector(server, survivalWorker.botUsername);
                } catch {}
              }
            }
          });
        } catch (dErr: any) {
          console.error('[TrainLoop] Error in Dragon Survival Run:', dErr);
        }
      }

      if (!state.history) state.history = [];
      const slmMetricsPath = path.resolve(resultsDir, 'slm_training_metrics.json');
      let slmInitLoss: number | undefined;
      let slmFinLoss: number | undefined;
      if (fs.existsSync(slmMetricsPath)) {
        try {
          const m = JSON.parse(fs.readFileSync(slmMetricsPath, 'utf-8'));
          slmInitLoss = m.initialLoss;
          slmFinLoss = m.finalLoss;
        } catch {}
      }
      if (actAimErrorDeg === undefined) {
        const actMetricsPath = path.resolve(process.cwd(), 'neural_lab', 'results', 'action_head_metrics.json');
        if (fs.existsSync(actMetricsPath)) {
          try {
            const am = JSON.parse(fs.readFileSync(actMetricsPath, 'utf-8'));
            actAimErrorDeg = am.aim_error_deg;
            actDiscreteAccPct = am.discrete_accuracy_pct;
          } catch {}
        }
      }

      const heldOutMeanScore = heldOutTotal > 0 ? Number((heldOutScoreSum / heldOutTotal).toFixed(1)) : undefined;
      const heldOutPassRate = heldOutTotal > 0 ? Number(((heldOutPassed / heldOutTotal) * 100).toFixed(1)) : undefined;

      state.history.push({
        iteration: iter,
        phase: `Exhaustive Loop (${numTasks} Tasks across ${concurrency} Sharded Bots${survivalWorker ? ' + Survival' : ''})`,
        timestamp: new Date().toISOString(),
        meanScore: iterMeanScore,
        passRate: iterPassRate,
        tasksPassed: iterPassed,
        totalTasks: numTasks,
        heldOutMeanScore,
        heldOutPassRate,
        heldOutPassed: heldOutTotal > 0 ? heldOutPassed : undefined,
        heldOutTotal: heldOutTotal > 0 ? heldOutTotal : undefined,
        rawReward: iterTotalReward,
        durationSeconds: Math.round((Date.now() - iterStartTime) / 1000),
        slmInitialLoss: slmInitLoss,
        slmFinalLoss: slmFinLoss,
        actAimErrorDeg,
        actDiscreteAccPct,
        domainBreakdown,
        survivalCampaign: survivalCampaignRes,
        dragonSurvivalScore: dragonScore,
        dragonMilestones: dragonMilestonesCount,
        milestone: dragonScore !== undefined ? `Dragon Run: ${dragonScore.toFixed(0)}% (${dragonMilestonesCount}/8)` : undefined,
      });

      saveTrainingState(isNewBest);

      const elapsedMinutes = ((Date.now() - iterStartTime) / 60000).toFixed(1);
      console.log(`\n----------------------------------------------------------------------`);
      console.log(
        `✓ [Iteration #${iter} Finished in ${elapsedMinutes}m] Pass Rate: ${iterPassed}/${numTasks} (${iterPassRate}%) | Mean Score: ${iterMeanScore}/100`
      );
      if (heldOutMeanScore !== undefined) {
        console.log(
          `  20% Held-Out Eval:    ${heldOutPassed}/${heldOutTotal} (${heldOutPassRate}%) | Mean Score: ${heldOutMeanScore}/100`
        );
      }
      console.log(
        `  All-Time Best: ${state.bestMeanScore}/100 (${state.bestPassRate}% at Iteration #${state.bestIteration})`
      );
      if (survivalCampaignRes) {
        console.log(
          `  Open-World Survival: Goal "${survivalCampaignRes.goalTitle}" | Tier: ${survivalCampaignRes.techTierReached} | Advancements: ${survivalCampaignRes.advancementsUnlocked.length}`
        );
      }
      if (rlLossDelta) console.log(`  System 1 (PPO Reflex): ${rlLossDelta}`);
      if (actLossDelta) console.log(`  System 1 (80.8M ACT):  ${actLossDelta}`);
      if (slmLossDelta) console.log(`  System 2 (SLM LoRA):   ${slmLossDelta}`);
      console.log(`  Weights Updated: voxel_reflex_policy_latest.pt, action_head_latest.pt, & slm_minecraft_savant_latest/`);
      if (isNewBest) {
        console.log(`  ⭐ NEW HIGH SCORE! Saved to: neural_lab/checkpoints/voxel_reflex_policy_best.pt`);
      }
      console.log(`  (Press Ctrl+C anytime to safely pause and save)`);
      console.log(`----------------------------------------------------------------------\n`);

      cyclesCompleted++;
      const elapsedHours = (Date.now() - sessionStartTime) / (1000 * 60 * 60);
      if (cyclesCompleted >= maxCycles) {
        console.log(`[TrainLoop] Target of ${maxCycles} cycle(s) reached. Stopping loop cleanly.`);
        break;
      }
      if (elapsedHours >= maxHours) {
        console.log(`[TrainLoop] Target runtime of ${maxHours} hour(s) reached (${elapsedHours.toFixed(2)}h elapsed). Stopping loop cleanly.`);
        break;
      }
    }
  } finally {
    if (!isShuttingDown) {
      saveTrainingState(false);
      for (const w of workers) {
        try {
          await w.sparringPartner.disconnect().catch(() => {});
        } catch {}
        try {
          w.system1.bot?.quit();
        } catch {}
      }
      try {
        survivalWorker?.system1.bot?.quit();
      } catch {}
      await server.stopServer();
      if (spawnedDaemonProc) {
        try {
          await fetch('http://127.0.0.1:8765/shutdown', { method: 'POST', signal: AbortSignal.timeout(1500) }).catch(() => {});
        } catch {}
        forceKillPid(spawnedDaemonProc.pid);
        try {
          spawnedDaemonProc.kill();
        } catch {}
      }
    }
  }
}

if (require.main === module) {
  runContinuousTrainingLoop()
    .then(() => {
      process.exit(0);
    })
    .catch((err) => {
      console.error(`[TrainLoop Fatal Error]`, err);
      process.exit(1);
    });
}
