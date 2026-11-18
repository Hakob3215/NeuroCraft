import * as fs from 'fs';
import * as path from 'path';
import { execFileSync, spawn } from 'child_process';
import { Vec3 } from 'vec3';
import {
  MINEBENCH_TASKS,
  MULTIPLAYER_TASKS,
  MinebenchDomain,
  MinebenchTaskSpec,
} from './benchmarkSpec';
import { SparringPartnerBot } from './sparringPartner';
import { evaluateMinebenchEpisode, MinebenchEvaluationSnapshot } from './evaluators';
import { HeadlessMinecraftServerManager } from './headlessServer';
import { System1BotController, HOSTILE_MOBS } from '../src/system1/botController';
import { DynamicMCPToolRegistry } from '../src/system2/mcp/toolRegistry';
import { NeuroCraftMCPServer } from '../src/system2/mcp/mcpServer';
import { registerBaseMCPTools } from '../src/system2/skills/baseSkills';
import { ExecutedStepRecord, UnifiedLLMClient } from '../src/system2/llmClient';
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
    .filter((e: any) => e && e !== bot.entity && e.position && e.isValid !== false)
    .sort((a: any, b: any) => a.position.distanceTo(botPos) - b.position.distanceTo(botPos))
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

function sampleDiverseOmniscienceTasks(tasksPath: string, count = 50): MinebenchTaskSpec[] {
  if (!fs.existsSync(tasksPath)) return [];
  try {
    const rawAll: any[] = JSON.parse(fs.readFileSync(tasksPath, 'utf-8'));
    const omniOnly = rawAll.filter((t) => t.taskId && t.taskId.startsWith('omni_'));
    const byCat: Record<string, any[]> = {};
    for (const t of omniOnly) {
      const cat = t.category || 'tool_affordance';
      if (!byCat[cat]) byCat[cat] = [];
      byCat[cat].push(t);
    }

    for (const cat in byCat) {
      byCat[cat].sort(() => Math.random() - 0.5);
    }

    const quotas: Record<string, number> = {
      tool_affordance: 8,
      nutrition: 7,
      equipment: 6,
      crafting: 7,
      block_mechanic: 4,
      mob_interaction: 15,
      structure_raid: 3,
    };

    const selected: any[] = [];
    for (const [cat, quota] of Object.entries(quotas)) {
      const pool = byCat[cat] || [];
      selected.push(...pool.slice(0, quota));
    }

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
      buildArenaCommands: (orig: { x: number; y: number; z: number }) => [
        `gamerule commandModificationBlockLimit 100000`,
        `fill ${orig.x - 12} ${orig.y - 3} ${orig.z - 12} ${orig.x + 12} ${orig.y - 1} ${orig.z + 12} minecraft:stone`,
        `fill ${orig.x - 12} ${orig.y - 4} ${orig.z - 12} ${orig.x + 12} ${orig.y - 4} ${orig.z + 12} minecraft:bedrock`,
        `fill ${orig.x - 12} ${orig.y} ${orig.z - 12} ${orig.x + 12} ${orig.y + 10} ${orig.z + 12} minecraft:air`,
        ...(t.setupCommands || [])
          .filter((cmd: string) => !cmd.startsWith('tp @p') && !cmd.startsWith('tp NeuroAgent'))
          .map((cmd: string) => resolveMinecraftCoordinates(cmd, orig)),
        `fill ${orig.x - 12} ${orig.y} ${orig.z - 12} ${orig.x + 12} ${orig.y + 2} ${orig.z + 12} minecraft:glass outline`,
        `fill ${orig.x - 11} ${orig.y} ${orig.z - 11} ${orig.x + 11} ${orig.y + 2} ${orig.z + 11} minecraft:air`,
        `tp NeuroAgent ${orig.x} ${orig.y} ${orig.z} 0 0`,
      ],
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
  await server.startServer(serverPort);

  console.log(`[TrainLoop] Initializing ${concurrency} parallel worker bot(s) on port ${serverPort}...`);
  const workers: WorkerContext[] = [];
  for (let i = 0; i < concurrency; i++) {
    const botUsername = getWorkerBotName(i, concurrency);
    const deltaX = getWorkerSpatialOffset(i);
    const sparringUsername = concurrency === 1 ? 'DQN_Opponent' : `DQN_Opponent_${i + 1}`;

    const system1 = new System1BotController();
    console.log(`[TrainLoop] Connecting worker #${i} (${botUsername}) at X-offset +${deltaX}...`);
    await system1.connect('127.0.0.1', serverPort, botUsername);
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
  console.log(`[TrainLoop] All ${concurrency} worker(s) successfully connected and ready!\n`);

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
      const sampledOmniTasks = sampleDiverseOmniscienceTasks(tasksPath, omniTaskCount);
      const loopTasks: MinebenchTaskSpec[] = [
        ...MINEBENCH_TASKS,
        ...sampledOmniTasks,
        ...MULTIPLAYER_TASKS,
      ];
      const numTasks = loopTasks.length;

      console.log(`\n======================================================================`);
      console.log(`>>> [Iteration #${iter}] Starting Exhaustive Anti-Collapse Curriculum Pass:`);
      console.log(`    - 16 Canonical MINEbench Challenges`);
      console.log(`    - ${sampledOmniTasks.length} Stratified Omniscience Affordances across 7 categories`);
      console.log(`    - ${MULTIPLAYER_TASKS.length} Guaranteed Multiplayer Scenarios`);
      console.log(`    - Total In-Game Physical Scenarios: ${numTasks}`);
      console.log(`    - Active Bot Concurrency: ${concurrency} Worker(s) (Spatial Arena Sharding ΔX=4000)`);
      console.log(`======================================================================\n`);

      let iterPassed = 0;
      let iterScoreSum = 0;
      let iterRewardSum = 0;
      const iterStepTransitions: any[] = [];

      const arenaResetMutex = new AsyncMutex();
      const stateWriteMutex = new AsyncMutex();
      const taskQueue = new ConcurrentTaskQueue(loopTasks);

      const workerPromises = workers.map(async (worker) => {
        while (!isShuttingDown) {
          const next = taskQueue.dequeue();
          if (!next) break;

          const { item: task, index: taskIndex } = next;
          const taskT0 = Date.now();
          const shardedTask = createShardedTaskSpec(task, worker.id, concurrency);

          // Verify worker bot connection and server readiness
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
              }
            }
          }

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
            await new Promise((r) => setTimeout(r, 300));
          });

          if (task.sparringConfig) {
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
          let ironPickaxeDurabilityUsedOnStone = 0;
          const killedMobsMap: Record<string, number> = {};

          const targetEntityName = String((task as any).targetObject || '').toLowerCase();
          const initialTargetEntities = targetEntityName ? countEntitiesByName(targetEntityName) : 0;

          const onDeath = () => {
            deaths++;
            worker.sparringPartner.recordAgentDeath();
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
            if (newBlock.name === 'lever') {
              const props = typeof newBlock.getProperties === 'function' ? newBlock.getProperties() : {};
              if (props?.powered === true || props?.powered === 'true' || ((newBlock.metadata ?? 0) & 8) !== 0) {
                leversFlipped++;
              }
            }
            if (newBlock.position?.distanceTo(bot.entity.position) <= 6.0) {
              affordanceActions++;
            }
          };

          bot.on('death', onDeath);
          bot.on('entityDead', onEntityDead);
          bot.on('entityGone', onEntityGone);
          bot._client?.on('explosion', onExplosion);
          bot.on('diggingCompleted', onDiggingCompleted);
          bot.on('blockUpdate', onBlockUpdate);

          const buildSnapshot = (): MinebenchEvaluationSnapshot => {
            const aliveCreepers = countEntitiesByName('creeper');
            const aliveSkeletons = countEntitiesByName('skeleton');
            const aliveSpiders = countEntitiesByName('spider');
            const aliveEndermen = countEntitiesByName('enderman');
            const aliveCrystals = countEntitiesByName('end_crystal');
            const aliveDragons = countEntitiesByName('ender_dragon');

            const leverPos = new Vec3(shardedTask.origin.x + 3, shardedTask.origin.y + 1, shardedTask.origin.z - 1);
            const leverBlock = bot.blockAt(leverPos);
            if (leverBlock && leverBlock.name === 'lever') {
              const props = typeof leverBlock.getProperties === 'function' ? leverBlock.getProperties() : {};
              if (props?.powered === true || props?.powered === 'true' || ((leverBlock.metadata ?? 0) & 8) !== 0) {
                leversFlipped = Math.max(1, leversFlipped);
              }
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
                ironPickaxeDurabilityUsedOnStone,
              },
              multiplayer: task.sparringConfig ? worker.sparringPartner.getSnapshot() : undefined,
            };
          };

          const stepHistory: ExecutedStepRecord[] = [];
          const executedTools: any[] = [];
          const episodeTransitions: any[] = [];
          let activeGoal = task.prompt;

          try {
            for (let stepIdx = 0; stepIdx < task.maxSteps; stepIdx++) {
              if (isShuttingDown || Date.now() - taskT0 > task.maxDurationSeconds * 1000) break;

              const preRlState = sampleLiveRLTransitionState(worker.system1);
              const preSnap = buildSnapshot();
              const preGrading = evaluateMinebenchEpisode(shardedTask, preSnap);
              if (stepIdx > 0 && preGrading.passed) break;

              const worldState = worker.system1.getLatestWorldState();
              const schemas = worker.registry.listSchemas();

              const variants = PROMPT_VARIANTS[task.id];
              let effectivePrompt =
                variants && variants.length > 0
                  ? variants[Math.floor(Math.random() * variants.length)]
                  : task.prompt;

              effectivePrompt = effectivePrompt
                .replace(/\bDQN_Opponent\b/g, worker.sparringPartner.username)
                .replace(/\bMirrorAgent\b/g, worker.sparringPartner.username)
                .replace(/\bCompanion_Player\b/g, worker.sparringPartner.username);

              const decision =
                stepIdx === 0
                  ? await worker.llm.generateCognitiveDecision('Trainer', effectivePrompt, worldState, schemas, [])
                  : await worker.llm.generateGoalContinuationDecision(activeGoal, stepHistory, worldState, schemas, []);

              if (decision.activeGoal) activeGoal = decision.activeGoal;

              if (!decision.toolCall || !decision.toolCall.name) {
                if (decision.goalCompleted && stepIdx > 0) break;
                continue;
              }

              const toolName = decision.toolCall.name;
              const toolArgs = decision.toolCall.arguments || {};
              const isBuildTool = toolName.includes('construct') || toolName.includes('build') || toolName.includes('house');
              const stepDurationMs = isBuildTool ? 55000 : 25000;
              const abortController = new AbortController();
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

              try {
                const prevYaw = bot.entity?.yaw || 0;
                const prevPitch = bot.entity?.pitch || 0;

                const rpcRes = await worker.mcpServer.handleRpcRequest(
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

                const payloadText = rpcRes.result?.content?.[0]?.text;
                if (payloadText) {
                  try {
                    const parsed = JSON.parse(payloadText);
                    stepSuccess = Boolean(parsed.success);
                    stepSummary = String(parsed.summary || '');
                    if (toolName === 'tame_animal' && stepSuccess) wolvesTamed = Math.max(1, wolvesTamed);
                    if (toolName === 'shoot_bow' && parsed.data?.fired) arrowsFired += Number(parsed.data.fired) || 0;
                    if ((toolName === 'eat_food' || toolName === 'consume_item') && stepSuccess) foodConsumed++;
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
                const stepReward = Number((postGrading.rawReward - preGrading.rawReward).toFixed(3));

                if (preRlState) {
                  const dYaw = Math.max(-0.35, Math.min(0.35, (bot.entity?.yaw || 0) - prevYaw));
                  const dPitch = Math.max(-0.35, Math.min(0.35, (bot.entity?.pitch || 0) - prevPitch));
                  const transitionRecord = {
                    provider: 'local_gpu',
                    taskId: task.id,
                    workerId: worker.id,
                    stepIdx,
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
                clearTimeout(stepTimeout);
                try {
                  bot.pathfinder?.setGoal(null);
                  bot.clearControlStates?.();
                } catch {}
              }

              if (stepSuccess) affordanceActions++;
              executedTools.push({ name: toolName, arguments: toolArgs, args: toolArgs, success: stepSuccess });
              stepHistory.push({
                stepNumber: stepIdx + 1,
                toolName,
                args: toolArgs,
                success: stepSuccess,
                summary: stepSummary,
              });

              if (decision.goalCompleted) break;
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

          await stateWriteMutex.runExclusive(async () => {
            state.totalEpisodesPlayed++;
            if (grading.passed) iterPassed++;
            iterScoreSum += grading.score;
            iterRewardSum += grading.rawReward;

            for (const tr of episodeTransitions) {
              fs.appendFileSync(transitionsPath, JSON.stringify(tr) + '\n', 'utf-8');
              iterStepTransitions.push(tr);
            }

            const replayRecord = {
              timestamp: new Date().toISOString(),
              iteration: iter,
              provider: 'local_gpu',
              taskId: task.id,
              workerId: worker.id,
              botUsername: worker.botUsername,
              domain: task.domain,
              tier: task.tier,
              prompt: task.prompt,
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

      if (isShuttingDown) break;

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
        await new Promise((r) => setTimeout(r, 600));

        // 1. Trigger RTX 5060 PPO + GRPO gradient update for System 1 Motor Reflex
        if (fs.existsSync(venvPython) && fs.existsSync(rlScript)) {
          try {
            const pyArgs = [rlScript, '--epochs', '8', '--loop', String(iter)];
            if (isNewBest) pyArgs.push('--is-best');
            const stdout = execFileSync(venvPython, pyArgs, { encoding: 'utf-8', timeout: 90000 });
            const parsed = JSON.parse(stdout.trim().split('\n').pop() || '{}');
            rlLossDelta = `PPO: ${parsed.initialLoss} -> ${parsed.finalLoss} (${parsed.transitionsPerSecond} trans/s)`;
          } catch (err: any) {
            rlLossDelta = `RL note: ${err.message}`;
          }
        }

        // 2. Trigger RTX 5060 System 2 Cognitive SLM QLoRA Fine-Tuning ('Minecraft Savant')
        if (fs.existsSync(venvPython) && fs.existsSync(slmScript)) {
          try {
            const slmArgs = [slmScript, '--epochs', '1', '--loop', String(iter), '--max-trajectories', '256'];
            if (isNewBest) slmArgs.push('--is-best');

            // Offload daemon's SLM to CPU to free VRAM for trainer
            try {
              const pauseRes = await fetch('http://127.0.0.1:8765/pause_slm', { method: 'POST' });
              await pauseRes.json();
              await new Promise((r) => setTimeout(r, 3500));
            } catch {}

            const slmOut = execFileSync(venvPython, slmArgs, { encoding: 'utf-8', timeout: 360000 });
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
            const actOut = execFileSync(venvPython, actArgs, { encoding: 'utf-8', timeout: 180000 });
            const parsedAct = JSON.parse(actOut.trim().split('\n').pop() || '{}');
            actLossDelta = `ACT (80.8M): ${parsedAct.initial_loss} -> ${parsedAct.final_loss} (aim err: ${parsedAct.aim_error_deg}°, disc: ${parsedAct.discrete_accuracy_pct}%)`;
          } catch (err: any) {
            actLossDelta = `ACT note: ${err.message}`;
          } finally {
            // Hot-reload updated adapter and action head in running neural daemon
            try {
              await fetch('http://127.0.0.1:8765/reload_adapter', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ adapter_name: 'slm_minecraft_savant_latest' }),
              });
            } catch {}
          }
        } else {
          // Hot-reload updated adapter in running neural daemon if actScript was skipped
          try {
            await fetch('http://127.0.0.1:8765/reload_adapter', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ adapter_name: 'slm_minecraft_savant_latest' }),
            });
          } catch {}
        }

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
          // Pause curriculum server and bots
          for (const w of workers) {
            try { await w.sparringPartner.disconnect().catch(() => {}); } catch {}
            try { w.system1.bot?.quit(); } catch {}
          }
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
          }
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

      state.history.push({
        iteration: iter,
        phase: `Exhaustive Loop (${numTasks} Tasks across ${concurrency} Sharded Bots)`,
        timestamp: new Date().toISOString(),
        meanScore: iterMeanScore,
        passRate: iterPassRate,
        tasksPassed: iterPassed,
        totalTasks: numTasks,
        rawReward: iterTotalReward,
        durationSeconds: Math.round((Date.now() - iterStartTime) / 1000),
        slmInitialLoss: slmInitLoss,
        slmFinalLoss: slmFinLoss,
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
      console.log(
        `  All-Time Best: ${state.bestMeanScore}/100 (${state.bestPassRate}% at Iteration #${state.bestIteration})`
      );
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
      if (cyclesCompleted >= maxCycles) {
        console.log(`[TrainLoop] Target of ${maxCycles} cycle(s) reached. Stopping loop cleanly.`);
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
      await server.stopServer();
      if (spawnedDaemonProc) {
        try {
          await fetch('http://127.0.0.1:8765/shutdown', { method: 'POST', signal: AbortSignal.timeout(1500) }).catch(() => {});
        } catch {}
        try {
          spawnedDaemonProc.kill();
        } catch {}
      }
    }
  }
}

if (require.main === module) {
  runContinuousTrainingLoop().catch((err) => {
    console.error(`[TrainLoop Fatal Error]`, err);
    process.exit(1);
  });
}
