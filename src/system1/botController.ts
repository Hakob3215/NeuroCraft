import mineflayer from 'mineflayer';
import { pathfinder, Movements, goals } from 'mineflayer-pathfinder';
import { Vec3 } from 'vec3';
import { logVerbose } from '../logger';
import {
  AreaTopologyScan,
  EquippedArmorState,
  MCPToolExecutionContext,
  PerceivedBlock,
  PerceivedEntity,
  ReflexInterruptEvent,
  ResourceClusterSummary,
  SanitizedWorldState,
  SubtaskStatus,
  TerrainContext,
} from '../types';
import { CONFIG } from '../config';
import { isWithinHumanFov } from './perception/frustum';
import { isEntityOccluded } from './perception/raycaster';
import { AuditoryPerceptionTracker, sampleHumanReactionDelayMs } from './perception/auditory';
import { HumanizedCameraController } from './kinematics/camera';
import { HumanizedMovementController } from './kinematics/movement';
import { ModernCombatController } from './kinematics/combat';
import { HumanizedInventoryController } from './kinematics/inventory';
import { ReflexEngine } from './reflexes/reflexEngine';
import { MINECRAFT_1_20_CUSTOM_PACKETS } from '../protocolFixes';
import { VLACompanionClient, getEntityPriority, isBossOrCrystalEntity } from '../system2/vlaClient';

export { getEntityPriority, isBossOrCrystalEntity };

export const HOSTILE_MOBS = new Set([
  'zombie',
  'zombie_villager',
  'skeleton',
  'wither_skeleton',
  'creeper',
  'spider',
  'cave_spider',
  'enderman',
  'witch',
  'slime',
  'magma_cube',
  'drowned',
  'husk',
  'stray',
  'bogged',
  'breeze',
  'pillager',
  'vindicator',
  'evoker',
  'ravager',
  'vex',
  'phantom',
  'blaze',
  'ghast',
  'hoglin',
  'zoglin',
  'piglin_brute',
  'warden',
  'guardian',
  'elder_guardian',
  'silverfish',
  'endermite',
  'shulker',
  'ender_dragon',
  'end_crystal',
]);

const NOTABLE_BLOCKS = new Set([
  'oak_log',
  'spruce_log',
  'birch_log',
  'jungle_log',
  'acacia_log',
  'dark_oak_log',
  'cherry_log',
  'mangrove_log',
  'crafting_table',
  'furnace',
  'blast_furnace',
  'chest',
  'coal_ore',
  'deepslate_coal_ore',
  'iron_ore',
  'deepslate_iron_ore',
  'copper_ore',
  'gold_ore',
  'diamond_ore',
  'deepslate_diamond_ore',
  'stone',
  'cobblestone',
  'sand',
  'dirt',
  'farmland',
  'wheat',
  'lever',
  'gold_block',
  'emerald_block',
  'obsidian',
  'water',
  'lava',
  'campfire',
  'soul_campfire',
  'grindstone',
  'ancient_debris',
  'pumpkin',
  'carved_pumpkin',
  'tnt',
  'nether_portal',
  'smoker',
  'anvil',
  'enchanting_table',
  'brewing_stand',
  'beacon',
  'barrel',
  'spawner',
]);

function getCompassDirection(dx: number, dz: number): string {
  // In Minecraft: -Z is North, +Z is South, +X is East, -X is West
  const angleDeg = (Math.atan2(dx, -dz) * 180) / Math.PI; // 0 = N, 90 = E, 180/-180 = S, -90 = W
  const normalized = (angleDeg + 360) % 360;
  const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  return dirs[Math.round(normalized / 45) % 8];
}

function hasAirNeighbor(bot: any, pos: Vec3): boolean {
  const offsets = [
    new Vec3(0, 1, 0),
    new Vec3(1, 0, 0),
    new Vec3(-1, 0, 0),
    new Vec3(0, 0, 1),
    new Vec3(0, 0, -1),
    new Vec3(0, -1, 0),
  ];
  for (const off of offsets) {
    const nb = bot.blockAt(pos.plus(off));
    if (!nb || nb.name === 'air' || nb.boundingBox === 'empty') {
      return true;
    }
  }
  return false;
}

/**
 * Detects whether an incoming message is a Vanilla server admin broadcast or command result
 * (e.g. "[ThePlayer: Gave 64 Steak to NeuroAgent]" or "Gave 64 [Torch] to TheJack3215]")
 * rather than genuine player chat.
 */
export function isServerCommandOrAdminMessage(message: string, translate?: string): boolean {
  if (translate === 'chat.type.admin' || (translate && translate.startsWith('commands.'))) {
    return true;
  }
  const trimmed = message.trim();
  if (trimmed.endsWith(']')) {
    if (
      /^(Gave\s|Applied\s|Set\s|Teleported\s|Killed\s|Summoned\s|Cleared\s|Enchanted\s|Filled\s|Played\s|Saved\s|Seed:\s|Time\sis\s|Whispered\s|Unknown\s|Usage:)/i.test(
        trimmed
      )
    ) {
      return true;
    }
  }
  if (/^\[.+:\s(Gave|Applied|Set|Teleported|Killed|Summoned|Cleared|Enchanted)/i.test(trimmed)) {
    return true;
  }
  return false;
}

/**
 * Robustly formats any Minecraft kick, disconnect packet, or ChatMessage object into
 * human-readable text without ever returning "[object Object]".
 */
export function formatKickReason(reason: any): string {
  if (!reason) return 'Clean socket disconnect or timeout (no server kick packet)';
  if (typeof reason === 'string') {
    try {
      const parsed = JSON.parse(reason);
      return formatKickReason(parsed);
    } catch {
      return reason;
    }
  }
  if (typeof reason === 'object') {
    // 0. Simplify NBT compound tag structures (common in Minecraft 1.20.4+ disconnect packets)
    try {
      if (reason.type && reason.value) {
        const nbt = require('prismarine-nbt');
        const simplified = nbt.simplify(reason);
        return formatKickReason(simplified);
      }
    } catch {}

    // 1. Check for prismarine-chat / modern ChatMessage methods
    if (typeof reason.toAnsi === 'function') {
      const ansi = reason.toAnsi();
      if (ansi && ansi !== '[object Object]') return ansi;
    }
    if (typeof reason.toMotd === 'function') {
      const motd = reason.toMotd();
      if (motd && motd !== '[object Object]') return motd;
    }
    // 2. Check if toString is overridden (NOT Object.prototype.toString)
    if (typeof reason.toString === 'function' && reason.toString !== Object.prototype.toString) {
      const str = reason.toString();
      if (str && str !== '[object Object]') return str;
    }
    // 3. Extract standard Minecraft JSON component fields: { translate, with, text, extra }
    if (reason.translate) {
      const transStr = typeof reason.translate === 'object' ? formatKickReason(reason.translate) : String(reason.translate);
      const withArgs = Array.isArray(reason.with)
        ? ` [${reason.with.map((w: any) => (typeof w === 'object' ? formatKickReason(w) : String(w))).join(', ')}]`
        : '';
      return `${transStr}${withArgs}`;
    }
    if (reason.text) {
      const textStr = typeof reason.text === 'object' ? formatKickReason(reason.text) : String(reason.text);
      const extraText = Array.isArray(reason.extra)
        ? reason.extra.map((e: any) => (typeof e === 'object' ? formatKickReason(e) : String(e))).join('')
        : '';
      return `${textStr}${extraText}`;
    }
    if (reason.value !== undefined) {
      return formatKickReason(reason.value);
    }
    try {
      return JSON.stringify(reason);
    } catch {
      return String(reason);
    }
  }
  return String(reason);
}

/**
 * System 1: 20Hz Kinematic & Perceptual Bot Controller.
 * Connects as an offline LAN companion player (zero extra Minecraft account needed)
 * and runs the 50ms tick loop completely decoupled from System 2 LLM latency.
 */
export class System1BotController {
  public bot: any = null;
  public camera!: HumanizedCameraController;
  public movement!: HumanizedMovementController;
  public combat!: ModernCombatController;
  public inventory!: HumanizedInventoryController;
  public reflexEngine!: ReflexEngine;
  public auditory = new AuditoryPerceptionTracker();
  public vlaClient = new VLACompanionClient('http://127.0.0.1:8765');

  private tickCounter = 0;
  private tickInterval: NodeJS.Timeout | null = null;
  private latestWorldState: SanitizedWorldState | null = null;
  private activeSubtask: SubtaskStatus | null = null;
  private entityPerceptionTimestamps = new Map<number, number>();
  private cachedNotableBlocks: PerceivedBlock[] = [];
  private cachedTerrainContext: TerrainContext | undefined = undefined;
  private lastBlockScanTick = -100;
  private lastBlockScanPos: Vec3 | null = null;
  private disconnectHandlers: Array<(reason: string) => void> = [];

  public invalidatePerceptionCache(): void {
    this.lastBlockScanTick = -999;
    this.lastBlockScanPos = null;
    this.cachedTerrainContext = undefined;
    this.cachedNotableBlocks = [];
    this.latestWorldState = null;
    this.entityPerceptionTimestamps.clear();
  }

  public refreshWorldStateNow(immediateEntityPerception = false): SanitizedWorldState | null {
    if (!this.bot?.entity) return null;
    if (immediateEntityPerception) {
      const entitiesObj = this.bot.entities || {};
      for (const idStr of Object.keys(entitiesObj)) {
        const ent = entitiesObj[idStr];
        if (!ent || ent === this.bot.entity || !ent.position) continue;
        if (ent.isValid === false || (typeof (ent as any).health === 'number' && (ent as any).health <= 0)) {
          continue;
        }
        this.entityPerceptionTimestamps.set(ent.id, 0);
      }
    }
    this.latestWorldState = this.buildSanitizedWorldState();
    return this.latestWorldState;
  }

  public onDisconnect(handler: (reason: string) => void): void {
    this.disconnectHandlers.push(handler);
  }

  private readonly onChatReceived: (username: string, message: string) => void;
  private readonly onReflexInterrupt: (event: ReflexInterruptEvent) => void;
  private readonly onSubtaskProgress: (subtask: SubtaskStatus) => void;

  constructor(
    onChatReceived?: ((username: string, message: string) => void) | any,
    onReflexInterrupt?: (event: ReflexInterruptEvent) => void,
    onSubtaskProgress?: (subtask: SubtaskStatus) => void,
    botInstance?: any
  ) {
    if (onChatReceived && typeof onChatReceived === 'object' && ((onChatReceived as any).entity || (onChatReceived as any).blockAt)) {
      this.bot = onChatReceived as any;
      this.onChatReceived = () => {};
      this.onReflexInterrupt = () => {};
      this.onSubtaskProgress = () => {};
    } else {
      this.onChatReceived = typeof onChatReceived === 'function' ? onChatReceived : () => {};
      this.onReflexInterrupt = onReflexInterrupt || (() => {});
      this.onSubtaskProgress = onSubtaskProgress || (() => {});
      if (botInstance) {
        this.bot = botInstance;
      }
    }
  }

  /**
   * Connects to the local Minecraft Singleplayer LAN world in offline mode.
   */
  public async connect(host: string, port: number, username = CONFIG.botUsername): Promise<void> {
    return new Promise((resolve, reject) => {
      const safeUsername = String(username || CONFIG.botUsername).slice(0, 16);
      console.log(`[System 1] Connecting "${safeUsername}" to ${host}:${port} (Offline LAN Mode)...`);

      this.bot = mineflayer.createBot({
        host,
        port,
        username: safeUsername,
        auth: 'offline',
        version: CONFIG.minecraftVersion,
        hideErrors: true,
        checkTimeoutInterval: 600000,
        customPackets: MINECRAFT_1_20_CUSTOM_PACKETS,
      } as any);

      this.bot.loadPlugin(pathfinder);

      this.camera = new HumanizedCameraController(this.bot);
      this.movement = new HumanizedMovementController(this.bot, this.camera);
      this.combat = new ModernCombatController(this.bot, this.camera);
      this.inventory = new HumanizedInventoryController(this.bot);
      this.reflexEngine = new ReflexEngine(
        this.bot,
        this.camera,
        this.combat,
        this.inventory,
        this.onReflexInterrupt
      );

      let connected = false;
      let settled = false;
      const connectTimer = setTimeout(() => {
        if (!connected && !settled) {
          settled = true;
          try { this.bot?.quit(); } catch {}
          reject(new Error(`Timed out waiting for "${safeUsername}" to spawn on ${host}:${port}`));
        }
      }, 60000);

      this.bot.once('spawn', () => {
        connected = true;
        if (!settled) {
          settled = true;
          clearTimeout(connectTimer);
        }
        console.log(`[System 1] "${safeUsername}" spawned into world at ${this.formatPos(this.bot.entity.position)}!`);

        // Eliminate float vs double hitbox rounding desync against block boundaries (Issue #3911)
        if (this.bot.physics) {
          this.bot.physics.playerHalfWidth = Math.fround(0.6) / 2;
        }

        const defaultMovements = new Movements(this.bot);
        defaultMovements.allowSprinting = true;
        defaultMovements.allowParkour = false;
        defaultMovements.canDig = false;
        defaultMovements.scafoldingBlocks = [];
        this.bot.pathfinder.setMovements(defaultMovements);

        this.start20HzTickLoop();
        resolve();
      });

      // Defensive knockback & damage handlers
      this.bot.on('entityHurt', (entity: any) => {
        if (entity === this.bot?.entity) {
          // Temporarily yield pathfinding for 250ms so authoritative server knockback resolves cleanly
          // without conflicting client-side A* path steps triggering an anti-cheat / moved-wrongly kick
          if (this.bot.pathfinder?.isMoving?.()) {
            const currentGoal = this.bot.pathfinder.goal;
            this.bot.pathfinder.setGoal(null);
            setTimeout(() => {
              if (currentGoal && !this.bot.pathfinder.goal && this.bot.entity?.isValid) {
                this.bot.pathfinder.setGoal(currentGoal);
              }
            }, 250);
          }
        }
      });

      // Ensure entity velocity received from server is always sanitized and never NaN
      this.bot._client?.on('entity_velocity', (packet: any) => {
        if (packet.entityId === this.bot?.entity?.id && this.bot?.entity?.velocity) {
          const vel = this.bot.entity.velocity;
          if (!Number.isFinite(vel.x)) vel.x = 0;
          if (!Number.isFinite(vel.y)) vel.y = 0;
          if (!Number.isFinite(vel.z)) vel.z = 0;
        }
      });

      this.bot.on('chat', (speaker: string, message: string, translate?: string) => {
        if (speaker === this.bot.username) return;
        if (isServerCommandOrAdminMessage(message, translate)) {
          logVerbose('SYSTEM_CHAT', `Ignored server admin/command broadcast from <${speaker}>: ${message}`);
          return;
        }
        this.onChatReceived(speaker, message);
      });

      this.bot.on('soundEffectHeard', (soundName: string, position: Vec3) => {
        if (!this.bot?.entity) return;
        this.auditory.registerSound(soundName, this.bot.entity.position, position);
      });

      const emitDisconnect = (reason: string) => {
        if (this.tickInterval) {
          clearInterval(this.tickInterval);
          this.tickInterval = null;
        }
        for (const handler of this.disconnectHandlers) {
          try {
            handler(reason);
          } catch (err: any) {
            console.error(`[System 1] Error in disconnect handler: ${err.message}`);
          }
        }
      };

      this.bot.on('error', (err: Error) => {
        const msg = err?.stack || err?.message || String(err);
        if (!connected && !settled) {
          settled = true;
          clearTimeout(connectTimer);
          reject(err);
        } else {
          console.warn(`[System 1 Network Warning] ${msg}`);
        }
      });

      this.bot.on('kicked', (reason: any) => {
        const formatted = formatKickReason(reason);
        console.warn(`[System 1] Kicked from server: ${formatted}`);
        if (!connected && !settled) {
          settled = true;
          clearTimeout(connectTimer);
          reject(new Error(`Kicked before spawn: ${formatted}`));
          return;
        }
        emitDisconnect(`Kicked: ${formatted}`);
      });

      this.bot.on('end', (reason?: any) => {
        const formatted = formatKickReason(reason);
        console.log(`[System 1] Disconnected from Minecraft world. Reason: ${formatted}`);
        if (!connected && !settled) {
          settled = true;
          clearTimeout(connectTimer);
          reject(new Error(`Disconnected before spawn: ${formatted}`));
          return;
        }
        emitDisconnect(`Disconnected: ${formatted}`);
      });

      // Attach low-level protocol & socket diagnostics
      const client = (this.bot as any)._client;
      if (client) {
        client.on('error', (err: any) => {
          const msg = err?.stack || err?.message || String(err);
          console.warn(`[System 1 Protocol Error] ${msg}`);
        });

        if (client.socket) {
          client.socket.on('error', (err: any) => {
            const msg = err?.stack || err?.message || String(err);
            console.warn(`[System 1 TCP Socket Error] ${msg}`);
          });

          client.socket.on('close', (hadError: boolean) => {
            logVerbose('NETWORK', `Underlying Minecraft TCP socket closed (hadError=${hadError})`);
          });
        }
      }
    });
  }

  /**
   * Performs a smooth, natural 360° 4-quadrant gaze sweep (N -> E -> S -> W)
   * and immediately refreshes the 360° Spatial Terrain & Resource Cluster map.
   */
  public async perform360EnvironmentScan(stepDurationMs = 150): Promise<SanitizedWorldState> {
    if (!this.bot?.entity || this.bot.targetDigBlock) {
      return this.getLatestWorldState();
    }

    const pos = this.bot.entity.position;
    const lookTargets = [
      new Vec3(pos.x, pos.y + 1.5, pos.z - 6), // North
      new Vec3(pos.x + 6, pos.y + 1.5, pos.z), // East
      new Vec3(pos.x, pos.y + 1.5, pos.z + 6), // South
      new Vec3(pos.x - 6, pos.y + 1.5, pos.z), // West
    ];

    for (const target of lookTargets) {
      if (!this.bot?.entity || this.bot.targetDigBlock) break;
      await this.camera.lookAtSmooth(target, stepDurationMs);
    }

    // Force an immediate full terrain & cluster scan after completing the 360 sweep
    this.lastBlockScanTick = -100;
    this.latestWorldState = this.buildSanitizedWorldState();
    return this.latestWorldState;
  }

  private start20HzTickLoop(): void {
    if (this.tickInterval) clearInterval(this.tickInterval);

    this.tickInterval = setInterval(() => {
      if (!this.bot?.entity) return;
      this.tickCounter++;

      // Guard against corrupted / NaN physics vectors
      if (this.bot.entity.velocity) {
        if (!Number.isFinite(this.bot.entity.velocity.x)) this.bot.entity.velocity.x = 0;
        if (!Number.isFinite(this.bot.entity.velocity.y)) this.bot.entity.velocity.y = 0;
        if (!Number.isFinite(this.bot.entity.velocity.z)) this.bot.entity.velocity.z = 0;
      }
      if (this.bot.entity.position) {
        if (!Number.isFinite(this.bot.entity.position.x) || !Number.isFinite(this.bot.entity.position.y) || !Number.isFinite(this.bot.entity.position.z)) {
          console.warn('[System 1 Physics Warning] Detected NaN in bot position! Restoring finite coordinates.');
          if (this.latestWorldState) {
            this.bot.entity.position.set(this.latestWorldState.position.x, this.latestWorldState.position.y, this.latestWorldState.position.z);
          }
        }
      }

      this.camera.updateTick();
      this.movement.updateTick();

      const state = this.buildSanitizedWorldState();
      this.latestWorldState = state;

      this.reflexEngine.evaluateTick(state);
    }, 50);
  }

  public buildSanitizedWorldState(): SanitizedWorldState {
    const now = Date.now();
    const botPos = this.bot?.entity?.position;
    if (!botPos || isNaN(botPos.x) || isNaN(botPos.y) || isNaN(botPos.z)) {
      if (this.latestWorldState) {
        return this.latestWorldState;
      }
      return {
        tick: this.tickCounter,
        timestamp: now,
        botUsername: this.bot?.username || CONFIG.botUsername,
        position: { x: 0, y: 64, z: 0 },
        velocity: { x: 0, y: 0, z: 0 },
        yaw: 0,
        pitch: 0,
        health: 20,
        food: 20,
        oxygen: 20,
        onGround: true,
        isInWater: false,
        isInLava: false,
        heldItem: null,
        equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
        inventory: [],
        visibleEntities: [],
        audibleCues: [],
        nearbyNotableBlocks: [],
        terrainContext: undefined,
        activeHazard: null,
        activeSubtask: this.activeSubtask,
      };
    }
    const eyePos = { x: botPos.x, y: botPos.y + 1.62, z: botPos.z };
    const botYaw = this.bot?.entity?.yaw ?? 0;
    const botPitch = this.bot?.entity?.pitch ?? 0;

    const isSolidBlock = (x: number, y: number, z: number): boolean => {
      const b = this.bot.blockAt(new Vec3(x, y, z));
      if (!b) return false;
      return b.boundingBox === 'block' && !b.name.includes('glass') && !b.name.includes('leaves');
    };

    const visibleEntities: PerceivedEntity[] = [];
    const entitiesObj = this.bot.entities || {};

    for (const idStr of Object.keys(entitiesObj)) {
      const ent = entitiesObj[idStr];
      if (!ent || ent === this.bot.entity || !ent.position) continue;
      if (ent.isValid === false || (typeof (ent as any).health === 'number' && (ent as any).health <= 0)) continue;

      const rawName = (ent.name || ent.username || ent.displayName || 'unknown').toLowerCase();
      const isBoss = isBossOrCrystalEntity(rawName);
      const dist = botPos.distanceTo(ent.position);
      const maxSight = isBoss ? 128.0 : CONFIG.perception.maxSightDistance;
      if (dist > maxSight) continue;

      const strictlyInFov = isWithinHumanFov(eyePos, botYaw, botPitch, ent.position);
      // 360° Auditory & Peripheral Radar within 32 blocks registers unoccluded mobs/players/animals;
      // Bosses and End Crystals have global sky/tower prominence across 128m;
      // Beyond 32m, standard entities require direct 90° foveal FOV cone.
      const inPerceptualRange = isBoss || dist <= 32.0 || strictlyInFov;
      if (!inPerceptualRange) continue;

      // Bosses in the sky and End Crystals on open towers are not occluded by ground terrain
      const isOccluded = isBoss ? false : isEntityOccluded(eyePos, ent.position, ent.height || 1.8, isSolidBlock);
      if (isOccluded) continue;

      if (!this.entityPerceptionTimestamps.has(ent.id)) {
        const reactionDelayMs = sampleHumanReactionDelayMs();
        this.entityPerceptionTimestamps.set(ent.id, now + reactionDelayMs);
      }

      const perceivedAt = this.entityPerceptionTimestamps.get(ent.id)!;
      if (now < perceivedAt) {
        continue;
      }

      let entityCategory: PerceivedEntity['type'] = 'other';
      if (ent.type === 'player') {
        entityCategory = 'player';
      } else if (HOSTILE_MOBS.has(rawName) || ent.type === 'hostile') {
        entityCategory = 'hostile';
      } else if (
        ent.type === 'mob' ||
        ent.type === 'animal' ||
        ent.type === 'passive' ||
        ent.type === 'water_creature'
      ) {
        entityCategory = 'passive';
      }

      visibleEntities.push({
        id: ent.id,
        name: rawName,
        username: ent.username,
        type: entityCategory,
        position: {
          x: Number(ent.position.x.toFixed(2)),
          y: Number(ent.position.y.toFixed(2)),
          z: Number(ent.position.z.toFixed(2)),
        },
        velocity: {
          x: Number((ent.velocity?.x ?? 0).toFixed(2)),
          y: Number((ent.velocity?.y ?? 0).toFixed(2)),
          z: Number((ent.velocity?.z ?? 0).toFixed(2)),
        },
        distance: Number(dist.toFixed(2)),
        isOccluded: false,
        inFov: strictlyInFov,
        perceivedAt,
      });
    }

    visibleEntities.sort((a, b) => {
      const pA = getEntityPriority(a.name, a.type);
      const pB = getEntityPriority(b.name, b.type);
      if (pA !== pB) return pA - pB;
      return a.distance - b.distance;
    });

    // Refresh comprehensive 360° Terrain & Resource Cluster Context every 20 ticks (1s) OR on >10m teleport
    if (
      this.tickCounter - this.lastBlockScanTick >= 20 ||
      !this.cachedTerrainContext ||
      !this.lastBlockScanPos ||
      botPos.distanceTo(this.lastBlockScanPos) > 10
    ) {
      this.lastBlockScanTick = this.tickCounter;
      this.lastBlockScanPos = botPos.clone();
      const scanResult = this.scanEnvironmentAndClusters();
      this.cachedNotableBlocks = scanResult.notableBlocks;
      this.cachedTerrainContext = scanResult.terrainContext;
    }

    const footBlock = this.bot.blockAt(botPos);
    const headBlock = this.bot.blockAt(new Vec3(botPos.x, botPos.y + 1.5, botPos.z));
    const isInLava =
      footBlock?.name?.includes('lava') || headBlock?.name?.includes('lava') || false;
    const isInWater =
      Boolean(this.bot.entity.isInWater) ||
      footBlock?.name?.includes('water') ||
      headBlock?.name?.includes('water') ||
      false;

    const invItems = (this.bot.inventory?.items() || []).map((item: any) => {
      const maxDur = item.maxDurability ?? null;
      const durUsed = item.durabilityUsed ?? 0;
      const remDur = maxDur ? Math.max(0, maxDur - durUsed) : undefined;
      const pct = maxDur ? Math.round((remDur! / maxDur) * 100) : undefined;
      return {
        name: item.name,
        count: item.count,
        slot: item.slot,
        durability: remDur,
        maxDurability: maxDur ?? undefined,
        durabilityPercent: pct,
      };
    });

    const slots = this.bot.inventory?.slots || [];
    const equippedArmor: EquippedArmorState = {
      head: slots[5]?.name ?? null,
      torso: slots[6]?.name ?? null,
      legs: slots[7]?.name ?? null,
      feet: slots[8]?.name ?? null,
      offHand: slots[45]?.name ?? null,
    };

    let heldItemStr: string | null = null;
    const held = this.bot.heldItem;
    if (held?.name) {
      if (held.maxDurability) {
        const rem = Math.max(0, held.maxDurability - (held.durabilityUsed ?? 0));
        const pct = Math.round((rem / held.maxDurability) * 100);
        heldItemStr = `${held.name} (${rem}/${held.maxDurability} durability, ${pct}%)`;
      } else {
        heldItemStr = held.name;
      }
    }

    return {
      tick: this.tickCounter,
      timestamp: now,
      botUsername: this.bot.username,
      position: {
        x: Number(botPos.x.toFixed(2)),
        y: Number(botPos.y.toFixed(2)),
        z: Number(botPos.z.toFixed(2)),
      },
      velocity: {
        x: Number((this.bot.entity.velocity?.x ?? 0).toFixed(2)),
        y: Number((this.bot.entity.velocity?.y ?? 0).toFixed(2)),
        z: Number((this.bot.entity.velocity?.z ?? 0).toFixed(2)),
      },
      yaw: Number(botYaw.toFixed(2)),
      pitch: Number(botPitch.toFixed(2)),
      health: this.bot.health ?? 20,
      food: this.bot.food ?? 20,
      oxygen: this.bot.oxygenLevel ?? 20,
      onGround: Boolean(this.bot.entity.onGround),
      isInWater,
      isInLava,
      heldItem: heldItemStr,
      equippedArmor,
      inventory: invItems,
      visibleEntities,
      audibleCues: this.auditory.getRecentCues(now),
      nearbyNotableBlocks: this.cachedNotableBlocks,
      terrainContext: this.cachedTerrainContext,
      activeHazard: this.reflexEngine.getActiveHazard(),
      activeSubtask: this.activeSubtask,
    };
  }

  /**
   * Scans the 3D voxel environment around the bot to build both:
   * 1. Immediate tactile surroundings (adjacent blocks, sky visibility, depth to stone below feet).
   * 2. Aggregated 360° `ResourceClusterSummary` across a 24-block radius so the LLM knows
   *    exactly how much wood, exposed stone, buried stone, coal_ore, iron_ore, crafting_table,
   *    or furnace is around it in every compass direction!
   */
  private scanEnvironmentAndClusters(): {
    notableBlocks: PerceivedBlock[];
    terrainContext: TerrainContext;
  } {
    const center = this.bot.entity.position.floored();
    const blockNameAt = (offsetX: number, offsetY: number, offsetZ: number): string =>
      this.bot.blockAt(center.offset(offsetX, offsetY, offsetZ))?.name || 'air';

    // Check sky visibility (up to 18 blocks above head)
    let canSeeSky = true;
    for (let dy = 2; dy <= 18; dy++) {
      const b = this.bot.blockAt(center.offset(0, dy, 0));
      if (b && b.boundingBox === 'block' && !b.name.includes('leaves') && !b.name.includes('glass')) {
        canSeeSky = false;
        break;
      }
    }

    // Check how many blocks of dirt/grass below feet until solid stone/deepslate starts
    let depthToStoneBlocks: number | null = null;
    for (let depth = 1; depth <= 14; depth++) {
      const b = this.bot.blockAt(center.offset(0, -depth, 0));
      if (b && (b.name === 'stone' || b.name === 'cobblestone' || b.name === 'deepslate')) {
        depthToStoneBlocks = depth;
        break;
      }
    }

    // Time of day classification
    const tod = (this.bot.time?.timeOfDay ?? 6000) % 24000;
    let timeOfDay: TerrainContext['timeOfDay'] = 'day';
    if (tod >= 13000 && tod < 22500) timeOfDay = 'night';
    else if (tod >= 12000 && tod < 13000) timeOfDay = 'dusk';
    else if (tod >= 22500) timeOfDay = 'dawn';

    // Scan 24-block radius for resource clusters (step by 1 within 8m for 100% precision, step by 2 from 9..24m)
    const clusters = new Map<
      string,
      {
        totalNearby: number;
        exposedCount: number;
        nearestPos: { x: number; y: number; z: number };
        nearestDist: number;
        nearestDir: string;
        nearestExposedPos?: { x: number; y: number; z: number };
        nearestExposedDist?: number;
      }
    >();

    const notableBlocks: PerceivedBlock[] = [];
    const radius = 22;

    for (let dx = -radius; dx <= radius; dx += Math.abs(dx) <= 8 ? 1 : 2) {
      for (let dy = -8; dy <= 10; dy += Math.abs(dy) <= 4 ? 1 : 2) {
        for (let dz = -radius; dz <= radius; dz += Math.abs(dz) <= 8 ? 1 : 2) {
          const pos = center.offset(dx, dy, dz);
          const block = this.bot.blockAt(pos);
          if (!block || !NOTABLE_BLOCKS.has(block.name)) continue;

          const dist = Number(this.bot.entity.position.distanceTo(pos).toFixed(1));
          const isExposed = hasAirNeighbor(this.bot, pos);

          // Group logs into their exact name, and group deepslate ores with their ore type
          const category = block.name;
          const existing = clusters.get(category);

          if (!existing) {
            clusters.set(category, {
              totalNearby: 1,
              exposedCount: isExposed ? 1 : 0,
              nearestPos: { x: pos.x, y: pos.y, z: pos.z },
              nearestDist: dist,
              nearestDir: getCompassDirection(dx, dz),
              nearestExposedPos: isExposed ? { x: pos.x, y: pos.y, z: pos.z } : undefined,
              nearestExposedDist: isExposed ? dist : undefined,
            });
          } else {
            existing.totalNearby++;
            if (isExposed) {
              existing.exposedCount++;
              if (
                existing.nearestExposedDist === undefined ||
                dist < existing.nearestExposedDist
              ) {
                existing.nearestExposedDist = dist;
                existing.nearestExposedPos = { x: pos.x, y: pos.y, z: pos.z };
              }
            }
            if (dist < existing.nearestDist) {
              existing.nearestDist = dist;
              existing.nearestPos = { x: pos.x, y: pos.y, z: pos.z };
              existing.nearestDir = getCompassDirection(dx, dz);
            }
          }

          if (
            isExposed &&
            block.name !== 'stone' &&
            block.name !== 'dirt' &&
            block.name !== 'sand' &&
            block.name !== 'cobblestone'
          ) {
            notableBlocks.push({
              name: block.name,
              position: { x: pos.x, y: pos.y, z: pos.z },
              distance: dist,
            });
          }
        }
      }
    }

    notableBlocks.sort((a, b) => a.distance - b.distance);
    if (notableBlocks.length > 24) {
      notableBlocks.splice(24);
    }

    const resourceClusters: ResourceClusterSummary[] = Array.from(clusters.entries())
      .map(([blockType, info]) => ({
        blockType,
        totalNearby: info.totalNearby,
        exposedCount: info.exposedCount,
        nearestPos: info.nearestPos,
        nearestDistance: info.nearestDist,
        nearestDirection: info.nearestDir,
        nearestExposedPos: info.nearestExposedPos,
        nearestExposedDistance: info.nearestExposedDist,
      }))
      .sort((a, b) => (a.nearestExposedDistance ?? a.nearestDistance) - (b.nearestExposedDistance ?? b.nearestDistance));

    const terrainContext: TerrainContext = {
      timeOfDay,
      isRaining: Boolean(this.bot.isRaining),
      canSeeSky,
      blockBelowFeet: blockNameAt(0, -1, 0),
      depthToStoneBlocks,
      adjacentBlocks: {
        north: blockNameAt(0, 0, -1),
        south: blockNameAt(0, 0, 1),
        east: blockNameAt(1, 0, 0),
        west: blockNameAt(-1, 0, 0),
        aboveHead: blockNameAt(0, 2, 0),
        belowFeet: blockNameAt(0, -1, 0),
      },
      resourceClusters: resourceClusters.slice(0, 14),
    };

    return {
      notableBlocks: notableBlocks.slice(0, 12),
      terrainContext,
    };
  }

  public getLatestWorldState(): SanitizedWorldState {
    if (!this.latestWorldState && this.bot?.entity) {
      this.latestWorldState = this.buildSanitizedWorldState();
    }
    return (
      this.latestWorldState || {
        tick: 0,
        timestamp: Date.now(),
        botUsername: CONFIG.botUsername,
        position: { x: 0, y: 64, z: 0 },
        velocity: { x: 0, y: 0, z: 0 },
        yaw: 0,
        pitch: 0,
        health: 20,
        food: 20,
        oxygen: 20,
        onGround: true,
        isInWater: false,
        isInLava: false,
        heldItem: null,
        inventory: [],
        visibleEntities: [],
        audibleCues: [],
        nearbyNotableBlocks: [],
        activeHazard: null,
        activeSubtask: null,
      }
    );
  }

  public setSubtaskStatus(subtask: SubtaskStatus | null): void {
    this.activeSubtask = subtask;
    if (subtask) {
      this.onSubtaskProgress(subtask);
    }
  }

  public createToolExecutionContext(
    subtask: SubtaskStatus,
    sayHumanized: (msg: string) => Promise<void>,
    abortSignal: AbortSignal
  ): MCPToolExecutionContext {
    return {
      bot: this.bot,
      kinematics: {
        lookAtSmooth: (target: Vec3, durationMs?: number, targetVelocity?: any) =>
          this.camera.lookAtSmooth(target, durationMs, targetVelocity),
        moveAlongSpline: (waypoints: Vec3[], sprint?: boolean) =>
          this.movement.moveAlongSpline(waypoints, sprint),
        pathfindTo: (target: Vec3, range = 2.2, timeoutMs = 8000) =>
          this.pathfindToPosition(target, range, timeoutMs, abortSignal),
        safeDig: (blockOrPos: any) =>
          this.safeDigBlock(blockOrPos, abortSignal),
        safePlaceBlock: (targetPos: Vec3, itemName: string, allowScaffold = true) =>
          this.safePlaceBlock(targetPos, itemName, allowScaffold, abortSignal),
        strikeTarget: (entity: any) => this.combat.strikeTarget(entity),
        equipItemHumanized: (itemName: string, dest?: any) =>
          this.inventory.equipItemHumanized(itemName, dest),
        executeNeuralMotorChunk: (prompt: string, targetPos?: Vec3, maxTicks?: number) =>
          this.executeNeuralMotorChunk(prompt, targetPos, maxTicks),
      },
      world: {
        getGroundHeight: (x: number, z: number, startY?: number) =>
          this.getGroundHeightAt(x, z, startY),
        getTopSolidBlock: (x: number, z: number, startY?: number) =>
          this.getTopSolidBlockAt(x, z, startY),
        scanAreaTopology: (minX: number, minZ: number, maxX: number, maxZ: number) =>
          this.scanAreaTopology(minX, minZ, maxX, maxZ),
      },
      worldState: () => this.getLatestWorldState(),
      reportProgress: (progress: number) => {
        subtask.progress = Math.max(0, Math.min(1, progress));
        this.setSubtaskStatus({ ...subtask });
      },
      sayHumanized,
      abortSignal,
    };
  }

  /**
   * Pure Sensorimotor Steering (Brain-Body Connection):
   * Eliminates procedural script hijacks from mineflayer-pathfinder.
   * System 1 continuously steers the bot at 20Hz towards target (x, y, z):
   * camera aim deltas, planar velocity (vx, vz), sprint-jumping obstacles, and crosshair locking.
   * Cross-attends to the 80.8M Temporal Action Chunking Transformer via executeNeuralMotorChunk.
   */
  public async pathfindToPosition(
    target: Vec3,
    range = 2.2,
    timeoutMs = 7000,
    abortSignal?: AbortSignal
  ): Promise<boolean> {
    if (!this.bot?.entity?.position) return false;

    // Ensure pathfinder goal is cleared so pathfinder does not hijack control
    this.bot.pathfinder?.setGoal(null);

    const start = Date.now();
    let lastPos = this.bot.entity.position.clone();
    let lastActiveTime = Date.now();
    let lastChunkTime = 0;
    let stuckTicks = 0;
    let reached = false;

    while (Date.now() - start < timeoutMs) {
      if (abortSignal?.aborted || !this.bot?.entity?.position) break;

      const botPos = this.bot.entity.position;
      const dx = target.x - botPos.x;
      const dy = target.y - botPos.y;
      const dz = target.z - botPos.z;
      const horizontalDist = Math.hypot(dx, dz);

      // Arrival condition: within target range horizontally and vertically
      if (horizontalDist <= range && Math.abs(dy) <= Math.max(1.8, range)) {
        reached = true;
        break;
      }

      // 1. Camera Aim Deltas & Crosshair Locking (System 1 20Hz continuous look)
      const targetYaw = Math.atan2(-dx, -dz);
      const targetPitch = Math.atan2(dy, Math.max(0.1, horizontalDist));
      try {
        await this.bot.look(targetYaw, targetPitch, true);
      } catch {}

      // 2. Continuous Planar Velocity (vx, vz)
      this.bot.setControlState('forward', horizontalDist > 0.4);
      this.bot.setControlState('sprint', horizontalDist > 3.0);

      // 3. Obstacle Hop, Parkour Gap Jump & Sprint-Jumping (System 1 Sub-Tick Voxel Reflex)
      const forwardDir = new Vec3(-Math.sin(targetYaw), 0, -Math.cos(targetYaw));
      const frontFootPos = botPos.plus(forwardDir.scaled(0.85)).floored();
      const blockAhead = this.bot.blockAt ? this.bot.blockAt(frontFootPos) : null;
      const blockAbove = this.bot.blockAt ? this.bot.blockAt(frontFootPos.offset(0, 1, 0)) : null;
      const blockBelowAhead = this.bot.blockAt ? this.bot.blockAt(frontFootPos.offset(0, -1, 0)) : null;
      const blockBelowAhead2 = this.bot.blockAt ? this.bot.blockAt(frontFootPos.offset(0, -2, 0)) : null;
      const hasObstacleAhead =
        (blockAhead && blockAhead.boundingBox === 'block' && (!blockAbove || blockAbove.boundingBox !== 'block')) ||
        Boolean(this.bot.entity.isCollidedHorizontally);
      const isGapAhead =
        (!blockAhead || blockAhead.boundingBox !== 'block') &&
        (!blockBelowAhead || blockBelowAhead.boundingBox !== 'block') &&
        (!blockBelowAhead2 || blockBelowAhead2.boundingBox !== 'block') &&
        target.y >= botPos.y - 1.2 &&
        horizontalDist > 1.2;

      if (isGapAhead && this.bot.entity.onGround) {
        this.bot.setControlState('sprint', true);
        this.bot.setControlState('jump', true);
      } else if (hasObstacleAhead && this.bot.entity.onGround) {
        this.bot.setControlState('jump', true);
      } else {
        this.bot.setControlState('jump', false);
      }

      // 4. Action Chunk Execution via 80.8M Temporal Action Chunking Transformer
      // Query lock-free /act/chunk every >= 400ms (8 ticks) to tighten continuous motor trajectory
      if (this.vlaClient && Date.now() - lastChunkTime >= 400 && horizontalDist > 1.5) {
        lastChunkTime = Date.now();
        try {
          await this.executeNeuralMotorChunk(
            `Navigate towards target at (${target.x.toFixed(1)}, ${target.y.toFixed(1)}, ${target.z.toFixed(1)})`,
            target,
            2,
            { allowItemDrop: false, allowDig: false }
          );
        } catch {}
      }

      // Check movement progress to prevent infinite loops
      const distFromLast = this.bot.entity.position.distanceTo(lastPos);
      if (distFromLast > 0.08) {
        lastActiveTime = Date.now();
        lastPos = this.bot.entity.position.clone();
        stuckTicks = 0;
      } else {
        stuckTicks++;
        if (stuckTicks > 4 && this.bot.entity.onGround) {
          this.bot.setControlState('jump', true);
        }
        if (Date.now() - lastActiveTime > 2500) {
          break;
        }
      }

      // 20Hz server physics tick rate (50ms)
      await new Promise((r) => setTimeout(r, 50));
    }

    // Clean up motor controls on completion
    this.bot.pathfinder?.setGoal(null);
    this.bot.clearControlStates?.();

    if (!this.bot?.entity?.position) return reached;
    const finalHoriz = Math.hypot(target.x - this.bot.entity.position.x, target.z - this.bot.entity.position.z);
    const finalDy = Math.abs(target.y - this.bot.entity.position.y);
    return reached || (finalHoriz <= range + 0.8 && finalDy <= Math.max(2.0, range + 0.8));
  }

  /**
   * Queries the 80.8M Action Chunking Transformer via fast lock-free /act/chunk (~1.5ms)
   * to execute continuous motor trajectory ticks oriented cleanly toward the optional targetPos.
   */
  public async executeNeuralMotorChunk(
    prompt: string,
    targetPos?: Vec3,
    maxTicks = 2,
    options?: { allowItemDrop?: boolean; allowDig?: boolean }
  ): Promise<boolean> {
    if (!this.bot?.entity) return false;
    try {
      const motorControls = await this.vlaClient.predictActionChunk(prompt, this.bot, targetPos);
      if (motorControls?.action_chunk && motorControls.action_chunk.length > 0) {
        await this.vlaClient.executeMotorActionChunk(
          this.bot,
          motorControls.action_chunk,
          targetPos,
          maxTicks,
          options
        );
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }

  /**
   * Selects and equips the optimal tool from inventory for the given block name.
   */
  public async equipBestToolForBlock(blockName: string): Promise<void> {
    if (!this.bot?.entity || !this.bot.inventory) return;
    const n = (blockName || '').toLowerCase();
    const inv = this.bot.inventory.items();

    // 1. Wood logs, stems, planks, crafting table
    if (n.includes('log') || n.includes('stem') || n.includes('planks') || n === 'crafting_table') {
      const axeOrder = ['netherite_axe', 'diamond_axe', 'iron_axe', 'stone_axe', 'wooden_axe'];
      for (const a of axeOrder) {
        if (inv.some((i: any) => i.name === a)) {
          await this.inventory.equipItemHumanized(a, 'hand');
          return;
        }
      }
      return;
    }

    // 2. Dirt, grass, podzol, sand, gravel, clay
    if (
      n.includes('dirt') ||
      n.includes('grass_block') ||
      n.includes('podzol') ||
      (n.includes('sand') && !n.includes('sandstone')) ||
      n.includes('gravel') ||
      n.includes('clay')
    ) {
      const shovelOrder = ['netherite_shovel', 'diamond_shovel', 'iron_shovel', 'stone_shovel', 'wooden_shovel'];
      for (const s of shovelOrder) {
        if (inv.some((i: any) => i.name === s)) {
          await this.inventory.equipItemHumanized(s, 'hand');
          return;
        }
      }
      // If no shovel, avoid wasting pickaxe/sword durability on dirt
      const held = this.bot.heldItem?.name || '';
      if (held.includes('pickaxe') || held.includes('sword')) {
        const nonTool = inv.find(
          (i: any) =>
            !i.name.includes('pickaxe') &&
            !i.name.includes('sword') &&
            !i.name.includes('axe') &&
            !i.name.includes('shovel')
        );
        if (nonTool) {
          try {
            await this.bot.equip(nonTool, 'hand');
          } catch {}
        } else {
          try {
            await this.bot.unequip('hand');
          } catch {}
        }
      }
      return;
    }

    // 3. Stone, cobblestone, ores, deepslate
    if (
      n.includes('stone') ||
      n.includes('cobble') ||
      n.includes('deepslate') ||
      n.includes('_ore') ||
      n === 'furnace' ||
      n === 'blast_furnace' ||
      n === 'smoker'
    ) {
      const pickOrder = ['netherite_pickaxe', 'diamond_pickaxe', 'iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe'];
      for (const p of pickOrder) {
        if (inv.some((i: any) => i.name === p)) {
          await this.inventory.equipItemHumanized(p, 'hand');
          return;
        }
      }
      return;
    }

    // 4. Leaves, cobweb
    if (n.includes('leaves') || n.includes('cobweb')) {
      const shearsOrSword = ['shears', 'netherite_sword', 'diamond_sword', 'iron_sword', 'stone_sword', 'wooden_sword'];
      for (const item of shearsOrSword) {
        if (inv.some((i: any) => i.name === item)) {
          await this.inventory.equipItemHumanized(item, 'hand');
          return;
        }
      }
    }
  }

  /**
   * Reliable server-verified block digging + drop pickup helper exposed on `ctx.kinematics.safeDig`.
   */
  public async safeDigBlock(blockOrPos: any, abortSignal?: AbortSignal): Promise<boolean> {
    if (!this.bot?.entity || abortSignal?.aborted) return false;
    const pos: Vec3 = blockOrPos?.position ? blockOrPos.position : blockOrPos;
    if (!pos) return false;

    const isNonSolidOrFluid = (b: any) =>
      !b ||
      b.name === 'air' ||
      b.name === 'cave_air' ||
      b.name === 'void_air' ||
      String(b.name || '').includes('water') ||
      String(b.name || '').includes('lava');

    const block = this.bot.blockAt(pos);
    if (isNonSolidOrFluid(block)) return true;
    if (block.diggable === false || block.name === 'bedrock') return false;

    this.bot.pathfinder?.setGoal(null);
    this.bot.clearControlStates?.();
    this.camera?.clearQueue();

    // Ensure within reach (<= 4.2m)
    const eyePos = this.bot.entity.position.offset(0, 1.62, 0);
    const blockCenter = pos.offset(0.5, 0.5, 0.5);
    if (eyePos.distanceTo(blockCenter) > 4.2) {
      await this.pathfindToPosition(pos, 2.2, 5500, abortSignal);
    }

    for (let retry = 0; retry < 2; retry++) {
      if (abortSignal?.aborted || !this.bot?.entity) return false;

      this.bot.pathfinder?.setGoal(null);
      this.bot.clearControlStates?.();

      // Settle on ground to avoid 5x airborne mining penalty
      const gStart = Date.now();
      while (!this.bot.entity.onGround && Date.now() - gStart < 350) {
        await new Promise((r) => setTimeout(r, 40));
      }

      const freshBlock = this.bot.blockAt(pos);
      if (isNonSolidOrFluid(freshBlock)) {
        return true;
      }
      if (freshBlock.diggable === false || freshBlock.name === 'bedrock') {
        return false;
      }

      // Automatically select and equip the best tool for this block type
      await this.equipBestToolForBlock(freshBlock.name);
      await new Promise((r) => setTimeout(r, 70));

      const dist = this.bot.entity.position.offset(0, 1.62, 0).distanceTo(blockCenter);
      if (dist > 4.8 || !this.bot.canDigBlock(freshBlock)) {
        logVerbose('DIG', `Cannot dig ${freshBlock.name} at ${this.formatPos(pos)} (dist=${dist.toFixed(2)}m)`);
        return false;
      }

      let serverReverted = false;
      const onBlockChange = (packet: any) => {
        if (
          packet?.location &&
          packet.location.x === pos.x &&
          packet.location.y === pos.y &&
          packet.location.z === pos.z &&
          packet.type !== 0
        ) {
          serverReverted = true;
        }
      };
      this.bot._client?.on('block_change', onBlockChange);

      logVerbose('DIG', `Starting dig (try #${retry + 1}) on ${freshBlock.name} at ${this.formatPos(pos)} (dist=${dist.toFixed(2)}m, held=${this.bot.heldItem?.name || 'hand'})`);

      let timeoutId: ReturnType<typeof setTimeout> | null = null;
      let digTimeoutMs = 8000;
      if (
        freshBlock.name.includes('obsidian') ||
        freshBlock.name.includes('ancient_debris') ||
        freshBlock.name.includes('respawn_anchor')
      ) {
        digTimeoutMs = 20000;
      } else if (typeof (this.bot as any).digTime === 'function') {
        const estTime = (this.bot as any).digTime(freshBlock);
        if (typeof estTime === 'number' && estTime > 0) {
          digTimeoutMs = Math.max(8000, estTime + 3000);
        }
      }

      try {
        // Crosshair locking onto block center and arm swinging (System 1 Sensorimotor driving)
        try {
          await this.bot.lookAt(blockCenter, true);
          this.bot.swingArm('right');
        } catch {}

        await Promise.race([
          this.bot.dig(freshBlock, true),
          new Promise((_, reject) => {
            timeoutId = setTimeout(() => {
              try {
                this.bot.stopDigging();
              } catch {}
              reject(new Error(`Dig timeout ${digTimeoutMs}ms`));
            }, digTimeoutMs);
          }),
        ]);
      } catch (e: any) {
        logVerbose('DIG', `Dig error on ${freshBlock.name} (try #${retry + 1}): ${e.message}`);
      } finally {
        if (timeoutId) clearTimeout(timeoutId);
        this.bot._client?.removeListener('block_change', onBlockChange);
      }

      // Wait 140ms (~3 server ticks) for server block-change confirmation
      await new Promise((r) => setTimeout(r, 140));
      const after = this.bot.blockAt(pos);
      const broke = isNonSolidOrFluid(after) && !serverReverted;
      logVerbose('DIG', `Dig check on ${freshBlock.name} at ${this.formatPos(pos)} -> broke=${broke} (serverReverted=${serverReverted}, now=${after?.name || 'air'})`);
      if (broke) {
        return true;
      }
    }

    return false;
  }

  /**
   * Scans downwards from startY at coordinate (x, z) to find the topmost solid ground block.
   */
  public getGroundHeightAt(x: number, z: number, startY?: number): number {
    const fallbackY = Math.floor(this.bot?.entity?.position?.y ?? 64) - 1;
    if (!this.bot) return fallbackY;

    const maxY = startY ?? Math.min(256, Math.floor(this.bot.entity?.position?.y ?? 64) + 12);
    const minY = Math.max(-64, maxY - 30);

    for (let y = maxY; y >= minY; y--) {
      const b = this.bot.blockAt(new Vec3(x, y, z));
      if (!b) continue;
      // Skip air, water, lava, tall grass, flowers, saplings, leaves
      if (
        b.name === 'air' ||
        b.name === 'cave_air' ||
        b.name.includes('water') ||
        b.name.includes('lava') ||
        b.name === 'grass' ||
        b.name.includes('short_grass') ||
        b.name.includes('tall_grass') ||
        b.name.includes('seagrass') ||
        b.name.includes('flower') ||
        b.name.includes('tulip') ||
        b.name.includes('orchid') ||
        b.name.includes('dandelion') ||
        b.name.includes('rose') ||
        b.name.includes('fern') ||
        b.name.includes('sapling') ||
        b.name.includes('bush') ||
        b.name.includes('leaves') ||
        b.name.includes('mushroom')
      ) {
        continue;
      }
      if (b.boundingBox === 'block') {
        return y;
      }
    }
    return fallbackY;
  }

  /**
   * Returns the topmost solid block at (x, z).
   */
  public getTopSolidBlockAt(x: number, z: number, startY?: number): any | null {
    const gy = this.getGroundHeightAt(x, z, startY);
    return this.bot?.blockAt ? this.bot.blockAt(new Vec3(x, gy, z)) : null;
  }

  /**
   * Comprehensive area topology scan:
   * Returns a 2D height grid, min/max/median elevations, flatness, and obstacle detection.
   */
  public scanAreaTopology(
    minX: number,
    minZ: number,
    maxX: number,
    maxZ: number
  ): AreaTopologyScan {
    const x0 = Math.min(minX, maxX);
    const x1 = Math.max(minX, maxX);
    const z0 = Math.min(minZ, maxZ);
    const z1 = Math.max(minZ, maxZ);

    const botY = Math.floor(this.bot?.entity?.position?.y ?? 64);
    const heights: number[] = [];
    const heightGrid: number[][] = [];
    const groundBlockTypes: Record<string, number> = {};
    let obstructionCount = 0;
    let airGapCount = 0;

    for (let x = x0; x <= x1; x++) {
      const row: number[] = [];
      for (let z = z0; z <= z1; z++) {
        const gy = this.getGroundHeightAt(x, z, botY + 8);
        row.push(gy);
        heights.push(gy);

        const groundBlock = this.bot?.blockAt ? this.bot.blockAt(new Vec3(x, gy, z)) : null;
        if (groundBlock?.name) {
          groundBlockTypes[groundBlock.name] = (groundBlockTypes[groundBlock.name] || 0) + 1;
        }

        // Check for obstructions above ground (between gy + 1 and gy + 3)
        if (this.bot?.blockAt) {
          for (let dy = 1; dy <= 3; dy++) {
            const ab = this.bot.blockAt(new Vec3(x, gy + dy, z));
            if (ab && ab.name !== 'air' && ab.name !== 'cave_air') {
              obstructionCount++;
            }
          }
        }

        if (gy < botY - 1) {
          airGapCount++;
        }
      }
      heightGrid.push(row);
    }

    heights.sort((a, b) => a - b);
    const minHeight = heights[0] ?? botY - 1;
    const maxHeight = heights[heights.length - 1] ?? botY - 1;
    const medianHeight = heights[Math.floor(heights.length / 2)] ?? botY - 1;
    const isFlat = maxHeight - minHeight <= 1;

    return {
      minX: x0,
      minZ: z0,
      maxX: x1,
      maxZ: z1,
      minHeight,
      maxHeight,
      medianHeight,
      isFlat,
      obstructionCount,
      airGapCount,
      groundBlockTypes,
      heightGrid,
    };
  }

  /**
   * Reach-aware, collision-free, robust block placement helper.
   * 1. If target is already desired block, returns true.
   * 2. Finds adjacent solid reference block & placement face normal vector.
   *    If floating and allowScaffold=true, recurses once to place a foundation block below.
   * 3. Calculates a safe standing position within 1.5 - 3.2m of targetPos that does not
   *    intersect targetPos or reference block.
   * 4. Dynamically pathfinds to that position if bot is > 3.2m away or colliding.
   * 5. Equips itemName, aims smoothly at the face, and places the block via bot.placeBlock.
   * 6. Confirms placement on server.
   */
  public async safePlaceBlock(
    targetPos: Vec3,
    itemName: string,
    allowScaffold = true,
    abortSignal?: AbortSignal
  ): Promise<boolean> {
    if (abortSignal?.aborted || !this.bot?.entity) return false;
    const targetFloored = targetPos.floored();

    const cleanName = itemName.replace('minecraft:', '').toLowerCase();
    const currentBlock = this.bot.blockAt(targetFloored);
    if (currentBlock && currentBlock.name.includes(cleanName)) {
      return true; // Already placed!
    }

    // Directions to find solid reference block to place against
    const dirs: Array<{ offset: Vec3; faceVec: Vec3 }> = [
      { offset: new Vec3(0, -1, 0), faceVec: new Vec3(0, 1, 0) },   // Place on top of block below
      { offset: new Vec3(0, 1, 0), faceVec: new Vec3(0, -1, 0) },   // Place against ceiling
      { offset: new Vec3(1, 0, 0), faceVec: new Vec3(-1, 0, 0) },   // Place against East neighbor
      { offset: new Vec3(-1, 0, 0), faceVec: new Vec3(1, 0, 0) },   // Place against West neighbor
      { offset: new Vec3(0, 0, 1), faceVec: new Vec3(0, 0, -1) },   // Place against South neighbor
      { offset: new Vec3(0, 0, -1), faceVec: new Vec3(0, 0, 1) },   // Place against North neighbor
    ];

    let refBlock: any = null;
    let faceVector: Vec3 | null = null;

    for (const d of dirs) {
      const neighborPos = targetFloored.plus(d.offset);
      const nb = this.bot.blockAt(neighborPos);
      if (
        nb &&
        nb.name !== 'air' &&
        nb.name !== 'cave_air' &&
        !nb.name.includes('water') &&
        !nb.name.includes('lava') &&
        nb.boundingBox === 'block'
      ) {
        refBlock = nb;
        faceVector = d.faceVec;
        break;
      }
    }

    // If no solid neighbor exists (floating block in mid-air):
    if (!refBlock && allowScaffold) {
      const belowPos = targetFloored.offset(0, -1, 0);
      const bBlock = this.bot.blockAt(belowPos);
      if (!bBlock || bBlock.name === 'air' || bBlock.name === 'cave_air') {
        // Place scaffold/foundation block beneath
        await this.safePlaceBlock(belowPos, itemName, false, abortSignal);
      }
      // Re-check after scaffold
      for (const d of dirs) {
        const neighborPos = targetFloored.plus(d.offset);
        const nb = this.bot.blockAt(neighborPos);
        if (
          nb &&
          nb.name !== 'air' &&
          nb.name !== 'cave_air' &&
          !nb.name.includes('water') &&
          !nb.name.includes('lava') &&
          nb.boundingBox === 'block'
        ) {
          refBlock = nb;
          faceVector = d.faceVec;
          break;
        }
      }
    }

    if (!refBlock || !faceVector) {
      return false;
    }

    // Compute reach and body collision
    // (If bot is standing/jumping above targetFloored.y + 0.9, e.g. pillaring up, it is not inside the block volume)
    const botPos = this.bot.entity.position;
    const dist = botPos.distanceTo(targetFloored.offset(0.5, 0.5, 0.5));
    const isCollidingWithBot =
      Math.abs(botPos.x - (targetFloored.x + 0.5)) < 0.75 &&
      Math.abs(botPos.z - (targetFloored.z + 0.5)) < 0.75 &&
      botPos.y >= targetFloored.y - 0.5 &&
      botPos.y < targetFloored.y + 0.9;

    if (dist > 3.4 || isCollidingWithBot) {
      // Find a safe standing position within 1.5 - 3.2m of targetFloored
      let bestStand: Vec3 | null = null;
      let bestScore = 999;

      const standOffsets = [
        new Vec3(1.6, 0, 0),
        new Vec3(-1.6, 0, 0),
        new Vec3(0, 0, 1.6),
        new Vec3(0, 0, -1.6),
        new Vec3(1.6, 0, 1.6),
        new Vec3(-1.6, 0, -1.6),
        new Vec3(2.2, 0, 0),
        new Vec3(-2.2, 0, 0),
        new Vec3(0, 0, 2.2),
        new Vec3(0, 0, -2.2),
      ];

      for (const offset of standOffsets) {
        const checkPos = targetFloored.plus(offset);
        const groundY = this.getGroundHeightAt(Math.floor(checkPos.x), Math.floor(checkPos.z), Math.floor(targetFloored.y + 2));
        const candidate = new Vec3(Math.floor(checkPos.x) + 0.5, groundY + 1, Math.floor(checkPos.z) + 0.5);
        const cDist = candidate.distanceTo(targetFloored.offset(0.5, 0.5, 0.5));

        if (Math.floor(candidate.x) === targetFloored.x && Math.floor(candidate.z) === targetFloored.z) {
          continue;
        }

        if (cDist >= 1.3 && cDist <= 3.4) {
          const moveDist = botPos.distanceTo(candidate);
          if (moveDist < bestScore) {
            bestScore = moveDist;
            bestStand = candidate;
          }
        }
      }

      if (bestStand) {
        await this.pathfindToPosition(bestStand, 1.0, 4000, abortSignal);
      }
    }

    // Equip item
    let equipped = await this.inventory.equipItemHumanized(itemName, 'hand');
    if (!equipped) {
      const invItem = (this.bot.inventory?.items() || []).find((i: any) =>
        i.name.includes(cleanName)
      );
      if (invItem) {
        equipped = await this.inventory.equipItemHumanized(invItem.name, 'hand');
      }
      if (!equipped) return false;
    }

    try {
      const aimTarget = refBlock.position.offset(
        0.5 + faceVector.x * 0.45,
        0.5 + faceVector.y * 0.45,
        0.5 + faceVector.z * 0.45
      );
      await this.camera.lookAtSmooth(aimTarget, 120);
      await this.bot.placeBlock(refBlock, faceVector);
      await new Promise((r) => setTimeout(r, 60));

      const afterBlock = this.bot.blockAt(targetFloored);
      return Boolean(
        afterBlock &&
          afterBlock.name !== 'air' &&
          afterBlock.name !== 'cave_air' &&
          afterBlock.name !== 'void_air'
      );
    } catch {
      return false;
    }
  }

  private formatPos(p?: { x: number; y: number; z: number } | null): string {
    if (!p || isNaN(p.x) || isNaN(p.y) || isNaN(p.z)) return '(unknown)';
    return `(${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)})`;
  }
}
