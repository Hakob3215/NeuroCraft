import mineflayer from 'mineflayer';
import { Vec3 } from 'vec3';
import { MINECRAFT_1_20_CUSTOM_PACKETS } from '../src/protocolFixes';

export type SparringMode =
  | 'idle'
  | 'pvp_dqn'
  | 'pvp_self'
  | 'companion_wait'
  | 'bridge_race'
  | 'escort_race'
  | 'resource_race';

export interface SparringTaskOptions {
  mode: SparringMode;
  targetUsername?: string;
  goalPos?: { x: number; y: number; z: number };
  targetBlockName?: string;
}

export interface MultiplayerSnapshot {
  opponentName: string;
  opponentHealth: number;
  opponentAlive: boolean;
  opponentKills: number;
  opponentAtGoal: boolean;
  opponentBlocksCollected: number;
  opponentInventory?: Array<{ name: string; count: number }>;
}

const HAZARD_BLOCKS = new Set([
  'lava',
  'flowing_lava',
  'fire',
  'soul_fire',
  'magma_block',
  'cactus',
  'sweet_berry_bush',
]);

const PASSABLE_BLOCKS = new Set([
  'air',
  'cave_air',
  'void_air',
  'water',
  'flowing_water',
  'short_grass',
  'grass',
  'tall_grass',
  'fern',
  'torch',
  'wall_torch',
]);

const ACTION_NAMES = ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'attack'] as const;

export class SparringPartnerBot {
  public readonly username: string;
  private readonly daemonUrl: string;
  public bot: mineflayer.Bot | null = null;

  private mode: SparringMode = 'idle';
  private targetUsername = 'NeuroAgent';
  private goalPos: { x: number; y: number; z: number } | null = null;
  private targetBlockName = 'oak_log';

  private tickTimer: NodeJS.Timeout | null = null;
  private isTicking = false;
  private isDigging = false;
  private lastAttackTime = 0;
  private lastWeaponCheck = 0;

  public killsAgainstAgent = 0;
  public blocksCollected = 0;
  public atGoal = false;
  public alive = true;
  private defeatedThisEpisode = false;

  constructor(username = 'DQN_Opponent', daemonUrl = 'http://127.0.0.1:8765') {
    this.username = username;
    this.daemonUrl = daemonUrl;
  }

  public async connect(host = '127.0.0.1', port = 25575): Promise<void> {
    if (this.bot) return;

    await new Promise<void>((resolve, reject) => {
      const bot = mineflayer.createBot({
        host,
        port,
        username: this.username,
        version: '1.20.4',
        auth: 'offline',
        customPackets: MINECRAFT_1_20_CUSTOM_PACKETS,
      } as any);
      this.bot = bot;

      const onError = (err: Error) => {
        cleanup();
        this.bot = null;
        reject(err);
      };
      const onSpawn = () => {
        cleanup();
        this.alive = true;
        resolve();
      };
      const cleanup = () => {
        bot.removeListener('error', onError);
        bot.removeListener('spawn', onSpawn);
      };

      bot.once('error', onError);
      bot.once('spawn', onSpawn);

      bot.on('death', () => {
        this.alive = false;
        this.defeatedThisEpisode = true;
        bot.clearControlStates();
      });

      bot.on('respawn', () => {
        this.alive = true;
      });

      bot.on('error', (err) => {
        console.warn(`[SparringPartner:${this.username}] socket/bot error: ${err.message}`);
      });

      bot.on('kicked', (reason) => {
        const reasonStr = typeof reason === 'object' ? JSON.stringify(reason) : String(reason);
        console.warn(`[SparringPartner:${this.username}] kicked: ${reasonStr}`);
      });
    });

    if (!this.tickTimer) {
      this.tickTimer = setInterval(() => {
        void this.step();
      }, 50);
    }
  }

  public async disconnect(): Promise<void> {
    if (this.tickTimer) {
      clearInterval(this.tickTimer);
      this.tickTimer = null;
    }
    this.mode = 'idle';
    if (this.bot) {
      try {
        this.bot.clearControlStates();
        this.bot.quit('Training teardown');
      } catch {
        try {
          this.bot.end();
        } catch {
          // ignore
        }
      }
      this.bot = null;
    }
  }

  public resetEpisodeCounters(): void {
    this.killsAgainstAgent = 0;
    this.blocksCollected = 0;
    this.atGoal = false;
    this.alive = true;
    this.defeatedThisEpisode = false;
    this.isDigging = false;
    this.lastAttackTime = 0;
  }

  public setMode(options: SparringTaskOptions): void {
    this.mode = options.mode;
    this.targetUsername = options.targetUsername ?? 'NeuroAgent';
    this.goalPos = options.goalPos ?? null;
    this.targetBlockName = options.targetBlockName ?? 'oak_log';
    this.isDigging = false;

    if (this.bot) {
      try {
        this.bot.clearControlStates();
      } catch {
        // ignore
      }
    }
  }

  public recordAgentDeath(): void {
    if (this.mode !== 'idle') {
      this.killsAgainstAgent += 1;
    }
  }

  public getSnapshot(): MultiplayerSnapshot {
    const health = this.bot?.health ?? 0;
    const inv = (this.bot?.inventory?.items() || []).map((i) => ({
      name: i.name,
      count: i.count,
    }));
    return {
      opponentName: this.username,
      opponentHealth: this.defeatedThisEpisode ? 0 : health,
      opponentAlive: !this.defeatedThisEpisode && this.alive && health > 0,
      opponentKills: this.killsAgainstAgent,
      opponentAtGoal: this.atGoal,
      opponentBlocksCollected: this.blocksCollected,
      opponentInventory: inv,
    };
  }

  private async step(): Promise<void> {
    if (this.isTicking || !this.bot || !this.bot.entity || this.mode === 'idle') {
      return;
    }
    if (this.defeatedThisEpisode || !this.alive || (this.bot.health ?? 20) <= 0) {
      return;
    }

    this.isTicking = true;
    try {
      if (this.mode === 'pvp_dqn') {
        await this.stepPvpDqn();
      } else if (this.mode === 'pvp_self') {
        await this.stepPvpSelf();
      } else if (this.mode === 'companion_wait') {
        const targetPlayer = this.bot.players[this.targetUsername]?.entity;
        const nearbyItem = this.bot.nearestEntity(
          (e: any) =>
            e &&
            (e.name === 'item' || e.type === 'object') &&
            this.bot!.entity.position.distanceTo(e.position) <= 12.0
        );

        if (nearbyItem) {
          const dx = nearbyItem.position.x - this.bot.entity.position.x;
          const dz = nearbyItem.position.z - this.bot.entity.position.z;
          const dist = Math.sqrt(dx * dx + dz * dz);
          if (dist > 0.25) {
            await this.bot.lookAt(nearbyItem.position.offset(0, 0.1, 0), true).catch(() => {});
            this.bot.setControlState('forward', true);
            this.bot.setControlState('sprint', false);
          } else {
            this.bot.clearControlStates();
          }
        } else {
          this.bot.clearControlStates();
          if (targetPlayer) {
            await this.bot.lookAt(targetPlayer.position.offset(0, 1.5, 0), true).catch(() => {});
          }
        }
      } else if (this.mode === 'bridge_race') {
        await this.stepBridgeRace();
      } else if (this.mode === 'escort_race') {
        await this.stepEscortRace();
      } else if (this.mode === 'resource_race') {
        await this.stepResourceRace();
      }
    } catch {
      // Keep sparring loop resilient on transient tick errors
    } finally {
      this.isTicking = false;
    }
  }

  private async equipBestWeapon(): Promise<void> {
    if (!this.bot) return;
    const now = Date.now();
    if (now - this.lastWeaponCheck < 1500) return;
    this.lastWeaponCheck = now;

    const priority = [
      'netherite_sword',
      'diamond_sword',
      'iron_sword',
      'stone_sword',
      'wooden_sword',
      'diamond_axe',
      'iron_axe',
      'stone_axe',
    ];
    const items = this.bot.inventory.items();
    for (const name of priority) {
      const match = items.find((i) => i.name === name);
      if (match) {
        if (this.bot.heldItem?.name !== match.name) {
          await this.bot.equip(match, 'hand').catch(() => {});
        }
        return;
      }
    }
  }

  private extractDqnState55(enemy: mineflayer.Bot['entity'] | null): number[] {
    const bot = this.bot!;
    const state: number[] = [];
    const base = bot.entity.position.floored();

    // 3x5x3 voxel grid matching GeorgeDudau/Minecraft-AI-Agent voxelScanner.js
    for (let x = -1; x <= 1; x++) {
      for (let y = -1; y <= 3; y++) {
        for (let z = -1; z <= 1; z++) {
          const blk = bot.blockAt(base.offset(x, y, z));
          const name = blk?.name ?? 'air';
          if (HAZARD_BLOCKS.has(name)) {
            state.push(-1.0);
          } else if (PASSABLE_BLOCKS.has(name) || blk?.boundingBox === 'empty') {
            state.push(0.0);
          } else {
            state.push(1.0);
          }
        }
      }
    }

    const health = (bot.health ?? 20) / 20.0;
    const onGround = bot.entity.onGround ? 1.0 : 0.0;
    const cooldownActive = Date.now() - this.lastAttackTime < 625 ? 1.0 : 0.0;
    const velY = bot.entity.onGround ? 0.0 : (bot.entity.velocity?.y ?? 0.0) / 3.92;
    state.push(health, onGround, cooldownActive, velY);

    if (enemy) {
      const dx = enemy.position.x - bot.entity.position.x;
      const dy = enemy.position.y - bot.entity.position.y;
      const dz = enemy.position.z - bot.entity.position.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const yawDiff = ((enemy.yaw ?? 0) - (bot.entity.yaw ?? 0)) / 6.284;
      const rawHp = typeof (enemy as any).metadata?.[9] === 'number' ? (enemy as any).metadata[9] : 20;
      const enemyHealth = Math.max(0, Math.min(1, rawHp / 20.0));
      state.push(dist / 16.0, dx / 16.0, dy / 16.0, dz / 16.0, yawDiff, enemyHealth);
    } else {
      state.push(1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    }

    return state;
  }

  private async stepPvpDqn(): Promise<void> {
    const bot = this.bot!;
    await this.equipBestWeapon();

    const targetPlayer = bot.players[this.targetUsername]?.entity ?? null;
    if (!targetPlayer) {
      bot.clearControlStates();
      return;
    }

    const dist = bot.entity.position.distanceTo(targetPlayer.position);
    await bot.lookAt(targetPlayer.position.offset(0, 1.5, 0), true).catch(() => {});

    const state = this.extractDqnState55(targetPlayer);
    let chosenAction: (typeof ACTION_NAMES)[number] = 'forward';

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 150);
      const resp = await fetch(`${this.daemonUrl}/dqn_sparring/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state }),
        signal: controller.signal,
      });
      clearTimeout(timeout);
      if (resp.ok) {
        const data = (await resp.json()) as { action_name?: string; action?: number; action_id?: number };
        const actionIdx = data.action ?? data.action_id;
        if (
          data.action_name &&
          (ACTION_NAMES as readonly string[]).includes(data.action_name)
        ) {
          chosenAction = data.action_name as (typeof ACTION_NAMES)[number];
        } else if (
          typeof actionIdx === 'number' &&
          actionIdx >= 0 &&
          actionIdx < ACTION_NAMES.length
        ) {
          chosenAction = ACTION_NAMES[actionIdx];
        }
      }
    } catch {
      // Deterministic fallback if daemon is temporarily paused/reloading
      chosenAction = dist <= 3.3 ? 'attack' : 'sprint';
    }

    bot.clearControlStates();

    if (chosenAction === 'attack') {
      if (dist <= 3.5 && Date.now() - this.lastAttackTime > 600) {
        bot.attack(targetPlayer);
        this.lastAttackTime = Date.now();
      } else if (dist > 3.5) {
        bot.setControlState('forward', true);
        bot.setControlState('sprint', true);
      }
    } else if (chosenAction === 'sprint') {
      bot.setControlState('forward', true);
      bot.setControlState('sprint', true);
    } else if (chosenAction === 'jump') {
      bot.setControlState('forward', true);
      bot.setControlState('jump', true);
    } else {
      bot.setControlState(chosenAction, true);
    }

    // Opportunistic hit when in melee reach and cooldown is ready
    if (dist <= 3.1 && Date.now() - this.lastAttackTime > 650) {
      bot.attack(targetPlayer);
      this.lastAttackTime = Date.now();
    }
  }

  private async stepPvpSelf(): Promise<void> {
    const bot = this.bot!;
    await this.equipBestWeapon();

    const targetPlayer = bot.players[this.targetUsername]?.entity ?? null;
    if (!targetPlayer) {
      bot.clearControlStates();
      return;
    }

    const dist = bot.entity.position.distanceTo(targetPlayer.position);
    await bot.lookAt(targetPlayer.position.offset(0, 1.5, 0), true).catch(() => {});

    bot.clearControlStates();

    // Humanized mirror sparring kinematics
    if (dist > 3.2) {
      bot.setControlState('forward', true);
      bot.setControlState('sprint', true);
      if ((bot.entity as any).isCollidedHorizontally) {
        bot.setControlState('jump', true);
      }
    } else {
      // Within melee engagement: dynamic lateral circle-strafe + attack on 625ms cooldown
      const strafe = Math.sin(Date.now() / 280) > 0 ? 'left' : 'right';
      bot.setControlState(strafe, true);
      if (dist > 2.4) bot.setControlState('forward', true);
      else if (dist < 1.6) bot.setControlState('back', true);

      // 30% chance to raise shield if holding shield in off-hand
      const shield = bot.inventory.items().find((i) => i.name === 'shield');
      if (shield && Math.random() < 0.3 && Date.now() - this.lastAttackTime > 450) {
        bot.activateItem(true);
        setTimeout(() => {
          try {
            bot.deactivateItem();
          } catch {}
        }, 300);
      }

      if (dist <= 3.5 && Date.now() - this.lastAttackTime > 625) {
        bot.attack(targetPlayer);
        this.lastAttackTime = Date.now();
      }
    }
  }

  private async stepBridgeRace(): Promise<void> {
    const bot = this.bot!;
    if (!this.goalPos) return;
    const goal = new Vec3(this.goalPos.x, this.goalPos.y, this.goalPos.z);
    if (bot.entity.position.distanceTo(goal) <= 2.5) {
      this.atGoal = true;
      bot.clearControlStates();
      return;
    }

    await bot.lookAt(goal.offset(0, 1, 0), true).catch(() => {});

    const dir = goal.minus(bot.entity.position);
    const len = Math.hypot(dir.x, dir.z) || 1;
    const aheadX = Math.floor(bot.entity.position.x + (dir.x / len) * 1.1);
    const aheadZ = Math.floor(bot.entity.position.z + (dir.z / len) * 1.1);
    const supportPos = new Vec3(aheadX, Math.floor(bot.entity.position.y) - 1, aheadZ);
    const supportBlock = bot.blockAt(supportPos);

    if (supportBlock && supportBlock.name === 'air') {
      bot.setControlState('sneak', true);
      const blockItem = bot.inventory
        .items()
        .find((i) => !i.name.includes('sword') && !i.name.includes('pickaxe'));
      if (blockItem) {
        await bot.equip(blockItem, 'hand').catch(() => {});
        const underFeet = bot.blockAt(bot.entity.position.floored().offset(0, -1, 0));
        if (underFeet && underFeet.name !== 'air') {
          const dx = supportPos.x - underFeet.position.x;
          const dz = supportPos.z - underFeet.position.z;
          if (dx !== 0 || dz !== 0) {
            const face =
              Math.abs(dz) >= Math.abs(dx)
                ? new Vec3(0, 0, Math.sign(dz))
                : new Vec3(Math.sign(dx), 0, 0);
            await bot.placeBlock(underFeet, face).catch(() => {});
          }
        }
      }
    } else {
      bot.setControlState('sneak', false);
    }

    bot.setControlState('forward', true);
  }

  private async stepEscortRace(): Promise<void> {
    const bot = this.bot!;
    if (!this.goalPos) return;
    const goal = new Vec3(this.goalPos.x, this.goalPos.y, this.goalPos.z);
    if (bot.entity.position.distanceTo(goal) <= 3.0) {
      this.atGoal = true;
      bot.clearControlStates();
      return;
    }

    await bot.lookAt(goal.offset(0, 1, 0), true).catch(() => {});
    bot.setControlState('forward', true);
    bot.setControlState('sprint', true);
  }

  private async stepResourceRace(): Promise<void> {
    const bot = this.bot!;
    if (this.isDigging) return;

    const target = bot.findBlock({
      matching: (b) => b.name === this.targetBlockName,
      maxDistance: 16,
    });
    if (!target) {
      bot.clearControlStates();
      return;
    }

    const dist = bot.entity.position.distanceTo(target.position.offset(0.5, 0.5, 0.5));
    await bot.lookAt(target.position.offset(0.5, 0.5, 0.5), true).catch(() => {});

    if (dist > 3.5) {
      bot.setControlState('forward', true);
      bot.setControlState('sprint', true);
      return;
    }

    bot.clearControlStates();
    this.isDigging = true;
    try {
      await bot.dig(target);
      this.blocksCollected += 1;
    } catch {
      // ignore dig interruption
    } finally {
      this.isDigging = false;
    }
  }
}
