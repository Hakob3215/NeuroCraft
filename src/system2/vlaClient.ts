/**
 * NeuroCraft Dual-Channel VLA (Vision-Language-Action) Client
 * ==========================================================
 * Connects the high-level System 2 multi-modal cognitive core to Minecraft.
 * Features a dual-channel architecture:
 *   - Channel 1 (Natural Dialogue): Speaks to players via chat ("Sure, dropping 5 torches!").
 *   - Channel 2 (Action Execution): High-speed discrete inventory actions and continuous 20Hz motor controls.
 *
 * Guaranteed ZERO short-circuiting on everyday companion requests.
 */

import { Bot } from 'mineflayer';
import { Vec3 } from 'vec3';

export interface VLAParsedAction {
  action: string;
  item?: string;
  quantity?: number;
  target?: string;
  slot?: string;
}

export interface VLAMotorControls {
  delta_yaw: number;
  delta_pitch: number;
  vx: number;
  vz: number;
  jump: boolean;
  sprint: boolean;
  sneak: boolean;
  attack: boolean;
  use_item: boolean;
  swap_slot: boolean;
  dig_block?: boolean;
  drop_item?: boolean;
  recommended_tool?: string;
  recommended_tool_confidence?: number;
  should_invoke_tool?: boolean;
  subgoal_completed?: boolean;
  should_replan?: boolean;
  state_value: number;
  action_chunk?: Array<{
    tick: number;
    delta_yaw: number;
    delta_pitch: number;
    vx: number;
    vz: number;
    jump: boolean;
    sprint: boolean;
    sneak: boolean;
    attack: boolean;
    use_item: boolean;
    swap_slot: boolean;
    dig_block?: boolean;
    drop_item?: boolean;
  }>;
}

const SEMANTIC_BLOCK_VOCAB: Record<string, number> = {
  air: 0,
  cave_air: 0,
  void_air: 0,
  stone: 1,
  deepslate: 1,
  andesite: 1,
  diorite: 1,
  granite: 1,
  tuff: 1,
  cobblestone: 2,
  cobbled_deepslate: 2,
  dirt: 3,
  coarse_dirt: 3,
  podzol: 3,
  mycelium: 3,
  grass_block: 4,
  sand: 5,
  red_sand: 5,
  sandstone: 5,
  gravel: 6,
  clay: 6,
  oak_log: 7,
  spruce_log: 7,
  birch_log: 7,
  oak_leaves: 8,
  oak_planks: 9,
  water: 10,
  flowing_water: 10,
  lava: 11,
  flowing_lava: 11,
  coal_ore: 12,
  deepslate_coal_ore: 12,
  iron_ore: 13,
  deepslate_iron_ore: 13,
  gold_ore: 14,
  deepslate_gold_ore: 14,
  diamond_ore: 15,
  deepslate_diamond_ore: 15,
  redstone_ore: 16,
  emerald_ore: 16,
  lapis_ore: 16,
  copper_ore: 16,
  crafting_table: 17,
  furnace: 18,
  smoker: 18,
  blast_furnace: 18,
  chest: 19,
  trapped_chest: 19,
  barrel: 19,
  ender_chest: 19,
  torch: 20,
  wall_torch: 20,
  lantern: 20,
  bedrock: 21,
  obsidian: 22,
  crying_obsidian: 22,
  glass: 23,
  stone_bricks: 24,
  bricks: 24,
  ladder: 25,
  vine: 25,
  farmland: 26,
  wheat: 27,
  carrots: 27,
  potatoes: 27,
  beetroots: 27,
  gold_block: 28,
  emerald_block: 29,
  diamond_block: 29,
  end_stone: 30,
  netherrack: 30,
  soul_sand: 30,
  lever: 31,
};

/**
 * Maps Minecraft 1.20.4 block names onto the 32-class semantic vocabulary (0..31)
 * consumed by the 211M Action Head's `voxel_embed` 3D CNN encoder.
 */
export function mapBlockNameToSemanticId(name?: string): number {
  if (!name) return 0;
  const n = name.toLowerCase().trim();
  if (SEMANTIC_BLOCK_VOCAB[n] !== undefined) return SEMANTIC_BLOCK_VOCAB[n];
  if (n.includes('air')) return 0;
  if (n.includes('cobble')) return 2;
  if (n.includes('dirt')) return 3;
  if (n.includes('sand')) return 5;
  if (n.endsWith('_log') || n.endsWith('_stem') || n.endsWith('_wood')) return 7;
  if (n.includes('leaves')) return 8;
  if (n.endsWith('_planks')) return 9;
  if (n.includes('water')) return 10;
  if (n.includes('lava')) return 11;
  if (n.includes('coal_ore')) return 12;
  if (n.includes('iron_ore')) return 13;
  if (n.includes('gold_ore')) return 14;
  if (n.includes('diamond_ore')) return 15;
  if (n.includes('redstone_ore') || n.includes('emerald_ore') || n.includes('lapis_ore') || n.includes('copper_ore')) return 16;
  if (n.includes('furnace') || n.includes('smoker')) return 18;
  if (n.includes('chest') || n.includes('barrel')) return 19;
  if (n.includes('torch') || n.includes('lantern')) return 20;
  if (n.includes('obsidian')) return 22;
  if (n.includes('glass')) return 23;
  if (n.includes('stone') || n.includes('brick') || n.includes('deepslate')) return 24;
  if (n.includes('ladder') || n.includes('vine')) return 25;
  if (n.includes('lever') || n.includes('button') || n.includes('door') || n.includes('bed') || n.includes('pressure_plate')) return 31;
  return 24;
}

export interface VLAPredictionResponse {
  intent: 'chat' | 'action' | 'hybrid';
  speech: string;
  action_tokens: string[];
  parsed_actions: VLAParsedAction[];
  motor_controls: VLAMotorControls;
  raw_text: string;
  tokens_generated: number;
  latency_ms: number;
}

export const CRITICAL_BOSS_NAMES = new Set([
  'ender_dragon',
  'end_crystal',
  'wither',
  'elder_guardian',
  'warden',
]);

export function isBossOrCrystalEntity(name: string): boolean {
  return CRITICAL_BOSS_NAMES.has(String(name || '').toLowerCase());
}

export function getEntityPriority(name: string, type?: string): number {
  const n = String(name || '').toLowerCase();
  // Tier 0: Critical Objectives (Ender Dragon, End Crystals, Wither, Elder Guardian, Warden)
  if (isBossOrCrystalEntity(n)) return 0;
  // Tier 1: Lethal / Immediate Explosive Threats (Creeper)
  if (n === 'creeper') return 1;
  // Tier 2: Ranged Threats & Snipers (Skeleton, Stray, Blaze, Ghast, Pillager)
  if (n === 'skeleton' || n === 'stray' || n === 'blaze' || n === 'ghast' || n === 'pillager') return 2;
  // Tier 3: Standard Hostile Mobs (Enderman, Zombie, Spider, Husk, Drowned, etc.)
  if (type === 'hostile' || n === 'enderman' || n === 'zombie' || n === 'spider' || n === 'cave_spider') return 3;
  // Tier 4: Other Players & Sparring Partners
  if (type === 'player') return 4;
  // Tier 5: Passive animals, items, dropped loot, objects
  return 5;
}

export class VLACompanionClient {
  private daemonUrl: string;

  constructor(daemonUrl = 'http://127.0.0.1:8765') {
    this.daemonUrl = daemonUrl;
  }

  /**
   * Extracts the 24-dimensional sensorimotor proprioception vector from the live bot.
   */
  public extractProprioception(bot: Bot, targetPos?: Vec3): number[] {
    const v = new Array(24).fill(0.0);
    if (!bot || !bot.entity) return v;

    // 0: Health [0..20] -> [0..1]
    v[0] = Math.max(0, Math.min(1, (bot.health || 20) / 20));
    // 1: Food [0..20] -> [0..1]
    v[1] = Math.max(0, Math.min(1, (bot.food || 20) / 20));
    // 2-3: Normalized Yaw [-pi..pi] & Pitch [-pi/2..pi/2] -> [-1..1]
    let rawYaw = (bot.entity.yaw || 0) % (2 * Math.PI);
    if (rawYaw > Math.PI) rawYaw -= 2 * Math.PI;
    if (rawYaw < -Math.PI) rawYaw += 2 * Math.PI;
    v[2] = rawYaw / Math.PI;
    v[3] = Math.max(-1, Math.min(1, (bot.entity.pitch || 0) / (Math.PI / 2)));
    // 4-6: Velocity (vx, vy, vz)
    v[4] = Math.max(-1, Math.min(1, bot.entity.velocity.x));
    v[5] = Math.max(-1, Math.min(1, bot.entity.velocity.y));
    v[6] = Math.max(-1, Math.min(1, bot.entity.velocity.z));
    // 7: On ground
    v[7] = bot.entity.onGround ? 1.0 : 0.0;
    // 8: In water or submerged
    v[8] = (bot.entity as any).isInWater ? 1.0 : 0.0;

    // Find nearest hostile mob
    const hostiles = Object.values(bot.entities || {}).filter(
      (e) => e && e !== bot.entity && (e.type === 'hostile' || ['creeper', 'skeleton', 'zombie', 'spider', 'enderman'].includes(e.name || ''))
    );
    let nearestHostile: any = null;
    let minDist = 32.0;

    for (const h of hostiles) {
      const d = bot.entity.position.distanceTo(h.position);
      if (d < minDist) {
        minDist = d;
        nearestHostile = h;
      }
    }

    if (nearestHostile) {
      const dx = nearestHostile.position.x - bot.entity.position.x;
      const dy = nearestHostile.position.y - bot.entity.position.y;
      const dz = nearestHostile.position.z - bot.entity.position.z;
      // 9-11: Relative offset to nearest hostile
      v[9] = Math.max(-1, Math.min(1, dx / 16.0));
      v[10] = Math.max(-1, Math.min(1, dy / 8.0));
      v[11] = Math.max(-1, Math.min(1, dz / 16.0));
      // 12: Distance
      v[12] = Math.min(1.0, minDist / 32.0);
      // 13: Creeper fuse active
      v[13] = nearestHostile.name === 'creeper' && minDist < 4.5 ? 1.0 : 0.0;
    } else if (targetPos) {
      const dx = targetPos.x - bot.entity.position.x;
      const dy = targetPos.y - bot.entity.position.y;
      const dz = targetPos.z - bot.entity.position.z;
      const dist = bot.entity.position.distanceTo(targetPos);
      v[9] = Math.max(-1, Math.min(1, dx / 16.0));
      v[10] = Math.max(-1, Math.min(1, dy / 8.0));
      v[11] = Math.max(-1, Math.min(1, dz / 16.0));
      v[12] = Math.min(1.0, dist / 32.0);
    }

    // 14: Equipped item in mainhand is sword
    const mainHand = bot.heldItem;
    v[14] = mainHand && mainHand.name.includes('sword') ? 1.0 : 0.0;
    // 15: Equipped item is shield
    const offHand = (bot.inventory as any)?.slots?.[45];
    v[15] = offHand && offHand.name.includes('shield') ? 1.0 : 0.0;
    // 16: Equipped item is bow
    v[16] = mainHand && mainHand.name.includes('bow') ? 1.0 : 0.0;
    // 17: Equipped item is pickaxe
    v[17] = mainHand && mainHand.name.includes('pickaxe') ? 1.0 : 0.0;

    return v;
  }

  /**
   * Extracts a 15x15x15 semantic 3D voxel occupancy grid (3,375 ints in 0..31) centered on the bot.
   */
  public extractLocalVoxels(bot: Bot): number[] {
    const voxels = new Array(3375).fill(0);
    if (!bot?.entity?.position || !bot.blockAt) return voxels;
    const pos = bot.entity.position.floored();
    let idx = 0;
    for (let dx = -7; dx <= 7; dx++) {
      for (let dy = -3; dy <= 11; dy++) {
        for (let dz = -7; dz <= 7; dz++) {
          const b = bot.blockAt(new Vec3(pos.x + dx, pos.y + dy, pos.z + dz));
          voxels[idx++] = mapBlockNameToSemanticId(b?.name);
        }
      }
    }
    return voxels;
  }

  /**
   * Extracts the top 16 nearest entities around the bot into a 16x12 radar matrix:
   * [dx/32, dy/16, dz/32, dist/32, vx, vy, vz, isHostile, isCreeper, isSkeleton, isBossOrCrystal, fuseOrItemFlag].
   */
  public extractEntityRadar(bot: Bot): number[][] {
    const radar: number[][] = Array.from({ length: 16 }, () => new Array(12).fill(0));
    if (!bot?.entity?.position) return radar;
    const botPos = bot.entity.position;
    const HOSTILE_NAMES = new Set([
      'zombie', 'husk', 'drowned', 'skeleton', 'stray', 'wither_skeleton',
      'creeper', 'spider', 'cave_spider', 'enderman', 'witch', 'blaze',
      'pillager', 'vindicator', 'evoker', 'ravager', 'phantom', 'slime',
    ]);

    const sorted = Object.values(bot.entities || {})
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

    for (let i = 0; i < sorted.length; i++) {
      const e: any = sorted[i];
      const name = String(e.name || e.displayName || '').toLowerCase();
      const dx = Math.max(-1, Math.min(1, (e.position.x - botPos.x) / 32.0));
      const dy = Math.max(-1, Math.min(1, (e.position.y - botPos.y) / 16.0));
      const dz = Math.max(-1, Math.min(1, (e.position.z - botPos.z) / 32.0));
      const dist = Math.min(1.0, e.position.distanceTo(botPos) / 32.0);
      const vx = Number.isFinite(e.velocity?.x) ? e.velocity.x : 0;
      const vy = Number.isFinite(e.velocity?.y) ? e.velocity.y : 0;
      const vz = Number.isFinite(e.velocity?.z) ? e.velocity.z : 0;
      const isHostile = e.type === 'hostile' || HOSTILE_NAMES.has(name) ? 1.0 : 0.0;
      const isCreeper = name === 'creeper' ? 1.0 : 0.0;
      const isSkeleton = name === 'skeleton' || name === 'stray' ? 1.0 : 0.0;
      const isBossOrCrystal = isBossOrCrystalEntity(name) ? 1.0 : 0.0;
      const fuseOrItemFlag =
        isCreeper > 0 && Number(e.metadata?.[16]) > 0
          ? 1.0
          : name === 'item' || e.type === 'object'
          ? 0.5
          : 0.0;

      radar[i] = [
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
        isBossOrCrystal,
        fuseOrItemFlag,
      ];
    }

    return radar;
  }

  /**
   * Fast lock-free (<3ms) 211M Multi-Modal Action Transformer query using cached System 2 SLM latent h,
   * 24-dim proprioception, 32-class semantic 3D voxels, and 16-entity radar.
   */
  public async predictActionChunk(
    intent: string,
    bot: Bot,
    targetPos?: Vec3
  ): Promise<VLAMotorControls | null> {
    if (!bot?.entity) return null;
    const proprioception = this.extractProprioception(bot, targetPos);
    const voxels = this.extractLocalVoxels(bot);
    const entities = this.extractEntityRadar(bot);

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 600);
      const res = await fetch(`${this.daemonUrl}/act/chunk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          bot_id: bot.username || 'NeuroAgent',
          intent,
          proprioception,
          voxels,
          entities,
        }),
      });
      clearTimeout(timeout);
      if (!res.ok) return null;
      const data = (await res.json()) as { motor_controls?: VLAMotorControls };
      return data.motor_controls || null;
    } catch {
      return null;
    }
  }

  /**
   * Predicts next dual-channel VLA step (speech + motor/inventory actions).
   */
  public async predict(userPrompt: string, bot: Bot): Promise<VLAPredictionResponse> {
    const proprioception = this.extractProprioception(bot);
    const voxels = this.extractLocalVoxels(bot);
    const entities = this.extractEntityRadar(bot);

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2500);
      const defaultTemp = process.env.SLM_TEMPERATURE ? parseFloat(process.env.SLM_TEMPERATURE) : 0.0;
      const defaultTopP = process.env.SLM_TOP_P ? parseFloat(process.env.SLM_TOP_P) : 1.0;
      const defaultSample = process.env.SLM_DO_SAMPLE === 'true' || defaultTemp > 0.0;

      const res = await fetch(`${this.daemonUrl}/vla/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          user_prompt: userPrompt,
          proprioception,
          voxels,
          entities,
          max_new_tokens: 48,
          temperature: defaultTemp,
          top_p: defaultTopP,
          do_sample: defaultSample,
        }),
      });
      clearTimeout(timeout);

      if (!res.ok) {
        throw new Error(`Neural daemon VLA error: HTTP ${res.status}`);
      }

      return (await res.json()) as VLAPredictionResponse;
    } catch (err: any) {
      // Robust offline fallback so companion never crashes
      return {
        intent: 'chat',
        speech: "I heard you, let me take care of that!",
        action_tokens: [],
        parsed_actions: [],
        motor_controls: {
          delta_yaw: 0,
          delta_pitch: 0,
          vx: 0,
          vz: 0,
          jump: false,
          sprint: false,
          sneak: false,
          attack: false,
          use_item: false,
          swap_slot: false,
          state_value: 0,
        },
        raw_text: "I heard you, let me take care of that!",
        tokens_generated: 10,
        latency_ms: 1.0,
      };
    }
  }

  /**
   * Executes parsed companion inventory actions with zero short-circuiting.
   */
  public async executeCompanionActions(
    bot: Bot,
    actions: VLAParsedAction[],
    playerEntity?: any
  ): Promise<string[]> {
    const logs: string[] = [];

    for (const act of actions) {
      try {
        if (act.action === 'drop_item' || act.action === 'give_to_player') {
          const itemName = act.item || 'cobblestone';
          const qty = act.quantity || 1;
          const item = bot.inventory.items().find((i) => i.name.toLowerCase().includes(itemName.toLowerCase()));

          if (item) {
            if (playerEntity) {
              await bot.lookAt(playerEntity.position.offset(0, 1.2, 0), true);
            }
            await bot.toss(item.type, null, Math.min(qty, item.count));
            logs.push(`Dropped ${Math.min(qty, item.count)}x ${item.name}`);
          } else {
            logs.push(`Item ${itemName} not found in inventory.`);
          }
        } else if (act.action === 'equip') {
          const itemName = act.item || 'iron_sword';
          const item = bot.inventory.items().find((i) => i.name.toLowerCase().includes(itemName.toLowerCase()));
          if (item) {
            const destination = act.slot === 'offhand' ? 'off-hand' : (act.slot as any) || 'hand';
            await bot.equip(item, destination);
            logs.push(`Equipped ${item.name} to ${destination}`);
          }
        } else if (act.action === 'consume') {
          const food = bot.inventory.items().find((i) =>
            ['bread', 'cooked_beef', 'apple', 'baked_potato', 'carrot', 'golden_apple'].includes(i.name)
          );
          if (food) {
            await bot.equip(food, 'hand');
            await bot.consume();
            logs.push(`Consumed ${food.name}`);
          }
        }
      } catch (err: any) {
        logs.push(`Action ${act.action} warning: ${err.message}`);
      }
    }

    return logs;
  }

  /**
   * Executes an 8-tick continuous motor trajectory chunk from the 80.8M Action Chunking Transformer.
   * Drives continuous camera aiming, planar velocity vectors, and discrete motor activations at 20Hz.
   */
  public async executeMotorActionChunk(
    bot: Bot,
    chunk?: VLAMotorControls['action_chunk'],
    targetPos?: Vec3,
    maxTicks = 4,
    options?: { allowItemDrop?: boolean; allowDig?: boolean }
  ): Promise<void> {
    if (!chunk || chunk.length === 0 || !bot?.entity) return;

    let baseYaw = bot.entity.yaw;
    let basePitch = bot.entity.pitch;
    if (targetPos) {
      const dx = targetPos.x - bot.entity.position.x;
      const dy = targetPos.y + 1.0 - (bot.entity.position.y + 1.62);
      const dz = targetPos.z - bot.entity.position.z;
      const horiz = Math.hypot(dx, dz);
      if (horiz > 0.2) {
        baseYaw = Math.atan2(-dx, -dz);
        basePitch = Math.atan2(dy, horiz);
      }
    }

    const allowDig = options?.allowDig !== false;
    const allowItemDrop = options?.allowItemDrop === true;
    let itemActivated = false;

    const slice = chunk.slice(0, Math.max(1, maxTicks));
    try {
      for (const step of slice) {
        if (!bot.entity) break;

        // 1. Aim adjustment (continuous camera aim relative to target-aligned orientation)
        if (step.delta_yaw !== 0 || step.delta_pitch !== 0 || targetPos) {
          const clampedDyaw = Math.max(-25, Math.min(25, step.delta_yaw || 0));
          const clampedDpitch = Math.max(-20, Math.min(20, step.delta_pitch || 0));
          const targetYaw = baseYaw + (clampedDyaw * Math.PI) / 180;
          const targetPitch = Math.max(
            -Math.PI / 2,
            Math.min(Math.PI / 2, basePitch + (clampedDpitch * Math.PI) / 180)
          );
          try {
            await bot.look(targetYaw, targetPitch, true);
          } catch {}
        }

        // 2. Continuous velocity / movement control states
        bot.setControlState('forward', step.vz > 0.1);
        bot.setControlState('back', step.vz < -0.1);
        bot.setControlState('left', step.vx < -0.1);
        bot.setControlState('right', step.vx > 0.1);
        if (step.jump && bot.entity.onGround) {
          bot.setControlState('jump', true);
        } else {
          bot.setControlState('jump', false);
        }
        bot.setControlState('sprint', step.sprint);
        bot.setControlState('sneak', step.sneak);

        // 3. Discrete motor actions
        if (step.attack) {
          try {
            bot.swingArm('right');
            const target = bot.nearestEntity(
              (e: any) =>
                e &&
                e !== bot.entity &&
                (e.type === 'hostile' || e.type === 'mob' || (e.type === 'player' && /opponent|dqn|rival|mirror/i.test(e.username || '')))
            );
            if (target && bot.entity.position.distanceTo(target.position) < 3.5) {
              (bot as any)._lastAttackAt = Date.now();
              bot.attack(target);
            }
          } catch {}
        }

        if (step.use_item) {
          if (!itemActivated) {
            try {
              bot.activateItem();
              itemActivated = true;
            } catch {}
          }
        } else if (itemActivated) {
          try {
            bot.deactivateItem();
          } catch {}
          itemActivated = false;
        }

        if (allowDig && step.dig_block) {
          try {
            const targetBlk = typeof (bot as any).blockAtCursor === 'function' ? (bot as any).blockAtCursor(4.5) : null;
            if (
              targetBlk &&
              targetBlk.name !== 'air' &&
              targetBlk.name !== 'bedrock' &&
              !(bot as any).targetDigBlock &&
              typeof bot.dig === 'function'
            ) {
              void bot.dig(targetBlk, 'ignore').catch(() => {});
            }
          } catch {}
        }

        if (allowItemDrop && step.drop_item && bot.heldItem && typeof bot.toss === 'function') {
          try {
            (bot as any)._lastDropItemAt = Date.now();
            void bot.toss(bot.heldItem.type, null, 1).catch(() => {});
          } catch {}
        }

        // 20Hz tick timing (50ms)
        await new Promise((r) => setTimeout(r, 50));
      }
    } finally {
      if (itemActivated) {
        try {
          bot.deactivateItem();
        } catch {}
      }
      if (!targetPos) {
        try {
          bot.clearControlStates();
        } catch {}
      }
    }
  }
}
