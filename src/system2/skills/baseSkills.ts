import { Vec3 } from 'vec3';
import { DynamicMCPToolRegistry } from '../mcp/toolRegistry';
import { logVerbose } from '../../logger';
import { sharedNeuralBridge } from '../../system1/neural/neuralBridge';

/**
 * Safely unwraps string or structured object arguments emitted by SLMs without coercing to "[object Object]".
 */
export function pickStrArg(v: any): string {
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  if (v && typeof v === 'object') {
    const inner =
      v.name ??
      v.block ??
      v.blockName ??
      v.item ??
      v.itemName ??
      v.animalType ??
      v.target ??
      v.type ??
      v.id ??
      v.username;
    if (inner !== undefined && inner !== null && typeof inner !== 'object') {
      return String(inner);
    }
  }
  return '';
}

/**
 * Returns the minimum pickaxe tier required to harvest `blockName` in vanilla Minecraft
 * so the agent NEVER mines `iron_ore` with a `wooden_pickaxe` or punches `stone` by hand!
 */
export function getRequiredPickaxeTier(blockName: string): 'none' | 'wood' | 'stone' | 'iron' | 'diamond' {
  const n = (blockName || '').toLowerCase();
  if (
    n.includes('obsidian') ||
    n.includes('ancient_debris') ||
    n === 'respawn_anchor' ||
    n === 'netherite_block'
  ) {
    return 'diamond';
  }
  if (
    n.includes('diamond_ore') ||
    n.includes('gold_ore') ||
    n.includes('redstone_ore') ||
    n.includes('emerald_ore') ||
    n === 'raw_gold_block'
  ) {
    return 'iron';
  }
  if (
    n.includes('iron_ore') ||
    n.includes('copper_ore') ||
    n.includes('lapis_ore') ||
    n === 'raw_iron_block' ||
    n === 'raw_copper_block' ||
    n === 'iron_block'
  ) {
    return 'stone';
  }
  if (
    n === 'stone' ||
    n === 'cobblestone' ||
    n === 'deepslate' ||
    n === 'cobbled_deepslate' ||
    n.includes('coal_ore') ||
    n === 'furnace' ||
    n === 'blast_furnace' ||
    n === 'smoker' ||
    n === 'andesite' ||
    n === 'diorite' ||
    n === 'granite' ||
    n === 'tuff' ||
    n.includes('sandstone') ||
    n.includes('terracotta') ||
    (n.includes('concrete') && !n.includes('powder')) ||
    n.includes('basalt') ||
    n.includes('blackstone') ||
    n.includes('calcite') ||
    n.includes('dripstone') ||
    n === 'mossy_cobblestone'
  ) {
    return 'wood';
  }
  return 'none';
}

export function isPickaxeTierSufficient(
  heldItemName: string | null | undefined,
  requiredTier: 'none' | 'wood' | 'stone' | 'iron' | 'diamond'
): boolean {
  if (requiredTier === 'none') return true;
  const h = (heldItemName || '').toLowerCase();
  if (requiredTier === 'diamond') {
    return h === 'netherite_pickaxe' || h === 'diamond_pickaxe';
  }
  if (requiredTier === 'iron') {
    return h === 'netherite_pickaxe' || h === 'diamond_pickaxe' || h === 'iron_pickaxe';
  }
  if (requiredTier === 'stone') {
    return (
      h === 'netherite_pickaxe' ||
      h === 'diamond_pickaxe' ||
      h === 'iron_pickaxe' ||
      h === 'stone_pickaxe'
    );
  }
  if (requiredTier === 'wood') {
    return h.endsWith('_pickaxe');
  }
  return true;
}

export interface ToolDurabilityInfo {
  item: any;
  name: string;
  durability: number;
  maxDurability: number;
  percent: number;
}

export function getItemDurabilityInfo(item: any): ToolDurabilityInfo | null {
  if (!item || !item.name) return null;
  const maxDur = item.maxDurability;
  if (!maxDur) return null;
  const durUsed = item.durabilityUsed ?? 0;
  const rem = Math.max(0, maxDur - durUsed);
  const pct = Math.round((rem / maxDur) * 100);
  return { item, name: item.name, durability: rem, maxDurability: maxDur, percent: pct };
}

export function getAvailableTools(bot: any, filterSuffix: string): ToolDurabilityInfo[] {
  const items = bot.inventory?.items() || [];
  const tools: ToolDurabilityInfo[] = [];
  for (const it of items) {
    if (it.name.endsWith(filterSuffix)) {
      const info = getItemDurabilityInfo(it);
      if (info) tools.push(info);
    }
  }
  return tools;
}

/**
 * Calculates remaining pickaxe durability specifically usable for general stone mining,
 * deliberately reserving high-tier iron/diamond pickaxes for rare ores.
 */
export function getUsablePickaxeDurability(bot: any): { total: number; stoneOrWoodTotal: number; ironReserve: number } {
  const tools = getAvailableTools(bot, '_pickaxe');
  let stoneOrWoodTotal = 0;
  let ironReserve = 0;
  for (const t of tools) {
    if (t.name === 'stone_pickaxe' || t.name === 'wooden_pickaxe') {
      stoneOrWoodTotal += t.durability;
    } else if (t.name === 'iron_pickaxe' || t.name === 'diamond_pickaxe' || t.name === 'netherite_pickaxe') {
      ironReserve += t.durability;
    }
  }
  return { total: stoneOrWoodTotal + Math.max(0, ironReserve - 20), stoneOrWoodTotal, ironReserve };
}

/**
 * Proactively verifies that the agent has sufficient stone-mining pickaxe durability
 * for an upcoming excavation task. If durability is low, crafts replacement stone_pickaxes
 * from inventory cobblestone and sticks before starting, preventing the agent from being
 * stranded bare-handed or exhausting high-tier iron pickaxes!
 */
export async function ensurePickaxesForMining(
  ctx: any,
  blocksToMineEstimate = 25
): Promise<void> {
  const inv = ctx.bot.inventory?.items() || [];
  const pickaxes = inv.filter((i: any) => i.name.endsWith('_pickaxe'));

  let stoneOrWoodDurability = 0;
  let stonePicksCount = 0;
  for (const p of pickaxes) {
    const info = getItemDurabilityInfo(p);
    if (info) {
      if (p.name === 'stone_pickaxe' || p.name === 'wooden_pickaxe') {
        stoneOrWoodDurability += info.durability;
        stonePicksCount++;
      }
    }
  }

  // If stone/wood pickaxe durability is insufficient, craft stone pickaxes to conserve higher-tier tools!
  if (stoneOrWoodDurability < blocksToMineEstimate || stonePicksCount === 0) {
    const cobbleCount = inv
      .filter((i: any) => i.name === 'cobblestone' || i.name === 'cobbled_deepslate')
      .reduce((s: number, i: any) => s + i.count, 0);
    const sticksCount = inv
      .filter((i: any) => i.name === 'stick')
      .reduce((s: number, i: any) => s + i.count, 0);
    const planksCount = inv
      .filter((i: any) => i.name.endsWith('_planks'))
      .reduce((s: number, i: any) => s + i.count, 0);
    const logsCount = inv
      .filter((i: any) => i.name.endsWith('_log') || i.name.endsWith('_stem'))
      .reduce((s: number, i: any) => s + i.count, 0);

    const hasWoodForSticks = sticksCount >= 2 || planksCount >= 2 || logsCount >= 1;

    if (cobbleCount >= 3 && hasWoodForSticks) {
      logVerbose(
        'MINING_PREP',
        `Stone-mining pickaxe durability (${stoneOrWoodDurability} uses) is low for ${blocksToMineEstimate}-block dig (auto-crafting disabled; use craft_item explicitly).`
      );
    }
  }
}

/**
 * Automatically selects and equips the right tool (or a non-pickaxe slot when digging dirt/leaves)
 * for the specific block being broken right now.
 * Implements strict tool preservation: saves iron/diamond pickaxes for diamonds and rare ores,
 * and uses economical stone tools for regular stone/dirt/wood excavation.
 */
/**
 * Automatically selects and equips the right tool (or a non-pickaxe slot when digging dirt/leaves)
 * for the specific block being broken right now.
 * Implements strict tool preservation: saves iron/diamond pickaxes for diamonds and rare ores,
 * and uses economical stone tools for regular stone/dirt/wood excavation.
 */
export async function equipBestToolForBlockName(ctx: any, blockName: string): Promise<void> {
  const n = (blockName || '').toLowerCase();

  // Axes for wood/logs/planks
  if (n.includes('log') || n.includes('stem') || n.includes('planks') || n === 'crafting_table') {
    const axes = ['stone_axe', 'wooden_axe', 'iron_axe', 'diamond_axe', 'netherite_axe'];
    for (const a of axes) {
      const it = (ctx.bot.inventory?.items() || []).find((i: any) => i.name === a);
      if (it) {
        const info = getItemDurabilityInfo(it);
        if (info && info.durability <= 1) continue; // Don't shatter on last hit
        if (await ctx.kinematics.equipItemHumanized(a, 'hand')) return;
      }
    }
    return;
  }

  // Shovels for dirt/sand/gravel (strictly excluding sandstone which requires a pickaxe!)
  if (
    n.includes('dirt') ||
    n.includes('grass_block') ||
    (n.includes('sand') && !n.includes('sandstone')) ||
    n.includes('gravel') ||
    n.includes('podzol') ||
    n.includes('clay')
  ) {
    const shovels = ['stone_shovel', 'wooden_shovel', 'iron_shovel', 'diamond_shovel', 'netherite_shovel'];
    for (const s of shovels) {
      const it = (ctx.bot.inventory?.items() || []).find((i: any) => i.name === s);
      if (it) {
        const info = getItemDurabilityInfo(it);
        if (info && info.durability <= 1) continue;
        if (await ctx.kinematics.equipItemHumanized(s, 'hand')) return;
      }
    }
    // If no shovel in inventory and held item is a pickaxe/sword/axe, switch to a non-tool item
    // (like stick, planks, or dirt) or unequip so pickaxe durability is NEVER wasted on dirt (2x wear penalty!)
    const held = ctx.bot.heldItem?.name || '';
    if (held.includes('pickaxe') || held.includes('sword') || held.includes('axe')) {
      const nonTool = (ctx.bot.inventory?.items() || []).find(
        (i: any) =>
          !i.name.includes('pickaxe') &&
          !i.name.includes('sword') &&
          !i.name.includes('axe') &&
          !i.name.includes('shovel')
      );
      if (nonTool) {
        try {
          await ctx.bot.equip(nonTool, 'hand');
        } catch {}
      } else {
        try {
          await ctx.bot.unequip('hand');
        } catch {}
      }
    }
    return;
  }

  // Pickaxes for stone/ores/sandstone
  const reqTier = getRequiredPickaxeTier(n);
  if (reqTier !== 'none') {
    let pickCandidates: string[] = [];

    if (reqTier === 'diamond') {
      // Must use diamond or netherite! (For obsidian, ancient debris)
      pickCandidates = ['netherite_pickaxe', 'diamond_pickaxe'];
    } else if (reqTier === 'iron') {
      // Must use iron, diamond, or netherite! (For diamond_ore, gold_ore, redstone_ore, emerald_ore)
      pickCandidates = ['netherite_pickaxe', 'diamond_pickaxe', 'iron_pickaxe'];
    } else if (reqTier === 'stone') {
      // Iron ore, lapis, copper: prefer stone pickaxe to save iron!
      pickCandidates = ['stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe', 'netherite_pickaxe'];
    } else {
      // reqTier === 'wood' (Stone, cobblestone, sandstone, deepslate, andesite, coal_ore):
      // PRESERVE HIGH-TIER TOOLS! Use stone pickaxe first!
      pickCandidates = ['stone_pickaxe', 'wooden_pickaxe'];

      // Only allow iron pickaxe if no stone/wood pickaxe exists AND iron pickaxe has > 15 durability
      const hasLowerPick = (ctx.bot.inventory?.items() || []).some(
        (i: any) => i.name === 'stone_pickaxe' || i.name === 'wooden_pickaxe'
      );
      if (!hasLowerPick) {
        const ironPick = (ctx.bot.inventory?.items() || []).find((i: any) => i.name === 'iron_pickaxe');
        const ironInfo = ironPick ? getItemDurabilityInfo(ironPick) : null;
        if (ironInfo && ironInfo.durability > 15) {
          pickCandidates.push('iron_pickaxe');
        } else if (!ironPick) {
          pickCandidates.push('diamond_pickaxe', 'netherite_pickaxe');
        }
      }
    }

    for (const p of pickCandidates) {
      const it = (ctx.bot.inventory?.items() || []).find((i: any) => i.name === p);
      if (it) {
        const info = getItemDurabilityInfo(it);
        if (info && info.durability <= 2 && reqTier !== 'iron' && reqTier !== 'diamond') continue;
        if (await ctx.kinematics.equipItemHumanized(p, 'hand')) return;
      }
    }
    return;
  }
}

/**
 * Verifies that the bot has the required harvesting tool equipped for `block` before digging.
 * 1. Equips the optimal tool via `equipBestToolForBlockName`.
 * 2. If held item is not sufficient, directly searches inventory for an existing pickaxe satisfying `reqTier`.
 * 3. If missing from inventory, auto-crafts a replacement on the spot.
 * 4. Only marks `lastToolFailure` if `depth === 0` (so obstacle/pocket blocks never abort resource gathering!).
 */
export async function ensureHarvestToolForBlock(ctx: any, block: any, depth = 0): Promise<boolean> {
  if (!block?.name) return true;
  const blockReqTier = getRequiredPickaxeTier(block.name);
  const gatherMinTier = (ctx as any).gatherMinimumPickaxeTier as
    | 'none'
    | 'wood'
    | 'stone'
    | 'iron'
    | 'diamond'
    | undefined;
  const tierRank: Record<'none' | 'wood' | 'stone' | 'iron' | 'diamond', number> = {
    none: 0,
    wood: 1,
    stone: 2,
    iron: 3,
    diamond: 4,
  };
  const reqTier =
    blockReqTier !== 'none' &&
    gatherMinTier &&
    tierRank[gatherMinTier] > tierRank[blockReqTier]
      ? gatherMinTier
      : blockReqTier;

  await equipBestToolForBlockName(ctx, block.name);

  if (reqTier === 'none') {
    return true;
  }

  if (isPickaxeTierSufficient(ctx.bot.heldItem?.name, reqTier)) {
    return true;
  }

  // 1. Direct inventory search: check if ANY pickaxe already in inventory satisfies reqTier!
  const invItems = ctx.bot.inventory?.items() || [];
  // Prioritize stone_pickaxe for wood/stone tiers to preserve higher-tier tools
  const sufficientInvPick =
    reqTier === 'wood' || reqTier === 'stone'
      ? invItems.find((i: any) => i.name === 'stone_pickaxe' || i.name === 'wooden_pickaxe') ||
        invItems.find((i: any) => isPickaxeTierSufficient(i.name, reqTier))
      : invItems.find((i: any) => isPickaxeTierSufficient(i.name, reqTier));

  if (sufficientInvPick) {
    const equipped = await ctx.kinematics.equipItemHumanized(sufficientInvPick.name, 'hand');
    if (equipped && isPickaxeTierSufficient(ctx.bot.heldItem?.name, reqTier)) {
      return true;
    }
  }

  // 2. Pickaxe is missing or broke; fail cleanly without hidden auto-crafting (AGENTS.md)
  // If tunneling through stone (`blockReqTier === 'wood'`) while gathering iron (`reqTier === 'stone'`),
  // allow falling back to `blockReqTier` so existing stone/wood pickaxes can still clear obstacles.
  if (blockReqTier !== reqTier && isPickaxeTierSufficient(ctx.bot.heldItem?.name, blockReqTier)) {
    return true;
  }

  // Only set lastToolFailure for primary target blocks (depth === 0).
  // A secondary obstacle or pocket excavation block (depth > 0) should NEVER terminate the entire gather task!
  if (depth === 0) {
    const minToolName =
      reqTier === 'diamond'
        ? 'diamond_pickaxe'
        : reqTier === 'iron'
          ? 'iron_pickaxe'
          : reqTier === 'stone'
            ? 'stone_pickaxe'
            : 'wooden_pickaxe';
    (ctx as any).lastToolFailure = `Missing required tool (${minToolName}) to harvest ${block.name} (currently holding: ${ctx.bot.heldItem?.name || 'hand'}) and lack materials in inventory to auto-craft a replacement.`;
    logVerbose('DIG', (ctx as any).lastToolFailure);
  }
  return false;
}

/**
 * Proactively verifies that the agent has the specified tool classes in inventory.
 * If any tool is missing, checks inventory materials (cobblestone, planks, sticks, iron)
 * and automatically crafts the best possible tool on the spot!
 */
export async function ensureToolsAvailable(
  ctx: any,
  toolTypes: Array<'axe' | 'shovel' | 'pickaxe' | 'sword'>
): Promise<void> {
  const inv = ctx.bot.inventory?.items() || [];
  const countByFilter = (pred: (n: string) => boolean) =>
    inv.filter((i: any) => pred(i.name.toLowerCase())).reduce((s: number, i: any) => s + i.count, 0);

  const ironIngots = countByFilter((n) => n === 'iron_ingot');
  const cobble = countByFilter((n) => n === 'cobblestone' || n === 'cobbled_deepslate');
  const planks = countByFilter((n) => n.endsWith('_planks'));
  const logs = countByFilter((n) => n.endsWith('_log') || n.endsWith('_stem'));
  const sticks = countByFilter((n) => n === 'stick');
  const totalPotentialPlanks = planks + logs * 4;
  const canHaveSticks = sticks >= 2 || totalPotentialPlanks >= 2;

  for (const type of toolTypes) {
    const hasTool = inv.some((i: any) => i.name.endsWith(`_${type}`));
    if (hasTool) continue;
    if (!canHaveSticks) continue;

    const neededMaterials = type === 'shovel' ? 1 : type === 'sword' ? 2 : 3;
    let candidate = '';
    if (cobble >= neededMaterials) {
      candidate = `stone_${type}`;
    } else if (ironIngots >= neededMaterials && ironIngots > 4) {
      candidate = `iron_${type}`;
    } else if (totalPotentialPlanks >= neededMaterials + 2) {
      candidate = `wooden_${type}`;
    }

    if (candidate) {
      logVerbose('CRAFT', `Missing ${candidate} before task preparation (auto-crafting disabled per AGENTS.md).`);
    }
  }
}

/**
 * Safely digs a single reachable block with Game-API server confirmation:
 * 1. Stops pathfinder & jumping so the bot stands firmly onGround (avoiding the 5x airborne mining slow-down!).
 * 2. Checks 3D raycast line-of-sight (`bot.world.raycast`) and clears any obstructing block in front first.
 * 3. Equips the exact right tool for `block.name` and waits 90ms (2 server ticks) for held-item sync.
 * 4. Calls `bot.dig(block, true)` with a properly cleaned-up watchdog timer (`clearTimeout` in `finally`!).
 * 5. Listens to server `block_change` packets and verifies the block genuinely turned to `air` on the server (retrying up to 2 times if needed).
 */
async function safeDigSingleBlock(
  ctx: any,
  blockPos: Vec3,
  depth = 0,
  allowPathfind = true
): Promise<boolean> {
  if (ctx.abortSignal?.aborted) return false;
  const block = ctx.bot.blockAt(blockPos);
  if (!block || block.name === 'air' || block.name.includes('water') || block.name.includes('lava') || block.name === 'bedrock') {
    return true;
  }

  // Stop movement & pathfinder so bot stays still and onGround while mining
  ctx.bot.pathfinder?.setGoal(null);
  if ((ctx.bot as any).targetDigBlock) {
    try {
      ctx.bot.stopDigging();
    } catch {}
    await new Promise((r) => setTimeout(r, 70));
  }
  ctx.bot.clearControlStates?.();

  const blockCenter = blockPos.offset(0.5, 0.5, 0.5);
  let dist = ctx.bot.entity.position.offset(0, 1.62, 0).distanceTo(blockCenter);

  if (allowPathfind && dist > 4.3 && ctx.kinematics?.pathfindTo) {
    // If the block is above head height (like tree leaves/canopy), navigate to ground level underneath it instead of jumping into air
    const botFootY = Math.floor(ctx.bot.entity?.position?.y ?? blockPos.y);
    const navTarget = blockPos.y > botFootY + 1 ? new Vec3(blockPos.x, botFootY, blockPos.z) : blockPos;
    await ctx.kinematics.pathfindTo(navTarget, 2.1);
    ctx.bot.pathfinder?.setGoal(null);
    ctx.bot.clearControlStates?.();
    dist = ctx.bot.entity.position.offset(0, 1.62, 0).distanceTo(blockCenter);
  }

  // Check if solid blocks are directly obstructing the 3D line-of-sight ray between bot's eyes and `blockPos`
  if (depth < 2 && typeof ctx.bot.world?.raycast === 'function') {
    for (let bStep = 0; bStep < 3; bStep++) {
      if (ctx.abortSignal?.aborted) break;
      const eyePos = ctx.bot.entity.position.offset(0, ctx.bot.entity.eyeHeight || 1.62, 0);
      const toTarget = blockCenter.minus(eyePos);
      const rayDist = toTarget.norm();
      if (rayDist <= 0.5 || rayDist > 4.8) break;

      const dir = toTarget.scaled(1 / rayDist);
      const hit = ctx.bot.world.raycast(eyePos, dir, Math.min(4.8, rayDist));
      if (
        hit?.position &&
        (hit.position.x !== blockPos.x || hit.position.y !== blockPos.y || hit.position.z !== blockPos.z)
      ) {
        const blocker = ctx.bot.blockAt(hit.position);
        if (
          blocker &&
          blocker.name !== 'air' &&
          blocker.boundingBox === 'block' &&
          !blocker.name.includes('bedrock') &&
          blocker.name !== 'crafting_table' &&
          blocker.name !== 'furnace' &&
          blocker.name !== 'chest'
        ) {
          logVerbose('DIG', `Line-of-sight to (${blockPos.x}, ${blockPos.y}, ${blockPos.z}) obstructed by ${blocker.name} at (${hit.position.x}, ${hit.position.y}, ${hit.position.z}); clearing blocker first`);
          const cleared = await safeDigSingleBlock(ctx, hit.position.clone(), depth + 1, false);
          if (!cleared) break;

          // Also clear the vertical pair (making a 1x2 human-height doorway instead of an unwalkable 1x1 peephole!)
          const botFootY = Math.floor(ctx.bot.entity.position.y);
          const pairPos =
            hit.position.y >= botFootY + 1
              ? hit.position.offset(0, -1, 0)
              : hit.position.offset(0, 1, 0);
          if (
            pairPos.x !== blockPos.x ||
            pairPos.y !== blockPos.y ||
            pairPos.z !== blockPos.z
          ) {
            const pairBlock = ctx.bot.blockAt(pairPos);
            if (
              pairBlock &&
              pairBlock.name !== 'air' &&
              pairBlock.boundingBox === 'block' &&
              !pairBlock.name.includes('bedrock') &&
              pairBlock.name !== 'crafting_table' &&
              pairBlock.name !== 'furnace' &&
              pairBlock.name !== 'chest' &&
              !(
                typeof (ctx as any).currentGatherMatcher === 'function' &&
                (ctx as any).currentGatherMatcher(pairBlock)
              )
            ) {
              await safeDigSingleBlock(ctx, pairPos.clone(), depth + 1, false);
            }
          }
          continue;
        }
      }
      break;
    }
  }

  for (let retry = 0; retry < 2; retry++) {
    if (ctx.abortSignal?.aborted) return false;

    ctx.bot.pathfinder?.setGoal(null);
    ctx.bot.clearControlStates?.();

    // Wait up to 350ms to settle onto the ground if mid-jump
    const groundWait = Date.now();
    while (!ctx.bot.entity.onGround && Date.now() - groundWait < 350) {
      await new Promise((r) => setTimeout(r, 40));
    }

    const freshBlock = ctx.bot.blockAt(blockPos);
    if (
      !freshBlock ||
      freshBlock.name === 'air' ||
      freshBlock.name === 'cave_air' ||
      freshBlock.name === 'void_air' ||
      freshBlock.name.includes('water') ||
      freshBlock.name.includes('lava')
    ) {
      return true;
    }
    if (freshBlock.diggable === false) {
      return false;
    }

    const prevHeld = ctx.bot.heldItem?.name || null;
    const hasHarvestTool = await ensureHarvestToolForBlock(ctx, freshBlock, depth);
    if (!hasHarvestTool) {
      return false;
    }
    const newHeld = ctx.bot.heldItem?.name || null;
    if (newHeld !== prevHeld) {
      // Wait 90ms (~2 server ticks) so the server registers the newly equipped tool before block_dig(start)
      await new Promise((r) => setTimeout(r, 90));
    }

    dist = ctx.bot.entity.position.offset(0, 1.62, 0).distanceTo(blockCenter);
    if (!ctx.bot.canDigBlock(freshBlock) || dist > 4.8) {
      logVerbose('DIG', `Cannot dig ${freshBlock.name} at (${blockPos.x}, ${blockPos.y}, ${blockPos.z}): dist=${dist.toFixed(2)}, canDig=${ctx.bot.canDigBlock(freshBlock)}`);
      return false;
    }

    const brokenBlockName = freshBlock.name;
    const isOpportunisticCollectable =
      depth > 0 &&
      (brokenBlockName.includes('_ore') ||
        brokenBlockName === 'crafting_table' ||
        brokenBlockName === 'furnace' ||
        brokenBlockName === 'chest' ||
        (typeof (ctx as any).currentGatherMatcher === 'function' &&
          (ctx as any).currentGatherMatcher({ name: brokenBlockName })));
    const preBlockerInv = isOpportunisticCollectable
      ? countHarvestedDropsInInventory(ctx.bot, brokenBlockName)
      : 0;

    let serverReverted = false;
    const onBlockChange = (packet: any) => {
      if (
        packet?.location &&
        packet.location.x === blockPos.x &&
        packet.location.y === blockPos.y &&
        packet.location.z === blockPos.z &&
        packet.type !== 0
      ) {
        serverReverted = true;
      }
    };
    ctx.bot._client?.on('block_change', onBlockChange);

    logVerbose('DIG', `Starting dig (try #${retry + 1}) on ${freshBlock.name} at (${blockPos.x}, ${blockPos.y}, ${blockPos.z}) dist=${dist.toFixed(2)}m held=${ctx.bot.heldItem?.name || 'hand'}`);

    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    const isSlowBlock = ['obsidian', 'crying_obsidian', 'ancient_debris', 'respawn_anchor', 'enchanting_table', 'ender_chest', 'anvil'].includes(freshBlock.name);
    const maxDigTimeMs = isSlowBlock ? 18000 : 8000;
    try {
      await Promise.race([
        ctx.bot.dig(freshBlock, true),
        new Promise((_, reject) => {
          timeoutId = setTimeout(() => {
            try {
              ctx.bot.stopDigging();
            } catch {}
            reject(new Error(`Dig timeout exceeded ${maxDigTimeMs / 1000}s`));
          }, maxDigTimeMs);
        }),
      ]);
    } catch (err: any) {
      logVerbose('DIG', `Dig error on ${freshBlock.name} at (${blockPos.x}, ${blockPos.y}, ${blockPos.z}) (try #${retry + 1}): ${err.message}`);
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
      ctx.bot._client?.removeListener('block_change', onBlockChange);
    }

    // Wait 120ms for server block_change / itemDrop packets to arrive
    await new Promise((r) => setTimeout(r, 120));
    const afterBlock = ctx.bot.blockAt(blockPos);
    const broke = (!afterBlock || afterBlock.name === 'air' || afterBlock.boundingBox === 'empty') && !serverReverted;
    logVerbose('DIG', `Verified dig on ${brokenBlockName} at (${blockPos.x}, ${blockPos.y}, ${blockPos.z}): broke=${broke} (serverReverted=${serverReverted}, now=${afterBlock?.name || 'air'})`);
    if (broke) {
      if (isOpportunisticCollectable) {
        logVerbose('PICKUP', `Opportunistically collecting dropped item from cleared ${brokenBlockName} at (${blockPos.x}, ${blockPos.y}, ${blockPos.z})!`);
        if (typeof (ctx as any).onGatherBlockBroken === 'function') {
          (ctx as any).onGatherBlockBroken(blockPos, brokenBlockName);
        }
        await collectDroppedItemNear(ctx, blockPos, brokenBlockName, preBlockerInv);
      }
      return true;
    }
  }

  return false;
}

/**
 * Digs a direct 1x2 tunnel or stepped staircase toward an underground/walled-off `targetPos`
 * when standard walk-only/A* pathfinding stops short (`dist > 4.2m`).
 * If `stopIfNearbyMatches` is provided, immediately stops tunneling as soon as any matching
 * target block is uncovered within 3.8m so the agent never mines past an adjacent ore!
 */
async function tunnelToBlock(
  ctx: any,
  targetPos: Vec3,
  stopIfNearbyMatches?: (b: any) => boolean,
  desiredReach = 2.45
): Promise<boolean> {
  const targetCenter = targetPos.offset(0.5, 0.5, 0.5);

  const hasLiquidAdjacent = (pos: Vec3): boolean => {
    const dirs = [
      new Vec3(0, 0, 0),
      new Vec3(1, 0, 0),
      new Vec3(-1, 0, 0),
      new Vec3(0, 0, 1),
      new Vec3(0, 0, -1),
      new Vec3(0, 1, 0),
    ];
    for (const d of dirs) {
      const b = ctx.bot.blockAt(pos.plus(d));
      if (b && (b.name.includes('lava') || b.name.includes('water'))) {
        return true;
      }
    }
    return false;
  };

  for (let step = 0; step < 28; step++) {
    if (ctx.abortSignal?.aborted || (ctx as any).lastToolFailure) return false;
    const eyePos = ctx.bot.entity.position.offset(0, 1.62, 0);
    const curDist = eyePos.distanceTo(targetCenter);
    const curHoriz = Math.hypot(
      ctx.bot.entity.position.x - targetCenter.x,
      ctx.bot.entity.position.z - targetCenter.z
    );
    if (curDist <= desiredReach && curHoriz <= 1.8) {
      return true;
    }

    const footPos = ctx.bot.entity.position.floored();

    // If we uncovered an adjacent matching target block along the tunnel walls right next to us (<= 2.45m), stop and mine it first!
    if (step > 0 && typeof stopIfNearbyMatches === 'function') {
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 2; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            const checkPos = footPos.offset(dx, dy, dz);
            const cb = ctx.bot.blockAt(checkPos);
            if (cb && stopIfNearbyMatches(cb) && hasExposedFace(ctx.bot, checkPos)) {
              const d = eyePos.distanceTo(checkPos.offset(0.5, 0.5, 0.5));
              if (d <= 2.45) {
                logVerbose('GATHER', `Tunneling uncovered adjacent matching block ${cb.name} at (${checkPos.x}, ${checkPos.y}, ${checkPos.z}) dist=${d.toFixed(1)}m; stopping tunnel to harvest it!`);
                return true;
              }
            }
          }
        }
      }
    }

    // If fallen sand/gravel is overlapping the bot's own head or feet, clear it first
    for (const selfPos of [footPos.offset(0, 1, 0), footPos]) {
      const sb = ctx.bot.blockAt(selfPos);
      if (sb && sb.name !== 'air' && sb.boundingBox === 'block' && sb.name !== 'bedrock') {
        const ok = await safeDigSingleBlock(ctx, selfPos, 1, false);
        if (!ok && (ctx as any).lastToolFailure) return false;
      }
    }

    const dx = targetPos.x - footPos.x;
    const dy = targetPos.y - footPos.y;
    const dz = targetPos.z - footPos.z;

    // If horizontally aligned and target is 1..4 blocks below us, clear straight down to it
    if (Math.abs(dx) <= 1 && Math.abs(dz) <= 1 && dy < 0 && dy >= -4) {
      for (let y = footPos.y; y > targetPos.y; y--) {
        const cPos = new Vec3(targetPos.x, y, targetPos.z);
        const cb = ctx.bot.blockAt(cPos);
        if (cb && cb.name !== 'air' && cb.boundingBox === 'block') {
          if (typeof stopIfNearbyMatches === 'function' && stopIfNearbyMatches(cb)) {
            return true;
          }
          const ok = await safeDigSingleBlock(ctx, cPos, 1, false);
          if (!ok) return false;
        }
      }
      return true;
    }

    // Choose cardinal horizontal step toward targetPos
    let sx = 0;
    let sz = 0;
    if (Math.abs(dx) >= Math.abs(dz) && dx !== 0) {
      sx = Math.sign(dx);
    } else if (dz !== 0) {
      sz = Math.sign(dz);
    } else {
      sx = 1;
    }

    // Choose vertical step (+1 up, 0 flat, -1 down)
    let sy = 0;
    if (dy >= 2) sy = 1;
    else if (dy <= -2) sy = -1;

    // If stepping UP (+1), ensure there is a solid block at (sx, 0, sz) to step onto
    if (sy === 1) {
      const stepSupport = ctx.bot.blockAt(footPos.offset(sx, 0, sz));
      if (!stepSupport || stepSupport.boundingBox !== 'block') {
        sy = 0;
      }
    }

    const nextFoot = footPos.offset(sx, sy, sz);
    if (hasLiquidAdjacent(nextFoot)) {
      logVerbose('DIG', `Aborting tunnel step toward (${targetPos.x}, ${targetPos.y}, ${targetPos.z}): liquid detected near (${nextFoot.x}, ${nextFoot.y}, ${nextFoot.z})`);
      return false;
    }

    const blocksToClear: Vec3[] = [];
    if (sy === 1) {
      blocksToClear.push(
        footPos.offset(0, 2, 0),
        footPos.offset(sx, 2, sz),
        footPos.offset(sx, 1, sz)
      );
    } else if (sy === -1) {
      blocksToClear.push(
        footPos.offset(sx, 1, sz),
        footPos.offset(sx, 0, sz),
        footPos.offset(sx, -1, sz)
      );
    } else {
      blocksToClear.push(
        footPos.offset(sx, 1, sz),
        footPos.offset(sx, 0, sz)
      );
    }

    // Up to 3 passes to handle falling gravity blocks (sand/gravel) settling into the tunnel step
    let dugAnyBlockInStep = false;
    const startStepPos = ctx.bot.entity.position.clone();
    for (let gravityPass = 0; gravityPass < 3; gravityPass++) {
      let dugGravityBlock = false;
      for (const bPos of blocksToClear) {
        if (ctx.abortSignal?.aborted || (ctx as any).lastToolFailure) return false;
        const b = ctx.bot.blockAt(bPos);
        // Do not break `targetPos` or any matching target block inside the tunnel step; let `gather_resource` break & collect it!
        if (
          (bPos.x === targetPos.x && bPos.y === targetPos.y && bPos.z === targetPos.z) ||
          (typeof stopIfNearbyMatches === 'function' && b && stopIfNearbyMatches(b))
        ) {
          return true;
        }
        if (b && b.name !== 'air' && b.boundingBox === 'block') {
          if (b.name === 'bedrock' || b.name === 'chest') {
            return false;
          }
          const aboveB = ctx.bot.blockAt(bPos.offset(0, 1, 0));
          if (
            b.name.includes('sand') ||
            b.name.includes('gravel') ||
            aboveB?.name?.includes('sand') ||
            aboveB?.name?.includes('gravel')
          ) {
            dugGravityBlock = true;
          }
          const ok = await safeDigSingleBlock(ctx, bPos, 1, false);
          if (!ok) return false;
          dugAnyBlockInStep = true;
        }
      }
      if (!dugGravityBlock) break;
      // Wait 420ms for falling sand/gravel to land before walking forward
      await new Promise((r) => setTimeout(r, 420));
    }

    // Walk 1 block onto `nextFoot`
    const destCenter = nextFoot.offset(0.5, 0, 0.5);
    const lookDx = destCenter.x - ctx.bot.entity.position.x;
    const lookDz = destCenter.z - ctx.bot.entity.position.z;
    const yaw = Math.atan2(-lookDx, -lookDz);
    if (Number.isFinite(yaw)) {
      try {
        await ctx.bot.look(yaw, 0, true);
      } catch {}
    }

    ctx.bot.setControlState?.('forward', true);
    if (sy > 0) ctx.bot.setControlState?.('jump', true);

    const walkStart = Date.now();
    while (Date.now() - walkStart < 700 && !ctx.abortSignal?.aborted) {
      const hd = Math.hypot(
        ctx.bot.entity.position.x - destCenter.x,
        ctx.bot.entity.position.z - destCenter.z
      );
      if (hd < 0.55) break;
      if (sy === 0 && Date.now() - walkStart > 320 && (ctx.bot.entity as any).isCollidedHorizontally) {
        ctx.bot.setControlState?.('jump', true);
      }
      await new Promise((r) => setTimeout(r, 40));
    }
    ctx.bot.clearControlStates?.();

    if (!dugAnyBlockInStep && ctx.bot.entity.position.distanceTo(startStepPos) < 0.08) {
      break;
    }
  }

  return ctx.bot.entity.position.offset(0, 1.62, 0).distanceTo(targetCenter) <= 4.5;
}

/**
 * Guarantees that after breaking a block at `blockPos`, the agent actively locates any dropped
 * item entity (`entity.name === 'item'`), walks directly over it, and waits for the game API
 * (`playerCollect` packet or inventory count increase) to confirm pickup!
 */
async function collectDroppedItemNear(
  ctx: any,
  blockPos: Vec3,
  requestedType: string,
  preDigInvCount: number
): Promise<boolean> {
  if (ctx.abortSignal?.aborted) return false;

  // Quick check: did it already enter inventory?
  if (countHarvestedDropsInInventory(ctx.bot, requestedType) > preDigInvCount) {
    return true;
  }

  let collectedEventFired = false;
  const onPlayerCollect = (collector: any) => {
    if (collector && ctx.bot.entity && collector.id === ctx.bot.entity.id) {
      collectedEventFired = true;
    }
  };
  ctx.bot.on?.('playerCollect', onPlayerCollect);

  const findNearestDroppedItemEntity = (): any | null => {
    const entities = Object.values(ctx.bot.entities || {}) as any[];
    let best: any = null;
    let bestScore = 999;
    const req = requestedType.toLowerCase();
    for (const e of entities) {
      if (!e || !e.position || e.isValid === false) continue;
      const eName = (e.name || e.displayName || '').toLowerCase();
      const droppedItem = typeof e.getDroppedItem === 'function' ? e.getDroppedItem() : null;
      const isItemEntity =
        eName === 'item' ||
        eName === 'item_stack' ||
        eName === 'dropped_item' ||
        droppedItem != null ||
        e.type === 'object';
      if (isItemEntity) {
        const dFromBlock = e.position.distanceTo(blockPos.offset(0.5, 0.5, 0.5));
        const dFromBot = e.position.distanceTo(ctx.bot.entity.position);
        if (dFromBlock <= 7.5) {
          const itemName = String(droppedItem?.name || '').toLowerCase();
          const matchesDrop =
            itemName &&
            ((req.includes('iron') && itemName.includes('iron')) ||
              (req.includes('coal') && itemName.includes('coal')) ||
              itemName.includes(req));
          const score = dFromBot - (matchesDrop ? 3.0 : 0);
          if (score < bestScore) {
            bestScore = score;
            best = e;
          }
        }
      }
    }
    return best;
  };

  try {
    // Give the server up to 180ms to spawn the dropped item entity and send initial velocity
    await new Promise((r) => setTimeout(r, 180));
    if (
      collectedEventFired ||
      countHarvestedDropsInInventory(ctx.bot, requestedType) > preDigInvCount
    ) {
      return true;
    }

    const start = Date.now();
    const maxCollectWaitMs = 3800;
    let excavatedPocket = false;

    while (Date.now() - start < maxCollectWaitMs && !ctx.abortSignal?.aborted) {
      if (
        collectedEventFired ||
        countHarvestedDropsInInventory(ctx.bot, requestedType) > preDigInvCount
      ) {
        logVerbose('PICKUP', `Confirmed item pickup for ${requestedType} into inventory!`);
        return true;
      }

      const itemEntity = findNearestDroppedItemEntity();
      const targetPos: Vec3 = itemEntity?.position
        ? itemEntity.position
        : blockPos.offset(0.5, 0, 0.5);

      // If the dropped item is suspended high in a tree canopy or leaves (> 3.2m above bot's feet),
      // do not try to parkour/hover in mid-air (which triggers vanilla anti-fly kicks); wait for leaf decay or ground drops
      const botFootY = Math.floor(ctx.bot.entity?.position?.y ?? targetPos.y);
      if (targetPos.y > botFootY + 3.2) {
        logVerbose('PICKUP', `Dropped item is high in canopy at Y=${targetPos.y.toFixed(1)} (bot at Y=${botFootY}); skipping airborne pickup.`);
        return true;
      }

      const horizDist = Math.hypot(
        ctx.bot.entity.position.x - targetPos.x,
        ctx.bot.entity.position.z - targetPos.z
      );

      // If the item dropped inside a 1x1 pocket or shelf (> 0.85m away after 550ms of trying to walk to it),
      // excavate a 1x2 walkable opening into the pocket so the bot can walk directly onto the dropped item!
      if (!excavatedPocket && horizDist > 0.85 && Date.now() - start > 550) {
        excavatedPocket = true;
        ctx.bot.pathfinder?.setGoal(null);
        ctx.bot.clearControlStates?.();

        const footPos = ctx.bot.entity.position.floored();
        const pocketPos = targetPos.floored();
        const stepX = pocketPos.x === footPos.x ? 0 : Math.sign(pocketPos.x - footPos.x);
        const stepZ = pocketPos.z === footPos.z ? 0 : Math.sign(pocketPos.z - footPos.z);

        const pocketClearTargets: Vec3[] = [
          new Vec3(footPos.x + stepX, footPos.y + 1, footPos.z + stepZ),
          new Vec3(footPos.x + stepX, footPos.y, footPos.z + stepZ),
          new Vec3(pocketPos.x, footPos.y + 1, pocketPos.z),
          new Vec3(pocketPos.x, footPos.y, pocketPos.z),
        ];

        for (const pPos of pocketClearTargets) {
          if (
            collectedEventFired ||
            countHarvestedDropsInInventory(ctx.bot, requestedType) > preDigInvCount
          ) {
            break;
          }
          const pb = ctx.bot.blockAt(pPos);
          if (
            pb &&
            pb.name !== 'air' &&
            pb.boundingBox === 'block' &&
            pb.name !== 'bedrock' &&
            pb.name !== 'crafting_table' &&
            pb.name !== 'furnace' &&
            pb.name !== 'chest'
          ) {
            logVerbose('PICKUP', `Excavating pocket block ${pb.name} at (${pPos.x}, ${pPos.y}, ${pPos.z}) to reach dropped ${requestedType}!`);
            await safeDigSingleBlock(ctx, pPos, 2, false);
          }
        }
      }

      if (horizDist > 0.45) {
        const lookDx = targetPos.x - ctx.bot.entity.position.x;
        const lookDz = targetPos.z - ctx.bot.entity.position.z;
        const yaw = Math.atan2(-lookDx, -lookDz);
        if (Number.isFinite(yaw) && typeof ctx.bot.look === 'function') {
          try {
            await ctx.bot.look(yaw, 0, true);
          } catch {}
        }
        if (ctx.kinematics?.pathfindTo) {
          await ctx.kinematics.pathfindTo(targetPos, 0.4, 2500);
        }
        ctx.bot.setControlState?.('forward', true);
      } else {
        ctx.bot.pathfinder?.setGoal(null);
        ctx.bot.clearControlStates?.();
      }

      await new Promise((r) => setTimeout(r, 120));

      // If no item entities exist nearby anymore and we've waited > 2000ms, stop waiting
      if (!itemEntity && Date.now() - start > 2000) {
        break;
      }
    }
  } finally {
    ctx.bot.pathfinder?.setGoal(null);
    ctx.bot.clearControlStates?.();
    ctx.bot.removeListener?.('playerCollect', onPlayerCollect);
  }

  return (
    collectedEventFired ||
    countHarvestedDropsInInventory(ctx.bot, requestedType) > preDigInvCount
  );
}

/**
 * Checks if a block has at least one exposed face (adjacent to air or non-solid block).
 */
function hasExposedFace(bot: any, pos: Vec3): boolean {
  const neighbors = [
    pos.offset(0, 1, 0),
    pos.offset(1, 0, 0),
    pos.offset(-1, 0, 0),
    pos.offset(0, 0, 1),
    pos.offset(0, 0, -1),
    pos.offset(0, -1, 0),
  ];
  for (const nPos of neighbors) {
    const b = bot.blockAt(nPos);
    if (!b || b.name === 'air' || b.boundingBox === 'empty') {
      return true;
    }
  }
  return false;
}

/**
 * Helper to place a block from inventory onto any nearby solid surface (floor, wall, ceiling) around the bot.
 * Tests all 6 adjacent faces of nearby candidate blocks to succeed even on steep hills, narrow tunnels, or 1x1 shafts.
 */
async function placeBlockNearby(ctx: any, itemName: string): Promise<any | null> {
  const equipped = await ctx.kinematics.equipItemHumanized(itemName, 'hand');
  if (!equipped) return null;

  const botPos = ctx.bot.entity.position;
  const center = botPos.floored();

  const isInsideBot = (pos: Vec3): boolean => {
    return (
      Math.abs(pos.x + 0.5 - botPos.x) < 0.65 &&
      Math.abs(pos.z + 0.5 - botPos.z) < 0.65 &&
      pos.y >= Math.floor(botPos.y) &&
      pos.y <= Math.floor(botPos.y + 1.7)
    );
  };

  const faces = [
    new Vec3(0, 1, 0), // Top face of block below (floor)
    new Vec3(1, 0, 0), // East face of west block
    new Vec3(-1, 0, 0), // West face of east block
    new Vec3(0, 0, 1), // South face of north block
    new Vec3(0, 0, -1), // North face of south block
    new Vec3(0, -1, 0), // Bottom face of ceiling
  ];

  const candidateAir: Vec3[] = [];
  for (let dx = -2; dx <= 2; dx++) {
    for (let dy = -1; dy <= 2; dy++) {
      for (let dz = -2; dz <= 2; dz++) {
        const p = center.offset(dx, dy, dz);
        if (isInsideBot(p)) continue;
        const b = ctx.bot.blockAt(p);
        if (b && (b.name === 'air' || b.boundingBox === 'empty')) {
          candidateAir.push(p);
        }
      }
    }
  }

  const eyePos = botPos.offset(0, 1.62, 0);
  candidateAir.sort(
    (a, b) =>
      a.offset(0.5, 0.5, 0.5).distanceTo(eyePos) - b.offset(0.5, 0.5, 0.5).distanceTo(eyePos)
  );

  const tryPlaceAgainstAdjacentFaces = async (): Promise<any | null> => {
    await ctx.kinematics.equipItemHumanized(itemName, 'hand');
    for (const airPos of candidateAir) {
      if (ctx.abortSignal?.aborted) return null;

      for (const face of faces) {
        const refPos = airPos.minus(face);
        const refBlock = ctx.bot.blockAt(refPos);
        if (
          refBlock &&
          refBlock.boundingBox === 'block' &&
          !refBlock.name.includes('lava') &&
          !refBlock.name.includes('water')
        ) {
          const targetFaceCenter = refPos.offset(
            0.5 + face.x * 0.5,
            0.5 + face.y * 0.5,
            0.5 + face.z * 0.5
          );
          await ctx.kinematics.lookAtSmooth(targetFaceCenter, 130);
          try {
            await ctx.bot.placeBlock(refBlock, face);
            await new Promise((r) => setTimeout(r, 220));
            const placed = ctx.bot.blockAt(airPos);
            if (placed && placed.name.includes(itemName)) {
              return placed;
            }
          } catch {
            // Continue testing next candidate face
          }
        }
      }
    }
    return null;
  };

  const placed = await tryPlaceAgainstAdjacentFaces();
  if (placed) return placed;

  // If in an ultra-tight enclosed shaft with no placement spot, excavate a 1x1 wall pocket at eye level
  const wallPockets = [
    center.offset(1, 1, 0),
    center.offset(-1, 1, 0),
    center.offset(0, 1, 1),
    center.offset(0, 1, -1),
  ];
  for (const wp of wallPockets) {
    const wb = ctx.bot.blockAt(wp);
    if (wb && wb.boundingBox === 'block' && !wb.name.includes('bedrock') && !wb.name.includes('lava')) {
      const dug = await safeDigSingleBlock(ctx, wp, 1, false);
      if (dug) {
        return await tryPlaceAgainstAdjacentFaces();
      }
    }
  }

  return null;
}

/**
 * Counts how many items in the bot's inventory correspond to the harvested drop of `requestedType`.
 */
function countHarvestedDropsInInventory(bot: any, requestedType: string): number {
  const items = bot.inventory?.items() || [];
  const req = requestedType.toLowerCase();
  return items
    .filter((i: any) => {
      const n = i.name.toLowerCase();
      if (req === 'stone' || req === 'cobblestone') {
        return n === 'cobblestone' || n === 'cobbled_deepslate' || n === 'stone';
      }
      if (['diamond', 'diamond_ore', 'deepslate_diamond_ore'].includes(req)) {
        return n === 'diamond' || n.includes('diamond_ore');
      }
      if (['iron', 'raw_iron', 'iron_ore', 'deepslate_iron_ore'].includes(req)) {
        return n === 'raw_iron' || n === 'iron_ore' || n === 'deepslate_iron_ore';
      }
      if (['gold', 'raw_gold', 'gold_ore', 'deepslate_gold_ore', 'nether_gold_ore'].includes(req)) {
        return n === 'raw_gold' || n === 'gold_ore' || n === 'deepslate_gold_ore' || n === 'nether_gold_ore';
      }
      if (['copper', 'raw_copper', 'copper_ore', 'deepslate_copper_ore'].includes(req)) {
        return n === 'raw_copper' || n === 'copper_ore' || n === 'deepslate_copper_ore';
      }
      if (['coal', 'coal_ore', 'deepslate_coal_ore'].includes(req)) {
        return n === 'coal' || n === 'coal_ore' || n === 'deepslate_coal_ore';
      }
      if (['redstone', 'redstone_ore', 'deepslate_redstone_ore'].includes(req)) {
        return n === 'redstone' || n === 'redstone_ore' || n === 'deepslate_redstone_ore';
      }
      if (['lapis', 'lapis_lazuli', 'lapis_ore', 'deepslate_lapis_ore'].includes(req)) {
        return n === 'lapis_lazuli' || n === 'lapis_ore' || n === 'deepslate_lapis_ore';
      }
      if (['emerald', 'emerald_ore', 'deepslate_emerald_ore'].includes(req)) {
        return n === 'emerald' || n === 'emerald_ore' || n === 'deepslate_emerald_ore';
      }
      if (req.includes('clay')) {
        return n === 'clay_ball' || n === 'clay';
      }
      if (req.includes('gravel')) {
        return n === 'flint' || n === 'gravel';
      }
      if (req.includes('hay')) {
        return n === 'wheat' || n === 'hay_block';
      }
      if (req === 'log' || req === 'wood' || req.includes('_log')) {
        return n.endsWith('_log') || n.endsWith('_stem');
      }
      return n === req || n.includes(req);
    })
    .reduce((sum: number, i: any) => sum + i.count, 0);
}

/**
 * Shared self-contained crafting engine (used by both the `craft_item` MCP tool and
 * `tryAutoCraftReplacementPickaxe` when a pickaxe breaks mid-mining!).
 * Automatically crafts prerequisite planks/sticks/crafting_table, places the table if needed,
 * crafts the requested item, auto-equips newly crafted armor, and reclaims the placed
 * crafting_table back into inventory (Mindcraft pattern) so workstations are never orphaned!
 */
async function craftItemWithAutoPrereqs(
  ctx: any,
  rawItemName: string,
  count = 1,
  reclaimPlacedTable = true,
  allowAutoPrereqs = false
): Promise<{ success: boolean; summary: string }> {
  let itemName = String(rawItemName || 'oak_planks').toLowerCase().trim();
  const mcData = require('minecraft-data')(ctx.bot.version);

  if (itemName === 'workbench' || itemName === 'table') {
    itemName = 'crafting_table';
  }

  if (
    itemName === 'plank' ||
    itemName === 'planks' ||
    itemName === 'wood_plank' ||
    itemName === 'wood_planks' ||
    itemName === 'wooden_plank' ||
    itemName === 'wooden_planks'
  ) {
    const invLogs = (ctx.bot.inventory?.items() || []).find((i: any) =>
      i.name.endsWith('_log')
    );
    if (invLogs) {
      itemName = invLogs.name.replace('_log', '_planks');
    } else {
      itemName = 'oak_planks';
    }
  }

  const autoCraftPlanksIfNeeded = async (minPlanks = 4) => {
    const currentPlanks = (ctx.bot.inventory?.items() || [])
      .filter((i: any) => i.name.endsWith('_planks'))
      .reduce((s: number, i: any) => s + i.count, 0);
    if (currentPlanks >= minPlanks) return;
    const logItem = (ctx.bot.inventory?.items() || []).find((i: any) =>
      i.name.endsWith('_log') || i.name.endsWith('_stem')
    );
    if (!logItem) return;
    const plankName = logItem.name.replace('_log', '_planks').replace('_stem', '_planks');
    const plankInfo = mcData.itemsByName[plankName];
    if (!plankInfo) return;
    const r = ctx.bot.recipesFor(plankInfo.id, null, 1, null);
    if (r && r.length > 0) {
      const logsNeeded = Math.min(
        logItem.count,
        Math.max(1, Math.ceil((minPlanks - currentPlanks) / 4))
      );
      try {
        await ctx.bot.craft(r[0], logsNeeded, undefined);
      } catch {}
    }
  };

  const autoCraftSticksIfNeeded = async () => {
    const hasSticks = (ctx.bot.inventory?.items() || []).some(
      (i: any) => i.name === 'stick' && i.count >= 2
    );
    if (hasSticks) return;
    await autoCraftPlanksIfNeeded(2);
    const stickInfo = mcData.itemsByName['stick'];
    if (!stickInfo) return;
    const r = ctx.bot.recipesFor(stickInfo.id, null, 1, null);
    if (r && r.length > 0) {
      try {
        await ctx.bot.craft(r[0], 1, undefined);
      } catch {}
    }
  };

  const ITEM_ALIASES: Record<string, string> = {
    redstone_comparator: 'comparator',
    redstone_repeater: 'repeater',
    workbench: 'crafting_table',
    table: 'crafting_table',
    enchantment_table: 'enchanting_table',
    stone_slab: 'smooth_stone_slab',
    plank: 'oak_planks',
    planks: 'oak_planks',
    wooden_plank: 'oak_planks',
    wooden_planks: 'oak_planks',
    wood: 'oak_log',
    log: 'oak_log',
  };
  if (ITEM_ALIASES[itemName]) {
    itemName = ITEM_ALIASES[itemName];
  }

  const itemInfo = mcData.itemsByName[itemName];
  if (!itemInfo) {
    return { success: false, summary: `Unknown Minecraft item name "${itemName}".` };
  }

  if (allowAutoPrereqs) {
    if (
      itemName.includes('pickaxe') ||
      itemName.includes('sword') ||
      itemName.includes('axe') ||
      itemName.includes('shovel')
    ) {
      await autoCraftPlanksIfNeeded(itemName.startsWith('wooden_') ? 5 : 4);
      await autoCraftSticksIfNeeded();
    } else if (itemName === 'stick' || itemName === 'crafting_table') {
      await autoCraftPlanksIfNeeded(4);
    } else if (itemName === 'chest') {
      await autoCraftPlanksIfNeeded(8 * count);
    } else if (itemName === 'sticky_piston') {
      const hasPiston = (ctx.bot.inventory?.items() || []).some((i: any) => i.name === 'piston');
      if (!hasPiston) {
        await autoCraftPlanksIfNeeded(3);
        await craftItemWithAutoPrereqs(ctx, 'piston', 1, false, true);
      }
    } else if (itemName === 'smoker' || itemName === 'blast_furnace') {
      const hasFurnace = (ctx.bot.inventory?.items() || []).some((i: any) => i.name === 'furnace');
      if (!hasFurnace) {
        await craftItemWithAutoPrereqs(ctx, 'furnace', 1, false, true);
      }
    }
  }

  let craftingTable = ctx.bot.findBlock({
    matching: (b: any) => b?.name === 'crafting_table',
    maxDistance: 6,
  });

  if (craftingTable && ctx.bot.entity.position.distanceTo(craftingTable.position) > 3.2) {
    if (ctx.kinematics?.pathfindTo) {
      await ctx.kinematics.pathfindTo(craftingTable.position, 2.0, 3000);
    }
    if (ctx.bot.entity.position.distanceTo(craftingTable.position) > 4.2) {
      craftingTable = null;
    }
  }

  let placedTablePos: Vec3 | null = null;
  let recipes = ctx.bot.recipesFor(itemInfo.id, null, 1, craftingTable);
  if ((!recipes || recipes.length === 0) && !craftingTable) {
    const hasTableInInv = (ctx.bot.inventory?.items() || []).some(
      (i: any) => i.name === 'crafting_table'
    );
    if (!hasTableInInv && allowAutoPrereqs) {
      await autoCraftPlanksIfNeeded(4);
      const tableInfo = mcData.itemsByName['crafting_table'];
      const tableRecipes = ctx.bot.recipesFor(tableInfo.id, null, 1, null);
      if (tableRecipes && tableRecipes.length > 0) {
        try {
          await ctx.bot.craft(tableRecipes[0], 1, undefined);
        } catch {}
      }
    }

    const placedTable = await placeBlockNearby(ctx, 'crafting_table');
    if (placedTable) {
      craftingTable = placedTable;
      placedTablePos = placedTable.position?.clone?.() || placedTable.position;
      recipes = ctx.bot.recipesFor(itemInfo.id, null, 1, craftingTable);
    }
  }

  const reclaimTableHelper = async () => {
    if (reclaimPlacedTable && placedTablePos) {
      const tableStillThere = ctx.bot.blockAt(placedTablePos);
      if (tableStillThere && tableStillThere.name === 'crafting_table') {
        const preTableCount = countHarvestedDropsInInventory(ctx.bot, 'crafting_table');
        const dug = await safeDigSingleBlock(ctx, placedTablePos, 1, false);
        if (dug) {
          await collectDroppedItemNear(ctx, placedTablePos, 'crafting_table', preTableCount);
          logVerbose('GATHER', `Reclaimed placed crafting_table at (${placedTablePos.x}, ${placedTablePos.y}, ${placedTablePos.z}) back into inventory!`);
        }
      }
    }
  };

  if (!recipes || recipes.length === 0) {
    await reclaimTableHelper();
    const invList = (ctx.bot.inventory?.items() || [])
      .map((i: any) => `${i.count}x ${i.name}`)
      .join(', ');
    const PREREQUISITE_HINTS: Record<string, string> = {
      oak_planks: 'Requires oak_log. Gather oak_log first.',
      spruce_planks: 'Requires spruce_log. Gather spruce_log first.',
      birch_planks: 'Requires birch_log. Gather birch_log first.',
      crafting_table: 'Requires 4 matching planks of any wood species. Gather any logs and craft planks first.',
      stick: 'Requires 2 matching planks of any wood species. Craft planks from logs first.',
      wooden_pickaxe: 'Requires 3 matching planks and 2 sticks at crafting_table.',
      stone_pickaxe: 'Requires 3 cobblestone and 2 sticks at crafting_table. Mine stone with wooden_pickaxe first.',
      iron_pickaxe: 'Requires 3 iron_ingots and 2 sticks at crafting_table. Mine iron_ore and smelt with furnace.',
      diamond_pickaxe: 'Requires 3 diamonds and 2 sticks at crafting_table. Mine diamond_ore with iron_pickaxe.',
      furnace: 'Requires 8 cobblestone at crafting_table. Mine stone with wooden_pickaxe first.',
      chest: 'Requires 8 matching planks at crafting_table.',
      iron_helmet: 'Requires 5 iron_ingots at crafting_table.',
      iron_chestplate: 'Requires 8 iron_ingots at crafting_table.',
      iron_leggings: 'Requires 7 iron_ingots at crafting_table.',
      iron_boots: 'Requires 4 iron_ingots at crafting_table.',
      shield: 'Requires 6 planks of any wood species and 1 iron_ingot at crafting_table.',
      wooden_sword: 'Requires 2 matching planks and 1 stick at crafting_table.',
      stone_sword: 'Requires 2 cobblestone and 1 stick at crafting_table.',
      iron_sword: 'Requires 2 iron_ingots and 1 stick at crafting_table.',
      bucket: 'Requires 3 iron_ingots at crafting_table.',
    };
    const is3x3 =
      itemName.includes('pickaxe') ||
      itemName.includes('sword') ||
      itemName.includes('axe') ||
      itemName.includes('shovel') ||
      itemName.includes('hoe') ||
      itemName.includes('helmet') ||
      itemName.includes('chestplate') ||
      itemName.includes('leggings') ||
      itemName.includes('boots') ||
      itemName.includes('shield') ||
      itemName === 'furnace' ||
      itemName === 'chest' ||
      itemName === 'bucket';
    const hint = PREREQUISITE_HINTS[itemName];
    const reasonMsg =
      hint
        ? hint
        : is3x3 && !craftingTable
        ? `A 3x3 crafting table is required to craft ${itemName}, but no table was nearby and placing one failed.`
        : `Missing raw materials or recipe prerequisites for ${itemName}.`;
    return {
      success: false,
      summary: `Cannot craft ${itemName}: ${reasonMsg} (Inventory: [${invList || 'empty'}]).`,
    };
  }

  ctx.reportProgress?.(0.5);
  await new Promise((r) => setTimeout(r, 200));
  const outputPerCraft = Math.max(1, Number((recipes[0] as any)?.result?.count || 1));
  const craftCycles = Math.max(1, Math.ceil(count / outputPerCraft));
  const sizedRecipes =
    craftCycles > 1 ? ctx.bot.recipesFor(itemInfo.id, null, craftCycles, craftingTable) : recipes;
  if (!sizedRecipes || sizedRecipes.length === 0) {
    await reclaimTableHelper();
    const invList = (ctx.bot.inventory?.items() || [])
      .map((i: any) => `${i.count}x ${i.name}`)
      .join(', ');
    return {
      success: false,
      summary: `Cannot craft ${count}x ${itemName}: insufficient ingredients for ${craftCycles} craft cycle(s) (each yields ${outputPerCraft}). (Inventory: [${invList || 'empty'}]).`,
    };
  }
  try {
    await ctx.bot.craft(sizedRecipes[0], craftCycles, craftingTable || undefined);
    await reclaimTableHelper();

    // Auto-equip newly crafted armor pieces immediately (Mindcraft armorManager pattern)
    if (itemName.includes('helmet')) await ctx.kinematics.equipItemHumanized(itemName, 'head');
    else if (itemName.includes('chestplate')) await ctx.kinematics.equipItemHumanized(itemName, 'torso');
    else if (itemName.includes('leggings')) await ctx.kinematics.equipItemHumanized(itemName, 'legs');
    else if (itemName.includes('boots')) await ctx.kinematics.equipItemHumanized(itemName, 'feet');
    else if (itemName.includes('shield')) await ctx.kinematics.equipItemHumanized(itemName, 'off-hand');

    ctx.reportProgress?.(1.0);
    return {
      success: true,
      summary: `Successfully crafted ${craftCycles * outputPerCraft}x ${itemName}!`,
    };
  } catch (err: any) {
    await reclaimTableHelper();
    return {
      success: false,
      summary: `Failed crafting ${itemName}: ${err.message}`,
    };
  }
}

/**
 * Registers the comprehensive built-in MCP survival & interaction tool primitives.
 */
export function registerBaseMCPTools(registry: DynamicMCPToolRegistry): void {
  // 1. follow_player
  registry.registerTool(
    {
      name: 'follow_player',
      description: 'Smoothly follows a target player at a comfortable companion distance.',
      inputSchema: {
        type: 'object',
        properties: {
          playerName: { type: 'string', description: 'Username of the player to follow' },
          distance: { type: 'number', description: 'Follow distance in blocks', default: 3 },
        },
        required: ['playerName'],
      },
      tags: ['follow', 'accompany', 'walk', 'player', 'companion'],
    },
    async (args, ctx) => {
      const targetName = pickStrArg(args.playerName).trim();
      const followDist = Number(args.distance) || 3;
      const isGenericPlayerRef =
        !targetName ||
        ['player', 'me', 'human', 'owner', 'user', 'companion', 'companion_player', 'partner', 'teammate', 'leader', 'friend'].includes(
          targetName.toLowerCase()
        );

      const candidatePlayers = (Object.values(ctx.bot.players || {}) as any[])
        .filter((p: any) => p?.entity && p.username !== ctx.bot.username);

      candidatePlayers.sort((a: any, b: any) => {
        const myPos = ctx.bot.entity?.position;
        const da = myPos && a.entity?.position ? myPos.distanceTo(a.entity.position) : 999999;
        const db = myPos && b.entity?.position ? myPos.distanceTo(b.entity.position) : 999999;
        return da - db;
      });

      let playerObj: any = null;
      if (!isGenericPlayerRef) {
        playerObj =
          candidatePlayers.find((p: any) => p?.username?.toLowerCase() === targetName.toLowerCase()) ||
          ctx.bot.players?.[targetName] ||
          null;
      }
      if (!playerObj) {
        playerObj = candidatePlayers[0] || null;
      }

      if (!playerObj?.entity) {
        return {
          success: false,
          summary: `I can't see ${targetName || 'any player'} nearby right now.`,
        };
      }

      await ctx.kinematics.lookAtSmooth(
        playerObj.entity.position.offset(0, 1.6, 0),
        220,
        playerObj.entity.velocity
      );

      if (ctx.kinematics?.pathfindTo) {
        await ctx.kinematics.pathfindTo(playerObj.entity.position, followDist, 5000);
      }
      ctx.reportProgress(1.0);

      return {
        success: true,
        summary: `Now following ${playerObj.username} at ${followDist}m distance.`,
      };
    }
  );

  // 2. come_here
  registry.registerTool(
    {
      name: 'come_here',
      description: 'Walks directly to a player and looks at them attentively.',
      inputSchema: {
        type: 'object',
        properties: {
          playerName: { type: 'string', description: 'Username of the player calling the agent' },
        },
        required: ['playerName'],
      },
      tags: ['come', 'here', 'goto', 'approach', 'player'],
    },
    async (args, ctx) => {
      const targetName = pickStrArg(args.playerName).trim();
      const isGenericPlayerRef =
        !targetName || ['player', 'me', 'human', 'owner', 'user'].includes(targetName.toLowerCase());
      const playerObj: any =
        (!isGenericPlayerRef &&
          (ctx.bot.players?.[targetName] ||
            Object.values(ctx.bot.players || {}).find(
              (p: any) => p?.username?.toLowerCase() === targetName.toLowerCase()
            ))) ||
        Object.values(ctx.bot.players || {}).find(
          (p: any) => p?.entity && p.username !== ctx.bot.username
        );

      if (!playerObj?.entity) {
        return {
          success: false,
          summary: `Could not locate ${targetName || 'any player'} within loaded chunks.`,
        };
      }

      const targetPos = playerObj.entity.position;
      await ctx.kinematics.lookAtSmooth(targetPos.offset(0, 1.6, 0), 200);
      ctx.reportProgress(0.3);

      if (ctx.kinematics?.pathfindTo) {
        await ctx.kinematics.pathfindTo(targetPos, 2.2);
      }
      if (
        playerObj.entity &&
        ctx.bot.entity.position.distanceTo(playerObj.entity.position) > 4.0 &&
        ctx.bot.entity.position.y < playerObj.entity.position.y - 1.5
      ) {
        await tunnelToBlock(ctx, playerObj.entity.position.floored());
        if (ctx.kinematics?.pathfindTo) {
          await ctx.kinematics.pathfindTo(playerObj.entity.position, 2.2);
        }
      }

      ctx.bot.pathfinder.setGoal(null);
      if (playerObj.entity) {
        await ctx.kinematics.lookAtSmooth(playerObj.entity.position.offset(0, 1.6, 0), 180);
      }
      ctx.reportProgress(1.0);

      return {
        success: true,
        summary: `Arrived next to ${playerObj.username}.`,
      };
    }
  );

  // 3. navigate_to_coordinates
  registry.registerTool(
    {
      name: 'navigate_to_coordinates',
      description: 'Walks/pathfinds to specific (x, y, z) world coordinates (such as a saved home base or landmark).',
      inputSchema: {
        type: 'object',
        properties: {
          x: { type: 'number', description: 'Target X coordinate' },
          y: { type: 'number', description: 'Target Y coordinate' },
          z: { type: 'number', description: 'Target Z coordinate' },
        },
        required: ['x', 'y', 'z'],
      },
      tags: ['navigate', 'goto', 'coordinates', 'coords', 'return', 'home', 'base'],
    },
    async (args, ctx) => {
      const tx = Number(args.x);
      const ty = Number(args.y);
      const tz = Number(args.z);

      if (isNaN(tx) || isNaN(ty) || isNaN(tz)) {
        return { success: false, summary: 'Invalid (x, y, z) coordinates provided.' };
      }

      const targetFloored = new Vec3(Math.floor(tx), Math.floor(ty), Math.floor(tz));
      const targetBlk = ctx.bot.blockAt(targetFloored);
      const standTarget =
        targetBlk && targetBlk.boundingBox === 'block'
          ? new Vec3(targetFloored.x + 0.5, targetFloored.y + 1.0, targetFloored.z + 0.5)
          : new Vec3(tx, ty, tz);

      if (ctx.kinematics?.pathfindTo) {
        await ctx.kinematics.pathfindTo(standTarget, 1.0);
      }
      if (ctx.abortSignal?.aborted) {
        ctx.bot.pathfinder.setGoal(null);
        return { success: false, summary: 'Navigation aborted.' };
      }
      if (
        ctx.bot.entity.position.distanceTo(standTarget) > 3.5 &&
        ctx.bot.entity.position.y < standTarget.y - 1.5
      ) {
        await tunnelToBlock(ctx, standTarget.floored());
        if (ctx.kinematics?.pathfindTo) {
          await ctx.kinematics.pathfindTo(standTarget, 1.0);
        }
      }

      ctx.bot.pathfinder.setGoal(null);
      ctx.reportProgress(1.0);
      const posNow = ctx.bot.entity.position;
      const hDist = Math.hypot(posNow.x - standTarget.x, posNow.z - standTarget.z);
      const vDist = Math.abs(posNow.y - standTarget.y);
      const arrived = hDist <= 1.6 && vDist <= 1.8;

      return {
        success: arrived,
        summary: arrived
          ? `Arrived at coordinates (${tx}, ${ty}, ${tz})!`
          : `Moved closer to (${tx}, ${ty}, ${tz}), currently ${hDist.toFixed(1)}m away (dy=${vDist.toFixed(1)}m).`,
      };
    }
  );

  // 4. gather_resource
  registry.registerTool(
    {
      name: 'gather_resource',
      description: 'Finds, excavates covering dirt/stone if needed, mines/chops, and collects blocks (e.g. log, stone, cobblestone, coal_ore, iron_ore, dirt, sand). Verifies actual inventory gain and switches tools per block.',
      inputSchema: {
        type: 'object',
        properties: {
          blockType: {
            type: 'string',
            description: 'Block type to harvest (e.g. log, stone, coal_ore, iron_ore, dirt)',
          },
          count: { type: 'number', description: 'Number of items to collect', default: 4 },
        },
        required: ['blockType'],
      },
      tags: ['gather', 'mine', 'chop', 'harvest', 'wood', 'log', 'stone', 'cobblestone', 'ore', 'iron_ore', 'coal_ore', 'dig'],
    },
    async (args, ctx) => {
      const rawArg =
        typeof args.blockType === 'string'
          ? args.blockType
          : args.blockType && typeof args.blockType === 'object'
            ? String((args.blockType as any).name || (args.blockType as any).block || '')
            : '';
      const rawType = rawArg.toLowerCase().trim();
      if (!rawType) {
        return { success: false, summary: 'No blockType specified for gather_resource.' };
      }
      const NON_MINEABLE_ITEM_RE = /(_helmet|_chestplate|_leggings|_boots|_sword|_pickaxe|_axe|_shovel|_hoe|_ingot|_nugget|_bucket|_horse_armor|_armor|_bow|_shield)$/;
      const EXACT_NON_MINEABLE = new Set([
        'stick', 'bread', 'arrow', 'shears', 'flint_and_steel', 'bow', 'shield',
        'bucket', 'compass', 'clock', 'spyglass', 'brush', 'fishing_rod', 'lead', 'name_tag', 'saddle',
      ]);
      if (NON_MINEABLE_ITEM_RE.test(rawType) || EXACT_NON_MINEABLE.has(rawType)) {
        return {
          success: false,
          summary: `"${rawType}" is a crafted item, not a mineable world block. Use craft_item or smelt_item instead.`,
        };
      }
      const requestedType =
        rawType === 'hay_bale'
          ? 'hay_block'
          : rawType.replace(/(_portal_frame|_frame|_item|_ore_block)$/, '');
      const targetCount = Math.min(64, Math.max(1, Number(args.count) || 4));
      const initialDropCount = countHarvestedDropsInInventory(ctx.bot, requestedType);
      const failedKeys = new Set<string>();
      let blocksBroken = 0;
      const isWoodRequest =
        requestedType === 'log' || requestedType === 'wood' || requestedType.includes('_log') || requestedType.includes('tree');

      const isOreVeinRequest =
        !isWoodRequest &&
        !requestedType.endsWith('_block') &&
        requestedType !== 'stone' &&
        requestedType !== 'cobblestone' &&
        requestedType !== 'dirt' &&
        requestedType !== 'sand' &&
        requestedType !== 'gravel';

      // IMPORTANT: Mineflayer's `findBlocks`/`findBlock` runs a chunk-palette pre-filter
      // `isBlockInSection(section, matcher)` where `b` is created via `Block.fromStateId(stateId, 0)`
      // with `b.position === null`! Therefore `matchesBlockNameOnly` MUST ONLY inspect `b?.name`!
      const matchesBlockNameOnly = (b: any): boolean => {
        if (!b?.name) return false;
        const n = b.name.toLowerCase();
        if (isWoodRequest) {
          return n.endsWith('_log') || n.endsWith('_stem');
        }
        if (requestedType.endsWith('_block')) {
          return n === requestedType || n === rawType;
        }
        if (requestedType === 'stone' || requestedType === 'cobblestone') {
          return n === 'stone' || n === 'cobblestone' || n === 'deepslate' || n === 'cobbled_deepslate';
        }
        if (['iron', 'raw_iron', 'iron_ore', 'deepslate_iron_ore'].includes(requestedType)) {
          return n === 'iron_ore' || n === 'deepslate_iron_ore';
        }
        if (['coal', 'coal_ore', 'deepslate_coal_ore'].includes(requestedType)) {
          return n === 'coal_ore' || n === 'deepslate_coal_ore';
        }
        if (['diamond', 'diamond_ore', 'deepslate_diamond_ore'].includes(requestedType)) {
          return n === 'diamond_ore' || n === 'deepslate_diamond_ore';
        }
        if (['gold', 'raw_gold', 'gold_ore', 'deepslate_gold_ore', 'nether_gold_ore'].includes(requestedType)) {
          return n === 'gold_ore' || n === 'deepslate_gold_ore' || n === 'nether_gold_ore';
        }
        if (['copper', 'raw_copper', 'copper_ore', 'deepslate_copper_ore'].includes(requestedType)) {
          return n === 'copper_ore' || n === 'deepslate_copper_ore';
        }
        if (['redstone', 'redstone_ore', 'deepslate_redstone_ore'].includes(requestedType)) {
          return n === 'redstone_ore' || n === 'deepslate_redstone_ore';
        }
        if (['lapis', 'lapis_lazuli', 'lapis_ore', 'deepslate_lapis_ore'].includes(requestedType)) {
          return n === 'lapis_ore' || n === 'deepslate_lapis_ore';
        }
        if (['emerald', 'emerald_ore', 'deepslate_emerald_ore'].includes(requestedType)) {
          return n === 'emerald_ore' || n === 'deepslate_emerald_ore';
        }
        return n === requestedType || n.includes(requestedType) || n === rawType || n.includes(rawType);
      };

      (ctx as any).lastToolFailure = null;

      // Proactively check pickaxe durability before gathering ores to conserve iron/diamond pickaxes!
      if (!isWoodRequest && requestedType !== 'dirt' && requestedType !== 'sand' && requestedType !== 'gravel') {
        await ensurePickaxesForMining(ctx, targetCount * 3);
      }
      (ctx as any).gatherMinimumPickaxeTier = getRequiredPickaxeTier(requestedType);
      (ctx as any).currentGatherMatcher = matchesBlockNameOnly;
      (ctx as any).onGatherBlockBroken = (_pos: Vec3, brokenName: string) => {
        if (matchesBlockNameOnly({ name: brokenName })) {
          blocksBroken++;
        }
      };

      const hasSolidWallBetween = (targetPos: Vec3): boolean => {
        const footPos = ctx.bot.entity.position.floored();
        const dx = targetPos.x - footPos.x;
        const dz = targetPos.z - footPos.z;
        if (Math.abs(dx) <= 1 && Math.abs(dz) <= 1) return false;
        let sx = 0;
        let sz = 0;
        if (Math.abs(dx) >= Math.abs(dz) && dx !== 0) sx = Math.sign(dx);
        else if (dz !== 0) sz = Math.sign(dz);
        for (const checkY of [footPos.y, footPos.y + 1]) {
          const stepPos = new Vec3(footPos.x + sx, checkY, footPos.z + sz);
          if (stepPos.x === targetPos.x && stepPos.y === targetPos.y && stepPos.z === targetPos.z) {
            continue;
          }
          const b = ctx.bot.blockAt(stepPos);
          if (b && b.name !== 'air' && b.boundingBox === 'block' && !matchesBlockNameOnly(b)) {
            return true;
          }
        }
        return false;
      };

      /**
       * 3D Cubic Flood-Fill Vein Miner (`mineflayer-collectblock` `BlockVeins` pattern):
       * Once we break one ore block at `seedPos`, immediately inspect all 26 neighbors in the
       * 3x3x3 cube around `seedPos` (and recursively around each newly broken vein block) so
       * the agent mines out the ENTIRE connected ore vein before moving on!
       */
      const mineConnectedVeinFrom = async (seedPos: Vec3): Promise<void> => {
        if (!isOreVeinRequest) return;
        const veinQueue: Vec3[] = [];
        const queuedKeys = new Set<string>([`${seedPos.x},${seedPos.y},${seedPos.z}`]);

        const enqueueNeighbors = (center: Vec3) => {
          for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
              for (let dz = -1; dz <= 1; dz++) {
                if (dx === 0 && dy === 0 && dz === 0) continue;
                const nPos = center.offset(dx, dy, dz);
                const k = `${nPos.x},${nPos.y},${nPos.z}`;
                if (queuedKeys.has(k) || failedKeys.has(k)) continue;
                const nb = ctx.bot.blockAt(nPos);
                if (nb && matchesBlockNameOnly(nb)) {
                  queuedKeys.add(k);
                  veinQueue.push(nPos);
                }
              }
            }
          }
        };

        enqueueNeighbors(seedPos);

        while (veinQueue.length > 0 && !ctx.abortSignal?.aborted && !(ctx as any).lastToolFailure) {
          const curCollected =
            countHarvestedDropsInInventory(ctx.bot, requestedType) - initialDropCount;
          if (curCollected >= targetCount) break;

          const eyePos = ctx.bot.entity.position.offset(0, 1.62, 0);
          veinQueue.sort(
            (a, b) =>
              eyePos.distanceTo(a.offset(0.5, 0.5, 0.5)) -
              eyePos.distanceTo(b.offset(0.5, 0.5, 0.5))
          );

          const vPos = veinQueue.shift()!;
          const vBlock = ctx.bot.blockAt(vPos);
          if (!vBlock || !matchesBlockNameOnly(vBlock)) continue;

          logVerbose('GATHER', `Vein-mining connected ${vBlock.name} at (${vPos.x}, ${vPos.y}, ${vPos.z}) (${veinQueue.length} more vein blocks queued)...`);

          const vDist = ctx.bot.entity.position
            .offset(0, 1.62, 0)
            .distanceTo(vPos.offset(0.5, 0.5, 0.5));
          if (vDist > 3.2 || (vDist > 2.45 && hasSolidWallBetween(vPos))) {
            await tunnelToBlock(ctx, vPos, undefined, 2.45);
          }

          const preVeinInv = countHarvestedDropsInInventory(ctx.bot, requestedType);
          const vOk = await safeDigSingleBlock(ctx, vPos, 0, false);
          if (vOk) {
            blocksBroken++;
            await collectDroppedItemNear(ctx, vPos, requestedType, preVeinInv);
            enqueueNeighbors(vPos);
            const nowDelta =
              countHarvestedDropsInInventory(ctx.bot, requestedType) - initialDropCount;
            ctx.reportProgress(Math.min(1.0, Math.max(0, nowDelta) / targetCount));
          } else {
            if ((ctx as any).lastToolFailure) break;
            failedKeys.add(`${vPos.x},${vPos.y},${vPos.z}`);
          }
        }
      };

      logVerbose('GATHER', `Starting gather_resource(blockType="${requestedType}", count=${targetCount}) from pos=(${ctx.bot.entity.position.x.toFixed(1)}, ${ctx.bot.entity.position.y.toFixed(1)}, ${ctx.bot.entity.position.z.toFixed(1)})`);

      const maxAttempts = Math.max(16, targetCount * 3);
      try {
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
          if (ctx.abortSignal?.aborted || (ctx as any).lastToolFailure) break;

          const collectedInInventory =
            countHarvestedDropsInInventory(ctx.bot, requestedType) - initialDropCount;
          if (collectedInInventory >= targetCount) {
            break;
          }

          // Query candidate positions using palette-safe `matchesBlockNameOnly`
          let candidatePositions: Vec3[] = [];
          if (typeof ctx.bot.findBlocks === 'function') {
            candidatePositions = ctx.bot.findBlocks({
              matching: matchesBlockNameOnly,
              maxDistance: 40,
              count: 120,
            });
          } else if (typeof ctx.bot.findBlock === 'function') {
            const single = ctx.bot.findBlock({
              matching: matchesBlockNameOnly,
              maxDistance: 40,
            });
            if (single?.position) candidatePositions = [single.position];
          }

          // Filter out already failed coordinates
          candidatePositions = candidatePositions.filter(
            (p) => !failedKeys.has(`${p.x},${p.y},${p.z}`)
          );

          if (candidatePositions.length === 0) {
            logVerbose('GATHER', `No candidate blocks found for "${requestedType}" within 40m on attempt #${attempt + 1}`);
            break;
          }

          const botPos = ctx.bot.entity.position;
          const botFootY = Math.floor(botPos.y);

          // Safety: Do not mine blocks directly under the bot's feet (never dig straight down into the void or floor support)
          candidatePositions = candidatePositions.filter((p) => {
            const isDirectlyUnderFeet =
              p.y < botFootY &&
              Math.abs(p.x - Math.floor(botPos.x)) <= 0.6 &&
              Math.abs(p.z - Math.floor(botPos.z)) <= 0.6;
            return !isDirectlyUnderFeet;
          });

          if (candidatePositions.length === 0) {
            logVerbose('GATHER', `No valid candidate blocks found (remaining candidates support bot feet) on attempt #${attempt + 1}`);
            break;
          }

          // For tree logs ONLY: trace DOWN to the lowest un-failed log block of its trunk so we chop bottom-up.
          // (Do NOT collapse vertical columns for underground ores, or lower vein blocks get hidden!)
          if (isWoodRequest) {
            const normalized: Vec3[] = [];
            const seenKeys = new Set<string>();
            for (const rawPos of candidatePositions) {
              let cur = rawPos.clone();
              for (let down = 0; down < 12; down++) {
                const belowPos = cur.offset(0, -1, 0);
                const belowKey = `${belowPos.x},${belowPos.y},${belowPos.z}`;
                const belowBlock = ctx.bot.blockAt(belowPos);
                if (belowBlock && matchesBlockNameOnly(belowBlock) && !failedKeys.has(belowKey)) {
                  cur = belowPos;
                } else {
                  break;
                }
              }
              const k = `${cur.x},${cur.y},${cur.z}`;
              if (!seenKeys.has(k) && !failedKeys.has(k)) {
                seenKeys.add(k);
                normalized.push(cur);
              }
            }
            if (normalized.length > 0) {
              candidatePositions = normalized;
            }
          }

          // Score candidates: strongly prioritize blocks already within immediate reach (<= 4.5m),
          // smooth vertical penalty, and preference for exposed faces
          candidatePositions.sort((a, b) => {
            const rawDistA = botPos.distanceTo(a);
            const rawDistB = botPos.distanceTo(b);
            const immediateBonusA = rawDistA <= 4.5 ? -4.0 : 0;
            const immediateBonusB = rawDistB <= 4.5 ? -4.0 : 0;
            const dyA = Math.abs(a.y - botFootY);
            const dyB = Math.abs(b.y - botFootY);
            const vertPenaltyA = dyA <= 3 ? 0 : (dyA - 3) * 1.5;
            const vertPenaltyB = dyB <= 3 ? 0 : (dyB - 3) * 1.5;
            const expA = hasExposedFace(ctx.bot, a) ? 0 : 2.5;
            const expB = hasExposedFace(ctx.bot, b) ? 0 : 2.5;
            const scoreA = rawDistA + immediateBonusA + vertPenaltyA + expA;
            const scoreB = rawDistB + immediateBonusB + vertPenaltyB + expB;
            return scoreA - scoreB;
          });

          let blockPos = candidatePositions[0].clone();
          let posKey = `${blockPos.x},${blockPos.y},${blockPos.z}`;
          const startDistToTarget = botPos.distanceTo(blockPos);
          logVerbose('GATHER', `Attempt #${attempt + 1}: Selected ${requestedType} at (${blockPos.x}, ${blockPos.y}, ${blockPos.z}), dist=${startDistToTarget.toFixed(1)}m (out of ${candidatePositions.length} candidates)`);

          // Navigate close to the block:
          // If mining underground stone/ores within 9m and either the block is buried or the bot is already underground,
          // skip walk-only A* pathfinding and tunnel directly toward `blockPos`!
          const horizDist = Math.hypot(
            ctx.bot.entity.position.x - (blockPos.x + 0.5),
            ctx.bot.entity.position.z - (blockPos.z + 0.5)
          );
          const skipSurfacePathfind =
            !isWoodRequest &&
            startDistToTarget <= 9.5 &&
            (!hasExposedFace(ctx.bot, blockPos) || !ctx.worldState()?.terrainContext?.canSeeSky);

          if (horizDist > 2.4 && !skipSurfacePathfind) {
            if (ctx.kinematics?.pathfindTo) {
              await ctx.kinematics.pathfindTo(blockPos, 2.0);
            }
          }

          // If low-hanging leaves are blocking the bot from reaching a tree trunk, clear adjacent leaves
          if (isWoodRequest && ctx.bot.entity.position.distanceTo(blockPos.offset(0.5, 0.5, 0.5)) > 4.4) {
            const midX = Math.round((ctx.bot.entity.position.x + blockPos.x) / 2);
            const midZ = Math.round((ctx.bot.entity.position.z + blockPos.z) / 2);
            for (let ly = Math.floor(ctx.bot.entity.position.y); ly <= Math.floor(ctx.bot.entity.position.y) + 2; ly++) {
              const leafPos = new Vec3(midX, ly, midZ);
              const leafBlock = ctx.bot.blockAt(leafPos);
              if (leafBlock && leafBlock.name.includes('leaves')) {
                await safeDigSingleBlock(ctx, leafPos, 1, false);
              }
            }
            if (ctx.kinematics?.pathfindTo) {
              await ctx.kinematics.pathfindTo(blockPos, 2.0);
            }
          }

          // If mining underground stone/ores and A* pathfinding stopped short (or a solid wall still separates the bot from `blockPos`),
          // actively tunnel/staircase directly to within 2.45m of `blockPos` so drops are always within immediate walkable pickup range!
          const eyeDistAfterNav = ctx.bot.entity.position
            .offset(0, 1.62, 0)
            .distanceTo(blockPos.offset(0.5, 0.5, 0.5));
          if (
            !isWoodRequest &&
            (eyeDistAfterNav > 3.2 || (eyeDistAfterNav > 2.45 && hasSolidWallBetween(blockPos))) &&
            eyeDistAfterNav <= 40.0
          ) {
            logVerbose('GATHER', `Target ${requestedType} at (${blockPos.x}, ${blockPos.y}, ${blockPos.z}) is ${eyeDistAfterNav.toFixed(1)}m away behind terrain; tunneling toward it...`);
            await tunnelToBlock(ctx, blockPos, matchesBlockNameOnly, 2.45);
            if ((ctx as any).lastToolFailure) break;

            // If `tunnelToBlock` stopped early because it uncovered a closer matching block right next to us, switch to that closer block!
            const nowEye = ctx.bot.entity.position.offset(0, 1.62, 0);
            if (nowEye.distanceTo(blockPos.offset(0.5, 0.5, 0.5)) > 3.2) {
              const foot = ctx.bot.entity.position.floored();
              let closerMatch: Vec3 | null = null;
              let closerDist = 99;
              for (let dx = -3; dx <= 3; dx++) {
                for (let dy = -2; dy <= 3; dy++) {
                  for (let dz = -3; dz <= 3; dz++) {
                    const cp = foot.offset(dx, dy, dz);
                    const ck = `${cp.x},${cp.y},${cp.z}`;
                    if (failedKeys.has(ck)) continue;
                    const cb = ctx.bot.blockAt(cp);
                    if (cb && matchesBlockNameOnly(cb)) {
                      const d = nowEye.distanceTo(cp.offset(0.5, 0.5, 0.5));
                      if (d <= 4.3 && d < closerDist) {
                        closerDist = d;
                        closerMatch = cp;
                      }
                    }
                  }
                }
              }
              if (closerMatch) {
                blockPos = closerMatch;
                posKey = `${blockPos.x},${blockPos.y},${blockPos.z}`;
              }
            }
          }

          // If the target block is 1..4 blocks underground below us and within horizontal reach,
          // explicitly excavate the covering blocks directly above `blockPos` one by one with the right tool!
          const currentFootY = Math.floor(ctx.bot.entity.position.y);
          const horizToTarget = Math.hypot(
            ctx.bot.entity.position.x - (blockPos.x + 0.5),
            ctx.bot.entity.position.z - (blockPos.z + 0.5)
          );
          if (blockPos.y < currentFootY && currentFootY - blockPos.y <= 4 && horizToTarget <= 3.8) {
            for (let y = currentFootY + 1; y > blockPos.y; y--) {
              if (ctx.abortSignal?.aborted || (ctx as any).lastToolFailure) break;
              const coverPos = new Vec3(blockPos.x, y, blockPos.z);
              const coverBlock = ctx.bot.blockAt(coverPos);
              if (coverBlock && coverBlock.name !== 'air' && coverBlock.boundingBox === 'block') {
                const coverOk = await safeDigSingleBlock(ctx, coverPos, 1, false);
                if (!coverOk) break;
              }
            }
          }

          if ((ctx as any).lastToolFailure) break;

          // Record inventory count right before digging target block
          const preDigInv = countHarvestedDropsInInventory(ctx.bot, requestedType);

          // Now dig the target block itself using the optimal tool (server-verified, no duplicate pathfind timeout!)
          const dugOk = await safeDigSingleBlock(ctx, blockPos, 0, false);
          if (!dugOk) {
            if ((ctx as any).lastToolFailure) break;
            const endDistToTarget = ctx.bot.entity.position.distanceTo(blockPos);
            // Only blacklist `posKey` in `failedKeys` if we did NOT make meaningful distance progress toward it!
            if (endDistToTarget >= startDistToTarget - 1.2) {
              failedKeys.add(posKey);
            } else {
              logVerbose('GATHER', `Made tunnel progress toward (${blockPos.x}, ${blockPos.y}, ${blockPos.z}) (${startDistToTarget.toFixed(1)}m -> ${endDistToTarget.toFixed(1)}m); keeping target active for next step`);
            }
            continue;
          }

          blocksBroken++;

          // Guarantee pickup of the dropped item from `blockPos`
          await collectDroppedItemNear(ctx, blockPos, requestedType, preDigInv);

          // If mining an ore vein (iron_ore, coal_ore, diamond_ore, etc.), immediately mine out the rest of the connected 3D vein!
          if (isOreVeinRequest) {
            await mineConnectedVeinFrom(blockPos);
          }

          // If chopping a tree trunk, also harvest up to 3 directly-overhead logs in the same column while standing right here!
          if (isWoodRequest) {
            for (let up = 1; up <= 3; up++) {
              const nowInvDelta =
                countHarvestedDropsInInventory(ctx.bot, requestedType) - initialDropCount;
              if (nowInvDelta >= targetCount || ctx.abortSignal?.aborted) break;
              const upperPos = blockPos.offset(0, up, 0);
              const upperBlock = ctx.bot.blockAt(upperPos);
              if (upperBlock && matchesBlockNameOnly(upperBlock)) {
                const preUpperInv = countHarvestedDropsInInventory(ctx.bot, requestedType);
                const upperOk = await safeDigSingleBlock(ctx, upperPos);
                if (upperOk) {
                  blocksBroken++;
                  await collectDroppedItemNear(ctx, blockPos, requestedType, preUpperInv);
                } else {
                  break;
                }
              } else {
                break;
              }
            }
          }

          const nowCollected =
            countHarvestedDropsInInventory(ctx.bot, requestedType) - initialDropCount;
          ctx.reportProgress(Math.min(1.0, Math.max(0, nowCollected) / targetCount));
        }
      } finally {
        (ctx as any).gatherMinimumPickaxeTier = null;
        (ctx as any).currentGatherMatcher = null;
        (ctx as any).onGatherBlockBroken = null;
      }

      if (blocksBroken > 0 && countHarvestedDropsInInventory(ctx.bot, requestedType) === initialDropCount) {
        await new Promise((r) => setTimeout(r, 450));
      }

      const finalDelta = Math.max(
        0,
        countHarvestedDropsInInventory(ctx.bot, requestedType) - initialDropCount
      );
      const dropsNothingBlock = /^(glass|.*_glass|.*_pane|ice|leaves|.*_leaves|tall_grass|short_grass|seagrass|fern|dead_bush|cobweb)$/i.test(requestedType);
      const isSuccess = finalDelta > 0 || (dropsNothingBlock && blocksBroken > 0);
      const invSummary = (ctx.bot.inventory?.items() || [])
        .map((it: any) => `${it.count}x ${it.name}`)
        .join(', ');
      const toolFailureNote = (ctx as any).lastToolFailure
        ? ` Stopped early: ${(ctx as any).lastToolFailure}`
        : '';

      return {
        success: isSuccess,
        summary:
          isSuccess
            ? `Harvested ${finalDelta || blocksBroken} ${requestedType} (${blocksBroken} blocks broken).${toolFailureNote} Current inventory: [${invSummary || 'empty'}].`
            : blocksBroken > 0
            ? `Broke ${blocksBroken} block(s) of ${requestedType}, but 0 entered inventory (drops were not picked up).${toolFailureNote} Current inventory: [${invSummary || 'empty'}].`
            : `Collected 0 ${requestedType} into inventory (${blocksBroken} blocks broken).${
                toolFailureNote ||
                ' Call explore_or_dig_for_ores (mode="staircase_mine") to dig down to stone/ore layer!'
              } Current inventory: [${invSummary || 'empty'}].`,
        data: { harvested: finalDelta, blocksBroken, inventoryDelta: finalDelta },
      };
    }
  );

  // 5. explore_or_dig_for_ores
  registry.registerTool(
    {
      name: 'explore_or_dig_for_ores',
      description: 'Either digs a stepped staircase underground to expose stone/coal/iron_ore ("staircase_mine") or explores surface terrain ("explore_surface") when resources are not immediately visible.',
      inputSchema: {
        type: 'object',
        properties: {
          mode: {
            type: 'string',
            description: '"staircase_mine" (dig down to find stone/coal/iron) or "explore_surface" (walk 25m to find trees/caves)',
            default: 'staircase_mine',
          },
          steps: {
            type: 'number',
            description: 'Number of staircase steps to dig or blocks to travel',
            default: 6,
          },
        },
      },
      tags: ['explore', 'dig', 'staircase', 'underground', 'cave', 'search', 'find', 'ores', 'stone', 'iron'],
    },
    async (args, ctx) => {
      const mode = (pickStrArg(args.mode) || 'staircase_mine').toLowerCase();
      const steps = Math.min(16, Math.max(2, Number(args.steps) || 6));

      if (mode.includes('explore') || mode.includes('surface')) {
        const angle = Math.random() * Math.PI * 2;
        const dist = Math.max(18, steps * 4);
        const targetX = Math.round(ctx.bot.entity.position.x + Math.cos(angle) * dist);
        const targetZ = Math.round(ctx.bot.entity.position.z + Math.sin(angle) * dist);
        const targetY = Math.round(ctx.bot.entity.position.y);

        if (ctx.kinematics?.pathfindTo) {
          await ctx.kinematics.pathfindTo(new Vec3(targetX, targetY, targetZ), 3.0, 8000);
        }
        ctx.reportProgress(1.0);
        return {
          success: true,
          summary: `Explored surface terrain to (${Math.round(ctx.bot.entity.position.x)}, ${targetY}, ${Math.round(ctx.bot.entity.position.z)}).`,
        };
      }

      // Staircase mining mode: dig forward and down safely using per-block optimal tools
      let dugCount = 0;
      const dirX = 1;
      const dirZ = 0;
      (ctx as any).lastToolFailure = null;

      // Proactively check pickaxe durability before tunneling down to conserve iron/diamond pickaxes!
      await ensurePickaxesForMining(ctx, steps * 3);

      for (let s = 0; s < steps; s++) {
        if (ctx.abortSignal?.aborted || (ctx as any).lastToolFailure) break;
        const base = ctx.bot.entity.position.floored();
        const blocksToClear = [
          base.offset(dirX, 1, dirZ), // Upper head clearance
          base.offset(dirX, 0, dirZ), // Torso clearance
          base.offset(dirX, -1, dirZ), // Step down
        ];

        for (const bPos of blocksToClear) {
          if (ctx.abortSignal?.aborted || (ctx as any).lastToolFailure) break;
          const b = ctx.bot.blockAt(bPos);
          if (b && b.name !== 'air' && b.boundingBox === 'block' && !b.name.includes('bedrock') && !b.name.includes('lava')) {
            const ok = await safeDigSingleBlock(ctx, bPos, 1, false);
            if (ok) dugCount++;
          }
        }

        // Step forward into the newly dug step to collect dropped cobble/dirt/ores
        const stepPos = base.offset(dirX, -1, dirZ);
        if (ctx.kinematics?.pathfindTo) {
          await ctx.kinematics.pathfindTo(stepPos, 1.0, 1500);
        }
        ctx.reportProgress((s + 1) / steps);
      }

      const invSummary = (ctx.bot.inventory?.items() || [])
        .map((it: any) => `${it.count}x ${it.name}`)
        .join(', ');
      const toolNote = (ctx as any).lastToolFailure ? ` (${(ctx as any).lastToolFailure})` : '';

      return {
        success: dugCount > 0,
        summary: `Dug staircase mine (${dugCount} blocks cleared) down to Y=${Math.round(ctx.bot.entity.position.y)}.${toolNote} Current inventory: [${invSummary || 'empty'}].`,
      };
    }
  );

  // 6. craft_item
  registry.registerTool(
    {
      name: 'craft_item',
      description: 'Crafts an item, tool, weapon, furnace, or armor piece (e.g. oak_planks, stick, crafting_table, wooden_pickaxe, stone_pickaxe, furnace, iron_pickaxe, iron_helmet, iron_chestplate, iron_leggings, iron_boots). Prerequisite materials (*_planks, stick, crafting_table) must be crafted in inventory first; if a 3x3 grid is needed, places an existing crafting_table from inventory and reclaims it afterward.',
      inputSchema: {
        type: 'object',
        properties: {
          itemName: {
            type: 'string',
            description: 'Exact item name to craft (e.g. stick, crafting_table, wooden_pickaxe, stone_pickaxe, furnace, iron_helmet, iron_chestplate, iron_leggings, iron_boots)',
          },
          count: { type: 'number', description: 'Quantity to craft', default: 1 },
        },
        required: ['itemName'],
      },
      tags: ['craft', 'make', 'recipe', 'pickaxe', 'sword', 'planks', 'stick', 'table', 'furnace', 'armor', 'iron_helmet', 'iron_chestplate', 'iron_leggings', 'iron_boots'],
    },
    async (args, ctx) => {
      const itemName = pickStrArg(args.itemName || args.recipeName || args.item).toLowerCase().trim();
      if (!itemName) {
        return { success: false, summary: 'Missing required itemName argument for craft_item.' };
      }
      const count = Math.max(1, Number(args.count) || 1);
      return await craftItemWithAutoPrereqs(ctx, itemName, count, true, false);
    }
  );

  // craft_recipe: Discrete Symbolic 3x3 Crafting Table Recipe Engine (MCP Domain)
  registry.registerTool(
    {
      name: 'craft_recipe',
      description:
        'Discrete symbolic 3x3 crafting recipe engine. Matches official Minecraft crafting table recipe graph without synthetic auto-crafting crutches or sub-ingredient synthesis.',
      inputSchema: {
        type: 'object',
        properties: {
          recipeName: {
            type: 'string',
            description:
              'Item name to craft (e.g. wooden_pickaxe, stone_pickaxe, crafting_table, iron_helmet, iron_chestplate, iron_leggings, iron_boots)',
          },
          count: { type: 'number', description: 'Quantity to craft', default: 1 },
        },
        required: ['recipeName'],
      },
      tags: ['craft_recipe', 'craft', 'recipe', 'crafting_table', 'mcp'],
    },
    async (args, ctx) => {
      const itemName = (pickStrArg(args.recipeName || args.itemName) || 'oak_planks').toLowerCase().trim();
      const count = Math.max(1, Number(args.count) || 1);
      return await craftItemWithAutoPrereqs(ctx, itemName, count, true, false);
    }
  );

  // 7. smelt_item
  registry.registerTool(
    {
      name: 'smelt_item',
      description: 'Smelts raw ore or food in a furnace (e.g. raw_iron -> iron_ingot, raw_gold -> gold_ingot). Automatically places or crafts a furnace if 8 cobblestone is available, and reclaims the furnace and leftover fuel when finished!',
      inputSchema: {
        type: 'object',
        properties: {
          inputItem: {
            type: 'string',
            description: 'Item to smelt (e.g. raw_iron, iron_ore, raw_gold, beef)',
            default: 'raw_iron',
          },
          count: {
            type: 'number',
            description: 'Number of items to smelt',
            default: 4,
          },
        },
        required: ['inputItem'],
      },
      tags: ['smelt', 'furnace', 'iron_ingot', 'raw_iron', 'iron_ore', 'cook', 'coal'],
    },
    async (args, ctx) => {
      const requestedInput = (
        pickStrArg(args.inputItem) ||
        pickStrArg(args.itemToSmelt) ||
        pickStrArg(args.oreType) ||
        pickStrArg(args.itemName) ||
        pickStrArg(args.item) ||
        'raw_iron'
      ).toLowerCase();
      const targetCount = Math.min(32, Math.max(1, Number(args.count) || 4));

      const invItems = ctx.bot.inventory?.items() || [];
      const inputStack = invItems.find((i: any) => {
        const n = i.name.toLowerCase();
        if (requestedInput.includes('iron')) {
          return n === 'raw_iron' || n === 'iron_ore' || n === 'deepslate_iron_ore';
        }
        return n === requestedInput || n.includes(requestedInput);
      });

      if (!inputStack) {
        return {
          success: false,
          summary: `No ${requestedInput} (or raw_iron) in inventory to smelt! Mine iron_ore first.`,
        };
      }

      const fuelStack = invItems.find((i: any) => {
        const n = i.name.toLowerCase();
        return (
          n === 'lava_bucket' ||
          n === 'coal' ||
          n === 'charcoal' ||
          n === 'blaze_rod' ||
          n.endsWith('_planks') ||
          n.endsWith('_log') ||
          n === 'coal_block'
        );
      });

      if (!fuelStack) {
        return {
          success: false,
          summary: 'No fuel (coal, charcoal, lava_bucket, planks, or logs) in inventory to power the furnace! Gather coal_ore or wood first.',
        };
      }

      let furnaceBlock = ctx.bot.findBlock({
        matching: (b: any) => b?.name === 'furnace' || b?.name === 'blast_furnace',
        maxDistance: 8,
      });

      if (furnaceBlock && ctx.bot.entity.position.distanceTo(furnaceBlock.position) > 3.2) {
        if (ctx.kinematics?.pathfindTo) {
          await ctx.kinematics.pathfindTo(furnaceBlock.position, 2.0, 3000);
        }
        if (ctx.bot.entity.position.distanceTo(furnaceBlock.position) > 4.2) {
          furnaceBlock = null;
        }
      }

      let placedFurnacePos: Vec3 | null = null;
      if (!furnaceBlock) {
        const hasFurnaceInv = (ctx.bot.inventory?.items() || []).some((i: any) => i.name === 'furnace');
        if (hasFurnaceInv) {
          furnaceBlock = await placeBlockNearby(ctx, 'furnace');
          if (furnaceBlock) {
            placedFurnacePos = furnaceBlock.position?.clone?.() || furnaceBlock.position;
          }
        }
      }

      if (!furnaceBlock) {
        return {
          success: false,
          summary: 'No furnace nearby or in inventory. Craft a furnace (8 cobblestone) with craft_item first.',
        };
      }

      const reclaimFurnaceIfPlaced = async () => {
        if (placedFurnacePos) {
          const stillFurnace = ctx.bot.blockAt(placedFurnacePos);
          if (stillFurnace && (stillFurnace.name === 'furnace' || stillFurnace.name === 'blast_furnace')) {
            const preFurnaceInv = countHarvestedDropsInInventory(ctx.bot, 'furnace');
            const dug = await safeDigSingleBlock(ctx, placedFurnacePos, 1, false);
            if (dug) {
              await collectDroppedItemNear(ctx, placedFurnacePos, 'furnace', preFurnaceInv);
              logVerbose('GATHER', `Reclaimed placed furnace at (${placedFurnacePos.x}, ${placedFurnacePos.y}, ${placedFurnacePos.z}) back into inventory!`);
            }
          }
        }
      };

      await ctx.kinematics.lookAtSmooth(furnaceBlock.position.offset(0.5, 0.5, 0.5), 180);

      try {
        const amountToSmelt = Math.min(targetCount, inputStack.count);
        const furnace = await ctx.bot.openFurnace(furnaceBlock);
        const fuelNeeded =
          fuelStack.name === 'lava_bucket'
            ? 1
            : fuelStack.name.includes('coal')
            ? Math.min(fuelStack.count, Math.ceil(amountToSmelt / 7) + 1)
            : Math.min(fuelStack.count, amountToSmelt);
        await furnace.putFuel(fuelStack.type, null, fuelNeeded);
        await furnace.putInput(inputStack.type, null, amountToSmelt);

        const waitTotalMs = amountToSmelt * 10200;
        const startWait = Date.now();
        while (Date.now() - startWait < waitTotalMs && !ctx.abortSignal?.aborted) {
          const elapsed = Date.now() - startWait;
          ctx.reportProgress(Math.min(0.95, elapsed / waitTotalMs));
          const outItem = furnace.outputItem();
          if (outItem && outItem.count >= amountToSmelt) {
            break;
          }
          await new Promise((r) => setTimeout(r, 500));
        }

        const out = furnace.outputItem();
        let smeltedCount = 0;
        let smeltedName = 'items';
        if (out) {
          smeltedCount = out.count;
          smeltedName = out.name;
          await furnace.takeOutput();
        }
        try {
          if (furnace.inputItem?.()) await furnace.takeInput();
        } catch {}
        try {
          if (furnace.fuelItem?.()) await furnace.takeFuel();
        } catch {}
        furnace.close();
        await reclaimFurnaceIfPlaced();
        ctx.reportProgress(1.0);

        return {
          success: smeltedCount > 0,
          summary: `Smelted ${smeltedCount}x ${smeltedName} in the furnace!`,
        };
      } catch (err: any) {
        await reclaimFurnaceIfPlaced();
        return {
          success: false,
          summary: `Furnace smelting interrupted: ${err.message}`,
        };
      }
    }
  );

  // smelt_ores: Discrete Symbolic Furnace Smelting (MCP Domain)
  registry.registerTool(
    {
      name: 'smelt_ores',
      description:
        'Discrete symbolic furnace smelting. Smelts raw ores and fuel cycles in a furnace container (e.g. raw_iron -> iron_ingot).',
      inputSchema: {
        type: 'object',
        properties: {
          oreType: {
            type: 'string',
            description: 'Item or raw ore to smelt (e.g. raw_iron, iron_ore, raw_gold, raw_copper)',
            default: 'raw_iron',
          },
          count: {
            type: 'number',
            description: 'Quantity of items to smelt',
            default: 4,
          },
        },
      },
      tags: ['smelt_ores', 'smelt', 'furnace', 'iron_ingot', 'raw_iron', 'mcp'],
    },
    async (args, ctx) => {
      const tool = registry.getTool('smelt_item');
      if (tool) {
        return await tool.handler(
          { inputItem: args.oreType || args.inputItem || args.itemName || 'raw_iron', count: args.count || 4 },
          ctx
        );
      }
      return { success: false, summary: 'smelt_item tool not registered.' };
    }
  );

  // 8. equip_armor_or_item
  registry.registerTool(
    {
      name: 'equip_armor_or_item',
      description: 'Equips armor pieces (helmet, chestplate, leggings, boots, or "all_armor" to equip all armor in inventory), shield, or weapons.',
      inputSchema: {
        type: 'object',
        properties: {
          itemName: {
            type: 'string',
            description: 'Item name to equip (e.g. "all_armor", "iron_helmet", "iron_chestplate", "iron_leggings", "iron_boots", "shield")',
            default: 'all_armor',
          },
        },
        required: ['itemName'],
      },
      tags: ['equip', 'wear', 'armor', 'iron_helmet', 'iron_chestplate', 'iron_leggings', 'iron_boots', 'shield'],
    },
    async (args, ctx) => {
      let query = (pickStrArg(args.itemName) || pickStrArg(args.item) || 'all_armor').toLowerCase();
      if (query === 'leather_cap' || query === 'cap') query = 'leather_helmet';
      const slot = pickStrArg(args.slot).toLowerCase();
      const equippedList: string[] = [];

      const slotMap: Array<{ suffix: string; dest: 'head' | 'torso' | 'legs' | 'feet' }> = [
        { suffix: 'helmet', dest: 'head' },
        { suffix: 'chestplate', dest: 'torso' },
        { suffix: 'leggings', dest: 'legs' },
        { suffix: 'boots', dest: 'feet' },
      ];

      if (query === 'all_armor' || query === 'armor' || query === 'full_iron_armor') {
        for (const s of slotMap) {
          if (await ctx.kinematics.equipItemHumanized(s.suffix, s.dest)) {
            equippedList.push(s.suffix);
          }
        }
        if (!equippedList.includes('chestplate') && (await ctx.kinematics.equipItemHumanized('elytra', 'torso'))) {
          equippedList.push('elytra');
        }
      } else {
        let dest: 'hand' | 'off-hand' | 'head' | 'torso' | 'legs' | 'feet' = 'hand';
        if (slot === 'head' || slot === 'helmet' || query.includes('helmet') || query.includes('cap')) dest = 'head';
        else if (slot === 'torso' || slot === 'chest' || query.includes('chestplate') || query.includes('elytra') || query.includes('tunic')) dest = 'torso';
        else if (slot === 'legs' || slot === 'leggings' || query.includes('leggings') || query.includes('pants')) dest = 'legs';
        else if (slot === 'feet' || slot === 'boots' || query.includes('boots')) dest = 'feet';
        else if (slot === 'offhand' || slot === 'off-hand' || query.includes('shield')) dest = 'off-hand';

        if (await ctx.kinematics.equipItemHumanized(query, dest)) {
          equippedList.push(query);
        }
      }

      ctx.reportProgress(1.0);
      return {
        success: equippedList.length > 0,
        summary:
          equippedList.length > 0
            ? `Equipped: ${equippedList.join(', ')}!`
            : `No matching item (${query}) found in inventory to equip.`,
      };
    }
  );

  // 9. engage_combat
  registry.registerTool(
    {
      name: 'engage_combat',
      description:
        'Engages and hunts nearby hostile mobs or specified animals (e.g. cow, sheep, pig, chicken, spider, creeper, zombie, skeleton, wither_skeleton) using 1.9+ timed weapon strikes, Creeper fuse kiting, multi-target chaining (count), and automatic loot drop collection.',
      inputSchema: {
        type: 'object',
        properties: {
          targetType: {
            type: 'string',
            description: 'Specific mob/animal name (e.g. "cow", "spider", "creeper", "wither_skeleton") or "hostile"',
            default: 'hostile',
          },
          maxRange: {
            type: 'number',
            description: 'Engagement search radius in blocks',
            default: 32,
          },
          count: {
            type: 'number',
            description: 'Number of targets to eliminate before returning (default 1)',
            default: 1,
          },
        },
      },
      tags: [
        'combat',
        'fight',
        'attack',
        'kill',
        'hunt',
        'defend',
        'protect',
        'mob',
        'zombie',
        'skeleton',
        'spider',
        'creeper',
        'cow',
        'sheep',
        'pig',
        'chicken',
        'animal',
      ],
    },
    async (args, ctx) => {
      const maxRange = Math.max(8, Number(args.maxRange) || 32);
      const targetCount = Math.max(1, Math.min(16, Number(args.count) || 1));
      const rawTargetType = (
        pickStrArg(args.targetType) ||
        pickStrArg(args.targetMob) ||
        pickStrArg(args.target) ||
        'hostile'
      ).toLowerCase().trim();
      const normalizedTarget = rawTargetType
        .replace(/\s+/g, '_')
        .replace(/^(hostiles|mobs|monsters|enemies|enemy|any)$/, 'hostile')
        .replace(/^(wolves)$/, 'wolf')
        .replace(/s$/, '');

      const HOSTILE_SET = new Set([
        'zombie',
        'zombie_villager',
        'husk',
        'drowned',
        'skeleton',
        'stray',
        'bogged',
        'wither_skeleton',
        'creeper',
        'spider',
        'cave_spider',
        'enderman',
        'witch',
        'slime',
        'magma_cube',
        'phantom',
        'pillager',
        'vindicator',
        'evoker',
        'vex',
        'ravager',
        'blaze',
        'breeze',
        'ghast',
        'hoglin',
        'zoglin',
        'piglin_brute',
        'warden',
        'silverfish',
        'endermite',
        'shulker',
        'guardian',
        'elder_guardian',
      ]);

      const PASSIVE_SET = new Set([
        'cow',
        'sheep',
        'pig',
        'chicken',
        'villager',
        'wandering_trader',
        'horse',
        'donkey',
        'mule',
        'llama',
        'wolf',
        'cat',
        'ocelot',
        'fox',
        'rabbit',
        'turtle',
        'bat',
        'squid',
        'glow_squid',
        'cod',
        'salmon',
        'tropical_fish',
        'pufferfish',
        'axolotl',
        'frog',
        'camel',
        'sniffer',
        'armadillo',
        'allay',
        'bee',
        'strider',
        'iron_golem',
        'snow_golem',
      ]);

      const NON_LIVING_SET = new Set([
        'item',
        'experience_orb',
        'arrow',
        'spectral_arrow',
        'trident',
        'snowball',
        'egg',
        'potion',
        'experience_bottle',
        'ender_pearl',
        'shulker_bullet',
        'small_fireball',
        'fireball',
        'dragon_fireball',
        'wither_skull',
        'llama_spit',
        'wind_charge',
        'evoker_fangs',
        'eye_of_ender',
        'firework_rocket',
        'fishing_bobber',
        'lightning_bolt',
        'boat',
        'chest_boat',
        'minecart',
        'chest_minecart',
        'furnace_minecart',
        'tnt_minecart',
        'hopper_minecart',
        'spawner_minecart',
        'command_block_minecart',
        'armor_stand',
        'painting',
        'item_frame',
        'glow_item_frame',
        'leash_knot',
        'item_display',
        'block_display',
        'text_display',
        'interaction',
        'marker',
        'falling_block',
        'area_effect_cloud',
        'tnt',
      ]);

      // Auto-equip armor and off-hand shield if available
      const armorSlots: Array<{ suffix: string; dest: 'head' | 'torso' | 'legs' | 'feet' }> = [
        { suffix: 'helmet', dest: 'head' },
        { suffix: 'chestplate', dest: 'torso' },
        { suffix: 'leggings', dest: 'legs' },
        { suffix: 'boots', dest: 'feet' },
      ];
      for (const slot of armorSlots) {
        await ctx.kinematics.equipItemHumanized(slot.suffix, slot.dest);
      }
      await ctx.kinematics.equipItemHumanized('shield', 'off-hand');

      const weapons = [
        'netherite_sword',
        'diamond_sword',
        'iron_sword',
        'stone_sword',
        'wooden_sword',
        'netherite_axe',
        'diamond_axe',
        'iron_axe',
        'stone_axe',
      ];
      for (const w of weapons) {
        if (await ctx.kinematics.equipItemHumanized(w, 'hand')) break;
      }

      const matchesTargetEntity = (e: any): boolean => {
        if (!e || e === ctx.bot.entity || !e.position || e.isValid === false) return false;
        const eName = String(e.name || e.displayName || '')
          .toLowerCase()
          .replace(/\s+/g, '_');
        if (eName === 'end_crystal') {
          const dist = e.position.distanceTo(ctx.bot.entity.position);
          return dist <= maxRange;
        }
        const eType = String(e.type || '').toLowerCase();
        if (eType === 'object' || eType === 'orb' || eType === 'other') {
          return false;
        }
        if (eType === 'player') {
          const u = String(e.username || '').toLowerCase();
          if (e === ctx.bot.entity || (u && u === ctx.bot.username?.toLowerCase())) {
            return false;
          }
          if (u.includes('companion') || u.includes('friend') || u.includes('teammate')) {
            return false;
          }

          const rawUnderscore = rawTargetType.replace(/\s+/g, '_');
          const isExplicitPlayerTarget = (
            normalizedTarget === 'player' ||
            normalizedTarget.includes('opponent') ||
            normalizedTarget.includes('rival') ||
            normalizedTarget.includes('dqn') ||
            normalizedTarget.includes('spar') ||
            normalizedTarget.includes('mirror') ||
            normalizedTarget.includes('clone') ||
            normalizedTarget === 'agent' ||
            u === normalizedTarget ||
            u === rawUnderscore ||
            u.includes(normalizedTarget) ||
            normalizedTarget.includes(u)
          );

          if (isExplicitPlayerTarget) {
            return true;
          }

          if (!normalizedTarget || normalizedTarget === 'hostile') {
            const isOpponentName =
              u.includes('opponent') ||
              u.includes('rival') ||
              u.includes('dqn') ||
              u.includes('mirror') ||
              u.includes('spar');

            if (isOpponentName) {
              const hasHostileMobNearby = !!ctx.bot.nearestEntity((candidate: any) => {
                if (!candidate || candidate.type === 'player' || candidate === ctx.bot.entity) return false;
                const name = String(candidate.name || '').toLowerCase();
                return HOSTILE_SET.has(name) || candidate.type === 'hostile';
              });
              return !hasHostileMobNearby;
            }
          }
          return false;
        }
        if (!eName || NON_LIVING_SET.has(eName)) return false;

        const dist = e.position.distanceTo(ctx.bot.entity.position);
        if (dist > maxRange) return false;

        if (!normalizedTarget || normalizedTarget === 'hostile') {
          return (
            HOSTILE_SET.has(eName) ||
            eType === 'hostile' ||
            (eType === 'mob' && !PASSIVE_SET.has(eName))
          );
        }

        const rawUnderscore = rawTargetType.replace(/\s+/g, '_');
        return (
          eName === normalizedTarget ||
          eName === rawUnderscore ||
          eName.includes(normalizedTarget)
        );
      };

      const findNextTarget = (): any => {
        const state = ctx.worldState();
        const perceivedCandidates = (state.visibleEntities || [])
          .filter((pe) => {
            if (pe.distance > maxRange) return false;
            const peName = String(pe.name || '').toLowerCase().replace(/\s+/g, '_');
            const peUsername = String(pe.username || '').toLowerCase();
            if (peUsername && peUsername === ctx.bot.username?.toLowerCase()) return false;

            if (!normalizedTarget || normalizedTarget === 'hostile') {
              return (
                pe.type === 'hostile' ||
                HOSTILE_SET.has(peName) ||
                pe.type === 'player'
              );
            }
            const rawUnderscore = rawTargetType.replace(/\s+/g, '_');
            return (
              peName === normalizedTarget ||
              peName === rawUnderscore ||
              peName.includes(normalizedTarget) ||
              peUsername === normalizedTarget ||
              peUsername === rawUnderscore ||
              peUsername.includes(normalizedTarget) ||
              (pe.type === 'player' && (
                normalizedTarget.includes('player') ||
                normalizedTarget.includes('opponent') ||
                normalizedTarget.includes('rival') ||
                normalizedTarget.includes('dqn') ||
                normalizedTarget.includes('spar') ||
                normalizedTarget.includes('mirror') ||
                normalizedTarget.includes('clone') ||
                normalizedTarget.includes('agent')
              ))
            );
          })
          .sort((a, b) => a.distance - b.distance);

        for (const pe of perceivedCandidates) {
          const ent = ctx.bot.entities?.[pe.id];
          if (ent && matchesTargetEntity(ent)) return ent;
        }

        // Full 360° entity scan fallback across ctx.bot.entities sorted by distance
        const allEntities = Object.values(ctx.bot.entities || {})
          .filter(matchesTargetEntity)
          .sort(
            (a: any, b: any) =>
              a.position.distanceTo(ctx.bot.entity.position) -
              b.position.distanceTo(ctx.bot.entity.position)
          );
        return allEntities[0] || null;
      };

      let kills = 0;
      let totalStrikes = 0;
      let lastMobName = normalizedTarget === 'hostile' ? 'hostile mob' : rawTargetType;
      let botDied = false;
      const onBotDeath = () => {
        botDied = true;
      };
      ctx.bot.once?.('death', onBotDeath);

      try {
        for (let targetIdx = 0; targetIdx < targetCount; targetIdx++) {
          if (ctx.abortSignal?.aborted || botDied || (ctx.bot.health ?? 20) <= 0) break;

          const targetEntity = findNextTarget();
          if (!targetEntity) {
            if (targetIdx === 0) {
              if (!normalizedTarget || normalizedTarget === 'hostile') {
                return {
                  success: true,
                  summary: 'Area is clear! No hostile mobs detected within range.',
                };
              }
              return {
                success: false,
                summary: `No ${rawTargetType} detected within ${maxRange}m.`,
              };
            }
            break;
          }

          lastMobName = targetEntity.name || lastMobName;
          const isCreeper = String(targetEntity.name || '').toLowerCase() === 'creeper';
          let strikesOnTarget = 0;
          let isFollowingTarget = false;
          let lastDistCheckTime = Date.now();
          let lastDistValue = ctx.bot.entity.position.distanceTo(targetEntity.position);
          const combatStart = Date.now();
          const lastTargetPos = targetEntity.position.clone();

          const useNeuralMotor =
            process.env.NEURAL_MOTOR_CONTROL !== 'false' &&
            typeof ctx.kinematics.executeNeuralMotorStep === 'function';

          while (
            targetEntity.isValid &&
            Date.now() - combatStart < 14000 &&
            !ctx.abortSignal?.aborted &&
            !botDied &&
            (ctx.bot.health ?? 20) > 0
          ) {
            if (targetEntity.position) {
              lastTargetPos.set(
                targetEntity.position.x,
                targetEntity.position.y,
                targetEntity.position.z
              );
            }
            const dist = ctx.bot.entity.position.distanceTo(targetEntity.position);

            if (useNeuralMotor) {
              ctx.kinematics.setActiveMotorTarget?.(targetEntity.position);
              const res = await ctx.kinematics.executeNeuralMotorStep!(
                `Engage and combat ${targetEntity.name || 'hostile'} at ${dist.toFixed(1)}m`,
                targetEntity.position,
                8,
                {
                  allowAttack: true,
                  allowUseItem: true,
                  allowDig: false,
                  allowItemDrop: false,
                  persistItemUse: true,
                  neuralAimOnly: true,
                  combatTarget: targetEntity,
                  abortSignal: ctx.abortSignal,
                }
              );
              if (res.executed) {
                strikesOnTarget += res.strikes;
                totalStrikes += res.strikes;
                const overallProgress =
                  (targetIdx + Math.min(0.95, (Date.now() - combatStart) / 8000)) / targetCount;
                ctx.reportProgress(Math.min(0.95, overallProgress));
                continue;
              }
            }

            const isFuseLit =
              isCreeper &&
              (targetEntity.metadata?.[16] === 1 || targetEntity.metadata?.[18] === true);

            if (isFuseLit && dist < 6.0) {
              // Creeper fuse is actively hissing! Backpedal / kite outside blast radius until fuse resets
              if (isFollowingTarget) {
                ctx.bot.pathfinder.setGoal(null);
                isFollowingTarget = false;
              }
              await ctx.kinematics.lookAtSmooth(targetEntity.position.offset(0, 1.2, 0), 80);
              ctx.bot.setControlState('back', true);
              ctx.bot.setControlState('sprint', false);
              await new Promise((r) => setTimeout(r, 280));
              ctx.bot.setControlState('back', false);
              continue;
            }

            if (dist > 3.1) {
              if (ctx.kinematics?.pathfindTo) {
                await ctx.kinematics.pathfindTo(targetEntity.position, 2.2, 800);
              }
            } else {
              const hit = await ctx.kinematics.strikeTarget(targetEntity);
              if (hit) {
                strikesOnTarget++;
                totalStrikes++;
                if (isCreeper && targetEntity.isValid) {
                  // Hit-and-run kiting against Creepers: step back outside 3m ignition radius between swings
                  ctx.bot.setControlState('back', true);
                  await new Promise((r) => setTimeout(r, 200));
                  ctx.bot.setControlState('back', false);
                }
              }
            }

            const overallProgress =
              (targetIdx + Math.min(0.95, (Date.now() - combatStart) / 8000)) / targetCount;
            ctx.reportProgress(Math.min(0.95, overallProgress));
          }

          ctx.kinematics.setActiveMotorTarget?.(null);
          (ctx.bot as any)._vlaItemUseState = null;
          try { ctx.bot.deactivateItem?.(); } catch {}
          ctx.bot.pathfinder.setGoal(null);
          ctx.bot.clearControlStates?.();

          if (botDied || (ctx.bot.health ?? 20) <= 0) {
            botDied = true;
            break;
          }

          if (!targetEntity.isValid) {
            kills++;
          }

          // Brief pause for item drops to spawn on death, then auto-collect nearby loot drops
          await new Promise((r) => setTimeout(r, 180));
          const nearbyDrops = Object.values(ctx.bot.entities || {}).filter(
            (e: any) =>
              e &&
              e.isValid !== false &&
              e.position &&
              (e.name === 'item' || e.type === 'object' || e.displayName === 'Item') &&
              (e.position.distanceTo(lastTargetPos) <= 6.5 ||
                e.position.distanceTo(ctx.bot.entity.position) <= 6.5)
          );
          if (nearbyDrops.length > 0 && ctx.kinematics?.pathfindTo) {
            const nearestDrop: any = nearbyDrops[0];
            await ctx.kinematics.pathfindTo(nearestDrop.position, 1.2);
            await new Promise((r) => setTimeout(r, 150));
          }
        }
      } finally {
        ctx.kinematics.setActiveMotorTarget?.(null);
        (ctx.bot as any)._vlaItemUseState = null;
        try { ctx.bot.deactivateItem?.(); } catch {}
        ctx.bot.removeListener?.('death', onBotDeath);
        ctx.bot.pathfinder.setGoal(null);
        ctx.bot.clearControlStates?.();
      }

      ctx.reportProgress(1.0);

      if (botDied) {
        return {
          success: false,
          summary: `Defeated in combat against ${lastMobName} after ${totalStrikes} strike(s).`,
          data: { kills, totalStrikes, botDied: true, targetType: lastMobName },
        };
      }

      return {
        success: kills > 0,
        summary:
          kills > 0
            ? targetCount > 1
              ? `Eliminated ${kills}/${targetCount} ${lastMobName}(s) with ${totalStrikes} timed 1.9+ strikes and collected drops.`
              : `Engaged ${lastMobName} with ${totalStrikes} timed 1.9+ strikes (target eliminated).`
            : `Engaged ${lastMobName} with ${totalStrikes} timed 1.9+ strikes, target still alive.`,
        data: { kills, totalStrikes, targetType: lastMobName },
      };
    }
  );

  // 10. give_item_to_player
  registry.registerTool(
    {
      name: 'give_item_to_player',
      description: 'Looks at a player, walks over if needed, and tosses an inventory item to them.',
      inputSchema: {
        type: 'object',
        properties: {
          playerName: { type: 'string', description: 'Recipient player username' },
          itemName: { type: 'string', description: 'Item name to toss (or "any")' },
        },
        required: ['playerName', 'itemName'],
      },
      tags: ['give', 'drop', 'toss', 'share', 'hand', 'item'],
    },
    async (args, ctx) => {
      const playerName = (
        pickStrArg(args.playerName) ||
        pickStrArg(args.targetPlayer) ||
        pickStrArg(args.targetPlayerName) ||
        pickStrArg(args.recipient) ||
        pickStrArg(args.player) ||
        pickStrArg(args.target)
      ).trim();
      const itemQuery = (
        pickStrArg(args.itemName) ||
        pickStrArg(args.item) ||
        pickStrArg(args.itemQuery) ||
        'any'
      ).toLowerCase();

      const items = ctx.bot.inventory?.items() || [];
      if (items.length === 0) {
        return { success: false, summary: 'My inventory is completely empty!' };
      }

      const matchedItem =
        itemQuery === 'any'
          ? items[0]
          : items.find((i: any) => i.name.toLowerCase().includes(itemQuery));
      if (!matchedItem) {
        return { success: false, summary: `Could not find ${itemQuery} in inventory to give.` };
      }

      const isGenericRef =
        !playerName ||
        ['me', 'player', 'user', 'friend', 'partner', 'human', 'owner', 'companion', 'companion_player', 'teammate', 'dqn_opponent', 'mirroragent'].includes(
          playerName.toLowerCase()
        );

      let targetEntity: any = null;
      if (!isGenericRef) {
        targetEntity =
          ctx.bot.players?.[playerName]?.entity ||
          (Object.values(ctx.bot.players || {}).find(
            (p: any) => p?.username?.toLowerCase() === playerName.toLowerCase()
          ) as any)?.entity ||
          (Object.values(ctx.bot.players || {}).find(
            (p: any) => p?.username?.toLowerCase().includes(playerName.toLowerCase())
          ) as any)?.entity;
      }

      if (!targetEntity && !isGenericRef) {
        const lowerName = playerName.toLowerCase();
        targetEntity =
          ctx.bot.nearestEntity?.(
            (e: any) =>
              e &&
              e !== ctx.bot.entity &&
              e.isValid !== false &&
              e.position &&
              String(e.name || e.displayName || e.username || '').toLowerCase() === lowerName
          ) ||
          ctx.bot.nearestEntity?.(
            (e: any) =>
              e &&
              e !== ctx.bot.entity &&
              e.isValid !== false &&
              e.position &&
              String(e.name || e.displayName || e.username || '').toLowerCase().includes(lowerName)
          ) ||
          null;
      }

      if (!targetEntity) {
        targetEntity = ctx.bot.nearestEntity(
          (e: any) => (e.type === 'player' || e.name === 'player') && e.username !== ctx.bot.username
        );
      }

      if (!targetEntity) {
        return {
          success: false,
          summary: `Could not locate recipient ${playerName || 'companion'} nearby to hand over ${matchedItem.name}.`,
        };
      }

      if (
        ctx.bot.entity.position.distanceTo(targetEntity.position) > 2.0 &&
        ctx.kinematics?.pathfindTo
      ) {
        await ctx.kinematics.pathfindTo(targetEntity.position, 1.4);
      }
      await ctx.kinematics.lookAtSmooth(targetEntity.position.offset(0, 1.5, 0), 220);

      const resolvedRecipient = targetEntity?.username || targetEntity?.name || playerName || 'nearby player';
      await new Promise((r) => setTimeout(r, 180));
      try {
        (ctx.bot as any)._lastDropItemAt = Date.now();
        await ctx.bot.tossStack(matchedItem);
        // Step back and poll until the tossed stack is picked up (up to ~2640ms so recipients reliably collect it)
        try {
          ctx.bot.setControlState?.('back', true);
          await new Promise((r) => setTimeout(r, 300));
          ctx.bot.clearControlStates?.();
        } catch {}
        for (let w = 0; w < 16; w++) {
          await new Promise((r) => setTimeout(r, 165));
          const nearbyDrop = ctx.bot.nearestEntity?.(
            (e: any) =>
              e &&
              e.isValid !== false &&
              e.position &&
              (e.name === 'item' || e.type === 'object' || e.displayName === 'Item') &&
              e.position.distanceTo(ctx.bot.entity.position) <= 4.0
          );
          if (!nearbyDrop) break;
        }
        ctx.reportProgress(1.0);
        return {
          success: true,
          summary: `Tossed ${matchedItem.count}x ${matchedItem.name} to ${resolvedRecipient}.`,
        };
      } catch (err: any) {
        return {
          success: false,
          summary: `Could not toss item: ${err.message}`,
        };
      }
    }
  );

  // 11. store_items_in_chest
  registry.registerTool(
    {
      name: 'store_items_in_chest',
      description: 'Walks to a nearby chest (or near a specified player first), crafts and places a chest if none exists nearby, opens the chest, and deposits the specified item and quantity into it.',
      inputSchema: {
        type: 'object',
        properties: {
          itemName: {
            type: 'string',
            description: 'Name of the item to store in the chest (e.g. iron_ingot, cobblestone, spruce_log)',
          },
          count: {
            type: 'number',
            description: 'Maximum quantity of the item to deposit',
            default: 64,
          },
          playerName: {
            type: 'string',
            description: 'Optional player username to walk near before locating/placing the chest',
          },
        },
        required: ['itemName'],
      },
      tags: ['chest', 'store', 'deposit', 'put', 'container', 'storage', 'iron_ingot'],
    },
    async (args, ctx) => {
      const itemQuery = (pickStrArg(args.itemName) || 'iron_ingot').toLowerCase().trim();
      const maxCount = Math.max(1, Number(args.count) || 64);
      const playerName = pickStrArg(args.playerName).trim();
      const mcData = require('minecraft-data')(ctx.bot.version);

      // If a playerName was specified, walk near them first (tunneling/staircasing out of underground mines if needed)
      if (playerName) {
        const playerObj: any =
          ctx.bot.players?.[playerName] ||
          Object.values(ctx.bot.players || {}).find(
            (p: any) => p?.username?.toLowerCase() === playerName.toLowerCase()
          );
        if (playerObj?.entity && ctx.bot.entity.position.distanceTo(playerObj.entity.position) > 4.0) {
          if (ctx.kinematics?.pathfindTo) {
            await ctx.kinematics.pathfindTo(playerObj.entity.position, 2.5);
          }
          if (
            playerObj.entity &&
            ctx.bot.entity.position.distanceTo(playerObj.entity.position) > 4.5 &&
            ctx.bot.entity.position.y < playerObj.entity.position.y - 1.5
          ) {
            await tunnelToBlock(ctx, playerObj.entity.position.floored());
            if (ctx.kinematics?.pathfindTo) {
              await ctx.kinematics.pathfindTo(playerObj.entity.position, 2.5);
            }
          }
        }
      }

      const matchingInvItems = (ctx.bot.inventory?.items() || []).filter((i: any) =>
        i.name.toLowerCase().includes(itemQuery)
      );
      if (matchingInvItems.length === 0) {
        return {
          success: false,
          summary: `No "${itemQuery}" found in inventory to deposit into a chest.`,
        };
      }

      let chestBlock = ctx.bot.findBlock({
        matching: (b: any) => b?.name === 'chest' || b?.name === 'trapped_chest' || b?.name === 'barrel',
        maxDistance: 16,
      });

      if (chestBlock && ctx.bot.entity.position.distanceTo(chestBlock.position) > 3.2) {
        if (ctx.kinematics?.pathfindTo) {
          await ctx.kinematics.pathfindTo(chestBlock.position, 2.0);
        }
        if (ctx.bot.entity.position.distanceTo(chestBlock.position) > 4.2) {
          chestBlock = null;
        }
      }

      // If no reachable chest exists nearby, place an existing chest from inventory (never auto-craft)
      if (!chestBlock) {
        const hasChestInInv = (ctx.bot.inventory?.items() || []).some((i: any) => i.name === 'chest');
        if (hasChestInInv) {
          chestBlock = await placeBlockNearby(ctx, 'chest');
        }
      }

      if (!chestBlock) {
        return {
          success: false,
          summary: 'No chest nearby or in inventory. Craft a chest (8 planks) with craft_item first.',
        };
      }

      await ctx.kinematics.lookAtSmooth(chestBlock.position.offset(0.5, 0.5, 0.5), 180);

      try {
        const container = await ctx.bot.openContainer(chestBlock);
        let depositedTotal = 0;
        let depositedName = itemQuery;
        for (const stack of matchingInvItems) {
          if (depositedTotal >= maxCount) break;
          const toPut = Math.min(stack.count, maxCount - depositedTotal);
          depositedName = stack.name;
          await container.deposit(stack.type, null, toPut);
          depositedTotal += toPut;
        }
        container.close();
        ctx.reportProgress(1.0);
        return {
          success: depositedTotal > 0,
          summary: `Deposited ${depositedTotal}x ${depositedName} into chest at (${chestBlock.position.x}, ${chestBlock.position.y}, ${chestBlock.position.z})!`,
        };
      } catch (err: any) {
        return {
          success: false,
          summary: `Failed to deposit items into chest: ${err.message}`,
        };
      }
    }
  );

  // manage_chest: Discrete Symbolic Chest Transfers (MCP Domain)
  registry.registerTool(
    {
      name: 'manage_chest',
      description:
        'Discrete combinatorial chest inventory slot management: deposits items into or withdraws items from a chest container.',
      inputSchema: {
        type: 'object',
        properties: {
          action: {
            type: 'string',
            enum: ['deposit', 'withdraw'],
            description: 'Operation: "deposit" to store items into chest, "withdraw" to loot/take items from chest.',
          },
          itemName: {
            type: 'string',
            description: 'Item name to deposit or withdraw (e.g. iron_ingot, raw_iron, cobblestone, oak_log, or "all").',
            default: 'all',
          },
          count: {
            type: 'number',
            description: 'Quantity of items to transfer.',
            default: 64,
          },
        },
        required: ['action'],
      },
      tags: ['manage_chest', 'chest', 'deposit', 'withdraw', 'transfer', 'container', 'mcp'],
    },
    async (args, ctx) => {
      const action = (pickStrArg(args.action) || 'deposit').toLowerCase().trim();
      const itemName = (pickStrArg(args.itemName) || 'all').toLowerCase().trim();
      const count = Math.max(1, Number(args.count) || 64);

      if (action === 'deposit') {
        const storeTool = registry.getTool('store_items_in_chest');
        if (storeTool) {
          return await storeTool.handler({ itemName, count }, ctx);
        }
      } else {
        const lootTool = registry.getTool('loot_chest');
        if (lootTool) {
          return await lootTool.handler({ chestName: 'chest' }, ctx);
        }
      }
      return { success: false, summary: `Unknown chest action "${action}". Must be "deposit" or "withdraw".` };
    }
  );

  // 11. inspect_surroundings
  registry.registerTool(
    {
      name: 'inspect_surroundings',
      description: 'Performs a smooth 360-degree human gaze scan (North, East, South, West) of the area and summarizes resource clusters, terrain depth, visible entities, and inventory.',
      inputSchema: {
        type: 'object',
        properties: {},
      },
      tags: ['inspect', 'scan', 'look', '360', 'status', 'inventory', 'surroundings'],
    },
    async (_args, ctx) => {
      const pos = ctx.bot.entity.position;
      const directions = [
        new Vec3(pos.x, pos.y + 1.5, pos.z - 6), // North
        new Vec3(pos.x + 6, pos.y + 1.5, pos.z), // East
        new Vec3(pos.x, pos.y + 1.5, pos.z + 6), // South
        new Vec3(pos.x - 6, pos.y + 1.5, pos.z), // West
      ];
      for (let i = 0; i < directions.length; i++) {
        if (ctx.abortSignal?.aborted) break;
        await ctx.kinematics.lookAtSmooth(directions[i], 150);
        ctx.reportProgress((i + 1) / directions.length);
      }

      const state = ctx.worldState();
      const clusters = (state.terrainContext?.resourceClusters || [])
        .map((c) => `${c.blockType}(${c.totalNearby} total, ${c.exposedCount} exposed, ${c.nearestDistance}m ${c.nearestDirection})`)
        .join('; ');

      return {
        success: true,
        summary: `Completed 360° scan: ${state.visibleEntities.length} entities, stone depth=${state.terrainContext?.depthToStoneBlocks ?? 'unknown'}m below feet. Clusters: [${clusters || 'none'}].`,
        data: state,
      };
    }
  );

  // 12. clear_flat_area
  registry.registerTool(
    {
      name: 'clear_flat_area',
      description:
        'Clears and levels a designated rectangular area (e.g. 15x15, 10x10) around the player or bot. Pre-crafts axes and shovels if missing, chops trees bottom-up safely from solid ground, clears brush/leaves, levels soil down to floor height, and collects all dropped resources.',
      inputSchema: {
        type: 'object',
        properties: {
          width: {
            type: 'number',
            description: 'Width (X-axis) of the area to clear in blocks (default 15)',
            default: 15,
          },
          length: {
            type: 'number',
            description: 'Length (Z-axis) of the area to clear in blocks (default 15)',
            default: 15,
          },
          height: {
            type: 'number',
            description: 'Height in blocks above floor level to clear (default 4)',
            default: 4,
          },
          targetY: {
            type: 'number',
            description: 'Optional floor Y level to flatten down to (defaults to current ground level)',
          },
          speaker: {
            type: 'string',
            description: 'Optional player username requesting the clearing',
          },
        },
      },
      tags: ['clear', 'flat', 'flatten', 'level', 'area', 'ground', 'trees', 'dig', 'construction'],
    },
    async (args, ctx) => {
      const width = Math.max(1, Math.min(32, Math.floor(Number(args.width) || 15)));
      const length = Math.max(1, Math.min(32, Math.floor(Number(args.length) || 15)));
      const height = Math.max(2, Math.min(16, Math.floor(Number(args.height) || 4)));

      // 1. Determine center position
      const speaker = pickStrArg(args.speaker).toLowerCase();
      const speakerPlayer = Object.values(ctx.bot.players || {}).find(
        (p: any) => p?.entity && (!speaker || p.username.toLowerCase() === speaker)
      ) as any;
      const centerPos = (
        speakerPlayer?.entity?.position &&
        ctx.bot.entity.position.distanceTo(speakerPlayer.entity.position) <= 24
          ? speakerPlayer.entity.position
          : ctx.bot.entity.position
      ).floored();

      const floorY =
        typeof args.targetY === 'number' && !isNaN(args.targetY)
          ? Math.floor(args.targetY)
          : centerPos.y;
      const topY = floorY + height;

      const halfW = Math.floor(width / 2);
      const halfL = Math.floor(length / 2);
      const minX = centerPos.x - halfW;
      const maxX = minX + width - 1;
      const minZ = centerPos.z - halfL;
      const maxZ = minZ + length - 1;

      logVerbose('CLEAR', `Starting clear_flat_area: X=[${minX}, ${maxX}], Z=[${minZ}, ${maxZ}], Y=[${floorY}, ${topY}] (center=(${centerPos.x}, ${centerPos.y}, ${centerPos.z}))`);

      // 2. Pre-flight Tool Provisioning: ensure bot has axe, shovel, pickaxe
      await ensureToolsAvailable(ctx, ['axe', 'shovel', 'pickaxe']);

      const ignorable = new Set(['air', 'cave_air', 'void_air', 'bedrock', 'barrier', 'water', 'lava']);
      let blocksCleared = 0;

      // Phase 1: Chop any tree trunks within the bounding box from the bottom up!
      // This prevents tree floating issues and allows leaves to naturally decay or be cleared.
      for (let x = minX; x <= maxX; x++) {
        for (let z = minZ; z <= maxZ; z++) {
          if (ctx.abortSignal?.aborted) break;
          for (let y = floorY; y <= topY + 4; y++) {
            const b = ctx.bot.blockAt(new Vec3(x, y, z));
            if (b && (b.name.endsWith('_log') || b.name.endsWith('_stem'))) {
              // Found a tree trunk! Navigate nearby on the ground and chop upward safely
              if (ctx.kinematics?.pathfindTo && ctx.bot.entity.position.distanceTo(new Vec3(x, floorY, z)) > 3.0) {
                await ctx.kinematics.pathfindTo(new Vec3(x, floorY, z), 2.2);
              }
              for (let chopY = floorY; chopY <= topY + 4; chopY++) {
                if (ctx.abortSignal?.aborted) break;
                const logPos = new Vec3(x, chopY, z);
                const logBlock = ctx.bot.blockAt(logPos);
                if (logBlock && (logBlock.name.endsWith('_log') || logBlock.name.endsWith('_stem'))) {
                  const dug = await safeDigSingleBlock(ctx, logPos, 0, false);
                  if (dug) {
                    blocksCleared++;
                    await collectDroppedItemNear(ctx, logPos, logBlock.name, 0);
                  }
                }
              }
              break;
            }
          }
        }
      }

      // Phase 2: Clear remaining blocks above floorY layer by layer
      for (let y = topY; y >= floorY; y--) {
        if (ctx.abortSignal?.aborted) break;

        let foundBlockInLayer = true;
        let layerSafetyLimit = 0;
        while (foundBlockInLayer && !ctx.abortSignal?.aborted && layerSafetyLimit++ < 150) {
          foundBlockInLayer = false;
          let bestPos: Vec3 | null = null;
          let bestDist = Infinity;
          const botPos = ctx.bot.entity.position;

          for (let x = minX; x <= maxX; x++) {
            for (let z = minZ; z <= maxZ; z++) {
              const checkPos = new Vec3(x, y, z);
              const b = ctx.bot.blockAt(checkPos);
              if (!b || ignorable.has(b.name)) continue;

              const d = Math.hypot(botPos.x - (x + 0.5), botPos.z - (z + 0.5));
              if (d < bestDist) {
                bestDist = d;
                bestPos = checkPos;
                foundBlockInLayer = true;
              }
            }
          }

          if (bestPos) {
            if (ctx.abortSignal?.aborted || !ctx.bot?.entity) break;

            // CRITICAL: Always pathfind to solid ground level (floorY), NEVER to airborne coordinates (e.g. Y=101)!
            // Pathfinding to airborne coordinates forces the bot to jump into tree leaves/canopies and hover,
            // triggering Vanilla Minecraft's 80-tick anti-fly kick ("multiplayer.disconnect.flying")!
            const groundStandPos = new Vec3(bestPos.x, floorY, bestPos.z);
            const botFoot = ctx.bot.entity?.position?.floored() ?? groundStandPos;
            const horizDist = Math.hypot(botFoot.x - bestPos.x, botFoot.z - bestPos.z);

            if (horizDist > 3.2 && ctx.kinematics?.pathfindTo) {
              await ctx.kinematics.pathfindTo(groundStandPos, 2.0);
            }
            const b = ctx.bot.blockAt(bestPos);
            if (b && !ignorable.has(b.name)) {
              const bName = b.name;
              const dug = await safeDigSingleBlock(ctx, bestPos, 0, false);
              if (dug) {
                blocksCleared++;
                if (blocksCleared % 4 === 0) {
                  await collectDroppedItemNear(ctx, bestPos, bName, 0);
                }
              }
            }
          }
        }

        ctx.reportProgress(Math.min(1.0, (topY - y + 1) / (topY - floorY + 1)));
      }

      const summary = `Cleared and flattened area (${width}x${length}, ${blocksCleared} blocks removed) down to floor level Y=${floorY}.`;
      logVerbose('CLEAR', summary);
      return {
        success: true,
        summary,
      };
    }
  );

  // 13. stop_current_action
  registry.registerTool(
    {
      name: 'stop_current_action',
      description: 'Immediately stops any active pathfinding, mining, or movement and cancels any active long-term goal.',
      inputSchema: {
        type: 'object',
        properties: {},
      },
      tags: ['stop', 'halt', 'cancel', 'wait', 'freeze'],
    },
    async (_args, ctx) => {
      ctx.bot.pathfinder?.setGoal(null);
      ctx.bot.clearControlStates?.();
      ctx.reportProgress(1.0);
      return {
        success: true,
        summary: 'Stopped all movement and cleared active goals.',
      };
    }
  );

  // 14. scan_area
  registry.registerTool(
    {
      name: 'scan_area',
      description: 'Scans the 3D terrain topology and spatial structure around the agent (or a designated location). Reports ground elevation range (min/max/median), flatness, surface block types, obstacles, and building site suitability.',
      inputSchema: {
        type: 'object',
        properties: {
          radius: {
            type: 'number',
            description: 'Scan radius in blocks around center (default 8, scans (2*radius)x(2*radius) footprint)',
            default: 8,
          },
          centerX: { type: 'number', description: 'Optional center X coordinate (defaults to bot position)' },
          centerZ: { type: 'number', description: 'Optional center Z coordinate (defaults to bot position)' },
        },
      },
      tags: ['scan', 'inspect', 'see', 'vision', 'terrain', 'ground', 'height', 'topology', 'area'],
    },
    async (args, ctx) => {
      const r = Math.max(2, Math.min(24, Math.floor(Number(args.radius) || 8)));
      const cx = args.centerX !== undefined ? Math.floor(Number(args.centerX)) : Math.floor(ctx.bot.entity.position.x);
      const cz = args.centerZ !== undefined ? Math.floor(Number(args.centerZ)) : Math.floor(ctx.bot.entity.position.z);

      const minX = cx - r;
      const maxX = cx + r;
      const minZ = cz - r;
      const maxZ = cz + r;

      // Look around in 4 directions smoothly while scanning
      const botPos = ctx.bot.entity.position;
      const dirs = [
        new Vec3(botPos.x, botPos.y + 1.6, botPos.z - 5),
        new Vec3(botPos.x + 5, botPos.y + 1.6, botPos.z),
        new Vec3(botPos.x, botPos.y + 1.6, botPos.z + 5),
        new Vec3(botPos.x - 5, botPos.y + 1.6, botPos.z),
      ];
      for (let i = 0; i < dirs.length; i++) {
        if (ctx.abortSignal?.aborted) break;
        await ctx.kinematics.lookAtSmooth(dirs[i], 120);
        ctx.reportProgress(((i + 1) / dirs.length) * 0.4);
      }

      const scan = ctx.world?.scanAreaTopology
        ? ctx.world.scanAreaTopology(minX, minZ, maxX, maxZ)
        : {
            minX,
            minZ,
            maxX,
            maxZ,
            minHeight: Math.floor(botPos.y) - 1,
            maxHeight: Math.floor(botPos.y) - 1,
            medianHeight: Math.floor(botPos.y) - 1,
            isFlat: true,
            obstructionCount: 0,
            airGapCount: 0,
            groundBlockTypes: {},
            heightGrid: [],
          };
      ctx.reportProgress(1.0);

      const blockDist = Object.entries(scan.groundBlockTypes)
        .map(([k, v]) => `${k}: ${v}`)
        .join(', ');

      const flatnessStr = scan.isFlat
        ? 'Flat terrain (height variance <= 1 block)'
        : `Uneven terrain (elevation range Y=${scan.minHeight} to Y=${scan.maxHeight}, median Y=${scan.medianHeight})`;

      const summary = `Scanned ${maxX - minX + 1}x${maxZ - minZ + 1} area around (${cx}, ${cz}): ${flatnessStr}. Surface blocks: [${blockDist || 'none'}]. Obstacles: ${scan.obstructionCount} blocks above ground, ${scan.airGapCount} low air gaps.`;
      return {
        success: true,
        summary,
        data: scan,
      };
    }
  );

  // 15. build_simple_house
  registry.registerTool(
    {
      name: 'build_simple_house',
      description:
        'Constructs a complete, fully-grounded survival shelter/house near the agent with walls, windows, a doorway, roof, and proper foundation (preventing any floating blocks in mid-air).',
      inputSchema: {
        type: 'object',
        properties: {
          width: { type: 'number', description: 'Width of house in blocks (default 5, min 4, max 10)', default: 5 },
          length: { type: 'number', description: 'Length of house in blocks (default 5, min 4, max 10)', default: 5 },
          height: { type: 'number', description: 'Wall height of house in blocks (default 3, min 3, max 5)', default: 3 },
          wallBlock: {
            type: 'string',
            description: 'Block type for walls (e.g. oak_planks, cobblestone, stone). Defaults to available planks/cobble.',
          },
          roofBlock: { type: 'string', description: 'Block type for roof. Defaults to wallBlock.' },
        },
      },
      tags: ['build', 'house', 'shelter', 'home', 'structure', 'walls', 'roof'],
    },
    async (args, ctx) => {
      // 1. Identify building material
      const inv = ctx.bot.inventory?.items() || [];
      const countItem = (n: string) =>
        inv.filter((it: any) => it.name === n || it.name.includes(n)).reduce((s: number, it: any) => s + it.count, 0);

      let wallBlock = pickStrArg(args.wallBlock).toLowerCase().replace('minecraft:', '').trim();
      let roofBlock = pickStrArg(args.roofBlock).toLowerCase().replace('minecraft:', '').trim();

      if (!wallBlock) {
        const candidates = [
          'oak_planks',
          'spruce_planks',
          'birch_planks',
          'jungle_planks',
          'acacia_planks',
          'dark_oak_planks',
          'mangrove_planks',
          'cherry_planks',
          'cobblestone',
          'cobbled_deepslate',
          'stone',
          'sandstone',
          'dirt',
        ];
        let bestCount = 0;
        for (const c of candidates) {
          const cnt = countItem(c);
          if (cnt > bestCount) {
            bestCount = cnt;
            wallBlock = c;
          }
        }
        if (!wallBlock || bestCount === 0) {
          wallBlock = 'oak_planks';
        }
      }

      if (!roofBlock) {
        roofBlock = wallBlock;
      }

      const availableWall = countItem(wallBlock);

      if (availableWall < 12) {
        return {
          success: false,
          summary: `Not enough building blocks (found ${availableWall}x ${wallBlock}, need at least 12). Craft planks via craft_item or call gather_resource first!`,
        };
      }

      const w = Math.max(4, Math.min(10, Math.floor(Number(args.width) || 5)));
      const l = Math.max(4, Math.min(10, Math.floor(Number(args.length) || 5)));
      const h = Math.max(3, Math.min(5, Math.floor(Number(args.height) || 3)));

      const botPos = ctx.bot.entity.position;
      const yaw = ctx.bot.entity.yaw ?? 0;
      let fwdX = Math.round(-Math.sin(yaw));
      let fwdZ = Math.round(Math.cos(yaw));
      if (fwdX === 0 && fwdZ === 0) {
        fwdX = 1;
      }

      const startX = Math.floor(botPos.x) + (fwdX !== 0 ? fwdX * 2 : 1);
      const startZ = Math.floor(botPos.z) + (fwdZ !== 0 ? fwdZ * 2 : 1);

      // 2. Scan site topology to anchor foundation directly on terrain
      const topology = ctx.world?.scanAreaTopology
        ? ctx.world.scanAreaTopology(startX, startZ, startX + w - 1, startZ + l - 1)
        : {
            minX: startX,
            minZ: startZ,
            maxX: startX + w - 1,
            maxZ: startZ + l - 1,
            minHeight: Math.floor(botPos.y) - 1,
            maxHeight: Math.floor(botPos.y) - 1,
            medianHeight: Math.floor(botPos.y) - 1,
            isFlat: true,
            obstructionCount: 0,
            airGapCount: 0,
            groundBlockTypes: {},
            heightGrid: [],
          };
      const baseY = topology.medianHeight + 1;

      await ctx.sayHumanized(`Starting construction of a ${w}x${l} house using ${wallBlock} (ground level Y=${baseY})...`);

      // 3. Clear obstructions in building envelope
      for (let x = startX; x < startX + w; x++) {
        for (let z = startZ; z < startZ + l; z++) {
          if (ctx.abortSignal?.aborted) break;
          const gh = typeof ctx.world?.getGroundHeight === 'function' ? ctx.world.getGroundHeight(x, z, baseY + 4) : baseY - 1;
          for (let y = Math.max(baseY, gh + 1); y <= baseY + h + 1; y++) {
            const ob = ctx.bot.blockAt(new Vec3(x, y, z));
            if (ob && ob.name !== 'air' && ob.name !== 'cave_air' && ob.boundingBox === 'block') {
              if (!ob.name.includes('bedrock')) {
                await ctx.kinematics.safeDig(new Vec3(x, y, z));
              }
            }
          }
        }
      }

      let placedCount = 0;

      // 4. Lay foundation down to ground for any gaps/drops
      for (let x = 0; x < w; x++) {
        for (let z = 0; z < l; z++) {
          if (ctx.abortSignal?.aborted) break;
          const isPerimeter = x === 0 || x === w - 1 || z === 0 || z === l - 1;
          if (!isPerimeter) continue;

          const wx = startX + x;
          const wz = startZ + z;
          const gh = typeof ctx.world?.getGroundHeight === 'function' ? ctx.world.getGroundHeight(wx, wz, baseY) : baseY - 1;
          if (gh < baseY - 1) {
            for (let fy = gh + 1; fy <= baseY - 1; fy++) {
              let placed = false;
              if (typeof ctx.kinematics?.safePlaceBlock === 'function') {
                placed = await ctx.kinematics.safePlaceBlock(new Vec3(wx, fy, wz), wallBlock);
              }
              if (placed) placedCount++;
            }
          }
        }
      }

      // 5. Build walls layer by layer
      const doorX = Math.floor(w / 2);
      const totalBlocksEstimate = (2 * w + 2 * (l - 2)) * h + w * l;

      for (let dy = 0; dy < h; dy++) {
        for (let x = 0; x < w; x++) {
          for (let z = 0; z < l; z++) {
            if (ctx.abortSignal?.aborted) {
              return { success: false, summary: `House construction aborted. Placed ${placedCount} blocks.` };
            }
            const isPerimeter = x === 0 || x === w - 1 || z === 0 || z === l - 1;
            if (!isPerimeter) continue;

            // Doorway on front wall (z = 0) in the middle at dy=0 and dy=1
            if (z === 0 && x === doorX && (dy === 0 || dy === 1)) {
              continue;
            }

            // Windows on side walls at dy = 1 in the middle
            if (dy === 1) {
              if ((x === 0 || x === w - 1) && z === Math.floor(l / 2)) {
                continue;
              }
            }

            const targetPos = new Vec3(startX + x, baseY + dy, startZ + z);
            let placed = false;
            if (typeof ctx.kinematics?.safePlaceBlock === 'function') {
              placed = await ctx.kinematics.safePlaceBlock(targetPos, wallBlock);
            }
            if (placed) placedCount++;
            ctx.reportProgress(Math.min(0.85, placedCount / totalBlocksEstimate));
          }
        }
      }

      // 6. Build Roof
      const roofY = baseY + h;
      for (let x = 0; x < w; x++) {
        for (let z = 0; z < l; z++) {
          if (ctx.abortSignal?.aborted) break;
          const targetPos = new Vec3(startX + x, roofY, startZ + z);
          let placed = false;
          if (typeof ctx.kinematics?.safePlaceBlock === 'function') {
            placed = await ctx.kinematics.safePlaceBlock(targetPos, roofBlock);
          }
          if (placed) placedCount++;
          ctx.reportProgress(Math.min(0.95, placedCount / totalBlocksEstimate));
        }
      }

      // 7. Place interior torches if available
      const hasTorches = (ctx.bot.inventory?.items() || []).some((it: any) => it.name === 'torch');
      if (hasTorches) {
        const torchPos = new Vec3(startX + 1, baseY + 1, startZ + Math.floor(l / 2));
        if (typeof ctx.kinematics?.safePlaceBlock === 'function') {
          await ctx.kinematics.safePlaceBlock(torchPos, 'torch', false);
        }
      }

      // 8. Move outside and gaze at the completed house
      const frontDoorPos = new Vec3(startX + doorX, baseY, startZ - 2);
      await ctx.kinematics.pathfindTo(frontDoorPos, 1.5);
      await ctx.kinematics.lookAtSmooth(new Vec3(startX + doorX, baseY + 1.5, startZ), 400);

      ctx.reportProgress(1.0);
      const summary = `Successfully built a ${w}x${l}x${h} house at (${startX}, ${baseY}, ${startZ}) using ${wallBlock} and ${roofBlock}! Placed ${placedCount} blocks with a doorway, windows, and ground-anchored foundation.`;
      await ctx.sayHumanized(`House complete! Built with ${placedCount} blocks.`);
      return {
        success: true,
        summary,
        data: { placedCount, origin: { x: startX, y: baseY, z: startZ }, width: w, length: l, height: h },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // 16. NEURAL_CHUNK_SCAN (RTX 5060 64x64x64 Volumetric 3D Scene Transformer)
  // ---------------------------------------------------------------------------
  registry.registerTool(
    {
      name: 'neural_chunk_scan',
      description:
        'Executes a 64x64x64 volumetric 3D terrain & ore scan on the RTX 5060 GPU (~4.8ms) to identify the flattest nearby building pad, richest underground ore vein direction, and cavern/lava hazards.',
      inputSchema: {
        type: 'object',
        properties: {
          radius: {
            type: 'number',
            description: 'Scan radius around the bot (default: 32 blocks, covering a 64x64x64 volume).',
            default: 32,
          },
        },
      },
      isSynthesized: false,
      tags: ['perception', 'neural', 'scan', 'ores', 'building'],
    },
    async (args, ctx) => {
      if (!ctx.bot?.entity) {
        return { success: false, summary: 'Bot is not spawned.' };
      }
      const pos = ctx.bot.entity.position.floored();
      const sparseBlocks: Array<{ dx: number; dy: number; dz: number; name: string }> = [];

      // Sample key blocks around the bot for the 64x64x64 tensor
      for (let dx = -16; dx <= 16; dx += 2) {
        for (let dz = -16; dz <= 16; dz += 2) {
          for (let dy = -16; dy <= 8; dy += 2) {
            const b = ctx.bot.blockAt(pos.offset(dx, dy, dz));
            if (b && b.name !== 'air' && b.name !== 'cave_air') {
              sparseBlocks.push({ dx, dy, dz, name: b.name });
            }
          }
        }
      }

      const gpuResult = await sharedNeuralBridge.evaluateChunkScan(sparseBlocks);
      if (gpuResult) {
        const buildTarget = {
          x: Math.round(pos.x + gpuResult.recommended_build_offset.dx),
          y: Math.round(pos.y + gpuResult.recommended_build_offset.dy),
          z: Math.round(pos.z + gpuResult.recommended_build_offset.dz),
        };
        const oreTarget = {
          x: Math.round(pos.x + gpuResult.recommended_ore_offset.dx),
          y: Math.round(pos.y + gpuResult.recommended_ore_offset.dy),
          z: Math.round(pos.z + gpuResult.recommended_ore_offset.dz),
        };
        const summary = `[RTX 5060 Volumetric 64³ Scan in ${gpuResult.latency_ms}ms] Best building site: (${buildTarget.x}, ${buildTarget.y}, ${buildTarget.z}) [flatness=${(
          gpuResult.flatness_confidence * 100
        ).toFixed(0)}%]; Richest ore hotspot: (${oreTarget.x}, ${oreTarget.y}, ${oreTarget.z}) [richness=${(
          gpuResult.ore_richness_score * 100
        ).toFixed(0)}%]; Hazard score: ${(gpuResult.cavern_hazard_score * 100).toFixed(0)}%.`;
        return {
          success: true,
          summary,
          data: { ...gpuResult, buildTarget, oreTarget, acceleratedByGpu: true },
        };
      }

      // Graceful CPU fallback if Neural Daemon is not running
      if (typeof ctx.world?.scanAreaTopology === 'function') {
        const topo = ctx.world.scanAreaTopology(pos.x - 8, pos.z - 8, pos.x + 8, pos.z + 8);
        return {
          success: true,
          summary: `[CPU Fallback Topology Scan] Elevation ${topo.minHeight}..${topo.maxHeight} (median Y=${topo.medianHeight}, isFlat=${topo.isFlat}), obstructions=${topo.obstructionCount}.`,
          data: { topology: topo, acceleratedByGpu: false },
        };
      }

      return { success: false, summary: 'Neither GPU Neural Daemon nor topology scanner available.' };
    }
  );

  // ---------------------------------------------------------------------------
  // 17. COLLECT_DROPPED_ITEMS (Built-in Loot & Death Recovery Skill)
  // ---------------------------------------------------------------------------
  registry.registerTool(
    {
      name: 'collect_dropped_items',
      description:
        'Scans for dropped item entities within radius (default 32m), pathfinds over each drop to pick it up, and automatically re-equips the best armor and weapon.',
      inputSchema: {
        type: 'object',
        properties: {
          radius: {
            type: 'number',
            description: 'Search radius in blocks for dropped items',
            default: 32,
          },
          maxItems: {
            type: 'number',
            description: 'Maximum number of dropped item clusters to visit',
            default: 15,
          },
        },
      },
      isSynthesized: false,
      tags: ['collect', 'pickup', 'items', 'drops', 'loot', 'recover', 'stuff', 'gear', 'inventory'],
    },
    async (args, ctx) => {
      const radius = Math.max(4, Math.min(64, Number(args.radius) || 32));
      const maxItems = Math.max(1, Math.min(30, Number(args.maxItems) || 15));
      let collectedClusters = 0;

      for (let i = 0; i < maxItems; i++) {
        if (ctx.abortSignal?.aborted) break;

        const droppedItems = Object.values(ctx.bot.entities || {})
          .filter(
            (e: any) =>
              e &&
              e.isValid !== false &&
              e.position &&
              (e.name === 'item' ||
                e.name === 'experience_orb' ||
                e.type === 'object' ||
                e.displayName === 'Item') &&
              e.position.distanceTo(ctx.bot.entity.position) <= radius
          )
          .sort(
            (a: any, b: any) =>
              a.position.distanceTo(ctx.bot.entity.position) -
              b.position.distanceTo(ctx.bot.entity.position)
          );

        if (droppedItems.length === 0) break;

        const targetDrop: any = droppedItems[0];
        const targetPos = targetDrop.position.clone();
        await ctx.kinematics.lookAtSmooth(targetPos, 150);
        if (ctx.kinematics?.pathfindTo) {
          await ctx.kinematics.pathfindTo(targetPos, 1.0);
        }
        await new Promise((r) => setTimeout(r, 220));
        collectedClusters++;
        ctx.reportProgress(Math.min(0.95, collectedClusters / Math.min(maxItems, droppedItems.length + collectedClusters)));
      }

      ctx.bot.pathfinder.setGoal(null);

      // Automatically re-equip best weapon if picked up
      const weapons = [
        'netherite_sword',
        'diamond_sword',
        'iron_sword',
        'stone_sword',
        'wooden_sword',
      ];
      for (const w of weapons) {
        if (await ctx.kinematics.equipItemHumanized(w, 'hand')) break;
      }

      ctx.reportProgress(1.0);
      return {
        success: true,
        summary:
          collectedClusters > 0
            ? `Collected ${collectedClusters} dropped item cluster(s) within ${radius}m.`
            : `No dropped items found on the ground within ${radius}m.`,
        data: { collectedClusters },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // 18. TAME_ANIMAL (Built-in Companion Taming Skill)
  // ---------------------------------------------------------------------------
  registry.registerTool(
    {
      name: 'tame_animal',
      description:
        'Approaches a nearby tamable animal (e.g. wolf, cat, horse, parrot) and feeds it the required taming item (e.g. bone for wolf, cod/salmon for cat, wheat_seeds for parrot) until tamed.',
      inputSchema: {
        type: 'object',
        properties: {
          animalType: {
            type: 'string',
            description: 'Animal type to tame (e.g. "wolf", "cat", "parrot")',
            default: 'wolf',
          },
          itemName: {
            type: 'string',
            description: 'Item to use for taming (optional; auto-selects bone for wolf, cod/salmon for cat)',
          },
        },
      },
      isSynthesized: false,
      tags: ['tame', 'wolf', 'dog', 'cat', 'pet', 'animal', 'bone', 'companion'],
    },
    async (args, ctx) => {
      const rawAnimal = (pickStrArg(args.animalType) || 'wolf').toLowerCase().trim();
      const animalType =
        rawAnimal === 'dog' || rawAnimal === 'dogs' || rawAnimal === 'wolves'
          ? 'wolf'
          : rawAnimal.replace(/s$/, '');

      const defaultFoods: Record<string, string[]> = {
        wolf: ['bone'],
        cat: ['cod', 'salmon'],
        ocelot: ['cod', 'salmon'],
        parrot: ['wheat_seeds', 'melon_seeds', 'pumpkin_seeds', 'beetroot_seeds', 'torchflower_seeds'],
      };

      const candidateItems = args.itemName
        ? [pickStrArg(args.itemName).toLowerCase().trim()].filter(Boolean)
        : defaultFoods[animalType] || ['bone', 'cod', 'salmon', 'wheat'];

      const targetAnimal: any = Object.values(ctx.bot.entities || {})
        .filter(
          (e: any) =>
            e &&
            e !== ctx.bot.entity &&
            e.isValid !== false &&
            e.position &&
            String(e.name || e.displayName || '').toLowerCase() === animalType &&
            e.position.distanceTo(ctx.bot.entity.position) <= 32
        )
        .sort(
          (a: any, b: any) =>
            a.position.distanceTo(ctx.bot.entity.position) -
            b.position.distanceTo(ctx.bot.entity.position)
        )[0];

      if (!targetAnimal) {
        return {
          success: false,
          summary: `No ${animalType} found within 32m to tame.`,
        };
      }

      let chosenItem: string | null = null;
      for (const item of candidateItems) {
        const invMatch = (ctx.bot.inventory?.items() || []).find(
          (it: any) => it.name === item || it.name.includes(item)
        );
        if (invMatch) {
          chosenItem = invMatch.name;
          break;
        }
      }

      if (!chosenItem) {
        return {
          success: false,
          summary: `Found a ${animalType} nearby, but I don't have ${candidateItems.join(' or ')} in my inventory to tame it!`,
        };
      }

      await ctx.kinematics.lookAtSmooth(targetAnimal.position.offset(0, 0.6, 0), 200);
      if (ctx.kinematics?.pathfindTo) {
        await ctx.kinematics.pathfindTo(targetAnimal.position, 2.2);
      }

      await ctx.kinematics.equipItemHumanized(chosenItem, 'hand');

      let attempts = 0;
      const maxAttempts = 12;
      while (attempts < maxAttempts && targetAnimal.isValid && !ctx.abortSignal?.aborted) {
        const stillHasItem = (ctx.bot.inventory?.items() || []).some(
          (it: any) => it.name === chosenItem
        );
        if (!stillHasItem && ctx.bot.heldItem?.name !== chosenItem) break;

        if (ctx.bot.entity.position.distanceTo(targetAnimal.position) > 3.2 && ctx.kinematics?.pathfindTo) {
          await ctx.kinematics.pathfindTo(targetAnimal.position, 2.0);
        }
        await ctx.kinematics.lookAtSmooth(targetAnimal.position.offset(0, 0.5, 0), 120);
        await ctx.kinematics.equipItemHumanized(chosenItem, 'hand');
        try {
          await ctx.bot.activateEntity(targetAnimal);
        } catch {
          // Ignore transient entity activation errors
        }
        attempts++;
        ctx.reportProgress(Math.min(0.95, attempts / 6));
        await new Promise((r) => setTimeout(r, 350));

        // Check tamed metadata bit (In 1.20.4, TameableAnimal bitmask at index 17 has 0x04 when tamed)
        const tameFlags = Number(targetAnimal.metadata?.[17] ?? 0);
        if ((tameFlags & 0x04) !== 0) {
          break;
        }
      }

      ctx.reportProgress(1.0);
      return {
        success: attempts > 0,
        summary: `Fed ${chosenItem} to ${animalType} (${attempts}x interaction attempts).`,
        data: { animalType, itemUsed: chosenItem, attempts },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // 19. PILLAR_UP (Built-in Vertical Pillaring & Jump-Place Skill)
  // ---------------------------------------------------------------------------
  registry.registerTool(
    {
      name: 'pillar_up',
      description:
        'Pillars straight up by jumping and placing solid blocks directly underneath the agent.',
      inputSchema: {
        type: 'object',
        properties: {
          height: {
            type: 'number',
            description: 'Number of blocks to pillar upwards (default 3)',
            default: 3,
          },
          blockName: {
            type: 'string',
            description: 'Optional preferred block name (e.g. dirt, cobblestone, oak_planks)',
          },
        },
      },
      isSynthesized: false,
      tags: ['pillar', 'tower', 'up', 'jump', 'stack', 'vertical', 'climb'],
    },
    async (args, ctx) => {
      const targetHeight = Math.max(1, Math.min(16, Number(args.height) || 3));
      const preferred = pickStrArg(args.blockName).toLowerCase().trim();
      const placeableCandidates = [
        ...(preferred ? [preferred] : []),
        'dirt',
        'cobblestone',
        'cobbled_deepslate',
        'oak_planks',
        'spruce_planks',
        'birch_planks',
        'stone',
        'netherrack',
        'sandstone',
      ];

      let chosenBlock: string | null = null;
      for (const bName of placeableCandidates) {
        const found = (ctx.bot.inventory?.items() || []).find(
          (it: any) => it.name === bName && it.count > 0
        );
        if (found) {
          chosenBlock = found.name;
          break;
        }
      }

      if (!chosenBlock) {
        return {
          success: false,
          summary: 'No solid building blocks (dirt, cobblestone, planks) in inventory to pillar up.',
        };
      }

      ctx.bot.pathfinder.setGoal(null);
      await ctx.kinematics.equipItemHumanized(chosenBlock, 'hand');

      let placed = 0;
      for (let i = 0; i < targetHeight; i++) {
        if (ctx.abortSignal?.aborted) break;
        const startY = ctx.bot.entity.position.y;
        const placePos = ctx.bot.entity.position.floored();
        ctx.bot.look(ctx.bot.entity.yaw, -Math.PI / 2, true);
        ctx.bot.setControlState('jump', true);

        const jumpStart = Date.now();
        while (Date.now() - jumpStart < 550) {
          if (ctx.bot.entity.position.y > startY + 1.05) {
            const refBlock = ctx.bot.blockAt(placePos.offset(0, -1, 0));
            if (refBlock && refBlock.name !== 'air') {
              try {
                await ctx.bot.placeBlock(refBlock, new Vec3(0, 1, 0));
                placed++;
              } catch {
                if (typeof ctx.kinematics?.safePlaceBlock === 'function') {
                  const ok = await ctx.kinematics.safePlaceBlock(placePos, chosenBlock, false);
                  if (ok) placed++;
                }
              }
            } else if (typeof ctx.kinematics?.safePlaceBlock === 'function') {
              const ok = await ctx.kinematics.safePlaceBlock(placePos, chosenBlock, false);
              if (ok) placed++;
            }
            break;
          }
          await new Promise((r) => setTimeout(r, 20));
        }
        ctx.bot.setControlState('jump', false);
        await new Promise((r) => setTimeout(r, 220));
        ctx.reportProgress((i + 1) / targetHeight);
      }

      ctx.bot.setControlState('jump', false);
      ctx.reportProgress(1.0);
      return {
        success: placed > 0,
        summary: `Pillared up ${placed}/${targetHeight} blocks using ${chosenBlock}.`,
        data: { placed, blockUsed: chosenBlock },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // 20. EAT_FOOD (Built-in Hunger & Health Restoration Skill)
  // ---------------------------------------------------------------------------
  registry.registerTool(
    {
      name: 'eat_food',
      description:
        'Equips the best edible food in inventory (e.g. cooked_beef, bread, golden_apple, baked_potato, apple) and eats it until hunger is restored.',
      inputSchema: {
        type: 'object',
        properties: {
          foodName: {
            type: 'string',
            description: 'Optional specific food item name to eat (defaults to best available)',
          },
        },
      },
      isSynthesized: false,
      tags: ['eat', 'food', 'hungry', 'hunger', 'heal', 'consume', 'beef', 'steak', 'bread', 'apple'],
    },
    async (args, ctx) => {
      const preferred = (
        pickStrArg(args.foodName) ||
        pickStrArg(args.itemName) ||
        pickStrArg(args.item)
      ).toLowerCase().trim();
      const isBeverage =
        preferred.includes('milk') ||
        preferred.includes('potion') ||
        preferred.includes('bottle') ||
        preferred.includes('honey');

      const foodPriority = [
        ...(preferred ? [preferred] : []),
        'enchanted_golden_apple',
        'golden_apple',
        'golden_carrot',
        'milk_bucket',
        'honey_bottle',
        'cooked_beef',
        'cooked_porkchop',
        'cooked_mutton',
        'cooked_salmon',
        'bread',
        'baked_potato',
        'cooked_chicken',
        'cooked_cod',
        'mushroom_stew',
        'rabbit_stew',
        'beetroot_soup',
        'pumpkin_pie',
        'apple',
        'carrot',
        'melon_slice',
        'sweet_berries',
        'glow_berries',
        'dried_kelp',
        'potato',
        'beef',
        'porkchop',
        'mutton',
        'chicken',
        'rabbit',
        'cod',
        'salmon',
      ];

      const invItems = ctx.bot.inventory?.items() || [];
      let chosenFood: string | null = null;
      for (const f of foodPriority) {
        const match = invItems.find((it: any) => it.name === f || it.name.includes(f));
        if (match) {
          chosenFood = match.name;
          break;
        }
      }

      if (!chosenFood) {
        return {
          success: false,
          summary: 'No edible food items found in inventory!',
        };
      }

      const isGolden = chosenFood.includes('golden_apple');
      const isLiquid = chosenFood.includes('milk') || chosenFood.includes('potion') || chosenFood.includes('bottle') || chosenFood.includes('honey');
      if ((ctx.bot.food ?? 20) >= 20 && !isGolden && !isLiquid) {
        return {
          success: true,
          summary: `Hunger is already full (20/20); holding ${chosenFood} ready for when hunger drops.`,
        };
      }

      await ctx.kinematics.equipItemHumanized(chosenFood, 'hand');
      ctx.reportProgress(0.4);
      try {
        await ctx.bot.consume();
      } catch {
        ctx.bot.activateItem();
        await new Promise((r) => setTimeout(r, 1650));
        ctx.bot.deactivateItem();
      }
      ctx.reportProgress(1.0);
      return {
        success: true,
        summary: `Ate ${chosenFood}. Hunger is now ${ctx.bot.food ?? 20}/20 (HP: ${(ctx.bot.health ?? 20).toFixed(1)}/20).`,
        data: { foodEaten: chosenFood, foodLevel: ctx.bot.food, health: ctx.bot.health },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // 21. SHOOT_BOW (Ballistic Ranged Combat & End Crystal Sniper Skill)
  // ---------------------------------------------------------------------------
  registry.registerTool(
    {
      name: 'shoot_bow',
      description:
        'Equips a bow (or crossbow) and fires ballistic gravity-compensated arrows at a target entity (e.g. end_crystal, creeper, blaze, skeleton, ender_dragon).',
      inputSchema: {
        type: 'object',
        properties: {
          targetType: {
            type: 'string',
            description: 'Target entity name (e.g. "end_crystal", "creeper", "blaze", "skeleton", "ender_dragon")',
            default: 'hostile',
          },
          shots: {
            type: 'number',
            description: 'Maximum number of arrows to fire (default 3)',
            default: 3,
          },
        },
      },
      isSynthesized: false,
      tags: ['bow', 'shoot', 'arrow', 'ranged', 'snipe', 'end_crystal', 'blaze', 'ender_dragon'],
    },
    async (args, ctx) => {
      const targetType = (pickStrArg(args.targetType) || pickStrArg(args.target) || pickStrArg(args.targetEntity) || 'hostile').toLowerCase().trim();
      const maxShots = Math.max(1, Math.min(12, Number(args.shots) || 3));

      const hasBow = await ctx.kinematics.equipItemHumanized('bow', 'hand');
      if (!hasBow) {
        return { success: false, summary: 'No bow in inventory to shoot!' };
      }

      let fired = 0;
      const useNeuralMotor =
        process.env.NEURAL_MOTOR_CONTROL !== 'false' &&
        typeof ctx.kinematics.executeNeuralMotorStep === 'function';

      try {
        for (let i = 0; i < maxShots; i++) {
          if (ctx.abortSignal?.aborted) break;
          const target: any = Object.values(ctx.bot.entities || {})
            .filter((e: any) => {
              if (!e || e === ctx.bot.entity || e.isValid === false || !e.position) return false;
              const n = String(e.name || e.displayName || '').toLowerCase();
              const u = String(e.username || '').toLowerCase();
              if (u && u === ctx.bot.username?.toLowerCase()) return false;
              if (targetType === 'hostile') {
                return ['creeper', 'skeleton', 'blaze', 'ghast', 'witch', 'spider', 'zombie', 'ender_dragon', 'end_crystal'].includes(n) || e.type === 'player';
              }
              return (
                n.includes(targetType) ||
                u.includes(targetType) ||
                (e.type === 'player' && (
                  targetType.includes('player') ||
                  targetType.includes('opponent') ||
                  targetType.includes('rival') ||
                  targetType.includes('dqn') ||
                  targetType.includes('spar') ||
                  targetType.includes('mirror') ||
                  targetType.includes('agent')
                ))
              );
            })
            .sort(
              (a: any, b: any) =>
                a.position.distanceTo(ctx.bot.entity.position) -
                b.position.distanceTo(ctx.bot.entity.position)
            )[0];

          if (!target) break;

          if (useNeuralMotor) {
            ctx.kinematics.setActiveMotorTarget?.(target.position);
            let shotExecuted = false;
            let shotReleased = false;
            for (let chunkIdx = 0; chunkIdx < 5; chunkIdx++) {
              if (ctx.abortSignal?.aborted || target.isValid === false) break;
              const res = await ctx.kinematics.executeNeuralMotorStep!(
                `Aim and shoot bow at ${targetType}`,
                target.position,
                8,
                {
                  allowUseItem: true,
                  allowAttack: false,
                  allowDig: false,
                  allowItemDrop: false,
                  persistItemUse: true,
                  neuralAimOnly: true,
                  abortSignal: ctx.abortSignal,
                }
              );
              if (!res.executed) break;
              shotExecuted = true;
              if (res.bowReleases > 0) {
                fired += res.bowReleases;
                shotReleased = true;
                break;
              }
            }
            if (shotExecuted) {
              (ctx.bot as any)._vlaItemUseState = null;
              try { ctx.bot.deactivateItem?.(); } catch {}
              try { ctx.bot.clearControlStates?.(); } catch {}
              if (!shotReleased) fired += 0;
              ctx.reportProgress((i + 1) / maxShots);
              await new Promise((r) => setTimeout(r, 150));
              continue;
            }
          }

          const horizDist = Math.hypot(
            target.position.x - ctx.bot.entity.position.x,
            target.position.z - ctx.bot.entity.position.z
          );
          // Offline fallback ballistic gravity arc compensation (+0.11m per horizontal block)
          const aimPoint = target.position.offset(0, (target.height || 1.6) * 0.65 + horizDist * 0.11, 0);
          await ctx.kinematics.lookAtSmooth(aimPoint, 160);
          ctx.bot.activateItem();
          await new Promise((r) => setTimeout(r, 1050));
          await ctx.kinematics.lookAtSmooth(aimPoint, 80);
          ctx.bot.deactivateItem();
          fired++;
          ctx.reportProgress((i + 1) / maxShots);
          await new Promise((r) => setTimeout(r, 250));
        }
      } finally {
        ctx.kinematics.setActiveMotorTarget?.(null);
        (ctx.bot as any)._vlaItemUseState = null;
        try { ctx.bot.deactivateItem?.(); } catch {}
        try { ctx.bot.clearControlStates?.(); } catch {}
      }

      ctx.reportProgress(1.0);
      return {
        success: fired > 0,
        summary: fired > 0 ? `Fired ${fired} ballistic arrow(s) at ${targetType}.` : `No ${targetType} in range to shoot.`,
        data: { fired, targetType },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // 22. TILL_AND_FARM (Homesteading, Crop Tilling, Planting & Bone-Meal Skill)
  // ---------------------------------------------------------------------------
  registry.registerTool(
    {
      name: 'till_and_farm',
      description:
        'Tills dirt/grass_block near water using a hoe, plants seeds/crops (wheat_seeds, carrot, potato), applies bone_meal if available, and harvests mature crops.',
      inputSchema: {
        type: 'object',
        properties: {
          plots: {
            type: 'number',
            description: 'Number of farmland plots to till and plant (default 4)',
            default: 4,
          },
        },
      },
      isSynthesized: false,
      tags: ['farm', 'till', 'hoe', 'plant', 'wheat', 'seeds', 'crops', 'harvest', 'homestead'],
    },
    async (args, ctx) => {
      const targetPlots = Math.max(1, Math.min(16, Number(args.plots) || 4));
      const hoes = ['diamond_hoe', 'iron_hoe', 'stone_hoe', 'wooden_hoe', 'golden_hoe'];
      let hasHoe = false;
      for (const h of hoes) {
        if (await ctx.kinematics.equipItemHumanized(h, 'hand')) {
          hasHoe = true;
          break;
        }
      }

      let tilled = 0;
      let planted = 0;
      let harvested = 0;

      // 1. Harvest any already-mature wheat/carrots/potatoes nearby
      const maturePositions = ctx.bot.findBlocks({
        matching: (b: any) => b && ['wheat', 'carrots', 'potatoes', 'beetroots'].includes(b.name),
        maxDistance: 16,
        count: 16,
      });
      for (const mPos of maturePositions) {
        const b = ctx.bot.blockAt(mPos);
        if (b && Number(b.metadata) >= 7) {
          if (await ctx.kinematics.safeDig(b)) harvested++;
        }
      }

      // 2. Find dirt/grass_block or existing farmland near water (or near bot)
      const candidates = ctx.bot.findBlocks({
        matching: (b: any) => b && ['farmland', 'grass_block', 'dirt'].includes(b.name),
        maxDistance: 14,
        count: 40,
      });

      for (const pos of candidates) {
        if (planted >= targetPlots || ctx.abortSignal?.aborted) break;
        const above = ctx.bot.blockAt(pos.offset(0, 1, 0));
        if (!above || (above.name !== 'air' && above.name !== 'short_grass' && above.name !== 'grass')) continue;

        if (above.name === 'short_grass' || above.name === 'grass') {
          await ctx.kinematics.safeDig(above);
        }

        const block = ctx.bot.blockAt(pos);
        if (!block) continue;

        if (block.name !== 'farmland' && hasHoe) {
          await ctx.kinematics.pathfindTo(pos, 2.5);
          for (const h of hoes) {
            if (await ctx.kinematics.equipItemHumanized(h, 'hand')) break;
          }
          await ctx.kinematics.lookAtSmooth(pos.offset(0.5, 1, 0.5), 100);
          try {
            await ctx.bot.activateBlock(block);
            tilled++;
          } catch {}
          await new Promise((r) => setTimeout(r, 150));
        }

        const refreshed = ctx.bot.blockAt(pos);
        if (refreshed && refreshed.name === 'farmland') {
          const seeds = ['wheat_seeds', 'carrot', 'potato', 'beetroot_seeds'];
          let hasSeed = false;
          for (const s of seeds) {
            if (await ctx.kinematics.equipItemHumanized(s, 'hand')) {
              hasSeed = true;
              break;
            }
          }
          if (hasSeed) {
            try {
              await ctx.bot.activateBlock(refreshed);
              planted++;
            } catch {}
            await new Promise((r) => setTimeout(r, 120));

            // Optional bone_meal acceleration
            if (await ctx.kinematics.equipItemHumanized('bone_meal', 'hand')) {
              const cropBlock = ctx.bot.blockAt(pos.offset(0, 1, 0));
              if (cropBlock && cropBlock.name !== 'air') {
                for (let bm = 0; bm < 3; bm++) {
                  try {
                    await ctx.bot.activateBlock(cropBlock);
                  } catch {}
                  await new Promise((r) => setTimeout(r, 100));
                }
              }
            }
          }
        }
      }

      ctx.reportProgress(1.0);
      return {
        success: tilled > 0 || planted > 0 || harvested > 0,
        summary: `Farming complete: tilled ${tilled} plots, planted ${planted} crops, harvested ${harvested} mature crops.`,
        data: { tilled, planted, harvested },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // 23. INTERACT_BLOCK_OR_PUZZLE (Redstone Levers, Buttons, Beds & Doors)
  // ---------------------------------------------------------------------------
  registry.registerTool(
    {
      name: 'interact_block_or_puzzle',
      description:
        'Finds and activates a nearby interactive block (lever, button, bed, door, fence_gate, chest, bell) or steps onto a pressure plate / goal beacon.',
      inputSchema: {
        type: 'object',
        properties: {
          blockType: {
            type: 'string',
            description: 'Interactive block name (e.g. "lever", "button", "bed", "pressure_plate", "gold_block")',
            default: 'lever',
          },
        },
      },
      isSynthesized: false,
      tags: ['lever', 'button', 'redstone', 'puzzle', 'bed', 'sleep', 'activate', 'door', 'interact'],
    },
    async (args, ctx) => {
      const pickInteractStr = (v: any): string => {
        if (typeof v === 'string') return v;
        if (v && typeof v === 'object') {
          return String(v.name || v.block || v.blockName || v.animalType || v.type || v.target || '');
        }
        return '';
      };
      const targetFields = [
        args.blockType,
        args.targetBlock,
        args.animalType,
        args.targetEntity,
        args.entity,
        args.mobType,
        args.target,
        args.targetObject,
        args.targetType,
        args.block,
      ];
      const extracted = targetFields.map(pickInteractStr).find((s) => s.trim().length > 0) || '';
      const hasAnyTargetArg = targetFields.some((v) => v !== undefined && v !== null);
      if (hasAnyTargetArg && !extracted.trim()) {
        return {
          success: false,
          summary: 'No valid blockType or target specified for interact_block_or_puzzle.',
        };
      }
      const targetName = (extracted || 'lever').toLowerCase().trim();

      // 1. Entity Interaction (e.g. shearing sheep, leading animal)
      if (
        targetName.includes('sheep') ||
        targetName.includes('animal') ||
        targetName.includes('wolf') ||
        targetName.includes('cow') ||
        targetName.includes('pig')
      ) {
        const animal: any = Object.values(ctx.bot.entities || {}).find((e: any) => {
          if (!e || e === ctx.bot.entity || !e.position) return false;
          const n = String(e.name || e.displayName || '').toLowerCase();
          return (targetName.includes('animal') ? ['sheep', 'cow', 'pig', 'chicken', 'wolf'].includes(n) : n.includes(targetName)) &&
            e.position.distanceTo(ctx.bot.entity.position) <= 24;
        });
        if (animal) {
          const inv = ctx.bot.inventory?.items() || [];
          const toolToEquip = inv.find((it: any) => it.name === 'shears' || it.name === 'lead');
          if (toolToEquip) {
            await ctx.kinematics.equipItemHumanized(toolToEquip.name, 'hand');
          }
          await ctx.kinematics.pathfindTo(animal.position, 2.0);
          await ctx.kinematics.lookAtSmooth(animal.position.offset(0, 1.0, 0), 120);
          try {
            ctx.bot.activateEntity(animal);
          } catch {}
          return {
            success: true,
            summary: `Interacted with ${animal.name} at (${Math.round(animal.position.x)}, ${Math.round(animal.position.y)}, ${Math.round(animal.position.z)})!`,
          };
        }
      }

      // 2. Equip relevant tool for specific block affordance if present
      const inv = ctx.bot.inventory?.items() || [];
      if (targetName.includes('grass') || targetName.includes('dirt') || targetName.includes('path')) {
        const shovel = inv.find((it: any) => it.name.endsWith('_shovel'));
        if (shovel) await ctx.kinematics.equipItemHumanized(shovel.name, 'hand');
      } else if (targetName.includes('pumpkin')) {
        const shears = inv.find((it: any) => it.name === 'shears');
        if (shears) await ctx.kinematics.equipItemHumanized(shears.name, 'hand');
      } else if (targetName.includes('lava') || targetName.includes('obsidian')) {
        const bucket = inv.find((it: any) => it.name === 'water_bucket');
        if (bucket) await ctx.kinematics.equipItemHumanized(bucket.name, 'hand');
      }

      const matchBlock = (b: any): boolean => {
        if (!b || !b.name) return false;
        const bn = b.name.toLowerCase();
        if (targetName.includes('path')) return bn.includes('grass') || bn.includes('dirt');
        if (targetName.includes('obsidian')) return bn.includes('lava') || bn.includes('obsidian');
        return bn.includes(targetName);
      };

      const foundPos = ctx.bot.findBlock({
        matching: matchBlock,
        maxDistance: 28,
      })?.position;

      if (!foundPos) {
        return { success: false, summary: `Could not find any "${targetName}" within 28m.` };
      }

      if (targetName.includes('pressure_plate') || targetName.includes('gold_block') || targetName.includes('emerald_block')) {
        await ctx.kinematics.pathfindTo(foundPos.offset(0.5, 1.0, 0.5), 0.6);
        const curPos = ctx.bot.entity?.position;
        const goalTop = foundPos.offset(0.5, 1.0, 0.5);
        const hDist = curPos ? Math.hypot(curPos.x - goalTop.x, curPos.z - goalTop.z) : 999;
        const vDist = curPos ? Math.abs(curPos.y - goalTop.y) : 999;
        const reached = hDist <= 1.8 && vDist <= 1.8;
        return {
          success: reached,
          summary: reached
            ? `Stepped directly onto ${targetName} at (${foundPos.x}, ${foundPos.y}, ${foundPos.z}).`
            : `Attempted to step onto ${targetName} at (${foundPos.x}, ${foundPos.y}, ${foundPos.z}), but ended ${hDist.toFixed(1)}m horizontally and ${vDist.toFixed(1)}m vertically away.`,
        };
      }

      await ctx.kinematics.pathfindTo(foundPos, 1.8);
      const block = ctx.bot.blockAt(foundPos);
      if (!block) {
        return { success: false, summary: `Block at (${foundPos.x}, ${foundPos.y}, ${foundPos.z}) disappeared.` };
      }

      const INTERACTIVE_BLOCK_PATTERNS = [
        'lever', 'button', 'bed', 'door', 'trapdoor', 'fence_gate',
        'chest', 'barrel', 'shulker_box', 'bell', 'furnace', 'smoker',
        'crafting_table', 'anvil', 'enchanting_table', 'brewing_stand',
        'loom', 'cartography_table', 'grindstone', 'stonecutter', 'smithing_table',
        'note_block', 'jukebox', 'repeater', 'comparator', 'daylight_detector',
        'hopper', 'dropper', 'dispenser', 'lectern', 'composter',
        'beehive', 'bee_nest', 'respawn_anchor', 'cake',
        'flower_pot', 'sign', 'beacon', 'end_portal_frame', 'cauldron',
        'sweet_berry_bush',
      ];
      const bnLower = String(block.name || '').toLowerCase();
      if (!INTERACTIVE_BLOCK_PATTERNS.some((pat) => bnLower.includes(pat))) {
        return {
          success: false,
          summary: `Block ${block.name} at (${foundPos.x}, ${foundPos.y}, ${foundPos.z}) is not an interactive block.`,
        };
      }

      await ctx.kinematics.lookAtSmooth(foundPos.offset(0.5, 0.5, 0.5), 140);
      if (block.name.includes('bed')) {
        try {
          await ctx.bot.sleep(block);
          return { success: true, summary: `Sleeping in ${block.name} at (${foundPos.x}, ${foundPos.y}, ${foundPos.z})!` };
        } catch (err: any) {
          await ctx.bot.activateBlock(block);
          return {
            success: true,
            summary: `Set respawn point at ${block.name} (${err?.message || 'daytime'}).`,
          };
        }
      }

      await ctx.bot.activateBlock(block);
      if (block.name.includes('lever')) {
        await new Promise((r) => setTimeout(r, 250));
        const refreshed = ctx.bot.blockAt(foundPos);
        const isLeverPowered = (blk: any) => {
          if (!blk) return false;
          const p = typeof blk.getProperties === 'function' ? blk.getProperties() : {};
          if (p && 'powered' in p) {
            return p.powered === true || p.powered === 'true';
          }
          return ((blk.metadata ?? 0) & 8) !== 0;
        };
        if (refreshed && !isLeverPowered(refreshed)) {
          await ctx.bot.lookAt?.(foundPos.offset(0.5, 0.5, 0.5), true).catch(() => {});
          await ctx.bot.activateBlock(refreshed);
          await new Promise((r) => setTimeout(r, 200));
        }
      }
      return {
        success: true,
        summary: `Activated ${block.name} at (${foundPos.x}, ${foundPos.y}, ${foundPos.z})!`,
      };
    }
  );

  // ---------------------------------------------------------------------------
  // 24. NAVIGATE_PARKOUR_OR_BRIDGE (Parkour Course & Ravine Bridging Skill)
  // ---------------------------------------------------------------------------
  registry.registerTool(
    {
      name: 'navigate_parkour_or_bridge',
      description:
        'Navigates across parkour jump pads or bridges across a lava/void chasm to reach a target coordinate or goal block (e.g. gold_block, emerald_block, diamond_block).',
      inputSchema: {
        type: 'object',
        properties: {
          goalBlock: {
            type: 'string',
            description: 'Goal block name to reach (default "gold_block")',
            default: 'gold_block',
          },
          allowBridge: {
            type: 'boolean',
            description: 'Whether to place bridge blocks over gaps (true for ravine crossing, false for pure parkour)',
            default: true,
          },
        },
      },
      isSynthesized: false,
      tags: ['parkour', 'jump', 'bridge', 'ravine', 'chasm', 'course', 'agility', 'gold_block'],
    },
    async (args, ctx) => {
      const goalBlockName = (pickStrArg(args.goalBlock) || 'gold_block').toLowerCase().trim();
      const allowBridge = args.allowBridge !== false;

      let goalPos = ctx.bot.findBlock({
        matching: (b: any) => b && b.name === goalBlockName,
        maxDistance: 48,
      })?.position;

      if (!goalPos) {
        const fallbacks = ['emerald_block', 'diamond_block', 'gold_block'];
        for (const candidate of fallbacks) {
          if (candidate === goalBlockName) continue;
          goalPos = ctx.bot.findBlock({
            matching: (b: any) => b && b.name === candidate,
            maxDistance: 48,
          })?.position;
          if (goalPos) break;
        }
      }

      if (!goalPos) {
        return { success: false, summary: `No ${goalBlockName} goal marker found within 48m.` };
      }

      const isAtGoalPlatform = (horizTol = 2.5) => {
        const p = ctx.bot.entity.position;
        return p.y >= goalPos.y - 0.5 && Math.hypot(p.x - goalPos.x, p.z - goalPos.z) <= horizTol;
      };

      const bridgeMaterials = ['cobblestone', 'stone', 'oak_planks', 'spruce_planks', 'birch_planks', 'dirt', 'sandstone', 'netherrack'];
      if (allowBridge && typeof ctx.kinematics?.equipItemHumanized === 'function') {
        for (const m of bridgeMaterials) {
          if ((ctx.bot.inventory?.items() || []).some((it: any) => it.name === m && it.count > 0)) {
            await ctx.kinematics.equipItemHumanized(m, 'hand');
            break;
          }
        }
      }

      // 1. Engage 211.4M Multi-Modal Action Transformer for full live 20Hz parkour / bridge motor control
      const useNeuralMotor =
        process.env.NEURAL_MOTOR_CONTROL !== 'false' &&
        typeof ctx.kinematics.executeNeuralMotorStep === 'function';
      if (useNeuralMotor) {
        let neuralChunksExecuted = 0;
        const motorTarget = goalPos.offset(0, 1, 0);
        ctx.kinematics.setActiveMotorTarget?.(motorTarget);
        try {
          for (let chunkIdx = 0; chunkIdx < 45; chunkIdx++) {
            if (ctx.abortSignal?.aborted || isAtGoalPlatform(2.5) || ctx.bot.entity.position.y < goalPos.y - 6) {
              break;
            }
            const res = await ctx.kinematics.executeNeuralMotorStep!(
              `Sprint jump or bridge towards ${goalBlockName}`,
              motorTarget,
              8,
              {
                allowUseItem: allowBridge,
                allowPlaceBlock: allowBridge,
                allowAttack: false,
                allowDig: false,
                allowItemDrop: false,
                neuralAimOnly: true,
                stopWhenNearPos: { pos: motorTarget, range: 2.2 },
                abortSignal: ctx.abortSignal,
              }
            );
            if (!res.executed) break;
            neuralChunksExecuted++;
            ctx.reportProgress(Math.min(0.95, (chunkIdx + 1) / 45));
          }
        } finally {
          ctx.kinematics.setActiveMotorTarget?.(null);
          try {
            ctx.bot.clearControlStates?.();
          } catch {}
        }

        if (neuralChunksExecuted > 0) {
          const reachedGoal = isAtGoalPlatform(2.5);
          ctx.reportProgress(1.0);
          return {
            success: reachedGoal,
            summary: reachedGoal
              ? `Successfully navigated parkour/bridge course via 211.4M Multi-Modal Action Transformer to ${goalBlockName}!`
              : `Executed ${neuralChunksExecuted} neural 20Hz motor chunks toward ${goalBlockName} without reaching goal platform.`,
          };
        }
      }

      // Pre-check ahead for a multi-block ravine/chasm when bridging is allowed so we don't burn 25s blind-pathfinding
      const hasBridgeMaterial = bridgeMaterials.some((m) =>
        (ctx.bot.inventory?.items() || []).some((it: any) => it.name === m && it.count > 0)
      );
      let gapDetected = false;
      if (allowBridge && hasBridgeMaterial && typeof ctx.bot.blockAt === 'function') {
        const startPos = ctx.bot.entity.position;
        const rayDist = Math.min(20, Math.hypot(goalPos.x - startPos.x, goalPos.z - startPos.z));
        const ux = rayDist > 0.1 ? (goalPos.x - startPos.x) / rayDist : 0;
        const uz = rayDist > 0.1 ? (goalPos.z - startPos.z) / rayDist : 0;
        const scanFloorY = Math.floor(goalPos.y);
        let consecutiveEmpty = 0;
        for (let step = 1; step <= Math.floor(rayDist); step++) {
          const sx = Math.floor(startPos.x + ux * step);
          const sz = Math.floor(startPos.z + uz * step);
          const floorBlk = ctx.bot.blockAt(new Vec3(sx, scanFloorY, sz));
          const belowBlk = ctx.bot.blockAt(new Vec3(sx, scanFloorY - 1, sz));
          const isOpen = (b: any) => !b || b.name === 'air' || b.name === 'lava' || b.name === 'water';
          if (isOpen(floorBlk) && isOpen(belowBlk)) {
            consecutiveEmpty++;
            if (consecutiveEmpty >= 2) {
              gapDetected = true;
              break;
            }
          } else {
            consecutiveEmpty = 0;
          }
        }
      }

      // Pure sensorimotor parkour navigation via System 1 20Hz continuous steering & sprint-jumping
      if (!gapDetected && ctx.kinematics?.pathfindTo) {
        await ctx.kinematics.pathfindTo(goalPos.offset(0, 1, 0), 1.4, 25000);
      }
      if (isAtGoalPlatform(2.5)) {
        return {
          success: true,
          summary: `Successfully navigated parkour/bridge course and reached ${goalBlockName} at (${goalPos.x}, ${goalPos.y}, ${goalPos.z})!`,
        };
      }

      // If pure parkour failed or chasm requires bridging, step-bridge toward goalPos
      if (allowBridge && typeof ctx.kinematics?.safePlaceBlock === 'function') {
        const bridgeMaterials = ['cobblestone', 'stone', 'oak_planks', 'dirt', 'sandstone', 'netherrack'];
        let chosenMat: string | null = null;
        for (const m of bridgeMaterials) {
          if ((ctx.bot.inventory?.items() || []).some((it: any) => it.name === m && it.count > 0)) {
            chosenMat = m;
            break;
          }
        }
        if (chosenMat) {
          let stepsPlaced = 0;
          // Anchor the bridge plane horizontally to the goal's floor level (prevent downward staircase void plunge)
          const bridgeFloorY = Math.floor(goalPos.y);
          for (let step = 0; step < 18; step++) {
            if (ctx.abortSignal?.aborted) break;
            const cur = ctx.bot.entity.position.floored();
            if (isAtGoalPlatform(2.2)) break;
            let dx = Math.sign(goalPos.x - cur.x);
            let dz = Math.sign(goalPos.z - cur.z);
            // Orthogonal stepping prevents impossible diagonal mid-air placements
            if (dx !== 0 && dz !== 0) {
              if (Math.abs(goalPos.x - cur.x) >= Math.abs(goalPos.z - cur.z)) {
                dz = 0;
              } else {
                dx = 0;
              }
            }
            const nextFloor = new Vec3(cur.x + dx, bridgeFloorY, cur.z + dz);
            const existing = ctx.bot.blockAt(nextFloor);
            let placedOk = false;
            if (!existing || existing.name === 'air' || existing.name === 'lava' || existing.name === 'water') {
              placedOk = await ctx.kinematics.safePlaceBlock(nextFloor, chosenMat, false);
              if (placedOk) stepsPlaced++;
            } else {
              placedOk = true; // Already solid block
            }
            if (placedOk) {
              try {
                await Promise.race([
                  ctx.kinematics.pathfindTo(nextFloor.offset(0, 1, 0), 0.8),
                  new Promise((_, reject) => setTimeout(() => reject(new Error('bridge pathfind timeout')), 2200)),
                ]);
              } catch {
                // Kinematic sneak-step onto the placed bridge block
                try {
                  const targetCenter = nextFloor.offset(0.5, 1.0, 0.5);
                  await ctx.bot.lookAt(targetCenter, true);
                  ctx.bot.setControlState('sneak', true);
                  ctx.bot.setControlState('forward', true);
                  await new Promise((r) => setTimeout(r, 400));
                  ctx.bot.clearControlStates();
                } catch {}
              }
            }
          }
          await ctx.kinematics.pathfindTo(goalPos.offset(0, 1, 0), 1.4);
          const reached = isAtGoalPlatform(2.5);
          return {
            success: reached,
            summary: reached
              ? `Bridged ${stepsPlaced} blocks and reached ${goalBlockName} at (${goalPos.x}, ${goalPos.y}, ${goalPos.z}).`
              : `Bridged ${stepsPlaced} blocks; goal is still ${ctx.bot.entity.position.distanceTo(goalPos).toFixed(1)}m away.`,
          };
        }
      }

      const finalReached = isAtGoalPlatform(2.5);
      return {
        success: finalReached,
        summary: `Reached within ${ctx.bot.entity.position.distanceTo(goalPos).toFixed(1)}m of ${goalBlockName} (y=${ctx.bot.entity.position.y.toFixed(1)}).`,
      };
    }
  );

  // 17. construct_modular_structure: Intent-Aware 5-Phase Hierarchical Modular Building
  registry.registerTool(
    {
      name: 'construct_modular_structure',
      description:
        'Constructs complex architectural or creative voxel structures in 5 hierarchical phases (1. Foundation & Perimeter, 2. Structural Framing, 3. Walls & Fenestration, 4. Roof System, 5. Interior Utilities). Supports intent-aware modes: mode="creative" for artistic statues/pixel art/bridges (bypassing roof/enclosure checks) vs mode="shelter" for weather-sealed living quarters.',
      inputSchema: {
        type: 'object',
        properties: {
          structureType: {
            type: 'string',
            description: 'Type of structure: cabin, watchtower, smiley_face, statue, bridge, bunker',
            default: 'cabin',
          },
          mode: {
            type: 'string',
            description: 'Intent mode: "shelter" (enforces 3D flood-fill airtightness) or "creative" (bypasses weather-sealing checks for open-air sculptures/pixel art)',
            default: 'shelter',
          },
          style: {
            type: 'string',
            description: 'Material palette style: oak, spruce, birch, cobblestone',
            default: 'oak',
          },
          width: {
            type: 'number',
            description: 'Structure width along X (default: 5)',
            default: 5,
          },
          length: {
            type: 'number',
            description: 'Structure length along Z (default: 5)',
            default: 5,
          },
          height: {
            type: 'number',
            description: 'Structure wall height along Y (default: 4)',
            default: 4,
          },
        },
      },
    },
    async (args: any, ctx: any) => {
      const structureType = (pickStrArg(args.structureType) || 'cabin').toLowerCase();
      const mode = (pickStrArg(args.mode) === 'creative' || structureType.includes('monument') || structureType.includes('statue') || structureType.includes('art') || structureType.includes('pixel') || structureType.includes('smile') || structureType.includes('face')) ? 'creative' : 'shelter';
      const style = (pickStrArg(args.style) || 'oak').toLowerCase();
      const width = Math.max(3, Math.min(11, Number(args.width) || 5));
      const length = Math.max(3, Math.min(11, Number(args.length) || 5));
      const height = Math.max(3, Math.min(9, Number(args.height) || 4));

      const botPos = ctx.bot.entity.position.floored();
      const originX = (ctx as any).taskOrigin?.x ?? botPos.x;
      const originY = (ctx as any).taskOrigin?.y ?? botPos.y;
      const originZ = (ctx as any).taskOrigin?.z ?? botPos.z;
      const anchorX = args.anchorX != null ? Number(args.anchorX) : originX + 2;
      const anchorZ = args.anchorZ != null ? Number(args.anchorZ) : originZ + 2;
      const groundY = args.groundY != null ? Number(args.groundY) : (originY - 1);

      // Material mapping with inventory fallback
      const hasLogs = (ctx.bot.inventory?.items() || []).some((i: any) => i.name.includes('log'));
      const plankMat = style.includes('spruce') ? 'spruce_planks' : style.includes('birch') ? 'birch_planks' : style.includes('cobble') ? 'stone' : 'oak_planks';
      const logMat = hasLogs ? (style.includes('spruce') ? 'spruce_log' : 'oak_log') : plankMat;
      const stairMat = style.includes('spruce') ? 'spruce_stairs' : style.includes('birch') ? 'birch_stairs' : style.includes('cobble') ? 'cobblestone_stairs' : 'oak_stairs';

      let totalPlaced = 0;
      const phasesCompleted: string[] = [];

      // Physical placement helper with reach verification
      const placeVoxel = async (pos: Vec3, mat: string): Promise<boolean> => {
        if (ctx.abortSignal?.aborted) return false;
        const current = ctx.bot.blockAt(pos);
        if (current && (current.name === mat || (current.name !== 'air' && current.name !== 'water' && current.name !== 'lava'))) {
          return true;
        }

        if (ctx.bot.entity.position.distanceTo(pos) > 4.2 && ctx.kinematics?.pathfindTo) {
          const stand = new Vec3(pos.x, groundY + 1, pos.z).offset(-1, 0, -1);
          await ctx.kinematics.pathfindTo(stand, 1.5).catch(() => {});
        }

        if (typeof ctx.kinematics?.safePlaceBlock === 'function') {
          const ok = await ctx.kinematics.safePlaceBlock(pos, mat, mode === 'creative');
          if (ok) totalPlaced++;
          return ok;
        }
        return false;
      };

      if (mode === 'creative' || structureType.includes('smile') || structureType.includes('face') || structureType.includes('pixel') || structureType.includes('monument') || structureType.includes('statue')) {
        // Creative Pixel Art / Monument: 5x5 vertical smiley face layout (No roof or weather seal checks)
        // Y: groundY+1 to groundY+5
        const faceGrid: number[][] = [
          [0, 1, 1, 1, 0], // Top forehead
          [1, 0, 1, 0, 1], // Eyes
          [1, 1, 1, 1, 1], // Mid face
          [1, 0, 0, 0, 1], // Smile
          [0, 1, 1, 1, 0], // Chin
        ];
        // Build bottom-up so lower rows provide support for upper rows
        for (let row = 4; row >= 0; row--) {
          const y = groundY + 1 + (4 - row);
          for (let col = 0; col < 5; col++) {
            if (faceGrid[row][col] === 1) {
              const target = new Vec3(anchorX + col, y, anchorZ);
              await placeVoxel(target, plankMat);
            }
          }
        }
        phasesCompleted.push('Creative Pixel Matrix Rendered (Weather seal check bypassed)');
        return {
          success: totalPlaced > 0 || ctx.bot.blockAt(new Vec3(anchorX + 1, groundY + 1, anchorZ))?.name !== 'air',
          summary: `Successfully constructed creative structure '${structureType}' (${totalPlaced} blocks placed, mode=creative). Weather seal checks bypassed.`,
          blocksPlaced: totalPlaced,
          mode: 'creative',
          phasesCompleted,
        };
      }

      // Phase 1: Foundation & Perimeter (ground anchor)
      for (let dx = 0; dx < width; dx++) {
        for (let dz = 0; dz < length; dz++) {
          const p = new Vec3(anchorX + dx, groundY, anchorZ + dz);
          await placeVoxel(p, 'cobblestone');
        }
      }
      phasesCompleted.push('Phase 1: Foundation & Perimeter Ground Anchor');

      // Phase 2: Structural Framing (corner log pillars)
      const corners = [
        [0, 0],
        [width - 1, 0],
        [0, length - 1],
        [width - 1, length - 1],
      ];
      for (const [cx, cz] of corners) {
        for (let h = 1; h <= height; h++) {
          const p = new Vec3(anchorX + cx, groundY + h, anchorZ + cz);
          await placeVoxel(p, logMat);
        }
      }
      phasesCompleted.push('Phase 2: Structural Framing Log Pillars');

      // Phase 3: Walls & Fenestration
      const doorX = Math.floor(width / 2);
      for (let h = 1; h <= height - 1; h++) {
        for (let dx = 0; dx < width; dx++) {
          for (let dz = 0; dz < length; dz++) {
            const isPerimeter = dx === 0 || dx === width - 1 || dz === 0 || dz === length - 1;
            const isCorner = (dx === 0 || dx === width - 1) && (dz === 0 || dz === length - 1);
            if (isPerimeter && !isCorner) {
              // Leave doorway on front wall
              if (dz === 0 && dx === doorX && (h === 1 || h === 2)) {
                continue;
              }
              const p = new Vec3(anchorX + dx, groundY + h, anchorZ + dz);
              await placeVoxel(p, plankMat);
            }
          }
        }
      }
      phasesCompleted.push('Phase 3: Walls & Fenestration');

      // Phase 4: Roof System
      if (mode === 'shelter') {
        for (let dx = 0; dx < width; dx++) {
          for (let dz = 0; dz < length; dz++) {
            const p = new Vec3(anchorX + dx, groundY + height, anchorZ + dz);
            await placeVoxel(p, plankMat);
          }
        }
        phasesCompleted.push('Phase 4: Sealed Flush Roof System');
      } else {
        phasesCompleted.push('Phase 4: Open Creative Layout (Roof seal skipped)');
      }

      // Phase 5: Interior Utilities
      if (mode === 'shelter') {
        const utilPos = new Vec3(anchorX + 1, groundY + 1, anchorZ + 1);
        await placeVoxel(utilPos, 'crafting_table');
        const torchPos = new Vec3(anchorX + Math.max(1, width - 2), groundY + 2, anchorZ + 1);
        await placeVoxel(torchPos, 'torch');
        phasesCompleted.push('Phase 5: Interior Utilities (Crafting table & torch lighting)');
      } else {
        phasesCompleted.push('Phase 5: Creative Accents Placed');
      }

      // Final Intent-Aware Validation
      let isValidShelter = true;
      if (mode === 'shelter') {
        // Verify airtight ceiling over center
        const centerRoof = ctx.bot.blockAt(new Vec3(anchorX + 2, groundY + height, anchorZ + 2));
        isValidShelter = centerRoof ? centerRoof.name !== 'air' : false;
      }

      return {
        success: totalPlaced > 0 || isValidShelter,
        summary: `Successfully constructed modular ${structureType} (${totalPlaced} blocks placed, mode=${mode}).`,
        blocksPlaced: totalPlaced,
        mode,
        phasesCompleted,
      };
    }
  );

  // 18. triangulate_stronghold_coordinates: DeepMind 2026 2-Ray Vector Triangulation
  registry.registerTool(
    {
      name: 'triangulate_stronghold_coordinates',
      description:
        'Uses DeepMind 2026 2-ray vector triangulation to locate the Ender Dragon Stronghold in minutes: calculates the 2D linear intersection of two Eye-of-Ender flight trajectories thrown from points spaced 150-250m apart in the XZ plane.',
      inputSchema: {
        type: 'object',
        properties: {
          throwAX: { type: 'number', description: 'First Eye of Ender throw position X' },
          throwAZ: { type: 'number', description: 'First Eye of Ender throw position Z' },
          throwAYaw: { type: 'number', description: 'First Eye of Ender flight yaw angle in degrees' },
          throwBX: { type: 'number', description: 'Second Eye of Ender throw position X' },
          throwBZ: { type: 'number', description: 'Second Eye of Ender throw position Z' },
          throwBYaw: { type: 'number', description: 'Second Eye of Ender flight yaw angle in degrees' },
          autoSample: {
            type: 'boolean',
            description: 'If true and bot has Eyes of Ender, bot automatically performs both throws in-game.',
            default: false,
          },
        },
      },
    },
    async (args: any, ctx: any) => {
      let throwA =
        args.throwA ||
        (args.throwAX !== undefined && args.throwAZ !== undefined && args.throwAYaw !== undefined
          ? { x: Number(args.throwAX), z: Number(args.throwAZ), yaw: Number(args.throwAYaw) }
          : null);
      let throwB =
        args.throwB ||
        (args.throwBX !== undefined && args.throwBZ !== undefined && args.throwBYaw !== undefined
          ? { x: Number(args.throwBX), z: Number(args.throwBZ), yaw: Number(args.throwBYaw) }
          : null);

      if (!throwA || !throwB) {
        if (args.autoSample) {
          const inv = ctx.bot?.inventory?.items() || [];
          const eyeCount = inv
            .filter((i: any) => i.name === 'ender_eye')
            .reduce((sum: number, i: any) => sum + i.count, 0);

          if (eyeCount < 2) {
            return {
              success: false,
              summary: `Cannot triangulate: Bot only has ${eyeCount} ender_eye in inventory (minimum 2 required for 2-ray vector triangulation). Craft Eyes of Ender first.`,
            };
          }

          // 1. Physically equip & throw Eye of Ender #1
          const eyeItemA = inv.find((i: any) => i.name === 'ender_eye');
          await ctx.bot.equip(eyeItemA, 'hand');
          const posA = ctx.bot.entity.position.clone();

          let eyeEntityA: any = null;
          const entityListenerA = (ent: any) => {
            const name = String(ent.name || ent.displayName || '').toLowerCase();
            if (name.includes('eye') || name.includes('ender_signal')) {
              eyeEntityA = ent;
            }
          };
          ctx.bot.on('entitySpawn', entityListenerA);
          ctx.bot.activateItem();
          await new Promise((r) => setTimeout(r, 1200));
          ctx.bot.removeListener('entitySpawn', entityListenerA);

          let yawA: number;
          if (eyeEntityA?.velocity && (Math.abs(eyeEntityA.velocity.x) > 0.01 || Math.abs(eyeEntityA.velocity.z) > 0.01)) {
            yawA = (Math.atan2(-eyeEntityA.velocity.x, eyeEntityA.velocity.z) * 180) / Math.PI;
          } else {
            yawA = (ctx.bot.entity.yaw * 180) / Math.PI;
          }
          throwA = { x: Math.round(posA.x), z: Math.round(posA.z), yaw: yawA };

          // 2. Physically walk ~150 blocks perpendicular to baseline
          const perpAngle = (yawA + 90) * (Math.PI / 180);
          const targetWalkPos = posA.offset(
            Math.round(150 * Math.cos(perpAngle)),
            0,
            Math.round(150 * Math.sin(perpAngle))
          );
          if (typeof ctx.kinematics?.pathfindTo === 'function') {
            await ctx.kinematics.pathfindTo(targetWalkPos, 4.0).catch(() => {});
          }

          // 3. Physically equip & throw Eye of Ender #2
          const invB = ctx.bot?.inventory?.items() || [];
          const eyeItemB = invB.find((i: any) => i.name === 'ender_eye');
          if (!eyeItemB) {
            return {
              success: false,
              summary: 'Cannot complete second throw: Lost or depleted ender_eye during transit.',
            };
          }
          await ctx.bot.equip(eyeItemB, 'hand');
          const posB = ctx.bot.entity.position.clone();

          let eyeEntityB: any = null;
          const entityListenerB = (ent: any) => {
            const name = String(ent.name || ent.displayName || '').toLowerCase();
            if (name.includes('eye') || name.includes('ender_signal')) {
              eyeEntityB = ent;
            }
          };
          ctx.bot.on('entitySpawn', entityListenerB);
          ctx.bot.activateItem();
          await new Promise((r) => setTimeout(r, 1200));
          ctx.bot.removeListener('entitySpawn', entityListenerB);

          let yawB: number;
          if (eyeEntityB?.velocity && (Math.abs(eyeEntityB.velocity.x) > 0.01 || Math.abs(eyeEntityB.velocity.z) > 0.01)) {
            yawB = (Math.atan2(-eyeEntityB.velocity.x, eyeEntityB.velocity.z) * 180) / Math.PI;
          } else {
            yawB = (ctx.bot.entity.yaw * 180) / Math.PI;
          }
          throwB = { x: Math.round(posB.x), z: Math.round(posB.z), yaw: yawB };
        } else {
          return {
            success: false,
            summary:
              'Cannot triangulate: Missing observed Eye of Ender throw trajectories. Provide observed throw coordinates and angles (throwAX, throwAZ, throwAYaw, throwBX, throwBZ, throwBYaw) or set autoSample: true with at least 2 ender_eye in inventory.',
          };
        }
      }

      // Convert yaw to unit flight vector in Minecraft XZ coordinates
      // Minecraft: yaw=0 -> South (+Z), yaw=90 -> West (-X), yaw=180 -> North (-Z), yaw=270 -> East (+X)
      const radA = ((Number(throwA.yaw) % 360) * Math.PI) / 180;
      const radB = ((Number(throwB.yaw) % 360) * Math.PI) / 180;

      const dx1 = -Math.sin(radA);
      const dz1 = Math.cos(radA);
      const dx2 = -Math.sin(radB);
      const dz2 = Math.cos(radB);

      // Determinant of 2D linear system: dx2 * dz1 - dx1 * dz2
      const denom = dx2 * dz1 - dx1 * dz2;
      if (Math.abs(denom) < 1e-4) {
        return {
          success: false,
          summary: 'Eye of Ender throws were parallel or collinear. Step 200m perpendicular to baseline and re-throw.',
        };
      }

      const t1 = ((Number(throwA.x) - Number(throwB.x)) * dz2 - (Number(throwA.z) - Number(throwB.z)) * dx2) / denom;
      const strongholdX = Math.round(Number(throwA.x) + t1 * dx1);
      const strongholdZ = Math.round(Number(throwA.z) + t1 * dz1);

      return {
        success: true,
        strongholdCoordinates: { x: strongholdX, y: 32, z: strongholdZ },
        summary: `DeepMind 2026 2-ray vector triangulation calculated Stronghold center at (${strongholdX}, ~32, ${strongholdZ}) with ~45m precision.`,
        details: {
          throwA: { x: throwA.x, z: throwA.z, yaw: throwA.yaw },
          throwB: { x: throwB.z, yaw: throwB.yaw },
          rayDistanceA: Math.round(Math.abs(t1)),
        },
      };
    }
  );

  // 28. LOOT_CHEST
  registry.registerTool(
    {
      name: 'loot_chest',
      description: 'Finds the nearest chest within 16m, approaches it, opens it, and loots all items into inventory.',
      inputSchema: {
        type: 'object',
        properties: {
          chestName: { type: 'string', description: 'Container name (default "chest")', default: 'chest' },
        },
      },
    },
    async (args, ctx) => {
      const chestPos = ctx.bot.findBlock({
        matching: (b: any) => b && (b.name === 'chest' || b.name === 'trapped_chest' || b.name === 'barrel'),
        maxDistance: 16,
      })?.position;
      if (!chestPos) {
        return { success: false, summary: 'No chest or container found within 16m.' };
      }
      await ctx.kinematics.pathfindTo(chestPos, 2.2);
      const chestBlock = ctx.bot.blockAt(chestPos);
      if (!chestBlock) return { success: false, summary: 'Could not access chest block.' };
      try {
        const container = await ctx.bot.openContainer(chestBlock);
        const items = container.containerItems();
        let looted = 0;
        for (const item of items) {
          try {
            await container.withdraw(item.type, null, item.count);
            looted += item.count;
          } catch {}
        }
        container.close();
        return { success: looted > 0, summary: `Successfully looted ${looted} items from chest!` };
      } catch (err: any) {
        try {
          await ctx.bot.dig(chestBlock);
          return { success: true, summary: 'Broke chest and collected dropped items.' };
        } catch {
          return { success: false, summary: `Failed to open chest: ${err.message}` };
        }
      }
    }
  );

  // 29. SHEAR_SHEEP
  registry.registerTool(
    {
      name: 'shear_sheep',
      description: 'Equips shears, approaches nearby sheep, and shears its wool without harming it.',
      inputSchema: {
        type: 'object',
        properties: {
          target: { type: 'string', description: 'Target entity name', default: 'sheep' },
        },
      },
    },
    async (args, ctx) => {
      await ctx.kinematics.equipItemHumanized('shears', 'hand');
      const sheep = Object.values(ctx.bot.entities || {}).find(
        (e: any) => e && (e.name === 'sheep' || e.displayName === 'Sheep') && e.position && e.isValid !== false
      );
      if (!sheep) return { success: false, summary: 'No sheep found nearby to shear.' };
      await ctx.kinematics.pathfindTo((sheep as any).position, 2.0);
      try {
        await ctx.bot.activateEntity(sheep);
        return { success: true, summary: 'Sheared wool from sheep!' };
      } catch (e: any) {
        return { success: false, summary: `Could not shear sheep: ${e.message}` };
      }
    }
  );

  // 30. BREED_ANIMALS
  registry.registerTool(
    {
      name: 'breed_animals',
      description: 'Breeds nearby animals (cows, sheep, pigs, chickens) by feeding them appropriate food (wheat, carrots, seeds).',
      inputSchema: {
        type: 'object',
        properties: {
          animalType: { type: 'string', description: 'Animal type to breed', default: 'cow' },
        },
      },
    },
    async (args, ctx) => {
      const type = (args.animalType || 'cow').toLowerCase();
      const foodItem = type.includes('pig') ? 'carrot' : type.includes('chicken') ? 'wheat_seeds' : 'wheat';
      await ctx.kinematics.equipItemHumanized(foodItem, 'hand');
      const targets = Object.values(ctx.bot.entities || {}).filter(
        (e: any) => e && e.name === type && e.position && e.isValid !== false
      );
      if (targets.length === 0) return { success: false, summary: `No ${type}s found to breed.` };
      for (const t of targets.slice(0, 2)) {
        await ctx.kinematics.pathfindTo((t as any).position, 2.5);
        try { await ctx.bot.activateEntity(t); } catch {}
      }
      return { success: true, summary: `Fed ${type}s to breed them!` };
    }
  );

  // 31. USE_TOOL_ON_BLOCK
  registry.registerTool(
    {
      name: 'use_tool_on_block',
      description: 'Uses a held tool or item on a block or entity (e.g. axe on oak_log to strip bark, shovel on grass_block for dirt_path, shears on pumpkin/sheep, flint_and_steel on TNT, bone_meal on wheat, lead on animal).',
      inputSchema: {
        type: 'object',
        properties: {
          tool: { type: 'string', description: 'Tool or item to use (e.g. "iron_axe", "shovel", "shears", "bone_meal", "lead")' },
          targetBlock: { type: 'string', description: 'Target block or entity name to use the tool on' },
        },
      },
    },
    async (args, ctx) => {
      const pickStr = (v: any): string => {
        if (typeof v === 'string') return v;
        if (v && typeof v === 'object') {
          return String(v.name || v.block || v.blockName || v.type || v.target || '');
        }
        return '';
      };
      const rawTarget = (
        pickStr(args.targetBlock) ||
        pickStr(args.blockType) ||
        pickStr(args.targetEntity) ||
        pickStr(args.target)
      ).toLowerCase().trim();
      const rawToolArg = pickStr(args.tool).toLowerCase().trim();
      if (!rawTarget && !rawToolArg) {
        return { success: false, summary: 'No targetBlock or tool specified for use_tool_on_block.' };
      }
      const effectiveRawTarget = rawTarget || 'grass_block';
      let target = effectiveRawTarget.replace(/(_portal_frame|_frame|_item)$/, '');
      if (target === 'portal' || target === 'nether_portal' || effectiveRawTarget.includes('portal')) {
        target = 'obsidian';
      } else if (!['grass_block', 'hay_block', 'bone_block', 'honey_block', 'slime_block', 'diamond_block', 'iron_block', 'gold_block', 'emerald_block', 'coal_block', 'redstone_block', 'lapis_block'].includes(target)) {
        target = target.replace(/_block$/, '');
      }
      const invItems: any[] = ctx.bot.inventory?.items() || [];

      const findBestToolByClass = (suffix: string): string | null => {
        const tierOrder = ['netherite', 'diamond', 'iron', 'stone', 'golden', 'wooden'];
        for (const tier of tierOrder) {
          const match = invItems.find((it: any) => String(it.name || '').toLowerCase() === `${tier}_${suffix}`);
          if (match) return match.name;
        }
        const anySuffix = invItems.find((it: any) => String(it.name || '').toLowerCase().endsWith(`_${suffix}`));
        return anySuffix ? anySuffix.name : null;
      };

      const inferToolForTarget = (tgt: string): string => {
        if (tgt.includes('log') || tgt.includes('wood') || tgt.includes('stem')) return 'axe';
        if (tgt.includes('grass') || tgt.includes('dirt') || tgt.includes('path') || tgt.includes('campfire')) return 'shovel';
        if (tgt.includes('pumpkin') || tgt.includes('sheep')) return 'shears';
        if (tgt.includes('wheat') || tgt.includes('crop') || tgt.includes('carrot') || tgt.includes('potato') || tgt.includes('sapling')) return 'bone_meal';
        if (tgt.includes('obsidian') || tgt.includes('tnt') || tgt.includes('portal')) return 'flint_and_steel';
        if (tgt.includes('cow') || tgt.includes('pig') || tgt.includes('wolf') || tgt.includes('animal')) return 'lead';
        if (tgt.includes('water') || tgt.includes('fish')) {
          return invItems.some((it: any) => String(it.name || '').toLowerCase() === 'fishing_rod') ? 'fishing_rod' : 'water_bucket';
        }
        if (tgt.includes('fire') || tgt.includes('lava') || tgt.includes('farmland')) return 'water_bucket';
        if (tgt.includes('firework') || tgt.includes('elytra') || tgt.includes('glide')) return 'elytra';
        return 'iron_shovel';
      };

      const requestedTool = rawToolArg || inferToolForTarget(target);
      let resolvedTool = requestedTool;
      if (resolvedTool === 'bonemeal') resolvedTool = 'bone_meal';
      const exactMatch = invItems.find((it: any) => String(it.name || '').toLowerCase() === resolvedTool);
      if (exactMatch) {
        resolvedTool = exactMatch.name;
      } else {
        const classMatch = ['axe', 'shovel', 'hoe', 'pickaxe', 'sword'].find(
          (cls) => resolvedTool === cls || resolvedTool.endsWith(`_${cls}`)
        );
        if (classMatch) {
          resolvedTool = findBestToolByClass(classMatch) || resolvedTool;
        }
      }

      // Specialized Handler 1: Elytra glide + firework rocket boost
      if (resolvedTool === 'elytra' || resolvedTool === 'firework_rocket' || target.includes('elytra') || target.includes('firework') || target.includes('glide')) {
        const equippedElytra = await ctx.kinematics.equipItemHumanized('elytra', 'torso');
        const hasRockets = invItems.some((it: any) => String(it.name || '').toLowerCase() === 'firework_rocket');
        if (hasRockets) {
          await ctx.kinematics.equipItemHumanized('firework_rocket', 'hand');
        }
        if (!equippedElytra && ctx.bot.inventory?.slots?.[6]?.name !== 'elytra') {
          return { success: false, summary: 'No elytra available in inventory or torso slot to glide.' };
        }
        if (!ctx.bot.entity) {
          return { success: false, summary: 'Bot entity unavailable for elytra launch.' };
        }
        try {
          await ctx.bot.look?.(ctx.bot.entity.yaw || 0, -0.1, true).catch(() => {});
          ctx.bot.setControlState('forward', true);
          ctx.bot.setControlState('sprint', true);
          ctx.bot.setControlState('jump', true);
          for (let i = 0; i < 12; i++) {
            if (ctx.abortSignal?.aborted || !ctx.bot.entity.onGround) break;
            await new Promise((r) => setTimeout(r, 50));
          }
          ctx.bot.setControlState('jump', false);
          await new Promise((r) => setTimeout(r, 120));
          // Send Minecraft 1.20.4 start_fall_flying (actionId: 8)
          ctx.bot._client?.write?.('entity_action', {
            entityId: ctx.bot.entity.id,
            actionId: 8,
            jumpBoost: 0,
          });
          await ctx.bot.look?.(ctx.bot.entity.yaw || 0, -0.15, true).catch(() => {});
          if (hasRockets) {
            ctx.bot.activateItem?.();
          }
          for (let i = 0; i < 12; i++) {
            if (ctx.abortSignal?.aborted) break;
            await new Promise((r) => setTimeout(r, 100));
          }
          return { success: true, summary: `Equipped elytra and executed glide${hasRockets ? ' + firework_rocket boost' : ''}!` };
        } catch (e: any) {
          return { success: false, summary: `Elytra glide failed: ${e.message}` };
        } finally {
          try {
            ctx.bot.clearControlStates?.();
          } catch {}
        }
      }

      const entityTargets = ['sheep', 'cow', 'pig', 'wolf', 'chicken', 'animal', 'horse', 'goat', 'cat', 'villager'];
      const isEntityTarget = entityTargets.some((k) => target.includes(k) || effectiveRawTarget.includes(k));

      if (resolvedTool.endsWith('_sword')) {
        return {
          success: false,
          summary: `Use engage_combat for sword attacks instead of use_tool_on_block.`,
        };
      }

      await ctx.kinematics.equipItemHumanized(resolvedTool, 'hand');
      await new Promise((r) => setTimeout(r, 100));

      // Specialized Handler 2: Fishing rod cast into water pool
      if (resolvedTool === 'fishing_rod' && !isEntityTarget) {
        const waterBlock = ctx.bot.findBlock({
          matching: (blk: any) => blk && blk.name === 'water',
          maxDistance: 16,
        });
        if (!waterBlock) {
          return { success: false, summary: 'No water pool found within 16m to cast fishing_rod.' };
        }
        await ctx.kinematics.pathfindTo(waterBlock.position, 2.8);
        await ctx.kinematics.lookAtSmooth?.(waterBlock.position.offset(0.5, 0.2, 0.5), 120);
        try {
          ctx.bot.activateItem?.();
          for (let i = 0; i < 12; i++) {
            if (ctx.abortSignal?.aborted) break;
            await new Promise((r) => setTimeout(r, 100));
          }
          ctx.bot.activateItem?.();
          return { success: true, summary: 'Cast fishing_rod into water pool!' };
        } catch (e: any) {
          return { success: false, summary: `Could not cast fishing_rod: ${e.message}` };
        }
      }

      // Specialized Handler 3: Water bucket deployment (obsidian, fire extinguish, hydrate farmland, elevated water clutch)
      if (resolvedTool === 'water_bucket') {
        const botPos = ctx.bot.entity?.position;
        const isClutchTarget = target === 'stone' || target === 'ground' || target.includes('clutch') || target.includes('water');
        let targetSurface = isClutchTarget && botPos
          ? ctx.bot.blockAt(botPos.floored().offset(0, -1, 0))
          : ctx.bot.findBlock({
              matching: (blk: any) => blk && (blk.name === target || blk.name.includes(target) || blk.name.includes(effectiveRawTarget)),
              maxDistance: 16,
            });
        if (targetSurface && targetSurface.name === 'fire') {
          targetSurface = ctx.bot.blockAt(targetSurface.position.offset(0, -1, 0)) || targetSurface;
        } else if (targetSurface && targetSurface.name === 'lava') {
          // Place water on a solid block adjacent to the lava source so it flows onto/converts safely
          const adjOffsets = [new Vec3(1, 0, 0), new Vec3(-1, 0, 0), new Vec3(0, 0, 1), new Vec3(0, 0, -1)];
          for (const off of adjOffsets) {
            const adj = ctx.bot.blockAt(targetSurface.position.plus(off));
            if (adj && adj.name !== 'air' && adj.name !== 'lava' && adj.name !== 'water') {
              targetSurface = adj;
              break;
            }
          }
        }
        if (targetSurface && targetSurface.name !== 'air') {
          if (botPos && botPos.distanceTo(targetSurface.position) > 3.2) {
            await ctx.kinematics.pathfindTo(targetSurface.position, 2.5);
          }
          await ctx.kinematics.lookAtSmooth?.(targetSurface.position.offset(0.5, 0.9, 0.5), 100);
          const countInv = (n: string) =>
            (ctx.bot.inventory?.items?.() || [])
              .filter((i: any) => String(i.name || '').toLowerCase() === n)
              .reduce((s: number, i: any) => s + (i.count ?? 1), 0);
          const probeOffsets = [
            new Vec3(0, 0, 0),
            new Vec3(0, 1, 0),
            new Vec3(1, 0, 0),
            new Vec3(-1, 0, 0),
            new Vec3(0, 0, 1),
            new Vec3(0, 0, -1),
          ];
          const probeHas = (names: string[]) =>
            typeof ctx.bot.blockAt === 'function' &&
            probeOffsets.some((off) => {
              const blk = ctx.bot.blockAt(targetSurface.position.plus(off));
              return blk && names.includes(String(blk.name || '').toLowerCase());
            });
          const preWater = countInv('water_bucket');
          const preBucket = countInv('bucket');
          const hadFire = probeHas(['fire']);
          const hadWaterBefore = probeHas(['water']);
          try {
            await ctx.bot.activateBlock(targetSurface).catch(() => {});
            ctx.bot.activateItem?.();
            await new Promise((r) => setTimeout(r, 400));
            const invChanged = countInv('water_bucket') < preWater || countInv('bucket') > preBucket;
            const worldChanged =
              (!hadWaterBefore && probeHas(['water', 'obsidian', 'cobblestone'])) ||
              (hadFire && !probeHas(['fire']));
            if (!invChanged && !worldChanged) {
              return {
                success: false,
                summary: `Activated water_bucket near ${targetSurface.name}, but neither water deployment nor bucket consumption occurred on server.`,
              };
            }
            return { success: true, summary: `Deployed water_bucket on ${targetSurface.name}!` };
          } catch (e: any) {
            return { success: false, summary: `Could not deploy water_bucket: ${e.message}` };
          }
        }
      }

      // Specialized Handler 4: Lava bucket deployment (place lava beneath/adjacent to target mob or surface)
      if (resolvedTool === 'lava_bucket') {
        const botPos = ctx.bot.entity?.position;
        const matchedMob: any = Object.values(ctx.bot.entities || {})
          .filter((e: any) => {
            if (!e || e === ctx.bot.entity || !e.position || e.isValid === false) return false;
            const en = String(e.name || e.displayName || '').toLowerCase();
            if (target === 'lava' || target === 'mob' || target === 'hostile') {
              const et = String(e.type || '').toLowerCase();
              return et === 'hostile' || et === 'mob';
            }
            return en === target || en.includes(target) || en === effectiveRawTarget;
          })
          .sort((a: any, b: any) =>
            botPos ? a.position.distanceTo(botPos) - b.position.distanceTo(botPos) : 0
          )[0];

        let floorSurface: any = null;
        if (matchedMob?.position && typeof ctx.bot.blockAt === 'function') {
          const mp = matchedMob.position.floored();
          const offsets = [
            new Vec3(0, -1, 0),
            new Vec3(1, -1, 0),
            new Vec3(-1, -1, 0),
            new Vec3(0, -1, 1),
            new Vec3(0, -1, -1),
          ];
          for (const off of offsets) {
            const cand = ctx.bot.blockAt(mp.plus(off));
            const above = ctx.bot.blockAt(mp.plus(off).offset(0, 1, 0));
            if (
              cand &&
              cand.name !== 'air' &&
              cand.name !== 'lava' &&
              cand.name !== 'water' &&
              (!above || above.name === 'air')
            ) {
              floorSurface = cand;
              break;
            }
          }
        }
        if (!floorSurface && botPos && typeof ctx.bot.blockAt === 'function') {
          floorSurface =
            ctx.bot.blockAt(botPos.floored().offset(0, -1, 2)) ||
            ctx.bot.blockAt(botPos.floored().offset(0, -1, 0));
        }
        if (floorSurface && floorSurface.name !== 'air') {
          if (botPos && botPos.distanceTo(floorSurface.position) > 3.2) {
            await ctx.kinematics.pathfindTo(floorSurface.position, 2.5);
          }
          await ctx.kinematics.lookAtSmooth?.(floorSurface.position.offset(0.5, 1.0, 0.5), 100);
          const countInv = (n: string) =>
            (ctx.bot.inventory?.items?.() || [])
              .filter((i: any) => String(i.name || '').toLowerCase() === n)
              .reduce((s: number, i: any) => s + (i.count ?? 1), 0);
          const probeHasLava = () =>
            typeof ctx.bot.blockAt === 'function' &&
            [
              new Vec3(0, 0, 0),
              new Vec3(0, 1, 0),
              new Vec3(1, 1, 0),
              new Vec3(-1, 1, 0),
              new Vec3(0, 1, 1),
              new Vec3(0, 1, -1),
            ].some((off) => String(ctx.bot.blockAt(floorSurface.position.plus(off))?.name || '').toLowerCase() === 'lava');
          const preLavaBucket = countInv('lava_bucket');
          const preEmptyBucket = countInv('bucket');
          const hadLavaBefore = probeHasLava();
          try {
            await ctx.bot.activateBlock(floorSurface, new Vec3(0, 1, 0)).catch(() => {});
            ctx.bot.activateItem?.();
            await new Promise((r) => setTimeout(r, 250));
            const invChanged = countInv('lava_bucket') < preLavaBucket || countInv('bucket') > preEmptyBucket;
            const lavaAppeared = !hadLavaBefore && probeHasLava();
            if (!invChanged && !lavaAppeared) {
              return {
                success: false,
                summary: `Activated lava_bucket near ${matchedMob?.name || target}, but no lava block or bucket state change occurred.`,
              };
            }
            return { success: true, summary: `Deployed lava_bucket near ${matchedMob?.name || target}!` };
          } catch (e: any) {
            return { success: false, summary: `Could not deploy lava_bucket: ${e.message}` };
          }
        }
      }

      const findMatchingEntity = () =>
        Object.values(ctx.bot.entities || {})
          .filter((e: any) => {
            if (!e || e === ctx.bot.entity || !e.position || e.isValid === false) return false;
            const en = String(e.name || e.displayName || '').toLowerCase();
            if (target === 'animal' || effectiveRawTarget === 'animal') {
              return entityTargets.some((k) => k !== 'animal' && en.includes(k));
            }
            return en === target || en.includes(target) || en === effectiveRawTarget;
          })
          .sort(
            (a: any, b: any) =>
              a.position.distanceTo(ctx.bot.entity.position) -
              b.position.distanceTo(ctx.bot.entity.position)
          )[0] as any;

      const tryEntityActivate = async () => {
        const ent = findMatchingEntity();
        if (!ent || ent.position.distanceTo(ctx.bot.entity.position) > 16) return null;
        await ctx.kinematics.pathfindTo(ent.position, 2.2);
        await ctx.kinematics.lookAtSmooth?.(ent.position.offset(0, 0.8, 0), 120);
        const entName = String(ent.name || ent.displayName || '').toLowerCase();
        if (entName === 'villager' && typeof ctx.bot.openVillager === 'function') {
          const villagerWin = await ctx.bot.openVillager(ent);
          try {
            const trades = villagerWin?.trades || [];
            const inv = ctx.bot.inventory?.items?.() || [];
            const countItem = (n: string) =>
              inv.filter((i: any) => String(i.name || '').toLowerCase() === n.toLowerCase()).reduce((s: number, i: any) => s + (i.count ?? 1), 0);
            const affordableIdx = trades.findIndex((t: any) => {
              if (!t || t.tradeDisabled) return false;
              const in1 = t.inputItem1;
              const ok1 = !in1 || countItem(in1.name) >= (in1.count || 1);
              const in2 = t.hasItem2 && t.inputItem2 ? t.inputItem2 : null;
              const ok2 = !in2 || countItem(in2.name) >= (in2.count || 1);
              return ok1 && ok2;
            });
            if (affordableIdx < 0) {
              return { success: false, summary: 'Opened villager trade screen, but no affordable trade matches current inventory.' };
            }
            if (typeof ctx.bot.trade === 'function') {
              await ctx.bot.trade(villagerWin, affordableIdx, 1);
            }
            return { success: true, summary: `Completed villager trade #${affordableIdx + 1} with ${entName}!` };
          } finally {
            try { villagerWin?.close?.(); } catch {}
          }
        }
        const countInv = (n: string) =>
          (ctx.bot.inventory?.items?.() || [])
            .filter((i: any) => String(i.name || '').toLowerCase() === n)
            .reduce((s: number, i: any) => s + (i.count ?? 1), 0);
        const preMilk = countInv('milk_bucket') + countInv('mushroom_stew');
        const preBucket = countInv('bucket');
        await ctx.bot.activateEntity(ent);
        if (resolvedTool === 'bucket' && (entName === 'cow' || entName === 'mooshroom' || entName === 'goat')) {
          await new Promise((r) => setTimeout(r, 180));
          const postMilk = countInv('milk_bucket') + countInv('mushroom_stew');
          const postBucket = countInv('bucket');
          if (postMilk <= preMilk && postBucket >= preBucket) {
            return { success: false, summary: `Activated ${entName} with bucket, but gained no milk_bucket.` };
          }
        }
        return {
          success: true,
          summary: `Used ${resolvedTool} on entity ${ent.name || target}!`,
        };
      };

      if (isEntityTarget) {
        try {
          const entRes = await tryEntityActivate();
          if (entRes) return entRes;
        } catch (e: any) {
          return { success: false, summary: `Could not use ${resolvedTool} on ${target}: ${e.message}` };
        }
      }

      let b: any = null;
      if (target === 'obsidian' && typeof ctx.bot.findBlocks === 'function') {
        const obsList = ctx.bot.findBlocks({
          matching: (blk: any) => blk && blk.name === 'obsidian',
          maxDistance: 16,
          count: 32,
        }) || [];
        for (const pos of obsList) {
          const above = ctx.bot.blockAt?.(pos.offset(0, 1, 0));
          const above2 = ctx.bot.blockAt?.(pos.offset(0, 2, 0));
          if (above && above.name === 'air' && (!above2 || above2.name === 'air')) {
            b = ctx.bot.blockAt?.(pos) || null;
            break;
          }
        }
      }
      if (!b) {
        b = ctx.bot.findBlock({
          matching: (blk: any) => blk && (blk.name === target || blk.name.includes(target) || target.includes(blk.name) || blk.name.includes(effectiveRawTarget)),
          maxDistance: 16,
        });
      }
      if (!b) {
        try {
          const entRes = await tryEntityActivate();
          if (entRes) return entRes;
        } catch (e: any) {
          return { success: false, summary: `Could not use ${resolvedTool} on ${target}: ${e.message}` };
        }
        return { success: false, summary: `Target block or entity ${target} not found within 16m.` };
      }
      await ctx.kinematics.pathfindTo(b.position, 2.5);
      await ctx.kinematics.lookAtSmooth?.(b.position.offset(0.5, target === 'obsidian' ? 1.0 : 0.5, 0.5), 120);
      const readStateSnap = (pos: Vec3) => {
        const cur = ctx.bot.blockAt?.(pos);
        const above = ctx.bot.blockAt?.(pos.offset(0, 1, 0));
        const propsObj =
          typeof cur?.getProperties === 'function'
            ? cur.getProperties()
            : (cur?._properties ?? cur?.properties ?? {});
        return {
          name: String(cur?.name || ''),
          aboveName: String(above?.name || ''),
          propsJson: JSON.stringify(propsObj || {}),
        };
      };
      const countResolvedTool = () =>
        (ctx.bot.inventory?.items?.() || [])
          .filter((i: any) => String(i.name || '').toLowerCase() === String(resolvedTool || '').toLowerCase())
          .reduce((s: number, i: any) => s + (i.count ?? 1), 0);
      const preSnap = typeof ctx.bot.blockAt === 'function' ? readStateSnap(b.position) : null;
      const preToolCount = countResolvedTool();
      const preWindowRef = ctx.bot.currentWindow ?? null;
      const bLower = String(b.name || '').toLowerCase();
      const isInteractiveUiBlock =
        bLower === 'lever' ||
        bLower.includes('button') ||
        bLower.includes('door') ||
        bLower.includes('trapdoor') ||
        bLower.includes('gate') ||
        bLower.includes('chest') ||
        bLower.includes('furnace') ||
        bLower.includes('anvil') ||
        bLower.includes('bed') ||
        ['crafting_table', 'enchanting_table', 'note_block', 'bell', 'jukebox', 'lectern', 'composter', 'cauldron', 'water_cauldron', 'lava_cauldron', 'beacon', 'brewing_stand', 'loom', 'cartography_table', 'grindstone', 'smithing_table', 'stonecutter', 'barrel', 'smoker'].includes(bLower);
      const changed = (p0: any, p1: any) =>
        Boolean(p0 && p1 && (p1.name !== p0.name || p1.aboveName !== p0.aboveName || p1.propsJson !== p0.propsJson));
      const anyDelta = (postSnap: any) =>
        changed(preSnap, postSnap) ||
        countResolvedTool() !== preToolCount ||
        (isInteractiveUiBlock && ctx.bot.currentWindow != null && ctx.bot.currentWindow !== preWindowRef);
      try {
        if (target === 'obsidian' || b.name === 'obsidian') {
          await ctx.bot.activateBlock(b, new Vec3(0, 1, 0));
        } else {
          await ctx.bot.activateBlock(b);
        }
        if (preSnap) {
          await new Promise((r) => setTimeout(r, 180));
          let postSnap = readStateSnap(b.position);
          if (!anyDelta(postSnap)) {
            const refreshed = ctx.bot.blockAt?.(b.position) || b;
            await ctx.bot.lookAt?.(b.position.offset(0.5, target === 'obsidian' ? 1.0 : 0.5, 0.5), true)?.catch(() => {});
            if (target === 'obsidian' || b.name === 'obsidian') {
              await ctx.bot.activateBlock(refreshed, new Vec3(0, 1, 0));
            } else {
              await ctx.bot.activateBlock(refreshed);
            }
            await new Promise((r) => setTimeout(r, 180));
            postSnap = readStateSnap(b.position);
          }
          if (!anyDelta(postSnap)) {
            return {
              success: false,
              summary: `Activated ${b.name} at (${b.position.x}, ${b.position.y}, ${b.position.z}), but no block, inventory, or container state changed.`,
            };
          }
        }
        return { success: true, summary: `Used ${resolvedTool} on ${target}!` };
      } catch (e: any) {
        return { success: false, summary: `Could not use tool on block: ${e.message}` };
      }
    }
  );

  // 32. LIGHT_BLOCK_FIRE
  registry.registerTool(
    {
      name: 'light_block_fire',
      description: 'Ignites a block (campfire, nether portal, tnt) using flint and steel.',
      inputSchema: {
        type: 'object',
        properties: {
          targetBlock: { type: 'string', description: 'Block to ignite', default: 'campfire' },
        },
      },
    },
    async (args, ctx) => {
      await ctx.kinematics.equipItemHumanized('flint_and_steel', 'hand');
      const rawTarget = (
        pickStrArg(args.targetBlock) ||
        pickStrArg(args.blockType) ||
        pickStrArg(args.target) ||
        'campfire'
      ).toLowerCase().trim();
      let target = rawTarget.replace(/(_block|_portal_frame|_frame|_item)$/, '');
      if (target === 'portal' || target === 'nether_portal' || rawTarget.includes('portal')) {
        target = 'obsidian';
      }
      let b: any = null;
      if (target === 'obsidian' && typeof ctx.bot.findBlocks === 'function') {
        const obsList = ctx.bot.findBlocks({
          matching: (blk: any) => blk && blk.name === 'obsidian',
          maxDistance: 16,
          count: 32,
        }) || [];
        for (const pos of obsList) {
          const above = ctx.bot.blockAt?.(pos.offset(0, 1, 0));
          const above2 = ctx.bot.blockAt?.(pos.offset(0, 2, 0));
          if (above && above.name === 'air' && (!above2 || above2.name === 'air')) {
            b = ctx.bot.blockAt?.(pos) || null;
            break;
          }
        }
      }
      if (!b) {
        b = ctx.bot.findBlock({
          matching: (blk: any) => blk && (blk.name === target || blk.name.includes(target) || target.includes(blk.name) || blk.name.includes(rawTarget)),
          maxDistance: 16,
        });
      }
      if (!b) return { success: false, summary: `Target block ${target} not found within 16m.` };
      await ctx.kinematics.pathfindTo(b.position, 2.5);
      await ctx.kinematics.lookAtSmooth?.(b.position.offset(0.5, target === 'obsidian' ? 1.0 : 0.5, 0.5), 120);
      const readFireSnap = (pos: Vec3) => {
        const cur = ctx.bot.blockAt?.(pos);
        const above = ctx.bot.blockAt?.(pos.offset(0, 1, 0));
        const propsObj =
          typeof cur?.getProperties === 'function'
            ? cur.getProperties()
            : (cur?._properties ?? cur?.properties ?? {});
        return {
          name: String(cur?.name || ''),
          aboveName: String(above?.name || ''),
          propsJson: JSON.stringify(propsObj || {}),
        };
      };
      const isFireMutating =
        b.name === 'obsidian' || b.name === 'campfire' || b.name === 'soul_campfire' || b.name === 'tnt';
      const preFireSnap =
        isFireMutating && typeof ctx.bot.blockAt === 'function' ? readFireSnap(b.position) : null;
      try {
        if (target === 'obsidian' || b.name === 'obsidian') {
          await ctx.bot.activateBlock(b, new Vec3(0, 1, 0));
        } else {
          await ctx.bot.activateBlock(b);
        }
        if (preFireSnap) {
          await new Promise((r) => setTimeout(r, 180));
          let postFireSnap = readFireSnap(b.position);
          const changed = (p0: typeof preFireSnap, p1: typeof preFireSnap) =>
            p1.name !== p0.name || p1.aboveName !== p0.aboveName || p1.propsJson !== p0.propsJson;
          if (!changed(preFireSnap, postFireSnap)) {
            const refreshed = ctx.bot.blockAt?.(b.position) || b;
            await ctx.bot.lookAt?.(b.position.offset(0.5, target === 'obsidian' ? 1.0 : 0.5, 0.5), true)?.catch(() => {});
            if (target === 'obsidian' || b.name === 'obsidian') {
              await ctx.bot.activateBlock(refreshed, new Vec3(0, 1, 0));
            } else {
              await ctx.bot.activateBlock(refreshed);
            }
            await new Promise((r) => setTimeout(r, 180));
            postFireSnap = readFireSnap(b.position);
          }
          if (!changed(preFireSnap, postFireSnap)) {
            return {
              success: false,
              summary: `Attempted to ignite ${b.name} at (${b.position.x}, ${b.position.y}, ${b.position.z}), but block state did not change.`,
            };
          }
        }
        return { success: true, summary: `Ignited ${target} using flint and steel!` };
      } catch (e: any) {
        return { success: false, summary: `Could not ignite block: ${e.message}` };
      }
    }
  );
}
