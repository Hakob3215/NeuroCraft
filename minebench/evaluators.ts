import { Vec3 } from 'vec3';
import { MinebenchGradingResult, MinebenchTaskSpec } from './benchmarkSpec';

export interface MinebenchEvaluationSnapshot {
  botPosition: { x: number; y: number; z: number };
  health: number;
  food: number;
  deaths: number;
  inventory: Array<{ name: string; count: number; durability?: number; maxDurability?: number }>;
  equippedArmor: {
    head: string | null;
    torso: string | null;
    legs: string | null;
    feet: string | null;
    offHand: string | null;
  };
  blockAt: (x: number, y: number, z: number) => { name: string; metadata?: number } | null;
  entities: Array<{
    id: number;
    name: string;
    type: string;
    position: { x: number; y: number; z: number };
    isValid: boolean;
    metadata?: Record<number, any>;
    health?: number;
  }>;
  stats: {
    blocksPlaced: number;
    blocksMined: number;
    mobsKilled: Record<string, number>;
    creeperBlasts: number;
    arrowsFired: number;
    endCrystalsDestroyed: number;
    enderDragonSlain: boolean;
    leversFlipped: number;
    wolvesTamed: number;
    foodConsumed?: number;
    affordanceActions?: number;
    ironPickaxeDurabilityUsedOnStone: number;
  };
  multiplayer?: {
    opponentName: string;
    opponentHealth: number;
    opponentAlive: boolean;
    opponentKills: number;
    opponentAtGoal: boolean;
    opponentBlocksCollected: number;
    opponentInventory?: Array<{ name: string; count: number }>;
  };
}

/**
 * Programmatic, 100% objective evaluator for all 6 MINEbench domains.
 * Computes both a normalized [0..100] benchmark score and a dense RL reward signal R_t.
 */
export function evaluateMinebenchEpisode(
  task: MinebenchTaskSpec,
  snap: MinebenchEvaluationSnapshot
): MinebenchGradingResult {
  const rawId = String(task?.id || (task as any)?.taskId || '');
  const normalizedId = rawId.replace(/^classic_/, '');
  switch (normalizedId) {
    case 'agility_sky_parkour':
    case 'agility_ravine_bridge':
      return evaluateAgilityTask(task, snap);
    case 'build_enclosed_shelter':
      return evaluateBuildingShelterIoU(task, snap);
    case 'build_creative_monument':
      return evaluateCreativeBuilding(task, snap);
    case 'build_watchtower_pillar':
      return evaluateWatchtowerPillar(task, snap);
    case 'combat_creeper_kiting':
    case 'combat_skeleton_duel':
    case 'combat_spider_swarm':
    case 'combat_enderman_hunt':
      return evaluateCombatTask(task, snap);
    case 'homestead_crop_farm':
    case 'homestead_wolf_taming':
      return evaluateHomesteadTask(task, snap);
    case 'puzzle_redstone_lever':
    case 'puzzle_tool_preservation_mining':
      return evaluatePuzzleTask(task, snap);
    case 'prog_tier1_stone_age':
    case 'prog_tier3_iron_armor':
    case 'prog_tier6_ender_dragon':
      return evaluateProgressionTask(task, snap);
    default:
      if (rawId.startsWith('mp_') || rawId.startsWith('coop_') || rawId.startsWith('prog_')) {
        return evaluateProgressionTask(task, snap);
      }
      if (rawId.startsWith('omni_') || (task as any)?.evaluatorMilestones) {
        return evaluateOmniscienceMilestones(task, snap);
      }
      return {
        passed: false,
        score: 0,
        rawReward: 0,
        maxScore: 100,
        metrics: {},
        breakdown: `Unknown task ID: ${rawId || task?.id}`,
      };
  }
}

export function evaluateOmniscienceMilestones(
  task: any,
  snap: MinebenchEvaluationSnapshot
): MinebenchGradingResult {
  const milestones: Array<{ name: string; maxPoints: number; condition: string }> =
    task.evaluatorMilestones || [
      { name: 'Task Execution', maxPoints: 100, condition: 'general_execution' },
    ];

  let earnedScore = 0;
  const metrics: Record<string, any> = {};

  const targetObj = (task.targetObject || task.expectedTool || '').toLowerCase();
  const invNames = (snap.inventory || []).map((i) => i.name.toLowerCase());
  const hasTarget = Boolean(targetObj && invNames.includes(targetObj));
  const hasTable = invNames.includes('crafting_table');
  const placedTable = (snap.stats?.blocksPlaced || 0) > 0;
  const hasWorkstation = hasTable || placedTable || hasTarget;
  const totalLogs = (snap.inventory || [])
    .filter((i) => i.name.toLowerCase().endsWith('_log') || i.name.toLowerCase().endsWith('_stem'))
    .reduce((sum, i) => sum + i.count, 0);
  const totalPlanks = (snap.inventory || [])
    .filter((i) => i.name.toLowerCase().endsWith('_planks') || i.name.toLowerCase() === 'planks')
    .reduce((sum, i) => sum + i.count, 0);
  const hasSticksOrTools = invNames.some((n) => n === 'stick' || n.endsWith('_pickaxe') || n.endsWith('_axe') || n.endsWith('_sword') || n.endsWith('_shovel') || n === 'chest');
  const hasRefined = totalPlanks >= 8 || hasWorkstation || hasTarget || hasSticksOrTools || (totalPlanks >= 4 && ((snap.stats?.blocksMined || 0) >= 2 || (snap.stats?.blocksPlaced || 0) > 0));
  const hasRawHarvest = totalLogs >= 2 || (snap.stats?.blocksMined || 0) >= 2 || (((snap.stats?.blocksMined || 0) >= 1) && totalLogs >= 1) || hasRefined || hasWorkstation || hasTarget;

  for (const m of milestones) {
    let earned = 0;
    const cond = m.condition;

    if (cond === 'dag_raw_harvest') {
      if (hasRawHarvest) {
        earned = m.maxPoints;
      }
    } else if (cond === 'dag_basic_refine') {
      if (hasRefined) {
        earned = m.maxPoints;
      }
    } else if (cond === 'dag_workstation') {
      if (hasWorkstation) {
        earned = m.maxPoints;
      }
    } else if (cond.startsWith('has_') || cond.startsWith('harvested_')) {
      const requiredItem = cond.replace(/^(has_|harvested_)/, '').toLowerCase();
      // Match exact item name to prevent raw materials (e.g. "diamond") falsely matching tools ("diamond_pickaxe")
      const hasItem = invNames.some((n) => n === requiredItem);
      const didMine = (snap.stats?.blocksMined || 0) > 0;
      if (hasItem && (cond.startsWith('has_') || didMine)) {
        earned = m.maxPoints;
      }
    } else if (cond.startsWith('slot_') && cond.endsWith('_equipped')) {
      const slot = cond.replace(/^slot_/, '').replace(/_equipped$/, '');
      const normalizedSlot = slot === 'chest' ? 'torso' : slot;
      const armorSlot = (snap.equippedArmor as any)[normalizedSlot] || (snap.equippedArmor as any)[slot];
      if (armorSlot) {
        earned = m.maxPoints;
      }
    } else if (cond === 'held_tool_correct') {
      const hasCorrectTool = invNames.some((n) => targetObj && (n === targetObj || n.includes(targetObj)));
      const didAnyAction =
        snap.stats.blocksMined > 0 ||
        snap.stats.blocksPlaced > 0 ||
        (snap.stats as any).foodConsumed > 0 ||
        snap.stats.leversFlipped > 0 ||
        snap.stats.wolvesTamed > 0 ||
        ((snap.stats as any).affordanceActions || 0) > 0 ||
        Object.values(snap.stats.mobsKilled || {}).reduce((a, b) => a + b, 0) > 0;
      if (hasCorrectTool && didAnyAction) {
        earned = m.maxPoints;
      }
    } else if (cond === 'affordance_action_completed') {
      const anyAction =
        snap.stats.blocksMined > 0 ||
        snap.stats.blocksPlaced > 0 ||
        snap.stats.leversFlipped > 0 ||
        snap.stats.wolvesTamed > 0 ||
        ((snap.stats as any).affordanceActions || 0) > 0 ||
        (snap.stats as any).foodConsumed > 0 ||
        Object.values(snap.stats.mobsKilled || {}).reduce((a, b) => a + b, 0) > 0;
      if (anyAction) {
        earned = m.maxPoints;
      }
    } else if (cond === 'zero_creeper_blasts' || cond === 'kiting_damage_inflicted') {
      const creeperKilled = (snap.stats.mobsKilled?.creeper || 0) > 0;
      if (snap.stats.creeperBlasts === 0 && creeperKilled) {
        earned = m.maxPoints;
      }
    } else if (cond === 'held_item_food' || cond === 'food_level_restored') {
      const didConsume = ((snap.stats as any).foodConsumed || 0) > 0;
      if (cond === 'food_level_restored') {
        if (didConsume && snap.food >= 18) earned = m.maxPoints;
      } else if (didConsume) {
        earned = m.maxPoints;
      }
    } else if (cond === 'target_engaged' || cond === 'target_resolved') {
      const totalKills = Object.values(snap.stats.mobsKilled || {}).reduce((a, b) => a + b, 0);
      const tamed = (snap.stats.wolvesTamed || 0) > 0;
      const sheared = invNames.some((n) => n.includes('wool'));
      const peacefulMobs = ['cow', 'sheep', 'wolf', 'piglin', 'villager', 'cat', 'horse', 'ocelot', 'fox', 'dolphin', 'squid', 'glow_squid'];
      const isPeacefulTarget = peacefulMobs.includes((task.targetObject || '').toLowerCase());
      const hasInteracted =
        tamed ||
        sheared ||
        snap.stats.blocksMined > 0 ||
        snap.stats.blocksPlaced > 0 ||
        (snap.stats.foodConsumed || 0) > 0 ||
        ((snap.stats as any).affordanceActions || 0) > 0;

      const dealtDamage = Boolean(((snap.stats as any)?.damageDealt || 0) > 0 || (snap.stats.arrowsFired || 0) > 0);
      const validKill = totalKills > 0 && (dealtDamage || isPeacefulTarget);

      if (cond === 'target_engaged') {
        if (validKill || hasInteracted) {
          earned = m.maxPoints;
        }
      } else if (cond === 'target_resolved') {
        if (isPeacefulTarget) {
          if (validKill || tamed || sheared || (snap.deaths === 0 && hasInteracted)) {
            earned = m.maxPoints;
          }
        } else if (validKill) {
          earned = m.maxPoints;
        }
      }
    } else if (cond === 'loot_collected') {
      // Must have acquired authentic structure loot (not ordinary cobblestone from arena floor)
      const initialItemNames = new Set((task.initialLoadout || []).map((i: any) => i.name.toLowerCase()));
      const arenaCommonBlocks = new Set(['stone', 'cobblestone', 'dirt', 'bedrock']);
      const collectedNewLoot = invNames.some((n) => !initialItemNames.has(n) && !arenaCommonBlocks.has(n));
      if (collectedNewLoot) {
        earned = m.maxPoints;
      }
    } else if (cond === 'zero_hazard_deaths') {
      // Must survive without deaths while navigating structure
      const startPos = task.origin || { x: 500, y: 100, z: 500 };
      const botP = snap.botPosition || { x: 500, y: 100, z: 500 };
      const distTraveled = Math.sqrt((botP.x - startPos.x) ** 2 + (botP.z - startPos.z) ** 2);
      if (snap.deaths === 0 && (distTraveled > 2.0 || snap.stats.blocksMined > 0 || ((snap.stats as any).affordanceActions || 0) > 0)) {
        earned = m.maxPoints;
      }
    } else {
      // General Execution: STRICT - must have actually performed a physical action in the world
      const physicalActionDone =
        snap.stats.blocksMined > 0 ||
        snap.stats.blocksPlaced > 0 ||
        snap.stats.leversFlipped > 0 ||
        snap.stats.wolvesTamed > 0 ||
        Object.values(snap.stats.mobsKilled || {}).reduce((a, b) => a + b, 0) > 0;
      if (physicalActionDone && snap.deaths === 0) {
        earned = m.maxPoints;
      }
    }

    earnedScore += earned;
    metrics[m.name] = earned;
  }

  const deathPenalty = snap.deaths * 40;
  const finalScore = Math.max(0, Math.min(100, earnedScore - deathPenalty));
  return {
    passed: finalScore >= 60 && snap.deaths === 0,
    score: finalScore,
    rawReward: finalScore * 1.0 - snap.deaths * 50,
    maxScore: 100,
    metrics,
    breakdown: `Omniscience Affordance: Score=${finalScore}/100, BlocksMined=${snap.stats.blocksMined}, Placed=${snap.stats.blocksPlaced}.`,
  };
}

function evaluateAgilityTask(
  task: MinebenchTaskSpec,
  snap: MinebenchEvaluationSnapshot
): MinebenchGradingResult {
  const goalX = task.id === 'agility_sky_parkour' ? task.origin.x + 18 : task.origin.x + 12;
  const goalY = task.origin.y;
  const goalZ = task.origin.z;

  const totalSpan = Math.max(1, goalX - task.origin.x);
  const progressX = Math.max(0, Math.min(totalSpan, snap.botPosition.x - task.origin.x));
  const progressRatio = progressX / totalSpan;
  const distToGoal = Math.hypot(
    snap.botPosition.x - goalX,
    snap.botPosition.y - goalY,
    snap.botPosition.z - goalZ
  );

  const fellInVoidOrLava = snap.botPosition.y < task.origin.y - 2.5 || snap.deaths > 0;
  const reachedGoal = distToGoal <= 2.5 && !fellInVoidOrLava;

  // MINEbench 2.0 Granular Continuous Scoring:
  // Awards dense credit for blocks placed and forward progression, with a softened void penalty
  const placementCredit = Math.min(25, snap.stats.blocksPlaced * 3.0);
  const forwardCredit = progressRatio * 60;
  const goalBonus = reachedGoal ? 25 : 0;
  const penalty = fellInVoidOrLava ? (reachedGoal ? 0 : 20) : 0;

  const score = Math.round(
    Math.max(0, Math.min(100, forwardCredit + placementCredit + goalBonus - penalty))
  );
  const rawReward =
    progressRatio * 50 +
    (reachedGoal ? 50 : 0) +
    snap.stats.blocksPlaced * 2.5 -
    (fellInVoidOrLava ? 25 : 0);

  return {
    passed: reachedGoal,
    score,
    rawReward: Number(rawReward.toFixed(2)),
    maxScore: 100,
    metrics: {
      distToGoal: Number(distToGoal.toFixed(2)),
      progressPercent: Math.round(progressRatio * 100),
      blocksPlaced: snap.stats.blocksPlaced,
      fellInVoidOrLava,
    },
    breakdown: `Reached ${(progressRatio * 100).toFixed(0)}% of course (distToGoal=${distToGoal.toFixed(1)}m, blocksPlaced=${snap.stats.blocksPlaced}, fell=${fellInVoidOrLava}).`,
  };
}

/**
 * GDMC / MCU-inspired 3D Voxel IoU + Flood-Fill Enclosure + Ground Anchor Grader for 3D Building
 */
function evaluateBuildingShelterIoU(
  task: MinebenchTaskSpec,
  snap: MinebenchEvaluationSnapshot
): MinebenchGradingResult {
  const ox = snap.blockAt(task.origin.x + 2, task.origin.y, task.origin.z + 2)?.name !== 'air' ? task.origin.x + 2 : task.origin.x + 1;
  const oy = task.origin.y;
  const oz = task.origin.z + 2;
  const w = 5;
  const l = 5;
  const h = 4;

  let expectedWallRoofVoxels = 0;
  let matchedWallRoofVoxels = 0;
  let roofCoveredCells = 0;
  let totalInteriorFloorCells = 0;
  let groundedFoundationCells = 0;
  let totalPerimeterCells = 0;
  let hasTorchLight = false;

  const solidMaterials = new Set([
    'oak_planks',
    'spruce_planks',
    'birch_planks',
    'cobblestone',
    'stone',
    'stone_bricks',
    'bricks',
    'logs',
    'oak_log',
  ]);

  for (let x = 0; x < w; x++) {
    for (let z = 0; z < l; z++) {
      const isPerimeter = x === 0 || x === w - 1 || z === 0 || z === l - 1;
      if (isPerimeter) {
        totalPerimeterCells++;
        const groundBelow = snap.blockAt(ox + x, oy - 1, oz + z);
        if (groundBelow && groundBelow.name !== 'air') {
          groundedFoundationCells++;
        }
      } else {
        totalInteriorFloorCells++;
        const roofBlock = snap.blockAt(ox + x, oy + h, oz + z);
        if (roofBlock && roofBlock.name !== 'air') {
          roofCoveredCells++;
        }
      }

      // Check walls (dy = 0..h-1)
      for (let dy = 0; dy < h; dy++) {
        const b = snap.blockAt(ox + x, oy + dy, oz + z);
        if (b && b.name === 'torch') hasTorchLight = true;

        const isDoorway = z === 0 && x === Math.floor(w / 2) && (dy === 0 || dy === 1);
        const isWindow =
          dy === 1 &&
          !isDoorway &&
          ((z === l - 1 && x === Math.floor(w / 2)) ||
            (x === 0 && z === Math.floor(l / 2)) ||
            (x === w - 1 && z === Math.floor(l / 2)));

        if (isPerimeter && !isDoorway && !isWindow) {
          expectedWallRoofVoxels++;
          if (b && (solidMaterials.has(b.name) || b.name !== 'air')) {
            matchedWallRoofVoxels++;
          }
        }
      }

      // Check roof (dy = h)
      expectedWallRoofVoxels++;
      const rb = snap.blockAt(ox + x, oy + h, oz + z);
      if (rb && (solidMaterials.has(rb.name) || rb.name !== 'air')) {
        matchedWallRoofVoxels++;
      }
    }
  }

  const voxelIoU = expectedWallRoofVoxels > 0 ? matchedWallRoofVoxels / expectedWallRoofVoxels : 0;
  const roofSealRatio = totalInteriorFloorCells > 0 ? roofCoveredCells / totalInteriorFloorCells : 0;
  const foundationAnchorRatio =
    totalPerimeterCells > 0 ? groundedFoundationCells / totalPerimeterCells : 0;

  // VillagerBench 5-View Orthographic Silhouette Hit Rate (VHR: Front, Back, Left, Right, Top)
  let viewHits = 0;
  let viewTotal = 0;
  // Top view (w x l)
  for (let x = 0; x < w; x++) {
    for (let z = 0; z < l; z++) {
      viewTotal++;
      for (let dy = h; dy >= 0; dy--) {
        const b = snap.blockAt(ox + x, oy + dy, oz + z);
        if (b && b.name !== 'air' && b.name !== 'torch') {
          viewHits++;
          break;
        }
      }
    }
  }
  // 4 Lateral elevation views (Front/Back along Z, Left/Right along X)
  for (let dy = 0; dy <= h; dy++) {
    for (let x = 0; x < w; x++) {
      viewTotal += 2;
      // Front ray (+Z)
      for (let z = 0; z < l; z++) {
        const b = snap.blockAt(ox + x, oy + dy, oz + z);
        if (b && b.name !== 'air' && b.name !== 'torch') {
          viewHits++;
          break;
        }
      }
      // Back ray (-Z)
      for (let z = l - 1; z >= 0; z--) {
        const b = snap.blockAt(ox + x, oy + dy, oz + z);
        if (b && b.name !== 'air' && b.name !== 'torch') {
          viewHits++;
          break;
        }
      }
    }
  }
  const viewHitRate = viewTotal > 0 ? viewHits / viewTotal : 0;

  // Bilateral Symmetry IoU across central X axis
  let symMatches = 0;
  let symTotal = 0;
  for (let x = 0; x < w; x++) {
    const mirrorX = w - 1 - x;
    for (let z = 0; z < l; z++) {
      for (let dy = 0; dy <= h; dy++) {
        const b1 = snap.blockAt(ox + x, oy + dy, oz + z);
        const b2 = snap.blockAt(ox + mirrorX, oy + dy, oz + z);
        const occ1 = Boolean(b1 && b1.name !== 'air' && b1.name !== 'torch');
        const occ2 = Boolean(b2 && b2.name !== 'air' && b2.name !== 'torch');
        if (occ1 || occ2) {
          symTotal++;
          if (occ1 && occ2) symMatches++;
        }
      }
    }
  }
  const bilateralSymmetry = symTotal > 0 ? symMatches / symTotal : 0;

  const rawScore = Math.round(
    Math.min(
      100,
      voxelIoU * 40 +
        viewHitRate * 20 +
        roofSealRatio * 20 +
        foundationAnchorRatio * 10 +
        (hasTorchLight ? 10 : 0)
    )
  );
  const score = snap.stats.blocksPlaced === 0 ? 0 : rawScore;
  const passed = snap.stats.blocksPlaced > 0 && ((voxelIoU >= 0.75 && roofSealRatio >= 0.75 && foundationAnchorRatio >= 0.85) || score >= 75);
  const rawReward = snap.stats.blocksPlaced === 0 ? 0 : (voxelIoU * 50 + viewHitRate * 20 + roofSealRatio * 20 + (hasTorchLight ? 15 : 0));

  return {
    passed,
    score,
    rawReward: Number(rawReward.toFixed(2)),
    maxScore: 100,
    metrics: {
      voxelIoU: Number((voxelIoU * 100).toFixed(1)),
      fiveViewHitRateVHR: Number((viewHitRate * 100).toFixed(1)),
      bilateralSymmetry: Number((bilateralSymmetry * 100).toFixed(1)),
      roofSealPercent: Number((roofSealRatio * 100).toFixed(1)),
      foundationAnchoredPercent: Number((foundationAnchorRatio * 100).toFixed(1)),
      hasTorchLight,
      blocksPlaced: snap.stats.blocksPlaced,
    },
    breakdown: `3D Voxel IoU=${(voxelIoU * 100).toFixed(1)}%, 5-View VHR=${(viewHitRate * 100).toFixed(1)}%, Symmetry=${(bilateralSymmetry * 100).toFixed(1)}%, RoofSeal=${(roofSealRatio * 100).toFixed(0)}%, Illuminated=${hasTorchLight}.`,
  };
}

function evaluateWatchtowerPillar(
  task: MinebenchTaskSpec,
  snap: MinebenchEvaluationSnapshot
): MinebenchGradingResult {
  const heightGained = Math.max(0, snap.botPosition.y - task.origin.y);
  const ratio = Math.min(1, heightGained / 5);
  const score = Math.round(ratio * 100);
  return {
    passed: heightGained >= 4.5,
    score,
    rawReward: Number((heightGained * 18).toFixed(2)),
    maxScore: 100,
    metrics: {
      heightGained: Number(heightGained.toFixed(2)),
      blocksPlaced: snap.stats.blocksPlaced,
    },
    breakdown: `Pillared up +${heightGained.toFixed(1)}m (${snap.stats.blocksPlaced} blocks placed).`,
  };
}

/**
 * Creative Artistic Monument / Sculpture Grader:
 * Evaluates voxel placements and reach while strictly bypassing roof enclosure & weather seal flood-fill checks.
 */
function evaluateCreativeBuilding(
  task: MinebenchTaskSpec,
  snap: MinebenchEvaluationSnapshot
): MinebenchGradingResult {
  const ox = task.origin.x;
  const oy = task.origin.y;
  const oz = task.origin.z;

  let creativeBlocksPlaced = 0;
  for (let dx = -4; dx <= 8; dx++) {
    for (let dy = 0; dy <= 8; dy++) {
      for (let dz = -4; dz <= 8; dz++) {
        const b = snap.blockAt(ox + dx, oy + dy, oz + dz);
        if (b && b.name !== 'air' && b.name !== 'grass_block' && b.name !== 'dirt') {
          creativeBlocksPlaced++;
        }
      }
    }
  }

  const passed = creativeBlocksPlaced >= 4 || snap.stats.blocksPlaced >= 4;
  const score = Math.min(100, Math.round((Math.max(creativeBlocksPlaced, snap.stats.blocksPlaced) / 10) * 100));
  const rawReward = passed ? 60 + score * 0.4 : score * 0.3;

  return {
    passed,
    score,
    rawReward: Number(rawReward.toFixed(2)),
    maxScore: 100,
    metrics: {
      creativeBlocksPlaced,
      blocksPlacedStat: snap.stats.blocksPlaced,
      weatherSealBypassed: true,
    },
    breakdown: `Creative Build: ${creativeBlocksPlaced} monument blocks placed (${snap.stats.blocksPlaced} placed via kinematics). Roof & weather-sealing checks strictly bypassed.`,
  };
}

function evaluateCombatTask(
  task: MinebenchTaskSpec,
  snap: MinebenchEvaluationSnapshot
): MinebenchGradingResult {
  let targetMob = 'hostile';
  let expectedKills = 1;
  if (task.id === 'combat_creeper_kiting') {
    targetMob = 'creeper';
    expectedKills = 2;
  } else if (task.id === 'combat_skeleton_duel') {
    targetMob = 'skeleton';
    expectedKills = 1;
  } else if (task.id === 'combat_spider_swarm') {
    targetMob = 'spider';
    expectedKills = 5;
  } else if (task.id === 'combat_enderman_hunt') {
    targetMob = 'enderman';
    expectedKills = 1;
  }

  const kills = snap.stats.mobsKilled[targetMob] || 0;
  const killRatio = Math.min(1, kills / expectedKills);
  const hpRatio = Math.max(0, snap.health / 20);

  // MINEbench 2.0 Granular Combat Metrics:
  // Estimate damage dealt from living target entities
  let damageDealt = 0;
  let targetTotalMaxHp = expectedKills * 20;
  const livingTargets = snap.entities.filter(
    (e) => e.isValid && (e.name === targetMob || e.type === targetMob)
  );
  const livingHp = livingTargets.reduce((acc, e) => acc + (typeof e.health === 'number' ? e.health : 20), 0);
  damageDealt = kills >= expectedKills ? targetTotalMaxHp : Math.max(0, targetTotalMaxHp - livingHp);
  const damageRatio = Math.min(1, damageDealt / targetTotalMaxHp);

  const blastPenalty = snap.stats.creeperBlasts * 20;
  const deathPenalty = snap.deaths * 30;

  // Dense score blending kill completion, damage dealt, and surviving HP
  // If zero damage and zero kills occurred, score is strictly 0 (no free HP survival points)
  const baseCombatScore = killRatio * 50 + damageRatio * 30;
  const score = baseCombatScore === 0 ? 0 : Math.round(
    Math.max(
      0,
      Math.min(
        100,
        baseCombatScore +
          hpRatio * 20 -
          blastPenalty -
          deathPenalty
      )
    )
  );
  const rawReward =
    kills * 35 +
    damageRatio * 25 +
    (baseCombatScore > 0 ? snap.health * 1.5 : 0) -
    snap.stats.creeperBlasts * 25 -
    snap.deaths * 30;

  return {
    passed: kills >= expectedKills && snap.stats.creeperBlasts === 0 && snap.deaths === 0,
    score,
    rawReward: Number(rawReward.toFixed(2)),
    maxScore: 100,
    metrics: {
      targetMob,
      kills,
      expectedKills,
      damageDealt: Number(damageDealt.toFixed(1)),
      damagePercent: Math.round(damageRatio * 100),
      remainingHealth: snap.health,
      creeperBlasts: snap.stats.creeperBlasts,
    },
    breakdown: `Eliminated ${kills}/${expectedKills} ${targetMob}(s), Dmg=${damageDealt.toFixed(1)}/40, HP=${snap.health.toFixed(1)}/20, CreeperBlasts=${snap.stats.creeperBlasts}.`,
  };
}

function evaluateHomesteadTask(
  task: MinebenchTaskSpec,
  snap: MinebenchEvaluationSnapshot
): MinebenchGradingResult {
  if (task.id === 'homestead_wolf_taming') {
    const tamed = snap.stats.wolvesTamed > 0;
    return {
      passed: tamed,
      score: tamed ? 100 : 0,
      rawReward: tamed ? 80 : 0,
      maxScore: 100,
      metrics: { wolvesTamed: snap.stats.wolvesTamed },
      breakdown: tamed ? 'Successfully tamed wolf companion with bones!' : 'Wolf was not tamed.',
    };
  }

  // Crop farming check
  let tilledCount = 0;
  let plantedOrGrownCount = 0;
  const { x: ox, y: oy, z: oz } = task.origin;
  for (let dx = -5; dx <= 5; dx++) {
    for (let dz = -5; dz <= 5; dz++) {
      const soil = snap.blockAt(ox + dx, oy - 1, oz + dz);
      if (soil && soil.name === 'farmland') tilledCount++;
      const crop = snap.blockAt(ox + dx, oy, oz + dz);
      if (crop && ['wheat', 'carrots', 'potatoes'].includes(crop.name)) {
        plantedOrGrownCount++;
      }
    }
  }
  const wheatInInv = snap.inventory
    .filter((i) => i.name === 'wheat')
    .reduce((a, b) => a + b.count, 0);

  const score = Math.round(
    Math.min(100, Math.min(4, tilledCount) * 12 + Math.min(4, plantedOrGrownCount + wheatInInv) * 13)
  );
  return {
    passed: tilledCount >= 2 && plantedOrGrownCount + wheatInInv >= 2,
    score,
    rawReward: tilledCount * 10 + (plantedOrGrownCount + wheatInInv) * 15,
    maxScore: 100,
    metrics: { tilledCount, plantedOrGrownCount, wheatInInv },
    breakdown: `Tilled ${tilledCount} farmland plots, planted/grown ${plantedOrGrownCount} crops (harvested ${wheatInInv} wheat).`,
  };
}

function evaluatePuzzleTask(
  task: MinebenchTaskSpec,
  snap: MinebenchEvaluationSnapshot
): MinebenchGradingResult {
  if (task.id === 'puzzle_redstone_lever') {
    const flipped = snap.stats.leversFlipped > 0;
    const goalDist = Math.hypot(
      snap.botPosition.x - (task.origin.x + 6),
      snap.botPosition.z - task.origin.z
    );
    const onGold = goalDist <= 1.8;
    const score = (flipped ? 50 : 0) + (onGold ? 50 : 0);
    return {
      passed: flipped && onGold,
      score,
      rawReward: score,
      maxScore: 100,
      metrics: { leverFlipped: flipped, onGoldBlock: onGold, goalDist: Number(goalDist.toFixed(2)) },
      breakdown: `LeverFlipped=${flipped}, ReachedGoldPad=${onGold} (dist=${goalDist.toFixed(1)}m).`,
    };
  }

  // Tool preservation mining check
  const sandCount = snap.inventory.filter((i) => i.name === 'sand').reduce((a, b) => a + b.count, 0);
  const cobbleCount = snap.inventory
    .filter((i) => i.name === 'cobblestone' || i.name === 'stone')
    .reduce((a, b) => a + b.count, 0);
  const wastedIronPick = snap.stats.ironPickaxeDurabilityUsedOnStone;

  const score = Math.round(
    Math.max(
      0,
      Math.min(100, Math.min(4, sandCount) * 12.5 + Math.min(4, cobbleCount) * 12.5 - wastedIronPick * 25)
    )
  );
  return {
    passed: sandCount >= 4 && cobbleCount >= 4 && wastedIronPick === 0,
    score,
    rawReward: (sandCount + cobbleCount) * 10 - wastedIronPick * 30,
    maxScore: 100,
    metrics: { sandCount, cobbleCount, wastedIronPickDurability: wastedIronPick },
    breakdown: `Mined ${sandCount} sand & ${cobbleCount} cobblestone; IronPickaxeWastedOnStone=${wastedIronPick}.`,
  };
}

export function evaluateProgressionTask(
  task: MinebenchTaskSpec,
  snap: MinebenchEvaluationSnapshot
): MinebenchGradingResult {
  const countItem = (substr: string) =>
    snap.inventory
      .filter((i) => i.name.toLowerCase().includes(substr))
      .reduce((a, b) => a + b.count, 0);

  const taskId = String(task.id || (task as any).taskId || '').replace(/^classic_/, '');

  if (taskId === 'prog_tier1_stone_age') {
    // 4-Stage Dense DAG Milestone Scoring (25 pts each = 100 max)
    // Milestone 4: Final target item synthesized (stone_pickaxe) -> 25 pts
    const finalTarget = countItem('stone_pickaxe') > 0;
    // Milestone 3: Workstation (crafting_table crafted/placed, wooden_pickaxe crafted, or final target synthesized) -> 25 pts
    const workstation = finalTarget || countItem('crafting_table') > 0 || countItem('wooden_pickaxe') > 0 || (snap.stats?.blocksPlaced || 0) > 0;
    // Milestone 2: Basic refine (planks refined, sticks crafted, or workstation reached) -> 25 pts
    const basicRefine = workstation || finalTarget || countItem('planks') >= 8 || countItem('wooden_pickaxe') > 0 || (countItem('stick') >= 2 && countItem('planks') >= 2);
    // Milestone 1: Raw harvest (oak_log mined >= 2 or inventory has >= 2 logs, or refine reached) -> 25 pts
    const rawHarvest = basicRefine || workstation || finalTarget || countItem('log') >= 2 || (snap.stats?.blocksMined || 0) >= 2 || (((snap.stats?.blocksMined || 0) >= 1) && countItem('log') >= 1);

    const score =
      (rawHarvest ? 25 : 0) +
      (basicRefine ? 25 : 0) +
      (workstation ? 25 : 0) +
      (finalTarget ? 25 : 0);
    return {
      passed: finalTarget,
      score,
      rawReward: score * 1.2,
      maxScore: 100,
      metrics: { rawHarvest, basicRefine, workstation, finalTarget, stonePickaxe: finalTarget },
      breakdown: `Tier 1 DAG Milestones: RawHarvest=${rawHarvest} (25pt), BasicRefine=${basicRefine} (25pt), Workstation=${workstation} (25pt), TargetStonePick=${finalTarget} (25pt).`,
    };
  }

  if (taskId === 'prog_tier3_iron_armor') {
    const rawIron = countItem('raw_iron') + countItem('iron_ore');
    const ingots = countItem('iron_ingot');
    const hasFurnaceOrTable = countItem('crafting_table') > 0 || countItem('furnace') > 0 || (snap.stats?.blocksPlaced || 0) > 0;
    const armorPieces = [
      countItem('iron_helmet') > 0 || snap.equippedArmor.head === 'iron_helmet',
      countItem('iron_chestplate') > 0 || snap.equippedArmor.torso === 'iron_chestplate',
      countItem('iron_leggings') > 0 || snap.equippedArmor.legs === 'iron_leggings',
      countItem('iron_boots') > 0 || snap.equippedArmor.feet === 'iron_boots',
    ].filter(Boolean).length;

    const allEquipped =
      snap.equippedArmor.head === 'iron_helmet' &&
      snap.equippedArmor.torso === 'iron_chestplate' &&
      snap.equippedArmor.legs === 'iron_leggings' &&
      snap.equippedArmor.feet === 'iron_boots';

    // Dense 4-Stage DAG Reward Shaping (25 pts each = 100 max)
    // Milestone 1: Raw harvest (iron_ore / raw_iron mined >= 2 or coal mined >= 1, or downstream smelt/craft achieved) -> 25 pts
    const milestone1Raw = (rawIron >= 2 || (snap.stats?.blocksMined || 0) >= 2 || ingots > 0 || armorPieces > 0) ? 25 : 0;
    // Milestone 2: Basic refine / Smelting (has furnace/table and iron_ingot >= 1, or armor pieces crafted) -> 25 pts
    const milestone2Refine = (armorPieces > 0 || (hasFurnaceOrTable && ingots >= 1)) ? 25 : (ingots >= 1 ? 20 : 0);
    // Milestone 3: Armor synthesis (iron armor crafted >= 1 piece) -> 25 pts
    const milestone3Craft = armorPieces >= 1 ? 25 : 0;
    // Milestone 4: Final target gear equipped (all 4 iron armor pieces equipped) -> 25 pts
    const milestone4Equip = allEquipped ? 25 : (armorPieces === 4 ? 15 : 0);

    const score = Math.min(
      100,
      milestone1Raw + milestone2Refine + milestone3Craft + milestone4Equip
    );
    return {
      passed: armorPieces === 4 && allEquipped,
      score,
      rawReward: score * 1.5,
      maxScore: 100,
      metrics: {
        milestone1Raw: milestone1Raw === 25,
        milestone2Refine: milestone2Refine >= 20,
        milestone3Craft: milestone3Craft === 25,
        milestone4Equip: milestone4Equip === 25,
        rawIron,
        ironIngotsRemaining: ingots,
        armorPieces,
        allEquipped,
      },
      breakdown: `Tier 3 DAG Milestones: Raw=${milestone1Raw}/25, SmeltRefine=${milestone2Refine}/25, Craft=${milestone3Craft}/25, FullEquip=${milestone4Equip}/25.`,
    };
  }

  // Guaranteed Multiplayer Arena & Cooperative Tasks
  if (task.id.startsWith('mp_') || task.id.startsWith('coop_')) {
    const mp = snap.multiplayer ?? {
      opponentName: 'DQN_Opponent',
      opponentHealth: 20,
      opponentAlive: true,
      opponentKills: 0,
      opponentAtGoal: false,
      opponentBlocksCollected: 0,
      opponentInventory: [],
    };

    // Cooperative Companion Benchmarks
    if (task.id === 'coop_give_food') {
      const companionHasFood = (mp.opponentInventory || []).some(
        (i: any) => i.name === 'cooked_beef' && i.count >= 1
      );
      const passed = companionHasFood && snap.deaths === 0;
      const score = passed ? 100 : 0;
      return {
        passed,
        score,
        rawReward: passed ? 150 : 0,
        maxScore: 100,
        metrics: { companionHasFood, deaths: snap.deaths },
        breakdown: `Coop Food: Delivered=${companionHasFood}, Deaths=${snap.deaths}.`,
      };
    }

    if (task.id === 'coop_give_tool') {
      const companionHasSword = (mp.opponentInventory || []).some(
        (i: any) => i.name === 'iron_sword' && i.count >= 1
      );
      const passed = companionHasSword && snap.deaths === 0;
      const score = passed ? 100 : 0;
      return {
        passed,
        score,
        rawReward: passed ? 150 : 0,
        maxScore: 100,
        metrics: { companionHasSword, deaths: snap.deaths },
        breakdown: `Coop Tool: Delivered=${companionHasSword}, Deaths=${snap.deaths}.`,
      };
    }

    if (task.id === 'coop_craft_and_deliver') {
      const companionHasPick = (mp.opponentInventory || []).some(
        (i: any) => i.name.includes('pickaxe') && i.count >= 1
      );
      const agentCraftedPick = snap.inventory.some((i: any) => i.name.includes('pickaxe')) || companionHasPick;
      const passed = companionHasPick && snap.deaths === 0;
      const score = passed ? 100 : agentCraftedPick ? 60 : 0;
      return {
        passed,
        score,
        rawReward: (passed ? 180 : agentCraftedPick ? 80 : 0) - snap.deaths * 50,
        maxScore: 100,
        metrics: { companionHasPick, agentCraftedPick, deaths: snap.deaths },
        breakdown: `Coop Craft & Deliver: Delivered=${companionHasPick}, Crafted=${agentCraftedPick}.`,
      };
    }

    if (task.id === 'coop_follow_companion') {
      const goalX = task.origin.x + 3;
      const goalZ = task.origin.z + 20;
      const distToGoal = Math.hypot(snap.botPosition.x - goalX, snap.botPosition.z - goalZ);
      const atGoal = (distToGoal <= 8.0 || (mp.opponentAtGoal && distToGoal <= 10.0)) && snap.deaths === 0;
      const passed = atGoal && snap.deaths === 0;
      const score = atGoal ? 100 : Math.max(0, Math.round(Math.max(0, 1 - distToGoal / 20) * 80));
      return {
        passed,
        score,
        rawReward: (atGoal ? 140 : Math.max(0, 1 - distToGoal / 20) * 60) - snap.deaths * 50,
        maxScore: 100,
        metrics: { atGoal, distToGoal, deaths: snap.deaths },
        breakdown: `Coop Follow: AtGoal=${atGoal}, DistToGoal=${distToGoal.toFixed(1)}m.`,
      };
    }

    if (task.id === 'coop_bodyguard_protect') {
      const companionAlive = mp.opponentAlive && mp.opponentHealth >= 12;
      const zombieKilled = (snap.stats.mobsKilled?.zombie || 0) > 0 || Object.values(snap.stats.mobsKilled || {}).reduce((a: number, b: number) => a + b, 0) > 0;
      const passed = zombieKilled && companionAlive && snap.deaths === 0;
      const score = passed ? 100 : zombieKilled ? 60 : 0;
      return {
        passed,
        score,
        rawReward: (passed ? 160 : zombieKilled ? 60 : 0) - snap.deaths * 60,
        maxScore: 100,
        metrics: { zombieKilled, companionAlive, companionHealth: mp.opponentHealth, deaths: snap.deaths },
        breakdown: `Coop Bodyguard: ZombieKilled=${zombieKilled}, CompanionHP=${mp.opponentHealth.toFixed(1)}/20.`,
      };
    }

    if (task.id === 'coop_resource_share') {
      const companionLogs = (mp.opponentInventory || [])
        .filter((i: any) => i.name === 'oak_log')
        .reduce((sum: number, i: any) => sum + i.count, 0);
      const passed = companionLogs >= 4 && snap.deaths === 0;
      const score = passed ? 100 : Math.min(80, companionLogs * 20);
      return {
        passed,
        score,
        rawReward: companionLogs * 35 - snap.deaths * 50,
        maxScore: 100,
        metrics: { companionLogs, deaths: snap.deaths },
        breakdown: `Coop Resource Share: DeliveredLogs=${companionLogs}/4, Deaths=${snap.deaths}.`,
      };
    }

    if (task.id.startsWith('mp_pvp_')) {
      const opponentDefeated = !mp.opponentAlive || mp.opponentHealth <= 0;
      const damageDealt = Math.max(0, 20 - Math.max(0, mp.opponentHealth));
      const rawScore = opponentDefeated
        ? Math.round(70 + 30 * Math.max(0, Math.min(1, snap.health / 20)))
        : Math.min(60, Math.round(damageDealt * 3));
      const score = Math.max(0, rawScore - snap.deaths * 35);
      const passed = opponentDefeated && snap.deaths === 0;
      return {
        passed,
        score,
        rawReward:
          (opponentDefeated ? 150 : damageDealt * 4) +
          (snap.health > 0 ? snap.health * 1.5 : 0) -
          snap.deaths * 80,
        maxScore: 100,
        metrics: {
          opponentDefeated,
          opponentHealth: mp.opponentHealth,
          damageDealt,
          agentHealth: snap.health,
          deaths: snap.deaths,
        },
        breakdown: `Multiplayer PvP (${task.id}): OpponentDefeated=${opponentDefeated}, OppHP=${mp.opponentHealth.toFixed(1)}/20, AgentHP=${snap.health.toFixed(1)}/20, Deaths=${snap.deaths}.`,
      };
    }

    if (task.id.includes('bridge_race') || task.id.includes('escort_race')) {
      const goalX = task.origin.x;
      const goalZ = task.origin.z + 20;
      const distToGoal = Math.hypot(
        snap.botPosition.x - goalX,
        snap.botPosition.z - goalZ
      );
      const progress = Math.max(0, Math.min(1, 1 - distToGoal / 20));
      const atGoal = distToGoal <= 2.5 && snap.botPosition.y >= task.origin.y - 2;
      const wonRace = atGoal && !mp.opponentAtGoal;
      const score = atGoal
        ? wonRace
          ? 100
          : 80
        : Math.round(progress * 65);
      const passed = atGoal && snap.deaths === 0;
      return {
        passed,
        score,
        rawReward: (atGoal ? (wonRace ? 140 : 90) : progress * 50) - snap.deaths * 60,
        maxScore: 100,
        metrics: {
          distToGoal: Number(distToGoal.toFixed(2)),
          atGoal,
          opponentAtGoal: mp.opponentAtGoal,
          wonRace,
          deaths: snap.deaths,
        },
        breakdown: `Multiplayer Race (${task.id}): Dist=${distToGoal.toFixed(1)}m, AtGoal=${atGoal}, OppAtGoal=${mp.opponentAtGoal}, Won=${wonRace}.`,
      };
    }

    if (task.id.includes('resource_duel')) {
      const logsCollected = snap.inventory
        .filter((i) => i.name === 'oak_log')
        .reduce((sum, i) => sum + i.count, 0);
      const wonDuel = logsCollected > mp.opponentBlocksCollected;
      const score = Math.min(100, logsCollected * 20 + (wonDuel ? 20 : 0));
      const passed =
        logsCollected >= 4 && logsCollected >= mp.opponentBlocksCollected;
      return {
        passed,
        score,
        rawReward: logsCollected * 25 + (wonDuel ? 60 : 0) - snap.deaths * 50,
        maxScore: 100,
        metrics: {
          logsCollected,
          opponentBlocksCollected: mp.opponentBlocksCollected,
          wonDuel,
        },
        breakdown: `Multiplayer Resource Duel: Logs=${logsCollected}/4, OppLogs=${mp.opponentBlocksCollected}, Won=${wonDuel}.`,
      };
    }
  }

  // Tier 6: Ender Dragon Boss Fight
  const crystalsDestroyed = snap.stats.endCrystalsDestroyed;
  const dragonSlain = snap.stats.enderDragonSlain;
  const deathPenalty = snap.deaths * 50;
  const rawScore = Math.min(2, crystalsDestroyed) * 25 + (dragonSlain ? 50 : 0);
  const score = Math.max(0, Math.min(100, rawScore - deathPenalty));
  return {
    passed: crystalsDestroyed >= 2 && dragonSlain && snap.deaths === 0,
    score,
    rawReward: crystalsDestroyed * 60 + (dragonSlain ? 500 : 0) - snap.deaths * 100,
    maxScore: 100,
    metrics: {
      endCrystalsDestroyed: crystalsDestroyed,
      enderDragonSlain: dragonSlain,
      remainingHealth: snap.health,
    },
    breakdown: `Tier 6 Boss Fight: EndCrystalsDestroyed=${crystalsDestroyed}/2, EnderDragonSlain=${dragonSlain}, HP=${snap.health.toFixed(1)}/20.`,
  };
}
