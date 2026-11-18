import { SanitizedWorldState } from '../types';

export interface GoalStepRecord {
  stepNumber: number;
  toolName: string;
  args: Record<string, any>;
  success: boolean;
  summary: string;
}

export interface MotorLifecycleFeedback {
  subgoal_completed?: boolean;
  should_replan?: boolean;
  recommended_tool?: string;
}

export interface GroundedAffordances {
  equippedSummary: {
    mainHand: string;
    offHand: string;
    armorWorn: string[];
    lowestToolDurabilityPercent: number | null;
    shortString: string;
  };
  nearbyWorkstations: string[];
  craftableNow: string[];
  smeltableNow: boolean;
  smeltableItems: string[];
}

export interface GoalTransitionResult {
  advanced: boolean;
  completedAll: boolean;
  needsReplan: boolean;
  reason: string;
  currentSubGoal: string | null;
}

/**
 * Inspects current 46-slot inventory counts, tool durability, equipped gear, and nearby
 * world blocks to compute exact grounded affordances available to the agent right now.
 */
export function computeGroundedAffordances(state: SanitizedWorldState): GroundedAffordances {
  const inv = state?.inventory || [];
  const countByName = (pred: (name: string) => boolean): number =>
    inv.reduce((sum, it) => (pred((it.name || '').toLowerCase()) ? sum + (it.count || 0) : sum), 0);

  const logs = countByName((n) => n.endsWith('_log') || n.endsWith('_stem'));
  const planks = countByName((n) => n.endsWith('_planks'));
  const sticks = countByName((n) => n === 'stick');
  const cobble = countByName((n) => n === 'cobblestone' || n === 'cobbled_deepslate');
  const iron = countByName((n) => n === 'iron_ingot');
  const coal = countByName((n) => n === 'coal' || n === 'charcoal');
  const wheat = countByName((n) => n === 'wheat');
  const rawOresOrFood = inv
    .filter((i) => {
      const n = (i.name || '').toLowerCase();
      return (
        n === 'raw_iron' ||
        n === 'iron_ore' ||
        n === 'deepslate_iron_ore' ||
        n === 'raw_gold' ||
        n === 'gold_ore' ||
        n === 'raw_copper' ||
        n === 'beef' ||
        n === 'porkchop' ||
        n === 'mutton' ||
        n === 'chicken' ||
        n === 'cobblestone'
      );
    })
    .map((i) => i.name);

  const hasFuel = logs > 0 || planks > 0 || coal > 0 || countByName((n) => n === 'coal_block') > 0;
  const smeltableItems = [...new Set(rawOresOrFood)];
  const smeltableNow = smeltableItems.length > 0 && hasFuel;

  // Detect nearby workstations and liquids
  const stationSet = new Set<string>();
  const adj = state?.terrainContext?.adjacentBlocks;
  const checkBlockName = (rawName?: string) => {
    const n = (rawName || '').toLowerCase();
    if (!n || n === 'air') return;
    if (n.includes('crafting_table')) stationSet.add('crafting_table');
    else if (n.includes('furnace') || n.includes('smoker')) stationSet.add('furnace');
    else if (n.includes('chest') || n.includes('barrel')) stationSet.add('chest');
    else if (n.includes('bed')) stationSet.add('bed');
    else if (n.includes('water')) stationSet.add('water');
    else if (n.includes('lava')) stationSet.add('lava');
  };

  for (const nb of state?.nearbyNotableBlocks || []) {
    checkBlockName(nb.name);
  }
  if (adj) {
    checkBlockName(adj.north);
    checkBlockName(adj.south);
    checkBlockName(adj.east);
    checkBlockName(adj.west);
    checkBlockName(adj.aboveHead);
    checkBlockName(adj.belowFeet);
  }
  if (state?.isInWater) stationSet.add('water');
  if (state?.isInLava) stationSet.add('lava');

  const nearbyWorkstations = Array.from(stationSet);
  const hasTableAccess =
    nearbyWorkstations.includes('crafting_table') || countByName((n) => n === 'crafting_table') > 0;

  // Immediate craftability based strictly on current inventory items + table access
  const craftableNow: string[] = [];
  if (logs >= 1) craftableNow.push('oak_planks');
  if (planks >= 2) craftableNow.push('stick');
  if (planks >= 4) craftableNow.push('crafting_table');
  if ((coal >= 1) && sticks >= 1) craftableNow.push('torch');

  if (hasTableAccess) {
    if (planks >= 3 && sticks >= 2) craftableNow.push('wooden_pickaxe');
    if (cobble >= 3 && sticks >= 2) craftableNow.push('stone_pickaxe');
    if (iron >= 3 && sticks >= 2) craftableNow.push('iron_pickaxe');
    if (cobble >= 8) craftableNow.push('furnace');
    if (planks >= 8) craftableNow.push('chest');
    if (planks >= 6 && iron >= 1) craftableNow.push('shield');
    if (iron >= 5) craftableNow.push('iron_helmet');
    if (iron >= 8) craftableNow.push('iron_chestplate');
    if (iron >= 7) craftableNow.push('iron_leggings');
    if (iron >= 4) craftableNow.push('iron_boots');
    if (wheat >= 3) craftableNow.push('bread');
  }

  // Equipped gear + durability summary
  const armor = state?.equippedArmor;
  const armorWorn = [
    armor?.head ? `head:${armor.head}` : null,
    armor?.torso ? `torso:${armor.torso}` : null,
    armor?.legs ? `legs:${armor.legs}` : null,
    armor?.feet ? `feet:${armor.feet}` : null,
  ].filter((x): x is string => Boolean(x));

  const durabilityValues = inv
    .map((i) => i.durabilityPercent)
    .filter((d): d is number => typeof d === 'number' && Number.isFinite(d));
  const lowestToolDurabilityPercent =
    durabilityValues.length > 0 ? Math.min(...durabilityValues) : null;

  const mainHand = state?.heldItem || 'none';
  const offHand = armor?.offHand || 'none';
  const parts = [
    `Main:${mainHand}`,
    `Off:${offHand}`,
    `Armor:[${armorWorn.length > 0 ? armorWorn.join(',') : 'none'}]`,
    lowestToolDurabilityPercent !== null ? `MinDur:${lowestToolDurabilityPercent}%` : null,
  ].filter(Boolean);

  return {
    equippedSummary: {
      mainHand,
      offHand,
      armorWorn,
      lowestToolDurabilityPercent,
      shortString: parts.join(' '),
    },
    nearbyWorkstations,
    craftableNow,
    smeltableNow,
    smeltableItems,
  };
}

/**
 * Stateful Hierarchical Long-Term Goal & Planning Manager:
 * Bridges multi-step SLM plans ("Get full iron armor", "Beat Ender Dragon") with
 * step outcomes and real-time System 1 motor replan triggers.
 */
export class HierarchicalGoalPlanner {
  public macroGoal: string | null = null;
  public subGoalQueue: string[] = [];
  public currentSubGoalIndex = 0;
  public activeIntent = '';
  public stepHistory: GoalStepRecord[] = [];
  private consecutiveFailures = 0;

  public setMacroGoal(goal: string, planSteps?: string[]): void {
    const cleanedGoal = (goal || '').trim();
    this.macroGoal = cleanedGoal || null;
    this.subGoalQueue =
      Array.isArray(planSteps) && planSteps.length > 0
        ? planSteps.map((s) => String(s).trim()).filter(Boolean)
        : cleanedGoal
          ? [cleanedGoal]
          : [];
    this.currentSubGoalIndex = 0;
    this.activeIntent = this.subGoalQueue[0] || cleanedGoal;
    this.stepHistory = [];
    this.consecutiveFailures = 0;
  }

  public getCurrentSubGoal(): string | null {
    if (this.currentSubGoalIndex >= 0 && this.currentSubGoalIndex < this.subGoalQueue.length) {
      return this.subGoalQueue[this.currentSubGoalIndex];
    }
    return this.macroGoal;
  }

  public advanceOrReplan(
    stepResult: {
      toolName: string;
      args?: Record<string, any>;
      success: boolean;
      summary: string;
    },
    motorFeedback?: MotorLifecycleFeedback | null,
    worldState?: SanitizedWorldState
  ): GoalTransitionResult {
    this.stepHistory.push({
      stepNumber: this.stepHistory.length + 1,
      toolName: stepResult.toolName,
      args: stepResult.args || {},
      success: stepResult.success,
      summary: stepResult.summary,
    });

    const summaryLower = (stepResult.summary || '').toLowerCase();
    const brokenOrMissing =
      summaryLower.includes('missing') ||
      summaryLower.includes('cannot craft') ||
      summaryLower.includes('broke') ||
      summaryLower.includes('no furnace') ||
      summaryLower.includes('no fuel');

    const healthEmergency = Boolean(worldState && worldState.health <= 5);

    if (!stepResult.success || motorFeedback?.should_replan || brokenOrMissing || healthEmergency) {
      this.consecutiveFailures++;
      const reason = motorFeedback?.should_replan
        ? 'Motor head signaled should_replan'
        : healthEmergency
          ? `Critical low health (${worldState?.health}/20)`
          : !stepResult.success
            ? `Step ${stepResult.toolName} failed: ${stepResult.summary}`
            : `Prerequisite issue: ${stepResult.summary}`;
      return {
        advanced: false,
        completedAll: false,
        needsReplan: true,
        reason,
        currentSubGoal: this.getCurrentSubGoal(),
      };
    }

    this.consecutiveFailures = 0;
    if (this.currentSubGoalIndex + 1 < this.subGoalQueue.length) {
      this.currentSubGoalIndex++;
      this.activeIntent = this.subGoalQueue[this.currentSubGoalIndex];
      return {
        advanced: true,
        completedAll: false,
        needsReplan: false,
        reason: motorFeedback?.subgoal_completed
          ? 'Motor & tool confirmed sub-goal completed; advancing queue.'
          : 'Step succeeded; advancing to next sub-goal.',
        currentSubGoal: this.getCurrentSubGoal(),
      };
    }

    return {
      advanced: false,
      completedAll: this.subGoalQueue.length > 0,
      needsReplan: false,
      reason: 'Final sub-goal executed.',
      currentSubGoal: this.getCurrentSubGoal(),
    };
  }

  public formatGoalHorizonContext(worldState?: SanitizedWorldState): string {
    if (!this.macroGoal && this.subGoalQueue.length === 0) {
      return 'Goal Horizon: Idle (no active macro goal).';
    }
    const idx = Math.min(this.currentSubGoalIndex + 1, Math.max(1, this.subGoalQueue.length));
    const total = Math.max(1, this.subGoalQueue.length);
    const current = this.getCurrentSubGoal() || 'none';
    const next = this.subGoalQueue[this.currentSubGoalIndex + 1];
    const lines = [
      `Macro Goal: "${this.macroGoal || current}"`,
      `Active Sub-Goal (${idx}/${total}): "${current}"${next ? ` | Next: "${next}"` : ''}`,
    ];
    if (this.consecutiveFailures > 0) {
      lines.push(`Replan Alert: ${this.consecutiveFailures} consecutive failure(s) on current sub-goal.`);
    }
    if (worldState) {
      const aff = computeGroundedAffordances(worldState);
      lines.push(`Craftable Now: [${aff.craftableNow.join(', ') || 'none'}]`);
    }
    return lines.join('\n');
  }
}
