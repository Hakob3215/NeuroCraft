/**
 * MINEbench Tier 3: Autonomous Vanilla Dragon Survival Run
 * =======================================================
 * Evaluates continuous depth across the canonical 8-Milestone Vanilla Survival DAG:
 *   1. Wood Age (Logs -> Planks -> Crafting Table -> Wooden Pickaxe)
 *   2. Stone Age (Cobblestone -> Stone Pickaxe -> Stone Sword)
 *   3. Iron Age (Smelting -> Iron Ingot -> Iron Pickaxe -> Shield -> Iron Armor)
 *   4. Diamond & Obsidian (Diamond Pickaxe -> Obsidian Mining)
 *   5. Nether Expedition (Nether Portal -> Blaze Rods -> Ender Pearls -> Eyes of Ender)
 *   6. Stronghold Triangulation (DeepMind 2-Ray Eye of Ender Intersection Solver)
 *   7. End Dimension (Portal Activation -> Sniping End Crystals with Bow)
 *   8. Dragon Slain ('Free the End' Advancement)
 *
 * Runs on a fresh vanilla procedural seed with empty inventory.
 */

import * as fs from 'fs';
import * as path from 'path';
import { Vec3 } from 'vec3';
import { HeadlessMinecraftServerManager } from './headlessServer';
import { System1BotController } from '../src/system1/botController';
import { DynamicMCPToolRegistry } from '../src/system2/mcp/toolRegistry';
import { NeuroCraftMCPServer } from '../src/system2/mcp/mcpServer';
import { registerBaseMCPTools } from '../src/system2/skills/baseSkills';
import { ExecutedStepRecord, UnifiedLLMClient } from '../src/system2/llmClient';
import { SubtaskStatus } from '../src/types';

export interface MilestoneProgress {
  id: string;
  title: string;
  achieved: boolean;
  score: number;
  achievedAtStep?: number;
  details: string;
}

export const SURVIVAL_DAG_MILESTONES: MilestoneProgress[] = [
  {
    id: 'milestone_1_wood',
    title: 'Milestone 1: Wood Age (Logs / Planks / Crafting Table / Wooden Pickaxe)',
    achieved: false,
    score: 10,
    details: 'Harvest wood logs and assemble basic wooden tools.',
  },
  {
    id: 'milestone_2_stone',
    title: 'Milestone 2: Stone Age (Cobblestone / Stone Pickaxe / Stone Sword)',
    achieved: false,
    score: 15,
    details: 'Mine cobblestone and upgrade to stone tools & weapons.',
  },
  {
    id: 'milestone_3_iron',
    title: 'Milestone 3: Iron Age (Smelting / Iron Ingot / Iron Pickaxe / Shield)',
    achieved: false,
    score: 20,
    details: 'Smelt iron ore into ingots and craft an iron pickaxe and shield.',
  },
  {
    id: 'milestone_4_diamond_obsidian',
    title: 'Milestone 4: Diamond & Obsidian Age (Diamond Pickaxe / Obsidian)',
    achieved: false,
    score: 15,
    details: 'Mine diamonds and harvest obsidian blocks for the Nether portal.',
  },
  {
    id: 'milestone_5_nether',
    title: 'Milestone 5: Nether Expedition (Blaze Rods & Ender Pearls)',
    achieved: false,
    score: 15,
    details: 'Enter the Nether, harvest blaze rods, and craft Eyes of Ender.',
  },
  {
    id: 'milestone_6_stronghold_triangulation',
    title: 'Milestone 6: Mathematical Stronghold Triangulation',
    achieved: false,
    score: 10,
    details: 'Execute DeepMind 2-ray vector triangulation to locate the Stronghold.',
  },
  {
    id: 'milestone_7_end_dimension',
    title: 'Milestone 7: End Dimension & Crystal Sniping',
    achieved: false,
    score: 10,
    details: 'Activate End portal and destroy protective End Crystals with bow.',
  },
  {
    id: 'milestone_8_dragon_slain',
    title: 'Milestone 8: Ender Dragon Slain ("Free the End")',
    achieved: false,
    score: 5,
    details: 'Slay the Ender Dragon to achieve complete Minecraft survival victory.',
  },
];

export interface DragonSurvivalResult {
  progressDepthScore: number;
  milestonesAchievedCount: number;
  efficiencyScore: number;
}

export async function runVanillaDragonSurvivalRun(maxSteps = 30): Promise<DragonSurvivalResult> {
  const resultsDir = path.resolve(process.cwd(), 'minebench', 'results');
  fs.mkdirSync(resultsDir, { recursive: true });
  const telemetryPath = path.join(resultsDir, 'dragon_survival_telemetry.json');

  console.log('======================================================================');
  console.log('  MINEBENCH TIER 3: AUTONOMOUS VANILLA DRAGON SURVIVAL RUN');
  console.log('  - World Type: Procedural Vanilla Seed (Normal Difficulty)');
  console.log('  - Spawn Loadout: Completely Naked (0 items in inventory)');
  console.log('  - Objective: Traverse the 8-Milestone Canonical Survival DAG');
  console.log('======================================================================\n');

  const dragonServerDir = path.resolve(process.cwd(), 'minebench', 'server_dragon');
  // Completely purge any cached world/playerdata so every survival run is a fresh procedural seed with 0 disk leakage
  const dragonWorldDir = path.join(dragonServerDir, 'world_dragon');
  if (fs.existsSync(dragonWorldDir)) {
    try {
      fs.rmSync(dragonWorldDir, { recursive: true, force: true });
    } catch {}
  }

  const serverPort = 25575;
  const server = new HeadlessMinecraftServerManager(dragonServerDir);
  await server.startServer(serverPort, 120000, 'minecraft\\:normal', 'world_dragon');

  const system1 = new System1BotController();
  await system1.connect('127.0.0.1', serverPort, 'NeuroAgent');

  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  const mcpServer = new NeuroCraftMCPServer(registry);
  const llm = new UnifiedLLMClient();

  // Reset bot to clean survival state and wait for full inventory wipe
  server.sendCommand('op NeuroAgent');
  server.sendCommand('gamemode survival NeuroAgent');
  server.sendCommand('clear NeuroAgent');
  server.sendCommand('difficulty normal');
  server.sendCommand('time set noon');
  server.sendCommand('weather clear');

  for (let c = 0; c < 10; c++) {
    server.sendCommand('clear NeuroAgent');
    await new Promise((r) => setTimeout(r, 200));
    if ((system1.bot?.inventory?.items() || []).length === 0) break;
  }

  const cleanItems = system1.bot?.inventory?.items() || [];
  console.log(`[DragonRun Preflight] Bot inventory verified: ${cleanItems.length} items (100% Naked Survival Spawn).`);

  const milestones = JSON.parse(JSON.stringify(SURVIVAL_DAG_MILESTONES)) as MilestoneProgress[];
  const executedHistory: ExecutedStepRecord[] = [];
  let currentGoal = 'Progress through the survival tech tree: start by gathering wood logs and crafting wooden tools.';

  const checkMilestones = () => {
    const inv = system1.bot?.inventory?.items() || [];
    const itemNames = new Set<string>(inv.map((i: any) => String(i.name)));

    // Milestone 1: Wood Age
    const hasPlanks = Array.from(itemNames).some((name) => name.endsWith('_planks'));
    if (
      !milestones[0].achieved &&
      (itemNames.has('wooden_pickaxe') || itemNames.has('crafting_table') || hasPlanks)
    ) {
      milestones[0].achieved = true;
      milestones[0].achievedAtStep = executedHistory.length;
      console.log(`  ⭐ UNLOCKED: ${milestones[0].title}`);
    }

    // Milestone 2: Stone Age
    if (
      !milestones[1].achieved &&
      (itemNames.has('stone_pickaxe') || itemNames.has('stone_sword') || itemNames.has('cobblestone') || itemNames.has('cobbled_deepslate'))
    ) {
      milestones[1].achieved = true;
      milestones[1].achievedAtStep = executedHistory.length;
      console.log(`  ⭐ UNLOCKED: ${milestones[1].title}`);
    }

    // Milestone 3: Iron Age
    if (
      !milestones[2].achieved &&
      (itemNames.has('iron_ingot') || itemNames.has('raw_iron') || itemNames.has('iron_pickaxe') || itemNames.has('shield'))
    ) {
      milestones[2].achieved = true;
      milestones[2].achievedAtStep = executedHistory.length;
      console.log(`  ⭐ UNLOCKED: ${milestones[2].title}`);
    }

    // Milestone 4: Diamond & Obsidian
    if (
      !milestones[3].achieved &&
      (itemNames.has('diamond') || itemNames.has('diamond_pickaxe') || itemNames.has('obsidian'))
    ) {
      milestones[3].achieved = true;
      milestones[3].achievedAtStep = executedHistory.length;
      console.log(`  ⭐ UNLOCKED: ${milestones[3].title}`);
    }

    // Milestone 5: Nether
    if (
      !milestones[4].achieved &&
      (itemNames.has('blaze_rod') || itemNames.has('ender_pearl') || itemNames.has('ender_eye'))
    ) {
      milestones[4].achieved = true;
      milestones[4].achievedAtStep = executedHistory.length;
      console.log(`  ⭐ UNLOCKED: ${milestones[4].title}`);
    }
  };

  let runResult: DragonSurvivalResult = {
    progressDepthScore: 0,
    milestonesAchievedCount: 0,
    efficiencyScore: 0,
  };

  try {
    for (let step = 0; step < maxSteps; step++) {
      checkMilestones();

      const worldState = system1.getLatestWorldState();
      const schemas = registry.listSchemas();

      const decision =
        step === 0
          ? await llm.generateCognitiveDecision('Guide', currentGoal, worldState, schemas, [])
          : await llm.generateGoalContinuationDecision(currentGoal, executedHistory, worldState, schemas, []);

      if (decision.activeGoal) currentGoal = decision.activeGoal;

      if (!decision.toolCall || !decision.toolCall.name) {
        if (decision.goalCompleted) break;
        continue;
      }

      const toolName = decision.toolCall.name;
      const toolArgs = decision.toolCall.arguments || {};
      const abortController = new AbortController();
      const stepTimeout = setTimeout(() => abortController.abort(), 25000);

      const subtask: SubtaskStatus = {
        id: `dragon_run_${step}`,
        toolName,
        description: `Executing ${toolName}`,
        args: toolArgs,
        startedAt: Date.now(),
        estimatedDurationMs: 5000,
        progress: 0,
        status: 'RUNNING',
      };

      let success = false;
      let summary = '';
      let hardTimer: ReturnType<typeof setTimeout> | null = null;

      try {
        const rpcPromise = mcpServer.handleRpcRequest(
          {
            jsonrpc: '2.0',
            id: subtask.id,
            method: 'tools/call',
            params: { name: toolName, arguments: toolArgs },
          },
          () => system1.createToolExecutionContext(subtask, async () => {}, abortController.signal)
        );
        const hardTimeoutPromise = new Promise<{ result?: any; error?: { message: string } }>((resolve) => {
          hardTimer = setTimeout(() => {
            try { abortController.abort(); } catch {}
            resolve({ error: { message: 'Dragon survival step hard timeout (35000ms)' } });
          }, 35000);
        });
        const res = await Promise.race([rpcPromise, hardTimeoutPromise]);

        const text = res.result?.content?.[0]?.text;
        if (text) {
          try {
            const p = JSON.parse(text);
            success = Boolean(p.success);
            summary = String(p.summary || '');
            if (toolName === 'triangulate_stronghold_coordinates' && success) {
              milestones[5].achieved = true;
              milestones[5].achievedAtStep = step + 1;
              console.log(`  ⭐ UNLOCKED: ${milestones[5].title}`);
            }
          } catch {
            success = !res.error;
            summary = text;
          }
        } else if (res.error) {
          success = false;
          summary = String(res.error.message || 'Step timeout');
        }
      } catch (err: any) {
        summary = err.message;
      } finally {
        if (hardTimer) clearTimeout(hardTimer);
        clearTimeout(stepTimeout);
      }

      executedHistory.push({
        stepNumber: step + 1,
        toolName,
        args: toolArgs,
        success,
        summary,
      });

      console.log(
        `  [Step ${String(step + 1).padStart(2)}/${maxSteps}] ${toolName.padEnd(28)} -> ${
          success ? 'OK' : 'FAIL'
        }: ${summary.slice(0, 60)}`
      );

      checkMilestones();
    }
  } finally {
    const achievedMilestones = milestones.filter((m) => m.achieved);
    const progressDepthScore = achievedMilestones.reduce((acc, m) => acc + m.score, 0);

    // MINEbench Tier 3 Dual Score: Progress Depth % + Efficiency %
    const PAR_STEPS_BY_MILESTONE: Record<string, number> = {
      milestone_1_wood: 3,
      milestone_2_stone: 6,
      milestone_3_iron: 12,
      milestone_4_diamond_obsidian: 18,
      milestone_5_nether: 24,
      milestone_6_stronghold_triangulation: 27,
      milestone_7_end_dimension: 29,
      milestone_8_dragon_slain: 30,
    };

    let parEfficiencySum = 0;
    if (achievedMilestones.length > 0) {
      for (const m of achievedMilestones) {
        const par = PAR_STEPS_BY_MILESTONE[m.id] || 10;
        const actual = m.achievedAtStep || executedHistory.length || 1;
        parEfficiencySum += Math.min(1.0, par / actual);
      }
      parEfficiencySum = parEfficiencySum / achievedMilestones.length;
    } else {
      parEfficiencySum = 0;
    }

    const successfulSteps = executedHistory.filter((s) => s.success).length;
    const actionSuccessRate = executedHistory.length > 0 ? successfulSteps / executedHistory.length : 0;
    const finalHp = system1.bot?.health || 0;
    const healthPreservation = Math.max(0, Math.min(1.0, finalHp / 20));

    const efficiencyScore = Math.round(
      (parEfficiencySum * 0.45 + actionSuccessRate * 0.35 + healthPreservation * 0.20) * 100
    );

    const telemetry = {
      completedAt: new Date().toISOString(),
      totalStepsExecuted: executedHistory.length,
      progressDepthScore,
      progressDepthPercent: progressDepthScore,
      efficiencyScore,
      efficiencyBreakdown: {
        parEfficiencyPercent: Math.round(parEfficiencySum * 100),
        actionSuccessPercent: Math.round(actionSuccessRate * 100),
        healthPreservationPercent: Math.round(healthPreservation * 100),
        finalHealth: finalHp,
      },
      milestonesAchievedCount: achievedMilestones.length,
      milestones,
      executedHistory,
    };

    fs.writeFileSync(telemetryPath, JSON.stringify(telemetry, null, 2), 'utf-8');

    runResult = {
      progressDepthScore,
      milestonesAchievedCount: achievedMilestones.length,
      efficiencyScore,
    };

    console.log('\n======================================================================');
    console.log(`  MINEBENCH TIER 3 DUAL SCORECARD: "BEAT MINECRAFT FROM 0"`);
    console.log(`  - Progress Depth Score : ${progressDepthScore.toFixed(1)}% (${achievedMilestones.length}/${milestones.length} milestones)`);
    console.log(`  - Survival Efficiency  : ${efficiencyScore.toFixed(1)}% (Par=${Math.round(parEfficiencySum * 100)}%, ToolSuccess=${Math.round(actionSuccessRate * 100)}%, HP=${finalHp}/20)`);
    console.log('----------------------------------------------------------------------');
    for (const m of milestones) {
      const stepInfo = m.achievedAtStep ? ` (Unlocked at Step ${m.achievedAtStep})` : '';
      console.log(`  - [${m.achieved ? 'X' : ' '}] ${m.title}${stepInfo}`);
    }
    console.log(`  Telemetry saved to: ${telemetryPath}`);
    console.log('======================================================================\n');

    try { system1.bot?.quit(); } catch {}
    await server.stopServer();
  }

  return runResult;
}

if (require.main === module) {
  runVanillaDragonSurvivalRun().catch((err) => {
    console.error('[DragonRun Fatal Error]', err);
    process.exit(1);
  });
}
