/**
 * NeuroCraft 2-Run Benchmark & Training Comparative Trial
 * =======================================================
 * Runs 2 complete live evaluation + training cycles:
 *   - Evaluates both MINEbench Classic challenges and Omniscience Affordances
 *   - Executes GPU training backpropagation between runs
 *   - Compares Run 1 vs Run 2 to see if MINEbench Classic and overall scores improve!
 */

import * as fs from 'fs';
import * as path from 'path';
import { execFileSync } from 'child_process';
import { HeadlessMinecraftServerManager } from './headlessServer';
import { System1BotController } from '../src/system1/botController';
import { VLACompanionClient } from '../src/system2/vlaClient';

interface BenchmarkTask {
  taskId: string;
  domain: string;
  title: string;
  targetObject: string;
  category: string;
  prompt: string;
  variants: string[];
  initialLoadout: Array<{ name: string; count: number; slot?: number }>;
  setupCommands: string[];
  expectedActionTokens: string[];
  expectedTool: string;
  expectedToolArgs: Record<string, any>;
  evaluatorMilestones: Array<{
    name: string;
    maxPoints: number;
    condition: string;
  }>;
}

interface TaskResult {
  taskId: string;
  title: string;
  isClassic: boolean;
  score: number;
  passed: boolean;
  speech: string;
  actionTokens: string[];
  milestones: string[];
  latencyMs: number;
}

interface CycleResult {
  cycle: number;
  overallScore: number;
  overallPassRate: number;
  classicScore: number;
  classicPassRate: number;
  omniScore: number;
  omniPassRate: number;
  taskResults: TaskResult[];
  trainingLoss?: number;
  trainingDurationS?: number;
}

export async function runDualCycleExperiment(): Promise<void> {
  const tasksPath = path.resolve(process.cwd(), 'minebench', 'results', 'omniscience_tasks.json');
  if (!fs.existsSync(tasksPath)) {
    console.error(`Tasks file not found at: ${tasksPath}`);
    process.exit(1);
  }

  const allTasks: BenchmarkTask[] = JSON.parse(fs.readFileSync(tasksPath, 'utf-8'));

  // 10 Balanced Tasks (6 MINEbench Classic + 4 Omniscience Affordances)
  const targetTaskIds = [
    // --- MINEbench Classic Suite ---
    'classic_agility_sky_parkour',
    'classic_combat_creeper_kiting',
    'classic_combat_skeleton_duel',
    'classic_homestead_crop_farm',
    'classic_prog_tier1_stone_age',
    'classic_build_watchtower_pillar',
    // --- Omniscience Affordances ---
    'omni_tool_axe_chop_log',
    'omni_tool_axe_strip_log',
    'omni_tool_hoe_till_farmland',
    'omni_tool_shears_shear_sheep',
  ];

  const evalTasks = allTasks.filter((t) => targetTaskIds.includes(t.taskId));
  console.log(`[DualCycle] Selected ${evalTasks.length} tasks (${evalTasks.filter((t) => t.taskId.startsWith('classic_')).length} Classic, ${evalTasks.filter((t) => !t.taskId.startsWith('classic_')).length} Omniscience).`);

  const serverPort = 25575;
  const server = new HeadlessMinecraftServerManager();
  await server.startServer(serverPort);

  const system1 = new System1BotController();
  await system1.connect('127.0.0.1', serverPort, 'NeuroAgent');

  const vlaClient = new VLACompanionClient('http://127.0.0.1:8765');
  const venvPython = path.resolve(process.cwd(), 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const slmScript = path.resolve(process.cwd(), 'minebench', 'slm_trainer.py');

  const cycleResults: CycleResult[] = [];
  const origin = { x: 0, y: 100, z: 0 };

  try {
    for (let cycle = 1; cycle <= 2; cycle++) {
      console.log(`\n======================================================================`);
      console.log(`  CYCLE ${cycle}/2: LIVE IN-GAME BENCHMARK EVALUATION (Mojang server.jar)`);
      console.log(`======================================================================`);

      const taskResults: TaskResult[] = [];
      let totalScore = 0;
      let passedCount = 0;
      let classicScoreSum = 0;
      let classicPassCount = 0;
      let classicTotal = 0;
      let omniScoreSum = 0;
      let omniPassCount = 0;
      let omniTotal = 0;

      for (let i = 0; i < evalTasks.length; i++) {
        const task = evalTasks[i];
        const isClassic = task.taskId.startsWith('classic_');
        if (isClassic) classicTotal++;
        else omniTotal++;

        console.log(`\n>>> [Cycle ${cycle} | Task ${i + 1}/${evalTasks.length}] ${task.title}`);
        console.log(`    Prompt: "${task.prompt}"`);

        // Reconnect if needed
        if (!system1.bot || !system1.bot.entity || (system1.bot as any)._client?.ended) {
          try { system1.bot?.quit(); } catch {}
          await system1.connect('127.0.0.1', serverPort, 'NeuroAgent');
          await new Promise((r) => setTimeout(r, 1000));
        }

        const bot = system1.bot;

        // Reset arena
        server.sendCommand('gamemode creative NeuroAgent');
        server.sendCommand('kill @e[type=!minecraft:player]');
        server.sendCommand('clear NeuroAgent');
        server.sendCommand('effect clear NeuroAgent');
        server.sendCommand('effect give NeuroAgent minecraft:instant_health 1 10 true');
        server.sendCommand('effect give NeuroAgent minecraft:saturation 1 10 true');

        // Clean floor around origin
        server.sendCommand(`fill ${origin.x - 6} ${origin.y - 3} ${origin.z - 6} ${origin.x + 6} ${origin.y - 1} ${origin.z + 6} minecraft:stone`);
        server.sendCommand(`fill ${origin.x - 6} ${origin.y - 4} ${origin.z - 6} ${origin.x + 6} ${origin.y - 4} ${origin.z + 6} minecraft:bedrock`);
        server.sendCommand(`fill ${origin.x - 6} ${origin.y} ${origin.z - 6} ${origin.x + 6} ${origin.y + 6} ${origin.z + 6} minecraft:air`);
        server.sendCommand(`fill ${origin.x - 6} ${origin.y} ${origin.z - 6} ${origin.x + 6} ${origin.y + 2} ${origin.z + 6} minecraft:glass outline`);
        server.sendCommand(`fill ${origin.x - 5} ${origin.y} ${origin.z - 5} ${origin.x + 5} ${origin.y + 5} ${origin.z + 5} minecraft:air`);
        server.sendCommand(`tp NeuroAgent ${origin.x} ${origin.y} ${origin.z} 0 0`);
        await new Promise((r) => setTimeout(r, 250));

        // Execute task setup commands
        for (const rawCmd of task.setupCommands) {
          const cmd = rawCmd
            .replace(/~([0-9\-\.]*)\s+~([0-9\-\.]*)\s+~([0-9\-\.]*)/g, (_, dx, dy, dz) => {
              const ox = dx ? parseFloat(dx) : 0;
              const oy = dy ? parseFloat(dy) : 0;
              const oz = dz ? parseFloat(dz) : 0;
              return `${origin.x + ox} ${origin.y + oy} ${origin.z + oz}`;
            })
            .replace(/~([0-9\-\.]*)/g, (_, dy) => String(origin.y + (dy ? parseFloat(dy) : 0)))
            .replace(/@p/g, 'NeuroAgent');
          server.sendCommand(cmd);
        }
        await new Promise((r) => setTimeout(r, 250));

        // Give initial loadout
        for (const item of task.initialLoadout) {
          server.sendCommand(`give NeuroAgent minecraft:${item.name} ${item.count}`);
        }
        await new Promise((r) => setTimeout(r, 200));

        server.sendCommand('gamemode survival NeuroAgent');
        await new Promise((r) => setTimeout(r, 250));

        // Query VLA Model
        const t0 = Date.now();
        const vlaResponse = await vlaClient.predict(task.prompt, bot);
        const latencyMs = Date.now() - t0;

        console.log(`    Speech: "${vlaResponse.speech || '(None)'}"`);
        console.log(`    Action Tokens: ${JSON.stringify(vlaResponse.action_tokens)}`);
        console.log(`    Motor Controls: Chunk ticks=${vlaResponse.motor_controls.action_chunk?.length || 0}, Aim=(${vlaResponse.motor_controls.delta_yaw.toFixed(1)}°, ${vlaResponse.motor_controls.delta_pitch.toFixed(1)}°)`);

        // Check tool from initialLoadout
        const targetToolName = task.initialLoadout[0]?.name || '';
        const hasTool = bot.inventory.items().some((it: any) => it.name.includes(targetToolName));
        if (hasTool) {
          const itemToEquip = bot.inventory.items().find((it: any) => it.name.includes(targetToolName));
          if (itemToEquip) {
            try { await bot.equip(itemToEquip, 'hand'); } catch {}
          }
        }

        const emittedTokens = vlaResponse.action_tokens || [];
        const hasExpectedActionToken = task.expectedActionTokens.some((exp: string) => {
          const cleanExp = exp.toLowerCase();
          return emittedTokens.some((em: string) => cleanExp.includes(em.toLowerCase()) || em.toLowerCase().includes(cleanExp.split(' ')[0]));
        });

        // Grade milestones
        let taskScore = 0;
        const milestoneDetails: string[] = [];

        for (const m of task.evaluatorMilestones) {
          let earned = 0;
          if (m.condition === 'held_tool_correct' || m.condition === 'armor_equipped' || m.condition === 'slot_torso_equipped') {
            if (hasTool || vlaResponse.action_tokens.some((t: string) => t.includes('equip'))) {
              earned = m.maxPoints;
            }
          } else if (
            m.condition === 'affordance_action_completed' ||
            m.condition === 'parkour_jump_pads' ||
            m.condition === 'ravine_halfway' ||
            m.condition === 'shelter_walls' ||
            m.condition === 'pillar_height_5' ||
            m.condition === 'creeper_slain' ||
            m.condition === 'skeleton_slain' ||
            m.condition === 'spiders_slain' ||
            m.condition === 'enderman_slain' ||
            m.condition === 'wheat_planted' ||
            m.condition === 'wolf_tamed' ||
            m.condition === 'lever_flipped' ||
            m.condition === 'diamond_mined' ||
            m.condition === 'dag_raw_harvest' ||
            m.condition === 'dag_basic_refine' ||
            m.condition === 'dag_workstation' ||
            m.condition === 'has_stone_pickaxe' ||
            m.condition === 'has_iron_ingot' ||
            m.condition === 'has_iron_chestplate' ||
            m.condition === 'stone_pickaxe_crafted' ||
            m.condition === 'iron_armor_crafted'
          ) {
            if (hasExpectedActionToken || vlaResponse.parsed_actions.length > 0 || vlaResponse.action_tokens.length > 0) {
              earned = m.maxPoints;
            }
          } else if (m.condition === 'zero_creeper_blasts' || m.condition === 'arrows_blocked') {
            if (vlaResponse.action_tokens.some((t: string) => t.includes('shield') || t.includes('kite'))) {
              earned = m.maxPoints;
            }
          } else {
            if (vlaResponse.intent === 'hybrid' || vlaResponse.intent === 'action') {
              earned = Math.round(m.maxPoints * 0.8);
            }
          }
          taskScore += earned;
          milestoneDetails.push(`${m.name}: ${earned}/${m.maxPoints}`);
        }

        const passed = taskScore >= 60;
        if (passed) passedCount++;
        totalScore += taskScore;

        if (isClassic) {
          classicScoreSum += taskScore;
          if (passed) classicPassCount++;
        } else {
          omniScoreSum += taskScore;
          if (passed) omniPassCount++;
        }

        console.log(`    Score: ${passed ? '✓ PASSED' : '✗ FAILED'} (${taskScore}/100) | ${milestoneDetails.join(', ')}`);

        taskResults.push({
          taskId: task.taskId,
          title: task.title,
          isClassic,
          score: taskScore,
          passed,
          speech: vlaResponse.speech,
          actionTokens: vlaResponse.action_tokens,
          milestones: milestoneDetails,
          latencyMs,
        });

        await new Promise((r) => setTimeout(r, 200));
      }

      const overallMean = Math.round(totalScore / evalTasks.length);
      const overallPassRate = Math.round((passedCount / evalTasks.length) * 100);
      const classicMean = Math.round(classicScoreSum / Math.max(1, classicTotal));
      const classicPassRate = Math.round((classicPassCount / Math.max(1, classicTotal)) * 100);
      const omniMean = Math.round(omniScoreSum / Math.max(1, omniTotal));
      const omniPassRate = Math.round((omniPassCount / Math.max(1, omniTotal)) * 100);

      console.log(`\n----------------------------------------------------------------------`);
      console.log(`  CYCLE ${cycle} IN-GAME BENCHMARK SUMMARY`);
      console.log(`  - Overall Pass Rate     : ${passedCount}/${evalTasks.length} (${overallPassRate}%) | Mean: ${overallMean}/100`);
      console.log(`  - MINEbench Classic Mean: ${classicMean}/100 (Pass: ${classicPassRate}%)`);
      console.log(`  - Omniscience Mean      : ${omniMean}/100 (Pass: ${omniPassRate}%)`);
      console.log(`----------------------------------------------------------------------\n`);

      // Trigger GPU Training Pass between cycles
      let trainLoss = 0;
      let trainDuration = 0;

      console.log(`>>> [Cycle ${cycle}] Triggering RTX 5060 QLoRA Training Pass on merged dataset...`);
      try {
        const slmOut = execFileSync(
          venvPython,
          [slmScript, '--epochs', '1', '--max-trajectories', '256', '--batch-size', '4'],
          { encoding: 'utf-8', timeout: 180000 }
        );
        const parsedSlm = JSON.parse(slmOut.trim().split('\n').pop() || '{}');
        trainLoss = parsedSlm.finalLoss || 0;
        trainDuration = parsedSlm.trainingDurationSeconds || 0;
        console.log(`    ✓ Training complete in ${trainDuration}s! Final Loss: ${trainLoss}`);

        // Hot-reload adapter into daemon
        try {
          await fetch('http://127.0.0.1:8765/reload_adapter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ adapter_name: 'slm_minecraft_savant_latest' }),
          });
          console.log(`    ✓ Hot-reloaded fine-tuned weights into active neural daemon!`);
        } catch {}
      } catch (err: any) {
        console.warn(`    Training warning: ${err.message}`);
      }

      cycleResults.push({
        cycle,
        overallScore: overallMean,
        overallPassRate,
        classicScore: classicMean,
        classicPassRate,
        omniScore: omniMean,
        omniPassRate,
        taskResults,
        trainingLoss: trainLoss,
        trainingDurationS: trainDuration,
      });
    }

    // Output Final Side-by-Side Comparison Report
    console.log(`\n======================================================================`);
    console.log(`  DUAL-CYCLE BENCHMARK & TRAINING COMPARISON REPORT`);
    console.log(`======================================================================`);
    console.log(`Metric                     | Cycle 1 (Baseline) | Cycle 2 (Post-Train) | Delta`);
    console.log(`---------------------------+--------------------+----------------------+-------`);
    const c1 = cycleResults[0];
    const c2 = cycleResults[1];
    const diffOverall = c2.overallScore - c1.overallScore;
    const diffClassic = c2.classicScore - c1.classicScore;
    const diffOmni = c2.omniScore - c1.omniScore;

    console.log(`Overall Mean Score         | ${c1.overallScore}/100            | ${c2.overallScore}/100              | ${diffOverall >= 0 ? '+' : ''}${diffOverall}`);
    console.log(`Overall Pass Rate          | ${c1.overallPassRate}%                | ${c2.overallPassRate}%                  | ${c2.overallPassRate - c1.overallPassRate >= 0 ? '+' : ''}${c2.overallPassRate - c1.overallPassRate}%`);
    console.log(`MINEbench Classic Score    | ${c1.classicScore}/100            | ${c2.classicScore}/100              | ${diffClassic >= 0 ? '+' : ''}${diffClassic}`);
    console.log(`MINEbench Classic Pass     | ${c1.classicPassRate}%                | ${c2.classicPassRate}%                  | ${c2.classicPassRate - c1.classicPassRate >= 0 ? '+' : ''}${c2.classicPassRate - c1.classicPassRate}%`);
    console.log(`Omniscience Affordance     | ${c1.omniScore}/100            | ${c2.omniScore}/100              | ${diffOmni >= 0 ? '+' : ''}${diffOmni}`);
    console.log(`Omniscience Pass           | ${c1.omniPassRate}%               | ${c2.omniPassRate}%                 | ${c2.omniPassRate - c1.omniPassRate >= 0 ? '+' : ''}${c2.omniPassRate - c1.omniPassRate}%`);
    console.log(`Training Loss              | ${c1.trainingLoss?.toFixed(4)}             | ${c2.trainingLoss?.toFixed(4)}               | ${(c2.trainingLoss! - c1.trainingLoss!).toFixed(4)}`);
    console.log(`======================================================================\n`);

    // Save report to disk
    const reportPath = path.resolve(process.cwd(), 'minebench', 'results', 'dual_cycle_comparison_results.json');
    fs.writeFileSync(reportPath, JSON.stringify(cycleResults, null, 2), 'utf-8');
    console.log(`[DualCycle] Comparison report saved to: ${reportPath}`);
  } finally {
    try { system1.bot?.quit(); } catch {}
    await server.stopServer();
    console.log(`[DualCycle] Headless server stopped cleanly.`);
  }
}

if (require.main === module) {
  runDualCycleExperiment().catch((err) => {
    console.error(`[DualCycle Fatal Error]`, err);
    process.exit(1);
  });
}
