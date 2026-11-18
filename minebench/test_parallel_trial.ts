import * as fs from 'fs';
import * as path from 'path';
import { HeadlessMinecraftServerManager } from './headlessServer';
import { System1BotController } from '../src/system1/botController';
import { MINEBENCH_TASKS } from './benchmarkSpec';
import { DynamicMCPToolRegistry } from '../src/system2/mcp/toolRegistry';
import { NeuroCraftMCPServer } from '../src/system2/mcp/mcpServer';
import { registerBaseMCPTools } from '../src/system2/skills/baseSkills';
import { UnifiedLLMClient } from '../src/system2/llmClient';

async function testParallelTrial(workerCount: number) {
  console.log(`\n======================================================`);
  console.log(`  TRIAL & ERROR: Testing ${workerCount} Parallel Headless Workers`);
  console.log(`======================================================\n`);

  process.env.DISABLE_FAST_INTENT = 'true';
  process.env.STRICT_LLM_BENCHMARK = 'true';
  process.env.NEURAL_FORCE_LOCAL_SLM = 'true';
  process.env.LLM_PROVIDER = 'local_gpu';

  const basePort = 25575;
  const servers: HeadlessMinecraftServerManager[] = [];
  const controllers: System1BotController[] = [];

  const t0 = Date.now();

  try {
    for (let i = 0; i < workerCount; i++) {
      const port = basePort + i;
      const workerDir = path.resolve(process.cwd(), 'minebench', 'servers', `worker_${i}`);
      fs.mkdirSync(workerDir, { recursive: true });
      const server = new HeadlessMinecraftServerManager(workerDir);
      servers.push(server);

      console.log(`[Worker ${i}] Starting server on port ${port}...`);
      await server.startServer(port);
      console.log(`[Worker ${i}] Server is online! Connecting bot NeuroAgent_${i}...`);

      const controller = new System1BotController();
      controllers.push(controller);
      await controller.connect('127.0.0.1', port, `NeuroAgent_${i}`);
      console.log(`[Worker ${i}] Bot connected successfully!`);
    }

    console.log(`\nAll ${workerCount} workers booted in ${((Date.now() - t0) / 1000).toFixed(1)}s!`);
    console.log(`Running concurrent tasks across all workers...`);

    const taskRunners = controllers.map(async (ctl, i) => {
      const server = servers[i];
      const task = MINEBENCH_TASKS[i % MINEBENCH_TASKS.length];
      console.log(`[Worker ${i}] Resetting arena for task: ${task.id}...`);
      await server.resetArenaOnLiveServer(task, `NeuroAgent_${i}`);

      const registry = new DynamicMCPToolRegistry();
      registerBaseMCPTools(registry);
      const mcpServer = new NeuroCraftMCPServer(registry);
      const llm = new UnifiedLLMClient();

      server.unfreezeAndAccelerateTicks(80);
      const worldState = ctl.getLatestWorldState();
      const schemas = registry.listSchemas();

      console.log(`[Worker ${i}] Requesting SLM decision...`);
      const decT0 = Date.now();
      const decision = await llm.generateCognitiveDecision(`Worker_${i}`, task.prompt, worldState, schemas, []);
      const decMs = Date.now() - decT0;
      console.log(`[Worker ${i}] SLM decision received in ${decMs}ms:`, decision.toolCall?.name);

      return { workerId: i, taskId: task.id, decMs, toolName: decision.toolCall?.name };
    });

    const results = await Promise.all(taskRunners);
    console.log(`\nConcurrent test results:`, results);
    console.log(`\nSUCCESS: ${workerCount} parallel workers executed cleanly!`);
    return true;
  } catch (err: any) {
    console.error(`\nFAILED for ${workerCount} workers:`, err.message);
    return false;
  } finally {
    console.log(`Shutting down test workers...`);
    for (let i = 0; i < controllers.length; i++) {
      try { controllers[i].bot?.quit(); } catch {}
    }
    for (let i = 0; i < servers.length; i++) {
      try { await servers[i].stopServer(); } catch {}
    }
  }
}

async function main() {
  const count = process.argv[2] ? parseInt(process.argv[2], 10) : 2;
  const ok = await testParallelTrial(count);
  process.exit(ok ? 0 : 1);
}

main();
