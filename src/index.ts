import * as readline from 'readline';
import { CONFIG } from './config';
import { discoverMinecraftLanWorld } from './discovery/lanScanner';
import { System1BotController } from './system1/botController';
import { CognitiveController } from './system2/cognitiveController';
import { installGlobalFileLogger } from './logger';

async function main(): Promise<void> {
  const logPaths = installGlobalFileLogger();
  console.log('╔════════════════════════════════════════════════════════════════════╗');
  console.log('║   NEUROCRAFT: Real-Time Human-Mimicking Minecraft AI Agent         ║');
  console.log('║   - System 1: 20Hz Reflexes, Bézier Camera, 90° FOV & Raycasting   ║');
  console.log('║   - System 2: Meta-Agent & Self-Synthesizing Dynamic MCP Server    ║');
  console.log('║   - Mode: 100% Vanilla Singleplayer LAN (Zero Extra Account)       ║');
  console.log('╚════════════════════════════════════════════════════════════════════╝');
  console.log(`[Logger] Full telemetry & LLM trace dumping to: ${logPaths.latestLogPath}`);

  // Parse optional CLI flags: --port <num> --host <ip> --name <username>
  const args = process.argv.slice(2);
  let targetHost = CONFIG.minecraftHost;
  let targetPort = CONFIG.minecraftPort;
  let botName = CONFIG.botUsername;

  for (let i = 0; i < args.length; i++) {
    if ((args[i] === '--port' || args[i] === '-p') && args[i + 1]) {
      targetPort = parseInt(args[i + 1], 10);
      i++;
    } else if ((args[i] === '--host' || args[i] === '-h') && args[i + 1]) {
      targetHost = args[i + 1];
      i++;
    } else if ((args[i] === '--name' || args[i] === '-n') && args[i + 1]) {
      botName = args[i + 1];
      i++;
    }
  }

  let cognitiveController: CognitiveController | null = null;

  const system1 = new System1BotController(
    (speaker: string, message: string) => {
      if (cognitiveController) {
        cognitiveController.handlePlayerChat(speaker, message).catch((err) => {
          console.warn(`[Main] Chat handler error: ${err.message}`);
        });
      }
    },
    (reflexEvent) => {
      if (cognitiveController) {
        cognitiveController.handleReflexInterrupt(reflexEvent);
      }
    },
    (subtask) => {
      if (cognitiveController) {
        cognitiveController.handleSubtaskProgress(subtask);
      }
    }
  );

  cognitiveController = new CognitiveController(system1);

  if (!targetPort || targetPort <= 0) {
    const discovered = await discoverMinecraftLanWorld(300000); // Wait up to 5 minutes for user to click Open to LAN
    targetHost = discovered.host;
    targetPort = discovered.port;
    console.log(
      `\n[LAN Scanner] Found LAN World: "${discovered.motd}" at ${targetHost}:${targetPort}!`
    );
  }

  await system1.connect(targetHost, targetPort, botName);

  // Perform an initial 360° human gaze sweep on spawn to populate spatial terrain & resource clusters
  await system1.perform360EnvironmentScan(140);

  // Send humanized greeting once spawned in world
  await cognitiveController.chatEngine.sayHumanized(
    `Hey! ${botName} reporting in. Tell me what to do in chat (or ask for something unique and I'll synthesize a new MCP skill for it)!`
  );

  // Also allow typing commands directly in the console window
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log('\n[Console Ready] Talk to the agent in Minecraft chat, or type a command right here:');
  rl.on('line', (line) => {
    const text = line.trim();
    if (!text) return;
    cognitiveController?.handlePlayerChat('ConsolePlayer', text);
  });
}

if (require.main === module) {
  main().catch((err) => {
    console.error(`\n[NeuroCraft Error] ${err.message}`);
    process.exit(1);
  });
}
