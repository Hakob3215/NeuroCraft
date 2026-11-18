import * as path from 'path';
import * as fs from 'fs';
import * as dotenv from 'dotenv';

// Load .env from project root if present
const envPath = path.resolve(__dirname, '..', '.env');
if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
}

export interface NeuroCraftConfig {
  botUsername: string;
  minecraftHost: string;
  minecraftPort: number; // 0 means auto-discover via UDP LAN broadcast
  minecraftVersion: string | undefined; // undefined = auto-detect
  perception: {
    horizontalFovDeg: number;
    verticalFovDeg: number;
    maxSightDistance: number;
    maxHearingDistance: number;
    reactionMedianMs: number;
    reactionSigma: number;
  };
  kinematics: {
    maxAngularAccelDegPerSec2: number;
    microTremorAmplitudeDeg: number;
    fittsOvershootRatio: number;
    inventoryBaseDelayMs: number;
    inventoryFittsSlopeMs: number;
    typingWpm: number;
  };
  speculative: {
    prefetchProgressThreshold: number; // 0.75 (75%)
    prefetchRemainingMsThreshold: number; // 1500ms
  };
  llm: {
    provider: 'gemini' | 'openai' | 'ollama' | 'local_gpu' | 'auto';
    apiKey: string;
    model: string;
    ollamaUrl: string;
    ollamaModel: string;
    neuralDaemonUrl: string;
  };
  paths: {
    dataDir: string;
    synthesizedSkillsDir: string;
    memoryFile: string;
    skillRegistryFile: string;
  };
}

const projectRoot = path.resolve(__dirname, '..');
const dataDir = path.join(projectRoot, 'data');
const synthesizedSkillsDir = path.join(dataDir, 'synthesized_skills');

export const CONFIG: NeuroCraftConfig = {
  botUsername: process.env.BOT_USERNAME || 'NeuroAgent',
  minecraftHost: process.env.MC_HOST || '127.0.0.1',
  minecraftPort: parseInt(process.env.MC_PORT || '0', 10),
  minecraftVersion: process.env.MC_VERSION || undefined,
  perception: {
    horizontalFovDeg: 90,
    verticalFovDeg: 70,
    maxSightDistance: 48,
    maxHearingDistance: 16,
    reactionMedianMs: 220, // LogNormal(mu=5.393, sigma=0.25) -> ~220ms median
    reactionSigma: 0.25,
  },
  kinematics: {
    maxAngularAccelDegPerSec2: 1100,
    microTremorAmplitudeDeg: 0.28,
    fittsOvershootRatio: 0.04,
    inventoryBaseDelayMs: 130,
    inventoryFittsSlopeMs: 65,
    typingWpm: 68,
  },
  speculative: {
    prefetchProgressThreshold: 0.75,
    prefetchRemainingMsThreshold: 1500,
  },
  llm: {
    provider: (process.env.LLM_PROVIDER as any) || 'auto',
    apiKey: process.env.GEMINI_API_KEY || process.env.OPENAI_API_KEY || '',
    model: process.env.LLM_MODEL || 'gemini-3.8-flash',
    ollamaUrl: process.env.OLLAMA_URL || 'http://127.0.0.1:11434',
    ollamaModel: process.env.OLLAMA_MODEL || 'qwen2.5-coder:7b',
    neuralDaemonUrl: process.env.NEURAL_DAEMON_URL || 'http://127.0.0.1:8765',
  },
  paths: {
    dataDir,
    synthesizedSkillsDir,
    memoryFile: path.join(dataDir, 'episodic_memory.json'),
    skillRegistryFile: path.join(dataDir, 'synthesized_registry.json'),
  },
};
