import * as fs from 'fs';
import * as path from 'path';

const projectRoot = path.resolve(__dirname, '..');
const logsDir = path.join(projectRoot, 'logs');
const latestLogFile = path.join(logsDir, 'agent_latest.log');

let initialized = false;
let sessionLogFile = '';

function ensureLogFiles(): void {
  if (initialized) return;
  initialized = true;

  try {
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    sessionLogFile = path.join(logsDir, `agent_session_${stamp}.log`);

    const header = `================================================================================\nNEUROCRAFT FULL DIAGNOSTIC LOG\nStarted At: ${new Date().toISOString()}\nLatest Log Path: ${latestLogFile}\nSession Log Path: ${sessionLogFile}\n================================================================================\n\n`;
    fs.writeFileSync(latestLogFile, header, 'utf8');
    fs.writeFileSync(sessionLogFile, header, 'utf8');
  } catch {
    // Ignore file init error
  }
}

function appendToLogFiles(line: string): void {
  ensureLogFiles();
  try {
    fs.appendFileSync(latestLogFile, line + '\n', 'utf8');
    if (sessionLogFile) {
      fs.appendFileSync(sessionLogFile, line + '\n', 'utf8');
    }
  } catch {
    // Ignore write error
  }
}

/**
 * Writes a detailed diagnostic entry (such as full LLM prompts/responses, block dig traces,
 * world state snapshots, or tool execution details) directly into `logs/agent_latest.log`.
 */
export function logVerbose(category: string, message: string, data?: any): void {
  const ts = new Date().toISOString();
  let serialized = '';
  if (data !== undefined) {
    try {
      serialized =
        typeof data === 'string' ? `\n${data}` : `\n${JSON.stringify(data, null, 2)}`;
    } catch {
      serialized = `\n[Unserializable data]`;
    }
  }
  appendToLogFiles(`[${ts}] [VERBOSE:${category}] ${message}${serialized}`);
}

/**
 * Hooks global `console.log`, `console.warn`, and `console.error` so every single line
 * printed to the terminal is also automatically saved to `logs/agent_latest.log`.
 */
export function installGlobalFileLogger(): { latestLogPath: string; sessionLogPath: string } {
  ensureLogFiles();

  const origLog = console.log.bind(console);
  const origWarn = console.warn.bind(console);
  const origError = console.error.bind(console);

  const formatArgs = (args: any[]): string =>
    args
      .map((a) => {
        if (typeof a === 'string') return a;
        try {
          return JSON.stringify(a);
        } catch {
          return String(a);
        }
      })
      .join(' ');

  console.log = (...args: any[]) => {
    origLog(...args);
    appendToLogFiles(`[${new Date().toISOString()}] [INFO] ${formatArgs(args)}`);
  };

  console.warn = (...args: any[]) => {
    origWarn(...args);
    appendToLogFiles(`[${new Date().toISOString()}] [WARN] ${formatArgs(args)}`);
  };

  console.error = (...args: any[]) => {
    origError(...args);
    appendToLogFiles(`[${new Date().toISOString()}] [ERROR] ${formatArgs(args)}`);
  };

  process.on('uncaughtException', (err: any) => {
    const msg = err?.stack || err?.message || String(err);
    console.error(`[Process UncaughtException] ${msg}`);
  });

  process.on('unhandledRejection', (reason: any) => {
    const msg = reason?.stack || reason?.message || String(reason);
    console.error(`[Process UnhandledRejection] ${msg}`);
  });

  return { latestLogPath: latestLogFile, sessionLogPath: sessionLogFile };
}
