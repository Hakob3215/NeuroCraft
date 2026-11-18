import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import { MINEBENCH_ARENA_GAMERULES, MinebenchTaskSpec } from './benchmarkSpec';

export const MOJANG_1_20_4_SERVER_URL =
  'https://piston-data.mojang.com/v1/objects/8dd1a28015f51b1803213892b50b7b4fc76e594d/server.jar';
export const MOJANG_1_20_4_SERVER_SHA1 = '8dd1a28015f51b1803213892b50b7b4fc76e594d';

function resolveJavaExecutable(): string {
  const candidates = [
    process.env.JAVA_HOME ? path.join(process.env.JAVA_HOME, 'bin', 'java.exe') : '',
    'C:\\Program Files\\Eclipse Adoptium\\jdk-21.0.12.101-hotspot\\bin\\java.exe',
  ].filter(Boolean);

  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }

  const adoptiumRoot = 'C:\\Program Files\\Eclipse Adoptium';
  if (fs.existsSync(adoptiumRoot)) {
    try {
      const subdirs = fs.readdirSync(adoptiumRoot);
      for (const d of subdirs) {
        const p = path.join(adoptiumRoot, d, 'bin', 'java.exe');
        if (fs.existsSync(p)) return p;
      }
    } catch {}
  }

  return 'java';
}

/**
 * Headless Minecraft 1.20.4 Dedicated Server Manager & Tick-Rate Controller.
 * Runs the real Mojang 1.20.4 `server.jar` in nogui mode with `/tick freeze` and `/tick rate`.
 */
export class HeadlessMinecraftServerManager {
  public readonly serverDir: string;
  public readonly jarPath: string;
  private proc: ChildProcessWithoutNullStreams | null = null;
  public isReady = false;
  private logListeners: Array<(line: string) => void> = [];

  constructor(baseDir = path.resolve(process.cwd(), 'minebench', 'server')) {
    this.serverDir = baseDir;
    this.jarPath = path.join(this.serverDir, 'server.jar');
  }

  public onServerLog(listener: (line: string) => void): void {
    this.logListeners.push(listener);
  }

  public async ensureServerJarDownloaded(): Promise<string> {
    fs.mkdirSync(this.serverDir, { recursive: true });

    const sharedJar = path.resolve(process.cwd(), 'minebench', 'server', 'server.jar');
    if (this.jarPath !== sharedJar && fs.existsSync(sharedJar)) {
      if (!fs.existsSync(this.jarPath)) {
        fs.copyFileSync(sharedJar, this.jarPath);
      }
      return this.jarPath;
    }

    if (fs.existsSync(this.jarPath)) {
      const hash = crypto
        .createHash('sha1')
        .update(fs.readFileSync(this.jarPath))
        .digest('hex');
      if (hash.toLowerCase() === MOJANG_1_20_4_SERVER_SHA1) {
        return this.jarPath;
      }
    }

    console.log(`[MINEbench Server] Downloading official Minecraft 1.20.4 server.jar (~46.9 MB)...`);
    const res = await fetch(MOJANG_1_20_4_SERVER_URL);
    if (!res.ok) {
      throw new Error(`Failed to download 1.20.4 server.jar: HTTP ${res.status}`);
    }
    const buf = Buffer.from(await res.arrayBuffer());
    fs.writeFileSync(this.jarPath, buf);
    console.log(`[MINEbench Server] Verified & saved server.jar to ${this.jarPath}`);
    return this.jarPath;
  }

  public writeServerConfig(port = 25565, levelType = 'minecraft\\:flat', levelName = 'world'): void {
    fs.mkdirSync(this.serverDir, { recursive: true });
    fs.writeFileSync(path.join(this.serverDir, 'eula.txt'), 'eula=true\n', 'utf-8');
    const props = [
      `server-port=${port}`,
      `allow-flight=true`,
      `level-name=${levelName}`,
      `online-mode=false`,
      `spawn-protection=0`,
      `level-type=${levelType}`,
      `gamemode=survival`,
      `difficulty=normal`,
      `enable-command-block=true`,
      `max-players=20`,
      `view-distance=6`,
      `simulation-distance=6`,
      `max-tick-time=-1`,
      `enforce-secure-profile=false`,
      `motd=MINEbench Headless 1.20.4 RL Server`,
    ].join('\n');
    fs.writeFileSync(path.join(this.serverDir, 'server.properties'), props + '\n', 'utf-8');
  }

  public async startServer(
    port = 25565,
    timeoutMs = 120000,
    levelType = 'minecraft\\:flat',
    levelName = 'world'
  ): Promise<void> {
    await this.ensureServerJarDownloaded();
    this.writeServerConfig(port, levelType, levelName);
    const javaExe = resolveJavaExecutable();
    console.log(`[MINEbench Server] Launching Headless Vanilla 1.20.4 Server on port ${port} using: ${javaExe}`);

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Headless Minecraft 1.20.4 server timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      this.proc = spawn(javaExe, ['-Xmx2048M', '-Xms512M', '-jar', 'server.jar', 'nogui'], {
        cwd: this.serverDir,
      });

      this.proc.stdout.on('data', (chunk: Buffer) => {
        const text = chunk.toString();
        const lines = text.split(/\r?\n/).filter(Boolean);
        for (const line of lines) {
          if (line.includes('moved too quickly!') || line.includes('moved wrongly!')) {
            continue;
          }
          for (const l of this.logListeners) {
            try {
              l(line);
            } catch {}
          }
          if (line.includes('Done (') && line.includes('For help, type "help"')) {
            this.isReady = true;
            clearTimeout(timer);
            for (const rule of MINEBENCH_ARENA_GAMERULES) {
              this.sendCommand(rule);
            }
            this.sendCommand('time set noon');
            this.sendCommand('weather clear');
            console.log(`[MINEbench Server] Headless 1.20.4 Server is ONLINE and ready!`);
            resolve();
          }
        }
      });

      this.proc.stderr.on('data', (chunk: Buffer) => {
        const line = chunk.toString().trim();
        if (line && !line.includes('Warning')) {
          console.warn(`[Server Stderr] ${line}`);
        }
      });

      this.proc.on('exit', (code, signal) => {
        this.isReady = false;
        this.proc = null;
        if (code !== 0 && code !== null) {
          console.warn(`[MINEbench Server] Server process exited with code ${code}, signal ${signal}`);
        }
      });

      this.proc.on('error', (err) => {
        clearTimeout(timer);
        this.isReady = false;
        this.proc = null;
        reject(err);
      });
    });
  }

  public sendCommand(cmd: string): void {
    if (!this.proc || !this.isReady) return;
    const clean = cmd.replace(/^\//, '').trim();
    this.proc.stdin.write(clean + '\n');
  }

  /**
   * Freezes server gameplay ticks while the Local LLM / Gemini Planner is thinking
   * so 0 game ticks pass during inference.
   */
  public freezeTicks(): void {
    this.sendCommand('tick freeze');
  }

  /**
   * Unfreezes server ticks and sets TPS rate (default 40 TPS = 2x real-time speed so Mineflayer physics stay synced).
   */
  public unfreezeAndAccelerateTicks(tps = 40): void {
    this.sendCommand('tick unfreeze');
    this.sendCommand(`tick rate ${Math.max(20, Math.min(200, tps))}`);
  }

  /**
   * Atomically resets and constructs the task's 3D arena on the live headless 1.20.4 server.
   */
  public async resetArenaOnLiveServer(
    task: MinebenchTaskSpec,
    botUsername = 'NeuroAgent',
    sparringConfig?: {
      username: string;
      mode: 'active' | 'parked';
      spawnOffset?: { dx: number; dy: number; dz: number };
      loadout?: Array<{ name: string; count: number }>;
    },
    scopedCleanup = false
  ): Promise<void> {
    if (!scopedCleanup) {
      this.sendCommand('tick unfreeze');
      this.sendCommand('tick rate 20');
      for (const rule of MINEBENCH_ARENA_GAMERULES) {
        this.sendCommand(rule);
      }
      this.sendCommand('difficulty normal');
      this.sendCommand('time set noon');
      this.sendCommand('weather clear');
    }
    this.sendCommand(`op ${botUsername}`);
    this.sendCommand(`gamemode creative ${botUsername}`);

    const { x, y, z } = task.origin;

    // Scoped cleanup within 128m radius so parallel shards and Persistent Survival Sector (X=20000) are untouched
    this.sendCommand(`kill @e[type=minecraft:experience_orb,x=${x},y=${y},z=${z},distance=..128]`);
    this.sendCommand(`kill @e[type=minecraft:item,x=${x},y=${y},z=${z},distance=..128]`);
    this.sendCommand(`kill @e[type=!minecraft:player,x=${x},y=${y},z=${z},distance=..128]`);
    await new Promise((r) => setTimeout(r, 50));

    this.sendCommand(`clear ${botUsername}`);
    this.sendCommand(`effect clear ${botUsername}`);
    this.sendCommand(`effect give ${botUsername} minecraft:instant_health 1 10 true`);
    this.sendCommand(`effect give ${botUsername} minecraft:saturation 1 10 true`);

    // Move bot (in creative mode) first to force chunk loading, then place temporary safety floor
    this.sendCommand(`tp ${botUsername} ${x} ${y} ${z} 0 0`);
    await new Promise((r) => setTimeout(r, 120));
    this.sendCommand(`fill ${x - 2} ${y - 1} ${z - 2} ${x + 2} ${y - 1} ${z + 2} minecraft:stone`);

    // Execute the task's procedural 1.20.4 arena commands
    const cmds = task.buildArenaCommands(task.origin);
    for (let cmd of cmds) {
      if (botUsername !== 'NeuroAgent') {
        cmd = cmd.replace(/\bNeuroAgent\b/g, botUsername).replace(/@p\b/g, botUsername);
      }
      this.sendCommand(cmd);
    }
    await new Promise((r) => setTimeout(r, 150));

    // Give initial loadout items
    for (const item of task.initialLoadout) {
      this.sendCommand(`give ${botUsername} minecraft:${item.name} ${item.count}`);
    }

    this.sendCommand(`gamemode survival ${botUsername}`);

    if (sparringConfig) {
      const opp = sparringConfig.username;
      this.sendCommand(`clear ${opp}`);
      this.sendCommand(`effect clear ${opp}`);
      if (sparringConfig.mode === 'parked') {
        this.sendCommand(`gamemode spectator ${opp}`);
        this.sendCommand(`tp ${opp} ${x} 200 ${z}`);
      } else {
        const dx = sparringConfig.spawnOffset?.dx ?? 8;
        const dy = sparringConfig.spawnOffset?.dy ?? 0;
        const dz = sparringConfig.spawnOffset?.dz ?? 0;
        const ox = x + dx;
        const oy = y + dy;
        const oz = z + dz;
        this.sendCommand(`fill ${ox - 1} ${oy - 1} ${oz - 1} ${ox + 1} ${oy - 1} ${oz + 1} minecraft:stone`);
        this.sendCommand(`effect give ${opp} minecraft:instant_health 1 10 true`);
        this.sendCommand(`effect give ${opp} minecraft:saturation 1 10 true`);
        this.sendCommand(`tp ${opp} ${ox} ${oy} ${oz} 180 0`);
        if (sparringConfig.loadout) {
          for (const item of sparringConfig.loadout) {
            this.sendCommand(`give ${opp} minecraft:${item.name} ${item.count}`);
          }
        }
        this.sendCommand(`gamemode survival ${opp}`);
      }
    }

    await new Promise((r) => setTimeout(r, 80));
  }

  public pruneMovementSpamFromLog(): void {
    const logFile = path.join(this.serverDir, 'logs', 'latest.log');
    if (!fs.existsSync(logFile)) return;
    try {
      const content = fs.readFileSync(logFile, 'utf-8');
      const filtered = content
        .split(/\r?\n/)
        .filter((l) => !l.includes('moved too quickly!') && !l.includes('moved wrongly!'))
        .join('\n');
      fs.writeFileSync(logFile, filtered, 'utf-8');
    } catch {}
  }

  public async stopServer(): Promise<void> {
    if (!this.proc) return;
    try {
      this.sendCommand('stop');
    } catch {}
    await new Promise((r) => setTimeout(r, 2000));
    if (this.proc && !this.proc.killed) {
      this.proc.kill();
    }
    this.pruneMovementSpamFromLog();
    this.proc = null;
    this.isReady = false;
  }
}
