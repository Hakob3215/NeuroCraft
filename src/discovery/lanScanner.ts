import * as dgram from 'dgram';
import * as net from 'net';

export interface DiscoveredLanWorld {
  motd: string;
  host: string;
  port: number;
}

const LAN_MULTICAST_ADDR = '224.0.2.60';
const LAN_UDP_PORT = 4445;

/**
 * Parses a raw Minecraft UDP LAN broadcast payload:
 * Format: [MOTD]Player - World Name[/MOTD][AD]54321[/AD]
 */
export function parseLanBroadcastPacket(payload: string, senderAddress = '127.0.0.1'): DiscoveredLanWorld | null {
  const motdMatch = payload.match(/\[MOTD\]([\s\S]*?)\[\/MOTD\]/);
  const portMatch = payload.match(/\[AD\](\d+)\[\/AD\]/);

  if (!portMatch) {
    return null;
  }

  const port = parseInt(portMatch[1], 10);
  if (isNaN(port) || port <= 0 || port > 65535) {
    return null;
  }

  const motd = motdMatch ? motdMatch[1].trim() : 'Minecraft LAN World';
  // If broadcast came from local machine interface, prefer 127.0.0.1 for lowest latency
  const host = senderAddress.startsWith('127.') || senderAddress === '0.0.0.0' ? '127.0.0.1' : senderAddress;

  return { motd, host, port };
}

/**
 * Checks if a TCP port is currently accepting connections on the given host.
 */
export function probeTcpPort(host: string, port: number, timeoutMs = 600): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let settled = false;

    const finish = (ok: boolean) => {
      if (!settled) {
        settled = true;
        socket.destroy();
        resolve(ok);
      }
    };

    socket.setTimeout(timeoutMs);
    socket.once('connect', () => finish(true));
    socket.once('timeout', () => finish(false));
    socket.once('error', () => finish(false));

    socket.connect(port, host);
  });
}

/**
 * Listens for Minecraft's native UDP LAN broadcast on 224.0.2.60:4445
 * while also probing standard port 25565.
 */
export async function discoverMinecraftLanWorld(timeoutMs = 120000): Promise<DiscoveredLanWorld> {
  // First quick check: is localhost:25565 already running a server?
  const isDefaultPortOpen = await probeTcpPort('127.0.0.1', 25565, 400);
  if (isDefaultPortOpen) {
    return {
      motd: 'Local Minecraft Server (25565)',
      host: '127.0.0.1',
      port: 25565,
    };
  }

  return new Promise((resolve, reject) => {
    const socket = dgram.createSocket({ type: 'udp4', reuseAddr: true });
    let resolved = false;

    const cleanup = () => {
      try {
        socket.close();
      } catch {
        // Ignore close errors
      }
    };

    const timer = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        cleanup();
        reject(new Error(`Timed out after ${timeoutMs / 1000}s waiting for Minecraft "Open to LAN" broadcast.`));
      }
    }, timeoutMs);

    socket.on('error', (err) => {
      if (!resolved) {
        console.warn(`[LAN Scanner] UDP socket warning: ${err.message}`);
      }
    });

    socket.on('message', (msg, rinfo) => {
      if (resolved) return;
      const text = msg.toString('utf8');
      const parsed = parseLanBroadcastPacket(text, rinfo.address);
      if (parsed) {
        resolved = true;
        clearTimeout(timer);
        cleanup();
        // Connect via 127.0.0.1 when running on the same PC for zero network hop overhead
        resolve({
          motd: parsed.motd,
          host: '127.0.0.1',
          port: parsed.port,
        });
      }
    });

    socket.bind(LAN_UDP_PORT, () => {
      try {
        socket.addMembership(LAN_MULTICAST_ADDR);
      } catch (e: any) {
        console.warn(`[LAN Scanner] Multicast membership note: ${e.message}`);
      }
      console.log('======================================================================');
      console.log(' [NeuroCraft] Listening for Minecraft Singleplayer LAN Broadcast...');
      console.log('   -> UDP Multicast: 224.0.2.60:4445');
      console.log('   -> In Minecraft Singleplayer: Press [Esc] -> [Open to LAN]');
      console.log('   -> Click [Start LAN World] and NeuroAgent will join within 2s!');
      console.log('======================================================================');
    });
  });
}
