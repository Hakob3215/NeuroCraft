import { AuditoryCue, Vector3D } from '../../types';
import { CONFIG } from '../../config';

/**
 * Generates a human perceptual reaction delay in milliseconds modeled by a Log-Normal distribution:
 *   t_react ~ LogNormal(mu = ln(medianMs), sigma)
 * For unexpected stimuli, human visual/auditory reaction time has a median of ~220ms
 * with a right-skewed tail (160ms - 340ms).
 */
export function sampleHumanReactionDelayMs(
  medianMs = CONFIG.perception.reactionMedianMs,
  sigma = CONFIG.perception.reactionSigma
): number {
  // Box-Muller transform for standard normal Z ~ N(0, 1)
  const u1 = Math.max(1e-7, Math.random());
  const u2 = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);

  const mu = Math.log(medianMs);
  const sample = Math.exp(mu + sigma * z);

  // Clamp to physiological human bounds [110ms, 550ms]
  return Math.max(110, Math.min(550, Math.round(sample)));
}

/**
 * Manages spatial sound cues so the agent can react to footsteps, explosions, or creeper hisses
 * behind its back without cheating via raw entity coordinate packets.
 */
export class AuditoryPerceptionTracker {
  private cues: AuditoryCue[] = [];
  private readonly maxAgeMs = 4000;

  public registerSound(
    soundName: string,
    botPos: Vector3D,
    soundPos: Vector3D,
    maxDistance = CONFIG.perception.maxHearingDistance
  ): void {
    const dx = soundPos.x - botPos.x;
    const dy = soundPos.y - botPos.y;
    const dz = soundPos.z - botPos.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (dist > maxDistance && !soundName.includes('explode') && !soundName.includes('thunder')) {
      return;
    }

    const norm = dist > 1e-4 ? dist : 1;
    // Add slight human directional estimation noise (+/- 8%)
    const jitter = () => (Math.random() - 0.5) * 0.16;

    const dirX = dx / norm + jitter();
    const dirY = dy / norm + jitter();
    const dirZ = dz / norm + jitter();
    const dirLen = Math.sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ) || 1;

    this.cues.push({
      soundName,
      directionVector: {
        x: Number((dirX / dirLen).toFixed(2)),
        y: Number((dirY / dirLen).toFixed(2)),
        z: Number((dirZ / dirLen).toFixed(2)),
      },
      estimatedDistance: Math.round(dist),
      timestamp: Date.now(),
    });

    if (this.cues.length > 25) {
      this.cues.shift();
    }
  }

  public getRecentCues(now = Date.now()): AuditoryCue[] {
    this.cues = this.cues.filter((c) => now - c.timestamp <= this.maxAgeMs);
    return [...this.cues];
  }
}
