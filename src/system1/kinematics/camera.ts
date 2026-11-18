import { Vec3 } from 'vec3';
import { Vector3D } from '../../types';
import { calculateLookAngles, normalizeRadians } from '../perception/frustum';
import { CONFIG } from '../../config';

/**
 * Evaluates a 1D cubic Bézier curve at parameter t in [0, 1].
 */
export function evaluateCubicBezier(t: number, p0: number, p1: number, p2: number, p3: number): number {
  const clampedT = Math.max(0, Math.min(1, t));
  const u = 1 - clampedT;
  return (
    u * u * u * p0 +
    3 * u * u * clampedT * p1 +
    3 * u * clampedT * clampedT * p2 +
    clampedT * clampedT * clampedT * p3
  );
}

/**
 * Continuous 1D pseudo-Perlin noise for human hand micro-tremors (+/- 0.28 degrees).
 */
export function sampleHandTremorRadians(timeSeconds: number, seedOffset = 0): number {
  const t = timeSeconds * 11.5 + seedOffset;
  const wave1 = Math.sin(t) * 0.5;
  const wave2 = Math.sin(t * 2.37 + 1.7) * 0.35;
  const wave3 = Math.cos(t * 4.13 + 3.1) * 0.15;
  const amplitudeRad = (CONFIG.kinematics.microTremorAmplitudeDeg * Math.PI) / 180;
  return (wave1 + wave2 + wave3) * amplitudeRad;
}

export interface CameraKeyframe {
  yaw: number;
  pitch: number;
}

/**
 * Generates a sequence of 20Hz (50ms) camera angle keyframes from `(startYaw, startPitch)`
 * to `(targetYaw, targetPitch)` using cubic Bézier smoothing with Fitts' Law slight overshoot
 * on large flicks and subtle hand micro-tremor.
 */
export function generateBezierCameraTrajectory(
  startYaw: number,
  startPitch: number,
  targetYaw: number,
  targetPitch: number,
  durationMs?: number
): CameraKeyframe[] {
  const deltaYaw = normalizeRadians(targetYaw - startYaw);
  const deltaPitch = targetPitch - startPitch;
  const totalAngularDistRad = Math.sqrt(deltaYaw * deltaYaw + deltaPitch * deltaPitch);
  const totalAngularDistDeg = (totalAngularDistRad * 180) / Math.PI;

  // Fitts' Law duration scaling: 150ms minimum, up to 550ms for 180-degree turns
  const computedDurationMs =
    durationMs ?? Math.min(600, Math.max(150, Math.round(140 + 2.4 * totalAngularDistDeg)));

  const numTicks = Math.max(3, Math.ceil(computedDurationMs / 50));
  const keyframes: CameraKeyframe[] = [];

  // For flicks > 35 degrees, human gaze slightly overshoots by ~3-4% at control point P2
  const overshootFactor = totalAngularDistDeg > 35 ? 1 + CONFIG.kinematics.fittsOvershootRatio : 1.0;

  const yawP0 = startYaw;
  const yawP1 = startYaw + deltaYaw * 0.25;
  const yawP2 = startYaw + deltaYaw * overshootFactor;
  const yawP3 = startYaw + deltaYaw;

  const pitchP0 = startPitch;
  const pitchP1 = startPitch + deltaPitch * 0.25;
  const pitchP2 = startPitch + deltaPitch * overshootFactor;
  const pitchP3 = startPitch + deltaPitch;

  const baseTimeSec = Date.now() / 1000;

  for (let i = 1; i <= numTicks; i++) {
    const t = i / numTicks;
    const rawYaw = evaluateCubicBezier(t, yawP0, yawP1, yawP2, yawP3);
    const rawPitch = evaluateCubicBezier(t, pitchP0, pitchP1, pitchP2, pitchP3);

    // Taper tremor at the very end of target acquisition so block interaction doesn't slip
    const tremorScale = i === numTicks ? 0.25 : 1.0;
    const yawTremor = sampleHandTremorRadians(baseTimeSec + i * 0.05, 0) * tremorScale;
    const pitchTremor = sampleHandTremorRadians(baseTimeSec + i * 0.05, 42) * tremorScale;

    keyframes.push({
      yaw: normalizeRadians(rawYaw + yawTremor),
      pitch: Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rawPitch + pitchTremor)),
    });
  }

  return keyframes;
}

/**
 * Manages smooth, non-blocking 20Hz camera rotation for the Mineflayer bot.
 */
export class HumanizedCameraController {
  private queue: CameraKeyframe[] = [];
  private currentResolve: (() => void) | null = null;

  constructor(private readonly bot: any) {}

  /**
   * Queues a smooth Bézier look trajectory toward a world coordinate.
   * Optionally leads a moving target's velocity vector.
   */
  public lookAtSmooth(
    targetPos: Vec3 | Vector3D,
    durationMs?: number,
    targetVelocity?: Vector3D
  ): Promise<void> {
    if (
      !this.bot?.entity?.position ||
      !targetPos ||
      !Number.isFinite(targetPos.x) ||
      !Number.isFinite(targetPos.y) ||
      !Number.isFinite(targetPos.z)
    ) {
      return Promise.resolve();
    }

    // Cancel any previous pending look promise cleanly
    if (this.currentResolve) {
      this.currentResolve();
      this.currentResolve = null;
    }

    const eyePos: Vector3D = {
      x: this.bot.entity.position.x,
      y: this.bot.entity.position.y + 1.62,
      z: this.bot.entity.position.z,
    };

    // Predictive velocity leading (lead by ~150ms for moving mobs/players)
    const leadSec = 0.15;
    const aimedPos: Vector3D = {
      x: targetPos.x + (targetVelocity?.x ?? 0) * leadSec,
      y: targetPos.y + (targetVelocity?.y ?? 0) * leadSec,
      z: targetPos.z + (targetVelocity?.z ?? 0) * leadSec,
    };

    const { yaw: targetYaw, pitch: targetPitch } = calculateLookAngles(eyePos, aimedPos);
    this.queue = generateBezierCameraTrajectory(
      this.bot.entity.yaw,
      this.bot.entity.pitch,
      targetYaw,
      targetPitch,
      durationMs
    );

    return new Promise((resolve) => {
      this.currentResolve = resolve;
    });
  }

  /**
   * Called every 50ms tick by the System 1 loop to advance the camera smoothly.
   * Pauses automatic camera interpolation while Mineflayer is actively digging a block
   * so the crosshair never slips off the target voxel mid-dig!
   */
  public updateTick(): void {
    if (!this.bot?.entity) return;
    if (this.bot.targetDigBlock) {
      // Flush pending look queue if bot is actively breaking a block
      if (this.queue.length > 0) {
        this.queue = [];
        if (this.currentResolve) {
          const res = this.currentResolve;
          this.currentResolve = null;
          res();
        }
      }
      return;
    }

    if (this.queue.length > 0) {
      const next = this.queue.shift()!;
      if (Number.isFinite(next.yaw) && Number.isFinite(next.pitch)) {
        this.bot.look(next.yaw, next.pitch, true);
      }
      if (this.queue.length === 0 && this.currentResolve) {
        const res = this.currentResolve;
        this.currentResolve = null;
        res();
      }
    }
  }

  public clearQueue(): void {
    this.queue = [];
    if (this.currentResolve) {
      const res = this.currentResolve;
      this.currentResolve = null;
      res();
    }
  }

  public isRotating(): boolean {
    return this.queue.length > 0;
  }
}
