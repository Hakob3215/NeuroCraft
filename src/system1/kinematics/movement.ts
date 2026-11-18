import { Vec3 } from 'vec3';
import { Vector3D } from '../../types';
import { HumanizedCameraController } from './camera';

/**
 * Evaluates a Catmull-Rom spline segment between P1 and P2 given control points P0, P1, P2, P3
 * and parameter t in [0, 1].
 */
export function evaluateCatmullRomPoint(
  t: number,
  p0: Vector3D,
  p1: Vector3D,
  p2: Vector3D,
  p3: Vector3D
): Vector3D {
  const t2 = t * t;
  const t3 = t2 * t;

  const calcAxis = (v0: number, v1: number, v2: number, v3: number): number =>
    0.5 *
    (2 * v1 +
      (-v0 + v2) * t +
      (2 * v0 - 5 * v1 + 4 * v2 - v3) * t2 +
      (-v0 + 3 * v1 - 3 * v2 + v3) * t3);

  return {
    x: calcAxis(p0.x, p1.x, p2.x, p3.x),
    y: calcAxis(p0.y, p1.y, p2.y, p3.y),
    z: calcAxis(p0.z, p1.z, p2.z, p3.z),
  };
}

/**
 * String-pulling (collinear pruning) + Catmull-Rom spline interpolation over discrete A* waypoints.
 * Transforms jagged block-center steps into smooth curved human trajectories.
 */
export function generateSmoothSplineTrajectory(
  waypoints: Vector3D[],
  samplesPerSegment = 4
): Vector3D[] {
  if (waypoints.length <= 1) {
    return [...waypoints];
  }

  // 1. Prune redundant collinear grid waypoints (Funnel string-pulling simplification)
  const pruned: Vector3D[] = [waypoints[0]];
  for (let i = 1; i < waypoints.length - 1; i++) {
    const prev = pruned[pruned.length - 1];
    const curr = waypoints[i];
    const next = waypoints[i + 1];

    const v1x = curr.x - prev.x;
    const v1z = curr.z - prev.z;
    const v2x = next.x - curr.x;
    const v2z = next.z - curr.z;

    const len1 = Math.hypot(v1x, v1z) || 1;
    const len2 = Math.hypot(v2x, v2z) || 1;
    const dot = (v1x * v2x + v1z * v2z) / (len1 * len2);

    // Keep waypoint if there is a turn > 12 degrees or vertical height change
    if (dot < 0.978 || Math.abs(curr.y - prev.y) > 0.2 || Math.abs(next.y - curr.y) > 0.2) {
      pruned.push(curr);
    }
  }
  pruned.push(waypoints[waypoints.length - 1]);

  if (pruned.length === 2) {
    return pruned;
  }

  // 2. Interpolate Catmull-Rom spline points
  const splinePoints: Vector3D[] = [];
  for (let i = 0; i < pruned.length - 1; i++) {
    const p0 = pruned[Math.max(0, i - 1)];
    const p1 = pruned[i];
    const p2 = pruned[i + 1];
    const p3 = pruned[Math.min(pruned.length - 1, i + 2)];

    for (let s = 0; s < samplesPerSegment; s++) {
      const t = s / samplesPerSegment;
      splinePoints.push(evaluateCatmullRomPoint(t, p0, p1, p2, p3));
    }
  }
  splinePoints.push(pruned[pruned.length - 1]);

  return splinePoints;
}

/**
 * Executes humanized movement, sprint-jumping synchronization, and smooth gaze steering.
 */
export class HumanizedMovementController {
  private activeSpline: Vector3D[] = [];
  private isSprinting = false;

  constructor(
    private readonly bot: any,
    private readonly camera: HumanizedCameraController
  ) {}

  /**
   * Synchronizes sprint-jumping when moving over open terrain, while strictly avoiding
   * jumping when the bot or pathfinder is digging a block (jumping incurs a 5x mining penalty in Minecraft!).
   */
  public updateTick(): void {
    if (!this.bot?.entity) return;

    // Never jump while actively digging a block (airborne mining is 5x slower and aborts digs!)
    if (this.bot.targetDigBlock || this.bot.pathfinder?.isMining?.()) {
      return;
    }

    const vel = this.bot.entity.velocity;
    const horizSpeed = Math.hypot(vel.x, vel.z);

    // Only sprint-jump when executing our own spline movement (do not interfere with pathfinder's step/dig physics)
    if (
      this.activeSpline.length > 0 &&
      this.bot.getControlState('forward') &&
      this.bot.getControlState('sprint')
    ) {
      if (this.bot.entity.onGround && horizSpeed > 0.14 && Math.random() < 0.18) {
        const headAbove = this.bot.blockAt(this.bot.entity.position.offset(0, 2.1, 0));
        if (!headAbove || headAbove.boundingBox === 'empty') {
          this.bot.setControlState('jump', true);
        }
      } else if (!this.bot.entity.onGround) {
        this.bot.setControlState('jump', false);
      }
    }
  }

  /**
   * Moves the agent along a smoothed spline curve of waypoints.
   */
  public async moveAlongSpline(waypoints: Vec3[], sprint = true): Promise<void> {
    if (!this.bot?.entity || waypoints.length === 0) return;

    this.activeSpline = generateSmoothSplineTrajectory(waypoints);
    this.isSprinting = sprint;

    for (const pt of this.activeSpline) {
      if (!this.bot?.entity) break;
      const targetVec = new Vec3(pt.x, pt.y + 1.5, pt.z);
      await this.camera.lookAtSmooth(targetVec, 160);

      this.bot.setControlState('forward', true);
      this.bot.setControlState('sprint', this.isSprinting);

      const startWait = Date.now();
      while (Date.now() - startWait < 2500) {
        if (!this.bot?.entity) break;
        const dist = Math.hypot(
          this.bot.entity.position.x - pt.x,
          this.bot.entity.position.z - pt.z
        );
        if (dist < 1.15) break;
        await new Promise((r) => setTimeout(r, 50));
      }
    }

    this.stopMovement();
  }

  public stopMovement(): void {
    if (!this.bot?.setControlState) return;
    this.bot.setControlState('forward', false);
    this.bot.setControlState('back', false);
    this.bot.setControlState('left', false);
    this.bot.setControlState('right', false);
    this.bot.setControlState('sprint', false);
    this.bot.setControlState('jump', false);
    this.activeSpline = [];
  }
}
