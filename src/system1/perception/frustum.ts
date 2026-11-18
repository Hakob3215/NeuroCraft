import { Vector3D } from '../../types';
import { CONFIG } from '../../config';

/**
 * Normalizes an angle in radians to the range [-PI, PI].
 */
export function normalizeRadians(angle: number): number {
  let a = angle % (2 * Math.PI);
  if (a > Math.PI) a -= 2 * Math.PI;
  if (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

/**
 * Calculates the yaw and pitch required for an observer at `eyePos` to look at `targetPos`
 * using Mineflayer's coordinate convention:
 * - yaw = 0 faces North (-Z), yaw = -PI/2 faces East (+X), yaw = PI faces South (+Z), yaw = PI/2 faces West (-X)
 * - pitch = 0 is horizontal, positive is looking up, negative is looking down
 */
export function calculateLookAngles(eyePos: Vector3D, targetPos: Vector3D): { yaw: number; pitch: number; distance: number } {
  if (
    !eyePos || !targetPos ||
    !Number.isFinite(eyePos.x) || !Number.isFinite(eyePos.y) || !Number.isFinite(eyePos.z) ||
    !Number.isFinite(targetPos.x) || !Number.isFinite(targetPos.y) || !Number.isFinite(targetPos.z)
  ) {
    return { yaw: 0, pitch: 0, distance: 0 };
  }
  const dx = targetPos.x - eyePos.x;
  const dy = targetPos.y - eyePos.y;
  const dz = targetPos.z - eyePos.z;
  const groundDist = Math.sqrt(dx * dx + dz * dz);
  const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

  const yaw = normalizeRadians(Math.atan2(-dx, -dz));
  const pitch = groundDist < 1e-6 ? (dy >= 0 ? Math.PI / 2 : -Math.PI / 2) : Math.atan2(dy, groundDist);

  return {
    yaw: Number.isFinite(yaw) ? yaw : 0,
    pitch: Number.isFinite(pitch) ? pitch : 0,
    distance: Number.isFinite(distance) ? distance : 0,
  };
}

/**
 * Checks whether `targetPos` falls inside the agent's synthetic human camera frustum.
 * Default: 90° horizontal FOV (+/- 45°) and 70° vertical FOV (+/- 35°).
 */
export function isWithinHumanFov(
  eyePos: Vector3D,
  botYaw: number,
  botPitch: number,
  targetPos: Vector3D,
  horizontalFovDeg = CONFIG.perception.horizontalFovDeg,
  verticalFovDeg = CONFIG.perception.verticalFovDeg,
  maxDistance = CONFIG.perception.maxSightDistance
): boolean {
  const { yaw: targetYaw, pitch: targetPitch, distance } = calculateLookAngles(eyePos, targetPos);

  if (distance > maxDistance) {
    return false;
  }

  // Within 1.5 blocks, human peripheral/tactile awareness always registers contact
  if (distance <= 1.5) {
    return true;
  }

  const deltaYaw = Math.abs(normalizeRadians(targetYaw - botYaw));
  const deltaPitch = Math.abs(targetPitch - botPitch);

  const maxHalfHorizRad = ((horizontalFovDeg / 2) * Math.PI) / 180;
  const maxHalfVertRad = ((verticalFovDeg / 2) * Math.PI) / 180;

  return deltaYaw <= maxHalfHorizRad && deltaPitch <= maxHalfVertRad;
}
