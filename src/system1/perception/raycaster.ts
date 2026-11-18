import { Vector3D } from '../../types';

export type VoxelBlockCheckFn = (x: number, y: number, z: number) => boolean;

/**
 * Performs fast 3D voxel raycasting (Amanatides & Woo / 3D DDA traversal) from `origin` to `target`.
 * Returns true if the line-of-sight is obstructed by any opaque/solid block (`isSolidBlock(x, y, z) === true`).
 */
export function isLineOfSightOccluded(
  origin: Vector3D,
  target: Vector3D,
  isSolidBlock: VoxelBlockCheckFn
): boolean {
  const dx = target.x - origin.x;
  const dy = target.y - origin.y;
  const dz = target.z - origin.z;
  const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

  if (distance < 0.25) {
    return false;
  }

  // Step along ray in 0.35-block increments (fast and catches all full/slab voxel obstructions)
  const steps = Math.ceil(distance / 0.35);
  const stepX = dx / steps;
  const stepY = dy / steps;
  const stepZ = dz / steps;

  const targetBlockX = Math.floor(target.x);
  const targetBlockY = Math.floor(target.y);
  const targetBlockZ = Math.floor(target.z);

  const originBlockX = Math.floor(origin.x);
  const originBlockY = Math.floor(origin.y);
  const originBlockZ = Math.floor(origin.z);

  let cx = origin.x;
  let cy = origin.y;
  let cz = origin.z;

  for (let i = 1; i < steps; i++) {
    cx += stepX;
    cy += stepY;
    cz += stepZ;

    const bx = Math.floor(cx);
    const by = Math.floor(cy);
    const bz = Math.floor(cz);

    // Ignore the observer's own head block and the target's exact occupied block
    if (bx === originBlockX && by === originBlockY && bz === originBlockZ) {
      continue;
    }
    if (bx === targetBlockX && by === targetBlockY && bz === targetBlockZ) {
      continue;
    }

    if (isSolidBlock(bx, by, bz)) {
      return true;
    }
  }

  return false;
}

/**
 * Multi-point entity visibility check: tests rays from the bot's eyes to the target entity's
 * head, torso, and feet. The entity is only considered occluded if ALL sample rays are blocked.
 */
export function isEntityOccluded(
  botEyePos: Vector3D,
  entityBasePos: Vector3D,
  entityHeight = 1.8,
  isSolidBlock: VoxelBlockCheckFn
): boolean {
  const samplePoints: Vector3D[] = [
    { x: entityBasePos.x, y: entityBasePos.y + entityHeight * 0.85, z: entityBasePos.z }, // Head
    { x: entityBasePos.x, y: entityBasePos.y + entityHeight * 0.5, z: entityBasePos.z },  // Torso
    { x: entityBasePos.x, y: entityBasePos.y + 0.2, z: entityBasePos.z },                 // Feet
  ];

  for (const pt of samplePoints) {
    if (!isLineOfSightOccluded(botEyePos, pt, isSolidBlock)) {
      return false; // At least one part of the entity is visible
    }
  }

  return true;
}
