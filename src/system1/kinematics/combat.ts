import { Vec3 } from 'vec3';
import { HumanizedCameraController } from './camera';

/**
 * Returns the Minecraft 1.9+ attack recharge duration in milliseconds (and ticks)
 * for the given held item name.
 * - Swords: attack speed 1.6 -> 625ms (12.5 ticks)
 * - Tridents: attack speed 1.1 -> 910ms (18.2 ticks)
 * - Axes (Wood/Stone): attack speed 0.8 -> 1250ms (25 ticks)
 * - Axes (Iron): attack speed 0.9 -> 1111ms (22.2 ticks)
 * - Axes (Diamond/Netherite/Gold): attack speed 1.0 -> 1000ms (20 ticks)
 * - Pickaxes: attack speed 1.2 -> 833ms (16.7 ticks)
 * - Shovels: attack speed 1.0 -> 1000ms (20 ticks)
 * - Unarmed / Other: attack speed 4.0 -> 250ms (5 ticks)
 */
export function getWeaponCooldownMs(itemName: string | null | undefined): number {
  if (!itemName) return 250;
  const name = itemName.toLowerCase();

  if (name.includes('sword')) return 625;
  if (name.includes('trident')) return 910;
  if (name.includes('_axe')) {
    if (name.startsWith('wooden') || name.startsWith('stone')) return 1250;
    if (name.startsWith('iron')) return 1115;
    return 1000;
  }
  if (name.includes('pickaxe')) return 835;
  if (name.includes('shovel')) return 1000;
  return 250;
}

/**
 * Checks whether the player's current physical state qualifies for a Minecraft 1.9+ Critical Strike:
 * - Airborne (`!onGround`)
 * - Falling downward (`velocity.y < -0.05`)
 * - Not in water or climbing a ladder/vine
 */
export function isCriticalStrikeWindow(
  onGround: boolean,
  velocityY: number,
  isInWater = false
): boolean {
  return !onGround && !isInWater && velocityY < -0.05;
}

/**
 * Modern 1.9+ Combat Controller with attack cooldown tracking, falling critical hit synchronization,
 * strafe kiting, and 5-tick (250ms) off-hand shield warm-up anticipation.
 */
export class ModernCombatController {
  private lastStrikeTimestamp = 0;
  private shieldRaisedAt: number | null = null;

  constructor(
    private readonly bot: any,
    private readonly camera: HumanizedCameraController
  ) {}

  /**
   * Returns current attack cooldown charge in [0.0, 1.0].
   */
  public getAttackCharge(now = Date.now()): number {
    const heldName = this.bot?.heldItem?.name ?? null;
    const cooldownMs = getWeaponCooldownMs(heldName);
    const elapsed = now - this.lastStrikeTimestamp;
    return Math.min(1.0, Math.max(0.0, elapsed / cooldownMs));
  }

  /**
   * Raises the off-hand shield (requires 5 ticks / 250ms warm-up before damage is blocked).
   */
  public raiseShield(): void {
    if (!this.bot?.inventory) return;
    const offHandItem = this.bot.inventory.slots[45]; // Off-hand slot in Minecraft 1.9+
    const mainHandItem = this.bot.heldItem;
    const hasShield =
      offHandItem?.name?.includes('shield') || mainHandItem?.name?.includes('shield');

    if (hasShield && this.shieldRaisedAt === null) {
      this.bot.activateItem(offHandItem?.name?.includes('shield') ? true : false);
      this.shieldRaisedAt = Date.now();
    }
  }

  public lowerShield(): void {
    if (this.shieldRaisedAt !== null) {
      this.bot?.deactivateItem();
      this.shieldRaisedAt = null;
    }
  }

  /**
   * Executes a single humanized 1.9+ weapon strike against `targetEntity` when recharge >= 98%.
   */
  public async strikeTarget(targetEntity: any, attemptCritical = true): Promise<boolean> {
    if (!this.bot?.entity || !targetEntity?.position) return false;

    const dist = this.bot.entity.position.distanceTo(targetEntity.position);
    if (dist > 3.35) {
      return false; // Out of survival melee reach (~3.0 - 3.3 blocks)
    }

    // Smoothly aim at target upper torso with predictive velocity lead
    const aimPoint = new Vec3(
      targetEntity.position.x,
      targetEntity.position.y + (targetEntity.height ? targetEntity.height * 0.72 : 1.25),
      targetEntity.position.z
    );
    await this.camera.lookAtSmooth(aimPoint, 130, targetEntity.velocity);

    // Wait until weapon cooldown reaches >= 98% charge
    while (this.getAttackCharge() < 0.98) {
      await new Promise((r) => setTimeout(r, 30));
    }

    // Optionally synchronize with a jump-fall critical strike window
    if (
      attemptCritical &&
      this.bot.entity.onGround &&
      !this.bot.entity.isInWater &&
      Math.random() < 0.55
    ) {
      this.bot.setControlState('jump', true);
      await new Promise((r) => setTimeout(r, 50));
      this.bot.setControlState('jump', false);

      // Wait up to 350ms for the apex of the jump to transition into downward descent (vy < -0.05)
      const waitStart = Date.now();
      while (
        Date.now() - waitStart < 350 &&
        !isCriticalStrikeWindow(
          this.bot.entity.onGround,
          this.bot.entity.velocity.y,
          this.bot.entity.isInWater
        )
      ) {
        await new Promise((r) => setTimeout(r, 25));
      }
    }

    // Lower shield briefly right before swinging (cannot attack while blocking in vanilla)
    const wasBlocking = this.shieldRaisedAt !== null;
    if (wasBlocking) {
      this.lowerShield();
      await new Promise((r) => setTimeout(r, 40));
    }

    this.bot.attack(targetEntity);
    this.lastStrikeTimestamp = Date.now();

    if (wasBlocking) {
      setTimeout(() => this.raiseShield(), 100);
    }

    return true;
  }
}
