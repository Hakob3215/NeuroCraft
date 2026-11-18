import { Vec3 } from 'vec3';
import { HazardType, ReflexInterruptEvent, SanitizedWorldState } from '../../types';
import { HumanizedCameraController } from '../kinematics/camera';
import { ModernCombatController } from '../kinematics/combat';
import { HumanizedInventoryController } from '../kinematics/inventory';
import { sharedNeuralBridge } from '../neural/neuralBridge';

/**
 * System 1 Sub-50ms Reflex Engine.
 * Evaluates immediate survival hazards on every 20Hz tick and executes instantaneous
 * motor preemption without waiting for System 2 LLM inference.
 */
export class ReflexEngine {
  private lastHealth: number | null = null;
  private lastDamageTime = 0;
  private activeHazard: HazardType | null = null;
  private lastInterruptTime = 0;
  private isExecutingReflex = false;
  private lastNeuralCheckTime = 0;
  private isEvaluatingNeuralReflex = false;
  private hasActiveSubtask = false;

  constructor(
    private readonly bot: any,
    private readonly camera: HumanizedCameraController,
    private readonly combat: ModernCombatController,
    private readonly inventory: HumanizedInventoryController,
    private readonly onReflexInterrupt: (event: ReflexInterruptEvent) => void
  ) {}

  public getActiveHazard(): HazardType | null {
    return this.activeHazard;
  }

  /**
   * Synchronous 20Hz tick evaluation. Detects lethal hazards in <5ms.
   */
  public evaluateTick(state: SanitizedWorldState): void {
    if (!this.bot?.entity) return;

    const now = Date.now();
    const currentHealth = this.bot.health ?? 20;

    // Initialize baseline health on first tick without triggering a false-positive damage spike on spawn
    if (this.lastHealth === null) {
      this.lastHealth = currentHealth;
      return;
    }

    const healthDrop = this.lastHealth - currentHealth;
    this.lastHealth = currentHealth;

    if (healthDrop > 0) {
      this.lastDamageTime = now;
    }

    // 1. CRITICAL: Creeper fuse or dangerous proximity
    // When actively running `engage_combat`, `engage_combat` manages Creeper hit-and-run kiting & shield blocking directly;
    // do not preempt `engage_combat` with a CRITICAL abort!
    const inActiveCombat = state.activeSubtask?.toolName === 'engage_combat';
    const closeCreeper = !inActiveCombat
      ? state.visibleEntities.find((e) => {
          if (!e.name.toLowerCase().includes('creeper')) return false;
          const rawEnt = this.bot?.entities?.[e.id];
          const isIgnited = rawEnt?.metadata?.[16] === 1 || rawEnt?.metadata?.[18] === true;
          return e.distance < 4.5 || (isIgnited && e.distance < 6.0);
        })
      : undefined;
    if (closeCreeper) {
      this.triggerReflex(
        'CREEPER_FUSE',
        'CRITICAL',
        `Creeper detected at ${closeCreeper.distance.toFixed(1)}m! Executing evasive sprint-jump & shield block.`,
        () => this.executeCreeperEvasion(closeCreeper.position, closeCreeper.distance)
      );
      return;
    }

    // 2. CRITICAL: Lava contact or burning
    if (state.isInLava) {
      this.triggerReflex(
        'LAVA_CONTACT',
        'CRITICAL',
        'Bot is in lava! Executing emergency swim-up and water bucket douse.',
        () => this.executeLavaEscape()
      );
      return;
    }

    // 3. HIGH: Drowning underwater (oxygen <= 6 bubbles)
    if (state.isInWater && state.oxygen <= 6) {
      this.triggerReflex(
        'DROWNING',
        'HIGH',
        'Low oxygen underwater! Swimming immediately to surface.',
        () => this.executeSurfaceSwim()
      );
      return;
    }

    // 4. HIGH: Sudden damage spike (> 2.5 HP lost in a single tick)
    if (healthDrop >= 2.5) {
      this.triggerReflex(
        'SUDDEN_DAMAGE',
        'HIGH',
        `Took ${healthDrop.toFixed(1)} sudden damage! Raising shield and strafing.`,
        () => this.executeDamageEvasion(state)
      );
      return;
    }

    // 5. MEDIUM: Starvation / Low HP during combat (food < 14 and health < 12)
    if (state.health < 12 && state.food < 16 && !this.isExecutingReflex) {
      if (now - this.lastInterruptTime > 6000) {
        this.executeAutoEat();
      }
    }

    this.hasActiveSubtask = Boolean(state.activeSubtask);

    // 6. System 1 Closed-Loop Neural Voxel Reflex (~2ms on RTX 5060)
    // Periodically checks VoxelReflexPolicy for sub-tick shield blocking & evasions against real hostiles
    if (
      !this.isExecutingReflex &&
      !this.isEvaluatingNeuralReflex &&
      now - this.lastNeuralCheckTime > 250 &&
      state.visibleEntities &&
      state.visibleEntities.some(
        (e) =>
          e.type === 'hostile' ||
          (e as any).isHostile ||
          ['creeper', 'skeleton', 'zombie', 'spider', 'enderman', 'warden', 'blaze', 'wither', 'witch', 'pillager', 'drowned', 'piglin'].includes(
            e.name?.toLowerCase?.() || ''
          )
      )
    ) {
      this.lastNeuralCheckTime = now;
      this.isEvaluatingNeuralReflex = true;
      sharedNeuralBridge
        .evaluateReflex(state)
        .then((reflexOut) => {
          if (!reflexOut || !this.bot?.entity) return;
          if (reflexOut.hazard_urgency > 0.55 && !this.activeHazard) {
            // 1. Continuous camera aim tracking
            if (reflexOut.aim_delta && (reflexOut.aim_delta.dyaw !== 0 || reflexOut.aim_delta.dpitch !== 0)) {
              const curYaw = this.bot.entity.yaw;
              const curPitch = this.bot.entity.pitch;
              const targetYaw = curYaw + (reflexOut.aim_delta.dyaw * Math.PI) / 180;
              const targetPitch = Math.max(
                -Math.PI / 2,
                Math.min(Math.PI / 2, curPitch + (reflexOut.aim_delta.dpitch * Math.PI) / 180)
              );
              this.bot.look(targetYaw, targetPitch, true).catch(() => {});
            }

            // 2. Shield blocking
            const hasShield = Boolean(this.bot?.inventory?.items()?.some((i: any) => i.name.includes('shield')));
            if (reflexOut.actions.use_shield && hasShield) {
              this.combat.raiseShield();
            }

            // 3. Directional movement states (only override locomotion when no active subtask/pathfinder holds control)
            if (!this.hasActiveSubtask && !this.bot.pathfinder?.isMoving?.()) {
              this.bot.setControlState('forward', Boolean(reflexOut.actions.move_forward));
              this.bot.setControlState('back', Boolean(reflexOut.actions.move_back));
              this.bot.setControlState('left', Boolean(reflexOut.actions.strafe_left));
              this.bot.setControlState('right', Boolean(reflexOut.actions.strafe_right));
              this.bot.setControlState('sprint', Boolean(reflexOut.actions.sprint));
              this.bot.setControlState('sneak', Boolean(reflexOut.actions.sneak));
              if (reflexOut.actions.jump && this.bot.entity.onGround) {
                this.bot.setControlState('jump', true);
              }
            }

            // 4. Attack nearest hostile or rival opponent in reach
            if (reflexOut.actions.attack) {
              const target = this.bot.nearestEntity(
                (e: any) =>
                  e &&
                  e !== this.bot.entity &&
                  (e.type === 'hostile' ||
                    ['creeper', 'skeleton', 'zombie', 'spider', 'enderman', 'warden', 'blaze', 'wither'].includes(e.name || '') ||
                    (e.type === 'player' &&
                      e.username &&
                      e.username.toLowerCase() !== this.bot.username?.toLowerCase() &&
                      !e.username.toLowerCase().includes('companion') &&
                      !e.username.toLowerCase().includes('friend') &&
                      !e.username.toLowerCase().includes('teammate') &&
                      (e.username.toLowerCase().includes('opponent') ||
                        e.username.toLowerCase().includes('rival') ||
                        e.username.toLowerCase().includes('dqn') ||
                        e.username.toLowerCase().includes('mirror') ||
                        e.username.toLowerCase().includes('spar'))))
              );
              if (target && this.bot.entity.position.distanceTo(target.position) <= 3.5) {
                this.bot.attack(target);
              }
            }

            // Reset control states after 200ms sub-tick window only if no active subtask/pathfinder owns movement
            setTimeout(() => {
              try {
                if (this.bot?.entity && !this.hasActiveSubtask && !this.bot.pathfinder?.isMoving?.()) {
                  this.bot.setControlState('forward', false);
                  this.bot.setControlState('back', false);
                  this.bot.setControlState('left', false);
                  this.bot.setControlState('right', false);
                  this.bot.setControlState('jump', false);
                  this.bot.setControlState('sprint', false);
                  this.bot.setControlState('sneak', false);
                }
              } catch {}
            }, 200);
          }
        })
        .catch(() => {})
        .finally(() => {
          this.isEvaluatingNeuralReflex = false;
        });
    }

    if (now - this.lastInterruptTime > 2500) {
      this.activeHazard = null;
    }
  }

  private triggerReflex(
    hazard: HazardType,
    severity: 'CRITICAL' | 'HIGH' | 'MEDIUM',
    details: string,
    routine: () => Promise<void>
  ): void {
    const now = Date.now();
    this.activeHazard = hazard;

    // Throttle duplicate interrupt dispatches to once per 1.2s
    if (now - this.lastInterruptTime > 1200) {
      this.lastInterruptTime = now;
      this.onReflexInterrupt({
        hazard,
        severity,
        details,
        position: {
          x: this.bot.entity.position.x,
          y: this.bot.entity.position.y,
          z: this.bot.entity.position.z,
        },
        timestamp: now,
      });
    }

    if (!this.isExecutingReflex) {
      this.isExecutingReflex = true;
      routine()
        .catch(() => {})
        .finally(() => {
          this.isExecutingReflex = false;
        });
    }
  }

  private async executeCreeperEvasion(
    creeperPos: { x: number; y: number; z: number },
    dist = 4.0
  ): Promise<void> {
    if (!this.bot?.entity) return;

    if (this.bot.pathfinder?.isMoving?.()) {
      this.bot.pathfinder.setGoal(null);
    }

    // If point-blank (<= 2.2m), face the Creeper directly and raise shield (shields only block within a forward 90° cone).
    // Otherwise (> 2.2m), keep shield lowered so sprinting works at full 5.6 m/s speed and sprint-jump away!
    if (dist <= 2.2) {
      await this.camera.lookAtSmooth(new Vec3(creeperPos.x, creeperPos.y + 1.2, creeperPos.z), 60);
      this.combat.raiseShield();
      this.bot.setControlState('back', true);
      await new Promise((r) => setTimeout(r, 550));
      this.bot.setControlState('back', false);
      this.combat.lowerShield();
      return;
    }

    this.combat.lowerShield();

    const dx = this.bot.entity.position.x - creeperPos.x;
    const dz = this.bot.entity.position.z - creeperPos.z;
    const escapeTarget = new Vec3(
      this.bot.entity.position.x + dx * 3,
      this.bot.entity.position.y + 1.6,
      this.bot.entity.position.z + dz * 3
    );

    await this.camera.lookAtSmooth(escapeTarget, 90);
    this.bot.setControlState('forward', true);
    this.bot.setControlState('sprint', true);
    this.bot.setControlState('jump', true);

    await new Promise((r) => setTimeout(r, 600));

    this.bot.setControlState('forward', false);
    this.bot.setControlState('sprint', false);
    this.bot.setControlState('jump', false);
  }

  private async executeLavaEscape(): Promise<void> {
    if (!this.bot?.entity) return;
    this.bot.setControlState('jump', true);
    this.bot.setControlState('forward', true);

    // Attempt to equip water bucket if in inventory
    const hasWater = await this.inventory.equipItemHumanized('water_bucket', 'hand');
    if (hasWater) {
      await this.camera.lookAtSmooth(
        new Vec3(
          this.bot.entity.position.x,
          this.bot.entity.position.y - 1,
          this.bot.entity.position.z
        ),
        100
      );
      this.bot.activateItem();
    }

    await new Promise((r) => setTimeout(r, 800));
    this.bot.setControlState('jump', false);
    this.bot.setControlState('forward', false);
  }

  private async executeSurfaceSwim(): Promise<void> {
    if (!this.bot?.entity) return;
    this.bot.setControlState('jump', true);
    await new Promise((r) => setTimeout(r, 900));
    this.bot.setControlState('jump', false);
  }

  private async executeDamageEvasion(state: SanitizedWorldState): Promise<void> {
    if (!this.bot?.entity) return;

    // Yield pathfinding so authoritative server knockback resolves cleanly without client fight
    if (this.bot.pathfinder?.isMoving?.()) {
      this.bot.pathfinder.setGoal(null);
    }
    this.bot.clearControlStates?.();

    this.combat.raiseShield();

    // Do NOT jump while airborne from server knockback (prevents anti-cheat / flight kicks).
    // Only perform ground strafe when on solid ground.
    if (this.bot.entity.onGround) {
      const strafeDir = Math.random() < 0.5 ? 'left' : 'right';
      this.bot.setControlState(strafeDir, true);
      this.bot.setControlState('back', true);

      await new Promise((r) => setTimeout(r, 350));

      this.bot.setControlState(strafeDir, false);
      this.bot.setControlState('back', false);
    } else {
      // Allow airborne knockback arc to finish smoothly without contradictory player controls
      await new Promise((r) => setTimeout(r, 350));
    }

    this.combat.lowerShield();
  }

  private async executeAutoEat(): Promise<void> {
    const foods = [
      'cooked_beef',
      'cooked_porkchop',
      'golden_apple',
      'Golden_carrot',
      'cooked_mutton',
      'cooked_chicken',
      'bread',
      'baked_potato',
      'apple',
    ];
    for (const food of foods) {
      if (await this.inventory.equipItemHumanized(food, 'hand')) {
        this.bot.activateItem();
        await new Promise((r) => setTimeout(r, 1650)); // Standard Minecraft eating duration (32 ticks = 1.6s)
        this.bot.deactivateItem();
        break;
      }
    }
  }
}
