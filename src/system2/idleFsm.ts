import { Vec3 } from 'vec3';
import { System1BotController } from '../system1/botController';

export type IdleActionExecutor = (
  toolName: string,
  args: Record<string, any>
) => Promise<{ success: boolean; summary: string }>;

/**
 * Human Idle State Machine (`IdleFSM`).
 * 1. Attentive Micro-Gestures: Turns gaze toward the speaker and nods subtly while LLM reasons.
 * 2. Autonomous Idle Companion Loop: When the agent has no active goal and has been idle
 *    for >= 15 seconds, autonomously performs productive companion tasks:
 *    - Crafting missing essential tools (axes, shovels, pickaxes, swords)
 *    - Smelting unsmelted raw ores into ingots
 *    - Crafting and equipping armor pieces
 *    - Looking at nearby players warmly and holding the right tool in hand
 *    - Eating food if hungry
 */
export class HumanIdleSimulator {
  private thinkingTimer: NodeJS.Timeout | null = null;
  private autonomousTimer: NodeJS.Timeout | null = null;
  private isThinking = false;
  private isGoalActive = false;
  private isIdleActionRunning = false;
  private lastActivityTimestamp = Date.now();
  private lastActionType = '';

  constructor(
    private readonly system1: System1BotController,
    private executeToolCallback?: IdleActionExecutor
  ) {}

  public setToolExecutor(executor: IdleActionExecutor): void {
    this.executeToolCallback = executor;
  }

  public notifyActivity(): void {
    this.lastActivityTimestamp = Date.now();
    this.isIdleActionRunning = false;
  }

  public setGoalActive(active: boolean): void {
    this.isGoalActive = active;
    if (active) {
      this.notifyActivity();
    }
  }

  /**
   * Starts subtle micro-nods while System 2 is thinking or synthesizing a new MCP tool.
   */
  public startThinkingBehavior(speakerUsername?: string): void {
    this.notifyActivity();
    if (this.isThinking) return;
    this.isThinking = true;

    const bot = this.system1.bot;
    if (!bot?.entity) return;

    const speakerEntity = speakerUsername ? bot.players?.[speakerUsername]?.entity : null;
    if (speakerEntity?.position) {
      this.system1.camera
        .lookAtSmooth(speakerEntity.position.offset(0, 1.62, 0), 220, speakerEntity.velocity)
        .catch(() => {});
    }

    let step = 0;
    this.thinkingTimer = setInterval(() => {
      if (!this.isThinking || !bot?.entity || bot.targetDigBlock) return;
      step++;

      const currentPos = bot.entity.position;
      const yaw = bot.entity.yaw + (step % 2 === 0 ? 0.08 : -0.08);
      const pitchOffset = step % 2 === 0 ? 0.14 : -0.1;
      const target = new Vec3(
        currentPos.x - Math.sin(yaw) * 3,
        currentPos.y + 1.62 + pitchOffset,
        currentPos.z - Math.cos(yaw) * 3
      );

      this.system1.camera.lookAtSmooth(target, 200).catch(() => {});
    }, 450);
  }

  public stopThinkingBehavior(): void {
    this.isThinking = false;
    if (this.thinkingTimer) {
      clearInterval(this.thinkingTimer);
      this.thinkingTimer = null;
    }
    this.notifyActivity();
  }

  /**
   * Starts the Autonomous Companion Idle Loop.
   * Runs a lightweight check every 5 seconds. If the agent has been idle for >= 15 seconds,
   * performs a self-improvement or companion interaction.
   */
  public startAutonomousIdleLoop(): void {
    if (this.autonomousTimer) return;

    this.autonomousTimer = setInterval(async () => {
      if (
        this.isThinking ||
        this.isGoalActive ||
        this.isIdleActionRunning ||
        !this.system1.bot?.entity
      ) {
        return;
      }

      const idleMs = Date.now() - this.lastActivityTimestamp;
      if (idleMs < 14000) {
        return;
      }

      await this.runAutonomousIdleStep();
    }, 4500);
  }

  public stopAutonomousIdleLoop(): void {
    if (this.autonomousTimer) {
      clearInterval(this.autonomousTimer);
      this.autonomousTimer = null;
    }
  }

  private async runAutonomousIdleStep(): Promise<void> {
    if (
      this.isThinking ||
      this.isGoalActive ||
      this.isIdleActionRunning ||
      !this.executeToolCallback
    ) {
      return;
    }

    this.isIdleActionRunning = true;
    try {
      const bot = this.system1.bot;
      const inv = bot.inventory?.items() || [];
      const countItem = (pred: (n: string) => boolean) =>
        inv.filter((i: any) => pred(i.name.toLowerCase())).reduce((s: number, i: any) => s + i.count, 0);

      const cobble = countItem((n) => n === 'cobblestone' || n === 'cobbled_deepslate');
      const ironIngots = countItem((n) => n === 'iron_ingot');
      const rawIron = countItem((n) => n === 'raw_iron');
      const sticks = countItem((n) => n === 'stick');
      const planks = countItem((n) => n.endsWith('_planks'));
      const logs = countItem((n) => n.endsWith('_log') || n.endsWith('_stem'));
      const coal = countItem((n) => n === 'coal' || n === 'charcoal');
      const totalPotentialPlanks = planks + logs * 4;
      const canCraftTools = sticks >= 2 || totalPotentialPlanks >= 2;

      // 1. Tool check & upgrade: Missing essential tools?
      if (canCraftTools) {
        const hasAxe = inv.some((i: any) => i.name.endsWith('_axe'));
        const hasShovel = inv.some((i: any) => i.name.endsWith('_shovel'));
        const hasPick = inv.some((i: any) => i.name.endsWith('_pickaxe'));
        const hasSword = inv.some((i: any) => i.name.endsWith('_sword'));

        let toolToCraft = '';
        if (!hasPick && (cobble >= 3 || totalPotentialPlanks >= 5)) {
          toolToCraft = cobble >= 3 ? 'stone_pickaxe' : 'wooden_pickaxe';
        } else if (!hasAxe && (cobble >= 3 || totalPotentialPlanks >= 5)) {
          toolToCraft = cobble >= 3 ? 'stone_axe' : 'wooden_axe';
        } else if (!hasShovel && (cobble >= 1 || totalPotentialPlanks >= 3)) {
          toolToCraft = cobble >= 1 ? 'stone_shovel' : 'wooden_shovel';
        } else if (!hasSword && (cobble >= 2 || totalPotentialPlanks >= 4)) {
          toolToCraft = cobble >= 2 ? 'stone_sword' : 'wooden_sword';
        }

        if (toolToCraft && this.lastActionType !== `craft_${toolToCraft}`) {
          this.lastActionType = `craft_${toolToCraft}`;
          console.log(`[IdleCompanion] Noticed missing ${toolToCraft} while idle—crafting it to be prepared!`);
          await this.executeToolCallback('craft_item', { itemName: toolToCraft, count: 1 });
          this.notifyActivity();
          return;
        }
      }

      // 2. Smelting unsmelted raw iron if we have cobblestone/furnace and fuel
      const hasFurnace = inv.some((i: any) => i.name === 'furnace') || cobble >= 8;
      const hasFuel = coal >= 1 || planks >= 2 || logs >= 1;
      if (rawIron >= 1 && hasFurnace && hasFuel && this.lastActionType !== 'smelt_iron') {
        this.lastActionType = 'smelt_iron';
        console.log(`[IdleCompanion] Smelting ${rawIron} raw iron in inventory while idle...`);
        await this.executeToolCallback('smelt_item', {
          inputItem: 'raw_iron',
          count: Math.min(8, rawIron),
        });
        this.notifyActivity();
        return;
      }

      // 3. Crafting & equipping armor if we have iron ingots
      if (ironIngots >= 4) {
        const equipped = (bot as any).equipment || [];
        const invPieces = inv.map((i: any) => i.name);
        let armorToCraft = '';
        if (
          ironIngots >= 4 &&
          !equipped[3] &&
          !invPieces.includes('iron_boots') &&
          !invPieces.includes('diamond_boots')
        ) {
          armorToCraft = 'iron_boots';
        } else if (
          ironIngots >= 5 &&
          !equipped[0] &&
          !invPieces.includes('iron_helmet') &&
          !invPieces.includes('diamond_helmet')
        ) {
          armorToCraft = 'iron_helmet';
        } else if (
          ironIngots >= 7 &&
          !equipped[2] &&
          !invPieces.includes('iron_leggings') &&
          !invPieces.includes('diamond_leggings')
        ) {
          armorToCraft = 'iron_leggings';
        } else if (
          ironIngots >= 8 &&
          !equipped[1] &&
          !invPieces.includes('iron_chestplate') &&
          !invPieces.includes('diamond_chestplate')
        ) {
          armorToCraft = 'iron_chestplate';
        }

        if (armorToCraft && this.lastActionType !== `craft_${armorToCraft}`) {
          this.lastActionType = `craft_${armorToCraft}`;
          console.log(`[IdleCompanion] Crafting and equipping ${armorToCraft} while idle!`);
          await this.executeToolCallback('craft_item', { itemName: armorToCraft, count: 1 });
          await this.executeToolCallback('equip_armor_or_item', { itemName: 'all_armor' });
          this.notifyActivity();
          return;
        }
      }

      // 4. Companion Interaction: Look at nearby player or hold proper tool in hand
      const nearbyPlayers = Object.values(bot.players || {}).filter(
        (p: any) =>
          p?.entity &&
          p.username !== bot.username &&
          bot.entity.position.distanceTo(p.entity.position) <= 16
      ) as any[];

      if (nearbyPlayers.length > 0) {
        const nearestPlayer = nearbyPlayers.sort(
          (a, b) =>
            bot.entity.position.distanceTo(a.entity.position) -
            bot.entity.position.distanceTo(b.entity.position)
        )[0];

        const pDist = bot.entity.position.distanceTo(nearestPlayer.entity.position);
        if (pDist <= 14) {
          // Look at player attentively
          await this.system1.camera.lookAtSmooth(
            nearestPlayer.entity.position.offset(0, 1.62, 0),
            260,
            nearestPlayer.entity.velocity
          );

          // If holding a non-tool item like dirt or cobblestone while idle, switch to sword or tool
          const held = bot.heldItem?.name || '';
          if (
            !held.includes('pickaxe') &&
            !held.includes('sword') &&
            !held.includes('axe') &&
            !held.includes('shovel')
          ) {
            const toolItem = inv.find(
              (i: any) =>
                i.name.includes('sword') ||
                i.name.includes('pickaxe') ||
                i.name.includes('axe')
            );
            if (toolItem) {
              try {
                await bot.equip(toolItem, 'hand');
              } catch {}
            }
          }

          // If player is a bit far (8-12m), take a few steps closer
          if (pDist > 7.5 && pDist < 12.0 && (this.system1 as any).pathfindToPosition) {
            await (this.system1 as any).pathfindToPosition(
              nearestPlayer.entity.position,
              3.5,
              3500
            );
          }

          this.notifyActivity();
          return;
        }
      }

      // 5. Environmental Glance (look at scenic horizon)
      const currentYaw = bot.entity.yaw;
      const glanceYaw = currentYaw + 0.45;
      const eyePos = bot.entity.position.offset(0, 1.62, 0);
      const lookTarget = new Vec3(
        eyePos.x - Math.sin(glanceYaw) * 6,
        eyePos.y + 0.5,
        eyePos.z - Math.cos(glanceYaw) * 6
      );
      await this.system1.camera.lookAtSmooth(lookTarget, 300);
      this.notifyActivity();
    } catch {
      // Ignore idle action errors
    } finally {
      this.isIdleActionRunning = false;
    }
  }
}
