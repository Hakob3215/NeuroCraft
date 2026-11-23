import { CONFIG } from '../../config';

/**
 * Converts a Minecraft inventory slot index (0-45) to normalized 2D GUI grid coordinates (row, col)
 * so we can compute Fitts' Law virtual mouse cursor travel distances.
 */
export function slotIndexToGridCoords(slot: number): { row: number; col: number } {
  if (slot >= 36 && slot <= 44) {
    // Hotbar row at bottom of GUI
    return { row: 4, col: slot - 36 };
  }
  if (slot >= 9 && slot <= 35) {
    // Main 3x9 player inventory
    return { row: Math.floor((slot - 9) / 9) + 1, col: (slot - 9) % 9 };
  }
  if (slot >= 5 && slot <= 8) {
    // Armor slots on top-left
    return { row: 0, col: slot - 5 };
  }
  return { row: 0, col: slot % 9 };
}

/**
 * Calculates Fitts' Law cursor movement + click delay in milliseconds between two GUI slots:
 *   T = BaseDelay + Slope * log2(1 + DistanceInSlots) + Jitter
 * Prevents single-tick (0ms) robotic inventory sorting that triggers anti-cheat checks.
 */
export function calculateFittsInventoryDelayMs(fromSlot: number, toSlot: number): number {
  const p1 = slotIndexToGridCoords(fromSlot);
  const p2 = slotIndexToGridCoords(toSlot);
  const slotDist = Math.hypot(p2.col - p1.col, p2.row - p1.row);

  const fittsTime =
    CONFIG.kinematics.inventoryBaseDelayMs +
    CONFIG.kinematics.inventoryFittsSlopeMs * Math.log2(1 + slotDist);

  // Add +/- 25ms physiological click jitter
  const jitter = (Math.random() - 0.5) * 50;
  return Math.max(95, Math.round(fittsTime + jitter));
}

/**
 * Manages humanized item equipping and GUI interactions with realistic cursor movement delays.
 */
export class HumanizedInventoryController {
  private lastCursorSlot = 36;

  constructor(private readonly bot: any) {}

  /**
   * Equips an item by name (or partial match) with realistic Fitts' Law delay.
   */
  public async equipItemHumanized(
    itemNameQuery: string,
    destination: 'hand' | 'off-hand' | 'head' | 'torso' | 'legs' | 'feet' = 'hand'
  ): Promise<boolean> {
    if (!this.bot?.inventory) return false;

    let queryNorm = itemNameQuery.toLowerCase();
    if (queryNorm === 'leather_cap' || queryNorm === 'cap') queryNorm = 'leather_helmet';
    if (queryNorm === 'tunic') queryNorm = 'leather_chestplate';
    if (queryNorm === 'pants') queryNorm = 'leather_leggings';

    const items = this.bot.inventory.items();
    const targetItem = items.find(
      (i: any) =>
        i.name.toLowerCase() === queryNorm ||
        i.name.toLowerCase().includes(queryNorm)
    );

    if (!targetItem) {
      return false;
    }

    const delayMs = calculateFittsInventoryDelayMs(this.lastCursorSlot, targetItem.slot);
    this.lastCursorSlot = targetItem.slot;

    await new Promise((r) => setTimeout(r, delayMs));

    try {
      (this.bot as any)._lastSwapSlotAt = Date.now();
      await this.bot.equip(targetItem, destination);
      return true;
    } catch {
      return false;
    }
  }
}
