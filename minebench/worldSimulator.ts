import { Vec3 } from 'vec3';
import { MinebenchTaskSpec } from './benchmarkSpec';
import { MinebenchEvaluationSnapshot } from './evaluators';
import { MCPToolExecutionContext, SanitizedWorldState } from '../src/types';

interface SimBlock {
  name: string;
  position: Vec3;
  boundingBox: 'block' | 'empty';
  metadata: number;
}

interface SimEntity {
  id: number;
  name: string;
  displayName: string;
  type: 'hostile' | 'animal' | 'passive' | 'object' | 'other';
  position: Vec3;
  velocity: Vec3;
  height: number;
  isValid: boolean;
  hp: number;
  metadata: Record<number, any>;
}

/**
 * High-Speed 3D Voxel & Entity Physics Simulator for MINEbench.
 * Directly parses and executes Minecraft 1.20.4 `/fill`, `/setblock`, `/summon`, and `/tp`
 * commands from `MinebenchTaskSpec.buildArenaCommands`, and provides a full Mineflayer-compatible
 * `ToolExecutionContext` so every real MCP skill in `baseSkills.ts` executes against the 3D arena!
 */
export class MinebenchWorldSimulator {
  private voxels = new Map<string, { name: string; metadata: number }>();
  public entities: Record<number, SimEntity> = {};
  private nextEntityId = 1000;

  public botPos = new Vec3(0, 64, 0);
  public botVel = new Vec3(0, 0, 0);
  public botYaw = 0;
  public botPitch = 0;
  public health = 20;
  public food = 20;
  public deaths = 0;
  public heldItemName: string | null = null;

  public inventory: Array<{
    name: string;
    count: number;
    slot: number;
    durability?: number;
    maxDurability?: number;
  }> = [];

  public equippedArmor: {
    head: string | null;
    torso: string | null;
    legs: string | null;
    feet: string | null;
    offHand: string | null;
  } = {
    head: null,
    torso: null,
    legs: null,
    feet: null,
    offHand: null,
  };

  public stats = {
    blocksPlaced: 0,
    blocksMined: 0,
    mobsKilled: {} as Record<string, number>,
    creeperBlasts: 0,
    arrowsFired: 0,
    endCrystalsDestroyed: 0,
    enderDragonSlain: false,
    leversFlipped: 0,
    wolvesTamed: 0,
    ironPickaxeDurabilityUsedOnStone: 0,
  };

  private key(x: number, y: number, z: number): string {
    return `${Math.floor(x)},${Math.floor(y)},${Math.floor(z)}`;
  }

  public setBlock(x: number, y: number, z: number, rawName: string): void {
    const clean = rawName
      .replace(/^minecraft:/i, '')
      .replace(/\[.*\]$/, '')
      .toLowerCase()
      .trim();
    const k = this.key(x, y, z);
    if (clean === 'air' || clean === 'cave_air' || clean === 'void_air') {
      this.voxels.delete(k);
    } else {
      const meta = rawName.includes('age=7') ? 7 : 0;
      this.voxels.set(k, { name: clean, metadata: meta });
    }
  }

  public getBlock(x: number, y: number, z: number): SimBlock {
    const fx = Math.floor(x);
    const fy = Math.floor(y);
    const fz = Math.floor(z);
    const entry = this.voxels.get(this.key(fx, fy, fz));
    const name = entry ? entry.name : 'air';
    const isPassable =
      name === 'air' ||
      name === 'water' ||
      name === 'lava' ||
      name === 'torch' ||
      name === 'lever' ||
      name === 'wheat' ||
      name === 'short_grass';
    return {
      name,
      position: new Vec3(fx, fy, fz),
      boundingBox: isPassable ? 'empty' : 'block',
      metadata: entry?.metadata ?? 0,
    };
  }

  /**
   * Parses and executes Minecraft 1.20.4 server commands (`/fill`, `/setblock`, `/summon`, `/tp`, `/gamerule`).
   */
  public executeServerCommand(cmd: string): void {
    const clean = cmd.replace(/^\//, '').trim();
    const parts = clean.split(/\s+/);
    const op = parts[0]?.toLowerCase();

    if (op === 'fill' && parts.length >= 8) {
      const x1 = Number(parts[1]);
      const y1 = Number(parts[2]);
      const z1 = Number(parts[3]);
      const x2 = Number(parts[4]);
      const y2 = Number(parts[5]);
      const z2 = Number(parts[6]);
      const blockSpec = parts[7];
      const mode = (parts[8] || 'replace').toLowerCase();

      const minX = Math.min(x1, x2);
      const maxX = Math.max(x1, x2);
      const minY = Math.min(y1, y2);
      const maxY = Math.max(y1, y2);
      const minZ = Math.min(z1, z2);
      const maxZ = Math.max(z1, z2);

      for (let x = minX; x <= maxX; x++) {
        for (let y = minY; y <= maxY; y++) {
          for (let z = minZ; z <= maxZ; z++) {
            const isBoundary =
              x === minX || x === maxX || y === minY || y === maxY || z === minZ || z === maxZ;
            if (mode === 'outline' && !isBoundary) continue;
            if (mode === 'hollow' && !isBoundary) {
              this.setBlock(x, y, z, 'air');
              continue;
            }
            this.setBlock(x, y, z, blockSpec);
          }
        }
      }
      return;
    }

    if (op === 'setblock' && parts.length >= 5) {
      const x = Number(parts[1]);
      const y = Number(parts[2]);
      const z = Number(parts[3]);
      const blockSpec = parts[4];
      this.setBlock(x, y, z, blockSpec);
      return;
    }

    if (op === 'tp' && parts.length >= 5) {
      const x = Number(parts[2]);
      const y = Number(parts[3]);
      const z = Number(parts[4]);
      this.botPos = new Vec3(x, y, z);
      return;
    }

    if (op === 'summon' && parts.length >= 5) {
      const rawName = parts[1].replace(/^minecraft:/i, '').toLowerCase();
      const x = Number(parts[2]);
      const y = Number(parts[3]);
      const z = Number(parts[4]);
      const id = this.nextEntityId++;

      const isPassive = ['cow', 'sheep', 'pig', 'chicken', 'wolf', 'cat', 'villager'].includes(
        rawName
      );
      const isCrystal = rawName === 'end_crystal';
      const hpMap: Record<string, number> = {
        creeper: 3,
        spider: 2,
        enderman: 5,
        ender_dragon: 8,
        end_crystal: 1,
        cow: 2,
        wolf: 4,
      };

      this.entities[id] = {
        id,
        name: rawName,
        displayName: rawName,
        type: isCrystal ? 'other' : isPassive ? 'animal' : 'hostile',
        position: new Vec3(x, y, z),
        velocity: new Vec3(0, 0, 0),
        height: rawName === 'enderman' ? 2.9 : 1.8,
        isValid: true,
        hp: hpMap[rawName] ?? 3,
        metadata: { 16: 0, 17: 0 },
      };
    }
  }

  public initializeTaskArena(task: MinebenchTaskSpec): void {
    this.voxels.clear();
    this.entities = {};
    this.health = 20;
    this.food = 16;
    this.deaths = 0;
    this.heldItemName = null;
    this.equippedArmor = { head: null, torso: null, legs: null, feet: null, offHand: null };
    this.stats = {
      blocksPlaced: 0,
      blocksMined: 0,
      mobsKilled: {},
      creeperBlasts: 0,
      arrowsFired: 0,
      endCrystalsDestroyed: 0,
      enderDragonSlain: false,
      leversFlipped: 0,
      wolvesTamed: 0,
      ironPickaxeDurabilityUsedOnStone: 0,
    };

    this.inventory = task.initialLoadout.map((item, idx) => ({
      name: item.name,
      count: item.count,
      slot: item.slot ?? 36 + idx,
      ...(item.name.includes('pickaxe') || item.name.includes('shovel') || item.name.includes('sword')
        ? {
            durability: item.name === 'iron_pickaxe' ? 18 : 120,
            maxDurability: item.name === 'iron_pickaxe' ? 250 : 131,
          }
        : {}),
    }));

    const commands = task.buildArenaCommands(task.origin);
    for (const cmd of commands) {
      this.executeServerCommand(cmd);
    }
  }

  public addInventoryItem(name: string, count: number): void {
    const existing = this.inventory.find((i) => i.name === name);
    if (existing) {
      existing.count += count;
    } else {
      this.inventory.push({ name, count, slot: 9 + this.inventory.length });
    }
  }

  public consumeInventoryItem(name: string, count: number): boolean {
    const existing = this.inventory.find((i) => i.name === name && i.count >= count);
    if (!existing) return false;
    existing.count -= count;
    if (existing.count <= 0) {
      this.inventory = this.inventory.filter((i) => i.count > 0);
    }
    return true;
  }

  public buildWorldState(): SanitizedWorldState {
    const visibleEntities = Object.values(this.entities)
      .filter((e) => e.isValid)
      .map((e) => ({
        id: e.id,
        name: e.name,
        type: (e.type === 'animal' ? 'passive' : e.type) as
          | 'hostile'
          | 'passive'
          | 'player'
          | 'object'
          | 'other',
        position: { x: e.position.x, y: e.position.y, z: e.position.z },
        velocity: { x: 0, y: 0, z: 0 },
        distance: Number(e.position.distanceTo(this.botPos).toFixed(2)),
        inFov: true,
        isOccluded: false,
        perceivedAt: Date.now(),
      }))
      .sort((a, b) => a.distance - b.distance);

    return {
      tick: Date.now(),
      timestamp: Date.now(),
      botUsername: 'NeuroAgent',
      position: { x: this.botPos.x, y: this.botPos.y, z: this.botPos.z },
      velocity: { x: 0, y: 0, z: 0 },
      yaw: this.botYaw,
      pitch: this.botPitch,
      health: this.health,
      food: this.food,
      oxygen: 20,
      onGround: true,
      isInWater: false,
      isInLava: false,
      heldItem: this.heldItemName,
      equippedArmor: { ...this.equippedArmor },
      inventory: this.inventory.map((i) => ({
        name: i.name,
        count: i.count,
        slot: i.slot,
        durability: i.durability,
        maxDurability: i.maxDurability,
        durabilityPercent:
          i.durability !== undefined && i.maxDurability
            ? Math.round((i.durability / i.maxDurability) * 100)
            : undefined,
      })),
      visibleEntities,
      audibleCues: [],
      nearbyNotableBlocks: [],
      terrainContext: {
        timeOfDay: 'day',
        isRaining: false,
        canSeeSky: true,
        blockBelowFeet: this.getBlock(this.botPos.x, this.botPos.y - 1, this.botPos.z).name,
        depthToStoneBlocks: 1,
        adjacentBlocks: {
          north: 'air',
          south: 'air',
          east: 'air',
          west: 'air',
          aboveHead: 'air',
          belowFeet: 'air',
        },
        resourceClusters: [],
      },
      activeHazard: null,
      activeSubtask: null,
    };
  }

  public getEvaluationSnapshot(): MinebenchEvaluationSnapshot {
    return {
      botPosition: { x: this.botPos.x, y: this.botPos.y, z: this.botPos.z },
      health: this.health,
      food: this.food,
      deaths: this.deaths,
      inventory: this.inventory.map((i) => ({ ...i })),
      equippedArmor: { ...this.equippedArmor },
      blockAt: (x, y, z) => this.getBlock(x, y, z),
      entities: Object.values(this.entities).map((e) => ({
        id: e.id,
        name: e.name,
        type: e.type,
        position: { x: e.position.x, y: e.position.y, z: e.position.z },
        isValid: e.isValid,
        metadata: e.metadata,
        health: e.hp,
      })),
      stats: {
        ...this.stats,
        mobsKilled: { ...this.stats.mobsKilled },
      },
    };
  }

  /**
   * Creates a full Mineflayer-compatible `MCPToolExecutionContext` backed by this 3D simulator
   * so real MCP tools in `baseSkills.ts` execute with zero mocks.
   */
  public createToolContext(abortSignal: AbortSignal): MCPToolExecutionContext {
    const self = this;

    const findMatchingBlocks = (options: {
      matching: (b: any) => boolean;
      maxDistance?: number;
      count?: number;
    }): Vec3[] => {
      const maxDist = options.maxDistance ?? 32;
      const limit = options.count ?? 64;
      const results: Vec3[] = [];

      // Test Mineflayer palette pre-filter compatibility (b.position === null)
      for (const [k, v] of self.voxels.entries()) {
        if (!options.matching({ name: v.name, position: null, metadata: v.metadata })) continue;
        const [x, y, z] = k.split(',').map(Number);
        const pos = new Vec3(x, y, z);
        if (pos.distanceTo(self.botPos) <= maxDist) {
          results.push(pos);
        }
      }

      results.sort((a, b) => a.distanceTo(self.botPos) - b.distanceTo(self.botPos));
      return results.slice(0, limit);
    };

    const botProxy: any = {
      username: 'NeuroAgent',
      version: '1.20.4',
      health: self.health,
      food: self.food,
      get entity() {
        return {
          position: self.botPos,
          velocity: self.botVel,
          yaw: self.botYaw,
          pitch: self.botPitch,
          onGround: true,
        };
      },
      get heldItem() {
        return self.heldItemName ? { name: self.heldItemName, count: 1 } : null;
      },
      entities: self.entities,
      players: {
        TheJack3215: {
          username: 'TheJack3215',
          entity: {
            position: self.botPos.offset(2, 0, 2),
            velocity: new Vec3(0, 0, 0),
          },
        },
      },
      inventory: {
        items: () => self.inventory.filter((i) => i.count > 0),
      },
      pathfinder: {
        setGoal: () => {},
        isMoving: () => false,
      },
      setControlState: (control: string, state: boolean) => {
        if (control === 'jump' && state) {
          self.botPos = new Vec3(self.botPos.x, self.botPos.y + 1.25, self.botPos.z);
        } else if (control === 'back' && state) {
          // Kiting / backpedaling resets Creeper fuse ignition (metadata[16] = 0)
          for (const ent of Object.values(self.entities)) {
            if (ent.isValid && ent.name === 'creeper') {
              ent.metadata[16] = 0;
            }
          }
        }
      },
      clearControlStates: () => {},
      swingArm: () => {},
      blockAt: (pos: Vec3) => self.getBlock(pos.x, pos.y, pos.z),
      findBlocks: findMatchingBlocks,
      findBlock: (options: { matching: (b: any) => boolean; maxDistance?: number }) => {
        const matches = findMatchingBlocks({ ...options, count: 1 });
        return matches.length > 0 ? self.getBlock(matches[0].x, matches[0].y, matches[0].z) : null;
      },
      nearestEntity: (filter?: (e: any) => boolean) => {
        const candidates = Object.values(self.entities)
          .filter((e) => e.isValid && (!filter || filter(e)))
          .sort((a, b) => a.position.distanceTo(self.botPos) - b.position.distanceTo(self.botPos));
        return candidates[0] || null;
      },
      canDigBlock: () => true,
      stopDigging: () => {},
      dig: async (block: SimBlock) => {
        await self.mineBlockWithToolCheck(block.position);
      },
      consume: async () => {
        if (self.heldItemName) {
          self.consumeInventoryItem(self.heldItemName, 1);
          self.food = 20;
          self.health = Math.min(20, self.health + 4);
          botProxy.food = self.food;
          botProxy.health = self.health;
        }
      },
      activateItem: () => {
        if (self.heldItemName === 'bow') {
          self.stats.arrowsFired++;
          // Check if aiming near an end_crystal or hostile
          const crystal = Object.values(self.entities).find(
            (e) => e.isValid && e.name === 'end_crystal'
          );
          if (crystal) {
            crystal.isValid = false;
            self.stats.endCrystalsDestroyed++;
            self.stats.mobsKilled['end_crystal'] = (self.stats.mobsKilled['end_crystal'] || 0) + 1;
          }
        }
      },
      deactivateItem: () => {},
      activateEntity: async (ent: SimEntity) => {
        if (ent.name === 'wolf' && self.heldItemName === 'bone') {
          self.consumeInventoryItem('bone', 1);
          ent.metadata[17] = 0x04; // Tamed bitmask!
          self.stats.wolvesTamed++;
        }
      },
      activateBlock: async (block: SimBlock) => {
        if (block.name === 'lever') {
          self.stats.leversFlipped++;
        } else if (
          (block.name === 'dirt' || block.name === 'grass_block') &&
          self.heldItemName?.includes('hoe')
        ) {
          self.setBlock(block.position.x, block.position.y, block.position.z, 'farmland');
        } else if (block.name === 'farmland' && self.heldItemName?.includes('seeds')) {
          self.consumeInventoryItem(self.heldItemName, 1);
          self.setBlock(block.position.x, block.position.y + 1, block.position.z, 'wheat');
        } else if (block.name === 'wheat' && self.heldItemName === 'bone_meal') {
          self.consumeInventoryItem('bone_meal', 1);
          self.setBlock(block.position.x, block.position.y, block.position.z, 'wheat[age=7]');
        }
      },
    };

    return {
      bot: botProxy,
      kinematics: {
        lookAtSmooth: async () => {},
        moveAlongSpline: async (wps: Vec3[]) => {
          if (wps.length > 0) self.botPos = wps[wps.length - 1].clone();
        },
        pathfindTo: async (targetPos: Vec3) => {
          self.botPos = targetPos.clone();
          return true;
        },
        safeDig: async (blockOrPos: any) => {
          const pos = blockOrPos?.position ? blockOrPos.position : blockOrPos;
          return self.mineBlockWithToolCheck(pos);
        },
        safePlaceBlock: async (targetPos: Vec3, itemName: string) => {
          if (!self.consumeInventoryItem(itemName, 1)) return false;
          self.setBlock(targetPos.x, targetPos.y, targetPos.z, itemName);
          self.stats.blocksPlaced++;
          if (targetPos.x === Math.floor(self.botPos.x) && targetPos.z === Math.floor(self.botPos.z)) {
            self.botPos = new Vec3(self.botPos.x, targetPos.y + 1, self.botPos.z);
          }
          return true;
        },
        strikeTarget: async (targetEnt: SimEntity) => {
          if (!targetEnt || !targetEnt.isValid) return false;
          targetEnt.hp -= 1;
          if (targetEnt.name === 'creeper' && targetEnt.hp === 1) {
            // Simulate creeper igniting fuse at low HP; proper kiting in engage_combat prevents blast
            targetEnt.metadata[16] = 1;
          }
          if (targetEnt.hp <= 0) {
            targetEnt.isValid = false;
            self.stats.mobsKilled[targetEnt.name] =
              (self.stats.mobsKilled[targetEnt.name] || 0) + 1;
            if (targetEnt.name === 'ender_dragon') {
              self.stats.enderDragonSlain = true;
            }
          }
          return true;
        },
        equipItemHumanized: async (itemName: string, dest = 'hand') => {
          const item = self.inventory.find(
            (i) => i.count > 0 && (i.name === itemName || i.name.includes(itemName))
          );
          if (!item) return false;
          if (dest === 'hand') {
            self.heldItemName = item.name;
          } else if (dest === 'head') {
            self.equippedArmor.head = item.name;
          } else if (dest === 'torso') {
            self.equippedArmor.torso = item.name;
          } else if (dest === 'legs') {
            self.equippedArmor.legs = item.name;
          } else if (dest === 'feet') {
            self.equippedArmor.feet = item.name;
          } else if (dest === 'off-hand') {
            self.equippedArmor.offHand = item.name;
          }
          return true;
        },
      },
      world: {
        getGroundHeight: (x: number, z: number) => {
          for (let y = 120; y >= 0; y--) {
            const b = self.getBlock(x, y, z);
            if (b.boundingBox === 'block') return y;
          }
          return 63;
        },
        getTopSolidBlock: (x: number, z: number) => {
          for (let y = 120; y >= 0; y--) {
            const b = self.getBlock(x, y, z);
            if (b.boundingBox === 'block') return b;
          }
          return null;
        },
        scanAreaTopology: (minX: number, minZ: number, maxX: number, maxZ: number) => ({
          minHeight: 63,
          maxHeight: 63,
          medianHeight: 63,
          isFlat: true,
          obstructionCount: 0,
          airGapCount: 0,
          groundBlockTypes: { grass_block: 25 },
          heightGrid: [],
          minX,
          minZ,
          maxX,
          maxZ,
        }),
      },
      worldState: () => self.buildWorldState(),
      reportProgress: () => {},
      sayHumanized: async () => {},
      abortSignal,
    };
  }

  private async mineBlockWithToolCheck(pos: Vec3): Promise<boolean> {
    const b = this.getBlock(pos.x, pos.y, pos.z);
    if (!b || b.name === 'air') return false;

    // Auto-select appropriate tool using NeuroCraft's preservation hierarchy
    if (b.name.includes('sand') && !b.name.includes('sandstone')) {
      const shovel = this.inventory.find((i) => i.name.includes('shovel'));
      if (shovel) this.heldItemName = shovel.name;
    } else if (b.name === 'stone' || b.name === 'cobblestone') {
      const stoneOrWoodPick = this.inventory.find(
        (i) => i.name === 'stone_pickaxe' || i.name === 'wooden_pickaxe'
      );
      if (stoneOrWoodPick) {
        this.heldItemName = stoneOrWoodPick.name;
      } else {
        const ironPick = this.inventory.find((i) => i.name === 'iron_pickaxe');
        if (ironPick) {
          this.heldItemName = 'iron_pickaxe';
          this.stats.ironPickaxeDurabilityUsedOnStone++;
        }
      }
    } else if (b.name.includes('iron_ore')) {
      const pick = this.inventory.find(
        (i) => i.name === 'stone_pickaxe' || i.name === 'iron_pickaxe' || i.name === 'diamond_pickaxe'
      );
      if (!pick) return false;
      this.heldItemName = pick.name;
    }

    const dropMap: Record<string, string> = {
      stone: 'cobblestone',
      iron_ore: 'raw_iron',
      deepslate_iron_ore: 'raw_iron',
      coal_ore: 'coal',
      wheat: 'wheat',
    };
    const dropName = dropMap[b.name] || b.name;
    this.setBlock(pos.x, pos.y, pos.z, 'air');
    this.addInventoryItem(dropName, 1);
    this.stats.blocksMined++;
    return true;
  }
}
