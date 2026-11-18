import { Vec3 } from 'vec3';

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export interface PerceivedEntity {
  id: number;
  name: string;
  username?: string;
  type: 'player' | 'mob' | 'hostile' | 'passive' | 'object' | 'other';
  position: Vector3D;
  velocity: Vector3D;
  distance: number;
  health?: number;
  isOccluded: boolean;
  inFov: boolean;
  perceivedAt: number; // Timestamp after applying LogNormal human reaction delay
  creeperFuseState?: number;
}

export interface PerceivedBlock {
  name: string;
  position: Vector3D;
  distance: number;
}

export interface ResourceClusterSummary {
  blockType: string;
  totalNearby: number;
  exposedCount: number;
  nearestPos: Vector3D;
  nearestDistance: number;
  nearestDirection: string;
  nearestExposedPos?: Vector3D;
  nearestExposedDistance?: number;
}

export interface TerrainContext {
  timeOfDay: 'day' | 'dusk' | 'night' | 'dawn';
  isRaining: boolean;
  canSeeSky: boolean;
  blockBelowFeet: string;
  depthToStoneBlocks: number | null;
  adjacentBlocks: {
    north: string;
    south: string;
    east: string;
    west: string;
    aboveHead: string;
    belowFeet: string;
  };
  resourceClusters: ResourceClusterSummary[];
}

export interface EquippedArmorState {
  head: string | null;
  torso: string | null;
  legs: string | null;
  feet: string | null;
  offHand: string | null;
}

export interface AuditoryCue {
  soundName: string;
  directionVector: Vector3D;
  estimatedDistance: number;
  timestamp: number;
}

export interface InventoryItem {
  name: string;
  count: number;
  slot: number;
  durability?: number; // Remaining durability uses before item breaks
  maxDurability?: number; // Total max durability of the item type
  durabilityPercent?: number; // Integer percentage remaining (0-100)
}

export interface SanitizedWorldState {
  tick: number;
  timestamp: number;
  botUsername: string;
  position: Vector3D;
  velocity: Vector3D;
  yaw: number;
  pitch: number;
  health: number;
  food: number;
  oxygen: number;
  onGround: boolean;
  isInWater: boolean;
  isInLava: boolean;
  heldItem: string | null;
  equippedArmor?: EquippedArmorState;
  inventory: InventoryItem[];
  visibleEntities: PerceivedEntity[];
  audibleCues: AuditoryCue[];
  nearbyNotableBlocks: PerceivedBlock[];
  terrainContext?: TerrainContext;
  activeHazard: HazardType | null;
  activeSubtask: SubtaskStatus | null;
}

export type HazardType =
  | 'CREEPER_FUSE'
  | 'LAVA_CONTACT'
  | 'BURNING'
  | 'DROWNING'
  | 'SUFFOCATION'
  | 'SUDDEN_DAMAGE'
  | 'FALLING_HIGH'
  | 'STARVATION';

export interface ReflexInterruptEvent {
  hazard: HazardType;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM';
  details: string;
  position: Vector3D;
  timestamp: number;
}

export interface SubtaskStatus {
  id: string;
  toolName: string;
  description: string;
  args: Record<string, any>;
  startedAt: number;
  estimatedDurationMs: number;
  progress: number; // 0.0 to 1.0
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'INTERRUPTED' | 'FAILED';
  error?: string;
}

export interface MCPToolSchema {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, { type: string; description: string; default?: any; enum?: string[] }>;
    required?: string[];
  };
  isSynthesized?: boolean;
  synthesizedAt?: string;
  sourceCode?: string;
  tags?: string[];
}

export interface AreaTopologyScan {
  minX: number;
  minZ: number;
  maxX: number;
  maxZ: number;
  minHeight: number;
  maxHeight: number;
  medianHeight: number;
  isFlat: boolean;
  obstructionCount: number;
  airGapCount: number;
  groundBlockTypes: Record<string, number>;
  heightGrid: number[][]; // [dx][dz]
}

export interface MCPToolExecutionContext {
  bot: any; // Mineflayer bot instance
  kinematics: {
    lookAtSmooth: (target: Vec3, durationMs?: number, targetVelocity?: Vector3D) => Promise<void>;
    moveAlongSpline: (waypoints: Vec3[], sprint?: boolean) => Promise<void>;
    pathfindTo: (target: Vec3, range?: number, timeoutMs?: number) => Promise<boolean>;
    safeDig: (blockOrPos: any) => Promise<boolean>;
    safePlaceBlock?: (targetPos: Vec3, itemName: string, allowScaffold?: boolean) => Promise<boolean>;
    strikeTarget: (entity: any) => Promise<boolean>;
    equipItemHumanized: (itemName: string, destination?: 'hand' | 'off-hand' | 'head' | 'torso' | 'legs' | 'feet') => Promise<boolean>;
    executeNeuralMotorChunk?: (prompt: string) => Promise<boolean>;
  };
  world?: {
    getGroundHeight: (x: number, z: number, startY?: number) => number;
    getTopSolidBlock: (x: number, z: number, startY?: number) => any | null;
    scanAreaTopology: (minX: number, minZ: number, maxX: number, maxZ: number) => AreaTopologyScan;
  };
  worldState: () => SanitizedWorldState;
  reportProgress: (progress: number) => void;
  sayHumanized: (message: string) => Promise<void>;
  abortSignal: AbortSignal;
}

export type MCPToolHandler = (
  args: Record<string, any>,
  ctx: MCPToolExecutionContext
) => Promise<{ success: boolean; summary: string; data?: any }>;

export interface RegisteredMCPTool {
  schema: MCPToolSchema;
  handler: MCPToolHandler;
  embeddingKeywords: string[];
}

export interface CognitiveDecision {
  thought: string;
  speechResponse?: string;
  activeGoal?: string;
  goalCompleted?: boolean;
  planSteps?: string[];
  rememberLandmark?: {
    name: string;
    position: Vector3D;
  };
  toolCall?: {
    name: string;
    arguments: Record<string, any>;
  };
  synthesizeRequest?: {
    desiredToolName: string;
    taskDescription: string;
    requiredParameters: Record<string, string>;
    triggeringAsk: string;
  };
}

export interface EpisodicMemoryEntry {
  id: string;
  type: 'LANDMARK' | 'CHEST' | 'DEATH' | 'PLAYER_INTERACTION' | 'SYNTHESIZED_SKILL' | 'GOAL_COMPLETED';
  summary: string;
  position?: Vector3D;
  timestamp: number;
  tags: string[];
}
