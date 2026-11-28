/**
 * MINEbench: Multi-Faceted Minecraft AI Benchmark & Curriculum RL Specification
 * Covers 6 Orthogonal Capability Domains & 18 Programmatically Graded Challenges:
 *   1. AGILITY_PARKOUR       (Sky Parkour, Lava Ravine Bridging, High-Altitude Clutch)
 *   2. ARCHITECTURE_BUILDING (Functional Cabin Enclosure + 3D Voxel IoU + Pillar Tower)
 *   3. TACTICAL_COMBAT       (Zero-Blast Creeper Kiting, Skeleton Duel, Enderman, Spider Swarm)
 *   4. HOMESTEADING_ECONOMY  (Crop Tilling/Bone-Meal/Harvest, Wolf Taming, Cattle Hunt & Cook)
 *   5. PUZZLE_ENGINEERING    (Redstone Lever Circuit, Multi-Layer Tool-Hierarchy Excavation)
 *   6. PROGRESSION_DRAGON    (Tier 1 Stone Age, Tier 3 Full Iron Armor, Tier 6 Ender Dragon Boss Fight)
 */

export type MinebenchDomain =
  | 'AGILITY_PARKOUR'
  | 'ARCHITECTURE_BUILDING'
  | 'TACTICAL_COMBAT'
  | 'HOMESTEADING_ECONOMY'
  | 'PUZZLE_ENGINEERING'
  | 'PROGRESSION_DRAGON'
  | 'MULTIPLAYER_ARENA';

export interface MinebenchSetupCommand {
  description: string;
  command: string;
}

export interface MinebenchInventoryLoadoutItem {
  name: string;
  count: number;
  slot?: number;
}

export interface MinebenchGradingResult {
  passed: boolean;
  score: number; // 0 to 100 normalized score
  rawReward: number; // Dense RL reward signal R_t
  maxScore: number;
  metrics: Record<string, number | string | boolean>;
  breakdown: string;
}

export interface MinebenchTaskSpec {
  id: string;
  title: string;
  domain: MinebenchDomain;
  tier: number; // 1 (Basic) to 6 (Grandmaster / Ender Dragon)
  maxDurationSeconds: number;
  maxSteps: number;
  prompt: string;
  origin: { x: number; y: number; z: number };
  initialLoadout: MinebenchInventoryLoadoutItem[];
  /**
   * Exact Minecraft 1.20.4+ server commands (/fill, /setblock, /summon, /gamerule, /tp)
   * used to procedurally construct and reset the arena in <50ms before the episode starts.
   */
  buildArenaCommands: (origin: { x: number; y: number; z: number }) => string[];
  sparringConfig?: {
    mode: 'pvp_dqn' | 'pvp_self' | 'companion_wait' | 'bridge_race' | 'escort_race' | 'resource_race';
    spawnOffset: { dx: number; dy: number; dz: number };
    loadout: MinebenchInventoryLoadoutItem[];
    goalOffset?: { dx: number; dy: number; dz: number };
    targetBlockName?: string;
  };
}

export const MINEBENCH_ARENA_GAMERULES: string[] = [
  'gamerule commandModificationBlockLimit 1000000',
  'gamerule sendCommandFeedback false',
  'gamerule logAdminCommands false',
  'gamerule commandBlockOutput false',
  'gamerule doDaylightCycle false',
  'gamerule doWeatherCycle false',
  'gamerule doMobSpawning false',
  'gamerule doPatrolSpawning false',
  'gamerule doTraderSpawning false',
  'gamerule doInsomnia false',
  'gamerule mobGriefing false',
  'gamerule doFireTick false',
  'gamerule doImmediateRespawn true',
  'gamerule keepInventory false',
  'gamerule spawnRadius 0',
  // Official Vanilla Server Scoreboard Tracking Objectives
  'scoreboard objectives add deaths deathCount "Total Deaths"',
  'scoreboard objectives add mob_kills totalKillCount "Total Kills"',
  'scoreboard objectives add dmg_dealt minecraft.custom:minecraft.damage_dealt "Damage Dealt"',
  'scoreboard objectives add dmg_taken minecraft.custom:minecraft.damage_taken "Damage Taken"',
  'scoreboard objectives add dmg_blocked minecraft.custom:minecraft.damage_blocked_by_shield "Damage Blocked"',
  'scoreboard objectives add sprint_cm minecraft.custom:minecraft.sprint_one_cm "Sprint Dist"',
  'scoreboard objectives add jump_cnt minecraft.custom:minecraft.jump "Jump Count"',
  'scoreboard objectives add mine_log minecraft.mined:minecraft.oak_log "Mined Logs"',
  'scoreboard objectives add mine_stone minecraft.mined:minecraft.stone "Mined Stone"',
  'scoreboard objectives add mine_iron minecraft.mined:minecraft.iron_ore "Mined Iron"',
  'scoreboard objectives add craft_bench minecraft.crafted:minecraft.crafting_table "Crafted Table"',
  'scoreboard objectives add craft_pick minecraft.crafted:minecraft.stone_pickaxe "Crafted Pick"',
  'scoreboard objectives setdisplay sidebar mob_kills',
];

export const MINEBENCH_TASKS: MinebenchTaskSpec[] = [
  // ===========================================================================
  // DOMAIN 1: AGILITY, PARKOUR & CLUTCH MECHANICS
  // ===========================================================================
  {
    id: 'agility_sky_parkour',
    title: 'Sky Parkour Sprint-Jump Course',
    domain: 'AGILITY_PARKOUR',
    tier: 2,
    maxDurationSeconds: 45,
    maxSteps: 4,
    prompt: 'Complete the parkour course and reach the gold_block at the end without falling!',
    origin: { x: 100, y: 80, z: 100 },
    initialLoadout: [{ name: 'cooked_beef', count: 8 }],
    buildArenaCommands: ({ x, y, z }) => [
      `gamerule commandModificationBlockLimit 100000`,
      `fill ${x - 4} ${y - 4} ${z - 4} ${x + 28} ${y + 8} ${z + 6} minecraft:air`,
      `fill ${x - 3} ${y - 4} ${z - 3} ${x + 22} ${y - 4} ${z + 3} minecraft:red_stained_glass`,
      // Starting platform
      `fill ${x - 1} ${y - 1} ${z - 1} ${x + 1} ${y - 1} ${z + 1} minecraft:quartz_block`,
      // Jump Pad 1 (2-block gap)
      `setblock ${x + 4} ${y - 1} ${z} minecraft:stone_bricks`,
      `setblock ${x + 5} ${y - 1} ${z} minecraft:stone_bricks`,
      // Jump Pad 2 (2-block gap + +1 elevation)
      `setblock ${x + 8} ${y} ${z} minecraft:stone_bricks`,
      `setblock ${x + 9} ${y} ${z} minecraft:stone_bricks`,
      // Jump Pad 3 (3-block sprint jump)
      `setblock ${x + 13} ${y} ${z} minecraft:stone_bricks`,
      `setblock ${x + 14} ${y} ${z} minecraft:stone_bricks`,
      // Goal platform with gold_block
      `fill ${x + 17} ${y} ${z - 1} ${x + 19} ${y} ${z + 1} minecraft:emerald_block`,
      `setblock ${x + 18} ${y} ${z} minecraft:gold_block`,
      `tp NeuroAgent ${x} ${y} ${z} -90 0`,
    ],
  },
  {
    id: 'agility_ravine_bridge',
    title: 'Lava Ravine Sneak-Bridging Crossing',
    domain: 'AGILITY_PARKOUR',
    tier: 3,
    maxDurationSeconds: 60,
    maxSteps: 4,
    prompt: 'Bridge across the lava ravine using your cobblestone and reach the gold_block on the far side!',
    origin: { x: 200, y: 70, z: 100 },
    initialLoadout: [
      { name: 'cobblestone', count: 32 },
      { name: 'water_bucket', count: 1 },
    ],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 3} ${y - 5} ${z - 4} ${x + 16} ${y + 6} ${z + 4} minecraft:air`,
      // Lava chasm below
      `fill ${x - 2} ${y - 4} ${z - 3} ${x + 15} ${y - 4} ${z + 3} minecraft:stone`,
      `fill ${x + 2} ${y - 3} ${z - 3} ${x + 11} ${y - 3} ${z + 3} minecraft:lava`,
      // Start ledge
      `fill ${x - 2} ${y - 1} ${z - 2} ${x + 1} ${y - 1} ${z + 2} minecraft:stone_bricks`,
      // Far goal ledge (10-block chasm)
      `fill ${x + 11} ${y - 1} ${z - 2} ${x + 14} ${y - 1} ${z + 2} minecraft:stone_bricks`,
      `setblock ${x + 12} ${y - 1} ${z} minecraft:gold_block`,
      `tp NeuroAgent ${x} ${y} ${z} -90 0`,
    ],
  },

  // ===========================================================================
  // DOMAIN 2: ARCHITECTURE & 3D CONSTRUCTION (Voxel IoU + Flood-Fill Enclosure)
  // ===========================================================================
  {
    id: 'build_enclosed_shelter',
    title: 'Functional Enclosed Cabin (3D Voxel IoU & Flood-Fill Grade)',
    domain: 'ARCHITECTURE_BUILDING',
    tier: 3,
    maxDurationSeconds: 90,
    maxSteps: 4,
    prompt: 'Build a simple house right here with walls, a roof, windows, a doorway, and interior torch lighting.',
    origin: { x: 300, y: 64, z: 100 },
    initialLoadout: [
      { name: 'oak_planks', count: 128 },
      { name: 'oak_log', count: 16 },
      { name: 'crafting_table', count: 1 },
      { name: 'cobblestone', count: 64 },
      { name: 'torch', count: 8 },
      { name: 'iron_pickaxe', count: 1 },
      { name: 'iron_axe', count: 1 },
    ],
    buildArenaCommands: ({ x, y, z }) => [
      // Flat grass building pad (20x20)
      `fill ${x - 6} ${y - 1} ${z - 6} ${x + 16} ${y - 1} ${z + 16} minecraft:grass_block`,
      `fill ${x - 6} ${y} ${z - 6} ${x + 16} ${y + 10} ${z + 16} minecraft:air`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
    ],
  },
  {
    id: 'build_creative_monument',
    title: 'Creative Artistic Monument / Pixel Art (Weather Seal Bypassed)',
    domain: 'ARCHITECTURE_BUILDING',
    tier: 2,
    maxDurationSeconds: 60,
    maxSteps: 3,
    prompt: 'Construct a creative smiley face monument right here in creative mode without worrying about roof seals!',
    origin: { x: 320, y: 64, z: 100 },
    initialLoadout: [
      { name: 'oak_planks', count: 128 },
      { name: 'yellow_wool', count: 32 },
      { name: 'cobblestone', count: 32 },
    ],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 6} ${y - 1} ${z - 6} ${x + 10} ${y - 1} ${z + 10} minecraft:grass_block`,
      `fill ${x - 6} ${y} ${z - 6} ${x + 10} ${y + 8} ${z + 10} minecraft:air`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
    ],
  },
  {
    id: 'build_watchtower_pillar',
    title: 'Vertical Watchtower Pillaring (5-Block Elevation)',
    domain: 'ARCHITECTURE_BUILDING',
    tier: 2,
    maxDurationSeconds: 40,
    maxSteps: 3,
    prompt: 'Pillar straight up 5 blocks high using cobblestone!',
    origin: { x: 350, y: 64, z: 100 },
    initialLoadout: [{ name: 'cobblestone', count: 16 }],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 4} ${y - 1} ${z - 4} ${x + 4} ${y - 1} ${z + 4} minecraft:stone_bricks`,
      `fill ${x - 4} ${y} ${z - 4} ${x + 4} ${y + 12} ${z + 4} minecraft:air`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
    ],
  },

  // ===========================================================================
  // DOMAIN 3: TACTICAL COMBAT & SURVIVAL ARENA
  // ===========================================================================
  {
    id: 'combat_creeper_kiting',
    title: 'Zero-Detonation Creeper Hit-and-Run Kiting',
    domain: 'TACTICAL_COMBAT',
    tier: 3,
    maxDurationSeconds: 50,
    maxSteps: 3,
    prompt: 'Kill the 2 creepers without letting them blow up!',
    origin: { x: 400, y: 64, z: 100 },
    initialLoadout: [
      { name: 'diamond_sword', count: 1 },
      { name: 'shield', count: 1 },
      { name: 'iron_helmet', count: 1 },
      { name: 'iron_chestplate', count: 1 },
      { name: 'iron_leggings', count: 1 },
      { name: 'iron_boots', count: 1 },
      { name: 'cooked_beef', count: 8 },
    ],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 12} ${y - 1} ${z - 12} ${x + 12} ${y - 1} ${z + 12} minecraft:stone_bricks`,
      `fill ${x - 12} ${y} ${z - 12} ${x + 12} ${y + 4} ${z + 12} minecraft:stone_brick_wall outline`,
      `fill ${x - 11} ${y} ${z - 11} ${x + 11} ${y + 6} ${z + 11} minecraft:air`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
      `summon minecraft:creeper ${x + 5} ${y} ${z} {PersistenceRequired:1b}`,
      `summon minecraft:creeper ${x - 5} ${y} ${z + 3} {PersistenceRequired:1b}`,
    ],
  },
  {
    id: 'combat_skeleton_duel',
    title: 'Armed Skeleton Ranged Sniper Duel (Shield + Counter-Strike)',
    domain: 'TACTICAL_COMBAT',
    tier: 3,
    maxDurationSeconds: 55,
    maxSteps: 3,
    prompt: 'Kill the nearest skeleton!',
    origin: { x: 425, y: 64, z: 100 },
    initialLoadout: [
      { name: 'diamond_sword', count: 1 },
      { name: 'shield', count: 1 },
      { name: 'iron_helmet', count: 1 },
      { name: 'iron_chestplate', count: 1 },
      { name: 'cooked_beef', count: 8 },
    ],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 12} ${y - 1} ${z - 12} ${x + 12} ${y - 1} ${z + 12} minecraft:stone_bricks`,
      `fill ${x - 12} ${y} ${z - 12} ${x + 12} ${y + 1} ${z + 12} minecraft:glass outline`,
      `fill ${x - 11} ${y} ${z - 11} ${x + 11} ${y + 6} ${z + 11} minecraft:air`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
      `summon minecraft:skeleton ${x + 7} ${y} ${z} {PersistenceRequired:1b,HandItems:[{id:"minecraft:bow",Count:1b},{}],ArmorItems:[{},{},{},{id:"minecraft:stone_button",Count:1b}]}`,
    ],
  },
  {
    id: 'combat_spider_swarm',
    title: 'Multi-Target Spider Swarm Elimination & Loot Pickup',
    domain: 'TACTICAL_COMBAT',
    tier: 2,
    maxDurationSeconds: 60,
    maxSteps: 3,
    prompt: 'can you take care of the 5 spiders',
    origin: { x: 450, y: 64, z: 100 },
    initialLoadout: [
      { name: 'diamond_sword', count: 1 },
      { name: 'shield', count: 1 },
      { name: 'iron_chestplate', count: 1 },
      { name: 'cooked_beef', count: 8 },
    ],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 14} ${y - 1} ${z - 14} ${x + 14} ${y - 1} ${z + 14} minecraft:sandstone`,
      `fill ${x - 14} ${y} ${z - 14} ${x + 14} ${y + 1} ${z + 14} minecraft:glass outline`,
      `fill ${x - 13} ${y} ${z - 13} ${x + 13} ${y + 5} ${z + 13} minecraft:air`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
      `summon minecraft:spider ${x + 4} ${y} ${z + 2} {PersistenceRequired:1b}`,
      `summon minecraft:spider ${x + 6} ${y} ${z - 2} {PersistenceRequired:1b}`,
      `summon minecraft:spider ${x - 4} ${y} ${z + 3} {PersistenceRequired:1b}`,
      `summon minecraft:spider ${x - 6} ${y} ${z - 3} {PersistenceRequired:1b}`,
      `summon minecraft:spider ${x + 2} ${y} ${z + 6} {PersistenceRequired:1b}`,
    ],
  },
  {
    id: 'combat_enderman_hunt',
    title: 'Enderman Duel + Health Restoration',
    domain: 'TACTICAL_COMBAT',
    tier: 4,
    maxDurationSeconds: 60,
    maxSteps: 4,
    prompt: 'can you fight this enderman?',
    origin: { x: 500, y: 64, z: 100 },
    initialLoadout: [
      { name: 'diamond_sword', count: 1 },
      { name: 'shield', count: 1 },
      { name: 'diamond_helmet', count: 1 },
      { name: 'diamond_chestplate', count: 1 },
      { name: 'diamond_leggings', count: 1 },
      { name: 'diamond_boots', count: 1 },
      { name: 'cooked_beef', count: 16 },
    ],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 14} ${y - 1} ${z - 14} ${x + 14} ${y - 1} ${z + 14} minecraft:end_stone_bricks`,
      `fill ${x - 14} ${y} ${z - 14} ${x + 14} ${y + 1} ${z + 14} minecraft:glass outline`,
      `fill ${x - 13} ${y} ${z - 13} ${x + 13} ${y + 6} ${z + 13} minecraft:air`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
      `summon minecraft:enderman ${x + 5} ${y} ${z} {PersistenceRequired:1b}`,
    ],
  },

  // ===========================================================================
  // DOMAIN 4: HOMESTEADING, FARMING & COMPANION TAMING
  // ===========================================================================
  {
    id: 'homestead_crop_farm',
    title: 'Irrigated Farmland Tilling, Planting & Bone-Meal Harvest',
    domain: 'HOMESTEADING_ECONOMY',
    tier: 2,
    maxDurationSeconds: 50,
    maxSteps: 4,
    prompt: 'Till the dirt next to the water, plant wheat seeds, and use bone meal to grow crops!',
    origin: { x: 600, y: 64, z: 100 },
    initialLoadout: [
      { name: 'iron_hoe', count: 1 },
      { name: 'wheat_seeds', count: 16 },
      { name: 'bone_meal', count: 24 },
    ],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 5} ${y - 1} ${z - 5} ${x + 5} ${y - 1} ${z + 5} minecraft:dirt`,
      `fill ${x - 5} ${y} ${z - 5} ${x + 5} ${y + 4} ${z + 5} minecraft:air`,
      `setblock ${x + 2} ${y - 1} ${z} minecraft:water`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
    ],
  },
  {
    id: 'homestead_wolf_taming',
    title: 'Companion Wolf Taming with Bones',
    domain: 'HOMESTEADING_ECONOMY',
    tier: 2,
    maxDurationSeconds: 40,
    maxSteps: 3,
    prompt: 'can you tame that wolf',
    origin: { x: 650, y: 64, z: 100 },
    initialLoadout: [{ name: 'bone', count: 16 }],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 6} ${y - 1} ${z - 6} ${x + 6} ${y - 1} ${z + 6} minecraft:grass_block`,
      `fill ${x - 6} ${y} ${z - 6} ${x + 6} ${y + 3} ${z + 6} minecraft:oak_fence outline`,
      `fill ${x - 5} ${y} ${z - 5} ${x + 5} ${y + 4} ${z + 5} minecraft:air`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
      `summon minecraft:wolf ${x + 2} ${y} ${z + 1}`,
    ],
  },

  // ===========================================================================
  // DOMAIN 5: PUZZLE ESCAPE ROOMS & REDSTONE ENGINEERING
  // ===========================================================================
  {
    id: 'puzzle_redstone_lever',
    title: 'Redstone Lever Activation & Escape Puzzle',
    domain: 'PUZZLE_ENGINEERING',
    tier: 2,
    maxDurationSeconds: 45,
    maxSteps: 4,
    prompt: 'Flip the lever to unlock the puzzle room and step onto the gold_block!',
    origin: { x: 700, y: 64, z: 100 },
    initialLoadout: [{ name: 'cooked_beef', count: 4 }],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 5} ${y - 1} ${z - 5} ${x + 8} ${y - 1} ${z + 5} minecraft:polished_andesite`,
      `fill ${x - 5} ${y} ${z - 5} ${x + 8} ${y + 4} ${z + 5} minecraft:air`,
      `setblock ${x + 3} ${y + 1} ${z} minecraft:stone_bricks`,
      `setblock ${x + 3} ${y + 1} ${z - 1} minecraft:lever[face=wall,facing=north,powered=false]`,
      `setblock ${x + 6} ${y - 1} ${z} minecraft:gold_block`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
    ],
  },
  {
    id: 'puzzle_tool_preservation_mining',
    title: 'Multi-Strata Excavation & Iron Pickaxe Durability Preservation',
    domain: 'PUZZLE_ENGINEERING',
    tier: 3,
    maxDurationSeconds: 60,
    maxSteps: 4,
    prompt: 'Mine 4 sand and 4 stone without wasting your low-durability iron_pickaxe!',
    origin: { x: 750, y: 64, z: 100 },
    initialLoadout: [
      { name: 'stone_shovel', count: 1 },
      { name: 'stone_pickaxe', count: 1 },
      { name: 'iron_pickaxe', count: 1 },
    ],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 6} ${y - 3} ${z - 6} ${x + 8} ${y - 2} ${z + 6} minecraft:bedrock`,
      `fill ${x - 6} ${y - 1} ${z - 6} ${x + 8} ${y - 1} ${z + 6} minecraft:stone`,
      `fill ${x - 6} ${y} ${z - 6} ${x + 8} ${y + 1} ${z + 6} minecraft:glass outline`,
      `fill ${x - 5} ${y} ${z - 5} ${x + 7} ${y + 5} ${z + 5} minecraft:air`,
      `fill ${x + 2} ${y} ${z - 1} ${x + 3} ${y + 1} ${z + 1} minecraft:sand`,
      `fill ${x + 5} ${y} ${z - 1} ${x + 6} ${y + 1} ${z + 1} minecraft:stone`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
    ],
  },

  // ===========================================================================
  // DOMAIN 6: SURVIVAL PROGRESSION & THE ENDER DRAGON BOSS FIGHT
  // ===========================================================================
  {
    id: 'prog_tier1_stone_age',
    title: 'Tier 1: Timber to Stone Pickaxe Bootstrapping',
    domain: 'PROGRESSION_DRAGON',
    tier: 1,
    maxDurationSeconds: 90,
    maxSteps: 8,
    prompt: 'Chop oak logs, craft a wooden_pickaxe, mine stone, and craft a stone_pickaxe!',
    origin: { x: 800, y: 64, z: 100 },
    initialLoadout: [],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 10} ${y - 4} ${z - 10} ${x + 12} ${y - 3} ${z + 10} minecraft:bedrock`,
      `fill ${x - 10} ${y - 2} ${z - 10} ${x + 12} ${y - 2} ${z + 10} minecraft:stone`,
      `fill ${x - 10} ${y - 1} ${z - 10} ${x + 12} ${y - 1} ${z + 10} minecraft:grass_block`,
      `fill ${x - 10} ${y} ${z - 10} ${x + 12} ${y + 1} ${z + 10} minecraft:glass outline`,
      `fill ${x - 9} ${y} ${z - 9} ${x + 11} ${y + 8} ${z + 9} minecraft:air`,
      // Exposed stone outcrop & coal vein
      `fill ${x - 5} ${y} ${z - 2} ${x - 3} ${y + 1} ${z + 2} minecraft:stone`,
      `fill ${x - 4} ${y} ${z + 1} ${x - 3} ${y + 1} ${z + 2} minecraft:coal_ore`,
      // Natural oak trees with canopies
      `fill ${x + 4} ${y} ${z} ${x + 4} ${y + 4} ${z} minecraft:oak_log`,
      `fill ${x + 3} ${y + 3} ${z - 1} ${x + 5} ${y + 5} ${z + 1} minecraft:oak_leaves`,
      `fill ${x + 7} ${y} ${z + 2} ${x + 7} ${y + 4} ${z + 2} minecraft:oak_log`,
      `fill ${x + 6} ${y + 3} ${z + 1} ${x + 8} ${y + 5} ${z + 3} minecraft:oak_leaves`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
    ],
  },
  {
    id: 'prog_tier3_iron_armor',
    title: 'Tier 3: Full Iron Armor Progression (Mine -> Smelt -> Craft -> Equip)',
    domain: 'PROGRESSION_DRAGON',
    tier: 3,
    maxDurationSeconds: 300,
    maxSteps: 20,
    prompt: 'Get full iron armor and equip it!',
    origin: { x: 850, y: 64, z: 100 },
    initialLoadout: [{ name: 'stone_pickaxe', count: 2 }],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 10} ${y - 4} ${z - 10} ${x + 12} ${y - 2} ${z + 10} minecraft:bedrock`,
      `fill ${x - 10} ${y - 1} ${z - 10} ${x + 12} ${y - 1} ${z + 10} minecraft:grass_block`,
      `fill ${x - 10} ${y} ${z - 10} ${x + 12} ${y + 1} ${z + 10} minecraft:glass outline`,
      `fill ${x - 9} ${y} ${z - 9} ${x + 11} ${y + 8} ${z + 9} minecraft:air`,
      // Natural trees for wood, fuel, and crafting table bootstrapping
      `fill ${x - 6} ${y} ${z + 5} ${x - 6} ${y + 4} ${z + 5} minecraft:oak_log`,
      `fill ${x - 7} ${y + 3} ${z + 4} ${x - 5} ${y + 5} ${z + 6} minecraft:oak_leaves`,
      `fill ${x + 7} ${y} ${z + 6} ${x + 7} ${y + 4} ${z + 6} minecraft:oak_log`,
      `fill ${x + 6} ${y + 3} ${z + 5} ${x + 8} ${y + 5} ${z + 7} minecraft:oak_leaves`,
      // Exposed stone and cobblestone outcrop for furnace crafting
      `fill ${x - 5} ${y} ${z - 3} ${x - 2} ${y + 2} ${z - 1} minecraft:stone`,
      `fill ${x - 4} ${y} ${z - 2} ${x - 3} ${y + 1} ${z - 2} minecraft:cobblestone`,
      // Coal ore vein for furnace fuel
      `fill ${x + 1} ${y} ${z + 4} ${x + 3} ${y + 1} ${z + 5} minecraft:coal_ore`,
      // Rich iron ore vein (28 blocks)
      `fill ${x + 3} ${y} ${z - 2} ${x + 6} ${y + 1} ${z + 1} minecraft:iron_ore`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
    ],
  },
  {
    id: 'prog_tier6_ender_dragon',
    title: 'Tier 6: The End — Destroy End Crystals & Slay the Ender Dragon',
    domain: 'PROGRESSION_DRAGON',
    tier: 6,
    maxDurationSeconds: 180,
    maxSteps: 12,
    prompt:
      'Destroy the end_crystal towers with your bow, eat golden_apple when hurt, and slay the ender_dragon when it perches!',
    origin: { x: 0, y: 64, z: 0 },
    initialLoadout: [
      { name: 'diamond_sword', count: 1 },
      { name: 'bow', count: 1 },
      { name: 'arrow', count: 64 },
      { name: 'shield', count: 1 },
      { name: 'golden_apple', count: 12 },
      { name: 'cooked_beef', count: 32 },
      { name: 'cobblestone', count: 64 },
      { name: 'water_bucket', count: 1 },
      { name: 'diamond_helmet', count: 1 },
      { name: 'diamond_chestplate', count: 1 },
      { name: 'diamond_leggings', count: 1 },
      { name: 'diamond_boots', count: 1 },
    ],
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 24} ${y - 1} ${z - 24} ${x + 24} ${y - 1} ${z + 24} minecraft:end_stone`,
      `fill ${x - 24} ${y} ${z - 24} ${x + 24} ${y + 1} ${z + 24} minecraft:glass outline`,
      `fill ${x - 23} ${y} ${z - 23} ${x + 23} ${y + 18} ${z + 23} minecraft:air`,
      // Obsidian pillars with End Crystals
      `fill ${x + 10} ${y} ${z + 8} ${x + 12} ${y + 6} ${z + 10} minecraft:obsidian`,
      `summon minecraft:end_crystal ${x + 11} ${y + 7} ${z + 9}`,
      `fill ${x - 12} ${y} ${z + 8} ${x - 10} ${y + 6} ${z + 10} minecraft:obsidian`,
      `summon minecraft:end_crystal ${x - 11} ${y + 7} ${z + 9}`,
      // Central bedrock perch fountain
      `fill ${x - 2} ${y} ${z - 2} ${x + 2} ${y} ${z + 2} minecraft:bedrock`,
      `fill ${x} ${y + 1} ${z} ${x} ${y + 3} ${z} minecraft:bedrock`,
      `tp NeuroAgent ${x} ${y} ${z - 6} 0 0`,
      `summon minecraft:ender_dragon ${x} ${y + 4} ${z + 2} {DragonPhase:0,PersistenceRequired:1b}`,
    ],
  },
];

/**
 * Guaranteed Multiplayer Sparring & Competitive Benchmarks:
 * 1. DQN Opponent Baseline Tasks (6 Tasks) - GeorgeDudau Dueling DQN frozen benchmark
 * 2. Self-Sparring / Mirror Tasks (6 Tasks) - Matched clone for dense gradient climbing & self-play
 */
export const DQN_MULTIPLAYER_TASKS: MinebenchTaskSpec[] = [
  {
    id: 'mp_pvp_dqn_melee',
    title: 'Multiplayer 1v1: Sword Duel vs DQN_Opponent',
    domain: 'MULTIPLAYER_ARENA',
    tier: 3,
    maxDurationSeconds: 60,
    maxSteps: 12,
    prompt:
      'Engage and defeat the rival player DQN_Opponent in a 1v1 iron sword arena duel while surviving with positive HP.',
    origin: { x: 1000, y: 64, z: 1000 },
    initialLoadout: [
      { name: 'iron_sword', count: 1 },
      { name: 'iron_chestplate', count: 1 },
      { name: 'iron_leggings', count: 1 },
      { name: 'cooked_beef', count: 8 },
    ],
    sparringConfig: {
      mode: 'pvp_dqn',
      spawnOffset: { dx: 0, dy: 0, dz: 10 },
      loadout: [
        { name: 'iron_sword', count: 1 },
        { name: 'iron_chestplate', count: 1 },
        { name: 'iron_leggings', count: 1 },
      ],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 14} ${y - 1} ${z - 14} ${x + 14} ${y - 1} ${z + 18} minecraft:stone_bricks`,
      `fill ${x - 14} ${y} ${z - 14} ${x + 14} ${y + 8} ${z + 18} minecraft:air`,
      `fill ${x - 15} ${y} ${z - 15} ${x + 15} ${y + 4} ${z - 15} minecraft:obsidian`,
      `fill ${x - 15} ${y} ${z + 19} ${x + 15} ${y + 4} ${z + 19} minecraft:obsidian`,
      `fill ${x - 15} ${y} ${z - 15} ${x - 15} ${y + 4} ${z + 19} minecraft:obsidian`,
      `fill ${x + 15} ${y} ${z - 15} ${x + 15} ${y + 4} ${z + 19} minecraft:obsidian`,
    ],
  },
  {
    id: 'mp_pvp_dqn_axe_shield',
    title: 'Multiplayer 1v1: Axe & Shield Duel vs DQN_Opponent',
    domain: 'MULTIPLAYER_ARENA',
    tier: 4,
    maxDurationSeconds: 60,
    maxSteps: 12,
    prompt:
      'Defeat DQN_Opponent in tactical 1v1 combat using your iron axe and shield, timing blocks and heavy strikes.',
    origin: { x: 1100, y: 64, z: 1000 },
    initialLoadout: [
      { name: 'iron_axe', count: 1 },
      { name: 'shield', count: 1 },
      { name: 'iron_helmet', count: 1 },
      { name: 'iron_boots', count: 1 },
      { name: 'golden_apple', count: 2 },
    ],
    sparringConfig: {
      mode: 'pvp_dqn',
      spawnOffset: { dx: 0, dy: 0, dz: 9 },
      loadout: [
        { name: 'iron_axe', count: 1 },
        { name: 'shield', count: 1 },
        { name: 'iron_helmet', count: 1 },
        { name: 'iron_boots', count: 1 },
      ],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 12} ${y - 1} ${z - 12} ${x + 12} ${y - 1} ${z + 16} minecraft:polished_deepslate`,
      `fill ${x - 12} ${y} ${z - 12} ${x + 12} ${y + 8} ${z + 16} minecraft:air`,
      `fill ${x - 13} ${y} ${z - 13} ${x + 13} ${y + 4} ${z - 13} minecraft:deepslate_bricks`,
      `fill ${x - 13} ${y} ${z + 17} ${x + 13} ${y + 4} ${z + 17} minecraft:deepslate_bricks`,
      `fill ${x - 13} ${y} ${z - 13} ${x - 13} ${y + 4} ${z + 17} minecraft:deepslate_bricks`,
      `fill ${x + 13} ${y} ${z - 13} ${x + 13} ${y + 4} ${z + 17} minecraft:deepslate_bricks`,
    ],
  },
  {
    id: 'mp_pvp_dqn_bow_arena',
    title: 'Multiplayer 1v1: Pillar Arena Skirmish vs DQN_Opponent',
    domain: 'MULTIPLAYER_ARENA',
    tier: 4,
    maxDurationSeconds: 65,
    maxSteps: 14,
    prompt:
      'Use cover pillars, bow shots, and melee strikes to eliminate DQN_Opponent in the skirmish arena.',
    origin: { x: 1200, y: 64, z: 1000 },
    initialLoadout: [
      { name: 'bow', count: 1 },
      { name: 'arrow', count: 24 },
      { name: 'stone_sword', count: 1 },
      { name: 'leather_chestplate', count: 1 },
      { name: 'cooked_beef', count: 6 },
    ],
    sparringConfig: {
      mode: 'pvp_dqn',
      spawnOffset: { dx: 0, dy: 0, dz: 14 },
      loadout: [
        { name: 'stone_sword', count: 1 },
        { name: 'leather_chestplate', count: 1 },
      ],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 14} ${y - 1} ${z - 10} ${x + 14} ${y - 1} ${z + 22} minecraft:sandstone`,
      `fill ${x - 14} ${y} ${z - 10} ${x + 14} ${y + 9} ${z + 22} minecraft:air`,
      `fill ${x - 15} ${y} ${z - 11} ${x + 15} ${y + 4} ${z - 11} minecraft:cut_sandstone`,
      `fill ${x - 15} ${y} ${z + 23} ${x + 15} ${y + 4} ${z + 23} minecraft:cut_sandstone`,
      `fill ${x - 15} ${y} ${z - 11} ${x - 15} ${y + 4} ${z + 23} minecraft:cut_sandstone`,
      `fill ${x + 15} ${y} ${z - 11} ${x + 15} ${y + 4} ${z + 23} minecraft:cut_sandstone`,
      // Cover pillars
      `fill ${x - 5} ${y} ${z + 5} ${x - 4} ${y + 3} ${z + 6} minecraft:cut_sandstone`,
      `fill ${x + 4} ${y} ${z + 8} ${x + 5} ${y + 3} ${z + 9} minecraft:cut_sandstone`,
    ],
  },
  {
    id: 'mp_bridge_race',
    title: 'Multiplayer Race: Parallel Bridge vs DQN_Opponent',
    domain: 'MULTIPLAYER_ARENA',
    tier: 3,
    maxDurationSeconds: 55,
    maxSteps: 14,
    prompt:
      'Bridge across the 18-block gap to the emerald_block goal platform ahead before DQN_Opponent reaches its goal.',
    origin: { x: 1300, y: 72, z: 1000 },
    initialLoadout: [{ name: 'cobblestone', count: 64 }],
    sparringConfig: {
      mode: 'bridge_race',
      spawnOffset: { dx: 6, dy: 0, dz: 0 },
      goalOffset: { dx: 6, dy: 0, dz: 20 },
      loadout: [{ name: 'cobblestone', count: 64 }],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 4} ${y - 6} ${z - 4} ${x + 10} ${y + 8} ${z + 25} minecraft:air`,
      // Enclosed catch floor at y-4 with a 2-block-high perimeter glass wall (y-3..y-2) carved hollow inside
      `fill ${x - 4} ${y - 4} ${z - 4} ${x + 10} ${y - 2} ${z + 25} minecraft:glass`,
      `fill ${x - 3} ${y - 3} ${z - 3} ${x + 9} ${y - 2} ${z + 24} minecraft:air`,
      // Start platforms
      `fill ${x - 2} ${y - 1} ${z - 2} ${x + 2} ${y - 1} ${z + 2} minecraft:stone`,
      `fill ${x + 4} ${y - 1} ${z - 2} ${x + 8} ${y - 1} ${z + 2} minecraft:stone`,
      // Goal platforms
      `fill ${x - 2} ${y - 1} ${z + 18} ${x + 2} ${y - 1} ${z + 22} minecraft:emerald_block`,
      `fill ${x + 4} ${y - 1} ${z + 18} ${x + 8} ${y - 1} ${z + 22} minecraft:emerald_block`,
    ],
  },
  {
    id: 'mp_mob_escort_race',
    title: 'Multiplayer Race: Extraction Sprint vs DQN_Opponent',
    domain: 'MULTIPLAYER_ARENA',
    tier: 3,
    maxDurationSeconds: 50,
    maxSteps: 12,
    prompt:
      'Navigate past the obstacle lane and reach the gold_block extraction beacon ahead before DQN_Opponent.',
    origin: { x: 1400, y: 64, z: 1000 },
    initialLoadout: [
      { name: 'iron_sword', count: 1 },
      { name: 'cooked_beef', count: 4 },
    ],
    sparringConfig: {
      mode: 'escort_race',
      spawnOffset: { dx: 5, dy: 0, dz: 0 },
      goalOffset: { dx: 5, dy: 0, dz: 20 },
      loadout: [{ name: 'iron_sword', count: 1 }],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 4} ${y - 1} ${z - 3} ${x + 9} ${y - 1} ${z + 23} minecraft:smooth_stone`,
      `fill ${x - 4} ${y} ${z - 3} ${x + 9} ${y + 1} ${z + 23} minecraft:glass outline`,
      `fill ${x - 3} ${y} ${z - 2} ${x + 8} ${y + 6} ${z + 22} minecraft:air`,
      // Obstacles in lane
      `fill ${x - 2} ${y} ${z + 7} ${x + 2} ${y} ${z + 7} minecraft:oak_fence`,
      `fill ${x + 3} ${y} ${z + 12} ${x + 7} ${y} ${z + 12} minecraft:oak_fence`,
      // Goal beacons
      `fill ${x - 1} ${y - 1} ${z + 19} ${x + 1} ${y - 1} ${z + 21} minecraft:gold_block`,
      `fill ${x + 4} ${y - 1} ${z + 19} ${x + 6} ${y - 1} ${z + 21} minecraft:gold_block`,
    ],
  },
  {
    id: 'mp_resource_duel',
    title: 'Multiplayer Duel: Timber Harvest Race vs DQN_Opponent',
    domain: 'MULTIPLAYER_ARENA',
    tier: 2,
    maxDurationSeconds: 50,
    maxSteps: 12,
    prompt:
      'Harvest at least 4 oak logs from the shared timber grove faster than DQN_Opponent.',
    origin: { x: 1500, y: 64, z: 1000 },
    initialLoadout: [{ name: 'iron_axe', count: 1 }],
    sparringConfig: {
      mode: 'resource_race',
      spawnOffset: { dx: 6, dy: 0, dz: 0 },
      targetBlockName: 'oak_log',
      loadout: [{ name: 'iron_axe', count: 1 }],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 8} ${y - 1} ${z - 6} ${x + 14} ${y - 1} ${z + 14} minecraft:grass_block`,
      `fill ${x - 8} ${y} ${z - 6} ${x + 14} ${y + 1} ${z + 14} minecraft:glass outline`,
      `fill ${x - 7} ${y} ${z - 5} ${x + 13} ${y + 10} ${z + 13} minecraft:air`,
      // Shared oak log pillars (fast, reliable 1.20.4 timber columns)
      `fill ${x - 3} ${y} ${z + 4} ${x - 3} ${y + 4} ${z + 4} minecraft:oak_log`,
      `fill ${x + 2} ${y} ${z + 5} ${x + 2} ${y + 4} ${z + 5} minecraft:oak_log`,
      `fill ${x + 7} ${y} ${z + 4} ${x + 7} ${y + 4} ${z + 4} minecraft:oak_log`,
      `fill ${x} ${y} ${z + 8} ${x} ${y + 4} ${z + 8} minecraft:oak_log`,
      `fill ${x + 5} ${y} ${z + 8} ${x + 5} ${y + 4} ${z + 8} minecraft:oak_log`,
    ],
  },
];

/**
 * Self-Sparring / Mirror Clone Tasks (6 Tasks).
 * Pits the agent against a mirror clone (MirrorAgent) matching its own speed,
 * humanized equipment, and kinematics, providing a climbable reward gradient and ~50% win dynamics.
 */
export const SELF_SPARRING_TASKS: MinebenchTaskSpec[] = [
  {
    id: 'mp_pvp_self_melee',
    title: 'Multiplayer 1v1: Mirror Sword Duel (Self-Play)',
    domain: 'MULTIPLAYER_ARENA',
    tier: 3,
    maxDurationSeconds: 60,
    maxSteps: 12,
    prompt:
      'Engage and spar with your mirror clone MirrorAgent in a 1v1 iron sword arena duel while surviving with positive HP.',
    origin: { x: 2000, y: 64, z: 1000 },
    initialLoadout: [
      { name: 'iron_sword', count: 1 },
      { name: 'iron_chestplate', count: 1 },
      { name: 'iron_leggings', count: 1 },
      { name: 'cooked_beef', count: 8 },
    ],
    sparringConfig: {
      mode: 'pvp_self',
      spawnOffset: { dx: 0, dy: 0, dz: 10 },
      loadout: [
        { name: 'iron_sword', count: 1 },
        { name: 'iron_chestplate', count: 1 },
        { name: 'iron_leggings', count: 1 },
      ],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 14} ${y - 1} ${z - 14} ${x + 14} ${y - 1} ${z + 18} minecraft:stone_bricks`,
      `fill ${x - 14} ${y} ${z - 14} ${x + 14} ${y + 8} ${z + 18} minecraft:air`,
      `fill ${x - 15} ${y} ${z - 15} ${x + 15} ${y + 4} ${z - 15} minecraft:obsidian`,
      `fill ${x - 15} ${y} ${z + 19} ${x + 15} ${y + 4} ${z + 19} minecraft:obsidian`,
      `fill ${x - 15} ${y} ${z - 15} ${x - 15} ${y + 4} ${z + 19} minecraft:obsidian`,
      `fill ${x + 15} ${y} ${z - 15} ${x + 15} ${y + 4} ${z + 19} minecraft:obsidian`,
    ],
  },
  {
    id: 'mp_pvp_self_axe_shield',
    title: 'Multiplayer 1v1: Mirror Axe & Shield Duel (Self-Play)',
    domain: 'MULTIPLAYER_ARENA',
    tier: 4,
    maxDurationSeconds: 60,
    maxSteps: 12,
    prompt:
      'Engage MirrorAgent in tactical 1v1 combat using iron axe and shield, timing blocks and counter-strikes.',
    origin: { x: 2100, y: 64, z: 1000 },
    initialLoadout: [
      { name: 'iron_axe', count: 1 },
      { name: 'shield', count: 1 },
      { name: 'iron_helmet', count: 1 },
      { name: 'iron_boots', count: 1 },
      { name: 'golden_apple', count: 2 },
    ],
    sparringConfig: {
      mode: 'pvp_self',
      spawnOffset: { dx: 0, dy: 0, dz: 9 },
      loadout: [
        { name: 'iron_axe', count: 1 },
        { name: 'shield', count: 1 },
        { name: 'iron_helmet', count: 1 },
        { name: 'iron_boots', count: 1 },
      ],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 12} ${y - 1} ${z - 12} ${x + 12} ${y - 1} ${z + 16} minecraft:polished_deepslate`,
      `fill ${x - 12} ${y} ${z - 12} ${x + 12} ${y + 8} ${z + 16} minecraft:air`,
      `fill ${x - 13} ${y} ${z - 13} ${x + 13} ${y + 4} ${z - 13} minecraft:deepslate_bricks`,
      `fill ${x - 13} ${y} ${z + 17} ${x + 13} ${y + 4} ${z + 17} minecraft:deepslate_bricks`,
      `fill ${x - 13} ${y} ${z - 13} ${x - 13} ${y + 4} ${z + 17} minecraft:deepslate_bricks`,
      `fill ${x + 13} ${y} ${z - 13} ${x + 13} ${y + 4} ${z + 17} minecraft:deepslate_bricks`,
    ],
  },
  {
    id: 'mp_pvp_self_bow_arena',
    title: 'Multiplayer 1v1: Mirror Pillar Arena Skirmish (Self-Play)',
    domain: 'MULTIPLAYER_ARENA',
    tier: 4,
    maxDurationSeconds: 65,
    maxSteps: 14,
    prompt:
      'Use cover pillars, bow shots, and melee strikes to spar with MirrorAgent in the skirmish arena.',
    origin: { x: 2200, y: 64, z: 1000 },
    initialLoadout: [
      { name: 'bow', count: 1 },
      { name: 'arrow', count: 24 },
      { name: 'stone_sword', count: 1 },
      { name: 'leather_chestplate', count: 1 },
      { name: 'cooked_beef', count: 6 },
    ],
    sparringConfig: {
      mode: 'pvp_self',
      spawnOffset: { dx: 0, dy: 0, dz: 14 },
      loadout: [
        { name: 'stone_sword', count: 1 },
        { name: 'leather_chestplate', count: 1 },
      ],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 14} ${y - 1} ${z - 10} ${x + 14} ${y - 1} ${z + 22} minecraft:sandstone`,
      `fill ${x - 14} ${y} ${z - 10} ${x + 14} ${y + 9} ${z + 22} minecraft:air`,
      `fill ${x - 15} ${y} ${z - 11} ${x + 15} ${y + 4} ${z - 11} minecraft:cut_sandstone`,
      `fill ${x - 15} ${y} ${z + 23} ${x + 15} ${y + 4} ${z + 23} minecraft:cut_sandstone`,
      `fill ${x - 15} ${y} ${z - 11} ${x - 15} ${y + 4} ${z + 23} minecraft:cut_sandstone`,
      `fill ${x + 15} ${y} ${z - 11} ${x + 15} ${y + 4} ${z + 23} minecraft:cut_sandstone`,
      `fill ${x - 5} ${y} ${z + 5} ${x - 4} ${y + 3} ${z + 6} minecraft:cut_sandstone`,
      `fill ${x + 4} ${y} ${z + 8} ${x + 5} ${y + 3} ${z + 9} minecraft:cut_sandstone`,
    ],
  },
  {
    id: 'mp_bridge_race_self',
    title: 'Multiplayer Race: Parallel Bridge vs Mirror Partner (Self-Play)',
    domain: 'MULTIPLAYER_ARENA',
    tier: 3,
    maxDurationSeconds: 55,
    maxSteps: 14,
    prompt:
      'Bridge across the 18-block gap to the emerald_block goal platform ahead in friendly competition with MirrorAgent.',
    origin: { x: 2300, y: 72, z: 1000 },
    initialLoadout: [{ name: 'cobblestone', count: 64 }],
    sparringConfig: {
      mode: 'bridge_race',
      spawnOffset: { dx: 6, dy: 0, dz: 0 },
      goalOffset: { dx: 6, dy: 0, dz: 20 },
      loadout: [{ name: 'cobblestone', count: 64 }],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 4} ${y - 6} ${z - 4} ${x + 10} ${y + 8} ${z + 25} minecraft:air`,
      // Enclosed catch floor at y-4 with a 2-block-high perimeter glass wall (y-3..y-2) carved hollow inside
      `fill ${x - 4} ${y - 4} ${z - 4} ${x + 10} ${y - 2} ${z + 25} minecraft:glass`,
      `fill ${x - 3} ${y - 3} ${z - 3} ${x + 9} ${y - 2} ${z + 24} minecraft:air`,
      `fill ${x - 2} ${y - 1} ${z - 2} ${x + 2} ${y - 1} ${z + 2} minecraft:stone`,
      `fill ${x + 4} ${y - 1} ${z - 2} ${x + 8} ${y - 1} ${z + 2} minecraft:stone`,
      `fill ${x - 2} ${y - 1} ${z + 18} ${x + 2} ${y - 1} ${z + 22} minecraft:emerald_block`,
      `fill ${x + 4} ${y - 1} ${z + 18} ${x + 8} ${y - 1} ${z + 22} minecraft:emerald_block`,
    ],
  },
  {
    id: 'mp_mob_escort_race_self',
    title: 'Multiplayer Race: Extraction Sprint vs Mirror Partner (Self-Play)',
    domain: 'MULTIPLAYER_ARENA',
    tier: 3,
    maxDurationSeconds: 50,
    maxSteps: 12,
    prompt:
      'Navigate past the obstacle lane and reach the gold_block extraction beacon ahead in competition with MirrorAgent.',
    origin: { x: 2400, y: 64, z: 1000 },
    initialLoadout: [
      { name: 'iron_sword', count: 1 },
      { name: 'cooked_beef', count: 4 },
    ],
    sparringConfig: {
      mode: 'escort_race',
      spawnOffset: { dx: 5, dy: 0, dz: 0 },
      goalOffset: { dx: 5, dy: 0, dz: 20 },
      loadout: [{ name: 'iron_sword', count: 1 }],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 4} ${y - 1} ${z - 3} ${x + 9} ${y - 1} ${z + 23} minecraft:smooth_stone`,
      `fill ${x - 4} ${y} ${z - 3} ${x + 9} ${y + 1} ${z + 23} minecraft:glass outline`,
      `fill ${x - 3} ${y} ${z - 2} ${x + 8} ${y + 6} ${z + 22} minecraft:air`,
      `fill ${x - 2} ${y} ${z + 7} ${x + 2} ${y} ${z + 7} minecraft:oak_fence`,
      `fill ${x + 3} ${y} ${z + 12} ${x + 7} ${y} ${z + 12} minecraft:oak_fence`,
      `fill ${x - 1} ${y - 1} ${z + 19} ${x + 1} ${y - 1} ${z + 21} minecraft:gold_block`,
      `fill ${x + 4} ${y - 1} ${z + 19} ${x + 6} ${y - 1} ${z + 21} minecraft:gold_block`,
    ],
  },
  {
    id: 'mp_resource_duel_self',
    title: 'Multiplayer Duel: Timber Harvest Race vs Mirror Partner (Self-Play)',
    domain: 'MULTIPLAYER_ARENA',
    tier: 2,
    maxDurationSeconds: 50,
    maxSteps: 12,
    prompt:
      'Harvest at least 4 oak logs from the shared timber grove in competition with MirrorAgent.',
    origin: { x: 2500, y: 64, z: 1000 },
    initialLoadout: [{ name: 'iron_axe', count: 1 }],
    sparringConfig: {
      mode: 'resource_race',
      spawnOffset: { dx: 6, dy: 0, dz: 0 },
      targetBlockName: 'oak_log',
      loadout: [{ name: 'iron_axe', count: 1 }],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 8} ${y - 1} ${z - 6} ${x + 14} ${y - 1} ${z + 14} minecraft:grass_block`,
      `fill ${x - 8} ${y} ${z - 6} ${x + 14} ${y + 1} ${z + 14} minecraft:glass outline`,
      `fill ${x - 7} ${y} ${z - 5} ${x + 13} ${y + 10} ${z + 13} minecraft:air`,
      `fill ${x - 3} ${y} ${z + 4} ${x - 3} ${y + 4} ${z + 4} minecraft:oak_log`,
      `fill ${x + 2} ${y} ${z + 5} ${x + 2} ${y + 4} ${z + 5} minecraft:oak_log`,
      `fill ${x + 7} ${y} ${z + 4} ${x + 7} ${y + 4} ${z + 4} minecraft:oak_log`,
      `fill ${x} ${y} ${z + 8} ${x} ${y + 4} ${z + 8} minecraft:oak_log`,
      `fill ${x + 5} ${y} ${z + 8} ${x + 5} ${y + 4} ${z + 8} minecraft:oak_log`,
    ],
  },
];

/**
 * Cooperative Companion / Teammate Tasks (6 Tasks).
 * Evaluates the agent as a faithful, responsive companion to human players on any server:
 * item sharing, DAG crafting & delivery, companion following, bodyguard defense, and resource sharing.
 */
export const COOPERATIVE_COMPANION_TASKS: MinebenchTaskSpec[] = [
  {
    id: 'coop_give_food',
    title: 'Cooperative Companion: Feed Hungry Teammate',
    domain: 'MULTIPLAYER_ARENA',
    tier: 1,
    maxDurationSeconds: 45,
    maxSteps: 8,
    prompt:
      'Companion_Player says: "I am starving! Can you give me some cooked beef?" Deliver cooked beef to Companion_Player.',
    origin: { x: 3000, y: 64, z: 1000 },
    initialLoadout: [{ name: 'cooked_beef', count: 8 }],
    sparringConfig: {
      mode: 'companion_wait',
      spawnOffset: { dx: 4, dy: 0, dz: 4 },
      loadout: [],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 8} ${y - 1} ${z - 8} ${x + 8} ${y - 1} ${z + 8} minecraft:smooth_stone`,
      `fill ${x - 8} ${y} ${z - 8} ${x + 8} ${y + 5} ${z + 8} minecraft:air`,
      `fill ${x - 8} ${y} ${z - 8} ${x + 8} ${y + 2} ${z + 8} minecraft:glass outline`,
      `fill ${x - 7} ${y} ${z - 7} ${x + 7} ${y + 2} ${z + 7} minecraft:air`,
    ],
  },
  {
    id: 'coop_give_tool',
    title: 'Cooperative Companion: Deliver Weapon to Teammate',
    domain: 'MULTIPLAYER_ARENA',
    tier: 1,
    maxDurationSeconds: 45,
    maxSteps: 8,
    prompt:
      'Companion_Player says: "I need a weapon! Toss me an iron sword." Deliver an iron sword to Companion_Player.',
    origin: { x: 3100, y: 64, z: 1000 },
    initialLoadout: [{ name: 'iron_sword', count: 2 }],
    sparringConfig: {
      mode: 'companion_wait',
      spawnOffset: { dx: 4, dy: 0, dz: 4 },
      loadout: [],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 8} ${y - 1} ${z - 8} ${x + 8} ${y - 1} ${z + 8} minecraft:smooth_stone`,
      `fill ${x - 8} ${y} ${z - 8} ${x + 8} ${y + 5} ${z + 8} minecraft:air`,
      `fill ${x - 8} ${y} ${z - 8} ${x + 8} ${y + 2} ${z + 8} minecraft:glass outline`,
      `fill ${x - 7} ${y} ${z - 7} ${x + 7} ${y + 2} ${z + 7} minecraft:air`,
    ],
  },
  {
    id: 'coop_craft_and_deliver',
    title: 'Cooperative Companion: Craft & Deliver Tool on Demand',
    domain: 'MULTIPLAYER_ARENA',
    tier: 3,
    maxDurationSeconds: 65,
    maxSteps: 12,
    prompt:
      'Companion_Player says: "I need a wooden pickaxe, please craft one and give it to me." Craft a wooden pickaxe and give it to Companion_Player.',
    origin: { x: 3200, y: 64, z: 1000 },
    initialLoadout: [{ name: 'oak_log', count: 4 }],
    sparringConfig: {
      mode: 'companion_wait',
      spawnOffset: { dx: 4, dy: 0, dz: 4 },
      loadout: [],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 8} ${y - 1} ${z - 8} ${x + 8} ${y - 1} ${z + 8} minecraft:smooth_stone`,
      `fill ${x - 8} ${y} ${z - 8} ${x + 8} ${y + 5} ${z + 8} minecraft:air`,
      `fill ${x - 8} ${y} ${z - 8} ${x + 8} ${y + 2} ${z + 8} minecraft:glass outline`,
      `fill ${x - 7} ${y} ${z - 7} ${x + 7} ${y + 2} ${z + 7} minecraft:air`,
    ],
  },
  {
    id: 'coop_follow_companion',
    title: 'Cooperative Companion: Follow & Escort Teammate',
    domain: 'MULTIPLAYER_ARENA',
    tier: 2,
    maxDurationSeconds: 50,
    maxSteps: 10,
    prompt:
      'Companion_Player says: "Follow me and stay close while we travel to the beacon." Follow Companion_Player and stay within 4 blocks.',
    origin: { x: 3300, y: 64, z: 1000 },
    initialLoadout: [],
    sparringConfig: {
      mode: 'escort_race',
      spawnOffset: { dx: 3, dy: 0, dz: 0 },
      goalOffset: { dx: 3, dy: 0, dz: 20 },
      loadout: [],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 4} ${y - 1} ${z - 3} ${x + 8} ${y - 1} ${z + 23} minecraft:smooth_stone`,
      `fill ${x - 4} ${y} ${z - 3} ${x + 8} ${y + 5} ${z + 23} minecraft:air`,
      `fill ${x - 4} ${y} ${z - 3} ${x + 8} ${y + 2} ${z + 23} minecraft:glass outline`,
      `fill ${x - 3} ${y} ${z - 2} ${x + 7} ${y + 2} ${z + 22} minecraft:air`,
      `fill ${x + 2} ${y - 1} ${z + 19} ${x + 4} ${y - 1} ${z + 21} minecraft:gold_block`,
    ],
  },
  {
    id: 'coop_bodyguard_protect',
    title: 'Cooperative Companion: Bodyguard Defense Against Hostile Mobs',
    domain: 'MULTIPLAYER_ARENA',
    tier: 3,
    maxDurationSeconds: 55,
    maxSteps: 10,
    prompt:
      'Companion_Player says: "Protect me from that approaching zombie!" Eliminate the hostile zombie before it hurts Companion_Player.',
    origin: { x: 3400, y: 64, z: 1000 },
    initialLoadout: [
      { name: 'iron_sword', count: 1 },
      { name: 'shield', count: 1 },
    ],
    sparringConfig: {
      mode: 'companion_wait',
      spawnOffset: { dx: 3, dy: 0, dz: 3 },
      loadout: [],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 8} ${y - 1} ${z - 8} ${x + 8} ${y - 1} ${z + 14} minecraft:smooth_stone`,
      `fill ${x - 8} ${y} ${z - 8} ${x + 8} ${y + 6} ${z + 14} minecraft:air`,
      `fill ${x - 9} ${y} ${z - 9} ${x + 9} ${y + 3} ${z + 15} minecraft:glass outline`,
      `fill ${x - 8} ${y} ${z - 8} ${x + 8} ${y + 3} ${z + 14} minecraft:air`,
      `summon zombie ${x + 3} ${y} ${z + 11} {ArmorItems:[{},{},{},{id:"leather_helmet",Count:1b}]}`,
    ],
  },
  {
    id: 'coop_resource_share',
    title: 'Cooperative Companion: Harvest & Share Resources',
    domain: 'MULTIPLAYER_ARENA',
    tier: 2,
    maxDurationSeconds: 55,
    maxSteps: 12,
    prompt:
      'Companion_Player says: "Can you harvest 4 oak logs from that grove and bring them to me?" Harvest oak logs and deliver them to Companion_Player.',
    origin: { x: 3500, y: 64, z: 1000 },
    initialLoadout: [{ name: 'iron_axe', count: 1 }],
    sparringConfig: {
      mode: 'companion_wait',
      spawnOffset: { dx: 4, dy: 0, dz: 4 },
      loadout: [],
    },
    buildArenaCommands: ({ x, y, z }) => [
      `fill ${x - 8} ${y - 1} ${z - 6} ${x + 14} ${y - 1} ${z + 14} minecraft:grass_block`,
      `fill ${x - 8} ${y} ${z - 6} ${x + 14} ${y + 10} ${z + 14} minecraft:air`,
      `fill ${x - 8} ${y} ${z - 6} ${x + 14} ${y + 2} ${z + 14} minecraft:glass outline`,
      `fill ${x - 7} ${y} ${z - 5} ${x + 13} ${y + 2} ${z + 13} minecraft:air`,
      `fill ${x - 3} ${y} ${z + 6} ${x - 3} ${y + 4} ${z + 6} minecraft:oak_log`,
      `fill ${x + 2} ${y} ${z + 7} ${x + 2} ${y + 4} ${z + 7} minecraft:oak_log`,
    ],
  },
];

export const MULTIPLAYER_TASKS: MinebenchTaskSpec[] = [
  ...COOPERATIVE_COMPANION_TASKS,
  ...SELF_SPARRING_TASKS,
  ...DQN_MULTIPLAYER_TASKS,
];
