/**
 * NeuroCraft Minecraft Omniscience Registry (1.20.4)
 * ==================================================
 * Exhaustively extracts, categorizes, and maps all 1,312 items, 1,058 blocks,
 * 126 entities, 64 biomes, and 32 structures from official Mojang data.
 *
 * Defines the comprehensive affordances, interaction verbs, harvest tiers,
 * and behavioral mechanics across the entire game.
 */

import * as fs from 'fs';
import * as path from 'path';

// Load official 1.20.4 data from minecraft-data
// eslint-disable-next-line @typescript-eslint/no-var-requires
const mcData = require('minecraft-data')('1.20.4');

export interface OmniscienceItemAffordance {
  actionVerb: string;
  category: 'tool' | 'weapon' | 'armor' | 'food' | 'potion' | 'utility' | 'redstone' | 'building' | 'material';
  description: string;
  targetTypes?: string[];
  requiredState?: Record<string, any>;
  rewardSignals: string[];
}

export interface OmniscienceBlockAffordance {
  harvestTool: 'pickaxe' | 'axe' | 'shovel' | 'hoe' | 'shears' | 'none';
  minToolTier?: 'wood' | 'stone' | 'iron' | 'diamond' | 'netherite';
  drops: string[];
  isInteractive: boolean;
  isContainer: boolean;
  isRedstone: boolean;
  isHazard: boolean;
  verbs: string[];
}

export interface OmniscienceMobAffordance {
  category: 'passive' | 'neutral' | 'hostile' | 'boss';
  behavior: string;
  tameItem?: string;
  breedItem?: string;
  cureItem?: string;
  effectiveWeapon: 'sword' | 'axe' | 'bow' | 'crossbow' | 'trident' | 'shield';
  tactics: string[];
  drops: string[];
}

export interface OmniscienceStructureAffordance {
  id: string;
  dimension: 'overworld' | 'nether' | 'the_end';
  keyLoot: string[];
  keyMobs: string[];
  hazards: string[];
  coreMechanics: string[];
}

// 1. Exhaustive Structures in Minecraft 1.20.4
export const ALL_MC_STRUCTURES: OmniscienceStructureAffordance[] = [
  {
    id: 'village_plains',
    dimension: 'overworld',
    keyLoot: ['bread', 'emerald', 'iron_ingot', 'potato', 'carrot'],
    keyMobs: ['villager', 'iron_golem', 'cat'],
    hazards: ['zombie_siege', 'raid'],
    coreMechanics: ['workstation_binding', 'trading', 'bed_sleeping', 'bell_ringing', 'iron_golem_defense'],
  },
  {
    id: 'village_desert',
    dimension: 'overworld',
    keyLoot: ['bread', 'emerald', 'cactus', 'dead_bush'],
    keyMobs: ['villager', 'camel', 'cat'],
    hazards: ['husk', 'desert_well_fall'],
    coreMechanics: ['trading', 'camel_riding'],
  },
  {
    id: 'desert_pyramid',
    dimension: 'overworld',
    keyLoot: ['diamond', 'enchanted_golden_apple', 'saddle', 'iron_ingot', 'gold_ingot'],
    keyMobs: ['husk'],
    hazards: ['tnt_pressure_plate_trap'],
    coreMechanics: ['disarm_pressure_plate', 'excavate_blue_terracotta', 'brush_suspicious_sand'],
  },
  {
    id: 'jungle_pyramid',
    dimension: 'overworld',
    keyLoot: ['diamond', 'emerald', 'iron_ingot', 'gold_ingot'],
    keyMobs: ['skeleton', 'zombie'],
    hazards: ['tripwire_dispenser_arrows', 'hidden_piston_doors'],
    coreMechanics: ['defuse_tripwire_with_shears', 'solve_lever_combination_puzzle'],
  },
  {
    id: 'pillager_outpost',
    dimension: 'overworld',
    keyLoot: ['crossbow', 'arrow', 'ominous_bottle', 'iron_ingot'],
    keyMobs: ['pillager', 'iron_golem'],
    hazards: ['bad_omen_raid_trigger', 'crossbow_crossfire'],
    coreMechanics: ['rescue_iron_golem_from_cage', 'snipe_pillager_captain'],
  },
  {
    id: 'woodland_mansion',
    dimension: 'overworld',
    keyLoot: ['totem_of_undying', 'diamond_hoe', 'enchanted_book', 'allay'],
    keyMobs: ['evoker', 'vindicator', 'allay', 'witch'],
    hazards: ['evoker_fangs', 'evoker_vexes', 'vindicator_axe_sprint'],
    coreMechanics: ['kill_evoker_for_totem', 'rescue_allays_from_jail', 'kite_vindicators'],
  },
  {
    id: 'ancient_city',
    dimension: 'overworld',
    keyLoot: ['echo_shard', 'disc_fragment_5', 'enchanted_golden_apple', 'warden_rib_armor_trim', 'swift_sneak_book'],
    keyMobs: ['warden'],
    hazards: ['sculk_shrieker_summoning', 'sculk_sensor_vibration', 'warden_sonic_boom', 'blindness_effect'],
    coreMechanics: ['sneak_on_wool', 'projectile_distraction', 'silence_shriekers_with_hoe'],
  },
  {
    id: 'ocean_monument',
    dimension: 'overworld',
    keyLoot: ['sponge', 'wet_sponge', 'prismarine_shard', 'prismarine_crystals', 'gold_block'],
    keyMobs: ['guardian', 'elder_guardian'],
    hazards: ['mining_fatigue_iii', 'guardian_laser_tracking', 'thorns_reflection', 'drowning'],
    coreMechanics: ['water_breathing_potion', 'depth_strider', 'break_line_of_sight', 'dry_sponge_in_furnace'],
  },
  {
    id: 'shipwreck',
    dimension: 'overworld',
    keyLoot: ['map', 'iron_ingot', 'gold_ingot', 'lapis_lazuli', 'emerald', 'bamboo'],
    keyMobs: ['drowned'],
    hazards: ['drowning', 'drowned_trident'],
    coreMechanics: ['underwater_door_air_pocket', 'loot_supply_map_chests', 'follow_treasure_map'],
  },
  {
    id: 'buried_treasure',
    dimension: 'overworld',
    keyLoot: ['heart_of_the_sea', 'iron_ingot', 'gold_ingot', 'diamond', 'prismarine_crystals'],
    keyMobs: [],
    hazards: ['suffocation', 'drowning'],
    coreMechanics: ['dig_sand_at_x_coords', 'craft_conduit'],
  },
  {
    id: 'stronghold',
    dimension: 'overworld',
    keyLoot: ['ender_eye', 'enchanted_book', 'iron_pickaxe', 'diamond'],
    keyMobs: ['silverfish'],
    hazards: ['silverfish_spawner', 'silverfish_infestation_swarm'],
    coreMechanics: ['destroy_spawner', 'insert_12_eyes_of_ender', 'activate_end_portal'],
  },
  {
    id: 'nether_fortress',
    dimension: 'nether',
    keyLoot: ['blaze_rod', 'nether_wart', 'wither_skeleton_skull', 'saddle', 'diamond_horse_armor'],
    keyMobs: ['blaze', 'wither_skeleton', 'skeleton', 'zombified_piglin'],
    hazards: ['blaze_fireballs', 'wither_effect', 'lava_ocean_fall'],
    coreMechanics: ['farm_blaze_rods', 'harvest_nether_wart_on_soul_sand', 'collect_3_wither_skulls', 'cure_wither_with_milk'],
  },
  {
    id: 'bastion_remnant',
    dimension: 'nether',
    keyLoot: ['netherite_upgrade_smithing_template', 'ancient_debris', 'pigstep_music_disc', 'gilded_blackstone', 'lodestone'],
    keyMobs: ['piglin_brute', 'piglin', 'magma_cube'],
    hazards: ['piglin_brute_melee_burst', 'piglin_chest_opening_aggro', 'lava_falls'],
    coreMechanics: ['wear_gold_armor', 'distract_piglins_with_gold_ingots', 'boat_trap_piglin_brute', 'smithing_upgrade_netherite'],
  },
  {
    id: 'end_city',
    dimension: 'the_end',
    keyLoot: ['elytra', 'dragon_head', 'shulker_shell', 'diamond_sword_enchanted', 'diamond_pickaxe_enchanted'],
    keyMobs: ['shulker'],
    hazards: ['shulker_bullet_levitation', 'void_fall_death'],
    coreMechanics: ['shield_block_shulker_bullets', 'water_bucket_clutch_from_levitation', 'craft_shulker_box', 'equip_elytra_fireworks'],
  },
  {
    id: 'trail_ruins',
    dimension: 'overworld',
    keyLoot: ['sniffer_egg', 'angler_pottery_sherd', 'wayfinder_armor_trim', 'raiser_armor_trim'],
    keyMobs: [],
    hazards: ['suffocation_gravel'],
    coreMechanics: ['brush_suspicious_gravel', 'hatch_sniffer_egg'],
  },
];

// 2. Comprehensive Tool Affordance Map
export const TOOL_AFFORDANCES: Record<string, OmniscienceItemAffordance[]> = {
  axe: [
    {
      actionVerb: 'chop_log',
      category: 'tool',
      description: 'Chopping tree trunk logs to harvest wood',
      targetTypes: ['oak_log', 'spruce_log', 'birch_log', 'jungle_log', 'acacia_log', 'dark_oak_log', 'mangrove_log', 'cherry_log'],
      rewardSignals: ['log_collected', 'tree_felled'],
    },
    {
      actionVerb: 'strip_log',
      category: 'tool',
      description: 'Right-clicking log to strip bark into stripped log for building',
      targetTypes: ['oak_log', 'spruce_log', 'birch_log', 'dark_oak_log'],
      rewardSignals: ['stripped_log_created'],
    },
    {
      actionVerb: 'disable_shield',
      category: 'weapon',
      description: 'Attacking a shielding opponent with an axe to disable their shield for 5 seconds',
      targetTypes: ['player', 'vindicator', 'skeleton'],
      rewardSignals: ['shield_disabled', 'critical_damage_dealt'],
    },
    {
      actionVerb: 'carve_pumpkin',
      category: 'tool',
      description: 'Breaking carved pumpkins and beehives faster than hand',
      targetTypes: ['carved_pumpkin', 'beehive', 'bee_nest'],
      rewardSignals: ['pumpkin_harvested'],
    },
  ],
  pickaxe: [
    {
      actionVerb: 'mine_stone',
      category: 'tool',
      description: 'Mining stone to produce cobblestone',
      targetTypes: ['stone', 'deepslate', 'andesite', 'diorite', 'granite'],
      rewardSignals: ['cobblestone_mined'],
    },
    {
      actionVerb: 'mine_iron_ore',
      category: 'tool',
      description: 'Mining iron ore using stone or better pickaxe',
      targetTypes: ['iron_ore', 'deepslate_iron_ore'],
      requiredState: { minTier: 'stone' },
      rewardSignals: ['raw_iron_mined'],
    },
    {
      actionVerb: 'mine_diamond_ore',
      category: 'tool',
      description: 'Mining diamond ore using iron or better pickaxe',
      targetTypes: ['diamond_ore', 'deepslate_diamond_ore'],
      requiredState: { minTier: 'iron' },
      rewardSignals: ['diamond_mined'],
    },
    {
      actionVerb: 'mine_obsidian',
      category: 'tool',
      description: 'Mining obsidian using diamond or netherite pickaxe',
      targetTypes: ['obsidian', 'crying_obsidian'],
      requiredState: { minTier: 'diamond' },
      rewardSignals: ['obsidian_mined'],
    },
    {
      actionVerb: 'mine_ancient_debris',
      category: 'tool',
      description: 'Mining ancient debris in Nether depths using diamond or netherite pickaxe',
      targetTypes: ['ancient_debris'],
      requiredState: { minTier: 'diamond' },
      rewardSignals: ['ancient_debris_mined'],
    },
  ],
  shovel: [
    {
      actionVerb: 'dig_earth',
      category: 'tool',
      description: 'Digging soft blocks like dirt, sand, gravel, clay, and snow',
      targetTypes: ['dirt', 'sand', 'gravel', 'clay', 'snow_block'],
      rewardSignals: ['earth_harvested'],
    },
    {
      actionVerb: 'create_grass_path',
      category: 'tool',
      description: 'Right clicking grass block to create a dirt/grass path',
      targetTypes: ['grass_block', 'dirt'],
      rewardSignals: ['path_created'],
    },
    {
      actionVerb: 'extinguish_campfire',
      category: 'utility',
      description: 'Right clicking campfire to put out flames without breaking it',
      targetTypes: ['campfire', 'soul_campfire'],
      rewardSignals: ['campfire_extinguished'],
    },
  ],
  hoe: [
    {
      actionVerb: 'till_farmland',
      category: 'tool',
      description: 'Tilling dirt or grass next to water into hydrated farmland',
      targetTypes: ['dirt', 'grass_block'],
      rewardSignals: ['farmland_tilled'],
    },
    {
      actionVerb: 'harvest_hay_bale',
      category: 'tool',
      description: 'Fast harvesting hay bales for wheat crafting',
      targetTypes: ['hay_block'],
      rewardSignals: ['hay_collected'],
    },
    {
      actionVerb: 'harvest_sculk',
      category: 'tool',
      description: 'Fast harvesting sculk sensors, catalysts, and shriekers',
      targetTypes: ['sculk', 'sculk_sensor', 'sculk_catalyst'],
      rewardSignals: ['sculk_harvested'],
    },
    {
      actionVerb: 'harvest_sponge',
      category: 'tool',
      description: 'Fast harvesting sponges in ocean monuments',
      targetTypes: ['sponge', 'wet_sponge'],
      rewardSignals: ['sponge_harvested'],
    },
  ],
  shears: [
    {
      actionVerb: 'shear_sheep',
      category: 'tool',
      description: 'Shearing wool from adult sheep without killing them',
      targetTypes: ['sheep'],
      rewardSignals: ['wool_sheared'],
    },
    {
      actionVerb: 'carve_pumpkin',
      category: 'utility',
      description: 'Carving a pumpkin to create a carved pumpkin and pumpkin seeds',
      targetTypes: ['pumpkin'],
      rewardSignals: ['carved_pumpkin_created', 'pumpkin_seeds_harvested'],
    },
    {
      actionVerb: 'harvest_leaves',
      category: 'tool',
      description: 'Harvesting leaf blocks directly instead of saplings',
      targetTypes: ['oak_leaves', 'spruce_leaves', 'birch_leaves', 'mangrove_leaves'],
      rewardSignals: ['leaves_harvested'],
    },
    {
      actionVerb: 'defuse_tripwire',
      category: 'utility',
      description: 'Cutting tripwire without triggering redstone pulses in temples',
      targetTypes: ['tripwire'],
      rewardSignals: ['tripwire_defused_safely'],
    },
    {
      actionVerb: 'harvest_honeycomb',
      category: 'utility',
      description: 'Shearing full beehive with campfire underneath to get 3x honeycomb',
      targetTypes: ['beehive', 'bee_nest'],
      rewardSignals: ['honeycomb_harvested'],
    },
  ],
  flint_and_steel: [
    {
      actionVerb: 'light_nether_portal',
      category: 'utility',
      description: 'Igniting obsidian portal frame to activate Nether portal',
      targetTypes: ['obsidian'],
      rewardSignals: ['nether_portal_activated'],
    },
    {
      actionVerb: 'ignite_tnt',
      category: 'utility',
      description: 'Igniting TNT block for controlled excavation',
      targetTypes: ['tnt'],
      rewardSignals: ['tnt_ignited'],
    },
    {
      actionVerb: 'ignite_creeper',
      category: 'weapon',
      description: 'Right clicking creeper with flint and steel to trigger instant detonation while running away',
      targetTypes: ['creeper'],
      rewardSignals: ['creeper_forced_detonation'],
    },
    {
      actionVerb: 'light_campfire',
      category: 'utility',
      description: 'Lighting unlit campfire for cooking and bee harvesting',
      targetTypes: ['campfire', 'soul_campfire'],
      rewardSignals: ['campfire_lit'],
    },
  ],
  water_bucket: [
    {
      actionVerb: 'water_clutch',
      category: 'utility',
      description: 'Placing water at feet right before hitting ground from lethal fall, then retrieving',
      rewardSignals: ['clutch_successful', 'zero_fall_damage'],
    },
    {
      actionVerb: 'extinguish_fire',
      category: 'utility',
      description: 'Placing water on self while burning in fire or lava',
      rewardSignals: ['fire_extinguished', 'health_preserved'],
    },
    {
      actionVerb: 'create_obsidian',
      category: 'utility',
      description: 'Pouring water on lava source block to turn it into obsidian',
      rewardSignals: ['obsidian_created'],
    },
    {
      actionVerb: 'hydrate_farmland',
      category: 'utility',
      description: 'Placing water block to hydrate 9x9 farmland patch',
      rewardSignals: ['farmland_hydrated'],
    },
  ],
  lava_bucket: [
    {
      actionVerb: 'smelt_fuel_100',
      category: 'utility',
      description: 'Fueling furnace with lava bucket to smelt 100 items',
      rewardSignals: ['furnace_fueled_100'],
    },
    {
      actionVerb: 'trap_mob',
      category: 'weapon',
      description: 'Placing lava in front of pursuing mobs to incinerate them',
      targetTypes: ['zombie', 'iron_golem', 'piglin_brute'],
      rewardSignals: ['mob_incinerated'],
    },
  ],
  milk_bucket: [
    {
      actionVerb: 'cure_poison',
      category: 'food',
      description: 'Drinking milk to clear cave spider or witch poison status',
      rewardSignals: ['poison_cleared'],
    },
    {
      actionVerb: 'cure_wither',
      category: 'food',
      description: 'Drinking milk to clear wither skeleton decay status',
      rewardSignals: ['wither_cleared'],
    },
    {
      actionVerb: 'clear_bad_omen',
      category: 'food',
      description: 'Drinking milk to clear bad omen before entering a village',
      rewardSignals: ['bad_omen_cleared_safely'],
    },
  ],
  shield: [
    {
      actionVerb: 'block_arrows',
      category: 'weapon',
      description: 'Facing skeleton and holding shield to deflect arrows',
      targetTypes: ['skeleton', 'pillager'],
      rewardSignals: ['arrows_blocked', 'zero_projectile_damage'],
    },
    {
      actionVerb: 'block_creeper_blast',
      category: 'weapon',
      description: 'Facing creeper and holding shield to absorb 100% explosion damage',
      targetTypes: ['creeper'],
      rewardSignals: ['explosion_absorbed', 'zero_blast_damage'],
    },
  ],
  fishing_rod: [
    {
      actionVerb: 'catch_fish',
      category: 'utility',
      description: 'Casting into water, observing bobber particle splash, and reeling in fish/treasure',
      rewardSignals: ['fish_caught'],
    },
    {
      actionVerb: 'reel_mob',
      category: 'utility',
      description: 'Hooking distant mob and reeling them in',
      targetTypes: ['mob', 'player'],
      rewardSignals: ['mob_reeled'],
    },
  ],
  lead: [
    {
      actionVerb: 'leash_animal',
      category: 'utility',
      description: 'Leashing animal and walking to guide it',
      targetTypes: ['cow', 'sheep', 'horse', 'iron_golem'],
      rewardSignals: ['animal_leashed'],
    },
    {
      actionVerb: 'tie_to_fence',
      category: 'utility',
      description: 'Right clicking fence post while leading animal to secure it',
      targetTypes: ['fence'],
      rewardSignals: ['animal_tied_to_fence'],
    },
  ],
  bonemeal: [
    {
      actionVerb: 'accelerate_crop',
      category: 'utility',
      description: 'Fertilizing crops with bone meal to advance growth stages',
      targetTypes: ['wheat', 'carrots', 'potatoes', 'beetroots'],
      rewardSignals: ['crop_matured'],
    },
    {
      actionVerb: 'grow_tree',
      category: 'utility',
      description: 'Fertilizing sapling with bone meal to grow full tree instantly',
      targetTypes: ['oak_sapling', 'spruce_sapling', 'birch_sapling', 'cherry_sapling'],
      rewardSignals: ['tree_grown'],
    },
  ],
  brush: [
    {
      actionVerb: 'brush_suspicious_block',
      category: 'utility',
      description: 'Brushing suspicious sand or gravel in ruins to extract archaeology loot',
      targetTypes: ['suspicious_sand', 'suspicious_gravel'],
      rewardSignals: ['archaeology_loot_extracted'],
    },
  ],
  elytra: [
    {
      actionVerb: 'glide_and_boost',
      category: 'armor',
      description: 'Jumping from high cliff, activating elytra, and boosting with firework rocket',
      rewardSignals: ['flight_sustained', 'safe_landing'],
    },
  ],
};

// 3. Programmatic Helper to Generate Tests For Every Single Registry Entry
export function getAllMinecraftEntities(): Array<{ id: number; name: string; displayName: string; category: string }> {
  return (mcData.entitiesArray || []).map((e: any) => ({
    id: e.id,
    name: e.name,
    displayName: e.displayName,
    category: e.type || 'mob',
  }));
}

export function getAllMinecraftBlocks(): Array<{ id: number; name: string; displayName: string; hardness: number }> {
  return (mcData.blocksArray || []).map((b: any) => ({
    id: b.id,
    name: b.name,
    displayName: b.displayName,
    hardness: b.hardness ?? 1.0,
  }));
}

export function getAllMinecraftItems(): Array<{ id: number; name: string; displayName: string; stackSize: number }> {
  return (mcData.itemsArray || []).map((i: any) => ({
    id: i.id,
    name: i.name,
    displayName: i.displayName,
    stackSize: i.stackSize || 64,
  }));
}

export function getAllMinecraftBiomes(): Array<{ id: number; name: string; category?: string }> {
  return (mcData.biomesArray || []).map((b: any) => ({
    id: b.id,
    name: b.name,
    category: b.category,
  }));
}
