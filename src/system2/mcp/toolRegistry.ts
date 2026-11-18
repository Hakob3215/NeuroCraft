import { EventEmitter } from 'events';
import { MCPToolHandler, MCPToolSchema, RegisteredMCPTool } from '../../types';
import { computeTokenSimilarity, tokenizeText } from '../memory/memoryStore';

/**
 * Dynamic Hot-Reloading MCP Tool Registry.
 * Holds both built-in primitive tools and runtime agent-synthesized tools.
 * Emits `notifications/tools/list_changed` whenever a new tool is added or updated.
 */
export class DynamicMCPToolRegistry extends EventEmitter {
  private tools = new Map<string, RegisteredMCPTool>();

  /**
   * Canonical alias mappings so models and callers can use concise verbs
   * without failing when calling underlying compound MCP skill names.
   */
  public static readonly TOOL_ALIASES: Record<string, string> = {
    // Equipment & armor
    equip: 'equip_armor_or_item',
    equip_armor: 'equip_armor_or_item',
    equip_item: 'equip_armor_or_item',
    wear: 'equip_armor_or_item',
    glide_and_boost: 'equip_armor_or_item',

    // Crafting & smelting
    craft: 'craft_item',
    craft_recipe: 'craft_item',
    make: 'craft_item',
    make_item: 'craft_item',
    smelt: 'smelt_item',
    smelt_or_cook: 'smelt_item',
    cook: 'smelt_item',

    // Nutrition & consumables & milk/potions
    eat: 'eat_food',
    consume: 'eat_food',
    consume_food: 'eat_food',
    consume_item: 'eat_food',
    drink: 'eat_food',
    drink_milk: 'eat_food',
    drink_potion: 'eat_food',
    drink_honey: 'eat_food',
    clear_bad_omen: 'eat_food',
    cure_poison: 'eat_food',
    cure_wither: 'eat_food',

    // Mining & harvesting resources
    mine: 'gather_resource',
    gather: 'gather_resource',
    harvest: 'gather_resource',
    chop_wood: 'gather_resource',
    mine_wood: 'gather_resource',
    mine_stone: 'gather_resource',
    mine_iron_ore: 'gather_resource',
    mine_diamond_ore: 'gather_resource',
    mine_obsidian: 'gather_resource',
    mine_ancient_debris: 'gather_resource',
    mine_ore: 'gather_resource',
    mine_log: 'gather_resource',
    chop_log: 'gather_resource',
    dig: 'gather_resource',
    dig_earth: 'gather_resource',
    harvest_hay_bale: 'gather_resource',
    harvest_crop: 'gather_resource',
    harvest_sculk: 'gather_resource',
    harvest_sponge: 'gather_resource',

    // Combat & weapons & shields
    attack: 'engage_combat',
    attack_mob: 'engage_combat',
    combat: 'engage_combat',
    melee_attack: 'engage_combat',
    fight: 'engage_combat',
    kill_creeper: 'engage_combat',
    kill_skeleton: 'engage_combat',
    kill_spider: 'engage_combat',
    kill_enderman: 'engage_combat',
    kill_mob: 'engage_combat',
    slay_creeper: 'engage_combat',
    slay_dragon: 'engage_combat',
    shield_block: 'engage_combat',
    block_arrows: 'engage_combat',
    block_creeper_blast: 'engage_combat',
    disable_shield: 'engage_combat',
    interact_with_or_defeat_entity: 'engage_combat',
    interact_with_or_defeat: 'engage_combat',
    defeat_entity: 'engage_combat',
    defeat: 'engage_combat',
    spar: 'engage_combat',
    duel: 'engage_combat',
    pvp: 'engage_combat',
    shoot_bow: 'shoot_bow',
    shoot_crystal: 'shoot_bow',
    snipe_crystal: 'shoot_bow',
    destroy_crystal: 'shoot_bow',

    // Agility & navigation
    parkour: 'navigate_parkour_or_bridge',
    bridge: 'navigate_parkour_or_bridge',
    parkour_jump: 'navigate_parkour_or_bridge',
    bridge_sneak: 'navigate_parkour_or_bridge',
    bridge_chasm: 'navigate_parkour_or_bridge',
    parkour_course: 'navigate_parkour_or_bridge',
    navigate_to: 'navigate_to_coordinates',
    walk_to: 'navigate_to_coordinates',
    go_to: 'navigate_to_coordinates',
    pillar: 'pillar_up',
    tower_up: 'pillar_up',

    // Farming & crops
    farm: 'till_and_farm',
    farm_crop: 'till_and_farm',
    farm_crops: 'till_and_farm',
    till: 'till_and_farm',
    till_soil: 'till_and_farm',
    till_and_plant: 'till_and_farm',
    till_farmland: 'till_and_farm',

    // Extra aliases for robust dispatch
    swipe_or_hunt: 'engage_combat',
    chop_oak_logs: 'gather_resource',
    chop_oak_log: 'gather_resource',
    breed_cows: 'breed_animals',
    loot_structure: 'loot_chest',
    open_chest: 'loot_chest',
    explore_structure: 'loot_chest',

    // Block & item interactions
    use_shovel: 'interact_block_or_puzzle',
    create_grass_path: 'interact_block_or_puzzle',
    create_path: 'interact_block_or_puzzle',
    make_path: 'interact_block_or_puzzle',
    path: 'interact_block_or_puzzle',
    shovel: 'interact_block_or_puzzle',
    carve: 'interact_block_or_puzzle',
    carve_pumpkin: 'interact_block_or_puzzle',
    shear: 'interact_block_or_puzzle',
    shear_sheep: 'interact_block_or_puzzle',
    shears: 'interact_block_or_puzzle',
    use_shears: 'interact_block_or_puzzle',
    strip_log: 'interact_block_or_puzzle',
    leash_animal: 'interact_block_or_puzzle',
    tie_to_fence: 'interact_block_or_puzzle',
    lead: 'interact_block_or_puzzle',
    leash: 'interact_block_or_puzzle',
    create_obsidian: 'interact_block_or_puzzle',
    extinguish_campfire: 'interact_block_or_puzzle',
    extinguish_fire: 'interact_block_or_puzzle',
    hydrate_farmland: 'interact_block_or_puzzle',
    water_clutch: 'use_tool_on_block',
    use_tool_on_entity: 'interact_block_or_puzzle',
    catch_fish: 'interact_block_or_puzzle',
    reel_mob: 'interact_block_or_puzzle',
    brush_suspicious_block: 'interact_block_or_puzzle',
    accelerate_crop: 'interact_block_or_puzzle',
    grow_tree: 'interact_block_or_puzzle',

    // Building
    construct_simple_house: 'build_simple_house',
    build_cabin: 'build_simple_house',
    build_shelter: 'build_simple_house',

    // Cooperative & item sharing
    give_item: 'give_item_to_player',
    give: 'give_item_to_player',
    drop_item: 'give_item_to_player',
    drop: 'give_item_to_player',
    toss_item: 'give_item_to_player',
    toss: 'give_item_to_player',
    share_item: 'give_item_to_player',
    deliver_item: 'give_item_to_player',
  };

  /**
   * Registers or hot-reloads an MCP tool and broadcasts `notifications/tools/list_changed`.
   */
  public registerTool(schema: MCPToolSchema, handler: MCPToolHandler): void {
    const keywords = tokenizeText(
      `${schema.name} ${schema.description} ${(schema.tags || []).join(' ')}`
    );

    this.tools.set(schema.name, {
      schema,
      handler,
      embeddingKeywords: keywords,
    });

    this.emit('notifications/tools/list_changed', {
      toolName: schema.name,
      isSynthesized: Boolean(schema.isSynthesized),
      totalTools: this.tools.size,
    });
  }

  public getTool(name: string): RegisteredMCPTool | undefined {
    if (!name) return undefined;
    if (this.tools.has(name)) {
      return this.tools.get(name);
    }
    const lower = name.toLowerCase().trim();
    if (this.tools.has(lower)) {
      return this.tools.get(lower);
    }
    // Normalize camelCase / PascalCase to snake_case (e.g. gatherResource -> gather_resource)
    const snake = name
      .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
      .toLowerCase()
      .trim();
    if (this.tools.has(snake)) {
      return this.tools.get(snake);
    }
    const resolved =
      DynamicMCPToolRegistry.TOOL_ALIASES[snake] ||
      DynamicMCPToolRegistry.TOOL_ALIASES[lower];
    if (resolved && this.tools.has(resolved)) {
      return this.tools.get(resolved);
    }

    // Prefix morphology & semantic fallback for SLM variations
    if (snake.startsWith('farm_') || snake.includes('crop')) return this.tools.get('till_and_farm');
    if (snake.startsWith('tie_') || snake.includes('leash')) return this.tools.get('interact_block_or_puzzle');
    if (snake.startsWith('loot_') || snake.startsWith('raid_')) return this.tools.get('loot_chest');
    if (snake.startsWith('mine_') || snake.startsWith('chop_') || snake.startsWith('harvest_') || snake.startsWith('dig_')) return this.tools.get('gather_resource');
    if (snake.startsWith('craft_') || snake.startsWith('make_')) return this.tools.get('craft_item');
    if (snake.startsWith('smelt_') || snake.startsWith('cook_')) return this.tools.get('smelt_item');
    if (snake.startsWith('equip_') || snake.startsWith('wear_')) return this.tools.get('equip_armor_or_item');
    if (snake.startsWith('kill_') || snake.startsWith('attack_') || snake.startsWith('fight_') || snake.startsWith('hunt_') || snake.startsWith('slay_')) return this.tools.get('engage_combat');
    if (snake.startsWith('parkour_') || snake.startsWith('bridge_')) return this.tools.get('navigate_parkour_or_bridge');

    const matches = this.searchMatchingTools(snake, 1);
    if (matches && matches.length > 0 && matches[0].similarity >= 0.25) {
      return this.tools.get(matches[0].schema.name);
    }

    return undefined;
  }

  public listSchemas(): MCPToolSchema[] {
    return Array.from(this.tools.values()).map((t) => t.schema);
  }

  public listSynthesizedSchemas(): MCPToolSchema[] {
    return this.listSchemas().filter((s) => s.isSynthesized);
  }

  /**
   * Semantic search over registered MCP tools to check if an existing tool
   * satisfies the user's natural language request.
   */
  public searchMatchingTools(
    naturalLanguageAsk: string,
    topK = 5
  ): Array<{ schema: MCPToolSchema; similarity: number }> {
    const queryTokens = tokenizeText(naturalLanguageAsk);
    const results: Array<{ schema: MCPToolSchema; similarity: number }> = [];

    for (const tool of this.tools.values()) {
      const sim = computeTokenSimilarity(queryTokens, tool.embeddingKeywords);
      results.push({ schema: tool.schema, similarity: sim });
    }

    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK);
  }
}
