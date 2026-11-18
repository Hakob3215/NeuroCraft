"""
MINEbench Gold Dataset Synthesizer
==================================
Generates a balanced, high-advantage gold trajectory dataset for all 16 MINEbench tasks
to prevent imitation-learning mode collapse and seed the System 2 SLM with clean,
canonical tool-calling capabilities across all gameplay domains.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "minebench" / "results" / "grpo_verified_trajectories.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

GOLD_SPEC = [
    # Domain 1: Agility & Parkour
    {
        "taskId": "agility_sky_parkour",
        "domain": "AGILITY_PARKOUR",
        "prompts": [
            "Complete the parkour course and reach the gold_block at the end without falling!",
            "Leap across the floating platforms to touch the gold_block at the finish line!",
            "Sprint-jump across the elevated parkour pillars to reach the gold_block goal!",
            "Parkour across the sky platforms to the gold block!",
        ],
        "tool": "navigate_parkour_or_bridge",
        "args": {"goalBlock": "gold_block", "allowBridge": False},
    },
    {
        "taskId": "agility_ravine_bridge",
        "domain": "AGILITY_PARKOUR",
        "prompts": [
            "Bridge across the ravine chasm using cobblestone to touch the emerald_block on the other side!",
            "Cross the deep ravine by bridging with blocks and reach the emerald_block safely!",
            "Bridge over the ravine gap using cobblestone to reach the emerald block!",
            "Place blocks across the chasm to reach the emerald_block!",
        ],
        "tool": "navigate_parkour_or_bridge",
        "args": {"goalBlock": "emerald_block", "allowBridge": True},
    },

    # Domain 2: Architecture & 3D Building
    {
        "taskId": "build_enclosed_shelter",
        "domain": "ARCHITECTURE_BUILDING",
        "prompts": [
            "Build a simple house right here with walls, a roof, windows, a doorway, and interior torch lighting.",
            "Construct an enclosed survival cabin with solid walls, a roof overhead, a front door, and a torch inside.",
            "Erect a sealed shelter before nightfall with walls, a roof, windows, and interior utilities.",
            "Build a survival cabin nearby with a roof and walls.",
        ],
        "tool": "construct_modular_structure",
        "args": {"structureType": "cabin", "mode": "shelter", "style": "oak", "width": 5, "length": 5, "height": 4},
    },
    {
        "taskId": "build_creative_monument",
        "domain": "ARCHITECTURE_BUILDING",
        "prompts": [
            "Construct a creative smiley face monument right here in creative mode without worrying about roof seals!",
            "Sculpt an artistic face monument on this plot in creative mode (weather seal checks bypassed)!",
            "Build a creative pixel art display with blocks on this platform!",
            "Make a creative smiley monument using blocks!",
        ],
        "tool": "construct_modular_structure",
        "args": {"structureType": "smiley_face", "mode": "creative", "style": "oak", "width": 5, "length": 5, "height": 5},
    },
    {
        "taskId": "build_watchtower_pillar",
        "domain": "ARCHITECTURE_BUILDING",
        "prompts": [
            "Pillar straight up 5 blocks high using cobblestone!",
            "Tower up 5 blocks into the air using your cobblestone!",
            "Pillar up 5 blocks using cobblestone right here!",
            "Erect a 5-block vertical cobblestone pillar beneath you!",
        ],
        "tool": "pillar_up",
        "args": {"height": 5, "material": "cobblestone"},
    },

    # Domain 3: Tactical Combat
    {
        "taskId": "combat_creeper_kiting",
        "domain": "TACTICAL_COMBAT",
        "prompts": [
            "Kill the 2 creepers without letting them blow up!",
            "Defeat the creepers without taking explosion damage by kiting and using your shield!",
            "Engage and slay the creepers safely; raise your shield during fuses!",
            "Fight and kill both creepers!",
        ],
        "tool": "engage_combat",
        "args": {"targetType": "creeper", "maxRange": 20, "count": 2},
    },
    {
        "taskId": "combat_skeleton_duel",
        "domain": "TACTICAL_COMBAT",
        "prompts": [
            "Kill the nearest skeleton!",
            "Defeat the armed skeleton using timed melee strikes and your shield to block arrows!",
            "Duel and slay the skeleton while deflecting incoming arrows with your shield!",
            "Fight the skeleton and eliminate it!",
        ],
        "tool": "engage_combat",
        "args": {"targetType": "skeleton", "maxRange": 25, "count": 1},
    },
    {
        "taskId": "combat_spider_swarm",
        "domain": "TACTICAL_COMBAT",
        "prompts": [
            "can you take care of the 5 spiders",
            "Eliminate the swarm of 5 spiders attacking you!",
            "Defeat all 5 aggressive spiders using your sword!",
            "Clear the area of all 5 spiders!",
        ],
        "tool": "engage_combat",
        "args": {"targetType": "spider", "maxRange": 25, "count": 5},
    },
    {
        "taskId": "combat_enderman_hunt",
        "domain": "TACTICAL_COMBAT",
        "prompts": [
            "can you fight this enderman?",
            "Hunt and defeat the enderman to obtain ender pearls!",
            "Engage the teleporting enderman with critical melee hits!",
            "Slay the enderman!",
        ],
        "tool": "engage_combat",
        "args": {"targetType": "enderman", "maxRange": 30, "count": 1},
    },

    # Domain 4: Homesteading & Economy
    {
        "taskId": "homestead_crop_farm",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "Till the dirt next to the water, plant wheat seeds, and use bone meal to grow crops!",
            "Farm the land: till soil near water, plant seeds, and fertilize with bone meal!",
            "Till a farmland plot, plant the wheat seeds, and apply bone meal!",
            "Establish a crop farm right here!",
        ],
        "tool": "till_and_farm",
        "args": {"cropType": "wheat_seeds", "areaRadius": 4, "useBoneMeal": True},
    },
    {
        "taskId": "homestead_wolf_taming",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "can you tame that wolf",
            "Tame the nearby wild wolf using bones from your inventory!",
            "Befriend the wolf by feeding it bones until hearts appear!",
            "Tame that wolf with bones!",
        ],
        "tool": "tame_animal",
        "args": {"animalType": "wolf", "itemName": "bone"},
    },

    # Domain 5: Puzzle & Engineering
    {
        "taskId": "puzzle_redstone_lever",
        "domain": "PUZZLE_ENGINEERING",
        "prompts": [
            "Flip the lever to unlock the puzzle room and step onto the gold_block!",
            "Interact with the lever to activate redstone and step onto the gold_block!",
            "Toggle the lever switch to open the gate and walk to the gold_block!",
            "Flip that lever and touch the gold block!",
        ],
        "tool": "interact_block_or_puzzle",
        "args": {"targetBlock": "lever", "action": "use"},
    },
    {
        "taskId": "puzzle_tool_preservation_mining",
        "domain": "PUZZLE_ENGINEERING",
        "prompts": [
            "Mine 4 sand and 4 stone without wasting your low-durability iron_pickaxe!",
            "Preserve your iron pickaxe: mine sand with shovel and stone with stone pickaxe!",
            "Harvest 4 sand and 4 stone using the correct durable tools!",
            "Mine sand and stone while preserving iron pickaxe durability!",
        ],
        "tool": "gather_resource",
        "args": {"blockType": "sand", "count": 4, "maxRadius": 10},
    },

    # Domain 6: Progression & Boss DAG
    {
        "taskId": "prog_tier1_stone_age",
        "domain": "PROGRESSION_DRAGON",
        "prompts": [
            "Chop oak logs, craft a wooden_pickaxe, mine stone, and craft a stone_pickaxe!",
            "Progress to the Stone Age: gather logs, craft wooden pickaxe, and mine cobblestone!",
            "Chop wood, make a crafting table and wooden pickaxe, then mine stone!",
            "Begin progression by harvesting oak logs and crafting tools!",
        ],
        "tool": "gather_resource",
        "args": {"blockType": "oak_log", "count": 4, "maxRadius": 15},
    },
    {
        "taskId": "prog_tier3_iron_armor",
        "domain": "PROGRESSION_DRAGON",
        "prompts": [
            "Get full iron armor and equip it!",
            "Mine iron ore, smelt ingots in furnace, and craft full iron armor!",
            "Smelt raw iron into ingots to craft iron armor pieces!",
            "Smelt iron and craft iron armor!",
        ],
        "tool": "smelt_item",
        "args": {"inputItem": "raw_iron", "count": 8},
    },
    {
        "taskId": "prog_tier6_ender_dragon",
        "domain": "PROGRESSION_DRAGON",
        "prompts": [
            "Destroy the end_crystal towers with your bow, eat golden_apple when hurt, and slay the ender_dragon when it perches!",
            "Shoot down the End Crystals on obsidian pillars with your bow and fight the Ender Dragon!",
            "Snipe the End Crystals with bow and arrow, then attack the dragon!",
            "Free the End: destroy End Crystals and slay the dragon!",
        ],
        "tool": "shoot_bow",
        "args": {"targetType": "end_crystal", "maxRange": 50},
    },

    # Domain 7: Nutrition & Consumption (eat_food)
    {
        "taskId": "omni_food_consume_cooked_beef",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "You are hungry with low saturation. Consume Cooked Beef to restore hunger points.",
            "Eat the cooked_beef.",
            "Consume cooked_beef to heal up.",
            "Eat food to restore hunger!",
        ],
        "tool": "eat_food",
        "args": {"foodName": "cooked_beef"},
    },
    {
        "taskId": "omni_tool_milk_bucket_cure_poison",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "You are poisoned by a cave spider! Drink the milk bucket immediately to cure it!",
            "Drink milk_bucket to clear your poison effect.",
            "Consume milk bucket to cure status effects!",
        ],
        "tool": "eat_food",
        "args": {"foodName": "milk_bucket"},
    },

    # Domain 8: Equipment & Armor (equip_armor_or_item)
    {
        "taskId": "omni_equip_iron_chestplate",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "Equip the Iron Chestplate into your chest armor slot for protection.",
            "Put on the iron_chestplate.",
            "Equip your iron chestplate for defense.",
        ],
        "tool": "equip_armor_or_item",
        "args": {"itemName": "iron_chestplate"},
    },
    {
        "taskId": "omni_equip_shield",
        "domain": "TACTICAL_COMBAT",
        "prompts": [
            "Equip your shield and block incoming skeleton arrows.",
            "Put your shield into your offhand slot.",
            "Equip your shield for arrow defense!",
        ],
        "tool": "equip_armor_or_item",
        "args": {"itemName": "shield"},
    },

    # Domain 9: Companion Cooperation (give_item_to_player)
    {
        "taskId": "coop_give_bread",
        "domain": "MULTIPLAYER_ARENA",
        "prompts": [
            "Give some bread to Trainer.",
            "Share bread from your inventory with the player.",
            "Drop me 3 bread please!",
        ],
        "tool": "give_item_to_player",
        "args": {"playerName": "Trainer", "itemName": "bread"},
    },
    {
        "taskId": "coop_give_torches",
        "domain": "MULTIPLAYER_ARENA",
        "prompts": [
            "Give torches to Trainer.",
            "Drop me torches from your inventory.",
            "Share a torch with Trainer.",
        ],
        "tool": "give_item_to_player",
        "args": {"playerName": "Trainer", "itemName": "torch"},
    },

    # Domain 10: Tool Affordances & Block/Entity Interactions
    {
        "taskId": "omni_tool_shears_shear_sheep",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "Shear the white wool off the sheep without hurting it.",
            "Use your shears to collect wool from the sheep.",
            "Shear the nearby sheep!",
        ],
        "tool": "shear_sheep",
        "args": {"target": "sheep"},
    },
    {
        "taskId": "omni_tool_axe_strip_log",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "Strip the bark off the oak log right in front of you with your axe.",
            "Use your iron axe to strip the oak_log.",
            "Strip bark from the oak log!",
        ],
        "tool": "use_tool_on_block",
        "args": {"tool": "iron_axe", "targetBlock": "oak_log"},
    },
    {
        "taskId": "omni_tool_shovel_create_grass_path",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "Use your shovel to turn the grass block into a dirt path.",
            "Flatten the grass_block into a path using iron_shovel.",
            "Make a grass path right here with your shovel.",
        ],
        "tool": "use_tool_on_block",
        "args": {"tool": "iron_shovel", "targetBlock": "grass_block"},
    },
    {
        "taskId": "omni_tool_shears_carve_pumpkin",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "Carve the pumpkin using shears to create a carved pumpkin and seeds.",
            "Use shears on the pumpkin block.",
        ],
        "tool": "use_tool_on_block",
        "args": {"tool": "shears", "targetBlock": "pumpkin"},
    },
    {
        "taskId": "omni_tool_flint_and_steel_ignite_tnt",
        "domain": "PUZZLE_ENGINEERING",
        "prompts": [
            "Ignite the TNT with your flint and steel.",
            "Use flint_and_steel on the tnt block!",
        ],
        "tool": "use_tool_on_block",
        "args": {"tool": "flint_and_steel", "targetBlock": "tnt"},
    },
    {
        "taskId": "omni_tool_water_bucket_extinguish_fire",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "Extinguish the fire using your water bucket.",
            "Use your water_bucket on the fire block!",
        ],
        "tool": "use_tool_on_block",
        "args": {"tool": "water_bucket", "targetBlock": "fire"},
    },
    {
        "taskId": "omni_tool_bone_meal_accelerate_crop",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "Use bone meal on the wheat crop in front of you to accelerate its growth.",
            "Apply bone_meal to the wheat crop.",
        ],
        "tool": "use_tool_on_block",
        "args": {"tool": "bone_meal", "targetBlock": "wheat"},
    },
    {
        "taskId": "omni_tool_lead_leash_animal",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "Attach a lead to the nearby animal and tie the pig to the oak fence.",
            "Use your lead on the cow or pig.",
        ],
        "tool": "use_tool_on_block",
        "args": {"tool": "lead", "targetBlock": "cow"},
    },
    {
        "taskId": "omni_tool_flint_and_steel_light_nether_portal",
        "domain": "PUZZLE_ENGINEERING",
        "prompts": [
            "Ignite the obsidian portal frame with your flint and steel.",
            "Light the obsidian Nether portal using flint_and_steel.",
        ],
        "tool": "light_block_fire",
        "args": {"targetBlock": "obsidian"},
    },
    {
        "taskId": "omni_tool_flint_and_steel_light_campfire",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "Light the campfire using flint and steel.",
            "Ignite the unlit campfire with flint_and_steel.",
        ],
        "tool": "light_block_fire",
        "args": {"targetBlock": "campfire"},
    },
    {
        "taskId": "omni_structure_desert_pyramid",
        "domain": "TACTICAL_COMBAT",
        "prompts": [
            "Explore the desert pyramid, overcome its hazards (tnt_pressure_plate_trap), and retrieve key loot (diamond, enchanted_golden_apple, saddle)!",
            "Loot the desert_pyramid chest safely.",
            "Open the chest and loot its items!",
        ],
        "tool": "loot_chest",
        "args": {"chestName": "chest"},
    },
    {
        "taskId": "omni_mob_cow",
        "domain": "HOMESTEADING_ECONOMY",
        "prompts": [
            "Feed wheat to the two cows to breed a baby calf.",
            "Breed the nearby cows using wheat.",
        ],
        "tool": "breed_animals",
        "args": {"animalType": "cow"},
    },
]

MULTI_STEP_GOLD_SPEC = [
    {
        "taskId": "prog_tier1_stone_age",
        "domain": "PROGRESSION_DRAGON",
        "activeGoal": "Chop oak logs, craft a wooden_pickaxe, mine stone, and craft a stone_pickaxe!",
        "steps": [
            {
                "tool": "gather_resource",
                "args": {"blockType": "oak_log", "count": 3},
                "summary": "Harvested 3 oak_log into inventory",
                "thought": "Step 1: Chop 3 oak_log to bootstrap wood prerequisites.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "oak_planks", "count": 12},
                "summary": "Successfully crafted 12x oak_planks",
                "thought": "Step 2: Craft oak_planks from oak_log before sticks and crafting_table.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "crafting_table", "count": 1},
                "summary": "Successfully crafted 1x crafting_table",
                "thought": "Step 3: Craft a crafting_table for 3x3 tool recipes.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "stick", "count": 4},
                "summary": "Successfully crafted 4x stick",
                "thought": "Step 4: Craft sticks from oak_planks for pickaxe handles.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "wooden_pickaxe", "count": 1},
                "summary": "Successfully crafted 1x wooden_pickaxe",
                "thought": "Step 5: Craft wooden_pickaxe using planks and sticks at crafting_table.",
                "goalCompleted": False,
            },
            {
                "tool": "gather_resource",
                "args": {"blockType": "stone", "count": 3},
                "summary": "Harvested 3 stone as cobblestone",
                "thought": "Step 6: Mine 3 stone using wooden_pickaxe to collect cobblestone.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "stone_pickaxe", "count": 1},
                "summary": "Successfully crafted 1x stone_pickaxe",
                "thought": "Step 7: Craft stone_pickaxe from cobblestone and sticks to complete Stone Age.",
                "goalCompleted": True,
            },
        ],
    },
    {
        "taskId": "omni_craft_chest",
        "domain": "HOMESTEADING_ECONOMY",
        "activeGoal": "Harvest wood, craft planks and a crafting table, and craft 1x Chest.",
        "steps": [
            {
                "tool": "gather_resource",
                "args": {"blockType": "oak_log", "count": 3},
                "summary": "Harvested 3 oak_log into inventory",
                "thought": "Step 1: Gather oak_log so we can craft planks and a crafting_table.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "oak_planks", "count": 12},
                "summary": "Successfully crafted 12x oak_planks",
                "thought": "Step 2: Craft oak_planks from oak_log.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "crafting_table", "count": 1},
                "summary": "Successfully crafted 1x crafting_table",
                "thought": "Step 3: Craft crafting_table for 3x3 chest synthesis.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "chest", "count": 1},
                "summary": "Successfully crafted 1x chest",
                "thought": "Step 4: Craft 1x chest using remaining oak_planks on the crafting_table.",
                "goalCompleted": True,
            },
        ],
    },
    {
        "taskId": "prog_tier3_iron_armor",
        "domain": "PROGRESSION_DRAGON",
        "activeGoal": "Get full iron armor and equip it!",
        "steps": [
            {
                "tool": "gather_resource",
                "args": {"blockType": "oak_log", "count": 3},
                "summary": "Harvested 3 oak_log for crafting table and fuel",
                "thought": "Step 1: Gather wood for crafting table and sticks.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "oak_planks", "count": 12},
                "summary": "Successfully crafted 12x oak_planks",
                "thought": "Step 2: Craft oak_planks from oak_log.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "crafting_table", "count": 1},
                "summary": "Successfully crafted 1x crafting_table",
                "thought": "Step 3: Craft a crafting_table.",
                "goalCompleted": False,
            },
            {
                "tool": "gather_resource",
                "args": {"blockType": "stone", "count": 8},
                "summary": "Harvested 8 cobblestone",
                "thought": "Step 4: Gather 8 cobblestone for a furnace.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "furnace", "count": 1},
                "summary": "Successfully crafted 1x furnace",
                "thought": "Step 5: Craft a furnace from 8 cobblestone.",
                "goalCompleted": False,
            },
            {
                "tool": "gather_resource",
                "args": {"blockType": "coal_ore", "count": 4},
                "summary": "Harvested 4 coal",
                "thought": "Step 6: Mine coal for smelting fuel.",
                "goalCompleted": False,
            },
            {
                "tool": "gather_resource",
                "args": {"blockType": "iron_ore", "count": 24},
                "summary": "Harvested 24 raw_iron",
                "thought": "Step 7: Mine iron_ore with stone_pickaxe.",
                "goalCompleted": False,
            },
            {
                "tool": "smelt_item",
                "args": {"inputItem": "raw_iron", "count": 24},
                "summary": "Smelted 24x iron_ingot",
                "thought": "Step 8: Smelt raw_iron into iron_ingot in furnace.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "iron_helmet", "count": 1},
                "summary": "Successfully crafted 1x iron_helmet",
                "thought": "Step 9: Craft iron_helmet from 5 iron_ingots.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "iron_chestplate", "count": 1},
                "summary": "Successfully crafted 1x iron_chestplate",
                "thought": "Step 10: Craft iron_chestplate from 8 iron_ingots.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "iron_leggings", "count": 1},
                "summary": "Successfully crafted 1x iron_leggings",
                "thought": "Step 11: Craft iron_leggings from 7 iron_ingots.",
                "goalCompleted": False,
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "iron_boots", "count": 1},
                "summary": "Successfully crafted 1x iron_boots",
                "thought": "Step 12: Craft iron_boots from 4 iron_ingots.",
                "goalCompleted": False,
            },
            {
                "tool": "equip_armor_or_item",
                "args": {"itemName": "all_armor"},
                "summary": "Equipped: helmet, chestplate, leggings, boots!",
                "thought": "Step 13: Equip the full iron armor set.",
                "goalCompleted": True,
            },
        ],
    },
    {
        "taskId": "prog_tier6_ender_dragon",
        "domain": "PROGRESSION_DRAGON",
        "activeGoal": "Destroy the end_crystal towers with your bow, eat golden_apple when hurt, and slay the ender_dragon when it perches!",
        "steps": [
            {
                "tool": "shoot_bow",
                "args": {"targetType": "end_crystal", "maxRange": 50},
                "summary": "Destroyed end_crystal atop obsidian pillar",
                "thought": "Step 1: Snipe the healing end crystal with bow and arrow.",
                "goalCompleted": False,
            },
            {
                "tool": "shoot_bow",
                "args": {"targetType": "end_crystal", "maxRange": 50},
                "summary": "Destroyed second end_crystal",
                "thought": "Step 2: Destroy remaining end crystals to stop dragon regeneration.",
                "goalCompleted": False,
            },
            {
                "tool": "engage_combat",
                "args": {"targetMob": "ender_dragon", "maxRange": 25},
                "summary": "Engaged and attacked ender_dragon",
                "thought": "Step 3: Attack the Ender Dragon as it perches upon the bedrock fountain.",
                "goalCompleted": True,
            },
        ],
    },
    {
        "taskId": "puzzle_tool_preservation_mining",
        "domain": "PUZZLE_ENGINEERING",
        "activeGoal": "Mine 4 sand and 4 stone without wasting your low-durability iron_pickaxe!",
        "steps": [
            {
                "tool": "gather_resource",
                "args": {"blockType": "sand", "count": 4},
                "summary": "Gathered 4 sand using shovel",
                "thought": "Step 1: Mine sand with shovel to preserve pickaxe durability.",
                "goalCompleted": False,
            },
            {
                "tool": "gather_resource",
                "args": {"blockType": "stone", "count": 4},
                "summary": "Mined 4 stone using stone pickaxe",
                "thought": "Step 2: Mine stone using stone pickaxe without wasting low-durability iron pickaxe.",
                "goalCompleted": True,
            },
        ],
    },
    {
        "taskId": "homestead_crop_farm",
        "domain": "HOMESTEAD_FARMING",
        "activeGoal": "Till the dirt next to the water, plant wheat seeds, and use bone meal to grow crops!",
        "steps": [
            {
                "tool": "till_and_farm",
                "args": {"plots": 4},
                "summary": "Tilled 4 farmland plots and planted wheat seeds",
                "thought": "Step 1: Till soil next to water and plant wheat seeds.",
                "goalCompleted": False,
            },
            {
                "tool": "till_and_farm",
                "args": {"plots": 4},
                "summary": "Applied bone meal and harvested mature wheat",
                "thought": "Step 2: Accelerate growth with bone meal and harvest wheat.",
                "goalCompleted": True,
            },
        ],
    },
    {
        "taskId": "prog_tier2_iron_pickaxe",
        "domain": "PROGRESSION_DRAGON",
        "activeGoal": "Mine iron_ore with stone_pickaxe, smelt raw_iron in a furnace, and craft an iron_pickaxe!",
        "steps": [
            {
                "tool": "gather_resource",
                "args": {"blockType": "iron_ore", "count": 4},
                "summary": "Harvested 4x raw_iron using stone_pickaxe",
                "thought": "Step 1: Mine 4 iron_ore using stone_pickaxe.",
                "goalCompleted": False,
                "state_hint": {
                    "held": "stone_pickaxe",
                    "inv": ["1x stone_pickaxe", "4x coal", "4x stick"],
                    "workstations": ["crafting_table", "furnace"],
                    "blocks": ["iron_ore@(2,58,1)"],
                    "clusters": ["iron_ore(4x,2.2m north)"],
                },
            },
            {
                "tool": "smelt_item",
                "args": {"inputItem": "raw_iron", "count": 4},
                "summary": "Smelted 4x iron_ingot in furnace",
                "thought": "Step 2: Smelt raw_iron into iron_ingot using coal in the furnace.",
                "goalCompleted": False,
                "state_hint": {
                    "held": "raw_iron",
                    "inv": ["1x stone_pickaxe", "4x raw_iron", "4x coal", "4x stick"],
                    "workstations": ["crafting_table", "furnace"],
                    "smeltable": "raw_iron",
                },
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "iron_pickaxe", "count": 1},
                "summary": "Successfully crafted 1x iron_pickaxe",
                "thought": "Step 3: Craft iron_pickaxe at the crafting_table using 3 iron_ingot and 2 stick.",
                "goalCompleted": True,
                "state_hint": {
                    "held": "iron_ingot",
                    "inv": ["1x stone_pickaxe", "4x iron_ingot", "1x coal", "4x stick"],
                    "workstations": ["crafting_table", "furnace"],
                    "craftable": ["iron_pickaxe", "iron_sword"],
                },
            },
        ],
    },
    {
        "taskId": "prog_tier4_diamond_and_obsidian",
        "domain": "PROGRESSION_DRAGON",
        "activeGoal": "Mine diamond_ore, craft a diamond_pickaxe, mine obsidian, and ignite the Nether portal!",
        "steps": [
            {
                "tool": "gather_resource",
                "args": {"blockType": "diamond_ore", "count": 3},
                "summary": "Harvested 3x diamond using iron_pickaxe",
                "thought": "Step 1: Mine 3 diamond_ore at deep level using iron_pickaxe.",
                "goalCompleted": False,
                "state_hint": {
                    "pos": "(10,-54,6)",
                    "held": "iron_pickaxe",
                    "inv": ["1x iron_pickaxe", "4x stick", "1x flint_and_steel"],
                    "workstations": ["crafting_table"],
                    "blocks": ["diamond_ore@(12,-54,6)"],
                    "clusters": ["diamond_ore(3x,2.0m east)"],
                },
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "diamond_pickaxe", "count": 1},
                "summary": "Successfully crafted 1x diamond_pickaxe",
                "thought": "Step 2: Craft a diamond_pickaxe from 3 diamonds and 2 sticks so we can mine obsidian.",
                "goalCompleted": False,
                "state_hint": {
                    "pos": "(10,-54,6)",
                    "held": "iron_pickaxe",
                    "inv": ["1x iron_pickaxe", "3x diamond", "4x stick", "1x flint_and_steel"],
                    "workstations": ["crafting_table"],
                    "craftable": ["diamond_pickaxe"],
                },
            },
            {
                "tool": "gather_resource",
                "args": {"blockType": "obsidian", "count": 10},
                "summary": "Harvested 10x obsidian using diamond_pickaxe",
                "thought": "Step 3: Mine 10 obsidian blocks with the diamond_pickaxe.",
                "goalCompleted": False,
                "state_hint": {
                    "pos": "(14,-54,8)",
                    "held": "diamond_pickaxe",
                    "inv": ["1x diamond_pickaxe", "1x flint_and_steel"],
                    "blocks": ["obsidian@(16,-54,8)"],
                    "clusters": ["obsidian(10x,2.0m east)"],
                },
            },
            {
                "tool": "light_block_fire",
                "args": {"targetBlock": "obsidian"},
                "summary": "Ignited obsidian Nether portal with flint_and_steel",
                "thought": "Step 4: Ignite the obsidian portal frame using flint_and_steel.",
                "goalCompleted": True,
                "state_hint": {
                    "pos": "(18,-54,8)",
                    "held": "flint_and_steel",
                    "inv": ["1x diamond_pickaxe", "1x flint_and_steel", "10x obsidian"],
                    "blocks": ["obsidian@(19,-54,8)"],
                },
            },
        ],
    },
    {
        "taskId": "prog_tier5_nether_and_stronghold",
        "domain": "PROGRESSION_DRAGON",
        "activeGoal": "Slay blazes and endermen, craft Eyes of Ender, and triangulate the Stronghold!",
        "steps": [
            {
                "tool": "engage_combat",
                "args": {"targetType": "blaze", "maxRange": 32, "count": 4},
                "summary": "Slayed 4 blazes and collected blaze_rod drops",
                "thought": "Step 1: Fight nearby blazes in the Nether Fortress to collect blaze rods.",
                "goalCompleted": False,
                "state_hint": {
                    "held": "iron_sword",
                    "offhand": "shield",
                    "armor": "iron_chestplate, iron_leggings",
                    "inv": ["1x iron_sword", "8x cooked_beef", "6x ender_pearl"],
                    "entities": ["blaze(7.5m)", "blaze(11.0m)"],
                },
            },
            {
                "tool": "engage_combat",
                "args": {"targetType": "enderman", "maxRange": 32, "count": 4},
                "summary": "Slayed endermen and gathered ender_pearls",
                "thought": "Step 2: Hunt endermen to gather remaining ender_pearls.",
                "goalCompleted": False,
                "state_hint": {
                    "held": "iron_sword",
                    "offhand": "shield",
                    "armor": "iron_chestplate, iron_leggings",
                    "inv": ["1x iron_sword", "6x blaze_rod", "6x ender_pearl", "8x cooked_beef"],
                    "entities": ["enderman(6.2m)"],
                },
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "blaze_powder", "count": 12},
                "summary": "Crafted 12x blaze_powder from blaze_rod",
                "thought": "Step 3: Craft blaze_powder from blaze_rod.",
                "goalCompleted": False,
                "state_hint": {
                    "held": "blaze_rod",
                    "inv": ["1x iron_sword", "6x blaze_rod", "12x ender_pearl"],
                    "craftable": ["blaze_powder"],
                },
            },
            {
                "tool": "craft_item",
                "args": {"itemName": "ender_eye", "count": 12},
                "summary": "Crafted 12x ender_eye",
                "thought": "Step 4: Combine blaze_powder and ender_pearl into 12x ender_eye.",
                "goalCompleted": False,
                "state_hint": {
                    "held": "blaze_powder",
                    "inv": ["1x iron_sword", "12x blaze_powder", "12x ender_pearl"],
                    "craftable": ["ender_eye"],
                },
            },
            {
                "tool": "triangulate_stronghold_coordinates",
                "args": {"autoSample": True},
                "summary": "Triangulated Stronghold coordinates using Eyes of Ender",
                "thought": "Step 5: Throw Eyes of Ender and triangulate the Stronghold portal location.",
                "goalCompleted": True,
                "state_hint": {
                    "held": "ender_eye",
                    "inv": ["1x diamond_pickaxe", "1x iron_sword", "12x ender_eye"],
                    "workstations": ["none"],
                },
            },
        ],
    },
]


def build_grounded_gold_system_prompt(spec_or_step: dict, history: list | None = None) -> str:
    """
    Constructs a realistic state-grounded system prompt matching
    UnifiedLLMClient.buildCompactLocalSystemPrompt so QLoRA trains on
    world state + affordances rather than blind task memorization.
    """
    hint = dict(spec_or_step.get("state_hint") or {})
    tool = str(spec_or_step.get("tool") or "")
    args = dict(spec_or_step.get("args") or {})

    pos = str(hint.get("pos", "(0,64,0)"))
    hp = int(hint.get("hp", 14 if tool == "eat_food" else 20))
    food = int(hint.get("food", 10 if tool == "eat_food" else 20))

    held = hint.get("held")
    inv = list(hint.get("inv") or [])
    workstations = list(hint.get("workstations") or [])
    craftable = list(hint.get("craftable") or [])
    smeltable = str(hint.get("smeltable") or "")
    clusters = list(hint.get("clusters") or [])
    entities = list(hint.get("entities") or [])
    blocks = list(hint.get("blocks") or [])
    offhand = str(hint.get("offhand") or "shield")
    armor = str(hint.get("armor") or "none")

    if not inv and not hint:
        if tool == "craft_item":
            item = str(args.get("itemName", "oak_planks"))
            craftable = [item]
            if "planks" in item:
                inv = ["4x oak_log"]
            elif "crafting_table" in item:
                inv = ["8x oak_planks"]
            elif "stick" in item:
                inv = ["4x oak_planks"]
            elif "wooden_pickaxe" in item:
                inv = ["6x oak_planks", "4x stick"]
                workstations = ["crafting_table"]
            elif "stone_pickaxe" in item:
                inv = ["4x cobblestone", "4x stick"]
                workstations = ["crafting_table"]
            elif "iron_" in item:
                inv = ["8x iron_ingot", "4x stick"]
                workstations = ["crafting_table"]
            elif "diamond_" in item:
                inv = ["4x diamond", "4x stick"]
                workstations = ["crafting_table"]
            elif "ender_eye" in item:
                inv = ["12x blaze_powder", "12x ender_pearl"]
            elif "blaze_powder" in item:
                inv = ["6x blaze_rod"]
            else:
                inv = ["8x oak_planks", "8x cobblestone", "4x stick"]
                workstations = ["crafting_table"]
        elif tool == "smelt_item":
            inp = str(args.get("inputItem", "raw_iron"))
            inv = [f"8x {inp}", "8x coal"]
            workstations = ["furnace"]
            smeltable = inp
            held = held or inp
        elif tool == "gather_resource":
            blk = str(args.get("blockType", "oak_log"))
            blocks = [f"{blk}@(3,64,2)"]
            clusters = [f"{blk}(4x,3.2m north)"]
            if "diamond" in blk or "obsidian" in blk:
                best_pick = "diamond_pickaxe" if "obsidian" in blk else "iron_pickaxe"
                inv = [f"1x {best_pick}"]
                held = held or best_pick
            elif "iron" in blk or "stone" in blk:
                pick = "wooden_pickaxe" if blk == "stone" else "stone_pickaxe"
                inv = [f"1x {pick}"]
                held = held or pick
            elif "sand" in blk:
                inv = ["1x stone_shovel", "1x stone_pickaxe", "1x iron_pickaxe"]
                held = held or "stone_shovel"
        elif tool == "engage_combat":
            target = str(args.get("targetType", "zombie"))
            entities = [f"{target}(4.5m)"]
            inv = ["1x iron_sword", "1x shield", "8x cooked_beef"]
            held = held or "iron_sword"
        elif tool == "shoot_bow":
            target = str(args.get("targetType", "end_crystal"))
            entities = [f"{target}(18.0m)"]
            inv = ["1x bow", "32x arrow", "2x golden_apple"]
            held = held or "bow"
        elif tool == "eat_food":
            food_name = str(args.get("foodName", "cooked_beef"))
            inv = [f"4x {food_name}"]
            held = held or food_name
        elif tool == "equip_armor_or_item":
            eq = str(args.get("itemName", "iron_chestplate"))
            inv = [f"1x {eq}"]
            held = held or eq
        elif tool == "give_item_to_player":
            player = str(args.get("playerName", "Trainer"))
            item = str(args.get("itemName", "bread"))
            entities = [f"{player}(player)(3.2m)"]
            inv = [f"4x {item}"]
            held = held or item
        elif tool == "shear_sheep":
            entities = ["sheep(2.5m)"]
            inv = ["1x shears"]
            held = held or "shears"
        elif tool == "breed_animals":
            animal = str(args.get("animalType", "cow"))
            entities = [f"{animal}(2.5m)", f"{animal}(3.4m)"]
            inv = ["8x wheat"]
            held = held or "wheat"
        elif tool == "tame_animal":
            animal = str(args.get("animalType", "wolf"))
            item = str(args.get("itemName", "bone"))
            entities = [f"{animal}(3.0m)"]
            inv = [f"6x {item}"]
            held = held or item
        elif tool == "use_tool_on_block":
            t_item = str(args.get("tool", "iron_axe"))
            blk = str(args.get("targetBlock", "oak_log"))
            blocks = [f"{blk}@(2,64,1)"]
            inv = [f"1x {t_item}"]
            held = held or t_item
        elif tool == "light_block_fire":
            blk = str(args.get("targetBlock", "obsidian"))
            blocks = [f"{blk}@(2,64,1)"]
            inv = ["1x flint_and_steel"]
            held = held or "flint_and_steel"
        elif tool == "till_and_farm":
            inv = ["1x iron_hoe", "8x wheat_seeds", "8x bone_meal"]
            blocks = ["dirt@(2,63,0)", "water@(3,63,0)"]
            held = held or "iron_hoe"
        elif tool == "loot_chest":
            blocks = ["chest@(2,64,1)"]
        elif tool == "interact_block_or_puzzle":
            blk = str(args.get("targetBlock", "lever"))
            blocks = [f"{blk}@(2,65,1)", "gold_block@(5,64,1)"]
        elif tool == "navigate_parkour_or_bridge":
            goal_blk = str(args.get("goalBlock", "gold_block"))
            blocks = [f"{goal_blk}@(14,64,0)"]
            if args.get("allowBridge"):
                inv = ["32x cobblestone"]
                held = held or "cobblestone"
        elif tool == "pillar_up":
            mat = str(args.get("material", "cobblestone"))
            inv = [f"16x {mat}"]
            held = held or mat
        elif tool == "construct_modular_structure":
            inv = ["64x oak_planks", "8x glass", "4x torch"]
            held = held or "oak_planks"
        elif tool == "triangulate_stronghold_coordinates":
            inv = ["12x ender_eye", "1x diamond_pickaxe"]
            held = held or "ender_eye"

    held_str = str(held or "none")
    recent_str = (
        ", ".join(f"{h['tool']}(OK)" for h in history[-3:])
        if history
        else "none"
    )
    return (
        "You are NeuroAgent, an autonomous Minecraft AI companion.\n"
        f"State: Pos{pos} HP:{hp}/20 Food:{food}/20 Held:{held_str}\n"
        f"Gear & Durability: Main:{held_str} Off:{offhand} Armor:[{armor}]\n"
        f"Inventory: [{', '.join(inv) if inv else 'empty'}]\n"
        f"Workstations Nearby: [{', '.join(workstations) if workstations else 'none'}] | "
        f"Immediately Craftable: [{', '.join(craftable) if craftable else 'none'}] | "
        f"Smeltable: [{smeltable if smeltable else 'none'}]\n"
        f"Resource Clusters: [{', '.join(clusters) if clusters else 'none'}]\n"
        f"Nearby Entities (360°): [{', '.join(entities) if entities else 'none'}]\n"
        f"Nearby Blocks: [{', '.join(blocks) if blocks else 'none'}]\n"
        f"Recent Actions: [{recent_str}]\n"
        "Respond ONLY with valid JSON."
    )


def generate_gold_dataset():
    total_samples = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # 1. Single-Turn Gold Trajectories across all 16 Canonical Tasks + Omniscience Affordances
        for spec in GOLD_SPEC:
            grounded_sys = build_grounded_gold_system_prompt(spec)
            for prompt in spec["prompts"]:
                # Standard format
                sample = {
                    "provider": "gold_curriculum",
                    "taskId": spec["taskId"],
                    "domain": spec["domain"],
                    "advantage": 1.5,
                    "rawReward": 100.0,
                    "messages": [
                        {
                            "role": "system",
                            "content": grounded_sys,
                        },
                        {"role": "user", "content": prompt},
                        {
                            "role": "assistant",
                            "content": json.dumps({
                                "thought": f"Executing {spec['tool']} for {spec['taskId']}",
                                "toolCall": {
                                    "name": spec["tool"],
                                    "arguments": spec["args"],
                                },
                            }),
                        },
                    ],
                }
                f.write(json.dumps(sample) + "\n")
                total_samples += 1

                # Formatted with Player "Trainer" says framing
                framed_sample = {
                    "provider": "gold_curriculum",
                    "taskId": spec["taskId"],
                    "domain": spec["domain"],
                    "advantage": 1.5,
                    "rawReward": 100.0,
                    "messages": [
                        {
                            "role": "system",
                            "content": grounded_sys,
                        },
                        {
                            "role": "user",
                            "content": f'Player "Trainer" says: "{prompt}"\nReturn ONLY a valid JSON object: {{"thought":"...", "toolCall":{{"name":"...", "arguments":{{}}}}}}',
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps({
                                "thought": f"Executing {spec['tool']} for {spec['taskId']}",
                                "toolCall": {
                                    "name": spec["tool"],
                                    "arguments": spec["args"],
                                },
                            }),
                        },
                    ],
                }
                f.write(json.dumps(framed_sample) + "\n")
                total_samples += 1

        # 2. Multi-Step Goal Continuation Trajectories (including state-grounded Step 0 + Steps 1..N)
        for chain in MULTI_STEP_GOLD_SPEC:
            history: list = []
            for step_idx, step in enumerate(chain["steps"]):
                grounded_sys = build_grounded_gold_system_prompt(step, history)
                if step_idx == 0:
                    user_prompt = (
                        f'Player "Trainer" says: "{chain["activeGoal"]}"\n'
                        'Return ONLY a valid JSON object: {"thought":"...", "toolCall":{"name":"...", "arguments":{}}}'
                    )
                else:
                    history_text = "\n".join(
                        f"Step {i + 1}: {h['tool']}({json.dumps(h['args'])}) -> SUCCESS: {h['summary']}"
                        for i, h in enumerate(history)
                    )
                    user_prompt = (
                        f'Active Goal: "{chain["activeGoal"]}"\n'
                        f'Recent Steps:\n{history_text}\n'
                        'Return ONLY a valid JSON object: {"thought":"...", "toolCall":{"name":"...", "arguments":{}}}'
                    )
                sample = {
                    "provider": "gold_continuation",
                    "taskId": chain["taskId"],
                    "domain": chain["domain"],
                    "advantage": 1.8,
                    "rawReward": 100.0,
                    "step": step_idx,
                    "messages": [
                        {
                            "role": "system",
                            "content": grounded_sys,
                        },
                        {"role": "user", "content": user_prompt},
                        {
                            "role": "assistant",
                            "content": json.dumps({
                                "thought": step["thought"],
                                "toolCall": {
                                    "name": step["tool"],
                                    "arguments": step["args"],
                                },
                                "goalCompleted": step.get("goalCompleted", False),
                            }),
                        },
                    ],
                }
                f.write(json.dumps(sample) + "\n")
                total_samples += 1

                history.append(step)

    print(f"[Gold Synthesizer] Generated {total_samples} state-grounded gold trajectories (including multi-step continuations) to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_gold_dataset()
