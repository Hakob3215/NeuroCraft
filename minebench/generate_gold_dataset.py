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
]


def generate_gold_dataset():
    total_samples = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # 1. Single-Turn Gold Trajectories across all 16 Canonical Tasks
        for spec in GOLD_SPEC:
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
                            "content": "You are NeuroAgent, an autonomous Minecraft 1.20.4 AI companion.",
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
                            "content": "You are NeuroAgent, an autonomous Minecraft 1.20.4 AI companion. Respond ONLY with valid JSON.",
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

        # 2. Multi-Step Goal Continuation Trajectories
        for chain in MULTI_STEP_GOLD_SPEC:
            history: list = []
            for step_idx, step in enumerate(chain["steps"]):
                if step_idx > 0:
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
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are NeuroAgent, an autonomous Minecraft 1.20.4 AI companion. Respond ONLY with valid JSON.",
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

    print(f"[Gold Synthesizer] Generated {total_samples} balanced gold trajectories (including multi-step continuations) to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_gold_dataset()
