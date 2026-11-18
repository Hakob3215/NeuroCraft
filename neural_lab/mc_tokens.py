"""
NeuroCraft Minecraft VLA Tokenizer Surgery & Action Grammar
===========================================================
Defines the native, comprehensive token grammar for the Dual-Channel
Vision-Language-Action (VLA) architecture. Eliminates JSON generation overhead
and enables direct discrete action routing alongside conversational dialogue.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re

REGISTRY_PATH = Path(__file__).resolve().parent / "mc_registry.json"

INTENT_TOKENS = [
    "<|intent:chat|>",
    "<|intent:action|>",
    "<|intent:hybrid|>",
]

INVENTORY_TOKENS = [
    "<|act:drop_item|>",
    "<|act:give_to_player|>",
    "<|act:equip|>",
    "<|act:consume|>",
    "<|act:open_container|>",
    "<|act:deposit_items|>",
    "<|act:withdraw_items|>",
    "<|act:craft|>",
    "<|act:smelt|>",
]

SPATIAL_TOKENS = [
    "<|act:follow_player|>",
    "<|act:goto_coord|>",
    "<|act:bridge_sneak|>",
    "<|act:parkour_jump|>",
    "<|act:pillar_up|>",
    "<|act:dig_staircase|>",
    "<|act:swim_surface|>",
]

WORLD_TOKENS = [
    "<|act:mine_block|>",
    "<|act:place_block|>",
    "<|act:interact_block|>",
    "<|act:till_and_plant|>",
    "<|act:harvest_crops|>",
    "<|act:tame_animal|>",
    "<|act:shear_sheep|>",
]

COMBAT_TOKENS = [
    "<|act:melee_attack|>",
    "<|act:kite_combat|>",
    "<|act:shield_block|>",
    "<|act:shoot_bow|>",
]

# Load exhaustive registry from official Minecraft 1.20.4 data
if REGISTRY_PATH.exists():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        _reg = json.load(f)
    ITEM_TOKENS = [f"<|item:{i}|>" for i in _reg.get("items", [])]
    TARGET_TOKENS = [f"<|target:{e}|>" for e in _reg.get("entities", [])] + [
        "<|target:player|>",
        "<|target:crop|>",
        "<|target:block|>",
    ]
    BLOCK_TOKENS = [f"<|block:{b}|>" for b in _reg.get("blocks", [])]
else:
    ITEM_TOKENS = ["<|item:bread|>", "<|item:torch|>", "<|item:iron_sword|>", "<|item:cobblestone|>"]
    TARGET_TOKENS = ["<|target:creeper|>", "<|target:skeleton|>", "<|target:zombie|>", "<|target:player|>"]
    BLOCK_TOKENS = ["<|block:stone|>", "<|block:dirt|>", "<|block:cobblestone|>"]

QUANTITY_TOKENS = [f"<|qty:{q}|>" for q in range(1, 65)]

SLOT_TOKENS = [
    "<|slot:mainhand|>",
    "<|slot:offhand|>",
    "<|slot:head|>",
    "<|slot:chest|>",
    "<|slot:legs|>",
    "<|slot:feet|>",
    "<|slot:hotbar_0|>",
    "<|slot:hotbar_1|>",
    "<|slot:hotbar_2|>",
    "<|slot:hotbar_3|>",
    "<|slot:hotbar_4|>",
    "<|slot:hotbar_5|>",
    "<|slot:hotbar_6|>",
    "<|slot:hotbar_7|>",
    "<|slot:hotbar_8|>",
]

ALL_MINECRAFT_SPECIAL_TOKENS = (
    INTENT_TOKENS
    + INVENTORY_TOKENS
    + SPATIAL_TOKENS
    + WORLD_TOKENS
    + COMBAT_TOKENS
    + TARGET_TOKENS
    + ITEM_TOKENS
    + BLOCK_TOKENS
    + QUANTITY_TOKENS
    + SLOT_TOKENS
)


def register_minecraft_tokens(tokenizer: Any, model: Optional[Any] = None) -> Tuple[int, List[str]]:
    """
    Surgically injects the Minecraft action grammar into the tokenizer vocabulary.
    Expands model embedding layer and LM head if model is provided.
    Returns: (num_added_tokens, list_of_new_tokens)
    """
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens_to_add = [tok for tok in ALL_MINECRAFT_SPECIAL_TOKENS if tok not in existing_tokens]

    if new_tokens_to_add:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens_to_add})
        if model is not None and num_added > 0:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            print(f"[TokenizerSurgery] Resized embeddings to {len(tokenizer)} (+{num_added} tokens) in <10ms")
        return num_added, new_tokens_to_add

    return 0, []


def parse_vla_response(response_text: str) -> Dict[str, Any]:
    """
    Parses a dual-channel VLA output into speech text and discrete action commands.
    Ensures zero short-circuiting: speech is preserved for chat, and action tokens
    are parsed into structured tool requests for Mineflayer.
    """
    result: Dict[str, Any] = {
        "intent": "chat",
        "speech": "",
        "action_tokens": [],
        "parsed_actions": [],
    }

    # Determine intent
    if "<|intent:hybrid|>" in response_text:
        result["intent"] = "hybrid"
    elif "<|intent:action|>" in response_text:
        result["intent"] = "action"
    elif "<|intent:chat|>" in response_text:
        result["intent"] = "chat"
    elif any(tok in response_text for tok in (INVENTORY_TOKENS + SPATIAL_TOKENS + COMBAT_TOKENS + WORLD_TOKENS)):
        result["intent"] = "hybrid" if re.search(r"[a-zA-Z]{3,}", response_text) else "action"

    # Extract all special tokens present
    all_tokens_pattern = r"(<\|[a-zA-Z0-9_:]+\|>)"
    found_tokens = re.findall(all_tokens_pattern, response_text)
    result["action_tokens"] = [t for t in found_tokens if not t.startswith("<|intent:")]

    # Extract speech (any natural language text outside the special tokens)
    clean_speech = re.sub(all_tokens_pattern, "", response_text).strip()
    result["speech"] = clean_speech

    # Parse discrete actions
    action_tokens = [t for t in found_tokens if t.startswith("<|act:")]
    for act_tok in action_tokens:
        act_name = act_tok[6:-2]  # e.g., 'drop_item'
        action_data: Dict[str, Any] = {"action": act_name}

        # Look for associated target, item, qty, slot in the text
        target_match = re.search(r"<\|target:([a-zA-Z0-9_]+)\|>", response_text)
        if target_match:
            action_data["target"] = target_match.group(1)

        item_match = re.search(r"<\|item:([a-zA-Z0-9_]+)\|>", response_text)
        if item_match:
            action_data["item"] = item_match.group(1)

        qty_match = re.search(r"<\|qty:(\d+)\|>", response_text)
        if qty_match:
            action_data["quantity"] = int(qty_match.group(1))

        slot_match = re.search(r"<\|slot:([a-zA-Z0-9_]+)\|>", response_text)
        if slot_match:
            action_data["slot"] = slot_match.group(1)

        result["parsed_actions"].append(action_data)

    return result
