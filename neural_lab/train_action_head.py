"""
NeuroCraft Scaled-Up Multi-Modal Action Head Trainer (RTX 5060 GPU)
==================================================================
Trains the ~211-Million-parameter Multi-Modal Temporal Action Chunking Transformer
(MinecraftActionHead) on multi-tick continuous trajectories, semantic 3D voxels,
16-entity radar states, extended motor controls, and discrete tool dispatch signals.

Outputs:
  neural_lab/checkpoints/action_head_latest.pt
  neural_lab/results/action_head_metrics.json
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from neural_lab.action_head import (
    MinecraftActionHead,
    CANONICAL_MOTOR_TOOLS,
    PROPRIOCEPTION_DIM,
    LATENT_DIM,
    CHUNK_SIZE,
    NUM_ENTITIES,
    ENTITY_FEAT_DIM,
)

CHECKPOINT_DIR = ROOT_DIR / "neural_lab" / "checkpoints"
RESULTS_DIR = ROOT_DIR / "neural_lab" / "results"
TOOL_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_MOTOR_TOOLS)}


def generate_temporal_action_dataset(
    num_samples: int = 8000,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Synthesizes diverse multi-modal behavioral scenarios with 8-tick temporal trajectories,
    32-class semantic 3D voxels, 16-entity radar tensors, extended motor signals, and tool targets.
    """
    torch.manual_seed(1337)

    latent_dim = LATENT_DIM
    proprio_dim = PROPRIOCEPTION_DIM
    chunk_size = CHUNK_SIZE

    # Input Tensors
    latents = torch.randn(num_samples, latent_dim, device=device) * 0.1
    proprios = torch.zeros(num_samples, proprio_dim, device=device)
    voxels = torch.zeros(num_samples, 15, 15, 15, dtype=torch.long, device=device)
    voxels[:, :, 0:4, :] = 1  # Solid stone floor (semantic class 1)
    voxels[:, :, 4, :] = 4    # Surface grass_block (semantic class 4)
    entities = torch.zeros(num_samples, NUM_ENTITIES, ENTITY_FEAT_DIM, device=device)

    # Target Tensors across 8-tick trajectory chunk
    target_aim = torch.zeros(num_samples, chunk_size, 2, device=device)
    target_move = torch.zeros(num_samples, chunk_size, 2, device=device)
    target_discrete = torch.zeros(num_samples, chunk_size, 6, device=device)   # [jump, sprint, sneak, attack, use_item, swap_slot]
    target_extended = torch.zeros(num_samples, chunk_size, 2, device=device)   # [dig_block, drop_item]
    target_tools = torch.zeros(num_samples, dtype=torch.long, device=device)
    target_tool_gate = torch.zeros(num_samples, device=device)
    target_values = torch.zeros(num_samples, device=device)
    actor_weights = torch.ones(num_samples, dtype=torch.float32, device=device)

    for i in range(num_samples):
        mode = i % 10

        proprios[i, 0] = 0.9   # health
        proprios[i, 1] = 0.9   # food
        proprios[i, 7] = 1.0   # onGround
        proprios[i, 14] = 1.0  # sword equipped
        proprios[i, 15] = 1.0  # shield equipped

        if mode == 0:
            # Mode 0: Creeper Tactical Kiting (Backpedal + Shield on Fuse)
            latents[i, 0:32] += 1.5
            proprios[i, 12] = 3.5 / 32.0
            proprios[i, 13] = 1.0
            entities[i, 0] = torch.tensor([0.0, 0.0, 3.5 / 32.0, 3.5 / 32.0, 0.0, 0.0, -0.05, 1.0, 1.0, 0.0, 0.0, 1.0], device=device)
            target_tools[i] = TOOL_TO_IDX["engage_combat"]
            target_tool_gate[i] = 1.0

            for t in range(chunk_size):
                target_move[i, t, 0] = 0.3 * math.sin(t * 0.5)
                target_move[i, t, 1] = -0.9
                target_aim[i, t, 0] = 0.0
                target_aim[i, t, 1] = -5.0
                target_discrete[i, t, 1] = 1.0  # sprint
                target_discrete[i, t, 4] = 1.0  # use_item (shield)
            target_values[i] = 0.85

        elif mode == 1:
            # Mode 1: Critical Strike Attack on Hostile Mob
            latents[i, 32:64] += 1.5
            proprios[i, 12] = 2.4 / 32.0
            entities[i, 0] = torch.tensor([0.0, 0.0, 2.4 / 32.0, 2.4 / 32.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], device=device)
            target_tools[i] = TOOL_TO_IDX["engage_combat"]
            target_tool_gate[i] = 1.0

            for t in range(chunk_size):
                target_move[i, t, 1] = 1.0
                target_aim[i, t, 1] = 12.0
                if t < 2:
                    target_discrete[i, t, 0] = 1.0  # jump
                elif 2 <= t <= 4:
                    target_discrete[i, t, 3] = 1.0  # attack
            target_values[i] = 0.90

        elif mode == 2:
            # Mode 2: Lava Ravine Edge & Parkour Sprint-Jump toward Gold Goal
            latents[i, 64:96] += 1.5
            voxels[i, 6:10, 0:5, 8:13] = 0    # Air chasm ahead
            voxels[i, 6:10, 0:2, 8:13] = 11   # Lava pool below chasm (semantic class 11)
            voxels[i, 7:9, 4, 13:15] = 28     # gold_block landing target (semantic class 28)
            target_tools[i] = TOOL_TO_IDX["navigate_parkour_or_bridge"]
            target_tool_gate[i] = 1.0

            for t in range(chunk_size):
                target_move[i, t, 1] = 1.0
                target_discrete[i, t, 1] = 1.0  # sprint
                if 2 <= t <= 5:
                    target_discrete[i, t, 0] = 1.0  # jump
            target_values[i] = 0.78

        elif mode == 3:
            # Mode 3: Precision Mining / Excavation (stone/iron_ore)
            latents[i, 96:128] += 1.5
            proprios[i, 14] = 0.0
            proprios[i, 15] = 0.0
            proprios[i, 17] = 1.0  # pickaxe
            voxels[i, 7, 4:6, 9] = 13  # iron_ore directly in front (class 13)
            target_tools[i] = TOOL_TO_IDX["gather_resource"]
            target_tool_gate[i] = 1.0

            for t in range(chunk_size):
                target_aim[i, t, 1] = 35.0
                target_discrete[i, t, 3] = 1.0  # attack/mine
                target_extended[i, t, 0] = 1.0  # dig_block
            target_values[i] = 0.65

        elif mode == 4:
            # Mode 4: MLG Water Clutch from High Fall (pure emergency motor reflex)
            latents[i, 128:160] += 2.0
            proprios[i, 5] = -0.8
            proprios[i, 7] = 0.0
            voxels[i, :, 0:4, :] = 0
            voxels[i, :, 0, :] = 1
            target_tools[i] = TOOL_TO_IDX["none"]
            target_tool_gate[i] = 0.0

            for t in range(chunk_size):
                target_aim[i, t, 1] = 88.0
                if t == 0:
                    target_discrete[i, t, 5] = 1.0  # swap_slot
                if t >= 6:
                    target_discrete[i, t, 4] = 1.0  # use_item
            target_values[i] = 0.95

        elif mode == 5:
            # Mode 5: Companion Interaction (Facing player, dropping item)
            latents[i, 160:192] += 1.5
            entities[i, 0] = torch.tensor([0.1, 0.0, 2.0 / 32.0, 2.1 / 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
            target_tools[i] = TOOL_TO_IDX["give_item_to_player"]
            target_tool_gate[i] = 1.0

            for t in range(chunk_size):
                target_aim[i, t, 0] = 12.0 * math.cos(t * 0.4)
                if t == 2:
                    target_discrete[i, t, 5] = 1.0  # swap_slot
                if t >= 4:
                    target_discrete[i, t, 2] = 1.0  # sneak
                    target_extended[i, t, 1] = 1.0  # drop_item
            target_values[i] = 0.72

        elif mode == 6:
            # Mode 6: Archery Drawing & Ranged Aim Arc
            latents[i, 192:224] += 1.5
            proprios[i, 12] = 18.0 / 32.0
            proprios[i, 14] = 0.0
            proprios[i, 15] = 0.0
            proprios[i, 16] = 1.0
            entities[i, 0] = torch.tensor([0.2, 0.3, 16.0 / 32.0, 18.0 / 32.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], device=device)
            target_tools[i] = TOOL_TO_IDX["shoot_bow"]
            target_tool_gate[i] = 1.0

            for t in range(chunk_size):
                target_move[i, t, 0] = 0.1 * math.sin(t)
                target_aim[i, t, 1] = -14.0
                if t <= 5:
                    target_discrete[i, t, 4] = 1.0  # draw bow
            target_values[i] = 0.80

        elif mode == 7:
            # Mode 7: Safe Exploration & Timber Locating
            latents[i, 224:256] += 1.0
            voxels[i, 5, 4:8, 10] = 7  # oak_log nearby (class 7)
            target_tools[i] = TOOL_TO_IDX["none"]
            target_tool_gate[i] = 0.0

            for t in range(chunk_size):
                target_move[i, t, 1] = 0.7
                target_aim[i, t, 0] = 5.0 * math.sin(t * 0.3)
            target_values[i] = 0.50

        elif mode == 8:
            # Mode 8: Crafting Station Dispatch (`craft_item`)
            latents[i, 160:192] += 1.5
            voxels[i, 7, 4, 8] = 17  # crafting_table adjacent (class 17)
            target_tools[i] = TOOL_TO_IDX["craft_item"]
            target_tool_gate[i] = 1.0
            for t in range(chunk_size):
                target_discrete[i, t, 4] = 1.0 if t < 2 else 0.0
            target_values[i] = 0.82

        else:
            # Mode 9: Smelting Furnace Dispatch (`smelt_item`)
            latents[i, 160:192] += 1.5
            voxels[i, 8, 4, 8] = 18  # furnace adjacent (class 18)
            target_tools[i] = TOOL_TO_IDX["smelt_item"]
            target_tool_gate[i] = 1.0
            for t in range(chunk_size):
                target_discrete[i, t, 4] = 1.0 if t < 2 else 0.0
            target_values[i] = 0.80

    return (
        latents, proprios, voxels, entities,
        target_aim, target_move, target_discrete, target_extended,
        target_tools, target_tool_gate, target_values, actor_weights
    )


def _map_action_idx_to_tool_idx(act_idx: int, tool_name: Optional[str] = None) -> Tuple[int, float]:
    if tool_name and tool_name in TOOL_TO_IDX:
        return TOOL_TO_IDX[tool_name], 1.0
    if act_idx == 3:
        return TOOL_TO_IDX["engage_combat"], 1.0
    if act_idx == 2:
        return TOOL_TO_IDX["navigate_parkour_or_bridge"], 1.0
    if act_idx == 4:
        return TOOL_TO_IDX["shoot_bow"], 1.0
    if act_idx == 8:
        return TOOL_TO_IDX["eat_food"], 1.0
    return TOOL_TO_IDX["gather_resource"], 1.0


def read_jsonl_tail(path: Path, max_lines: int = 3000, max_bytes: int = 48 * 1024 * 1024) -> List[str]:
    """Reads up to the last `max_lines` complete lines from a JSONL file in O(1) time/memory."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()  # discard partial first line
        raw = f.read()
    lines = raw.decode("utf-8", errors="ignore").splitlines()
    return lines[-max_lines:]


def load_real_transitions_dataset(
    max_samples: int = 3000,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> Optional[Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]]:
    """Loads recent real Mojang server transitions into multi-modal 8-tick temporal chunks."""
    trans_path = ROOT_DIR / "minebench" / "results" / "real_rl_transitions.jsonl"
    if not trans_path.exists():
        return None

    records = []
    try:
        for line in read_jsonl_tail(trans_path, max_lines=max_samples):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if len(rec.get("voxels", [])) == 3375 and len(rec.get("botState", [])) >= 10:
                    records.append(rec)
            except Exception:
                pass
    except Exception:
        return None

    if not records:
        return None

    n = len(records)
    latents = torch.randn(n, LATENT_DIM, device=device) * 0.1
    proprios = torch.zeros(n, PROPRIOCEPTION_DIM, device=device)
    voxels = torch.zeros(n, 15, 15, 15, dtype=torch.long, device=device)
    entities = torch.zeros(n, NUM_ENTITIES, ENTITY_FEAT_DIM, device=device)

    tgt_aim = torch.zeros(n, CHUNK_SIZE, 2, device=device)
    tgt_move = torch.zeros(n, CHUNK_SIZE, 2, device=device)
    tgt_disc = torch.zeros(n, CHUNK_SIZE, 6, device=device)
    tgt_ext = torch.zeros(n, CHUNK_SIZE, 2, device=device)
    tgt_tools = torch.zeros(n, dtype=torch.long, device=device)
    tgt_gate = torch.zeros(n, device=device)
    tgt_val = torch.zeros(n, device=device)
    actor_weights = torch.ones(n, dtype=torch.float32, device=device)

    for i, r in enumerate(records):
        bs = r["botState"]
        proprios[i, 0] = float(bs[0])
        proprios[i, 1] = float(bs[1])
        proprios[i, 4] = float(bs[2])
        proprios[i, 5] = float(bs[3])
        proprios[i, 6] = float(bs[4])
        proprios[i, 7] = float(bs[8])
        proprios[i, 8] = float(bs[9])

        ents = r.get("entities", [])
        for ei in range(min(NUM_ENTITIES, len(ents))):
            if len(ents[ei]) == ENTITY_FEAT_DIM:
                entities[i, ei] = torch.tensor(ents[ei], dtype=torch.float32, device=device)

        if ents and len(ents[0]) >= 8 and ents[0][7] > 0.5:
            proprios[i, 9] = max(-1.0, min(1.0, float(ents[0][0]) * 2.0))
            proprios[i, 10] = max(-1.0, min(1.0, float(ents[0][1]) * 2.0))
            proprios[i, 11] = max(-1.0, min(1.0, float(ents[0][2]) * 2.0))
            proprios[i, 12] = float(ents[0][3])
            proprios[i, 13] = float(ents[0][11]) if len(ents[0]) > 11 else 0.0

        # Preserve full 32-class semantic voxel IDs
        voxels[i] = torch.tensor(r["voxels"], dtype=torch.long, device=device).view(15, 15, 15).clamp(0, 31)

        aim = r.get("aimDelta", [0.0, 0.0])
        dyaw_deg = max(-180.0, min(180.0, float(aim[0]) * 180.0 / math.pi))
        dpitch_deg = max(-90.0, min(90.0, float(aim[1]) * 180.0 / math.pi))
        act_idx = int(r.get("actionIdx", 1))
        tool_idx, gate_val = _map_action_idx_to_tool_idx(act_idx, r.get("toolName"))
        tgt_tools[i] = tool_idx
        tgt_gate[i] = gate_val

        if act_idx == 3:
            latents[i, 32:64] += 1.5
            tgt_move[i, :, 1] = 1.0
            tgt_disc[i, 2:5, 3] = 1.0
        elif act_idx == 2:
            latents[i, 64:96] += 1.5
            tgt_move[i, :, 1] = 1.0
            tgt_disc[i, :, 1] = 1.0
            tgt_disc[i, 2:5, 0] = 1.0
        elif act_idx == 4:
            latents[i, 192:224] += 1.5
            tgt_disc[i, 0:6, 4] = 1.0
        else:
            latents[i, 96:128] += 1.0
            tgt_move[i, :, 1] = 0.5
            tgt_ext[i, :, 0] = 1.0  # dig_block active during gathering

        tgt_aim[i, :, 0] = dyaw_deg
        tgt_aim[i, :, 1] = dpitch_deg
        rew = float(r.get("reward", 0.0))
        tgt_val[i] = 1.0 / (1.0 + math.exp(-max(-5.0, min(5.0, rew / 25.0))))
        if rew > 0.0:
            actor_weights[i] = min(3.0, 1.2 + rew / 20.0)
        elif rew == 0.0:
            actor_weights[i] = 0.6
        else:
            actor_weights[i] = 0.05

    return (
        latents, proprios, voxels, entities,
        tgt_aim, tgt_move, tgt_disc, tgt_ext,
        tgt_tools, tgt_gate, tgt_val, actor_weights
    )


def load_real_motor_chunks_dataset(
    max_samples: int = 2000,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    chunks_path: Optional[Path] = None,
) -> Optional[Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]]:
    """Loads real System 2 latent_h + 24-dim proprio + 20Hz 8-tick motor chunks."""
    target_path = chunks_path or (ROOT_DIR / "minebench" / "results" / "real_motor_chunks.jsonl")
    if not target_path.exists():
        return None

    records = []
    try:
        for line in read_jsonl_tail(target_path, max_lines=max_samples):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if (
                    isinstance(rec.get("latent_h"), list) and len(rec["latent_h"]) == LATENT_DIM
                    and isinstance(rec.get("proprio24"), list) and len(rec["proprio24"]) == PROPRIOCEPTION_DIM
                    and isinstance(rec.get("voxels"), list) and len(rec["voxels"]) == 3375
                    and isinstance(rec.get("aimTargets"), list) and len(rec["aimTargets"]) == CHUNK_SIZE
                    and isinstance(rec.get("moveTargets"), list) and len(rec["moveTargets"]) == CHUNK_SIZE
                    and isinstance(rec.get("discreteTargets"), list) and len(rec["discreteTargets"]) == CHUNK_SIZE
                    and isinstance(rec.get("extendedTargets"), list) and len(rec["extendedTargets"]) == CHUNK_SIZE
                    and all(isinstance(row, list) and len(row) >= 2 for row in rec["aimTargets"])
                    and all(isinstance(row, list) and len(row) >= 2 for row in rec["moveTargets"])
                    and all(isinstance(row, list) and len(row) >= 6 for row in rec["discreteTargets"])
                    and all(isinstance(row, list) and len(row) >= 2 for row in rec["extendedTargets"])
                ):
                    records.append(rec)
            except Exception:
                pass
    except Exception:
        return None

    if not records:
        return None

    n = len(records)
    latents = torch.zeros(n, LATENT_DIM, dtype=torch.float32, device=device)
    proprios = torch.zeros(n, PROPRIOCEPTION_DIM, dtype=torch.float32, device=device)
    voxels = torch.zeros(n, 15, 15, 15, dtype=torch.long, device=device)
    entities = torch.zeros(n, NUM_ENTITIES, ENTITY_FEAT_DIM, dtype=torch.float32, device=device)

    tgt_aim = torch.zeros(n, CHUNK_SIZE, 2, dtype=torch.float32, device=device)
    tgt_move = torch.zeros(n, CHUNK_SIZE, 2, dtype=torch.float32, device=device)
    tgt_disc = torch.zeros(n, CHUNK_SIZE, 6, dtype=torch.float32, device=device)
    tgt_ext = torch.zeros(n, CHUNK_SIZE, 2, dtype=torch.float32, device=device)
    tgt_tools = torch.zeros(n, dtype=torch.long, device=device)
    tgt_gate = torch.zeros(n, dtype=torch.float32, device=device)
    tgt_val = torch.zeros(n, dtype=torch.float32, device=device)
    actor_weights = torch.ones(n, dtype=torch.float32, device=device)

    for i, r in enumerate(records):
        latents[i] = torch.tensor(r["latent_h"], dtype=torch.float32, device=device).clamp(-20.0, 20.0)
        proprios[i] = torch.tensor(r["proprio24"], dtype=torch.float32, device=device)
        voxels[i] = torch.tensor(r["voxels"], dtype=torch.long, device=device).view(15, 15, 15).clamp(0, 31)

        ents = r.get("entities", [])
        for ei in range(min(NUM_ENTITIES, len(ents))):
            if isinstance(ents[ei], list) and len(ents[ei]) == ENTITY_FEAT_DIM:
                entities[i, ei] = torch.tensor(ents[ei], dtype=torch.float32, device=device)

        aim_t = torch.tensor([row[:2] for row in r["aimTargets"]], dtype=torch.float32, device=device)
        aim_t[:, 0] = aim_t[:, 0].clamp(-180.0, 180.0)
        aim_t[:, 1] = aim_t[:, 1].clamp(-90.0, 90.0)
        tgt_aim[i] = aim_t

        tgt_move[i] = torch.tensor([row[:2] for row in r["moveTargets"]], dtype=torch.float32, device=device).clamp(-1.0, 1.0)
        tgt_disc[i] = torch.tensor([row[:6] for row in r["discreteTargets"]], dtype=torch.float32, device=device).clamp(0.0, 1.0)
        tgt_ext[i] = torch.tensor([row[:2] for row in r["extendedTargets"]], dtype=torch.float32, device=device).clamp(0.0, 1.0)

        t_name = str(r.get("toolName") or "none")
        if t_name in TOOL_TO_IDX and t_name != "none":
            tgt_tools[i] = TOOL_TO_IDX[t_name]
            tgt_gate[i] = 1.0
        else:
            tgt_tools[i] = TOOL_TO_IDX["none"]
            tgt_gate[i] = 0.0

        st_rew = float(r.get("reward", 0.0))
        ep_rew = float(r.get("episodeReward", 0.0))
        tgt_val[i] = max(-1.0, min(1.0, (0.5 * st_rew + 0.5 * ep_rew) / 100.0))

        ep_passed = bool(r.get("episodePassed", False))
        st_ok = bool(r.get("stepSuccess", False))
        if ep_passed and st_ok:
            actor_weights[i] = 1.5
        elif st_ok:
            actor_weights[i] = 1.0
        else:
            actor_weights[i] = 0.25

    return (
        latents, proprios, voxels, entities,
        tgt_aim, tgt_move, tgt_disc, tgt_ext,
        tgt_tools, tgt_gate, tgt_val, actor_weights
    )


def train_action_transformer(
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    loop_idx: int = 1,
    is_best: bool = False,
) -> Dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 74)
    print("  NEUROCRAFT 211M MULTI-MODAL TEMPORAL ACTION TRANSFORMER TRAINER")
    print("  Architecture: MinecraftActionHead (12 Layers, 16 Heads, 85-Token Context)")
    print(f"  Device: {gpu_name} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 74)

    t0 = time.perf_counter()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = MinecraftActionHead().to(device=device, dtype=dtype)
    ckpt_path = CHECKPOINT_DIR / "action_head_latest.pt"
    effective_lr = lr
    if ckpt_path.exists():
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.to(device=device, dtype=dtype)
            effective_lr = min(lr, 1e-4)
            print(f"[Action Head] Resumed existing weights from {ckpt_path} (fine-tune lr={effective_lr:.1e})")
        except Exception as e:
            print(f"[Action Head] Could not resume checkpoint ({e}); starting from scratch.")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Action Head] Model has {total_params:,} parameters ({total_params * 2 / 1e6:.1f} MB in BF16).")

    real_ds = load_real_transitions_dataset(max_samples=3000, device=str(device))
    num_real = int(real_ds[0].shape[0]) if real_ds is not None else 0
    motor_chunks_ds = load_real_motor_chunks_dataset(max_samples=2000, device=str(device))
    num_motor_chunks = int(motor_chunks_ds[0].shape[0]) if motor_chunks_ds is not None else 0
    num_synth = max(800, 4000 - num_real - num_motor_chunks)
    print(
        f"[Action Head] Generating {num_synth:,} annealed synthetic + {num_real:,} real Mojang transitions "
        f"+ {num_motor_chunks:,} real 20Hz SLM-latent motor chunks..."
    )
    ds = list(generate_temporal_action_dataset(num_samples=num_synth, device=str(device)))
    if real_ds is not None:
        print(f"[Action Head] Appended {num_real} reward-weighted real server transition trajectories!")
        for idx in range(len(ds)):
            ds[idx] = torch.cat([ds[idx], real_ds[idx]], dim=0)
    if motor_chunks_ds is not None:
        print(f"[Action Head] Appended {num_motor_chunks} real-latent 20Hz motor chunks (real_motor_chunks.jsonl)!")
        for idx in range(len(ds)):
            ds[idx] = torch.cat([ds[idx], motor_chunks_ds[idx]], dim=0)

    dataset = TensorDataset(*ds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    initial_loss = 0.0
    final_loss = 0.0
    avg_acc = 0.0
    avg_aim_deg = 0.0
    avg_tool_acc = 0.0

    print(f"[Action Head] Starting optimization across {len(loader)} batches per epoch...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_aim_err = 0.0
        epoch_move_err = 0.0
        epoch_discrete_acc = 0.0
        epoch_tool_acc = 0.0
        batches = 0

        for b_lat, b_prop, b_vox, b_ent, b_aim, b_move, b_disc, b_ext, b_tool, b_gate, b_val, b_w in loader:
            optimizer.zero_grad()

            out = model(
                b_lat.to(dtype),
                b_prop.to(dtype),
                voxels=b_vox,
                entities=b_ent.to(dtype),
            )

            pred_aim = out["trajectory_aim"]
            pred_move = out["trajectory_movement"]
            pred_discrete_logits = out["trajectory_discrete_logits"]
            pred_ext_logits = out["trajectory_extended_logits"]
            pred_tool_logits = out["tool_logits"]
            pred_gate_logit = out["tool_gate_logit"]
            pred_val = out["value"]

            w_actor = b_w.to(dtype=dtype)
            w_actor_f32 = b_w.float()
            loss_aim = (F.mse_loss(pred_aim, b_aim.to(dtype), reduction="none").mean(dim=(1, 2)) * w_actor).mean() * 0.001
            loss_move = (F.mse_loss(pred_move, b_move.to(dtype), reduction="none").mean(dim=(1, 2)) * w_actor).mean() * 2.0
            loss_disc = (F.binary_cross_entropy_with_logits(pred_discrete_logits, b_disc.to(dtype), reduction="none").mean(dim=(1, 2)) * w_actor).mean() * 2.5
            loss_ext = (F.binary_cross_entropy_with_logits(pred_ext_logits, b_ext.to(dtype), reduction="none").mean(dim=(1, 2)) * w_actor).mean() * 1.0
            loss_tool = (F.cross_entropy(pred_tool_logits.float(), b_tool, reduction="none") * w_actor_f32).mean() * 0.75
            loss_gate = (F.binary_cross_entropy_with_logits(pred_gate_logit, b_gate.to(dtype), reduction="none") * w_actor).mean() * 0.5
            loss_val = criterion_mse(pred_val, b_val.to(dtype)) * 0.5

            total_loss = loss_aim + loss_move + loss_disc + loss_ext + loss_tool + loss_gate + loss_val
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(total_loss.item())
            with torch.no_grad():
                epoch_aim_err += float(torch.abs(pred_aim - b_aim).mean().item())
                epoch_move_err += float(torch.abs(pred_move - b_move).mean().item())
                disc_preds = (torch.sigmoid(pred_discrete_logits) >= 0.5).float()
                epoch_discrete_acc += float((disc_preds == b_disc).float().mean().item())
                tool_preds = torch.argmax(pred_tool_logits, dim=-1)
                epoch_tool_acc += float((tool_preds == b_tool).float().mean().item())
            batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, batches)
        avg_aim_deg = epoch_aim_err / max(1, batches)
        avg_move_err = epoch_move_err / max(1, batches)
        avg_acc = (epoch_discrete_acc / max(1, batches)) * 100.0
        avg_tool_acc = (epoch_tool_acc / max(1, batches)) * 100.0

        if epoch == 1:
            initial_loss = avg_loss
        final_loss = avg_loss

        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        print(
            f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | Aim Err: {avg_aim_deg:.2f}° | "
            f"Move Err: {avg_move_err:.3f} | Disc Acc: {avg_acc:.1f}% | Tool Acc: {avg_tool_acc:.1f}% | VRAM: {vram_mb:.0f} MB"
        )

    ckpt_path = CHECKPOINT_DIR / "action_head_latest.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n[Action Head] Successfully saved trained 211M weights to {ckpt_path}!")

    if is_best:
        best_path = CHECKPOINT_DIR / "action_head_best.pt"
        torch.save(model.state_dict(), best_path)
        print(f"[Action Head] Saved new all-time best weights to {best_path}!")

    total_time = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    metrics = {
        "status": "success",
        "model": "MinecraftActionHead",
        "parameters": total_params,
        "epochs": epochs,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "discrete_accuracy_pct": round(avg_acc, 2),
        "tool_dispatch_accuracy_pct": round(avg_tool_acc, 2),
        "aim_error_deg": round(avg_aim_deg, 2),
        "realTransitionsLoaded": num_real,
        "realMotorChunksLoaded": num_motor_chunks,
        "peak_vram_mb": round(peak_vram, 1),
        "training_time_s": round(total_time, 2),
        "checkpoint_path": str(ckpt_path),
        "loopIdx": loop_idx,
    }

    metrics_path = RESULTS_DIR / "action_head_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 74)
    print(f"  TRAINING COMPLETE in {total_time:.1f}s! Peak VRAM: {peak_vram:.0f} MB | Final Loss: {final_loss:.4f}")
    print("=" * 74)
    print(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeuroCraft 211M Multi-Modal Action Transformer Trainer")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--loop", type=int, default=1, help="Current training loop index")
    parser.add_argument("--is-best", action="store_true", help="Flag if this iteration achieved a new high score")
    args = parser.parse_args()

    train_action_transformer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loop_idx=args.loop,
        is_best=args.is_best,
    )
