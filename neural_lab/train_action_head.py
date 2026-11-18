"""
NeuroCraft Scaled-Up Action Head Trainer (RTX 5060 GPU)
======================================================
Trains the 80.8-Million-Parameter Temporal Action Chunking Transformer
(MinecraftActionHead) on multi-tick continuous trajectories and discrete motor affordances.

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
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from neural_lab.action_head import (
    MinecraftActionHead,
    PROPRIOCEPTION_DIM,
    LATENT_DIM,
    CHUNK_SIZE,
)

CHECKPOINT_DIR = ROOT_DIR / "neural_lab" / "checkpoints"
RESULTS_DIR = ROOT_DIR / "neural_lab" / "results"


def generate_temporal_action_dataset(
    num_samples: int = 8000,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Synthesizes diverse behavioral scenarios with 8-tick temporal trajectories
    covering combat kiting, MLG clutching, shield blocking, mining, and parkour.
    """
    torch.manual_seed(1337)

    latent_dim = LATENT_DIM
    proprio_dim = PROPRIOCEPTION_DIM
    chunk_size = CHUNK_SIZE

    # Input Tensors
    latents = torch.randn(num_samples, latent_dim, device=device) * 0.1
    proprios = torch.zeros(num_samples, proprio_dim, device=device)
    voxels = torch.zeros(num_samples, 15, 15, 15, device=device)
    voxels[:, :, 0:7, :] = 1.0  # Solid stone ground

    # Target Tensors across 8-tick trajectory chunk
    target_aim = torch.zeros(num_samples, chunk_size, 2, device=device)         # [B, 8, 2] in [-180, 180]
    target_move = torch.zeros(num_samples, chunk_size, 2, device=device)        # [B, 8, 2] in [-1, 1]
    target_discrete = torch.zeros(num_samples, chunk_size, 6, device=device)    # [B, 8, 6] in {0, 1}
    target_values = torch.zeros(num_samples, device=device)                     # [B]

    for i in range(num_samples):
        mode = i % 8

        # Base proprioception: healthy grounded bot
        proprios[i, 0] = 0.9   # health
        proprios[i, 1] = 0.9   # food
        proprios[i, 7] = 1.0   # onGround
        proprios[i, 14] = 1.0  # sword equipped (aligned with vlaClient.ts)
        proprios[i, 15] = 1.0  # shield equipped (aligned with vlaClient.ts)

        if mode == 0:
            # Mode 0: Creeper Tactical Kiting (Backpedal + Shield on Fuse)
            latents[i, 0:32] += 1.5   # Combat intent signature
            proprios[i, 12] = 3.5 / 32.0  # Normalized Creeper dist (index 12)
            proprios[i, 13] = 1.0     # Creeper fuse active flag (index 13)

            for t in range(chunk_size):
                # Backpedal smoothly away: vz = -1.0, slight circle strafe vx = 0.3
                target_move[i, t, 0] = 0.3 * math.sin(t * 0.5)
                target_move[i, t, 1] = -0.9
                # Face creeper directly
                target_aim[i, t, 0] = 0.0
                target_aim[i, t, 1] = -5.0
                # Discrete: sprint=1, use_item=1 (shield block)
                target_discrete[i, t, 1] = 1.0  # sprint
                target_discrete[i, t, 4] = 1.0  # use_item (shield)

            target_values[i] = 0.85

        elif mode == 1:
            # Mode 1: Critical Strike Attack on Zombie / Skeleton
            latents[i, 32:64] += 1.5   # Melee aggression signature
            proprios[i, 12] = 2.4 / 32.0  # Target dist (index 12)

            for t in range(chunk_size):
                # Advance forward
                target_move[i, t, 1] = 1.0
                target_aim[i, t, 0] = 0.0
                target_aim[i, t, 1] = 12.0  # Look slightly down at torso (+12 deg in Minecraft is down)
                # Jump at tick 0, attack at falling tick 3 & 4
                if t < 2:
                    target_discrete[i, t, 0] = 1.0  # jump
                elif 2 <= t <= 4:
                    target_discrete[i, t, 3] = 1.0  # attack (crit)

            target_values[i] = 0.90

        elif mode == 2:
            # Mode 2: Ravine / Cliff Edge Detection & Parkour Sprint-Jump
            latents[i, 64:96] += 1.5  # Navigation / Parkour signature
            voxels[i, 7:15, 0:7, 9:15] = 0.0  # Void pit ahead at Z >= 9

            for t in range(chunk_size):
                # Full forward sprint
                target_move[i, t, 1] = 1.0
                target_discrete[i, t, 1] = 1.0  # sprint
                # Jump right before pit boundary (tick 2..4)
                if 2 <= t <= 5:
                    target_discrete[i, t, 0] = 1.0  # jump

            target_values[i] = 0.75

        elif mode == 3:
            # Mode 3: Precision Mining / Excavation / Stripping
            latents[i, 96:128] += 1.5  # Harvesting signature
            proprios[i, 14] = 0.0      # unequip sword
            proprios[i, 15] = 0.0      # unequip shield
            proprios[i, 17] = 1.0      # pickaxe equipped (index 17)

            for t in range(chunk_size):
                # Stationary
                target_move[i, t, 0] = 0.0
                target_move[i, t, 1] = 0.0
                target_aim[i, t, 0] = 0.0
                target_aim[i, t, 1] = 35.0  # Pitch down at ground block
                target_discrete[i, t, 3] = 1.0  # attack/mine continuously

            target_values[i] = 0.60

        elif mode == 4:
            # Mode 4: MLG Water Clutch from High Fall
            latents[i, 128:160] += 2.0  # Emergency clutch signature
            proprios[i, 5] = -0.8       # Falling fast vy = -0.8
            proprios[i, 7] = 0.0        # in air

            for t in range(chunk_size):
                target_move[i, t, 0] = 0.0
                target_move[i, t, 1] = 0.0
                target_aim[i, t, 1] = 88.0  # Look straight down at feet
                # Swap to bucket at t=0, place water at t=6..7
                if t == 0:
                    target_discrete[i, t, 5] = 1.0  # swap_slot
                if t >= 6:
                    target_discrete[i, t, 4] = 1.0  # use_item (water place)

            target_values[i] = 0.95

        elif mode == 5:
            # Mode 5: Companion Interaction (Facing player, dropping items)
            latents[i, 160:192] += 1.5  # Companion speech & interaction
            for t in range(chunk_size):
                target_move[i, t, 0] = 0.0
                target_move[i, t, 1] = 0.0
                target_aim[i, t, 0] = 15.0 * math.cos(t * 0.4)
                target_aim[i, t, 1] = 0.0
                if t == 2:
                    target_discrete[i, t, 5] = 1.0  # swap slot to item
                if t >= 4:
                    target_discrete[i, t, 2] = 1.0  # sneak (polite companion nod)

            target_values[i] = 0.70

        elif mode == 6:
            # Mode 6: Archery Drawing & Distance Aim Arc
            latents[i, 192:224] += 1.5  # Ranged combat signature
            proprios[i, 12] = 18.0 / 32.0  # Target dist (index 12)
            proprios[i, 14] = 0.0         # unequip sword
            proprios[i, 15] = 0.0         # unequip shield
            proprios[i, 16] = 1.0         # bow equipped (index 16)
            for t in range(chunk_size):
                target_move[i, t, 0] = 0.1 * math.sin(t)
                target_move[i, t, 1] = 0.0
                target_aim[i, t, 1] = -14.0  # Elevated ballistic arc
                # Draw bow (hold use_item for ticks 0..5, release at tick 6)
                if t <= 5:
                    target_discrete[i, t, 4] = 1.0

            target_values[i] = 0.80

        elif mode == 7:
            # Mode 7: Safe Calm Exploration & Patrolling
            latents[i, 224:256] += 1.0
            for t in range(chunk_size):
                target_move[i, t, 1] = 0.7  # Gentle walk
                target_aim[i, t, 0] = 5.0 * math.sin(t * 0.3)
                target_aim[i, t, 1] = 0.0

            target_values[i] = 0.50

    return latents, proprios, voxels, target_aim, target_move, target_discrete, target_values


def load_real_transitions_dataset(
    max_samples: int = 2000,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Loads recent real Mojang server transitions into 8-tick temporal chunks."""
    trans_path = ROOT_DIR / "minebench" / "results" / "real_rl_transitions.jsonl"
    if not trans_path.exists():
        return None

    records = []
    try:
        with open(trans_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-max_samples:]
        for line in lines:
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
    voxels = torch.zeros(n, 15, 15, 15, device=device)
    tgt_aim = torch.zeros(n, CHUNK_SIZE, 2, device=device)
    tgt_move = torch.zeros(n, CHUNK_SIZE, 2, device=device)
    tgt_disc = torch.zeros(n, CHUNK_SIZE, 6, device=device)
    tgt_val = torch.zeros(n, device=device)

    for i, r in enumerate(records):
        bs = r["botState"]
        proprios[i, 0] = float(bs[0])  # health
        proprios[i, 1] = float(bs[1])  # food
        proprios[i, 4] = float(bs[2])  # vx
        proprios[i, 5] = float(bs[3])  # vy
        proprios[i, 6] = float(bs[4])  # vz
        proprios[i, 7] = float(bs[8])  # onGround
        proprios[i, 8] = float(bs[9])  # inWater

        # Nearest hostile entity into proprio slots 9..12
        ents = r.get("entities", [])
        if ents and len(ents[0]) >= 8 and ents[0][7] > 0.5:
            proprios[i, 9] = max(-1.0, min(1.0, float(ents[0][0]) * 2.0))
            proprios[i, 10] = max(-1.0, min(1.0, float(ents[0][1]) * 2.0))
            proprios[i, 11] = max(-1.0, min(1.0, float(ents[0][2]) * 2.0))
            proprios[i, 12] = float(ents[0][3])
            proprios[i, 13] = float(ents[0][11]) if len(ents[0]) > 11 else 0.0

        vox_t = torch.tensor(r["voxels"], dtype=torch.float32, device=device).view(15, 15, 15)
        voxels[i] = (vox_t > 0).float()

        aim = r.get("aimDelta", [0.0, 0.0])
        dyaw_deg = max(-180.0, min(180.0, float(aim[0]) * 180.0 / math.pi))
        dpitch_deg = max(-90.0, min(90.0, float(aim[1]) * 180.0 / math.pi))
        act_idx = int(r.get("actionIdx", 1))

        # Match semantic slice conditioning
        if act_idx == 3:  # melee combat
            latents[i, 32:64] += 1.5
            tgt_move[i, :, 1] = 1.0
            tgt_disc[i, 2:5, 3] = 1.0  # attack
        elif act_idx == 2:  # parkour/navigation
            latents[i, 64:96] += 1.5
            tgt_move[i, :, 1] = 1.0
            tgt_disc[i, :, 1] = 1.0  # sprint
            tgt_disc[i, 2:5, 0] = 1.0  # jump
        elif act_idx == 4:  # bow / build / use
            latents[i, 192:224] += 1.5
            tgt_disc[i, 0:6, 4] = 1.0  # use_item
        else:
            latents[i, 96:128] += 1.0
            tgt_move[i, :, 1] = 0.5

        tgt_aim[i, :, 0] = dyaw_deg
        tgt_aim[i, :, 1] = dpitch_deg
        rew = float(r.get("reward", 0.0))
        tgt_val[i] = 1.0 / (1.0 + math.exp(-max(-5.0, min(5.0, rew / 25.0))))

    return latents, proprios, voxels, tgt_aim, tgt_move, tgt_disc, tgt_val


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
    print("  NEUROCRAFT 80.8M TEMPORAL ACTION CHUNKING TRANSFORMER TRAINER")
    print(f"  Target Architecture: MinecraftActionHead (8 Layers, 12 Heads, 8-Tick Horizon)")
    print(f"  Device: {gpu_name} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 74)

    t0 = time.perf_counter()

    # 1. Instantiate or resume the 80.8M parameter Action Transformer
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

    # 2. Generate physically grounded training dataset + real server transitions
    print("[Action Head] Generating 4,000 synthetic + loading real Mojang server transitions...")
    latents, proprios, voxels, tgt_aim, tgt_move, tgt_discrete, tgt_val = generate_temporal_action_dataset(
        num_samples=4000, device=str(device)
    )
    real_ds = load_real_transitions_dataset(max_samples=2000, device=str(device))
    if real_ds is not None:
        r_lat, r_prop, r_vox, r_aim, r_move, r_disc, r_val = real_ds
        print(f"[Action Head] Appended {r_lat.shape[0]} real server transition trajectories!")
        latents = torch.cat([latents, r_lat], dim=0)
        proprios = torch.cat([proprios, r_prop], dim=0)
        voxels = torch.cat([voxels, r_vox], dim=0)
        tgt_aim = torch.cat([tgt_aim, r_aim], dim=0)
        tgt_move = torch.cat([tgt_move, r_move], dim=0)
        tgt_discrete = torch.cat([tgt_discrete, r_disc], dim=0)
        tgt_val = torch.cat([tgt_val, r_val], dim=0)

    dataset = TensorDataset(latents, proprios, voxels, tgt_aim, tgt_move, tgt_discrete, tgt_val)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    initial_loss = 0.0
    final_loss = 0.0

    print(f"[Action Head] Starting optimization across {len(loader)} batches per epoch...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_aim_err = 0.0
        epoch_move_err = 0.0
        epoch_discrete_acc = 0.0
        batches = 0

        for b_lat, b_prop, b_vox, b_aim, b_move, b_disc, b_val in loader:
            optimizer.zero_grad()

            out = model(b_lat.to(dtype), b_prop.to(dtype), voxels=b_vox.to(dtype))

            pred_aim = out["trajectory_aim"]                           # [B, 8, 2]
            pred_move = out["trajectory_movement"]                     # [B, 8, 2]
            pred_discrete_logits = out["trajectory_discrete_logits"]   # [B, 8, 6]
            pred_val = out["value"]                                    # [B]

            loss_aim = criterion_mse(pred_aim, b_aim.to(dtype)) * 0.001
            loss_move = criterion_mse(pred_move, b_move.to(dtype)) * 2.0
            loss_disc = criterion_bce(pred_discrete_logits, b_disc.to(dtype)) * 2.5
            loss_val = criterion_mse(pred_val, b_val.to(dtype)) * 0.5

            total_loss = loss_aim + loss_move + loss_disc + loss_val
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(total_loss.item())
            with torch.no_grad():
                epoch_aim_err += float(torch.abs(pred_aim - b_aim).mean().item())
                epoch_move_err += float(torch.abs(pred_move - b_move).mean().item())
                disc_preds = (torch.sigmoid(pred_discrete_logits) >= 0.5).float()
                epoch_discrete_acc += float((disc_preds == b_disc).float().mean().item())
            batches += 1

        scheduler.step()
        avg_loss = epoch_loss / batches
        avg_aim_deg = epoch_aim_err / batches
        avg_move_err = epoch_move_err / batches
        avg_acc = (epoch_discrete_acc / batches) * 100.0

        if epoch == 1:
            initial_loss = avg_loss
        final_loss = avg_loss

        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        print(
            f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | Aim Err: {avg_aim_deg:.2f}° | "
            f"Move Err: {avg_move_err:.3f} | Discrete Acc: {avg_acc:.1f}% | VRAM: {vram_mb:.0f} MB"
        )

    # 3. Save Checkpoints
    ckpt_path = CHECKPOINT_DIR / "action_head_latest.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n[Action Head] Successfully saved trained 80.8M weights to {ckpt_path}!")

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
        "aim_error_deg": round(avg_aim_deg, 2),
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
    parser = argparse.ArgumentParser(description="NeuroCraft 80.8M Action Chunking Transformer Trainer")
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
