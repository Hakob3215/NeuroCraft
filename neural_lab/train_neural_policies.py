"""
NEUROCRAFT GPU TRAINING PIPELINE
Trains the three System 1 Neural Policies on the NVIDIA GeForce RTX 5060 (Blackwell sm_120):
1. VoxelReflexPolicy     -> checkpoints/voxel_reflex_policy.pt
2. VolumetricChunkAnalyzer -> checkpoints/chunk_analyzer.pt
3. VisuomotorPolicy      -> checkpoints/visuomotor_policy.pt

Logs full training telemetry, loss curves, and validation accuracy to:
  neural_lab/training_results.txt
"""

import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from models import VoxelReflexPolicy, VolumetricChunkAnalyzer, VisuomotorPolicy, REFLEX_ACTION_NAMES


def generate_reflex_dataset(num_samples: int = 6000, device: str = "cuda:0"):
    """
    Generates physically grounded 3D voxel + entity + bot_state scenarios and expert ground-truth actions:
      0: Creeper Fuse Evasion (<4.8m)
      1: Skeleton Projectile Strafe & Shield (5-14m)
      2: Melee Hostile Critical Strike Combo (2-4m, weapon ready)
      3: Melee Hostile Kiting / Cooldown Recharge (2-4m, weapon on cooldown)
      4: Lava / Deep Cliff Edge Hazard Ahead
      5: Parkour 1-2 Block Gap Sprint-Jump
      6: Calm Exploration / Safe Navigation
    """
    torch.manual_seed(42)
    voxels = torch.zeros((num_samples, 15, 15, 15), dtype=torch.long, device=device)
    # Default solid ground floor at Y=0..6 (bot feet at center [7, 7, 7])
    voxels[:, :, 0:7, :] = 6  # grass_block / stone

    entities = torch.zeros((num_samples, 16, 12), dtype=torch.float32, device=device)
    bot_state = torch.zeros((num_samples, 10), dtype=torch.float32, device=device)
    # Default healthy grounded bot
    bot_state[:, 0] = 0.9   # health_norm
    bot_state[:, 1] = 0.9   # food_norm
    bot_state[:, 2] = 1.0   # oxygen_norm
    bot_state[:, 3] = 1.0   # on_ground
    bot_state[:, 6] = 1.0   # has_shield
    bot_state[:, 7] = 1.0   # weapon_cooldown (ready)

    target_aim = torch.zeros((num_samples, 2), dtype=torch.float32, device=device)
    target_actions = torch.zeros((num_samples, len(REFLEX_ACTION_NAMES)), dtype=torch.float32, device=device)
    target_urgency = torch.zeros((num_samples, 1), dtype=torch.float32, device=device)
    scenario_labels = torch.zeros((num_samples,), dtype=torch.long, device=device)

    for i in range(num_samples):
        mode = i % 7
        scenario_labels[i] = mode

        if mode == 0:
            # Creeper Fuse Evasion (<4.5m)
            rx = (torch.rand(1, device=device).item() - 0.5) * 5.0
            rz = (torch.rand(1, device=device).item() - 0.5) * 5.0
            if abs(rx) < 0.8 and abs(rz) < 0.8:
                rx = 2.0
            dist = math.sqrt(rx * rx + rz * rz)
            entities[i, 0] = torch.tensor([rx, 0.0, rz, -rx * 0.2, 0.0, -rz * 0.2, dist, 1.0, 1.0, 0.0, 1.0, 1.0], device=device)
            # Aim toward creeper to block with shield OR turn away and sprint-jump back
            dyaw = math.atan2(-rx, -rz) * 0.5
            target_aim[i] = torch.tensor([max(-1.4, min(1.4, dyaw)), 0.0], device=device)
            # move_back=1, jump=1, sprint=1, use_shield=1
            target_actions[i, 1] = 1.0  # move_back
            target_actions[i, 4] = 1.0  # jump
            target_actions[i, 5] = 1.0  # sprint
            target_actions[i, 8] = 1.0  # use_shield
            target_urgency[i, 0] = 0.96

        elif mode == 1:
            # Skeleton Projectile Strafe & Shield
            rx = (torch.rand(1, device=device).item() - 0.5) * 14.0
            rz = -6.0 - torch.rand(1, device=device).item() * 6.0
            dist = math.sqrt(rx * rx + rz * rz)
            entities[i, 0] = torch.tensor([rx, 0.0, rz, 0.0, 0.0, 0.0, dist, 1.0, 0.0, 1.0, 1.0, 1.0], device=device)
            dyaw = math.atan2(-rx, -rz) * 0.4
            target_aim[i] = torch.tensor([max(-1.4, min(1.4, dyaw)), -0.05], device=device)
            target_actions[i, 0] = 1.0  # move_forward
            target_actions[i, 2 if rx >= 0 else 3] = 1.0  # strafe
            target_actions[i, 8] = 1.0  # use_shield
            target_urgency[i, 0] = 0.82

        elif mode == 2:
            # Melee Hostile Critical Strike (weapon ready)
            rx = (torch.rand(1, device=device).item() - 0.5) * 3.5
            rz = -1.5 - torch.rand(1, device=device).item() * 2.0
            dist = math.sqrt(rx * rx + rz * rz)
            entities[i, 0] = torch.tensor([rx, 0.0, rz, 0.0, 0.0, 0.2, dist, 1.0, 0.0, 0.0, 0.8, 1.0], device=device)
            bot_state[i, 7] = 1.0  # weapon ready
            dyaw = math.atan2(-rx, -rz) * 0.5
            target_aim[i] = torch.tensor([max(-1.4, min(1.4, dyaw)), -0.1], device=device)
            target_actions[i, 0] = 1.0  # move_forward
            target_actions[i, 4] = 1.0  # jump (for falling critical strike)
            target_actions[i, 7] = 1.0  # attack
            target_urgency[i, 0] = 0.78

        elif mode == 3:
            # Melee Hostile Kiting while weapon on cooldown (<0.4)
            rx = (torch.rand(1, device=device).item() - 0.5) * 3.0
            rz = -1.2 - torch.rand(1, device=device).item() * 1.5
            dist = math.sqrt(rx * rx + rz * rz)
            entities[i, 0] = torch.tensor([rx, 0.0, rz, 0.0, 0.0, 0.3, dist, 1.0, 0.0, 0.0, 0.8, 1.0], device=device)
            bot_state[i, 7] = 0.2  # weapon recharging!
            dyaw = math.atan2(-rx, -rz) * 0.5
            target_aim[i] = torch.tensor([max(-1.4, min(1.4, dyaw)), -0.1], device=device)
            target_actions[i, 1] = 1.0  # move_back (spacing)
            target_actions[i, 2] = 1.0  # strafe_left
            target_actions[i, 8] = 1.0  # use_shield while recharging
            target_urgency[i, 0] = 0.74

        elif mode == 4:
            # Lava / Deep Cliff Ahead (voxels ahead in -Z [7, 0:7, 3:6] are lava=11 or air=0)
            voxels[i, 5:10, 0:7, 3:6] = 11  # Lava pool ahead
            bot_state[i, 5] = 0.0
            target_aim[i] = torch.tensor([0.8, 0.2], device=device)  # Turn away
            target_actions[i, 1] = 1.0  # move_back
            target_actions[i, 6] = 1.0  # sneak (don't fall off edge!)
            target_urgency[i, 0] = 0.94

        elif mode == 5:
            # Parkour 1-2 block gap ahead (air at Z=5..6, solid ground at Z=0..4)
            voxels[i, 5:10, 0:7, 5:7] = 0  # Air trench
            target_aim[i] = torch.tensor([0.0, 0.0], device=device)
            target_actions[i, 0] = 1.0  # move_forward
            target_actions[i, 4] = 1.0  # jump
            target_actions[i, 5] = 1.0  # sprint
            target_urgency[i, 0] = 0.68

        else:
            # Mode 6: Safe Calm Navigation
            target_aim[i] = torch.tensor([0.0, 0.0], device=device)
            target_actions[i, 0] = 1.0  # move_forward
            target_urgency[i, 0] = 0.05

    return voxels, entities, bot_state, target_aim, target_actions, target_urgency, scenario_labels


def generate_chunk_dataset(num_chunks: int = 512, device: str = "cuda:0"):
    """
    Generates 64x64x64 chunks with:
    - Embedded flat grass building pad at known offset (bx, by, bz)
    - Embedded rich iron/diamond ore cluster at known offset (ox, oy, oz)
    - Optional lava/cavern hazard
    """
    torch.manual_seed(101)
    chunks = torch.zeros((num_chunks, 64, 64, 64), dtype=torch.long, device=device)
    # Base subterranean stone Y=0..28, dirt Y=28..31, grass Y=31
    chunks[:, :, 0:28, :] = 1
    chunks[:, :, 28:31, :] = 5
    chunks[:, :, 31, :] = 6

    targets = torch.zeros((num_chunks, 9), dtype=torch.float32, device=device)

    for i in range(num_chunks):
        # Random flat building pad center in [12..52]
        bx = torch.randint(12, 52, (1,), device=device).item()
        bz = torch.randint(12, 52, (1,), device=device).item()
        by = 32
        chunks[i, bx - 4 : bx + 4, by, bz - 4 : bz + 4] = 14  # Oak planks / cleared foundation marker

        # Random rich ore vein in underground stone [8..56, 6..24, 8..56]
        ox = torch.randint(8, 56, (1,), device=device).item()
        oy = torch.randint(6, 24, (1,), device=device).item()
        oz = torch.randint(8, 56, (1,), device=device).item()
        chunks[i, ox - 2 : ox + 2, oy - 2 : oy + 2, oz - 2 : oz + 2] = 20  # Diamond / Iron ore cluster

        # Hazard lava pocket on half of chunks
        has_lava = (i % 2 == 0)
        if has_lava:
            chunks[i, 28:36, 29:32, 28:36] = 11  # Surface lava pool

        targets[i, 0] = float(bx - 32)
        targets[i, 1] = float(by - 32)
        targets[i, 2] = float(bz - 32)
        targets[i, 3] = 0.92
        targets[i, 4] = float(ox - 32)
        targets[i, 5] = float(oy - 32)
        targets[i, 6] = float(oz - 32)
        targets[i, 7] = 0.95
        targets[i, 8] = 0.88 if has_lava else 0.08

    return chunks, targets


def train_all():
    os.makedirs("checkpoints", exist_ok=True)
    lines = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log("=" * 78)
    log("  NEUROCRAFT GPU POLICY TRAINING PIPELINE (RTX 5060 BLACKWELL)")
    log("=" * 78)
    log(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # -------------------------------------------------------------------------
    # 1. TRAIN VOXEL REFLEX POLICY
    # -------------------------------------------------------------------------
    log("\n[1/3] Training VoxelReflexPolicy (6,000 3D Combat & Hazard Scenarios)...")
    t0 = time.perf_counter()
    voxels, entities, bot_state, t_aim, t_act, t_urg, labels = generate_reflex_dataset(6000, device=device)

    # Split 5000 train / 1000 val
    train_ds = TensorDataset(voxels[:5000], entities[:5000], bot_state[:5000], t_aim[:5000], t_act[:5000], t_urg[:5000])
    val_v, val_e, val_b, val_aim, val_act, val_urg = (
        voxels[5000:], entities[5000:], bot_state[5000:], t_aim[5000:], t_act[5000:], t_urg[5000:]
    )

    loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    reflex_model = VoxelReflexPolicy().to(device)
    optimizer = torch.optim.AdamW(reflex_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    for epoch in range(1, 16):
        reflex_model.train()
        total_loss = 0.0
        for bv, be, bb, b_aim, b_act, b_urg in loader:
            optimizer.zero_grad()
            p_aim, p_logits, p_urg = reflex_model(bv, be, bb)
            loss_aim = F.mse_loss(p_aim, b_aim)
            loss_act = F.binary_cross_entropy_with_logits(p_logits, b_act)
            loss_urg = F.mse_loss(p_urg, b_urg)
            loss = loss_aim + loss_act * 2.0 + loss_urg * 2.0
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch in (1, 5, 10, 15):
            reflex_model.eval()
            with torch.no_grad():
                vp_aim, vp_logits, vp_urg = reflex_model(val_v, val_e, val_b)
                pred_bits = (torch.sigmoid(vp_logits) > 0.5).float()
                bit_acc = (pred_bits == val_act).float().mean().item() * 100.0
                urg_err = F.l1_loss(vp_urg, val_urg).item()
            log(f"  Epoch {epoch:2d}/15 | Loss: {total_loss/len(loader):.4f} | Val Action Acc: {bit_acc:.2f}% | Urgency MAE: {urg_err:.4f}")

    reflex_time = time.perf_counter() - t0
    torch.save(reflex_model.state_dict(), "checkpoints/voxel_reflex_policy.pt")
    log(f"  -> Saved checkpoints/voxel_reflex_policy.pt in {reflex_time:.2f}s!")

    del voxels, entities, bot_state, t_aim, t_act, t_urg, train_ds, loader
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 2. TRAIN VOLUMETRIC CHUNK ANALYZER (64x64x64)
    # -------------------------------------------------------------------------
    log("\n[2/3] Training VolumetricChunkAnalyzer (64x64x64 3D Terrain & Ore Scanner)...")
    t0 = time.perf_counter()
    chunks, chunk_targets = generate_chunk_dataset(320, device="cpu")
    chunk_ds = TensorDataset(chunks[:256], chunk_targets[:256])
    val_chunks = chunks[256:].to(device)
    val_ctargets = chunk_targets[256:].to(device)

    chunk_loader = DataLoader(chunk_ds, batch_size=16, shuffle=True)
    chunk_model = VolumetricChunkAnalyzer().to(device)
    c_opt = torch.optim.AdamW(chunk_model.parameters(), lr=3e-4, weight_decay=1e-4)

    for epoch in range(1, 13):
        chunk_model.train()
        c_loss_sum = 0.0
        for bc, bt in chunk_loader:
            bc = bc.to(device)
            bt = bt.to(device)
            c_opt.zero_grad()
            pred = chunk_model(bc)
            # Coordinate offset loss (normalized by 32) + confidence score loss
            coord_loss = F.smooth_l1_loss(pred[:, [0, 1, 2, 4, 5, 6]] / 32.0, bt[:, [0, 1, 2, 4, 5, 6]] / 32.0)
            score_loss = F.mse_loss(pred[:, [3, 7, 8]], bt[:, [3, 7, 8]])
            loss = coord_loss + score_loss * 2.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(chunk_model.parameters(), 1.0)
            c_opt.step()
            c_loss_sum += loss.item()

        if epoch in (1, 4, 8, 12):
            chunk_model.eval()
            with torch.no_grad():
                vpred = chunk_model(val_chunks)
                hazard_acc = ((vpred[:, 8] > 0.5) == (val_ctargets[:, 8] > 0.5)).float().mean().item() * 100.0
                score_mae = F.l1_loss(vpred[:, [3, 7, 8]], val_ctargets[:, [3, 7, 8]]).item()
            log(f"  Epoch {epoch:2d}/12 | Loss: {c_loss_sum/len(chunk_loader):.4f} | Hazard Detection Acc: {hazard_acc:.1f}% | Score MAE: {score_mae:.4f}")

    chunk_time = time.perf_counter() - t0
    torch.save(chunk_model.state_dict(), "checkpoints/chunk_analyzer.pt")
    log(f"  -> Saved checkpoints/chunk_analyzer.pt in {chunk_time:.2f}s!")

    del chunks, chunk_targets, val_chunks, val_ctargets, chunk_ds, chunk_loader
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 3. TRAIN VISUOMOTOR POLICY (8x128x128 RGB + Goal Vector)
    # -------------------------------------------------------------------------
    log("\n[3/3] Training VisuomotorPolicy (Spatiotemporal Visual Target Tracker)...")
    t0 = time.perf_counter()
    num_vis = 240
    # Synthetic visual frames with bright target patch at (tx, ty)
    vis_frames = torch.randn((num_vis, 8, 3, 128, 128), dtype=torch.float16, device="cpu") * 0.1
    vis_goals = torch.randn((num_vis, 512), dtype=torch.float16, device="cpu")
    vis_mouse = torch.zeros((num_vis, 2), dtype=torch.float32, device="cpu")
    vis_keys = torch.zeros((num_vis, 16), dtype=torch.float32, device="cpu")

    for i in range(num_vis):
        tx = torch.randint(20, 108, (1,)).item()
        ty = torch.randint(20, 108, (1,)).item()
        vis_frames[i, :, 0, ty - 8 : ty + 8, tx - 8 : tx + 8] = 2.0  # Salient target in frame
        dx = (tx - 64.0) / 64.0
        dy = (ty - 64.0) / 64.0
        vis_mouse[i, 0] = dx
        vis_mouse[i, 1] = dy
        vis_keys[i, 0] = 1.0  # W (forward)
        if abs(dx) < 0.25 and abs(dy) < 0.25:
            vis_keys[i, 7] = 1.0  # Attack / harvest when centered on target

    vis_ds = TensorDataset(vis_frames[:200], vis_goals[:200], vis_mouse[:200], vis_keys[:200])
    vis_loader = DataLoader(vis_ds, batch_size=16, shuffle=True)

    vis_model = VisuomotorPolicy().to(device)
    v_opt = torch.optim.AdamW(vis_model.parameters(), lr=5e-4, weight_decay=1e-4)

    for epoch in range(1, 9):
        vis_model.train()
        v_loss_sum = 0.0
        for bf, bg, bm, bk in vis_loader:
            bf = bf.to(device, dtype=torch.float32)
            bg = bg.to(device, dtype=torch.float32)
            bm = bm.to(device)
            bk = bk.to(device)
            v_opt.zero_grad()
            pm, pk = vis_model(bf, bg)
            loss = F.mse_loss(pm, bm) + F.binary_cross_entropy_with_logits(pk, bk)
            loss.backward()
            v_opt.step()
            v_loss_sum += loss.item()
        if epoch in (1, 4, 8):
            log(f"  Epoch {epoch:2d}/8  | Loss: {v_loss_sum/len(vis_loader):.4f}")

    vis_time = time.perf_counter() - t0
    torch.save(vis_model.state_dict(), "checkpoints/visuomotor_policy.pt")
    log(f"  -> Saved checkpoints/visuomotor_policy.pt in {vis_time:.2f}s!")

    log("\n" + "=" * 78)
    log("  ALL 3 NEURAL POLICIES TRAINED & SAVED TO neural_lab/checkpoints/")
    log("=" * 78)

    with open("training_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    train_all()
