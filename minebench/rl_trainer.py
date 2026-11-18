"""
MINEbench Dual-System Reinforcement Learning Trainer (RTX 5060 GPU)
===================================================================
Trains directly on REAL gameplay transitions recorded from the live Mineflayer
agent connected to the official Mojang 1.20.4 Headless Server (`server.jar`):
1. System 1 Motor/Combat RL:
   - Clipped PPO (Proximal Policy Optimization) on `VoxelReflexPolicy` using
     real 3D voxel grids (`real_rl_transitions.jsonl`), real entity tensors,
     real bot telemetry, and real step rewards.
   - Zero-Bloat Checkpoint Management:
     - Always updates `voxel_reflex_policy_latest.pt` (and active `voxel_reflex_policy.pt`).
     - If `--is-best` is flagged, also updates `voxel_reflex_policy_best.pt`.
2. System 2 Cognitive Planner RL:
   - GRPO (Group Relative Policy Optimization) advantage calculation and
     trajectory distillation from `rl_replay_buffer.jsonl`.
"""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from neural_lab.models import REFLEX_ACTION_NAMES, VoxelReflexPolicy

NUM_DISCRETE_ACTIONS = len(REFLEX_ACTION_NAMES)

CHECKPOINT_DIR = ROOT_DIR / "neural_lab" / "checkpoints"
RESULTS_DIR = ROOT_DIR / "minebench" / "results"
REPLAY_BUFFER_PATH = RESULTS_DIR / "rl_replay_buffer.jsonl"
REAL_TRANSITIONS_PATH = RESULTS_DIR / "real_rl_transitions.jsonl"
GRPO_DATASET_PATH = RESULTS_DIR / "grpo_verified_trajectories.jsonl"
RL_METRICS_PATH = RESULTS_DIR / "rl_training_metrics.json"


def build_rl_batch_from_real_transitions(
    step_transitions: List[Dict],
    episode_records: List[Dict],
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    voxels = torch.zeros((batch_size, 15, 15, 15), dtype=torch.long, device=device)
    entities = torch.zeros((batch_size, 16, 12), dtype=torch.float32, device=device)
    bot_state = torch.zeros((batch_size, 10), dtype=torch.float32, device=device)
    target_actions = torch.zeros((batch_size, NUM_DISCRETE_ACTIONS), dtype=torch.float32, device=device)
    target_aim = torch.zeros((batch_size, 2), dtype=torch.float32, device=device)
    rewards = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    ep_reward_map = {
        (r.get("provider", ""), r.get("taskId", "")): float(r.get("rawReward", 0.0))
        for r in episode_records
    }

    for i in range(batch_size):
        tr = step_transitions[i % len(step_transitions)]
        raw_vox = tr.get("voxels", [])
        if len(raw_vox) == 3375:
            voxels[i] = torch.tensor(raw_vox, dtype=torch.long, device=device).view(15, 15, 15).clamp(min=0, max=31)
        elif len(raw_vox) == 4096:
            vox_t = torch.tensor(raw_vox, dtype=torch.long, device=device).view(16, 16, 16)
            voxels[i] = vox_t[:15, :15, :15].clamp(min=0, max=31)

        raw_ent = tr.get("entities", [])
        if len(raw_ent) == 16:
            entities[i] = torch.tensor(raw_ent, dtype=torch.float32, device=device)

        raw_state = tr.get("botState", [])
        if len(raw_state) == 10:
            bot_state[i] = torch.tensor(raw_state, dtype=torch.float32, device=device)

        act_idx = int(tr.get("actionIdx", 0)) % NUM_DISCRETE_ACTIONS
        target_actions[i, act_idx] = 1.0

        aim = tr.get("aimDelta", [0.0, 0.0])
        if len(aim) == 2:
            target_aim[i, 0] = float(aim[0])
            target_aim[i, 1] = float(aim[1])

        step_r = float(tr.get("reward", 0.0))
        ep_r = ep_reward_map.get((tr.get("provider", ""), tr.get("taskId", "")), 0.0)
        rewards[i] = step_r + 0.25 * ep_r

    mean_r = rewards.mean()
    std_r = rewards.std().clamp(min=1e-4)
    advantages = ((rewards - mean_r) / std_r).clamp(min=-2.5, max=2.5)
    positive_weights = torch.sigmoid(advantages) + 0.25

    return voxels, entities, bot_state, target_actions, target_aim, positive_weights


VALID_MCP_TOOLS = {
    "follow_player", "come_here", "navigate_to_coordinates", "gather_resource",
    "explore_or_dig_for_ores", "craft_item", "craft_recipe", "smelt_item",
    "smelt_ores", "equip_armor_or_item", "engage_combat", "give_item_to_player",
    "store_items_in_chest", "retrieve_items_from_chest", "scan_area_topology",
    "build_simple_house", "neural_chunk_scan", "collect_dropped_items",
    "tame_animal", "pillar_up", "eat_food", "shoot_bow", "till_and_farm",
    "interact_block_or_puzzle", "navigate_parkour_or_bridge", "construct_modular_structure",
    "triangulate_stronghold_coordinates", "loot_chest", "shear_sheep",
    "breed_animals", "use_tool_on_block", "light_block_fire",
}


def is_valid_tool_for_task(task_id: str, prompt: str, tool_name: str) -> bool:
    """Rejects hallucinated tools and cross-domain poisoned replay matches before QLoRA training."""
    t = (tool_name or "").strip()
    tid = (task_id or "").lower()
    p = (prompt or "").lower()

    if not t or t not in VALID_MCP_TOOLS:
        return False
    if t == "follow_player" and "follow" not in p:
        return False

    # Prevent equip_armor_or_item pollution onto food, sheepshearing, mobs, or crafting tasks
    if t == "equip_armor_or_item":
        if tid.startswith("omni_food_") or tid.startswith("omni_mob_") or tid.startswith("omni_craft_"):
            return False
        if not (tid.startswith("omni_equip_") or "equip" in p or "armor" in p or "shield" in p or "wear" in p):
            return False

    # Prevent explore_or_dig_for_ores pollution onto food, mob, shears, lead, or path tasks
    if t == "explore_or_dig_for_ores":
        if tid.startswith("omni_food_") or tid.startswith("omni_mob_") or "shear" in tid or "leash" in tid:
            return False

    # Prevent crafting/harvesting pollution when consuming food
    if tid.startswith("omni_food_") and t not in {"eat_food", "craft_item", "smelt_item", "gather_resource"}:
        return False

    return True


def run_minebench_rl_update(epochs: int = 8, is_best: bool = False, loop_idx: int = None) -> Dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    records: List[Dict] = []
    if REPLAY_BUFFER_PATH.exists():
        with open(REPLAY_BUFFER_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

    step_transitions: List[Dict] = []
    if REAL_TRANSITIONS_PATH.exists():
        with open(REAL_TRANSITIONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        step_transitions.append(json.loads(line))
                    except Exception:
                        pass

    if not step_transitions:
        raise RuntimeError("No real server step transitions found in real_rl_transitions.jsonl!")

    batch_size = min(64, max(8, len(step_transitions)))

    policy = VoxelReflexPolicy().to(device)
    old_policy = VoxelReflexPolicy().to(device)
    
    # Priority load order: latest checkpoint -> base checkpoint
    latest_ckpt = CHECKPOINT_DIR / "voxel_reflex_policy_latest.pt"
    base_ckpt = CHECKPOINT_DIR / "voxel_reflex_policy.pt"
    ckpt_to_load = latest_ckpt if latest_ckpt.exists() else base_ckpt
    if ckpt_to_load.exists():
        state_dict = torch.load(ckpt_to_load, map_location=device, weights_only=True)
        policy.load_state_dict(state_dict)
        old_policy.load_state_dict(state_dict)
    old_policy.eval()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    clip_eps = 0.2

    t0 = time.perf_counter()
    initial_loss = 0.0
    final_loss = 0.0
    total_steps = 0

    policy.train()
    for ep in range(epochs):
        ep_loss = 0.0
        steps_per_epoch = 8
        for _ in range(steps_per_epoch):
            voxels, entities, bot_state, target_actions, target_aim, adv_weights = (
                build_rl_batch_from_real_transitions(step_transitions, records, batch_size, device)
            )

            with torch.no_grad():
                _, old_action_logits, _ = old_policy(voxels, entities, bot_state)
                old_probs = torch.sigmoid(old_action_logits).detach()

            aim_delta, action_logits, _ = policy(voxels, entities, bot_state)
            new_probs = torch.sigmoid(action_logits)

            p_new = torch.where(target_actions > 0.5, new_probs, 1.0 - new_probs).clamp(min=1e-5)
            p_old = torch.where(target_actions > 0.5, old_probs, 1.0 - old_probs).clamp(min=1e-5)
            ratio = (p_new / p_old).mean(dim=-1)

            surr1 = ratio * adv_weights
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_weights
            ppo_policy_loss = -torch.min(surr1, surr2).mean()

            bce_loss = F.binary_cross_entropy_with_logits(
                action_logits, target_actions, reduction="none"
            ).mean(dim=-1)
            weighted_bce = (bce_loss * adv_weights).mean()
            aim_loss = (
                F.smooth_l1_loss(aim_delta, target_aim, reduction="none").mean(dim=-1)
                * adv_weights
            ).mean()

            loss = 0.4 * ppo_policy_loss + 0.6 * weighted_bce + 0.5 * aim_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            ep_loss += float(loss.item())
            total_steps += batch_size

        avg_ep_loss = ep_loss / steps_per_epoch
        if ep == 0:
            initial_loss = avg_ep_loss
        final_loss = avg_ep_loss

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    # Save weights with Zero-Disk-Bloat policy (latest + best + distinct loop checkpoints up to 4)
    latest_path = CHECKPOINT_DIR / "voxel_reflex_policy_latest.pt"
    active_path = CHECKPOINT_DIR / "voxel_reflex_policy.pt"
    best_path = CHECKPOINT_DIR / "voxel_reflex_policy_best.pt"

    torch.save(policy.state_dict(), latest_path)
    torch.save(policy.state_dict(), active_path)
    if is_best:
        torch.save(policy.state_dict(), best_path)
    if loop_idx is not None and loop_idx <= 4:
        loop_path = CHECKPOINT_DIR / f"voxel_reflex_policy_loop{loop_idx}.pt"
        torch.save(policy.state_dict(), loop_path)

    # Export GRPO trajectories: Stratified Reservoir Sampling (Task Gold Bank)
    # 1. Load & scrub existing verified trajectories (filter out cross-domain poison & hallucinations)
    task_gold_bank: Dict[Tuple[str, int], Dict] = {}
    gold_seed_samples: List[Dict] = []

    def _extract_tool_and_prompt(entry: Dict) -> Tuple[str, str]:
        msgs = entry.get("messages", [])
        u_msg = str(msgs[1].get("content", "")) if len(msgs) > 1 else ""
        a_msg = str(msgs[2].get("content", "")) if len(msgs) > 2 else ""
        t_name = ""
        try:
            parsed_a = json.loads(a_msg)
            t_name = str(parsed_a.get("toolCall", {}).get("name", ""))
        except Exception:
            pass
        return u_msg, t_name

    if GRPO_DATASET_PATH.exists():
        with open(GRPO_DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        tid = d.get("taskId")
                        step = int(d.get("step", 0))
                        u_msg, t_name = _extract_tool_and_prompt(d)
                        if not tid or not is_valid_tool_for_task(tid, u_msg, t_name):
                            continue
                        prov = d.get("provider", "")
                        if prov in ("gold_curriculum", "gold_continuation"):
                            gold_seed_samples.append(d)
                        else:
                            key = (tid, step)
                            if key not in task_gold_bank or float(d.get("rawReward", 0)) > float(task_gold_bank[key].get("rawReward", 0)):
                                task_gold_bank[key] = d
                    except Exception:
                        pass

    # Also seed from omniscience_gold_dataset.jsonl if present and not yet in bank
    omni_gold_path = RESULTS_DIR / "omniscience_gold_dataset.jsonl"
    if omni_gold_path.exists():
        with open(omni_gold_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        tid = d.get("taskId")
                        step = int(d.get("step", 0))
                        u_msg, t_name = _extract_tool_and_prompt(d)
                        if tid and is_valid_tool_for_task(tid, u_msg, t_name):
                            key = (tid, step)
                            if key not in task_gold_bank:
                                task_gold_bank[key] = d
                    except Exception:
                        pass

    # 2. Mine all clean verified passed trajectories across replay buffer records
    rewards_list = [float(r.get("rawReward", 0.0)) for r in records]
    mean_rew = sum(rewards_list) / max(1, len(rewards_list))
    var_rew = sum((x - mean_rew) ** 2 for x in rewards_list) / max(1, len(rewards_list))
    std_rew = max(1e-3, math.sqrt(var_rew))

    recent_verified: List[Dict] = []

    for r_idx, r in enumerate(records):
        if not r.get("passed", False):
            continue

        raw_r = float(r.get("rawReward", 0.0))
        tid = r.get("taskId")
        prompt = str(r.get("prompt", "Execute task."))
        if not tid:
            continue

        adv = (raw_r - mean_rew) / std_rew
        exec_tools = r.get("executedTools", [])
        is_recent = r_idx >= len(records) - 300
        prior_steps: List[Dict] = []

        for step_idx, tool_entry in enumerate(exec_tools):
            tool_name = str(tool_entry.get("name", ""))
            tool_args = tool_entry.get("args") or tool_entry.get("arguments") or {}
            step_ok = bool(tool_entry.get("success", True))
            step_summary = str(tool_entry.get("summary") or f"Executed {tool_name}")

            # STRICT PURITY: Only teach successful, task-valid tool invocations
            if not step_ok or not is_valid_tool_for_task(tid, prompt, tool_name):
                prior_steps.append({"name": tool_name, "args": tool_args, "ok": step_ok, "summary": step_summary})
                continue

            if len(prior_steps) == 0:
                user_content = (
                    f'Player "Trainer" says: "{prompt}"\n'
                    'Return ONLY a valid JSON object: {"thought":"...", "toolCall":{"name":"...", "arguments":{}}}'
                )
            else:
                history_lines = "\n".join(
                    f"Step {i + 1}: {ps['name']}({json.dumps(ps['args'])}) -> {'SUCCESS' if ps['ok'] else 'FAILED'}: {ps['summary']}"
                    for i, ps in enumerate(prior_steps[-10:])
                )
                user_content = (
                    f'Active Goal: "{prompt}"\n'
                    f'Recent Steps:\n{history_lines}\n'
                    'Return ONLY a valid JSON object: {"thought":"...", "toolCall":{"name":"...", "arguments":{}}}'
                )

            sample = {
                "provider": r.get("provider", "local_gpu"),
                "taskId": tid,
                "domain": r.get("domain"),
                "advantage": round(adv, 4),
                "rawReward": raw_r,
                "step": step_idx,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are NeuroAgent, an autonomous Minecraft 1.20.4 AI companion. Respond ONLY with valid JSON.",
                    },
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "thought": f"Executing {tool_name} to progress toward objective.",
                                "toolCall": {
                                    "name": tool_name,
                                    "arguments": tool_args,
                                },
                                "goalCompleted": step_idx == len(exec_tools) - 1,
                            }
                        ),
                    },
                ],
            }

            key = (tid, step_idx)
            # Stratified Gold Bank: Keep best trajectory per (taskId, step)
            if key not in task_gold_bank or raw_r >= float(task_gold_bank[key].get("rawReward", 0.0)):
                task_gold_bank[key] = sample

            if is_recent:
                recent_verified.append(sample)

            prior_steps.append({"name": tool_name, "args": tool_args, "ok": step_ok, "summary": step_summary})

    # 3. Assemble clean balanced dataset: canonical gold seeds + task bank anchors + recent verified rollouts
    bank_samples = list(task_gold_bank.values())
    combined_dataset = gold_seed_samples + bank_samples + recent_verified[-80:]

    with open(GRPO_DATASET_PATH, "w", encoding="utf-8") as gf:
        for s in combined_dataset:
            gf.write(json.dumps(s) + "\n")

    print(f"[RL Trainer] Clean Stratified Task Bank: {len(gold_seed_samples)} seeds + {len(task_gold_bank)} task anchors = {len(combined_dataset)} pure samples saved.")

    metrics = {
        "device": gpu_name,
        "epochs": epochs,
        "batchSize": batch_size,
        "totalTransitionsTrained": total_steps,
        "realStepTransitionsLoaded": len(step_transitions),
        "transitionsPerSecond": round(total_steps / max(1e-4, elapsed_s), 1),
        "initialLoss": round(initial_loss, 4),
        "finalLoss": round(final_loss, 4),
        "trainingDurationSeconds": round(elapsed_s, 3),
        "replayEpisodesProcessed": len(records),
        "grpoVerifiedTrajectories": len(combined_dataset),
        "latestCheckpointSaved": str(latest_path),
        "isBestCheckpoint": is_best,
    }

    with open(RL_METRICS_PATH, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2)

    print(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--is-best", action="store_true")
    parser.add_argument("--loop", type=int, default=None)
    args = parser.parse_args()
    run_minebench_rl_update(epochs=args.epochs, is_best=args.is_best, loop_idx=args.loop)
