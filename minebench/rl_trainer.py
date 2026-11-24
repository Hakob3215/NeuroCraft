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
import re
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
DPO_PAIRS_PATH = RESULTS_DIR / "dpo_pairs.jsonl"
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
    "store_items_in_chest", "manage_chest", "inspect_surroundings", "clear_flat_area",
    "stop_current_action", "scan_area", "build_simple_house", "neural_chunk_scan",
    "collect_dropped_items", "tame_animal", "pillar_up", "eat_food", "shoot_bow",
    "till_and_farm", "interact_block_or_puzzle", "navigate_parkour_or_bridge",
    "construct_modular_structure", "triangulate_stronghold_coordinates",
    "loot_chest", "shear_sheep", "breed_animals", "use_tool_on_block",
    "light_block_fire",
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


REQUIRED_TOOL_ARG_ALIASES = {
    "gather_resource": ("blockType", "resource", "block"),
    "craft_item": ("itemName", "recipeName", "item"),
    "smelt_item": ("inputItem", "itemName", "oreType"),
    "equip_armor_or_item": ("itemName", "item"),
    "eat_food": ("foodName", "itemName", "food"),
    "give_item_to_player": ("itemName", "item"),
}


def has_required_args(tool_name: str, tool_args: Dict) -> bool:
    """Rejects empty/missing required arguments so degenerate calls are excluded from positive GRPO training."""
    t = (tool_name or "").strip()
    required = REQUIRED_TOOL_ARG_ALIASES.get(t)
    if not required:
        return True
    if not isinstance(tool_args, dict):
        return False
    return any(isinstance(tool_args.get(k), str) and len(tool_args.get(k).strip()) > 0 for k in required)


def compact_step_summary(summary: str) -> str:
    """Strips redundant Current inventory tails from historical step summaries for train/inference parity."""
    cleaned = re.sub(r"\s*Current inventory:\s*\[[^\]]*\]\.?", "", str(summary or ""), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def should_mark_goal_completed(
    is_passed: bool,
    score_val: float,
    provider_name: str,
    num_steps: int,
    step_idx: int,
) -> bool:
    """Only mark goalCompleted=True on genuinely completed multi-step final steps, never on Step 0 or open-world survival."""
    return bool(
        is_passed
        and score_val >= 95.0
        and provider_name != "open_world_survival"
        and num_steps >= 2
        and step_idx > 0
        and step_idx == num_steps - 1
    )


def sanitize_loaded_grpo_entry(entry: Dict) -> Dict:
    """Scrubs legacy Step-0 or open-world goalCompleted=True flags from loaded non-seed GRPO entries."""
    if entry.get("provider") in ("gold_curriculum", "gold_continuation"):
        return entry
    msgs = entry.get("messages", [])
    if isinstance(msgs, list) and len(msgs) > 0:
        last_msg = msgs[-1]
        if isinstance(last_msg, dict) and isinstance(last_msg.get("content"), str):
            try:
                parsed = json.loads(last_msg["content"])
                if isinstance(parsed, dict) and (
                    int(entry.get("step", 0)) <= 0
                    or entry.get("provider") == "open_world_survival"
                    or not bool(entry.get("verifiedFinalStep", False))
                ):
                    parsed["goalCompleted"] = False
                    last_msg["content"] = json.dumps(parsed)
            except Exception:
                pass
    return entry


def read_jsonl_tail(path: Path, max_lines: int = 5000, max_bytes: int = 48 * 1024 * 1024) -> List[str]:
    """Reads up to the last `max_lines` complete lines from a JSONL file in O(1) time/memory."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()  # discard partial first line
        raw = f.read()
    lines = raw.decode("utf-8", errors="ignore").splitlines()
    return lines[-max_lines:]


def run_minebench_rl_update(epochs: int = 8, is_best: bool = False, loop_idx: int = None) -> Dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    records: List[Dict] = []
    if REPLAY_BUFFER_PATH.exists():
        for line in read_jsonl_tail(REPLAY_BUFFER_PATH, max_lines=4000):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    step_transitions: List[Dict] = []
    if REAL_TRANSITIONS_PATH.exists():
        for line in read_jsonl_tail(REAL_TRANSITIONS_PATH, max_lines=5000):
            line = line.strip()
            if line:
                try:
                    step_transitions.append(json.loads(line))
                except Exception:
                    pass

    if not step_transitions:
        print("[RL Trainer] Note: No real server step transitions found in real_rl_transitions.jsonl. Skipping PPO gradient step.", flush=True)
        metrics = {
            "status": "skipped",
            "reason": "no_transitions",
            "device": gpu_name,
            "transitionsTrained": 0,
            "initialLoss": 0.0,
            "finalLoss": 0.0,
            "transitionsPerSecond": 0.0,
            "isBest": is_best,
            "loopIdx": loop_idx,
        }
        with open(RL_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics))
        return metrics

    batch_size = min(64, max(8, len(step_transitions)))
    latest_path = CHECKPOINT_DIR / "voxel_reflex_policy_latest.pt"
    active_path = CHECKPOINT_DIR / "voxel_reflex_policy.pt"
    best_path = CHECKPOINT_DIR / "voxel_reflex_policy_best.pt"

    initial_loss = 0.0
    final_loss = 0.0
    total_steps = 0
    elapsed_s = 0.0

    if epochs > 0:
        policy = VoxelReflexPolicy().to(device)
        old_policy = VoxelReflexPolicy().to(device)

        # Priority load order: latest checkpoint -> base checkpoint
        ckpt_to_load = latest_path if latest_path.exists() else (CHECKPOINT_DIR / "voxel_reflex_policy.pt")
        if ckpt_to_load.exists():
            state_dict = torch.load(ckpt_to_load, map_location=device, weights_only=True)
            policy.load_state_dict(state_dict)
            old_policy.load_state_dict(state_dict)
        old_policy.eval()

        optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
        clip_eps = 0.2

        t0 = time.perf_counter()

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
                            d = sanitize_loaded_grpo_entry(d)
                            key = (tid, step)
                            if key not in task_gold_bank or float(d.get("rawReward", 0)) > float(task_gold_bank[key].get("rawReward", 0)):
                                task_gold_bank[key] = d
                    except Exception:
                        pass

    def _infer_domain_from_tid(tid: str) -> str:
        t = str(tid or "").lower()
        if t.startswith("omni_tool_"):
            return "TOOL_AFFORDANCES"
        if t.startswith("omni_food_"):
            return "NUTRITION"
        if t.startswith("omni_equip_"):
            return "EQUIPMENT"
        if t.startswith("omni_block_"):
            return "BLOCK_MINING"
        if t.startswith("omni_mob_"):
            return "MOB_INTERACTION"
        if t.startswith("omni_structure_"):
            return "STRUCTURE_RAID"
        if t.startswith("prog_"):
            return "PROGRESSION_DRAGON"
        if t.startswith("combat_"):
            return "COMBAT_TACTICAL"
        return "SURVIVAL"

    def _normalize_omni_gold_entry(d: Dict, tid: str) -> Dict:
        if not d.get("domain"):
            d["domain"] = _infer_domain_from_tid(tid)
        msgs = list(d.get("messages") or [])
        if len(msgs) >= 3:
            sys_c = str(msgs[0].get("content") or "")
            if "NeuroAgent" not in sys_c or "Use native action tokens" in sys_c or sys_c.strip() in ("", "sys"):
                msgs[0] = {
                    "role": "system",
                    "content": (
                        "You are NeuroAgent, an autonomous Minecraft AI companion.\n"
                        f"State: Domain:{d['domain']} HP:20/20 Food:20/20\n"
                        "Recent Actions: [none]\n"
                        "Respond ONLY with valid JSON."
                    ),
                }
            u_c = str(msgs[1].get("content") or "").strip()
            if not u_c.startswith('Player "') and not u_c.startswith("Active Goal:"):
                msgs[1] = {
                    "role": "user",
                    "content": (
                        f'Player "Trainer" says: "{u_c}"\n'
                        'Return ONLY a valid JSON object: {"thought":"...", "toolCall":{"name":"...", "arguments":{}}}'
                    ),
                }
            try:
                parsed_a = json.loads(str(msgs[2].get("content") or "{}"))
                if isinstance(parsed_a, dict) and "goalCompleted" not in parsed_a:
                    parsed_a["goalCompleted"] = False
                    msgs[2] = {"role": "assistant", "content": json.dumps(parsed_a)}
            except Exception:
                pass
            d["messages"] = msgs
        return d

    # Also seed from omniscience_gold_dataset.jsonl if present and not yet in bank
    omni_gold_path = RESULTS_DIR / "omniscience_gold_dataset.jsonl"
    if omni_gold_path.exists():
        with open(omni_gold_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        raw_tid = d.get("taskId") or d.get("id")
                        tid = re.sub(r"^omni_gold_\d+_", "", str(raw_tid or "")).strip()
                        step = int(d.get("step", 0))
                        u_msg, t_name = _extract_tool_and_prompt(d)
                        if tid.startswith("omni_craft_") or t_name in ("craft_item", "craft_recipe"):
                            continue
                        if tid and is_valid_tool_for_task(tid, u_msg, t_name):
                            d["taskId"] = tid
                            d["provider"] = d.get("provider") or "omniscience_gold"
                            d["passed"] = False
                            d = _normalize_omni_gold_entry(d, tid)
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

    def _build_fallback_grounded_system_prompt(rec: Dict, prior: List[Dict]) -> str:
        recent_str = (
            ", ".join(
                f"{ps['name']}({'OK' if ps['ok'] else 'FAIL'})" for ps in prior[-3:]
            )
            if prior
            else "none"
        )
        hp = rec.get("health", 20)
        food = rec.get("food", 20)
        dom = rec.get("domain", "SURVIVAL")
        return (
            "You are NeuroAgent, an autonomous Minecraft AI companion.\n"
            f"State: Domain:{dom} HP:{hp}/20 Food:{food}/20\n"
            f"Recent Actions: [{recent_str}]\n"
            "Respond ONLY with valid JSON."
        )

    recent_verified: List[Dict] = []

    for r_idx, r in enumerate(records):
        if bool(r.get("isHeldOut", False)):
            continue
        is_passed = bool(r.get("passed", False))
        score_val = float(r.get("score", 0.0))
        raw_r = float(r.get("rawReward", 0.0))
        provider_r = str(r.get("provider", "local_gpu"))

        tid = r.get("taskId")
        prompt = str(r.get("prompt", "Execute task."))
        if not tid:
            continue

        if not is_passed:
            if provider_r == "open_world_survival":
                continue
            tid_s = str(tid)
            if not (tid_s.startswith("prog_") or tid_s.startswith("omni_craft_") or tid_s.startswith("build_")):
                continue
            if score_val < 25.0 and raw_r < 25.0:
                continue

        exec_tools = r.get("executedTools", [])
        adv = (raw_r - mean_rew) / std_rew
        if not is_passed and adv < 0.0:
            continue

        max_valid_step_idx = len(exec_tools) - 1
        if not is_passed:
            # Distill only verified productive prefixes for positive-advantage multi-step DAG partial episodes
            max_valid_step_idx = -1
            for s_i, t_entry in enumerate(exec_tools):
                if bool(t_entry.get("success", False)) and float(t_entry.get("stepReward", 0.0)) > 0.0:
                    max_valid_step_idx = s_i
            if max_valid_step_idx < 0:
                continue

        is_recent = r_idx >= len(records) - 300
        prior_steps: List[Dict] = []

        for step_idx, tool_entry in enumerate(exec_tools):
            if not is_passed and step_idx > max_valid_step_idx:
                break

            tool_name = str(tool_entry.get("name", ""))
            tool_args = tool_entry.get("args") or tool_entry.get("arguments") or {}
            step_ok = bool(tool_entry.get("success", True))
            step_summary = compact_step_summary(str(tool_entry.get("summary") or f"Executed {tool_name}"))

            step_reward_val = float(tool_entry.get("stepReward", 0.0) or 0.0)

            # STRICT PURITY: Only teach successful, task-valid, non-empty-arg tool invocations (and >0 stepReward in failed episodes)
            if (
                not step_ok
                or (not is_passed and step_reward_val <= 0.0)
                or not is_valid_tool_for_task(tid, prompt, tool_name)
                or not has_required_args(tool_name, tool_args)
            ):
                prior_steps.append({"name": tool_name, "args": tool_args, "ok": step_ok, "summary": step_summary})
                continue

            rec_speaker = str(r.get("speaker") or "Trainer").strip() or "Trainer"
            if len(prior_steps) == 0:
                user_content = (
                    f'Player "{rec_speaker}" says: "{prompt}"\n'
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

            recorded_sys_prompt = str(tool_entry.get("systemPrompt") or "").strip()
            sys_content = (
                recorded_sys_prompt
                if recorded_sys_prompt
                else _build_fallback_grounded_system_prompt(r, prior_steps)
            )

            provider_name = str(r.get("provider", "local_gpu"))
            can_mark_goal_completed = should_mark_goal_completed(
                is_passed=is_passed,
                score_val=score_val,
                provider_name=provider_name,
                num_steps=len(exec_tools),
                step_idx=step_idx,
            )

            real_thought = str(tool_entry.get("thought") or "").strip()
            assistant_obj: Dict = {
                "thought": real_thought if real_thought else f"Executing {tool_name} to progress toward objective.",
                "toolCall": {
                    "name": tool_name,
                    "arguments": tool_args,
                },
                "goalCompleted": can_mark_goal_completed,
            }
            plan_steps = tool_entry.get("planSteps")
            if step_idx == 0 and isinstance(plan_steps, list) and len(plan_steps) > 0:
                assistant_obj["planSteps"] = plan_steps

            has_explicit_rew = "stepReward" in tool_entry and tool_entry.get("stepReward") is not None

            sample = {
                "provider": provider_name,
                "taskId": tid,
                "domain": r.get("domain"),
                "workerId": r.get("workerId"),
                "basePrompt": str(prompt or "").strip(),
                "passed": is_passed,
                "advantage": round(adv, 4),
                "rawReward": raw_r,
                "stepReward": step_reward_val,
                "hasExplicitStepReward": has_explicit_rew,
                "step": step_idx,
                "priorTools": [ps["name"] for ps in prior_steps],
                "verifiedFinalStep": can_mark_goal_completed,
                "messages": [
                    {
                        "role": "system",
                        "content": sys_content,
                    },
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant_obj),
                    },
                ],
            }

            key = (tid, step_idx)
            # Stratified Gold Bank: Keep best PASSED trajectory per (taskId, step); never let partial fails overwrite gold anchors
            prev_best = task_gold_bank.get(key)
            prev_passed = bool(prev_best.get("passed", False)) if prev_best else False
            prev_is_gold = bool(prev_best and prev_best.get("provider") in ("gold_curriculum", "gold_continuation", "omniscience_gold"))
            prev_raw = float(prev_best.get("rawReward", 0.0)) if prev_best else -1e9
            if (
                prev_best is None
                or (is_passed and not prev_passed)
                or (not prev_is_gold and is_passed == prev_passed and raw_r >= prev_raw)
            ):
                task_gold_bank[key] = sample

            if is_recent:
                recent_verified.append(sample)

            prior_steps.append({"name": tool_name, "args": tool_args, "ok": step_ok, "summary": step_summary})

    # 3. Assemble clean balanced deduplicated dataset: canonical gold seeds + task bank anchors + recent verified rollouts
    bank_samples = list(task_gold_bank.values())
    raw_combined = gold_seed_samples + bank_samples + recent_verified[-80:]
    seen_grpo = set()
    combined_dataset: List[Dict] = []
    for s in raw_combined:
        msgs = s.get("messages", [])
        asst_c = msgs[-1].get("content", "") if isinstance(msgs, list) and msgs else ""
        dedup_key = (s.get("provider"), s.get("taskId"), s.get("step"), asst_c)
        if dedup_key not in seen_grpo:
            seen_grpo.add(dedup_key)
            combined_dataset.append(s)

    with open(GRPO_DATASET_PATH, "w", encoding="utf-8") as gf:
        for s in combined_dataset:
            gf.write(json.dumps(s) + "\n")

    # 4. Mine contrastive DPO pairs (strictly verified PASSED winner vs actual FAILED loser on matching prefix/shard)
    coord_or_worker_sensitive_tools = {
        "navigate_to_coordinates",
        "till_and_farm",
        "engage_combat",
        "give_item_to_player",
        "follow_player",
    }
    dpo_candidates: Dict[Tuple[str, int], Dict[Tuple[str, str], Dict]] = {}
    for r in records:
        if bool(r.get("isHeldOut", False)):
            continue
        tid = r.get("taskId")
        if not tid:
            continue
        tid_lower = str(tid).lower()
        rej_base_prompt = str(r.get("prompt") or "").strip()
        is_passed = bool(r.get("passed", False))
        raw_r = float(r.get("rawReward", 0.0))
        exec_tools = r.get("executedTools", [])
        for step_idx, tool_entry in enumerate(exec_tools):
            winner = task_gold_bank.get((tid, step_idx))
            if not winner or not bool(winner.get("passed", False)):
                continue
            winner_raw_r = float(winner.get("rawReward", 0.0))
            if winner_raw_r < 50.0:
                continue
            if bool(winner.get("hasExplicitStepReward", False)):
                if float(winner.get("stepReward", 0.0)) <= 0.0 and not bool(winner.get("verifiedFinalStep", False)):
                    continue

            tool_name = str(tool_entry.get("name", ""))
            tool_args = tool_entry.get("args") or tool_entry.get("arguments") or {}
            step_ok = bool(tool_entry.get("success", True))
            rej_has_explicit_rew = "stepReward" in tool_entry and tool_entry.get("stepReward") is not None
            rej_step_rew = float(tool_entry.get("stepReward", 0.0) or 0.0)
            is_loser_step = (not step_ok) or (not is_passed) or (rej_has_explicit_rew and rej_step_rew <= 0.0)
            if not is_loser_step:
                continue

            if step_idx > 0:
                rej_prior_tools = [str(pe.get("name", "")) for pe in exec_tools[:step_idx]]
                if rej_prior_tools != list(winner.get("priorTools") or []):
                    continue

            try:
                w_parsed = json.loads(winner["messages"][2]["content"])
                w_tool = str(w_parsed.get("toolCall", {}).get("name", ""))
                w_args_json = json.dumps(w_parsed.get("toolCall", {}).get("arguments") or {}, sort_keys=True)
            except Exception:
                continue

            winner_domain_lower = str(winner.get("domain") or r.get("domain") or "").lower()
            is_combat_or_mob_task = (
                "combat" in tid_lower
                or tid_lower.startswith("omni_mob_")
                or "combat" in winner_domain_lower
                or "mob" in winner_domain_lower
            )
            if w_tool == "navigate_to_coordinates" and is_combat_or_mob_task:
                continue

            if tool_name in coord_or_worker_sensitive_tools or w_tool in coord_or_worker_sensitive_tools:
                if (
                    winner.get("workerId") is not None
                    and r.get("workerId") is not None
                    and winner.get("workerId") != r.get("workerId")
                ):
                    continue
                winner_base_prompt = str(winner.get("basePrompt") or "").strip()
                if winner_base_prompt and rej_base_prompt and winner_base_prompt != rej_base_prompt:
                    continue
                if step_idx == 0 and len(winner.get("messages", [])) > 1:
                    rej_speaker = str(r.get("speaker") or "Trainer").strip() or "Trainer"
                    expected_rej_user = (
                        f'Player "{rej_speaker}" says: "{rej_base_prompt}"\n'
                        'Return ONLY a valid JSON object: {"thought":"...", "toolCall":{"name":"...", "arguments":{}}}'
                    )
                    if expected_rej_user != str(winner["messages"][1].get("content", "")):
                        continue

            rej_args_json = json.dumps(tool_args if isinstance(tool_args, dict) else {}, sort_keys=True)
            if tool_name == w_tool and rej_args_json == w_args_json:
                continue

            rej_thought = str(tool_entry.get("thought") or "").strip() or f"Executing {tool_name}."
            rejected_content = json.dumps(
                {
                    "thought": rej_thought,
                    "toolCall": {
                        "name": tool_name,
                        "arguments": tool_args if isinstance(tool_args, dict) else {},
                    },
                    "goalCompleted": False,
                }
            )
            margin = round(winner_raw_r - raw_r, 2)
            key = (tid, step_idx)
            sig = (tool_name, rej_args_json)
            bucket = dpo_candidates.setdefault(key, {})
            if sig not in bucket or margin > float(bucket[sig].get("margin", 0.0)):
                bucket[sig] = {
                    "taskId": tid,
                    "domain": winner.get("domain") or r.get("domain"),
                    "step": step_idx,
                    "margin": margin,
                    "prompt_messages": winner["messages"][:2],
                    "chosen": winner["messages"][2]["content"],
                    "rejected": rejected_content,
                }

    dpo_pairs: List[Dict] = []
    for bucket in dpo_candidates.values():
        top_two = sorted(bucket.values(), key=lambda x: float(x.get("margin", 0.0)), reverse=True)[:2]
        dpo_pairs.extend(top_two)

    with open(DPO_PAIRS_PATH, "w", encoding="utf-8") as df:
        for p in dpo_pairs:
            df.write(json.dumps(p) + "\n")

    print(f"[RL Trainer] Clean Stratified Task Bank: {len(gold_seed_samples)} seeds + {len(task_gold_bank)} task anchors = {len(combined_dataset)} pure samples | DPO pairs: {len(dpo_pairs)}.")

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
        "dpoPairsSaved": len(dpo_pairs),
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
