"""
MINEbench System 2 Cognitive SLM QLoRA Trainer (RTX 5060 GPU)
============================================================
Fine-tunes the System 2 cognitive LLM (Qwen2.5-Coder-1.5B-Instruct) using
PEFT QLoRA (Rank 16) directly on high-advantage Minecraft gameplay trajectories
distilled into `minebench/results/grpo_verified_trajectories.jsonl`.

Transforms the frozen zero-shot model into an authentic "Minecraft Savant"
capable of high-efficiency inventory management, crafting paths, and spatial building.
"""

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure PyTorch allocator and cuBLAS workspace for VRAM fragmentation resistance and tensor core GEMMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from neural_lab.mc_tokens import register_minecraft_tokens

CHECKPOINT_DIR = ROOT_DIR / "neural_lab" / "checkpoints"
RESULTS_DIR = ROOT_DIR / "minebench" / "results"
GRPO_DATASET_PATH = RESULTS_DIR / "grpo_verified_trajectories.jsonl"
REPLAY_BUFFER_PATH = RESULTS_DIR / "rl_replay_buffer.jsonl"
DPO_PAIRS_PATH = RESULTS_DIR / "dpo_pairs.jsonl"
SLM_METRICS_PATH = RESULTS_DIR / "slm_training_metrics.json"

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def stratified_sample_by_domain(items: List[Dict], max_items: int) -> List[Dict]:
    """Balances samples round-robin across gameplay domains to prevent majority-domain collapse."""
    import random

    if not items or max_items <= 0:
        return []

    buckets: Dict[str, List[Dict]] = {}
    for item in items:
        dom = str(item.get("domain") or "general").strip().lower() or "general"
        buckets.setdefault(dom, []).append(item)

    for bucket in buckets.values():
        random.shuffle(bucket)

    domain_keys = list(buckets.keys())
    random.shuffle(domain_keys)

    selected: List[Dict] = []
    while len(selected) < max_items:
        progressed = False
        for dom in domain_keys:
            bucket = buckets[dom]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
                if len(selected) >= max_items:
                    break
        if not progressed:
            break

    random.shuffle(selected)
    return selected


def load_training_trajectories(max_trajectories: int = 512) -> List[Dict]:
    """Loads domain-stratified high-advantage trajectories from Omniscience, VLA, and GRPO datasets."""
    grpo_trajectories: List[Dict] = []
    if GRPO_DATASET_PATH.exists():
        with open(GRPO_DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        grpo_trajectories.append(json.loads(line))
                    except Exception:
                        pass
        print(f"[SLM Trainer] Loaded {len(grpo_trajectories)} real-world trajectories from GRPO Dataset.")

    gold_trajectories: List[Dict] = []
    omni_path = RESULTS_DIR / "omniscience_gold_dataset.jsonl"
    if omni_path.exists():
        with open(omni_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        gold_trajectories.append(json.loads(line))
                    except Exception:
                        pass

    vla_path = RESULTS_DIR / "vla_gold_dataset.jsonl"
    if vla_path.exists():
        with open(vla_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        gold_trajectories.append(json.loads(line))
                    except Exception:
                        pass

    import random
    # Prioritize real game experience balanced across domains, then fill remaining quota with gold demos
    selected_grpo = stratified_sample_by_domain(grpo_trajectories, max_trajectories)
    remaining_quota = max(0, max_trajectories - len(selected_grpo))
    selected_gold = stratified_sample_by_domain(gold_trajectories, remaining_quota)
    trajectories = selected_grpo + selected_gold
    random.shuffle(trajectories)

    # Ensure all trajectories have canonical toolCall in assistant message
    for traj in trajectories:
        msgs = traj.get("messages", [])
        if len(msgs) >= 3 and msgs[2].get("role") == "assistant":
            try:
                content = json.loads(msgs[2]["content"])
                if "toolCall" not in content and "executedTools" in content and content["executedTools"]:
                    first_t = content["executedTools"][0]
                    content["toolCall"] = {
                        "name": first_t.get("name", "idle"),
                        "arguments": first_t.get("args") or first_t.get("arguments") or {},
                    }
                    msgs[2]["content"] = json.dumps(content)
            except Exception:
                pass

    print(f"[SLM Trainer] Active balanced training dataset: {len(trajectories)} samples ({len(selected_grpo)} real experience, {len(trajectories) - len(selected_grpo)} gold demonstrations).")
    return trajectories


def load_dpo_pairs(max_pairs: int = 256) -> List[Dict]:
    """Loads domain-stratified contrastive DPO pairs (chosen vs rejected) mined from replay rollouts."""
    pairs: List[Dict] = []
    if not DPO_PAIRS_PATH.exists():
        return pairs
    with open(DPO_PAIRS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if (
                    isinstance(obj.get("prompt_messages"), list)
                    and len(obj["prompt_messages"]) >= 2
                    and isinstance(obj.get("chosen"), str)
                    and isinstance(obj.get("rejected"), str)
                ):
                    pairs.append(obj)
            except Exception:
                pass
    if not pairs:
        return []
    sampled = stratified_sample_by_domain(pairs, max_pairs)
    print(f"[SLM Trainer] Loaded {len(sampled)} domain-stratified contrastive DPO pairs (from {len(pairs)} mined).")
    return sampled


def compute_masked_causal_lm_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Computes exact causal LM cross-entropy ONLY on unmasked (labels != -100) target tokens.

    Avoids projecting masked prompt tokens (~95% of sequence) through the 151,936-vocab lm_head
    and FP32 CrossEntropyLoss, cutting peak VRAM by ~2.2 GiB at max_length=1280 with identical gradients.
    """
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(causal_lm, "model") and hasattr(causal_lm, "lm_head"):
        backbone_out = causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = backbone_out[0]
        shift_hidden = hidden_states[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return None
        valid_hidden = shift_hidden[valid_mask]
        valid_labels = shift_labels[valid_mask]
        valid_logits = causal_lm.lm_head(valid_hidden)
        return torch.nn.functional.cross_entropy(valid_logits.float(), valid_labels)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


def compute_masked_logp_and_nll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Computes length-normalized assistant log-prob and smoothed NLL on unmasked tokens only."""
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels != -100
    if not valid_mask.any():
        return None, None

    valid_labels = shift_labels[valid_mask]
    if hasattr(causal_lm, "model") and hasattr(causal_lm, "lm_head"):
        backbone_out = causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        shift_hidden = backbone_out[0][:, :-1, :].contiguous()
        valid_logits = causal_lm.lm_head(shift_hidden[valid_mask]).float()
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        valid_logits = shift_logits[valid_mask].float()

    log_probs = torch.nn.functional.log_softmax(valid_logits, dim=-1)
    token_logp = log_probs.gather(dim=-1, index=valid_labels.unsqueeze(-1)).squeeze(-1)
    mean_logp = token_logp.mean()
    nll = torch.nn.functional.cross_entropy(
        valid_logits,
        valid_labels,
        label_smoothing=max(0.0, float(label_smoothing)),
    )
    return mean_logp, nll


def _tokenize_masked_turn(
    tokenizer,
    prompt_msgs: List[Dict],
    assistant_content: str,
    max_length: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    full_msgs = list(prompt_msgs) + [{"role": "assistant", "content": assistant_content}]
    full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

    enc_full = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
    enc_prompt = tokenizer(prompt_text, truncation=True, max_length=max_length, return_tensors="pt")

    input_ids = enc_full.input_ids[0]
    labels = input_ids.clone()
    prompt_len = enc_prompt.input_ids.shape[1]
    labels[:prompt_len] = -100
    if prompt_len >= input_ids.shape[0] or not (labels != -100).any():
        return None
    return input_ids, labels


def _pad_1d_multiple_of_8(
    ids: torch.Tensor,
    lbls: torch.Tensor,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padded_ids = ids.unsqueeze(0)
    padded_lbls = lbls.unsqueeze(0)
    pad_rem = padded_ids.shape[1] % 8
    if pad_rem != 0:
        pad_amount = 8 - pad_rem
        padded_ids = torch.nn.functional.pad(padded_ids, (0, pad_amount), value=pad_id)
        padded_lbls = torch.nn.functional.pad(padded_lbls, (0, pad_amount), value=-100)
    mask = (padded_ids != pad_id).long()
    return padded_ids, padded_lbls, mask


def train_slm_qlora(
    epochs: int = 2,
    lr: float = 2e-4,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_length: int = 1280,
    is_best: bool = False,
    loop_idx: Optional[int] = None,
    max_trajectories: int = 512,
    fresh: bool = False,
    use_token_surgery: bool = False,
    dpo_beta: float = 0.1,
    nll_weight: float = 0.3,
    label_smoothing: float = 0.04,
) -> Dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

    trajectories = load_training_trajectories(max_trajectories=max_trajectories)
    dpo_pairs = load_dpo_pairs(max_pairs=max(64, max_trajectories // 2))
    if not trajectories and not dpo_pairs:
        print("[SLM Trainer] WARNING: No verified trajectories or DPO pairs found to train on. Skipping.")
        return {"status": "skipped", "reason": "no_trajectories"}

    print("=" * 74)
    print("  NEUROCRAFT SYSTEM 2 SLM HYBRID DPO+NLL QLoRA TRAINER ('MINECRAFT SAVANT')")
    print(f"  Device: {gpu_name} | SFT: {len(trajectories)} | DPO Pairs: {len(dpo_pairs)} | Epochs: {epochs}")
    print(f"  beta={dpo_beta} | nll_weight={nll_weight} | label_smoothing={label_smoothing} | lr={lr}")
    print("=" * 74)

    t0 = time.perf_counter()

    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model in BF16
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            attn_implementation="sdpa",
            local_files_only=True,
        )
    except Exception:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            attn_implementation="sdpa",
        )

    # 2b. Optional Tokenizer Surgery (subword BPE natively encodes action syntax with pretrained semantics)
    if use_token_surgery:
        num_added, _ = register_minecraft_tokens(tokenizer, base_model)
        if num_added > 0:
            print(f"[SLM Trainer] Registered {num_added} Minecraft special tokens. Vocabulary size: {len(tokenizer)}")
    else:
        print(f"[SLM Trainer] Using pristine subword BPE tokenizer ({len(tokenizer)} tokens) preserving pretrained representations.")

    # 3. Setup PEFT LoRA or resume from existing adapter
    latest_adapter = CHECKPOINT_DIR / "slm_minecraft_savant_latest"
    if (latest_adapter / "adapter_config.json").exists() and not fresh:
        print(f"[SLM Trainer] Resuming from existing adapter at {latest_adapter}")
        model = PeftModel.from_pretrained(base_model, str(latest_adapter), is_trainable=True)
    else:
        print(f"[SLM Trainer] Initializing v2 r=16 Full-MLP LoRA adapter (r=16, alpha=32) on base model...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    # Ensure all LoRA parameters strictly match base model compute dtype (BF16)
    model = model.to(dtype=dtype)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    covered_dpo_keys = {
        (p.get("taskId"), p.get("step"))
        for p in dpo_pairs
        if p.get("taskId") is not None and p.get("step") is not None
    }

    # 4. Tokenize Contrastive DPO Pairs + Unpaired SFT Winners with Target Masking
    dpo_tokenized: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    sft_tokenized: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
    dropped_too_long = 0

    for pair in dpo_pairs:
        try:
            prompt_msgs = pair.get("prompt_messages", [])
            chosen_tok = _tokenize_masked_turn(tokenizer, prompt_msgs, str(pair["chosen"]), max_length)
            rejected_tok = _tokenize_masked_turn(tokenizer, prompt_msgs, str(pair["rejected"]), max_length)
            if chosen_tok is None or rejected_tok is None:
                dropped_too_long += 1
                continue
            ch_ids, ch_lbls = chosen_tok
            rej_ids, rej_lbls = rejected_tok
            dpo_tokenized.append((ch_ids, ch_lbls, rej_ids, rej_lbls))
        except Exception:
            continue

    for traj in trajectories:
        t_key = (traj.get("taskId"), traj.get("step"))
        if t_key[0] is not None and t_key[1] is not None and t_key in covered_dpo_keys:
            continue
        msgs = traj.get("messages", [])
        if len(msgs) < 2:
            continue
        try:
            tok = _tokenize_masked_turn(
                tokenizer,
                msgs[:-1],
                str(msgs[-1].get("content", "")),
                max_length,
            )
            if tok is None:
                dropped_too_long += 1
                continue

            input_ids, labels = tok
            adv = float(traj.get("advantage", 0.0))
            # Weight loss by trajectory advantage (min 0.3, max 2.5)
            adv_weight = max(0.3, min(2.5, 1.0 + 0.5 * adv))
            sft_tokenized.append((input_ids, labels, adv_weight))
        except Exception:
            continue

    if not dpo_tokenized and not sft_tokenized:
        print("[SLM Trainer] Could not tokenize any valid DPO pairs or trajectory samples.")
        return {"status": "failed", "reason": "no_valid_tokenized_data"}

    print(
        f"[SLM Trainer] Prepared {len(dpo_tokenized)} contrastive DPO pairs + "
        f"{len(sft_tokenized)} unpaired SFT samples (dropped {dropped_too_long} exceeding max_length={max_length})."
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate_fn(batch):
        inps = [item[0] for item in batch]
        lbls = [item[1] for item in batch]
        weights = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        padded_inps = torch.nn.utils.rnn.pad_sequence(inps, batch_first=True, padding_value=pad_id)
        padded_lbls = torch.nn.utils.rnn.pad_sequence(lbls, batch_first=True, padding_value=-100)

        # Pad sequence length to nearest multiple of 8 for cuBLAS tensor core alignment
        seq_len = padded_inps.shape[1]
        pad_rem = seq_len % 8
        if pad_rem != 0:
            pad_amount = 8 - pad_rem
            padded_inps = torch.nn.functional.pad(padded_inps, (0, pad_amount), value=pad_id)
            padded_lbls = torch.nn.functional.pad(padded_lbls, (0, pad_amount), value=-100)

        mask = (padded_inps != pad_id).long()
        return padded_inps, padded_lbls, mask, weights

    import random

    actual_batch_size = max(1, min(batch_size, 8))
    sft_loader = (
        DataLoader(sft_tokenized, batch_size=actual_batch_size, shuffle=True, collate_fn=collate_fn)
        if sft_tokenized
        else None
    )

    initial_loss = 0.0
    final_loss = 0.0
    step_count = 0
    total_samples = len(dpo_tokenized) + len(sft_tokenized)
    batch_losses: list[float] = []

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        grad_pending = 0

        schedule_items: List[Tuple[str, object]] = [("dpo", p) for p in dpo_tokenized]
        if sft_loader is not None:
            for sft_batch in sft_loader:
                schedule_items.append(("sft", sft_batch))
        random.shuffle(schedule_items)

        for b_idx, (mode, payload) in enumerate(schedule_items):
            ch_ids = ch_lbls = ch_mask = None
            rej_ids = rej_lbls = rej_mask = None
            ref_w_logp = ref_l_logp = pi_w_logp = chosen_nll = pi_l_logp = None
            logits_diff = dpo_loss = None
            b_inps = b_lbls = b_mask = b_weights = None
            raw_loss = loss = None
            try:
                if mode == "dpo":
                    ch_ids_1d, ch_lbls_1d, rej_ids_1d, rej_lbls_1d = payload  # type: ignore[misc]
                    ch_ids, ch_lbls, ch_mask = _pad_1d_multiple_of_8(ch_ids_1d, ch_lbls_1d, pad_id)
                    rej_ids, rej_lbls, rej_mask = _pad_1d_multiple_of_8(rej_ids_1d, rej_lbls_1d, pad_id)
                    ch_ids = ch_ids.to(device)
                    ch_lbls = ch_lbls.to(device)
                    ch_mask = ch_mask.to(device)
                    rej_ids = rej_ids.to(device)
                    rej_lbls = rej_lbls.to(device)
                    rej_mask = rej_mask.to(device)

                    # Reference log-probs with adapter disabled (0 extra VRAM copy)
                    with torch.no_grad():
                        with model.disable_adapter():
                            ref_w_logp, _ = compute_masked_logp_and_nll(
                                model, ch_ids, ch_mask, ch_lbls, label_smoothing=0.0
                            )
                            ref_l_logp, _ = compute_masked_logp_and_nll(
                                model, rej_ids, rej_mask, rej_lbls, label_smoothing=0.0
                            )
                    if ref_w_logp is None or ref_l_logp is None:
                        del ch_ids, ch_lbls, ch_mask, rej_ids, rej_lbls, rej_mask, ref_w_logp, ref_l_logp
                        continue

                    pi_w_logp, chosen_nll = compute_masked_logp_and_nll(
                        model, ch_ids, ch_mask, ch_lbls, label_smoothing=label_smoothing
                    )
                    pi_l_logp, _ = compute_masked_logp_and_nll(
                        model, rej_ids, rej_mask, rej_lbls, label_smoothing=0.0
                    )
                    if pi_w_logp is None or chosen_nll is None or pi_l_logp is None:
                        del ch_ids, ch_lbls, ch_mask, rej_ids, rej_lbls, rej_mask
                        del ref_w_logp, ref_l_logp, pi_w_logp, chosen_nll, pi_l_logp
                        continue

                    logits_diff = (pi_w_logp - ref_w_logp) - (pi_l_logp - ref_l_logp)
                    dpo_loss = -torch.nn.functional.logsigmoid(dpo_beta * logits_diff)
                    raw_loss = dpo_loss + nll_weight * chosen_nll
                    if torch.isnan(raw_loss) or torch.isinf(raw_loss):
                        del ch_ids, ch_lbls, ch_mask, rej_ids, rej_lbls, rej_mask
                        del ref_w_logp, ref_l_logp, pi_w_logp, chosen_nll, pi_l_logp, logits_diff, dpo_loss, raw_loss
                        continue

                    b_loss = float(raw_loss.item())
                    loss = raw_loss / gradient_accumulation_steps
                    loss.backward()
                    del ch_ids, ch_lbls, ch_mask, rej_ids, rej_lbls, rej_mask
                    del ref_w_logp, ref_l_logp, pi_w_logp, chosen_nll, pi_l_logp, logits_diff, dpo_loss, raw_loss, loss
                else:
                    b_inps, b_lbls, b_mask, b_weights = payload  # type: ignore[misc]
                    b_inps = b_inps.to(device)
                    b_lbls = b_lbls.to(device)
                    b_mask = b_mask.to(device)
                    b_weights = b_weights.to(device)

                    # Unpaired winner fallback: smoothed NLL preserves valid JSON syntax
                    _, raw_loss = compute_masked_logp_and_nll(
                        model, b_inps, b_mask, b_lbls, label_smoothing=label_smoothing
                    )
                    if raw_loss is None or torch.isnan(raw_loss) or torch.isinf(raw_loss):
                        del b_inps, b_lbls, b_mask, b_weights, raw_loss
                        continue

                    b_loss = float(raw_loss.item())
                    # Scale loss by advantage weight and gradient accumulation
                    loss = (raw_loss * b_weights.mean()) / gradient_accumulation_steps
                    loss.backward()
                    del b_inps, b_lbls, b_mask, b_weights, raw_loss, loss
            except (RuntimeError, Exception) as e:
                err_name = type(e).__name__.lower()
                err_msg = str(e).lower()
                if "cublas" in err_msg or "out of memory" in err_msg or "acceleratorerror" in err_name:
                    print(f"  [SLM Trainer Warning] Caught transient CUDA/cuBLAS error during batch {b_idx}: {e}. Clearing cache and skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    grad_pending = 0
                    ch_ids = ch_lbls = ch_mask = rej_ids = rej_lbls = rej_mask = None
                    ref_w_logp = ref_l_logp = pi_w_logp = chosen_nll = pi_l_logp = None
                    logits_diff = dpo_loss = b_inps = b_lbls = b_mask = b_weights = raw_loss = loss = None
                    gc.collect()
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                else:
                    raise e

            epoch_loss += b_loss
            batches += 1
            grad_pending += 1
            step_count += 1
            batch_losses.append(b_loss)

            if grad_pending >= gradient_accumulation_steps or b_idx == len(schedule_items) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_pending = 0

        if grad_pending > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = epoch_loss / max(1, batches)
        window = max(2, min(len(batch_losses), max(4, len(batch_losses) // 4)))
        if epoch == 0 and len(batch_losses) > 0:
            initial_loss = sum(batch_losses[:window]) / window
        if len(batch_losses) > 0:
            final_loss = sum(batch_losses[-window:]) / window
        print(f"  [Epoch {epoch + 1}/{epochs}] Mean Loss: {avg_loss:.4f} (Window: {initial_loss:.4f} -> {final_loss:.4f}, {batches} steps)")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    train_duration = time.perf_counter() - t0
    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    )

    # 5. Save LoRA Adapters (Zero-Disk-Bloat Policy: latest + best + loop checkpoint)
    model.save_pretrained(str(latest_adapter))
    tokenizer.save_pretrained(str(latest_adapter))
    print(f"[SLM Trainer] Saved latest adapter to: {latest_adapter}")

    if is_best:
        best_adapter = CHECKPOINT_DIR / "slm_minecraft_savant_best"
        model.save_pretrained(str(best_adapter))
        tokenizer.save_pretrained(str(best_adapter))
        print(f"[SLM Trainer] NEW HIGH SCORE! Saved best adapter to: {best_adapter}")

    if loop_idx is not None and loop_idx < 5:
        loop_adapter = CHECKPOINT_DIR / f"slm_minecraft_savant_loop{loop_idx}"
        model.save_pretrained(str(loop_adapter))
        tokenizer.save_pretrained(str(loop_adapter))

    metrics = {
        "status": "success",
        "device": gpu_name,
        "epochs": epochs,
        "samplesTrained": total_samples,
        "sftSamplesTrained": len(sft_tokenized),
        "dpoPairsTrained": len(dpo_tokenized),
        "dpoBeta": dpo_beta,
        "nllWeight": nll_weight,
        "labelSmoothing": label_smoothing,
        "initialLoss": round(initial_loss, 4),
        "finalLoss": round(final_loss, 4),
        "peakVramMb": round(peak_vram, 1),
        "trainingDurationSeconds": round(train_duration, 2),
        "savedAdapter": str(latest_adapter),
        "isBest": is_best,
        "loopIdx": loop_idx,
    }

    with open(SLM_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroCraft System 2 Cognitive SLM Hybrid DPO+NLL QLoRA Trainer")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1280)
    parser.add_argument("--max-trajectories", type=int, default=512)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--nll-weight", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--is-best", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Start training from pristine LoRA instead of resuming")
    parser.add_argument("--use-token-surgery", action="store_true", help="Perform tokenizer surgery instead of subword BPE")
    parser.add_argument("--loop", type=int, default=None)
    args = parser.parse_args()

    train_slm_qlora(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        is_best=args.is_best,
        loop_idx=args.loop,
        max_trajectories=args.max_trajectories,
        fresh=args.fresh,
        use_token_surgery=args.use_token_surgery,
        dpo_beta=args.dpo_beta,
        nll_weight=args.nll_weight,
        label_smoothing=args.label_smoothing,
    )
