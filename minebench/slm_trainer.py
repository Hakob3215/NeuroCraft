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
SLM_METRICS_PATH = RESULTS_DIR / "slm_training_metrics.json"

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def load_training_trajectories(max_trajectories: int = 512) -> List[Dict]:
    """Loads high-advantage trajectories from Omniscience, VLA, and GRPO verified datasets."""
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
    random.shuffle(gold_trajectories)
    random.shuffle(grpo_trajectories)

    # Prioritize real game experience, then fill remaining quota with diverse gold dataset
    remaining_quota = max(0, max_trajectories - len(grpo_trajectories))
    trajectories = grpo_trajectories + gold_trajectories[:remaining_quota]
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

    print(f"[SLM Trainer] Active balanced training dataset: {len(trajectories)} samples ({len(grpo_trajectories)} real experience, {len(trajectories) - len(grpo_trajectories)} gold demonstrations).")
    return trajectories


def train_slm_qlora(
    epochs: int = 2,
    lr: float = 2e-4,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_length: int = 512,
    is_best: bool = False,
    loop_idx: Optional[int] = None,
    max_trajectories: int = 512,
    fresh: bool = False,
    use_token_surgery: bool = False,
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
    if not trajectories:
        print("[SLM Trainer] WARNING: No verified trajectories found to train on. Skipping.")
        return {"status": "skipped", "reason": "no_trajectories"}

    print("=" * 74)
    print("  NEUROCRAFT SYSTEM 2 SLM QLoRA TRAINER ('MINECRAFT SAVANT')")
    print(f"  Device: {gpu_name} | Dataset: {len(trajectories)} trajectories | Epochs: {epochs}")
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
        print(f"[SLM Trainer] Initializing pristine LoRA adapter (r=8, alpha=16) on base model...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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

    # 4. Tokenize Trajectories with Target Masking
    tokenized_data: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
    for traj in trajectories:
        msgs = traj.get("messages", [])
        if len(msgs) < 2:
            continue
        try:
            full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
            prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)

            enc_full = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
            enc_prompt = tokenizer(prompt_text, truncation=True, max_length=max_length, return_tensors="pt")

            input_ids = enc_full.input_ids[0]
            labels = input_ids.clone()
            prompt_len = enc_prompt.input_ids.shape[1]

            # Mask prompt tokens with -100 so loss is strictly computed on assistant actions
            labels[:prompt_len] = -100
            if prompt_len >= input_ids.shape[0] or not (labels != -100).any():
                continue

            adv = float(traj.get("advantage", 0.0))
            # Weight loss by trajectory advantage (min 0.3, max 2.5)
            adv_weight = max(0.3, min(2.5, 1.0 + 0.5 * adv))

            tokenized_data.append((input_ids, labels, adv_weight))
        except Exception:
            continue

    if not tokenized_data:
        print("[SLM Trainer] Could not tokenize any valid trajectory samples.")
        return {"status": "failed", "reason": "no_valid_tokenized_data"}

    print(f"[SLM Trainer] Prepared {len(tokenized_data)} tokenized samples for training.")

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

    actual_batch_size = max(1, min(batch_size, 8))
    loader = DataLoader(tokenized_data, batch_size=actual_batch_size, shuffle=True, collate_fn=collate_fn)

    initial_loss = 0.0
    final_loss = 0.0
    step_count = 0
    total_samples = len(tokenized_data)
    batch_losses: list[float] = []

    optimizer.zero_grad()

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0

        for b_inps, b_lbls, b_mask, b_weights in loader:
            b_inps = b_inps.to(device)
            b_lbls = b_lbls.to(device)
            b_mask = b_mask.to(device)
            b_weights = b_weights.to(device)

            outputs = model(input_ids=b_inps, attention_mask=b_mask, labels=b_lbls)
            raw_loss = outputs.loss
            if raw_loss is None or torch.isnan(raw_loss):
                continue

            # Scale loss by advantage weight and gradient accumulation
            loss = (raw_loss * b_weights.mean()) / gradient_accumulation_steps
            try:
                loss.backward()
            except RuntimeError as e:
                if "cublas" in str(e).lower() or "out of memory" in str(e).lower():
                    print(f"  [SLM Trainer Warning] Caught transient CUDA/cuBLAS error during backward: {e}. Clearing cache and skipping batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e

            b_loss = float(raw_loss.item())
            epoch_loss += b_loss
            batches += 1
            step_count += 1
            batch_losses.append(b_loss)

            if batches % gradient_accumulation_steps == 0 or batches == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, batches)
        window = max(2, min(len(batch_losses), max(4, len(batch_losses) // 4)))
        if epoch == 0 and len(batch_losses) > 0:
            initial_loss = sum(batch_losses[:window]) / window
        if len(batch_losses) > 0:
            final_loss = sum(batch_losses[-window:]) / window
        print(f"  [Epoch {epoch + 1}/{epochs}] Mean Loss: {avg_loss:.4f} (Window: {initial_loss:.4f} -> {final_loss:.4f}, {batches} batches)")

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
    parser = argparse.ArgumentParser(description="NeuroCraft System 2 Cognitive SLM QLoRA Trainer")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-trajectories", type=int, default=512)
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
    )
