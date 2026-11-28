"""
NEUROCRAFT RTX 5060 LOCAL NEURAL DAEMON
Hosts the entire All-Neural Stack in VRAM on the NVIDIA GeForce RTX 5060 (Blackwell sm_120):
  1. System 1 Motor Cortex: VoxelReflexPolicy (~2.0 ms latency, 20Hz sub-tick reflexes)
  2. System 1 3D Vision:    VolumetricChunkAnalyzer (~4.8 ms latency, 64x64x64 chunk scan)
  3. System 1 Visuomotor:   VisuomotorPolicy (~8.3 ms latency, 120 FPS visual control)
  4. System 2 Local Brain:  Qwen/Qwen2.5-Coder-1.5B-Instruct (BF16 Tensor Core SLM)

Exposes a zero-overhead HTTP/JSON-RPC bridge on http://127.0.0.1:8765 for the Node.js Bot.
"""

import os
import sys
import gc
import time
import json
import re
import asyncio
import argparse
from typing import Dict, Any

import torch
from aiohttp import web
from transformers import StoppingCriteria, StoppingCriteriaList

from models import (
    VoxelReflexPolicy,
    VolumetricChunkAnalyzer,
    VisuomotorPolicy,
    REFLEX_ACTION_NAMES,
    block_name_to_id,
)
from mc_tokens import register_minecraft_tokens, parse_vla_response, ALL_MINECRAFT_SPECIAL_TOKENS
from action_head import MinecraftActionHead

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


STANDALONE_ACTIONS = {
    "shield_block", "swim_surface", "bridge_sneak", "parkour_jump",
    "pillar_up", "dig_staircase", "shear_sheep", "harvest_crops"
}
ACTIONS_REQUIRING_SLOT = {"equip"}
ACTIONS_REQUIRING_QTY = {"drop_item", "give_to_player", "deposit_items", "withdraw_items", "craft", "smelt"}
TERMINATING_PARAM_PREFIXES = ("<|qty:", "<|slot:", "<|target:", "<|block:")


class FastTokenStoppingCriteria(StoppingCriteria):
    """
    Ultra-low-overhead token-level stopping criteria.
    - Zero full-sequence string decoding inside Python GIL (O(1) per token).
    - Caches token-to-string mappings in an O(1) dict.
    - Directly tracks JSON brace depth outside quotes with escape handling.
    - Halts immediately the millisecond the top-level JSON closes (depth == 0 after seeing '{').
    - Halts immediately on EOS tokens (<|im_end|>, <|endoftext|>).
    - Halts immediately on complete discrete VLA action trajectories (standalone actions & terminating params).
    - Prevents premature cutoff on chat dialogues, companion speech, or long tool synthesis codes.
    - Latching state ensures consistent stopping across subsequent calls.
    """
    _global_token_str_cache: Dict[int, str] = {}

    def __init__(self, tokenizer: Any, prompt_len: int, stop_on_json: bool = True, stop_on_action: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.last_checked_len = prompt_len
        self.stop_on_json = stop_on_json
        self.stop_on_action = stop_on_action

        # JSON state tracking
        self.depth = 0
        self.seen_first_brace = False
        self.in_str = False
        self.escape = False

        # Action tracking state
        self.action_name = None
        self.action_state = "NONE"  # "NONE" -> "SEEN_ACT" -> "SEEN_ITEM" -> "DONE"
        self.action_tail = ""
        self.is_done = False

        # EOS token IDs
        self.eos_token_ids = set()
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                self.eos_token_ids.update(tokenizer.eos_token_id)
            else:
                self.eos_token_ids.add(tokenizer.eos_token_id)
        # Standard Qwen / ChatML EOS tokens
        self.eos_token_ids.update([151643, 151645])

    def _get_token_str(self, token_id: int) -> str:
        s = self._global_token_str_cache.get(token_id)
        if s is None:
            s = self.tokenizer.decode([token_id], skip_special_tokens=False)
            self._global_token_str_cache[token_id] = s
        return s

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor = None, **kwargs) -> bool:
        if self.is_done:
            return True

        seq_len = input_ids.shape[1]
        if seq_len <= self.prompt_len:
            return False

        start_idx = max(self.last_checked_len, self.prompt_len)
        self.last_checked_len = seq_len

        for idx in range(start_idx, seq_len):
            tok_id = input_ids[0, idx].item()

            # 1. Fast EOS check by token ID
            if tok_id in self.eos_token_ids:
                self.is_done = True
                return True

            piece = self._get_token_str(tok_id)

            # 2. Check for EOS string markers
            if "<|im_end|>" in piece or "<|endoftext|>" in piece:
                self.is_done = True
                return True

            # 3. VLA Action Token Tracking
            if self.stop_on_action:
                self.action_tail = (self.action_tail + piece)[-160:]
                clean_tail = self.action_tail.strip()

                last_open = self.action_tail.rfind("<|")
                last_close = self.action_tail.rfind("|>")
                in_tag = last_open > last_close

                if self.action_state == "NONE":
                    if "<|act:" in self.action_tail:
                        m = re.search(r"<\|act:([a-zA-Z0-9_]+)\|>", self.action_tail)
                        if m:
                            self.action_name = m.group(1)
                            if self.action_name in STANDALONE_ACTIONS and clean_tail.endswith("|>"):
                                self.is_done = True
                                return True
                            self.action_state = "SEEN_ACT"

                elif self.action_state == "SEEN_ACT":
                    if any(p in self.action_tail for p in TERMINATING_PARAM_PREFIXES) and clean_tail.endswith("|>"):
                        self.is_done = True
                        return True
                    if "<|item:" in self.action_tail and clean_tail.endswith("|>"):
                        if self.action_name not in ACTIONS_REQUIRING_SLOT and self.action_name not in ACTIONS_REQUIRING_QTY:
                            self.is_done = True
                            return True
                        self.action_state = "SEEN_ITEM"

                elif self.action_state == "SEEN_ITEM":
                    if any(p in self.action_tail for p in TERMINATING_PARAM_PREFIXES) and clean_tail.endswith("|>"):
                        self.is_done = True
                        return True
                    if not in_tag and piece.strip():
                        self.is_done = True
                        return True

            # 4. JSON brace depth tracking
            if self.stop_on_json:
                for ch in piece:
                    if self.escape:
                        self.escape = False
                        continue
                    if ch == "\\":
                        if self.in_str:
                            self.escape = True
                        continue
                    if ch == '"':
                        self.in_str = not self.in_str
                        continue
                    if not self.in_str:
                        if ch == '{':
                            self.seen_first_brace = True
                            self.depth += 1
                        elif ch == '}':
                            if self.seen_first_brace:
                                self.depth -= 1
                                if self.depth == 0:
                                    self.is_done = True
                                    return True

        return False


JsonBraceStoppingCriteria = FastTokenStoppingCriteria


class NeuralStackServer:
    def __init__(self, slm_model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct", load_slm: bool = True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.slm_model_id = slm_model_id
        self.load_slm_flag = load_slm

        self.reflex_policy: VoxelReflexPolicy | None = None
        self.chunk_analyzer: VolumetricChunkAnalyzer | None = None
        self.visuomotor_policy: VisuomotorPolicy | None = None
        self.action_head: MinecraftActionHead | None = None
        self.dqn_opponent: Any = None
        self.tokenizer: Any = None
        self.slm_model: Any = None
        self.active_adapter: str | None = None
        self.llm_lock: asyncio.Lock | None = None
        self.bot_latents: Dict[str, torch.Tensor] = {}
        self.latest_latent_h: torch.Tensor = torch.zeros(1, 1536, dtype=torch.float32, device=self.device)

        self.telemetry = {
            "reflex_calls": 0,
            "reflex_avg_ms": 0.0,
            "chunk_calls": 0,
            "chunk_avg_ms": 0.0,
            "llm_calls": 0,
            "llm_avg_ms": 0.0,
        }

    def initialize_models(self):
        print("=" * 74)
        print("  NEUROCRAFT RTX 5060 NEURAL DAEMON INITIALIZING")
        print("=" * 74)
        if torch.cuda.is_available():
            print(f"[GPU] Device: {torch.cuda.get_device_name(0)} (Blackwell sm_120)")
        else:
            print("[GPU] WARNING: Running on CPU fallback")

        # 1. Load VoxelReflexPolicy
        self.reflex_policy = VoxelReflexPolicy().to(self.device)
        reflex_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "voxel_reflex_policy.pt")
        if os.path.exists(reflex_ckpt):
            self.reflex_policy.load_state_dict(torch.load(reflex_ckpt, map_location=self.device, weights_only=True))
            print(f"[System 1] Loaded trained VoxelReflexPolicy from {reflex_ckpt}")
        self.reflex_policy.eval()

        # 2. Load VolumetricChunkAnalyzer
        self.chunk_analyzer = VolumetricChunkAnalyzer().to(self.device)
        chunk_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "chunk_analyzer.pt")
        if os.path.exists(chunk_ckpt):
            self.chunk_analyzer.load_state_dict(torch.load(chunk_ckpt, map_location=self.device, weights_only=True))
            print(f"[System 1] Loaded trained VolumetricChunkAnalyzer from {chunk_ckpt}")
        self.chunk_analyzer.eval()

        # 3. Load VisuomotorPolicy
        self.visuomotor_policy = VisuomotorPolicy().to(self.device)
        vis_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "visuomotor_policy.pt")
        if os.path.exists(vis_ckpt):
            self.visuomotor_policy.load_state_dict(torch.load(vis_ckpt, map_location=self.device, weights_only=True))
            print(f"[System 1] Loaded trained VisuomotorPolicy from {vis_ckpt}")
        self.visuomotor_policy.eval()

        # 4. Load Local SLM (Qwen2.5-Coder) onto RTX 5060 in BF16
        if self.load_slm_flag:
            print(f"[System 2] Loading Local SLM '{self.slm_model_id}' onto {self.device} (BF16)...")
            t0 = time.perf_counter()
            from transformers import AutoModelForCausalLM, AutoTokenizer

            target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.slm_model_id, local_files_only=True)
                self.slm_model = AutoModelForCausalLM.from_pretrained(
                    self.slm_model_id,
                    torch_dtype=target_dtype,
                    attn_implementation="sdpa",
                    local_files_only=True,
                ).to(self.device)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.slm_model_id)
                self.slm_model = AutoModelForCausalLM.from_pretrained(
                    self.slm_model_id,
                    torch_dtype=target_dtype,
                    attn_implementation="sdpa",
                ).to(self.device)
            # 4b. Check if adapter requires tokenizer surgery or uses pristine subword BPE
            adapter_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "slm_minecraft_savant_latest")
            adapter_vocab_resized = False
            tok_cfg_path = os.path.join(adapter_dir, "tokenizer_config.json")
            if os.path.exists(tok_cfg_path):
                try:
                    with open(tok_cfg_path, "r", encoding="utf-8") as f:
                        cfg_data = json.load(f)
                    if cfg_data.get("added_tokens_decoder") and len(cfg_data.get("added_tokens_decoder")) > 50:
                        adapter_vocab_resized = True
                except Exception:
                    pass

            if adapter_vocab_resized:
                added, _ = register_minecraft_tokens(self.tokenizer, self.slm_model)
                if added > 0:
                    print(f"[System 2 VLA] Registered {added} native Minecraft action tokens into vocabulary.")
            else:
                print(f"[System 2 VLA] Preserving pristine subword BPE vocabulary ({len(self.tokenizer)} tokens).")

            # 4c. Check for fine-tuned 'Minecraft Savant' LoRA Adapter
            adapter_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "slm_minecraft_savant_latest")
            if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
                from peft import PeftModel
                self.slm_model = PeftModel.from_pretrained(self.slm_model, adapter_dir)
                try:
                    self.slm_model.merge_adapter()
                    print(f"[System 2] Merged 'Minecraft Savant' LoRA weights reversibly into PeftModel (32.5+ tok/s)")
                except Exception as fe:
                    print(f"[System 2] Warning: Could not merge LoRA weights at startup: {fe}")
                self.slm_model = self.slm_model.to(dtype=target_dtype).to(self.device)
                self.active_adapter = adapter_dir
                print(f"[System 2] Attached fine-tuned 'Minecraft Savant' LoRA adapter from {adapter_dir}")

            self.slm_model.eval()
            param_dev = next(self.slm_model.parameters()).device
            print(f"[System 2] SLM ready on {param_dev} (Active VRAM: {torch.cuda.memory_allocated() / (1024*1024):.1f} MiB)")

            # 4d. Initialize / Load Neural Action Head
            act_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.action_head = MinecraftActionHead().to(device=self.device, dtype=act_dtype)
            head_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "action_head_latest.pt")
            if os.path.exists(head_ckpt):
                try:
                    self.action_head.load_state_dict(torch.load(head_ckpt, map_location=self.device, weights_only=True), strict=False)
                    self.action_head.to(device=self.device, dtype=act_dtype)
                    print(f"[System 2 VLA] Loaded trained MinecraftActionHead (211M Multi-Modal ACT) from {head_ckpt}")
                except Exception as he:
                    print(f"[System 2 VLA] Incompatible legacy ActionHead checkpoint ({he}); starting fresh 211M Multi-Modal ACT.")
            else:
                print(f"[System 2 VLA] Initialized pristine MinecraftActionHead from scratch (211M Multi-Modal Spatial Action Transformer)")
            self.action_head.eval()

            # Warmup CUDA kernels so first chat message has zero JIT lag
            dummy = self.tokenizer(["{}"], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                self.slm_model.generate(**dummy, max_new_tokens=2, pad_token_id=self.tokenizer.eos_token_id)

            # Pre-warm common JSON & syntax tokens into FastTokenStoppingCriteria cache
            common_tokens = ["{", "}", '{"', '{\n', '}\n', '}\n\n', '"', '":', '": "', '",', ':', ',', ' ', '\n',
                             "true", "false", "null", "<|im_end|>", "<|im_start|>", "<|endoftext|>"]
            for s in common_tokens:
                try:
                    for tid in self.tokenizer.encode(s, add_special_tokens=False):
                        if tid not in FastTokenStoppingCriteria._global_token_str_cache:
                            FastTokenStoppingCriteria._global_token_str_cache[tid] = self.tokenizer.decode([tid], skip_special_tokens=False)
                except Exception:
                    pass

            for s in ALL_MINECRAFT_SPECIAL_TOKENS:
                try:
                    for tid in self.tokenizer.encode(s, add_special_tokens=False):
                        if tid not in FastTokenStoppingCriteria._global_token_str_cache:
                            FastTokenStoppingCriteria._global_token_str_cache[tid] = self.tokenizer.decode([tid], skip_special_tokens=False)
                except Exception:
                    pass

            dt = time.perf_counter() - t0
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            adapter_msg = f" (LoRA Adapter: {os.path.basename(self.active_adapter)})" if self.active_adapter else ""
            print(f"[System 2] Local SLM loaded & warmed up in {dt:.2f}s!{adapter_msg} Total VRAM resident: {vram_mb:.1f} MiB")

        # 5. Load External Sparring Partner Model (GeorgeDudau/Minecraft-AI-Agent Dueling DQN) on CPU (0 VRAM)
        ext_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external_agents", "Minecraft-AI-Agent"))
        dqn_ckpt = os.path.join(ext_dir, "Previous AI models", "Model 1 (0.13).pt")
        if os.path.exists(os.path.join(ext_dir, "dqn.py")) and os.path.exists(dqn_ckpt):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("ext_dqn", os.path.join(ext_dir, "dqn.py"))
                ext_dqn = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ext_dqn)
                self.dqn_opponent = ext_dqn.DQN(55, 7, duellingDQN=True).to("cpu")
                self.dqn_opponent.load_state_dict(torch.load(dqn_ckpt, map_location="cpu", weights_only=True))
                self.dqn_opponent.eval()
                print(f"[Sparring AI] Loaded external Dueling DQN opponent (GeorgeDudau/Minecraft-AI-Agent) on CPU (0 MiB VRAM)")
            except Exception as de:
                print(f"[Sparring AI] Warning: Could not load external DQN opponent: {de}")

        print("=" * 74)

    async def handle_health(self, request: web.Request) -> web.Response:
        vram_alloc = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        free_b, total_b = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        slm_dev = str(next(self.slm_model.parameters()).device) if self.slm_model is not None else "none"
        return web.json_response({
            "status": "ok",
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "slm_device": slm_dev,
            "slm_model": self.slm_model_id if self.slm_model is not None else None,
            "active_adapter": self.active_adapter,
            "vram_allocated_mb": round(vram_alloc, 1),
            "vram_free_mb": round(free_b / (1024 * 1024), 1),
            "telemetry": self.telemetry,
        })

    async def handle_reflex(self, request: web.Request) -> web.Response:
        t0 = time.perf_counter()
        data = await request.json()

        # Construct or parse 15x15x15 voxel tensor
        if "voxels" in data and isinstance(data["voxels"], list):
            v_list = data["voxels"]
            if len(v_list) == 3375:
                voxels = torch.tensor(v_list, dtype=torch.long, device=self.device).view(1, 15, 15, 15).clamp(0, 63)
            elif len(v_list) == 4096:
                vox_t = torch.tensor(v_list, dtype=torch.long, device=self.device).view(16, 16, 16)
                voxels = vox_t[:15, :15, :15].unsqueeze(0).clamp(0, 63)
            else:
                voxels = torch.zeros((1, 15, 15, 15), dtype=torch.long, device=self.device)
        else:
            voxels = torch.zeros((1, 15, 15, 15), dtype=torch.long, device=self.device)
            voxels[:, :, 0:7, :] = 6
            # Populate sparse blocks if provided
            for b in data.get("sparse_blocks", []):
                dx = int(b.get("dx", 0)) + 7
                dy = int(b.get("dy", 0)) + 7
                dz = int(b.get("dz", 0)) + 7
                if 0 <= dx < 15 and 0 <= dy < 15 and 0 <= dz < 15:
                    voxels[0, dx, dy, dz] = block_name_to_id(b.get("name", "stone"))

        # Construct 16x12 entity tensor
        entities = torch.zeros((1, 16, 12), dtype=torch.float32, device=self.device)
        raw_ents = data.get("entities", [])
        for idx, e in enumerate(raw_ents[:16]):
            entities[0, idx, 0] = float(e.get("rx", 0.0))
            entities[0, idx, 1] = float(e.get("ry", 0.0))
            entities[0, idx, 2] = float(e.get("rz", 0.0))
            entities[0, idx, 3] = float(e.get("vx", 0.0))
            entities[0, idx, 4] = float(e.get("vy", 0.0))
            entities[0, idx, 5] = float(e.get("vz", 0.0))
            entities[0, idx, 6] = float(e.get("distance", 10.0))
            entities[0, idx, 7] = 1.0 if e.get("isHostile", False) else 0.0
            entities[0, idx, 8] = 1.0 if "creeper" in str(e.get("name", "")).lower() else 0.0
            entities[0, idx, 9] = 1.0 if ("skeleton" in str(e.get("name", "")).lower() or "arrow" in str(e.get("name", "")).lower()) else 0.0
            entities[0, idx, 10] = float(e.get("healthNorm", 1.0))
            entities[0, idx, 11] = 1.0

        # Construct 1x10 bot state tensor
        bs = data.get("bot_state", {})
        bot_tensor = torch.tensor([[
            float(bs.get("health", 20.0)) / 20.0,
            float(bs.get("food", 20.0)) / 20.0,
            float(bs.get("oxygen", 20.0)) / 20.0,
            1.0 if bs.get("onGround", True) else 0.0,
            1.0 if bs.get("isInWater", False) else 0.0,
            1.0 if bs.get("isInLava", False) else 0.0,
            1.0 if bs.get("hasShield", True) else 0.0,
            float(bs.get("weaponCooldown", 1.0)),
            float(bs.get("pitch", 0.0)) / 1.57,
            float(bs.get("yaw", 0.0)) / 3.14,
        ]], dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            aim_delta, logits, urgency = self.reflex_policy(voxels, entities, bot_tensor)
            probs = torch.sigmoid(logits)[0].tolist()
            dyaw, dpitch = aim_delta[0].tolist()
            urg_val = float(urgency[0, 0].item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.telemetry["reflex_calls"] += 1
        n = self.telemetry["reflex_calls"]
        self.telemetry["reflex_avg_ms"] = round(((self.telemetry["reflex_avg_ms"] * (n - 1)) + dt_ms) / n, 2)

        actions_dict = {name: (probs[i] > 0.5) for i, name in enumerate(REFLEX_ACTION_NAMES)}
        action_probs = {name: round(probs[i], 3) for i, name in enumerate(REFLEX_ACTION_NAMES)}

        return web.json_response({
            "aim_delta": {"dyaw": round(dyaw, 4), "dpitch": round(dpitch, 4)},
            "actions": actions_dict,
            "action_probs": action_probs,
            "hazard_urgency": round(urg_val, 4),
            "latency_ms": round(dt_ms, 2),
        })

    async def handle_chunk_scan(self, request: web.Request) -> web.Response:
        t0 = time.perf_counter()
        data = await request.json()

        chunk = torch.zeros((1, 64, 64, 64), dtype=torch.long, device=self.device)
        chunk[:, :, 0:28, :] = 1
        chunk[:, :, 28:31, :] = 5
        chunk[:, :, 31, :] = 6

        for b in data.get("sparse_blocks", []):
            dx = int(b.get("dx", 0)) + 32
            dy = int(b.get("dy", 0)) + 32
            dz = int(b.get("dz", 0)) + 32
            if 0 <= dx < 64 and 0 <= dy < 64 and 0 <= dz < 64:
                chunk[0, dx, dy, dz] = block_name_to_id(b.get("name", "stone"))

        with torch.inference_mode():
            out = self.chunk_analyzer(chunk)[0].tolist()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.telemetry["chunk_calls"] += 1
        n = self.telemetry["chunk_calls"]
        self.telemetry["chunk_avg_ms"] = round(((self.telemetry["chunk_avg_ms"] * (n - 1)) + dt_ms) / n, 2)

        return web.json_response({
            "recommended_build_offset": {
                "dx": round(out[0], 1),
                "dy": round(out[1], 1),
                "dz": round(out[2], 1),
            },
            "flatness_confidence": round(out[3], 3),
            "recommended_ore_offset": {
                "dx": round(out[4], 1),
                "dy": round(out[5], 1),
                "dz": round(out[6], 1),
            },
            "ore_richness_score": round(out[7], 3),
            "cavern_hazard_score": round(out[8], 3),
            "latency_ms": round(dt_ms, 2),
        })

    def _run_slm_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        repetition_penalty: float = 1.12,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        bot_id: str = "default",
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_device = next(self.slm_model.parameters()).device
        # Self-healing: if daemon is configured for CUDA but SLM remained on CPU, restore to CUDA immediately
        if self.device.type == "cuda" and model_device.type != "cuda":
            print("[System 2] Self-healing: Restoring SLM from CPU to CUDA before inference...", flush=True)
            target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.slm_model = self.slm_model.to(dtype=target_dtype).to(self.device)
            self.slm_model.eval()
            model_device = next(self.slm_model.parameters()).device

        model_inputs = self.tokenizer([text], return_tensors="pt").to(model_device)
        input_len = model_inputs.input_ids.shape[1]

        stop_criteria = StoppingCriteriaList([
            FastTokenStoppingCriteria(self.tokenizer, input_len, stop_on_json=True, stop_on_action=True)
        ])

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": [self.tokenizer.eos_token_id, 151643, 151645],
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": 8,
            "return_dict_in_generate": True,
            "output_hidden_states": False,
            "logits_to_keep": 1,
            "stopping_criteria": stop_criteria,
        }
        if do_sample and temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(0.01, float(temperature))
            gen_kwargs["top_p"] = min(1.0, max(0.01, float(top_p)))
        else:
            gen_kwargs["do_sample"] = False

        causal_lm = self.slm_model.get_base_model() if hasattr(self.slm_model, "get_base_model") else self.slm_model
        norm_mod = getattr(getattr(causal_lm, "model", None), "norm", None)
        captured_h: list = [None]
        hook_handle = None
        if norm_mod is not None:
            def _capture_last_norm(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, torch.Tensor) and t.ndim == 3:
                    captured_h[0] = t[:, -1, :].detach()
            hook_handle = norm_mod.register_forward_hook(_capture_last_norm)

        gen_outputs = None
        try:
            with torch.inference_mode():
                gen_outputs = self.slm_model.generate(
                    **model_inputs,
                    **gen_kwargs,
                )
        except (RuntimeError, Exception) as e:
            if "out of memory" in str(e).lower() or "acceleratorerror" in type(e).__name__.lower():
                del model_inputs, gen_outputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        new_tokens = gen_outputs.sequences[0][input_len:]
        num_tokens = len(new_tokens)

        if captured_h[0] is not None:
            latent_h = captured_h[0].to(dtype=torch.float32, device=self.device)
            self.latest_latent_h = latent_h
            self.bot_latents[bot_id] = latent_h

        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        response_text = (
            response_text.replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )
        del model_inputs, gen_outputs
        if input_len > 1500 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return response_text, num_tokens, input_len

    async def handle_llm_chat(self, request: web.Request) -> web.Response:
        if self.slm_model is None or self.tokenizer is None:
            return web.json_response({"error": "SLM not loaded"}, status=503)

        t_enqueue = time.perf_counter()
        data = await request.json()
        system_prompt = data.get("system_prompt", "You are a helpful Minecraft AI assistant. Respond with valid JSON.")
        user_prompt = data.get("user_prompt", "")
        max_new_tokens = int(data.get("max_new_tokens", 160))
        repetition_penalty = float(data.get("repetition_penalty", 1.12))
        bot_id = str(data.get("bot_id", "default"))

        default_temp = float(os.environ.get("SLM_TEMPERATURE", "0.0"))
        default_top_p = float(os.environ.get("SLM_TOP_P", "1.0"))
        default_sample = os.environ.get("SLM_DO_SAMPLE", "false").lower() in ("true", "1") or default_temp > 0.0

        temperature = float(data.get("temperature", default_temp))
        top_p = float(data.get("top_p", default_top_p))
        do_sample = bool(data.get("do_sample", default_sample or temperature > 0.0))

        if self.llm_lock is None:
            self.llm_lock = asyncio.Lock()

        async with self.llm_lock:
            t0 = time.perf_counter()
            wait_ms = (t0 - t_enqueue) * 1000.0
            response_text, num_tokens, input_len = await asyncio.to_thread(
                self._run_slm_sync, system_prompt, user_prompt, max_new_tokens, repetition_penalty, temperature, top_p, do_sample, bot_id
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0

        tok_per_sec = (num_tokens / (dt_ms / 1000.0)) if dt_ms > 0 else 0.0

        self.telemetry["llm_calls"] += 1
        n = self.telemetry["llm_calls"]
        self.telemetry["llm_avg_ms"] = round(((self.telemetry["llm_avg_ms"] * (n - 1)) + dt_ms) / n, 2)

        model_device = next(self.slm_model.parameters()).device
        print(
            f"[System 2 {model_device.type.upper()} SLM] Prompt={input_len} tok -> Generated={num_tokens} tok in {dt_ms:.0f}ms ({tok_per_sec:.1f} tok/s, queue_wait={wait_ms:.0f}ms)",
            flush=True,
        )

        return web.json_response({
            "content": response_text,
            "tokens_generated": num_tokens,
            "latency_ms": round(dt_ms, 1),
            "queue_wait_ms": round(wait_ms, 1),
            "tokens_per_sec": round(tok_per_sec, 1),
        })

    async def handle_vla_predict(self, request: web.Request) -> web.Response:
        """Dual-Channel VLA inference: language speech + continuous 20Hz motor commands."""
        if self.slm_model is None or self.tokenizer is None or self.action_head is None:
            return web.json_response({"error": "VLA components not fully loaded"}, status=503)

        t0 = time.perf_counter()
        data = await request.json()
        system_prompt = data.get(
            "system_prompt",
            "You are NeuroAgent, an intelligent Minecraft 1.20.4 companion. Use native action tokens for physical tasks and speak naturally."
        )
        user_prompt = data.get("user_prompt", "")
        proprio_list = data.get("proprioception", [0.0] * 24)
        max_new_tokens = int(data.get("max_new_tokens", 48))

        default_temp = float(os.environ.get("SLM_TEMPERATURE", "0.0"))
        default_top_p = float(os.environ.get("SLM_TOP_P", "1.0"))
        default_sample = os.environ.get("SLM_DO_SAMPLE", "false").lower() in ("true", "1") or default_temp > 0.0

        temperature = float(data.get("temperature", default_temp))
        top_p = float(data.get("top_p", default_top_p))
        do_sample = bool(data.get("do_sample", default_sample or temperature > 0.0))

        if self.llm_lock is None:
            self.llm_lock = asyncio.Lock()

        async with self.llm_lock:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_device = next(self.slm_model.parameters()).device
            if self.device.type == "cuda" and model_device.type != "cuda":
                print("[System 2 VLA] Self-healing: Restoring SLM from CPU to CUDA before inference...", flush=True)
                target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                self.slm_model = self.slm_model.to(dtype=target_dtype).to(self.device)
                self.slm_model.eval()
                model_device = next(self.slm_model.parameters()).device

            model_inputs = self.tokenizer([text], return_tensors="pt").to(model_device)
            input_len = model_inputs.input_ids.shape[1]
            stop_criteria = StoppingCriteriaList([
                FastTokenStoppingCriteria(self.tokenizer, input_len, stop_on_json=False, stop_on_action=True)
            ])

            vla_gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": [self.tokenizer.eos_token_id, 151643, 151645],
                "repetition_penalty": 1.1,
                "return_dict_in_generate": True,
                "output_hidden_states": True,
                "stopping_criteria": stop_criteria,
            }
            if do_sample and temperature > 0.0:
                vla_gen_kwargs["do_sample"] = True
                vla_gen_kwargs["temperature"] = max(0.01, float(temperature))
                vla_gen_kwargs["top_p"] = min(1.0, max(0.01, float(top_p)))
            else:
                vla_gen_kwargs["do_sample"] = False

            with torch.inference_mode():
                gen_outputs = self.slm_model.generate(
                    **model_inputs,
                    **vla_gen_kwargs,
                )

                new_tokens = gen_outputs.sequences[0][input_len:]
                num_tokens = len(new_tokens)
                response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
                response_text = (
                    response_text.replace("<|im_end|>", "")
                    .replace("<|im_start|>", "")
                    .replace("<|endoftext|>", "")
                    .strip()
                )

                # Extract last hidden state (1536-dim)
                if gen_outputs.hidden_states and len(gen_outputs.hidden_states) > 0:
                    last_step_states = gen_outputs.hidden_states[-1]
                    latent_h = last_step_states[-1][:, -1, :].detach().to(dtype=torch.float32, device=self.device)  # [1, 1536]
                    bot_id = str(data.get("bot_id", "default"))
                    self.latest_latent_h = latent_h
                    self.bot_latents[bot_id] = latent_h
                else:
                    latent_h = torch.zeros(1, 1536, device=self.device)

                proprio_tensor = torch.tensor(proprio_list, dtype=torch.float32, device=self.device)
                voxels_list = data.get("voxels", None)
                entities_list = data.get("entities", None)
                voxel_tensor = None
                if isinstance(voxels_list, list) and len(voxels_list) == 3375:
                    voxel_tensor = torch.tensor(voxels_list, dtype=torch.long, device=self.device).view(1, 15, 15, 15)
                entity_tensor = None
                if isinstance(entities_list, list) and len(entities_list) == 16:
                    entity_tensor = torch.tensor(entities_list, dtype=torch.float32, device=self.device).view(1, 16, 12)
                latent_h_act = (latent_h * 0.1).clamp(-10.0, 10.0)
                motor_controls = self.action_head.predict_action_dict(
                    latent_h_act, proprio_tensor, voxels=voxel_tensor, entities=entity_tensor
                )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        parsed = parse_vla_response(response_text)

        return web.json_response({
            "intent": parsed["intent"],
            "speech": parsed["speech"],
            "action_tokens": parsed["action_tokens"],
            "parsed_actions": parsed["parsed_actions"],
            "motor_controls": motor_controls,
            "raw_text": response_text,
            "tokens_generated": num_tokens,
            "latency_ms": round(dt_ms, 1),
        })

    async def handle_act_chunk(self, request: web.Request) -> web.Response:
        """
        Lock-free ultra-low-latency (<3ms) 20Hz continuous 8-tick Action Chunk inference.
        Cross-attends to the bot's cached 1,536-dim System 2 SLM latent vector + live proprioception
        + 16-entity radar + 3D semantic voxels without acquiring self.llm_lock!
        """
        if self.action_head is None:
            return web.json_response({"error": "ActionHead not loaded"}, status=503)

        t0 = time.perf_counter()
        data = await request.json()
        bot_id = str(data.get("bot_id", "default"))
        intent = str(data.get("intent", "")).lower()
        proprio_list = data.get("proprioception", [0.0] * 24)
        voxels_list = data.get("voxels", None)
        entities_list = data.get("entities", None)

        base_latent = self.bot_latents.get(bot_id, self.latest_latent_h)
        has_real_latent = base_latent is not None and float(base_latent.abs().max().item()) >= 1e-4
        if has_real_latent:
            latent_h = (base_latent * 0.1).clamp(-10.0, 10.0).clone()
        else:
            latent_h = torch.zeros(1, 1536, device=self.device)
            # Bootstrap fallback slice conditioning ONLY before first real SLM latent exists
            if "kite" in intent or "creeper" in intent:
                latent_h[:, 0:32] += 1.5
            elif "combat" in intent or "attack" in intent or "engage" in intent or "slay" in intent:
                latent_h[:, 32:64] += 1.5
            elif "parkour" in intent or "navigate" in intent or "bridge" in intent or "sprint" in intent:
                latent_h[:, 64:96] += 1.5
            elif "mine" in intent or "gather" in intent or "dig" in intent or "chop" in intent:
                latent_h[:, 96:128] += 1.5
            elif any(k in intent for k in ("craft", "smelt", "equip", "chest", "eat", "drop")):
                latent_h[:, 160:192] += 1.5
            else:
                latent_h[:, 224:256] += 1.0

        proprio_tensor = torch.nan_to_num(
            torch.tensor(proprio_list, dtype=torch.float32, device=self.device),
            nan=0.0, posinf=1.0, neginf=-1.0,
        )
        voxel_tensor = None
        if isinstance(voxels_list, list) and len(voxels_list) == 3375:
            voxel_tensor = torch.tensor(voxels_list, dtype=torch.long, device=self.device).view(1, 15, 15, 15)
        entity_tensor = None
        if isinstance(entities_list, list) and len(entities_list) == 16:
            entity_tensor = torch.nan_to_num(
                torch.tensor(entities_list, dtype=torch.float32, device=self.device).view(1, 16, 12),
                nan=0.0, posinf=1.0, neginf=-1.0,
            )

        explore = bool(data.get("explore", os.environ.get("NEURAL_MOTOR_EXPLORE", "false").lower() == "true"))
        motor_controls = self.action_head.predict_action_dict(
            latent_h, proprio_tensor, voxels=voxel_tensor, entities=entity_tensor, explore=explore
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return web.json_response({
            "motor_controls": motor_controls,
            "latency_ms": round(dt_ms, 2),
        })

    async def handle_shutdown(self, request: web.Request) -> web.Response:
        loop = asyncio.get_running_loop()
        loop.call_later(0.5, lambda: sys.exit(0))
        return web.json_response({"status": "shutting_down"})

    async def handle_pause_slm(self, request: web.Request) -> web.Response:
        """Temporarily moves ALL GPU models to CPU and flushes CUDA cache during QLoRA training passes."""
        if torch.cuda.is_available():
            if self.llm_lock is None:
                self.llm_lock = asyncio.Lock()
            async with self.llm_lock:
                for attr_name in ("slm_model", "action_head", "reflex_policy", "chunk_analyzer", "visuomotor_policy"):
                    model_obj = getattr(self, attr_name, None)
                    if model_obj is not None:
                        setattr(self, attr_name, model_obj.to("cpu"))
                gc.collect()
                torch.cuda.empty_cache()
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                print(f"[System 2] All GPU models offloaded to CPU for training (Active VRAM: {vram_mb:.1f} MiB)", flush=True)
        return web.json_response({
            "status": "paused",
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0,
        })

    async def handle_reload_adapter(self, request: web.Request) -> web.Response:
        if self.slm_model is None:
            return web.json_response({"error": "SLM not loaded"}, status=503)

        try:
            data = await request.json() if request.can_read_body else {}
        except Exception:
            data = {}

        adapter_name = data.get("adapter_name", "slm_minecraft_savant_latest")
        adapter_dir = os.path.join(os.path.dirname(__file__), "checkpoints", adapter_name)
        has_adapter = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))

        if self.llm_lock is None:
            self.llm_lock = asyncio.Lock()

        def _sync_reload() -> Dict:
            from peft import PeftModel
            target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            if has_adapter:
                # Reversibly unmerge & unload previous PeftModel so base weights W_0 NEVER stack across loops
                if isinstance(self.slm_model, PeftModel):
                    try:
                        if hasattr(self.slm_model, "unmerge_adapter"):
                            self.slm_model.unmerge_adapter()
                    except Exception:
                        pass
                    try:
                        base = self.slm_model.unload()
                    except Exception:
                        base = self.slm_model.get_base_model()
                    self.slm_model = PeftModel.from_pretrained(base, adapter_dir)
                else:
                    self.slm_model = PeftModel.from_pretrained(self.slm_model, adapter_dir)

                try:
                    self.slm_model.merge_adapter()
                    print(f"[System 2] Merged reloaded LoRA adapter reversibly into PeftModel (zero weight stacking).", flush=True)
                except Exception as fe:
                    print(f"[System 2] Warning: Could not merge reloaded LoRA: {fe}", flush=True)

            # Always ensure SLM is safely restored to target device and in eval mode
            self.slm_model = self.slm_model.to(dtype=target_dtype)
            self.slm_model = self.slm_model.to(self.device)
            self.slm_model.eval()
            if has_adapter:
                self.active_adapter = adapter_dir
            param_device = next(self.slm_model.parameters()).device

            # Hot-reload 211M Multi-Modal Action Head weights and restore to GPU
            head_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "action_head_latest.pt")
            if self.action_head is not None:
                act_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                try:
                    if os.path.exists(head_ckpt):
                        self.action_head.load_state_dict(torch.load(head_ckpt, map_location=self.device, weights_only=True), strict=False)
                        print(f"[System 1] Hot-reloaded ActionHead (211M Multi-Modal ACT) from {head_ckpt}", flush=True)
                except Exception as he:
                    print(f"[System 1] Warning: Failed to hot-reload ActionHead: {he}", flush=True)
                finally:
                    self.action_head = self.action_head.to(device=self.device, dtype=act_dtype)
                    self.action_head.eval()

            # Hot-reload System 1 PPO VoxelReflexPolicy weights and restore to GPU
            reflex_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "voxel_reflex_policy_latest.pt")
            reflex_reloaded = False
            if self.reflex_policy is not None:
                try:
                    if os.path.exists(reflex_ckpt):
                        self.reflex_policy.load_state_dict(torch.load(reflex_ckpt, map_location=self.device, weights_only=True))
                        reflex_reloaded = True
                        print(f"[System 1] Hot-reloaded VoxelReflexPolicy from {reflex_ckpt}", flush=True)
                except Exception as re_err:
                    print(f"[System 1] Warning: Failed to hot-reload VoxelReflexPolicy: {re_err}", flush=True)
                finally:
                    self.reflex_policy = self.reflex_policy.to(self.device)
                    self.reflex_policy.eval()

            # Restore VolumetricChunkAnalyzer & VisuomotorPolicy to GPU
            if self.chunk_analyzer is not None:
                self.chunk_analyzer = self.chunk_analyzer.to(self.device)
                self.chunk_analyzer.eval()
            if self.visuomotor_policy is not None:
                self.visuomotor_policy = self.visuomotor_policy.to(self.device)
                self.visuomotor_policy.eval()

            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            print(f"[System 2] Hot-reloaded LoRA adapter on {param_device} from {adapter_dir} (Active VRAM: {vram_mb:.1f} MiB)", flush=True)
            return {
                "status": "ok",
                "active_adapter": adapter_dir,
                "slm_device": str(param_device),
                "action_head_reloaded": bool(self.action_head is not None),
                "reflex_reloaded": reflex_reloaded,
                "vram_allocated_mb": round(vram_mb, 1),
                "reloaded_at": time.time(),
            }

        async with self.llm_lock:
            try:
                payload = await asyncio.to_thread(_sync_reload)
                return web.json_response(payload)
            except Exception as e:
                print(f"[System 2] Failed to hot-reload adapter: {e}")
                return web.json_response({"error": str(e)}, status=500)

    async def handle_dqn_sparring_predict(self, request: web.Request) -> web.Response:
        """Runs CPU inference (<0.2ms, 0 VRAM) for the external GeorgeDudau/Minecraft-AI-Agent Dueling DQN."""
        if self.dqn_opponent is None:
            return web.json_response({"error": "External DQN opponent not loaded"}, status=503)

        data = await request.json()
        state_vec = data.get("state", [])
        if not isinstance(state_vec, list) or len(state_vec) != 55:
            return web.json_response({"error": f"Expected 55-dim state vector, got {len(state_vec) if isinstance(state_vec, list) else type(state_vec).__name__}"}, status=400)

        dqn_actions = ["forward", "back", "left", "right", "jump", "sprint", "attack"]
        with torch.inference_mode():
            t = torch.tensor(state_vec, dtype=torch.float32, device="cpu").unsqueeze(0)
            q_vals = self.dqn_opponent(t).squeeze(0)
            action_idx = int(torch.argmax(q_vals).item())

        return web.json_response({
            "action": action_idx,
            "action_name": dqn_actions[action_idx],
            "q_values": [round(float(v), 4) for v in q_vals.tolist()],
        })

    async def handle_vla_latest_latent(self, request: web.Request) -> web.Response:
        bot_id = str(request.query.get("bot_id", "default"))
        latent = self.bot_latents.get(bot_id, self.latest_latent_h)
        if latent is None:
            return web.json_response({"bot_id": bot_id, "latent_h": None})
        vec = [round(float(x), 3) for x in latent.detach().view(-1).cpu().tolist()[:1536]]
        return web.json_response({"bot_id": bot_id, "latent_h": vec})


def main():
    parser = argparse.ArgumentParser(description="NeuroCraft RTX 5060 Local Neural Daemon")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--slm-model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--no-slm", action="store_true", help="Only load System 1 Voxel & Visuomotor policies")
    args = parser.parse_args()

    server = NeuralStackServer(slm_model_id=args.slm_model, load_slm=not args.no_slm)
    server.initialize_models()

    # Suppress harmless Windows Proactor client disconnection noise (WinError 10054 on aborted HTTP sockets)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    orig_handler = loop.get_exception_handler()
    def proactor_handler(lp, context):
        exc = context.get("exception")
        msg = context.get("message", "")
        if isinstance(exc, ConnectionResetError) or "[WinError 10054]" in str(exc) or "[WinError 10054]" in msg:
            return
        if orig_handler:
            orig_handler(lp, context)
        else:
            lp.default_exception_handler(context)
    loop.set_exception_handler(proactor_handler)

    app = web.Application()
    app.router.add_get("/health", server.handle_health)
    app.router.add_get("/vla/latest_latent", server.handle_vla_latest_latent)
    app.router.add_post("/reflex", server.handle_reflex)
    app.router.add_post("/chunk_scan", server.handle_chunk_scan)
    app.router.add_post("/llm/chat", server.handle_llm_chat)
    app.router.add_post("/vla/predict", server.handle_vla_predict)
    app.router.add_post("/act/chunk", server.handle_act_chunk)
    app.router.add_post("/dqn_sparring/predict", server.handle_dqn_sparring_predict)
    app.router.add_post("/pause_slm", server.handle_pause_slm)
    app.router.add_post("/reload_adapter", server.handle_reload_adapter)
    app.router.add_post("/shutdown", server.handle_shutdown)

    print(f"[NeuralDaemon] Listening on http://127.0.0.1:{args.port}")
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
