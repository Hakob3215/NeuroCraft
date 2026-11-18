# NeuroCraft Neural Lab: Multi-Modal VLA Daemon & GPU Trainers

This module (`neural_lab/`) hosts the local PyTorch neural architectures, HTTP inference daemon, trainers, and GPU profiling benchmarks optimized for an **NVIDIA GeForce RTX 5060 Laptop GPU** (8GB VRAM, `<= 6.0 GB` active ceiling).

---

## Core Neural Stack

1. **`action_head.py` — Scaled-Up 211.4M Multi-Modal Action Transformer (`MinecraftActionHead`)**:
   - **Parameters**: `211,357,480` (~422.7 MB BF16), 12 Transformer Decoder layers, 16 heads, `d_model = 1024`.
   - **85-Token Multi-Modal Context**:
     - `4` SLM Intention Tokens projected from per-bot hidden state $h \in \mathbb{R}^{1536}$
     - `1` Bodily Proprioception Token (`24-dim`: HP, food, velocity, aim, target vector, fuse/tool state)
     - `16` Entity Radar Tokens (`[16, 12]`: nearest entities, velocities, hostile/creeper/skeleton/boss/item flags)
     - `64` Spatial Semantic 3D Voxel Patch Tokens ($15 \times 15 \times 15$ 32-class semantic block IDs $\to$ 3D CNN `4×4×4`)
   - **Outputs**: 8-tick ($400\text{ms}$ @ 20Hz) continuous aim (`Δyaw, Δpitch`), velocity (`vx, vz`), core & extended motor bits (`jump, sprint, sneak, attack, use_item, swap_slot, dig_block, drop_item`), 24-class `tool_dispatch_head` + `tool_gate_head`, and `subgoal_head` (`subgoal_completed`, `should_replan`) + Critic $V(s)$.
2. **`models.py` — Sub-Tick Reflex & 3D Scene Networks**:
   - **`VoxelReflexPolicy` (~1.1M params)**: ~2ms hazard & combat reflex network over $15 \times 15 \times 15$ voxels + 16 entities.
   - **`VolumetricChunkAnalyzer`**: $64 \times 64 \times 64$ 3D volumetric scanner for flat building sites, ore clusters, and cavern hazards.
   - **`VisuomotorPolicy`**: Spatiotemporal transformer benchmark architecture.
3. **`neural_daemon.py` — Local RTX 5060 Inference & Hot-Reload Server (`http://127.0.0.1:8765`)**:
   - Serves `/llm/chat` (`Qwen2.5-Coder-1.5B-Instruct` with stochastic RL sampling `SLM_TEMPERATURE=0.7`, $O(1)$ JSON/token stopping criteria, and per-bot cached 1,536-dim latent vectors `self.bot_latents[bot_id]`).
   - Serves lock-free `/act/chunk` for ultra-low-latency 20Hz Action Head queries without blocking on `llm_lock`.
   - Supports dynamic CPU offloading (`/pause_slm`) during QLoRA backprop and reversible LoRA + ACT + PPO hot-reloading (`/reload_adapter` via `merge_adapter()` / `unmerge_adapter()`).
4. **`train_action_head.py`**:
   - Trains `MinecraftActionHead` jointly on 32-class semantic voxels, 16-entity radar states, 8-tick motor trajectories, extended dig/drop bits, and discrete tool dispatch targets, blending synthetic scenarios with real server transitions from `minebench/results/real_rl_transitions.jsonl`.

---

## Hardware Profiler Benchmarks

To run the standalone GPU microbenchmarks manually:
```bat
cd neural_lab
run_benchmarks.bat
```
Results are written to `neural_lab/benchmark_results.txt` and `neural_lab/benchmark_results.json`.
