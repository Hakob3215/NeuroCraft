# NeuroCraft Autonomous Agent Workspace Rules & Execution Standards

This file defines the mandatory operational directives, benchmark standards, architecture specifications, and runtime protocols for the **NeuroCraft** Minecraft AI agent project. All agents operating in this workspace must adhere to these rules without exception.

---

## 1. Foundational Philosophy: The Bitter Lesson

All development, training, and benchmarking in NeuroCraft are governed by Richard Sutton's principle **The Bitter Lesson**:
> *"The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, by a large margin... Human knowledge and procedural shortcuts always get washed away by compute."*

### Core Directives:
1. **Never "Solve It for the Model"**: If training results or benchmark scores drop, never inject procedural shortcuts, regex pattern routers, synthetic state machines, or hand-coded safety nets to artificially boost scores.
2. **Embrace Exploratory Failure**: If something *can* be learned through more training iterations, cleaner loss functions, and grounded reinforcement learning, it *must* be learned that way.
3. **The 500-Iteration Maturation Curve**:
   - **Iterations 1–50 (Exploratory Fumbling & Grounding Phase)**: Raw unassisted exploration (`SLM_TEMPERATURE=0.7`, `SLM_TOP_P=0.9`, `SLM_DO_SAMPLE=true`, `allowAutoPrereqs=false`). Expect low pass rates (~15–30%), prerequisite DAG misses, and clumsy motor attempts while the agent discovers basic affordances, gathers early successes into the Stratified Task Gold Bank, and fills 3D transition buffers. This fumbling phase is the natural progression of reinforcement learning, **not a system defect**.
   - **Iterations 51–200 (Getting Its Footing — "Kinda Good" Ramp)**: As verified trajectories and real 3D semantic voxel/entity transitions accumulate, QLoRA adapters, the 211.4M Multi-Modal Action Transformer, and PPO reflex gradients gain traction. The agent reliably chains multi-step crafting DAGs (`oak_log` $\to$ `oak_planks` $\to$ `crafting_table` $\to$ `stick` $\to$ `pickaxe`), wins standard combat encounters, and unlocks early open-world survival advancements.
   - **Iterations 201–400 (Accelerated Multi-Domain Competence — "Wow" Phase)**: Strong multi-step error recovery, tool durability preservation, smooth 20Hz motor kiting/parkour, and deep tech-tree progression (Iron & Diamond tiers) across both the standardized testbench and the persistent open-world survival campaign.
   - **Iterations 401–500+ (Emergent Generalist Mastery)**: High, well-rounded competence across all gameplay domains, long-horizon survival campaign goals (Nether fortress, Stronghold triangulation, Wither, Beacon, Ender Dragon), and tight neural coordination between System 2 and the 211.4M Multi-Modal Motor Head.

---

## 2. The Unified Hierarchical VLA Architecture: Rich Presence, SLM Intents & 211.4M Motor Head

NeuroCraft connects a **cognitive LLM planner (System 2)** and a **211.4M multi-modal motor transformer (System 1)** through shared rich sensory presence and strict combinatorial MCP tools:

```
Prompt / Macro Goal + Rich Grounded World State (46-slot Inv, Affordances, 360° Radar, Horizon Stack)
        │
        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  SYSTEM 2: Cognitive SLM (Qwen2.5-Coder-1.5B + QLoRA + HierarchicalGoalPlanner)        │
│   • Decomposes long-term goals & sub-goals; monitors failure / replan triggers          │
│   • Produces discrete JSON Intention + per-bot 1,536-dim Latent Intention Vector (h)   │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │ h ∈ ℝ¹⁵³⁶ + Live 3D Sensory Context
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  85-TOKEN MULTI-MODAL CONTEXT MEMORY [B, 85, 1024]                                     │
│   • 4 SLM Latent Intention Tokens (projected from per-bot h ∈ ℝ¹⁵³⁶)                  │
│   • 1 Bodily Proprioception Token (24-dim HP, food, velocity, aim, target, gear flags) │
│   • 16 Entity Radar Tokens (360° top-16 mobs, players, items, crystals: [16, 12])      │
│   • 64 Spatial 3D Voxel Patch Tokens (15×15×15 32-class semantic blocks -> 4×4×4 grid) │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  SYSTEM 1: 211.4M Multi-Modal Action Transformer (`action_head.py`) + 1.1M Reflex      │
│   • Uses BOTH the SLM intent h AND live 3D data to predict 8-tick (400ms @ 20Hz) chunks│
│   • Continuous Physics: (Δyaw, Δpitch, vx, vz) + [jump, sprint, sneak, attack,         │
│     use_item, swap_slot, dig_block, drop_item]                                         │
│   • Discrete Tool Dispatch (`tool_dispatch_head` across 24 tools + `tool_gate_head`)   │
│   • Lifecycle Feedback (`subgoal_completed`, `should_replan`, Critic V(s))              │
└───────────────────┬───────────────────────────────────────────┬────────────────────────┘
                    │ Combinatorial GUI / Recipe Math           │ Continuous 20Hz Control
                    ▼                                           ▼
        ┌──────────────────────────┐                ┌──────────────────────────┐
        │     MCP TOOL LAYER       │                │   SUB-TICK REFLEX CORE   │
        │  - 3x3 Crafting Matching │                │  - 20Hz Camera & Motion  │
        │  - Furnace Smelt Cycles  │                │  - 1.1M VoxelReflexPolicy│
        │  - Chest & Slot Swaps    │                │  - Shield / Creeper Kite │
        └──────────────────────────┘                └──────────────────────────┘
```

### 1. Rich Informative Presence for Both Brains
- **System 2 Presence (`llmClient.ts` & `goalPlanner.ts`)**: `computeGroundedAffordances(state)` surfaces objective, non-prescriptive situational awareness: gear & minimum tool durability %, nearby workstations, items immediately craftable from current inventory, smeltable ores + fuel availability, 360° resource clusters, nearby entities, and the active `HierarchicalGoalPlanner` horizon stack (`Macro Goal`, `Sub-Goal [idx/total]`, failure alerts).
- **System 1 Multi-Modal Presence (`vlaClient.ts` & `action_head.py`)**: `MinecraftActionHead` (~211.4M parameters, 12 Transformer Decoder layers, 16 heads, `d_model=1024`) cross-attends over **85 rich tokens** combining 4 SLM intent tokens (`h`), 24-dim proprioception, 16 entity radar vectors, and 64 3D spatial patches built from **32-class semantic block IDs** (`0..31` — air, stone, cobble, water, lava, ores, logs, crafting tables, furnaces, gold/emerald goal blocks, levers/doors, etc.), never blind binary cubes.
- **No Procedural Crutches (`allowAutoPrereqs = false`)**: The execution layer must never silently auto-craft sub-ingredients (planks, sticks, crafting tables) or inject synthetic heuristic nudges solving gameplay for the model. Missing prerequisite items must fail cleanly with grounded inventory feedback.

### 2. Dual Outputs: Tick-Future Physical Control + Tool Dispatch + Replanning
- **Tick-Future Actions & Tool Dispatch**: Guided by both $h$ and live 3D voxels/entities, the 211.4M Motor Head outputs continuous camera/movement trajectories alongside discrete action activations (`dig_block`, `drop_item`, `attack`, `use_item`, `swap_slot`), discrete MCP tool dispatch logits (`24 CANONICAL_MOTOR_TOOLS` for crafting, smelting, equipping, chest transfers, etc.), and `should_replan` / `subgoal_completed` signals.
- **Full Reflex Output Consumption**: In `reflexEngine.ts`, all outputs from `VoxelReflexPolicy` (continuous aim deltas, directional strafing/backpedaling, sprint-jumping, shields, and reactive attacks) remain active at sub-tick speed.

---

## 3. Mandatory Benchmark & Runtime Flags

To guarantee that benchmarks measure genuine neural capabilities and explore multi-step trajectories natively rather than hitting deterministic greedy traps, the following environment flags must remain active:

```cmd
set DISABLE_FAST_INTENT=true
set STRICT_LLM_BENCHMARK=true
set NEURAL_FORCE_LOCAL_SLM=true
set LLM_PROVIDER=local_gpu
set SLM_TEMPERATURE=0.7
set SLM_TOP_P=0.9
set SLM_DO_SAMPLE=true
```

- `DISABLE_FAST_INTENT=true`: Completely disables regex intent interception so the SLM parses all prompts.
- `STRICT_LLM_BENCHMARK=true`: Disables heuristic fallback state machines on JSON parse failures.
- `NEURAL_FORCE_LOCAL_SLM=true`: Routes System 2 inference to the local RTX 5060 GPU daemon.
- `SLM_TEMPERATURE=0.7`, `SLM_TOP_P=0.9`, `SLM_DO_SAMPLE=true`: Enables stochastic RL policy exploration rather than argmax mode collapse.

---

## 4. Definition & Requirements for a "Full Run" (Dual-Track Training & Telemetry)

Whenever the user instructs to perform a **"full run"**, **"full benchmark"**, or **"full training run"**:

### Mandatory Dual-Tier & Dual-Track Execution
Never substitute a single-step policy probe or abbreviated shortcut for a full run unless the user explicitly asks for a "smoke test" or "quick check". Every full run must execute both brain tiers against real Mojang `server.jar` 1.20.4 physics across **both training tracks**:

1. **Track A — Standardized MINEbench Sharded Workers (`NeuroAgent_1..3`, $\Delta X = 4,000$)**:
   - Executes curated + Omniscience tasks across Agility/Parkour, Architecture/3D, Tactical Combat, Homesteading/Affordances, Crafting & Tech DAGs, Puzzles/Tooling, and Multiplayer Sparring.
2. **Track B — Persistent Open-World Survival Worker (`NeuroAgent_Surv` at `(20000, 100, 20000)`)**:
   - Runs persistently alongside Track A without resetting world state between iterations, pursuing long-horizon `SURVIVAL_CAMPAIGN_GOALS` (Ender Dragon, Wither, Beacon, Ocean Monument Sponges, Trail Ruins/Ancient City, Iron/Diamond progression) anchored by Vanilla Minecraft Advancements.
3. **Decomposed 4-Panel Dashboard (`generate_training_graphs.ts`)**:
   - Reports outcomes cleanly without muddying metrics: (1) Standardized Exam Over Time, (2) Per-Domain Pass Rate Grouped Bars at Time Increments $t$, (3) Open-World Survival Tech Tier & Advancements, and (4) System 2 SLM Loss + System 1 ACT Aim Error.

---

## 5. Autonomous Runner & Graceful Shutdown Protocol

### Command
```cmd
train_agent.bat
```
or
```bash
node .runtime/node/node.exe minebench/dist/minebench/train_loop.js
```

To reset all replay buffers, checkpoints, and metrics cleanly back to Loop 0:
```cmd
reset_agent.bat
```

### Mandatory Graceful Shutdown Standards (`Ctrl+C` / `SIGINT` / `SIGTERM`)
Any script running continuous loops must handle interruption cleanly and without data loss:
1. **Atomic State Flushing**: Flush `training_state.json`, `grpo_verified_trajectories.jsonl`, and scorecard/dashboard artifacts immediately without truncation.
2. **Clean Bot Disconnect**: Send `disconnect.quitting` packets for all connected worker bots.
3. **Headless Server Teardown**: Issue `/stop` to `server.jar` and wait for clean exit.
4. **VRAM & Port Safety**: Release CUDA memory, ensure ports `25575` (Minecraft) and `8765` (VLA Daemon) are completely free, and leave zero orphaned `node.exe`, `python.exe`, or `java.exe` processes.

---

## 6. Hardware & Memory Envelope (RTX 5060 Laptop GPU, 8GB)

- **VRAM Ceiling**: Never allocate more than 6.0 GB of VRAM. Maintain at least **2.0 GB of free headroom** during both inference and QLoRA/ACT/PPO updates (`SLM ~3.05 GB` + `211.4M Multi-Modal Action Head ~422.7 MB BF16` + `1.1M Reflex / ChunkScan ~20 MB` $\approx$ `3.5–3.8 GB` active inference footprint).
- **Dynamic Offloading & Reversible LoRA Hot-Reload**: During QLoRA backpropagation, send `POST /pause_slm` to offload the SLM to CPU memory, run gradient updates, and hot-reload updated LoRA/ACT/PPO weights with `POST /reload_adapter` using reversible `merge_adapter()` / `unmerge_adapter()` (never stacking $\Delta W$ into $W_0$ across loops).
- **Zero-Disk-Bloat Checkpoints**: Keep only `*_latest`, `*_best`, base `voxel_reflex_policy.pt`, and early diagnostic loop checkpoints (`loop <= 4`).

---

## 7. Official Minecraft Server Scoreboard Telemetry Standards

The following 12 vanilla scoreboard objectives must remain registered during live testing so `train_loop.ts` logs verifiable server ground truth chronologically to `latest.log`:

1. `deaths` (`deathCount`)
2. `mob_kills` (`totalKillCount`, displayed on sidebar)
3. `dmg_dealt` (`minecraft.custom:minecraft.damage_dealt`)
4. `dmg_taken` (`minecraft.custom:minecraft.damage_taken`)
5. `dmg_blocked` (`minecraft.custom:minecraft.damage_blocked_by_shield`)
6. `sprint_cm` (`minecraft.custom:minecraft.sprint_one_cm`)
7. `jump_cnt` (`minecraft.custom:minecraft.jump`)
8. `mine_log` (`minecraft.mined:minecraft.oak_log`)
9. `mine_stone` (`minecraft.mined:minecraft.stone`)
10. `mine_iron` (`minecraft.mined:minecraft.iron_ore`)
11. `craft_bench` (`minecraft.crafted:minecraft.crafting_table`)
12. `craft_pick` (`minecraft.crafted:minecraft.stone_pickaxe`)

---

## 8. Two-Phase Operational Lifecycle: Testing Mode vs. Long-Term Training Mode

### Phase 1: Testing Mode (Iterative Architectural Audit & Wipe Loop)
1. Execute 1–2 iterations (`--iterations 1` or `--iterations 2`).
2. Launch a dedicated Architecture Auditor subagent to inspect **ARCHITECTURE and BENCHMARK SUITE** defects across `src/`, `minebench/`, `neural_lab/`, and `minebench/server/logs/latest.log`.
   - **Mandatory Directive**: The auditor must NOT complain about low early scores (expected during Iterations 1–50 of the 500-Iteration Maturation Curve). Focus solely on systemic perception blind spots, entity filter bugs, server/packet failures, broken DAG prerequisites, or invalid evaluator conditions.
3. Filter recommendations against **The Bitter Lesson** (never add auto-crafting crutches or regex routers), implement verified fixes, compile cleanly, wipe probe logs via `reset_agent.bat`, and re-probe until clean.

### Phase 2: Long-Term Training Mode (Sustained Maturation Toward 500+ Iterations)
- Once the architecture is certified sound, transition to continuous multi-hour or overnight runs across the **500-Iteration Maturation Curve**, landing atomically on completion or deadline with synchronized dashboard visuals.
