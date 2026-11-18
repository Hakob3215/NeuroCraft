# NeuroCraft Autonomous Agent Workspace Rules & Execution Standards

This file defines the mandatory operational directives, benchmark standards, and runtime protocols for the **NeuroCraft** Minecraft AI agent project. All agents operating in this workspace must adhere to these rules without exception.

---

## 1. Foundational Philosophy: The Bitter Lesson

All development, training, and benchmarking in NeuroCraft are governed by Richard Sutton's principle **The Bitter Lesson**:
> *"The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, by a large margin... Human knowledge and procedural shortcuts always get washed away by compute."*

### Core Directives:
1. **Never "Solve It for the Model"**: If training results or benchmark scores drop, never inject procedural shortcuts, regex pattern routers, synthetic state machines, or hand-coded safety nets to artificially boost scores.
2. **Embrace Exploratory Failure**: If something *can* be learned through more training iterations, cleaner loss functions, and grounded reinforcement learning, it *must* be learned that way. 
3. **The 100-Iteration Maturation Curve**:
   - **Iterations 1–25 (Unassisted Baseline)**: Raw exploratory behavior, low benchmark pass rates (~50–60%), prerequisite DAG failures, and clumsy motor misses. This is the natural and necessary progression of reinforcement learning, **not a system defect**.
   - **Iterations 26–60 (Distillation & Pruning)**: PPO policy gradients, GRPO verified trajectories, and QLoRA fine-tuning prune invalid tool calls and tighten continuous motor chunk trajectories.
   - **Iterations 61–100+ (Emergent Mastery)**: High benchmark pass rates (85%+) reflecting genuine neural weight adaptation across your RTX 5060 GPU.

---

## 2. The Architectural Division of Labor (Discrete Symbolic vs. Continuous Sensorimotor)

NeuroCraft strictly divides cognitive thought and bodily action into two complementary domains:

```
┌─────────────────────────────────────────────────────────────┐
│                   SYSTEM 2 (Cognitive SLM)                  │
│       - Natural Language Intent & Multi-Step DAGs           │
│       - Produces 1,536-dim Latent Thought Vector (h)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│       MCP TOOL LAYER         │     │    SYSTEM 1 (80.8M ACT)      │
│   (Discrete / Combinatorial) │     │  (Continuous / Sensorimotor) │
│  - Crafting Recipe Graph     │     │  - 20Hz Camera Aim Tracking  │
│  - Smelting & Fuel Cycles    │     │  - Continuous Planar Velocity│
│  - Chest Inventory Transfers │     │  - Sprint-Jumping & Parkour  │
│  - Tool Durability Checks    │     │  - Melee Strike Timings      │
│  - Block Breaking / Placing  │     │  - Sub-Tick Voxel Reflexes   │
└──────────────────────────────┘     └──────────────────────────────┘
```

### 1. Discrete Symbolic Domain (MCP Tool Layer)
- **What Belongs Here**: 46-slot inventory management, official Minecraft crafting table recipe matching, furnace smelting, and durability thresholds.
- **Why**: Trying to force a continuous neural head to output raw window-click packets (`clickWindow(slot, mode)`) explodes the search space into a $46!$ combinatorial trap. The deterministic recipe engine guarantees valid crafts without hallucinations.
- **Strict Prohibition of Procedural Crutches**:
  - **No Auto-Crafting Crutches**: The execution layer must NEVER silently synthesize sub-ingredients (e.g. auto-crafting wood planks, sticks, or crafting tables). If the agent attempts to craft an advanced item without prerequisite items in its inventory, the tool must fail and return grounded inventory feedback, compelling the SLM to discover and plan the prerequisite DAG sequence (`gather_resource` $\to$ `craft planks` $\to$ `craft table` $\to$ `craft sticks` $\to$ `craft pickaxe`).
  - **No Synthetic Progression Nudges**: Prompts and context formatters must NEVER inject artificial stopping or progression hints. The cognitive policy must evaluate its own inventory and world state.
  - **No Monolithic Structure Macros**: 3D construction must not be collapsed into single-call procedural scripts that build an entire structure in one shot.

### 2. Continuous Sensorimotor Domain (System 1: ACT & Reflex)
- **What Belongs Here**: 20Hz continuous physics, 3D voxel spatial awareness, camera aiming, velocity vectors, sprinting, sneaking, jumping, and melee strike timing.
- **Direct Latent Coupling**:
  - The SLM (Qwen2.5-Coder-1.5B) produces a 1,536-dimensional latent representation ($h \in \mathbb{R}^{1536}$) on every decision.
  - The 80.8M Temporal Action Chunking Transformer (`action_head.py`) cross-attends directly to $h$, combining it with bodily proprioception (24-dim) and local 3D voxels ($15 \times 15 \times 15$) to output 8-tick ($400\text{ms}$) continuous action trajectories:
    - Continuous camera aim: $(\Delta\text{yaw}, \Delta\text{pitch})$
    - Continuous planar velocity vectors: $(v_x, v_z)$
    - Discrete motor activations: `[jump, sprint, sneak, attack, use_item, swap_slot]` across all 8 ticks.
- **Full Reflex Output Consumption**:
  - In `reflexEngine.ts`, all outputs from `VoxelReflexPolicy` must be actively applied: continuous camera aim deltas, directional strafing/backpedaling, sprint-jumping, and reactive attacks. Never castrate the neural reflex to a simple 2-button trigger.

---

## 3. Mandatory Benchmark & Runtime Flags

To guarantee that benchmarks measure real neural capabilities rather than regex pattern matching or hardcoded state machines, the following environment flags must remain permanently active:

```cmd
set DISABLE_FAST_INTENT=true
set STRICT_LLM_BENCHMARK=true
set NEURAL_FORCE_LOCAL_SLM=true
set LLM_PROVIDER=local_gpu
```

- `DISABLE_FAST_INTENT=true`: Completely disables regex intent interception so the SLM is forced to parse all player prompts.
- `STRICT_LLM_BENCHMARK=true`: Disables heuristic fallback state machines on JSON parse failures, exposing true policy accuracy.
- `NEURAL_FORCE_LOCAL_SLM=true`: Directs all inference to the local RTX 5060 GPU daemon.

---

## 4. Definition & Requirements for a "Full Run"

Whenever the user instructs to perform a **"full run"**, **"full benchmark"**, or **"full training run"**:

### Mandatory Dual-Tier Execution
Never substitute a single-step policy probe or abbreviated shortcut for a full run unless the user explicitly asks for a "smoke test" or "quick check". Every full run must exhaustively execute **both tiers**:

1. **Tier 1: Cognitive Policy Planning (System 2 VLA / SLM)**:
   - Full natural language prompt decoding and intent classification.
   - Generation of structured tool calls and 1,536-dim latent vectors.
   - Generation of 8-tick ($400\text{ms}$) continuous motor chunk trajectories from the 80.8M Action Chunking Transformer.
2. **Tier 2: Grounded Physical Simulation (System 1 + Mojang `server.jar` 1.20.4 Physics)**:
   - Live multi-step execution loop across physical game ticks.
   - Verification of real world state transitions:
     - **Affordance & Homesteading**: Physical block state transitions (`oak_log` $\to$ `stripped_oak_log`, dirt $\to$ farmland, sheep shearing, crop harvesting).
     - **Agility & Navigation**: Jump-by-jump parkour navigation, sneak-bridging over ravines, and reaching target coordinates without falling.
     - **Tactical Combat**: Live entity damage, shield blocking, kiting creeper fuses, dodging skeleton arrows, and verifying entity HP reaches 0 in server memory.
     - **Architecture & Progression**: 3D voxel placement (IoU verification), crafting tree progressions, smelting, and equipping gear.

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

### Mandatory Graceful Shutdown Standards (`Ctrl+C` / `SIGINT` / `SIGTERM`)
Any script running continuous loops must handle interruption cleanly and without data loss:
1. **Atomic State Flushing**:
   - Immediately flush and save `training_state.json`, `grpo_verified_trajectories.jsonl`, and the latest scorecard to disk.
   - Prevent JSON truncation or corrupted records.
2. **Clean Bot Disconnect**:
   - Send `disconnect.quitting` packet to cleanly disconnect `NeuroAgent` from the server.
3. **Headless Server Teardown**:
   - Issue `/stop` to `server.jar` and wait for clean process termination.
4. **VRAM & Port Safety**:
   - Terminate or pause background daemons, release CUDA allocations, and ensure ports `25575` (Minecraft) and `8765` (VLA Daemon) are completely free.
   - Zero orphaned `node.exe`, `python.exe`, or `java.exe` processes.

---

## 6. Hardware & Memory Envelope (RTX 5060 Laptop GPU, 8GB)

- **VRAM Ceiling**: Never allocate more than 6.0 GB of VRAM. Maintain a **minimum of 2.0 GB of free headroom** at all times to prevent system freezes or display driver crashes.
- **Dynamic Offloading During Backprop**: When running QLoRA training on System 2, send `POST /pause_slm` to offload the active inference model to CPU memory, execute gradient backpropagation safely, and hot-reload updated weights to GPU with `POST /reload_adapter`.
- **Action Head Kinematics**: All physical bot actions must use the 80.8M Temporal Action Chunking Transformer (`action_head.py`) generating 8-tick ($400\text{ms}$) trajectories to guarantee smooth, human-mimicking movement.

---

## 7. Official Minecraft Server Scoreboard Telemetry Standards

To guarantee 100% objective, incontrovertible physical logging directly within the Mojang `server.jar` engine (and in `latest.log`), the following 12 vanilla scoreboard objectives must remain permanently registered during live testing:

1. `deaths` (`deathCount`): Total player deaths recorded natively by server.
2. `mob_kills` (`totalKillCount`): Total mob & entity kills (displayed on sidebar).
3. `dmg_dealt` (`minecraft.custom:minecraft.damage_dealt`): Actual melee & ranged hit points dealt by `NeuroAgent` (tenths of HP).
4. `dmg_taken` (`minecraft.custom:minecraft.damage_taken`): Real physical damage sustained by `NeuroAgent`.
5. `dmg_blocked` (`minecraft.custom:minecraft.damage_blocked_by_shield`): Damage absorbed by shield blocking mechanics.
6. `sprint_cm` (`minecraft.custom:minecraft.sprint_one_cm`): Agility sprinting displacement in cm.
7. `jump_cnt` (`minecraft.custom:minecraft.jump`): Total jump counts across parkour and navigation obstacles.
8. `mine_log` (`minecraft.mined:minecraft.oak_log`): Grounded oak log block break events.
9. `mine_stone` (`minecraft.mined:minecraft.stone`): Grounded stone block break events.
10. `mine_iron` (`minecraft.mined:minecraft.iron_ore`): Grounded iron ore block break events.
11. `craft_bench` (`minecraft.crafted:minecraft.crafting_table`): Unassisted crafting table synthesis.
12. `craft_pick` (`minecraft.crafted:minecraft.stone_pickaxe`): Unassisted stone pickaxe crafting milestones.

At the conclusion of every benchmark episode, `train_loop.ts` queries all 12 scoreboard targets via server RCON/stdin so that verifiable server ground-truth is logged chronologically to `latest.log`.

---

## 8. Two-Phase Operational Lifecycle: Testing Mode vs. Long-Term Training Mode

To ensure the architecture, benchmark suite, physics engine, and telemetry are 100% sound before committing to extensive compute campaigns, execution must follow a strict two-phase operational protocol:

### Phase 1: Testing Mode (Iterative Architectural Audit & Wipe Loop)
1. **Short Batch Execution**:
   - Execute ONLY one or two iterations (`--iterations 1` or `--iterations 2`).
2. **Dedicated Architecture Auditor Subagent**:
   - Immediately upon batch completion, launch a specialized subagent whose sole task is to point out flaws with the **ARCHITECTURE and BENCHMARK SUITE**.
   - **Mandatory Directive**: The auditor must NOT complain that the model scored low due to early training infancy (this is expected under Sutton's Bitter Lesson and the 100-Iteration Maturation Curve). Instead, it must audit for genuine systemic, structural, and mechanical defects:
     - Entity filters, perception blind spots, or tool signature mismatches (e.g., players excluded from combat, missing tool schemas).
     - Minecraft server errors, keepalive drops, tick lag warnings, packet disconnects, or socket timeouts.
     - Broken crafting/smelting DAG prerequisites or unachievable win conditions in task evaluators.
   - **Scope of Sweep**: The subagent must extensively sweep the entire active, unarchived codebase (`src/`, `minebench/`, `neural_lab/`), runner telemetry, and Minecraft server logs (`minebench/server/logs/latest.log`).
3. **Coordinator Evaluation & Remediation**:
   - The coordinator evaluates the subagent's report, filtering out any suggestions that would violate the Bitter Lesson (no procedural auto-crafting crutches or regex routers).
   - If genuine architectural, perception, or suite flaws are identified, implement the proper code-level fixes and verify TypeScript/Python compilation.
4. **Clean Wipe & Re-Probe**:
   - After implementing verified fixes, wipe the previous probe logs/checkpoints and execute another 1–2 iteration probe.
   - Continue this iterative audit $\to$ evaluate $\to$ fix $\to$ wipe $\to$ re-probe loop for as long as necessary until either:
     - The subagent reports zero architectural/suite flaws, OR
     - The coordinator deems all remaining suggestions improper or invalid.

### Phase 2: Long-Term Training Mode (Sustained Maturation to User Deadline)
- Once Phase 1 settles cleanly and the architecture is certified sound, transition to long-term training mode.
- Run iterations continuously toward the user's specified time budget (either a duration in hours or a specific wall-clock deadline).
- Perform safe landings before the deadline, flushing atomic state, syncing dashboard artifacts, and presenting final scorecard telemetry.
