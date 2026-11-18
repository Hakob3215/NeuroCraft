# NeuroCraft: Dual-Brain Hierarchical VLA Minecraft AI Agent

A self-contained, zero-mod autonomous Minecraft AI agent & local RTX 5060 RL training platform featuring a **Hierarchical Vision-Language-Action (VLA)** architecture grounded in official Mojang Minecraft `1.20.4` physics:

1. **System 2 — Cognitive Planner & Long-Term Goal Horizon (`Qwen2.5-Coder-1.5B` QLoRA + `HierarchicalGoalPlanner`):**
   - Receives rich grounded presence via `computeGroundedAffordances`: gear & tool durability, nearby workstations, immediately craftable/smeltable items, 360° resource clusters, nearby entities, and the active Macro Goal / Sub-Goal Stack (`goalPlanner.ts`).
   - Outputs structured cognitive decisions and a per-bot `1,536-dim` latent intention vector ($h \in \mathbb{R}^{1536}$), dynamically advancing sub-goals or triggering replanning on failures/hazards.
2. **System 1 — 211.4M Multi-Modal Neural Motor Head (`action_head.py`) & Sub-Tick Reflex Core (`models.py`, `reflexEngine.ts`):**
   - Cross-attends over an **85-token multi-modal context memory** (`4` SLM latent tokens + `1` 24-dim bodily proprioception token + `16` 360° entity radar tokens + `64` spatial patches from a $15 \times 15 \times 15$ **32-class semantic 3D voxel grid**).
   - Uses **BOTH** the SLM's latent intent ($h$) and live 3D world perception to generate **8-tick (400ms @ 20Hz) motor trajectories** (`Δyaw`, `Δpitch`, `vx`, `vz`, `jump`, `sprint`, `sneak`, `attack`, `use_item`, `swap_slot`, `dig_block`, `drop_item`), discrete tool dispatches across 24 canonical tools, and sub-goal lifecycle/replan signals (`subgoal_completed`, `should_replan`, $V(s)$).
   - Paired with `VoxelReflexPolicy` (~2ms hazard reflexes for shield timing, Creeper kiting, lava avoidance) and human-mimicking perceptual FOV/raycasting, log-normal reaction delays, and 1.9+ combat cooldowns.
3. **Strict Combinatorial MCP Tool Layer (`baseSkills.ts`, `mcpServer.ts`):**
   - Handles discrete combinatorial GUI & container operations (`3×3` crafting matching with `allowAutoPrereqs = false`, furnace fuel/smelting cycles, chest/armor slot transfers) plus an isolated `node:vm` self-synthesizing tool factory (`toolFactory.ts`).

---

## Training & Dual-Track Curriculum (`train_agent.bat`)

Run continuous dual-brain reinforcement learning across the **500-Iteration Maturation Curve** (`1–50` exploratory fumbling $\to$ `51–200` ramping $\to$ `201–400+` generalist competence):

```cmd
train_agent.bat
```

- **Track A (Standardized Testbench Shards):** Parallel spatial arena workers (`NeuroAgent_1..3`, $\Delta X = 4,000$) evaluate Agility/Parkour, 3D Architecture, Tactical Combat, Homesteading/Affordances, Crafting DAGs, Puzzles/Tooling, and Multiplayer Sparring.
- **Track B (Persistent Open-World Survival Worker):** `NeuroAgent_Surv` operates persistently at `(20000, 100, 20000)` chasing long-horizon `SURVIVAL_CAMPAIGN_GOALS` (Ender Dragon, Wither, Beacon, Ocean Monument Sponges, Trail Ruins/Ancient City) anchored by Vanilla Advancements.
- **4-Panel Decomposed Dashboard:** Live interactive telemetry rendered to `minebench/results/training_dashboard.html` and `training_curves.svg`.
- **Clean Loop 0 Reset:** Run `reset_agent.bat` to wipe replay buffers, metrics, and checkpoints back to factory Loop 0 and re-seed the clean DAG gold curriculum.

---

## Singleplayer LAN Companion Mode (`run_singleplayer.bat`)

1. Launch **Minecraft Singleplayer**, open to LAN (`Esc` $\to$ `Open to LAN` $\to$ `Start LAN World`).
2. Run **`run_singleplayer.bat`** (`NeuroAgent` auto-discovers the UDP LAN broadcast on `224.0.2.60:4445`).
3. Chat naturally in-game (`"follow me"`, `"gather 8 oak_log"`, `"craft stone_pickaxe"`, `"fight mobs"`, `"!tools"`).
