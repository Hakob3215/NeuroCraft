# NeuroCraft: Real-Time Human-Mimicking Minecraft AI Agent

A self-contained, zero-mod autonomous AI companion player for **Minecraft Singleplayer** featuring:
1. **System 1 (20Hz / 50ms Reflex & Kinematic Core):**
   - **Synthetic Human 90° FOV & 3D Voxel Raycasting:** Eliminates packet-level "wallhacks" by culling any entity or block occluded behind solid walls.
   - **Log-Normal Perceptual Reaction Latency ($\sim 220\text{ms}$):** Models realistic human surprise reaction times.
   - **Cubic Bézier Camera Smoothing & Perlin Micro-Tremor:** Smooth gaze interpolation with Fitts' Law slight overshoot and subtle hand jitter ($\pm 0.28^\circ$).
   - **Catmull-Rom Spline Trajectories:** Smooth curved pathing and synchronized sprint-jumping instead of robotic 90-degree grid walking.
   - **Modern 1.9+ Combat Physics:** Weapon attack cooldown recharge tracking, airborne falling critical-hit synchronization ($v_y < -0.05$), and 5-tick off-hand shield warm-up.
   - **Sub-50ms Hazard Reflex Engine:** Instant preemption for Creeper fuses, lava contact, drowning, and sudden damage without waiting for LLM calls.
2. **System 2 (Meta-Agent Overseer & Self-Synthesizing Dynamic MCP Interface):**
   - **High-Level Meta-Agent (`metaAgent.ts`):** Oversees all in-game chat and commands.
   - **Self-Synthesizing Tool Factory (`toolFactory.ts`):** When given a novel ask not covered by built-in primitives, dynamically writes new JavaScript tool code, validates it in an isolated `node:vm` security sandbox (`sandbox.ts`), persists it to `data/synthesized_skills/`, and hot-reloads it into the active **Model Context Protocol (MCP)** server (`notifications/tools/list_changed`).
   - **Human Idle State Machine (`idleFsm.ts`):** While thinking or synthesizing code, turns to look at the speaker and nods attentively so the agent never freezes in place.
   - **Speculative Pipeline Prefetching (`speculative.ts`):** Triggers lookahead planning at 75% subtask completion for zero-latency transitions.

---

## Quick Start (100% Vanilla Singleplayer - Zero Mods & Zero Extra Accounts)

1. Launch **Minecraft Singleplayer** and load any world.
2. Press **`Esc`** $\to$ click **`Open to LAN`** $\to$ click **`Start LAN World`**.
3. Double-click **`run_singleplayer.bat`** (or run `.\run_singleplayer.bat` in terminal).
   - *Note:* On first run, `run_singleplayer.bat` automatically bootstraps a portable Node.js v20 runtime into `.runtime\node\`, installs local packages into `node_modules\`, and compiles TypeScript into `dist\`. Everything stays 100% inside this folder!
4. Within 2 seconds, **`NeuroAgent`** automatically detects the UDP LAN ping (`224.0.2.60:4445`) and spawns into your world!

---

## Talking to the Agent in Minecraft Chat

Just press **`T`** in Minecraft and type naturally:
- `"follow me"` / `"come here"` / `"stop"`
- `"gather 8 oak_log"` / `"mine 12 stone"`
- `"craft wooden_pickaxe"` / `"craft crafting_table"`
- `"fight mobs"` / `"defend me"`
- `"give me oak_log"` / `"status"`
- `"!tools"` — Lists all active MCP tools, including newly synthesized skills!
- **Any unique request** (e.g., `"do a celebration dance"`, `"patrol the perimeter"`, `"build a cobble shelter"`): Triggers the **Self-Synthesizing MCP Tool Factory** to write, sandbox-verify, and hot-reload a new MCP skill on the fly!
