@echo off
setlocal EnableDelayedExpansion
title NeuroCraft - Full Agent ^& Training Loop Reset (Factory Loop 0)

cd /d "%~dp0"

echo ======================================================================
echo   NEUROCRAFT: Full Agent, Replay Buffer ^& Checkpoint Reset (Loop 0)
echo ======================================================================
echo.

if exist "data\synthesized_skills\*.js" (
    echo [Reset] Deleting synthesized MCP skills in data\synthesized_skills\...
    del /q "data\synthesized_skills\*.js"
) else (
    echo [Reset] No synthesized MCP skill files found.
)

if exist "data\synthesized_registry.json" (
    echo [Reset] Deleting data\synthesized_registry.json...
    del /q "data\synthesized_registry.json"
)

if exist "data\episodic_memory.json" (
    echo [Reset] Deleting data\episodic_memory.json...
    del /q "data\episodic_memory.json"
)

if exist "logs\*.log" (
    echo [Reset] Clearing previous session logs in logs\...
    del /q "logs\*.log"
)

echo [Reset] Clearing legacy replay buffers and metrics in minebench\results\...
if exist "minebench\results\rl_replay_buffer.jsonl" del /q "minebench\results\rl_replay_buffer.jsonl"
if exist "minebench\results\real_rl_transitions.jsonl" del /q "minebench\results\real_rl_transitions.jsonl"
if exist "minebench\results\slm_training_metrics.json" del /q "minebench\results\slm_training_metrics.json"
if exist "minebench\results\rl_training_metrics.json" del /q "minebench\results\rl_training_metrics.json"

echo [Reset] Clearing legacy RL, ACT, and SLM LoRA checkpoints in neural_lab\checkpoints\...
if exist "neural_lab\checkpoints\voxel_reflex_policy_loop*.pt" del /q "neural_lab\checkpoints\voxel_reflex_policy_loop*.pt"
if exist "neural_lab\checkpoints\voxel_reflex_policy_latest.pt" del /q "neural_lab\checkpoints\voxel_reflex_policy_latest.pt"
if exist "neural_lab\checkpoints\voxel_reflex_policy_best.pt" del /q "neural_lab\checkpoints\voxel_reflex_policy_best.pt"
if exist "neural_lab\checkpoints\action_head_latest.pt" del /q "neural_lab\checkpoints\action_head_latest.pt"
if exist "neural_lab\checkpoints\action_head_best.pt" del /q "neural_lab\checkpoints\action_head_best.pt"
if exist "neural_lab\checkpoints\slm_minecraft_savant_latest" rmdir /s /q "neural_lab\checkpoints\slm_minecraft_savant_latest"
if exist "neural_lab\checkpoints\slm_minecraft_savant_best" rmdir /s /q "neural_lab\checkpoints\slm_minecraft_savant_best"
for /d %%D in ("neural_lab\checkpoints\slm_minecraft_savant_loop*") do rmdir /s /q "%%D"

echo [Reset] Resetting minebench\results\training_state.json to Loop 0...
set "PYTHON_EXE=python"
if exist "neural_lab\.venv\Scripts\python.exe" set "PYTHON_EXE=neural_lab\.venv\Scripts\python.exe"
"%PYTHON_EXE%" -c "import json, datetime; json.dump({'sessionCount': 0, 'currentIteration': 0, 'totalEpisodesPlayed': 0, 'bestPassRate': 0.0, 'bestMeanScore': 0.0, 'bestIteration': 0, 'latestMeanScore': 0.0, 'latestPassRate': 0.0, 'latestReward': 0.0, 'lastSavedAt': datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'), 'latestCheckpoint': 'voxel_reflex_policy_latest.pt', 'bestCheckpoint': 'voxel_reflex_policy_best.pt', 'history': []}, open('minebench/results/training_state.json', 'w', encoding='utf-8'), indent=2)"

echo [Reset] Regenerating clean DAG-aligned gold curriculum into grpo_verified_trajectories.jsonl...
"%PYTHON_EXE%" "minebench\generate_gold_dataset.py"

if exist ".runtime\node\node.exe" (
    echo [Reset] Rebuilding root and minebench TypeScript source...
    ".runtime\node\node.exe" "node_modules\typescript\bin\tsc" -p "tsconfig.json"
    ".runtime\node\node.exe" "node_modules\typescript\bin\tsc" -p "minebench\tsconfig.json"
    echo [Reset] Refreshing Loop 0 telemetry dashboard...
    ".runtime\node\node.exe" "minebench\dist\minebench\generate_training_graphs.js"
)

echo.
echo ======================================================================
echo  [DONE] NeuroAgent has been wiped clean to Loop 0 factory defaults!
echo  - Legacy replay buffers, metrics, and checkpoints removed
echo  - Clean prerequisite-DAG gold curriculum seeded
echo  - Loop 0 training_state.json and 4-panel dashboard refreshed
echo  You can now run train_agent.bat when plugged in for a clean Iteration #1.
echo ======================================================================
echo.
if "%1"=="--no-pause" goto :eof
pause
