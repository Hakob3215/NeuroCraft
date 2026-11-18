@echo off
title NEUROCRAFT: Continuous RL ^& Model Training Studio (RTX 5060)
cls
echo ===========================================================================
echo   NEUROCRAFT: CONTINUOUS REINFORCEMENT LEARNING ^& MODEL TRAINER
echo   - System 1: RTX 5060 PPO Voxel Reflexes (Latest + Best Weights)
echo   - System 2: Local RTX 5060 Qwen2.5-Coder SLM (QLoRA "Minecraft Savant" Adapter)
echo   - World   : Official Mojang Minecraft 1.20.4 server.jar (Real Physics)
echo ===========================================================================
echo.
echo [Workflow]:
echo   1. Runs continuous MINEbench training passes on your local RTX 5060.
echo   2. Updates "voxel_reflex_policy_latest.pt" (System 1) after each pass.
echo   3. Fine-tunes "slm_minecraft_savant_latest" (System 2 QLoRA) on winning trajectories.
echo   4. Hot-reloads updated brain weights dynamically into live Neural Daemon.
echo   5. Press [Ctrl + C] ANYTIME to pause! All progress is saved immediately.
echo   6. Pass --dragon-run to execute Tier 3 Procedural Vanilla Survival DAG.
echo   7. Pass --concurrency N (or --bots N) for parallel spatial arena workers (1-5, default 3).
echo ===========================================================================
echo.

set DISABLE_FAST_INTENT=true
set STRICT_LLM_BENCHMARK=true
set NEURAL_FORCE_LOCAL_SLM=true
set LLM_PROVIDER=local_gpu

rem Enable stochastic temperature sampling for active reinforcement learning exploration
if not defined SLM_TEMPERATURE set SLM_TEMPERATURE=0.7
if not defined SLM_TOP_P set SLM_TOP_P=0.9
if not defined SLM_DO_SAMPLE set SLM_DO_SAMPLE=true

set "CONCURRENCY_ARG="
if defined CONCURRENCY (
    set "CONCURRENCY_ARG=--concurrency %CONCURRENCY%"
    echo [Config] Multi-Bot Concurrency configured via %%CONCURRENCY%%: %CONCURRENCY%
)

echo [1/3] Compiling TypeScript suite...
.\.runtime\node\node.exe .\node_modules\typescript\bin\tsc -p .\tsconfig.json
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Root TypeScript compilation failed!
    exit /b 1
)
.\.runtime\node\node.exe .\node_modules\typescript\bin\tsc -p .\minebench\tsconfig.json
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Minebench TypeScript compilation failed!
    exit /b 1
)

echo [2/3] Checking RTX 5060 Neural Daemon status (http://127.0.0.1:8765)...
curl.exe -s http://127.0.0.1:8765/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [Daemon] Starting RTX 5060 Neural Daemon in background...
    start /B "" ".\neural_lab\.venv\Scripts\python.exe" ".\neural_lab\neural_daemon.py"
    echo [Daemon] Waiting for Qwen2.5-Coder to load into VRAM...
    for /L %%i in (1,1,30) do (
        curl.exe -s http://127.0.0.1:8765/health >nul 2>&1 && goto :daemon_ready
        ping 127.0.0.1 -n 2 >nul
    )
) else (
    echo [Daemon] RTX 5060 Neural Stack is already resident in VRAM!
)
:daemon_ready

if "%1"=="--dragon-run" (
    echo [3/3] Launching Tier 3 Autonomous Vanilla Dragon Survival Run...
    echo.
    .\.runtime\node\node.exe .\minebench\dist\minebench\dragon_survival_run.js %*
) else (
    echo [3/3] Launching Continuous Dual-Brain Training Loop...
    echo.
    .\.runtime\node\node.exe .\minebench\dist\minebench\train_loop.js %CONCURRENCY_ARG% %*
)

echo.
echo ===========================================================================
echo   Training Session Ended. Checkpoints saved to:
echo     - System 1 Reflex: neural_lab\checkpoints\voxel_reflex_policy_latest.pt
echo     - System 2 Savant: neural_lab\checkpoints\slm_minecraft_savant_latest\
echo     - State Telemetry: minebench\results\training_state.json
echo ===========================================================================
rem Done. Return cleanly so automated runners capture final exit code.
exit /b 0
