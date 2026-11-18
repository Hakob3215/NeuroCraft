@echo off
title MINEbench: Multi-Domain Minecraft AI Benchmark & RTX 5060 RL Trainer
echo ===========================================================================
echo   MINEBENCH: Multi-Faceted Minecraft 1.20.4 AI Benchmark ^& RL Suite
echo   - 6 Domains: Agility/Parkour, 3D Architecture, Combat, Homestead,
echo                Redstone/Puzzles, ^& Survival Tech-Tree / Ender Dragon
echo   - Dual-System RL: RTX 5060 Clipped PPO (System 1) + GRPO (System 2)
echo ===========================================================================

set DISABLE_FAST_INTENT=true
set STRICT_LLM_BENCHMARK=true

echo [1/2] Compiling TypeScript ^& MINEbench Suite...
.\.runtime\node\node.exe .\node_modules\typescript\bin\tsc -p .\minebench\tsconfig.json
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] TypeScript compilation failed!
    pause
    exit /b 1
)

echo [2/2] Running MINEbench Evaluation ^& RTX 5060 RL Update...
.\.runtime\node\node.exe .\minebench\dist\minebench\runner.js %*

echo.
echo [Complete] Full report saved to: minebench\results\minebench_report.txt
pause
