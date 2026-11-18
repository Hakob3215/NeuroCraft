@echo off
title NEUROCRAFT: Autonomous Indefinite Training & Simulation Loop (Ctrl+C to Stop)
cls
echo ===========================================================================
echo   NEUROCRAFT: AUTONOMOUS INDEFINITE TRAINING & SIMULATION ENGINE
echo   - Exhaustive physical simulation in Mojang server.jar (1.20.4)
echo   - System 2 QLoRA fine-tuning + System 1 80.8M Action Head optimization
echo   - Clean Ctrl+C Graceful Teardown: Saves checkpoints & stops all servers
echo ===========================================================================
echo.
call train_agent.bat %*
