@echo off
setlocal EnableDelayedExpansion
title NeuroCraft - Full Agent Reset (Wipe Synthesized Skills, Memory ^& Logs)

cd /d "%~dp0"

echo ======================================================================
echo   NEUROCRAFT: Full Agent State ^& Synthesized MCP Skill Reset
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

if exist ".runtime\node\node.exe" (
    echo [Reset] Rebuilding TypeScript source to dist\...
    ".runtime\node\node.exe" "node_modules\typescript\bin\tsc"
)

echo.
echo ======================================================================
echo  [DONE] NeuroAgent has been wiped clean to factory defaults!
echo  - All custom/synthesized MCP skills removed
echo  - Episodic memories cleared
echo  - Old logs cleared
echo  You can now double-click run_singleplayer.bat for a fresh test run.
echo ======================================================================
echo.
if "%1"=="--no-pause" goto :eof
pause
