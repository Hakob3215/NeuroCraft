@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

title NeuroCraft - RTX 5060 All-Neural Local AI Stack

echo ======================================================================
echo   NEUROCRAFT: RTX 5060 Hybrid Neural + Gemini Flash Launcher
echo   - System 1 (RTX 5060 GPU): 2ms 3D Voxel Reflexes + 5ms 64^3 Scanner
echo   - System 2 (Brain): ^<1ms Fast-Path + Gemini 3.8 Flash (~2s)
echo ======================================================================

:: 1. Start RTX 5060 Neural Daemon in background if not already listening on 8765
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8765/health' -TimeoutSec 1 -UseBasicParsing; exit 0 } catch { exit 1 }"
if %ERRORLEVEL% neq 0 (
    echo [GPU Daemon] Starting RTX 5060 Local Neural Daemon on http://127.0.0.1:8765...
    start "NeuroCraft RTX 5060 Neural Daemon" /min "%~dp0neural_lab\.venv\Scripts\python.exe" "%~dp0neural_lab\neural_daemon.py"
    echo [GPU Daemon] Waiting for RTX 5060 models to load into VRAM...
    for /L %%i in (1,1,60) do (
        powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8765/health' -TimeoutSec 1 -UseBasicParsing; exit 0 } catch { exit 1 }"
        if !ERRORLEVEL! equ 0 goto :daemon_ready
        timeout /t 2 /nobreak >nul
    )
)
:daemon_ready
echo [GPU Daemon] RTX 5060 Neural Stack is ONLINE!

:: 2. Configure Hybrid Mode: Gemini 3.8 Flash primary for System 2 + RTX 5060 GPU for System 1 & offline fallback
set "LLM_PROVIDER=gemini"
set "NEURAL_ALLOW_CLOUD_FALLBACK=true"

:: 3. Launch NeuroCraft Agent
call "%~dp0run_singleplayer.bat" %*
