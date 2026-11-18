@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

title NeuroCraft - 100%% Vanilla Singleplayer AI Companion

echo ======================================================================
echo   NEUROCRAFT: Real-Time Human-Mimicking Minecraft AI Agent
echo   1-Click Vanilla Singleplayer Launcher (Self-Contained)
echo ======================================================================

:: 1. Prefer project-local portable Node.js runtime (.runtime\node)
set "LOCAL_NODE_DIR=%~dp0.runtime\node"
if exist "%LOCAL_NODE_DIR%\node.exe" (
    set "PATH=%LOCAL_NODE_DIR%;%PATH%"
)

where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [Setup] Bootstrapping self-contained portable Node.js v20 LTS into .runtime\node...
    if not exist ".runtime" mkdir ".runtime"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://nodejs.org/dist/v20.18.0/node-v20.18.0-win-x64.zip' -OutFile '.runtime\node.zip'"
    tar.exe -xf ".runtime\node.zip" -C ".runtime"
    if exist ".runtime\node" rmdir /s /q ".runtime\node"
    move ".runtime\node-v20.18.0-win-x64" ".runtime\node" >nul
    del /q ".runtime\node.zip"
    set "PATH=%LOCAL_NODE_DIR%;%PATH%"
)

:: 2. Install dependencies inside project folder if node_modules is missing
if not exist "%~dp0node_modules" (
    echo [Setup] Installing local packages into .\node_modules...
    call npm install --no-fund --no-audit
)

:: 3. Build TypeScript if dist\index.js is missing
if not exist "%~dp0dist\index.js" (
    echo [Setup] Compiling TypeScript architecture...
    call npm run build
)

:: 4. Launch NeuroCraft Agent
echo.
node "%~dp0dist\index.js" %*

pause
