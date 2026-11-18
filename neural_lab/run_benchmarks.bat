@echo off
setlocal
cd /d "%~dp0"

echo ======================================================================
echo   NEUROCRAFT AI LAB: RTX 5060 BENCHMARK RUNNER
echo ======================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in neural_lab\.venv!
    echo Please make sure the venv is created and PyTorch is installed.
    pause
    exit /b 1
)

echo [Running] Executing benchmark_runner.py on NVIDIA GeForce RTX 5060...
.venv\Scripts\python.exe benchmark_runner.py

echo.
echo ======================================================================
echo   Benchmark Complete!
echo   Results saved to: neural_lab\benchmark_results.txt
echo ======================================================================
pause
