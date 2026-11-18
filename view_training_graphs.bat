@echo off
title NEUROCRAFT: Training Graphs & Convergence Visualizer
cls
echo [GraphGen] Updating latest training graphs and SVG curves...
.\.runtime\node\node.exe .\minebench\dist\minebench\generate_training_graphs.js
echo.
echo [Dashboard] Opening Interactive Training Dashboard in default browser...
start "" "%~dp0minebench\results\training_dashboard.html"
echo.
echo Done!
