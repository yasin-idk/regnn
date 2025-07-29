@echo off
title ML Regression Dashboard - Auto Launcher
echo.
echo ====================================================================
echo    ML REGRESSION DASHBOARD - WINDOWS LAUNCHER
echo ====================================================================
echo.
echo Starting the ML Regression Dashboard...
echo This will automatically install all required packages.
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Run the Python launcher
python start.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause >nul
) 