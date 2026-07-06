@echo off
cd /d "%~dp0"
if "%STABLEBOY_PASS%"=="" (
    echo STABLEBOY_PASS is not set. Run this once in PowerShell, then restart your terminal:
    echo   setx STABLEBOY_PASS "your-password-here"
    pause
    exit /b 1
)
poetry run python web_app.py
