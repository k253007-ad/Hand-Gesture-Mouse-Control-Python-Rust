@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Run the setup commands in README.md first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "hand_mouse.py" %*
if errorlevel 1 pause
