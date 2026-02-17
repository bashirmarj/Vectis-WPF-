@echo off
REM AAG Feature Recognition Viewer Launcher for Windows
REM Opens web-based viewer at http://localhost:8080

cd /d "%~dp0"
echo ============================================================
echo AAG Feature Recognition Viewer
echo ============================================================
echo Starting local viewer...
echo.

"C:\Users\BASH\miniconda3\python.exe" run_local.py

pause
