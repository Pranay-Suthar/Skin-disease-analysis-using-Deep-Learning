@echo off
echo ============================================
echo DermaAI - Frontend + Backend Startup
echo ============================================
echo.

REM Stop any existing processes
taskkill /F /IM node.exe 2>nul
taskkill /F /IM python.exe 2>nul

echo [1/2] Starting Flask Backend on http://127.0.0.1:5000/
start "Flask Backend" cmd /k "cd /d d:\Code-Editors\VS-Code\Python\Skin_App\backend && python app.py"

timeout /t 3

echo [2/2] Starting Frontend Dev Server on http://localhost:5173/
start "Frontend Dev Server" cmd /k "cd /d d:\Code-Editors\VS-Code\Python\Skin_App\frontend && npm run dev"

echo.
echo ============================================
echo Servers Starting...
echo Frontend: http://localhost:5173/
echo Backend:  http://127.0.0.1:5000/
echo ============================================
echo.
pause
