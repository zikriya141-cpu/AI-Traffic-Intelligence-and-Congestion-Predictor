@echo off
title Traffic Congestion System Server
color 0A
echo ============================================================
echo   Intelligent Traffic Congestion Prediction System
echo.
echo   The browser will open automatically at:  http://localhost:8000
echo   If it does not open, type this manually:  localhost:8000
echo.
echo   NOTE: Do NOT type 0.0.0.0 - it will not work in a browser
echo ============================================================
echo.

cd /d "%~dp0"
cd backend

echo Checking if port 8000 is already in use...
netstat -ano | findstr ":8000" | findstr "LISTENING"
if %errorlevel%==0 (
    echo.
    echo !! ERROR: Port 8000 is already in use !!
    echo This means the server is already running.
    echo First run 'stop.bat', then run this file again.
    echo.
    pause
    exit /b 1
)

echo Port is free. Starting the server...
echo Please wait 6-10 seconds while the model loads, then the browser will open.
echo.
echo --- Opening the browser automatically ---
start "" /min cmd /c "timeout /t 7 /nobreak >nul & start http://localhost:8000"

python -m uvicorn main:app --host 0.0.0.0 --port 8000

echo.
echo Server stopped.
pause