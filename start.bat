@echo off
rem Double-click this file to compile (if needed), train (first time only) and chat.
cd /d "%~dp0"

gcc -std=c11 -O2 -Isrc src/taskai.c src/matrix.c -o taskai.exe -lm
if errorlevel 1 (
    echo.
    echo Compile failed. Make sure gcc is installed and the files are saved.
    pause
    exit /b 1
)

if not exist model.tai taskai.exe train

taskai.exe chat
pause
