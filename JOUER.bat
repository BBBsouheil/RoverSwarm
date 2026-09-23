@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo RoverSwarm n'est pas encore installe.
    echo Lancez d'abord scripts\setup.ps1 depuis PowerShell.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m roverswarm.mission_control
if errorlevel 1 (
    echo.
    echo RoverSwarm s'est arrete avec une erreur.
    pause
)
