$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    python -m venv (Join-Path $Root ".venv")
}
& $Python -m pip install --upgrade pip
& $Python -m pip install -e "${Root}[dev]"
Write-Host "RoverSwarm est prêt. Activez avec: .\.venv\Scripts\Activate.ps1"
