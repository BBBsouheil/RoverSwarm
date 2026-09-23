$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Model = Join-Path $Root "models\roverswarm_v5_seed97.zip"
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Temp = Join-Path $Root "artifacts\test-temp-release-$Stamp"
$Capture = Join-Path $Root "artifacts\results\release-smoke.png"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Environnement absent. Lancez d'abord scripts\setup.ps1."
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Modèle final absent: $Model"
}

Push-Location $Root
try {
    & $Python -m pytest -p no:cacheprovider --basetemp $Temp
    if ($LASTEXITCODE -ne 0) { throw "Les tests ont échoué." }

    & $Python -m roverswarm.mission_control --scenario survey --no-briefing --steps 20 --screenshot $Capture
    if ($LASTEXITCODE -ne 0) { throw "La démonstration automatisée a échoué." }
}
finally {
    Pop-Location
}

Write-Host "RoverSwarm 1.0 validé: tests, modèle et rendu Mission Control."
