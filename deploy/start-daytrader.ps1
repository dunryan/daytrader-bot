# Start the daytrader-bot scheduler (set-and-forget daily lifecycle).
# Run manually or via Task Scheduler (see install-windows-task.ps1).

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
}

Set-Location $ProjectRoot
$env:GIT_SSH = "C:\Program Files\Git\usr\bin\ssh.exe"

Write-Host "Starting daytrader-bot from $ProjectRoot"
Write-Host "Schedule is in config/config.yaml (America/New_York)."
Write-Host "Press Ctrl+C to stop."
& $Python main.py
