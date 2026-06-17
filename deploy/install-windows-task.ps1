# Register a Windows Scheduled Task to start daytrader-bot on weekday mornings.
# Run once in PowerShell (as your user):  .\deploy\install-windows-task.ps1
#
# Requires: PC on and logged in (or task configured to run whether logged on or not).

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$StartScript = Join-Path $PSScriptRoot "start-daytrader.ps1"
$TaskName = "daytrader-bot"

if (-not (Test-Path $StartScript)) {
    throw "Missing $StartScript"
}

# 3:50 AM Pacific = 6:50 AM Eastern (10 min before premarket research at 07:00 ET).
# Task Scheduler uses your PC's local clock (Pacific).
$TriggerTime = "03:50"

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$StartScript`"" `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $TriggerTime

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Starts daytrader-bot scheduler (research, trading cycles, EOD report)." `
    -Force

Write-Host "Registered scheduled task: $TaskName"
Write-Host "  Runs: Mon-Fri at $TriggerTime (local PC clock)"
Write-Host "  Script: $StartScript"
Write-Host ""
Write-Host "Test now:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "Remove:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
