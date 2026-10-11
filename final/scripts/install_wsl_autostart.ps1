param([Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_.-]+$')][string]$Distribution)
$ErrorActionPreference = 'Stop'
$taskName = 'EdgeFace WSL runtime'
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$wslPath = Join-Path $env:WINDIR 'System32\wsl.exe'
# An active wsl.exe keeps WSL running; systemd alone does not do so.
$arguments = '-d "' + $Distribution + '" --exec /bin/sh -c "exec sleep infinity"'
$action = New-ScheduledTaskAction -Execute $wslPath -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
    -Description 'Start the EdgeFace WSL distribution at Windows sign-in and keep native Tailscale/backend services available.'
Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
