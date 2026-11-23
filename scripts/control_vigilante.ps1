<#
.SYNOPSIS
    Controla el vigilante multi-carpeta en segundo plano sin ventana de terminal (`pythonw.exe`).

.EXAMPLE
    # Iniciar en segundo plano en MODO SEGURO (Dry-Run) vigilando Screenshots + Downloads
    .\scripts\control_vigilante.ps1 start

    # Ver estado, uso de RAM y últimas líneas del log
    .\scripts\control_vigilante.ps1 status

    # Detener el vigilante en segundo plano
    .\scripts\control_vigilante.ps1 stop

    # Iniciar en MODO REAL (mueve archivos con confianza >= 80%)
    .\scripts\control_vigilante.ps1 start -Execute -Threshold 0.80

    # Registrar inicio automático al prender Windows (opcional)
    .\scripts\control_vigilante.ps1 install-startup -Execute

    # Quitar inicio automático
    .\scripts\control_vigilante.ps1 uninstall-startup
#>

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("start", "stop", "status", "install-startup", "uninstall-startup")]
    [string]$Action,

    [ValidateSet("pc", "inbox")]
    [string]$Profile = "pc",

    [switch]$Execute,

    [double]$Threshold = 0.80,

    [string]$DestRoot = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PidFile = Join-Path $ProjectRoot "logs\watcher.pid"
$LogFile = Join-Path $ProjectRoot "logs\organizer.log"
$TaskName = "OrganizadorMultimediaIA"

# Localizar pythonw.exe de forma portable (.venv -> PATH -> instalaciones estándar)
$PythonW = $null
$VenvPythonW = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
if (Test-Path $VenvPythonW) {
    $PythonW = $VenvPythonW
}
if (-not $PythonW) {
    $allCmds = Get-Command pythonw.exe -All -ErrorAction SilentlyContinue
    $realCmd = $allCmds | Where-Object { $_.Source -notlike "*\WindowsApps\*" } | Select-Object -First 1
    if ($realCmd) {
        $PythonW = $realCmd.Source
    } elseif ($allCmds) {
        $PythonW = ($allCmds | Select-Object -First 1).Source
    }
}
if (-not $PythonW) {
    foreach ($ver in @("Python314", "Python313", "Python312", "Python311", "Python310")) {
        $cand = "C:\$ver\pythonw.exe"
        if (Test-Path $cand) { $PythonW = $cand; break }
    }
}
if (-not $PythonW) {
    throw "No se encontró pythonw.exe. Instala Python 3.10+ y agrégalo al PATH."
}

function Build-WatcherArgs {
    $argsList = @(
        "-m", "src.organizer_watcher",
        "--profile", $Profile,
        "--on-demand",
        "--threshold", $Threshold.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    )
    if ($Execute) {
        $argsList += "--no-dry-run"
    } else {
        $argsList += "--dry-run"
    }
    if ($DestRoot -and $DestRoot.Trim() -ne "") {
        $argsList += @("--dest-root", $DestRoot)
    }
    return $argsList
}

function Get-RunningWatcherProcess {
    if (-not (Test-Path $PidFile)) {
        return $null
    }
    $pidText = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $pidText -or -not ($pidText -match "^\d+$")) {
        return $null
    }
    $proc = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
    if ($proc -and $proc.ProcessName -like "python*") {
        return $proc
    }
    return $null
}

switch ($Action) {
    "start" {
        $existing = Get-RunningWatcherProcess
        if ($existing) {
            Write-Host "El vigilante ya está corriendo (PID: $($existing.Id)). Usa 'stop' primero si deseas reiniciarlo." -ForegroundColor Yellow
            exit 0
        }

        $argList = Build-WatcherArgs
        $modeLabel = if ($Execute) { "EJECUCION REAL (>= $([math]::Round($Threshold*100))%)" } else { "DRY-RUN (MODO SEGURO)" }

        Start-Process -FilePath $PythonW -ArgumentList $argList -WorkingDirectory $ProjectRoot -WindowStyle Hidden
        Start-Sleep -Seconds 2

        $started = Get-RunningWatcherProcess
        if ($started) {
            $ramMb = [math]::Round($started.WorkingSet64 / 1MB, 1)
            Write-Host "[OK] Vigilante iniciado en segundo plano sin terminal." -ForegroundColor Green
            Write-Host "     PID:    $($started.Id)"
            Write-Host "     Perfil: $Profile | Modo: $modeLabel | RAM actual: $ramMb MB"
            Write-Host "     Log:    $LogFile"
        } else {
            Write-Host "[!] El proceso se lanzó, revisa $LogFile si no aparece activo." -ForegroundColor Yellow
        }
    }

    "stop" {
        $proc = Get-RunningWatcherProcess
        if (-not $proc) {
            Write-Host "El vigilante no estaba corriendo." -ForegroundColor Yellow
            if (Test-Path $PidFile) { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue }
            exit 0
        }
        Stop-Process -Id $proc.Id -Force
        Start-Sleep -Milliseconds 400
        if (Test-Path $PidFile) { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue }
        Write-Host "[OK] Vigilante detenido (PID $($proc.Id))." -ForegroundColor Green
    }

    "status" {
        $proc = Get-RunningWatcherProcess
        if ($proc) {
            $ramMb = [math]::Round($proc.WorkingSet64 / 1MB, 1)
            Write-Host "[ACTIVO] Vigilante corriendo en segundo plano | PID: $($proc.Id) | RAM: $ramMb MB" -ForegroundColor Green
        } else {
            Write-Host "[INACTIVO] El vigilante no está corriendo." -ForegroundColor Yellow
        }

        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            Write-Host "Inicio automático de Windows: INSTALADO ($($task.State))" -ForegroundColor Cyan
        } else {
            Write-Host "Inicio automático de Windows: No instalado" -ForegroundColor DarkGray
        }

        if (Test-Path $LogFile) {
            Write-Host "`n--- Últimas 10 líneas de organizer.log ---" -ForegroundColor Cyan
            Get-Content $LogFile -Tail 10
        }
    }

    "install-startup" {
        $argList = Build-WatcherArgs
        $quotedArgs = ($argList | ForEach-Object { if ($_ -match "\s") { "`"$_`"" } else { $_ } }) -join " "
        $taskAction = New-ScheduledTaskAction -Execute $PythonW -Argument $quotedArgs -WorkingDirectory $ProjectRoot
        $taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
        $taskSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Days 0)

        Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $taskTrigger -Settings $taskSettings -Force | Out-Null
        $modeLabel = if ($Execute) { "EJECUCION REAL" } else { "DRY-RUN (MODO SEGURO)" }
        Write-Host "[OK] Tarea programada '$TaskName' instalada para iniciar automáticamente con Windows." -ForegroundColor Green
        Write-Host "     Perfil: $Profile | Modo: $modeLabel | Modo RAM: On-Demand (~25 MB en reposo)"
        Write-Host "     Para quitarla en cualquier momento: .\scripts\control_vigilante.ps1 uninstall-startup"
    }

    "uninstall-startup" {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "[OK] Tarea programada '$TaskName' eliminada." -ForegroundColor Green
        } else {
            Write-Host "La tarea programada '$TaskName' no estaba instalada." -ForegroundColor Yellow
        }
    }
}
