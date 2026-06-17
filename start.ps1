$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$port = if ($env:VIDEODECK_PORT) { [int]$env:VIDEODECK_PORT } else { 8000 }

# Prefer py launcher, then fallback to python.
$pythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
    $pythonArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
    $pythonArgs = @()
} else {
    Write-Host "Python not found. Install Python 3.11+ and retry."
    exit 1
}

if (-not (Test-Path ".venv")) {
    & $pythonCmd @pythonArgs -m venv .venv
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment Python not found at .venv\Scripts\python.exe"
    exit 1
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

# Stop an already running server process on this port.
try {
    $existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty OwningProcess
    if ($existing) {
        Write-Host "Stopping existing server PID $existing on port $port"
        Stop-Process -Id $existing -Force
    }
} catch {
    # Ignore cleanup failures and continue startup.
}

& $venvPython main.py
