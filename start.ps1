$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

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
& $venvPython main.py
