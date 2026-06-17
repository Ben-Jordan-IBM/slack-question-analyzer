# Slack Question Analyzer — one-command setup for Windows.
# Right-click > Run with PowerShell, or:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Slack Question Analyzer setup ===" -ForegroundColor Cyan

# 1. Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python is not installed. Get it from https://www.python.org/downloads/ (3.10+)," -ForegroundColor Red
    Write-Host "check 'Add python.exe to PATH' during install, then run this script again."
    Read-Host "Press Enter to exit"
    exit 1
}
$pyVersion = (python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if ([version]$pyVersion -lt [version]"3.10") {
    Write-Host "Python $pyVersion found, but 3.10+ is required. Update from https://python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Python $pyVersion"

# 2. Install the package
Write-Host "Installing the analyzer (this can take a few minutes the first time)..."
python -m pip install --quiet -e .
Write-Host "[OK] Package installed"

# 3. Ollama
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Host "Ollama is not installed. Download and run the installer from:" -ForegroundColor Yellow
    Write-Host "    https://ollama.com/download" -ForegroundColor Yellow
    Write-Host "Then run this script again."
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Ollama installed"

# Make sure the Ollama server is running
try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null
} catch {
    Write-Host "Starting Ollama..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 4
}
Write-Host "[OK] Ollama running"

# 4. Pull the models (idempotent; skips anything already downloaded).
# Chat model is sized to the machine: 8B on >=12GB RAM, 3B otherwise.
$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
if ($ramGB -ge 12) {
    $chatModel = "llama3.1:8b"
    Write-Host "Detected ${ramGB}GB RAM - using the larger chat model for better topic names."
    Write-Host "Downloading models (first time only: ~270MB + ~5GB)..."
} else {
    $chatModel = "llama3.2"
    Write-Host "Detected ${ramGB}GB RAM - using the compact chat model."
    Write-Host "Downloading models (first time only: ~270MB + ~2GB)..."
}
ollama pull nomic-embed-text
ollama pull $chatModel
if ($chatModel -ne "llama3.2") {
    # The fast model: token-heavy extraction on large transcripts goes to
    # the 3B while the 8B handles the judgment calls
    ollama pull llama3.2
}
Write-Host "[OK] Models ready"

# 5. Desktop shortcut for daily use (best-effort)
try {
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut(
        [IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'Slack Question Analyzer.lnk'))
    $shortcut.TargetPath = Join-Path $PSScriptRoot 'start.bat'
    $shortcut.WorkingDirectory = $PSScriptRoot
    $shortcut.Save()
    Write-Host "[OK] Desktop shortcut created ('Slack Question Analyzer')"
} catch {
    Write-Host "[skip] Could not create a desktop shortcut (use start.bat instead)"
}

# 6. Launch — the dashboard opens in your browser automatically
Write-Host ""
Write-Host "Starting the analyzer at http://localhost:5000 ..." -ForegroundColor Green
python api_server.py
