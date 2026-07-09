# Slack Question Analyzer — one-command setup for Windows.
# Right-click > Run with PowerShell, or:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
# Console + child processes speak UTF-8: Slack text is full of emoji, and
# redirected output would otherwise crash on cp1252
$env:PYTHONUTF8 = "1"

Write-Host "=== Slack Question Analyzer setup ===" -ForegroundColor Cyan

function Fail($message, $hint) {
    Write-Host $message -ForegroundColor Red
    if ($hint) { Write-Host $hint }
    Read-Host "Press Enter to exit"
    exit 1
}

# 1. Python. Probe FUNCTIONALLY, not by path: the Microsoft Store alias
# stub lives in WindowsApps but so does genuine Store-installed Python —
# only actually running `-c` tells them apart (the stub exits without
# executing it). Routed through cmd so a broken interpreter's stderr can't
# become a script-terminating NativeCommandError under EAP=Stop on
# Windows PowerShell 5.1.
$py = $null
$pyVersion = $null
foreach ($candidate in @("python", "py")) {
    if (-not (Get-Command $candidate -ErrorAction SilentlyContinue)) { continue }
    $probe = cmd /c "$candidate -c ""import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))"" 2>nul"
    if ($LASTEXITCODE -eq 0 -and $probe) {
        $py = $candidate
        $pyVersion = "$probe".Trim()
        break
    }
}
if (-not $py) {
    Fail "Python is not installed (or only the Microsoft Store stub is on PATH)." `
         "Install it from https://www.python.org/downloads/ (3.10+), check 'Add python.exe to PATH' during install, then run this script again."
}
if ([version]$pyVersion -lt [version]"3.10") {
    Fail "Python $pyVersion found, but 3.10+ is required. Update from https://python.org"
}
Write-Host "[OK] Python $pyVersion (via '$py')"

# 2. Install the package. Old bundled pips can't editable-install a
# pyproject-only package, so upgrade pip first — and actually CHECK the
# exit codes ($ErrorActionPreference does not stop on native commands).
Write-Host "Installing the analyzer (this can take a few minutes the first time)..."
& $py -m pip install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "[warn] Could not upgrade pip - continuing with the current version"
}
& $py -m pip install --quiet -e .
if ($LASTEXITCODE -ne 0) {
    Fail "Package install failed (see pip's message above)." `
         "On a corporate network this is usually the proxy: set HTTPS_PROXY, or run  $py -m pip install --proxy <your-proxy> -e ."
}
Write-Host "[OK] Package installed"

# 3. Ollama
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Fail "Ollama is not installed. Download and run the installer from:  https://ollama.com/download" `
         "Then run this script again."
}
Write-Host "[OK] Ollama installed"

# Make sure the Ollama server is running — poll until it answers instead
# of hoping a fixed sleep was enough
function Test-Ollama {
    try {
        Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null
        return $true
    } catch { return $false }
}
if (-not (Test-Ollama)) {
    Write-Host "Starting Ollama..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    $up = $false
    foreach ($attempt in 1..10) {
        Start-Sleep -Seconds 2
        if (Test-Ollama) { $up = $true; break }
    }
    if (-not $up) {
        Fail "Ollama did not come up after 20 seconds." `
             "Start the Ollama app from the Start menu (it sits in the system tray), then run this script again."
    }
}
Write-Host "[OK] Ollama running"

# 4. Pull the models (idempotent; skips anything already downloaded).
# Chat model is sized to the machine: 8B on >=12GB RAM, 3B otherwise.
$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
if ($ramGB -ge 12) {
    $chatModel = "llama3.1:8b"
    $neededGB = 8
    Write-Host "Detected ${ramGB}GB RAM - using the larger chat model for better topic names."
    Write-Host "Downloading models (first time only: ~270MB + ~5GB + ~2GB)..."
} else {
    $chatModel = "llama3.2"
    $neededGB = 4
    Write-Host "Detected ${ramGB}GB RAM - using the compact chat model."
    Write-Host "Downloading models (first time only: ~270MB + ~2GB)..."
}
# Models land under OLLAMA_MODELS (default %USERPROFILE%\.ollama), NOT the
# repo folder — check the drive they'll actually be written to. Skip the
# check on exotic layouts (UNC paths etc.) rather than dying on it.
try {
    $modelDir = if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS }
                else { Join-Path $env:USERPROFILE ".ollama" }
    $drive = (Split-Path -Qualifier $modelDir).TrimEnd(':')
    $freeGB = [math]::Round((Get-PSDrive -Name $drive).Free / 1GB)
    if ($freeGB -lt $neededGB) {
        Fail "Only ${freeGB}GB free on drive ${drive}: - the models need ~${neededGB}GB there." `
             "Free up disk space and run this script again."
    }
} catch {
    Write-Host "[skip] Could not check free disk space - continuing"
}
function Pull-Model($model) {
    ollama pull $model
    if ($LASTEXITCODE -ne 0) {
        Fail "Downloading '$model' failed (see Ollama's message above)." `
             "Check your network connection and disk space, then run this script again - it resumes where it left off."
    }
}
Pull-Model nomic-embed-text
Pull-Model $chatModel
if ($chatModel -ne "llama3.2") {
    # The fast model: token-heavy extraction on large transcripts goes to
    # the 3B while the 8B handles the judgment calls
    Pull-Model llama3.2
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

# 6. Launch — the dashboard opens in your browser automatically (the server
# picks the next free port if 5000 is taken)
Write-Host ""
Write-Host "Starting the analyzer (the dashboard opens automatically; URL shown below)..." -ForegroundColor Green
& $py api_server.py
