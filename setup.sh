#!/usr/bin/env bash
# Slack Question Analyzer — one-command setup for macOS and Linux.
#   ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Slack Question Analyzer setup ==="

# 1. Python
PYTHON=python3
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: Python 3 is not installed. Install 3.10+ from https://python.org" >&2
    exit 1
fi
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "ERROR: Python 3.10+ is required (found $("$PYTHON" -V))." >&2
    exit 1
fi
echo "[OK] $("$PYTHON" -V)"

# 2. Install the package (old bundled pips can't editable-install a
# pyproject-only package, so upgrade pip first)
echo "Installing the analyzer (this can take a few minutes the first time)..."
"$PYTHON" -m pip install --quiet --upgrade pip || echo "[warn] Could not upgrade pip - continuing"
if ! "$PYTHON" -m pip install --quiet -e .; then
    echo "ERROR: package install failed (see pip's message above)." >&2
    echo "On a corporate network this is usually the proxy: set HTTPS_PROXY," >&2
    echo "or run: $PYTHON -m pip install --proxy <your-proxy> -e ." >&2
    exit 1
fi
echo "[OK] Package installed"

# 3. Ollama
if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is not installed. Install it with:" >&2
    echo "    curl -fsSL https://ollama.com/install.sh | sh     # Linux" >&2
    echo "    or download from https://ollama.com/download      # macOS" >&2
    echo "Then run this script again." >&2
    exit 1
fi
echo "[OK] Ollama installed"

# Make sure the Ollama server is running — poll until it answers instead
# of hoping a fixed sleep was enough. Probe with curl or wget, whichever
# this machine has; fall back to `ollama list` (talks to the same server).
if command -v curl >/dev/null 2>&1; then
    ollama_up() { curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null; }
elif command -v wget >/dev/null 2>&1; then
    ollama_up() { wget -q -T 3 -O /dev/null http://localhost:11434/api/tags; }
else
    ollama_up() { ollama list >/dev/null 2>&1; }
fi
if ! ollama_up; then
    echo "Starting Ollama..."
    (ollama serve >/dev/null 2>&1 &)
    up=""
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        sleep 2
        if ollama_up; then up=1; break; fi
    done
    if [ -z "$up" ]; then
        echo "ERROR: Ollama did not come up after 20 seconds." >&2
        echo "Start it manually (ollama serve) and run this script again." >&2
        exit 1
    fi
fi
echo "[OK] Ollama running"

# 4. Pull the models (idempotent; skips anything already downloaded).
# Chat model is sized to the machine: 8B on >=12GB RAM, 3B otherwise.
if [ "$(uname)" = "Darwin" ]; then
    ram_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
else
    ram_gb=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0) / 1048576 ))
fi
if [ "$ram_gb" -ge 12 ]; then
    chat_model="llama3.1:8b"
    echo "Detected ${ram_gb}GB RAM - using the larger chat model for better topic names."
    echo "Downloading models (first time only: ~270MB + ~5GB + ~2GB)..."
else
    chat_model="llama3.2"
    echo "Detected ${ram_gb}GB RAM - using the compact chat model."
    echo "Downloading models (first time only: ~270MB + ~2GB)..."
fi
# Free-disk check where the models actually land: OLLAMA_MODELS if set,
# otherwise ~/.ollama
model_dir="${OLLAMA_MODELS:-$HOME}"
needed_gb=8
[ "$ram_gb" -lt 12 ] && needed_gb=4
free_gb=$(df -Pk "$model_dir" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}')
if [ -n "$free_gb" ] && [ "$free_gb" -lt "$needed_gb" ]; then
    echo "ERROR: only ${free_gb}GB free in $model_dir - the models need ~${needed_gb}GB." >&2
    echo "Free up disk space and run this script again." >&2
    exit 1
fi

pull_model() {
    if ! ollama pull "$1"; then
        echo "ERROR: downloading '$1' failed (see Ollama's message above)." >&2
        echo "Check your network and disk space, then run this script again -" >&2
        echo "it resumes where it left off." >&2
        exit 1
    fi
}
pull_model nomic-embed-text
pull_model "$chat_model"
if [ "$chat_model" != "llama3.2" ]; then
    # The fast model: token-heavy extraction on large transcripts goes to
    # the 3B while the 8B handles the judgment calls
    pull_model llama3.2
fi
echo "[OK] Models ready"

# 5. Launch — the dashboard opens in your browser automatically (the server
# picks the next free port if 5000 is taken, e.g. by macOS AirPlay)
echo
echo "Starting the analyzer (the dashboard opens automatically; URL shown below)..."
exec "$PYTHON" api_server.py
