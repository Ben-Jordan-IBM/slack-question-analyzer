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

# 2. Install the package
echo "Installing the analyzer (this can take a few minutes the first time)..."
"$PYTHON" -m pip install --quiet -e .
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

# Make sure the Ollama server is running
if ! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null; then
    echo "Starting Ollama..."
    (ollama serve >/dev/null 2>&1 &)
    sleep 4
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
ollama pull nomic-embed-text
ollama pull "$chat_model"
if [ "$chat_model" != "llama3.2" ]; then
    # The fast model: token-heavy extraction on large transcripts goes to
    # the 3B while the 8B handles the judgment calls
    ollama pull llama3.2
fi
echo "[OK] Models ready"

# 5. Launch — the dashboard opens in your browser automatically
echo
echo "Starting the analyzer at http://localhost:5000 ..."
exec "$PYTHON" api_server.py
