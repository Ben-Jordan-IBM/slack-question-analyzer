#!/usr/bin/env bash
# Slack Question Analyzer — get the latest version (macOS/Linux).
# Pulls new code AND reinstalls dependencies: a pull alone breaks the app
# whenever an update adds a dependency.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v git >/dev/null 2>&1 || [ ! -d .git ]; then
    echo "This folder is not a git checkout (zip download?), so it cannot self-update." >&2
    echo "Re-download the project and run ./setup.sh again." >&2
    exit 1
fi

echo "=== Updating Slack Question Analyzer ==="
if git pull && python3 -m pip install --quiet -e .; then
    echo "[OK] Updated. Restart the app if it is running."
else
    echo "Update FAILED - see the message above. If you edited tracked files" >&2
    echo "like taxonomy.json, point TAXONOMY_PATH in .env at a copy instead." >&2
    exit 1
fi
