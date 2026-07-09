#!/usr/bin/env bash
# Slack Question Analyzer - double-click start on macOS (after ./setup.sh once).
cd "$(dirname "$0")"

# Fail with a pointer instead of a raw traceback when the install is
# missing or a git pull added a dependency
if ! python3 -c "import slack_question_analyzer" 2>/dev/null; then
    echo "The analyzer is not installed (or an update added new dependencies)."
    echo "Run ./setup.sh once, or:  python3 -m pip install -e ."
    echo "If that fails too, run:   python3 -m slack_question_analyzer.cli doctor"
    read -r -p "Press Enter to close..."
    exit 1
fi
exec python3 api_server.py
