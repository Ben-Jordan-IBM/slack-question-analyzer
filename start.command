#!/usr/bin/env bash
# Slack Question Analyzer — double-click start for macOS (after ./setup.sh has run once).
cd "$(dirname "$0")"
exec python3 api_server.py
