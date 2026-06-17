@echo off
REM Slack Question Analyzer - double-click start (after setup.bat has run once).
cd /d "%~dp0"
python api_server.py
pause
