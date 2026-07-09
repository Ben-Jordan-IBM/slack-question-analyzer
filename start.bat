@echo off
REM Slack Question Analyzer - double-click start (after setup.bat has run once).
cd /d "%~dp0"
REM Slack text is full of emoji; UTF-8 keeps redirected output from crashing
set PYTHONUTF8=1

REM Pick the interpreter the same way setup.ps1 does: 'python' may be the
REM Microsoft Store stub, in which case the py launcher is the real one
set "PY=python"
python -c "import sys" >nul 2>nul
if errorlevel 1 set "PY=py"

REM Fail with a pointer instead of a raw traceback when the install is
REM missing or a git pull added a dependency
%PY% -c "import slack_question_analyzer" 2>nul
if errorlevel 1 (
    echo The analyzer is not installed ^(or an update added new dependencies^).
    echo Run setup.bat once, or:  %PY% -m pip install -e .
    echo If that fails too, run:  %PY% -m slack_question_analyzer.cli doctor
    pause
    exit /b 1
)
%PY% api_server.py
pause
