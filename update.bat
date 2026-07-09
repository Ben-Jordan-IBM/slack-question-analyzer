@echo off
REM Slack Question Analyzer - get the latest version (double-click).
REM Pulls new code AND reinstalls dependencies: a pull alone breaks the app
REM whenever an update adds a dependency.
cd /d "%~dp0"
set PYTHONUTF8=1

REM Same interpreter pick as start.bat ('python' may be the Store stub)
set "PY=python"
python -c "import sys" >nul 2>nul
if errorlevel 1 set "PY=py"

echo === Updating Slack Question Analyzer ===
REM A zip-download copy has no git and no repo - say so instead of a
REM misleading pull error
where git >nul 2>nul
if errorlevel 1 (
    echo Git is required to update. Install it from https://git-scm.com/download/win
    echo ^(or re-download the project as a zip and run setup.bat again^).
    pause
    exit /b 1
)
if not exist ".git" (
    echo This folder is not a git checkout ^(zip download?^), so it cannot self-update.
    echo Re-download the project and run setup.bat again.
    pause
    exit /b 1
)
REM Everything from the pull onward lives on ONE line: cmd reads batch files
REM by byte offset, and git pull rewrites this very file - a multi-line tail
REM would resume mid-line inside the NEW file's bytes.
git pull && %PY% -m pip install --quiet -e . && echo [OK] Updated. Start the app with start.bat ^(restart it if it is running^). || echo Update FAILED - see the message above. If you edited tracked files like taxonomy.json, point TAXONOMY_PATH in .env at a copy instead. & pause
