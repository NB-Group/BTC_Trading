@echo off
setlocal enabledelayedexpansion

REM Prefer cz if available
where cz >NUL 2>&1
if %ERRORLEVEL%==0 (
    cz check --commit-msg-file %1
    exit /b %ERRORLEVEL%
)

REM Check if python has commitizen module before running
where python >NUL 2>&1
if %ERRORLEVEL%==0 (
    python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('commitizen') else 1)" >NUL 2>&1
    if %ERRORLEVEL%==0 (
        python -m commitizen check --commit-msg-file %1
        exit /b %ERRORLEVEL%
    ) else (
        echo [commit-msg] commitizen 未安装，跳过严格校验。建议: pip install commitizen
        exit /b 0
    )
)

echo [commit-msg] 未找到 Python，跳过严格校验。
exit /b 0

