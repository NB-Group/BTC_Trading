@echo off
setlocal enabledelayedexpansion

REM Prefer cz if available
where cz >NUL 2>&1
if %ERRORLEVEL%==0 (
    cz check --commit-msg-file %1
    exit /b %ERRORLEVEL%
)

REM Try python -m commitizen
where python >NUL 2>&1
if %ERRORLEVEL%==0 (
    python -m commitizen check --commit-msg-file %1
    exit /b %ERRORLEVEL%
)

echo [commit-msg] commitizen 未安装，跳过严格校验。建议: pip install commitizen
exit /b 0

