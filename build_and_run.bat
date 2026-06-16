@echo off
REM Launcher for build_and_run.ps1 (bypasses ExecutionPolicy, forwards all args).
REM Examples:
REM   build_and_run.bat
REM   build_and_run.bat -Reconfigure
REM   build_and_run.bat -ShowKernel
REM   build_and_run.bat -SkipBuild -ExtraArgs "--special-mult generic"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_and_run.ps1" %*
echo.
echo (exit code: %ERRORLEVEL%)
pause
