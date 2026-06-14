@echo off
REM ECM Factor Validation Test (batch wrapper)
REM Runs tools\test_validate_factors.ps1 via PowerShell

set ROOT=%~dp0
pushd "%ROOT%"

echo.
echo ============================================
echo   ECM Factor Validation Test
echo ============================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "tools\test_validate_factors.ps1"
set EXITCODE=%ERRORLEVEL%

popd

echo.
echo ============================================
echo   Test complete. Press any key to exit.
echo ============================================
pause
exit /b %EXITCODE%
