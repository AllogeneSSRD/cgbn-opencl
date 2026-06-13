@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ecm_benchmark.ps1" %1 %2
pause
