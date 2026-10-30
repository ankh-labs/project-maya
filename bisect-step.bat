@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bisect-step.ps1"
exit /b %ERRORLEVEL%
