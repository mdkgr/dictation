@echo off
title Greek Dictation
cd /d "%~dp0"
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do set "%%A=%%B"
)
python dictate.py
pause
