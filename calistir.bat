@echo off
call "%~dp0venv\Scripts\activate.bat"
if "%~1"=="" (
    python try_model1.py
) else (
    python %*
)
