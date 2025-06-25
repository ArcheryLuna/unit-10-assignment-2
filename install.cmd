@echo off
REM ---------------------------------------------------------------------------
REM  install.cmd — Windows CMD installer for the GCSE Analysis project
REM ---------------------------------------------------------------------------
REM  Creates a Python virtual environment (./venv) and installs the packages
REM  required to run gcse_analysis.py.
REM
REM  Usage (from Command Prompt):
REM      install.cmd          — run in project root alongside gcse_analysis.py
REM ---------------------------------------------------------------------------

setlocal enabledelayedexpansion

REM -------- SETTINGS ---------------------------------------------------------
set "VENV_DIR=venv"
set "PACKAGES=pandas numpy scipy matplotlib seaborn"

REM -------- 1. Locate Python 3 ------------------------------------------------
where py  >nul 2>&1
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set "PY=python"
    ) else (
        echo Error: Python 3 not found in PATH. Please install Python 3.9+ and retry.
        exit /b 1
    )
)

REM -------- 2. Create virtual environment -----------------------------------
if exist "%VENV_DIR%" (
    echo Virtual environment "%VENV_DIR%" already exists. Skipping creation.
) else (
    echo Creating virtual environment in .\%VENV_DIR% …
    %PY% -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Ensure that the venv module is available.^
 and that Python 3.9+ is correctly installed.
        exit /b 1
    )
)

REM -------- 3. Upgrade pip & install packages --------------------------------
set "PIP=%VENV_DIR%\Scripts\python.exe -m pip"

echo.
echo Upgrading pip, setuptools, wheel …
%PIP% install --quiet --upgrade pip setuptools wheel

echo Installing project packages: %PACKAGES% …
%PIP% install --quiet %PACKAGES%

REM -------- 4. Freeze exact versions ----------------------------------------
%PIP% freeze > requirements.txt

REM -------- 5. Finished ------------------------------------------------------
echo.
echo ✔ Installation complete.
echo To activate the environment, run:
echo   call %VENV_DIR%\Scripts\activate.bat
echo Then execute the analysis script with:
echo   python gcse_analysis.py

endlocal
