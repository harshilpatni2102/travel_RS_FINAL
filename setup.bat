@echo off
REM Setup script for Travel Destination Recommendation System
REM Windows PowerShell/CMD version

echo ========================================
echo Travel Recommendation System - Setup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Python detected successfully
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo Virtual environment created successfully
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [5/5] Installing dependencies...
echo This may take several minutes (downloading ML models)...
pip install -r requirements.txt
echo.

REM Check data file
echo Checking for dataset...
if exist "data\Worldwide-Travel-Cities-Dataset-Ratings-and-Climate.csv" (
    echo ✓ Dataset found successfully
) else (
    echo ⚠ WARNING: Dataset not found in data/ folder
    echo Please ensure the CSV file is in the correct location
)
echo.

echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To run the application:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run Streamlit: streamlit run app.py
echo.
echo Or simply run: run_app.bat
echo.
pause
