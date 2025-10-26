@echo off
REM Quick run script for Travel Destination Recommendation System

echo Starting Travel Recommendation System...
echo.

REM Check if virtual environment exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run Streamlit app
echo Opening application in browser...
echo Press Ctrl+C to stop the application
echo.
streamlit run app.py

pause
