@echo off
REM Keith Ngamphon McKenzie
REM keith@mckenzie.page
REM https://mckenzie.page
REM Python Simple Statistical Tests

echo ==========================================
echo Simple Py Statistical Testing
echo Author: Keith Ngamphon McKenzie
echo Website: https://mckenzie.page
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is required but not installed on this system.
    pause
    exit /b 1
)

echo Python found
python --version

REM Install dependencies
echo Installing dependencies...
pip install numpy scipy

REM Method 1: Direct Python execution
echo Method 1: Running Python version...
python main.py

echo.
echo ==========================================
echo Build completed!
echo Author: Keith Ngamphon McKenzie
echo Website: https://mckenzie.page
echo ==========================================
pause
