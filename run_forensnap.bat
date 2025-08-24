@echo off
:: ============================================================================
:: ForenSnap Ultimate Launcher
:: Automated AI-Powered Digital Forensics Suite
:: ============================================================================
title ForenSnap Ultimate - AI Digital Forensics

echo.
echo ===============================================================================
echo  🔬 FORENSNAP ULTIMATE - AI-POWERED DIGITAL FORENSICS SUITE
echo ===============================================================================
echo  Version: 2.0.0
echo  Advanced Screenshot Analysis for Digital Investigations
echo ===============================================================================
echo.

:: Set colors for better visibility
color 0A

:: Change to the script directory
cd /d "%~dp0"

:: Create organized directory structure
if not exist "data" md data
if not exist "logs" md logs
if not exist "src" md src

:: Check if Python is installed
echo 🐍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.8 or later from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Display Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% found

:: Check if pip is available
echo 🔧 Checking pip installation...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not available!
    echo Please ensure pip is installed with Python.
    pause
    exit /b 1
)
echo ✅ pip is available

:: Upgrade pip to latest version
echo 📦 Upgrading pip to latest version...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo ⚠️ Failed to upgrade pip, continuing anyway...
) else (
    echo ✅ pip upgraded successfully
)

:: Check if virtual environment exists, create if not
if not exist "venv" (
    echo 🌍 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

:: Activate virtual environment
echo 🚀 Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment!
    pause
    exit /b 1
)
echo ✅ Virtual environment activated

:: Install/update required system packages first
echo 📦 Installing system packages...
pip install --upgrade setuptools wheel --quiet
pip install --upgrade pip --quiet

:: Install ForenSnap dependencies
echo 🔧 Installing ForenSnap Ultimate dependencies...
echo    ⚠️  Python 3.13 detected - installing compatible versions...

:: First install core dependencies that are known to work
echo    📦 Installing core packages...
pip install pillow>=10.0.0 --quiet
pip install opencv-python>=4.8.0 --quiet
pip install numpy>=1.24.0 --quiet
pip install requests>=2.31.0 --quiet

:: Install OCR packages
echo    👁️  Installing OCR packages...
pip install pytesseract>=0.3.10 --quiet
pip install easyocr>=1.7.0 --quiet
pip install langdetect>=1.0.9 --quiet

:: Try to install PyTorch (CPU version for compatibility)
echo    🤖 Installing AI packages...
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 (
    echo    ⚠️  PyTorch installation failed, trying alternative...
    pip install torch torchvision --quiet
)

:: Install better CLIP version
echo    🔗 Installing CLIP model...
pip install git+https://github.com/openai/CLIP.git --quiet 2>nul
if errorlevel 1 (
    echo    ⚠️  OpenAI CLIP installation failed, trying alternative...
    pip install clip-by-openai --quiet 2>nul
)

:: Install text processing
echo    📝 Installing text processing...
pip install transformers>=4.35.0 --quiet
pip install spacy>=3.7.0 --quiet

:: Install other packages with error handling
echo    🔧 Installing additional packages...
pip install sqlalchemy>=2.0.0 --quiet 2>nul
pip install fastapi>=0.100.0 --quiet 2>nul
pip install reportlab>=4.0.0 --quiet 2>nul
pip install fuzzywuzzy>=0.18.0 --quiet 2>nul

echo    ✅ Core dependencies installed (some advanced features may be limited)

:: Check if Tesseract is installed (optional but recommended)
echo 🔍 Checking Tesseract OCR...
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Tesseract OCR not found - OCR will use EasyOCR only
    echo 💡 For better OCR accuracy, install Tesseract:
    echo    Download from: https://github.com/UB-Mannheim/tesseract/wiki
) else (
    echo ✅ Tesseract OCR found
)

:: Check available memory
echo 💾 Checking system resources...
for /f "skip=1" %%p in ('wmic computersystem get TotalPhysicalMemory ^| findstr [0-9]') do set TOTAL_MEMORY=%%p
set /a MEMORY_GB=!TOTAL_MEMORY!/1024/1024/1024
if %MEMORY_GB% LSS 4 (
    echo ⚠️ Low memory detected: %MEMORY_GB%GB
    echo    ForenSnap may run slower on systems with less than 4GB RAM
) else (
    echo ✅ Sufficient memory: %MEMORY_GB%GB
)

echo.
echo ===============================================================================
echo  🚀 LAUNCHING FORENSNAP ULTIMATE
echo ===============================================================================
echo.

:: Run ForenSnap Ultimate
echo 🔬 Starting ForenSnap Ultimate...
echo.

:: Check command line arguments
if "%1"=="" (
    echo 🖥️ Launching GUI Interface...
    python run_forensnap_ultimate.py
) else (
    echo 💻 Running command: %*
    python run_forensnap_ultimate.py %*
)

:: Check exit code
if errorlevel 1 (
    echo.
    echo ❌ ForenSnap Ultimate encountered an error!
    echo.
    echo 🔧 Troubleshooting tips:
    echo    • Check if all dependencies are installed correctly
    echo    • Ensure sufficient disk space and memory
    echo    • Check the logs/forensnap.log file for detailed error information
    echo    • Try running: python run_forensnap_ultimate.py --help
    echo.
    echo 📋 System Information:
    echo    • Python Version: %PYTHON_VERSION%
    echo    • Current Directory: %CD%
    echo    • Date/Time: %DATE% %TIME%
    echo.
    pause
) else (
    echo.
    echo ✅ ForenSnap Ultimate completed successfully!
    echo.
    echo 📊 Session Information:
    echo    • Date/Time: %DATE% %TIME%
    echo    • Results saved to data/ directory
    echo    • Logs available in logs/ directory
    echo.
)

:: Deactivate virtual environment
deactivate

echo.
echo ===============================================================================
echo  ForenSnap Ultimate Session Ended
echo ===============================================================================
echo.

:: Don't auto-close if there was an error
if errorlevel 1 (
    echo Press any key to close...
    pause >nul
)

exit /b
