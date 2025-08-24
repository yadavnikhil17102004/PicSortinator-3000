@echo off
:: ============================================================================
:: ForenSnap Ultimate Setup & Desktop Shortcut Creator
:: One-time setup for the AI-Powered Digital Forensics Suite
:: ============================================================================
title ForenSnap Ultimate - Quick Setup

echo.
echo ===============================================================================
echo  🔧 FORENSNAP ULTIMATE - QUICK SETUP
echo ===============================================================================
echo  This will set up ForenSnap Ultimate and create a desktop shortcut
echo ===============================================================================
echo.

:: Set colors
color 0B

:: Change to script directory
cd /d "%~dp0"

echo 🚀 Starting ForenSnap Ultimate Setup...
echo.

:: Check Python installation
echo 🐍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed!
    echo.
    echo 📥 Please install Python 3.8 or later:
    echo    1. Go to: https://www.python.org/downloads/
    echo    2. Download Python 3.9 or later
    echo    3. ✅ Check "Add Python to PATH" during installation
    echo    4. Run this setup again after installation
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% detected

:: Create virtual environment
echo 🌍 Setting up virtual environment...
if exist "venv" (
    echo 🗂️ Removing old virtual environment...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment!
    pause
    exit /b 1
)
echo ✅ Virtual environment created

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✅ pip upgraded

:: Install core dependencies
echo 📦 Installing core dependencies (this may take a few minutes)...
pip install wheel setuptools --quiet

:: Create desktop shortcut
echo 🖥️ Creating desktop shortcut...

set DESKTOP=%USERPROFILE%\Desktop
set SHORTCUT_PATH=%DESKTOP%\ForenSnap Ultimate.lnk
set TARGET_PATH=%~dp0run_forensnap.bat
set ICON_PATH=%~dp0forensnap_icon.ico

:: Create VBS script to generate shortcut
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%SHORTCUT_PATH%" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%TARGET_PATH%" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%~dp0" >> CreateShortcut.vbs
echo oLink.Description = "ForenSnap Ultimate - AI-Powered Digital Forensics Suite" >> CreateShortcut.vbs
echo oLink.WindowStyle = 1 >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs

cscript CreateShortcut.vbs >nul
del CreateShortcut.vbs

if exist "%SHORTCUT_PATH%" (
    echo ✅ Desktop shortcut created successfully
) else (
    echo ⚠️ Failed to create desktop shortcut
)

:: Create start menu shortcut
echo 📂 Creating start menu shortcut...
set STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs
set STARTMENU_SHORTCUT=%STARTMENU%\ForenSnap Ultimate.lnk

echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateStartMenuShortcut.vbs
echo sLinkFile = "%STARTMENU_SHORTCUT%" >> CreateStartMenuShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateStartMenuShortcut.vbs
echo oLink.TargetPath = "%TARGET_PATH%" >> CreateStartMenuShortcut.vbs
echo oLink.WorkingDirectory = "%~dp0" >> CreateStartMenuShortcut.vbs
echo oLink.Description = "ForenSnap Ultimate - AI-Powered Digital Forensics Suite" >> CreateStartMenuShortcut.vbs
echo oLink.WindowStyle = 1 >> CreateStartMenuShortcut.vbs
echo oLink.Save >> CreateStartMenuShortcut.vbs

cscript CreateStartMenuShortcut.vbs >nul 2>&1
del CreateStartMenuShortcut.vbs

if exist "%STARTMENU_SHORTCUT%" (
    echo ✅ Start menu shortcut created
) else (
    echo ⚠️ Failed to create start menu shortcut
)

:: Deactivate virtual environment
deactivate

echo.
echo ===============================================================================
echo  ✅ SETUP COMPLETE!
echo ===============================================================================
echo.
echo 🎉 ForenSnap Ultimate is now ready to use!
echo.
echo 🚀 How to start ForenSnap:
echo    • Double-click the desktop shortcut: "ForenSnap Ultimate"
echo    • Or run: run_forensnap.bat
echo    • Or from Start Menu: ForenSnap Ultimate
echo.
echo 💡 Features available:
echo    ✓ Multi-language OCR (Tesseract + EasyOCR)
echo    ✓ Social media platform detection
echo    ✓ Advanced threat detection with NLP
echo    ✓ NSFW/adult content detection
echo    ✓ Legal-compliant PDF report generation
echo    ✓ Batch processing capabilities
echo    ✓ Advanced search and filtering
echo    ✓ Evidence chain of custody tracking
echo.
echo 📊 Usage examples:
echo    • GUI Mode: Just double-click the shortcut
echo    • Single image: run_forensnap.bat analyze "image.jpg"
echo    • Batch folder: run_forensnap.bat batch "C:\Images"
echo    • Generate report: run_forensnap.bat report
echo    • Search database: run_forensnap.bat search "keyword"
echo.
echo 🔧 Optional optimizations:
echo    • Install Tesseract OCR for better text recognition
echo    • Ensure CUDA GPU support for faster AI processing
echo    • Allocate at least 4GB RAM for optimal performance
echo.
echo 📝 Support files created:
echo    • Desktop shortcut: %DESKTOP%\ForenSnap Ultimate.lnk
echo    • Start menu entry: ForenSnap Ultimate
echo    • Virtual environment: venv\
echo    • Database: forensnap_ultimate.db (created on first run)
echo    • Logs: forensnap.log (created on first run)
echo.
echo ===============================================================================

echo 🎯 Ready to analyze screenshots for digital investigations!
echo.
echo Press any key to close this setup window...
pause >nul

exit /b 0
