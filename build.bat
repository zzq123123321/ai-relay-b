@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo  Building AI_Relay_B (fixed) with PyInstaller
echo ============================================
echo.

py -3.12 -m PyInstaller --noconfirm --clean AI_Relay_B.spec

if errorlevel 1 (
    echo.
    echo BUILD FAILED
    exit /b 1
)

echo.
echo Build output: dist\AI_Relay_B\
echo.
echo To deploy, copy dist\AI_Relay_B\* to D:\AI_Relay_B\ (replacing the old files).
endlocal
