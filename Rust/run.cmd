@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "CARGO=%USERPROFILE%\.cargo\bin\cargo.exe"
if not exist "%CARGO%" (
    echo Rust is not installed. Install Rust from https://rustup.rs/
    pause
    exit /b 1
)

rem Discover installed C++ tools instead of assuming an edition or toolset version.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%V in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do call "%%V\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 -no_logo
)
rem Optional developer-specific SDK configuration; excluded from Git.
if exist "run.local.cmd" call "run.local.cmd"

:run
"%CARGO%" run --release --locked -- %*
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo.
    echo Launch failed. See README.md for setup and troubleshooting.
    pause
)
exit /b %APP_EXIT%
