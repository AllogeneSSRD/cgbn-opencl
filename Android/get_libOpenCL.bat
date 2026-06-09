@echo off
setlocal enabledelayedexpansion

REM Pull 64-bit vendor OpenCL for LOCAL INSPECTION ONLY (objdump NEEDED).
REM Do NOT copy into jniLibs — load from /vendor/lib64 at runtime on device.

set "ADB=D:\AppData\Android\Sdk\platform-tools\adb.exe"
if not exist "%ADB%" set "ADB=adb"

set "NDK=D:\AppData\Android\Sdk\ndk\28.2.13676358"
set "OBJDUMP=%NDK%\toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-objdump.exe"
if not exist "%OBJDUMP%" set "OBJDUMP=llvm-objdump"

set "ROOT=%~dp0"

echo === Pull 64-bit libOpenCL.so (reference copy) ===
"%ADB%" pull /system/vendor/lib64/libOpenCL.so "%ROOT%libOpenCL.so"
if errorlevel 1 (
    echo FAILED: adb pull lib64/libOpenCL.so
    echo Try: adb root
    pause
    exit /b 1
)

echo.
echo === NEEDED dependencies ===
"%OBJDUMP%" -p "%ROOT%libOpenCL.so" | findstr /i NEEDED

echo.
echo === 16 KB page alignment (LOAD align) ===
"%OBJDUMP%" -p "%ROOT%libOpenCL.so" | findstr /i "LOAD"

echo.
echo NOTE: Do not add libOpenCL.so to jniLibs.
echo       The app dlopen's /vendor/lib64/libOpenCL.so on the phone directly.
echo       Pulled copies are often 4 KB-aligned and crash on 16 KB-page devices if packaged in APK.
pause
