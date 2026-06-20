@echo off
cd /d D:\code\MPA-OpenCl

echo === CMake Configure === > _cmake_cfg.log 2>&1
cmake -G "Visual Studio 17 2022" -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE="D:/code/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -DOPENSSL_ROOT_DIR="D:/code/vcpkg/installed/x64-windows" -S D:\code\MPA-OpenCl >> _cmake_cfg.log 2>&1
set CFG_RC=%ERRORLEVEL%
echo CMAKE CONFIGURE EXIT CODE: %CFG_RC% >> _cmake_cfg.log

if not %CFG_RC%==0 goto :end

echo === CMake Build === > _cmake_build.log 2>&1
cmake --build build --config Debug --target cpu_mont_bench -- /v:minimal >> _cmake_build.log 2>&1
echo CMAKE BUILD EXIT CODE: %ERRORLEVEL% >> _cmake_build.log

:end