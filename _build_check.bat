@echo off
cd /d D:\code\MPA-OpenCl
cmake -G "Visual Studio 17 2022" -B "build" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE="D:/code/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -DOPENSSL_ROOT_DIR="D:/code/vcpkg/installed/x64-windows" -S "D:\code\MPA-OpenCl" > build_cfg_log.txt 2>&1
echo CMake exit code: %ERRORLEVEL% >> build_cfg_log.txt