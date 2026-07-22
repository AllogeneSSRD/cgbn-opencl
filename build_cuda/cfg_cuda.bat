@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\code\MPA-OpenCl
"D:\code\vcpkg\downloads\tools\cmake-4.3.2-windows\cmake-4.3.2-windows-x86_64\bin\cmake.exe" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=D:/code/vcpkg/installed/x64-windows -DECM_WINDOWS_GMP_ROOT=D:/code/vcpkg/installed/x64-windows -DCMAKE_CUDA_FLAGS="--ptxas-options=-v" -S . -B build_cuda_cmake
echo === cmake configure exit: %errorlevel% ===
