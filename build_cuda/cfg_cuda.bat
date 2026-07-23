@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\code\MPA-OpenCl
cmake -DECM_CUDA_FULL_BUILD=ON -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -DECM_CUDA_ARCHITECTURES=60 -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=D:/code/vcpkg/installed/x64-windows -DECM_WINDOWS_GMP_ROOT=D:/code/vcpkg/installed/x64-windows -DCMAKE_CUDA_FLAGS="--ptxas-options=-v" -S . -B build_cuda_cmake
echo === cmake configure exit: %errorlevel% ===
