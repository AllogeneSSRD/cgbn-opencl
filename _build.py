import subprocess, sys

def run(cmd_args, log_path):
    with open(log_path, 'w', encoding='utf-8') as f:
        p = subprocess.run(cmd_args, stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode

# Step 1: CMake configure
rc = run([
    'cmake', '-G', 'Visual Studio 17 2022',
    '-B', 'D:/code/MPA-OpenCl/build',
    '-DCMAKE_BUILD_TYPE=Debug',
    '-DCMAKE_TOOLCHAIN_FILE=D:/code/vcpkg/scripts/buildsystems/vcpkg.cmake',
    '-DVCPKG_TARGET_TRIPLET=x64-windows',
    '-DOPENSSL_ROOT_DIR=D:/code/vcpkg/installed/x64-windows',
    '-S', 'D:/code/MPA-OpenCl'
], 'D:/code/MPA-OpenCl/_cmake_cfg.log')

print(f"CMake configure: rc={rc}")

if rc != 0:
    sys.exit(1)

# Step 2: Build cpu_mont_bench
rc = run([
    'cmake', '--build', 'D:/code/MPA-OpenCl/build',
    '--config', 'Debug',
    '--target', 'cpu_mont_bench',
    '--', '/v:minimal'
], 'D:/code/MPA-OpenCl/_cmake_build.log')

print(f"CMake build: rc={rc}")