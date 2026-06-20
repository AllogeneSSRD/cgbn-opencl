"""Run cmake configure + build and write results to a log file for reading."""
import subprocess, sys, os

os.chdir(r'D:\code\MPA-OpenCl')

def run_and_log(cmd, logpath):
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600
        )
        with open(logpath, 'w', encoding='utf-8') as f:
            f.write(f"CMD: {' '.join(cmd)}\n")
            f.write(f"RC: {p.returncode}\n\n")
            f.write(p.stdout or '(no output)')
        return p.returncode
    except Exception as e:
        with open(logpath, 'w', encoding='utf-8') as f:
            f.write(f"CMD: {' '.join(cmd)}\n")
            f.write(f"EXCEPTION: {e}\n")
        return -1

# Step 1: cmake configure
rc = run_and_log([
    'cmake', '-G', 'Visual Studio 17 2022',
    '-B', r'D:\code\MPA-OpenCl\build',
    '-DCMAKE_BUILD_TYPE=Debug',
    '-DCMAKE_TOOLCHAIN_FILE=D:/code/vcpkg/scripts/buildsystems/vcpkg.cmake',
    '-DVCPKG_TARGET_TRIPLET=x64-windows',
    '-DOPENSSL_ROOT_DIR=D:/code/vcpkg/installed/x64-windows',
    '-S', r'D:\code\MPA-OpenCl'
], r'D:\code\MPA-OpenCl\_build_cfg_log.txt')

print(f"cmake configure rc={rc}")

if rc != 0:
    print("CONFIGURE FAILED - see _build_cfg_log.txt")
    sys.exit(1)

# Step 2: build cpu_mont_bench
rc = run_and_log([
    'cmake', '--build', r'D:\code\MPA-OpenCl\build',
    '--config', 'Debug',
    '--target', 'cpu_mont_bench',
    '--', '/v:minimal'
], r'D:\code\MPA-OpenCl\_build_make_log.txt')

print(f"cmake build rc={rc}")

if rc != 0:
    print("BUILD FAILED - see _build_make_log.txt")
    sys.exit(1)

print("BUILD SUCCESS")