# ECM-OpenCl

OpenCL implementation of **ECM stage-1** (Elliptic Curve Method factorization) with a large library of Montgomery and modular add/sub kernels. The project targets high-throughput GPU factorization on desktop GPUs and includes an **Android** port for on-device OpenCL microbenchmarks and probing.

Montgomery multiply/square dominate ECM stage-1 runtime; this repository invests heavily in multiple kernel variants (private-memory CIOS, loop unrolling, work-group tiling, FIPS-style paths, inline assembly on AMD) and provides standalone benches to compare them before selecting defaults for full stage-1 runs.

## Features

- **GPU ECM stage-1 driver** (`ecm.exe`) — batch curves on OpenCL, checkpoint/resume, optional kernel path overrides
- **Operator microbenchmarks** — `opencl_ecm_addsub`, `opencl_ecm_montsqr` sweep all registered kernel paths for a given modulus width
- **Pluggable kernel paths** — compile-time `-D` flags and runtime CLI/env selection for mul/sqr/add/sub implementations
- **OpenCL program binary cache** — FNV-1a keyed disk cache (Windows desktop and Android)
- **Android ECM app** — device probe, add/sub bench, mont mul/sqr bench on vendor `libOpenCL.so`
- **Codegen & ISA tooling** — Python generators and disassembly helpers under `tools/`

## Repository layout

```
ECM-OpenCl/
├── CMakeLists.txt              # Windows/desktop build (OpenCL, OpenSSL, GMP)
├── cgbn/backends/opencl/
│   ├── impl_opencl.cpp         # OpenCL context, program build, binary cache
│   └── kernels/                # .cl sources (ecm_stage1, mont_*, mp_*)
├── include/                    # Manifest headers, path enums, ECM API
├── src/
│   ├── cgbn_stage1_opencl.cpp  # Stage-1 OpenCL host driver
│   ├── ecm_driver.cpp          # ecm.exe CLI
│   ├── opencl_ecm_*_bench.cpp  # Standalone microbenchmarks
│   └── opencl_ecm_*_manifest.cpp
├── Android/
│   ├── README.md               # Android OpenCL loading notes (16 KB pages)
│   └── ECM/                    # Android Studio app — see ECM/README.md
├── tools/                      # Kernel/asm generators, disasm scripts
├── test/                       # CUDA/OpenCL correctness & throughput tests
├── ECM_OPERATOR_ANALYSIS.md    # Hotspot analysis and bench baselines
└── README_zh.md                # Chinese documentation
```

## Prerequisites (Windows / desktop)

| Dependency | Notes |
|------------|--------|
| CMake 3.20+ | Visual Studio 2022 (x64) recommended |
| OpenCL ICD | GPU vendor runtime (NVIDIA, AMD, Intel) |
| OpenSSL | Via CMake `find_package(OpenSSL)` |
| GMP | Currently linked via **hardcoded vcpkg paths** in `CMakeLists.txt` |

Before building, edit `CMakeLists.txt` if your vcpkg install is not at `D:/code/vcpkg/installed/x64-windows/` — update `link_directories(...)` and the `gmp.lib` paths for `ecm`, `main`, and the bench targets.

Optional: [Pari/GP](https://pari.math.u-bordeaux.fr/) for `--go` group-order diagnostics (`ECM_GP_BIN` env var).

## Build (Windows)

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

Binaries appear under `build/Debug/` (MSVC multi-config generator).

## Executables

| Target | Purpose |
|--------|---------|
| `ecm.exe` | ECM stage-1 factorization driver (reads `N` from stdin) |
| `opencl_ecm_addsub.exe` | Modular add/sub kernel throughput sweep |
| `opencl_ecm_montsqr.exe` | Montgomery mul/sqr kernel throughput sweep |
| `main.exe` | Low-risk OpenCL smoke tests |
| `opencl_asm_selftest.exe` | Inline-asm kernel self-test |
| `opencl_mont_isa_export.exe` | Export compiled ISA for mont kernels |
| `opencl_addsub_isa_export.exe` | Export compiled ISA for add/sub kernels |

## ECM driver (`ecm.exe`)

Reads composite **N** from stdin (decimal or expression), then runs stage-1 with optional GPU batching.

```powershell
echo "(2^991-1)" | build\Debug\ecm.exe -v --go -gpu -gpucurves 384 1e6 0
build\Debug\ecm.exe --showkernel
```

Common options:

| Option | Description |
|--------|-------------|
| `-gpu` | Enable GPU stage-1 (requires `-gpucurves`) |
| `-gpucurves <n>` | Curves per GPU launch |
| `-d <index>` | OpenCL device index |
| `--mul`, `--sqr`, `--add`, `--sub <path>` | Override kernel paths |
| `--showkernel` | List all registered paths and exit |

Add/sub path names include `default`, `fused`, `fused_unroll`, `fused_unroll_b32`, `asm_b32`, etc. Run `--showkernel` for the full Montgomery and add/sub lists.

## Microbenchmarks

Both benches share the argument pattern:

```text
<exe> [--bits <bits>] <kernel_iterations> <instances> <launch_repeats>
```

Total work ≈ `instances × kernel_iterations × launch_repeats` per path.

```powershell
# 512-bit modular add/sub — all manifest paths
build\Debug\opencl_ecm_addsub.exe --bits 512 10000 128 3

# 512-bit Montgomery mul/sqr — WG mode, tpi=4 default
build\Debug\opencl_ecm_montsqr.exe --bits 512 1000 128 1
```

Set `ECM_BENCH_CSV=<file>` to append CSV results. MSVC links `opencl_ecm_montsqr` with a 16 MB stack because large `.cl` JIT can overflow the default 1 MB thread stack on NVIDIA.

## Kernel architecture

1. **`.cl` sources** live in `cgbn/backends/opencl/kernels/`.
2. **Manifests** (`opencl_ecm_addsub_manifest`, `opencl_ecm_montsqr_manifest`) register named paths, source files, and extra `-D` compile flags.
3. **Path enums** (`opencl_ecm_*_path.h`) map CLI names to compile-time constants used in `ecm_stage1.cl` dispatch.
4. **Stage-1 host** (`cgbn_stage1_opencl.cpp`) concatenates kernel sources, applies modulus-width-specific defaults (256 / 512 / 4096 bit), and launches `kernel_double_add`.

Stage-1 selects default mul/sqr/add/sub paths by modulus size. For **512-bit** modulus on recent Adreno GPUs, prefer **`unroll_only_512`** / **`unroll_only_512_manual`** over `priv_opt` or generic work-group paths — see [Operator analysis](#operator-analysis) and `ECM_OPERATOR_ANALYSIS.md`.

## Environment variables

| Variable | Component | Description |
|----------|-----------|-------------|
| `CGBN_KERNEL_ROOT` | All | Override directory containing `.cl` kernel tree |
| `CGBN_OPENCL_DEVICE_INDEX` | All | Default OpenCL device index |
| `CGBN_OPENCL_CACHE_DIR` | Desktop cache | Directory for `.opencl_cache/opencl_{hash}.bin` |
| `CGBN_OPENCL_CACHE_DISABLE` | Desktop cache | Disable binary cache when set |
| `CGBN_OPENCL_CACHE_VERBOSE` | Desktop cache | Log cache hit/miss details |
| `CGBN_OPENCL_COMPILE_VERBOSE` | Desktop cache | Log full compile options |
| `ECM_OPENCL_TPI` | Stage-1 | Threads per instance (power of two; default 8) |
| `ECM_STAGE1_FORCE_NORMALIZE` | Stage-1 | Force normalize path |
| `ECM_MP_ADD_MOD_FUSED_UNROLL` | Stage-1 / addsub bench | Fused unroll variant |
| `ECM_PROFILE_OPS` | Stage-1 | Print per-op counters |
| `ECM_PROFILE_OPS_FILE` | Stage-1 | CSV path for op profile (default `ecm_ops_profile.csv`) |
| `ECM_VERIFY_GPU_RESULTS` | Stage-1 | CPU cross-check GPU results |
| `ECM_VERIFY_GPU_STRICT` | Stage-1 | Fail on verification mismatch |
| `ECM_BENCH_CSV` | Benches | Append bench results to CSV |
| `ECM_ADDSUB_ASM_DISABLE` | Addsub bench | Skip asm paths |
| `ECM_MONT_WG_IMPL` | Mont bench | Work-group implementation selector |
| `ECM_LOG_TIMESTAMP` | Logging | Prefix log lines with timestamps |
| `ECM_GP_BIN` | ecm driver | Path to `gp` executable for `--go` |

## OpenCL compile cache

Desktop builds use `cgbn::opencl::build_program_from_source` in `cgbn/backends/opencl/impl_opencl.cpp`:

- Cache file: `{CGBN_OPENCL_CACHE_DIR or cwd}/.opencl_cache/opencl_{fnv1a64}.bin`
- Key: GPU name/vendor/driver + build options + full concatenated source
- Flow: try `clCreateProgramWithBinary` on hit; on miss compile, query binary, write atomically

The Android app uses the same hash algorithm under `{codeCacheDir}/opencl_cache/`. When drivers cannot export binaries, a **live program cache** retains compiled `cl_program` objects for the app process lifetime. Details: [Android/ECM/README.md](Android/ECM/README.md).

## Android

Open **`Android/ECM`** in Android Studio (not the repo root). The app:

- Loads vendor OpenCL via `uses-native-library` (do **not** bundle `libOpenCL.so` — breaks 16 KB page devices)
- Runs device probe, ECM add/sub bench, and mont mul/sqr bench (same kernels as desktop, minus AMD asm)
- Passes `codeCacheDir` into native code for OpenCL binary cache

See [Android/README.md](Android/README.md) for 16 KB page constraints and [Android/ECM/README.md](Android/ECM/README.md) for UI parameters, cache debugging, and adb commands.

```bash
adb logcat ECM-OpenCL:I *:S
adb shell run-as com.example.ecm ls -la code_cache/opencl_cache/
```

## Operator analysis

ECM stage-1 spends most time in **Montgomery mul/sqr**, not modular add/sub. Measured operator mix and microbench numbers are documented in **[ECM_OPERATOR_ANALYSIS.md](ECM_OPERATOR_ANALYSIS.md)**.

Mobile GPU guidance (512-bit, typical bench config `128 × 1000 × 1`):

| GPU | Recommended mul path | Notes |
|-----|----------------------|-------|
| Adreno 830 | `unroll_only_512_manual` | ~8.5M ops/s; auto `unroll_only_512` close second |
| Adreno 642 | `unroll_only_512` (auto) | ~1.4M ops/s; **avoid** `manual` on older compiler (~5× slower) |

Do not use `priv_opt` or generic `wg` paths as 512-bit defaults on mobile when 512-specific paths are available.

## Tools

The `tools/` directory contains Python generators for unrolled mont/add/sub kernels, NPU test vectors, and PowerShell scripts for ISA disassembly (`disasm_mont_isa.ps1`, `disasm_addsub_isa.ps1`). See `tools/DISASM_SETUP.md` for installing objdump/llvm-objdump on Windows.

The `test/` directory holds Makefile-based CUDA/OpenCL correctness suites (`opencl_mont_tests.cpp`, `opencl_addsub_tests.cpp`, etc.) used during kernel development.

## Related documentation

| Document | Content |
|----------|---------|
| [README_zh.md](README_zh.md) | Chinese version of this file |
| [ECM_OPERATOR_ANALYSIS.md](ECM_OPERATOR_ANALYSIS.md) | Hotspot breakdown, path inventory, bench CSV |
| [Android/ECM/README.md](Android/ECM/README.md) | Android app build, cache, mont/addsub UI |
| [Android/README.md](Android/README.md) | OpenCL loading on Android 15+ |
