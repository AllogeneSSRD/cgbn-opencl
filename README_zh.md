# ECM-OpenCl

基于 **OpenCL** 的 **ECM 第一阶段**（椭圆曲线因子分解）实现，附带大量 Montgomery 乘/平方与模加/模减内核变体。项目面向桌面 GPU 的高吞吐因子分解，并提供 **Android** 端 OpenCL 探测与算子微基准。

ECM stage-1 的热点在 Montgomery 乘/平方，而非模加/模减。本仓库围绕多种内核实现（私有内存 CIOS、循环展开、工作组分块、FIPS 风格路径、AMD 内联汇编等）做了大量工作，并通过独立微基准在接入完整 stage-1 前对比选型。

## 功能概览

- **GPU ECM stage-1 驱动**（`ecm.exe`）— OpenCL 批量曲线、检查点/恢复、可选内核路径覆盖
- **算子微基准** — `opencl_ecm_addsub`、`opencl_ecm_montsqr` 按模数位宽遍历 manifest 中全部路径
- **可插拔内核路径** — 编译期 `-D` 与运行时 CLI/环境变量选择 mul/sqr/add/sub 实现
- **OpenCL 程序二进制缓存** — FNV-1a 键控磁盘缓存（Windows 桌面与 Android）
- **Android ECM 应用** — 在厂商 `libOpenCL.so` 上运行设备探测、add/sub 与 mont mul/sqr 基准
- **代码生成与 ISA 工具** — `tools/` 下 Python 生成器与反汇编脚本

## 目录结构

```
ECM-OpenCl/
├── CMakeLists.txt              # Windows/桌面构建（OpenCL、OpenSSL、GMP）
├── cgbn/backends/opencl/
│   ├── impl_opencl.cpp         # OpenCL 上下文、程序构建、二进制缓存
│   └── kernels/                # .cl 源码（ecm_stage1、mont_*、mp_*）
├── include/                    # Manifest 头文件、路径枚举、ECM API
├── src/
│   ├── cgbn_stage1_opencl.cpp  # Stage-1 OpenCL 主机端驱动
│   ├── ecm_driver.cpp          # ecm.exe 命令行
│   ├── opencl_ecm_*_bench.cpp  # 独立微基准
│   └── opencl_ecm_*_manifest.cpp
├── Android/
│   ├── README.md               # Android OpenCL 加载说明（16 KB 页）
│   └── ECM/                    # Android Studio 工程 — 见 ECM/README.md
├── tools/                      # 内核/汇编生成器、反汇编脚本
├── test/                       # CUDA/OpenCL 正确性与吞吐测试
├── ECM_OPERATOR_ANALYSIS.md    # 热点分析与基准数据
└── README.md                   # 英文文档
```

## 环境要求（Windows / 桌面）

| 依赖 | 说明 |
|------|------|
| CMake 3.20+ | 推荐 Visual Studio 2022（x64） |
| OpenCL ICD | GPU 厂商运行时（NVIDIA、AMD、Intel） |
| OpenSSL | CMake `find_package(OpenSSL)` |
| GMP | 当前在 `CMakeLists.txt` 中通过 **硬编码 vcpkg 路径** 链接 |

若 vcpkg 安装目录不是 `D:/code/vcpkg/installed/x64-windows/`，请先修改 `CMakeLists.txt` 中的 `link_directories(...)` 以及各目标的 `gmp.lib` 路径。

可选：[Pari/GP](https://pari.math.u-bordeaux.fr/) 用于 `--go` 群阶诊断（环境变量 `ECM_GP_BIN`）。

## 构建（Windows）

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

可执行文件位于 `build/Debug/`（MSVC 多配置生成器）。

## 可执行目标

| 目标 | 用途 |
|------|------|
| `ecm.exe` | ECM stage-1 因子分解驱动（从 stdin 读取 `N`） |
| `opencl_ecm_addsub.exe` | 模加/模减内核吞吐对比 |
| `opencl_ecm_montsqr.exe` | Montgomery 乘/平方内核吞吐对比 |
| `main.exe` | OpenCL 低风险冒烟测试 |
| `opencl_asm_selftest.exe` | 内联汇编内核自检 |
| `opencl_mont_isa_export.exe` | 导出 mont 内核编译 ISA |
| `opencl_addsub_isa_export.exe` | 导出 add/sub 内核编译 ISA |

## ECM 驱动（`ecm.exe`）

从标准输入读取合数 **N**（十进制或表达式），执行 stage-1，可选 GPU 批处理。

```powershell
echo "(2^991-1)" | build\Debug\ecm.exe -v --go -gpu -gpucurves 384 1e6 0
build\Debug\ecm.exe --showkernel
```

常用选项：

| 选项 | 说明 |
|------|------|
| `-gpu` | 启用 GPU stage-1（需配合 `-gpucurves`） |
| `-gpucurves <n>` | 每次 GPU launch 的曲线数 |
| `-d <index>` | OpenCL 设备索引 |
| `--mul`、`--sqr`、`--add`、`--sub <path>` | 覆盖内核路径 |
| `--showkernel` | 列出全部注册路径并退出 |

模加/模减路径名包括 `default`、`fused`、`fused_unroll`、`fused_unroll_b32`、`asm_b32` 等。完整 Montgomery 与 add/sub 列表请运行 `--showkernel`。

## 微基准

两个基准共用参数形式：

```text
<exe> [--bits <bits>] <kernel_iterations> <instances> <launch_repeats>
```

总工作量 ≈ 每条路径的 `instances × kernel_iterations × launch_repeats`。

```powershell
# 512 位模加/模减 — 遍历 manifest 全部路径
build\Debug\opencl_ecm_addsub.exe --bits 512 10000 128 3

# 512 位 Montgomery 乘/平方 — 默认 WG 模式、tpi=4
build\Debug\opencl_ecm_montsqr.exe --bits 512 1000 128 1
```

设置 `ECM_BENCH_CSV=<文件>` 可追加 CSV 结果。MSVC 为 `opencl_ecm_montsqr` 链接 16 MB 栈空间，因 NVIDIA 上大型 `.cl` JIT 可能撑爆默认 1 MB 线程栈。

## 内核架构

1. **`.cl` 源码**位于 `cgbn/backends/opencl/kernels/`。
2. **Manifest**（`opencl_ecm_addsub_manifest`、`opencl_ecm_montsqr_manifest`）注册路径名、源文件与额外 `-D` 编译选项。
3. **路径枚举**（`opencl_ecm_*_path.h`）将 CLI 名称映射为 `ecm_stage1.cl` 分发所用的编译期常量。
4. **Stage-1 主机**（`cgbn_stage1_opencl.cpp`）拼接内核源码，按模数位宽（256 / 512 / 4096 位）选择默认 mul/sqr/add/sub 路径，并 launch `kernel_double_add`。

Stage-1 按模数大小选择默认路径。**512 位**在较新 Adreno 上应优先 **`unroll_only_512`** / **`unroll_only_512_manual`**，而非 `priv_opt` 或通用工作组路径 — 见 [算子分析](#算子分析) 与 `ECM_OPERATOR_ANALYSIS.md`。

## 环境变量

| 变量 | 组件 | 说明 |
|------|------|------|
| `CGBN_KERNEL_ROOT` | 全局 | 覆盖 `.cl` 内核树目录 |
| `CGBN_OPENCL_DEVICE_INDEX` | 全局 | 默认 OpenCL 设备索引 |
| `CGBN_OPENCL_CACHE_DIR` | 桌面缓存 | `.opencl_cache/opencl_{hash}.bin` 目录 |
| `CGBN_OPENCL_CACHE_DISABLE` | 桌面缓存 | 设置后禁用二进制缓存 |
| `CGBN_OPENCL_CACHE_VERBOSE` | 桌面缓存 | 输出缓存命中/未命中详情 |
| `CGBN_OPENCL_COMPILE_VERBOSE` | 桌面缓存 | 输出完整编译选项 |
| `ECM_OPENCL_TPI` | Stage-1 | 每实例线程数（2 的幂；默认 8） |
| `ECM_STAGE1_FORCE_NORMALIZE` | Stage-1 | 强制 normalize 路径 |
| `ECM_MP_ADD_MOD_FUSED_UNROLL` | Stage-1 / addsub 基准 | 融合展开变体 |
| `ECM_PROFILE_OPS` | Stage-1 | 打印各算子计数 |
| `ECM_PROFILE_OPS_FILE` | Stage-1 | 算子 profile CSV（默认 `ecm_ops_profile.csv`） |
| `ECM_VERIFY_GPU_RESULTS` | Stage-1 | CPU 交叉校验 GPU 结果 |
| `ECM_VERIFY_GPU_STRICT` | Stage-1 | 校验不一致则失败 |
| `ECM_BENCH_CSV` | 微基准 | 追加基准结果到 CSV |
| `ECM_ADDSUB_ASM_DISABLE` | Addsub 基准 | 跳过 asm 路径 |
| `ECM_MONT_WG_IMPL` | Mont 基准 | 工作组实现选择 |
| `ECM_LOG_TIMESTAMP` | 日志 | 为日志行加时间戳前缀 |
| `ECM_GP_BIN` | ecm 驱动 | `--go` 所用 `gp` 可执行文件路径 |

## OpenCL 编译缓存

桌面构建使用 `cgbn/backends/opencl/impl_opencl.cpp` 中的 `cgbn::opencl::build_program_from_source`：

- 缓存文件：`{CGBN_OPENCL_CACHE_DIR 或 cwd}/.opencl_cache/opencl_{fnv1a64}.bin`
- 缓存键：GPU 名称/厂商/驱动版本 + 编译选项 + 完整拼接源码
- 流程：命中则 `clCreateProgramWithBinary`；未命中则编译、查询 binary、原子写入

Android 应用在 `{codeCacheDir}/opencl_cache/` 下使用相同哈希算法。部分驱动无法导出 binary 时，采用 **live program cache** 在进程内保留已编译的 `cl_program`。详见 [Android/ECM/README.md](Android/ECM/README.md)。

## Android

在 Android Studio 中打开 **`Android/ECM`**（不是仓库根目录）。应用功能：

- 通过 `uses-native-library` 加载厂商 OpenCL（**切勿**打包 `libOpenCL.so` — 16 KB 页设备会因对齐崩溃）
- 设备探测、ECM add/sub 基准、mont mul/sqr 基准（与桌面同源内核，不含 AMD asm）
- 将 `codeCacheDir` 传入 native 层供 OpenCL 二进制缓存使用

16 KB 页约束见 [Android/README.md](Android/README.md)；UI 参数、缓存调试与 adb 命令见 [Android/ECM/README.md](Android/ECM/README.md)。

```bash
adb logcat ECM-OpenCL:I *:S
adb shell run-as com.example.ecm ls -la code_cache/opencl_cache/
```

## 算子分析

ECM stage-1 的主要耗时在 **Montgomery 乘/平方**，而非模加/模减。算子混合比例与微基准测量见 **[ECM_OPERATOR_ANALYSIS.md](ECM_OPERATOR_ANALYSIS.md)**。

移动 GPU 选型建议（512 位，典型基准配置 `128 × 1000 × 1`）：

| GPU | 推荐 mul 路径 | 说明 |
|-----|---------------|------|
| Adreno 830 | `unroll_only_512_manual` | 约 850 万 ops/s；自动 `unroll_only_512` 接近 |
| Adreno 642 | `unroll_only_512`（auto） | 约 143 万 ops/s；旧编译器上 **勿用** `manual`（约慢 5 倍） |

在移动端有 512 专用路径时，勿将 `priv_opt` 或通用 `wg` 路径作为 512 位默认。

## 工具链

`tools/` 目录包含 Montgomery/add/sub 展开内核的 Python 生成器、NPU 测试向量，以及 ISA 反汇编 PowerShell 脚本（`disasm_mont_isa.ps1`、`disasm_addsub_isa.ps1`）。Windows 上安装 objdump/llvm-objdump 见 `tools/DISASM_SETUP.md`。

`test/` 目录为 Makefile 驱动的 CUDA/OpenCL 正确性套件（`opencl_mont_tests.cpp`、`opencl_addsub_tests.cpp` 等），用于内核开发阶段验证。

## 相关文档

| 文档 | 内容 |
|------|------|
| [README.md](README.md) | 英文版 |
| [ECM_OPERATOR_ANALYSIS.md](ECM_OPERATOR_ANALYSIS.md) | 热点分解、路径清单、基准 CSV |
| [Android/ECM/README.md](Android/ECM/README.md) | Android 应用构建、缓存、mont/addsub UI |
| [Android/README.md](Android/README.md) | Android 15+ OpenCL 加载 |
