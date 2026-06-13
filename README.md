# ECM-OpenCl

在 [GMP-ECM](https://gitlab.inria.fr/zimmerma/ecm) 因子分解框架基础上，实现 **ECM stage-1** 在多后端、多系统与多设备上的 GPU/NPU 加速与算子优化。本仓库以 **OpenCL** 为主要交付后端，围绕 Montgomery 乘/平方与模加/模减提供大量可切换内核路径；**针对 AMD GPU（GCN/RDNA）的汇编与 ISA 调优**是核心工作方向之一。

英文概览见 [README_en.md](README_en.md)。

## 参考与感谢

本仓库参考并感谢上游 **[ZIMMERMANN Paul / ecm · GitLab](https://gitlab.inria.fr/zimmerma/ecm)**（GMP-ECM）的算法、接口与 GPU 路线设计。

上游原始说明文档保存在本仓库 [`docs/`](docs/) 目录（自上游同步，便于离线查阅）：

| 文件 | 内容 |
|------|------|
| [docs/README](docs/README) | GMP-ECM 基本用法、B1/B2、表达式语法、`-param` / `-sigma` 等 |
| [docs/README.gpu](docs/README.gpu) | 上游 CUDA/CGBN GPU 版说明 |
| [docs/README.lib](docs/README.lib) | `libecm` 库接口与 `ecm_params` |
| [docs/README.dev](docs/README.dev) | 上游 autotools 开发构建 |
| [docs/README.dev.asm](docs/README.dev.asm) | 上游架构相关汇编说明 |

## 项目特点

- **同一套 ECM stage-1 数学流程**，可在不同后端与设备上运行或对照验证。
- **OpenCL 后端**（本仓库主线）：Windows 上 **NVIDIA 独显、AMD 独显、Intel 独显/核显（iGPU）**；**Android** 上通过厂商 `libOpenCL.so` 运行探测与微基准。
- **CUDA / CGBN 参考路径**：与上游 GPU-ECM 一致，CUDA stage-1 参考实现位于 `test/`（见 [docs/ECM_GPU_FLOW.md](docs/ECM_GPU_FLOW.md)）。
- **NPU 探索**：`RyzenAI/` 提供与 OpenCL 微基准对标的 add/sub 算子测试（ONNX + Vitis AI）。
- **优化重心**：Montgomery 与模运算内核的 **AMD 内联汇编**（`v_mad_u64_u32` 等）、RGA/ISA 反汇编闭环、4096-bit 路径与工作组协作框架；移动 Adreno 以 **`unroll_only_512*`** 等路径为主（详见 [bench/0530_report.md](bench/0530_report.md)）。

---

## 目录

| 章节 | 说明 |
|------|------|
| [Quick Start（Windows）](#quick-startwindows) | 最短路径：构建 → `ecm` → 微基准 |
| [Windows](#windows) | 桌面构建、使用与 OpenCL 能力 |
| [Android](#android) | 真机 App、微基准与缓存 |
| [开发与文档](#开发与文档) | 数学原理、param、算子分析、工具、bench、AMD 汇编 |
| [其他文档索引](#其他文档索引) | 正文未单独展开的子文档列表 |

---

## Quick Start（Windows）

```powershell
# 1. 构建（需 OpenCL、OpenSSL、GMP；见下方「构建」）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug

# 2. ECM stage-1（从 stdin 读 N）
echo "(2^991-1)" | build\Debug\ecm.exe -v -gpu -gpucurves 384 1e6 0

# 3. 算子微基准（先 ecm 驱动，再 bench — 与下文 Windows 章节一致）
build\Debug\opencl_ecm_addsub.exe --bits 512 10000 128 3
build\Debug\opencl_ecm_montsqr.exe --bits 512 1000 128 1
```

列出全部可切换内核路径：`build\Debug\ecm.exe --showkernel`

---

## Windows

### 构建

| 依赖 | 说明 |
|------|------|
| CMake 3.20+ | 推荐 Visual Studio 2022（x64） |
| OpenCL ICD | NVIDIA / AMD / Intel 运行时 |
| OpenSSL | `find_package(OpenSSL)` |
| GMP | `CMakeLists.txt` 中当前硬编码 vcpkg 路径，需按本机修改 |

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

产物在 `build/Debug/`。主要目标：`ecm.exe`、`opencl_ecm_addsub.exe`、`opencl_ecm_montsqr.exe`、`opencl_asm_selftest.exe`、`opencl_*_isa_export.exe` 等。

可选：[Pari/GP](https://pari.math.u-bordeaux.fr/) + 环境变量 `ECM_GP_BIN`，配合 `ecm.exe --go` 做群阶诊断。

### 使用：ECM stage-1（`ecm.exe`）

从标准输入读取合数 **N**（十进制或表达式），执行 stage-1；`-gpu` 启用 OpenCL 批处理曲线。

```powershell
echo "(2^991-1)" | build\Debug\ecm.exe -v --go -gpu -gpucurves 384 1e6 0
echo "(2^4003-1)" | build\Debug\ecm.exe -gpu -gpucurves 384 --add asm_b32 1e6 0
build\Debug\ecm.exe --showkernel
```

| 选项 | 说明 |
|------|------|
| `-gpu` / `-gpucurves <n>` | GPU stage-1 与每批曲线数 |
| `-d <index>` | OpenCL 设备（亦可用 `CGBN_OPENCL_DEVICE_INDEX`） |
| `--mul` / `--sqr` / `--add` / `--sub <path>` | 覆盖 Montgomery / 模加模减内核路径 |
| `--showkernel` | 列出 manifest 中全部路径 |

上游 `-param`、`-sigma` 等 ECM 参数语义见 [docs/README](docs/README) 第 6 节；本仓库 OpenCL stage-1 与 `ecm_params` / batch-32bit-`D` 参数的对应关系见 [docs/DEBUG_PARAMETERS_GUIDE.md](docs/DEBUG_PARAMETERS_GUIDE.md)。

### 使用：算子微基准

参数形式（两工具相同）：

```text
<exe> [--bits <bits>] <kernel_iterations> <instances> <launch_repeats>
```

```powershell
build\Debug\opencl_ecm_addsub.exe --bits 512 10000 128 3
build\Debug\opencl_ecm_montsqr.exe --bits 512 1000 128 1
```

- 追加 CSV：环境变量 `ECM_BENCH_CSV`
- 跨厂商 512/4096 对比报告：[bench/0530_report.md](bench/0530_report.md)

### 其他：OpenCL 与运行时

| 主题 | 说明 | 详细文档 |
|------|------|----------|
| OpenCL 实现总览 | stage-1 主机/内核分工、与 CUDA 差异 | [docs/OPENCL_IMPLEMENTATION.md](docs/OPENCL_IMPLEMENTATION.md) |
| 程序二进制缓存 | FNV-1a 键、`/.opencl_cache/` | 实现见 `cgbn/backends/opencl/impl_opencl.cpp`；变量见下表 |
| 内核树与 manifest | `.cl` 注册、路径枚举 | [kernels/opencl/bench/mp_addsub/README.md](kernels/opencl/bench/mp_addsub/README.md) |
| 调试与 env | `ECM_PROFILE_OPS`、`ECM_VERIFY_GPU_*` 等 | [docs/DEBUG_PARAMETERS_GUIDE.md](docs/DEBUG_PARAMETERS_GUIDE.md) |

### 环境变量

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

OpenCL 后端骨架说明：[cgbn/backends/opencl/README.md](cgbn/backends/opencl/README.md)

---

## Android

完整 stage-1 **`ecm` 驱动当前为 Windows 桌面目标**；Android 侧提供 **OpenCL 可用性探测**与 **ECM 同源算子微基准**（add/sub、mont mul/sqr），用于真机选型与编译缓存验证。

### 构建

1. 用 Android Studio 打开 **`Android/ECM`**（非仓库根目录）。
2. 确认 **`jniLibs/` 内无** 从手机 `adb pull` 的 `libOpenCL.so`（16 KB 页设备会因对齐崩溃）。
3. 真机 **arm64-v8a** 构建并 Run。

Gradle 会在构建前同步 OpenCL 内核到 APK assets（`syncAddsubKernels`）。总览与 16 KB 页约束：[Android/README.md](Android/README.md)。

### 使用：探测与微基准

| 步骤 | 说明 |
|------|------|
| 设备探测 | 启动 App 自动枚举平台/设备；成功标志 `RESULT: PASS (OpenCL usable)` |
| ECM add/sub | UI 四参数对应桌面 `opencl_ecm_addsub.exe` |
| ECM mont mul/sqr | 对应桌面 `opencl_ecm_montsqr.exe`（WG、tpi=4；不含 AMD asm） |

桌面命令对照与默认参数、512-bit 路径列表输出格式：[Android/ECM/README.md](Android/ECM/README.md)。

```bash
adb logcat ECM-OpenCL:I *:S
adb shell run-as com.example.ecm ls -la code_cache/opencl_cache/
```

### 其他：Android 特有行为

- **OpenCL 加载**：`uses-native-library` + 运行时 `dlopen`，不打包 vendor `.so` — [Android/README.md](Android/README.md)
- **编译缓存**：`codeCacheDir/opencl_cache/`；驱动无法导出 binary 时使用 **live program cache** — [Android/ECM/README.md](Android/ECM/README.md)「OpenCL 编译缓存」
- **与桌面差异**：无 AMD 汇编路径；首次编译大型 `mont_priv*.cl` 可能需数分钟

---

## 开发与文档

以下按主题索引子目录文档；**正文仅作入口，细节以子文档为准**。

### 数学原理与 GPU-ECM 流程

| 文档 | 简介 |
|------|------|
| [docs/ECM_GPU_FLOW.md](docs/ECM_GPU_FLOW.md) | stage-1 数学流程：Montgomery ladder、`s` 比特扫描、检查点 |
| [docs/README.gpu](docs/README.gpu) | 上游 CUDA/CGBN GPU-ECM 启用与用法 |
| [docs/README](docs/README) | 上游 ECM/P-1/P+1 基础与 `-param` 选项 |

### GPU-ECM `param` 与调试

| 文档 | 简介 |
|------|------|
| [docs/DEBUG_PARAMETERS_GUIDE.md](docs/DEBUG_PARAMETERS_GUIDE.md) | `cgbn_ecm_stage1` / batch 参数、`gpu_ecm()` 调试输出 |
| [docs/README.lib](docs/README.lib) | `ecm_params` 结构与 `ecm_factor()` 返回值 |

### 算子分析

| 文档 | 简介 |
|------|------|
| [docs/ECM_OPERATOR_ANALYSIS.md](docs/ECM_OPERATOR_ANALYSIS.md) | stage-1 算子混合比、微基准数据、优化优先级（Montgomery 为首要热点） |

### 工具（`tools/`）

| 文档 / 入口 | 简介 |
|-------------|------|
| [tools/DISASM_SETUP.md](tools/DISASM_SETUP.md) | Windows 安装 objdump / llvm-objdump，配合 ISA 导出 |
| [kernels/opencl/bench/mp_addsub/README.md](kernels/opencl/bench/mp_addsub/README.md) | add/sub 内核布局、`gen_all.py` 再生成、bench 优先级 |
| `tools/gen_*.py`、`disasm_*_isa.ps1` | Montgomery/addsub 展开与 asm 块生成；反汇编脚本 |

### 性能测试（`bench/`）

跨厂商总报告：[bench/0530_report.md](bench/0530_report.md)（512 / 4096-bit，NVIDIA / AMD / Intel iGPU）。

| 系列 | 文档 | 主题 |
|------|------|------|
| Montgomery WG | [MONT_WG_SWITCHABLE_FRAMEWORK_CN.md](bench/MONT_WG_SWITCHABLE_FRAMEWORK_CN.md) | 可切换 WG 框架 |
| | [MONT_WG_IMPL4_CROSS_VENDOR_TUNING_CN.md](bench/MONT_WG_IMPL4_CROSS_VENDOR_TUNING_CN.md) | impl4 跨厂商 unroll 调参 |
| | [MONT_WG_MINIMAL_IMPL4_PLAN_CN.md](bench/MONT_WG_MINIMAL_IMPL4_PLAN_CN.md) | 最小 impl4 方案 |
| | [MONT_ISA_4096_ANALYSIS.md](bench/MONT_ISA_4096_ANALYSIS.md) | 4096-bit Montgomery ISA |
| Add/Sub 优化 | [ADDSUB_BASELINE_CN.md](bench/ADDSUB_BASELINE_CN.md) | 4096-bit 纯核基线（AMD gfx1150） |
| | [ADDSUB_ADDMOD_SPECULATIVE_CN.md](bench/ADDSUB_ADDMOD_SPECULATIVE_CN.md) | 投机减模 |
| | [ADDSUB_ADDMOD_FULL_UNROLL_CN.md](bench/ADDSUB_ADDMOD_FULL_UNROLL_CN.md) | 全展开 |
| | [ADDSUB_ADDMOD_ASM_4096_CN.md](bench/ADDSUB_ADDMOD_ASM_4096_CN.md) | 4096-bit asm |
|  profiling / TPI | [RadeonGPUProfiler_1.md](bench/RadeonGPUProfiler_1.md) | RGP 分析记录 |
| | [TPI_1.md](bench/TPI_1.md) | TPI 相关测试 |
|  Intel iGPU | [0530_Intel.md](bench/0530_Intel.md) | 2026-05-30 Intel 核显 raw 记录 |

### AMD 汇编优化

| 文档 | 简介 |
|------|------|
| [docs/README.dev.asm](docs/README.dev.asm) | 上游 asm-redc 目录约定（历史参考） |
| [bench/ADDSUB_ADDMOD_ASM_4096_CN.md](bench/ADDSUB_ADDMOD_ASM_4096_CN.md) | add/sub-mod 4096-bit AMDGCN asm |
| [bench/MONT_ISA_4096_ANALYSIS.md](bench/MONT_ISA_4096_ANALYSIS.md) | Montgomery 4096 ISA 与 asm 路径 |
| `tools/disasm_mont_isa.ps1` | 配合 `opencl_mont_isa_export` 反汇编 |

### IM Compiler（整数乘法代码生成）

| 文档 | 简介 |
|------|------|
| [docs/IM_Compiler/分段整数乘法.md](docs/IM_Compiler/分段整数乘法.md) | 分段整数乘法思路 |
| [docs/IM_Compiler/IMCompiler论文.md](docs/IM_Compiler/IMCompiler论文.md) | 论文摘要 |
| [docs/IM_Compiler/IMCompiler：面向密码学整数乘法的高性能GPU内核自动生成框架.md](docs/IM_Compiler/IMCompiler：面向密码学整数乘法的高性能GPU内核自动生成框架.md) | 框架总述 |

### NPU（Ryzen AI）

| 文档 | 简介 |
|------|------|
| [RyzenAI/README_ADDSUB.md](RyzenAI/README_ADDSUB.md) | NPU add/sub 微基准，对标 OpenCL `opencl_ecm_addsub` |
| [RyzenAI/quicktest/README.md](RyzenAI/quicktest/README.md) | 快速验证脚本 |

### 仓库布局（简图）

```
ECM-OpenCl/
├── src/                    # ecm driver, stage-1 OpenCL host, micro-benchmarks
├── include/                # public headers
├── kernels/opencl/         # OpenCL kernel sources
│   ├── common/             #   shared helpers, operator interface, mp primitives
│   ├── mont_mul/           #   Montgomery multiply kernels
│   ├── add_mod/            #   modular addition kernels
│   ├── sub_mod/            #   modular subtraction kernels
│   ├── bench/              #   micro-benchmark kernels (addsub, mont, asm selftest)
│   └── ecm_stage1*.cl      #   stage-1 ladder entry points
├── cgbn/backends/opencl/   # OpenCL backend runtime (context, build, cache)
├── docs/                   # principles, debug, upstream README copies
├── bench/                  # performance records and tuning notes
├── tools/                  # generators and disassembly
├── Android/ECM/            # Android App
├── RyzenAI/                # NPU micro-benchmarks
└── test/                   # CUDA/OpenCL correctness suite (Makefile)
```

---

## 其他文档索引

以下为仓库内 **未在上文单独展开** 的 Markdown / 说明文件（已排除 `.gitignore` 中的 `.refactor/`、`.github/`、`build/`、`docs/ecm/` 等）：

| 路径 | 说明 |
|------|------|
| [README_en.md](README_en.md) | 英文项目说明 |
| [docs/README.dev](docs/README.dev) | 上游 autotools 开发说明 |

`test/` 下 Makefile 驱动的 CUDA/OpenCL 测试源文件（无独立 `.md` 索引）用于内核正确性验证；CUDA bench 头文件见 `test/bench_cgbn_*.h`。
