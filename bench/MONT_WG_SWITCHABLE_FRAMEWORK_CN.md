# OpenCL Montgomery WG 统一可切换框架 + GMP 护栏（中文说明）

本文档说明当前 `mont_wg` 的三种实现如何切换、如何在命令行调用，以及如何使用 GMP 对结果进行护栏校验，避免“提速但算错”。

## 1. 设计目标

- 在同一套代码中支持多种 `mont_wg` 实现，便于 A/B 性能对比。
- 保持 Stage1 主流程与 microbench 使用同一数学核心。
- 每次改动后都能用 GMP 快速验证 WG 乘法/平方正确性。

## 2. 实现模式（MONT_WG_IMPL）

在 `cgbn/backends/opencl/kernels/mont_wg.cl` 中通过宏 `MONT_WG_IMPL` 选择实现：

- `0`：旧版（`tid==0` 串行 CIOS）
- `1`：当前版（并行 base 项 + 串行全归并）
- `2`：分块归并版（并行 base 项 + 分块串行归并）
- `3`：并行前缀进位（scan）实验版

> 说明：`2/3` 为实验实现，当前阶段可能显著退化，部分设备上可能导致图形输出异常。

相关宏：

- `MONT_WG_IMPL`：实现模式
- `MONT_WG_MERGE_CHUNK`：`impl=2` 分块大小（默认 32）

## 3. 运行时切换方式（无需改代码）

Host 侧已支持通过环境变量把宏注入 OpenCL 编译参数：

- `ECM_MONT_WG_IMPL`
- `ECM_MONT_WG_MERGE_CHUNK`

并增加了实验保护开关：

- `ECM_ENABLE_EXPERIMENTAL_WG=1` 才允许 `ECM_MONT_WG_IMPL=2/3`
- 未开启时若设置 `2/3`，会自动回退到 `WG_IMPL=1`

### 3.1 Stage1 主程序（ecm.exe）

示例（PowerShell）：

```powershell
$env:ECM_DISABLE_MONT_WG = "0"
$env:ECM_OPENCL_TPI = "16"
$env:ECM_MONT_WG_IMPL = "1"
$env:ECM_MONT_WG_MERGE_CHUNK = "32"
'(2^3919-1)' | .\build\Debug\ecm.exe -v -gpu -d 1 -sigma 3:12345678 -gpucurves 128 1e3 0
```

日志中会打印类似：

`OpenCL: built kernel MAX_LIMBS=... TPI=... WG_IMPL=... CHUNK=...`

### 3.2 Microbench（opencl_ecm_addsub.exe）

示例（PowerShell）：

```powershell
$env:ECM_MONT_WG_IMPL = "1"
$env:ECM_MONT_WG_MERGE_CHUNK = "32"
.\build\Debug\opencl_ecm_addsub.exe --bits 4096 --use-wg --tpi 16 1000 128 3
```

参数说明：

- 位置参数：`kernel_iterations instances launch_repeats`
- 示例里的 `1000 128 3` 含义：
  - 每次 kernel 内循环 1000
  - 128 个实例（适合你当前 4096-bit 测试）
  - 重复发射 3 轮（建议 3~5）

## 4. GMP 护栏（正确性校验）

`opencl_ecm_addsub` 在 `--use-wg` 模式下会自动执行：

- `cgbn_mont_mul_wg_bench` 对 GMP 校验
- `cgbn_mont_sqr_wg_bench` 对 GMP 校验

通过标志：

- `GMP verify: PASS`

若不一致会打印 mismatch 明细并返回失败。

## 5. 推荐 A/B 流程（固定口径）

建议固定：

- `bits=4096`
- `instances=128`
- `launch_repeats=3~5`
- `tpi=16`

依次执行：

1. `ECM_MONT_WG_IMPL=0`（旧版）
2. `ECM_MONT_WG_IMPL=1`（当前）
3. `ECM_MONT_WG_IMPL=2/3`（新实验）

每组记录：

- `mont_mul_wg` 时间
- `mont_sqr_wg` 时间
- `GMP verify` 是否 PASS
- ECM 端到端 `gputime`

## 6. 注意事项

- `cmd.exe` 里 `^` 需要转义，推荐优先用 PowerShell 输入梅森表达式。
- 先确认 GPU 空闲再做 A/B，避免并发任务污染结果。
- 若你只想对比 WG 路径，请确保：
  - `ECM_DISABLE_MONT_WG=0`
  - `ECM_GPU_DUMP=0`（避免每批同步影响计时）
- 建议常规跑分只使用 `WG_IMPL=0/1`；`WG_IMPL=2/3` 仅在实验环境下测试。

