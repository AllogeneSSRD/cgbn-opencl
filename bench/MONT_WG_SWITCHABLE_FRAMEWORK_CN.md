# OpenCL Montgomery WG 统一可切换框架 + GMP 护栏（中文说明）

本文档说明当前 `mont_wg` 的实现如何切换、如何在命令行调用，以及如何使用 GMP 对结果进行护栏校验。

## 1. 设计目标

- 在同一套代码中支持多种 `mont_wg` 实现，便于 A/B 性能对比。
- 保持 Stage1 主流程与 microbench 使用同一数学核心。
- 每次改动后都能用 GMP 快速验证 WG 乘法/平方正确性。

## 2. 实现模式（MONT_WG_IMPL）

在 `cgbn/backends/opencl/kernels/mont_wg.cl` 中通过宏 `MONT_WG_IMPL` 选择实现：

- `0`：旧版（`tid==0` 串行 CIOS）
- `1`：并行 base 项 + 串行全归并
- `4`：并行 base 项 + 串行全归并（2-limb 展开，默认）

> `impl=2/3`（分块归并、并行前缀 scan）已移除，不再支持。

相关宏：

- `MONT_WG_IMPL`：实现模式（0 / 1 / 4）
- `MONT_WG_IMPL4_UNROLL`：`impl=4` 时 merge 展开因子（`1` 或 `2`，默认按 GPU vendor 自动选择）

## 3. 运行时切换方式

Host 侧通过环境变量注入 OpenCL 编译参数：

- `ECM_MONT_WG_IMPL`：0 / 1 / 4（默认 4）
- `ECM_MONT_WG_IMPL4_UNROLL`：1 / 2（可选，覆盖 vendor 自动策略）

### 3.1 Stage1 主程序（ecm.exe）

```powershell
$env:ECM_DISABLE_MONT_WG = "0"
$env:ECM_OPENCL_TPI = "16"
$env:ECM_MONT_WG_IMPL = "4"
'(2^3919-1)' | .\build\Debug\ecm.exe -v -gpu -d 1 -sigma 3:12345678 -gpucurves 128 1e3 0
```

日志示例：

`OpenCL: built kernel MAX_LIMBS=... TPI=... WG_IMPL=4 IMPL4_UNROLL=2 NORM=1 (...)`

### 3.2 Microbench（opencl_ecm_addsub.exe）

```powershell
$env:ECM_MONT_WG_IMPL = "4"
.\build\Debug\opencl_ecm_addsub.exe -d 0 --bits 4096 --use-wg --tpi 16 1000 128 3
```

## 4. GMP 护栏

`opencl_ecm_addsub --use-wg` 会自动校验 `mont_mul_wg` / `mont_sqr_wg`，通过标志为 `GMP verify: PASS`。

## 5. 推荐 A/B 流程

固定 `bits`、`instances`、`tpi` 后依次对比：

1. `ECM_MONT_WG_IMPL=0`（旧版基线）
2. `ECM_MONT_WG_IMPL=1`（串行 merge）
3. `ECM_MONT_WG_IMPL=4`（当前默认）

记录 `mont_mul_wg` / `mont_sqr_wg` 时间与 `GMP verify` 结果。

## 6. 回滚

- 临时切回 impl1：`$env:ECM_MONT_WG_IMPL = "1"`
- 禁用 WG 路径：`$env:ECM_DISABLE_MONT_WG = "1"`
