# CPU Add/Sub AVX 微基准 — 开发文档

## 概述

`cpu_addsub_bench.exe` 是独立于 OpenCL 的 CPU 端模加/模减性能基准。与
`opencl_ecm_addsub.exe` 使用相同的 CLI 接口和 GMP 操作数生成逻辑，结果可直接
对标。

### 命令行

```powershell
# 位置参数: [bits] [kernel-iterations] [ipt] [repeats]

# 延迟测试（1 thread × 16 ipt，overflow 用例）
.\build\Debug\cpu_addsub_bench.exe 512 1e6 16 1

# 延迟测试（no-overflow 用例）
.\build\Debug\cpu_addsub_bench.exe 512 1e6 16 1 --no-overflow

# 吞吐量测试（12 threads × 16 ipt）
.\build\Debug\cpu_addsub_bench.exe 512 5e4 16 5 -t 12 -a 1,3,5,7,9,11,13,15,17,19,21,23

# 科学计数法 + 命名参数混合
.\build\Debug\cpu_addsub_bench.exe -b 512 -k 1e6 -i 16 -r 1

# 写入 CSV
.\build\Debug\cpu_addsub_bench.exe 512 5000 64 3 --csv bench.csv
```

### 参数

| 参数 | 短形式 | 说明 | 默认值 |
|------|--------|------|--------|
| `[bits]` | `-b` / `--bits` | 位宽（32 的倍数，≤16384） | 1024 |
| `[kernel_iterations]` | `-k` / `--kernel-iters` | 每个 instance 的内层循环次数，支持科学计数法（1e6） | 1000 |
| `[ipt]` | `-i` / `--ipt` | 每个线程的 instance 数（ipt × threads = total instances） | 16 |
| `[repeats]` | `-r` / `--repeats` | 测量重复次数 | 10 |
| `--threads <N>` | `-t` | 线程数 | 1 |
| `--affinity c1,c2` | `-a` | Pin 线程 t 到核心 c_t | auto |
| `--no-overflow` | — | 使用 a+b < N 测试数据（默认 a+b >= N） | false |
| `--unroll` | — | 仅测试固定位宽 unroll 路径 | false |
| `--csv <file>` | — | 写结果 CSV | — |

### 测试数据生成

与 `opencl_ecm_addsub` 使用完全相同的确定性 GMP 随机数方案：
- 两个测试用例：`overflow (a+b>=N)` 和 `no-overflow (a+b<N)`
- 确定性 seed: `bits × 31337 + case_index × 0x9e3779b9`
- `--no-overflow` 标志选择用例（默认为 overflow）
- 所有线程/instance 复用同一个 (a,b,N)，保证结果可复现和对标

## 可用算子

### Fused（通用，任何位宽）

`cpu_add_fused_c` / `cpu_sub_fused_c`：纯 C 融合模加/模减，add + 条件减 N 单 pass 完成。
是 ECM stage1 中 `add_mod_fused` 的 CPU 等价实现。

### Unroll（固定位宽）

`cpu_add_mod_unroll_<W>b` / `cpu_sub_mod_unroll_<W>b`：仅当 `limbs == W/32` 时执行
（guard 检查）。12 个注册宽度：

192b(6)、256b(8)、384b(12)、512b(16)、768b(24)、1024b(32)、
1536b(48)、2048b(64)、2560b(80)、3072b(96)、3584b(112)、4096b(128)

## 实现架构

```
cpu_addsub_bench.cpp           (main + benchmark loop)
  ├── cpu_addsub_impl.h        (C+AVX2+AVX512 implementations)
  │     ├── cpu_add_fused_c    (scalar fallback)
  │     ├── cpu_sub_fused_c    (scalar fallback)
  │     └── cpu_add_mod_unroll_*b()  (width-guarded wrappers)
  └── GMP                       (random operand generation)
```

### 与 OpenCL addsub bench 的对应关系

| CPU | OpenCL | 说明 |
|-----|--------|------|
| `cpu_add_fused_c` | `add_mod_fused_body` | 通用融合 add-sub，任何 limbs |
| `cpu_sub_fused_c` | `sub_mod_fused` | 融合 sub，borrow 链 |
| `cpu_add_mod_unroll_512b` | `add_mod_unroll_512b` | 512b 固定位宽 guard |
| `cpu_sub_mod_unroll_1024b` | `sub_mod_asm_1024b_body` | 1024b guard（asm 路径在 CPU 上等价 unroll） |

### AVX2/AVX512 编译门控

CMakeLists.txt 自动检测并设置：

- MSVC: `/arch:AVX2`（启用 AVX2+FMA）
- GCC/Clang: `-mavx2 -mfma`
- AVX512 可选：`-mavx512f -mavx512dq`（当前未默认启用，需手动添加）

`cpu_addsub_impl.h` 通过 `#if defined(__AVX512F__)` / `#if defined(__AVX2__)` 条件
编译 SIMD 路径。当编译器标志生效时，`cpu_add_fused_c` 被相应的向量化版本替换。

## 延迟 vs 吞吐量设计

```powershell
# 延迟：1 thread × 16 ipt（最小并行）
cpu_addsub_bench -b 512 -k 1e6 -i 16

# 吞吐量：12 threads × 16 ipt = 192 total instances
cpu_addsub_bench -b 512 -k 5e4 -i 16 -t 12 -a 1,3,...,23
```

## 扩展 AVX2 / AVX512（已完成）

### 实现的算法变体

`include/cpu_addsub_impl.h` 现提供以下实现，benchmark 自动运行所有可用变体并对比：

| 变体 | 说明 | 位宽选择 |
|------|------|----------|
| `cpu_add_fused_scalar` | 纯 C 融合 add/sub，一条 lane 一条 carry | 任意 |
| `cpu_add_fused_avx2_manual` | AVX2 纵向 SIMD：批 8 limbs 做 load/add/store，carry 链标量传播 | 任意 |
| `cpu_add_fused_avx2_lookahead` | AVX2 纵向 SIMD + 进位预测（overflow/propagation mask） | 任意 |
| `cpu_sub_fused_*` | 对应 sub 变体 | 任意 |

AVX2 编译门控：MSVC 已默认启用 `/arch:AVX2`，`cpu_addsub_bench` Debug 构建即可使用 AVX2 变体。

### 性能结论 (512-bit, Zen5)

```
cpu_add_scalar:        6.1 ms, 26 M ops/s   (baseline)
cpu_add_avx2_manual:  12.8 ms, 12.5 M ops/s (0.48×)
cpu_add_avx2_lookahead: 31 ms, 5 M ops/s    (0.20×)
```

**垂直 SIMD 对 fused add/sub 无益**。原因：
1. Fused add/sub 有**两条串行进位链**（add-carry + sub-borrow），无法 SIMD 化
2. SIMD 仅加速 bulk load/add（工作量 <10%），但引入 store→scalar-ripple→load 往返开销（~40% overhead）
3. 进位预测（lookahead）在 8-lane 内仍需串行扫描，无算力优势

### 横向 SoA 批处理（后续）

`cpu_mont_bench` 的 SoA 批处理模式对 addsub 同样适用：
- 16 个 instance 的同一 limb 打包成 `__m512i`，各 lane 独立进位
- 无跨 lane 进位依赖，可充分利用 SIMD 算力
- 适用于多 instance 批量 benchmark 场景

## 与其他性能基准的关系

| 基准 | 依赖 | 测试对象 | 对标 |
|------|------|----------|------|
| `cpu_addsub_bench.exe` | GMP | CPU fused/unroll | `opencl_ecm_addsub.exe` |
| `opencl_ecm_addsub.exe` | OpenCL + GMP | GPU asm/unroll/fused | `ecm_stage1.cl` |
| `cpu_mont_bench.exe` | GMP | CPU mont mul/sqr | `opencl_ecm_montsqr.exe` |
