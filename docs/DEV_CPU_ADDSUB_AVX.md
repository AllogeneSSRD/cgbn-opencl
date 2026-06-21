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
| `--csv <file>` | — | 写结果 CSV | — |

### 测试数据生成

与 `opencl_ecm_addsub` 使用完全相同的确定性 GMP 随机数方案：
- 两个测试用例：`overflow (a+b>=N)` 和 `no-overflow (a+b<N)`
- 确定性 seed: `bits × 31337 + case_index × 0x9e3779b9`
- `--no-overflow` 标志选择用例（默认为 overflow）
- 所有线程/instance 复用同一个 (a,b,N)，保证结果可复现和对标

## 可用算子

### Fused（通用，任何位宽）

`cpu_add_fused_scalar` / `cpu_sub_fused_scalar`：纯 C 融合模加/模减，add + 条件减 N 单 pass 完成。
是 ECM stage1 中 `add_mod_fused` 的 CPU 等价实现。

受限于 fused add/sub 的双串行进位链特性，**当前仅标量实现有实际价值**。

## 实现架构

```
cpu_addsub_bench.cpp           (main + benchmark loop)
  ├── cpu_addsub_impl.h        (标量融合 add/sub)
  │     ├── cpu_add_fused_scalar
  │     └── cpu_sub_fused_scalar
  └── GMP                       (random operand generation)
```

### 与 OpenCL addsub bench 的对应关系

| CPU | OpenCL | 说明 |
|-----|--------|------|
| `cpu_add_fused_scalar` | `add_mod_fused_body` | 通用融合 add-sub，任何 limbs |
| `cpu_sub_fused_scalar` | `sub_mod_fused` | 融合 sub，borrow 链 |

## 扩展记录: AVX2/AVX512 纵向 SIMD（已放弃）

### 尝试的方案

| 变体 | 策略 | 状态 |
|------|------|------|
| `avx2_manual` | 纵向：每 8 limbs SIMD load/add/store，carry 链在标量端传播 | **已删除** |
| `avx2_lookahead` | 纵向 + overflow/propagation mask 进位预测 | **已删除** |
| `avx512_manual` | AVX512 纵向：每 16 limbs SIMD load/add | **已删除** |
| `unroll_*b` | 固定位宽 guard wrapper（标量调用），与标量等价 | **已删除** |

### 性能测量 (512-bit, Zen5 AVX2)

```
scalar:           6.1 ms, 26.3 M ops/s   (baseline)
avx2_manual:     12.8 ms, 12.5 M ops/s   (0.48×, 衰退 2.1×)
avx2_lookahead:  31.1 ms,  5.1 M ops/s   (0.20×, 衰退 5.1×)
```

4096-bit 测试结论一致——垂直 SIMD 在位宽增大后无改善。

### 放弃理由

Fused modular add/sub 算法有 **两条串行进位链**（add-carry + sub-borrow），这与 SIMD 的"无数据依赖并行"模型根本冲突：

1. **核心瓶颈不可向量化**：72% 的周期消耗在两条 carry/borrow 链的标量传播上，SIMD 无法触及
2. **SIMD 引入负摊还**：bulk load/add 仅占 ~8% 工作量，但 store→scalar riple→reload 往返消耗额外的 15-35% 周期
3. **lookahead 无优势**：8-lane 内 overflow/propagation mask 仍需串行扫描（退化为 O(8) scalar），加上 SIMD ↔ mask 格式转换开销
4. **固定位宽 unroll 等价于标量**：limb guard 检查 + 直接调用标量循环，无任何加速路径

### 正确的向量化方向：横向 SoA 批处理（已实现）

`cpu_mont_bench` 的 **Structure-of-Arrays 批处理** 已在 `cpu_addsub_impl.h` 中实现、
在 `cpu_addsub_bench` 中与标量路径并行测试。ipt 强制对齐 K (8 或 16)，默认保持 16。

**SoA 性能 (Zen5 AVX2, Release /O2)**：

| 算子 | 512-bit | 1024-bit | 4096-bit |
|------|---------|----------|----------|
| cpu_add_fused (scalar) | 209 ms | 390 ms | 1959 ms |
| cpu_add_avx2_soa | 147 ms (**1.42×**) | 377 ms (1.04×) | 1147 ms (**1.71×**) |
| cpu_sub_fused (scalar) | 156 ms | 493 ms | 1589 ms |
| cpu_sub_avx2_soa | 87.5 ms (**1.79×**) | 231 ms (**2.14×**) | 652 ms (**2.44×**) |

**多线程 (96 ipt × 4t, 512-bit)**：
| add scalar: 1418 ms | add SoA: 1016 ms (**1.39×**) |
| sub scalar: 1054 ms | sub SoA: 646 ms (**1.63×**) |

位宽越大 SoA 加速越显著。**必须用 Release 编译测试**——Debug 下不内联 intrinsic、寄存器分配退化。

## 与其他性能基准的关系

| 基准 | 依赖 | 测试对象 | 对标 |
|------|------|----------|------|
| `cpu_addsub_bench.exe` | GMP | CPU fused add/sub (scalar + SoA) | `opencl_ecm_addsub.exe` |
| `opencl_ecm_addsub.exe` | OpenCL + GMP | GPU asm/unroll/fused | `ecm_stage1.cl` |
| `cpu_mont_bench.exe` | GMP | CPU mont mul/sqr | `opencl_ecm_montsqr.exe` |
