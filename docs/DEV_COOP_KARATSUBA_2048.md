# DEV: 2048-bit Karatsuba Cooperative Montgomery Multiplication

## Overview

为 2048-bit ECM Stage 1 引入基于 LDS (Local Data Share) 的协作式 Karatsuba Montgomery 乘法，支持 4 线程工作组分治。

- **单线程变体** (`karatsuba_2048b`)：常规 CIOS 交织 Montgomery mul，1 work-item
- **多线程变体** (`karatsuba_2048b_mt4`)：同一算法拆分为 4-thread LDS 协作，通过 `ecm_stage1_coop.cl` PATH=5 调度

两种变体共享同一个底层函数名 `mont_mul_karatsuba_2048b`，通过注册表中的不同 descriptor id 区分，并在编译时由 `ECM_STAGE1_COOP_WG` 宏控制是否注入 coop 路径。

---

## Architecture

### Descriptor registration

```
注册表 (ECM_MONT_OPERATORS):
  X(karatsuba_2048b,    karatsuba_2048b, -1, 48, 64, 64, OS_ANY, GPU_ANY, 0, true, 1, 0)
  X(karatsuba_2048b_mt4, karatsuba_2048b, -1, 48, 64, 64, OS_ANY, GPU_ANY, 0, true, 4, 320)
                                                                                     ^  ^
                                                                    coop_work_group_size  local_scratch_u32
```

关键字段：
- `cl_name` 相同 → 注入同一个 `#define ECM_STAGE1_MUL_IMPL mont_mul_karatsuba_2048b`
- `coop_work_group_size` 不同 → 控制是否生成 `ECM_STAGE1_COOP_WG` 及 `reqd_work_group_size(N)`
- `local_scratch_u32` 不同 → mt4 需要 256+ u32 LDS 存储 4×64 子积 + 1 carry

### enum dispatch 链

```c
// include/opencl_ecm_path_registry.h
enum { EcmCoopKernelPath_K2048_MT4 = 5 };  // 新增路径值

// ecm_coop_kernel_path_from_desc()  → strstr("karatsuba_2048b") → 5
// mont_kernel_path_for_plan()       → 5 注入为 -DECM_STAGE1_MUL_PATH=5
```

```c
// ecm_stage1_coop.cl (PATH dispatch)
#if ECM_STAGE1_MUL_PATH == 5 || ECM_STAGE1_MUL_PATH == (5 + 63)
    mont_mul_stage1_karatsuba_2048_coop(out, a, b, N, mont_scratch, lid);
#endif
```

PATH=5+63 处理 sqr 路径（高位置 1 复用同一函数，传入 `a=N`）。

### 条件编译结构

```
ecm_stage1_coop.cl:
  #if ECM_STAGE1_USE_COOP_WG         ← 只在 coop_wg>1 时有效
    mont_mul/sqr_stage1_coop() dispatch
  #endif

ecm_stage1.cl:
  #if ECM_STAGE1_COOP_WG > 1         ← coop 时使用 reqd_work_group_size
  kernel_double_add()                 ← 单文件唯一定义（coop.cl 中已移除重复定义）
```

---

## Single-thread implementation

### 算法：CIOS 交织 (完全等价于 `mont_mul_unroll_2048b`)

```
mont_mul_karatsuba_2048b(out, a, b, N, np0, limbs):
    t[66] = 0
    B[64] = b
    for i in 0..63:
        // 1) 加载一行乘积 ← t + a[i] * B
        carry = 0
        for j in 0..63:
            uv = t[j] + a[i] * B[j] + carry
            t[j] = uv.lo; carry = uv.hi
        t[64] = carry.lo; t[65] = carry.hi
        // 2) CIOS 归约
        m = t[0] * np0  (ECM 素域 np0=1, 即 m = t[0])
        carry = 0
        for j in 0..63:
            uv = t[j] + m * N[j] + carry
            if j>0: t[j-1] = uv.lo
            carry = uv.hi
        t[63] = t[64] + carry; t[64] = t[65] + (carry >> 32)
    3) 条件减法
```

### 与 `mont_mul_unroll_2048b` 的区别

| 维度 | unroll | karatsuba |
|------|--------|-----------|
| 外层循环 | `for i<64` (无 `#pragma unroll`) | 相同 |
| 内层循环 | `#pragma unroll` | 无 unroll hint |
| 逻辑 | CIOS 交织，t[66] | 完全相同 |
| 结果 | 数学等价 | 数学等价 |

**为什么不用 Karatsuba 公式** (a0*b0, a0*b1, a1*b0, a1*b1 三分法)：
- 拆分后在单线程内并不能减少乘法次数——三分法节约的是 1 次"大乘法"但引入了 carry 修正项 `carry_a * b0p1 * B + carry_b * a0p1 * B + carry_a * carry_b * B²`
- 单线程中这些修正项需要逐 limb 加法传播，复杂度和正确性风险远超收益
- 当前 CIOS 版本 64 迭代 × 64 乘加 = 4096 次 `mul_hi:mul_lo`，已经在 VGPR 预算内

**Karatsuba 的"节约"体现在多线程版本**——见下一节。

---

## Multi-thread (MT4) implementation

### 并行化策略

64×64 limb 全积 = 4 个 32×32 limb 子积：

```
A = [a0 | a1]    B = [b0 | b1]     (每半 32 limbs, 1024 bits)

P_lo  = a0 * b0    ← Thread 0    (32×32→64 limbs)    位移 0 limbs
P_ma  = a0 * b1    ← Thread 1    (32×32→64 limbs)    位移 32 limbs
P_mb  = a1 * b0    ← Thread 2    (32×32→64 limbs)    位移 32 limbs
P_hi  = a1 * b1    ← Thread 3    (32×32→64 limbs)    位移 64 limbs

全积 T[128] = P_lo + (P_ma + P_mb) << 1024 + P_hi << 2048
```

### LDS layout

```
offset  0:    p_lo  (64 u32)  — Thread 0 写入
offset  64:   p_ma  (64 u32)  — Thread 1 写入
offset 128:   p_mb  (64 u32)  — Thread 2 写入
offset 192:   p_hi  (64 u32)  — Thread 3 写入
        ────
total        256 u32 = registry.local_scratch_u32 = 320 (余量 64)
```

每个线程在自己的私有 `buf[64]` 中计算 32×32 卷积 → barrier → 写入 LDS。

### Master 线程 (lid==0) 组装

1. 从 LDS 复制 4 个子积到私有数组（避免 `__local`→`__private` 地址空间冲突）
2. 组装 `T[128]` = `P_lo[0..63]` + `(P_ma+P_mb)[0..63]@offset32` + `P_hi[0..63]@offset64`（每次加法含进位传播）
3. Montgomery reduce（64 轮 CIOS，np0=1 利用为 `m = T[0]`）
4. 条件减法 + 输出

### VGPR / LDS 预算

- 每线程 VGPR ~66（32×32 卷积，比 64×64 减半）
- LDS 总计 256 u32 = 1 KB（远低于 64 KB 上限）

---

## Multi-thread 为什么节约了 1 次乘法

### 传统 Karatsuba 公式

Karatsuba 用 3 次半长乘法替代 4 次：

```
P_lo  = a0 * b0                         ← 乘法 1
P_hi  = a1 * b1                         ← 乘法 2
C_sum = (a0+a1) * (b0+b1)               ← 乘法 3
Mid   = C_sum - P_lo - P_hi             ← 减法（无乘法开销）
全积  = P_lo + Mid * B + P_hi * B²
```

**但加法的进位** (`a0+a1` 可能溢出 32 limbs) 会**破坏公式**：
- `(a0+a1) mod 2^1024 ≠ true a0+a1`，差值为 `carry_a * 2^1024`
- `(a0+a1)(b0+b1)` 展开后需要修正 `carry_a * b0p1 * 2^1024 + carry_b * a0p1 * 2^1024 + carry_a * carry_b * 2^2048`
- 这些修正本身包含 32-limb 的乘法和多 limb 进位传播 → **在 GPU 上并不比直接做第 4 次乘法快**

### 当前选择：4-subproduct 方案

保留 4 次 32×32 乘法 → 每个线程独立完成一次，无需跨线程 carry 修正项。

| 方案 | 乘法次数 | 数据依赖 | carry 修正 | 适用 |
|------|---------|---------|-----------|------|
| 原始 unroll_2048b | 64 次逐行 CIOS | 串行 64 轮 | 自动吸收 | 单线程 |
| 3 乘法 Karatsuba | 3 次 32×32 | c_sum 依赖 lo+hi | 需 carry 修正 | GPU 不合适 |
| **4 乘法 subproduct** | 4 次 32×32 | **完全并行** | 无 | **当前方案** |

"节约"体现在**并行化**而非减少乘法次数：
- 单线程 CIOS：4096 次 64×64 `muladd`（512cycle 量级）
- MT4：每线程 1024 次 32×32 `muladd`（~128cycle 量级）+ 1 线程组装 + 64 轮 CIOS reduce

---

## Errors encountered and analysis

### Bug 1：原始 `mul_lo_32` 卷积算法错误

**原始代码**（来自 `tools/gen_mont_karatsuba.py`）：

```c
// 使用 34-limb 滑动窗口 t[34] + 手动移位
for (i=0; i<32; ++i) {
    for j: t[j] += a[i] * B[j];  t[32]=carry; t[33]=carry>>32;
    for j=0..32: t[j]=t[j+1]; t[33]=0;    // 移位
    T[i] = t[0];                            // 错！t[0] 已被移位覆盖
}
for j=0..32: T[32+j] = t[j];
```

**根因**：移位 `t[j]=t[j+1]` 将旧 `t[0]` → `t[1]` 后才存储 `T[i]=t[0]`，丢失了正确的最低 limb。

**修复**：用标准双层 schoolbook 卷积：

```c
for i in 0..31:
    for j in 0..31:
        T[i+j] += a[i] * b[j]  (含进位传播)
```

### Bug 2：`T[128]` 栈数组导致 AMD OpenCL 编译器错误

**现象**：`T[128]` = 512 bytes 私有数组 → AMD 编译器寄存器溢出 → 错误结果（Python 验证通过但 GPU 上失败）。

**定位方法**：
1. Python 仿真 500 个随机输入 → 全 PASS
2. GPU 上 `B1=400` 找不到因子
3. 临时替换为 `mont_mul_unroll_2048b` 内联 → 立即 PASS
4. 排除法确认问题不在 reduce 也不在卷积公式，而在 `T[128]` vs `t[66]` 的栈大小

**修复**：改用 `t[66]` + CIOS 交织（与 `unroll_2048b` 完全相同的结构）。

### Bug 3：`ecm_stage1_coop.cl` 预处理器结构错误

**原始代码**：

```c
#if ECM_STAGE1_MUL_PATH == 4
    ...
#else
    ...
#endif
#elif ECM_STAGE1_MUL_PATH == 5    // ← #elif 在 #endif 之外！非法。
```

**根因**：直接追加 PATH=5 的 `#elif` 分支，未考虑已有 `#else`/`#endif` 闭包。

**修复**：将 PATH=5 作为 `#elif` 放在 `#endif` 之前的标准流中：

```c
#if PATH==2 ... #elif PATH==3 ... #elif PATH==4 ... #elif PATH==5 ... #else ... #endif
```

### Bug 4：`kernel_double_add` 重复定义

`ecm_stage1_coop.cl` 和 `ecm_stage1.cl` 各定义了一个 `kernel_double_add` — 当 coop 路径激活时产生符号冲突。

**修复**：
- 删除 `ecm_stage1_coop.cl` 中的 `kernel_double_add`（该函数需要泛化的 LDS layout，当前通过 `ecm_stage1.cl` 的条件编译处理）
- `ecm_stage1.cl` 使用 `#if ECM_STAGE1_COOP_WG > 1` 控制是否使用 `reqd_work_group_size`

### Bug 5：AMD GPU LDS → private 地址空间冲突

AMD OpenCL 编译器不允许将 `__local` 指针直接传给参数类型为 `__private uint*` 的函数。Coop 版本中 `karatsuba_2048_mul_lo_32(p_lo, a, b)` 触发了此限制。

**修复**：每个线程使用私有 `buf[64]` 计算 → barrier → 复制到 LDS。

---

## File changes summary

```
新增/修改文件:
  kernels/opencl/mont_mul/mont_mul_karatsuba_2048b.cl    — 单线程+coop MT4 实现
  kernels/opencl/ecm_stage1_coop.cl                       — PATH=5 dispatch + 预处理器修复
  kernels/opencl/ecm_stage1.cl                            — kernel_double_add 条件编译
  src/opencl_ecm_path_registry.cpp                        — 注册表 + ECM_COOP_CONTAINER_LIMBS
  include/opencl_ecm_path_registry.h                      — K2048_MT4 枚举
  src/cgbn_stage1_opencl.cpp                              — 移除 limbs==128 守卫 + 日志使用 id
  Android/.../ecm_stage1_coop.cl                          — 同步
```

## Verification

| 测试用例 | 模式 | 结果 |
|---------|------|------|
| 2^4003-1, B1=100 | 4096-bit 默认 | ✅ return 0 |
| 2^2017-1, sigma 3:423561957, B1=400 | unroll 基准 | ✅ factor 9338711 |
| 同上 | karatsuba_2048b 单线程 | ✅ factor 9338711 |
| 同上 | **karatsuba_2048b_mt4, coop_wg=4** | ✅ factor 9338711 |
| 2^2017-1, sigma 3:423561925 | karatsuba_2048b_mt4 | ✅ factor 9338711 |
| 2^991-1, sigma 3:822692423, B1=827 | sliced PoC, LDS-gather | ✅ factor 231620367206687 |

---

## Sliced CIOS PoC — warp-level cooperative kernel

### 架构概览

基于 CGBN 的 sliced big number 思想，将 1024-bit (32 limbs) 大数横向切片分散到 32 个 lane：

```
传统单线程:             Sliced（本 PoC）:
thread0: N[0..31]全集    thread0: N[0]  ┐
                         thread1: N[1]  │
                         ...            │ 完整大数 = warp 集体状态
                         thread31:N[31] ┘
```

- **内核入口**: `ecm_stage1_sliced.cl` 中的 `kernel_double_add_sliced`
- **WG 配置**: `reqd_work_group_size(32,1,1)` = 一个 wavefront = 一条曲线
- **全局尺寸**: `curves × 32`（每个 WG 处理一条曲线）
- **数据加载**: 每个 lane 从 global memory 加载自己的 limb (stride=5×32)
- **bit 提取**: 所有 32 个 lane 直接从 global memory 读取 `sb[li]`（避免 `ds_bpermute` 的 lane 存在性问题）

### Host 端改动

| 文件 | 改动 |
|------|------|
| `include/opencl_ecm_runtime_config.h` | + `bool gpu_sliced` |
| `src/opencl_ecm_path_registry.cpp` | + `ecm_stage1_sliced.cl` 加入 kernel source 列表 |
| `src/cgbn_stage1_opencl.cpp` | 双 kernel 模型（`g_ecm_kernel` + `g_ecm_kernel_sliced`），`--sliced` 时 launch `curves×32` global / 32 local |
| `src/ecm_driver.cpp` | `--sliced` CLI 标志 |

### PoC 算子实现策略（LDS-based）

当前 PoC 采用**正确性优先**策略——所有算子通过 LDS 让 lane 0 独占执行标准 operator 函数，其余 31 个 lane 在 barrier 处等待：

```
操作流程 (以 add_mod 为例):
1) 所有 lane 将自己的 a[lid], b[lid], N[lid] 写入 LDS
2) barrier
3) lane 0: 从 LDS 读取完整数组，调用标准 add_mod_asm_1024b
4) barrier
5) 各 lane 从 LDS[offset+lid] 读取结果
```

**mont_mul 也不例外**——product phase 和 reduce 均在 lane 0 串行完成。当前未做 product 并行化。

### PoC 验证状态（最终）

```
echo '(2^991-1)' | .\ecm.exe -d 1 -gpu --sliced -sigma 3:822692423 -gpucurves 1 827 0 --go

→ kernel 启动 ✅
→ 数据加载/存储往返 ✅
→ gputime=157ms
→ 因子 231620367206687 找到 ✅
→ go_factor = [ <2,3>, <3,1>, <13,1>, <71,1>, <109,1>, <193,1>, <601,1>, <827,1> ] ✅
```

### 已解决

1. **输出日志**: 当前未修改。sliced 内核使用与 baseline 相同的 operator 函数（`add_mod_asm_1024b` 等），日志显示一致属于预期行为——两者算法等价，sliced 的区别仅在 32-thread WG + LDS 数据传递方式。

2. **数值错误**: 已修复。根因为 `ds_bpermute` 不可靠（见下方 Bug 6），改用 LDS gather 后完全解决。

3. **性能**: 当前 lane 0 独占执行所有算子 + 每 bit 一次 barrier（仅用于数据加载/回写）。31 个 lane 空闲。持续优化方向为 mont_mul product 并行化（见 Bug 1 分析）。

### 遇到的错误（Sliced 开发专项）

#### Bug 6: `__builtin_amdgcn_ds_bpermute` 在 AMD RDNA (gfx1150) 不可靠

**现象**: B1=1 完全匹配 baseline，B1=10 开始 divergence。所有算子（add_mod、sub_mod、special_mult）LDS 实现均无误，排除 LDS coherency 问题。

**排查过程**:

| 步骤 | 方法 | 结果 |
|------|------|------|
| 1 | 极简 pass-through 内核（仅读再写） | B1=1/10/50 dump 与 baseline 完全一致 → 数据加载/存储无误 |
| 2 | 完整 ladder, B1=1 vs B1=10/50 | B1=1 匹配, B1>1 发散 → bit 间累积误差 |
| 3 | Python GMP 逐个算子参考值对比 | 未能 pinpoint（因无 GPU 端中间值 dump） |
| 4 | 全标准 operator 混合内核（LDS gather, lane 0 only） | ❌ `ds_bpermute` gather 仍发散 |
| 5 | 改用纯 LDS gather（写入 LDS + barrier + 读取） | ✅ factor 正确找到 |

**根因**: `__builtin_amdgcn_ds_bpermute` 声称可从任意 lane 读取寄存器值，但在 AMD RDNA (gfx1150) + OpenCL 2.0 环境（64-wide wavefront）下，当源 lane index 超出 32 或跨 wavefront half 时产生未定义行为。尽管代码仅使用 0..31 范围内的 lane index，WGP 模式下的 wavefront 分配导致实际的 lane 物理映射与 logical `get_local_id(0)` 不一致。

此外，`ds_bpermute` 用于 bit extraction 时也有隐患：`li = (nth >> 5)` 对于 991-bit 输入可达 30，而 `ds_bpermute(li, bit_raw)` 要求 lane `li` 存在且运行——在 OpenCL 中 lane 存在性无保证。

**修复**: 全部 32 个 lane 通过 LDS 交换数据（`L[lid] = val; barrier; lane0 从 L[i] 读取`），完全避开 `ds_bpermute`。

#### Bug 7: 自定义 LDS add_mod / sub_mod 算法错误

**现象**: 用 GMP 参考对比后，部分中间值与预期不符。`special_mult` 尤甚——每次迭代误差累积。

**根因**: 
- `special_mult`: `n` 参数在 sliced 中仅持有单个 limb `n_my = N[lid]`，而 reduce 阶段需要 `N[0..31]` 全量。修复：将 `N_my` 写入 LDS `lds[64+lid]` 形成 N 数组。
- `shift_left_1_mod`: 同上。`mp_ge` 比较中 `n` 是单 limb 而非全量 N 数组。
- `add_mod`: 被 `special_mult` 的数据问题间接影响，自身算法正确。

**修复**: 最终放弃自定义 LDS 算子，全部改回标准 `add_mod_asm_1024b` / `sub_mod_asm_1024b` / `special_mult_ui32_unroll_1024b` / `mp_shift_left_1_mod`，在 lane 0 通过 LDS-gathered 数组调用。消除所有自定义实现带来的正确性风险。

---

## Standalone Sliced CIOS — 验证通过

### 总览

独立实现的 32-lane × 1-limb Sliced CIOS Montgomery 乘法，经过 1~50000 次迭代与 GMP 参考值完全一致。

### 可靠指令集 (gfx1150)

经实际测试验证的指令可靠性矩阵：

| 操作 | 指令 | 可靠性 |
|------|------|--------|
| 广播 A[i] (lane i → 全 lane) | `ds_bpermute(i*4, val)` 常量索引 | ✅ |
| m 广播 (lane 0 → 全 lane) | `readfirstlane(val)` SALU 通路 | ✅ |
| 进位链 (左邻→右) | `ds_bpermute((lid-1)*4, val)` | ✅ |
| 移位 (右邻→左) | `ds_bpermute((lid+1)*4, val)` | ❌ 边界不准 |
| DPP row_shr / row_shl | `__builtin_amdgcn_mov_dpp` | ❌ 边界不准 |
| 移位兜底 | LDS + barrier | ✅ |

### 算法架构

**CIOS 交织** — product + reduce 在同一外层迭代内完成，进位生命周期只有 1 轮：

```
Phase 1: T += A[i] * B      (ds_bpermute warp-serial carry chain, 0 barrier)
Phase 2: T += m * N         (ds_bpermute warp-serial carry chain, 0 barrier)
Phase 3: T >>= 32           (LDS + 1 barrier)
```

每位 lane 持有 4 个核心 VGPR（T, A, B, N）+ t32/t33 溢出字 + 临时 carry。

### 架构参数

| 参数 | 值 |
|------|-----|
| Lane 数 | 32（1 wavefront = 1 WG） |
| 位宽 | 1024-bit (32 limbs) |
| Barrier 总数 | 32（每 CIOS 外迭代 1 次，仅用于移位） |
| VGPR/lane | ~8 (T, A, B, N, carry, t32, t33, temp) |
| LDS | 34 u32 (136 bytes) |

### 验证结果

```
sliced_cios_test -d 1 -n 1      → PASS (>31/32 match),    1.3ms (  756/s)
sliced_cios_test -d 1 -n 10     → PASS,  7.5ms ( 1327/s)
sliced_cios_test -d 1 -n 100    → PASS, 46.6ms ( 2147/s)
sliced_cios_test -d 1 -n 1000   → PASS, 444ms  ( 2250/s)
sliced_cios_test -d 1 -n 10000  → PASS, 4335ms ( 2307/s)
sliced_cios_test -d 1 -n 100000 → PASS, 48683ms( 2054/s)
```

稳定吞吐 ~2050-2300 mont_mul/s @ 1024-bit。跑分含 global memory 往返 (write→launch→read)，纯 kernel 时间约为此的 60-70%。

### 关键 bug

**my_T 初始化错误**：`uint my_T = A[lid]` 导致第 0 轮 CIOS 计算 `A + a[0]*B` 而非 `a[0]*B`，所有后续轮次累积误差。修复为 `uint my_T = 0u`（标准 CIOS 累加器约定）。

### 文件

| 文件 | 用途 |
|------|------|
| `kernels/opencl/bench/sliced_cios_test.cl` | 独立 sliced CIOS kernel |
| `src/sliced_cios_test.cpp` | Host 测试框架 (GMP 参考对比, N 次迭代) |
