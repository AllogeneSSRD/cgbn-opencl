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
