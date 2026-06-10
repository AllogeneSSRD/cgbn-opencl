# `mont_i24_mul_full`：mad24 融合优化（Level 1）

## 结论

| 维度 | 结论 |
|------|------|
| **正确性** | 数学等价于原 12-bit 拆分实现；CIOS 外层逻辑未改，仅替换内层 24×24→48 乘法语义。 |
| **改动范围** | `cgbn/backends/opencl/kernels/mont_mul_unroll_i24.cl` 中 `mont_i24_mul_full` 单函数。 |
| **830 @384** | mul **3.56M → 5.77M**（**+62%**，约 **1.62×**） |
| **830 @512** | mul **1.57M → 2.38M**（**+52%**，约 **1.52×**） |
| **642 @384** | mul **881K → 1.21M**（**+37%**，约 **1.37×**） |
| **642 @512** | mul **541K → 767K**（**+42%**，约 **1.42×**） |
| **生产选型 @512** | 仍应使用 32b `unroll_only_512*`；优化后 unroll_i24 约为 830 manual 最优的 **28%**（原 ~20%），差距缩小但未逆转。 |
| **生产选型 @384** | **830 上 unroll_i24 现为最强档**（5.77M > priv_opt 2.61M，约 **2.2×**）；642 上 1.21M > priv_opt 482K（约 **2.5×**）。 |

**建议**：保留 mad24 版本为默认；`@384` 及以下可积极评估 unroll_i24 作 Mont 路径；`@512` 继续禁止用于生产 Mont。

---

## 问题：为何旧实现吃不满 mul24

`mont_mul_unroll_i24` 将每个 24-bit limb 乘分解为四个 12-bit 半 limb，用 `mul24` 做四次部分积，再用 **`+` 在 `ulong` 上累加**：

```opencl
// 优化前（已移除）
const uint p00 = mul24(a0, b0);
const uint p01 = mul24(a0, b1);
const uint p10 = mul24(a1, b0);
const uint p11 = mul24(a1, b1);
ulong mid = (ulong)p01 + (ulong)p10 + (ulong)(p00 >> 12);
const ulong lo48 = ((ulong)(p00 & mask12)) | ((mid & mask12) << 12);
const ulong hi48 = (ulong)p11 + (mid >> 12);
return (hi48 << 24) | lo48;
```

在 Adreno Shader Compiler 上，这会带来：

1. **中间累加脱离 IMAD24 流水**：`+` 倾向于生成普通 32-bit ADD，甚至 64-bit 扩展。
2. **指令数偏多**：4× `mul24` + 多次宽位加法 + 64-bit 拼接。
3. **CIOS 热循环放大**：每个 limb 对内层乘调用 `mont_i24_mul_full`，22-limb @512 时调用次数极多，内层每多一条指令都会线性放大总耗时。

---

## 方案：mad24 融合进位链

OpenCL `mad24(a, b, c)` 在 Adreno 上映射为 **单周期 IMAD24**（24-bit 乘加，结果截断到 32-bit）。利用 12-bit 分解的溢出界，可把交叉项与进位全部塞进 `mad24`，**直到最后一步才拼成 48-bit**：

```opencl
static inline ulong mont_i24_mul_full(uint a, uint b) {
    const uint mask12 = 0xFFFu;
    const uint a0 = a & mask12;
    const uint a1 = a >> 12;
    const uint b0 = b & mask12;
    const uint b1 = b >> 12;

    const uint p00 = mul24(a0, b0);
    const uint mid1 = mad24(a0, b1, p00 >> 12);   // a0*b1 + 进位，< 2^32
    const uint mid2 = mad24(a1, b0, mid1);        // a1*b0 + mid1，< 2^32
    const uint lo48 = (p00 & mask12) | ((mid2 & mask12) << 12);
    const uint hi48 = mad24(a1, b1, mid2 >> 12);
    return ((ulong)hi48 << 24) | lo48;
}
```

### 溢出界（保证 mad24 安全）

| 步骤 | 上界 | 说明 |
|------|------|------|
| `p00` | 2^24 | 12×12 bit 积 |
| `mad24(a0,b1,p00>>12)` | 2^24 + 2^12 | < 2^32 |
| `mad24(a1,b0,mid1)` | 2^24 + 2^25 | < 2^32 |
| `mad24(a1,b1,mid2>>12)` | 2^24 + 2^13 | < 2^32 |

### 指令形态对比

| | 优化前 | 优化后 |
|---|--------|--------|
| mul24 | 4 | 1 |
| mad24 | 0 | 3 |
| ulong 中间加法 | 2+ | 0（仅最终 `\|` 与移位） |

---

## 实测（Android ECM bench）

**环境**：ECM OpenCL APK，`mont_mul_unroll_i24` / `mont_sqr_unroll_i24` 路径。

**参数**：`kernel_iterations=10000`，`instances=128`，`launch_repeats=1`（较历史文档 §2 的 `1000` 迭代更长，绝对 ops/s 略低属正常，**前后对比均用同配置**）。

### Adreno 830

| 位宽 | limbs | mul ops/s | sqr ops/s |
|------|-------|-----------|-----------|
| **384 @24** | 16 | **5.773M** | **5.789M** |
| **512 @24** | 22 | **2.382M** | **2.388M** |

### Adreno 642

| 位宽 | limbs | mul ops/s | sqr ops/s |
|------|-------|-----------|-----------|
| **384 @24** | 16 | **1.212M** | **1.255M** |
| **512 @24** | 22 | **767K** | **765K** |

### 相对优化前（同 GPU，历史 §2 基线，`kernel_iterations=1000`）

| 配置 | 830 优化前 → 后 | 提升 | 642 优化前 → 后 | 提升 |
|------|-----------------|------|-----------------|------|
| 384 @24 mul | 3.56M → **5.77M** | **~1.62×** | 881K → **1.21M** | **~1.37×** |
| 512 @24 mul | 1.57M → **2.38M** | **~1.52×** | 541K → **767K** | **~1.42×** |

sqr 与 mul 几乎持平（sqr 走 `mul(a,a)`，符合预期）。

### 优化后与 32b 路径对照（选型仍有效）

| GPU | 位宽 | unroll_i24（mad24） | 32b 参考最优 | 比值 |
|-----|------|---------------------|--------------|------|
| **830** | 512 | 2.38M | unroll_only_manual **8.47M** | **0.28×** |
| **830** | 384 | **5.77M** | priv_opt **2.61M** | **2.21×** |
| **642** | 512 | 767K | unroll_only **1.43M** | **0.54×** |
| **642** | 384 | **1.21M** | priv_opt **482K** | **2.52×** |

**解读**：

- **@512**：mad24 显著缩小与 32b unroll 的差距（830 从 ~20% 提到 ~28%），但 **仍不足以切换生产路径**。
- **@384**：830 上 unroll_i24 从「略快于 priv_opt」升级为「大幅领先」（**2.2×**），**值得作为 <512 且无专用 unroll 时的首选 Mont 路径**。
- **642 @384**：绝对算力仍约为 830 的 **1/4.8**；相对 priv_opt 优势扩大，端到端仍受 GPU 档次限制。

---

## 工程影响

| 项目 | 说明 |
|------|------|
| 修改文件 | `cgbn/backends/opencl/kernels/mont_mul_unroll_i24.cl` |
| Android 同步 | `syncMontsqrKernels` / assets 同名文件 |
| Bench 入口 | APK「Montgomery Mul / Sqr → 24-bit」 |
| 编译宏 | `-DMAX_LIMBS=<ceil(bits/24)>` `-DMP_LIMB_BITS=24` |

无需改 manifest、JNI 或 host 侧逻辑。

---

## 后续优化（Level 2+）

本次为 **Level 1：内层乘法语义**。CIOS 仍受 **limb 数平方** 约束，@512 有 22 limb vs 32b unroll 16 limb 的结构性差距。

| 优先级 | 方向 | 预期 |
|--------|------|------|
| P1 | `mont_i24_add3` 等 CIOS 累加也尽量 mad24 / 窄位 | 中等 |
| P1 | 专用 `mont_sqr_unroll_i24_body`（非 `mul(a,a)`） | 小幅 |
| P2 | 按位宽生成全手动展开体（仿 `unroll_only_512_manual`） | 大工程，@512 仍难追 32b |
| P2 | hot 内核（`inner=kernel_iterations`，对齐 addsub `fused_hot`） | 剥离 global 重载 |

---

## 复现

```bash
# Android logcat
adb logcat ECM-OpenCL:I *:S

# APK：ECM 微基准 → bits=384 或 512 → Montgomery Mul/Sqr → 24-bit
```

相关总览见 [`Android/性能测试mulsqr.md`](../Android/性能测试mulsqr.md) §2、§8。
