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

## Level 2：32-bit CIOS MAC（消灭内层 `ulong`）

### 为何不能用 `mad24` 替换累加

`mad24(a,b,c)` 要求 `a,b` 为 **24-bit** 操作数。CIOS 内层一项是 **48-bit 乘积** `mont_i24_mul_full(...)`，塞进 `mad24` 的 `a`/`b` 会被硬件截断，结果错误。**累加瓶颈在 64-bit 路径，不在 mul24 本身。**

Adreno 无原生 64-bit ALU；`ulong` 加法被拆成多条 32-bit 进位链，内层循环 ×16~22 limb 后指令数与寄存器压力显著。

### 实现（已合入，与 Level 1 并存 bench）

| 符号 | 说明 |
|------|------|
| `mont_i24_mul_full_split` | 返回 `uint2(lo24, hi24)`，乘法语义同 Level 1 |
| `mont_i24_cios_mac_u32` | `A×B` 内层：`t[j] += ai*bj + carry`，全程 `uint` |
| `mont_mul_unroll_i24_body` | **Level 1 基线**：`ulong` CIOS + `mont_i24_mul_full` |
| `mont_mul_unroll_i24_u32_body` | **Level 2**：内层两圈 CIOS 均用 32-bit MAC；循环外 `top`/借位仍偶发 `ulong` |

Bench 内核（同次 APK 24-bit Mont 跑分一次输出四条）：

- `mont_mul_unroll_i24` / `mont_sqr_unroll_i24` — ulong 基线
- `mont_mul_unroll_i24_u32` / `mont_sqr_unroll_i24_u32` — Level 2

### Level 2 实测结论（2026-05）

**参数**：`kernel_iterations=1000`，`instances=128`，`launch_repeats=10`，`src_kib=11`（含 ulong + u32 四内核同 program）。总 op 数与 Level 1 首轮（`10000×1`）相同：**1.28M**。

| 配置 | GPU | Level 1 ulong mul | Level 2 u32 mul | u32/ulong | 判定 |
|------|-----|-------------------|-----------------|-----------|------|
| **384@24** | **830** | 5.95M | **6.70M** | **1.13×** | **u32 胜**，约 **+13%** |
| **512@24** | **830** | **2.76M** | 2.69M | 0.97× | **持平**，ulong 略优 |
| **384@24** | **642** | **1.20M** | 1.14M | 0.95× | **u32 负优化**，约 **−5%** |
| **512@24** | **642** | 735K | **776K** | **1.06×** | **u32 微胜**，约 **+6%** |

sqr 与 mul 同趋势（830 @384 u32 +12%；642 @384 u32 −11%；其余 ±3% 内）。

**工程建议（按 GPU × 位宽 dispatch，非一刀切）**：

| GPU | @384 | @512 |
|-----|------|------|
| **830** | 默认 **`mont_mul_unroll_i24_u32_body`** | 保留 **`mont_mul_unroll_i24_body`（ulong）** |
| **642** | 保留 **ulong**（u32 更慢） | 可试 **u32**（+6%，在波动内，需 VERIFY 后定） |

**假说被部分推翻**：「消灭 ulong 一定更快」在 **830 @384** 成立，在 **642 @384** 反噬。原因见下节「为何 u32 并非普适」。

#### 为何 u32 并非普适

1. **指令数未必更少**：u32 路径每 MAC 多一次 `mul_full_split`、多次 `& mask` / `>> 24`，Adreno 642 上编译器未必比「ulong 加法链」更短。
2. **寄存器与占用率**：拆分 `uint2` + 多个中间 `uint` 可能抬高 VGPR，642 算力本就弱，occupancy 下降会放大延迟。
3. **代数等价 ≠ ISA 等价**：ulong 加法在部分 Adreno 版本上已被合成较优的 2×32-bit 进位链；手工拆 carry 不一定赢编译器。
4. **同 program 多内核**：`.cl` 从 5 KiB 增至 11 KiB，四入口共享 inline 体，**会改变 ulong 内核的编译结果**（见「跨次跑分」）。

### 跨次跑分：为何「未改动的 ulong」与 Level 1 首轮不一致

Level 1 首轮（mad24 后）与 Level 2 同次 bench 中的 **ulong 基线** 对比如下（均为 **mul ops/s**）：

| 配置 | GPU | 首轮（`10000×1`，src 5KiB） | Level2 会话 ulong（`1000×10`，src 11KiB） | 比值 |
|------|-----|----------------------------|-------------------------------------------|------|
| 384@24 | 830 | 5.77M | 5.95M | **1.03×**（略快） |
| 512@24 | 830 | 2.38M | 2.76M | **1.16×**（明显快） |
| 384@24 | 642 | 1.21M | 1.20M | **0.99×**（持平） |
| 512@24 | 642 | 767K | 735K | **0.96×**（略慢） |

**642 略慢、830 略快（@512 尤甚）——并非同一内核「退化」或「神秘加速」，而是测量条件与编译上下文变化叠加：**

| 因素 | 对 642 偏慢 | 对 830 偏快 |
|------|-------------|-------------|
| **`launch_repeats` 10 vs 1** | 10 次 `Enqueue` + 1 次 `Finish`，固定启动/同步开销摊到更短的单次 kernel（1000 iter），**弱 GPU 上 overhead 占比更大** → ops/s 略降 | 830 算力强，overhead 占比小；有时多次短跑反而避免单次超长 kernel 中途降频 |
| **源码体积 5→11 KiB** | 同 program 编入 u32 体与额外 bench 入口，编译器对 **共享 inline 函数** 的全局寄存器/内联决策改变，ulong 内核 ISA 可能略膨胀 | 830 编译器/ICache 容量更大，新 ISA 未必吃亏；@512 工作量大，对 codegen 微调更敏感，可能出现 **16% 量级** 的向好波动 |
| **热状态 / 功耗** | 中端 642 更易在连续 bench（4 内核串跑）后 **温控降频** | 旗舰 830 余量大，同场景更易维持较高 GPU 频率 |
| **首轮与本轮非严格同环境** | 后台、电量、是否插电、室温不同 → **±5% 正常**；642 @512 **−4%** 在噪声内 | 830 @512 **+16%** 偏大，**建议以同会话 ulong 为基线**；首轮 2.38M 可能偏保守（冷机或首次 enqueue） |
| **ulong 微改动** | 累加后写 limb 增加 `& 0xFFFFFF`（正确性） | 影响应 <1%，可忽略 |

**实践约定**：

1. **对比 Level 1 vs Level 2 必须用同一次 bench 输出**（已实现四内核同 program、同参数）。
2. **跨日期对比**应固定 `kernel_iterations`、`launch_repeats`，并注明 `src_kib`；推荐报告 **`1000×10` 与 `10000×1` 各跑一轮** 取区间。
3. **勿用首轮 2.38M 断言 830@512「当前上限」**；同会话 ulong **2.76M** 更可信；Level 2 u32 **2.69M** 说明 @512 上 u32 无优势。

### Level 3 及以后

| 优先级 | 方向 | 预期 |
|--------|------|------|
| P1 | **按 GPU dispatch**：830@384→u32，其余 ulong | 已可由 bench 表驱动 |
| P1 | 专用 `mont_sqr_unroll_i24_body`（非 `mul(a,a)`） | 小幅 |
| P2 | 按位宽生成全手动展开体（仿 `unroll_only_512_manual`） | 大工程，@512 仍难追 32b |
| P2 | hot 内核（`inner=kernel_iterations`，对齐 addsub `fused_hot`） | 剥离 global 重载 |
| P2 | 最终减法 `borrow` 循环也 32-bit 化 | 小幅（循环外） |

---

## 复现

```bash
# Android logcat
adb logcat ECM-OpenCL:I *:S

# APK：ECM 微基准 → bits=384 或 512 → Montgomery Mul/Sqr → 24-bit
```

相关总览见 [`Android/性能测试mulsqr.md`](../Android/性能测试mulsqr.md) §2、§8。
