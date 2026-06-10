# Android Montgomery mul/sqr 性能分析（Adreno 642 / 830）

完整 512-bit 路径跑通后，结论与早期（仅 `priv_opt` / `wg` 有效时）完全不同。本文汇总 **32-bit limb 全路径 bench** 与 **24-bit limb `mont_mul_unroll_i24`（mul24）** 实测，并给出选型与优化建议。

> **构建说明**：`fips512_mt*` 与 `wg` @512 已从 manifest / bench 中剔除，不再编译或跑分。

配置：`instances=128`，`kernel_iterations=1000`，`launch_repeats=1`，WG 模式 `tpi=4`。

> **数据说明**：24b 与 384@32 对照段曾将两台 GPU 对调；已按 **Adreno 830 绝对性能高于 642** 校正（与同文档 §1 历史 512@32 unroll 数据一致：830 为 M ops/s 档，642 为百 K～1.4M 档）。

---

## 1. 总览：512-bit @ 32-bit limb（历史完整 bench）

| 路径 | Adreno 642 mul | Adreno 830 mul | 830/642 |
|------|----------------|----------------|---------|
| **unroll_only_512_manual** | 323K | **8.47M** | 26× |
| **unroll_only_512** (auto) | **1.43M** | 7.73M | 5.4× |
| local_only_512 | 265K | 2.74M | 10× |
| opt2_512_local | 265K | 2.41M | 9× |
| fips512 | 406K | 2.07M | 5.1× |
| priv_opt | 282K | 1.60M | 5.7× |
| priv (legacy) | 273K | 1.05M | 3.8× |
| unroll32/64 | 241–285K | 857–897K | ~3× |
| wg | 201K | 1.05M | 5.2× |
| fips512_mt* / cs | 174–218K | 831K–1.85M | — |

**结论**：512-bit 必须用 **512 专用路径 `unroll_only_512*`**，不能用 generic `priv_opt` / `wg` / `unroll32`。与桌面 stage1 默认一致。

### sqr @ 512-bit / 32-bit limb

| 路径 | 642 sqr | 830 sqr |
|------|---------|---------|
| unroll_only_512 | 355K | 3.57M |
| fips512 | 407K | 2.19M |
| priv_opt | 279K | 1.58M |
| wg | 198K | 1.07M |

---

## 2. 新增：24-bit limb（`mont_mul_unroll_i24`）实测

内核：`mont_mul_unroll_i24.cl`，CIOS + `#pragma unroll`，内乘用 `mul24`（12-bit 分解实现 24×24→48）；支持任意 `bits % 24 == 0`（512 固定 22 limbs）。

| 配置 | limbs | Adreno **830** mul | Adreno **830** sqr | Adreno **642** mul | Adreno **642** sqr |
|------|-------|--------------------|--------------------|--------------------|--------------------|
| **512-bit @24** | 22 | **1.57M** | **1.59M** | 541K | 542K |
| **384-bit @24** | 16 | **3.56M** | **3.61M** | 881K | 881K |

**830/642 @24**：512-bit mul 约 **2.9×**；384-bit mul 约 **4.0×** — 与 §1 中 830 整体强于 642 一致。

对照：**384-bit @32**（同次 bench，通用路径，12 limbs）：

| 路径 | Adreno **830** mul | Adreno **642** mul |
|------|--------------------|--------------------|
| priv_opt | **2.61M** | 482K |
| unroll32 | 1.51M | 485K |
| priv | 1.83M | 476K |
| wg | 1.83M | 340K |

---

## 3. 24b vs 32b：核心结论

### 3.1 不能只看 bit 宽度，要看 limb 数与路径类型

| 对比 | 说明 |
|------|------|
| 512@32 **unroll_only** | 16 limbs，全展开专用体 → **两台机各自最快档** |
| 512@24 **unroll_i24** | 22 limbs（⌈512/24⌉），CIOS + mul24；limb 多 **37.5%**，`mont_i24_mul_full` 开销大 |
| 384@32 **priv_opt** | 12 limbs，运行时 `limbs` 循环 |
| 384@24 **unroll_i24** | 16 limbs → 比 384@32 多 **33%** limb，但无 runtime 循环 |

Montgomery CIOS 工作量近似 **∝ limbs²**：

- 512：22² / 16² ≈ **1.89×** 更多内层乘加（相对 32b unroll_only）
- 384：16² / 12² ≈ **1.78×**

### 3.2 与各自「同位宽、同 GPU 最优」对比（mul）

| GPU | 位宽 | 24b unroll_i24 | 32b 最优（本仓库） | unroll_i24 / 32b 最优 |
|-----|------|------------|-------------------|-------------------|
| **830** | 512 | **1.57M** | unroll_only **7.73–8.47M** | **~20%**（慢约 **5×**） |
| **830** | 384 | **3.56M** | priv_opt **2.61M** | **~136%**（快 **1.36×**） |
| **642** | 512 | 541K | unroll_only **1.43M** | **~38%**（慢约 **2.6×**） |
| **642** | 384 | 881K | priv_opt 482K | **~183%**（快 **1.83×**） |

### 3.3 修正后的判断（两台机共性 + 差异）

**共性（生产 @512-bit）**

- **两台机 @512-bit 均应使用 32b `unroll_only_512*`**，**不要**用 unroll_i24 作 Mont 生产路径。
- unroll_i24 @512 仅为 unroll_only 最优的 **20%（830）～38%（642）**；此前误写「642 上 unroll_i24 略快于 unroll_only」系 GPU 标反所致。

**830（绝对性能高）**

- @512：unroll_only manual **8.47M** 仍为终极档；unroll_i24 **1.57M** 有 mul24 但仍远落后。
- @384：unroll_i24 **3.56M** > priv_opt **2.61M**，绝对值也远高于 642 的 881K。
- 若模数 <512 且无专用 unroll，**830 上 unroll_i24 值得评估**；@512 仍坚持 32b unroll。

**642（绝对性能低）**

- @512：unroll_only auto **1.43M** >> unroll_i24 **541K**（约 **2.6×**）。
- @384：unroll_i24 **881K** 仍优于 priv_opt **482K**，但绝对值仅为 830 同配置的四分之一左右。
- **manual unroll** 在 642 上仍劣于 auto（§1），与 unroll_i24 选型无关。

### 3.4 为何 unroll_i24 @512 无法替代 32b unroll_only（两台机）

| 因素 | 说明 |
|------|------|
| Limb 数 | 22 vs 16 → CIOS 约 **1.9×** 工作量 |
| 乘法语义 | `u24_mul_full` 每步 **4× mul24** + 组合；32b unroll 为单条 32×32→64 |
| 专用展开 | `unroll_only_512_body` 为 16-limb 深度优化；unroll_i24 为通用 22-limb CIOS |
| 绝对算力 | 830 把 32b unroll 推到 **8M+** ops/s；unroll_i24 在 830 上虽达 **1.57M**（仍强于 642），但相对自身最优仍差 **5×** |

### 3.5 为何 unroll_i24 @384 能赢 generic 32b（两台机，830 赢得更多）

- `priv_opt` / `unroll32` 无固定位宽展开，带 `limbs` 循环。
- unroll_i24 全展开 + mul24，在 **384@24（16 limb）** 上弥补 limb 数增加的开销。
- **830** 上绝对 **3.56M vs 2.61M**；**642** 上 **881K vs 482K** — 相对提升类似，绝对差距由 GPU 算力决定。

### 3.6 sqr

两台机 **sqr ≈ mul**（unroll_i24 用 `mul(a,a)`）。32b 侧仍有 `mont_sqr_priv_unroll_only_512_body` 可挖。

---

## 4. 为什么 `unroll_only_512`（32b）远快于 `priv_opt`

`mont_mul_priv_unroll_only_512_body`（`mont_priv_opt.cl`）：

- 固定 **16 limb**，无运行时 `limbs` 分支
- `#pragma unroll` 内外层全展开 → 纯 ALU 链
- N/B 私有缓存，modulus 走 `__constant`
- 无 local memory、无 WG 协作

`priv_opt` 为通用 `limbs` 循环 + 部分展开，Adreno 上难以达到同等常量折叠与指令调度。

---

## 5. 路径分类（工程建议）

| 类别 | 路径 | 建议 |
|------|------|------|
| **两台机生产默认 @512** | `unroll_only_512` 32b | 830 mul→**manual**；642→**auto**；**均不用 unroll_i24** |
| **830 @384 及以下** | unroll_i24 或补 12-limb unroll | unroll_i24 已优于 priv_opt（3.56M vs 2.61M） |
| **642 @384 及以下** | unroll_i24 优于 priv_opt | 绝对性能低，端到端仍受 GPU 限制 |
| **禁止 @512 生产** | `mont_mul_unroll_i24` | 两台机均显著慢于 32b unroll |
| bench 已剔除 | `fips512_mt*`、`wg` @512 | 手机上全面落后 |
| 642 慎用 | `unroll_only_512_manual` | 实测比 auto 慢 4.4× |

---

## 6. 公平对比尚未完成项

| 对比 | 目的 |
|------|------|
| **288-bit @24**（12 limbs）vs **384-bit @32**（12 limbs） | 与 add/sub 相同「公平 limb 数」口径 |
| **504-bit @24**（21 limbs）vs **512-bit @32** unroll_only | ECM 常用 504@24 |
| unroll_i24 **hot 内核**（单次 enqueue，`inner=kernel_iterations`） | 剥离 global 重载 |
| unroll_i24 vs unroll_only @512 + **VERIFY** | 确认两台机均不切换生产 |

---

## 7. 优化方向（按优先级，含 unroll_i24）

### P0 — 算子选型

1. **@512-bit（830 / 642）**：`mont_mul_priv_unroll_only_512` + `mont_sqr_priv_unroll_only_512`（830 mul 可试 manual）
2. **勿将 unroll_i24 作 @512 生产路径**（两台机均已证实落后 32b unroll）
3. **@384 及以下**：可评估 unroll_i24；830 收益更大（绝对 3.56M 档）
4. micro-probe 仅保留：`unroll_only_512` vs `_manual`（按 GPU），**不必** probe unroll_i24@512

### P1 — unroll_i24 内核（research / 非 512 生产）

5. **`u24_mul_full` 减负**：`mad24`、更短 24×24→48 分解
6. **22-limb 全手动展开**（仿 512_manual 生成器）— 目标拉高 830 上 1.57M，仍难追 8M unroll
7. **专用 sqr body**（24-bit `sqr_basecase` + REDC）
8. **hot 内核**（对齐 addsub `fused_hot`）

### P2 — 编译与产物

9. 生产 APK：512@32 只链 `unroll_only_512*`；unroll_i24 仅 bench / 非 512 实验
10. 642 manual 退化：RGA/ISA 分析
11. 修正 bench 文案：384@24 勿显示「22 limbs for 512-bit」

### P3 — 端到端

12. Mont 选型后瓶颈或在 add/sub、global 带宽、curve 调度
13. 830 sqr：专用 `mont_sqr_priv_unroll_only_512_body`

### P4 — bench 工具

14. 输出「相对最优」列；按 GPU 标注 `recommended`

---

## 8. 数据速查表（校正后）

### mul ops/s

| 配置 | Adreno **830** | Adreno **642** | 830/642 |
|------|----------------|----------------|---------|
| 512 @32 unroll_only_manual | **8.47M** | 323K | 26× |
| 512 @32 unroll_only auto | 7.73M | **1.43M** | 5.4× |
| 512 @24 unroll_i24 | **1.57M** | 541K | 2.9× |
| 384 @24 unroll_i24 | **3.56M** | 881K | 4.0× |
| 384 @32 priv_opt | **2.61M** | 482K | 5.4× |
| 384 @32 unroll32 | 1.51M | 485K | 3.1× |
| 512 @32 priv_opt | 1.60M | 282K | 5.7× |
| 512 @32 wg | 1.05M | 201K | 5.2× |

### 相对各自 @512 最优 mul（830→manual，642→auto unroll）

| 配置 | 830 | 642 |
|------|-----|-----|
| 512 @24 unroll_i24 | **0.19×** | **0.38×** |
| 384 @24 unroll_i24 vs 512 最优 | 0.42× | 0.62× |
| 384 @24 unroll_i24 vs 384 priv_opt | **1.36×** | **1.83×** |

---

## 9. 一句话策略（校正后）

| GPU | 512-bit mul（生产） | 512-bit sqr | 384-bit 及以下 | 避免 @512 |
|-----|---------------------|-------------|----------------|-----------|
| **Adreno 830** | `unroll_only_512_manual` | `unroll_only_512` | unroll_i24 或补 unroll（**3.56M** 档） | **unroll_i24**、wg、priv_opt |
| **Adreno 642** | `unroll_only_512` auto | `unroll_only_512` | unroll_i24 > priv_opt（**881K** 档） | **unroll_i24**、manual、wg |

**两台机 @512-bit：Mont 一律 32b `unroll_only_512*`，不用 unroll_i24。**

桌面 stage1 已默认 `unroll_only_512`（32b）。Android port 与 §1 历史结论一致；unroll_i24 仅作 **<512 bit 或无专用 unroll 时** 的研究/备选，且 **830 上绝对收益远高于 642**。

---

## 10. 与 add/sub limb24 的联动

add/sub 在 Adreno 上可从 limb24 + `fused_hot` 获益。Montgomery CIOS **以内层乘为主**：

- **@512**：add/sub 可用 24b；**Mont 必须用 32b unroll_only**（两台机 `mont_mul_unroll_i24` 均远慢于 unroll）
- **@384**：两台机 Mont unroll_i24 均可优于 priv_opt；**830 绝对 3.56M vs 642 881K**，端到端瓶颈仍在 642 算力

stage1 应对 **add/sub 与 mont 分开配置**；勿假设 unroll_i24 对 Mont @512 有效。
