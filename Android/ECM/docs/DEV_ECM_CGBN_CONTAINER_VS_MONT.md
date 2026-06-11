# CGBN 容器位宽 vs Montgomery 算子路径 — 开发提醒

本文专门澄清 Stage-1 日志里 **`CGBN<BITS,TPI>`** 与 **`GPU: stage1 operators: mul=...`** 的关系，以及 **`unroll384`（12-limb）** 的合法 N 范围。避免把「512 容器」误读成「应用了错误的 Montgomery 内核」。

相关总览见 [`DEV_ECM_OPERATOR_PATHS.md`](DEV_ECM_OPERATOR_PATHS.md)。

---

## 1. 两类完全不同的「位数」

Stage-1 host 同时维护两套概念，**日志会分别打印**：

| 概念 | 典型日志 | 决定因素 | 含义 |
|------|----------|----------|------|
| **CGBN 容器 / template** | `GPU: CGBN<512,8> kernel` | `select_bits(n_log2)` → `BITS`；`limbs = BITS/32` | OpenCL 编译时的 **数据布局与模加/模减** 位宽；32-bit 路径最小档为 **512**（无单独 384 档） |
| **Montgomery mul/sqr 算子** | `mul=mont_mul_priv_unroll_only_384` | `opencl_ecm_resolve_stage1_mont_mode()` + `ecm_stage1.cl` 分发 | **CIOS 乘法** 使用的实现与展开宽度 |
| **N 的真实位数** | `371-bit N` | `mpz_sizeinbase(N, 2)` | 模数本身大小；**Auto 规则用这个**，不用容器 `BITS` |

**提醒：`CGBN<512,8>` 与 `mul=...unroll_only_384` 可以同时出现，这不是 bug。**

---

## 2. 示例：`2^371 − 1`（371-bit N）

典型日志：

```text
GPU: CGBN<512,8> kernel, 371-bit N, ...
GPU: stage1 operators: mul=mont_mul_priv_unroll_only_384, sqr=mont_sqr_priv_unroll_only_384, ...
```

解读：

1. **`371-bit N`** — 模数位数。
2. **`CGBN<512,8>`** — `371 + CARRY_BITS(6) ≤ 512` → `select_bits` 选 **512-bit 容器**，`limbs=16`，`TPI=8`。曲线状态、`addmod`/`submod` 仍按 **16×32-bit** 布局编译。
3. **`unroll_only_384`** — Auto 判定 `371 + 6 < 384` → 合法使用 **12-limb CIOS 全展开** 的 Montgomery mul/sqr；算子只处理 **前 12 个 limb**，结果 limb 12–15 写 0，与 512 容器布局兼容。

**不应把第一行改成 `CGBN<384,8>`。** Stage-1 没有 384-bit CGBN 容器档位；384 仅指 mul/sqr 内核的 CIOS 宽度。

---

## 3. `select_bits`：容器如何选

```cpp
// src/cgbn_stage1_opencl.cpp
static uint32_t select_bits(size_t n_log2) {
    static const uint32_t candidates[] = {
        512, 1024, 1280, 1536, ... , 9216};
    for (uint32_t b : candidates) {
        if (n_log2 + CARRY_BITS <= b)
            return b;
    }
    return 0;
}
```

- `CARRY_BITS` = `ECM_STAGE1_MONT_CARRY_BITS` = **6**（ladder 中间运算余量）。
- 32-bit 路径：**第一个 ≥ N+CARRY 的候选**即为 `BITS`（512 起跳）。
- i24 路径 **不用** `select_bits` 定 `limbs`，而用 `ecm_limb24_stage1_limbs()`（见 operator 文档 §3）。

---

## 4. `unroll384`：12-limb 算子 vs 512 容器

### 4.1 内核在做什么

`mont_mul_stage1_unroll_only_384` / `mont_mul_priv_unroll_only_384_body`（`ecm_stage1.cl`、`mont_priv_opt.cl`）：

- **B 缓存**：读满 **16 limb**（512-bit 布局）。
- **CIOS 主循环**：仅 **12×12 limb** 全展开（`ECM_STAGE1_384_LIMBS = 12`）。
- **约减 / 最终减**：只用 `N[0..11]`。
- **输出**：`out[0..11]` 为结果，`out[12..15] = 0`。

因此这是嵌在 **512-bit 私有布局** 里的 **384-bit CIOS 特化**，不是独立的 384-bit CGBN template。

### 4.2 合法 N 范围（数学正确性）

Host 判定（`src/opencl_ecm_mont_path.h`）：

```cpp
constexpr size_t ECM_STAGE1_UNROLL384_MAX_BITS = 384u;
constexpr size_t ECM_STAGE1_MONT_CARRY_BITS = 6u;

inline bool opencl_ecm_stage1_n_fits_unroll384(size_t n_bit_size) {
    return n_bit_size + ECM_STAGE1_MONT_CARRY_BITS < ECM_STAGE1_UNROLL384_MAX_BITS;
}
```

等价于 **`N < 384 − CARRY_BITS`**，即 **`N ≤ 377`**（整数 bit 数）。

| 条件 | 说明 |
|------|------|
| ✅ `371 + 6 < 384` | 可用 unroll384 |
| ✅ `377 + 6 = 383 < 384` | 可用 unroll384（上界） |
| ❌ `378 + 6 = 384 ≮ 384` | **禁止** unroll384；应 unroll512 / priv_opt |
| ❌ `421 + 6 > 384` | 421-bit preset 必须 **unroll512**（或更大容器上的 generic 路径） |

**切勿**用「512 容器上限 506」（`N + CARRY ≤ 512`）来判断 unroll384 是否合法；506 只表示 **Stage-1 仍用 512-bit 容器**，不表示 12-limb CIOS 足够。

### 4.3 显式选 `unroll_only_384` 但 N 过大

`opencl_ecm_resolve_stage1_mont_mode()` 会打印 warning 并回落 **`UNROLL512`**：

```text
Warning: unroll_only_384 requires N+6<384 bits (N<378), got 421; using unroll512
```

---

## 5. mul/sqr Auto 分档（当前实现）

常量（`opencl_ecm_mont_path.h`）：

- `ECM_STAGE1_AUTO_I24_BLSUB_MAX_BITS = 264`（i24 blsub 相对 unroll384 有性能优势的区间上界）
- `ECM_STAGE1_UNROLL384_MAX_BITS = 384`
- `ECM_STAGE1_MONT_CARRY_BITS = 6`

**mul 与 sqr 均为 auto** 时（优先级见 `opencl_ecm_resolve_stage1_mont_mode()`）：

| N 位数（约） | Montgomery 模式 | 典型日志 mul/sqr | 容器 |
|-------------|-----------------|------------------|------|
| `< 264` | `I24_U32_BLSUB` | `mont_mul_unroll_i24_u32_blsub` | i24 `limbs×24` |
| `264 … 377` | `UNROLL384` | `mont_mul_priv_unroll_only_384` | **512** / 16 limb（32-bit 布局） |
| `378 … 506` | `UNROLL512` | `mont_mul_priv_unroll_only_512` | 512 / 16 limb |
| `> 506` 或无固定 32-bit 路径 | **`I24_U32`**（兼容回退） | `mont_mul_unroll_i24_u32` | i24 `limbs×24` |
| i24 limb 数 > `OPENCL_ECM_MAX_LIMBS` | `PRIV_OPT`（最后手段） | `mont_mul_stage1_priv_opt` | 32-bit `select_bits` 容器 |

UI 文案（`arrays.xml`）：`自动 (<264b→i24 blsub; 264–377b→unroll384; 378–506b→unroll512; 更大→i24)`。

显式 path（`unroll512`、`priv_opt`、`unroll32` 等）**优先于** 上表 Auto 规则。

**371-bit auto 预期：** `unroll384` + `CGBN<512,*>`（不是 i24，也不是 priv_opt/unroll32）。

---

## 6. 兼容回退路径（无固定 bit 特化时）

固定路径（unroll384 / unroll512 / 4096 特化）只覆盖有限 N 区间。**Auto 在无法命中固定路径时**，不再使用 `unroll32` / `priv_opt` 作为默认回退（Adreno 上实测 ~8.5s vs i24 ~2.9s @371-bit）。

当前兼容回退链（`opencl_ecm_stage1_compatible_mont_fallback()`）：

1. **`mont_mul_unroll_i24_u32`** — 任意 N（在 `OPENCL_ECM_MAX_LIMBS` 内）均可；Stage-1 切换 i24 容器与 `-DECM_STAGE1_USE_I24_384=1`。
2. **`mont_mul_stage1_priv_opt`** — 仅当 i24 limb 数超过 host 缓冲上限时的最后手段。

`unroll32` / 显式 `priv_opt` 仍可在 UI 中手动选择（bench / 对照），**不会**再作为 auto 默认。

### 6.1 i24 相对 unroll384 的性能区间

bench 结论（见 `Android/性能测试mulsqr.md`）：**i24 仅在与 384-bit 固定路径对比时，在 `< 11 limb`（约 `< 264-bit` N）才有优势**。因此：

- **`< 264` auto → i24 blsub**（小 N 最快）
- **`264 … 377` auto → unroll384**（固定 12-limb CIOS，优于 i24 @371-bit）
- **`≥ 378` 且仍在 512 容器 → unroll512**

### 6.2 未来：固定路径阶梯 vs 兼容路径

若按约 **1.5× bit** 间隔部署全展开固定路径（384 → 512 → 768 → 1152 → …），且每个档位覆盖对应 N 区间，则 **兼容回退（i24 / priv_opt）可逐步淘汰**。当前尚未铺满阶梯，故保留 **i24_u32** 作为通用回退。

候选兼容路径（手动 / 应急，非 auto 默认）：

| 路径 | 适用 | 备注 |
|------|------|------|
| `i24_u32` / `i24_u32_blsub` | 任意 N（limb 上限内） | **当前 auto 大 N 默认** |
| `priv_opt` | i24 放不下时 | 32-bit 容器 + 运行时 limbs |
| `unroll32` | bench 对照 | 性能差，勿作 auto 回退 |

---

## 7. 读日志 checklist

1. **`CGBN<BITS,TPI>`** → 问：容器与 add/sub 布局多大？（`select_bits` 或 i24）
2. **`XXX-bit N`** → 问：模数真实位数？（Auto 阈值看这个）
3. **`stage1 operators: mul=...`** → 问：Montgomery 实际用哪段 CL？
4. 若 **`CGBN<512,*>` + `unroll384`** 且 **`N ≤ 377`** → **正常**
5. 若 **`> 506` auto** → 应见 **`mont_mul_unroll_i24_u32`** 与 **`CGBN<limbs×24,*>`**，不是 priv_opt

---

## 8. 相关源文件

| 文件 | 内容 |
|------|------|
| `src/opencl_ecm_mont_path.h` | 阈值、`opencl_ecm_stage1_compatible_mont_fallback()` |
| `src/opencl_ecm_mont_path.cpp` | Auto / 显式 path 解析 |
| `src/cgbn_stage1_opencl.cpp` | `select_bits`、`GPU: CGBN<...>` 打印、`ensure_ecm_kernel` |
| `cgbn/backends/opencl/kernels/ecm_stage1.cl` | `mont_mul_stage1_unroll_only_384`、`mont_mul_stage1()` 分发 |
| `cgbn/backends/opencl/kernels/mont_priv_opt.cl` | bench 同源 `mont_mul_priv_unroll_only_384_body` |

---

## 9. 历史误区（避免重复）

| 误区 | 正确理解 |
|------|----------|
| 「Auto 384–506 用 unroll384」 | **错误**。506 是 512 **容器**上限；unroll384 仅 **N+CARRY < 384** |
| 「12 limb = 可表示 506-bit N」 | **错误**。12×32=384 bit 是 CIOS 宽度；506 来自 512−CARRY |
| 「见 unroll384 就应显示 CGBN<384,*>`」 | **错误**。容器仍为 512；384 只在 operators 行体现 |
| 「371-bit auto 应走 priv_opt」 | **错误**。264–377 应 **unroll384**；991-bit 等应 **i24_u32** 回退 |
