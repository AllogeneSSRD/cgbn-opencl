# mp_add_mod 推测减法优化分析（4096-bit，AMD gfx1150）

## 结论（能否优化）

| 维度 | 结论 |
|------|------|
| **正确性** | 可以。融合版与 mask 版均通过 GMP 对照（与 legacy 一致）。 |
| **算法** | 消除 `mp_ge` 在数学上成立（a,b ∈ [0,N-1] ⇒ S=a+b ∈ [0,2N-2]）。 |
| **静态 ISA** | 融合版更小：ISA_SIZE 796 vs legacy 1052；动态指令行约 135 vs 187。 |
| **端到端吞吐（v1）** | 未提速：fused ~6% 慢。 |
| **端到端吞吐（v2）** | **略快**：branchless fix + 2-limb unroll 后 fused ≈ **1.03×** legacy（同配置波动内）。 |

**建议**：`mp_add_mod`（fused v2）可作为默认实现；`ecm_mp_add_mod_fused` 核函数已切换。保留 `ecm_mp_add_mod_legacy` 作回退对照。

---

## 原理简述

### Legacy（3 趟最坏情况）

1. `S = a + b`
2. `if (carry || mp_ge(S,N))` ← 完整比较循环
3. `S -= N`

### 融合推测减法（1～2 趟）

单趟：`sum = a+b`，同时 `r[i] = sum + ~N[i] + carry_sub`（即 S−N 的补码链）。

修正条件：`carry_add==0 && carry_sub==0` ⇔ `S < N`，此时 `r += N` 恢复为 S。

无 `mp_ge`。

### Mask 版（无 mp_ge，额外 S[]）

1. `S = a+b`
2. `D = S-N`，`need_sub = carry | (borrow==0)`
3. `r = need_sub ? D : S`（位掩码选择）

private 增至 2592 B（多一块 `S[MAX_LIMBS]`）。

---

## v2 改动（当前 `mp_add_mod`）

1. **主循环**：单趟同时算 `S=a+b` 与 `T=S-N`（`+~N` 补码链），无 `mp_ge`。
2. **修正**：仅当 `carry_add==0 && carry_sub==0`（即 `S<N`）时再跑一趟 `r += N`；否则直接 `return`（跳过原先 branchless 但每 limb 仍执行的 masked fix 循环）。
3. **2-limb 展开**：`MP_ADD_MOD_FUSED_UNROLL=2`（`ECM_MP_ADD_MOD_FUSED_UNROLL=1|2`；**勿设 1**，本机明显更慢）。

**已接入生产路径**：`ecm_stage1.cl` 中 `mp_add_mod` / `mp_add_mod_l` 与 bench 内联实现一致；stage1 编译日志含 `ADDMOD_UNROLL=`。

**v3（分趟 add 再 sub）**：在本机 890M 上慢于 v2，未采用。

## 实测（2026-05-22）

参数：`--addsub-only --bits 4096 1000 128 50`，设备 AMD 890M（`-d 1`）。

| 实现 | ops/s | vs legacy | 备注 |
|------|-------|-----------|------|
| legacy | 2.51M | 1.00x | 含 mp_ge |
| fused v1（有分支） | 2.28M | 0.94x | |
| **fused v2（unroll=2，skip-fix）** | **2.56M** | **~1.02x** | 当前默认 |
| fused v2（unroll=1） | 2.35M | 0.90x | 本机略慢于 unroll=2 |
| mask | 1.85M | 0.76x | 额外 S[] |

基线参考（优化前纯核函数）：`bench/addsub_baseline_4096_amd.csv` 中 legacy 约 **2.63M ops/s**（同配置，存在运行波动）。

---

## 为何 ISA 更小却更慢？

1. **每 limb 更重**：融合循环内同时维护 `carry_add` 与 `carry_sub`，依赖链更长。
2. **VGPR 压力**：fused USED_VGPR=15（legacy=12），可能降低 wave 占用。
3. **修正分支**：`S < N` 时仍要第二趟 `+N`；分支 + 额外访存抵消了去掉 `mp_ge` 的收益。
4. **mask 版**：双倍 private（S+D 语义）+ 三次遍历，吞吐最差。

---

## 工作流命令

```powershell
# 1) 构建
cmake --build build --config Debug --target opencl_ecm_addsub opencl_addsub_isa_export

# 2) RGA（含 legacy / fused / mask）
.\tools\disasm_addsub_isa.ps1 -Bits 4096

# 3) 吞吐 + GMP
$env:ECM_BENCH_CSV = "bench\addsub_opt_4096_amd.csv"
.\build\Debug\opencl_ecm_addsub.exe -d 1 --addsub-only --bits 4096 1000 128 50
```

核函数对照：

- `ecm_mp_add_mod_legacy` — 原算法
- `ecm_mp_add_mod_fused` — 融合推测减法
- `ecm_mp_add_mod_mask` — 掩码选择

---

## 后续可试方向

1. **循环展开 / 2-limb 步进**（仿 `mont_wg impl4`），减轻依赖链气泡。
2. **分支消除**：用掩码修正替代 `if (carry_add==0 && carry_sub==0)` 显式第二循环（常数时间风格）。
3. **仅在 ECM 热路径替换**：先 profile `ecm_stage1` 中 `mp_add_mod` 调用占比，再决定是否值得继续挖。
