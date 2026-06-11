# Android Stage-1 addmod / submod 路径与并行度

本文解释 Adreno 上 **`fused`** 与 **`fused_unroll_b16`** 为何随 `gpucurves` 变化出现交叉性能，以及 Auto 默认应如何选。

Montgomery mul/sqr 选型见 [`DEV_ECM_CGBN_CONTAINER_VS_MONT.md`](DEV_ECM_CGBN_CONTAINER_VS_MONT.md)。

---

## 1. 实测摘要（347-bit N，`unroll384`，`CGBN<512,8>`）

| gpucurves | s bits | add/sub 路径 | gputime (ms) | 相对更快 |
|-----------|--------|--------------|--------------|----------|
| 1 | 14447 | `fused_unroll_b16` | 1639 | — |
| 1 | 14447 | **`fused`** | **1434** | **−12.5%** |
| 128 | 14447 | `fused_unroll_b16` | 3431 | — |
| 128 | 14447 | **`fused`** | **3332** | **−2.9%** |
| 9216 | **1438** | **`fused_unroll_b16`** | **5842** | **−6.5%** vs fused |
| 9216 | 1438 | `fused` | 6244 | — |

**结论（同一 s 下）：**

- **1～128 curves**：`fused` 稳定更快（约 3%～13%）。
- **9216 curves**：`fused_unroll_b16` 略快（约 6%），但 **s 仅 1438 bit**（非 14447），不宜与上行直接比绝对 ms。

归一化 **ms / (curves × s_bits)**（bit-curve 成本）：

| 场景 | fused_unroll_b16 | fused | fused 优势 |
|------|-------------------|-------|------------|
| 1 × 14447 | 0.113 | **0.099** | +12% |
| 128 × 14447 | 0.00185 | **0.00180** | +3% |
| 9216 × 1438 | **0.000441** | 0.000471 | b16 +7% |

交叉点大约在 **极高并行 + 足够填满 GPU** 时偏向 `fused_unroll_b16`；常见 **1～512 curves** 偏向 `fused`。

---

## 2. 两条路径在 CL 里差什么

512-bit 容器（`limbs == 16`，`MAX_LIMBS == 16`）时，编译宏 `ECM_STAGE1_ADDMOD_PATH` / `SUBMOD_PATH` 决定分支（`ecm_stage1.cl`）：

| path id | 名称 | addmod @16 limb | submod @16 limb |
|---------|------|-----------------|-----------------|
| 0 | **`fused`** | 运行时 limb 循环（add 侧 2-limb 展开，`MP_ADD_MOD_FUSED_UNROLL=2`） | 运行时 limb 循环 |
| 4 | **`fused_unroll_b16`** | `mp_add_mod_fused_unroll_b16_512` → **`#pragma unroll 16` 全展开** | 同上 |

`fused_unroll_b16` 与 `fused_unroll` 在 `MAX_LIMBS==16` 时 **生成的 add/sub 代码相同**；与 `fused` 的差异是 **编译期全展开 vs 运行时循环**。

每个 `double_add_v2`（每条 s bit、每条 curve）固定调用：

- `mp_add_mod` × 4
- `mp_sub_mod` × 4
- `mont_mul` / `mont_sqr` × 4 各
- 以及 normalize、special_mult 等

Montgomery（`unroll384`）占大头，但 add/sub 在 **8 次/步** 仍可观；路径差异会随并行度放大。

---

## 3. 为何并行度会翻转胜负

### 3.1 调度模型

- 512-bit Stage-1：`global_size = curves`，**local_size = null**（驱动自组 WG）。
- 每条 curve 一个 work-item；`TPI=8` 在此 kernel 中 **不拆 curve**（与 CGBN template 名中的 TPI 不同层）。
- `double_add_v2` 私有数组已很大（`t, CB, DA, AA, BB, K, dK` 各 16 limb），**VGPR 压力本已偏高**。

### 3.2 低并行（1～128 curves）

- GPU **填不满**；单次 launch 有效 wave 少。
- `fused_unroll_b16` 把 add/sub 再 **inline 展开 16×**，kernel 体积与寄存器需求上升 → **ICache / 占用率** 变差。
- `fused` 循环体小，**延迟主导** 场景下反而更快。

### 3.3 高并行（9216 curves）

- 足够 work-item 填满 Adreno CU；**吞吐主导**。
- 全展开去掉循环分支，利于指令调度与 SIMD 流水线 → **`fused_unroll_b16` 略胜**。

### 3.4 与 Montgomery 路径叠加

347-bit auto 使用 **`unroll384`（12-limb CIOS 全展开）**，本身已增大 kernel。再叠 add/sub 全展开，**低并行时寄存器/代码膨胀更明显**——这解释了 fused 在 1 curve 上 **~12%** 的差距大于 128 curve 的 **~3%**。

---

## 4. 当前 Android Auto 默认（与实测冲突）

`opencl_ecm_resolve_addsub_path()`（`src/opencl_ecm_addsub_path.cpp`）：

| limbs | GPU | Auto add | Auto sub |
|-------|-----|----------|----------|
| 16 (512 容器) | Adreno（非 AMD） | **`fused`** | **`fused`** |
| 16 | AMD | add: `asm_b16` | `fused_unroll_b16` |

Adreno Auto 已改为 **`fused`**（`opencl_ecm_addsub_path.cpp`）。

i24 容器（`ECM_STAGE1_USE_I24_384`）下，`mp_add_mod` 走 **`mp_add_mod_fused_unroll_i24`**，与上述 32-bit fused/b16 选择无关。

---

## 5. 建议

### 5.1 当前默认（已实现）

**Adreno / 非 AMD + 512-bit 容器：Auto add/sub = `fused`。**

- 常见 **`gpucurves` 1～512** 更快 ~3–13%。
- **`gpucurves` 很大（约 ≥2048）** 时在 UI 手动选 **`fused_unroll_b16`** 可再榨 ~5–7%。

实现：`opencl_ecm_resolve_addsub_path()`，`limbs == 16u && !is_amd` → `ECM_ADDSUB_PATH_FUSED`。

### 5.2 中期：按 curves 选 add/sub（可选）

`resolve_addsub_path` 目前只有 `limbs`，**不知道 curves**。可在 `cgbn_ecm_stage1()` 已知 `curves` 后：

```text
if (limbs == 16 && auto && !is_amd)
  curves >= 2048 → fused_unroll_b16
  else            → fused
```

需扩展 API（`resolve_addsub_paths(..., curves)`）并在 kernel 缓存 key 中加入 curves 阈值或 path id。

### 5.3 长期：与 Montgomery 固定路径同一思路

若按 ~**1.5× bit** 部署 **固定 limb 全展开 add/sub**（512→768→…，与 `unroll384` / `unroll512` 对齐），则：

- 每个 N 区间用 **Dedicated add/sub unroll**，不再依赖通用 `fused` 循环；
- 与「Montgomery 固定路径铺满后取消 i24/priv_opt 兼容层」同一产品路线。

在此之前，**`fused` 作为 512 容器通用回退** 优于 `unroll32` 式 Montgomery 回退的思路，同样适用于 add/sub。

### 5.4 Bench 注意事项

1. **对比不同 gpucurves 时固定 `s` 位数**（你方 9216 用 `s=1438`，1/128 用 `s=14447`，总工作量不同）。
2. 报告 **ms / (curves × s_bits)** 或 **ms / curve**（同一 s）。
3. 记录 **`GPU: stage1 operators:`** 中 add/sub 名；mul 路径（unroll384/i24）会改变 add/sub 的相对权重。
4. 可用 `ECM_PROFILE_OPS=1` 查看算子计数（`cgbn_stage1_opencl.cpp`）。

---

## 6. 手动路径速查

| 场景 | 建议 add/sub |
|------|----------------|
| Adreno，512 容器，gpucurves ≤512，日常 ECM | **`fused`**（Auto 默认） |
| Adreno，512 容器，gpucurves ≥2048 | **`fused_unroll_b16`** |
| AMD，512 容器 | add: **`asm_b16`**（若可用）；sub: b16 / asm 视构建 |
| i24 容器 | 内核内 **i24 fused unroll**（与 UI add/sub 选择基本无关） |
| 4096 容器 | **`fused_unroll_b32`** / AMD **`asm_b32`** |

---

## 7. 相关源文件

| 文件 | 内容 |
|------|------|
| `src/opencl_ecm_addsub_path.cpp` | Auto 默认、`fused` / `fused_unroll_b16` 解析 |
| `cgbn/backends/opencl/kernels/ecm_stage1.cl` | `mp_add_mod` / `mp_sub_mod` 分发 |
| `src/cgbn_stage1_opencl.cpp` | `resolve_addsub_paths`、`global_size=curves`、batch 自适应 |
| `Android/ECM/docs/DEV_ECM_OPERATOR_PATHS.md` | §5 add/sub 总览 |
