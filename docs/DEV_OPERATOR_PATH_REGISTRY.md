# 算子路径注册表（数据驱动）

Stage-1 **Montgomery mul/sqr** 与 **add/sub-mod** 路径集中在注册表中定义、选择与编译宏推导。桌面与 Android 共用 `src/opencl_ecm_path_registry.cpp`。

**mul/sqr 与 add/sub 各自独立解析**，便于组合测试（例如 `--mul unroll_only_384 --sqr unroll_only_512`）。

## 文件

| 文件 | 内容 |
|------|------|
| `include/opencl_ecm_path_registry.h` | 分离描述符、`EcmStage1KernelBuildPlan`、`generate_build_options` |
| `src/opencl_ecm_path_registry.cpp` | `kMontMulRegistry` / `kMontSqrRegistry` / `kAddModRegistry` / `kSubModRegistry` |
| `src/opencl_ecm_mont_path.cpp` | resolve 薄封装、日志名 |
| `src/opencl_ecm_addsub_path.cpp` | add/sub 薄封装 |

## Montgomery 描述符（mul/sqr 共用）

```cpp
struct EcmMontPathDescriptor {
    EcmMontPathKind kind;
    int variant_id;
    const char *id;
    uint16_t min_n_bits, max_n_bits;
    bool max_n_strict;
    uint16_t required_container_limbs;
    // coop_wg_size, stage1_force_macro, ...
};
```

`kMontMulRegistry[]` 与 `kMontSqrRegistry[]` 为两个独立常量数组，条目镜像但 CL 名/别名不同；resolve 仍分 `opencl_ecm_resolve_stage1_mont_mul` / `_sqr`，可独立组合。

add/sub 共用 `EcmAddSubPathDescriptor`，分别挂在 `kAddModRegistry[]` / `kSubModRegistry[]`。

### 位宽判断（统一函数）

```cpp
bool ecm_path_n_bit_fits(min_n_bits, max_n_bits, max_n_strict, n_bit_size);
bool ecm_mont_path_n_fits(desc, n_bit_size);
bool ecm_mont_path_container_fits(desc, limbs, n_bit_size);
bool ecm_addsub_path_fits(desc, ctx);
```

carry 固定为 `ECM_STAGE1_MONT_CARRY_BITS = 6`，由 `ecm_path_n_bit_fits` 内部加上。

| 路径 | min_n | max_n | strict | container limbs |
|------|-------|-------|--------|-----------------|
| unroll_only_384 | 0 | 384 | yes | 16 |
| unroll_only_512 | 0 | 512 | no | 0 |
| 4096 专用 | 3072 | 4096 | no | 0 |
| priv_opt / i24 | 0 | 0 | — | 0 |

### Auto 优先级（专用 → 兼容）

| auto_priority | 路径 | dedicated | 适用 N（约） |
|---------------|------|-----------|--------------|
| 10 | unroll_only_384 | yes | &lt; 378 |
| 20 | unroll_only_512 | yes | 378 … 506 |
| 21–25 | 4096 专用路径 | yes | 3072 … 4090（limbs=128） |
| 30 | priv_opt | **no** | 通用兼容兜底 |

**两层解析（mul 与 sqr 各自调用）：**

1. **`opencl_ecm_resolve_stage1_mont_mul` / `_sqr`** — `kind==STAGE1`。4096 专用 N 区间时 stage1 模式为 **priv_opt**（`limbs==128` 时实际算子由 `ECM_STAGE1_MUL_PATH` / `ECM_STAGE1_SQR_PATH` 控制）。
2. **`opencl_ecm_resolve_mont4096_mul` / `_sqr`** — `kind==4096`。auto 时在专用路径中按 priority **21→25** 选第一个 `n_fits_4096_dedicated` 为真的项。

显式请求不满足 `n_fits` 时，从下一档 `auto_priority` 继续 fallback。

### 编译宏（描述符驱动）

每个 Montgomery 描述符可携带 `stage1_force_macro`（如 `ECM_STAGE1_MUL_FORCE_UNROLL384`）；选中路径时 `opencl_ecm_stage1_generate_build_options()` 仅注入 **该算子对应的一个** `-D`（默认路径 `nullptr` 则不注入，由 `ecm_stage1.cl` 按 limbs 分发）。

i24 路径用 `stage1_use_i24` / `stage1_i24_blsub`；add/sub 用 `path_id` + `needs_asm_b32` / `needs_asm_b16`。

流程：

1. **resolve 直接返回描述符**（编译宏的唯一来源）：
   - `opencl_ecm_resolve_stage1_mont_mul/sqr` → `const EcmMont*PathDescriptor*`
   - `opencl_ecm_resolve_mont4096_mul/sqr` → 4096 描述符或 `nullptr`（stage1 专用别名）
   - `opencl_ecm_resolve_addmod_path` / `opencl_ecm_resolve_submod_path`
2. `opencl_ecm_stage1_make_build_plan(...)` — 把上述指针填入 plan（无 enum/path_id 二次查表）
3. `opencl_ecm_stage1_generate_build_options(plan)` — 仅对非空 `stage1_force_macro` 等字段注入 `-D...=1`（其余宏默认 0）
4. `ensure_ecm_kernel(plan)` — 编译；缓存 key 为 plan 指针相等性

## Add / Sub 描述符（分离）

```cpp
struct EcmAddModPathDescriptor { ... };
struct EcmSubModPathDescriptor { ... };
```

入口：

- `opencl_ecm_resolve_addmod_path(path, limbs, is_amd)`
- `opencl_ecm_resolve_submod_path(path, limbs, is_amd)`

`opencl_ecm_resolve_addsub_path(..., is_add)` 为兼容包装。

Auto 规则见 `kAddModRegistry` / `kSubModRegistry` 中 `limbs_fits` + `auto_priority`。

## 新增路径清单

### Montgomery mul 或 sqr

1. 在 `kMontMulRegistry[]` 或 `kMontSqrRegistry[]` 增加一行。
2. `ecm_stage1.cl` 中 `mont_mul_stage1` / `mont_sqr_stage1` 分发。
3. `arrays.xml` value 对齐 `aliases`。

### Montgomery 4096

1. mul/sqr 注册表各增一行 + `ecm_stage1_mont4096_paths.cl`。
2. `ECM_STAGE1_MUL_PATH` / `ECM_STAGE1_SQR_PATH` 分支（coop 路径已分离）。

### AddSub

1. `kAddModRegistry[]` 或 `kSubModRegistry[]` 增加一行。
2. `ecm_stage1.cl` 中 `ECM_STAGE1_ADDMOD_PATH` / `ECM_STAGE1_SUBMOD_PATH` 分支。

## 桌面验证

```powershell
echo "(2^151-1)" | ecm -v -d 0 -gpu -sigma 3:2026  -gpucurves 16 1e4 0   # unroll384
echo "(2^347-1)" | ecm -v -d 0 -gpu -sigma 3:561219477 -gpucurves 32 1e5 0
echo "(2^421-1)" | ecm -v -d 0 -gpu -sigma 3:20260611  -gpucurves 256 1e5 0  # unroll512
echo "(2^641-1)" | ecm -v -d 0 -gpu -gpucurves 32 1e4 0                       # priv_opt
```

混合路径示例：`--mul unroll_only_384 --sqr priv_opt`（需 N 同时满足两者或触发 fallback）。

## 后续

- [ ] 从注册表生成 `arrays.xml` / CLI help
- [ ] addsub ASM 片段链接字段并入描述符
