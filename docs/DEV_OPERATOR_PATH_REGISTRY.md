# 算子路径注册表（数据驱动）

Stage-1 **Montgomery mul/sqr** 与 **add/sub-mod** 路径集中在注册表中定义、选择与编译宏推导。桌面与 Android 共用 `src/opencl_ecm_path_registry.cpp`。

命名规范见 [`kernels/opencl/README.md`](../kernels/opencl/README.md)。

## 目录与命名（统一）

| 算子族 | 目录 | 文件命名 | 示例 `kernel_path` |
|--------|------|----------|-------------------|
| Montgomery mul | `mont_mul/` | `unroll_{bits}.cl`, `priv_opt.cl` | `mont_mul/unroll_384.cl` |
| Montgomery 4096 | `mont_mul/4096/` | `unroll_64.cl`, `fips_4096.cl` | `mont_mul/4096/fips_4096.cl` |
| Montgomery sqr | `mont_sqr/` | `dispatch.cl`（实现来自 mul 算子文件） | — |
| Add mod | `add_mod/` | 与 CLI `id` 一致 | `add_mod/bits/unroll_384b.cl` |
| Sub mod | `sub_mod/` | 与 CLI `id` 一致 | `sub_mod/bits/unroll_384b.cl` |

每个算子 `.cl` 定义 `ecm_stage1_mont_mul` / `ecm_stage1_mont_sqr` / `ecm_stage1_add_mod` / `ecm_stage1_sub_mod`；`*/dispatch.cl` 暴露 ladder 使用的 `mont_mul_stage1`、`mp_add_mod` 等符号。

## 加载顺序

`opencl_ecm_stage1_kernel_source_paths(plan)`：

1. `common/stage1_config.h.cl`, `common/mp_priv.h.cl`
2. 已选 mul / sqr / add / sub 算子文件（去重）
3. `mont_mul/dispatch.cl`, `mont_sqr/dispatch.cl`, `add_mod/dispatch.cl`, `sub_mod/dispatch.cl`
4. `ecm_stage1.cl`（ladder + coop，无算子 `#if` 分发）

Host 通过 `load_ecm_stage1_kernel_file()` 加载；根目录 `kernels/opencl/`，环境变量 `ECM_KERNEL_ROOT` 可覆盖。

## 描述符字段

```cpp
const char *id;           // CLI / 日志
const char *cl_name;      // 运行时打印名
const char *const *aliases;
const char *kernel_path;  // 相对 kernels/opencl/
```

4096 coop 路径仍使用 `ECM_STAGE1_MUL_PATH` / `SQR_PATH`（仅 `ecm_stage1.cl` coop 胶水）。

## 生成器

| 工具 | 输出 |
|------|------|
| `tools/gen_mp_addsub_bits_stage1.py` | `add_mod/bits/*.cl`, `sub_mod/bits/*.cl` |
| `tools/gen_mp_addsub_asm_block16_stage1.py` | `add_mod/asm_512b.cl`, `sub_mod/asm_512b.cl` |
| `tools/gen_mp_addsub_asm_block32_stage1.py` | `add_mod/asm_4096b.cl`, `sub_mod/asm_4096b.cl` |

## 编译

`ECM_STAGE1_KERNEL_REV=12`。仅拼接注册表选中的算子文件，不再依赖 `ECM_STAGE1_ADDMOD_PATH` 数值枚举做 add/sub 分发。
