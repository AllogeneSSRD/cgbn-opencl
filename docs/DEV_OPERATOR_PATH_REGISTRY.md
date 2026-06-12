# 算子路径注册表（宏别名注入架构）

Stage-1 **Montgomery mul/sqr** 与 **add/sub-mod** 路径集中在注册表中定义与选择，Host 端通过宏别名注入将选定算子绑定到统一接口。桌面与 Android 共用 `src/opencl_ecm_path_registry.cpp`。

内核命名规范见 [`kernels/opencl/README.md`](../kernels/opencl/README.md)。

## 架构概览

```
Host (C++)                              Kernel (OpenCL)
──────────────────────────────────────  ──────────────────────────
汇编源码:
#define ECM_STAGE1_MUL_IMPL mont_mul_unroll_384b
#define ECM_STAGE1_SQR_IMPL mont_sqr_unroll_384b
#define ECM_STAGE1_ADD_IMPL add_mod_unroll_384b
#define ECM_STAGE1_SUB_IMPL sub_mod_unroll_384b
                                        ↓
                                       operator_iface.h.cl:
                                       #define mont_mul ECM_STAGE1_MUL_IMPL
                                       #define mont_sqr ECM_STAGE1_SQR_IMPL
                                       #define add_mod ECM_STAGE1_ADD_IMPL
                                       #define sub_mod ECM_STAGE1_SUB_IMPL
                                        ↓
                                       ecm_stage1.cl:
                                       void double_add_v2(...) {
                                           mont_mul(q, AA, BB, N, np0, limbs);
                                           add_mod(w, DA, CB, N, limbs);
                                           ...
                                       }
```

主内核 `ecm_stage1.cl` 只调用宏别名 `mont_mul`/`mont_sqr`/`add_mod`/`sub_mod`，无任何 `#ifdef` 算子分支，无 4096 硬编码。

## 目录与命名

| 算子族 | 目录 | 文件命名 | `cl_name` 示例 |
|--------|------|----------|---------------|
| Montgomery mul + sqr | `mont_mul/` | `mont_mul_{variant}_{bits}b.cl` | `mont_mul_unroll_384b` / `mont_sqr_unroll_384b` |
| Add mod | `add_mod/` | `add_mod_{variant}_{bits}b.cl` | `add_mod_unroll_384b` |
| Sub mod | `sub_mod/` | `sub_mod_{variant}_{bits}b.cl` | `sub_mod_unroll_384b` |
| 公共配置 | `common/` | `stage1_config.h.cl`, `mp_priv.h.cl`, `ladder_helpers.cl`, `operator_iface.h.cl` | — |
| 主入口 | — | `ecm_stage1.cl` | — |
| Coop 补充 | — | `ecm_stage1_coop.cl` | 仅 `ECM_STAGE1_COOP_WG > 1` 时加载 |

**命名规则：** `cl_name` == OpenCL 函数名 == 文件名主干（位数后缀带 `b`，如 `384b`）。同一个 mul 文件同时导出 `mont_mul_*` 和 `mont_sqr_*`（sqr 内部调用 mul）。

## 汇编加载顺序

`opencl_ecm_stage1_assemble_kernel_source(plan, load_file)` 按以下顺序拼接源码：

1. **注入宏**：`#define ECM_STAGE1_MUL_IMPL <plan.mul->cl_name>` 等 4 个宏
2. `common/stage1_config.h.cl` — 编译期配置（`MAX_LIMBS`, `TPI`, `COOP_WG` 等）
3. `common/mp_priv.h.cl` — 基础 limb 原语（`mp_ge`, `mp_sub_n`, `mp_copy` 等）
4. `common/ladder_helpers.cl` — 与算子无关的 ladder 辅助函数
5. 已选算子文件（mul, sqr, add, sub — 按 `kernel_path` 去重）
6. `common/operator_iface.h.cl` — 别名 `mont_mul`/`mont_sqr`/`add_mod`/`sub_mod`
7. `ecm_stage1_coop.cl` — 仅当 4096 位且 `coop_work_group_size > 1` 时加载
8. `ecm_stage1.cl` — 主 ladder 入口

Host 通过 `load_ecm_stage1_kernel_file()` 加载文件；根目录 `kernels/opencl/`，环境变量 `ECM_KERNEL_ROOT` 可覆盖。

## 描述符字段

```cpp
struct EcmMontPathDescriptor {
    const char *id;           // CLI 参数 / 别名匹配 (如 "unroll_only_384")
    const char *cl_name;      // OpenCL 函数名 (如 "mont_mul_unroll_384b")
    const char *const *aliases;
    const char *kernel_path;  // 相对 kernels/opencl/ (如 "mont_mul/mont_mul_unroll_384b.cl")
    int8_t auto_priority;     // 自动选择优先级 (值越小越优先，-1 表示仅手动)
    uint16_t min_n_bits;      // 最小 N 位宽
    uint16_t max_n_bits;      // 最大 N 位宽
    bool max_n_strict;        // max_n_bits 是否严格上限
    uint16_t max_container_bits;
    uint32_t os_mask;         // OS 过滤 (ECM_OS_*)
    uint32_t gpu_vendor_mask; // GPU 厂商过滤 (ECM_GPU_*)
    uint32_t gpu_vendor_exclude_mask;
    bool dedicated;           // 是否为固定宽度算子
    uint8_t coop_work_group_size; // 合作工作组大小 (4096 位专用)
    uint16_t local_scratch_u32;   // 本地内存占用量
    const char *force_macro;  // 已废弃 (宏别名注入后不再需要)
};

// add/sub 描述符结构类似，无 dedicated/coop/scratch 字段
struct EcmAddSubPathDescriptor { ... };
```

## 4096 位 Coop 路径

4096 位协作工作组（`ECM_STAGE1_COOP_WG > 1`）使用 `ecm_stage1_coop.cl` 作为补充：

- 定义 `kernel_double_add`（`reqd_work_group_size` 版本），使用 `__local` 内存 + barrier
- 内部通过 `ECM_STAGE1_MUL_PATH`/`ECM_STAGE1_SQR_PATH` 整数枚举分发到具体的 4096 协作算子
- 主入口 `ecm_stage1.cl` 的 `kernel_double_add` 被 `#if !ECM_STAGE1_USE_COOP_WG` 包围，二者互斥
- 构建选项中仍保留 `ECM_STAGE1_MUL_PATH` 等枚举（仅 coop 内部使用）

## 生成器

| 工具 | 输出目标 |
|------|---------|
| `tools/gen_mp_addsub_bits_stage1.py` | `add_mod/add_mod_unroll_{bits}b.cl`, `sub_mod/sub_mod_unroll_{bits}b.cl` 及 asm 变体 |
| `tools/gen_mp_addsub_asm_block16_stage1.py` | `add_mod/add_mod_asm_512b.cl`, `sub_mod/sub_mod_asm_512b.cl` |
| `tools/gen_mp_addsub_asm_block32_stage1.py` | `add_mod/add_mod_asm_4096b.cl`, `sub_mod/sub_mod_asm_4096b.cl` |

生成器输出函数命名遵循 `{op}_{variant}_{bits}b` 格式，包含 `*_body` 内部实现和带 `limbs` 校验的外层包装。

## 内核版本

`ECM_STAGE1_KERNEL_REV=13` —— 宏别名注入架构，算子文件独立，dispatch.cl 已移除。
