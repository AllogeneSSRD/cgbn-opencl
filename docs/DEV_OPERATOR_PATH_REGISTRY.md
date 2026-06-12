# 算子路径注册表（数据驱动）

Stage-1 **Montgomery mul/sqr** 与 **add/sub-mod** 路径集中在注册表中定义、选择与编译宏推导。桌面与 Android 共用 `src/opencl_ecm_path_registry.cpp`。

**mul/sqr 与 add/sub 各自独立解析**，可自由组合（例如 `--mul unroll_only_384 --sqr unroll64_4096`）。

## 文件

| 文件 | 内容 |
|------|------|
| `include/opencl_ecm_path_registry.h` | 描述符、`EcmStage1KernelBuildPlan`、`generate_build_options` |
| `src/opencl_ecm_path_registry.cpp` | `kMontMulRegistry` / `kMontSqrRegistry` / `kAddModRegistry` / `kSubModRegistry` |

## Montgomery 描述符

固定算子（`dedicated=true`）：算子宽度 = `max_n_bits / 32`（如 unroll384 → **12 limbs**）。  
兼容算子（`dedicated=false`）：容器须容纳 `N + CARRY`（由 `select_bits()` 选出的 CGBN 容器，可与算子宽度不同；见 `DEV_ECM_CGBN_CONTAINER_VS_MONT.md`）。

```cpp
struct EcmMontPathDescriptor {
    const char *id;
    const char *cl_name;
    const char *const *aliases;
    int8_t auto_priority;           // -1 = 仅显式指定

    uint16_t min_n_bits, max_n_bits;
    bool max_n_strict;
    bool dedicated;                 // true: 固定算子; false: 兼容容器

    uint8_t coop_work_group_size;
    uint16_t local_scratch_u32;
    uint8_t cl_dispatch_id;         // 4096 dedicated + 128-limb 容器时注入 ECM_STAGE1_*_PATH
    uint32_t kernel_includes;       // EcmKernelInclude 掩码
    const char *force_macro;
};
```

**匹配规则**（`ecm_mont_path_fits`）：

| 类型 | N 位宽 | 容器 |
|------|--------|------|
| dedicated | `ecm_path_n_bit_fits(min,max,strict,N)` | `container_bits >= max_n_bits` |
| compatible | 同上（max=0 不限） | `(N+CARRY) <= container_bits` |

i24 路径通过 `id` 前缀 `i24` 识别（无单独 `limb_bits` 字段）。

## Add/Sub 描述符

```cpp
struct EcmAddSubPathDescriptor {
  // ...
  uint16_t max_container_bits;      // 0=任意, 512, 4096
  uint32_t os_mask;                 // EcmPathOs 位掩码
  uint32_t gpu_vendor_mask;         // 须命中（0 / ANY = 不限）
  uint32_t gpu_vendor_exclude_mask; // 须未命中（如排除 AMD）
  uint32_t kernel_includes;
};
```

**平台 / GPU 掩码**（`EcmPathContext` 携带运行时 `os_mask` + `gpu_vendor_mask`）：

| 掩码 | 含义 |
|------|------|
| `ECM_OS_WINDOWS` / `ANDROID` / `LINUX` / `MACOS` | 宿主 OS |
| `ECM_GPU_AMD` / `NVIDIA` / `INTEL` / `QUALCOMM` / `HUAWEI` / `APPLE` | OpenCL 设备厂商 |

**Kernel 附加源**（`EcmKernelInclude`）：

| 位 | 文件 |
|----|------|
| `MONT_EXTENDED` | `ecm_stage1_mont4096_paths.cl` |
| `MP_ASM_U32` | `asm_block32_stage1.cl` |
| `MP_ASM_U16` | `asm_block16_stage1.cl` |

## 解析

```cpp
opencl_ecm_resolve_mont_mul(path, n_bit_size, container_limbs, &unknown);
opencl_ecm_resolve_mont_sqr(path, n_bit_size, container_limbs, &unknown);
EcmPathContext ctx{n, limbs, ecm_path_host_os_mask(), gpu_vendor_mask};
opencl_ecm_resolve_addmod_path(path, ctx);
```

- `container_limbs==0`：仅按 N 位宽匹配（i24 探测）
- auto：按 `auto_priority` 升序取第一个 `ecm_mont_path_fits` / `ecm_addsub_path_fits` 的项

## Auto 优先级（约）

| priority | 路径 | dedicated | 算子 limbs | N（约） |
|----------|------|-----------|------------|---------|
| 10 | unroll_only_384 | yes | 12 | N+6 &lt; 384 |
| 20 | unroll_only_512 | yes | 16 | 378…506 |
| 21–25 | unroll64/fips 系列 | yes | 128 | 3072…4090 |
| 30 | priv_opt | no | 随容器 | 兜底 |

容器由 `select_bits(N)` 决定（如 M151 → 512-bit / 16 limbs）；unroll384 不要求 `container_limbs==16`。

## 编译计划

1. resolve → 描述符指针
2. `opencl_ecm_stage1_make_build_plan(...)`
3. `opencl_ecm_stage1_collect_kernel_includes(plan)` → prepend 哪些 `.cl`
4. `opencl_ecm_stage1_generate_build_options(plan)` → `-D` 宏（`ECM_STAGE1_KERNEL_REV=7`）
