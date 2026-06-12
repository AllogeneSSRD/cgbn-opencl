# 算子路径注册表（数据驱动）

Stage-1 **Montgomery mul/sqr** 与 **add/sub-mod** 路径集中在注册表中定义、选择与编译宏推导。桌面与 Android 共用 `src/opencl_ecm_path_registry.cpp`。

**mul/sqr 与 add/sub 各自独立解析**，可自由组合（例如 `--mul unroll_only_384 --add asm_384b`）。

## 文件

| 文件 | 内容 |
|------|------|
| `include/opencl_ecm_path_registry.h` | 描述符、`EcmStage1KernelBuildPlan`、`generate_build_options` |
| `src/opencl_ecm_path_registry.cpp` | 各注册表 |
| `tools/gen_mp_addsub_bits_stage1.py` | 生成 `addsub_bits_stage1.cl`（128b–384b unroll + asm） |

## Montgomery 描述符

见前文；固定算子 `dedicated=true`，算子宽度 = `max_n_bits/32`。

## Add/Sub 描述符（按 Bits 命名）

```cpp
struct EcmAddSubPathDescriptor {
    int cl_dispatch_id;
    const char *id;              // 如 asm_384b, unroll_512b
    // ...
    uint16_t max_n_bits;         // 0=不限；128/192/256/378 等
    bool max_n_strict;
    uint16_t max_container_bits; // 最小容器位宽：512 或 4096
    uint32_t os_mask;
    uint32_t gpu_vendor_mask;
    uint32_t gpu_vendor_exclude_mask;
    uint32_t kernel_includes;
};
```

### 位宽专用路径（512-bit 容器内）

| ID / CLI | 算子位宽 | Limbs | 适用 N (约) | AMD asm | 全平台 unroll |
|----------|----------|-------|-------------|---------|---------------|
| `asm_128b` / `unroll_128b` | 128 | 4 | N ≤ 128 | ✓ | ✓ |
| `asm_192b` / `unroll_192b` | 192 | 6 | N ≤ 192 | ✓ | ✓ |
| `asm_256b` / `unroll_256b` | 256 | 8 | N ≤ 256 | ✓ | ✓ |
| `asm_384b` / `unroll_384b` | 384 | 12 | N ≤ 378 | ✓ | ✓ |

OpenCL 函数：`mp_add_mod_asm_384b` / `mp_add_mod_unroll_384b`（sub 同理）。

### 容器级路径（重命名）

| 新名 | 旧名 | 容器 | 说明 |
|------|------|------|------|
| `asm_512b` | `asm_b16` | 512-bit (16 limbs) | 全 16-limb AMD asm |
| `unroll_512b` | `fused_unroll_b16` | 512-bit | 全 16-limb 静态展开 |
| `asm_4096b` | `asm_b32` | 4096-bit (128 limbs) | 4×32-limb AMD asm |
| `unroll_4096b` | `fused_unroll_b32` | 4096-bit | 4×32-limb 静态展开 |

旧 CLI 名仍作 **alias** 保留。

### `ECM_ADDSUB_PATH_*` 枚举（与 `ecm_stage1.cl` 同步）

```
0 fused, 1 fused_unroll, 2 unroll_4096b, 3 asm_4096b,
4 unroll_512b, 5 asm_512b,
6 unroll_128b, 7 asm_128b, … 12 unroll_384b, 13 asm_384b
```

### Kernel includes

| 掩码 | 文件 |
|------|------|
| `ECM_KERNEL_INC_ADDSUB_BITS` | `mp_addsub/stage1/addsub_bits_stage1.cl` |
| `ECM_KERNEL_INC_MP_ASM_U16` | `asm_block16_stage1.cl` (512b asm) |
| `ECM_KERNEL_INC_MP_ASM_U32` | `asm_block32_stage1.cl` (4096b asm) |

## 解析

```cpp
EcmPathContext ctx{n_bit_size, limbs, os_mask, gpu_vendor_mask};
opencl_ecm_resolve_addmod_path(path, ctx);
```

Auto：按 `auto_priority` 升序，结合 `max_n_bits`、容器、GPU 掩码选首项。

## 编译计划

`ECM_STAGE1_KERNEL_REV=8`；`ECM_STAGE1_ADDMOD_PATH` / `SUBMOD_PATH` 注入 dispatch id。
