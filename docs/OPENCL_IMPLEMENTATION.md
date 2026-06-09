# OpenCL ECM Stage 1 实现总结

## 文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `cgbn/backends/opencl/kernels/ecm_stage1.cl` | GPU内核（Montgomery Ladder） | ✅ 创建 |
| `src/cgbn_stage1_opencl.cpp` | 主机端实现（OpenCL管理+初始化+因子提取） | ✅ 创建 |
| `ECM_GPU_FLOW.md` | 数学流程纲要 | ✅ 创建 |

## 核心差异: CUDA vs OpenCL

### CUDA 版本 (test/cgbn_stage1.cu)
```
优势：
- 使用 CGBN 库的完整模板系统
- 类型安全的大整数操作
- 自动内存管理
- 编译时内核选择（512~32768 bits）

结构：
kernel_double_add<cgbn_params_t>
  ├── curve_t 类
  ├── 内联 Montgomery Ladder
  └── 类型化的点操作
```

### OpenCL 版本 (新)
```
优势：
- 跨平台支持（Intel、AMD、NVIDIA）
- 与 CGBN 已有的 OpenCL 算子复用
- 灵活的编译时内核选择

结构：
ecm_stage1.cl (内核)
  ├── kernel_ecm_stage1()
  ├── point_double()
  ├── point_add()
  └── Montgomery Ladder 循环

cgbn_stage1_opencl.cpp (主机)
  ├── OpenCL 上下文初始化
  ├── set_curve_data() - Suyama 参数化
  ├── process_results() - GCD 因子提取
  └── GPU 缓冲区管理
```

## OpenCL 实现架构

### 1. 主机端初始化 (cgbn_stage1_opencl.cpp)

```cpp
opencl_init_context()
  └─→ clGetPlatformIDs()
      clGetDeviceIDs()
      clCreateContext()
      clCreateCommandQueue()

set_curve_data(N, curves, sigma)
  └─→ for each curve i:
        d = (sigma + i) mod N
        Compute (x_init, z_init) via Suyama
        Export to uint32_t array
```

### 2. GPU 内核 (ecm_stage1.cl)

```cpp
kernel_ecm_stage1()
  ├─ Per-work-item: 1 curve instance
  ├─ Montgomery Ladder:
  │   for bit in s_bits[s_start:s_end]:
  │     cond_swap()
  │     point_double()
  │     point_add()
  └─ Store (x_final, z_final)
```

### 3. 因子提取 (主机端)

```cpp
process_results()
  └─→ for each curve:
        z_final = GPU结果
        if mpz_invert(inv, z_final, N):
          factor = inv * x_final mod N
        else:
          factor = gcd(z_final, N)
```

## OpenCL 算子复用

已有的 CGBN OpenCL 算子：

| 函数 | 位置 | 用途 |
|------|------|------|
| `cgbn_mont_mul` | mont.cl | 蒙哥马利乘法 |
| `cgbn_mont_sqr` | mont.cl | 蒙哥马利平方 |
| `cgbn_add` | addsub.cl | 模加法 |
| (自实现) | ecm_stage1.cl | `mont_add`, `mont_sub`, `point_*` |

### Montgomery 操作集成

```cpp
// 从已有的 mont.cl 调用或内联
void cgbn_mont_mul(__global const uint *a,
                   __global const uint *b,
                   __global const uint *n,
                   __global uint *out,
                   uint np0,
                   uint limbs,
                   uint instance_idx) {
    // CIOS 方法 (Coarsely Integrated Operand Scanning)
    // 时间复杂度: O(limbs²)
}

void cgbn_mont_sqr(__global const uint *a,
                   __global const uint *n,
                   __global uint *out,
                   uint np0,
                   uint limbs,
                   uint instance_idx) {
    // 优化的平方（相比乘法快 ~30%）
}
```

## 执行流程

```
Host (ecm_driver.cpp)
  └─→ parse_expression('(2^991-1)/(8218291649)')
      N = 959 bits
      B1 = 11e6
      └─→ compute_batch_s(B1)
          s = 15869673 bits
          └─→ opencl_ecm_stage1(N, s, curves=2, ...)
              
              ├─ opencl_init_context()
              │  └─ Platform → Device → Context → Queue
              │
              ├─ set_curve_data()
              │  └─ For each curve: Suyama params
              │
              ├─ clCreateBuffer() × 4
              │  ├─ gpu_s_bits (15869673/32 limbs)
              │  ├─ gpu_curve_data (5*2*64 limbs)
              │  ├─ gpu_N (64 limbs)
              │  └─ gpu_results (2*2*64 limbs)
              │
              ├─ clCompileProgram("ecm_stage1.cl")
              │  └─ kernel_ecm_stage1 (with dynamic BITS selection)
              │
              ├─ clEnqueueNDRangeKernel()
              │  ├─ global_size = curves (2)
              │  ├─ local_size = 256 (TPB_DEFAULT)
              │  └─ kernel execution (Montgomery Ladder)
              │
              ├─ clEnqueueReadBuffer(gpu_results)
              │  └─ Fetch (x_final, z_final) per curve
              │
              └─ process_results()
                 └─ For each curve:
                    ├─ if mpz_invert(z_final, N):
                    │    factor = inv * x_final mod N
                    └─ else: factor = gcd(z_final, N)
                    
Output: factors[] with 0 (no factor) or ECM_FACTOR_FOUND_STEP1
```

## 关键参数对应

| 概念 | CUDA 参数 | OpenCL 参数 | 说明 |
|------|----------|-----------|------|
| 内核大小 | `cgbn_params_t<TPI,BITS>` | 动态 BITS 选择 | 1024~32768 bits |
| 每工作项线程 | `TPI` (4,8,16,32) | 工作组大小 | 控制寄存器压力 |
| 实例数 | `curves` | `get_global_id(0)` | 并行处理多条曲线 |
| Montgomery 参数 | `uint32_t np0` | 内核参数 | N^{-1} mod 2^32 |
| s 编码 | `uint32_t *s_bits` | GPU 缓冲区 | 批积分解式的比特表示 |

## 性能注意事项

### 内存访问优化

```cpp
// CUDA 版本: 全局内存 → 寄存器（CGBN 优化）
// OpenCL 版本: 需要手动优化

// 1. 局部缓存 Montgomery 参数
uint N[MAX_LIMBS];  // 从全局缓存一次
for (...) cgbn_mont_mul(..., N, ...);  // 重复使用

// 2. 向量化加载 (uint4)
// 已在 addsub.cl 中实现的技术

// 3. 工作组共享内存（可选，用于并行化）
__local uint local_sum[256];
```

### 计算强度

- **Montgomery 乘法**: 1 个内存读取 → 1 个乘法 (高计算强度)
- **整数点操作**: ~20 次乘法 + 10 次平方 (high throughput)
- **总吞吐**: 理论上接近 CUDA 版本 (取决于 OpenCL 编译器优化)

## 编译集成

### CMakeLists.txt 修改（待）

```cmake
# 添加 OpenCL 编译
find_package(OpenCL REQUIRED)

add_library(cgbn_opencl_ecm
  src/cgbn_stage1_opencl.cpp
  cgbn/backends/opencl/kernels/ecm_stage1.cl  # 会被内联为 C 字符串
)

target_link_libraries(cgbn_opencl_ecm
  PUBLIC cgbn_opencl OpenCL::OpenCL gmp
)

# 或直接在 ecm 目标中链接
target_link_libraries(ecm PRIVATE cgbn_opencl_ecm)
```

### 内核编译方式

**选项 1: 运行时编译（推荐）**
```cpp
// 在 opencl_init_context() 中
const char *kernel_src = ... // 从文件读取 ecm_stage1.cl
cl_program program = clCreateProgramWithSource(..., kernel_src, ...);
clBuildProgram(program, 1, &device, "-cl-opt-disable", ...);
```

**选项 2: 离线编译**
```bash
# 预编译内核
clang -cc1 -emit-llvm ecm_stage1.cl -o ecm_stage1.bc
llvm-spirv ecm_stage1.bc -o ecm_stage1.spv

# 在运行时加载
cl_program = clCreateProgramWithIL(context, spirv_data, ...);
```

## 下一步改进

1. **完整的 double-add 实现**
   - 当前 ecm_stage1.cl 中的 point_double/point_add 是占位符
   - 需要完整的 Montgomery 梯形公式

2. **动态内核选择**
   - 基于 N 的比特大小选择最优的 BITS/TPI 组合
   - 当前硬编码为 BITS=2048

3. **检查点支持**
   - 类似 CUDA 版本的断点恢复
   - 保存/恢复 GPU 状态

4. **多设备支持**
   - 支持多个 GPU 并行处理不同的曲线集
   - 工作负载分配优化

5. **性能基准测试**
   - 与 CUDA 版本对比
   - 不同 N 大小和 B1 值的性能曲线
