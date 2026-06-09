# ECM GPU Stage 1 数学流程纲要

基于 `test/cgbn_stage1.cu` CUDA 实现的完整分析

## 1. 输入参数

```
N         : 要分解的大整数 (target to factor)
s         : Batch product = ∏_{p≤B1} p^{⌊log_p(B1)⌋}
curves    : 椭圆曲线个数 (通常 2-1000)
sigma[]   : 每条曲线的随机参数 (sigma[i] = σ + i)
B1        : Stage 1 界限 (从 s 的比特数编码)
checkpoint_interval_ms : 检查点保存间隔
```

## 2. 预计算步骤 (Host)

### 2.1 初始化
- 计算 `np0 = -N^{-1} mod 2^{32}` (Montgomery 参数)
- 配置 CGBN 内核参数：选择合适的 BITS 和 TPI
  - TPI=4:  BITS=512
  - TPI=8:  BITS=1024, 1280, 1536, 1792, 2048
  - TPI=16: BITS=2560~8192
  - TPI=32: BITS=9216~32768
- 验证 N 的比特数 ≤ BITS - CARRY_BITS (CARRY_BITS=6)

### 2.2 s 比特编码
- 将 `s` (GMP mpz_t) 转换为 uint32_t 数组（小端）
- 提取所有比特：从 MSB 逐位扫描
- 总比特数 `s_num_bits` = mpz_sizeinbase(s, 2)

### 2.3 曲线初始化 (set_p_2p 函数)
对每条曲线 i = 0..curves-1：
```
sigma_i = sigma_ptr + i

计算 x 坐标 (使用Suyama参数化):
  d = (sigma_i / 2^32) mod N        [特殊乘法处理]
  
  B = (d^2 - 5) mod N              [基础算术]
  C = (B^2 - 2*B) mod N
  A = (B^3 - 3*B) mod N / (4*B*C)  [模逆]
  
  x_init = (C^2 - 1) mod N / 4
  z_init = 1
```

**存储格式** (每条曲线 5 个 limbs):
- data[5*i+0] : x 坐标
- data[5*i+1] : z 坐标  
- data[5*i+2] : 曲线参数 A
- data[5*i+3] : 曲线参数 B
- data[5*i+4] : 曲线参数 C

## 3. GPU 核心计算 (kernel_double_add)

### 3.1 线程映射
```
instance_i = (blockIdx.x * blockDim.x + threadIdx.x) / TPI
if instance_i >= curves: return  [不处理超额线程]
```

### 3.2 初始化 (Setup)
从全局内存加载：
```
modulus N (输入数字)
aX, aZ    (当前点的射影坐标)
bX, bZ    (辅助点)
曲线参数 A, B, C
```

**初始状态**:
- aX, aZ = x_init, z_init  (初始点)
- bX, bZ = 1, 0            (无穷远点)

### 3.3 S 位序列处理 (Double-and-Add Montgomery Ladder)

**算法**: 从 s 的第一个比特到最后一个比特
```
for bit_index from s_start to s_start + s_interval:
  bit_value = getBit(s_bits, bit_index)
  
  if bit_value != swapped:
    swapped = !swapped
    swap(aX, bX)
    swap(aZ, bZ)
  
  double_add_v2(aX, aZ, bX, bZ)  // 同时进行 point doubling 和 point addition
```

**Montgomery Ladder 的关键性质**：
- 不依赖于单个比特是 0 还是 1
- 恒定时间操作（抵抗侧信道攻击）
- 维护两个点的差为初始 x 坐标

### 3.4 Double-Add-v2 操作

对于射影坐标 (X:Z)，计算：
```
设输入: aX, aZ (点 A), bX, bZ (点 B)

Point Doubling: 2A
  U1 = (aX - aZ)^2
  U2 = (aX + aZ)^2
  U3 = U2 - U1
  ... 复杂的蒙哥马利梯形公式 ...
  aX_new = U2 * V3
  aZ_new = (U1 + (A+2)*U3/4) * V2

Point Addition: A + B (其中 A - B 已知)
  ... 使用加法差分的优化公式 ...
  bX_new = Z1 * ((aX - bX)^2)
  bZ_new = X1 * ((bX_new - bZ)^2) 等
```

**包含的模运算**:
- 加法/减法 (mod N)
- 乘法 (mod N)
- 平方 (特化的乘法)
- 条件取反 (处理负数)

### 3.5 最终点提取

处理 swapped 状态：
```
if swapped == 1:
  swap(aX, bX)
  swap(aZ, bZ)

// 现在 aX:aZ 包含乘积 s*P 的结果
```

## 4. 因子提取阶段 (Host - process_results)

对每条曲线 i = 0..curves-1：

### 4.1 从 GPU 数据恢复
```
从 data[5*i+0], data[5*i+1] 恢复 x_final, z_final
转换回 GMP mpz_t (使用 mpz_import)
```

### 4.2 模逆计算 & GCD
```
if mpz_invert(factor, z_final, N):
  // z_final 与 N 互质
  factor = factor * x_final mod N  // 恢复射影坐标
  // factor 是 P*s 在曲线上的 x 坐标
  // 如果触发异常，可能包含 N 的因子
else:
  // z_final 与 N 不互质 - 找到因子！
  mpz_gcd(factor, z_final, N)
  return ECM_FACTOR_FOUND_STEP1
```

### 4.3 结果处理
- 如果 factor ≠ N: 找到非平凡因子 ✓
- 如果 factor = N: 无因子（曲线无贡献）
- 如果 factor = 1: 无因子

## 5. 检查点系统 (可选)

### 5.1 保存
```
checkpoint_header_t:
  - magic: 0x45555047 ("GPUE")
  - version: 3
  - s_partial: 当前比特位置
  - s_num_bits: 总比特数
  - batches_complete: 已完成批次
  - BITS, TPI, curves, sigma 等配置
  
定期保存（每 checkpoint_interval_ms 毫秒）
文件名: .ecm_ckpt_<nbits>_<first8>_<last8>.dat
```

### 5.2 恢复
- 校验 magic 和 version
- 恢复 GPU 数据和进度指针
- 继续下一轮 s 位处理

## 6. 关键算术操作总结

| 操作 | 出现位置 | 复杂度 | 次数/比特 |
|------|---------|--------|----------|
| 加法 mod N | double-add | O(n) | ~4 |
| 乘法 mod N | double-add, 初始化 | O(n²) | ~8 |
| 平方 mod N | double-add | O(n²) | ~6 |
| 模逆 | 因子提取 | O(n³) | 1/all |
| GCD | 因子提取 | O(n³) | 1/all |

## 7. 数据流图

```
Host:
  N, s, sigma, curves, B1
       ↓
  [内核选择 & 初始化]
  [set_p_2p: 曲线参数化]
  [allocate GPU 内存]
       ↓
GPU:
  [kernel_double_add]
  for each bit in s:
    Montgomery Ladder step
  ↓
  (x_final, z_final) per curve
       ↓
Host:
  [process_results]
  for each curve:
    if found_factor(x_final, z_final, N):
      OUTPUT: factor
  
  [可选] save checkpoint
```

## 8. OpenCL 实现映射

需要移植的函数/操作：

| CUDA 概念 | OpenCL 等价物 |
|----------|---------------|
| `__global__ kernel_double_add<>` | `__kernel void kernel_double_add()` |
| CGBN context 初始化 | OpenCL 内存对象初始化 |
| `cgbn_mul`, `cgbn_add` 等 | 自定义 OpenCL 核函数或内联 |
| `__syncthreads()` | `barrier(CLK_GLOBAL_MEM_FENCE)` |
| thread indexing | `get_global_id()`, `get_local_id()` |
| 全局内存读写 | OpenCL buffer 读写 |
| 错误报告 | 结果数组中的状态字段 |
