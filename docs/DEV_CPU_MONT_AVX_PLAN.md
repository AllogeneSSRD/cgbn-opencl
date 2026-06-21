# CPU AVX/AVX512 Montgomery 乘法 Benchmark 计划

## 概述

对标 `opencl_ecm_montsqr` bench 模式，实现不依赖 OpenCL 的纯 CPU Montgomery 乘法 benchmark。
支持 AVX512F 和 AVX2 指令集，批处理模式（SoA 布局），逐位宽（512-bit, 1024-bit）性能测试。

## 最终决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 位宽范围 | 512-bit, 1024-bit 优先 | 快速迭代，验证批处理CIOS在CPU上可行 |
| SIMD策略 | 方案A: 批处理模式 (SoA) | 16/8 条曲线独立并行，匹配CIOS外循环特性 |
| 指令集 | AVX512F 优先 + AVX2 回退 | Zen5 支持 AVX512F（不含IFMA52）；无AVX512时降级AVX2 |
| 回退控制 | CPUID 自动检测 + `--avx2` 强制 | 开发/测试灵活 |
| 集成方式 | 独立模块 (`cpu_mont_bench`) | 不侵入 OpenCL 路径，对标 `opencl_ecm_montsqr` bench 模式 |
| 数据布局 | Structure of Arrays (SoA) | 最佳 SIMD load/store 效率 |
| 算法 | CIOS (Coarsely Integrated Operand Scanning) | 与 OpenCL priv_opt 一致 |
| 验证 | GMP 参考 + 固定测试向量 | benchmark 前自动 selftest |

## 文件清单

| 文件 | 作用 |
|------|------|
| `src/cpu_mont_bench.cpp` | main() 入口: CLI 解析 → selftest → benchmark 循环 |
| `src/cpu_mont_avx.h` | 对外接口声明 |
| `src/cpu_mont_avx.cpp` | AVX512/AVX2 手写 intrinsic 实现 (CIOS, 批处理) |
| `src/cpu_mont_scalar.cpp` | 标量 CIOS 参考实现 + GMP 验证辅助 |
| `src/cpu_mont_scalar.h` | 标量接口 + fill_to_gmp / fill_from_gmp 辅助函数 |
| `test/cpu_mont_test_vectors.cpp` | 固定测试向量 (512-bit, 1024-bit) |
| `test/cpu_mont_test_vectors.h` | 测试向量数据声明 |

## CLI 接口

```
# 位置参数: [bits] [iterations] [ipt] [repeats]

# 延迟测试
cpu_mont_bench 512 1e6 16 1

# 延迟测试 (no-overflow 用例)
cpu_mont_bench 512 1e6 16 1 --no-overflow

# 吞吐量测试
cpu_mont_bench 512 1e6 16 5 -t 12 -a 1,3,5,7,9,11,13,15,17,19,21,23

# 命名参数形式
cpu_mont_bench -b 512 -k 1e6 -i 16 -r 1 --no-verify
```

### 参数

| 参数 | 短形式 | 说明 | 默认值 |
|------|--------|------|--------|
| `[bits]` | `-b` / `--bits` | 位宽 | 512 |
| `[iterations]` | `-k` / `--kernel-iters` | 每线程 Montgomery 乘法次数，支持科学计数法 | 1000 |
| `[ipt]` | `-i` / `--ipt` | 每线程 instance 数（auto=16 AVX512 / 8 AVX2） | auto |
| `[repeats]` | `-r` / `--repeats` | Launch repeats | 1 |
| `--threads <N>` | `-t` | 线程数 | 1 |
| `--affinity MODE` | `-a` | auto / none / 逗号分隔逻辑 CPU | auto |
| `--no-overflow` | — | 使用小输入数据 | false |
| `--avx2` | — | 强制 AVX2（即使在 AVX512 CPU 上） | false |
| `--no-verify` | — | 跳过 self-test | false |

### 测试数据生成

使用 GMP 确定性随机数（与 addsub/montsqr 统一的 seed 方案）：
- 两个用例：`large-inputs` (a,b ∈ [N/2, N)) 和 `small-inputs` (a,b < N/4)
- 确定性 seed: `bits × 31337 + case_index × 0x9e3779b9`
- `--no-overflow` 选择 small-inputs
- LCG 保留用于 self-test 内部验证

## CMakeLists.txt 集成

```cmake
add_executable(cpu_mont_bench
    src/cpu_mont_bench.cpp
    src/cpu_mont_avx.cpp
    src/cpu_mont_scalar.cpp
    test/cpu_mont_test_vectors.cpp
)
target_compile_definitions(cpu_mont_bench PRIVATE BUILD_CPU_MONT_MAIN=1)
target_include_directories(cpu_mont_bench PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(cpu_mont_bench ${GMP_LIBRARY})
# AVX2 在 MSVC 上默认启用，AVX512 单独编译选项:
# set_source_files_properties(src/cpu_mont_avx.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
```

## 最后更新

2026-06-21: 已完成 `cpu_addsub_bench` 的 AVX2/AVX512 扩展实现和基准测试。

### 结论

**垂直 SIMD（纵向向量化）对 fused add/sub 无益**，因为该算法有两条串行进位链，SIMD 无法加速核心瓶颈。

`cpu_mont_bench` 的横向 SoA 批处理策略是 CPU 向量化的正确方向——每个 SIMD lane 处理独立 instance，无跨 lane 进位依赖。

### 当前架构

```
cpu_addsub_bench.cpp           (main + multi-variant benchmark loop)
  └── cpu_addsub_impl.h        (scalar + avx2_manual + avx2_lookahead)
        ├── cpu_add_fused_scalar       (标量基线)
        ├── cpu_add_fused_avx2_manual  (SIMD bulk + 标量 carry)
        ├── cpu_add_fused_avx2_lookahead (overflow+propagation mask)
        └── cpu_sub_fused_*    (对应 sub 变体)
```

Benchmark 自动注册并运行所有可用变体（基于编译时 ISA 检测），输出对比结果。

下列方案已讨论但未在当前阶段实施，记录供后续评估：

1. **AoS + gather/scatter 模式**: 对 prefetch-friendly 场景可能有优势
2. **AVX512 IFMA52**: 需 AVX512-IFMA52 指令集 (Intel Sapphire Rapids+), 可替代 CIOS 中的 64-bit 乘积
3. **更大位宽**: 2048-bit, 4096-bit — 需处理更多 limbs，SoA 缓冲区更大
4. **多线程并行**: 在 NUMA 节点上拆分曲线批处理，用 OpenMP 或线程池
5. **与 OpenCL GPU 交叉验证**: 同一输入在 CPU AVX 和 GPU 上运行并比对结果

## 实现顺序

- [ ] `src/cpu_mont_scalar.cpp` — 标量 CIOS 参考 + GMP 验证辅助
- [ ] `src/cpu_mont_avx.cpp` — AVX512 批处理 CIOS (16 条曲线, SoA)
- [ ] `src/cpu_mont_bench.cpp` — CLI + benchmark harness
- [ ] AVX2 回退路径 (8 条曲线)
- [ ] CMakeLists.txt 集成
- [ ] `test/cpu_mont_test_vectors.cpp` — 固定测试向量
- [ ] 编译验证 + 自测