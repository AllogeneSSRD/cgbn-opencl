# MONT_WG 最小 impl4 实施方案（中文）

本文目标是在现有 `MONT_WG_IMPL=1` 稳定路径上做最小增量，新增 `MONT_WG_IMPL=4`，并完成端到端可复现验证。

## 1) 目标与非目标

### 目标

- 提供一个**最小可行**的新实现：`impl4`。
- 保持数学路径与 `impl1` 一致，仅引入 1 个低风险优化思想。
- 通过 bench 的 GMP 护栏与 Stage1 烟测确认“可用 + 可回滚”。

### 非目标

- 不做大规模算法重写（不引入复杂 scan/prefix 结构）。
- 不追求一步到位超越所有场景性能。
- 不改动 `impl0/1/2/3` 的既有行为和语义。

## 2) 与当前 impl1 的差异点（只引入低风险思想）

`impl4` 基于 `impl1`，保留“并行 base 项 + tid0 串行归并”的结构。

仅引入一个优化思想：

- **串行归并双步展开（2-limb unroll）**  
  在 tid0 的 merge 循环中，每次处理两个 limb，减少循环控制开销和部分依赖链长度，逻辑不变、进位语义不变。

不引入以下高风险变化：

- 不改跨线程进位协议；
- 不改变 barrier 布局；
- 不引入新的 scratch 数据结构（沿用 `impl1` 布局）。

## 3) 数据流 / 线程协作草图（文字）

每轮 Montgomery 外层 `i`：

1. 所有线程并行计算 `base[j] = t[j] + ai*b[j]`（写 `sum_lo/sum_hi`）。
2. `barrier`。
3. `tid0` 执行归并（`impl4` 使用双步展开），写回 `t[j]` 并产生高位 carry。
4. `barrier`。
5. 所有线程并行计算 `base2[j] = t[j] + m*N[j]`（写 `sum_lo/sum_hi`）。
6. `barrier`。
7. `tid0` 执行右移语义归并（`j>0` 写 `t[j-1]`，`impl4` 双步展开）。
8. `barrier`，进入下一轮。

本质上 `impl4` 只改变 tid0 的“如何遍历 j”，不改变线程协作拓扑。

## 4) 正确性风险点与防护

风险点：

- 双步展开时最后奇数尾项处理错误（off-by-one）。
- 第二次归并中 `j==0` 不写 `t[-1]` 的保护被破坏。
- 进位链衔接顺序变化导致高位 `t[limbs] / t_hi` 不一致。

防护：

- 展开循环后保留显式尾项路径（`j < limbs`）。
- 保持与 `impl1` 一致的 `if (j > 0u)` 写回条件。
- 不改终态归约（`top/top2`）和最终 conditional subtract 路径。
- 验证时强制执行：
  - bench `mont_mul_wg_bench` 与 `mont_sqr_wg_bench` 的 GMP PASS；
  - Stage1 烟测对比 `impl1` 与 `impl4` 都能正常返回。

## 5) 验证矩阵

必跑：

1. **bench + GMP**  
   - `ECM_MONT_WG_IMPL=1`：`mont_mul_wg_bench` / `mont_sqr_wg_bench` PASS  
   - `ECM_MONT_WG_IMPL=4`：`mont_mul_wg_bench` / `mont_sqr_wg_bench` PASS

2. **Stage1 烟测（ecm.exe）**  
   - 固定 sigma、小参数，分别跑 `impl1` 与 `impl4`，确认：
     - 内核可构建；
     - 程序可运行并返回；
     - 无明显错误日志。

可选扩展：

- 针对 `M4007` / `M4019` 做短程 smoke，先验证稳定性再看耗时趋势。

## 6) 回滚策略

随时可回滚到稳定路径，无需改代码：

- 环境变量切回：`ECM_MONT_WG_IMPL=1`

若出现设备相关问题，建议同时固定：

- `ECM_DISABLE_MONT_WG=0`（仍走 WG）
- `ECM_OPENCL_TPI` 使用当前已验证值（例如 8 或 16）。

该策略确保 impl4 仅作为可选实验分支，不影响默认稳定可用性。
