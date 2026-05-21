# MONT 4096-bit ISA 分析（中文）

本文基于 `opencl_mont_isa_export` + RGA(`-s bin`) 对 4096-bit 的 `mont` 系列函数进行分析。

相关输入与产物：

- 二进制：`bench/mont_isa_4096_amd.bin`
- RGA 统计：`bench/gfx1150_*_mont_isa_4096_amd.rga.analysis.csv`
- ISA 文本：`bench/gfx1150_*_mont_isa_4096_amd.rga.isa.txt`
- 指令计数汇总：`bench/mont_isa_4096_amd.rga.inst_counts.csv`

---

## 1. 专有名词解释

### 1.1 Scratch（私有溢出内存）

- 含义：当内核中线程私有数据（寄存器需求）超出可用寄存器时，编译器会把一部分“溢出”到显存（scratch）。
- 特点：
  - 访问延迟远高于寄存器；
  - 常导致性能明显下降；
  - 大位宽/大数组内核容易触发。
- 一般结论：`scratch` 越大，通常越不利于吞吐与延迟。

### 1.2 SGPR（Scalar General Purpose Register）

- 含义：标量寄存器，主要承载控制流、地址、循环计数、标量常量等。
- 过高影响：
  - 可能降低并发驻留的 wave 数（occupancy）。
- 常见来源：
  - 分支复杂、循环控制复杂、地址计算多、等待/同步控制多。

### 1.3 VGPR（Vector General Purpose Register）

- 含义：向量寄存器，承载每个 lane 的数据通路（SIMD 运算数据）。
- 过高影响：
  - 同样会压低 occupancy；
  - 常见于大整数乘加链、临时变量多、向量中间态多。

### 1.4 ISA_SIZE

- RGA 给出的内核 ISA 大小（字节），可用于粗略衡量代码体量。
- 注意：代码体量大不一定慢，但通常意味着控制路径/指令数更复杂。

### 1.5 s_waitcnt

- 用于等待内存/流水线依赖满足的同步指令。
- 数量高通常说明：
  - 数据依赖链长；
  - memory/指令调度隐藏不够；
  - 可能存在等待气泡。

---

## 2. 4096-bit 结果汇总

## 2.1 资源视角（RGA analysis）

- `ecm_mont_mul_priv_bench`
  - `ISA_SIZE=1472`
  - `SCRATCH_MEM=2568`
  - `USED_SGPR=19`, `USED_VGPR=18`

- `ecm_mont_sqr_priv_bench`
  - `ISA_SIZE=1472`
  - `SCRATCH_MEM=2056`
  - `USED_SGPR=19`, `USED_VGPR=18`

- `cgbn_mont_mul_wg_bench`
  - `ISA_SIZE=2352`
  - `SCRATCH_MEM=0`
  - `USED_SGPR=39`, `USED_VGPR=37`

- `cgbn_mont_sqr_wg_bench`
  - `ISA_SIZE=2332`
  - `SCRATCH_MEM=0`
  - `USED_SGPR=38`, `USED_VGPR=37`

观察：

- `priv` 路径寄存器占用较低，但有明显 scratch 溢出；
- `wg` 路径寄存器占用更高、ISA 更大，但无 scratch。

这解释了为何在大位宽下 WG 往往更稳：避免了高代价的 scratch 访问。

## 2.2 指令类别视角（ISA 文本计数）

- `ecm_mont_mul_priv_bench`
  - total: `251`
  - `v_*`: `86`
  - `s_*`: `161`
  - `mul/mad/fma`: `6`
  - memory: `4`
  - branch: `31`
  - `s_waitcnt`: `18`

- `ecm_mont_sqr_priv_bench`
  - total: `255`
  - `v_*`: `82`
  - `s_*`: `170`
  - `mul/mad/fma`: `6`
  - memory: `3`
  - branch: `31`
  - `s_waitcnt`: `17`

- `cgbn_mont_mul_wg_bench`
  - total: `446`
  - `v_*`: `169`
  - `s_*`: `229`
  - `mul/mad/fma`: `9`
  - memory: `48`
  - branch: `33`
  - `s_waitcnt`: `26`

- `cgbn_mont_sqr_wg_bench`
  - total: `441`
  - `v_*`: `169`
  - `s_*`: `224`
  - `mul/mad/fma`: `10`
  - memory: `48`
  - branch: `32`
  - `s_waitcnt`: `25`

观察：

- WG 指令总量显著高于 priv（约 1.7x），且 memory 指令明显更多；
- 但 WG 避免了 scratch，实际性能可能仍优于 priv（取决于位宽与并行度）。

---

## 3. 如何解读这组数据

不要只看“指令总数”：

- `priv`: 指令少，但 scratch 高（慢内存风险）。
- `wg`: 指令多、同步多，但能避免 scratch，且更适配大位宽协作。

对 ECM Stage1（4096+ 位宽）而言，更重要的是：

1. 是否触发 scratch；
2. `s_waitcnt` 与 branch 密度；
3. memory traffic（尤其 local/global 往返）；
4. 实测吞吐与 occupancy。

---

## 4. 下一步优化优先级建议

按收益/风险排序：

1. **优先压低 priv 路径 scratch**
   - 减少临时数组与生命周期重叠；
   - 先做低风险变量复用。

2. **优化 WG 的等待与内存访问**
   - 降低不必要 barrier / waitcnt；
   - 合并或重排 local memory 访问。

3. **针对 WG 调参**
   - 结合 `TPI`、work-group 配置，观察 SGPR/VGPR 对 occupancy 的影响；
   - 找到“寄存器压力 vs 协作收益”的平衡点。

4. **阶段性复测**
   - 每次改动后同时记录：
     - RGA `ISA_SIZE/SGPR/VGPR/SCRATCH`
     - 算子微基准吞吐
     - 端到端 Stage1 时间

---

## 5. 备注

- 本文中的“指令条数”来自 ISA 文本的类别统计，属于工程上有用的近似指标；
- 若需要更精确 pipeline 级诊断，可结合更细粒度 profiler 指标（issue、stall、memory pipeline）。
