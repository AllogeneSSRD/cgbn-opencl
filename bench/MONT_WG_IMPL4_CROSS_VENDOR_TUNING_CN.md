# MONT_WG impl4 跨厂商调优说明（中文）

## 1) 发现的问题

- `impl4` 在 AMD 上有明显收益，但在 NVIDIA 上出现回退。
- `impl4` 的核心差异是 tid0 合并循环的双步展开，可能引入更高指令/寄存器压力，导致 NVIDIA 调度效率下降。
- 在现有工具链下，优先以可得指标（kernel private mem、pref_wg、max_wg、bench 吞吐）作为替代证据。

## 2) 本次新增策略

引入 `impl4` 的可调展开因子：

- 编译宏：`MONT_WG_IMPL4_UNROLL`
  - `1`：不展开（接近 impl1 的串行 merge 形态）
  - `2`：双步展开（原 impl4 行为）

并新增跨厂商自动默认策略（可回滚）：

- NVIDIA 默认：`impl4_unroll=1`
- 其他厂商（含 AMD）默认：`impl4_unroll=2`
- 可通过环境变量强制覆盖：
  - `ECM_MONT_WG_IMPL4_UNROLL=1|2`

## 3) AMD / NVIDIA 切换方式

### 自动策略（推荐）

- 仅设置 `ECM_MONT_WG_IMPL=4`，其余由 host 按 vendor 自动决定 `impl4_unroll`。

### 手工覆盖（A/B）

- NVIDIA 尝试稳态：
  - `ECM_MONT_WG_IMPL=4`
  - `ECM_MONT_WG_IMPL4_UNROLL=1`
- AMD 尝试收益：
  - `ECM_MONT_WG_IMPL=4`
  - `ECM_MONT_WG_IMPL4_UNROLL=2`

## 4) 回滚方法

最稳回滚（不走 impl4）：

- `ECM_MONT_WG_IMPL=1`

保留 impl4 但关闭展开：

- `ECM_MONT_WG_IMPL=4`
- `ECM_MONT_WG_IMPL4_UNROLL=1`

以上均无需改代码，可直接通过环境变量切换。
