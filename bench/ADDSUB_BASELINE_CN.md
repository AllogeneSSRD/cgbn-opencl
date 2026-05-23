# Add/Sub/Mod 纯核函数基线（4096-bit，AMD gfx1150）

优化前基线数据，对应三步工作流：纯核函数 → RGA 反汇编 → bench 吞吐。

## 1. 纯核函数（无 in-kernel 计时循环）

`ecm_addsub_bench.cl` 中每个 `__kernel` 仅执行 **一次** 算术操作：

| 核函数 | 语义 |
|--------|------|
| `ecm_mp_add_n` | r = a + b |
| `ecm_mp_sub_n` | r = a - N |
| `ecm_mp_add_mod` | r = (a + b) mod N |
| `ecm_mp_sub_mod` | r = (a - b) mod N |

与旧 `*_bench` 内核的区别：无 `iterations` 循环、无 `mp_copy` 反馈链，便于 ISA 分析反映真实算子成本。

## 2. 导出与 RGA 反汇编

```powershell
cmake --build build --config Debug --target opencl_addsub_isa_export
.\tools\disasm_addsub_isa.ps1 -Bits 4096
```

产物：

- 二进制：`bench/addsub_isa_4096_amd.bin`
- 每个核函数各一份：
  - `bench/gfx1150_ecm_mp_*_addsub_isa_4096_amd.rga.isa.txt`
  - `bench/gfx1150_ecm_mp_*_addsub_isa_4096_amd.rga.livereg.txt`
  - `bench/gfx1150_ecm_mp_*_addsub_isa_4096_amd.rga.analysis.csv`

## 3. 吞吐基准（纯核函数）

```powershell
$env:ECM_BENCH_CSV = "bench\addsub_baseline_4096_amd.csv"
.\build\Debug\opencl_ecm_addsub.exe -d 1 --addsub-only --bits 4096 1000 128 50
```

参数：`kernel_iterations=1000`, `instances=128`, `launch_repeats=50`  
总 ops = 128 × 1000 × 50 = **6.4M**

### 4096-bit 基线吞吐（AMD Radeon 890M，2026-05-22）

| 核函数 | 时间 (ms) | 吞吐 (ops/s) | private (B) |
|--------|-----------|--------------|-------------|
| ecm_mp_add_n | 2087.29 | 3.07M | 1568 |
| ecm_mp_sub_n | 1933.15 | 3.31M | 1568 |
| ecm_mp_add_mod | 2433.70 | 2.63M | 2080 |
| ecm_mp_sub_mod | 2061.72 | 3.10M | 2080 |

CSV：`bench/addsub_baseline_4096_amd.csv`

### RGA 资源摘要（4096-bit）

| 核函数 | ISA_SIZE | SCRATCH | USED_SGPR | USED_VGPR |
|--------|----------|---------|-----------|-----------|
| ecm_mp_add_n | 480 | 1540 | 16 | 9 |
| ecm_mp_sub_n | 484 | 1540 | 16 | 11 |
| ecm_mp_add_mod | 1052 | 2052 | 16 | 12 |
| ecm_mp_sub_mod | 660 | 2052 | 16 | 12 |

## 4. 后续优化对照

优化后重复同一命令，对比：

- `bench/addsub_baseline_4096_amd.csv` 吞吐变化
- 对应 `.rga.isa.txt` / `.rga.livereg.txt` 指令数与 live-register 压力
