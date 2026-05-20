# ECM OpenCL Operator Analysis (Hotspot Baseline)

This report captures the current operator implementation status and measured performance to guide next-step optimization and CL ASM work.

## 1) Main Kernel Operator Mix

Kernel: `kernel_double_add` in `cgbn/backends/opencl/kernels/ecm_stage1.cl`  
Core step: `double_add_v2(...)` per scalar bit.

Per one `double_add_v2` call (fixed mix):

- `mp_add_mod`: 4
- `mp_sub_mod`: 4
- `mont_mul_priv`: 4
- `mont_sqr_priv`: 4
- `mont_normalize`: 8
- `special_mult_ui32`: 1
- `mp_shift_left_1_mod`: 1

So total counts are:

`total_calls(op) = calls_per_bit(op) * total_processed_bits_over_all_curves`.

This shows immediately that `mont_mul_priv` / `mont_sqr_priv` dominate runtime, not add/sub.

## 2) Microbench Results (Measured)

Tool: `opencl_ecm_addsub.exe`  
Config: 1024-bit, `kernel_iterations=1000`, `instances=256`, `launch_repeats=50`.

Measured CSV: `ecm_operator_bench.csv`

- `ecm_mp_add_n_bench`: 80.2038 ms, `1.59593e+08 ops/s`, private mem 768 B
- `ecm_mp_add_mod_bench`: 75.6509 ms, `1.69198e+08 ops/s`, private mem 1024 B
- `ecm_mp_sub_mod_bench`: 69.2332 ms, `1.84882e+08 ops/s`, private mem 1024 B
- `ecm_mont_mul_priv_bench`: 4205.92 ms, `3.04333e+06 ops/s`, private mem 1808 B
- `ecm_mont_sqr_priv_bench`: 3970.06 ms, `3.22413e+06 ops/s`, private mem 1552 B

Conclusion: Montgomery mul/sqr are ~50x slower than add/sub primitives at the operator level and are the first optimization target.

## 3) Current Implementation Methods

### `mp_add_n`
- Multi-limb integer add in `Z` (base 2^32).
- Carry-propagation loop across limbs.
- No modular reduction.

### `mp_add_mod`
- Computes `a + b (mod N)`.
- Strategy: raw add, then conditional subtract `N` once if carry or `>= N`.
- Valid because inputs are expected reduced (`[0, N-1]`).

### `mp_sub_mod`
- Computes `a - b (mod N)`.
- Strategy: raw subtract; if underflow, add `N` (without re-reduction).

### `mont_mul_priv` (CIOS style)
- Coarsely Integrated Operand Scanning.
- Outer loop over limbs:
  1. accumulate `t += ai * b`
  2. compute reduction digit `m = t0 * np0`
  3. accumulate `t += m * N`
  4. shift/divide by radix (`2^32`) implicitly by index writes
- Final conditional subtraction by `N`.
- `mont_sqr_priv` currently aliases to `mont_mul_priv(a, a, ...)`.

## 4) Resource/Instruction Cost Notes

## 4.1 Register/private memory pressure

OpenCL-reported `CL_KERNEL_PRIVATE_MEM_SIZE` (bytes):

- add/sub kernels: 768~1024 B
- `mont_mul_priv` bench: 1808 B
- `mont_sqr_priv` bench: 1552 B

High private memory strongly suggests register pressure and likely spills on some architectures.

## 4.2 CIOS inner-loop arithmetic cost model

For one `(i, j)` pair in `mont_mul_priv`, a simplified scalar model is:

- `t[j] + ai*b[j] + carry`
- `t[j] + m*N[j] + carry`

A practical estimate per limb pair:

- ~2 multiply-accumulate chains + carry handling.
- If mapped to ISA FMA-like integer MAD paths, rough model:
  - **Cost ~= 2x FMAs + 1x ADDs** (plus carry/compare/shift overhead).

At algorithmic level, complexity is `O(limbs^2)`, so instruction count grows quadratically with modulus size.

## 5) Why `mp_` Prefix Matters

`mp_` means "multi-precision primitive":

- separates bignum limb math from curve-level formulas,
- clarifies arithmetic domain (`Z` vs `Z/NZ`),
- avoids confusion with OpenCL scalar/vector operations.

This naming is useful when introducing ASM-level replacements, because call sites stay semantically stable while implementations swap.

## 6) Optimization Directions (Priority Order)

1. **Montgomery mul/sqr specialization**
   - special-case `sqr` (symmetry) instead of aliasing to generic mul.
   - reduce temporaries (`B`, `Nloc`) and duplicate copies.
   - shorten live ranges to lower register pressure.

2. **Kernel structure / occupancy**
   - evaluate TPI/work-group cooperative mont kernels (`mont_wg`) for stage-1 path.
   - tune unrolling and launch size for occupancy vs ILP.

3. **Instruction-level improvements**
   - encourage MAD/mul_hi style integer codegen.
   - try backend-friendly explicit carry primitives.

4. **Memory traffic**
   - avoid repeated global<->private copies when possible.
   - consider storing invariants (`N`, `np0`) in better-cached spaces per wave/work-group.

5. **Profiling and verification workflow**
   - keep CSV output (`ECM_BENCH_CSV`) for regression tracking.
   - add stage-level operator-count print (`ECM_PROFILE_OPS=1`) on host side.

## 7) Next Step for CL ASM Prep

Recommended immediate path:

- Build a dedicated `mont_mul_priv` asm-prototype kernel with same interface.
- Compare against current CIOS kernel on:
  - ops/s,
  - private mem size,
  - max work-group size,
  - correctness over random vectors.

Use this report + `ecm_operator_bench.csv` as baseline before replacing stage-1 core paths.

