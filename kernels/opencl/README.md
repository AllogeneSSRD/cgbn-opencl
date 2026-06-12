# ECM Stage1 OpenCL kernels

Stage1 sources live under `kernels/opencl/` (peer to `src/`). Host concatenates only the
files required for the resolved operator paths; see `opencl_ecm_stage1_kernel_source_paths()`.

## Directory layout

| Directory | Role |
|-----------|------|
| `common/` | Shared config + limb primitives (`stage1_config.h.cl`, `mp_priv.h.cl`) |
| `mont_mul/` | Montgomery multiplication implementations |
| `mont_sqr/` | Montgomery square dispatch (reuses `ecm_stage1_mont_sqr` from mul operator file) |
| `add_mod/` | Modular addition implementations |
| `sub_mod/` | Modular subtraction implementations |
| `ecm_stage1.cl` | Ladder + curve logic only (`double_add_v2`, `kernel_double_add`, coop glue) |

## Naming convention

**Rule:** directory = operator family; filename = registry CLI `id` (or obvious width variant).

| Family | Directory | File pattern | Registry `id` example |
|--------|-----------|--------------|------------------------|
| Montgomery mul | `mont_mul/` | `unroll_{bits}.cl`, `priv_opt.cl` | `unroll_only_384` → `unroll_384.cl` |
| Montgomery 4096 | `mont_mul/4096/` | `unroll_64.cl`, `fips_4096.cl` | `unroll64_4096`, `fips4096` |
| Montgomery sqr | `mont_sqr/` | `dispatch.cl` only | same kernel file as mul |
| Add mod | `add_mod/` | `{variant}_{width}.cl` | `unroll_384b`, `asm_512b`, `fused_unroll` |
| Sub mod | `sub_mod/` | `{variant}_{width}.cl` | mirrors `add_mod/` |

**Variants:** `unroll`, `asm`, `fused`, `fused_unroll`, `priv_opt`, `fips`.

**Widths:** Montgomery fixed-operator sizes use bit width (`384`, `512`, `32`); add/sub use
registry suffix (`128b`, `384b`, `512b`, `4096b`).

## Operator file contract

Each loaded operator `.cl` defines one of:

- `ecm_stage1_mont_mul` / `ecm_stage1_mont_sqr` (montgomery)
- `ecm_stage1_add_mod` / `ecm_stage1_sub_mod` (modular add/sub)

Dispatch shells (`*/dispatch.cl`) expose the legacy symbols used by the ladder:

- `mont_mul_stage1`, `mont_sqr_stage1`, `mp_add_mod`, `mp_sub_mod`

## Load order

1. `common/stage1_config.h.cl`
2. `common/mp_priv.h.cl`
3. Selected operator files (mul, sqr, add, sub — deduped)
4. `mont_mul/dispatch.cl`, `mont_sqr/dispatch.cl`, `add_mod/dispatch.cl`, `sub_mod/dispatch.cl`
5. `ecm_stage1.cl`
