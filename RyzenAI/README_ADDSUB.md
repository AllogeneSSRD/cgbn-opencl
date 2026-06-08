# NPU add/sub microbench

`npu_addsub.py` mirrors OpenCL `opencl_ecm_addsub.exe --addsub-only` core ops:

| Op | Semantics |
|----|-----------|
| `mp_add_n` | `r = a + b` |
| `mp_sub_n` | `r = a - N` |
| `mp_add_mod` | `r = (a + b) mod N` (fused speculative subtract) |
| `mp_sub_mod` | `r = (a - b) mod N` |

Limb-wise add/sub use ONNX Runtime (VitisAI NPU when available); carry/borrow and mod fix use NumPy (same pattern as `npu_uint512.py`).

## Environment

```powershell
conda activate ryzen-ai-1.7.1
cd D:\code\MPA-OpenCl
```

Requires: `numpy`, `onnx`, `onnxruntime` (with `VitisAIExecutionProvider` for NPU).

Regenerate script after edits:

```powershell
python tools/gen_npu_addsub.py
```

## Usage (OpenCL-compatible positional args)

```powershell
# kernel_iterations instances launch_repeats
python RyzenAI/npu_addsub.py --bits 512 10000 128 2

# NumPy baseline only (no ONNX/NPU)
python RyzenAI/npu_addsub.py --numpy-only --bits 512 1000 64 1
```

Compare with OpenCL:

```powershell
.\build\Debug\opencl_ecm_addsub.exe -d 1 --addsub-only --bits 512 10000 128 2
```
