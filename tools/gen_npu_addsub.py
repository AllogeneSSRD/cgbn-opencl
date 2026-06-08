#!/usr/bin/env python3
"""Generate RyzenAI/npu_addsub.py (UTF-8). Run: python tools/gen_npu_addsub.py"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "RyzenAI" / "npu_addsub.py"

CONTENT = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU add/sub/mod microbench - mirrors opencl_ecm_addsub core ops.

Run environment:
  conda activate ryzen-ai-1.7.1
  python RyzenAI/npu_addsub.py --bits 512 10000 128 2
  python RyzenAI/npu_addsub.py --bits 2048 500 32 1

Operations (ecm_addsub_bench.cl):
  mp_add_n    r = a + b
  mp_sub_n    r = a - N
  mp_add_mod  r = (a + b) mod N   (fused speculative subtract)
  mp_sub_mod  r = (a - b) mod N
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from npu_uint512 import create_add_model, create_session, create_sub_model
except ImportError:
    create_add_model = None
    create_sub_model = None
    create_session = None


class MpWidth:
    """Multiprecision width: 512..4096 bits in 32-bit limbs (32/64-bit aligned widths)."""

    MIN_BITS = 512
    MAX_BITS = 4096
    LIMB_BITS = 32

    __slots__ = ("bits",)

    def __init__(self, bits: int) -> None:
        err = MpWidth.validate(bits)
        if err:
            raise ValueError(err)
        self.bits = bits

    @staticmethod
    def validate(bits: int) -> Optional[str]:
        if bits < MpWidth.MIN_BITS or bits > MpWidth.MAX_BITS:
            return f"bits must be in [{MpWidth.MIN_BITS}, {MpWidth.MAX_BITS}]"
        if bits % MpWidth.LIMB_BITS != 0:
            return f"bits must be a multiple of {MpWidth.LIMB_BITS}"
        return None

    @property
    def limbs(self) -> int:
        return self.bits // MpWidth.LIMB_BITS


def limbs_from_int(value: int, limbs: int) -> np.ndarray:
    mask = (1 << 32) - 1
    return np.array([(value >> (32 * i)) & mask for i in range(limbs)], dtype=np.uint32)


def int_from_limbs(arr: np.ndarray) -> int:
    result = 0
    flat = np.asarray(arr).reshape(-1)
    for i in range(flat.shape[0]):
        result |= int(flat[i] & 0xFFFFFFFF) << (32 * i)
    return result


def opencl_test_vectors(width: MpWidth) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same fixed values as opencl_ecm_addsub_bench.cpp."""
    bits = width.bits
    limbs = width.limbs
    mask = (1 << bits) - 1
    n_val = (1 << (bits - 1)) - 109
    a_val = ((1 << (bits - 1)) - 991) % n_val
    b_val = ((1 << (bits - 1)) - 8218291649) % n_val
    return (
        limbs_from_int(n_val & mask, limbs),
        limbs_from_int(a_val & mask, limbs),
        limbs_from_int(b_val & mask, limbs),
    )


def tile_vectors(a: np.ndarray, b: np.ndarray, n: np.ndarray, instances: int):
    a_batch = np.tile(a.astype(np.int64), (instances, 1))
    b_batch = np.tile(b.astype(np.int64), (instances, 1))
    n_batch = np.tile(n.astype(np.int64), (instances, 1))
    return a_batch, b_batch, n_batch


def _propagate_carry(sum_arr: np.ndarray) -> np.ndarray:
    result = sum_arr.copy()
    limbs = result.shape[1]
    for j in range(limbs):
        carry = result[:, j] >> 32
        result[:, j] &= 0xFFFFFFFF
        if j + 1 < limbs:
            result[:, j + 1] += carry
    return result


def _propagate_borrow(diff_arr: np.ndarray) -> np.ndarray:
    result = diff_arr.copy()
    borrow = np.zeros(result.shape[0], dtype=np.int64)
    for j in range(result.shape[1]):
        result[:, j] -= borrow
        neg_mask = result[:, j] < 0
        result[:, j] = np.where(neg_mask, result[:, j] + (1 << 32), result[:, j])
        borrow = np.where(neg_mask, 1, 0).astype(np.int64)
    return result & 0xFFFFFFFF


propagate_carry = _propagate_carry
propagate_borrow = _propagate_borrow


def ref_mp_add_n(a: int, b: int, bits: int) -> int:
    return (a + b) & ((1 << bits) - 1)


def ref_mp_sub_n(a: int, n: int, bits: int) -> int:
    return (a - n) & ((1 << bits) - 1)


def ref_mp_add_mod(a: int, b: int, n: int) -> int:
    return (a + b) % n


def ref_mp_sub_mod(a: int, b: int, n: int) -> int:
    return (a - b) % n


def numpy_mp_add_n(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return propagate_carry(a.astype(np.int64) + b.astype(np.int64))


def numpy_mp_sub_n(a: np.ndarray, n: np.ndarray) -> np.ndarray:
    return propagate_borrow(a.astype(np.int64) - n.astype(np.int64))


def numpy_mp_add_mod_legacy(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint32)
    for i in range(batch):
        carry_add = 0
        carry_sub = 1
        ri = np.zeros(limbs, dtype=np.uint64)
        ai = a[i].astype(np.uint64)
        bi = b[i].astype(np.uint64)
        ni = n[i].astype(np.uint64)
        for j in range(limbs):
            s = int(ai[j]) + int(bi[j]) + carry_add
            carry_add = s >> 32
            t = (s & 0xFFFFFFFF) + ((~int(ni[j])) & 0xFFFFFFFF) + carry_sub
            carry_sub = t >> 32
            ri[j] = t & 0xFFFFFFFF
        if (carry_add | carry_sub) == 0:
            c = 0
            for j in range(limbs):
                s = int(ri[j]) + int(ni[j]) + c
                ri[j] = s & 0xFFFFFFFF
                c = s >> 32
        r[i] = ri.astype(np.uint32)
    return r


def numpy_mp_sub_mod_legacy(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint32)
    for i in range(batch):
        borrow = 0
        ri = np.zeros(limbs, dtype=np.uint64)
        ai = a[i].astype(np.uint64)
        bi = b[i].astype(np.uint64)
        ni = n[i].astype(np.uint64)
        for j in range(limbs):
            av = int(ai[j])
            bv = int(bi[j])
            w = av - bv - borrow
            ri[j] = w & 0xFFFFFFFF
            borrow = 1 if av < bv + borrow else 0
        if borrow:
            c = 0
            for j in range(limbs):
                s = int(ri[j]) + int(ni[j]) + c
                ri[j] = s & 0xFFFFFFFF
                c = s >> 32
        r[i] = ri.astype(np.uint32)
    return r


def numpy_mp_add_mod_vec(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    carry_add = np.zeros(batch, dtype=np.uint64)
    carry_sub = np.ones(batch, dtype=np.uint64)
    mask32 = np.uint64(0xFFFFFFFF)
    for j in range(limbs):
        s = a[:, j] + b[:, j] + carry_add
        carry_add = s >> 32
        t = (s & mask32) + ((~n[:, j]) & mask32) + carry_sub
        carry_sub = t >> 32
        r[:, j] = t & mask32
    need_fix = (carry_add | carry_sub) == 0
    if np.any(need_fix):
        c = np.zeros(batch, dtype=np.uint64)
        for j in range(limbs):
            s = r[:, j] + n[:, j] + c
            new_r = s & mask32
            new_c = s >> 32
            r[:, j] = np.where(need_fix, new_r, r[:, j])
            c = np.where(need_fix, new_c, np.uint64(0))
    return r.astype(np.uint32)


def numpy_mp_sub_mod_vec(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    borrow = np.zeros(batch, dtype=np.uint64)
    mask32 = np.uint64(0xFFFFFFFF)
    for j in range(limbs):
        av = a[:, j]
        bv = b[:, j]
        w = av - bv - borrow
        r[:, j] = w & mask32
        borrow = (av < bv + borrow).astype(np.uint64)
    need_fix = borrow.astype(bool)
    if np.any(need_fix):
        c = np.zeros(batch, dtype=np.uint64)
        for j in range(limbs):
            s = r[:, j] + n[:, j] + c
            new_r = s & mask32
            new_c = s >> 32
            r[:, j] = np.where(need_fix, new_r, r[:, j])
            c = np.where(need_fix, new_c, np.uint64(0))
    return r.astype(np.uint32)


numpy_mp_add_mod = numpy_mp_add_mod_vec
numpy_mp_sub_mod = numpy_mp_sub_mod_vec


class NPUAddSubBackend:
    def __init__(self, width: MpWidth):
        self.width = width
        self.limbs = width.limbs
        self.add_session = None
        self.sub_session = None
        self.add_ep: Optional[str] = None
        self.sub_ep: Optional[str] = None
        self._init_sessions()

    def _create_binary_model(self, op: str, name: str):
        if not HAS_ONNX:
            return None
        X = helper.make_tensor_value_info("X", TensorProto.INT64, ["batch", self.limbs])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT64, ["batch", self.limbs])
        Z = helper.make_tensor_value_info("Z", TensorProto.INT64, ["batch", self.limbs])
        node = helper.make_node(op, ["X", "Y"], ["Z"])
        graph = helper.make_graph([node], name, [X, Y], [Z])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        model.ir_version = 8
        return model

    def _init_sessions(self) -> None:
        if not HAS_ONNX or not HAS_ORT or create_session is None:
            return
        add_model = None
        sub_model = None
        try:
            import npu_uint512 as u512

            saved = u512.LIMBS
            u512.LIMBS = self.limbs
            if create_add_model is not None:
                add_model = create_add_model()
            if create_sub_model is not None:
                sub_model = create_sub_model()
            u512.LIMBS = saved
        except Exception:
            pass
        if add_model is None:
            add_model = self._create_binary_model("Add", "add_graph")
        if sub_model is None:
            sub_model = self._create_binary_model("Sub", "sub_graph")
        if add_model is not None:
            self.add_session, self.add_ep = create_session(add_model)
        if sub_model is not None:
            self.sub_session, self.sub_ep = create_session(sub_model)

    @property
    def active(self) -> bool:
        return self.add_session is not None and self.sub_session is not None

    @property
    def is_npu(self) -> bool:
        return ("VitisAI" in (self.add_ep or "")) or ("VitisAI" in (self.sub_ep or ""))

    def _onnx_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.add_session is None:
            return numpy_mp_add_n(a, b)
        raw = self.add_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})[0]
        return propagate_carry(raw)

    def _onnx_sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.sub_session is None:
            return numpy_mp_sub_n(a, b)
        raw = self.sub_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})[0]
        return propagate_borrow(raw)

    def mp_add_n(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._onnx_add(a, b)

    def mp_sub_n(self, a: np.ndarray, n: np.ndarray) -> np.ndarray:
        return self._onnx_sub(a, n)

    def mp_add_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        return numpy_mp_add_mod_vec(a, b, n)

    def mp_sub_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        return numpy_mp_sub_mod_vec(a, b, n)


class NumpyBackend:
    mp_add_n = staticmethod(numpy_mp_add_n)
    mp_sub_n = staticmethod(numpy_mp_sub_n)
    mp_add_mod = staticmethod(numpy_mp_add_mod_vec)
    mp_sub_mod = staticmethod(numpy_mp_sub_mod_vec)


def verify_ops(width: MpWidth, a_vec, b_vec, n_vec, backends) -> bool:
    bits = width.bits
    limbs = width.limbs
    mask = (1 << bits) - 1
    a0 = int_from_limbs(a_vec) & mask
    b0 = int_from_limbs(b_vec) & mask
    n0 = int_from_limbs(n_vec) & mask
    expects = {
        "mp_add_n": ref_mp_add_n(a0, b0, bits),
        "mp_sub_n": ref_mp_sub_n(a0, n0, bits),
        "mp_add_mod": ref_mp_add_mod(a0, b0, n0),
        "mp_sub_mod": ref_mp_sub_mod(a0, b0, n0),
    }
    a_batch = a_vec.reshape(1, limbs).astype(np.int64)
    b_batch = b_vec.reshape(1, limbs).astype(np.int64)
    n_batch = n_vec.reshape(1, limbs).astype(np.int64)
    runners = {
        "mp_add_n": lambda be: be.mp_add_n(a_batch, b_batch),
        "mp_sub_n": lambda be: be.mp_sub_n(a_batch, n_batch),
        "mp_add_mod": lambda be: be.mp_add_mod(a_batch, b_batch, n_batch),
        "mp_sub_mod": lambda be: be.mp_sub_mod(a_batch, b_batch, n_batch),
    }
    all_ok = True
    for be_name, backend in backends:
        for op_name, expected in expects.items():
            got_arr = runners[op_name](backend)
            got = int_from_limbs(got_arr[0])
            if op_name in ("mp_add_mod", "mp_sub_mod"):
                ok = (got % n0) == expected
            else:
                ok = (got & mask) == (expected & mask)
            if ok:
                print(f"  [{be_name}:{op_name}] Python verify: PASS")
            else:
                print(f"  [{be_name}:{op_name}] verify: FAIL (got={got}, expect={expected})")
                all_ok = False
    return all_ok


def bench_op(fn: Callable[[], None], warmup: int, iters: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        for _ in range(iters):
            fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def run_benchmark(width: MpWidth, instances, kernel_iterations, launch_repeats, warmup, npu_backend):
    n_vec, a_vec, b_vec = opencl_test_vectors(width)
    a_batch, b_batch, n_batch = tile_vectors(a_vec, b_vec, n_vec, instances)
    numpy_be = NumpyBackend()
    op_count = float(instances * kernel_iterations * launch_repeats)
    timings: List[Tuple[str, str, float]] = []

    if npu_backend.active:
        timings.append(
            (
                "mp_add_n",
                "npu_add_n",
                bench_op(lambda: npu_backend.mp_add_n(a_batch, b_batch), warmup, kernel_iterations, launch_repeats),
            )
        )
        timings.append(
            (
                "mp_sub_n",
                "npu_sub_n",
                bench_op(lambda: npu_backend.mp_sub_n(a_batch, n_batch), warmup, kernel_iterations, launch_repeats),
            )
        )
        timings.append(
            (
                "mp_add_mod",
                "npu_fused",
                bench_op(
                    lambda: npu_backend.mp_add_mod(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_sub_mod",
                "npu_fused",
                bench_op(
                    lambda: npu_backend.mp_sub_mod(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )

    timings.append(
        ("mp_add_n", "numpy", bench_op(lambda: numpy_be.mp_add_n(a_batch, b_batch), warmup, kernel_iterations, launch_repeats))
    )
    timings.append(
        ("mp_sub_n", "numpy", bench_op(lambda: numpy_be.mp_sub_n(a_batch, n_batch), warmup, kernel_iterations, launch_repeats))
    )
    timings.append(
        (
            "mp_add_mod",
            "numpy_fused",
            bench_op(
                lambda: numpy_be.mp_add_mod(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )
    timings.append(
        (
            "mp_sub_mod",
            "numpy_fused",
            bench_op(
                lambda: numpy_be.mp_sub_mod(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )
    timings.append(
        (
            "mp_add_mod",
            "fused_legacy",
            bench_op(
                lambda: numpy_mp_add_mod_legacy(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )
    timings.append(
        (
            "mp_sub_mod",
            "fused_legacy",
            bench_op(
                lambda: numpy_mp_sub_mod_legacy(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )

    by_op: dict = {}
    for label, path, ms in timings:
        by_op.setdefault(label, []).append((path, ms))

    print()
    print("--- mp_add_n / mp_sub_n ---")
    for op in ("mp_add_n", "mp_sub_n"):
        rows = by_op.get(op, [])
        for i, (path, ms) in enumerate(rows):
            ops_s = op_count / (ms / 1000.0)
            line = f"  {op} ({path}): {ms:.4f} ms, {ops_s:.6g} ops/s"
            if i + 1 < len(rows):
                line += f" ({rows[i + 1][1] / ms:.6g}x vs next)"
            print(line)

    print("--- mp_add_mod (priority high -> low) ---")
    add_mod_rows = by_op.get("mp_add_mod", [])
    for i, (path, ms) in enumerate(add_mod_rows):
        ops_s = op_count / (ms / 1000.0)
        line = f"  [{i + 1}] {path}: {ms:.4f} ms, {ops_s:.6g} ops/s"
        if i + 1 < len(add_mod_rows):
            line += f" ({add_mod_rows[i + 1][1] / ms:.6g}x vs next tier)"
        print(line)

    print("--- mp_sub_mod (priority high -> low) ---")
    sub_mod_rows = by_op.get("mp_sub_mod", [])
    for i, (path, ms) in enumerate(sub_mod_rows):
        ops_s = op_count / (ms / 1000.0)
        line = f"  [{i + 1}] {path}: {ms:.4f} ms, {ops_s:.6g} ops/s"
        if i + 1 < len(sub_mod_rows):
            line += f" ({sub_mod_rows[i + 1][1] / ms:.6g}x vs next tier)"
        print(line)

    print()
    for op in ("mp_add_n", "mp_sub_n", "mp_add_mod", "mp_sub_mod"):
        rows = by_op.get(op, [])
        if not rows:
            continue
        path, ms = rows[0]
        ops_s = op_count / (ms / 1000.0)
        suffix = ""
        if len(rows) > 1:
            suffix = f" (vs {rows[1][0]}: {rows[1][1] / ms:.6g}x)"
        print(f"{op}: {ms:.4f} ms, {ops_s:.6g} ops/s [{path}]{suffix}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NPU add/sub/mod microbench (OpenCL ecm_addsub core ops)",
        epilog=(
            "Environment: conda activate ryzen-ai-1.7.1\n"
            "Example: python RyzenAI/npu_addsub.py --bits 512 10000 128 2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bits",
        type=int,
        default=512,
        help=f"Bit width, {MpWidth.MIN_BITS}..{MpWidth.MAX_BITS}, multiple of 32 (64 ok)",
    )
    p.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    p.add_argument("--numpy-only", action="store_true", help="Skip ONNX/NPU backend")
    p.add_argument("positional", nargs="*", help="kernel_iterations [instances] [launch_repeats]")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    err = MpWidth.validate(args.bits)
    if err:
        print(err, file=sys.stderr)
        return 1
    width = MpWidth(args.bits)

    kernel_iterations = int(args.positional[0]) if len(args.positional) >= 1 else 10000
    instances = int(args.positional[1]) if len(args.positional) >= 2 else 128
    launch_repeats = int(args.positional[2]) if len(args.positional) >= 3 else 2

    print(
        f"NPU add/sub microbench: {width.bits}-bit, kernel_iterations={kernel_iterations}, "
        f"instances={instances}, launch_repeats={launch_repeats}, limbs={width.limbs}"
    )
    if HAS_ORT:
        print(f"  onnxruntime={ort.__version__}, EPs={ort.get_available_providers()}")
    else:
        print("  onnxruntime: NOT INSTALLED (NumPy fallback only)")

    npu = NPUAddSubBackend(width)
    if args.numpy_only:
        npu.add_session = None
        npu.sub_session = None
    elif npu.active:
        print(f"  NPU backend: add_ep={npu.add_ep}, sub_ep={npu.sub_ep}")
        if npu.is_npu:
            print("  >>> VitisAI NPU acceleration ACTIVE")
        else:
            print(f"  >>> ONNX active on {npu.add_ep}")
    else:
        print("  >>> ONNX sessions unavailable; NumPy baseline only")

    n_vec, a_vec, b_vec = opencl_test_vectors(width)
    backends: List[Tuple[str, object]] = [("numpy", NumpyBackend())]
    if npu.active and not args.numpy_only:
        backends.insert(0, ("npu", npu))

    print()
    if not verify_ops(width, a_vec, b_vec, n_vec, backends):
        print("\nVerify FAILED")
        return 1

    run_benchmark(width, instances, kernel_iterations, launch_repeats, args.warmup, npu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(CONTENT, encoding="utf-8", newline="\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
