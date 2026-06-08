#!/usr/bin/env python3
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

DEFAULT_ORT_EPS = [
    "VitisAIExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]

SELF_TEST_WIDTHS = (512, 1024, 2048, 4096)
DEFAULT_LIMB_SWEEP = (8, 16, 32, 64)  # limb count at 32-bit/limb
SUPPORTED_LIMB_BITS = (8, 16, 32, 64)
DEFAULT_LIMB_BITS_SWEEP = SUPPORTED_LIMB_BITS


def create_ort_session(model, preferred_eps: Optional[List[str]] = None):
    """Create ONNX Runtime session; returns (session, ep_name) or (None, None)."""
    if not HAS_ORT or model is None:
        return None, None
    eps = preferred_eps if preferred_eps is not None else DEFAULT_ORT_EPS
    available = ort.get_available_providers()
    for ep in eps:
        if ep not in available:
            continue
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if ep == "VitisAIExecutionProvider":
                providers = [(ep, {"target_engine": "NPU"})]
            else:
                providers = [ep]
            session = ort.InferenceSession(
                model.SerializeToString(),
                sess_options=opts,
                providers=providers,
            )
            return session, session.get_providers()[0]
        except Exception:
            continue
    try:
        session = ort.InferenceSession(model.SerializeToString())
        return session, session.get_providers()[0]
    except Exception:
        return None, None


def create_limb_binary_model(op: str, limbs: int, graph_name: str):
    """Element-wise int64 Add/Sub over [batch, limbs]."""
    if not HAS_ONNX:
        return None
    X = helper.make_tensor_value_info("X", TensorProto.INT64, ["batch", limbs])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, ["batch", limbs])
    Z = helper.make_tensor_value_info("Z", TensorProto.INT64, ["batch", limbs])
    node = helper.make_node(op, ["X", "Y"], ["Z"])
    graph = helper.make_graph([node], graph_name, [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    return model


class MpWidth:
    """N-bit integer with configurable limb width (8/16/32/64 bits per limb)."""

    MIN_BITS = 512
    MAX_BITS = 4096
    DEFAULT_LIMB_BITS = 32
    MIN_LIMBS_32 = MIN_BITS // DEFAULT_LIMB_BITS
    MAX_LIMBS_32 = MAX_BITS // DEFAULT_LIMB_BITS
    SWEEP_MIN_LIMBS = 8  # 256-bit at 32b/limb

    __slots__ = ("bits", "limb_bits")

    def __init__(self, bits: int, limb_bits: int = DEFAULT_LIMB_BITS) -> None:
        err = MpWidth.validate(bits, limb_bits)
        if err:
            raise ValueError(err)
        self.bits = bits
        self.limb_bits = limb_bits

    @classmethod
    def from_limb_count(
        cls,
        limb_count: int,
        limb_bits: int = DEFAULT_LIMB_BITS,
        *,
        allow_sweep: bool = False,
    ) -> "MpWidth":
        err = cls.validate_limb_count(limb_count, limb_bits, allow_sweep=allow_sweep)
        if err:
            raise ValueError(err)
        bits = limb_count * limb_bits
        if not allow_sweep:
            err = cls.validate(bits, limb_bits)
            if err:
                raise ValueError(err)
        obj = object.__new__(cls)
        obj.bits = bits
        obj.limb_bits = limb_bits
        return obj

    from_limbs = from_limb_count

    @staticmethod
    def validate(bits: int, limb_bits: int = DEFAULT_LIMB_BITS) -> Optional[str]:
        if limb_bits not in SUPPORTED_LIMB_BITS:
            return f"limb_bits must be one of {SUPPORTED_LIMB_BITS}"
        if bits < MpWidth.MIN_BITS or bits > MpWidth.MAX_BITS:
            return f"bits must be in [{MpWidth.MIN_BITS}, {MpWidth.MAX_BITS}]"
        if bits % limb_bits != 0:
            return f"bits ({bits}) must be a multiple of limb_bits ({limb_bits})"
        return None

    @staticmethod
    def validate_limb_count(
        limb_count: int,
        limb_bits: int = DEFAULT_LIMB_BITS,
        *,
        allow_sweep: bool = False,
    ) -> Optional[str]:
        if limb_bits not in SUPPORTED_LIMB_BITS:
            return f"limb_bits must be one of {SUPPORTED_LIMB_BITS}"
        min_limbs = MpWidth.SWEEP_MIN_LIMBS if (allow_sweep and limb_bits == 32) else 1
        max_limbs = MpWidth.MAX_BITS // limb_bits
        if limb_count < min_limbs or limb_count > max_limbs:
            return f"limb count must be in [{min_limbs}, {max_limbs}] for {limb_bits}b/limb"
        return None

    @staticmethod
    def validate_limbs(limbs: int, *, allow_sweep: bool = False) -> Optional[str]:
        return MpWidth.validate_limb_count(limbs, MpWidth.DEFAULT_LIMB_BITS, allow_sweep=allow_sweep)

    @property
    def limbs(self) -> int:
        return self.bits // self.limb_bits

    @property
    def limb_mask(self) -> int:
        return (1 << self.limb_bits) - 1


def limbs_from_int(value: int, width: MpWidth) -> np.ndarray:
    mask = width.limb_mask
    lb = width.limb_bits
    return np.array([(value >> (lb * i)) & mask for i in range(width.limbs)], dtype=np.uint64)


def int_from_limbs(arr: np.ndarray, width: MpWidth) -> int:
    result = 0
    mask = np.uint64(width.limb_mask)
    lb = width.limb_bits
    flat = np.asarray(arr, dtype=np.uint64).reshape(-1)
    for i in range(flat.shape[0]):
        result |= int(flat[i] & mask) << (lb * i)
    return result


def opencl_test_vectors(width: MpWidth) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same fixed integer values as opencl_ecm_addsub_bench.cpp, any limb layout."""
    bits = width.bits
    mask = (1 << bits) - 1
    n_val = (1 << (bits - 1)) - 109
    a_val = ((1 << (bits - 1)) - 991) % n_val
    b_val = ((1 << (bits - 1)) - 8218291649) % n_val
    return (
        limbs_from_int(n_val & mask, width),
        limbs_from_int(a_val & mask, width),
        limbs_from_int(b_val & mask, width),
    )


def tile_vectors(a: np.ndarray, b: np.ndarray, n: np.ndarray, instances: int):
    a_batch = np.tile(a.astype(np.int64), (instances, 1))
    b_batch = np.tile(b.astype(np.int64), (instances, 1))
    n_batch = np.tile(n.astype(np.int64), (instances, 1))
    return a_batch, b_batch, n_batch


def limbwise_add(width: MpWidth, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise limb add with carry propagation."""
    lb = width.limb_bits
    mask = width.limb_mask
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    batch, limbs = a.shape
    result = np.zeros((batch, limbs), dtype=np.uint64)
    if lb == 64:
        for i in range(batch):
            c = 0
            for j in range(limbs):
                s = int(a[i, j]) + int(b[i, j]) + c
                result[i, j] = s & mask
                c = s >> lb
        return result
    mask_np = np.uint64(mask)
    carry = np.zeros(batch, dtype=np.uint64)
    for j in range(limbs):
        s = a[:, j] + b[:, j] + carry
        result[:, j] = s & mask_np
        carry = s >> lb
    return result


def propagate_borrow_sub(width: MpWidth, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiprecision subtract a - b with per-limb borrow (unsigned limbs)."""
    mask = np.uint64(width.limb_mask)
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    borrow = np.zeros(batch, dtype=np.uint64)
    for j in range(limbs):
        av = a[:, j]
        bv = b[:, j]
        r[:, j] = (av - bv - borrow) & mask
        borrow = ((av < bv) | ((av == bv) & (borrow > 0))).astype(np.uint64)
    return r


def ref_mp_add_n(a: int, b: int, bits: int) -> int:
    return (a + b) & ((1 << bits) - 1)


def ref_mp_sub_n(a: int, n: int, bits: int) -> int:
    return (a - n) & ((1 << bits) - 1)


def ref_mp_add_mod(a: int, b: int, n: int) -> int:
    return (a + b) % n


def ref_mp_sub_mod(a: int, b: int, n: int) -> int:
    return (a - b) % n


def numpy_mp_add_n(width: MpWidth, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return limbwise_add(width, a, b)


def numpy_mp_sub_n(width: MpWidth, a: np.ndarray, n: np.ndarray) -> np.ndarray:
    return propagate_borrow_sub(width, a, n)


def numpy_mp_add_mod_legacy(width: MpWidth, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    lb = width.limb_bits
    mask = width.limb_mask
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    for i in range(batch):
        carry_add = 0
        carry_sub = 1
        ri = np.zeros(limbs, dtype=np.uint64)
        ai = a[i].astype(np.uint64)
        bi = b[i].astype(np.uint64)
        ni = n[i].astype(np.uint64)
        for j in range(limbs):
            s = int(ai[j]) + int(bi[j]) + carry_add
            carry_add = s >> lb
            t = (s & mask) + ((~int(ni[j])) & mask) + carry_sub
            carry_sub = t >> lb
            ri[j] = t & mask
        if (carry_add | carry_sub) == 0:
            c = 0
            for j in range(limbs):
                s = int(ri[j]) + int(ni[j]) + c
                ri[j] = s & mask
                c = s >> lb
        r[i] = ri
    return r


def numpy_mp_sub_mod_legacy(width: MpWidth, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    lb = width.limb_bits
    mask = width.limb_mask
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
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
            ri[j] = w & mask
            borrow = 1 if av < bv + borrow else 0
        if borrow:
            c = 0
            for j in range(limbs):
                s = int(ri[j]) + int(ni[j]) + c
                ri[j] = s & mask
                c = s >> lb
        r[i] = ri
    return r


def numpy_mp_add_mod_vec(width: MpWidth, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    if width.limb_bits == 64:
        return numpy_mp_add_mod_legacy(width, a, b, n)
    lb = width.limb_bits
    mask = np.uint64(width.limb_mask)
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    carry_add = np.zeros(batch, dtype=np.uint64)
    carry_sub = np.ones(batch, dtype=np.uint64)
    for j in range(limbs):
        s = a[:, j] + b[:, j] + carry_add
        carry_add = s >> lb
        t = (s & mask) + ((~n[:, j]) & mask) + carry_sub
        carry_sub = t >> lb
        r[:, j] = t & mask
    need_fix = (carry_add | carry_sub) == 0
    if np.any(need_fix):
        c = np.zeros(batch, dtype=np.uint64)
        for j in range(limbs):
            s = r[:, j] + n[:, j] + c
            new_r = s & mask
            new_c = s >> lb
            r[:, j] = np.where(need_fix, new_r, r[:, j])
            c = np.where(need_fix, new_c, np.uint64(0))
    return r


def numpy_mp_sub_mod_vec(width: MpWidth, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    if width.limb_bits == 64:
        return numpy_mp_sub_mod_legacy(width, a, b, n)
    lb = width.limb_bits
    mask = np.uint64(width.limb_mask)
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    borrow = np.zeros(batch, dtype=np.uint64)
    for j in range(limbs):
        av = a[:, j]
        bv = b[:, j]
        w = av - bv - borrow
        r[:, j] = w & mask
        borrow = (av < bv + borrow).astype(np.uint64)
    need_fix = borrow.astype(bool)
    if np.any(need_fix):
        c = np.zeros(batch, dtype=np.uint64)
        for j in range(limbs):
            s = r[:, j] + n[:, j] + c
            new_r = s & mask
            new_c = s >> lb
            r[:, j] = np.where(need_fix, new_r, r[:, j])
            c = np.where(need_fix, new_c, np.uint64(0))
    return r


class NPUAddSubBackend:
    def __init__(self, width: MpWidth, preferred_eps: Optional[List[str]] = None):
        self.width = width
        self.limbs = width.limbs
        self.preferred_eps = preferred_eps
        self.add_session = None
        self.sub_session = None
        self.add_ep: Optional[str] = None
        self.sub_ep: Optional[str] = None
        self._init_sessions()

    def _init_sessions(self) -> None:
        if not HAS_ONNX or not HAS_ORT:
            return
        tag = f"{self.width.limb_bits}b_x{self.limbs}"
        add_model = create_limb_binary_model("Add", self.limbs, f"mp_add_n_{tag}")
        sub_model = create_limb_binary_model("Sub", self.limbs, f"mp_sub_n_{tag}")
        self.add_session, self.add_ep = create_ort_session(add_model, self.preferred_eps)
        self.sub_session, self.sub_ep = create_ort_session(sub_model, self.preferred_eps)

    @property
    def active(self) -> bool:
        return self.add_session is not None and self.sub_session is not None

    @property
    def is_npu(self) -> bool:
        return ("VitisAI" in (self.add_ep or "")) or ("VitisAI" in (self.sub_ep or ""))

    def _onnx_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.add_session is None:
            return numpy_mp_add_n(self.width, a, b)
        self.add_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})
        return limbwise_add(self.width, a, b)

    def _onnx_sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.sub_session is None:
            return numpy_mp_sub_n(self.width, a, b)
        self.sub_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})
        return propagate_borrow_sub(self.width, a, b)

    def mp_add_n(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._onnx_add(a, b)

    def mp_sub_n(self, a: np.ndarray, n: np.ndarray) -> np.ndarray:
        return self._onnx_sub(a, n)

    def mp_add_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        return numpy_mp_add_mod_vec(self.width, a, b, n)

    def mp_sub_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        return numpy_mp_sub_mod_vec(self.width, a, b, n)


class NumpyBackend:
    def __init__(self, width: MpWidth):
        self.width = width

    def mp_add_n(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return numpy_mp_add_n(self.width, a, b)

    def mp_sub_n(self, a: np.ndarray, n: np.ndarray) -> np.ndarray:
        return numpy_mp_sub_n(self.width, a, n)

    def mp_add_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        return numpy_mp_add_mod_vec(self.width, a, b, n)

    def mp_sub_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        return numpy_mp_sub_mod_vec(self.width, a, b, n)


def verify_ops(width: MpWidth, a_vec, b_vec, n_vec, backends) -> bool:
    bits = width.bits
    limbs = width.limbs
    mask = (1 << bits) - 1
    a0 = int_from_limbs(a_vec, width) & mask
    b0 = int_from_limbs(b_vec, width) & mask
    n0 = int_from_limbs(n_vec, width) & mask
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
            got = int_from_limbs(got_arr[0], width)
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


def verify_fused_vec_matches_legacy(width: MpWidth, a_vec, b_vec, n_vec) -> bool:
    limbs = width.limbs
    a_batch = a_vec.reshape(1, limbs).astype(np.int64)
    b_batch = b_vec.reshape(1, limbs).astype(np.int64)
    n_batch = n_vec.reshape(1, limbs).astype(np.int64)
    pairs = (
        ("mp_add_mod", numpy_mp_add_mod_vec, numpy_mp_add_mod_legacy),
        ("mp_sub_mod", numpy_mp_sub_mod_vec, numpy_mp_sub_mod_legacy),
    )
    all_ok = True
    for name, fast, slow in pairs:
        got = fast(width, a_batch, b_batch, n_batch)
        ref = slow(width, a_batch, b_batch, n_batch)
        if np.array_equal(got, ref):
            print(f"  [fused_vec:{name}] matches legacy: PASS")
        else:
            print(f"  [fused_vec:{name}] matches legacy: FAIL")
            all_ok = False
    return all_ok


def run_self_test(
    widths: Tuple[int, ...] = SELF_TEST_WIDTHS,
    limb_bits_list: Tuple[int, ...] = SUPPORTED_LIMB_BITS,
    preferred_eps: Optional[List[str]] = None,
    include_onnx: bool = True,
) -> bool:
    print("npu_addsub self-test")
    all_ok = True
    for bits in widths:
        for lb in limb_bits_list:
            err = MpWidth.validate(bits, lb)
            if err:
                print(f"  [{bits}-bit, {lb}b/limb] width: FAIL ({err})")
                all_ok = False
                continue
            width = MpWidth(bits, lb)
            n_vec, a_vec, b_vec = opencl_test_vectors(width)
            backends: List[Tuple[str, object]] = [("numpy", NumpyBackend(width))]
            if include_onnx and HAS_ORT:
                onnx_be = NPUAddSubBackend(width, preferred_eps=preferred_eps)
                if onnx_be.active:
                    backends.insert(0, ("onnx", onnx_be))
            print(f"\n--- {bits}-bit, {lb}b/limb (limbs={width.limbs}) ---")
            if not verify_ops(width, a_vec, b_vec, n_vec, backends):
                all_ok = False
            if not verify_fused_vec_matches_legacy(width, a_vec, b_vec, n_vec):
                all_ok = False
    print()
    if all_ok:
        print("self-test: ALL PASS")
    else:
        print("self-test: FAILED")
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


def parse_limb_list(text: str) -> Tuple[int, ...]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("empty limb list")
    return tuple(out)


def collect_timings(
    width: MpWidth,
    instances: int,
    kernel_iterations: int,
    launch_repeats: int,
    warmup: int,
    npu_backend: NPUAddSubBackend,
    *,
    include_legacy: bool = False,
) -> Tuple[float, List[Tuple[str, str, float]]]:
    n_vec, a_vec, b_vec = opencl_test_vectors(width)
    a_batch, b_batch, n_batch = tile_vectors(a_vec, b_vec, n_vec, instances)
    numpy_be = NumpyBackend(width)
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
    if include_legacy:
        timings.append(
            (
                "mp_add_mod",
                "fused_legacy",
                bench_op(
                    lambda: numpy_mp_add_mod_legacy(width, a_batch, b_batch, n_batch),
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
                    lambda: numpy_mp_sub_mod_legacy(width, a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )

    return op_count, timings


def timing_lookup(timings: List[Tuple[str, str, float]], op: str, path: str) -> Optional[float]:
    for label, p, ms in timings:
        if label == op and p == path:
            return ms
    return None


def print_timings(timings: List[Tuple[str, str, float]], op_count: float) -> None:
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


def run_benchmark(
    width: MpWidth,
    instances,
    kernel_iterations,
    launch_repeats,
    warmup,
    npu_backend,
    *,
    include_legacy: bool = False,
):
    op_count, timings = collect_timings(
        width,
        instances,
        kernel_iterations,
        launch_repeats,
        warmup,
        npu_backend,
        include_legacy=include_legacy,
    )
    print_timings(timings, op_count)


def run_limb_sweep(
    limb_counts: Tuple[int, ...],
    instances: int,
    kernel_iterations: int,
    launch_repeats: int,
    warmup: int,
    preferred_eps: Optional[List[str]] = None,
    numpy_only: bool = False,
    *,
    include_legacy: bool = False,
) -> bool:
    print(
        f"Limb sweep: counts={limb_counts} (32-bit limbs), "
        f"kernel_iterations={kernel_iterations}, instances={instances}, launch_repeats={launch_repeats}"
    )
    if HAS_ORT:
        print(f"  onnxruntime={ort.__version__}, EPs={ort.get_available_providers()}")
    rows: List[dict] = []
    all_ok = True

    for limbs in limb_counts:
        err = MpWidth.validate_limbs(limbs, allow_sweep=True)
        if err:
            print(f"  [limbs={limbs}] SKIP: {err}")
            all_ok = False
            continue
        width = MpWidth.from_limbs(limbs, allow_sweep=True)
        n_vec, a_vec, b_vec = opencl_test_vectors(width)
        npu = NPUAddSubBackend(width, preferred_eps=preferred_eps)
        if numpy_only:
            npu.add_session = None
            npu.sub_session = None
        backends: List[Tuple[str, object]] = [("numpy", NumpyBackend(width))]
        if npu.active and not numpy_only:
            backends.insert(0, ("onnx", npu))
        if not verify_ops(width, a_vec, b_vec, n_vec, backends):
            all_ok = False
            continue
        op_count, timings = collect_timings(
            width,
            instances,
            kernel_iterations,
            launch_repeats,
            warmup,
            npu,
            include_legacy=include_legacy,
        )
        row = {
            "limbs": limbs,
            "limb_bits": width.limb_bits,
            "bits": width.bits,
            "npu_add_n": timing_lookup(timings, "mp_add_n", "npu_add_n"),
            "numpy_add_n": timing_lookup(timings, "mp_add_n", "numpy"),
            "npu_sub_n": timing_lookup(timings, "mp_sub_n", "npu_sub_n"),
            "numpy_sub_n": timing_lookup(timings, "mp_sub_n", "numpy"),
            "npu_add_mod": timing_lookup(timings, "mp_add_mod", "npu_fused"),
            "npu_sub_mod": timing_lookup(timings, "mp_sub_mod", "npu_fused"),
            "op_count": op_count,
        }
        rows.append(row)

    if not rows:
        print("No sweep results.")
        return False

    hdr = (
        f"{'limbs':>5} {'bits':>5} | "
        f"{'add_n NPU':>10} {'add_n np':>10} | "
        f"{'sub_n NPU':>10} {'sub_n np':>10} | "
        f"{'add_mod':>10} {'sub_mod':>10} | "
        f"{'us/limb add':>11}"
    )
    print()
    print("--- limb count sweep (ms per batch-run, lower is better) ---")
    print(hdr)
    print("-" * len(hdr))
    def _fmt_ms(v: Optional[float]) -> str:
        return f"{v:10.4f}" if v is not None else f"{'n/a':>10}"

    base_row = next((r for r in rows if r["limbs"] == DEFAULT_LIMB_SWEEP[0]), rows[0])
    base_limbs = base_row["limbs"]
    base_add = base_row.get("npu_add_n") or base_row.get("numpy_add_n")
    for row in rows:
        add_npu = row.get("npu_add_n")
        add_np = row.get("numpy_add_n")
        sub_npu = row.get("npu_sub_n")
        sub_np = row.get("numpy_sub_n")
        add_mod = row.get("npu_add_mod")
        sub_mod = row.get("npu_sub_mod")
        add_ref = add_npu if add_npu is not None else add_np
        us_per_limb = (add_ref / row["limbs"] * 1000.0) if add_ref is not None else None
        print(
            f"{row['limbs']:5d} {row['bits']:5d} | "
            f"{_fmt_ms(add_npu)} {_fmt_ms(add_np)} | "
            f"{_fmt_ms(sub_npu)} {_fmt_ms(sub_np)} | "
            f"{_fmt_ms(add_mod)} {_fmt_ms(sub_mod)} | "
            f"{us_per_limb if us_per_limb is not None else float('nan'):11.2f}"
        )

    print()
    print("--- scaling vs limbs=8 (NPU mp_add_n, linear=1.0x per doubling) ---")
    if base_add is not None and base_limbs > 0:
        for row in rows:
            add_npu = row.get("npu_add_n")
            add_np = row.get("numpy_add_n")
            ref = add_npu if add_npu is not None else add_np
            if ref is None:
                continue
            limb_ratio = row["limbs"] / base_limbs
            time_ratio = ref / base_add
            linear_ratio = time_ratio / limb_ratio if limb_ratio > 0 else float("nan")
            backend = "NPU" if add_npu is not None else "numpy"
            print(
                f"  limbs={row['limbs']:3d} ({row['bits']:4d}b) {backend}: "
                f"{time_ratio:.3f}x wall time vs {base_limbs} limbs, "
                f"{linear_ratio:.3f}x vs linear limb scaling"
            )

    print()
    print("Notes:")
    print("  - limbs = 32-bit word count; bits = limbs * 32")
    print("  - us/limb add uses NPU mp_add_n when available, else numpy")
    print("  - linear scaling ~1.0x => overhead dominates; >>1.0x => per-limb work dominates")
    return all_ok


def parse_limb_bits_list(text: str) -> Tuple[int, ...]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lb = int(part)
        if lb not in SUPPORTED_LIMB_BITS:
            raise ValueError(f"unsupported limb_bits {lb}, want {SUPPORTED_LIMB_BITS}")
        out.append(lb)
    if not out:
        raise ValueError("empty limb_bits list")
    return tuple(out)


def run_limb_bits_sweep(
    bits: int,
    limb_bits_list: Tuple[int, ...],
    instances: int,
    kernel_iterations: int,
    launch_repeats: int,
    warmup: int,
    preferred_eps: Optional[List[str]] = None,
    numpy_only: bool = False,
    *,
    include_legacy: bool = False,
) -> bool:
    print(
        f"Limb-bits sweep: {bits}-bit fixed, limb_bits={limb_bits_list}, "
        f"kernel_iterations={kernel_iterations}, instances={instances}, launch_repeats={launch_repeats}"
    )
    if HAS_ORT:
        print(f"  onnxruntime={ort.__version__}, EPs={ort.get_available_providers()}")
    rows: List[dict] = []
    all_ok = True

    for lb in limb_bits_list:
        err = MpWidth.validate(bits, lb)
        if err:
            print(f"  [{bits}b, {lb}b/limb] SKIP: {err}")
            all_ok = False
            continue
        width = MpWidth(bits, lb)
        n_vec, a_vec, b_vec = opencl_test_vectors(width)
        npu = NPUAddSubBackend(width, preferred_eps=preferred_eps)
        if numpy_only:
            npu.add_session = None
            npu.sub_session = None
        backends: List[Tuple[str, object]] = [("numpy", NumpyBackend(width))]
        if npu.active and not numpy_only:
            backends.insert(0, ("onnx", npu))
        if not verify_ops(width, a_vec, b_vec, n_vec, backends):
            all_ok = False
            continue
        op_count, timings = collect_timings(
            width,
            instances,
            kernel_iterations,
            launch_repeats,
            warmup,
            npu,
            include_legacy=include_legacy,
        )
        rows.append(
            {
                "limb_bits": lb,
                "limbs": width.limbs,
                "bits": bits,
                "npu_add_n": timing_lookup(timings, "mp_add_n", "npu_add_n"),
                "numpy_add_n": timing_lookup(timings, "mp_add_n", "numpy"),
                "npu_sub_n": timing_lookup(timings, "mp_sub_n", "npu_sub_n"),
                "numpy_sub_n": timing_lookup(timings, "mp_sub_n", "numpy"),
                "npu_add_mod": timing_lookup(timings, "mp_add_mod", "npu_fused"),
                "npu_sub_mod": timing_lookup(timings, "mp_sub_mod", "npu_fused"),
                "op_count": op_count,
            }
        )

    if not rows:
        print("No sweep results.")
        return False

    def _fmt_ms(v: Optional[float]) -> str:
        return f"{v:10.4f}" if v is not None else f"{'n/a':>10}"

    hdr = (
        f"{'lb':>3} {'limbs':>5} {'bits':>5} | "
        f"{'add_n NPU':>10} {'add_n np':>10} | "
        f"{'sub_n NPU':>10} {'sub_n np':>10} | "
        f"{'add_mod':>10} {'sub_mod':>10} | "
        f"{'us/limb':>9}"
    )
    print()
    print("--- limb bit-width sweep, same N-bit (ms per batch-run) ---")
    print(hdr)
    print("-" * len(hdr))

    base_row = next((r for r in rows if r["limb_bits"] == DEFAULT_LIMB_BITS_SWEEP[0]), rows[0])
    base_add = base_row.get("npu_add_n") or base_row.get("numpy_add_n")
    for row in rows:
        add_npu = row.get("npu_add_n")
        add_np = row.get("numpy_add_n")
        sub_npu = row.get("npu_sub_n")
        sub_np = row.get("numpy_sub_n")
        add_mod = row.get("npu_add_mod")
        sub_mod = row.get("npu_sub_mod")
        add_ref = add_npu if add_npu is not None else add_np
        us_per_limb = (add_ref / row["limbs"] * 1000.0) if add_ref is not None else None
        print(
            f"{row['limb_bits']:3d} {row['limbs']:5d} {row['bits']:5d} | "
            f"{_fmt_ms(add_npu)} {_fmt_ms(add_np)} | "
            f"{_fmt_ms(sub_npu)} {_fmt_ms(sub_np)} | "
            f"{_fmt_ms(add_mod)} {_fmt_ms(sub_mod)} | "
            f"{us_per_limb if us_per_limb is not None else float('nan'):9.2f}"
        )

    print()
    print(f"--- scaling vs {base_row['limb_bits']}b/limb (mp_add_n wall time) ---")
    if base_add is not None:
        for row in rows:
            add_npu = row.get("npu_add_n")
            add_np = row.get("numpy_add_n")
            ref = add_npu if add_npu is not None else add_np
            if ref is None:
                continue
            limb_ratio = row["limbs"] / base_row["limbs"]
            time_ratio = ref / base_add
            backend = "NPU" if add_npu is not None else "numpy"
            print(
                f"  {row['limb_bits']:2d}b/limb, {row['limbs']:3d} limbs: {backend} add_n "
                f"{time_ratio:.3f}x vs {base_row['limb_bits']}b/limb "
                f"(limb count {limb_ratio:.2f}x)"
            )

    print()
    print("Notes:")
    print("  - Same total bits; wider limbs => fewer limbs, less carry-loop iterations")
    print("  - ONNX still does element-wise add/sub over [batch, limbs] int64 slots")
    return all_ok


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NPU add/sub/mod microbench (OpenCL ecm_addsub core ops)",
        epilog=(
            "Environment: conda activate ryzen-ai-1.7.1\n"
            "Example: python RyzenAI/npu_addsub.py --bits 512 10000 128 2\n"
            "Limb count sweep (32b/limb): --limbs-sweep 8,16,32,64 500 32 1\n"
            "Limb bits sweep (same N): --limb-bits-sweep --bits 512 500 32 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bits",
        type=int,
        default=None,
        help=f"Total bit width N, {MpWidth.MIN_BITS}..{MpWidth.MAX_BITS}",
    )
    p.add_argument(
        "--limb-bits",
        type=int,
        default=MpWidth.DEFAULT_LIMB_BITS,
        choices=list(SUPPORTED_LIMB_BITS),
        help="Bits per limb (8/16/32/64); limbs = bits / limb_bits",
    )
    p.add_argument(
        "--limb-bits-sweep",
        nargs="?",
        const=",".join(str(x) for x in DEFAULT_LIMB_BITS_SWEEP),
        default=None,
        help=f"Benchmark limb bit-widths at fixed --bits (default {DEFAULT_LIMB_BITS_SWEEP})",
    )
    p.add_argument(
        "--limbs",
        type=int,
        default=None,
        help=f"Limb count at 32b/limb; bits=limbs*32",
    )
    p.add_argument(
        "--limbs-sweep",
        nargs="?",
        const=",".join(str(x) for x in DEFAULT_LIMB_SWEEP),
        default=None,
        help=f"Benchmark limb counts (default {DEFAULT_LIMB_SWEEP}), e.g. --limbs-sweep 8,16,32,64",
    )
    p.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    p.add_argument("--numpy-only", action="store_true", help="Skip ONNX/NPU backend")
    p.add_argument(
        "--self-test",
        action="store_true",
        help=f"Run correctness tests for {SELF_TEST_WIDTHS} and exit (no benchmark)",
    )
    p.add_argument(
        "--onnx-ep",
        type=str,
        default=None,
        help="Comma-separated ONNX EP priority (e.g. CPUExecutionProvider)",
    )
    p.add_argument(
        "--fused-legacy",
        action="store_true",
        help="Include fused_legacy mod tier in benchmark (slow per-row Python loops)",
    )
    p.add_argument("positional", nargs="*", help="kernel_iterations [instances] [launch_repeats]")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    preferred_eps = None
    if args.onnx_ep:
        preferred_eps = [s.strip() for s in args.onnx_ep.split(",") if s.strip()]

    if args.self_test:
        return 0 if run_self_test(preferred_eps=preferred_eps, include_onnx=not args.numpy_only) else 1

    kernel_iterations = int(args.positional[0]) if len(args.positional) >= 1 else 10000
    instances = int(args.positional[1]) if len(args.positional) >= 2 else 128
    launch_repeats = int(args.positional[2]) if len(args.positional) >= 3 else 2

    if args.limb_bits_sweep is not None:
        bits = args.bits if args.bits is not None else 512
        try:
            lb_list = parse_limb_bits_list(args.limb_bits_sweep)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        ok = run_limb_bits_sweep(
            bits,
            lb_list,
            instances,
            kernel_iterations,
            launch_repeats,
            args.warmup,
            preferred_eps=preferred_eps,
            numpy_only=args.numpy_only,
            include_legacy=args.fused_legacy,
        )
        return 0 if ok else 1

    if args.limbs_sweep is not None:
        try:
            limb_counts = parse_limb_list(args.limbs_sweep)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        ok = run_limb_sweep(
            limb_counts,
            instances,
            kernel_iterations,
            launch_repeats,
            args.warmup,
            preferred_eps=preferred_eps,
            numpy_only=args.numpy_only,
            include_legacy=args.fused_legacy,
        )
        return 0 if ok else 1

    if args.limbs is not None and args.bits is not None:
        print("Use only one of --bits or --limbs", file=sys.stderr)
        return 1

    if args.limbs is not None:
        err = MpWidth.validate_limbs(args.limbs, allow_sweep=True)
        if err:
            print(err, file=sys.stderr)
            return 1
        width = MpWidth.from_limb_count(args.limbs, allow_sweep=True)
    else:
        bits = args.bits if args.bits is not None else 512
        err = MpWidth.validate(bits, args.limb_bits)
        if err:
            print(err, file=sys.stderr)
            return 1
        width = MpWidth(bits, args.limb_bits)

    print(
        f"NPU add/sub microbench: {width.bits}-bit ({width.limb_bits}b/limb x {width.limbs} limbs), "
        f"kernel_iterations={kernel_iterations}, instances={instances}, launch_repeats={launch_repeats}"
    )
    if HAS_ORT:
        print(f"  onnxruntime={ort.__version__}, EPs={ort.get_available_providers()}")
    else:
        print("  onnxruntime: NOT INSTALLED (NumPy fallback only)")

    npu = NPUAddSubBackend(width, preferred_eps=preferred_eps)
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
    backends: List[Tuple[str, object]] = [("numpy", NumpyBackend(width))]
    if npu.active and not args.numpy_only:
        backends.insert(0, ("npu", npu))

    print()
    if not verify_ops(width, a_vec, b_vec, n_vec, backends):
        print("\nVerify FAILED")
        return 1

    run_benchmark(
        width,
        instances,
        kernel_iterations,
        launch_repeats,
        args.warmup,
        npu,
        include_legacy=args.fused_legacy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
