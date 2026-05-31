#!/usr/bin/env python3
"""Regenerate all mp_addsub generated OpenCL sources."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
ROOT = TOOLS.parent

SCRIPTS = [
    "gen_mp_add_mod_unroll.py",
    "gen_mp_sub_mod_unroll.py",
    "gen_mp_addmod_asm_block16.py",
    "gen_mp_addmod_asm_block32.py",
    "gen_mp_addmod_asm_block64.py",
    "gen_mp_submod_asm_block32.py",
    "gen_mp_submod_asm_block64.py",
    "gen_mp_addmod_asm_fused.py",
    "gen_mp_submod_asm_fused.py",
    "gen_mp_addsub_asm_block32_stage1.py",
    "gen_mp_addsub_asm_block16_stage1.py",
]


def main() -> None:
    MP_GEN = ROOT / "cgbn/backends/opencl/kernels/mp_addsub/generated"
    MP_GEN.mkdir(parents=True, exist_ok=True)
    (ROOT / "cgbn/backends/opencl/kernels/mp_addsub/stage1").mkdir(parents=True, exist_ok=True)

    for name in SCRIPTS:
        path = TOOLS / name
        if not path.is_file():
            print(f"skip missing {name}")
            continue
        print(f"running {name}...")
        subprocess.check_call([sys.executable, str(path)], cwd=str(ROOT))
    print("done.")


if __name__ == "__main__":
    main()
