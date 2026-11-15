"""Regression test for the packaged mixed-precision fallback.

Run from the distribution root:
    python -m tests.portable_build

This forces GENERIC BLASFEO in this process, checks the unavailable-sweep ABI,
and runs the existing 336-check C/C++ analytic KKT smoke suite. It does not
change the user's shell environment or enable fast-math. On an x86 host this
is a GENERIC-branch test, not an ARM/macOS execution test.
"""
from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    header = root / "icore" / "src" / "mixed_sweep_stubs.h"
    if not header.is_file():
        raise RuntimeError(f"Required portable implementation is missing: {header}")

    os.environ["ICORE_BLASFEO_TARGET"] = "GENERIC"
    os.environ["ICORE_BLASFEO"] = "on"
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    from icore._build import build, build_info

    pointer = ct.POINTER(ct.c_double)
    workspace = ct.c_void_p
    signatures = {
        "available": ([workspace], ct.c_int, [None], 0),
        "active": ([workspace], ct.c_int, [None], 0),
        "invalidate": ([workspace], None, [None], None),
        "begin": ([workspace, pointer, ct.c_double, ct.c_double, ct.c_double],
                  ct.c_int, [None, None, 0.0, 0.0, 0.0], -6),
        "stage": ([workspace, ct.c_int, pointer, ct.c_double, ct.c_double, ct.c_double],
                  ct.c_int, [None, 0, None, 0.0, 0.0, 0.0], -6),
        "backward": ([workspace, ct.c_int, pointer, pointer, pointer, ct.c_int,
                      pointer, pointer, pointer, ct.c_int],
                     ct.c_int, [None, 0, None, None, None, 0, None, None, None, 0], -6),
        "forward": ([workspace, ct.c_int, pointer, pointer, pointer, pointer,
                     pointer, pointer, pointer],
                    ct.c_int, [None, 0, None, None, None, None, None, None, None], -6),
    }
    for backend in ("c", "cpp"):
        path = build(backend)
        info = build_info(backend, library=path)
        if info["target"] != "GENERIC":
            raise AssertionError(f"Expected GENERIC backend, got {info}")
        library = ct.CDLL(str(path))
        for suffix, (argtypes, restype, args, expected) in signatures.items():
            function = getattr(library, "icore_ocp_la_mixed_sweeps_" + suffix)
            function.argtypes, function.restype = argtypes, restype
            actual = function(*args)
            if actual != expected:
                raise AssertionError(f"{backend}/{suffix}: {actual!r} != {expected!r}")
        # The rejection contract must not overwrite caller-owned output arrays.
        arrays = [(ct.c_double * 4)(1.0, -2.0, 3.0, -4.0) for _ in range(3)]
        snapshots = [bytes(array) for array in arrays]
        backward = library.icore_ocp_la_mixed_sweeps_backward
        forward = library.icore_ocp_la_mixed_sweeps_forward
        if backward(None, 0, None, None, None, 0, *arrays, 0) != -6:
            raise AssertionError("Unavailable backward sweep returned the wrong status")
        if forward(None, 0, None, None, None, None, *arrays) != -6:
            raise AssertionError("Unavailable forward sweep returned the wrong status")
        if snapshots != [bytes(array) for array in arrays]:
            raise AssertionError("An unavailable sweep modified an output buffer")
        print(f"{backend}: GENERIC build and all seven fallback entry points passed", flush=True)

    subprocess.run([sys.executable, "-m", "tests.smoke"], cwd=root,
                   env=os.environ.copy(), check=True)
    print("Portable build regression test passed.", flush=True)


if __name__ == "__main__":
    main()
