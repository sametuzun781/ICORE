# Building and troubleshooting

## Requirements

Use Python 3.10+, a working C/C++ toolchain, `make`, and CMake 3.18 or newer (below 4
for the bundled comparator projects). Install the Python packages with:

```bash
python -m pip install -r requirements.txt
```

Use the same environment as the Jupyter kernel. A compiler is a system tool, not a
Python package. On macOS, `xcode-select --install` installs the Command Line Tools.
On Linux use your distribution's development-tool packages. Native Windows is not
a target of this build helper; use WSL.

## First run

Importing `Solver` does not solve a problem. The first `Solver(problem, backend=...)`
may build its native library. Subsequent compatible uses reuse the build. Dependencies
are supplied as source archives; do not install a separately configured HPIPM/BLASFEO
and assume it is the library used by the comparison notebook.

C and C++ share numerical source but may generate different machine code. Neither
fast-math nor relaxed stopping tolerances are enabled by this cleanup.

## Automatic build cache

Hidden `.build/` folders are generated locally. They track which source, compiler,
and dependency configuration produced each library. Automatic content checks prevent
Python from loading an old binary after source changes or mixing incompatible runs.
You do not need to read, copy, or edit these identifiers. They do not run inside the
native IPM iteration. Generated build folders and result folders are ignored by Git.

Do not remove these checks from the implementation to make the displayed repository
look simpler: they protect correctness. The old machine-specific manifests and
release receipts are not shipped.

## Apple Silicon / portable path

The existing conservative ARM target selection is unchanged. This is not a release
that enables an unvalidated optimized Apple kernel. The missing-header fix is included:

```text
icore/src/mixed_sweep_stubs.h
```

It supplies rejection stubs for unavailable mixed-precision sweeps so the existing
FP64 path remains available. It is not an empty placeholder. To check the GENERIC
build branch on your host:

```bash
python -m tests.portable_build
```

A GENERIC test on x86 is not an ARM/macOS execution test.

## Common issues

- **No compiler:** install the system toolchain and restart the notebook kernel.
- **An old package is imported:** open a fresh kernel from this checkout. Inspect
  `import icore; print(icore.__file__)` to confirm the path.
- **Edited source:** let the automatic build cache detect the change; restart the
  kernel when changing packages. Do not copy compiled libraries between hosts.
- **A comparison solver is unavailable:** the comparison preparation table and raw
  records retain the error. It is not counted as a timing win.
- **Optional JAX backend:** install JAX separately and select it explicitly. Native
  examples do not require it, and this repack does not add JAX validation coverage.

Environment options `ICORE_BLASFEO`, `ICORE_BLASFEO_TARGET`, and `ICORE_LTO` are
advanced build controls. Leave them unset for the normal supported path.
