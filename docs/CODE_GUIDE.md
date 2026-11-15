# Source map

Start with `tutorials/01_scalar_regulator.ipynb`, then follow a solve down these layers.
The tutorials construct their arrays inline; reporting helpers do not construct models.

| Layer | Main files | Responsibility |
|---|---|---|
| Public Python API | `icore/api.py`, `icore/model.py` | Validate options, prepare/update/solve, expose diagnostics |
| Native binding | `icore/_native.py` | Array preparation, ctypes interface, owned result arrays |
| Exact presolve | `icore/presolve.py` | Coefficient-certified OCP-to-chain reduction and original-coordinate recovery |
| Native dispatch | `icore/src/api.c` | Workspace ownership and selected numerical body |
| Shared IPM | `icore/src/driver.c`, `driver_loop.h` | Mehrotra predictor/corrector and convergence/step logic |
| Dense QP | `dense_qp.h`, `dense_qp_model.h`, `dense_bound_pairs.h`, `dense_la.c` | Dense structural path and its numerical operations |
| General OCP | `ocp.c`, `ocp_model.h`, `ocp_factor.h`, `ocp_newton.h` | Stage assembly, Riccati/QR factorization and directions |
| OCP state and initialization | `ocp_workspace.h`, `ocp_initialize.h`, `ocp_vectors.h` | Memory ownership, initialization and residual operations |
| Cone operations | `ocp_cones.h`, `orthant.h` | Orthant/SOC scaling and step computations |
| Packed / mixed precision | `ocp_la.c`, `mixed_factor_impl.h`, `mixed_sweep_impl.h`, `mixed_sweep_stubs.h` | Packed kernels, guarded precision paths, portable rejection contract |
| Adjacent-stage chain | `chain.c`, `chain_model.h`, `chain_driver.h`, `chain_sparse/` | Residual/epigraph structure, sparse recovery and dual reconstruction |
| Build support | `icore/_build.py`, `_blasfeo_build.py`, `vendor/` | Automatic local builds and safe cache reuse |
| Example and comparison layer | `example_tools/` | Loading, exact comparator adapters, audits, reports and untimed replay |

Paths without a directory in the table are under `icore/src/`.
The public C header is `icore/src/icore.h`; the low-level OCP header is
`icore/src/icore_ocp.h`.

## Why some branches remain

Structural checks, finite-value guards, factor recovery, exact reduction, mixed-precision
acceptance and cache invalidation are correctness/performance mechanisms, not cosmetic
clutter. This repack does not delete numerical branches merely to reduce line count.
The shared IPM and every native arithmetic source file are retained.

## Teaching versus infrastructure

- `tutorials/*.ipynb`: math, arrays, direct solve, plots.
- `example_tools/tutorial.py`: optional short summary and independent audit; no solver changes.
- `example_tools/trace.py`: an opt-in instrumented copy, never the production timing library.
- `tests/reference_models.py`: previous tutorial models retained only as regression oracles;
  the notebooks do not call them.
- `problems/`: stored coefficients, not source code or saved solver warm starts.

The existing source-linked HTML book is retained unchanged. Its numerical source
snapshots still match this numerical engine; build/archive filenames and the notebook
layout described in this guide supersede its older packaging descriptions.
