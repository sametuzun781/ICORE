# Validation of the GitHub source repack

## Documentation-only disclosure update

The current update adds an English development-transparency and human-review-status note to the main README, the HTML methods book, and the introductory Markdown of the seven notebooks. All other files are byte-for-byte unchanged from the preceding GitHub source archive. Notebook code cells, saved outputs, and metadata are unchanged.

The numerical and performance results below are retained from the preceding repack; they are not new measurements from this update. No solver run or performance benchmark was repeated for this disclosure-only update. Automated checks do not replace independent human review.

## Previous repack validation

This is a packaging, documentation, and notebook cleanup of the build-fixed ICORE
0.20.0 numerical engine. It is not a new performance release or a retuning of the
algorithm. The starting engine also matches the 97 ICORE files in the latest Rocket
and Quadrotor Fast Discretization v2 packages.

## What stayed unchanged

- All **57 native C/header files** are byte-for-byte unchanged.
- All **11 checked numerical/runtime Python files**, including the API, native
  binding, presolve, model definitions and optional JAX implementation, are unchanged.
- All **393 NPZ problem files** and the catalog are byte-for-byte unchanged.
- The three dependency source archives have readable outer filenames, but their
  bytes are unchanged. Their automatic content checks and original licenses remain.
- Numerical tolerances, IPM policies, structural certificates, refinement, mixed
  precision guards, and fallback behavior are not relaxed or removed.

Only three build-support files have archive-path/documentation edits. The separately
instrumented reporting helper accepts explicit replay settings without duplicate
keyword arguments. The production solve/update path is not changed.

## Compiled-code comparison

Fresh local C and C++ builds were compared with the preceding distribution under
the same compiler settings. **Both complete shared-library files were byte-for-byte
identical**, not merely their solutions or selected instruction sections.

Configuration: Linux/x86-64, GCC/G++ 14.2.0, the retained BLASFEO HASWELL target,
O3 and the existing LTO setting. C library size: 1,593,192 bytes; C++: 1,588,872 bytes.

This is stronger evidence for retaining the tested native solve performance than
judging tiny differences in noisy microbenchmark times. It is not a claim that every
wall-clock sample, compiler, machine, or future model must have identical timing.

## Numerical regression comparison

All 393 catalog instances were solved with both native backends in the old and
new checkouts. **786/786 comparisons** had identical numerical fields and native
diagnostics, and passed the independent original-model accuracy audit at 1e-8.
Timing fields were excluded from numerical equality. No initial guess, tolerance,
or problem coefficient was changed to get a passing result.

The analytic KKT smoke suite passed **336/336** checks, including coefficient updates
and structural transitions. All **18** catalog/report/tutorial unit tests passed.
The visible array-building cells of the five model tutorials exactly matched the
retained reference models; both backends solved and audited each model. Both
backends also passed 12 MPC updates against fresh-workspace solutions.

The ordinary teaching report was tested with the numerical solve disabled after
the production result was obtained: the report does not secretly solve again.
Four opt-in detailed replays (scalar and SOC models, C and C++) matched production
bit for bit, passed their audits, and dropped no recorded events. They used an
explicit nondefault iteration budget to check replay-setting forwarding.

## Complete comparison pipeline

All mode was run with **393 problems x 4 methods x 2 repetitions = 3,144 records**.
This was a functionality/accuracy check, not a controlled speedup experiment.

| Method | Passed records | Failed accuracy/status | Unsupported |
|---|---:|---:|---:|
| ICORE C | 786 | 0 | 0 |
| ICORE C++ | 786 | 0 | 0 |
| HPIPM BALANCE | 688 | 2 | 96 |
| QOCO | 724 | 62 | 0 |

HPIPM's 96 unsupported records are 48 SOC instances repeated twice. Failures remain
in the data and do not become speed wins. These are outcomes of these adapters and
settings, not general claims about upstream solver quality. Faster/slower/tied/
incomparable tables cover all 393 cases. Resuming the completed saved run reproduced
identical records without repeating completed problems.

## Notebook and portable-build checks

All **six independent tutorial notebooks plus Compare.ipynb** ran from fresh
Jupyter kernels: 42 code cells, no execution errors. Tutorial notebooks retain
computed plots/results for browsing. Comparison outputs are intentionally cleared
in the source distribution rather than presenting local timings as universal claims.
The existing HTML methods book was not rewritten in this repack.

A separate clean source copy was built with **Clang, C and C++, and GENERIC BLASFEO**.
All seven unavailable-sweep entry points satisfied their rejection contract, did not
overwrite outputs, and the 336-check analytic smoke suite passed. The previously
missing `mixed_sweep_stubs.h` is present in the archive.

No actual macOS/ARM execution or performance claim is made. The optional JAX numerical
source is preserved, but no new full-catalog JAX test was performed. Finite tests do
not constitute an all-input proof.

A later documentation-only update adds the root MIT License for ICORE-owned
portions. The numerical results above predate that license update; they are not
new benchmark or test results. All engine, example, and notebook files are
unchanged by the licensing update. See `../LICENSE-NOTE.md` for scope and
maintainer release responsibilities.

## Reproduce

From the repository root, with a working native toolchain:

```bash
python -m unittest tests.test_examples tests.test_tutorials
python -m tests.smoke
python -m tests.portable_build
jupyter lab
```

Run each tutorial top to bottom. In Compare.ipynb, set RUN_ALL=True and ALL_N=2 to
repeat the pipeline check. Use more repetitions and controlled conditions for
performance research. Generated runs and build caches are ignored by Git.
