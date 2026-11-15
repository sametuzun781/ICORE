# Saved-run benchmark reporting review

## Basis and scope

This update is based on the supplied `ICORE-main.zip` repository snapshot and the
previously supplied `runs.zip`, specifically `runs/all393_N20`. The uploaded archive,
not an assumed live GitHub commit, is the source of truth. No new measurements were
performed. QOCOGEN was not run.

The review covers the README, benchmark reporting script, runner timing/eligibility
code, related documentation, tutorial model definitions, and package integrity.
It is not a proof of the native numerical algorithm for all inputs, a new performance
study, or an independent full mathematical review of the solver implementation.

## Findings and changes

| File / location in uploaded snapshot | Finding | Updated behavior |
|---|---|---|
| `scripts/generate_benchmark_summary.py`, `summarize_problem` | Averaged only accepted rows, so a partly successful group could be ranked | Exactly N unique repetitions must pass, with finite positive solve times; the mean uses all N |
| Same script, `discover_runs_dir` | Silently selected a largest run; N=5 and N=20 both contain 393 files | Ambiguous parent directories are rejected; select `--runs-dir runs/all393_N20` |
| Same script, `main` | Used the maximum repetition count without validating a consistent run | Checks repetitions, methods, tolerance, configuration, inputs, completion, duplicate names, and recorded environment consistency |
| README, summary figure and generated documentation | Used “supported” for recorded successes | Uses “passed all repetitions”; failed, unsupported, unavailable, incomplete and invalid outcomes remain distinct |
| README benchmark table | Numbers were manually embedded and could become stale | Only the section between `BEGIN ICORE-BENCHMARK` and `END ICORE-BENCHMARK` is regenerated with figures/CSVs |
| Family figure | Family-size label could differ from the eligible pair count | Shows actual HPIPM and QOCO pair counts; keeps the size-based selection rule |
| Scatter figure | Two panels contained cramped repeated text | One log-log axes with separate rival markers and a non-overlapping header |
| `docs/BENCHMARKING.md` | Plotting script rules and limitations were implicit | Explicit timing scope, eligibility, aggregation, exclusions and regeneration instructions |
| New `tests/test_benchmark_summary.py` | No regression coverage for saved-run aggregation | 32 reporting-only tests, including incomplete/failed/duplicate groups and README preservation |

The numerical runtime and benchmark runner have not been changed. The generation
script does not import or call a native solver. The original headline ratios did
not change after enforcing complete-group eligibility on this particular run.

## Recomputed N=20 outcomes

| Solver | Problems passing 20/20 | Failed/error | Explicitly unsupported |
|---|---:|---:|---:|
| ICORE C | 393 | 0 | 0 |
| ICORE C++ | 393 | 0 | 0 |
| HPIPM | 344 | 1 | 48 |
| QOCO | 360 | 33 | 0 |

No unavailable, incomplete, or invalid groups were found in these saved data.
A failure here describes the recorded adapter/settings/audit outcome; it does not
assert that the upstream solver cannot represent that problem type.

| Rival | Eligible pairs | ICORE C++ lower mean | Rival lower mean | Geometric mean of rival / ICORE C++ |
|---|---:|---:|---:|---:|
| HPIPM BALANCE | 344 | 287 | 57 | 1.4159841604600987 |
| QOCO | 360 | 358 | 2 | 3.010955030728148 |

These are ratios of arithmetic mean solve times for each eligible problem,
aggregated geometrically across problems. No confidence or statistical-significance
claim is attached to the faster/slower counts. The paired subsets differ by rival.
Recorded platform: macOS-26.1-arm64-arm-64bit. ICORE's recorded target: GENERIC.
See [the full summary](BENCHMARK_SUMMARY.md) for recorded versions and timing scope.

## Checks performed for this update

- 32 new saved-run reporting regression tests passed.
- 12 existing notebook/helper tests passed, including all 393 model/canonical constructions.
- 3 existing non-native tutorial checks passed: inline model/reference equality, markdown text, and the portable header's presence.
- Cross-checked all **1,572 problem/solver groups and 31,440 records** against the existing notebook `mean_table`: eligibility and eligible means agree.
- Regenerating the README block from the real N=20 run is idempotent.
- Total: **47 tests passed**. Native solve tests and the full native benchmark were not rerun.
- All 37 Python files parsed; all 42 code cells in the seven notebooks parsed.
- Local README/document/notebook links were checked for existing file targets; no missing target was found.
- The three figures were generated from the saved N=20 run and visually inspected. The summary renderer's text-overlap and containment checks passed.
- CSV output uses LF, without CRLF conversions being left to Git.

Byte-for-byte preservation against the uploaded archive:

- All **96 files under `icore/`** are unchanged, including native sources, Python API, dependency sources and build logic.
- All **9 files under `example_tools/`** are unchanged.
- All **393 NPZ instances**, their catalog and problem README are unchanged.
- `Compare.ipynb`, all six tutorial notebooks and their README are unchanged.
- The approved README hero image, existing methods HTML, license and third-party notices are unchanged.
- The README's license and GPT-disclosure sections remain at the bottom, with their text unchanged.

No files are deleted. The small update archive contains only modified/new reporting,
documentation and test files. It contains no `.git`, `runs/`, or native build cache.
The full source archive is an alternative for a fresh checkout, not a replacement
for the `.git` directory of an existing repository.

## Reproduce reporting checks

```bash
python -m unittest tests.test_benchmark_summary
python scripts/generate_benchmark_summary.py --runs-dir runs/all393_N20 --repo-root .
```

Keep the saved run directory locally. It is not included in this source update.
