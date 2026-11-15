# What the comparison notebook measures

`Compare.ipynb` uses the retained comparison adapters and independent original-model
accuracy checks. Only the notebook interface was simplified.

## Modes

Single-problem mode shows the mathematical model, then runs each requested solver N
times. All mode runs the entire 393-instance catalog with the same method list and N.
The default methods are ICORE C, ICORE C++, HPIPM BALANCE, and QOCO.

Results use **arithmetic means in milliseconds**, not medians. Native compilation is
completed outside timing where available; optional JAX per-shape compilation is
reported separately. Each repetition creates a fresh workspace and uses a cold solve.
No detailed tracing, plotting, or correctness audit is hidden in the solve timer.

| Metric | Included |
|---|---|
| `setup_ms` | Fresh workspace construction, excluding compilation |
| `solve_ms` | Cold solve and owned original-coordinate outputs |
| `total_ms` | `setup_ms + solve_ms` |

A shared modeling conversion is reported separately. These metrics are not full
trajectory-optimization application time. The tutorials' native time has a narrower
scope and must not be directly substituted for these times.

## Accuracy and fair failure handling

Every ranked solver/problem pair must pass all N/N checks: native successful status,
original-model primal error, dual error, and absolute cone complementarity at the
requested tolerance. Where a reference objective is available it is checked too.
A method with an incomplete or failed set of repetitions remains visible but is not
ranked. Means do not silently exclude slow/failed attempts.

The HPIPM QP adapter does not support generic SOC models. It reports unsupported,
not an approximate QP and not an artificial speed win for ICORE. Comparator failures
are configuration/test outcomes, not universal judgments of the upstream libraries.

By default the reference is the fastest valid mean among requested rivals.
Incomplete rival coverage is marked. The notebook separates faster, slower, tied,
and incomparable rows. Change ICORE backend, rival, or metric without solving again.

## Saved runs and resume

All mode saves records after each completed problem in `runs/`. `RESUME=True` checks
that the model, requested repetitions, environment, and loaded implementation match.
An interrupted problem is rerun as a whole; partial repeats are not silently merged.
The identifiers that enforce this are internal, not user-facing settings.

A mean below a rival's mean is descriptive. For performance claims, use enough
repetitions, multiple sessions, matched accuracy, controlled CPU load, and a common
measurement scope. This source cleanup is not a new speedup claim.

## Publishing figures and a README snapshot

From the repository root, select one run explicitly:

```bash
python scripts/generate_benchmark_summary.py --runs-dir runs/all393_N20 --repo-root .
```

This command reads saved JSON; it does not execute a solver. It updates the three
figures, summary/family/coverage/outcome CSV files, `docs/BENCHMARK_SUMMARY.md`, and
only the README section between `BEGIN ICORE-BENCHMARK` and `END ICORE-BENCHMARK`.
Other README text, including the position of the license and GPT disclosure,
is preserved. Use `--no-update-readme` to generate reports without editing README.

The generator checks the configured problem list, repetitions, methods, tolerance,
input identities, completion record, and environment consistency when these records
are available. If a parent contains several runs (for example N=5 and N=20), it
stops and asks you to select one instead of choosing silently. Do not combine
measurements from different machines or builds in one saved-run directory.

### What enters a speed comparison?

A problem/solver group is eligible only when it contains exactly N unique IDs
1 through N, every record is accepted with outcome `passed`, any recorded
status/KKT/objective decision is true, and every `solve_ms` is finite and positive.
The script uses the recorded independent-audit decisions; it does not reconstruct
solution arrays or rerun the KKT audit.

The arithmetic mean includes all N repetitions. Failed attempts are not filtered
out to create a faster mean. A complete failed group may retain its mean in the
outcome CSV for inspection, but never enters ratios or faster/slower counts.

- **Passed all**: the complete group satisfies the rule above.
- **Failed/error**: at least one attempt failed the recorded audit/status or raised an error.
- **Unsupported**: every attempt explicitly reports an unsupported adapter case.
- **Unavailable**: every attempt reports an unavailable solver/build.
- **Incomplete**: missing, duplicated or invalid repetition IDs.
- **Invalid**: contradictory skipped records, or invalid timings in a passed group.
- **Not requested**: this method was not selected in the saved run.

These categories partition the displayed catalog for each default solver. A failed
accuracy check does not mean that the upstream library cannot represent the problem.

### Ratios and figure interpretation

For each eligible pair, divide the rival's arithmetic mean by ICORE C++'s mean.
The overall and family summaries are geometric means of those per-problem ratios,
with one equal weight per eligible instance. This is not a ratio of summed times.
Ratios above 1 favor ICORE; ratios below 1 favor the rival. Pair sets differ between
rivals, so their aggregates do not directly rank HPIPM against QOCO.

The family chart retains at most the 14 largest families containing at least five
instances, selected by descending size and then name (not by speed). Its labels
show actual pair counts for each rival; all families are present in the CSV.
The log scatter uses circles for HPIPM and triangles for QOCO; each marker is one
eligible problem/rival pair, with both axes in milliseconds.

The summary uses renderer-measured text spacing and checks label intersections.
CSV files use LF line endings. No figure/table is a statistical-significance
claim, and the recorded target/compiler/environment should accompany published
results. Run the reporting-only tests with:

```bash
python -m unittest tests.test_benchmark_summary
```
