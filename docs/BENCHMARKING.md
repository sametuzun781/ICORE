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
