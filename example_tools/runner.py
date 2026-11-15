"""Cold-solve comparisons with explicit arithmetic means and failure accounting.

No diagnostic replay, printing, model rendering, correctness check, or JIT/native
compilation is included in solve_ms. The numerical package is not modified.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import importlib.util
import hashlib
import os
import json
from numbers import Integral
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from icore import Solver, __version__
from icore._build import build, build_info
from icore._comparators import build_comparators
from icore.vendor.hpipm.adapter import HPIPMDense, HPIPMOCP, UnsupportedProblem
from icore.vendor.qoco.api import NativeQoco
from .catalog import Example, PROBLEMS, list_problems, load_problem

METHODS = ("ICORE C", "ICORE C++", "HPIPM", "QOCO")
OPTIONAL_METHODS = ("ICORE JAX", "HPIPM ROBUST")
METRICS = ("solve_ms", "setup_ms", "total_ms")


def clean(value):
    """JSON with null, rather than nonstandard NaN/Infinity tokens."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def prepare(methods: Sequence[str] = METHODS) -> dict:
    """Compile requested libraries once; report unavailable methods explicitly."""
    methods = _validate_methods(methods)
    libraries, errors, rows = {}, {}, []
    for method, backend in (("ICORE C", "c"), ("ICORE C++", "cpp")):
        if method not in methods:
            continue
        try:
            libraries[method] = build(backend)
            info = build_info(backend, library=libraries[method])
            rows.append(dict(solver=method, status="ready", version=__version__,
                             compiler=info["compiler"], target=info["target"],
                             precision="float64; optional guarded mixed OCP kernels"))
        except Exception as exc:
            errors[method] = f"{type(exc).__name__}: {exc}"
    rivals = [m for m in methods if m.startswith("HPIPM") or m == "QOCO"]
    if rivals:
        try:
            built = build_comparators()
            for method in rivals:
                libraries[method] = str(built["qoco" if method == "QOCO" else "hpipm"])
                rows.append(dict(solver=method, status="ready", version="0.3.2" if method == "QOCO" else method,
                                 compiler="see .build receipt", target="sparse LDL" if method == "QOCO" else "BLASFEO",
                                 precision="float64"))
        except Exception as exc:
            for method in rivals:
                errors[method] = f"{type(exc).__name__}: {exc}"
    if "ICORE JAX" in methods:
        if importlib.util.find_spec("jax") is None:
            errors["ICORE JAX"] = "JAX is not installed; install the optional dependency or omit this method."
        else:
            import jax
            jax.config.update("jax_enable_x64", True)
            rows.append(dict(solver="ICORE JAX", status="ready", version=jax.__version__,
                             compiler="XLA; per-shape compilation measured separately",
                             target=str(jax.devices()), precision="float64"))
    rows.extend(dict(solver=m, status="unavailable", error=error) for m, error in errors.items())
    return dict(methods=list(methods), libraries=libraries, errors=errors, solvers=rows,
                runtime_identity=_runtime_identity(libraries),
                platform=platform.platform(), machine=platform.machine(), python=sys.version.split()[0],
                timing_scope="fresh workspace; cold solve plus owned original-coordinate outputs; compilation/conversion/checks excluded")



def _runtime_identity(libraries):
    """Untimed identity check prevents merging checkpoints from changed builds."""
    import scipy
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return dict(helper_sources=digest.hexdigest(),
                native_libraries={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for name, path in libraries.items()},
                python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                threads={key: os.environ.get(key) for key in
                         ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})


def _validate_methods(methods):
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Choose a nonempty list of distinct solver names")
    unknown = set(methods) - set(METHODS + OPTIONAL_METHODS)
    if unknown:
        raise ValueError(f"Unknown solvers: {sorted(unknown)}")
    return methods


def _settings(repetitions, tolerance):
    if isinstance(repetitions, bool) or not isinstance(repetitions, Integral) or repetitions < 1:
        raise ValueError("repetitions/N must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")


class _Workspace:
    """One fresh solver, with exact existing OCP/chain coordinate adapters."""
    def __init__(self, method: str, problem: Example, tolerance: float, prepared: dict):
        self.method, self.problem, self.solver = method, problem, None
        self.compile_ms = 0.0
        if method.startswith("ICORE"):
            backend = {"ICORE C": "c", "ICORE C++": "cpp", "ICORE JAX": "jax"}[method]
            self.solver = Solver(problem.coefficients, backend=backend, tolerance=tolerance,
                                 max_iterations=200, **problem.options,
                                 **problem.metadata.get("solver_settings", {}))
            self.compile_ms = 1000 * self.solver.compile_seconds
        elif method.startswith("HPIPM"):
            if problem.metadata["soc_sizes"]:
                raise UnsupportedProblem("Generic SOC constraints are outside this HPIPM QP adapter; not approximated as a QP.")
            model, _ = problem.hpipm_model
            options = dict(profile="ROBUST" if method.endswith("ROBUST") else "BALANCE",
                           tolerance=tolerance, max_iter=200, complementarity_budget="component")
            self.solver = HPIPMDense(model, **options) if problem.dense else HPIPMOCP(model, **options)
        else:
            self.solver = NativeQoco(problem.canonical, tolerance=tolerance, max_iterations=200,
                                     library=prepared["libraries"][method])

    def solve(self):
        raw = self.solver.solve()
        if self.method == "QOCO":
            result = self.problem.restore_qoco(raw)
        elif self.method.startswith("HPIPM") and self.problem.chain:
            result = {**raw, **self.problem.chain_lift.decode(raw)}
        else:
            result = raw
        if self.method == "ICORE JAX":
            # The solver synchronizes device work; also own host outputs like native methods.
            def host(value):
                if isinstance(value, dict):
                    return {k: host(v) for k, v in value.items()}
                if isinstance(value, (tuple, list)):
                    return tuple(host(v) for v in value)
                return np.asarray(value).copy() if hasattr(value, "shape") else value
            result = host(result)
        return result, raw

    def close(self):
        if self.solver is not None:
            self.solver.close()


def _conversion(problem, methods):
    """Shared display/audit and comparator formulation preparation, once per model."""
    timings = {}
    start = perf_counter()
    _ = problem.canonical
    timings["canonical_model_ms"] = 1000 * (perf_counter() - start)
    if any(m.startswith("HPIPM") for m in methods) and not problem.metadata["soc_sizes"]:
        start = perf_counter()
        _ = problem.hpipm_model
        timings["hpipm_model_ms"] = 1000 * (perf_counter() - start)
    return timings


def run_problem(problem: Example | str, *, repetitions: int = 3, tolerance: float = 1e-8,
                methods: Sequence[str] = METHODS, prepared: dict | None = None,
                keep_solutions: bool = True, seed: int = 20261029) -> dict:
    """Exactly N timed attempts for each requested supported/available solver.

    No hidden warmup, automatic retry, or diagnostic solve. Unavailable and
    unsupported methods have N explicit skipped records, with no fabricated time.
    A new workspace is created for each repetition; order is seeded/rotated.
    """
    _settings(repetitions, tolerance)
    methods = _validate_methods(methods)
    problem = load_problem(problem) if isinstance(problem, str) else problem
    prepared = prepare(methods) if prepared is None else prepared
    conversion_ms = _conversion(problem, methods)
    rows, latest = [], {}
    rng = np.random.default_rng(seed)
    base_order = list(rng.permutation(len(methods)))
    for repetition in range(repetitions):
        indices = base_order[repetition % len(methods):] + base_order[:repetition % len(methods)]
        if (repetition // len(methods)) % 2:
            indices = indices[::-1]
        for position, index in enumerate(indices):
            method = methods[index]
            row = dict(problem=problem.name, cohort=problem.metadata["cohort"], family=problem.metadata["family"],
                       solver=method, repetition=repetition + 1, order=position + 1, setup_ms=None,
                       compile_ms=None, solve_ms=None, total_ms=None, status=None, iterations=None,
                       refinements=None, factor_retries=None, accepted=False, outcome="error", error="")
            workspace = None
            try:
                if method in prepared["errors"]:
                    row.update(outcome="unavailable", error=prepared["errors"][method])
                    continue
                if method.startswith("HPIPM") and problem.metadata["soc_sizes"]:
                    row.update(outcome="unsupported", error="HPIPM QP adapter does not support generic SOCs")
                    continue
                start = perf_counter()
                workspace = _Workspace(method, problem, tolerance, prepared)
                constructed_ms = 1000 * (perf_counter() - start)
                row["compile_ms"] = workspace.compile_ms
                row["setup_ms"] = max(0.0, constructed_ms - workspace.compile_ms)
                start = perf_counter()
                result, raw = workspace.solve()
                row["solve_ms"] = 1000 * (perf_counter() - start)
                row["total_ms"] = row["setup_ms"] + row["solve_ms"]
                row["status"] = int(np.asarray(result["status"]))
                row["iterations"] = int(np.asarray(result.get("iterations", -1)))
                row["refinements"] = int(np.asarray(result.get("refinements", -1)))
                row["factor_retries"] = int(np.asarray(result.get("factor_retries", -1)))
                metrics = problem.audit(result, method, tolerance=tolerance,
                                        canonical_raw=raw if method == "QOCO" else None)
                row.update({k: metrics.get(k) for k in ("primal", "dual", "gap", "objective", "objective_error",
                           "reference_available", "status_ok", "kkt_ok", "objective_ok", "accepted")})
                row["outcome"] = "passed" if metrics["accepted"] else "failed_accuracy"
                if method.startswith("ICORE"):
                    row["plan"] = workspace.solver.factorization_plan
                    row["kernel_diagnostics"] = workspace.solver.diagnostics
                if method.startswith("HPIPM"):
                    row["native_diagnostics"] = raw.get("native", {})
                    row["settings"] = raw.get("settings", {})
                if keep_solutions:
                    latest[method] = result
            except UnsupportedProblem as exc:
                row.update(outcome="unsupported", error=str(exc))
            except Exception as exc:
                row.update(outcome="error", error=f"{type(exc).__name__}: {exc}")
            finally:
                if workspace is not None:
                    try:
                        workspace.close()
                    except Exception as exc:
                        row.update(accepted=False, outcome="error", error=row["error"] + f"; close: {exc}")
                rows.append(clean(row))
    return dict(problem=problem.metadata, methods=list(methods), repetitions=int(repetitions),
                tolerance=tolerance, rows=rows, latest_solutions=latest,
                conversion_ms=conversion_ms, environment=prepared)


def mean_table(rows: list[dict], *, repetitions: int) -> pd.DataFrame:
    """Arithmetic means use all N measured attempts, not only successful samples.

    Incomplete/failed groups retain raw means but are excluded from rankings.
    """
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    records = []
    for (name, method), group in frame.groupby(["problem", "solver"], sort=False):
        record = dict(problem=name, solver=method, records=len(group),
                      passed=int(group.accepted.sum()), expected=int(repetitions))
        measured = pd.to_numeric(group.solve_ms, errors="coerce")
        unique_repeats = set(group.repetition) == set(range(1, repetitions + 1))
        record["measured"] = int(measured.notna().sum())
        record["eligible"] = bool(len(group) == repetitions and unique_repeats
                                  and group.accepted.all() and measured.notna().all())
        for field in ("setup_ms", "solve_ms", "total_ms", "compile_ms"):
            values = pd.to_numeric(group[field], errors="coerce")
            record[field.replace("_ms", "_mean_ms")] = float(values.mean()) if values.notna().any() else np.nan
        record["solve_std_ms"] = float(measured.std(ddof=1)) if measured.notna().sum() > 1 else np.nan
        record["outcomes"] = ", ".join(f"{k}:{v}" for k, v in Counter(group.outcome).items())
        errors = [str(x) for x in group.error if x]
        record["error"] = " | ".join(dict.fromkeys(errors))
        records.append(record)
    return pd.DataFrame(records)


def comparison_tables(rows: list[dict], *, repetitions: int, icore_solver: str = "ICORE C++",
                      rivals: Sequence[str] = ("HPIPM", "QOCO"), metric: str = "solve_ms",
                      tie_percent: float = 0.0) -> dict[str, pd.DataFrame]:
    """Split by arithmetic-mean ratio to the fastest VALID requested rival.

    Counts are descriptive, not significance tests. Missing rivals are disclosed.
    Failed ICORE/rival groups cannot create a false performance win.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if not np.isfinite(tie_percent) or tie_percent < 0:
        raise ValueError("tie_percent must be finite and nonnegative")
    rivals = tuple(rivals)
    if not rivals or icore_solver in rivals:
        raise ValueError("Choose at least one distinct rival")
    summary = mean_table(rows, repetitions=repetitions)
    groups = {key: [] for key in ("faster", "slower", "tie", "incomparable")}
    if summary.empty:
        return {**{key: pd.DataFrame() for key in groups}, "summary": summary}
    field = metric.replace("_ms", "_mean_ms")
    for name, group in summary.groupby("problem", sort=False):
        by_method = {r["solver"]: r for r in group.to_dict("records")}
        row = {"problem": name, "ICORE solver": icore_solver}
        for method in dict.fromkeys([*by_method, icore_solver, *rivals]):
            r = by_method.get(method)
            row[method + " mean ms"] = r[field] if r else np.nan
            row[method + " pass"] = f"{r['passed']}/{r['expected']}" if r else "not requested"
        valid = [m for m in rivals if by_method.get(m, {}).get("eligible", False)
                 and np.isfinite(by_method[m][field]) and by_method[m][field] > 0]
        row["valid rivals"] = ", ".join(valid)
        row["coverage"] = "all requested rivals valid" if len(valid) == len(rivals) else "partial / unavailable / failed rival"
        reference = by_method.get(icore_solver)
        if not reference or not reference["eligible"] or not valid:
            row["reason"] = "ICORE group failed/incomplete" if not reference or not reference["eligible"] else "No valid rival"
            groups["incomparable"].append(row)
            continue
        best = min(valid, key=lambda m: by_method[m][field])
        ratio = reference[field] / by_method[best][field]
        row.update({"fastest valid rival": best, "ICORE / rival": ratio,
                    "ICORE change %": 100 * (ratio - 1)})
        key = "tie" if abs(100 * (ratio - 1)) <= tie_percent else "faster" if ratio < 1 else "slower"
        groups[key].append(row)
    return {**{key: pd.DataFrame(value) for key, value in groups.items()}, "summary": summary}


def run_all(*, repetitions: int = 3, methods: Sequence[str] = METHODS, tolerance: float = 1e-8,
            names: Sequence[str] | None = None, directory: str | Path = PROBLEMS,
            output_dir: str | Path | None = None, resume: bool = False,
            prepared: dict | None = None, progress: Callable | None = None) -> dict:
    """Run the whole catalog with a per-problem checkpoint; retain every failure."""
    _settings(repetitions, tolerance)
    methods = _validate_methods(methods)
    entries = list_problems(directory)
    catalog = {r["name"]: r for r in entries}
    names = list(catalog) if names is None else list(names)
    if len(set(names)) != len(names) or set(names) - set(catalog):
        raise ValueError("names must contain distinct manifest problem names")
    prepared = prepare(methods) if prepared is None else prepared
    out = Path(output_dir) if output_dir else Path.cwd() / "runs" / datetime.now().strftime("all-%Y%m%d-%H%M%S-%f")
    out.mkdir(parents=True, exist_ok=True)
    config = dict(schema="icore-example-run-v1", names=names, methods=list(methods), repetitions=int(repetitions),
                  tolerance=float(tolerance), engine_version=__version__,
                  inputs={name: catalog[name]["coefficient_identity"] for name in names},
                  machine=prepared["machine"], platform=prepared["platform"],
                  runtime_identity=prepared.get("runtime_identity", {}))
    configuration = out / "configuration.json"
    if configuration.exists():
        if not resume:
            raise FileExistsError(f"Run exists: {out}; use a new directory or resume=True")
        if json.loads(configuration.read_text()) != config:
            raise ValueError("Resume settings, build/runtime, machine, solver list, or coefficients differ")
    else:
        configuration.write_text(json.dumps(config, indent=2))
        (out / "environment.json").write_text(json.dumps(clean(prepared), indent=2))
    rows, completed, interrupted = [], [], False
    try:
        for index, name in enumerate(names):
            record_file = out / f"problem-{index:04d}.json"
            if resume and record_file.exists():
                item = json.loads(record_file.read_text())
                if item["problem"]["name"] != name:
                    raise ValueError("Checkpoint problem order differs")
            else:
                problem = load_problem(name, directory)
                try:
                    item = run_problem(problem, repetitions=repetitions, tolerance=tolerance,
                                       methods=methods, prepared=prepared, keep_solutions=False,
                                       seed=20261029 + index)
                except Exception as exc:
                    # Preserve model/conversion-level failures too, with no fake time.
                    item = dict(problem=problem.metadata, rows=[dict(problem=name, solver=m,
                        repetition=r+1, accepted=False, outcome="error", error=f"model preparation: {type(exc).__name__}: {exc}",
                        setup_ms=None, solve_ms=None, total_ms=None, compile_ms=None)
                        for r in range(repetitions) for m in methods])
                item.pop("latest_solutions", None)
                item.pop("environment", None)
                temp = record_file.with_suffix(".tmp")
                temp.write_text(json.dumps(clean(item), indent=2))
                temp.replace(record_file)
            rows.extend(item["rows"])
            completed.append(name)
            state = dict(complete=len(completed) == len(names), completed=len(completed), total=len(names), last=name)
            (out / "progress.json").write_text(json.dumps(state, indent=2))
            if progress:
                progress(state)
    except KeyboardInterrupt:
        interrupted = True
    summary = mean_table(rows, repetitions=repetitions)
    (out / "means.json").write_text(summary.to_json(orient="records", indent=2))
    result = dict(rows=rows, repetitions=int(repetitions), methods=list(methods),
                  completed=len(completed), total=len(names), complete=len(completed) == len(names),
                  interrupted=interrupted, output_dir=str(out), environment=prepared)
    return result
