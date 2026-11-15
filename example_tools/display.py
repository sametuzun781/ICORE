"""Notebook presentation: formulation first, observed execution second.

Rendering and diagnostic replay never run inside a timed solve. Plans, production
counters, and the separately instrumented replay have different evidence labels.
"""
from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML, Markdown, Math, display

from .catalog import Example, load_problem, read_catalog
from .runner import comparison_tables, mean_table, clean

BODIES = {0: "General OCP / Riccati kernel", 1: "Dense-QP kernel", 2: "Adjacent-stage affine-chain kernel"}
COHORT_NAMES = {
    "main175": "Main benchmark set",
    "extra72": "Additional validation models",
    "matrix_updates32": "Distinct matrix-update states",
    "analytic_kkt": "Analytic KKT examples",
}


def _table(frame: pd.DataFrame):
    """Display every row without changing global pandas options."""
    if frame.empty:
        display(Markdown("No problems in this group."))
        return
    display(HTML('<div style="overflow:auto;max-height:620px">' +
                 frame.to_html(index=False, escape=True, float_format=lambda x: f"{x:.6f}", na_rep="—") + "</div>"))


def show_catalog():
    manifest = read_catalog()
    display(Markdown(f"### {manifest['count']} distinct coefficient instances"))
    _table(pd.DataFrame([{"Group": COHORT_NAMES.get(k, k), "Problem": v}
                         for k, v in manifest["unique_by_primary_cohort"].items()]))
    display(Markdown("Repeated backend runs and aliases of identical coefficients are not counted as new problems. "
                     "Matrix updates, analytic tests, and equivalent formulations are labeled separately; "
                     "this is not a count of independent application families."))


def formulation(problem: Example | str) -> dict:
    """Return the mathematical model that the adapters actually represent."""
    p = load_problem(problem) if isinstance(problem, str) else problem
    m, a = p.metadata, p.coefficients
    if p.chain:
        equations = [r"\begin{aligned}\underset{\{y_k\}_{k=0}^{T},\{t_k\}_{k=0}^{T-1}}{\operatorname{minimize}}\quad"
            r"&\sum_{k=0}^{T}\left[\frac12(y_k-\bar y_k)^\top D_k(y_k-\bar y_k)+c_k^\top y_k\right]"
            r"+\sum_{k=0}^{T-1}w_k^\top t_k\\"
            r"\text{subject to}\quad &-t_k\le L_k y_k+R_k y_{k+1}+d_k\le t_k,&& k=0,\ldots,T-1,\\"
            r"&E_k y_k+F_k y_{k+1}+e_k\le0,&& k=0,\ldots,T-1,\\"
            r"&C_k y_k+g_k\le0,&& k=0,\ldots,T,\\"
            r"&(y_k)_j=(y_k^{\mathrm{fixed}})_j,&&\text{where free}_{k,j}=0.\end{aligned}"]
        notes = ("This is an adjacent-stage OCP/chain model. The first two-sided inequality is an L1 epigraph; "
                 "it is not a hard dynamics equality. Native ICORE solves this original chain. "
                 "HPIPM uses an equivalent one-step-memory OCP lift; QOCO uses a sparse residual/epigraph form. "
                 "The lift below is a comparator formulation, not the selected native ICORE kernel.")
        equations += [r"\xi_{k+1}=y_k,\qquad v_k=(y_k,t_{k-1}),\qquad"
                      r"-t_{k-1}\le L_{k-1}\xi_k+R_{k-1}y_k+d_{k-1}\le t_{k-1}."]
        symbols = {"T": int(a["left"].shape[0]), "Dimension of y_k": int(a["center"].shape[1]),
                   "Free y coordinates": int(np.count_nonzero(a["free"])),
                   "L1/epigraph coordinates": int(a["offset"].size),
                   "D_k": "diag(hessian[k]); positive diagonal", "w_k": "l1_weight (broadcast)",
                   "L/R/d": "left / right / offset", "E/F/e": "pair_left / pair_right / pair_offset",
                   "C/g": "local_matrix / local_offset", "ȳ/c": "center / linear"}
        kind = "Adjacent-stage quadratic/L1 OCP"
    elif p.dense:
        model = p.canonical
        subject = r"Gv\le h"
        if model.A.shape[0]:
            subject = r"Av=b,\qquad " + subject
        equations = [r"\begin{aligned}\underset{v\in\mathbb R^n}{\operatorname{minimize}}\quad"
                     r"&\frac12v^\top Pv+c^\top v+\kappa\\\text{subject to}\quad&" + subject + r".\end{aligned}"]
        notes = ("This is a single-stage dense QP (a degenerate OCP) without dynamic coupling. "
                 "The displayed problem is the equivalent model after eliminating fixed-state contributions; "
                 "no artificial dynamics constraint is added. The objective constant κ is retained in the accuracy audit.")
        symbols = {"n": model.P.shape[0], "Equalities": model.A.shape[0], "Inequalities": model.G.shape[0],
                   "κ": float(model.constant), "P/c/G/h": "dense_* coefficients in problems/<name>.npz"}
        kind = "Single-stage dense QP"
    else:
        has_terminal = np.asarray(a.get("F", np.empty((0, 0)))).size > 0
        terminal = r"\\&F x_N=f" if has_terminal else ""
        equations = [r"\begin{aligned}\underset{\{x_k\}_{k=0}^{N},\{u_k\}_{k=0}^{N-1}}{\operatorname{minimize}}\quad"
            r"&\sum_{k=0}^{N-1}\left[\frac12\begin{bmatrix}u_k\\x_k\end{bmatrix}^{\!\top}"
            r"\begin{bmatrix}R_k&S_k\\S_k^\top&Q_k\end{bmatrix}\begin{bmatrix}u_k\\x_k\end{bmatrix}"
            r"+r_k^\top u_k+q_k^\top x_k\right]+\frac12x_N^\top Q_Nx_N+q_N^\top x_N\\"
            r"\text{subject to}\quad&x_0=\bar x_0,\\"
            r"&x_{k+1}=A_kx_k+B_ku_k+b_k,&&k=0,\ldots,N-1,\\"
            r"&h_k-G_k z_k\in\mathcal K_k,&&k=0,\ldots,N" + terminal + r".\end{aligned}",
            r"z_k=(u_k,x_k),\quad z_N=(0,x_N),\qquad"
            r"\mathcal K_k=\mathbb R_+^{\ell_k}\times\prod_j\mathcal Q^{d_{kj}},\quad"
            r"\mathcal Q^d=\{(s_0,\bar s):s_0\ge\|\bar s\|_2\}."]
        notes = ("The symbols are the selected input blocks Q/R/S/q/r/A/B/b/G/h/x0/F/f; "
                 "at stages without SOCs the cone condition reduces to G_k z_k ≤ h_k. For QOCO the same model "
                 "is assembled in sparse conic form, x₀ is substituted, and results are restored to original coordinates.")
        symbols = {"N": m["N"], "n_x": m["nx"], "n_u": m["nu"],
                   "Scalar inequality/conic rows": m["n_inequalities"], "Equalities": m["n_equalities"],
                   "Terminal F rows": int(np.asarray(a.get("f", [])).size),
                   "SOC blocks": m["soc_sizes"] or "none", "Orthant rows (per stage)": m["nonnegative"]}
        kind = "Stage-structured conic OCP" if m["soc_sizes"] else "Stage-structured quadratic OCP"
    return dict(name=p.name, kind=kind, equations=equations, notes=notes, symbols=symbols,
                cohort=m["cohort"], reference_available=m.get("reference_objective") is not None)


def show_problem(problem: Example | str):
    f = formulation(problem)
    display(Markdown(f"## Problem being solved: `{f['name']}`\n**{f['kind']}**"))
    for equation in f["equations"]:
        display(Math(equation))
    display(Markdown(f["notes"]))
    def compact(v):
        if isinstance(v, list):
            text = json.dumps(v)
            return text if len(text) <= 130 else f"{len(v)} stages; distribution: " + str(Counter(map(str, v)))
        return str(v)
    _table(pd.DataFrame([{"Symbol / property": k, "Value": compact(v)} for k, v in f["symbols"].items()]))
    display(Markdown(f"**Cohort:** {COHORT_NAMES.get(f['cohort'], f['cohort'])}. "
                     + ("A reference objective value is available." if f["reference_available"] else
                        "No independent reference objective is available; the common original-model KKT audit is applied.")))
    return f


def plan_problem(problem: Example | str, *, backend="cpp", tolerance=1e-8):
    """An untimed setup only: it does not add a solve to the N repetitions."""
    from icore import Solver
    p = load_problem(problem) if isinstance(problem, str) else problem
    with Solver(p.coefficients, backend=backend, tolerance=tolerance, max_iterations=200,
                **p.options, **p.metadata.get("solver_settings", {})) as solver:
        result = dict(plan=solver.factorization_plan, presolve_witness=solver._presolve_witness)
    display(Markdown("### Plan after compilation/setup — not a record of executed factorization"))
    _table(pd.DataFrame([{"Plan": k, "Value": str(v)} for k, v in result["plan"].items()]))
    if result["presolve_witness"]:
        display(Markdown("**Exact-chain presolve rejection witness:** `" + str(result["presolve_witness"]) +
                         "`. This original-coefficient location violates the zero-A / diagonal-H certificate."))
    return result


def body_reason(problem: Example, plan: dict, diagnostics: dict) -> str:
    body = diagnostics.get("last_kernel")
    if body == 1:
        return ("Native dense-QP certificate accepted: one transition, no effective second-stage objective/constraint; "
                "no SOC, terminal equality, or special epigraph. Fixed x₀ contributions were eliminated.")
    if body == 2:
        return ("The input explicitly uses the adjacent-stage affine-chain layout." if problem.chain else
                "The OCP was converted to an equivalent affine chain using an exact structural presolve certificate.")
    if diagnostics.get("dense_fallbacks", 0):
        return "The dense attempt was numerically rejected; execution restarted in the general OCP kernel (native counter)."
    return "General stage-structured OCP route. Executed rejection checkpoints for specialized routes appear in the diagnostic replay below."


def trace_tables(trace: dict) -> dict[str, pd.DataFrame]:
    grouped = trace["grouped"]
    factors, rejection, changes, values = [], [], [], []
    for entry in grouped:
        location = f"{entry['file']}:{entry['line']}"
        if entry["kind"] == "enter" and entry.get("category") in ("factorization", "policy", "recovery"):
            returns = [r for r in grouped if r["function"] == entry["function"] and r["file"] == entry["file"]
                       and r["kind"] == "return" and r["phase"] == entry["phase"]]
            factors.append({"Function": entry["function"], "Operation": entry["label"], "Phase": entry["phase"],
                "Calls": entry["count"], "Successful returns": sum(r["count"] for r in returns if r["success"]),
                "Rejected / unsuccessful returns": sum(r["count"] for r in returns if not r["success"])})
        if entry["kind"] == "return" and not entry["success"]:
            rejection.append({"Route": entry["function"], "Phase": entry["phase"], "Count": entry["count"],
                              "Return": entry["value"], "Source": location,
                              "Executed return checkpoint": entry["source_context"]})
        if entry["kind"] in ("fallback", "recovery", "rejection", "precision", "accepted"):
            changes.append({"Event": entry["label"], "Phase": entry["phase"], "Count": entry["count"], "Source": location})
        if entry["kind"] == "guard_value":
            values.append({"Route": entry["function"], "Guard input": entry["expression"], "Value": entry["value"],
                           "Count": entry["count"], "Phase": entry["phase"]})
    return {"factors": pd.DataFrame(factors), "rejections": pd.DataFrame(rejection),
            "changes": pd.DataFrame(changes), "guard_values": pd.DataFrame(values)}


def show_decisions(problem: Example | str, benchmark: dict, *, backend="cpp", trace: dict | None = None):
    p = load_problem(problem) if isinstance(problem, str) else problem
    method = "ICORE C++" if backend == "cpp" else "ICORE C"
    rows = [r for r in benchmark["rows"] if r["solver"] == method and "kernel_diagnostics" in r]
    if not rows:
        display(Markdown(f"**{method}:** no completed native diagnostic record."))
        return
    row = rows[-1]
    d, plan = row["kernel_diagnostics"], row["plan"]
    actual = BODIES.get(d.get("last_kernel"), f"Unknown code: {d.get('last_kernel')}")
    records = [
        {"Question": "Which kernel was selected?", "Answer": actual, "Evidence": f"Production solve #{row['repetition']}: last_kernel"},
        {"Question": "Which structural property justified it?", "Answer": body_reason(p, plan, d), "Evidence": "Input + native certificate/kernel counter"},
        {"Question": "Which factorization was attempted?", "Answer": "The observed call/return list below" if trace else
            "Production counters are shown below; they do not record every attempt in order. Enable diagnostic replay for details.", "Evidence": "Separate replay" if trace else "A plan is not treated as an executed operation"},
        {"Question": "Which fallback actually occurred?", "Answer": f"dense_fallbacks={d.get('dense_fallbacks', 0)}; "
            f"factor_retries={row.get('factor_retries')}; refinements={row.get('refinements')}", "Evidence": "Counters from the timed production solve"},
        {"Question": "Which arithmetic precision was used?", "Answer": "Outer variables/residuals use FP64. Inner factor/sweep precision is reported by replay events." if trace else
            "Outer variables/residuals use FP64; the production build does not expose every internal mixed-precision event.", "Evidence": "Interface dtype + observed acceptance events, when available"},
        {"Question": "Why was a fast path rejected?", "Answer": "Executed return locations and observed guard values are listed below." if trace else
            "Standard counters do not contain this detail; diagnostic replay is disabled.", "Evidence": "Separate replay; no inferred rejection reason"},
    ]
    display(Markdown("### Observed execution route"))
    _table(pd.DataFrame(records))
    _table(pd.DataFrame([{"Native counter": k, "Value": v} for k, v in d.items()]))
    if trace is None:
        return
    tables = trace_tables(trace)
    display(Markdown("### Separate, untimed diagnostic replay\n"
        "This observation uses **one additional solve**, excluded from the N repetitions and their mean. "
        "No probes were added to the timed library. Replay output bitwise matches production: **"
        + str(trace["bitwise_matches_production"]) + "**. "
        "The replay is a separate execution of the same input, not a complete trace of the timed call."))
    if not trace["bitwise_matches_production"] or trace["dropped_events"] or not trace["audit"]["accepted"]:
        display(Markdown("**Warning:** replay equality, accuracy, or event-integrity checks did not all pass. "
                         "Do not use this trace as confirmation of the production run."))
    precision = [e for e in trace["grouped"] if e["kind"] == "precision"]
    attempted_fp32 = any(e["kind"] == "enter" and e["function"].startswith(("icore_ocp_la_mixed_begin", "icore_ocp_la_mixed_stage", "icore_ocp_la_mixed_sweeps_"))
                         for e in trace["grouped"])
    display(Markdown("**Replay precision:** " +
        ("; ".join(dict.fromkeys(e["label"] for e in precision)) if precision else
         "FP64 acceptance route; an FP32 factor attempt was observed, but no FP32 acceptance event." if attempted_fp32 else
         "No FP32 attempt/acceptance on the observed factor/sweep routes; FP64.")))
    display(Markdown("**Factorization/dispatch attempts** — a policy call alone does not prove that a factorization ran."))
    _table(tables["factors"])
    display(Markdown("**Acceptance, fallback, and recovery events** — an eligibility rejection is not a numerical failure."))
    _table(tables["changes"])
    display(Markdown("**Rejected routes and executed return locations**"))
    _table(tables["rejections"])
    display(Markdown("**Observed integer/boolean inputs to rejection guards**"))
    _table(tables["guard_values"])
    display(Markdown("The trace covers selected dispatch, factorization, recovery, and precision checkpoints; "
                     "it is not a complete trace of every `if` or SIMD branch. Values of unrecorded operands in compound OR guards "
                     "are not inferred. The full chronological event list is in `trace['events']`."))


def show_run(benchmark: dict):
    display(Markdown(f"### {benchmark['repetitions']} repetitions — arithmetic means (ms)"))
    _table(mean_table(benchmark["rows"], repetitions=benchmark["repetitions"]))
    display(Markdown("`solve_mean_ms`: cold solve plus independently owned original-coordinate outputs. "
                     "`setup_mean_ms`: workspace creation; `total_mean_ms`: setup + solve. "
                     "Model conversion, accuracy audit, display, and compilation are excluded. "
                     "When JAX is selected, compilation is separately reported in `compile_mean_ms`."))
    display(Markdown("**One-time model conversion (separate):** `" + str(benchmark["conversion_ms"]) + "`"))


def show_comparisons(rows, *, repetitions, icore_solver="ICORE C++", rivals=("HPIPM", "QOCO"),
                     metric="solve_ms", tie_percent=0.0):
    tables = comparison_tables(rows, repetitions=repetitions, icore_solver=icore_solver, rivals=rivals,
                               metric=metric, tie_percent=tie_percent)
    display(Markdown(f"## All mode: {icore_solver} / {', '.join(rivals)}\n"
        f"**N={repetitions}; metric={metric}; arithmetic mean; time unit ms.** "
        "Each problem is compared with the fastest requested rival that passes N/N accuracy checks. "
        "A row with `coverage=partial` is not a win over an unavailable or unsuccessful rival. "
        "These are descriptive measurements, not significance tests or claims of superiority on all hardware."))
    for key, heading in (("faster", "ICORE faster"), ("slower", "ICORE slower"),
                         ("tie", "Tied within the selected threshold"), ("incomparable", "Incomparable / failed accuracy gate")):
        display(Markdown(f"### {heading} — {len(tables[key])} problems"))
        _table(tables[key])
    return tables


def export_tables(tables: dict, directory: str | Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, frame in tables.items():
        frame.to_csv(directory / f"{key}.csv", index=False)
    return directory
