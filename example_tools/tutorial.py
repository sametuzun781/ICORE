"""Small, optional notebook reports. These helpers never change the solver.

Model creation and Solver.solve() remain visible in each notebook. The normal
report only reads existing counters and independently checks the solution.
Detailed tracing performs one separate, explicitly untimed diagnostic solve.
"""
from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import HTML, Markdown, display

from .audit import audit_physical
from .catalog import Example
from .display import BODIES, trace_tables


def check_solution(problem: dict, solution: dict, *, nonnegative: int | None = None,
                   soc_sizes: tuple[int, ...] = (), tolerance: float = 1e-8) -> dict:
    """Check a block-form OCP solution in the original model's units.

    This is a separate correctness check, not part of native solve timing.
    A failed status or KKT test raises rather than displaying a plausible plot.
    """
    sizes = tuple(soc_sizes)
    count = problem["G"].shape[1] - sum(sizes) if nonnegative is None else nonnegative
    audit = audit_physical(problem, count, sizes, solution, tolerance=tolerance,
                           status_ok=int(solution["status"]) == 0)
    if not audit["accepted"]:
        raise RuntimeError(f"The returned solution did not pass the original-model check: {audit}")
    return audit


def _show_table(rows: list[dict]) -> None:
    display(pd.DataFrame(rows).style.hide(axis="index"))


def explain_solution(solver: Any, solution: dict, problem: dict, *,
                     objective_offset: float = 0.0, detailed: bool = False) -> dict:
    """Show actual kernel, presolve, counters, accuracy, and milliseconds.

    Call while the Solver context is open, immediately after solve().
    A factorization policy is labeled as a policy, not an executed operation.
    Setting detailed=True adds one separate instrumented replay; it does not
    instrument the production library and must not be included in solve timing.
    """
    if solver.backend not in ("c", "cpp"):
        raise ValueError("This native diagnostic report supports backend='c' or 'cpp'.")
    cones = tuple(solver.cones["soc_sizes"] or problem.get("soc_sizes", ()))
    count = solver.cones["nonnegative"]
    if count is None:
        count = problem.get("nonnegative", problem["G"].shape[1] - sum(cones))
    audit = check_solution(problem, solution, nonnegative=count, soc_sizes=cones,
                           tolerance=solver.settings["tolerance"])
    plan, counters = solver.factorization_plan, solver.diagnostics
    reduced = plan["selection_reason"] == "exact_coefficient_eligible_ocp_reduction"
    kernel = BODIES.get(counters.get("last_kernel"), "Unreported kernel")
    if reduced:
        reason = "Exact coefficient certificate accepted: the OCP was reduced to an affine chain."
    elif counters.get("last_kernel") == 1:
        reason = "Native one-transition dense-QP certificate accepted."
    elif plan["selection_reason"] == "explicit_adjacent_stage_chain":
        reason = "The input already has the adjacent-stage chain representation."
    else:
        reason = "The general stage-structured OCP representation was retained."
    info = solution["info"]
    rows = [
        {"Property": "Backend", "Value": "C++" if solver.backend == "cpp" else "C"},
        {"Property": "Status / iterations", "Value": f"Passed / {solution['iterations']}"},
        {"Property": "Native solve (ms)", "Value": f"{1000 * info['seconds']:.6f}"},
        {"Property": "Structural presolve requested", "Value": "Yes" if plan["structural_presolve"] else "No"},
        {"Property": "Exact OCP-to-chain reduction applied", "Value": "Yes" if reduced else "No"},
        {"Property": "Actual kernel", "Value": kernel},
        {"Property": "Reason", "Value": reason},
        {"Property": "Factorization policy (not a trace)", "Value": {
            "riccati_auto_with_equality_capable_qr": "Automatic Riccati; QR recovery supports terminal equalities",
            "riccati_classical": "Classical Riccati",
            "riccati_square_root": "Square-root Riccati",
            "chain_classical_with_augmented_ldl_and_qr_recovery": "Classical chain; augmented LDL and QR recovery available",
            "chain_classical_cholesky": "Classical chain Cholesky",
            "chain_orthogonal_qr": "Orthogonal QR chain",
        }.get(plan["factorization"], plan["factorization"])},
        {"Property": "Factor retries / refinement steps", "Value": f"{solution['factor_retries']} / {solution['refinements']}"},
        {"Property": "Dense-to-OCP numerical fallbacks", "Value": str(counters.get("dense_fallbacks", 0))},
        {"Property": "Packed stages / rejected attempts", "Value": f"{counters.get('ocp_packed_stages', 0)} / {counters.get('ocp_packed_rejected', 0)}"},
        {"Property": "Precision", "Value": "FP64 outer variables and residuals; use detailed replay for inner-factor precision."},
    ]
    display(Markdown("### What the solver actually did"))
    _show_table(rows)
    witness = getattr(solver, "_presolve_witness", None)
    if not plan["structural_presolve"]:
        display(Markdown("Optional structural presolve was disabled by the caller. Native structural checks still apply."))
    elif witness is not None and not reduced:
        display(Markdown("The optional chain-reduction certificate was rejected; keeping the general OCP is expected, not an error."))
    display(Markdown("### Accuracy in the original problem"))
    display(pd.DataFrame([{
        "primal error": audit["primal"], "dual error": audit["dual"],
        "complementarity": audit["gap"], "dynamics error": audit["dynamics"],
        "terminal error": audit["terminal"],
        "objective (including constant)": audit["objective"] + objective_offset,
        "passed": audit["accepted"],
    }]).style.hide(axis="index").format({
        "primal error": "{:.3e}", "dual error": "{:.3e}", "complementarity": "{:.3e}",
        "dynamics error": "{:.3e}", "terminal error": "{:.3e}",
        "objective (including constant)": "{:.6f}",
    }))
    display(Markdown("Native solve time excludes compilation, model setup, this audit, and plotting. "
                     "Counters do not list every factorization attempt or every reason a fast path was rejected."))
    report = {"audit": audit, "plan": plan, "counters": counters,
              "objective_with_constant": audit["objective"] + objective_offset}
    if detailed:
        from .trace import replay
        settings = dict(solver.settings)
        settings.pop("tolerance")
        settings["step_policy"] = ("auto", "fixed", "adaptive")[settings["step_policy"]]
        settings["structural_presolve"] = solver.structural_presolve
        options = {k: v for k, v in solver.cones.items() if v is not None}
        options.update(nonnegative=count, soc_sizes=cones)
        example = Example(metadata={"name": "user_ocp", "representation": "ocp",
            "family": "tutorial", "cohort": "tutorial", "source_groups": ["tutorial"],
            "options": options, "solver_settings": settings,
            "nonnegative": count, "soc_sizes": cones, "reference_objective": None},
            coefficients=problem)
        trace = replay(example, backend=solver.backend, tolerance=solver.settings["tolerance"],
                       production_result=solution)
        if not (trace["bitwise_matches_production"] and trace["audit"]["accepted"]
                and trace["dropped_events"] == 0):
            raise RuntimeError("Diagnostic replay did not match the production solve; do not interpret the trace.")
        display(Markdown("### Detailed replay — one additional untimed solve\n"
                         "The replay matched production bit for bit. It records selected calls and guards, "
                         "not every branch or a hardware profile."))
        for name, frame in trace_tables(trace).items():
            content = "No recorded events in this category." if frame.empty else frame.to_html(index=False, escape=True)
            display(HTML(f"<details><summary>{html.escape(name.replace('_', ' ').title())}</summary>"
                         f"<div style='overflow:auto;max-height:440px'>{content}</div></details>"))
        report["trace"] = trace
    return report
