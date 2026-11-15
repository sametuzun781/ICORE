"""Check the visible notebook models against the retained tutorial references.

Run: python -m unittest tests.test_tutorials
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from icore import Solver
from example_tools.tutorial import check_solution, explain_solution
from tests.reference_models import BUILDERS

ROOT = Path(__file__).resolve().parents[1]


def inline_model(path: Path) -> tuple[dict, dict]:
    """Run only the explicit array-building cells, without reports or plots."""
    notebook = json.loads(path.read_text())
    values = {"np": np}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        source = "".join(cell["source"])
        if set(tags) & {"dynamics", "objective", "constraints"}:
            exec(compile(source, str(path), "exec"), values)
        elif "solve" in tags:
            exec(compile(source.split("with Solver", 1)[0], str(path), "exec"), values)
            break
    return values["problem"], values


class TutorialModels(unittest.TestCase):
    def test_all_five_inline_models_match_reference(self):
        paths = sorted((ROOT / "tutorials").glob("0[1-5]_*.ipynb"))
        self.assertEqual(len(paths), 5)
        for path, builder in zip(paths, BUILDERS):
            actual, values = inline_model(path)
            expected = builder()
            with self.subTest(notebook=path.name):
                self.assertEqual(set(actual), set(expected.model))
                for field in actual:
                    np.testing.assert_array_equal(actual[field], expected.model[field], err_msg=field)
                self.assertAlmostEqual(values["objective_offset"], expected.objective_offset, places=10)

    def test_both_native_backends_and_no_reporting_solve(self):
        paths = sorted((ROOT / "tutorials").glob("0[1-5]_*.ipynb"))
        for path, builder in zip(paths, BUILDERS):
            problem, _ = inline_model(path)
            expected = builder()
            for backend in ("c", "cpp"):
                with self.subTest(notebook=path.name, backend=backend):
                    with Solver(problem, backend=backend, **expected.options) as solver:
                        solution = solver.solve()
                        with patch("example_tools.tutorial.display"), patch.object(
                                solver._engine, "solve", side_effect=AssertionError("Report must not solve")):
                            report = explain_solution(solver, solution, problem)
                        self.assertTrue(report["audit"]["accepted"])
                        self.assertEqual(report["counters"], solver.diagnostics)

    def test_failure_is_not_silently_plotted(self):
        example = BUILDERS[0]()
        with Solver(example.model, backend="cpp") as solver:
            solution = solver.solve()
            failed = {**solution, "status": 1}
            with self.assertRaises(RuntimeError):
                check_solution(example.model, failed)

    def test_mpc_model_and_updates(self):
        problem, _ = inline_model(ROOT / "tutorials/06_mpc_updates.ipynb")
        reference = BUILDERS[1]().model
        for key in problem:
            np.testing.assert_array_equal(problem[key], reference[key])
        for backend in ("c", "cpp"):
            state = problem["x0"].copy()
            with Solver(problem, backend=backend) as solver:
                for _ in range(12):
                    problem["x0"] = state.copy()
                    solution = solver.solve(problem)
                    check_solution(problem, solution)
                    with Solver(problem, backend=backend) as fresh:
                        rebuilt = fresh.solve()
                    check_solution(problem, rebuilt)
                    np.testing.assert_array_equal(solution["x"], rebuilt["x"])
                    np.testing.assert_array_equal(solution["u"], rebuilt["u"])
                    state = (problem["A"][0] @ state + problem["B"][0] @ solution["u"][0]
                             + problem["b"][0])
            problem["x0"] = reference["x0"].copy()

    def test_notebook_markdown_has_no_control_characters(self):
        for path in (ROOT / "tutorials").glob("*.ipynb"):
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "markdown":
                    source = "".join(cell["source"])
                    self.assertFalse(any(ord(c) < 32 and c != "\n" for c in source), path.name)

    def test_portable_header_is_shipped(self):
        header = ROOT / "icore/src/mixed_sweep_stubs.h"
        self.assertTrue(header.is_file())
        self.assertIn("icore_ocp_la_mixed_sweeps_begin", header.read_text())


if __name__ == "__main__":
    unittest.main()
