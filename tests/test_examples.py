"""Tests of the notebook layer. Run with: python -m unittest tests.test_examples."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from example_tools.catalog import read_catalog, list_problems, load_problem
from example_tools.display import formulation, trace_tables
from example_tools.runner import mean_table, comparison_tables, run_all, run_problem
from example_tools.trace import _instrument


def sample(name, solver, times, accepted=None):
    accepted = [True] * len(times) if accepted is None else accepted
    return [dict(problem=name, solver=solver, repetition=i+1, setup_ms=0.1,
                 compile_ms=0.0, solve_ms=t, total_ms=None if t is None else t+0.1,
                 accepted=ok, outcome="passed" if ok else "failed_accuracy", error="")
            for i, (t, ok) in enumerate(zip(times, accepted))]


class NotebookHelpers(unittest.TestCase):
    def test_catalog_and_aliases(self):
        manifest = read_catalog()
        self.assertEqual(manifest["count"], 393)
        self.assertEqual(len({x["coefficient_identity"] for x in manifest["cases"]}), 393)
        self.assertEqual(sum(manifest["unique_by_primary_cohort"].values()), 393)
        for alias, name in list(manifest["aliases"].items())[:3]:
            self.assertEqual(load_problem(alias).name, name)
        self.assertEqual(len(list_problems(cohort="main175")), 175)
        self.assertTrue(all("LIPM" in p["name"] for p in list_problems(contains="LIPM")))

    def test_every_formulation_and_canonical_model(self):
        for meta in list_problems():
            problem = load_problem(meta["name"])
            with self.subTest(problem=problem.name):
                form = formulation(problem)
                self.assertTrue(form["equations"])
                self.assertIn("minimize", form["equations"][0])
                canonical = problem.canonical
                self.assertEqual(canonical.P.shape[0], len(canonical.c))
                self.assertEqual(canonical.G.shape[0], len(canonical.h))
                self.assertEqual(canonical.A.shape[0], len(canonical.b))
                self.assertFalse(any(array.flags.writeable for array in problem.coefficients.values()))

    def test_mean_is_not_median(self):
        table = mean_table(sample("P", "ICORE C++", [1., 2., 9.]), repetitions=3)
        self.assertEqual(table.iloc[0].solve_mean_ms, 4.)
        self.assertTrue(table.iloc[0].eligible)

    def test_failures_not_filtered_into_fast_mean(self):
        rows = sample("P", "ICORE C++", [1., 2., 9.], [True, True, False])
        rows += sample("P", "QOCO", [5., 5., 5.])
        self.assertEqual(mean_table(rows, repetitions=3).iloc[0].solve_mean_ms, 4.)
        tables = comparison_tables(rows, repetitions=3, rivals=("QOCO",))
        self.assertEqual(len(tables["faster"]), 0)
        self.assertEqual(len(tables["incomparable"]), 1)

    def test_fastest_valid_rival_partial_is_labeled(self):
        rows = sample("P", "ICORE C++", [2., 2., 2.])
        rows += sample("P", "HPIPM", [1., 1., 1.], [False]*3)
        rows += sample("P", "QOCO", [3., 3., 3.])
        tables = comparison_tables(rows, repetitions=3)
        self.assertEqual(len(tables["faster"]), 1)
        self.assertEqual(tables["faster"].iloc[0]["fastest valid rival"], "QOCO")
        self.assertIn("partial", tables["faster"].iloc[0]["coverage"])
        only_hp = comparison_tables(rows, repetitions=3, rivals=("HPIPM",))
        self.assertEqual(len(only_hp["incomparable"]), 1)

    def test_all_requested_rivals_and_icore_backends(self):
        rows = sample("P", "ICORE C++", [2.]*3) + sample("P", "ICORE C", [0.5]*3)
        rows += sample("P", "HPIPM", [1.]*3) + sample("P", "QOCO", [3.]*3)
        self.assertEqual(len(comparison_tables(rows, repetitions=3)["slower"]), 1)
        self.assertEqual(len(comparison_tables(rows, repetitions=3, icore_solver="ICORE C")["faster"]), 1)
        self.assertEqual(len(comparison_tables(rows, repetitions=3, rivals=("QOCO",))["faster"]), 1)

    def test_unsupported_never_creates_win(self):
        rows = sample("P", "ICORE C++", [1.]*3)
        rows += sample("P", "HPIPM", [None]*3, [False]*3)
        for row in rows[3:]:
            row["outcome"] = "unsupported"
        tables = comparison_tables(rows, repetitions=3, rivals=("HPIPM",))
        self.assertEqual(len(tables["incomparable"]), 1)
        self.assertEqual(tables["summary"].iloc[1].measured, 0)

    def test_missing_and_duplicate_repeats_ineligible(self):
        rows = sample("P", "ICORE C++", [1., 2.])
        self.assertFalse(mean_table(rows, repetitions=3).iloc[0].eligible)
        rows = sample("P", "ICORE C++", [1., 2., 3.])
        rows[2]["repetition"] = 2
        self.assertFalse(mean_table(rows, repetitions=3).iloc[0].eligible)

    def test_settings_reject_zero_fraction_and_boolean(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                run_problem("LIPMWALK0", repetitions=value)

    def test_tie_and_bad_metric(self):
        rows = sample("P", "ICORE C++", [1.]*3) + sample("P", "QOCO", [1.]*3)
        self.assertEqual(len(comparison_tables(rows, repetitions=3, rivals=("QOCO",))["tie"]), 1)
        with self.assertRaises(ValueError):
            comparison_tables(rows, repetitions=3, metric="median")

    def test_checkpoint_resume_does_not_solve_again(self):
        problem = load_problem("LIPMWALK0")
        item = dict(problem=problem.metadata, rows=sample(problem.name, "ICORE C++", [1., 2., 3.]))
        prepared = dict(machine="unit-test", platform="unit-test", errors={})
        with tempfile.TemporaryDirectory() as directory:
            with patch("example_tools.runner.run_problem", return_value=item) as mocked:
                a = run_all(repetitions=3, methods=("ICORE C++",), names=[problem.name], prepared=prepared,
                            output_dir=directory)
                self.assertEqual(mocked.call_count, 1)
                b = run_all(repetitions=3, methods=("ICORE C++",), names=[problem.name], prepared=prepared,
                            output_dir=directory, resume=True)
                self.assertEqual(mocked.call_count, 1)
                self.assertEqual(a["rows"], b["rows"])
                with self.assertRaises(ValueError):
                    run_all(repetitions=4, methods=("ICORE C++",), names=[problem.name], prepared=prepared,
                            output_dir=directory, resume=True)

    def test_trace_edits_do_not_touch_numeric_expression(self):
        code = "static int cholesky(double *a) { if(a[0] > 0) return 1; return 0; }"
        catalog = []
        result = _instrument(code, "unit.c", catalog)
        self.assertIn("if(a[0] > 0)", result)
        self.assertEqual(len([x for x in catalog if x["kind"] == "return"]), 2)
        self.assertIn("example_trace_return", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
