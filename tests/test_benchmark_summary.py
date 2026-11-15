"""Regression tests for saved-run reporting only; no native solver is executed.

Run: python -m unittest tests.test_benchmark_summary
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("icore_benchmark_report", ROOT / "scripts/generate_benchmark_summary.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def fixture(name="P", repetitions=3, methods=None):
    methods = list(methods or report.ALL_SOLVERS)
    rows = []
    for method in methods:
        for i in range(repetitions):
            rows.append(dict(problem=name, solver=method, repetition=i + 1,
                             solve_ms=(1.0 if method.startswith("ICORE") else 2.0) * (i + 1),
                             accepted=True, outcome="passed", status_ok=True,
                             kkt_ok=True, objective_ok=True, status=1 if method == "QOCO" else 0))
    return dict(problem=dict(name=name, family="example", coefficient_identity="test-identity"),
                repetitions=repetitions, methods=methods, tolerance=1e-8, rows=rows)


class SavedRunReporting(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, data, path=None):
        path = path or self.root / "problem-0000.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def summary(self, data):
        return report.summarize_problem(self.save(data))

    def rows(self, data, method="ICORE C++"):
        return [r for r in data["rows"] if r["solver"] == method]

    def test_arithmetic_mean_of_all_runs_not_median(self):
        data = fixture()
        for row, value in zip(self.rows(data), (1, 2, 9)):
            row["solve_ms"] = value
        self.assertEqual(self.summary(data)["solver_means"]["ICORE C++"], 4)

    def test_one_failed_run_excludes_whole_pair(self):
        data = fixture()
        row = self.rows(data)[-1]
        row.update(accepted=False, outcome="failed_accuracy", solve_ms=90)
        summary = self.summary(data)
        self.assertNotIn("ICORE C++", summary["solver_means"])
        self.assertEqual(summary["groups"]["ICORE C++"]["state"], "failed")
        self.assertEqual(summary["groups"]["ICORE C++"]["mean_solve_ms"], 31)
        self.assertEqual(report.compute_summary([summary])[1]["QOCO"]["common_passed"], 0)

    def test_missing_repetition_is_incomplete(self):
        data = fixture()
        data["rows"].remove(self.rows(data)[-1])
        self.assertEqual(self.summary(data)["groups"]["ICORE C++"]["state"], "incomplete")

    def test_duplicate_repetition_is_incomplete(self):
        data = fixture()
        self.rows(data)[-1]["repetition"] = 1
        self.assertNotIn("ICORE C++", self.summary(data)["solver_means"])

    def test_extra_repetition_is_incomplete(self):
        data = fixture()
        data["rows"].append(copy.deepcopy(self.rows(data)[0]))
        self.assertEqual(self.summary(data)["groups"]["ICORE C++"]["state"], "incomplete")

    def test_invalid_repetition_ids(self):
        for value in (True, 1.0, "1", None, 0, 4):
            with self.subTest(value=value):
                data = fixture()
                self.rows(data)[0]["repetition"] = value
                self.assertEqual(self.summary(data)["groups"]["ICORE C++"]["state"], "incomplete")

    def test_out_of_order_is_valid(self):
        data = fixture()
        data["rows"].reverse()
        self.assertIn("ICORE C++", self.summary(data)["solver_means"])

    def test_invalid_times_never_rank(self):
        for value in (0, -1, float("inf"), float("nan"), None, "1.0", True):
            with self.subTest(value=value):
                data = fixture()
                self.rows(data)[1]["solve_ms"] = value
                summary = self.summary(data)
                self.assertEqual(summary["groups"]["ICORE C++"]["state"], "invalid")
                self.assertNotIn("ICORE C++", summary["solver_means"])

    def test_status_code_is_solver_specific(self):
        summary = self.summary(fixture())
        self.assertIn("QOCO", summary["solver_means"])
        self.assertIn("ICORE C++", summary["solver_means"])

    def test_redundant_audit_failure_is_not_ignored(self):
        for key in ("status_ok", "kkt_ok", "objective_ok"):
            data = fixture()
            self.rows(data)[0][key] = False
            self.assertNotIn("ICORE C++", self.summary(data)["solver_means"])

    def test_unsupported_unavailable_are_separate(self):
        data = fixture()
        for method, outcome in (("HPIPM", "unsupported"), ("QOCO", "unavailable")):
            for row in self.rows(data, method):
                row.update(accepted=False, outcome=outcome, solve_ms=None)
        summary = self.summary(data)
        self.assertEqual(summary["groups"]["HPIPM"]["state"], "unsupported")
        self.assertEqual(summary["groups"]["QOCO"]["state"], "unavailable")
        self.assertEqual(report.compute_summary([summary])[1]["HPIPM"]["common_passed"], 0)

    def test_contradictory_unsupported_is_invalid(self):
        data = fixture()
        for row in self.rows(data, "HPIPM"):
            row.update(outcome="unsupported")
        self.assertEqual(self.summary(data)["groups"]["HPIPM"]["state"], "invalid")

    def test_not_requested_is_not_failed(self):
        summary = self.summary(fixture(methods=["ICORE C++"]))
        self.assertEqual(summary["groups"]["QOCO"]["state"], "not_requested")

    def test_no_rows_for_requested_solver_is_incomplete(self):
        data = fixture()
        data["rows"] = [r for r in data["rows"] if r["solver"] != "QOCO"]
        self.assertEqual(self.summary(data)["groups"]["QOCO"]["state"], "incomplete")

    def test_invalid_n_methods_tolerance(self):
        for key, values in (("repetitions", (0, -1, True, 1.5)),
                            ("methods", ([], ["QOCO", "QOCO"], "QOCO")),
                            ("tolerance", (0, -1, None, float("inf")))):
            for value in values:
                with self.subTest(key=key, value=value):
                    data = fixture()
                    data[key] = value
                    with self.assertRaises(ValueError):
                        self.summary(data)

    def test_row_problem_or_method_mismatch_is_error(self):
        for key, value in (("problem", "other"), ("solver", "unknown")):
            data = fixture()
            data["rows"][0][key] = value
            with self.assertRaises(ValueError):
                self.summary(data)

    def test_model_preparation_error_uses_run_configuration(self):
        data = fixture()
        config = {key: data.pop(key) for key in ("repetitions", "methods", "tolerance")}
        for row in data["rows"]:
            row.update(accepted=False, outcome="error", solve_ms=None)
        summary = report.summarize_problem(self.save(data), config)
        self.assertEqual(summary["groups"]["QOCO"]["state"], "failed")

    def test_configuration_coefficient_mismatch_rejected(self):
        config = dict(inputs={"P": "another-input"})
        with self.assertRaises(ValueError):
            report.summarize_problem(self.save(fixture()), config)

    def test_configuration_settings_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            report.summarize_problem(self.save(fixture()), dict(repetitions=20))

    def test_duplicate_problem_rejected(self):
        self.save(fixture(), self.root / "problem-0000.json")
        self.save(fixture(), self.root / "problem-0001.json")
        with self.assertRaises(ValueError):
            report.load_run(self.root)

    def test_mixed_run_settings_rejected(self):
        for key, value in (("repetitions", 2), ("tolerance", 1e-7), ("methods", ["QOCO"])):
            with self.subTest(key=key):
                self.save(fixture(), self.root / "problem-0000.json")
                other = fixture("other", repetitions=2 if key == "repetitions" else 3,
                                methods=["QOCO"] if key == "methods" else None)
                other[key] = value
                self.save(other, self.root / "problem-0001.json")
                with self.assertRaises(ValueError):
                    report.load_run(self.root)

    def test_incomplete_configuration_rejected(self):
        self.save(fixture())
        self.save(dict(names=["P", "missing"]), self.root / "configuration.json")
        with self.assertRaises(ValueError):
            report.load_run(self.root)

    def test_progress_incomplete_rejected(self):
        self.save(fixture())
        self.save(dict(complete=False, completed=1, total=2), self.root / "progress.json")
        with self.assertRaises(ValueError):
            report.load_run(self.root)

    def test_environment_mismatch_rejected(self):
        self.save(fixture())
        self.save(dict(machine="arm64"), self.root / "configuration.json")
        self.save(dict(machine="x86_64"), self.root / "environment.json")
        with self.assertRaises(ValueError):
            report.load_run(self.root)

    def test_ambiguous_parent_must_not_select_a_run(self):
        for name in ("N5", "N20"):
            self.save(fixture(), self.root / name / "problem-0000.json")
            self.save({}, self.root / name / "environment.json")
        with self.assertRaises(ValueError):
            report.discover_runs_dir(self.root)
        self.assertEqual(report.discover_runs_dir(self.root / "N20"), (self.root / "N20").resolve())

    def test_unique_parent_and_missing_directory(self):
        self.save(fixture(), self.root / "one" / "problem-0000.json")
        self.save({}, self.root / "one" / "environment.json")
        self.assertEqual(report.discover_runs_dir(self.root), (self.root / "one").resolve())
        with self.assertRaises(ValueError):
            report.discover_runs_dir(self.root / "absent")

    def test_geometric_mean_and_zero_pairs(self):
        self.assertAlmostEqual(report.gmean([1, 9]), 3)
        self.assertTrue(math.isnan(report.gmean([])))
        with self.assertRaises(ValueError):
            report.gmean([1, 0])
        summary = self.summary(fixture(methods=["ICORE C++"]))
        comparisons = report.compute_summary([summary])[1]
        self.assertTrue(math.isnan(comparisons["QOCO"]["geomean_ratio"]))
        self.assertIn("N/A", report.comparison_markdown(comparisons))

    def test_coverage_partitions_every_problem(self):
        summaries = [self.summary(fixture()), self.summary(fixture("B", methods=["ICORE C++"]))]
        for row in report.coverage_table(summaries):
            self.assertEqual(sum(row[state] for state in report.OUTCOMES), 2)

    def test_readme_only_updates_marked_content_and_is_idempotent(self):
        path = self.root / "README.md"
        prefix, suffix = "# My introduction\n\n", "\n\n## License\nMIT\n## GPT\nKeep this.\n"
        path.write_text(prefix + report.BEGIN + "\nold\n" + report.END + suffix)
        block = report.BEGIN + "\nnew\n" + report.END
        report.update_readme(path, block)
        self.assertEqual(path.read_text(), prefix + block + suffix)
        first = path.read_bytes()
        report.update_readme(path, block)
        self.assertEqual(path.read_bytes(), first)

    def test_readme_block_stays_compact_and_links_to_details(self):
        summary = self.summary(fixture())
        comparisons = report.compute_summary([summary])[1]
        block = report.readme_block([summary], comparisons, {}, "example_N3")
        self.assertLess(len(block.split()), 150)
        self.assertEqual(block.count("!["), 1)
        self.assertNotIn("| Rival |", block)
        self.assertNotIn("<details>", block)
        self.assertIn("docs/BENCHMARK_SUMMARY.md", block)
        self.assertIn("docs/BENCHMARKING.md", block)
        self.assertIn("**every repetition**", block)
        self.assertIn("not support counts", block)
        self.assertIn("not significance claims", block)
        self.assertIn("geometric means of per-problem", block)

    def test_compact_readme_metadata_follows_selected_run(self):
        summary = self.summary(fixture(repetitions=7))
        comparisons = report.compute_summary([summary])[1]
        env = dict(platform="ExampleOS", machine="example-cpu")
        block = report.readme_block([summary], comparisons, env, "selected_N7")
        self.assertIn("7 repetitions", block)
        self.assertIn("audit tolerance `1e-08`", block)
        self.assertIn("ExampleOS / example-cpu", block)
        self.assertIn("selected_N7", block)
        self.assertNotIn("all393_N20", block)
        self.assertNotIn("20 repetitions", block)

    def test_compact_regeneration_preserves_license_and_gpt_text(self):
        summary = self.summary(fixture())
        comparisons = report.compute_summary([summary])[1]
        block = report.readme_block([summary], comparisons, {}, "example_N3")
        path = self.root / "README.md"
        intro = "# ICORE\n\nMy edited introduction.\n"
        footer = "\n## License & development\nMIT. GPT disclosure stays here.\n"
        path.write_text(intro + report.BEGIN + "old text" + report.END + footer,
                        encoding="utf-8")
        report.update_readme(path, block)
        self.assertEqual(path.read_text(encoding="utf-8"), intro + block + footer)
        first = path.read_bytes()
        report.update_readme(path, block)
        self.assertEqual(path.read_bytes(), first)

    def test_bad_readme_markers_do_not_overwrite(self):
        path = self.root / "README.md"
        for value in ("No markers", report.END + report.BEGIN, report.BEGIN * 2 + report.END):
            path.write_text(value)
            with self.assertRaises(ValueError):
                report.update_readme(path, report.BEGIN + report.END)
            self.assertEqual(path.read_text(), value)

    def test_csv_is_lf_and_missing_values_are_blank(self):
        path = self.root / "table.csv"
        report.write_csv(path, [dict(name="ICORE", value=float("nan"))], ["name", "value"])
        self.assertEqual(path.read_bytes(), b"name,value\nICORE,\n")

    def test_report_omits_private_paths_and_cache_ids(self):
        p = self.summary(fixture())
        env = dict(platform="macOS/arm64", libraries={"ICORE": "/Users/private/secret"},
                   runtime_identity={"native": "a_private_cache_key"}, solvers=[])
        output = self.root / "summary.md"
        comparisons = report.compute_summary([p])[1]
        report.write_markdown(output, self.root / "N20", [p], comparisons, env)
        text = output.read_text()
        self.assertNotIn("/Users/private", text)
        self.assertNotIn("a_private_cache_key", text)
        self.assertIn("macOS/arm64", text)


if __name__ == "__main__":
    unittest.main()
