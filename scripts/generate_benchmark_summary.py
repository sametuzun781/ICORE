#!/usr/bin/env python3
"""Regenerate README benchmark figures and tables from saved Compare.ipynb runs.

No solver is imported or executed. A comparison uses a problem/solver pair only
when every requested repetition passed and every solve time is finite and positive.
Ambiguous run directories, duplicate problems, mixed settings and partial saved
runs are rejected rather than silently combined. README content outside the
ICORE-BENCHMARK markers is never changed.

Run from the repository root:
    python scripts/generate_benchmark_summary.py --runs-dir runs/all393_N20 --repo-root .
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASELINE = "ICORE C++"
RIVALS = ("HPIPM", "QOCO")
ALL_SOLVERS = ("ICORE C++", "ICORE C", "HPIPM", "QOCO")
OUTCOMES = ("passed", "failed", "unsupported", "unavailable", "incomplete", "invalid", "not_requested")
BEGIN = "<!-- BEGIN ICORE-BENCHMARK -->"
END = "<!-- END ICORE-BENCHMARK -->"
# Preserve the blue/teal palette used by the existing README figures.
COLORS = {
    "blue": "#2F6BFF", "blue_dark": "#0F274D", "blue_light": "#DDE9FF",
    "teal": "#38A7D0", "gray_text": "#425466", "gray_muted": "#6B7280",
    "gray_light": "#E8EEF7", "gray_panel": "#F7F9FC",
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def discover_runs_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Run directory does not exist: {path}")
    if list(path.glob("problem-*.json")):
        return path
    candidates = sorted({p.parent for p in path.rglob("environment.json")
                         if list(p.parent.glob("problem-*.json"))})
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"No saved benchmark run found under {path}")
    choices = "\n".join(f"  --runs-dir {p}" for p in candidates)
    raise ValueError("More than one saved run was found; select exactly one:\n" + choices)


def load_problem_files(runs_dir: Path) -> list[Path]:
    files = sorted(runs_dir.glob("problem-*.json"))
    if not files:
        raise ValueError(f"No problem-*.json files in {runs_dir}")
    return files


def positive_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def mean(values):
    # Include every repetition, never just the successful subset.
    return math.fsum(values) / len(values) if values else float("nan")


def gmean(values):
    if not values:
        return float("nan")
    if not all(positive_number(x) for x in values):
        raise ValueError("A speed ratio is not finite and positive")
    return math.exp(math.fsum(math.log(x) for x in values) / len(values))


def audit_passed(row: dict) -> bool:
    """Use recorded independent-audit decisions, not a solver-specific status code.

    QOCO and ICORE use different native success codes. We do not reinterpret them,
    or claim to rerun the original-model KKT audit without the solution arrays.
    """
    return (row.get("accepted") is True and row.get("outcome") == "passed"
            and all(row[key] is True for key in ("status_ok", "kkt_ok", "objective_ok")
                    if key in row))


def group_result(rows: list[dict], repetitions: int) -> dict:
    ids = [r.get("repetition") for r in rows]
    values = [r.get("solve_ms") for r in rows]
    measured = [x for x in values if positive_number(x)]
    result = dict(records=len(rows), expected=repetitions,
                  passed_repetitions=sum(audit_passed(r) for r in rows),
                  measured_repetitions=len(measured), mean_solve_ms=None,
                  state="incomplete", reason="Missing, duplicate or invalid repetition IDs")
    complete = (len(rows) == repetitions
                and all(type(i) is int for i in ids)
                and set(ids) == set(range(1, repetitions + 1)))
    if not complete:
        return result
    outcomes = {r.get("outcome") for r in rows}
    if outcomes in ({"unsupported"}, {"unavailable"}):
        state = next(iter(outcomes))
        if any(r.get("accepted") is True or r.get("solve_ms") is not None for r in rows):
            result.update(state="invalid", reason=f"Contradictory {state} record")
        else:
            result.update(state=state, reason=f"Every requested attempt was {state}")
        return result
    if len(measured) == repetitions:
        # This is retained for diagnostics even for a failed group; it is not ranked.
        result["mean_solve_ms"] = mean(measured)
    if result["passed_repetitions"] != repetitions:
        result.update(state="failed", reason="One or more attempts failed the recorded audit/status or raised an error")
    elif len(measured) != repetitions:
        result.update(state="invalid", reason="A passed attempt has a missing, nonfinite or nonpositive solve time")
    else:
        result.update(state="passed", reason="All requested repetitions passed")
    return result


def summarize_problem(path: Path, configuration: dict | None = None) -> dict:
    data = read_json(path)
    config = configuration or {}
    problem = data["problem"]
    name = problem["name"]
    repetitions = data.get("repetitions", config.get("repetitions"))
    methods = data.get("methods", config.get("methods"))
    tolerance = data.get("tolerance", config.get("tolerance"))
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError(f"{name}: repetitions must be a positive integer")
    if (not isinstance(methods, list) or not methods
            or not all(isinstance(m, str) for m in methods) or len(set(methods)) != len(methods)):
        raise ValueError(f"{name}: methods must be distinct solver names")
    if not positive_number(tolerance):
        raise ValueError(f"{name}: tolerance must be finite and positive")
    for key, value in (("repetitions", repetitions), ("methods", methods), ("tolerance", tolerance)):
        if key in config and value != config[key]:
            raise ValueError(f"{name}: {key} differs from configuration.json")
    expected_identity = config.get("inputs", {}).get(name)
    if expected_identity is not None and problem.get("coefficient_identity") != expected_identity:
        raise ValueError(f"{name}: coefficient identity differs from configuration.json")
    grouped = defaultdict(list)
    for row in data["rows"]:
        if row.get("problem") != name or row.get("solver") not in methods:
            raise ValueError(f"{name}: row belongs to another problem or unrequested solver")
        grouped[row["solver"]].append(row)
    groups = {m: group_result(grouped[m], repetitions) for m in methods}
    for solver in ALL_SOLVERS:
        if solver not in groups:
            groups[solver] = dict(state="not_requested", reason="Not requested in this run",
                                  records=0, expected=0, passed_repetitions=0,
                                  measured_repetitions=0, mean_solve_ms=None)
    return dict(name=name, family=problem["family"], cohort=problem.get("cohort", ""),
                representation=problem.get("representation", ""), methods=methods,
                repetitions=repetitions, tolerance=tolerance, groups=groups,
                solver_means={m: g["mean_solve_ms"] for m, g in groups.items() if g["state"] == "passed"})


def load_run(runs_dir: Path) -> tuple[list[dict], dict]:
    config_path = runs_dir / "configuration.json"
    config = read_json(config_path) if config_path.exists() else {}
    summaries = [summarize_problem(p, config) for p in load_problem_files(runs_dir)]
    names = [p["name"] for p in summaries]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate problem names in saved run")
    if "names" in config and (len(config["names"]) != len(names) or set(config["names"]) != set(names)):
        raise ValueError("Saved run is incomplete or has unexpected problems; finish it before publishing")
    if len({(p["repetitions"], tuple(p["methods"]), p["tolerance"]) for p in summaries}) != 1:
        raise ValueError("Mixed repetitions, solver lists or tolerances in the same run")
    progress = runs_dir / "progress.json"
    if progress.exists():
        p = read_json(progress)
        if (p.get("complete") is not True or p.get("completed") != len(names)
                or p.get("total") != len(names)):
            raise ValueError("progress.json does not describe a complete matching run")
    environment_path = runs_dir / "environment.json"
    environment = read_json(environment_path) if environment_path.exists() else {}
    for key in ("machine", "platform", "runtime_identity", "methods"):
        if key in config and key in environment and config[key] != environment[key]:
            raise ValueError(f"{key} differs between configuration.json and environment.json")
    return summaries, environment


def compute_summary(problem_summaries):
    passed_counts = Counter()
    families = Counter()
    for problem in problem_summaries:
        families[problem["family"]] += 1
        passed_counts.update(problem["solver_means"] .keys())
    comparisons, family_ratios, pair_rows = {}, {}, {}
    for rival in RIVALS:
        rows, by_family = [], defaultdict(list)
        faster = slower = ties = 0
        for problem in problem_summaries:
            means = problem["solver_means"]
            if BASELINE not in means or rival not in means:
                continue
            baseline, other = means[BASELINE], means[rival]
            ratio = other / baseline
            if not positive_number(ratio):
                raise ValueError(f"Invalid speed ratio for {problem['name']}")
            faster += baseline < other
            slower += baseline > other
            ties += baseline == other
            by_family[problem["family"]].append(ratio)
            rows.append(dict(problem=problem["name"], family=problem["family"],
                             baseline_ms=baseline, rival_ms=other, ratio_rival_over_icore=ratio))
        ratios = [r["ratio_rival_over_icore"] for r in rows]
        comparisons[rival] = dict(common_passed=len(rows), icore_faster=faster,
                                  rival_faster=slower, ties=ties, geomean_ratio=gmean(ratios),
                                  median_ratio=statistics.median(ratios) if ratios else float("nan"), rows=rows)
        family_ratios[rival], pair_rows[rival] = by_family, rows
    family_summary = []
    for family in sorted(families):
        entry = dict(family=family, n_problems=families[family])
        for rival in RIVALS:
            ratios = family_ratios[rival].get(family, [])
            entry[f"{rival}_n"] = len(ratios)
            entry[f"{rival}_geomean_ratio"] = gmean(ratios)
        family_summary.append(entry)
    return passed_counts, comparisons, family_summary, pair_rows


def coverage_table(summaries):
    return [dict(solver=solver, **{state: sum(p["groups"][solver]["state"] == state
                                            for p in summaries) for state in OUTCOMES})
            for solver in ALL_SOLVERS]


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if isinstance(v, float) and not math.isfinite(v) else v
                             for k, v in row.items()})


def make_summary_figure(path: Path, passed_counts, rival_summaries, n_problems, repetitions):
    """Summary layout v2: renderer-measured rows and a dedicated PNG canvas.

    Adjacent rows are positioned from the previous row's actual bounding box,
    not guessed font heights. Rendering uses the same Agg canvas for measuring
    and saving, without changing the backend for the other figures.
    """
    import math
    from pathlib import Path
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import Bbox

    # Isolate this figure from GUI/Retina scaling and personal matplotlibrc files.
    settings = {
        'font.family': ['DejaVu Sans'], 'font.size': 11,
        'font.style': 'normal', 'font.weight': 'normal',
        'font.stretch': 'normal', 'text.usetex': False,
        'figure.autolayout': False, 'figure.constrained_layout.use': False,
        'axes.labelsize': 11, 'axes.labelweight': 'normal',
        'axes.grid': False, 'axes.linewidth': 0.8,
        'xtick.labelsize': 10, 'ytick.labelsize': 11,
        'savefig.bbox': None, 'savefig.transparent': False,
    }
    with mpl.rc_context(settings):
        fig = Figure(figsize=(14, 8.5), dpi=240, facecolor='white')
        canvas = FigureCanvasAgg(fig)
        renderer = canvas.get_renderer()
        width_pt, height_pt = 14 * 72, 8.5 * 72
        px_per_pt = fig.dpi / 72
        tracked = []
        baseline = globals().get('BASELINE', 'ICORE C++')

        def ratio_text(value):
            return f'{value:.2f}×' if math.isfinite(value) and value > 0 else 'N/A'

        def panel(x, top, width, height, facecolor='white', edgecolor='#DCE3F1'):
            """Coordinates are physical points, measured down from the top."""
            bounds = Bbox.from_bounds(
                x / width_pt, 1 - (top + height) / height_pt,
                width / width_pt, height / height_pt,
            )
            fig.add_artist(FancyBboxPatch(
                (bounds.x0, bounds.y0), bounds.width, bounds.height,
                boxstyle='round,pad=0,rounding_size=0.018',
                transform=fig.transFigure, linewidth=1,
                facecolor=facecolor, edgecolor=edgecolor, zorder=0,
            ))
            return bounds

        def label(x, top, text, *, width, size=11, weight='normal',
                  color=None, region=None, ha='left'):
            """Fit text to its width and return its *measured* bottom in points."""
            artist = fig.text(
                x / width_pt, 1 - top / height_pt, str(text),
                ha=ha, va='top', fontsize=size, fontweight=weight,
                fontfamily='DejaVu Sans', fontstyle='normal',
                color=color or COLORS['gray_text'], linespacing=1.25,
                usetex=False, zorder=2,
            )
            available = width * px_per_pt
            # Measured word wrapping; explicit newlines remain explicit.
            wrapped = []
            for paragraph in str(text).split('\n'):
                line = ''
                for word in paragraph.split():
                    trial = (line + ' ' + word).strip()
                    artist.set_text(trial)
                    if line and artist.get_window_extent(renderer).width > available:
                        wrapped.append(line)
                        line = word
                    else:
                        line = trial
                wrapped.append(line)
            artist.set_text('\n'.join(wrapped))
            box = artist.get_window_extent(renderer)
            if box.width > available:
                # Only an unusually long, unbroken word should require this.
                new_size = size * available / box.width * 0.98
                if new_size < 9:
                    raise ValueError(f'Summary label is too long: {text!r}')
                artist.set_fontsize(new_size)
                box = artist.get_window_extent(renderer)
            tracked.append((artist, region))
            return height_pt - box.y0 / px_per_pt

        def stack(x, top, width, items, region=None):
            """Place each next line below the measured bottom, with real padding."""
            for text, options, gap_pt in items:
                top = label(x, top, text, width=width, region=region, **options)
                top += gap_pt
            return top

        try:
            if len(RIVALS) != 2:
                raise ValueError('This summary layout expects two rivals.')
            if len(ALL_SOLVERS) > 4:
                raise ValueError('This summary layout supports up to four coverage rows.')

            cursor = label(36, 24, 'Benchmark snapshot', width=936,
                           size=24, weight='bold', color=COLORS['blue_dark'])
            label(36, cursor + 12,
                  f'{n_problems} stored problems  |  {repetitions} runs per problem  |  '
                  'Arithmetic mean solve time [ms]', width=936, size=11.5)

            coverage = panel(36, 108, 248, 186, facecolor=COLORS['gray_panel'])
            cursor = label(54, 126, f'Passed all {repetitions} runs', width=212, size=13.5,
                           weight='bold', color=COLORS['blue_dark'], region=coverage) + 17
            for solver in ALL_SOLVERS:
                left_bottom = label(54, cursor, solver, width=155,
                                    size=12, region=coverage)
                right_bottom = label(266, cursor, str(passed_counts.get(solver, 0)),
                                     width=45, size=12, ha='right', weight='bold',
                                     color=COLORS['blue_dark'], region=coverage)
                cursor = max(left_bottom, right_bottom) + 13

            for i, rival in enumerate(RIVALS):
                x = 300 + i * 344
                card = panel(x, 108, 328, 186)
                info = rival_summaries[rival]
                stack(x + 18, 126, 292, [
                    (f'{baseline} vs {rival}',
                     dict(size=14.5, weight='bold', color=COLORS['blue_dark']), 15),
                    (ratio_text(info['geomean_ratio']),
                     dict(size=30, weight='bold', color=COLORS['blue']), 10),
                    (f'Geometric mean of rival / {baseline}',
                     dict(size=10.5, color=COLORS['gray_muted']), 15),
                    (f"ICORE faster: {info['icore_faster']} / {info['common_passed']}",
                     dict(size=12), 10),
                    (f"Rival faster: {info['rival_faster']}  |  "
                     f"Median ratio: {ratio_text(info['median_ratio'])}",
                     dict(size=10.5), 0),
                ], region=card)

            explanation = panel(36, 314, 936, 82,
                                facecolor=COLORS['blue_light'], edgecolor='#C9D9FF')
            stack(54, 328, 900, [
                ('How to read this',
                 dict(size=13, weight='bold', color=COLORS['blue_dark']), 10),
                ('Ratios above 1 favor ICORE C++; ratios below 1 favor the rival.\n'
                 f'Both solvers must pass all {repetitions} runs. Counts show mean-time comparisons, '
                 'not statistical significance.', dict(size=11), 0),
            ], region=explanation)

            label(62, 417, 'Solve-time comparison', width=890, size=15,
                  weight='bold', color=COLORS['blue_dark'])
            chart = fig.add_axes([88 / width_pt, 60 / height_pt,
                                  866 / width_pt, 104 / height_pt])
            chart.set_facecolor('white')
            values = [rival_summaries[r]['geomean_ratio'] for r in RIVALS]
            scale = max([1.0] + [v for v in values if math.isfinite(v) and v > 0])
            bars = chart.barh(RIVALS,
                              [v if math.isfinite(v) and v > 0 else 0 for v in values],
                              height=0.56, color=[COLORS['teal'], COLORS['blue']])
            chart.axvline(1, color=COLORS['gray_muted'], lw=1.2, ls='--')
            chart.set_xlim(0, scale * 1.25)
            chart.set_xlabel(f'Geometric mean ratio (rival / {baseline})',
                             color=COLORS['gray_text'], labelpad=10)
            chart.set_axisbelow(True)
            chart.grid(axis='x', alpha=0.18)
            for side in ('top', 'right', 'left'):
                chart.spines[side].set_visible(False)
            chart.tick_params(axis='y', length=0, labelsize=11, pad=9,
                              colors=COLORS['gray_text'])
            chart.tick_params(axis='x', labelsize=10, colors=COLORS['gray_text'])
            chart.invert_yaxis()
            for bar, value in zip(bars, values):
                chart.text(bar.get_width() + scale * 0.025,
                           bar.get_y() + bar.get_height() / 2, ratio_text(value),
                           ha='left', va='center', fontsize=11,
                           color=COLORS['gray_text'])

            # Keep the guard: measurement and final output share the same canvas.
            canvas.draw()
            renderer = canvas.get_renderer()
            boxes = []
            for artist, region in tracked:
                box = artist.get_window_extent(renderer)
                bounds = fig.bbox if region is None else region.transformed(fig.transFigure)
                if not (bounds.x0 - 0.5 <= box.x0 and box.x1 <= bounds.x1 + 0.5
                        and bounds.y0 - 0.5 <= box.y0 and box.y1 <= bounds.y1 + 0.5):
                    raise RuntimeError(f'Summary text exceeds its container: {artist.get_text()!r}')
                boxes.append((artist.get_text(), box))
            for i, (text, box) in enumerate(boxes):
                for other, other_box in boxes[i + 1:]:
                    if box.overlaps(other_box):
                        raise RuntimeError(f'Overlapping summary labels: {text!r} / {other!r}')
            # No tight-layout or GUI draw after validation can move the labels.
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.print_png(str(path))
            print(f'Saved summary figure: {path}')
        finally:
            fig.clear()



def make_family_figure(path: Path, family_summary):
    """Show the same 14 largest families, with the actual paired sample counts."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = sorted((r for r in family_summary if r["n_problems"] >= 5),
                  key=lambda r: (-r["n_problems"], r["family"]))[:14]
    fig = Figure(figsize=(14, 8.8), dpi=220, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.38, 0.12, 0.59, 0.71])
    fig.text(0.04, 0.96, "Family-level benchmark view", va="top", fontsize=20,
             fontweight="bold", color=COLORS["blue_dark"])
    fig.text(0.04, 0.90, "14 largest families with at least 5 instances; selected by size, not performance.\n"
             "Labels show eligible pairs: HP = HPIPM, Q = QOCO. Missing pairs are not speed wins.",
             va="top", fontsize=11, color=COLORS["gray_text"], linespacing=1.5)
    for j, rival in enumerate(RIVALS):
        xs = [r[f"{rival}_geomean_ratio"] for r in rows]
        ys = [i + (-0.17 if j == 0 else 0.17) for i in range(len(rows))]
        ax.barh(ys, xs, height=0.32, label=f"{rival} / {BASELINE}",
                color=COLORS["teal" if j == 0 else "blue"])
    ax.axvline(1, linestyle="--", linewidth=1.2, color=COLORS["gray_muted"])
    ax.set_yticks(range(len(rows)), [f"{r['family']}  [HP {r['HPIPM_n']}, Q {r['QOCO_n']}]" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Geometric mean ratio (rival / ICORE C++); above 1 favors ICORE", labelpad=12)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    canvas.print_png(str(path))
    fig.clear()


def make_scatter_figure(path: Path, pair_rows):
    """One log-log chart; each marker is an eligible problem/rival pair."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10.6, 8.8), dpi=240, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    fig.text(0.08, 0.97, "Per-problem solve times", va="top", fontsize=22,
             fontweight="bold", color=COLORS["blue_dark"])
    fig.text(0.08, 0.91, "Each marker is one problem/rival pair passing every repetition.\n"
             "Above the diagonal: ICORE C++ has a lower mean solve time.",
             va="top", fontsize=11.5, color=COLORS["gray_text"], linespacing=1.5)
    ax = fig.add_axes([0.11, 0.12, 0.84, 0.67])
    all_values = []
    for j, rival in enumerate(RIVALS):
        rows = pair_rows[rival]
        xs, ys = [r["baseline_ms"] for r in rows], [r["rival_ms"] for r in rows]
        all_values.extend(xs + ys)
        ax.scatter(xs, ys, s=26, alpha=0.67, edgecolors="none",
                   marker="o" if j == 0 else "^", color=COLORS["teal" if j == 0 else "blue"],
                   label=f"{rival}: {len(rows)} pairs")
    lo, hi = (min(all_values) * 0.75, max(all_values) * 1.4) if all_values else (0.01, 100)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2,
            color=COLORS["gray_muted"], label="Equal mean time")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
           xlabel="ICORE C++ mean solve time [ms]", ylabel="Rival mean solve time [ms]")
    ax.grid(which="major", alpha=0.20)
    ax.grid(which="minor", alpha=0.06)
    ax.legend(loc="upper left", frameon=False)
    if not all_values:
        ax.text(0.5, 0.5, "No eligible pairs in this run", transform=ax.transAxes,
                ha="center", va="center")
    canvas.print_png(str(path))
    fig.clear()


def ratio_text(value, digits=3):
    return f"{value:.{digits}f}×" if math.isfinite(value) else "N/A"


def comparison_markdown(comparisons):
    rows = ["| Rival | Both pass all repetitions | ICORE C++ lower mean | Rival lower mean | Geometric mean ratio (rival / ICORE C++) |",
            "|---|---:|---:|---:|---:|"]
    for rival in RIVALS:
        r = comparisons[rival]
        label = "HPIPM BALANCE" if rival == "HPIPM" else rival
        rows.append(f"| {label} | {r['common_passed']} | {r['icore_faster']} | {r['rival_faster']} | {ratio_text(r['geomean_ratio'])} |")
    return "\n".join(rows)


def environment_description(environment):
    platform = environment.get("platform", "not recorded")
    machine = environment.get("machine", "not recorded")
    return f"{platform} / {machine}"


def readme_block(summaries, comparisons, environment, run_name):
    """Keep the README brief; detailed tables and methodology stay in docs/."""
    repetitions, tolerance = summaries[0]["repetitions"], summaries[0]["tolerance"]
    return f"""{BEGIN}
**{len(summaries)} stored instances · {repetitions} repetitions · audit tolerance `{tolerance:g}`**  
Recorded platform: `{environment_description(environment)}`; run: `{run_name}`.

[![Benchmark summary](docs/assets/benchmark-summary.png)](docs/BENCHMARK_SUMMARY.md)

Arithmetic mean cold-solve times include returned outputs; setup, compilation,
model conversion, and audits are excluded. Ratios are geometric means of per-problem
ratios, using only pairs passing **every repetition**; values above 1 favor ICORE.
Pass counts are not support counts, and lower sample means are not significance claims.

[Full results & extra plots](docs/BENCHMARK_SUMMARY.md) ·
[Timing & reproduction](docs/BENCHMARKING.md)
{END}"""


def update_readme(path: Path, block: str) -> bool:
    if not path.exists():
        print(f"README not found; generated reports are still available: {path}")
        return False
    text = path.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
        raise ValueError("README must contain exactly one ordered pair of ICORE-BENCHMARK markers; "
                         "no README text was changed. Use --no-update-readme to generate reports only.")
    start, stop = text.index(BEGIN), text.index(END) + len(END)
    text = text[:start] + block + text[stop:]
    path.write_text(text, encoding="utf-8")
    return True


def write_markdown(path, runs_dir, summaries, comparisons, environment):
    n, tolerance = summaries[0]["repetitions"], summaries[0]["tolerance"]
    coverage = coverage_table(summaries)
    lines = ["# Benchmark summary", "",
             f"Source: `{runs_dir.name}`; **{len(summaries)} coefficient instances**, **{n} repetitions per method**, audit tolerance **{tolerance:g}**.",
             "", f"Recorded environment: **{environment_description(environment)}**.", "",
             "All figures and tables below are recomputed from saved JSON, not from new solver runs.",
             "", "## Outcome coverage", "",
             f"A passed problem requires every one of its {n} repetitions to pass. These are not counts of mathematically supported problem classes.", "",
             "| Solver | Passed all | Failed/error | Unsupported | Unavailable | Incomplete | Invalid | Not requested |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in coverage:
        lines.append("| " + c["solver"] + " | " + " | ".join(str(c[s]) for s in OUTCOMES) + " |")
    lines += ["", "Unsupported means an explicit adapter outcome. Failed means a recorded status/audit failure or an error, not lack of mathematical support.",
              "Incomplete includes missing/duplicate repetition IDs. Invalid includes nonfinite, zero or negative timings in an otherwise passed group.",
              "", "## Pairwise comparisons", "", comparison_markdown(comparisons), "",
              "Times are arithmetic means over all N repetitions for each eligible problem/solver pair.",
              "For each pair, r = mean(rival solve_ms) / mean(ICORE C++ solve_ms).",
              "The aggregate is exp(mean(log(r))), with equal weight per problem. Above 1 favors ICORE; below 1 favors the rival.",
              "The paired sets differ between rivals. Missing and failed cases do not become speed wins.",
              "A lower mean is descriptive, not a confidence interval, a proof of superiority, or a measure of full application latency.",
              "", "## Timing scope and reproducibility", "",
              "The benchmark runner creates a fresh workspace and performs a cold solve for each repetition. `solve_ms` includes original-coordinate owned outputs, but excludes setup, compilation, shared model conversion, audits, and plotting.",
              "The order is seeded and rotated/reversed by the runner; repetitions are not claimed to be independent randomly sampled workloads.",
              "The catalog includes synthetic instances, coefficient updates, and equivalent formulations, not 393 independent application families.",
              "", "### Recorded software/build information", "",
              "| Solver | Recorded version | Recorded target | Recorded compiler |", "|---|---|---|---|"]
    for solver in environment.get("solvers", []):
        # Do not publish personal absolute library paths or internal cache IDs.
        values = [str(solver.get(k, "not recorded")).replace("|", "/")
                  for k in ("solver", "version", "target", "compiler")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", f"Python: `{environment.get('python', 'not recorded')}`.",
              "Hardware model, clock policy, and per-primitive backend choices are not fully recorded; they are not inferred here.",
              "In particular, ICORE's recorded target is shown as supplied. This run is not a claim of best-possible tuning for every solver.",
              "", "## Figures", "", "![Summary](assets/benchmark-summary.png)", "",
              "![Families](assets/benchmark-by-family.png)", "",
              "The family plot selects at most 14 families with at least 5 instances, by decreasing family size and then name. HP/Q sample counts can be smaller than the family size. All families, including smaller ones, are retained in the family CSV.",
              "", "![Per-problem means](assets/benchmark-scatter-log.png)", "",
              "## Files and regeneration", "",
              "- [Overall ratios](benchmark-summary.csv)", "- [Family ratios and eligible counts](benchmark-family-summary.csv)",
              "- [Coverage categories](benchmark-coverage.csv)", "- [Every problem/solver outcome](benchmark-outcomes.csv)",
              "", "From the repository root:", "", "```bash",
              f"python scripts/generate_benchmark_summary.py --runs-dir runs/{runs_dir.name} --repo-root .", "```", "",
              "The source run files are not included in this documentation update. Keep your local `runs/` directory to reproduce the aggregation. No solver is executed by this command.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(repo_root, runs_dir, summaries, comparisons, families, environment):
    docs = repo_root / "docs"
    coverage = coverage_table(summaries)
    write_csv(docs / "benchmark-coverage.csv", coverage, ["solver", *OUTCOMES])
    ratios = [dict(rival=rival, common_passed_problems=r["common_passed"],
                   icore_faster=r["icore_faster"], rival_faster=r["rival_faster"], ties=r["ties"],
                   geomean_ratio_rival_over_icore_cpp=r["geomean_ratio"],
                   median_ratio_rival_over_icore_cpp=r["median_ratio"])
              for rival, r in comparisons.items()]
    write_csv(docs / "benchmark-summary.csv", ratios, list(ratios[0]))
    write_csv(docs / "benchmark-family-summary.csv", families, list(families[0]))
    outcomes = [dict(problem=p["name"], family=p["family"], cohort=p["cohort"], solver=m, **g)
                for p in summaries for m, g in p["groups"].items()]
    write_csv(docs / "benchmark-outcomes.csv", outcomes, list(outcomes[0]))
    write_markdown(docs / "BENCHMARK_SUMMARY.md", runs_dir, summaries, comparisons, environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Exactly one saved run, or a parent containing only one run")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--no-update-readme", action="store_true", help="Generate reports without changing README")
    args = parser.parse_args()
    try:
        runs_dir = discover_runs_dir(args.runs_dir)
        summaries, environment = load_run(runs_dir)
        passed, comparisons, families, pairs = compute_summary(summaries)
        root = args.repo_root.expanduser().resolve()
        # Validate README markers before changing any output files.
        if not args.no_update_readme and (root / "README.md").exists():
            text = (root / "README.md").read_text(encoding="utf-8")
            if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
                raise ValueError("README benchmark markers are missing or invalid. Use the updated README or --no-update-readme.")
        assets = root / "docs/assets"
        assets.mkdir(parents=True, exist_ok=True)
        make_summary_figure(assets / "benchmark-summary.png", passed, comparisons, len(summaries), summaries[0]["repetitions"])
        make_family_figure(assets / "benchmark-by-family.png", families)
        make_scatter_figure(assets / "benchmark-scatter-log.png", pairs)
        write_outputs(root, runs_dir, summaries, comparisons, families, environment)
        if not args.no_update_readme:
            update_readme(root / "README.md", readme_block(summaries, comparisons, environment, runs_dir.name))
        print(f"Updated saved-run reports under {root / 'docs'}; no solvers were executed.")
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Benchmark report error: {exc}\n")


if __name__ == "__main__":
    main()
