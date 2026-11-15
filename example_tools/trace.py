"""Opt-in, untimed decision tracing without editing the production solver.

A separate library is built from a copy of the exact production source. Integer
return sites, key accepted-path flags, and recovery counters emit events into a
thread-local buffer. The production sources, ABI, and benchmark library remain
untouched. Every replay is compared with a normal solve; a mismatch is disclosed.
"""
from __future__ import annotations

from collections import Counter
import ctypes as ct
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from icore._build import build
from .catalog import Example, load_problem
from .runner import clean

# Trace scope is explicit. This is not a hardware branch/branch-miss profiler.
FUNCTIONS = {
    "dense_qp_plan": ("eligibility", "Dense-QP structural certificate"),
    "packed_stage_size_eligible": ("eligibility", "Packed OCP size/structure policy"),
    "factor_problem_requested": ("policy", "OCP factorization dispatch"),
    "factor_classical": ("factorization", "Classical Riccati / Cholesky"),
    "factor_square_root": ("factorization", "Whole-chain square-root QR"),
    "factor_square_root_range": ("factorization", "Square-root QR stage range"),
    "factor_classical_qr_suffix": ("recovery", "Classical-to-QR suffix recovery"),
    "factor_local_square_root": ("recovery", "Packed local QR recovery"),
    "factor_packed_square_root": ("policy", "Packed square-root Riccati attempt"),
    "factor_mixed_square_root": ("eligibility", "Guarded FP32-factor OCP route"),
    "factor_mixed_sweeps": ("eligibility", "Guarded FP32-factor/sweep route"),
    "dense_qp_factor": ("policy", "Dense Newton factorization dispatch"),
    "icore_la_factor": ("factorization", "BLASFEO dense Gram–Cholesky"),
    "cholesky": ("factorization", "Portable Cholesky"),
    "cholesky_equilibrated": ("recovery", "Equilibrated Cholesky recovery"),
    "icore_ocp_la_sqrt_begin": ("factorization", "FP64 packed terminal Cholesky"),
    "icore_ocp_la_sqrt_begin_cached_diagonal": ("factorization", "FP64 diagonal-cost packed terminal Cholesky"),
    "icore_ocp_la_sqrt_stage": ("factorization", "FP64 packed Riccati stage"),
    "icore_ocp_la_sqrt_stage_cached_diagonal": ("factorization", "FP64 diagonal-cost packed Riccati stage"),
    "icore_ocp_la_mixed_begin": ("factorization", "FP32 terminal factor attempt"),
    "icore_ocp_la_mixed_stage": ("factorization", "FP32 stage factor attempt"),
    "icore_ocp_la_mixed_sweeps_begin": ("factorization", "FP32 factor-and-sweep terminal attempt"),
    "icore_ocp_la_mixed_sweeps_stage": ("factorization", "FP32 factor-and-sweep stage attempt"),
    "icore_ocp_la_qr_upper": ("factorization", "Packed Householder QR"),
    "qr_upper": ("factorization", "Portable range-safe Householder QR"),
    "factor": ("policy", "Affine-chain factorization dispatch"),
    "factor_recovery": ("recovery", "Chain augmented-LDL / QR recovery"),
    "factor_qr": ("factorization", "Affine-chain QR"),
    "chain_saddle_factor": ("factorization", "Chain augmented sparse LDL"),
    "newton_direction": ("direction", "OCP Newton direction / refinement"),
}

# These events are emitted only when the corresponding existing assignment or
# counter increment actually executes, not merely when a path is configured.
FLAGS = {
    "w->qp_diagonal_factor = 1;": ("accepted", "Diagonal dense Newton factor accepted"),
    "w->qp_packed_factor = 1;": ("accepted", "Packed dense Newton factor accepted"),
    "w->mixed_factor_active = 1;": ("precision", "FP32 factors accepted; outer state/residuals FP64"),
    "w->mixed_sweeps_active = 1;": ("precision", "FP32 forward/backward sweeps activated"),
    "++w->qp_fallbacks;": ("fallback", "Dense numerical failure: restarted in general OCP body"),
    "++w->ocp_sqrt_rejected;": ("rejection", "Packed square-root factor chain rejected"),
    "++w->ocp_packed_rejected;": ("rejection", "Packed stage path rejected"),
    "++w->factor_retries;": ("recovery", "Native factor retry/recovery counter increment"),
    "w->mixed_disabled = 1;": ("fallback", "Mixed-precision route disabled by existing guard"),
}

SNAPSHOTS = {
    "dense_qp_plan": ("w->N", "w->nx", "w->nu", "w->nsoc", "w->nf", "w->ne"),
    "packed_stage_size_eligible": ("w->nx", "w->nvr", "w->nsoc", "w->nf", "w->ne", "w->qp_eligible"),
    "factor_mixed_square_root": ("diagonal", "zero_initial_direction", "affine_predictor",
        "w->mixed_disabled", "w->orthant_direct", "w->ne", "w->nf", "w->nsoc", "w->ocp_la != NULL"),
    "factor_packed_square_root": ("diagonal", "zero_initial_direction", "affine_predictor", "w->ocp_la != NULL"),
}

HEADER = r''' 
#ifndef EXAMPLE_TRACE_H
#define EXAMPLE_TRACE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void example_trace_reset(void);
void example_trace_phase(int phase);
void example_trace_event(int id, int value);
int example_trace_return(int id, int value);
int example_trace_size(void);
int example_trace_dropped(void);
int example_trace_copy(int *out, int capacity);
#ifdef __cplusplus
}
#endif
#endif
'''
RECORDER = r'''
#include "example_trace.h"
#ifdef __cplusplus
#define EX_TLS thread_local
#else
#define EX_TLS _Thread_local
#endif
#define EX_CAPACITY 262144
static EX_TLS int events[EX_CAPACITY][3];
static EX_TLS int used, dropped, phase;
void example_trace_reset(void) { used = dropped = phase = 0; }
void example_trace_phase(int value) { phase = value; }
void example_trace_event(int id, int value) {
    if (used < EX_CAPACITY) {
        events[used][0] = id; events[used][1] = value; events[used][2] = phase; ++used;
    } else ++dropped;
}
int example_trace_return(int id, int value) { example_trace_event(id, value); return value; }
int example_trace_size(void) { return used; }
int example_trace_dropped(void) { return dropped; }
int example_trace_copy(int *out, int capacity) {
    int n = used < capacity ? used : capacity;
    for (int i = 0; i < n; ++i) for (int j = 0; j < 3; ++j) out[3*i+j] = events[i][j];
    return n;
}
'''


def _mask(source):
    """Blank comments/strings while preserving every source offset and newline."""
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern, lambda m: "".join("\n" if c == "\n" else " " for c in m.group()), source)


def _close(mask, start, left, right):
    depth = 0
    for i in range(start, len(mask)):
        if mask[i] == left:
            depth += 1
        elif mask[i] == right:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unbalanced native source while building diagnostic copy")


def _instrument(source, relative, catalog):
    mask, edits = _mask(source), []

    def event(offset, function, kind, label, expression=None):
        line = source.count("\n", 0, offset) + 1
        context = "\n".join(source[:offset].splitlines()[-5:]).strip()
        index = len(catalog)
        catalog.append(dict(id=index, function=function, kind=kind, label=label,
                            file=relative, line=line, expression=expression,
                            source_context=context,
                            success_convention="zero" if function.startswith(("icore_la_", "icore_ocp_la_")) else "nonzero"))
        return index

    pattern = re.compile(r"(?m)^(?:static\s+)?(?:inline\s+)?(int|void)\s+(" + "|".join(map(re.escape, FUNCTIONS)) + r")\s*\(")
    for match in pattern.finditer(mask):
        return_type, name = match.groups()
        paren = mask.index("(", match.start())
        end_paren = _close(mask, paren, "(", ")")
        opening = end_paren + 1
        while opening < len(mask) and mask[opening].isspace():
            opening += 1
        if opening >= len(mask) or mask[opening] != "{":
            continue  # Declaration, not definition.
        closing = _close(mask, opening, "{", "}")
        category, label = FUNCTIONS[name]
        index = event(opening, name, "enter", label)
        catalog[index]["category"] = category
        entry_code = f"\n    example_trace_event({index}, -1);\n"
        for expression in SNAPSHOTS.get(name, ()):
            snapshot = event(opening, name, "guard_value", expression, expression)
            entry_code += f"    example_trace_event({snapshot}, (int)({expression}));\n"
        edits.append((opening + 1, opening + 1, entry_code))
        for ret in re.finditer(r"\breturn\b[^;]*;", mask[opening+1:closing]):
            start, end = opening+1+ret.start(), opening+1+ret.end()
            expression = source[start+6:end-1].strip()
            index = event(start, name, "return", label, expression or "void early return")
            catalog[index]["category"] = category
            if return_type == "void":
                replacement = f"{{ example_trace_event({index}, 0); return; }}"
            else:
                replacement = f"return example_trace_return({index}, ({expression}));"
            edits.append((start, end, replacement))
        if return_type == "void":
            index = event(closing, name, "return", label, "normal completion")
            catalog[index]["category"] = category
            edits.append((closing, closing, f"\n    example_trace_event({index}, 1);\n"))
    for literal, (kind, label) in FLAGS.items():
        for match in re.finditer(re.escape(literal), mask):
            index = event(match.start(), "state_change", kind, label, literal)
            # A compound statement preserves an enclosing unbraced if/else.
            edits.append((match.start(), match.end(),
                          "{ " + literal + f" example_trace_event({index}, 1); }}"))
    edits.sort(reverse=True)
    last = len(source) + 1
    for start, end, replacement in edits:
        if end > last:
            raise ValueError(f"Overlapping diagnostic instrumentation in {relative}")
        source = source[:start] + replacement + source[end:]
        last = start
    return source


def build_trace(backend="cpp"):
    """Build only a separate observer library; never patch the loaded production one."""
    if backend not in ("c", "cpp"):
        raise ValueError("Native detailed replay supports c/cpp, not JAX")
    production = Path(build(backend))
    receipt = json.loads(production.with_suffix(".json").read_text())
    identity = json.loads(receipt["identity"])
    import icore
    package = Path(icore.__file__).parent
    source = package / "src"
    tag = hashlib.sha256((receipt["identity"] + Path(__file__).read_text()).encode()).hexdigest()[:16]
    root = package.parent / ".build" / "example_trace" / (backend + "-" + tag)
    root.mkdir(parents=True, exist_ok=True)
    output = root / ("libtrace.dylib" if sys.platform == "darwin" else "libtrace.so")
    import fcntl
    with (root / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        catalog_file = root / "event_catalog.json"
        if not output.exists() or not catalog_file.exists():
            target = root / "src"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            catalog = []
            for path in sorted(target.rglob("*")):
                if path.suffix in (".c", ".h"):
                    path.write_text(_instrument(path.read_text(), str(path.relative_to(target)), catalog))
            header = target / "example_trace.h"
            header.write_text(HEADER)
            recorder = target / "example_trace.c"
            recorder.write_text(RECORDER)
            command = [str(x).replace(str(source), str(target)) for x in identity["command"]]
            first_source = next(i for i, x in enumerate(command) if x.endswith(".c"))
            command[first_source:first_source] = ["-DICORE_OCP_MIXED_DIAGNOSTICS=1", "-include", str(header), str(recorder)]
            temporary = output.with_suffix(output.suffix + ".tmp")
            command[command.index("-o") + 1] = str(temporary)
            compiled = subprocess.run(command, capture_output=True, text=True)
            (root / "build.log").write_text(compiled.stdout + compiled.stderr)
            if compiled.returncode:
                raise RuntimeError("Diagnostic-only build failed; production library unchanged:\n" + compiled.stderr[-6000:])
            temporary.replace(output)
            catalog_file.write_text(json.dumps(catalog, indent=2))
            (root / "receipt.json").write_text(json.dumps(dict(production=str(production),
                command=command, scope="instrumented copy; never used for performance ranking"), indent=2))
    return output, json.loads(catalog_file.read_text()), package, tag


def _module(package, tag, library, backend):
    alias = "_icore_example_trace_" + backend + "_" + tag
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, package / "__init__.py", submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        native = importlib.import_module(alias + "._native")
        native.build = lambda requested: str(library) if requested == backend else build(requested)
    return sys.modules[alias]


def _same(a, b):
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    aa, bb = np.asarray(a), np.asarray(b)
    return aa.dtype == bb.dtype and aa.shape == bb.shape and aa.tobytes() == bb.tobytes()


def replay(problem: Example | str, *, backend: str = "cpp", tolerance: float = 1e-8,
           production_result: dict | None = None) -> dict:
    """One explicit untimed diagnostic solve, not an extra benchmark repetition."""
    problem = load_problem(problem) if isinstance(problem, str) else problem
    library, catalog, package, tag = build_trace(backend)
    module = _module(package, tag, library, backend)
    recorder = ct.CDLL(str(library))
    recorder.example_trace_reset.argtypes = []
    recorder.example_trace_phase.argtypes = [ct.c_int]
    recorder.example_trace_copy.argtypes = [ct.POINTER(ct.c_int), ct.c_int]
    recorder.example_trace_size.restype = ct.c_int
    recorder.example_trace_dropped.restype = ct.c_int
    normal_extra = 0
    settings = {"backend": backend, "tolerance": tolerance, "max_iterations": 200,
                **problem.options, **problem.metadata.get("solver_settings", {})}
    if production_result is None:
        import icore
        with icore.Solver(problem.coefficients, **settings) as ordinary:
            production_result = ordinary.solve()
        normal_extra = 1
    recorder.example_trace_reset()
    with module.Solver(problem.coefficients, **settings) as solver:
        plan = solver.factorization_plan
        witness = solver._presolve_witness
        recorder.example_trace_phase(1)
        result = solver.solve()
        diagnostics = solver.diagnostics
    count = recorder.example_trace_size()
    events = np.empty((count, 3), dtype=np.int32)
    copied = recorder.example_trace_copy(events.ctypes.data_as(ct.POINTER(ct.c_int)), count)
    events = events[:copied]
    fields = [k for k in ("status", "iterations", "refinements", "factor_retries", "objective", "residuals",
                          "x", "u", "y", "nu", "s", "z", "stage_y", "epigraph")
              if k in result and k in production_result]
    equal_fields = {k: _same(result[k], production_result[k]) for k in fields}
    grouped = []
    for (index, value, phase), repeats in Counter(map(tuple, events.tolist())).items():
        entry = catalog[index]
        if entry["kind"] == "return":
            success = value == 0 if entry["success_convention"] == "zero" else value != 0
        else:
            success = None
        grouped.append(dict(entry, value=value, phase="solve" if phase else "setup", count=repeats, success=success))
    return clean(dict(problem=problem.name, backend=backend,
        evidence="separate instrumented replay; production sources/library unchanged; not timed", plan=plan,
        presolve_witness=witness, diagnostics=diagnostics, equal_fields=equal_fields,
        bitwise_matches_production=all(equal_fields.values()),
        audit=problem.audit(result, "ICORE C++" if backend == "cpp" else "ICORE C", tolerance=tolerance),
        diagnostic_solves=1, additional_production_solves=normal_extra, dropped_events=recorder.example_trace_dropped(),
        event_catalog=catalog, events=events, grouped=grouped))
