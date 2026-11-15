"""Read the complete, deduplicated example catalog. No solve happens here."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse

from .audit import canonicalize, restore_ocp, restore_chain, audit_canonical, audit_physical, audit_chain
from .chain_canonical import ChainCanonicalizer
from .chain_lift import ChainLift

PROBLEMS = Path(__file__).resolve().parents[1] / "problems"


def read_catalog(directory: str | Path = PROBLEMS) -> dict:
    """Return metadata; backend repetitions and aliases are not new problems."""
    path = Path(directory) / "catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "icore-examples-v1":
        raise ValueError(f"Unsupported example catalog: {path}")
    if len(data["cases"]) != data["count"]:
        raise ValueError("Example count does not match the manifest")
    return data


def list_problems(directory: str | Path = PROBLEMS, *, cohort: str | None = None,
                  family: str | None = None, contains: str = "") -> list[dict]:
    rows = read_catalog(directory)["cases"]
    return [row for row in rows
            if (cohort is None or cohort in row["source_groups"])
            and (family is None or row["family"] == family)
            and contains.lower() in row["name"].lower()]


def _qp(values: dict, constant: float = 0.0) -> SimpleNamespace:
    matrices = {}
    for key in ("P", "A", "G"):
        matrix = sparse.csc_matrix(values[key], dtype=np.float64)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        matrices[key] = matrix
    return SimpleNamespace(**matrices, c=values["c"], b=values["b"], h=values["h"],
                           constant=constant, l=len(values["h"]), cone_sizes=(),
                           nonnegative=len(values["h"]), soc_sizes=())


@dataclass
class Example:
    """One coefficient instance, independent of execution language and repeats."""
    metadata: dict
    coefficients: dict[str, np.ndarray]
    dense_data: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.metadata["name"]

    @property
    def chain(self) -> bool:
        return self.metadata["representation"] == "chain"

    @property
    def dense(self) -> bool:
        return self.metadata["representation"] == "dense"

    @property
    def options(self) -> dict:
        return dict(self.metadata["options"])

    @property
    def qp(self) -> SimpleNamespace:
        return SimpleNamespace(**self.coefficients)

    @cached_property
    def layout(self):
        if self.chain:
            return ChainCanonicalizer(self.qp)
        if self.dense:
            return None
        return canonicalize(self.coefficients, self.metadata["nonnegative"],
                            tuple(self.metadata["soc_sizes"]))

    @cached_property
    def canonical(self) -> SimpleNamespace:
        """Exact sparse counterpart; assembly is not inside the solve timer."""
        if self.dense:
            values = {key: self.dense_data["dense_" + key]
                      for key in ("P", "c", "A", "b", "G", "h")}
            return _qp(values, float(self.dense_data.get("dense_constant", 0.0)))
        if self.chain:
            model = self.layout.build(self.qp)
            p = self.qp
            model.constant = float(.5 * np.sum(p.hessian * (model.fixed - p.center) ** 2)
                                   + np.sum(p.linear * model.fixed))
            model.nonnegative, model.soc_sizes = model.l, ()
            return model
        model = self.layout.problem
        model.l, model.cone_sizes = model.nonnegative, model.soc_sizes
        # The upstream QOCO update path requires every diagonal slot to exist.
        model.P.setdiag(model.P.diagonal())
        model.P.sort_indices()
        return model

    @cached_property
    def hpipm_model(self) -> tuple[dict, dict]:
        """Retain OCP structure; only original dense inputs use dense HPIPM."""
        if self.chain:
            lift = self.chain_lift
            count = max(lift.nonnegative, default=0)
            p = dict(lift.coefficients)
            G = np.zeros((lift.T + 1, count, lift.nu + lift.nx))
            h = np.ones((lift.T + 1, count))
            for k, size in enumerate(lift.nonnegative):
                G[k, :size], h[k, :size] = p["G"][k], p["h"][k]
            p["G"], p["h"] = G, h
            return p, {**lift.receipt(), "form": "one-node memory OCP lift",
                       "constant_padding_rows": int(h.size - sum(lift.nonnegative))}
        if self.dense:
            p = self.canonical
            return dict(P=p.P, q=p.c, A=p.A, b=p.b, G=p.G, h=p.h), {"form": "original dense QP"}
        return self.coefficients, {"form": "original stage OCP"}

    @cached_property
    def chain_lift(self) -> ChainLift:
        if not self.chain:
            raise ValueError("Only an adjacent-stage chain needs a memory lift")
        return ChainLift(self.coefficients)

    def restore_qoco(self, raw: dict) -> dict:
        if self.dense:
            return raw
        if self.chain:
            return {**raw, **restore_chain(self, self.layout, self.canonical, raw)}
        return {**raw, **restore_ocp(self, self.layout, raw)}

    def canonical_vectors(self, result: dict, method: str) -> tuple:
        if method == "QOCO":
            return tuple(result[key] for key in ("x", "y", "s", "z"))
        if self.dense:
            if method.startswith("HPIPM"):
                return tuple(result[key] for key in ("v", "y", "s", "z"))
            return (np.asarray(result["u"])[0].copy(), np.empty(0),
                    np.asarray(result["s"])[0].copy(), np.asarray(result["z"])[0].copy())
        if not self.chain:
            return self.layout.vectors(result)
        raise ValueError("Chain native outputs use the original-chain audit")

    def audit(self, result: dict, method: str, *, tolerance: float = 1e-8,
              canonical_raw: dict | None = None) -> dict:
        """Common absolute KKT gate, computed outside timing, in original units."""
        status = int(np.asarray(result["status"]))
        status_ok = status == (1 if method == "QOCO" else 0)
        if self.chain:
            metrics = audit_chain(self.qp, result, tolerance=tolerance)
            # Also check the actual inequalities, not just Gv+s=h.
            q, y, t = self.qp, np.asarray(result["stage_y"]), np.asarray(result["epigraph"])
            defect = np.einsum("kij,kj->ki", q.left, y[:-1]) + np.einsum("kij,kj->ki", q.right, y[1:]) + q.offset
            pair = np.einsum("kij,kj->ki", q.pair_left, y[:-1]) + np.einsum("kij,kj->ki", q.pair_right, y[1:]) + q.pair_offset
            local = np.einsum("kij,kj->ki", q.local_matrix, y) + q.local_offset
            actual = max(float(np.max(a, initial=0.0)) for a in (defect-t, -defect-t, pair, local))
            metrics["primal"] = max(metrics["primal"], actual)
        else:
            vectors = self.canonical_vectors(canonical_raw if method == "QOCO" else result, method)
            metrics = audit_canonical(self.canonical, vectors)
            if not self.dense and method.startswith("ICORE"):
                physical = audit_physical(self.coefficients, self.metadata["nonnegative"],
                                          self.metadata["soc_sizes"], result)
                metrics["physical_audit"] = physical
                for key in ("primal", "dual", "gap"):
                    metrics[key] = max(metrics[key], physical[key])
        if canonical_raw is not None:
            checked = audit_canonical(self.canonical, tuple(canonical_raw[k] for k in ("x", "y", "s", "z")))
            metrics["canonical_audit"] = checked
            for key in ("primal", "dual", "gap"):
                metrics[key] = max(metrics[key], checked[key])
        reference = self.metadata.get("reference_objective")
        error = abs(metrics.get("objective", np.nan) - reference) if reference is not None else None
        limit = 10 * tolerance * (1 + abs(reference)) if reference is not None else None
        objective_ok = bool(np.isfinite(error) and error <= limit) if error is not None else True
        kkt_ok = all(np.isfinite(metrics[k]) and metrics[k] <= tolerance for k in ("primal", "dual", "gap"))
        return dict(metrics, status_ok=status_ok, kkt_ok=kkt_ok, objective_ok=objective_ok,
                    reference_available=reference is not None, objective_error=error,
                    objective_error_limit=limit, accepted=bool(status_ok and kkt_ok and objective_ok))


def load_problem(name: str, directory: str | Path = PROBLEMS) -> Example:
    """Load a manifest-listed model only. Stored solutions never enter solvers."""
    directory = Path(directory)
    manifest = read_catalog(directory)
    name = manifest["aliases"].get(name, name)
    by_name = {row["name"]: row for row in manifest["cases"]}
    if name not in by_name:
        raise KeyError(f"Unknown problem {name!r}; use list_problems()")
    with np.load(directory / by_name[name].get("file", name + ".npz"), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        p = {key: data[key].copy() for key in data.files
             if key != "metadata" and not key.startswith("dense_")}
        dense = {key: data[key].copy() for key in data.files if key.startswith("dense_")}
    digest = hashlib.sha256()
    for key, value in sorted(p.items()):
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    digest.update(json.dumps(by_name[name]["options"], sort_keys=True).encode())
    if digest.hexdigest() != by_name[name]["coefficient_identity"]:
        raise ValueError("Actual problem coefficients differ from the catalog")
    for array in (*p.values(), *dense.values()):
        array.flags.writeable = False
    # The manifest carries merged groups and aliases after deduplication.
    return Example(dict(by_name[name]), p, dense)
