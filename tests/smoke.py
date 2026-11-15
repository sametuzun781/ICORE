"""Analytic KKT oracle, newly generated exact structural transitions; not performance data."""
from pathlib import Path
import os, json
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import icore
R = Path(__file__).resolve().parents[1] / "runs"
R.mkdir(exist_ok=True)

def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2))
import numpy as np
mods = {'new': icore}
rng = np.random.default_rng(24110109)
rows = []
for n in [1, 3, 4, 7, 16, 33]:
    for kind in ['boxes', 'general', 'mixed', 'empty']:
        L = rng.normal(size=(n, n)) * 0.1
        Q = L.T @ L + np.eye(n)
        xs = rng.normal(size=n) * 0.2
        if kind == 'boxes':
            G = np.repeat(np.eye(n), 2, axis=0) * np.tile([1.0, -1.0], n)[:, None]
        elif kind == 'general':
            G = rng.normal(size=(n + 4, n))
        elif kind == 'mixed':
            G = np.vstack([np.eye(n)[:min(2, n)], rng.normal(size=(n + 4, n))])
        else:
            G = np.zeros((0, n))
        k = len(G)
        lam = np.zeros(k)
        lam[::3] = 0.3
        slack = np.ones(k)
        slack[::3] = 0.0
        hh = G @ xs + slack
        cv = -Q @ xs - G.T @ lam
        m = k + 8
        models = []
        for layout in ['prefix', 'offset1', 'offset3', 'offset7', 'holes', 'permuted', 'prefix_again']:
            if layout.startswith('offset'):
                idx = np.arange(k) + int(layout[6:])
            elif layout == 'holes':
                idx = np.sort(rng.choice(m, k, replace=False))
            elif layout == 'permuted':
                idx = rng.choice(m, k, replace=False)
            else:
                idx = np.arange(k)
            GG = np.zeros((m, n))
            hv = np.ones(m)
            GG[idx] = G
            hv[idx] = hh
            data = dict(H=np.zeros((2, n + 1, n + 1)), c=np.zeros((2, n + 1)), A=np.ones((1, 1, 1)), B=np.zeros((1, 1, n)), b=np.zeros((1, 1)), x0=np.array([0.2]), G=np.zeros((2, m, n + 1)), h=np.ones((2, m)))
            data['H'][0, :n, :n] = Q
            data['c'][0, :n] = cv
            data['G'][0, :, :n] = GG
            data['h'][0] = hv
            models.append((layout, data, GG, hv))
        for label, mod in mods.items():
            for be in ['c', 'cpp']:
                solver = mod.Solver(models[0][1], backend=be, tolerance=1e-08, structural_presolve=False)
                for layout, data, GG, hv in models:
                    solver.update(data)
                    res = solver.solve()
                    u = np.asarray(res['u'])[0]
                    z = np.asarray(res['z'])[0]
                    ss = np.asarray(res['s'])[0]
                    primal = max(0.0, float(np.max(GG @ u - hv)), float(np.max(np.abs(GG @ u + ss - hv))), float(np.max(-ss)))
                    dual = max(float(np.max(np.abs(Q @ u + cv + GG.T @ z))), float(np.max(-z)), 0.0)
                    gap = float(abs(ss @ z))
                    obj = float(0.5 * u @ Q @ u + cv @ u)
                    expected = float(0.5 * xs @ Q @ xs + cv @ xs)
                    passed = res['status'] == 0 and max(primal, dual, gap) <= 1e-08 and (abs(obj - expected) <= 2e-08)
                    rows.append(dict(n=n, kind=kind, layout=layout, solver=label, backend=be, status=int(res['status']), iterations=int(res['iterations']), primal=primal, dual=dual, gap=gap, objective_error=abs(obj - expected), primal_solution_error=float(np.max(abs(u - xs))), passed=passed))
                solver.close()
save(R / 'smoke-results.json', dict(seed=24110109, evidence='New analytic KKT correctness checks only; no unseen application/performance claim', cases=len(rows) // 2, solves=len(rows), all_passed=all((x['passed'] for x in rows)), rows=rows))
print('SPAN CONTRACT', len(rows), all((x['passed'] for x in rows)), flush=True)
assert all((x['passed'] for x in rows))
