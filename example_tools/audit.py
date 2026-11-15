"""Independent original-model audits and exact coordinate mappings.
Retained from the existing 175-case benchmark, not optimization kernels.
"""

from dataclasses import dataclass

from types import SimpleNamespace

import numpy as np

from scipy import sparse

@dataclass
class CanonicalQP:
    P: object
    c: np.ndarray
    A: object
    b: np.ndarray
    G: object
    h: np.ndarray
    nonnegative: int
    soc_sizes: tuple
    constant: float
    equality_rows: np.ndarray
    cone_rows: np.ndarray
    shape: tuple

@dataclass
class PhysicalCanonical:
    problem: CanonicalQP
    stage_indices: list
    cone_order: np.ndarray
    N: int
    nx: int
    nu: int

    def vectors(self,result):
        """Pack full native/JAX raw result into canonical order; owned copies."""
        if hasattr(result,'_asdict'):result=result._asdict()
        X=np.asarray(result.get('X',result.get('x')));U=np.asarray(result.get('U',result.get('u')))
        middle=np.concatenate((U[1:],X[1:self.N]),axis=1).ravel()
        x=np.concatenate((U[0],middle,X[self.N]))
        y=np.r_[np.asarray(result['y']).ravel(),np.asarray(result['nu']).ravel()]
        s=np.asarray(result['s']).ravel()[self.cone_order].copy()
        z=np.asarray(result['z']).ravel()[self.cone_order].copy()
        return x,y.copy(),s,z

def canonicalize(p,nlp,dims):
    N=len(p['Q'])-1;nx=p['Q'].shape[1];nu=p['R'].shape[1]
    ns,nc=p['h'].shape;nv=nx+nu
    if ns not in (N,N+1):raise ValueError('Unexpected cone stage count')
    blocks=[];linear=[];indices=[];cursor=0
    for k in range(N+1):
        idx=np.full(nv,-1,dtype=int)
        if k==0:
            idx[:nu]=np.arange(cursor,cursor+nu);cursor+=nu
            blocks.append(p['R'][k]);linear.append(p['r'][k]+p['S'][k]@p['x0'])
        elif k==N:
            idx[nu:]=np.arange(cursor,cursor+nx);cursor+=nx
            blocks.append(p['Q'][k]);linear.append(p['q'][k])
        else:
            idx[:]=np.arange(cursor,cursor+nv);cursor+=nv
            blocks.append(np.block([[p['R'][k],p['S'][k]],[p['S'][k].T,p['Q'][k]]]))
            linear.append(np.r_[p['r'][k],p['q'][k]])
        indices.append(idx)
    P=sparse.block_diag(blocks,format='csc');c=np.concatenate(linear)
    P.eliminate_zeros();P.sort_indices()
    constant=float(.5*p['x0']@p['Q'][0]@p['x0']+p['q'][0]@p['x0'])
    nf=len(p['f']);A=sparse.lil_matrix((N*nx+nf,cursor));b=np.r_[-p['b'][:N].ravel(),p['f']]
    for k in range(N):
        rr=slice(k*nx,(k+1)*nx)
        A[rr,indices[k][:nu]]=p['B'][k]
        if k:A[rr,indices[k][nu:]]=p['A'][k]
        else:b[rr]-=p['A'][k]@p['x0']
        A[rr,indices[k+1][nu:]]=-np.eye(nx)
    if nf:A[N*nx:,indices[N][nu:]]=p['F']
    G=sparse.lil_matrix((ns*nc,cursor));h=p['h'].copy()
    for k in range(ns):
        rr=slice(k*nc,(k+1)*nc)
        active=indices[k]>=0
        G[rr,indices[k][active]]=p['G'][k][:,active]
        if k==0:h[k]-=p['G'][k,:,nu:]@p['x0']
    grid=np.arange(ns*nc).reshape(ns,nc)
    order=np.r_[grid[:,:nlp].ravel(),grid[:,nlp:].ravel()]
    A=A.tocsc();G=G.tocsr()[order].tocsc()
    for m in (A,G):m.eliminate_zeros();m.sum_duplicates();m.sort_indices()
    problem=CanonicalQP(P,c,A,b,G,h.ravel()[order],ns*nlp,tuple(dims)*ns,
                        constant,np.arange(N*nx+nf),order,(cursor,))
    return PhysicalCanonical(problem,indices,order,N,nx,nu)

def audit_canonical(problem, vectors):
    """Identical, unscaled canonical KKT tests for both solvers.

    Absolute cone-pairing SUM, not maximum scalar complementarity. This is
    a numerical audit, not an exact objective-error certificate for infeasible data.
    """
    x, y, slack, dual = vectors
    if not all(np.isfinite(value).all() for value in vectors):
        return dict(primal=np.inf, dual=np.inf, gap=np.inf, objective=np.nan)
    largest = lambda a: float(np.max(np.abs(a), initial=0.))
    feasible_slack = problem.h - problem.G @ x
    stationarity = problem.P @ x + problem.c + problem.A.T @ y + problem.G.T @ dual
    primal = max(largest(problem.A @ x - problem.b), largest(feasible_slack - slack))
    l = problem.nonnegative
    for s in (slack, feasible_slack):
        primal = max(primal, float(np.max(-s[:l], initial=0.)))
    dual_cone = float(np.max(-dual[:l], initial=0.))
    gap = float(np.sum(np.abs(slack[:l] * dual[:l])))
    for size in problem.soc_sizes:
        s, z, f = slack[l:l+size], dual[l:l+size], feasible_slack[l:l+size]
        primal = max(primal, float(np.linalg.norm(s[1:]) - s[0]),
                     float(np.linalg.norm(f[1:]) - f[0]))
        dual_cone = max(dual_cone, float(np.linalg.norm(z[1:]) - z[0]))
        gap += abs(float(s @ z))
        l += size
    objective = float(.5 * x @ (problem.P @ x) + problem.c @ x + problem.constant)
    finite = all(np.isfinite(a).all() for a in (*vectors, stationarity, objective))
    return dict(primal=primal if finite else np.inf,
                dual=max(largest(stationarity), dual_cone) if finite else np.inf,
                gap=gap if finite else np.inf, objective=objective)

def _maximum_abs(x):
    return float(np.max(np.abs(x), initial=0.))

def _cone_violation(values, orthant, cones):
    violation = float(np.maximum(-values[:, :orthant], 0.).max(initial=0.))
    offset = orthant
    for dimension in cones:
        margin = values[:, offset] - np.linalg.norm(values[:, offset + 1:offset + dimension], axis=1)
        violation = max(violation, float(np.maximum(-margin, 0.).max(initial=0.)))
        offset += dimension
    if offset != values.shape[1]:
        raise ValueError("Cone layout does not match output")
    return violation

def audit_physical(p, orthant, cones, result, *, tolerance=None,
                   reference_objective=None, status_ok=None):
    """Original unscaled KKT audit using direct physical stage algebra.

    Dynamics multipliers correspond to Ax + Bu + b - x_next = 0.
    The initial state is fixed, so its stationarity equation is eliminated.
    This implementation uses no solver numerical routine or cached factor.
    """
    X, U, Y, Nu, Slack, Dual = (np.asarray(result[k]) for k in ("x", "u", "y", "nu", "s", "z"))
    if not all(np.isfinite(a).all() for a in (X, U, Y, Nu, Slack, Dual)):
        return dict(primal=float("inf"), dual=float("inf"), gap=float("inf"),
                    objective=float("nan"), objective_error=float("inf"), accepted=False)
    N, nu = len(U), U.shape[1]
    # Symmetric matrices are defined by their upper triangles in the public API.
    Q = np.triu(p["Q"]) + np.swapaxes(np.triu(p["Q"], 1), -1, -2)
    R = np.triu(p["R"][:N]) + np.swapaxes(np.triu(p["R"][:N], 1), -1, -2)
    state_gradient = np.einsum("kij,kj->ki", Q, X) + p["q"]
    control_gradient = np.einsum("kij,kj->ki", R, U) + p["r"][:N]
    state_gradient[:N] += np.einsum("kji,kj->ki", p["S"][:N], U)
    control_gradient += np.einsum("kij,kj->ki", p["S"][:N], X[:N])
    objective = float(.5 * np.einsum("ki,kij,kj->", X, Q, X)
                      + .5 * np.einsum("ki,kij,kj->", U, R, U)
                      + np.einsum("ki,kij,kj->", U, p["S"][:N], X[:N])
                      + np.sum(p["q"] * X) + np.sum(p["r"][:N] * U))
    dynamics = np.einsum("kij,kj->ki", p["A"][:N], X[:N])
    dynamics += np.einsum("kij,kj->ki", p["B"][:N], U) + p["b"][:N] - X[1:]
    terminal = p["F"] @ X[-1] - p["f"]
    state_gradient[:N] += np.einsum("kji,kj->ki", p["A"][:N], Y)
    state_gradient[1:] -= Y
    state_gradient[-1] += p["F"].T @ Nu
    control_gradient += np.einsum("kji,kj->ki", p["B"][:N], Y)
    stage = np.column_stack((np.vstack((U, np.zeros((1, nu)))), X))[:len(Slack)]
    slack_equation = np.einsum("kij,kj->ki", p["G"], stage) + Slack - p["h"]
    cone_gradient = np.einsum("kji,kj->ki", p["G"], Dual)
    control_gradient += cone_gradient[:N, :nu]
    state_gradient[:len(Slack)] += cone_gradient[:, nu:]
    primal_cone = _cone_violation(Slack, orthant, cones)
    dual_cone = _cone_violation(Dual, orthant, cones)
    primal = max(_maximum_abs(X[0] - p["x0"]), _maximum_abs(dynamics),
                 _maximum_abs(terminal), _maximum_abs(slack_equation), primal_cone)
    dual = max(_maximum_abs(control_gradient), _maximum_abs(state_gradient[1:]), dual_cone)
    # Absolute pairing per cone block prevents cancellation between blocks.
    # Every orthant coordinate is its own one-dimensional cone.
    gap = float(np.sum(np.abs(Slack[:, :orthant] * Dual[:, :orthant])))
    offset = orthant
    for dimension in cones:
        pairing = np.sum(Slack[:, offset:offset + dimension] * Dual[:, offset:offset + dimension], axis=1)
        gap += float(np.sum(np.abs(pairing)))
        offset += dimension
    error = abs(objective - reference_objective) if reference_objective is not None else None
    answer = dict(primal=primal, dual=dual, gap=gap, objective=objective,
                  objective_error=error, dynamics=_maximum_abs(dynamics),
                  terminal=_maximum_abs(terminal), slack_equation=_maximum_abs(slack_equation),
                  primal_cone=primal_cone, dual_cone=dual_cone)
    if tolerance is not None:
        answer["accepted"] = bool(max(primal, dual, gap) <= tolerance
            and (status_ok is None or status_ok)
            and (error is None or error <= 10 * tolerance * (1 + abs(reference_objective))))
    return answer

def audit_chain(qp,result,tolerance=1e-8,reference=None):
    y,t=np.asarray(result['stage_y']),np.asarray(result['epigraph'])
    s=tuple(np.asarray(v) for v in result['s'])
    z=tuple(np.asarray(v) for v in result['z'])
    maximum=lambda a:float(np.max(np.abs(a),initial=0.))
    if not all(np.isfinite(a).all() for a in (y,t,*s,*z)):
        return dict(primal=float('inf'),dual=float('inf'),gap=float('inf'),accepted=False)
    defect=np.einsum('kij,kj->ki',qp.left,y[:-1])+np.einsum('kij,kj->ki',qp.right,y[1:])+qp.offset
    pair=np.einsum('kij,kj->ki',qp.pair_left,y[:-1])+np.einsum('kij,kj->ki',qp.pair_right,y[1:])+qp.pair_offset
    local=np.einsum('kij,kj->ki',qp.local_matrix,y)+qp.local_offset
    constraints=(defect-t,-defect-t,pair,local)
    gradient=qp.hessian*(y-qp.center)+qp.linear
    gradient[:-1]+=np.einsum('kij,ki->kj',qp.left,z[0]-z[1])+np.einsum('kij,ki->kj',qp.pair_left,z[2])
    gradient[1:]+=np.einsum('kij,ki->kj',qp.right,z[0]-z[1])+np.einsum('kij,ki->kj',qp.pair_right,z[2])
    gradient+=np.einsum('kij,ki->kj',qp.local_matrix,z[3])
    primal=max(maximum((1-qp.free)*(y-qp.fixed)),*(maximum(g+v) for g,v in zip(constraints,s)),*(float(np.max(-v,initial=0.)) for v in s))
    dual=max(maximum(qp.free*gradient),maximum(qp.l1_weight-z[0]-z[1]),*(float(np.max(-v,initial=0.)) for v in z))
    gap=sum(float(np.sum(np.abs(a*b))) for a,b in zip(s,z))
    objective=float(.5*np.sum(qp.hessian*(y-qp.center)**2)+np.sum(qp.linear*y)+np.sum(qp.l1_weight*np.abs(defect)))
    hard=max(maximum((1-qp.free)*(y-qp.fixed)),float(np.max(pair,initial=0.)),float(np.max(local,initial=0.)))
    error=None if reference is None else abs(objective-reference)
    return dict(primal=primal,dual=dual,gap=gap,objective=objective,objective_error=error,
                hard_constraint_violation=hard,accepted=bool(int(result['status'])==0 and max(primal,dual,gap)<=tolerance
                and (error is None or error<=10*tolerance*(1+abs(reference)))))

def chain_objective(qp, y):
    defect = (np.einsum('kij,kj->ki', qp.left, y[:-1])
              + np.einsum('kij,kj->ki', qp.right, y[1:]) + qp.offset)
    return float(.5*np.sum(qp.hessian*(y-qp.center)**2)
                 + np.sum(qp.linear*y) + np.sum(qp.l1_weight*np.abs(defect)))

def restore_ocp(case, canonical, result):
    """Undo fixed-x0 elimination and cone ordering of the sparse QOCO model."""
    N, nx, nu = canonical.N, canonical.nx, canonical.nu
    stages = np.zeros((N+1, nx+nu))
    stages[0, nu:] = case.coefficients['x0']
    for k, indices in enumerate(canonical.stage_indices):
        active = indices >= 0
        stages[k, active] = result['x'][indices[active]]
    slack, dual = (np.empty_like(result[key]) for key in ('s', 'z'))
    slack[canonical.cone_order] = result['s']
    dual[canonical.cone_order] = result['z']
    shape = case.coefficients['h'].shape
    return dict(x=stages[:, nu:].copy(), u=stages[:-1, :nu].copy(),
        y=result['y'][:N*nx].reshape(N, nx).copy(), nu=result['y'][N*nx:].copy(),
        s=slack.reshape(shape), z=dual.reshape(shape))

def restore_chain(case, layout, canonical, result):
    """Undo residual/epigraph lifting and restore omitted feasible constant rows."""
    qp = case.qp
    y = layout.decode(result, canonical)
    count = canonical.epigraphs
    t = result['x'][layout.nfree+count:].reshape(qp.offset.shape).copy()
    defect = (np.einsum('kij,kj->ki', qp.left, y[:-1])
              + np.einsum('kij,kj->ki', qp.right, y[1:]) + qp.offset)
    pair = (np.einsum('kij,kj->ki', qp.pair_left, y[:-1])
            + np.einsum('kij,kj->ki', qp.pair_right, y[1:]) + qp.pair_offset)
    local = np.einsum('kij,kj->ki', qp.local_matrix, y) + qp.local_offset
    constraints = (defect-t, -defect-t, pair, local)
    slack = np.concatenate([-value.ravel() for value in constraints])
    dual = np.zeros_like(slack)
    slack[canonical.keep] = result['s']
    dual[canonical.keep] = result['z']
    cuts = np.cumsum([value.size for value in constraints])[:-1]
    groups = lambda value: tuple(part.reshape(shape.shape)
        for part, shape in zip(np.split(value, cuts), constraints))
    return dict(stage_y=y.copy(), epigraph=t, s=groups(slack), z=groups(dual))

def chain_canonical_audit(case, canonical, result):
    qp = case.qp
    fixed = canonical.fixed
    constant = float(.5*np.sum(qp.hessian*(fixed-qp.center)**2) + np.sum(qp.linear*fixed))
    model = SimpleNamespace(**vars(canonical), nonnegative=canonical.l,
                            soc_sizes=canonical.cone_sizes, constant=constant)
    return audit_canonical(model, tuple(result[key] for key in ('x', 'y', 's', 'z')))
