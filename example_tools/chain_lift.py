"""Exact, explicit ChainQP-to-OCP lifting for the unmodified historical solver.

This is a model adapter, not an alternative numerical solver. If the original
chain has T nodes y[k], the lifted OCP has T controls and states x[k+1]=y[k].
At stage k, control [w[k],t[k-1]] holds the current free node coordinates and
the previous edge's epigraphs, while x[k] holds the previous node. Fixed
coordinates are substituted. Unused padded control coordinates have independent
unit quadratic costs whose unique minimizer is zero. They never constrain or
regularize a physical variable. Stage-zero dummy epigraphs use the same rule.
Feasible constant inequalities are omitted and restored with exact slack and
zero dual. The original ChainQP and its independent audit remain unchanged.
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np


class ChainLift:
    def __init__(self, coefficients):
        self.original = coefficients
        self.qp = q = SimpleNamespace(**coefficients)
        self.T, self.d = T, d = q.center.shape
        self.r = r = q.offset.shape[1]
        self.nx, self.nu = d, d + r
        weights = np.broadcast_to(q.l1_weight, q.offset.shape)
        if T < 2 or not np.all((q.free == 0) | (q.free == 1)):
            raise ValueError("Expected at least two nodes and a binary free mask")
        if np.any(q.hessian < 0) or np.any(q.l1_weight < 0):
            raise ValueError("Expected convex diagonal objective and nonnegative L1 weights")
        self.anchor = (1. - q.free) * q.fixed
        nx, nu = self.nx, self.nu
        Q = np.zeros((T+1, nx, nx))
        R = np.zeros((T, nu, nu))
        S = np.zeros((T, nu, nx))
        linear_x = np.zeros((T+1, nx))
        linear_u = np.zeros((T, nu))
        A = np.zeros((T, nx, nx))
        B = np.zeros((T, nx, nu))
        b = self.anchor.copy()
        self.dummy = np.zeros((T, nu), dtype=bool)
        for k in range(T):
            R[k, np.arange(d), np.arange(d)] = np.where(q.free[k], q.hessian[k], 1.)
            linear_u[k, :d] = q.free[k] * (q.linear[k] - q.hessian[k] * q.center[k])
            B[k, np.arange(d), np.arange(d)] = q.free[k]
            self.dummy[k, :d] = q.free[k] == 0
            if k:
                linear_u[k, d:] = weights[k-1]
            else:
                R[k, d:, d:] = np.eye(r)
                self.dummy[k, d:] = True
        self.constant = float(np.sum(.5*q.hessian*(self.anchor-q.center)**2 + q.linear*self.anchor))
        G, h, self.rows = [], [], []
        self.omitted = []

        def append_stage(k, matrix, rhs, index):
            active = np.any(matrix != 0, axis=1)
            for j in np.flatnonzero(~active):
                if rhs[j] < 0:
                    raise ValueError(f"Infeasible constant inequality at stage {k}: {index[j]}")
                self.omitted.append(index[j])
            G.append(np.ascontiguousarray(matrix[active]))
            h.append(np.ascontiguousarray(rhs[active]))
            self.rows.append([item for item, keep in zip(index, active) if keep])

        for k in range(T):
            matrices, right_sides, indexes = [], [], []
            if k:
                j = k-1
                matrix = np.zeros((r, nu+nx))
                matrix[:, :d] = q.right[j] * q.free[k]
                matrix[:, d:nu] = -np.eye(r)
                matrix[:, nu:] = q.left[j] * q.free[j]
                shift = q.left[j] @ self.anchor[j] + q.right[j] @ self.anchor[k] + q.offset[j]
                matrices.append(matrix)
                right_sides.append(-shift)
                indexes.extend((0, j, a) for a in range(r))
                negative = -matrix.copy()
                negative[:, d:nu] = -np.eye(r)
                matrices.append(negative)
                right_sides.append(shift)
                indexes.extend((1, j, a) for a in range(r))
                n_pair = q.pair_offset.shape[1]
                matrix = np.zeros((n_pair, nu+nx))
                matrix[:, :d] = q.pair_right[j] * q.free[k]
                matrix[:, nu:] = q.pair_left[j] * q.free[j]
                matrices.append(matrix)
                right_sides.append(-(q.pair_left[j] @ self.anchor[j] + q.pair_right[j] @ self.anchor[k] + q.pair_offset[j]))
                indexes.extend((2, j, a) for a in range(n_pair))
            n_local = q.local_offset.shape[1]
            matrix = np.zeros((n_local, nu+nx))
            matrix[:, :d] = q.local_matrix[k] * q.free[k]
            matrices.append(matrix)
            right_sides.append(-(q.local_matrix[k] @ self.anchor[k] + q.local_offset[k]))
            indexes.extend((3, k, a) for a in range(n_local))
            append_stage(k, np.vstack(matrices), np.concatenate(right_sides), indexes)
        G.append(np.zeros((0, nu+nx)))
        h.append(np.zeros(0))
        self.rows.append([])
        self.coefficients = dict(Q=Q, R=R, S=S, q=linear_x, r=linear_u,
            A=A, B=B, b=b, G=G, h=h, x0=np.zeros(nx), F=np.zeros((0,nx)), f=np.zeros(0))
        self.nonnegative = [len(a) for a in h]
        self.soc_sizes = [() for _ in h]
        self.restore_maps = []
        for entries in self.rows:
            groups = []
            for group in range(4):
                selected = [(j,row,col) for j,(g,row,col) in enumerate(entries) if g==group]
                if selected:
                    source, row, col = (np.array(a,dtype=int) for a in zip(*selected))
                    groups.append((group, source, row, col))
            self.restore_maps.append(groups)

    def original_constraints(self, y, t):
        q = self.qp
        defect = np.einsum('kij,kj->ki', q.left, y[:-1])
        defect += np.einsum('kij,kj->ki', q.right, y[1:]) + q.offset
        pair = np.einsum('kij,kj->ki', q.pair_left, y[:-1])
        pair += np.einsum('kij,kj->ki', q.pair_right, y[1:]) + q.pair_offset
        local = np.einsum('kij,kj->ki', q.local_matrix, y) + q.local_offset
        return defect-t, -defect-t, pair, local

    def decode(self, result):
        q = self.qp
        control = np.asarray(result['u'])
        y = self.anchor + q.free * control[:, :self.d]
        t = control[1:, self.d:].copy()
        constraints = self.original_constraints(y, t)
        slacks = [-a.copy() for a in constraints]
        duals = [np.zeros_like(a) for a in constraints]
        for k, maps in enumerate(self.restore_maps):
            for group, source, row, column in maps:
                slacks[group][row,column] = np.asarray(result['s'][k])[source]
                duals[group][row,column] = np.asarray(result['z'][k])[source]
        return dict(stage_y=y.copy(), epigraph=t,
                    s=tuple(slacks), z=tuple(duals))

    def receipt(self):
        q = self.qp
        return dict(original_nodes=self.T, original_width=self.d,
            original_free_variables=int(np.sum(q.free)), original_epigraph_variables=int(q.offset.size),
            lifted_horizon=self.T, lifted_state=self.nx, lifted_control=self.nu,
            lifted_decision_variables=self.T*(self.nx+self.nu),
            lifted_equality_rows=self.T*self.nx,
            independent_dummy_controls=int(np.sum(self.dummy)),
            original_inequality_rows=2*q.offset.size+q.pair_offset.size+q.local_offset.size,
            lifted_inequality_rows=sum(self.nonnegative),
            omitted_feasible_constant_rows=len(self.omitted),
            objective_constant=self.constant,
            physical_coefficients_changed=False,
            method='one-node memory lift, fixed-coordinate substitution, independent zero-optimum dummy padding')

    def check_equivalence(self, seed=20261023, trials=3):
        """Check objective/constraint identities and dual pullback, no solver."""
        rng = np.random.default_rng(seed)
        q, p = self.qp, self.coefficients
        max_objective = max_constraint = max_gradient = 0.
        for _ in range(trials):
            u = rng.normal(size=(self.T, self.nu))
            # Include nonzero dummy controls and account for their explicit cost.
            y = self.anchor + q.free*u[:, :self.d]
            x = np.vstack((np.zeros((1, self.d)), y))
            t = u[1:, self.d:]
            constraints = self.original_constraints(y,t)
            original_obj = float(np.sum(.5*q.hessian*(y-q.center)**2 + q.linear*y)
                                 + np.sum(q.l1_weight*t))
            lifted_obj = float(.5*np.einsum('ki,kij,kj->',u,p['R'],u) + np.sum(p['r']*u))
            target = original_obj + .5*float(np.sum(u[self.dummy]**2))
            error = abs(lifted_obj+self.constant-target)/(1+abs(target)+abs(lifted_obj)+abs(self.constant))
            max_objective = max(max_objective,error)
            weights = [np.zeros_like(a) for a in constraints]
            pulled = np.zeros_like(y)
            for k, entries in enumerate(self.rows):
                v = np.r_[u[k] if k<self.T else np.zeros(self.nu),x[k]]
                actual = p['G'][k]@v-p['h'][k]
                expect = np.array([constraints[g][row,col] for g,row,col in entries])
                scale = 1+np.abs(p['G'][k])@np.abs(v)+np.abs(p['h'][k])
                if actual.size:
                    max_constraint=max(max_constraint,float(np.max(np.abs(actual-expect)/scale)))
                z = rng.normal(size=len(entries))
                for value,(g,row,col) in zip(z,entries):
                    weights[g][row,col]=value
                gv = p['G'][k].T@z
                if k<self.T:
                    pulled[k] += q.free[k]*gv[:self.d]
                if k:
                    pulled[k-1] += q.free[k-1]*gv[self.nu:]
            difference=weights[0]-weights[1]
            expected=np.einsum('kji,kj->ki',q.local_matrix,weights[3])
            expected[:-1] += np.einsum('kji,kj->ki',q.left,difference)
            expected[1:] += np.einsum('kji,kj->ki',q.right,difference)
            expected[:-1] += np.einsum('kji,kj->ki',q.pair_left,weights[2])
            expected[1:] += np.einsum('kji,kj->ki',q.pair_right,weights[2])
            expected *= q.free
            scale=1+np.max(np.abs(pulled),initial=0)+np.max(np.abs(expected),initial=0)
            max_gradient=max(max_gradient,float(np.max(np.abs(pulled-expected),initial=0)/scale))
        result=dict(objective_relative_error=max_objective, constraint_scaled_error=max_constraint,
                    dual_pullback_scaled_error=max_gradient, trials=trials)
        assert max(result[k] for k in result if k!='trials') < 2e-14, result
        return result
