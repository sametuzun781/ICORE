# The small public API

```python
from icore import Solver

with Solver(problem, backend="cpp", tolerance=1e-8) as solver:
    result = solver.solve()
    solver.update(updated_problem)
    next_result = solver.solve()
```

`solver.solve(updated_problem)` is equivalent to update then solve. Do not mutate
private workspace buffers. `close()` releases native memory; the `with` statement
handles this even if a check raises an exception.

## OCP convention

For k = 0,...,N-1:

- Dynamics: `x[k+1] = A[k] @ x[k] + B[k] @ u[k] + b[k]`.
- Initial state: `x[0] = x0`.
- Stage variables in constraints: **`[u, x]`**.
- Quadratic cost: `0.5*x@Q@x + u@S@x + 0.5*u@R@u + q@x + r@u`.
- Terminal cost: `0.5*x[N]@Q[N]@x[N] + q[N]@x[N]`.
- Optional terminal equality: `F @ x[N] = f`.
- Constraints: `h[k] - G[k] @ [u[k], x[k]]` lies in the specified cone.

The native tutorials use these explicit arrays:

| Array | Shape |
|---|---|
| `A`, `B`, `b` | `(N,nx,nx)`, `(N,nx,nu)`, `(N,nx)` |
| `x0` | `(nx,)` |
| `Q`, `q` | `(N+1,nx,nx)`, `(N+1,nx)` |
| `R`, `S`, `r` | `(N,nu,nu)`, `(N,nu,nx)`, `(N,nu)` |
| `G`, `h` | `(N+1,m,nu+nx)`, `(N+1,m)` |
| `F`, `f` | `(nf,nx)`, `(nf,)` |

An empty `F=np.empty((0,nx))`, `f=np.empty(0)` means no terminal equality.
At the terminal node there is no control; keep its control columns zero.
Supply symmetric cost matrices; their upper triangles define the symmetric operator.
Convexity concerns the full block Hessian including `S`, not just Q and R.
The coefficient interface is not an arbitrary symbolic/nonlinear modeling language.

Without cone options every row is a scalar nonnegative-cone row, i.e. `Gv <= h`.
For two orthant rows and two 3D SOCs use:

```python
solver = Solver(problem, nonnegative=2, soc_sizes=(3, 3))
```

Use the same cone layout per stage. Each SOC block is `(radius, vector)` with
`radius >= norm(vector)`. Tutorial 5 shows signs and terminal padding explicitly.
Other native representations include `H/c` in `[u,x]` order and `ChainQP` for
adjacent-stage affine residual/epigraph models; see the existing methods book.

## Result and timing

`result['x']` has N+1 state rows; `result['u']` has N control rows. Status zero denotes
the solver's success status. The tutorials also use an independent original-model
KKT check; do not infer success solely from an attractive plot.

`result['info']['seconds']` is native solver time in seconds. Multiply by 1000 for
milliseconds. It excludes Python setup, compilation, independent checks, and plots.
The comparison notebook's `solve_ms` has a broader documented scope, including owned
original-coordinate outputs. Do not mix the two measurements.

## Plan versus actual execution

`solver.factorization_plan` explains the policy and optional exact reduction.
`structural_presolve=True` means it was requested, not necessarily applied.
`selection_reason == 'exact_coefficient_eligible_ocp_reduction'` confirms that reduction.

`solver.diagnostics` reads counters from the last native solve. Its `last_kernel`
identifies general OCP/Riccati, dense QP, or affine chain. Use it rather than assuming
that every OCP-labeled input executed a Riccati factorization.

`example_tools.tutorial.explain_solution` combines these with an original-model audit.
It runs outside the solve. `detailed=True` uses a separate instrumented copy and one
extra solve to record selected factorization, precision, fallback and rejection events.
No event is inferred merely from a policy. Replay equality and event integrity are checked.

## Repeated solves

A reused workspace does not imply a previous-iterate warm start. Each solve follows
the existing cold-start policy. Updates refresh/invalidate cached data according to
coefficients, not just dimensions. See the independent MPC notebook.
