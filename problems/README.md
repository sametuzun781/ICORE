# Example coefficients

This directory contains **393 distinct coefficient instances** in compressed NumPy
`.npz` files. `catalog.json` records their names, dimensions, representations, options,
source groups, and aliases. It is read automatically; do not edit internal identity
fields or include stored reference solutions as solver inputs.

| Group | Count |
|---|---:|
| Main benchmark collection | 175 |
| Additional validation models | 72 |
| Distinct coefficient-update states | 32 |
| Analytic KKT examples | 114 |
| Total | 393 |

The original 100 files remain here; 293 additional files are under `extended/`.
This is a count of coefficient instances, not independent application families.
Numeric arrays are unchanged from the preceding ICORE distribution.

From the repository root:

```python
from example_tools.catalog import list_problems, load_problem
from icore import Solver

names = list_problems(contains="LIPM")
example = load_problem(names[0]["name"])
with Solver(example.coefficients, backend="cpp", **example.options,
            **example.metadata.get("solver_settings", {})) as solver:
    result = solver.solve()
```

For your own problem, use the independent notebooks in `tutorials/`: they show how
to construct arrays rather than loading an opaque saved example.
