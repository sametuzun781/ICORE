# Six independent notebooks

Start at `01_scalar_regulator.ipynb`. Each notebook can run in a fresh kernel;
none depends on executing another notebook or importing a hidden model builder.

The sequence is scalar regulation, terminal tracking, input-rate limits, soft
constraints, SOC limits, and an MPC loop. Open the repository root in Jupyter,
then run a notebook top to bottom. Imports also work when Jupyter starts here.

`BACKEND` selects C or C++; `USE_PRESOLVE` requests optional exact reduction.
Reports distinguish requesting presolve from actually applying it. `DETAILED=True`
adds a separate diagnostic replay. Model building, auditing, plots, and replay are
not part of reported native solver time.

A successful optimizer return is independently checked in the original model's
units before results are plotted. Saved plots are examples from the validated run,
not expected performance on every machine.
