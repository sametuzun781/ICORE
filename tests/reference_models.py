"""Fixed reference models retained for regression tests.

The notebooks build their own arrays inline and do not import this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class TutorialCase:
    name: str
    model: dict[str, np.ndarray]
    nonnegative: int
    soc_sizes: tuple[int, ...] = ()
    objective_offset: float = 0.0

    @property
    def options(self) -> dict[str, Any]:
        return dict(nonnegative=self.nonnegative, soc_sizes=self.soc_sizes)



def allocate_ocp(N: int, nx: int, nu: int, rows: int) -> dict[str, np.ndarray]:
    """Allocate a block-form model. Terminal control columns stay unused.

    A/B/b have N stages; Q/q/G/h have N+1. R/S/r have N.
    Zero G rows with strictly interior h are harmless terminal padding;
    this keeps the cone layout the same at every stage.
    """
    if min(N, nx, nu) < 1 or rows < 0:
        raise ValueError("Require N, nx, nu >= 1 and rows >= 0")
    return dict(
        Q=np.zeros((N + 1, nx, nx)), R=np.zeros((N, nu, nu)),
        S=np.zeros((N, nu, nx)), q=np.zeros((N + 1, nx)), r=np.zeros((N, nu)),
        A=np.zeros((N, nx, nx)), B=np.zeros((N, nx, nu)), b=np.zeros((N, nx)),
        G=np.zeros((N + 1, rows, nu + nx)), h=np.ones((N + 1, rows)),
        x0=np.zeros(nx), F=np.empty((0, nx)), f=np.empty(0),
    )


def scalar_regulator() -> TutorialCase:
    """T1: x[k+1] = x[k] + 0.25*u[k], |u[k]| <= 1."""
    N, dt = 20, 0.25
    p = allocate_ocp(N, nx=1, nu=1, rows=2)
    p["A"][:] = 1.0
    p["B"][:] = dt
    p["Q"][:] = 1.0
    p["Q"][-1] = 8.0
    p["R"][:] = 0.2
    p["x0"][:] = 2.5
    # G @ [u, x] <= h: +u <= 1 and -u <= 1.
    p["G"][:N, 0, 0] = 1.0
    p["G"][:N, 1, 0] = -1.0
    return TutorialCase("T1_scalar_regulator", p, nonnegative=2)


def terminal_tracking() -> TutorialCase:
    """T2: position/velocity tracking, bounds, and F @ x[N] = f."""
    N, dt = 30, 0.2
    p = allocate_ocp(N, nx=2, nu=1, rows=4)
    p["A"][:] = [[1.0, dt], [0.0, 1.0]]
    p["B"][:] = [[0.5 * dt**2], [dt]]
    p["Q"][:] = np.diag([4.0, 0.3])
    p["Q"][-1] = np.diag([20.0, 2.0])
    p["R"][:] = 0.15
    reference = np.array([2.0, 0.0])
    p["q"][:] = -np.einsum("kij,j->ki", p["Q"], reference)
    p["F"], p["f"] = np.eye(2), reference.copy()
    p["G"][:N, 0, 0], p["G"][:N, 1, 0] = 1.0, -1.0
    p["h"][:, :2] = 1.2
    p["G"][:, 2, 2], p["G"][:, 3, 2] = 1.0, -1.0  # velocity
    p["h"][:, 2:] = 1.0
    offset = float(0.5 * np.einsum("i,kij,j->", reference, p["Q"], reference))
    return TutorialCase("T2_terminal_tracking", p, 4, objective_offset=offset)


def rate_limited_tracking() -> TutorialCase:
    """T3: keep delta-u local by adding the previous input to the state."""
    N, dt, eta, delta_max = 30, 0.2, 1.5, 0.3
    p = allocate_ocp(N, nx=3, nu=1, rows=6)
    # x = [position, velocity, previous acceleration]; u = acceleration.
    p["A"][:] = [[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    p["B"][:] = [[0.5 * dt**2], [dt], [1.0]]
    p["Q"][:] = np.diag([4.0, 0.3, eta])
    p["Q"][-1] = np.diag([20.0, 2.0, 0.0])
    p["R"][:] = 0.15 + eta
    p["S"][:, 0, 2] = -eta  # 0.5*eta*(u - previous_u)^2
    reference = np.array([2.0, 0.0, 0.0])
    p["q"][:] = -np.einsum("kij,j->ki", p["Q"], reference)
    p["F"], p["f"] = np.eye(3)[:2], reference[:2].copy()
    p["G"][:N, 0, 0], p["G"][:N, 1, 0] = 1.0, -1.0
    p["h"][:, :2] = 1.2
    p["G"][:, 2, 2], p["G"][:, 3, 2] = 1.0, -1.0
    p["h"][:, 2:4] = 1.0
    p["G"][:N, 4, [0, 3]] = [1.0, -1.0]
    p["G"][:N, 5, [0, 3]] = [-1.0, 1.0]
    p["h"][:, 4:] = delta_max
    offset = float(0.5 * np.einsum("i,kij,j->", reference, p["Q"], reference))
    return TutorialCase("T3_rate_limited_tracking", p, 6, objective_offset=offset)


def soft_corridor() -> TutorialCase:
    """T4: soften the running corridor, not the terminal target or dynamics."""
    N, dt, rho, rho2 = 20, 0.25, 5.0, 0.1
    p = allocate_ocp(N, nx=1, nu=2, rows=5)
    # u = [physical input, epsilon]. Epsilon has no dynamics effect.
    p["A"][:] = 1.0
    p["B"][:, 0, 0] = dt
    p["Q"][:] = 1.0
    p["Q"][-1] = 8.0
    p["R"][:] = np.diag([0.2, rho2])
    p["r"][:, 1] = rho
    p["x0"][:] = 2.5
    p["F"], p["f"] = np.ones((1, 1)), np.zeros(1)
    p["G"][:N, 0, 0], p["G"][:N, 1, 0] = 1.0, -1.0
    p["G"][:N, 2, 1], p["h"][:N, 2] = -1.0, 0.0  # epsilon >= 0
    p["G"][:N, 3, [1, 2]] = [-1.0, 1.0]          # x - epsilon <= 1
    p["G"][:N, 4, [1, 2]] = [-1.0, -1.0]         # -x - epsilon <= 1
    # There is no terminal epsilon. Keep the terminal corridor hard.
    p["G"][N, 3, 2], p["G"][N, 4, 2] = 1.0, -1.0
    return TutorialCase("T4_soft_corridor", p, 5)


def planar_soc() -> TutorialCase:
    """T5: exact Euclidean acceleration/speed bounds plus two linear walls."""
    N, dt, amax, vmax = 30, 0.2, 1.2, 1.0
    p = allocate_ocp(N, nx=4, nu=2, rows=8)
    # x = [px, py, vx, vy], u = [ax, ay].
    p["A"][:] = np.block([[np.eye(2), dt * np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
    p["B"][:] = np.vstack((0.5 * dt**2 * np.eye(2), dt * np.eye(2)))
    p["Q"][:] = np.diag([2.0, 2.0, 0.2, 0.2])
    p["Q"][-1] = np.diag([10.0, 10.0, 1.0, 1.0])
    p["R"][:] = 0.1 * np.eye(2)
    reference = np.array([2.0, 1.0, 0.0, 0.0])
    p["q"][:] = -np.einsum("kij,j->ki", p["Q"], reference)
    p["F"], p["f"] = np.eye(4), reference.copy()
    p["h"][:] = 0.0
    p["G"][:, 0, 3], p["h"][:, 0] = 1.0, 1.4   # py <= 1.4
    p["G"][:, 1, 3], p["h"][:, 1] = -1.0, 0.4  # py >= -0.4
    # h-G[u,x] = [amax, ax, ay] in Q3.
    p["h"][:, 2] = amax
    p["G"][:N, 3:5, :2] = -np.eye(2)
    # h-G[u,x] = [vmax, vx, vy] in Q3.
    p["h"][:, 5] = vmax
    p["G"][:, 6:8, 4:6] = -np.eye(2)
    offset = float(0.5 * np.einsum("i,kij,j->", reference, p["Q"], reference))
    return TutorialCase("T5_planar_soc", p, 2, (3, 3), offset)


BUILDERS = (scalar_regulator, terminal_tracking, rate_limited_tracking, soft_corridor, planar_soc)

