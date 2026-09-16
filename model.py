"""Two-state nonlinear evaporation model used in Zanon & Gros (2020)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import ExperimentConfig


@dataclass
class EvaporatorFlows:
    f2: float
    f4: float
    f5: float
    f100: float
    q100: float
    q200: float


class EvaporatorModel:
    """Simplified nonlinear model with RK4 integration in minutes."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def algebraic(self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray) -> EvaporatorFlows:
        c = self.cfg
        x2, p2 = np.asarray(state, dtype=float)
        p100, f200 = np.asarray(control, dtype=float)
        f1, x1, t1, t200 = np.asarray(disturbance, dtype=float)
        f200 = max(float(f200), 1e-6)

        t2 = c.t2_p_coeff * p2 + c.t2_x_coeff * x2 + c.t2_offset
        t3 = c.t3_p_coeff * p2 + c.t3_offset
        t100 = c.t100_p_coeff * p100 + c.t100_offset
        ua1 = c.ua1_factor * (f1 + c.recirculation_f3)
        q100 = ua1 * (t100 - t2)
        f4 = (q100 - f1 * c.cp * (t2 - t1)) / c.latent_evaporation
        q200 = c.ua2 * (t3 - t200) / (1.0 + c.ua2 / (2.0 * c.cp * f200))
        f5 = q200 / c.latent_evaporation
        f2 = f1 - f4
        f100 = q100 / c.latent_steam
        return EvaporatorFlows(float(f2), float(f4), float(f5), float(f100), float(q100), float(q200))

    def derivative(self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        c = self.cfg
        x2 = float(state[0])
        f1, x1 = float(disturbance[0]), float(disturbance[1])
        flows = self.algebraic(state, control, disturbance)
        return np.array([
            (f1 * x1 - flows.f2 * x2) / c.liquid_holdup,
            (flows.f4 - flows.f5) / c.pressure_capacitance,
        ], dtype=float)

    def step(self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        h = self.cfg.dt_min
        x = np.asarray(state, dtype=float)
        u = np.asarray(control, dtype=float)
        d = np.asarray(disturbance, dtype=float)
        k1 = self.derivative(x, u, d)
        k2 = self.derivative(x + 0.5 * h * k1, u, d)
        k3 = self.derivative(x + 0.5 * h * k2, u, d)
        k4 = self.derivative(x + h * k3, u, d)
        return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def steady_input(self, state: np.ndarray, disturbance: np.ndarray | None = None) -> np.ndarray:
        """Closed-form steady input for a requested (X2,P2) and fixed F3."""
        c = self.cfg
        d = c.disturbance_nominal if disturbance is None else np.asarray(disturbance, dtype=float)
        f1, x1, t1, t200 = d
        x2, p2 = np.asarray(state, dtype=float)
        f4 = f1 * (1.0 - x1 / x2)
        t2 = c.t2_p_coeff * p2 + c.t2_x_coeff * x2 + c.t2_offset
        t3 = c.t3_p_coeff * p2 + c.t3_offset
        q100 = c.latent_evaporation * f4 + f1 * c.cp * (t2 - t1)
        ua1 = c.ua1_factor * (f1 + c.recirculation_f3)
        p100 = (t2 + q100 / ua1 - c.t100_offset) / c.t100_p_coeff
        q200 = c.latent_evaporation * f4
        denom = c.ua2 * (t3 - t200) / q200 - 1.0
        f200 = c.ua2 / (2.0 * c.cp * denom)
        return np.array([p100, f200], dtype=float)

    def linearized_steady_input(
        self,
        state: np.ndarray,
        a: np.ndarray | None = None,
        b: np.ndarray | None = None,
        affine: np.ndarray | None = None,
    ) -> np.ndarray:
        """Input making ``state`` steady for the normalized affine model.

        Zanon-Gros (2020) defines its evaporation terminal center this way.  It
        is intentionally distinct from :meth:`steady_input`, which solves the
        nonlinear nominal plant equilibrium.
        """
        if a is None or b is None or affine is None:
            a, b, affine = self.linearize(
                self.cfg.linearization_state,
                self.cfg.linearization_input,
            )
        z = self.normalized_state(state)
        rhs = (np.eye(len(z)) - np.asarray(a, dtype=float)) @ z - np.asarray(
            affine, dtype=float
        )
        v = np.linalg.solve(np.asarray(b, dtype=float), rhs)
        return self.physical_input(v)

    def economic_cost(self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray) -> float:
        """Stage cost from safe-rpi2020, using the model's algebraic flows."""
        flows = self.algebraic(state, control, disturbance)
        return float(10.09 * (flows.f2 + self.cfg.recirculation_f3) + 600.0 * flows.f100 + 0.6 * control[1])

    def normalized_state(self, state: np.ndarray) -> np.ndarray:
        return (np.asarray(state, dtype=float) - self.cfg.linearization_state) / self.cfg.state_scale

    def physical_state(self, normalized: np.ndarray) -> np.ndarray:
        return self.cfg.linearization_state + self.cfg.state_scale * np.asarray(normalized, dtype=float)

    def normalized_input(self, control: np.ndarray) -> np.ndarray:
        return (np.asarray(control, dtype=float) - self.cfg.linearization_input) / self.cfg.input_scale

    def physical_input(self, normalized: np.ndarray) -> np.ndarray:
        return self.cfg.linearization_input + self.cfg.input_scale * np.asarray(normalized, dtype=float)

    def nominal_normalized_step(self, state_n: np.ndarray, input_n: np.ndarray) -> np.ndarray:
        x = self.physical_state(state_n)
        u = self.physical_input(input_n)
        return self.normalized_state(self.step(x, u, self.cfg.disturbance_nominal))

    def linearize(self, state: np.ndarray | None = None, control: np.ndarray | None = None, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finite-difference normalized discrete affine model x+=A x+B u+b."""
        n, m = 2, 2
        x0 = np.zeros(n) if state is None else self.normalized_state(state)
        u0 = np.zeros(m) if control is None else self.normalized_input(control)
        f0 = self.nominal_normalized_step(x0, u0)
        a = np.zeros((n, n))
        bb = np.zeros((n, m))
        for i in range(n):
            xp, xm = x0.copy(), x0.copy()
            xp[i] += eps
            xm[i] -= eps
            a[:, i] = (self.nominal_normalized_step(xp, u0) - self.nominal_normalized_step(xm, u0)) / (2.0 * eps)
        for i in range(m):
            up, um = u0.copy(), u0.copy()
            up[i] += eps
            um[i] -= eps
            bb[:, i] = (self.nominal_normalized_step(x0, up) - self.nominal_normalized_step(x0, um)) / (2.0 * eps)
        affine = f0 - a @ x0 - bb @ u0
        return a, bb, affine
