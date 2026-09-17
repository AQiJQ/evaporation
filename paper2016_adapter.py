"""Read-only compatibility layer for the TuneMPC evaporation reference.

The numerical model transcription below is intentionally small and mirrors
``examples/evaporation_process/main.py``.  Controller construction, whenever
the legacy optional dependencies are available, is delegated to TuneMPC's
``Tuner``/``Pmpc`` implementation instead of reimplementing its tuning theory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np


DEFAULT_TUNEMPC_PATH = Path(r"C:\Users\cushy\PycharmProjects\tunempc")
REFERENCE_EXAMPLE = Path("examples/evaporation_process/main.py")
REQUIRED_REFERENCE_FILES = (
    REFERENCE_EXAMPLE,
    Path("tunempc/tuner.py"),
    Path("tunempc/pmpc.py"),
    Path("tunempc/convexifier.py"),
    Path("tunempc/closed_loop_tools.py"),
)


@dataclass(frozen=True)
class TuneMPCDiagnostics:
    path: str
    reference_files_present: bool
    missing_reference_files: tuple[str, ...]
    python_version: str
    legacy_python_range_satisfied: bool
    dependencies: dict[str, bool]
    direct_api_available: bool
    note: str


def dependency_diagnostics(path: Path | str = DEFAULT_TUNEMPC_PATH) -> dict[str, Any]:
    """Report availability without importing or modifying the reference repo."""
    root = Path(path).expanduser().resolve()
    missing_files = tuple(
        str(relative) for relative in REQUIRED_REFERENCE_FILES
        if not (root / relative).is_file()
    )
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("casadi", "scipy", "picos", "cvxopt")
    }
    legacy_python = (3, 5) <= sys.version_info[:2] < (3, 8)
    files_present = not missing_files
    direct = files_present and all(
        dependencies[name] for name in ("casadi", "scipy", "picos", "cvxopt")
    )
    note = (
        "TuneMPC declares Python >=3.5,<3.8 and pins casadi==3.5.1 and "
        "picos==1.2.0.post32. A newer interpreter may work only with a "
        "separately validated compatibility environment."
    )
    return asdict(TuneMPCDiagnostics(
        path=str(root),
        reference_files_present=files_present,
        missing_reference_files=missing_files,
        python_version=".".join(map(str, sys.version_info[:3])),
        legacy_python_range_satisfied=legacy_python,
        dependencies=dependencies,
        direct_api_available=direct,
        note=note,
    ))


def reference_intermediate(
    state: np.ndarray, control: np.ndarray
) -> dict[str, float]:
    """TuneMPC reference-example algebraic variables (nominal exogenous data)."""
    x2, p2 = np.asarray(state, dtype=float)
    p100, f200 = np.asarray(control, dtype=float)
    t2 = 0.5616 * p2 + 0.3126 * x2 + 48.43
    t3 = 0.507 * p2 + 55.0
    t100 = 0.1538 * p100 + 90.0
    ua1 = 0.16 * (10.0 + 50.0)
    q100 = ua1 * (t100 - t2)
    f100 = q100 / 36.6
    q200 = 6.84 * (t3 - 25.0) / (1.0 + 6.84 / (2.0 * 0.07 * f200))
    f5 = q200 / 38.5
    f4 = (q100 - 10.0 * 0.07 * (t2 - 40.0)) / 38.5
    f2 = 10.0 - f4
    return {
        "T2": float(t2), "T3": float(t3), "T100": float(t100),
        "UA1": float(ua1), "Q100": float(q100), "F100": float(f100),
        "Q200": float(q200), "F5": float(f5), "F4": float(f4),
        "F2": float(f2),
    }


def reference_derivative(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    values = reference_intermediate(state, control)
    x2 = float(np.asarray(state, dtype=float)[0])
    return np.array([
        (10.0 * 5.0 - values["F2"] * x2) / 20.0,
        (values["F4"] - values["F5"]) / 4.0,
    ])


def reference_economic_cost(state: np.ndarray, control: np.ndarray) -> float:
    values = reference_intermediate(state, control)
    return float(
        10.09 * (values["F2"] + 50.0)
        + 600.0 * values["F100"]
        + 0.6 * float(np.asarray(control, dtype=float)[1])
    )


def build_tunempc_baselines(
    path: Path | str = DEFAULT_TUNEMPC_PATH,
    *,
    horizon: int = 200,
    common_input_lower_bounds: bool = True,
    convexifier_solver: str = "cvxopt",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct the five paper-validation controllers through TuneMPC APIs.

    This follows the author-team example's ``Tuner.solve_ocp``, ``convexify``
    and ``create_mpc`` sequence.  It intentionally raises with diagnostics if
    the legacy dependency stack is unavailable; no approximate tuned MPC is
    silently substituted.
    """
    diagnostics = dependency_diagnostics(path)
    if not diagnostics["reference_files_present"]:
        raise RuntimeError(
            "TuneMPC reference path is incomplete: "
            + ", ".join(diagnostics["missing_reference_files"])
        )
    if not diagnostics["direct_api_available"]:
        missing = [
            name for name, available in diagnostics["dependencies"].items()
            if not available and name in {"casadi", "scipy", "picos", "cvxopt"}
        ]
        raise RuntimeError(
            "TuneMPC direct API unavailable; missing optional dependencies: "
            + ", ".join(missing)
        )

    root = Path(path).expanduser().resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    import casadi as ca  # type: ignore[import-not-found]
    import casadi.tools as ct  # type: ignore[import-not-found]
    import tunempc  # type: ignore[import-not-found]

    x = ct.struct_symMX(["X2", "P2"])
    u = ct.struct_symMX(["P100", "F200"])
    t2 = 0.5616 * x["P2"] + 0.3126 * x["X2"] + 48.43
    t3 = 0.507 * x["P2"] + 55.0
    t100 = 0.1538 * u["P100"] + 90.0
    q100 = 0.16 * (10.0 + 50.0) * (t100 - t2)
    f100 = q100 / 36.6
    f4 = (q100 - 10.0 * 0.07 * (t2 - 40.0)) / 38.5
    q200 = 6.84 * (t3 - 25.0) / (1.0 + 6.84 / (2.0 * 0.07 * u["F200"]))
    f5 = q200 / 38.5
    f2 = 10.0 - f4
    xdot = ca.vertcat(
        (10.0 * 5.0 - f2 * x["X2"]) / 20.0,
        (f4 - f5) / 4.0,
    )
    integrator = ca.integrator(
        "paper2016_F", "collocation", {"x": x, "p": u, "ode": xdot},
        {"tf": 1.0},
    )
    objective = ca.Function(
        "paper2016_economic_cost", [x, u],
        [10.09 * (f2 + 50.0) + 600.0 * f100 + 0.6 * u["F200"]],
    )
    constraint_terms = [
        x["X2"] - 25.0,
        x["P2"] - 40.0,
        80.0 - x["P2"],
        400.0 - u["P100"],
        400.0 - u["F200"],
    ]
    if common_input_lower_bounds:
        constraint_terms += [u["P100"] - 100.0, u["F200"] - 100.0]
    constraints = ca.Function(
        "paper2016_h", [x, u], [ca.vertcat(*constraint_terms)]
    )

    tuner = tunempc.Tuner(f=integrator, l=objective, h=constraints, p=1)
    w0 = ca.vertcat(25.0, 49.743, 191.713, 215.888)
    steady_solution = tuner.solve_ocp(w0)
    tuner.convexify(rho=1e-3, force=False, solver=convexifier_solver)
    q = tuner.S["q"]
    normal_h = [np.diag([10.0, 10.0, 0.1, 0.1])]
    tuned_diagonal_h = [np.diag(np.diag(np.asarray(tuner.S["Hc"][0])))]
    controllers = {
        "Economic_MPC": tuner.create_mpc("economic", N=horizon),
        "Normal_Tracking_MPC": tuner.create_mpc(
            "tracking", N=horizon, tuning={"H": normal_h, "q": q}
        ),
        "Tuned_Tracking_MPC": tuner.create_mpc("tuned", N=horizon),
        "Tuned_Diagonal": tuner.create_mpc(
            "tracking", N=horizon, tuning={"H": tuned_diagonal_h, "q": q}
        ),
        "Normal_Zero_Gradient": tuner.create_mpc(
            "tracking", N=horizon,
            tuning={"H": normal_h, "q": [np.zeros_like(np.asarray(q[0]))]},
        ),
    }
    metadata = {
        **diagnostics,
        "controller_source": "TuneMPC_reference",
        "tuned_cost_source": "recomputed_locally_by_TuneMPC",
        "horizon": int(horizon),
        "common_input_lower_bounds": bool(common_input_lower_bounds),
        "convexifier_solver": convexifier_solver,
        "steady_solution": np.asarray(steady_solution.cat).reshape(-1).tolist(),
        "tracking_gradient": [np.asarray(item).reshape(-1).tolist() for item in q],
        "tuned_hessian": [np.asarray(item).tolist() for item in tuner.S["Hc"]],
    }
    return controllers, metadata
