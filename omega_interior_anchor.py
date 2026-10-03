"""Offline-only baseline-preserving interior-anchor action mapping.

This controller is deliberately not selected by the SAC training entrypoint.
It reuses the existing Z/Omega safety polytopes and their final QP.
"""
from __future__ import annotations

from itertools import combinations
from functools import lru_cache

import numpy as np

from .omega_feasible_normalized import (
    FeasibleSetNormalizedController, ray_authority,
    action_for_feasible_point,
)


@lru_cache(maxsize=16)
def _chebyshev_vertex_systems(row_count, row_bytes):
    """Cache fixed facet geometry; online states change bounds, not normals."""
    h = np.frombuffer(row_bytes, dtype=np.float64).reshape(row_count, 2)
    lp_rows = np.vstack((
        np.column_stack((h, np.linalg.norm(h, axis=1))),
        [0.0, 0.0, -1.0],
    ))
    indices, inverses = [], []
    for triple in combinations(range(len(lp_rows)), 3):
        matrix = lp_rows[list(triple)]
        if abs(np.linalg.det(matrix)) >= 1e-12:
            indices.append(triple)
            inverses.append(np.linalg.inv(matrix))
    if not indices:
        raise RuntimeError("Chebyshev LP has no nonsingular vertex systems")
    return lp_rows, np.asarray(indices, dtype=int), np.asarray(inverses)


def chebyshev_center_2d(rows, bounds):
    """Solve the 2D Chebyshev LP by enumerating its 3-plane vertices.

    No additional optimizer dependency is needed. Rows and coordinates are
    those of the existing normalized safety-action polytope.
    """
    h = np.asarray(rows, dtype=float)
    b = np.asarray(bounds, dtype=float)
    if h.ndim != 2 or h.shape[1] != 2 or b.shape != (len(h),):
        raise ValueError("Expected H of shape (m,2) and h of shape (m,)")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(b)):
        raise ValueError("Safety-action polytope contains nonfinite values")
    contiguous = np.ascontiguousarray(h, dtype=np.float64)
    lp_rows, indices, inverses = _chebyshev_vertex_systems(
        len(h), contiguous.tobytes()
    )
    lp_bounds = np.concatenate((b, [0.0]))
    candidates = np.einsum("nij,nj->ni", inverses, lp_bounds[indices])
    feasible = np.max(candidates @ lp_rows.T - lp_bounds, axis=1) <= 1e-8
    if not np.any(feasible):
        raise RuntimeError("Chebyshev LP has no feasible vertex")
    best = candidates[np.flatnonzero(feasible)[
        np.argmax(candidates[feasible, 2])
    ]]
    center, radius = best[:2], float(max(0.0, best[2]))
    if np.max(h @ center + np.linalg.norm(h, axis=1) * radius - b) > 1e-8:
        raise RuntimeError("Chebyshev center failed its defining inequalities")
    return center, radius


def interior_anchor_target(base, anchor, action, rows, bounds):
    """Return the convex-combination target and its geometric diagnostics."""
    a = np.asarray(action, dtype=float)
    rho = float(np.max(np.abs(a)))
    direction = a / rho if rho > 1e-15 else np.zeros(2)
    tau = ray_authority(anchor, direction, rows, bounds) if rho > 1e-15 else 0.0
    boundary = anchor + tau * direction
    target = (1.0 - rho) * base + rho * boundary
    return target, boundary, rho, direction, tau


def action_for_interior_point(base, anchor, desired, rows, bounds,
                              input_scale=None):
    """Invert the anchor map for any point of the feasible 2D polytope."""
    scale = np.ones(2) if input_scale is None else np.asarray(input_scale)
    from .control import project_qp_2d
    physical, feasible = project_qp_2d(
        np.asarray(desired) * scale,
        np.asarray(rows) / scale[np.newaxis, :], bounds,
    )
    if not feasible:
        raise RuntimeError("Safety-action polytope is empty")
    point = physical / scale
    delta = point - base
    length = float(np.max(np.abs(delta)))
    if length <= 1e-12:
        return np.zeros(2), point
    from_base = delta / length
    base_tau = ray_authority(base, from_base, rows, bounds)
    if base_tau <= 1e-12:
        raise RuntimeError("Desired feasible point has zero baseline ray")
    boundary = base + base_tau * from_base
    rho = float(np.clip(length / base_tau, 0.0, 1.0))
    anchor_delta = boundary - anchor
    anchor_length = float(np.max(np.abs(anchor_delta)))
    if anchor_length <= 1e-12:
        raise RuntimeError("Boundary point coincides with interior anchor")
    direction = anchor_delta / anchor_length
    action = rho * direction
    return np.clip(action, -1.0, 1.0), point


class InteriorAnchorController(FeasibleSetNormalizedController):
    """Independent offline prototype; final projection is the original QP."""

    def act(self, state, actor_action, *, action_is_normalized=True):
        if not action_is_normalized:
            raise ValueError("Expected normalized actor action")
        action = np.asarray(actor_action, dtype=float)
        if action.shape != (2,) or np.any(np.abs(action) > 1 + 1e-9):
            raise ValueError("Actor action must be in [-1,1]^2")
        if np.max(np.abs(action)) <= 1e-15:
            control, info = super().act(
                state, np.zeros(2), action_is_normalized=True
            )
            if info["qp_feasible"] and info["mode"] in {
                "Z_mode_existing_controller", "Omega_safe_one_step_QP"
            }:
                anchor, radius = chebyshev_center_2d(
                    info["action_safe_rows"], info["action_safe_bounds"]
                )
                info["interior_anchor_coordinate"] = anchor.copy()
                info["interior_chebyshev_radius"] = radius
                info["interior_boundary_coordinate"] = np.full(2, np.nan)
                info["interior_actor_rho"] = 0.0
                info["interior_tau_max"] = 0.0
                info["interior_target_gap"] = 0.0
            info["interior_zero_action_exact_baseline"] = True
            return control, info

        z_saved = self.inner.z.copy()
        was_saved = self.was_in_omega
        try:
            _, baseline = super().act(
                state, np.zeros(2), action_is_normalized=True
            )
        finally:
            self.inner.z = z_saved
            self.was_in_omega = was_saved
        if not baseline["qp_feasible"] or baseline["mode"] not in {
            "Z_mode_existing_controller", "Omega_safe_one_step_QP"
        }:
            # Outside the certified action domain, retain the existing fallback.
            return super().act(state, np.zeros(2), action_is_normalized=True)

        rows = baseline["action_safe_rows"]
        bounds = baseline["action_safe_bounds"]
        base = baseline["action_center_coordinate"]
        anchor, radius = chebyshev_center_2d(rows, bounds)
        target, boundary, rho, direction, tau = interior_anchor_target(
            base, anchor, action, rows, bounds
        )
        if np.max(rows @ target - bounds) > 1e-8:
            raise RuntimeError("Convex interior-anchor target is not feasible")
        proxy, projected = action_for_feasible_point(
            base, target, rows, bounds, self.cfg.input_scale
        )
        if np.linalg.norm(projected - target) > 1e-8:
            raise RuntimeError("Interior target changed during radial inversion")
        control, info = super().act(
            state, proxy, action_is_normalized=True
        )
        info.update({
            "interior_zero_action_exact_baseline": False,
            "interior_anchor_coordinate": anchor.copy(),
            "interior_chebyshev_radius": radius,
            "interior_boundary_coordinate": boundary.copy(),
            "interior_target_coordinate": target.copy(),
            "interior_actor_rho": rho,
            "interior_actor_direction": direction.copy(),
            "interior_tau_max": tau,
            "interior_proxy_action": proxy.copy(),
            "interior_target_gap": float(np.linalg.norm(
                info["action_final_coordinate"] - target
            )),
        })
        return control, info
