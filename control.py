"""H-infinity synthesis, RPI construction, and one-step two-input QP filter."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import numpy as np

from .config import ExperimentConfig
from .model import EvaporatorModel


def spectral_radius(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def _hinf_riccati(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve the discrete full-information H-infinity dynamic-game Riccati equation."""
    n = a.shape[0]
    e = np.eye(n)
    p = q.copy()
    for _ in range(8000):
        hww = e.T @ p @ e - gamma * gamma * np.eye(n)
        huu = r + b.T @ p @ b
        huw = b.T @ p @ e
        if np.max(np.linalg.eigvalsh(hww)) >= -1e-9:
            return None
        schur_u = huu - huw @ np.linalg.solve(hww, huw.T)
        if np.min(np.linalg.eigvalsh(schur_u)) <= 1e-10:
            return None
        h = np.block([[huu, huw], [huw.T, hww]])
        g = np.vstack([b.T @ p @ a, e.T @ p @ a])
        try:
            solved = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            return None
        k = -solved[: b.shape[1], :]
        p_next = q + a.T @ p @ a - g.T @ solved
        p_next = 0.5 * (p_next + p_next.T)
        if not np.all(np.isfinite(p_next)) or np.max(np.abs(p_next)) > 1e12:
            return None
        if np.max(np.abs(p_next - p)) < 1e-8:
            if spectral_radius(a + b @ k) < 1.0:
                return k, p_next
            return None
        p = p_next
    return None


def estimate_hinf_norm(a: np.ndarray, b: np.ndarray, k: np.ndarray, q: np.ndarray, r: np.ndarray, points: int = 1024) -> float:
    """Frequency-grid closed-loop l2 gain from additive state disturbance."""
    acl = a + b @ k
    cq = np.linalg.cholesky(q).T
    cr = np.linalg.cholesky(r).T @ k
    cperf = np.vstack([cq, cr])
    eye = np.eye(a.shape[0])
    peak = 0.0
    for omega in np.linspace(0.0, np.pi, points):
        transfer = cperf @ np.linalg.solve(np.exp(1j * omega) * eye - acl, eye)
        peak = max(peak, float(np.linalg.svd(transfer, compute_uv=False)[0]))
    return peak


def synthesize_hinf(cfg: ExperimentConfig, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Synthesize a dynamic-game feedback at the configured attenuation gamma."""
    gamma = float(cfg.hinf_gamma)
    feasible = _hinf_riccati(a, b, cfg.hinf_q, cfg.hinf_r, gamma)
    if feasible is None:
        raise RuntimeError("H-infinity Riccati synthesis failed at the configured gamma.")
    best_k, _ = feasible
    sampled_norm = estimate_hinf_norm(a, b, best_k, cfg.hinf_q, cfg.hinf_r)
    if sampled_norm >= gamma:
        raise RuntimeError(f"Sampled closed-loop H-infinity norm {sampled_norm:.3f} exceeds gamma={gamma:.3f}.")
    return best_k, gamma, float(sampled_norm)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise 2-D convex hull (monotone chain)."""
    unique = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def four_facet_outer_approximation(
    points: np.ndarray,
    angle: float = 0.0,
    inflation: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a four-facet, generally asymmetric outer approximation.

    The returned tuple is ``(vertices, M, m)`` for
    ``W_theta = {w | M w <= m}``.  Following the evaporation example in the
    2020 paper, ``m`` is fixed to one and changes in support are absorbed by
    ``M``.  Including the origin makes every support positive and preserves
    the usual RPI convention ``0 in W_theta``.
    """
    cloud = np.vstack([np.zeros((1, 2)), np.asarray(points, dtype=float)])
    axis_1 = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    axis_2 = np.array([-np.sin(angle), np.cos(angle)], dtype=float)
    normals = np.vstack([axis_1, axis_2, -axis_1, -axis_2])
    support = np.maximum(
        float(inflation) * np.max(normals @ cloud.T, axis=1),
        1e-8,
    )
    matrix = normals / support[:, None]
    bounds = np.ones(4, dtype=float)
    # Intersections of adjacent facets in counter-clockwise order.
    order = [0, 1, 2, 3]
    vertices = []
    for index, next_index in zip(order, order[1:] + order[:1]):
        vertices.append(np.linalg.solve(
            matrix[[index, next_index]], bounds[[index, next_index]]
        ))
    vertices = _convex_hull(np.asarray(vertices, dtype=float))
    return vertices, matrix, bounds


def estimate_disturbance_set(
    cfg: ExperimentConfig,
    model: EvaporatorModel,
    a: np.ndarray,
    b: np.ndarray,
    affine: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Four-facet asymmetric envelope of corner and sampled model mismatch."""
    # Local domain used to sample nonlinear one-step mismatch.
    x_lo = np.maximum(
        cfg.state_lower,
        cfg.linearization_state - cfg.safety_design_state_half_range,
    )
    x_hi = np.minimum(
        cfg.state_upper,
        cfg.linearization_state + cfg.safety_design_state_half_range,
    )
    u_lo = np.maximum(
        cfg.input_lower,
        cfg.linearization_input - cfg.safety_design_input_half_range,
    )
    u_hi = np.minimum(
        cfg.input_upper,
        cfg.linearization_input + cfg.safety_design_input_half_range,
    )
    residuals: list[np.ndarray] = []

    def observe(x: np.ndarray, u: np.ndarray, d: np.ndarray) -> None:
        xn = model.normalized_state(x)
        un = model.normalized_input(u)
        actual = model.normalized_state(model.step(x, u, d))
        predicted = a @ xn + b @ un + affine
        mismatch = actual - predicted
        residuals.append(mismatch)

    # All 16 disturbance corners at representative state/input corners and centers.
    x_points = [0.5 * (x_lo + x_hi)] + [np.array(v) for v in product(*zip(x_lo, x_hi))]
    u_points = [0.5 * (u_lo + u_hi)] + [np.array(v) for v in product(*zip(u_lo, u_hi))]
    d_lo = cfg.disturbance_nominal - cfg.disturbance_half_range
    d_hi = cfg.disturbance_nominal + cfg.disturbance_half_range
    d_points = [np.array(v) for v in product(*zip(d_lo, d_hi))]
    for x in x_points:
        for u in u_points:
            for d in d_points:
                observe(x, u, d)

    # Also certify a compact neighborhood of the interior terminal/safety
    # anchor.  This prevents the initial economic-point linear model from
    # omitting the affine/nonlinear residual encountered at episode reset.
    safe_u = model.linearized_steady_input(
        cfg.safe_center_state, a, b, affine
    )
    safe_x_lo = np.maximum(
        cfg.state_lower,
        cfg.safe_center_state - cfg.theta_initial_safe_state_half_range,
    )
    safe_x_hi = np.minimum(
        cfg.state_upper,
        cfg.safe_center_state + cfg.theta_initial_safe_state_half_range,
    )
    safe_u_lo = np.maximum(
        cfg.input_lower,
        safe_u - cfg.theta_initial_safe_input_half_range,
    )
    safe_u_hi = np.minimum(
        cfg.input_upper,
        safe_u + cfg.theta_initial_safe_input_half_range,
    )
    safe_x_points = [cfg.safe_center_state] + [
        np.asarray(value, dtype=float)
        for value in product(*zip(safe_x_lo, safe_x_hi))
    ]
    safe_u_points = [safe_u] + [
        np.asarray(value, dtype=float)
        for value in product(*zip(safe_u_lo, safe_u_hi))
    ]
    for x in safe_x_points:
        for u in safe_u_points:
            for d in d_points:
                observe(x, u, d)

    for _ in range(cfg.disturbance_bound_samples):
        observe(
            rng.uniform(x_lo, x_hi),
            rng.uniform(u_lo, u_hi),
            rng.uniform(d_lo, d_hi),
        )
    data_hull = _convex_hull(np.vstack([
        np.zeros((1, 2)), np.asarray(residuals, dtype=float)
    ]))
    angle = 0.0
    vertices, matrix, bounds = four_facet_outer_approximation(
        data_hull,
        angle=angle,
        inflation=cfg.rpi_inflation,
    )
    bound = np.maximum(np.max(np.abs(vertices), axis=0), 1e-8)
    return bound, vertices, data_hull, matrix, bounds, angle


def rpi_generators(acl: np.ndarray, w_bound: np.ndarray, tol: float = 1e-11, max_terms: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Generators and coordinate support of Z=sum Acl^i W."""
    power = np.eye(acl.shape[0])
    generators: list[np.ndarray] = []
    support = np.zeros(acl.shape[0])
    for i in range(max_terms):
        block = power @ np.diag(w_bound)
        generators.extend([block[:, j].copy() for j in range(block.shape[1])])
        term = np.sum(np.abs(block), axis=1)
        support += term
        power = acl @ power
        if i > 20 and np.max(term) < tol:
            break
    else:
        raise RuntimeError("RPI series did not converge.")
    return np.column_stack(generators), support


def _rpi_support(
    acl: np.ndarray,
    w_vertices: np.ndarray,
    directions: np.ndarray,
    tol: float = 1e-11,
    max_terms: int = 5000,
) -> np.ndarray:
    """Support of sum Acl^i W in selected row-vector directions."""
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    power = np.eye(acl.shape[0])
    support = np.zeros(len(directions))
    for index in range(max_terms):
        transformed = directions @ power
        term = np.max(transformed @ w_vertices.T, axis=1)
        support += term
        power = power @ acl
        if index > 20 and float(np.max(np.abs(term))) < tol:
            break
    else:
        raise RuntimeError("RPI support series did not converge.")
    return support


def rpi_polygon(
    acl: np.ndarray,
    w_vertices: np.ndarray,
    directions: int = 720,
) -> np.ndarray:
    """Safe outer polygon from densely sampled RPI support half-spaces."""
    angles = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    supports = _rpi_support(acl, w_vertices, normals)
    vertices = np.empty((directions, 2), dtype=float)
    for index in range(directions):
        next_index = (index + 1) % directions
        matrix = np.vstack([normals[index], normals[next_index]])
        vertices[index] = np.linalg.solve(
            matrix,
            np.array([supports[index], supports[next_index]]),
        )
    return vertices


def point_in_convex_polygon(
    point: np.ndarray,
    vertices: np.ndarray,
    tol: float = 1e-9,
) -> bool:
    """Membership test for a counter-clockwise convex polygon."""
    point = np.asarray(point, dtype=float)
    edges = np.roll(vertices, -1, axis=0) - vertices
    relative = point - vertices
    cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
    return bool(np.all(cross >= -tol))


def zonotope_boundary(generators: np.ndarray, directions: int = 1440) -> np.ndarray:
    """Ordered support vertices for a centered two-dimensional zonotope."""
    angles = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    vertices = np.empty((directions, 2))
    for i, direction in enumerate(dirs):
        signs = np.where(direction @ generators >= 0.0, 1.0, -1.0)
        vertices[i] = generators @ signs
    rounded = np.round(vertices, 10)
    _, first = np.unique(rounded, axis=0, return_index=True)
    vertices = vertices[np.sort(first)]
    center = np.mean(vertices, axis=0)
    order = np.argsort(np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0]))
    return vertices[order]


def project_qp_2d(candidate: np.ndarray, a_ineq: np.ndarray, b_ineq: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, bool]:
    """Exact Euclidean projection onto A u <= b by active-set enumeration in 2D."""
    candidate = np.asarray(candidate, dtype=float)
    if np.all(a_ineq @ candidate <= b_ineq + tol):
        return candidate.copy(), True
    choices: list[np.ndarray] = []
    for row, bound in zip(a_ineq, b_ineq):
        denom = float(row @ row)
        if denom <= 1e-14:
            continue
        point = candidate - ((row @ candidate - bound) / denom) * row
        if np.all(a_ineq @ point <= b_ineq + tol):
            choices.append(point)
    for i, j in combinations(range(len(b_ineq)), 2):
        mat = np.vstack([a_ineq[i], a_ineq[j]])
        if abs(np.linalg.det(mat)) <= 1e-12:
            continue
        point = np.linalg.solve(mat, np.array([b_ineq[i], b_ineq[j]]))
        if np.all(a_ineq @ point <= b_ineq + tol):
            choices.append(point)
    if not choices:
        return candidate.copy(), False
    distances = [float(np.sum((point - candidate) ** 2)) for point in choices]
    return choices[int(np.argmin(distances))], True


def finite_horizon_nominal_policy(
    cfg: ExperimentConfig,
    a: np.ndarray,
    b: np.ndarray,
    affine: np.ndarray,
    h: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First affine action of the positive-definite quadratic value model.

    This is the unconstrained nominal part of the finite-horizon optimizer.
    The RPI/QP layer below imposes the hard robust constraints on its first
    action, so updating h or p cannot bypass the safety filter.
    """
    hessian = np.asarray(cfg.nominal_mpc_stage_hessian, dtype=float)
    q = hessian[:2, :2]
    n = hessian[:2, 2:]
    r = hessian[2:, 2:]
    h_x = np.asarray(h[:2], dtype=float)
    h_u = np.asarray(h[2:], dtype=float)
    value_hessian = np.asarray(cfg.nominal_mpc_terminal_hessian, dtype=float)
    value_linear = np.asarray(p, dtype=float).copy()
    first_gain = np.zeros((2, 2), dtype=float)
    first_offset = np.zeros(2, dtype=float)
    gamma = float(cfg.gamma_rl)
    for stage in reversed(range(int(cfg.nominal_mpc_horizon))):
        q_xx = q + gamma * a.T @ value_hessian @ a
        q_ux = n.T + gamma * b.T @ value_hessian @ a
        q_uu = r + gamma * b.T @ value_hessian @ b
        shifted = value_hessian @ affine + value_linear
        q_x = h_x + gamma * a.T @ shifted
        q_u = h_u + gamma * b.T @ shifted
        inverse_action = np.linalg.inv(0.5 * (q_uu + q_uu.T))
        gain = -inverse_action @ q_ux
        offset = -inverse_action @ q_u
        value_hessian = q_xx - q_ux.T @ inverse_action @ q_ux
        value_hessian = 0.5 * (value_hessian + value_hessian.T)
        value_linear = q_x - q_ux.T @ inverse_action @ q_u
        if stage == 0:
            first_gain = gain
            first_offset = offset
    return first_gain, first_offset


@dataclass
class SafetyDesign:
    a: np.ndarray
    b: np.ndarray
    affine: np.ndarray
    k: np.ndarray
    gamma_design: float
    gamma_sampled: float
    w_bound: np.ndarray
    w_vertices: np.ndarray
    w_data_hull: np.ndarray
    theta_m_matrix: np.ndarray
    theta_m_bound: np.ndarray
    theta_m_angle: float
    rpi_boundary: np.ndarray
    rpi_support: np.ndarray
    rpi_support_lower: np.ndarray
    rpi_support_upper: np.ndarray
    input_rpi_support: np.ndarray
    input_rpi_support_lower: np.ndarray
    input_rpi_support_upper: np.ndarray
    x_lower_tight: np.ndarray
    x_upper_tight: np.ndarray
    u_lower_tight: np.ndarray
    u_upper_tight: np.ndarray
    z_ref: np.ndarray
    v_ref: np.ndarray
    invariant_lower: np.ndarray
    invariant_upper: np.ndarray
    invariant_scale: float
    gain_state_weight_scale: float
    gain_input_weight_scale: float
    gain_candidates_feasible: int
    reference_rpi_width_physical: np.ndarray
    reference_rpi_area_physical: float
    reference_x_minus_z_width_physical: np.ndarray
    reference_x_minus_z_area_physical: float
    box_reference_rpi_width_physical: np.ndarray
    box_reference_rpi_area_physical: float
    theta_h: np.ndarray
    theta_p: np.ndarray
    nominal_policy_gain: np.ndarray
    nominal_policy_offset: np.ndarray


def _controlled_invariant_box(
    cfg: ExperimentConfig,
    a: np.ndarray,
    b: np.ndarray,
    affine: np.ndarray,
    z_ref: np.ndarray,
    v_ref: np.ndarray,
    state_lower: np.ndarray,
    state_upper: np.ndarray,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Largest homothetic controlled-invariant box, certified at its vertices.

    At every vertex there must exist one admissible current input whose next
    nominal state remains in the same box.  Convexity and affine dynamics then
    certify the full box.  This is a one-step RPI/QP construction, not MPC.
    """

    def scaled_box(scale: float) -> tuple[np.ndarray, np.ndarray]:
        lower = z_ref + scale * (state_lower - z_ref)
        upper = z_ref + scale * (state_upper - z_ref)
        return lower, upper

    def is_invariant(scale: float) -> bool:
        lower, upper = scaled_box(scale)
        for vertex in product(*zip(lower, upper)):
            z = np.asarray(vertex, dtype=float)
            center_next = a @ z + affine
            aq = np.vstack([np.eye(2), -np.eye(2), b, -b])
            bq = np.concatenate([
                input_upper,
                -input_lower,
                upper - center_next,
                -lower + center_next,
            ])
            _, feasible = project_qp_2d(v_ref, aq, bq, tol=1e-8)
            if not feasible:
                return False
        return True

    if not is_invariant(0.0):
        return None
    low, high = 0.0, 1.0
    if not is_invariant(1.0):
        for _ in range(60):
            middle = 0.5 * (low + high)
            if is_invariant(middle):
                low = middle
            else:
                high = middle
        scale = low
    else:
        scale = 1.0
    scale *= float(cfg.invariant_set_margin)
    lower, upper = scaled_box(scale)
    return lower, upper, scale


def build_safety_design(
    cfg: ExperimentConfig,
    model: EvaporatorModel,
    rng: np.random.Generator,
    *,
    w_data_hull: np.ndarray | None = None,
    theta_m_angle: float = 0.0,
    theta_h: np.ndarray | None = None,
    theta_p: np.ndarray | None = None,
    theta_k: np.ndarray | None = None,
) -> SafetyDesign:
    # Zanon-Gros evaporation setup: identify the affine nominal model at the
    # economic optimum, while keeping the terminal/safety center in the
    # interior of the constraints.
    a, b, affine = model.linearize(
        cfg.linearization_state, cfg.linearization_input
    )
    safe_u = model.linearized_steady_input(
        cfg.safe_center_state, a, b, affine
    )
    if w_data_hull is None:
        (
            w_bound,
            w_vertices,
            data_hull,
            theta_m_matrix,
            theta_m_bound,
            theta_m_angle,
        ) = estimate_disturbance_set(cfg, model, a, b, affine, rng)
    else:
        data_hull = _convex_hull(np.vstack([
            np.zeros((1, 2)), np.asarray(w_data_hull, dtype=float)
        ]))
        w_vertices, theta_m_matrix, theta_m_bound = (
            four_facet_outer_approximation(
                data_hull,
                angle=float(theta_m_angle),
                inflation=cfg.theta_w_online_inflation,
            )
        )
        w_bound = np.maximum(np.max(np.abs(w_vertices), axis=0), 1e-8)
    learned_h = (
        np.zeros(4, dtype=float)
        if theta_h is None else np.asarray(theta_h, dtype=float)
    )
    learned_p = (
        np.zeros(2, dtype=float)
        if theta_p is None else np.asarray(theta_p, dtype=float)
    )
    x_lo = model.normalized_state(cfg.state_lower)
    x_hi = model.normalized_state(cfg.state_upper)
    u_lo = model.normalized_input(cfg.input_lower)
    u_hi = model.normalized_input(cfg.input_upper)
    z_ref = model.normalized_state(cfg.safe_center_state)
    v_ref = model.normalized_input(safe_u)

    # K is an explicit continuous component of theta.  The configured seed was
    # obtained by a continuous constrained search for the paper center; later
    # calls provide arbitrary real-valued 2x2 proposals from the online learner.
    # Every proposal must still pass all H-infinity/RPI/QP checks below.
    k = np.asarray(
        cfg.theta_k_initial_gain if theta_k is None else theta_k,
        dtype=float,
    )
    if k.shape != (2, 2) or not np.all(np.isfinite(k)):
        raise ValueError("theta_k must be a finite 2x2 feedback matrix.")
    candidates: list[dict[str, object]] = []
    for state_weight_scale, input_weight_scale in [(1.0, 1.0)]:
        acl = a + b @ k
        if spectral_radius(acl) >= 1.0:
            continue
        sampled_norm = estimate_hinf_norm(
            a,
            b,
            k,
            float(state_weight_scale) * cfg.hinf_q,
            float(input_weight_scale) * cfg.hinf_r,
        )
        if sampled_norm >= cfg.hinf_gamma:
            continue
        try:
            boundary = rpi_polygon(acl, w_vertices)
            support_upper = _rpi_support(acl, w_vertices, np.eye(2))
            support_lower = _rpi_support(acl, w_vertices, -np.eye(2))
        except RuntimeError:
            continue
        input_support_upper = _rpi_support(acl, w_vertices, k)
        input_support_lower = _rpi_support(acl, w_vertices, -k)
        support = np.maximum(support_lower, support_upper)
        input_support = np.maximum(
            input_support_lower, input_support_upper
        )
        x_lo_t = x_lo + support_lower
        x_hi_t = x_hi - support_upper
        u_lo_t = u_lo + input_support_lower
        u_hi_t = u_hi - input_support_upper
        if np.any(z_ref <= x_lo_t) or np.any(z_ref >= x_hi_t):
            continue
        if np.any(v_ref <= u_lo_t) or np.any(v_ref >= u_hi_t):
            continue

        # Find a controlled-invariant subset of X-Z in the declared learning
        # region.  Runtime checks continue to report any sampled nonlinear
        # mismatch that leaves the offline disturbance polytope.
        learning_x_lo = model.normalized_state(
            cfg.safe_center_state - cfg.invariant_nominal_half_range_physical
        )
        learning_x_hi = model.normalized_state(
            cfg.safe_center_state + cfg.invariant_nominal_half_range_physical
        )
        verified_x_lo = np.maximum(x_lo_t, learning_x_lo)
        verified_x_hi = np.minimum(x_hi_t, learning_x_hi)
        if np.any(verified_x_hi <= verified_x_lo):
            continue
        invariant = _controlled_invariant_box(
            cfg,
            a,
            b,
            affine,
            z_ref,
            v_ref,
            verified_x_lo,
            verified_x_hi,
            u_lo_t,
            u_hi_t,
        )
        if invariant is None:
            continue
        invariant_lower, invariant_upper, invariant_scale = invariant

        physical_boundary = boundary * cfg.state_scale
        rpi_area = 0.5 * abs(float(np.sum(
            physical_boundary[:, 0] * np.roll(physical_boundary[:, 1], -1)
            - physical_boundary[:, 1] * np.roll(physical_boundary[:, 0], -1)
        )))
        rpi_width = (support_lower + support_upper) * cfg.state_scale
        x_minus_z_width = (x_hi_t - x_lo_t) * cfg.state_scale
        candidates.append({
            "state_weight_scale": float(state_weight_scale),
            "input_weight_scale": float(input_weight_scale),
            "k": k,
            "sampled_norm": float(sampled_norm),
            "boundary": boundary,
            "support": support,
            "support_lower": support_lower,
            "support_upper": support_upper,
            "input_support": input_support,
            "input_support_lower": input_support_lower,
            "input_support_upper": input_support_upper,
            "x_lo_t": x_lo_t,
            "x_hi_t": x_hi_t,
            "u_lo_t": u_lo_t,
            "u_hi_t": u_hi_t,
            "invariant_lower": invariant_lower,
            "invariant_upper": invariant_upper,
            "invariant_scale": invariant_scale,
            "invariant_area": float(np.prod(
                (invariant_upper - invariant_lower) * cfg.state_scale
            )),
            "input_rpi_fraction": float(np.sum(
                input_support / np.maximum(u_hi - u_lo, 1e-12)
            )),
            "rpi_width": rpi_width,
            "rpi_area": rpi_area,
            "x_minus_z_width": x_minus_z_width,
            "x_minus_z_area": float(np.prod(x_minus_z_width)),
        })

    if not candidates:
        raise RuntimeError(
            "Continuous K proposal failed the H-infinity/RPI/input-tightening/"
            "controlled-invariant safety gate."
        )
    reference = min(
        candidates,
        key=lambda item: (
            abs(float(item["state_weight_scale"]) - 1.0)
            + abs(float(item["input_weight_scale"]) - 1.0)
        ),
    )
    # Do not accept an RPI set larger than the original scale-1 design. Among the
    # remaining candidates, balance compact RPI sets against a large controlled
    # invariant region and less input tightening.
    pareto = [
        item for item in candidates
        if float(item["rpi_area"]) <= float(reference["rpi_area"]) + 1e-12
        and float(item["x_minus_z_area"]) >= float(reference["x_minus_z_area"]) - 1e-12
    ]
    invariant_qualified = [
        item for item in pareto
        if float(item["invariant_area"]) >= 0.75 * float(reference["invariant_area"])
    ]
    selected = min(
        invariant_qualified,
        key=lambda item: (
            float(item["rpi_area"]),
            float(item["input_rpi_fraction"]),
            -float(item["invariant_area"]),
        ),
    )
    box_generators, box_support = rpi_generators(
        a + b @ np.asarray(selected["k"]), w_bound
    )
    box_boundary_physical = zonotope_boundary(box_generators) * cfg.state_scale
    box_area = 0.5 * abs(float(np.sum(
        box_boundary_physical[:, 0] * np.roll(box_boundary_physical[:, 1], -1)
        - box_boundary_physical[:, 1] * np.roll(box_boundary_physical[:, 0], -1)
    )))
    nominal_gain, nominal_offset = finite_horizon_nominal_policy(
        cfg, a, b, affine, learned_h, learned_p
    )
    return SafetyDesign(
        a=a,
        b=b,
        affine=affine,
        k=np.asarray(selected["k"]),
        gamma_design=float(cfg.hinf_gamma),
        gamma_sampled=float(selected["sampled_norm"]),
        w_bound=w_bound,
        w_vertices=w_vertices,
        w_data_hull=data_hull,
        theta_m_matrix=theta_m_matrix,
        theta_m_bound=theta_m_bound,
        theta_m_angle=float(theta_m_angle),
        rpi_boundary=np.asarray(selected["boundary"]),
        rpi_support=np.asarray(selected["support"]),
        rpi_support_lower=np.asarray(selected["support_lower"]),
        rpi_support_upper=np.asarray(selected["support_upper"]),
        input_rpi_support=np.asarray(selected["input_support"]),
        input_rpi_support_lower=np.asarray(selected["input_support_lower"]),
        input_rpi_support_upper=np.asarray(selected["input_support_upper"]),
        x_lower_tight=np.asarray(selected["x_lo_t"]),
        x_upper_tight=np.asarray(selected["x_hi_t"]),
        u_lower_tight=np.asarray(selected["u_lo_t"]),
        u_upper_tight=np.asarray(selected["u_hi_t"]),
        z_ref=z_ref,
        v_ref=v_ref,
        invariant_lower=np.asarray(selected["invariant_lower"]),
        invariant_upper=np.asarray(selected["invariant_upper"]),
        invariant_scale=float(selected["invariant_scale"]),
        gain_state_weight_scale=float(selected["state_weight_scale"]),
        gain_input_weight_scale=float(selected["input_weight_scale"]),
        gain_candidates_feasible=len(candidates),
        reference_rpi_width_physical=np.asarray(reference["rpi_width"]),
        reference_rpi_area_physical=float(reference["rpi_area"]),
        reference_x_minus_z_width_physical=np.asarray(reference["x_minus_z_width"]),
        reference_x_minus_z_area_physical=float(reference["x_minus_z_area"]),
        box_reference_rpi_width_physical=2.0 * box_support * cfg.state_scale,
        box_reference_rpi_area_physical=box_area,
        theta_h=learned_h.copy(),
        theta_p=learned_p.copy(),
        nominal_policy_gain=nominal_gain,
        nominal_policy_offset=nominal_offset,
    )


class SafeController:
    """Learned nominal guidance -> SAC residual -> QP -> H-infinity feedback."""

    def __init__(self, cfg: ExperimentConfig, model: EvaporatorModel, design: SafetyDesign):
        self.cfg, self.model, self.d = cfg, model, design
        self.z = design.z_ref.copy()

    def reset(self, state: np.ndarray) -> None:
        self.z = self.model.normalized_state(state)

    def act(self, state: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        d = self.d
        x = self.model.normalized_state(state)
        e = x - self.z

        # The learned h,p policy supplies finite-horizon nominal guidance.
        # Current-input constraints and one-step membership in the certified
        # controlled-invariant set remain the final hard safety gate.
        z_center_next = d.a @ self.z + d.affine
        rows = [np.eye(2), -np.eye(2), d.b, -d.b]
        bounds = [
            d.u_upper_tight,
            -d.u_lower_tight,
            d.invariant_upper - z_center_next,
            -d.invariant_lower + z_center_next,
        ]
        aq = np.vstack(rows)
        bq = np.concatenate(bounds)
        theta_candidate = d.nominal_policy_gain @ self.z + d.nominal_policy_offset
        base, theta_feasible = project_qp_2d(theta_candidate, aq, bq)
        if not theta_feasible:
            base, theta_feasible = project_qp_2d(d.v_ref, aq, bq)
        if not theta_feasible:
            base = np.clip(d.v_ref, d.u_lower_tight, d.u_upper_tight)
        requested_residual = np.asarray(residual, dtype=float)
        requested_candidate = base + requested_residual

        # State-dependent safe action parameterization.  Move from the known
        # feasible base along the SAC-requested ray only as far as the current
        # tightened-input/invariant polytope permits.  Unlike an orthogonal QP
        # projection, this preserves the actor's requested direction and makes
        # the executed action a deterministic function of (observation, action).
        ray_denominator = aq @ requested_residual
        outward = ray_denominator > 1e-12
        mapping_scale = 1.0
        if np.any(outward):
            slack = bq - aq @ base
            mapping_scale = min(
                1.0,
                max(0.0, float(np.min(
                    slack[outward] / ray_denominator[outward]
                ))),
            )
        if mapping_scale < 1.0:
            mapping_scale *= float(self.cfg.invariant_set_margin)
        candidate = base + mapping_scale * requested_residual
        nominal, feasible = project_qp_2d(candidate, aq, bq)
        if not feasible:
            nominal = np.clip(base, d.u_lower_tight, d.u_upper_tight)
        ancillary = d.k @ e
        actual_n = nominal + ancillary
        actual_n = np.clip(actual_n, self.model.normalized_input(self.cfg.input_lower), self.model.normalized_input(self.cfg.input_upper))
        control = self.model.physical_input(actual_n)
        z_next = d.a @ self.z + d.b @ nominal + d.affine
        info = {
            "x_norm": x.copy(), "z": self.z.copy(), "e": e.copy(),
            "theta_candidate": theta_candidate.copy(),
            "theta_projection_gap": float(np.linalg.norm(base - theta_candidate)),
            "base": base.copy(), "requested_candidate": requested_candidate.copy(),
            "candidate": candidate.copy(), "nominal": nominal.copy(),
            "ancillary": ancillary.copy(), "actual_norm": actual_n.copy(),
            "z_next": z_next.copy(), "qp_feasible": bool(feasible),
            "projection_gap": float(np.linalg.norm(nominal - candidate)),
            "feasible_action_mapping_scale": float(mapping_scale),
            "feasible_action_mapping_gap": float(np.linalg.norm(
                candidate - requested_candidate
            )),
        }
        self.z = z_next
        return control, info
