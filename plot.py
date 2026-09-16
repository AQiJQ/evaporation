"""Matplotlib figures for the independent evaporator safe-SAC experiment."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np


# Colors and layout follow the supplied journal-style reference figure.
GREEN = "#2ca25f"
BLUE = "#5b8fd1"
ORANGE = "#ee9b00"
RED = "#d9485f"
BLACK = "#333333"
GRID = "#d9d9d9"


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "font.size": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
    "savefig.facecolor": "white",
})


def _style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.8)


def _legend(ax, **kwargs) -> None:
    defaults = {
        "loc": "upper left",
        "frameon": True,
        "facecolor": "white",
        "edgecolor": "#d0d0d0",
        "framealpha": 0.92,
        "ncol": 1,
    }
    defaults.update(kwargs)
    ax.legend(**defaults)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=170, bbox_inches="tight")


def show_all_plots(*, block: bool = True) -> None:
    """Display all figures created by this module in the Python GUI backend."""
    plt.show(block=block)


def close_all_plots() -> None:
    """Close figures, mainly for automated tests and noninteractive scripts."""
    plt.close("all")


def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values.copy()
    average = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), average])


def plot_learning(log: dict[str, np.ndarray], out_dir: Path) -> None:
    """Plot the true accumulated reward of every training episode."""
    mask = log["episode"] >= 1.0
    episodes = log["episode"][mask]
    rewards = log["return"][mask]
    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    ax.plot(episodes, moving_average(rewards), color=BLUE, linewidth=1.8,
            label="20-episode moving average")
    _style_axis(
        ax,
        "SAC 20-episode moving-average reward",
        "Episode",
        "Moving-average reward",
    )
    if len(episodes) > 1:
        ax.set_xlim(float(episodes[0]), float(episodes[-1]))
    _legend(ax)
    _save(fig, out_dir / "training_curves.png")


def plot_theta_learning(log: dict[str, np.ndarray], out_dir: Path) -> None:
    """Show the slow h, p, M, K adaptation separately from SAC reward."""
    episodes = log["episode"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    series = [
        ("theta_h_norm", r"Stage linear term $\|h\|_2$", BLUE),
        ("theta_p_norm", r"Terminal linear term $\|p\|_2$", ORANGE),
        ("theta_M_angle_deg", r"Uncertainty facets $M$ angle", GREEN),
        ("theta_K_change_norm", r"Feedback change $\|K-K_0\|_F$", RED),
    ]
    for ax, (key, title, color) in zip(axes.flat, series):
        ax.plot(episodes, log[key], color=color)
        ylabel = "Angle [deg]" if key == "theta_M_angle_deg" else "Norm"
        _style_axis(ax, title, "Episode", ylabel)
        if len(episodes) > 1:
            ax.set_xlim(float(episodes[0]), float(episodes[-1]))
    _save(fig, out_dir / "theta_learning.png")


def plot_empirical_regret(log: dict[str, np.ndarray], out_dir: Path) -> None:
    """Plot post-hoc fixed-seed evaluation gap, not the training reward."""
    mask = np.isfinite(log["evaluation_return_per_step"])
    episodes = log["episode"][mask]
    evaluation = log["evaluation_return_per_step"][mask]
    best_so_far = np.maximum.accumulate(evaluation)
    best_observed = float(np.max(evaluation))
    scale = max(best_observed - float(evaluation[0]), 1e-12)
    regret = (best_so_far - best_observed) / scale
    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    ax.plot(episodes, regret, color=GREEN, linewidth=1.7,
            label="Best-so-far evaluation gap")
    ax.axhline(0.0, color=BLACK, linewidth=0.9,
               label="Best observed in this run")
    _style_axis(
        ax,
        "Post-hoc empirical regret of deterministic evaluation",
        "Episode",
        "Normalized gap",
    )
    if len(episodes) > 1:
        ax.set_xlim(float(episodes[0]), float(episodes[-1]))
    _legend(ax)
    _save(fig, out_dir / "empirical_regret.png")


def plot_sets(
    cfg,
    model,
    design,
    rpi_boundary: np.ndarray,
    rollout: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    fig, (ax, text_ax) = plt.subplots(
        1, 2,
        figsize=(11.0, 5.3),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )
    x_lower, x_upper = cfg.state_lower, cfg.state_upper
    tight_lower = model.physical_state(design.x_lower_tight)
    tight_upper = model.physical_state(design.x_upper_tight)
    invariant_lower = model.physical_state(design.invariant_lower)
    invariant_upper = model.physical_state(design.invariant_upper)
    rpi_physical = cfg.safe_center_state + rpi_boundary * cfg.state_scale

    ax.add_patch(Rectangle(
        x_lower,
        *(x_upper - x_lower),
        facecolor="#c9e4f2", edgecolor=BLUE, alpha=0.72,
        label=r"$X$ (hard constraints)",
    ))
    ax.add_patch(Rectangle(
        tight_lower,
        *(tight_upper - tight_lower),
        facecolor="#9fe2dc", edgecolor="#30bfb7", alpha=0.78,
        label=r"$X\ominus Z$",
    ))
    ax.add_patch(Rectangle(
        invariant_lower,
        *(invariant_upper - invariant_lower),
        facecolor="#ffe48a", edgecolor=ORANGE, alpha=0.86,
        label=r"Controlled-invariant $S$",
    ))
    ax.add_patch(Polygon(
        rpi_physical,
        closed=True,
        facecolor="#ef8a8a", edgecolor=RED, alpha=0.82,
        label=r"Translated RPI $Z$",
    ))
    ax.plot(
        rollout["state"][:, 0], rollout["state"][:, 1],
        color=BLACK, linewidth=1.35, label="Safe SAC rollout",
    )
    ax.scatter(*cfg.safe_center_state, color=BLACK, s=22, zorder=5,
               label="Safety anchor")
    ax.set_xlim(24.5, 45.0)
    ax.set_ylim(39.0, 81.0)
    _style_axis(ax, "Robust sets and safe rollout", r"$X_2$ [%]", r"$P_2$ [kPa]")
    _legend(ax, ncol=2)

    z_width = (
        design.rpi_support_lower + design.rpi_support_upper
    ) * cfg.state_scale
    tight_width = (design.x_upper_tight - design.x_lower_tight) * cfg.state_scale
    invariant_width = (design.invariant_upper - design.invariant_lower) * cfg.state_scale
    area_z = 0.5 * abs(float(np.sum(
        rpi_physical[:, 0] * np.roll(rpi_physical[:, 1], -1)
        - rpi_physical[:, 1] * np.roll(rpi_physical[:, 0], -1)
    )))
    lines = [
        "Set sizes",
        "",
        f"Continuously learned H∞ K: ||K||F = {np.linalg.norm(design.k):.3f}",
        f"RPI bounding width: {z_width[0]:.3f} % × {z_width[1]:.3f} kPa",
        f"RPI polygon area: {area_z:.3f} %-kPa",
        f"Box-W RPI area (same K): {design.box_reference_rpi_area_physical:.3f} %-kPa",
        f"X−Z width: {tight_width[0]:.3f} % × {tight_width[1]:.3f} kPa",
        f"X−Z area: {np.prod(tight_width):.3f} %-kPa",
        f"Invariant S width: {invariant_width[0]:.3f} % × {invariant_width[1]:.3f} kPa",
        f"W bound (normalized): [{design.w_bound[0]:.5f}, {design.w_bound[1]:.5f}]",
        f"H∞ γ bound / measured norm: {design.gamma_design:.3f} / {design.gamma_sampled:.3f}",
    ]
    text_ax.axis("off")
    text_ax.text(
        0.0, 0.98, "\n".join(lines), va="top", ha="left",
        linespacing=1.55, fontsize=10, color=BLACK,
        transform=text_ax.transAxes,
    )
    _save(fig, out_dir / "rpi_x_minus_z.png")


def plot_paper_aligned_set_comparison(
    cfg,
    model,
    initial_design,
    learned_design,
    out_dir: Path,
    learned_title: str = r"After online $\theta=(h,p,M,K)$ learning",
) -> None:
    """Initial/final robust sets in the convention of the paper's Figure 4."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.3), constrained_layout=True)
    for index, (ax, design, title) in enumerate(zip(
        axes,
        (initial_design, learned_design),
        ("Initial safe design", learned_title),
    )):
        tight_lower = model.physical_state(design.x_lower_tight)
        tight_upper = model.physical_state(design.x_upper_tight)
        invariant_lower = model.physical_state(design.invariant_lower)
        invariant_upper = model.physical_state(design.invariant_upper)
        translated_rpi = (
            cfg.safe_center_state + design.rpi_boundary * cfg.state_scale
        )
        ax.add_patch(Rectangle(
            cfg.state_lower,
            *(cfg.state_upper - cfg.state_lower),
            facecolor="#c9e4f2", edgecolor=BLUE, alpha=0.60,
            label=r"$X$" if index == 0 else None,
        ))
        ax.add_patch(Rectangle(
            tight_lower,
            *(tight_upper - tight_lower),
            facecolor="#9fe2dc", edgecolor="#30bfb7", alpha=0.72,
            label=r"$X\ominus Z$" if index == 0 else None,
        ))
        ax.add_patch(Rectangle(
            invariant_lower,
            *(invariant_upper - invariant_lower),
            facecolor="#ffe48a", edgecolor=ORANGE, alpha=0.82,
            label=r"Terminal/invariant $S$" if index == 0 else None,
        ))
        ax.add_patch(Polygon(
            translated_rpi,
            closed=True,
            facecolor="#ef8a8a", edgecolor=RED, alpha=0.80,
            label=r"RPI $Z$" if index == 0 else None,
        ))
        ax.scatter(
            *cfg.safe_center_state,
            color=BLACK,
            s=24,
            zorder=5,
            label=r"Paper center $(29,53.57)$" if index == 0 else None,
        )
        ax.set_xlim(24.5, 45.0)
        ax.set_ylim(39.0, 81.0)
        _style_axis(ax, title, r"$X_2$ [%]", r"$P_2$ [kPa]")
    _legend(axes[0], loc="upper right", ncol=1)
    _save(fig, out_dir / "paper2020_initial_final_sets.png")


def plot_rollout(cfg, rollout: dict[str, np.ndarray], out_dir: Path) -> None:
    time = rollout["time"]
    state, control = rollout["state"], rollout["control"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    series = [
        (r"Product concentration $X_2$", state[:, 0], cfg.state_lower[0],
         cfg.state_upper[0], r"$X_2$ [%]"),
        (r"Operating pressure $P_2$", state[:, 1], cfg.state_lower[1],
         cfg.state_upper[1], r"$P_2$ [kPa]"),
        (r"Steam pressure $P_{100}$", control[:, 0], cfg.input_lower[0],
         cfg.input_upper[0], r"$P_{100}$ [kPa]"),
        (r"Cooling-water flow $F_{200}$", control[:, 1], cfg.input_lower[1],
         cfg.input_upper[1], r"$F_{200}$ [kg min$^{-1}$]"),
    ]
    for ax, (title, values, lower, upper, ylabel) in zip(axes.flat, series):
        ax.plot(time, values, color=BLUE, label="Safe SAC")
        ax.axhline(lower, color=RED, linestyle="--", linewidth=1.0,
                   label="Constraints")
        ax.axhline(upper, color=RED, linestyle="--", linewidth=1.0)
        padding = 0.05 * max(upper - lower, 1.0)
        ax.set_ylim(lower - padding, upper + padding)
        ax.set_xlim(float(time[0]), float(time[-1]))
        _style_axis(ax, title, "Time [min]", ylabel)
        _legend(ax)
    _save(fig, out_dir / "states_inputs.png")


def plot_disturbances(cfg, rollout: dict[str, np.ndarray], out_dir: Path) -> None:
    time = rollout["time"]
    disturbance = rollout["disturbance"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    series = [
        (r"Feed flow $F_1$", r"$F_1$ [kg min$^{-1}$]"),
        (r"Feed concentration $X_1$", r"$X_1$ [%]"),
        (r"Feed temperature $T_1$", r"$T_1$ [°C]"),
        (r"Cooling-water temperature $T_{200}$", r"$T_{200}$ [°C]"),
    ]
    for index, (ax, (title, ylabel)) in enumerate(zip(axes.flat, series)):
        nominal = cfg.disturbance_nominal[index]
        half_range = cfg.disturbance_half_range[index]
        lower, upper = nominal - half_range, nominal + half_range
        ax.plot(time, disturbance[:, index], color=BLUE, linewidth=0.65,
                label="Hidden plant disturbance")
        ax.axhline(lower, color=RED, linestyle="--", linewidth=1.0,
                   label="Bounds")
        ax.axhline(upper, color=RED, linestyle="--", linewidth=1.0)
        ax.set_xlim(float(time[0]), float(time[-1]))
        ax.set_ylim(lower - 0.08 * (upper - lower), upper + 0.08 * (upper - lower))
        _style_axis(ax, title, "Time [min]", ylabel)
        _legend(ax)
    _save(fig, out_dir / "disturbances.png")


def plot_disturbance_response(cfg, design, rollout: dict[str, np.ndarray], out_dir: Path) -> None:
    time = rollout["time"]
    d_norm = (
        rollout["disturbance"] - cfg.disturbance_nominal
    ) / cfg.disturbance_half_range
    state_deviation = rollout["state"] - cfg.safe_center_state
    w_upper = np.max(design.w_vertices, axis=0)
    w_lower = -np.min(design.w_vertices, axis=0)
    w_utilization = np.where(
        rollout["effective_w"] >= 0.0,
        rollout["effective_w"] / np.maximum(w_upper, 1e-12),
        -rollout["effective_w"] / np.maximum(w_lower, 1e-12),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    axes[0, 0].plot(time, d_norm[:, 0], color=GREEN, linewidth=0.65, label=r"$F_1$")
    axes[0, 0].plot(time, d_norm[:, 1], color=ORANGE, linewidth=0.65, label=r"$X_1$")
    axes[0, 0].set_ylim(-1.1, 1.1)
    _style_axis(axes[0, 0], "Hidden feed disturbances", "Time [min]", "Normalized value")
    _legend(axes[0, 0], ncol=2)

    axes[0, 1].plot(time, d_norm[:, 2], color=GREEN, linewidth=0.65, label=r"$T_1$")
    axes[0, 1].plot(time, d_norm[:, 3], color=ORANGE, linewidth=0.65, label=r"$T_{200}$")
    axes[0, 1].set_ylim(-1.1, 1.1)
    _style_axis(axes[0, 1], "Hidden temperature disturbances", "Time [min]", "Normalized value")
    _legend(axes[0, 1], ncol=2)

    axes[1, 0].plot(time, state_deviation[:, 0], color=GREEN, label=r"$\Delta X_2$")
    axes[1, 0].plot(time, state_deviation[:, 1], color=ORANGE, label=r"$\Delta P_2$")
    _style_axis(axes[1, 0], "State response about safety anchor", "Time [min]", "State deviation")
    _legend(axes[1, 0], ncol=2)

    axes[1, 1].plot(time, w_utilization[:, 0], color=GREEN, linewidth=0.8, label=r"directional $w_{X_2}$ utilization")
    axes[1, 1].plot(time, w_utilization[:, 1], color=ORANGE, linewidth=0.8, label=r"directional $w_{P_2}$ utilization")
    axes[1, 1].axhline(1.0, color=RED, linestyle="--", linewidth=1.0, label="Robust bounds")
    axes[1, 1].set_ylim(0.0, max(1.08, 1.05 * float(np.max(w_utilization))))
    _style_axis(axes[1, 1], "Effective mismatch and robust bound", "Time [min]", "Bound ratio")
    _legend(axes[1, 1], ncol=2)
    for ax in axes.flat:
        ax.set_xlim(float(time[0]), float(time[-1]))
    _save(fig, out_dir / "disturbance_response.png")


def plot_economic_performance(rollout: dict[str, np.ndarray], reference_cost: float, out_dir: Path) -> None:
    time = rollout["time"]
    cost = rollout["economic_cost"]
    cumulative_mean = np.cumsum(cost) / np.arange(1, len(cost) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, title, values in [
        (axes[0], "Instantaneous economic stage cost", cost),
        (axes[1], "Cumulative mean economic cost", cumulative_mean),
    ]:
        ax.plot(time, values, color=GREEN, label="Safe SAC")
        ax.axhline(reference_cost, color=ORANGE, linestyle="--", linewidth=1.2,
                   label="Safety-anchor nominal cost")
        ax.set_xlim(float(time[0]), float(time[-1]))
        _style_axis(ax, title, "Time [min]", "Economic stage cost")
        _legend(ax)
    _save(fig, out_dir / "economic_performance.png")


def plot_feedback_components(rollout: dict[str, np.ndarray], out_dir: Path) -> None:
    time = rollout["time"]
    residual = rollout.get("applied_residual", rollout["residual"])
    candidate = rollout.get("candidate", rollout["base"] + residual)
    qp_correction = rollout["nominal_control"] - candidate
    ancillary = rollout["ancillary"]
    values = np.concatenate([residual, qp_correction, ancillary], axis=1)
    limit = max(0.02, 1.08 * float(np.max(np.abs(values))))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for index, (ax, title) in enumerate(zip(
        axes, [r"$P_{100}$ channel", r"$F_{200}$ channel"]
    )):
        ax.plot(time, ancillary[:, index], color=GREEN, label="H∞")
        ax.plot(time, qp_correction[:, index], color=BLUE, label="QP")
        ax.plot(time, residual[:, index], color=ORANGE, label="SAC")
        ax.set_xlim(float(time[0]), float(time[-1]))
        ax.set_ylim(-limit, limit)
        _style_axis(ax, title, "Time [min]", "Normalized input contribution")
        _legend(ax, ncol=3)
    _save(fig, out_dir / "feedback_components.png")


def plot_sac_policy_map(policy_map: dict[str, np.ndarray], out_dir: Path) -> None:
    """Requested and safely executed residuals over the certified state set."""
    x2 = policy_map["X2"]
    p2 = policy_map["P2"]
    requested = policy_map["requested_physical"]
    applied = policy_map["applied_physical"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    panels = [
        (requested[:, :, 0], r"Actor-requested $\Delta P_{100}$", "kPa"),
        (requested[:, :, 1], r"Actor-requested $\Delta F_{200}$", r"kg min$^{-1}$"),
        (applied[:, :, 0], r"Safe applied $\Delta P_{100}$", "kPa"),
        (applied[:, :, 1], r"Safe applied $\Delta F_{200}$", r"kg min$^{-1}$"),
    ]
    for ax, (values, title, unit) in zip(axes.flat, panels):
        limit = max(float(np.max(np.abs(values))), 1e-8)
        levels = np.linspace(-limit, limit, 21)
        image = ax.contourf(
            x2, p2, values, levels=levels, cmap="coolwarm", extend="both"
        )
        fig.colorbar(image, ax=ax, label=unit)
        _style_axis(ax, title, r"$X_2$ [%]", r"$P_2$ [kPa]")
    _save(fig, out_dir / "sac_residual_policy_map.png")


def plot_rl_comparison(
    cfg,
    rl_rollout: dict[str, np.ndarray],
    theta_only_rollout: dict[str, np.ndarray],
    no_rl_rollout: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    time = rl_rollout["time"]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.0), constrained_layout=True)
    paired = [
        (r"Product concentration $X_2$", rl_rollout["state"][:, 0], theta_only_rollout["state"][:, 0], no_rl_rollout["state"][:, 0], r"$X_2$ [%]", cfg.state_lower[0], cfg.state_upper[0]),
        (r"Operating pressure $P_2$", rl_rollout["state"][:, 1], theta_only_rollout["state"][:, 1], no_rl_rollout["state"][:, 1], r"$P_2$ [kPa]", cfg.state_lower[1], cfg.state_upper[1]),
        (r"Steam pressure $P_{100}$", rl_rollout["control"][:, 0], theta_only_rollout["control"][:, 0], no_rl_rollout["control"][:, 0], r"$P_{100}$ [kPa]", cfg.input_lower[0], cfg.input_upper[0]),
        (r"Cooling-water flow $F_{200}$", rl_rollout["control"][:, 1], theta_only_rollout["control"][:, 1], no_rl_rollout["control"][:, 1], r"$F_{200}$ [kg min$^{-1}$]", cfg.input_lower[1], cfg.input_upper[1]),
    ]
    for ax, (title, rl_values, theta_values, base_values, ylabel, lower, upper) in zip(axes.flat[:4], paired):
        ax.plot(time, rl_values, color=GREEN, label="Safe SAC")
        ax.plot(time, theta_values, color=BLUE, label=r"$\theta$-only")
        ax.plot(time, base_values, color=ORANGE, label=r"Fixed $\theta$, no RL")
        ax.axhline(lower, color=RED, linestyle="--", linewidth=0.9, label="Constraints")
        ax.axhline(upper, color=RED, linestyle="--", linewidth=0.9)
        padding = 0.05 * max(upper - lower, 1.0)
        ax.set_ylim(lower - padding, upper + padding)
        _style_axis(ax, title, "Time [min]", ylabel)
        _legend(ax, ncol=3)

    rl_cost = np.cumsum(rl_rollout["economic_cost"]) / np.arange(1, len(time) + 1)
    theta_cost = np.cumsum(theta_only_rollout["economic_cost"]) / np.arange(1, len(time) + 1)
    no_rl_cost = np.cumsum(no_rl_rollout["economic_cost"]) / np.arange(1, len(time) + 1)
    rl_return = np.cumsum(rl_rollout["reward"])
    theta_return = np.cumsum(theta_only_rollout["reward"])
    no_rl_return = np.cumsum(no_rl_rollout["reward"])
    for ax, title, rl_values, theta_values, base_values, ylabel in [
        (axes[2, 0], "Cumulative mean economic cost", rl_cost, theta_cost, no_rl_cost, "Economic stage cost"),
        (axes[2, 1], "Cumulative reward", rl_return, theta_return, no_rl_return, "Return"),
    ]:
        ax.plot(time, rl_values, color=GREEN, label="Safe SAC")
        ax.plot(time, theta_values, color=BLUE, label=r"$\theta$-only")
        ax.plot(time, base_values, color=ORANGE, label=r"Fixed $\theta$, no RL")
        _style_axis(ax, title, "Time [min]", ylabel)
        _legend(ax)
    for ax in axes.flat:
        ax.set_xlim(float(time[0]), float(time[-1]))
    _save(fig, out_dir / "rl_vs_no_rl.png")
