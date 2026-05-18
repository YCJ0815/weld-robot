from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


Logger = Callable[[str], None]


@dataclass(frozen=True)
class SDFTrajOptConfig:
    num_waypoints: int = 14
    maxiter: int = 80
    collision_weight: float = 2.0e7
    smoothness_weight: float = 5.0
    arm_safe_distance: float = 0.05
    tool_safe_distance: float = 0.01
    penetration_tol: float = -0.001
    arm_step_size: float = 0.02
    tool_step_size: float = 0.01
    constraint_point_stride: int = 8
    dense_check_resolution: float = 0.025


def _interpolate_edge(q_from: np.ndarray, q_to: np.ndarray, resolution: float) -> np.ndarray:
    dist = float(np.linalg.norm(q_to - q_from))
    steps = max(1, int(math.ceil(dist / max(resolution, 1e-6))))
    return np.linspace(q_from, q_to, steps + 1)


def _densify_path(path: np.ndarray, resolution: float) -> np.ndarray:
    dense = [path[0]]
    for i in range(len(path) - 1):
        dense.extend(_interpolate_edge(path[i], path[i + 1], resolution)[1:])
    return np.array(dense)


def _resample_trajectory(path: np.ndarray, num_waypoints: int) -> np.ndarray:
    if len(path) <= 1 or len(path) == num_waypoints:
        return path.copy()
    seg_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = float(cum_len[-1])
    if total_len < 1e-12:
        return np.repeat(path[:1], num_waypoints, axis=0)
    target_s = np.linspace(0.0, total_len, num_waypoints)
    return np.column_stack([np.interp(target_s, cum_len, path[:, j]) for j in range(path.shape[1])])


def _collision_penalty_from_distances(dist_arr: np.ndarray, d_safe: float) -> float:
    if len(dist_arr) == 0:
        return 0.0
    return float(np.sum(np.square(np.maximum(0.0, d_safe - dist_arr))))


def _reconstruct_path(x: np.ndarray, q_start: np.ndarray, q_goal: np.ndarray, n_dof: int) -> np.ndarray:
    if len(x) == 0:
        return np.vstack([q_start, q_goal])
    return np.vstack([q_start, x.reshape(-1, n_dof), q_goal])


class KinematicSDFCollisionEvaluator:
    def __init__(self, kinematics: Any, sdf_layer: Any, config: SDFTrajOptConfig):
        self.kinematics = kinematics
        self.sdf_layer = sdf_layer
        self.config = config

    def sample_points(self, q: np.ndarray, step_size: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        step = self.config.arm_step_size if step_size is None else float(step_size)
        return self.kinematics.sample_collision_points(
            q,
            arm_step_size=step,
            tool_step_size=self.config.tool_step_size,
        )

    def evaluate_state(self, q: np.ndarray) -> dict[str, float | bool]:
        arm_pts, tool_pts = self.sample_points(q)
        dist_arm = self.sdf_layer.get_distances(arm_pts) if len(arm_pts) > 0 else np.array([999.0])
        dist_tool = self.sdf_layer.get_distances(tool_pts) if len(tool_pts) > 0 else np.array([999.0])
        arm_min = float(np.min(dist_arm))
        tool_min = float(np.min(dist_tool))
        return {
            "arm_min": arm_min,
            "tool_min": tool_min,
            "arm_pen": arm_min < self.config.penetration_tol,
            "tool_pen": tool_min < self.config.penetration_tol,
            "valid": arm_min >= self.config.arm_safe_distance and tool_min >= self.config.tool_safe_distance,
            "nonpenetrating": arm_min >= self.config.penetration_tol and tool_min >= self.config.penetration_tol,
        }

    def is_state_valid(self, q: np.ndarray) -> bool:
        return bool(self.evaluate_state(q)["valid"])

    def summarize_trajectory(self, path: np.ndarray) -> dict[str, float | int]:
        dense = _densify_path(path, self.config.dense_check_resolution)
        penetration_count = 0
        near_count = 0
        arm_min_global = np.inf
        tool_min_global = np.inf
        for q in dense:
            state_eval = self.evaluate_state(q)
            arm_min = float(state_eval["arm_min"])
            tool_min = float(state_eval["tool_min"])
            arm_min_global = min(arm_min_global, arm_min)
            tool_min_global = min(tool_min_global, tool_min)
            if state_eval["arm_pen"] or state_eval["tool_pen"]:
                penetration_count += 1
            elif arm_min < self.config.arm_safe_distance or tool_min < self.config.tool_safe_distance:
                near_count += 1
        return {
            "penetration_count": penetration_count,
            "near_count": near_count,
            "arm_min_global": float(arm_min_global),
            "tool_min_global": float(tool_min_global),
        }


def _trajopt_objective(
    x: np.ndarray,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    n_dof: int,
    evaluator: KinematicSDFCollisionEvaluator,
) -> float:
    if len(x) == 0:
        return 0.0
    config = evaluator.config
    path = _reconstruct_path(x, q_start, q_goal, n_dof)
    smoothness_cost = 0.0
    if len(path) >= 3:
        q_dd = path[2:] - 2.0 * path[1:-1] + path[:-2]
        smoothness_cost = float(np.sum(np.square(q_dd)))
    collision_cost = 0.0
    for q in path:
        arm_pts, tool_pts = evaluator.sample_points(q)
        collision_cost += _collision_penalty_from_distances(
            evaluator.sdf_layer.get_distances(arm_pts),
            config.arm_safe_distance,
        )
        collision_cost += _collision_penalty_from_distances(
            evaluator.sdf_layer.get_distances(tool_pts),
            config.tool_safe_distance,
        )
    return config.collision_weight * collision_cost + config.smoothness_weight * smoothness_cost


def _trajopt_nonpenetration_constraint(
    x: np.ndarray,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    n_dof: int,
    evaluator: KinematicSDFCollisionEvaluator,
) -> np.ndarray:
    path = _reconstruct_path(x, q_start, q_goal, n_dof)
    margins = []
    stride = max(1, int(evaluator.config.constraint_point_stride))
    for q in path[1:-1]:
        arm_pts, tool_pts = evaluator.sample_points(q)
        if len(arm_pts) > 0:
            margins.append(evaluator.sdf_layer.get_distances(arm_pts[::stride]))
        if len(tool_pts) > 0:
            margins.append(evaluator.sdf_layer.get_distances(tool_pts[::stride]))
    if not margins:
        return np.array([1.0], dtype=float)
    return np.concatenate(margins)


def run_sdf_trajopt(
    q_seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    evaluator: KinematicSDFCollisionEvaluator,
    logger: Logger | None = None,
) -> tuple[np.ndarray, bool]:
    if minimize is None:
        if logger is not None:
            logger("[SDF-TrajOpt] scipy.optimize is unavailable; skipping optimization.")
        return q_seed, False
    if len(q_seed) <= 2:
        if logger is not None:
            logger("[SDF-TrajOpt] Seed path has too few waypoints; skipping optimization.")
        return q_seed, False

    config = evaluator.config
    num_waypoints = max(3, min(config.num_waypoints, len(q_seed)))
    q_init = _resample_trajectory(q_seed, num_waypoints)
    q_start = q_init[0]
    q_goal = q_init[-1]
    x0 = q_init[1:-1].reshape(-1)
    if len(x0) == 0:
        if logger is not None:
            logger("[SDF-TrajOpt] Seed path has no internal waypoints; skipping optimization.")
        return q_seed, False

    bounds = []
    for _ in range(len(q_init) - 2):
        for low, high in zip(lower, upper):
            bounds.append((float(low), float(high)))

    if logger is not None:
        logger(
            f"[SDF-TrajOpt] Starting optimization: seed_points={len(q_seed)} "
            f"opt_points={len(q_init)} maxiter={config.maxiter} stride={config.constraint_point_stride}"
        )

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: _trajopt_nonpenetration_constraint(x, q_start, q_goal, len(lower), evaluator),
        }
    ]
    result = minimize(
        _trajopt_objective,
        x0,
        args=(q_start, q_goal, len(lower), evaluator),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": config.maxiter, "disp": False, "ftol": 1e-4},
    )

    q_opt = _reconstruct_path(result.x, q_start, q_goal, len(lower))
    summary = evaluator.summarize_trajectory(q_opt)
    nonpenetrating = (
        summary["arm_min_global"] >= evaluator.config.penetration_tol
        and summary["tool_min_global"] >= evaluator.config.penetration_tol
    )
    accepted = bool(result.success) and bool(nonpenetrating)
    if logger is not None:
        logger(
            f"[SDF-TrajOpt] Finished: success={result.success} accepted={accepted} "
            f"status={result.status} nit={getattr(result, 'nit', -1)} fun={float(result.fun):.4f} "
            f"penetrations={summary['penetration_count']} near={summary['near_count']} "
            f"arm_min={summary['arm_min_global']:.4f} tool_min={summary['tool_min_global']:.4f}"
        )
    return q_opt, accepted
