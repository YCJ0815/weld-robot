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
    num_waypoints: int = 20
    max_waypoints: int = 16
    maxiter: int = 800
    collision_weight: float = 2.0e8
    smoothness_weight: float = 8.0
    path_length_weight: float = 3.0
    arm_safe_distance: float = 0.05
    tool_safe_distance: float = 0.01
    penetration_tol: float = -0.001
    arm_step_size: float = 0.02
    tool_step_size: float = 0.01
    constraint_point_stride: int = 12
    dense_check_resolution: float = 0.025
    endpoint_relax_waypoints: int = 2
    endpoint_safe_distance_scale: float = 0.0
    initial_penetration_tol: float = -0.0025


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


def _normalized_curvature_scores(path: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(path), dtype=float)
    if len(path) <= 2:
        return scores
    for i in range(1, len(path) - 1):
        prev_step = path[i] - path[i - 1]
        next_step = path[i + 1] - path[i]
        denom = float(np.linalg.norm(prev_step) + np.linalg.norm(next_step))
        if denom < 1e-9:
            continue
        scores[i] = float(np.linalg.norm(path[i + 1] - 2.0 * path[i] + path[i - 1]) / denom)
    max_score = float(np.max(scores))
    if max_score > 1e-12:
        scores /= max_score
    return scores


def _collision_penalty_from_distances(dist_arr: np.ndarray, d_safe: float) -> float:
    if len(dist_arr) == 0:
        return 0.0
    return float(np.sum(np.square(np.maximum(0.0, d_safe - dist_arr))))


def _summary_within_tolerance(summary: dict[str, float | int], tol: float) -> bool:
    return float(summary["arm_min_global"]) >= tol and float(summary["tool_min_global"]) >= tol


def _endpoint_safe_distance_scale(path_index: int, num_points: int, config: SDFTrajOptConfig) -> float:
    relax_waypoints = max(0, int(config.endpoint_relax_waypoints))
    if relax_waypoints <= 0 or num_points <= 2:
        return 1.0
    distance_from_endpoint = min(path_index, num_points - 1 - path_index)
    if distance_from_endpoint >= relax_waypoints:
        return 1.0
    alpha = float(distance_from_endpoint) / float(relax_waypoints)
    endpoint_scale = float(np.clip(config.endpoint_safe_distance_scale, 0.0, 1.0))
    return endpoint_scale + (1.0 - endpoint_scale) * alpha


def _reconstruct_path(x: np.ndarray, q_start: np.ndarray, q_goal: np.ndarray, n_dof: int) -> np.ndarray:
    if len(x) == 0:
        return np.vstack([q_start, q_goal])
    return np.vstack([q_start, x.reshape(-1, n_dof), q_goal])


def _candidate_waypoint_counts(seed_len: int, requested: int, max_waypoints: int) -> list[int]:
    if seed_len <= 0:
        return [max(3, requested)]
    upper_bound = seed_len if max_waypoints <= 0 else min(seed_len, max(max_waypoints, requested))
    current = max(3, min(requested, upper_bound))
    return list(range(current, upper_bound + 1))


class KinematicSDFCollisionEvaluator:
    def __init__(self, kinematics: Any, sdf_layer: Any, config: SDFTrajOptConfig):
        self.kinematics = kinematics
        self.sdf_layer = sdf_layer
        self.config = config
        self._state_cache: dict[bytes, dict[str, float | bool]] = {}
        self._state_cache_max_size = 4096

    def _cache_key(self, q: np.ndarray) -> bytes:
        arr = np.asarray(q, dtype=np.float32).reshape(-1)
        return np.round(arr, decimals=5).tobytes()

    def _store_cached_state(self, key: bytes, value: dict[str, float | bool]) -> None:
        if len(self._state_cache) >= self._state_cache_max_size:
            self._state_cache.clear()
        self._state_cache[key] = value

    def sample_points(self, q: np.ndarray, step_size: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        step = self.config.arm_step_size if step_size is None else float(step_size)
        return self.kinematics.sample_collision_points(
            q,
            arm_step_size=step,
            tool_step_size=self.config.tool_step_size,
        )

    def evaluate_state(self, q: np.ndarray) -> dict[str, float | bool]:
        key = self._cache_key(q)
        cached = self._state_cache.get(key)
        if cached is not None:
            return cached
        arm_pts, tool_pts = self.sample_points(q)
        dist_arm = self.sdf_layer.get_distances(arm_pts) if len(arm_pts) > 0 else np.array([999.0])
        dist_tool = self.sdf_layer.get_distances(tool_pts) if len(tool_pts) > 0 else np.array([999.0])
        arm_min = float(np.min(dist_arm))
        tool_min = float(np.min(dist_tool))
        result = {
            "arm_min": arm_min,
            "tool_min": tool_min,
            "arm_pen": arm_min < self.config.penetration_tol,
            "tool_pen": tool_min < self.config.penetration_tol,
            "valid": arm_min >= self.config.arm_safe_distance and tool_min >= self.config.tool_safe_distance,
            "nonpenetrating": arm_min >= self.config.penetration_tol and tool_min >= self.config.penetration_tol,
        }
        self._store_cached_state(key, result)
        return result

    def evaluate_path_states(self, path: np.ndarray) -> list[dict[str, float | bool]]:
        return [self.evaluate_state(q) for q in np.asarray(path, dtype=float)]

    def is_state_valid(self, q: np.ndarray) -> bool:
        return bool(self.evaluate_state(q)["valid"])

    def is_state_nonpenetrating(self, q: np.ndarray) -> bool:
        return bool(self.evaluate_state(q)["nonpenetrating"])

    def summarize_trajectory(self, path: np.ndarray) -> dict[str, float | int]:
        dense = _densify_path(path, self.config.dense_check_resolution)
        dense_states = self.evaluate_path_states(dense)
        penetration_count = 0
        near_count = 0
        arm_min_global = np.inf
        tool_min_global = np.inf
        for state_eval in dense_states:
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


def _adaptive_select_waypoints(
    path: np.ndarray,
    num_waypoints: int,
    evaluator: KinematicSDFCollisionEvaluator,
) -> np.ndarray:
    if len(path) <= num_waypoints:
        return path.copy()

    num_waypoints = max(2, int(num_waypoints))
    curvature_scores = _normalized_curvature_scores(path)
    clearance_scores = np.zeros(len(path), dtype=float)
    state_evals = evaluator.evaluate_path_states(path)
    target_clearance = max(
        1e-6,
        float(max(evaluator.config.arm_safe_distance, evaluator.config.tool_safe_distance)),
    )
    for i, state_eval in enumerate(state_evals):
        clearance = float(min(state_eval["arm_min"], state_eval["tool_min"]))
        clearance_scores[i] = max(0.0, target_clearance - clearance) / target_clearance

    selected = {0, len(path) - 1}
    num_uniform_anchors = min(num_waypoints, max(2, int(math.ceil(num_waypoints * 0.35))))
    uniform_indices = np.linspace(0, len(path) - 1, num_uniform_anchors).round().astype(int)
    selected.update(int(idx) for idx in uniform_indices.tolist())

    importance = 1.5 * clearance_scores + curvature_scores
    ranked_indices = np.argsort(-importance)
    for idx in ranked_indices.tolist():
        if len(selected) >= num_waypoints:
            break
        selected.add(int(idx))

    if len(selected) < num_waypoints:
        for idx in range(len(path)):
            if len(selected) >= num_waypoints:
                break
            selected.add(idx)

    ordered_indices = np.array(sorted(selected), dtype=int)
    if len(ordered_indices) > num_waypoints:
        keep_mask = np.zeros(len(ordered_indices), dtype=bool)
        keep_mask[0] = True
        keep_mask[-1] = True
        interior = ordered_indices[1:-1]
        interior_scores = importance[interior]
        remaining = num_waypoints - 2
        if remaining > 0 and len(interior) > 0:
            top_positions = np.argsort(-interior_scores)[:remaining]
            keep_mask[1 + np.sort(top_positions)] = True
        ordered_indices = ordered_indices[keep_mask]
        ordered_indices.sort()
    return path[ordered_indices]


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
    path_length_cost = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    collision_cost = 0.0
    for path_index, q in enumerate(path):
        safe_scale = _endpoint_safe_distance_scale(path_index, len(path), config)
        arm_pts, tool_pts = evaluator.sample_points(q)
        collision_cost += _collision_penalty_from_distances(
            evaluator.sdf_layer.get_distances(arm_pts),
            config.arm_safe_distance * safe_scale,
        )
        collision_cost += _collision_penalty_from_distances(
            evaluator.sdf_layer.get_distances(tool_pts),
            config.tool_safe_distance * safe_scale,
        )
    return (
        config.collision_weight * collision_cost
        + config.smoothness_weight * smoothness_cost
        + config.path_length_weight * path_length_cost
    )


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
    full_seed_summary = evaluator.summarize_trajectory(q_seed)
    full_seed_nonpenetrating = _summary_within_tolerance(full_seed_summary, evaluator.config.penetration_tol)
    full_seed_initially_acceptable = _summary_within_tolerance(full_seed_summary, evaluator.config.initial_penetration_tol)
    waypoint_counts = _candidate_waypoint_counts(
        len(q_seed),
        requested=max(3, int(config.num_waypoints)),
        max_waypoints=int(config.max_waypoints),
    )
    q_init = _adaptive_select_waypoints(q_seed, waypoint_counts[0], evaluator)
    q_init_summary = evaluator.summarize_trajectory(q_init)
    selected_count = waypoint_counts[0]
    for count in waypoint_counts[1:]:
        if _summary_within_tolerance(q_init_summary, evaluator.config.initial_penetration_tol):
            break
        candidate = _adaptive_select_waypoints(q_seed, count, evaluator)
        candidate_summary = evaluator.summarize_trajectory(candidate)
        q_init = candidate
        q_init_summary = candidate_summary
        selected_count = count
    selected_from_full_seed = False
    if (
        not (
            _summary_within_tolerance(q_init_summary, evaluator.config.initial_penetration_tol)
        )
        and full_seed_initially_acceptable
        and len(q_seed) > len(q_init)
    ):
        q_init = q_seed.copy()
        q_init_summary = full_seed_summary
        selected_count = len(q_seed)
        selected_from_full_seed = True
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
        nonpenetrating_init = _summary_within_tolerance(q_init_summary, evaluator.config.penetration_tol)
        acceptable_init = _summary_within_tolerance(q_init_summary, evaluator.config.initial_penetration_tol)
        logger(
            f"[SDF-TrajOpt] Starting optimization: seed_points={len(q_seed)} "
            f"requested_opt_points={config.num_waypoints} actual_opt_points={len(q_init)} "
            f"max_opt_points={config.max_waypoints} selection=adaptive "
            f"init_nonpenetrating={nonpenetrating_init} init_acceptable={acceptable_init} "
            f"maxiter={config.maxiter} stride={config.constraint_point_stride} "
            f"endpoint_relax={config.endpoint_relax_waypoints} endpoint_scale={config.endpoint_safe_distance_scale:.2f}"
        )
        if selected_from_full_seed:
            logger(
                f"[SDF-TrajOpt] Adaptive compression remained penetrating up to {selected_count - 1} points; "
                f"falling back to full nonpenetrating seed with {len(q_seed)} points."
            )
        if not acceptable_init and selected_count == waypoint_counts[-1] and selected_count < len(q_seed):
            logger(
                f"[SDF-TrajOpt] Initial resampled path is still penetrating at capped waypoint count {selected_count}; "
                f"full seed has {len(q_seed)} points."
            )
        if not acceptable_init and not full_seed_initially_acceptable:
            logger(
                "[SDF-TrajOpt] Full RRT seed is also penetrating under SDF evaluation; "
                "this indicates an RRT-vs-SDF feasibility mismatch rather than waypoint compression loss."
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
