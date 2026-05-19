from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


StateValidator = Callable[[np.ndarray], bool]
EdgeValidator = Callable[[np.ndarray, np.ndarray], bool]
Logger = Callable[[str], None]
TrajOptRunner = Callable[[np.ndarray, Logger | None], tuple[np.ndarray, bool, dict[str, Any]]]


@dataclass(frozen=True)
class TrajOptConfig:
    num_waypoints: int = 20
    maxiter: int = 500
    ftol: float = 1e-3
    smoothness_weight: float = 8.0
    path_length_weight: float = 3.0
    seed_weight: float = 0.05
    constraint_edge_resolution: float = 0.08
    constraint_samples_per_segment: int = 0
    shortcut_iterations: int = 600
    shortcut_passes: int = 6
    averaging_passes: int = 6
    averaging_blend: float = 0.35
    validation_resolution: float = 0.05


def interpolate_edge(q_from: np.ndarray, q_to: np.ndarray, resolution: float) -> np.ndarray:
    dist = float(np.linalg.norm(q_to - q_from))
    steps = max(1, int(math.ceil(dist / resolution)))
    return np.linspace(q_from, q_to, steps + 1)


def densify_path(path: np.ndarray, resolution: float) -> np.ndarray:
    dense = [path[0]]
    for i in range(len(path) - 1):
        dense.extend(interpolate_edge(path[i], path[i + 1], resolution)[1:])
    return np.array(dense)


def _nearest_node_index(nodes: list[np.ndarray], q: np.ndarray) -> int:
    arr = np.array(nodes)
    return int(np.argmin(np.linalg.norm(arr - q, axis=1)))


def _extend_tree(
    nodes: list[np.ndarray],
    parents: list[int],
    q_target: np.ndarray,
    step_size: float,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
) -> tuple[int | None, str]:
    nearest = _nearest_node_index(nodes, q_target)
    q_near = nodes[nearest]
    delta = q_target - q_near
    dist = float(np.linalg.norm(delta))
    if dist < 1e-10:
        return None, "trapped"
    q_new = q_near + min(step_size, dist) * delta / dist
    if not is_state_valid(q_new) or not is_edge_valid(q_near, q_new):
        return None, "trapped"
    nodes.append(q_new)
    parents.append(nearest)
    return len(nodes) - 1, "reached" if dist <= step_size else "advanced"


def _connect_tree(
    nodes: list[np.ndarray],
    parents: list[int],
    q_target: np.ndarray,
    step_size: float,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
) -> tuple[int | None, str]:
    last_idx = None
    while True:
        new_idx, status = _extend_tree(nodes, parents, q_target, step_size, is_state_valid, is_edge_valid)
        if new_idx is None:
            return last_idx, "trapped"
        last_idx = new_idx
        if status == "reached":
            return last_idx, "reached"


def _reconstruct_path(nodes: list[np.ndarray], parents: list[int], idx: int) -> np.ndarray:
    path = []
    while idx != -1:
        path.append(nodes[idx])
        idx = parents[idx]
    return np.array(path[::-1])


def rrt_connect_plan(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
    step_size: float,
    max_iter: int,
    goal_bias: float,
    rng: np.random.Generator,
    logger: Logger | None = None,
) -> np.ndarray:
    if is_edge_valid(q_start, q_goal):
        if logger is not None:
            logger("[RRT-Connect] Direct start-goal edge is valid.")
        return np.vstack([q_start, q_goal])

    start_nodes, goal_nodes = [q_start.copy()], [q_goal.copy()]
    start_parents, goal_parents = [-1], [-1]
    best_goal_distance = float(np.linalg.norm(q_goal - q_start))
    for iteration in range(max_iter):
        q_rand = q_goal if rng.random() < goal_bias else rng.uniform(lower, upper)
        new_idx, _ = _extend_tree(start_nodes, start_parents, q_rand, step_size, is_state_valid, is_edge_valid)
        if new_idx is not None:
            best_goal_distance = min(best_goal_distance, float(np.linalg.norm(start_nodes[new_idx] - q_goal)))
            connect_idx, status = _connect_tree(
                goal_nodes,
                goal_parents,
                start_nodes[new_idx],
                step_size,
                is_state_valid,
                is_edge_valid,
            )
            if status == "reached" and connect_idx is not None:
                path_a = _reconstruct_path(start_nodes, start_parents, new_idx)
                path_b = _reconstruct_path(goal_nodes, goal_parents, connect_idx)
                path = np.vstack([path_a, path_b[::-1][1:]])
                if logger is not None:
                    logger(f"[RRT-Connect] Found path at iter={iteration}, waypoints={len(path)}")
                return path
        start_nodes, goal_nodes = goal_nodes, start_nodes
        start_parents, goal_parents = goal_parents, start_parents

    raise RuntimeError(
        f"RRT-Connect failed after {max_iter} iterations; "
        f"best_goal_distance={best_goal_distance:.4f}, "
        f"tree_sizes=({len(start_nodes)}, {len(goal_nodes)})"
    )


def rrt_connect_plan_with_restarts(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
    step_size: float,
    max_iter: int,
    restarts: int,
    goal_bias: float,
    rng: np.random.Generator,
    logger: Logger | None = None,
) -> np.ndarray:
    last_error: Exception | None = None
    total_attempts = max(1, restarts)
    for attempt in range(1, total_attempts + 1):
        if logger is not None:
            logger(f"[RRT-Connect] Planning attempt {attempt}/{total_attempts}")
        try:
            return rrt_connect_plan(
                q_start=q_start,
                q_goal=q_goal,
                lower=lower,
                upper=upper,
                is_state_valid=is_state_valid,
                is_edge_valid=is_edge_valid,
                step_size=step_size,
                max_iter=max_iter,
                goal_bias=goal_bias,
                rng=rng,
                logger=logger,
            )
        except RuntimeError as exc:
            last_error = exc
            if logger is not None:
                logger(f"[RRT-Connect] Attempt {attempt} failed: {exc}")
    raise RuntimeError(f"RRT-Connect failed after {total_attempts} restarts. Last error: {last_error}")


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


def _path_length(path: np.ndarray) -> float:
    if len(path) <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def validate_path(
    path: np.ndarray,
    is_state_valid: StateValidator,
    resolution: float,
) -> tuple[bool, int | None, np.ndarray | None]:
    for segment_idx in range(len(path) - 1):
        samples = interpolate_edge(path[segment_idx], path[segment_idx + 1], resolution)
        for sample in samples[1:]:
            if not is_state_valid(sample):
                return False, segment_idx, sample
    return True, None, None


def shortcut_smooth_path(
    path: np.ndarray,
    is_edge_valid: EdgeValidator,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(path) <= 2:
        return path.copy()

    best = path.copy()
    for _ in range(max(0, iterations)):
        if len(best) <= 2:
            break
        i = int(rng.integers(0, len(best) - 1))
        j = int(rng.integers(i + 1, len(best)))
        if j <= i + 1:
            continue
        if not is_edge_valid(best[i], best[j]):
            continue
        candidate = np.vstack([best[: i + 1], best[j:]])
        if _path_length(candidate) <= _path_length(best) + 1e-9:
            best = candidate
    return best


def local_average_smooth_path(
    path: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
    passes: int,
    blend: float,
) -> np.ndarray:
    if len(path) <= 2:
        return path.copy()

    smoothed = path.copy()
    alpha = float(np.clip(blend, 0.0, 1.0))
    for _ in range(max(0, passes)):
        updated = smoothed.copy()
        changed = False
        for i in range(1, len(smoothed) - 1):
            midpoint = 0.5 * (smoothed[i - 1] + smoothed[i + 1])
            candidate = np.clip((1.0 - alpha) * smoothed[i] + alpha * midpoint, lower, upper)
            if not is_state_valid(candidate):
                continue
            if not is_edge_valid(updated[i - 1], candidate):
                continue
            if not is_edge_valid(candidate, smoothed[i + 1]):
                continue
            updated[i] = candidate
            changed = True
        smoothed = updated
        if not changed:
            break
    return smoothed


def _reconstruct_trajopt_path(x: np.ndarray, q_start: np.ndarray, q_goal: np.ndarray, n_dof: int) -> np.ndarray:
    if len(x) == 0:
        return np.vstack([q_start, q_goal])
    return np.vstack([q_start, x.reshape(-1, n_dof), q_goal])


def _trajopt_objective(
    x: np.ndarray,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    q_seed: np.ndarray,
    n_dof: int,
    config: TrajOptConfig,
) -> float:
    if len(x) == 0:
        return 0.0

    path = _reconstruct_trajopt_path(x, q_start, q_goal, n_dof)
    smoothness_cost = 0.0
    if len(path) >= 3:
        q_dd = path[2:] - 2.0 * path[1:-1] + path[:-2]
        smoothness_cost = float(np.sum(np.square(q_dd)))

    path_length_cost = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    seed_cost = float(np.sum(np.square(path[1:-1] - q_seed[1:-1])))
    return (
        config.smoothness_weight * smoothness_cost
        + config.path_length_weight * path_length_cost
        + config.seed_weight * seed_cost
    )


def _trajopt_validity_constraint(
    x: np.ndarray,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    n_dof: int,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
    samples_per_segment: int,
) -> np.ndarray:
    path = _reconstruct_trajopt_path(x, q_start, q_goal, n_dof)
    margins = []
    for q in path[1:-1]:
        margins.append(1.0 if is_state_valid(q) else -1.0)
    num_samples = max(0, int(samples_per_segment))
    if num_samples > 0:
        for i in range(len(path) - 1):
            qa, qb = path[i], path[i + 1]
            for alpha in np.linspace(0.0, 1.0, num_samples + 2)[1:-1]:
                q = (1.0 - alpha) * qa + alpha * qb
                margins.append(1.0 if is_state_valid(q) else -1.0)
    if not margins:
        return np.array([1.0], dtype=float)
    return np.array(margins, dtype=float)


def run_trajopt(
    q_seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
    config: TrajOptConfig,
    logger: Logger | None = None,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    if minimize is None:
        if logger is not None:
            logger("[TrajOpt] scipy.optimize is unavailable; skipping optimization.")
        return q_seed, False, {"failure_reason": "solver_unavailable"}

    if len(q_seed) <= 2:
        if logger is not None:
            logger("[TrajOpt] Seed path has too few waypoints; skipping optimization.")
        return q_seed, False, {"failure_reason": "seed_too_short"}

    num_waypoints = max(3, int(config.num_waypoints))
    q_init = _resample_trajectory(q_seed, num_waypoints)
    q_start = q_init[0]
    q_goal = q_init[-1]
    x0 = q_init[1:-1].reshape(-1)
    if len(x0) == 0:
        if logger is not None:
            logger("[TrajOpt] Seed path has no internal waypoints; skipping optimization.")
        return q_seed, False, {"failure_reason": "seed_has_no_internal_waypoints"}

    bounds = []
    for _ in range(len(q_init) - 2):
        for low, high in zip(lower, upper):
            bounds.append((float(low), float(high)))

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: _trajopt_validity_constraint(
                x,
                q_start,
                q_goal,
                len(lower),
                is_state_valid,
                is_edge_valid,
                config.constraint_samples_per_segment,
            ),
        }
    ]
    if logger is not None:
        logger(
            f"[TrajOpt] Starting optimization: seed_points={len(q_seed)} "
            f"opt_points={len(q_init)} maxiter={config.maxiter} ftol={config.ftol:.1e}"
        )

    result = minimize(
        _trajopt_objective,
        x0,
        args=(q_start, q_goal, q_init, len(lower), config),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": config.maxiter, "disp": False, "ftol": config.ftol},
    )

    q_opt = _reconstruct_trajopt_path(result.x, q_start, q_goal, len(lower))
    waypoint_valid = bool(np.all(_trajopt_validity_constraint(
        result.x,
        q_start,
        q_goal,
        len(lower),
        is_state_valid,
        is_edge_valid,
        config.constraint_samples_per_segment,
    ) >= 0.0))
    edge_valid = True
    for i in range(len(q_opt) - 1):
        if not is_edge_valid(q_opt[i], q_opt[i + 1]):
            edge_valid = False
            break
    success = bool(result.success) and waypoint_valid and edge_valid
    failure_reason = "accepted"
    if not bool(result.success):
        failure_reason = "optimizer_failed"
    elif not waypoint_valid:
        failure_reason = "waypoint_constraint_failed"
    elif not edge_valid:
        failure_reason = "edge_constraint_failed"
    info = {
        "optimizer_success": bool(result.success),
        "accepted": bool(success),
        "failure_reason": failure_reason,
        "status": int(getattr(result, "status", -1)),
        "nit": int(getattr(result, "nit", -1)),
        "fun": float(result.fun),
        "waypoint_valid": bool(waypoint_valid),
        "edge_valid": bool(edge_valid),
        "opt_points": int(len(q_init)),
    }
    if logger is not None:
        logger(
            f"[TrajOpt] Finished: success={result.success} accepted={success} "
            f"waypoint_valid={waypoint_valid} edge_valid={edge_valid} "
            f"status={result.status} nit={getattr(result, 'nit', -1)} fun={float(result.fun):.4f}"
        )
    return q_opt, success, info


def optimize_path(
    q_seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
    config: TrajOptConfig,
    rng: np.random.Generator,
    logger: Logger | None = None,
    trajopt_runner: TrajOptRunner | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = q_seed.copy()
    trajopt_seed = q_seed.copy()
    stages: list[str] = ["rrt_seed"]
    accepted_stages: list[str] = ["rrt_seed"]
    validation_resolution = max(1e-6, float(config.validation_resolution))

    seed_valid, invalid_segment, invalid_sample = validate_path(current, is_state_valid, validation_resolution)
    if not seed_valid:
        raise RuntimeError(
            "RRT seed path is not collision-free under dense validation: "
            f"segment={invalid_segment}, sample={np.round(invalid_sample, 4) if invalid_sample is not None else None}"
        )

    def accept_if_safe(candidate: np.ndarray, stage_name: str) -> np.ndarray:
        nonlocal current
        is_valid, bad_segment, bad_sample = validate_path(candidate, is_state_valid, validation_resolution)
        if not is_valid:
            if logger is not None:
                rounded_sample = np.round(bad_sample, 4) if bad_sample is not None else None
                logger(
                    f"[PathOpt] Rejected {stage_name}: collision under dense validation "
                    f"(segment={bad_segment}, sample={rounded_sample})"
                )
            return current
        accepted_stages.append(stage_name)
        return candidate

    for _ in range(max(1, config.shortcut_passes)):
        current = accept_if_safe(
            shortcut_smooth_path(current, is_edge_valid, config.shortcut_iterations, rng),
            "shortcut",
        )
    stages.append("shortcut")

    current = accept_if_safe(
        local_average_smooth_path(
            current,
            lower=lower,
            upper=upper,
            is_state_valid=is_state_valid,
            is_edge_valid=is_edge_valid,
            passes=config.averaging_passes,
            blend=config.averaging_blend,
        ),
        "average",
    )
    stages.append("average")

    if trajopt_runner is None:
        q_trajopt, trajopt_success, trajopt_info = run_trajopt(
            q_seed=current,
            lower=lower,
            upper=upper,
            is_state_valid=is_state_valid,
            is_edge_valid=is_edge_valid,
            config=config,
            logger=logger,
        )
    else:
        if logger is not None and len(trajopt_seed) != len(current):
            logger(
                f"[PathOpt] Using conservative RRT seed for external trajopt: "
                f"rrt_waypoints={len(trajopt_seed)} geometric_waypoints={len(current)}"
            )
        q_trajopt, trajopt_success, trajopt_info = trajopt_runner(trajopt_seed, logger)
    if trajopt_success:
        current = accept_if_safe(q_trajopt, "trajopt")
        stages.append("trajopt")
    else:
        trajopt_info = dict(trajopt_info)
        trajopt_info.setdefault("accepted", False)

    current = accept_if_safe(
        local_average_smooth_path(
            current,
            lower=lower,
            upper=upper,
            is_state_valid=is_state_valid,
            is_edge_valid=is_edge_valid,
            passes=max(1, config.averaging_passes // 2),
            blend=min(0.5, config.averaging_blend),
        ),
        "post_average",
    )
    stages.append("post_average")

    if logger is not None:
        logger(
            f"[PathOpt] stages={'+'.join(stages)} accepted={'+'.join(accepted_stages)} "
            f"seed_waypoints={len(q_seed)} final_waypoints={len(current)} "
            f"seed_length={_path_length(q_seed):.4f} final_length={_path_length(current):.4f}"
        )
    return current, {
        "trajopt_success": trajopt_success,
        "trajopt_info": trajopt_info,
        "stages": stages,
        "accepted_stages": accepted_stages,
    }
