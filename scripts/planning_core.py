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


@dataclass(frozen=True)
class TrajOptConfig:
    num_waypoints: int = 20
    maxiter: int = 200
    smoothness_weight: float = 5.0
    path_length_weight: float = 1.0
    seed_weight: float = 0.15
    constraint_edge_resolution: float = 0.08


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
    edge_resolution: float,
) -> np.ndarray:
    path = _reconstruct_trajopt_path(x, q_start, q_goal, n_dof)
    margins = []
    for q in path[1:-1]:
        margins.append(1.0 if is_state_valid(q) else -1.0)
    for i in range(len(path) - 1):
        qa, qb = path[i], path[i + 1]
        if edge_resolution > 0:
            for q in interpolate_edge(qa, qb, edge_resolution)[1:-1]:
                margins.append(1.0 if is_state_valid(q) else -1.0)
        margins.append(1.0 if is_edge_valid(qa, qb) else -1.0)
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
) -> tuple[np.ndarray, bool]:
    if minimize is None:
        if logger is not None:
            logger("[TrajOpt] scipy.optimize is unavailable; skipping optimization.")
        return q_seed, False

    if len(q_seed) <= 2:
        if logger is not None:
            logger("[TrajOpt] Seed path has too few waypoints; skipping optimization.")
        return q_seed, False

    num_waypoints = max(3, min(config.num_waypoints, len(q_seed)))
    q_init = _resample_trajectory(q_seed, num_waypoints)
    q_start = q_init[0]
    q_goal = q_init[-1]
    x0 = q_init[1:-1].reshape(-1)
    if len(x0) == 0:
        if logger is not None:
            logger("[TrajOpt] Seed path has no internal waypoints; skipping optimization.")
        return q_seed, False

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
                config.constraint_edge_resolution,
            ),
        }
    ]
    if logger is not None:
        logger(
            f"[TrajOpt] Starting optimization: seed_points={len(q_seed)} "
            f"opt_points={len(q_init)} maxiter={config.maxiter}"
        )

    result = minimize(
        _trajopt_objective,
        x0,
        args=(q_start, q_goal, q_init, len(lower), config),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": config.maxiter, "disp": False, "ftol": 1e-4},
    )

    q_opt = _reconstruct_trajopt_path(result.x, q_start, q_goal, len(lower))
    valid = bool(np.all(_trajopt_validity_constraint(
        result.x,
        q_start,
        q_goal,
        len(lower),
        is_state_valid,
        is_edge_valid,
        config.constraint_edge_resolution,
    ) >= 0.0))
    success = bool(result.success) and valid
    if logger is not None:
        logger(
            f"[TrajOpt] Finished: success={result.success} accepted={success} "
            f"status={result.status} nit={getattr(result, 'nit', -1)} fun={float(result.fun):.4f}"
        )
    return q_opt, success
