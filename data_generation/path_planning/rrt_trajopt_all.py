import time
import os
import sys
import select
import argparse
import json
import numpy as np
import pybullet as p
import pybullet_data
import trimesh
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates
from weld_path_parser import load_weld_path_transitions

# ======================= 配置参数 =======================
urdf_path = "./config/urdf/zj-robot.urdf"
STL_OBSTACLE_PATH = "./zj.stl"
STL_OBSTACLE_SCALE = 7.5e-05
STL_OBSTACLE_POSITION_XYZ = [0.5, -3.3, 0.5] 
STL_OBSTACLE_ORIENTATION_XYZW = [0.0, 0.0, 0.0, 1.0]

END_EFFECTOR_LINK_NAME = "Circle_link"
TORCH_STL_PATH = "./config/meshes/stl/welding_torch.stl"
WELD_POSES_JSON_PATH = "./zj_all_weld_poses_scaled_0075.json"

# Circle_link_joint 变换（welding_torch_link → Circle_link）
TORCH_TO_EE_XYZ = [-0.01115, 0.08448, -0.71562]
TORCH_TO_EE_RPY = [2.5410, -0.0338, -2.9778]

INITIAL_EE_POSITION_XYZ = [
    -0.4850029884685424+ 0.5,
    1.9413121908323476 - 3.3,
    0.1316106889627408 + 0.5,
]
USE_INITIAL_EE_ORIENTATION = True
INITIAL_EE_ORIENTATION_XYZW = [
          -0.3460026116421203,
          0.9260879050786919,
          0.1409601513513542,
          -0.05266517383238341
]

TARGET_POSITION_XYZ = [
    -0.183749999999999977 + 0.50,
    1.679625 - 3.3,
    0.07499999999999989 + 0.5,
]
USE_TARGET_ORIENTATION = True
TARGET_ORIENTATION_XYZW = [
          -0.3981126085090628,
          0.7690945006604258,
          0.4440369169885576,
          -0.2298504216904915
]

PLAYBACK_DT = 0.15
PLAYBACK_EDGE_RESOLUTION = 0.02
PLAYBACK_PATH_RESOLUTION = 0.005
PLAYBACK_LOOP_PAUSE = 0.2
GOAL_RETREAT_STEP = 0.001
GOAL_RETREAT_MAX_STEPS = 2

CHECK_ARM_STEP_SIZE = 0.02

D_SAFE_ARM = 0.05
D_SAFE_EE = 0.01
SDF_PENETRATION_TOL = -0.001

RRT_STEP_SIZE = 0.30
RRT_EDGE_CHECK_RESOLUTION = 0.01
RRT_MAX_ITER = 5000
RRT_GOAL_BIAS = 0.20
RRT_INFORMED_BIAS = 0.85
RRT_GOAL_THRESHOLD = 0.20
RRT_IMPROVEMENT_PATIENCE = 50
TRAJECTORY_EDGE_CHECK_RESOLUTION = 0.01
DEFAULT_REPLAY_SAVE_PATH = "./outputs/rrt_replay_segments.npz"


class FastNumPyFK:
    """纯 NumPy 实现的正运动学求解器。"""
    def __init__(self, robot_id, joint_indices):
        self.num_joints = p.getNumJoints(robot_id)
        self.joint_indices = joint_indices
        self.q_index_map = {j: i for i, j in enumerate(self.joint_indices)}
        self.parents = {}
        self.local_T = {}
        self.joint_axes = {}
        self.joint_types = {}
        self.arm_alphas = {}

        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        self.base_T = np.eye(4)
        self.base_T[:3, :3] = Rotation.from_quat(base_orn).as_matrix()
        self.base_T[:3, 3] = base_pos

        for i in range(self.num_joints):
            info = p.getJointInfo(robot_id, i)
            self.joint_types[i] = info[2]
            self.joint_axes[i] = np.array(info[13])
            origin_xyz = np.array(info[14])
            origin_quat = np.array(info[15])
            self.parents[i] = info[16]

            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(origin_quat).as_matrix()
            T[:3, 3] = origin_xyz
            self.local_T[i] = T

            seg_len = float(np.linalg.norm(origin_xyz))
            n_chk = max(1, int(np.ceil(seg_len / CHECK_ARM_STEP_SIZE)))
            self.arm_alphas[i] = np.linspace(0.0, 1.0, n_chk + 1)[:, None]

    def _rodrigues_rotation(self, axis, angle):
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def compute_link_states(self, q):
        global_T = {-1: self.base_T}
        for i in range(self.num_joints):
            parent = self.parents[i]
            T_parent = global_T[parent]
            T_local = self.local_T[i]
            if i in self.q_index_map:
                q_val = q[self.q_index_map[i]]
                j_type = self.joint_types[i]
                T_motion = np.eye(4)
                if j_type == p.JOINT_REVOLUTE:
                    T_motion[:3, :3] = self._rodrigues_rotation(self.joint_axes[i], q_val)
                elif j_type == p.JOINT_PRISMATIC:
                    T_motion[:3, 3] = self.joint_axes[i] * q_val
                global_T[i] = T_parent @ T_local @ T_motion
            else:
                global_T[i] = T_parent @ T_local
        return global_T


class SDFCollisionLayer:
    def __init__(self, npz_path="workpiece_sdf.npz"):
        if not os.path.exists(npz_path):
            print(f"[错误] 找不到 SDF 预计算文件: {npz_path}")
            sys.exit(1)

        print(f"[SDF] 加载预计算碰撞矩阵: {npz_path}")
        data = np.load(npz_path)
        self.sdf_grid = data["sdf"]

        median_val = float(np.median(self.sdf_grid))
        if median_val < 0:
            print(f"[SDF] 检测到符号翻转（median={median_val:.4f}m），自动修正。")
            self.sdf_grid = -self.sdf_grid
        else:
            print(f"[SDF] 符号正常（median={median_val:.4f}m）。")

        self.x = data["x"]
        self.y = data["y"]
        self.z = data["z"]
        self.origin = np.array([self.x[0], self.y[0], self.z[0]])
        self.spacing = np.array([
            self.x[1] - self.x[0],
            self.y[1] - self.y[0],
            self.z[1] - self.z[0],
        ])

    def get_distances(self, points):
        if len(points) == 0:
            return np.array([])
        indices = ((points - self.origin) / self.spacing).T
        return map_coordinates(
            self.sdf_grid,
            indices,
            order=1,
            mode="constant",
            cval=1.0,
        )


def get_tool_points(
    stl_path,
    scale=1.0,
    torch_to_ee_xyz=None,
    torch_to_ee_rpy=None,
    ring_points=30,
    ring_spacing=0.001,
):
    def _build_rings(verts, centroid_global, axis, spacing):
        proj = (verts - centroid_global) @ axis
        s_min, s_max = proj.min(), proj.max()
        bin_edges = np.arange(s_min, s_max + spacing, spacing)
        if len(bin_edges) < 2:
            bin_edges = np.array([s_min, s_max])

        spine_pts, spine_R = [], []
        for k in range(len(bin_edges) - 1):
            mask = (proj >= bin_edges[k]) & (proj < bin_edges[k + 1])
            if mask.sum() < 3:
                continue
            layer_v = verts[mask]
            c = layer_v.mean(axis=0)
            diff = layer_v - c
            perp_vecs = diff - (diff @ axis)[:, None] * axis
            r = float(np.median(np.linalg.norm(perp_vecs, axis=1)))
            spine_pts.append(c)
            spine_R.append(max(r, 1e-3))

        if len(spine_pts) < 2:
            return np.zeros((ring_points, 3))

        spine_pts = np.array(spine_pts)
        spine_R = np.array(spine_R)

        seg = np.diff(spine_pts, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = float(cum_len[-1])
        if total_len > 1e-8:
            n_seg = max(1, int(np.round(total_len / spacing)))
            s_new = np.linspace(0.0, total_len, n_seg + 1)
            spine_pts = np.column_stack([
                np.interp(s_new, cum_len, spine_pts[:, 0]),
                np.interp(s_new, cum_len, spine_pts[:, 1]),
                np.interp(s_new, cum_len, spine_pts[:, 2]),
            ])
            spine_R = np.interp(s_new, cum_len, spine_R)
        else:
            spine_pts = spine_pts[:2]
            spine_R = spine_R[:2]

        tangents = np.zeros_like(spine_pts)
        tangents[0] = spine_pts[1] - spine_pts[0]
        tangents[-1] = spine_pts[-1] - spine_pts[-2]
        tangents[1:-1] = spine_pts[2:] - spine_pts[:-2]
        t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.where(t_norms < 1e-8, 1.0, t_norms)

        t0 = tangents[0]
        ref = np.array([0.0, 0.0, 1.0])
        if abs(t0 @ ref) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        n0 = ref - (ref @ t0) * t0
        n0 /= np.linalg.norm(n0)

        angles = np.linspace(0.0, 2.0 * np.pi, ring_points, endpoint=False)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        all_pts = []
        prev_n = n0
        for k in range(len(spine_pts)):
            t = tangents[k]
            n = prev_n - (prev_n @ t) * t
            n_len = np.linalg.norm(n)
            n = (n / n_len) if n_len > 1e-8 else prev_n
            b = np.cross(t, n)
            b_len = np.linalg.norm(b)
            if b_len > 1e-8:
                b = b / b_len
            prev_n = n
            ring = spine_pts[k] + spine_R[k] * (cos_a[:, None] * n + sin_a[:, None] * b)
            all_pts.append(ring)

        return np.vstack(all_pts)

    try:
        mesh = trimesh.load(stl_path, force="mesh")
        if scale != 1.0:
            tf = np.eye(4)
            tf[:3, :3] *= scale
            mesh.apply_transform(tf)
        verts = np.array(mesh.vertices, dtype=np.float64)
        centroid_global = verts.mean(axis=0)
        _, _, vt = np.linalg.svd(verts - centroid_global, full_matrices=False)
        axis = vt[0]
        pts = _build_rings(verts, centroid_global, axis, ring_spacing)
    except Exception as exc:
        print(f"[Tool] 焊枪 STL 加载失败，退化为线段采样: {exc}")
        pts = np.array([[0, 0, z] for z in np.linspace(0, 0.25, ring_points * 2)])

    if torch_to_ee_xyz is not None and torch_to_ee_rpy is not None:
        t_vec = np.array(torch_to_ee_xyz, dtype=np.float64)
        R = Rotation.from_euler("xyz", torch_to_ee_rpy).as_matrix()

        def _to_ee(pts):
            return (R.T @ (pts - t_vec).T).T

        pts = _to_ee(pts)

    print(f"[Tool] 焊枪采样点: check={len(pts)}")
    return pts


def get_dense_robot_points(robot_id, joint_indices, fk_solver, q, ee_link, local_tool_points):
    set_joint_positions(robot_id, joint_indices, q)

    base_pos, _ = p.getBasePositionAndOrientation(robot_id)
    base_pos = np.array(base_pos, dtype=np.float64)
    link_world_pos = {-1: base_pos}
    for i in range(fk_solver.num_joints):
        state = p.getLinkState(robot_id, i, computeForwardKinematics=True)
        link_world_pos[i] = np.array(state[4], dtype=np.float64)

    ee_state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    ee_pos = np.array(ee_state[4], dtype=np.float64)
    ee_orn = np.array(ee_state[5], dtype=np.float64)
    ee_rot = np.array(p.getMatrixFromQuaternion(ee_orn), dtype=np.float64).reshape(3, 3)
    ee_points = (ee_rot @ local_tool_points.T).T + ee_pos

    arm_segments = []
    for i in range(fk_solver.num_joints):
        if i == ee_link:
            continue
        p_start = link_world_pos[fk_solver.parents[i]]
        p_end = link_world_pos[i]
        alphas = fk_solver.arm_alphas[i]
        arm_segments.append((1 - alphas) * p_start + alphas * p_end)

    arm_points = np.vstack(arm_segments) if arm_segments else np.empty((0, 3))
    return arm_points, ee_points


def summarize_trajectory_collision(robot_id, joint_indices, fk_solver, Q, sdf_layer, ee_link, local_tool_points):
    penetration_count = 0
    near_count = 0
    arm_min_global = np.inf
    ee_min_global = np.inf

    for q in Q:
        arm_pts, ee_pts = get_dense_robot_points(
            robot_id, joint_indices, fk_solver, q, ee_link, local_tool_points
        )
        dist_arm_min = float(sdf_layer.get_distances(arm_pts).min()) if len(arm_pts) > 0 else 999.0
        dist_ee_min = float(sdf_layer.get_distances(ee_pts).min()) if len(ee_pts) > 0 else 999.0

        arm_min_global = min(arm_min_global, dist_arm_min)
        ee_min_global = min(ee_min_global, dist_ee_min)

        arm_pen = dist_arm_min < SDF_PENETRATION_TOL
        arm_near = (not arm_pen) and (dist_arm_min < D_SAFE_ARM)
        ee_pen = dist_ee_min < SDF_PENETRATION_TOL
        ee_near = (not ee_pen) and (dist_ee_min < D_SAFE_EE)

        if arm_pen or ee_pen:
            penetration_count += 1
        elif arm_near or ee_near:
            near_count += 1

    return {
        "penetration_count": penetration_count,
        "near_count": near_count,
        "arm_min_global": float(arm_min_global),
        "ee_min_global": float(ee_min_global),
    }


def evaluate_state_sdf(robot_id, joint_indices, fk_solver, sdf_layer, ee_link, local_tool_points, q):
    arm_pts, ee_pts = get_dense_robot_points(
        robot_id, joint_indices, fk_solver, q, ee_link, local_tool_points
    )
    dist_arm = sdf_layer.get_distances(arm_pts) if len(arm_pts) > 0 else np.array([999.0])
    dist_ee = sdf_layer.get_distances(ee_pts) if len(ee_pts) > 0 else np.array([999.0])
    dist_arm_min = float(np.min(dist_arm))
    dist_ee_min = float(np.min(dist_ee))

    arm_pen = dist_arm_min < SDF_PENETRATION_TOL
    arm_near = (not arm_pen) and (dist_arm_min < D_SAFE_ARM)
    ee_pen = dist_ee_min < SDF_PENETRATION_TOL
    ee_near = (not ee_pen) and (dist_ee_min < D_SAFE_EE)

    return {
        "arm_min": dist_arm_min,
        "ee_min": dist_ee_min,
        "arm_pen": arm_pen,
        "arm_near": arm_near,
        "ee_pen": ee_pen,
        "ee_near": ee_near,
        "valid": (dist_arm_min >= D_SAFE_ARM) and (dist_ee_min >= D_SAFE_EE),
    }


def densify_trajectory_for_collision_check(Q, resolution=TRAJECTORY_EDGE_CHECK_RESOLUTION):
    if len(Q) <= 1:
        return Q.copy()
    dense = [Q[0]]
    for i in range(len(Q) - 1):
        edge = interpolate_edge(Q[i], Q[i + 1], resolution)
        dense.extend(edge[1:])
    return np.array(dense)


def get_movable_joints(robot_id):
    joint_indices, joint_names, lower_limits, upper_limits = [], [], [], []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_indices.append(j)
            joint_names.append(info[1].decode("utf-8"))
            low, high = info[8], info[9]
            if low > high:
                low, high = -np.pi, np.pi
            lower_limits.append(low)
            upper_limits.append(high)
    return joint_indices, joint_names, np.array(lower_limits), np.array(upper_limits)


def solve_ik(robot_id, ee_link, target_pos, target_orn=None):
    if target_orn is None:
        return np.array(p.calculateInverseKinematics(robot_id, ee_link, target_pos))
    return np.array(
        p.calculateInverseKinematics(robot_id, ee_link, target_pos, targetOrientation=target_orn)
    )


def adjust_pose_out_of_collision(
    robot_id,
    joint_indices,
    lower_limits,
    upper_limits,
    fk_solver,
    sdf_layer,
    ee_link,
    local_tool_points,
    target_pos,
    target_orn,
    label="pose",
):
    target_pos = np.array(target_pos, dtype=np.float64)
    q_goal = solve_ik(robot_id, ee_link, target_pos, target_orn=target_orn)[: len(joint_indices)]
    q_goal = np.clip(q_goal, lower_limits, upper_limits)
    initial_eval = evaluate_state_sdf(
        robot_id, joint_indices, fk_solver, sdf_layer, ee_link, local_tool_points, q_goal
    )
    should_retreat = (
        initial_eval["arm_min"] < SDF_PENETRATION_TOL
        or initial_eval["ee_min"] < SDF_PENETRATION_TOL
    )
    if not should_retreat:
        return target_pos, q_goal, initial_eval, 0, initial_eval

    set_joint_positions(robot_id, joint_indices, q_goal)
    p.stepSimulation()
    ee_state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    ee_rot = np.array(p.getMatrixFromQuaternion(ee_state[5]), dtype=np.float64).reshape(3, 3)
    retreat_dir = -ee_rot[:, 2]
    retreat_norm = np.linalg.norm(retreat_dir)
    if retreat_norm < 1e-8:
        retreat_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        retreat_dir = retreat_dir / retreat_norm

    best_pos = target_pos.copy()
    best_q = q_goal.copy()
    best_eval = initial_eval
    for step_idx in range(1, GOAL_RETREAT_MAX_STEPS + 1):
        trial_pos = target_pos + retreat_dir * (GOAL_RETREAT_STEP * step_idx)
        trial_q = solve_ik(robot_id, ee_link, trial_pos, target_orn=target_orn)[: len(joint_indices)]
        trial_q = np.clip(trial_q, lower_limits, upper_limits)
        trial_eval = evaluate_state_sdf(
            robot_id, joint_indices, fk_solver, sdf_layer, ee_link, local_tool_points, trial_q
        )
        best_pos = trial_pos
        best_q = trial_q
        best_eval = trial_eval
        if trial_eval["valid"]:
            return best_pos, best_q, best_eval, step_idx, initial_eval

    return best_pos, best_q, best_eval, GOAL_RETREAT_MAX_STEPS, initial_eval


def set_joint_positions(robot_id, joint_indices, q):
    for idx, joint_id in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_id, q[idx])


def visualize_tool_points_pybullet(
    robot_id,
    ee_link,
    local_tool_points,
    color_rgb=(1.0, 0.2, 0.2),
    point_size=6,
    batch_size=1000,
):
    link_state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    ee_pos = np.array(link_state[4], dtype=np.float64)
    ee_orn = np.array(link_state[5], dtype=np.float64)
    ee_rot = np.array(p.getMatrixFromQuaternion(ee_orn), dtype=np.float64).reshape(3, 3)
    world_tool_points = (ee_rot @ local_tool_points.T).T + ee_pos

    color = np.array(color_rgb, dtype=np.float64)
    color_block = np.repeat(color[None, :], min(batch_size, len(world_tool_points)), axis=0)
    for start in range(0, len(world_tool_points), batch_size):
        end = min(start + batch_size, len(world_tool_points))
        p.addUserDebugPoints(
            pointPositions=world_tool_points[start:end].tolist(),
            pointColorsRGB=color_block[: end - start].tolist(),
            pointSize=point_size,
        )

    p.addUserDebugText(
        "Torch sample points",
        (ee_pos + np.array([0.0, 0.0, 0.08])).tolist(),
        textColorRGB=[1.0, 0.4, 0.4],
        textSize=1.2,
    )
    print(f"[可视化] 已显示焊枪采样点: {len(world_tool_points)}")


def visualize_sdf_zero_level_pybullet(
    sdf_layer,
    band=None,
    max_points=8000,
    color_rgb=(0.2, 0.55, 1.0),
    point_size=3,
    batch_size=1000,
    seed=0,
):
    if band is None:
        band = 0.75 * float(np.max(np.abs(sdf_layer.spacing)))

    mask = np.abs(sdf_layer.sdf_grid) <= band
    idx = np.argwhere(mask)
    if len(idx) == 0:
        print("[可视化] SDF 0 距离面点云为空。")
        return

    if len(idx) > max_points:
        rng = np.random.default_rng(seed)
        idx = idx[rng.choice(len(idx), size=max_points, replace=False)]

    points = sdf_layer.origin[None, :] + idx * sdf_layer.spacing[None, :]
    color = np.array(color_rgb, dtype=np.float64)
    color_block = np.repeat(color[None, :], min(batch_size, len(points)), axis=0)
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        p.addUserDebugPoints(
            pointPositions=points[start:end].tolist(),
            pointColorsRGB=color_block[: end - start].tolist(),
            pointSize=point_size,
        )

    center = points.mean(axis=0)
    p.addUserDebugText(
        f"SDF |d| <= {band:.4f} m",
        (center + np.array([0.0, 0.0, 0.08])).tolist(),
        textColorRGB=[0.3, 0.7, 1.0],
        textSize=1.2,
    )
    print(f"[可视化] 已显示 SDF 0 距离面近似点云: {len(points)}")


def play_trajectory(robot_id, joint_indices, Q, dt=0.08):
    for q in Q:
        set_joint_positions(robot_id, joint_indices, q)
        p.stepSimulation()
        time.sleep(dt)


def get_ee_path_world_positions(robot_id, joint_indices, Q, ee_link):
    pts = []
    for q in Q:
        set_joint_positions(robot_id, joint_indices, q)
        p.stepSimulation()
        state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
        pts.append(np.array(state[4], dtype=np.float64))
    return np.array(pts)


def draw_ee_trajectory_lines(robot_id, joint_indices, Q, ee_link, color_rgb, line_width=2.0):
    pts = get_ee_path_world_positions(robot_id, joint_indices, Q, ee_link)
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        p.addUserDebugLine(
            pts[i].tolist(),
            pts[i + 1].tolist(),
            lineColorRGB=list(color_rgb),
            lineWidth=line_width,
        )

def draw_world_trajectory_lines(points, color_rgb, line_width=2.5):
    if len(points) < 2:
        return []
    ids = []
    for i in range(len(points) - 1):
        ids.append(
            p.addUserDebugLine(
                points[i].tolist(),
                points[i + 1].tolist(),
                lineColorRGB=list(color_rgb),
                lineWidth=line_width,
                lifeTime=0,
            )
        )
    return ids


def wait_for_start_signal():
    print("\n[回放] 已到初始位姿。按 y 开始，按 q 退出。")
    while p.isConnected():
        p.stepSimulation()
        keys = p.getKeyboardEvents()
        if ord("y") in keys and keys[ord("y")] & p.KEY_WAS_TRIGGERED:
            return True
        if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
            return False
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if ready:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == "y":
                    return True
                if cmd == "q":
                    return False
        except Exception:
            pass
        time.sleep(1.0 / 240.0)
    return False


def setup_pybullet(gui=True):
    client = p.connect(p.GUI if gui else p.DIRECT)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)

    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=STL_OBSTACLE_PATH,
        meshScale=[STL_OBSTACLE_SCALE] * 3,
        rgbaColor=[0.6, 0.6, 0.6, 1.0],
    )
    obstacle_id = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=STL_OBSTACLE_POSITION_XYZ,
        baseOrientation=STL_OBSTACLE_ORIENTATION_XYZW,
    )
    return client, robot_id, [obstacle_id]


def transform_json_xyz(xyz):
    p_xyz = np.asarray(xyz, dtype=np.float64) * 0.001
    p_xyz[0] += 0.5
    p_xyz[1] -= 3.3
    p_xyz[2] += 0.5
    return p_xyz.tolist()


def draw_replay_debug_visual(seg):
    s = seg["start_pos"]
    g = seg["goal_pos"]
    ee_path = np.asarray(seg["ee_path"], dtype=np.float64)
    ids = []

    ids.extend(draw_world_trajectory_lines(ee_path, [0.12, 0.9, 0.75], line_width=2.8))

    d = 0.02
    ids.append(p.addUserDebugLine([s[0] - d, s[1], s[2]], [s[0] + d, s[1], s[2]], [0.0, 0.0, 0.0], 6.0, 0))
    ids.append(p.addUserDebugLine([s[0], s[1] - d, s[2]], [s[0], s[1] + d, s[2]], [0.0, 0.0, 0.0], 6.0, 0))
    ids.append(p.addUserDebugLine([s[0], s[1], s[2] - d], [s[0], s[1], s[2] + d], [0.0, 0.0, 0.0], 6.0, 0))
    ids.append(p.addUserDebugLine([s[0] - d, s[1], s[2]], [s[0] + d, s[1], s[2]], [1.0, 0.2, 0.2], 4.0, 0))
    ids.append(p.addUserDebugLine([s[0], s[1] - d, s[2]], [s[0], s[1] + d, s[2]], [1.0, 0.2, 0.2], 4.0, 0))
    ids.append(p.addUserDebugLine([s[0], s[1], s[2] - d], [s[0], s[1], s[2] + d], [1.0, 0.2, 0.2], 4.0, 0))

    start_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.009, rgbaColor=[0.1, 1.0, 0.1, 0.95])
    end_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.009, rgbaColor=[1.0, 0.1, 0.1, 0.95])
    start_body = p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=start_shape, basePosition=s)
    end_body = p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=end_shape, basePosition=g)
    return {"debug_ids": ids, "body_ids": [start_body, end_body]}


def clear_debug_visuals(vis_handles):
    for item_id in vis_handles.get("debug_ids", []):
        p.removeUserDebugItem(item_id)
    for body_id in vis_handles.get("body_ids", []):
        if body_id >= 0:
            p.removeBody(body_id)


def pack_segments(segments):
    if not segments:
        return np.empty((0, 0), dtype=np.float64), np.array([], dtype=np.int32)
    lengths = np.array([len(seg) for seg in segments], dtype=np.int32)
    packed = np.vstack(segments)
    return packed, lengths


def unpack_segments(packed, lengths):
    segments = []
    offset = 0
    for seg_len in lengths:
        next_offset = offset + int(seg_len)
        segments.append(np.array(packed[offset:next_offset], dtype=np.float64))
        offset = next_offset
    return segments


def save_replay_bundle(save_path, segments, segment_visual_data, metadata):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    q_packed, q_lengths = pack_segments([seg["q_path"] for seg in segments])
    playback_packed, playback_lengths = pack_segments([seg["q_playback"] for seg in segments])
    ee_packed, ee_lengths = pack_segments([seg["ee_path"] for seg in segment_visual_data])
    start_positions = np.array([seg["start_pos"] for seg in segment_visual_data], dtype=np.float64)
    goal_positions = np.array([seg["goal_pos"] for seg in segment_visual_data], dtype=np.float64)
    labels = np.array([seg["label"] for seg in segment_visual_data], dtype="<U16")
    np.savez_compressed(
        save_path,
        q_packed=q_packed,
        q_lengths=q_lengths,
        playback_packed=playback_packed,
        playback_lengths=playback_lengths,
        ee_packed=ee_packed,
        ee_lengths=ee_lengths,
        start_positions=start_positions,
        goal_positions=goal_positions,
        labels=labels,
        metadata=np.array(json.dumps(metadata), dtype="<U4096"),
    )
    print(f"[保存] 规划结果已写入: {save_path}")


def load_replay_bundle(save_path):
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"未找到回放文件: {save_path}")
    data = np.load(save_path, allow_pickle=False)
    q_segments = unpack_segments(data["q_packed"], data["q_lengths"])
    playback_segments = unpack_segments(data["playback_packed"], data["playback_lengths"])
    ee_segments = unpack_segments(data["ee_packed"], data["ee_lengths"])
    metadata = json.loads(str(data["metadata"]))
    segment_visual_data = []
    for idx, (start_pos, goal_pos, ee_path, label) in enumerate(
        zip(data["start_positions"], data["goal_positions"], ee_segments, data["labels"]),
        start=1,
    ):
        segment_visual_data.append(
            {
                "start_pos": np.array(start_pos, dtype=np.float64),
                "goal_pos": np.array(goal_pos, dtype=np.float64),
                "ee_path": np.array(ee_path, dtype=np.float64),
                "label": str(label) if str(label) else f"{idx:02d}",
            }
        )
    segments = []
    for q_path, q_playback in zip(q_segments, playback_segments):
        segments.append({"q_path": q_path, "q_playback": q_playback})
    print(f"[加载] 已读取回放文件: {save_path}")
    return segments, segment_visual_data, metadata


def check_state_valid_sdf(
    robot_id,
    joint_indices,
    fk_solver,
    sdf_layer,
    ee_link,
    local_tool_points,
    q,
):
    state_eval = evaluate_state_sdf(
        robot_id, joint_indices, fk_solver, sdf_layer, ee_link, local_tool_points, q
    )
    return state_eval["valid"]


def interpolate_edge(q_from, q_to, resolution):
    diff = q_to - q_from
    dist = float(np.linalg.norm(diff))
    steps = max(1, int(np.ceil(dist / resolution)))
    return np.linspace(q_from, q_to, steps + 1)


def check_edge_valid_sdf(
    robot_id,
    joint_indices,
    fk_solver,
    sdf_layer,
    ee_link,
    local_tool_points,
    q_from,
    q_to,
    resolution,
):
    for q in interpolate_edge(q_from, q_to, resolution)[1:]:
        if not check_state_valid_sdf(
            robot_id, joint_indices, fk_solver, sdf_layer, ee_link, local_tool_points, q
        ):
            return False
    return True


def sample_unit_n_ball(dim, rng):
    direction = rng.normal(size=dim)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        direction[0] = 1.0
        norm = 1.0
    direction /= norm
    radius = rng.random() ** (1.0 / dim)
    return direction * radius


def rotation_to_vector(unit_vec):
    dim = len(unit_vec)
    basis = np.eye(dim)
    if np.allclose(unit_vec, basis[:, 0]):
        return basis
    mat = basis.copy()
    mat[:, 0] = unit_vec
    for i in range(1, dim):
        v = mat[:, i]
        for j in range(i):
            v = v - np.dot(v, mat[:, j]) * mat[:, j]
        n = np.linalg.norm(v)
        if n < 1e-10:
            for candidate in basis.T:
                v = candidate.copy()
                for j in range(i):
                    v = v - np.dot(v, mat[:, j]) * mat[:, j]
                n = np.linalg.norm(v)
                if n >= 1e-10:
                    break
        mat[:, i] = v / max(n, 1e-10)
    return mat


def informed_sample(q_start, q_goal, c_best, lower_limits, upper_limits, rng):
    dim = len(q_start)
    c_min = float(np.linalg.norm(q_goal - q_start))
    if not np.isfinite(c_best) or c_best <= c_min + 1e-9:
        return rng.uniform(lower_limits, upper_limits)

    center = 0.5 * (q_start + q_goal)
    a1 = (q_goal - q_start) / max(c_min, 1e-12)
    C = rotation_to_vector(a1)

    r1 = c_best / 2.0
    if dim == 1:
        radii = np.array([r1])
    else:
        rn = np.sqrt(max(c_best ** 2 - c_min ** 2, 0.0)) / 2.0
        radii = np.array([r1] + [rn] * (dim - 1))

    L = np.diag(radii)
    for _ in range(100):
        x_ball = sample_unit_n_ball(dim, rng)
        sample = center + C @ (L @ x_ball)
        if np.all(sample >= lower_limits) and np.all(sample <= upper_limits):
            return sample
    return np.clip(sample, lower_limits, upper_limits)


def reconstruct_path(nodes, parents, goal_idx):
    path = []
    idx = goal_idx
    while idx != -1:
        path.append(nodes[idx])
        idx = parents[idx]
    path.reverse()
    return np.array(path)


def _nearest_node_index(nodes, q):
    nodes_arr = np.array(nodes)
    return int(np.argmin(np.linalg.norm(nodes_arr - q, axis=1)))


def _extend_tree(nodes, parents, q_target, step_size, is_state_valid, is_edge_valid):
    nearest_idx = _nearest_node_index(nodes, q_target)
    q_near = nodes[nearest_idx]

    direction = q_target - q_near
    dist = float(np.linalg.norm(direction))
    if dist < 1e-10:
        return None, "trapped"

    q_new = q_near + min(step_size, dist) * (direction / dist)
    if not is_state_valid(q_new):
        return None, "trapped"
    if not is_edge_valid(q_near, q_new):
        return None, "trapped"

    nodes.append(q_new)
    parents.append(nearest_idx)
    status = "reached" if dist <= step_size else "advanced"
    return len(nodes) - 1, status


def _connect_tree(nodes, parents, q_target, step_size, is_state_valid, is_edge_valid):
    last_new_idx = None
    while True:
        new_idx, status = _extend_tree(
            nodes, parents, q_target, step_size, is_state_valid, is_edge_valid
        )
        if new_idx is None:
            return last_new_idx, "trapped"
        last_new_idx = new_idx
        if status == "reached":
            return last_new_idx, "reached"


def _reconstruct_bidirectional_path(nodes_a, parents_a, idx_a, nodes_b, parents_b, idx_b):
    path_a = reconstruct_path(nodes_a, parents_a, idx_a)
    path_b = reconstruct_path(nodes_b, parents_b, idx_b)
    return np.vstack([path_a, path_b[::-1][1:]])


def rrt_connect_plan(
    q_start,
    q_goal,
    lower_limits,
    upper_limits,
    is_state_valid,
    is_edge_valid,
    step_size,
    max_iter,
    goal_bias,
    goal_threshold,
    informed_bias,
    rng,
):
    _ = goal_threshold
    _ = informed_bias

    start_nodes = [q_start.copy()]
    start_parents = [-1]
    goal_nodes = [q_goal.copy()]
    goal_parents = [-1]

    for it in range(max_iter):
        if rng.random() < goal_bias:
            q_rand = q_goal
        else:
            q_rand = rng.uniform(lower_limits, upper_limits)

        new_idx, extend_status = _extend_tree(
            start_nodes, start_parents, q_rand, step_size, is_state_valid, is_edge_valid
        )
        if new_idx is None:
            start_nodes, goal_nodes = goal_nodes, start_nodes
            start_parents, goal_parents = goal_parents, start_parents
            continue

        connect_idx, connect_status = _connect_tree(
            goal_nodes,
            goal_parents,
            start_nodes[new_idx],
            step_size,
            is_state_valid,
            is_edge_valid,
        )
        if connect_status == "reached" and connect_idx is not None:
            path = _reconstruct_bidirectional_path(
                start_nodes, start_parents, new_idx, goal_nodes, goal_parents, connect_idx
            )
            path_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
            print(f"[RRT-Connect] 迭代 {it:4d}: 找到可行解，路径长度={path_len:.4f}")
            return path, path_len

        start_nodes, goal_nodes = goal_nodes, start_nodes
        start_parents, goal_parents = goal_parents, start_parents

    raise RuntimeError(f"RRT-Connect 在 {max_iter} 次迭代内未找到无碰撞路径。")


def check_trajectory_collisions_sdf(
    robot_id, joint_indices, fk_solver, Q, sdf_layer, ee_link, local_tool_points, label="轨迹"
):
    penetration_count = 0
    near_count = 0
    print(f"\n{'=' * 56}\n[SDF 碰撞检查] {label}，共 {len(Q)} 个路径点")
    for i, q in enumerate(Q):
        arm_pts, ee_pts = get_dense_robot_points(
            robot_id, joint_indices, fk_solver, q, ee_link, local_tool_points
        )
        dist_arm_min = float(sdf_layer.get_distances(arm_pts).min()) if len(arm_pts) > 0 else 999.0
        dist_ee_min = float(sdf_layer.get_distances(ee_pts).min()) if len(ee_pts) > 0 else 999.0

        arm_pen = dist_arm_min < SDF_PENETRATION_TOL
        arm_near = (not arm_pen) and (dist_arm_min < D_SAFE_ARM)
        ee_pen = dist_ee_min < SDF_PENETRATION_TOL
        ee_near = (not ee_pen) and (dist_ee_min < D_SAFE_EE)

        if arm_pen or ee_pen:
            status = "[穿透!!]"
            penetration_count += 1
            print(
                f"  路径点 {i:>2d}: {status}  本体最小距={dist_arm_min:+.4f}m  末端最小距={dist_ee_min:+.4f}m"
            )
        elif arm_near or ee_near:
            status = "[近距离]"
            near_count += 1
            print(
                f"  路径点 {i:>2d}: {status}  本体最小距={dist_arm_min:+.4f}m  末端最小距={dist_ee_min:+.4f}m"
            )

    print("-" * 56)
    print(
        f"[汇总] 穿透路径点: {penetration_count}  近距离路径点: {near_count}  安全路径点: {len(Q) - penetration_count - near_count}"
    )
    print(f"{'=' * 56}\n")
    return penetration_count, near_count


def replay_saved_segments(robot_id, joint_indices, segments, segment_visual_data):
    q_wait = segments[0]["q_playback"][0]
    set_joint_positions(robot_id, joint_indices, q_wait)
    p.stepSimulation()
    if not wait_for_start_signal():
        return

    replay_dt = max(1.0 / 240.0, PLAYBACK_DT)
    current_vis_handles = {"debug_ids": [], "body_ids": []}
    while p.isConnected():
        for seg_vis, seg in zip(segment_visual_data, segments):
            clear_debug_visuals(current_vis_handles)
            current_vis_handles = draw_replay_debug_visual(seg_vis)
            set_joint_positions(robot_id, joint_indices, seg["q_playback"][0])
            play_trajectory(robot_id, joint_indices, seg["q_playback"], dt=replay_dt)
        clear_debug_visuals(current_vis_handles)
        time.sleep(PLAYBACK_LOOP_PAUSE)


def main(gui=True, seed=0, play_index=0, save_path=DEFAULT_REPLAY_SAVE_PATH, replay_file=None):
    rng = np.random.default_rng(seed)
    client, robot_id, _ = setup_pybullet(gui=gui)
    joint_indices, _, lower_limits, upper_limits = get_movable_joints(robot_id)

    ee_link = -1
    for j in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, j)[12].decode("utf-8") == END_EFFECTOR_LINK_NAME:
            ee_link = j
            break
    if ee_link < 0:
        raise RuntimeError(f"未找到末端连杆: {END_EFFECTOR_LINK_NAME}")

    if replay_file is not None:
        segments, segment_visual_data, metadata = load_replay_bundle(replay_file)
        print(f"[回放] 文件内成功段数: {metadata.get('planned_total', len(segments))}")
        if gui:
            replay_saved_segments(robot_id, joint_indices, segments, segment_visual_data)
        if p.isConnected():
            p.disconnect()
        return

    transitions = load_weld_path_transitions(WELD_POSES_JSON_PATH)
    print(f"[JSON] 解析到 {len(transitions)} 条相邻 weld_path 过渡记录（非闭环）。")
    if not transitions:
        print("[规划] 没有可规划的过渡段。")
        return
    if play_index < 0 or play_index > len(transitions):
        print(f"[错误] play_index 超出范围: {play_index}，应在 0..{len(transitions)}")
        return
    if play_index > 0:
        print(f"[回放] 仅规划并回放第 {play_index} 条过渡路径。")
    else:
        print("[回放] 规划并回放全部过渡路径。")

    t0 = transitions[0]
    print("[JSON] 示例过渡:")
    print(
        f"  {t0['current_weld_path']}:{t0['current_last_edge']} end -> "
        f"{t0['next_weld_path']}:{t0['next_first_edge']} start"
    )
    print(f"  end_pose={t0['current_end_pose']}  end_xyz(raw)={t0['current_end_xyz']}")
    print(f"  start_pose={t0['next_start_pose']}  start_xyz(raw)={t0['next_start_xyz']}")

    sdf_layer = SDFCollisionLayer("workpiece_sdf2.npz")
    fk_solver = FastNumPyFK(robot_id, joint_indices)

    local_tool_points = get_tool_points(
        TORCH_STL_PATH,
        torch_to_ee_xyz=TORCH_TO_EE_XYZ,
        torch_to_ee_rpy=TORCH_TO_EE_RPY,
    )

    print("\n[规划] 开始按过渡段进行 RRT-Connect 规划（仅 current_end -> next_start）...")
    total_planning_time = 0.0
    successful_segments = []
    successful_segment_visual_data = []
    planned_total = 0

    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    try:
        for i, tr in enumerate(transitions, start=1):
            if play_index > 0 and i != play_index:
                continue

            start_pos = transform_json_xyz(tr["current_end_xyz"])
            goal_pos = transform_json_xyz(tr["next_start_xyz"])
            start_orn = np.array(tr["current_end_pose"], dtype=np.float64)
            goal_orn = np.array(tr["next_start_pose"], dtype=np.float64)

            start_pos_adjusted, q_start, start_eval, start_retreat_steps, _ = adjust_pose_out_of_collision(
                robot_id,
                joint_indices,
                lower_limits,
                upper_limits,
                fk_solver,
                sdf_layer,
                ee_link,
                local_tool_points,
                start_pos,
                start_orn,
                label=f"start-{i}",
            )
            goal_pos_adjusted, q_goal, goal_eval, goal_retreat_steps, _ = adjust_pose_out_of_collision(
                robot_id,
                joint_indices,
                lower_limits,
                upper_limits,
                fk_solver,
                sdf_layer,
                ee_link,
                local_tool_points,
                goal_pos,
                goal_orn,
                label=f"goal-{i}",
            )

            start_in_collision = (
                start_eval["arm_min"] < SDF_PENETRATION_TOL or start_eval["ee_min"] < SDF_PENETRATION_TOL
            )
            goal_in_collision = (
                goal_eval["arm_min"] < SDF_PENETRATION_TOL or goal_eval["ee_min"] < SDF_PENETRATION_TOL
            )
            if start_retreat_steps > 0:
                print(
                    f"[起点退让][{i:02d}] steps={start_retreat_steps} "
                    f"retreat_dist={start_retreat_steps * GOAL_RETREAT_STEP:.4f}m "
                    f"adjusted_start={np.round(start_pos_adjusted, 6)}"
                )
            if goal_retreat_steps > 0:
                print(
                    f"[终点退让][{i:02d}] steps={goal_retreat_steps} "
                    f"retreat_dist={goal_retreat_steps * GOAL_RETREAT_STEP:.4f}m "
                    f"adjusted_target={np.round(goal_pos_adjusted, 6)}"
                )
            if start_in_collision or goal_in_collision:
                print(
                    f"[规划][{i:02d}/{len(transitions)}] "
                    f"{tr['current_weld_path']}:{tr['current_last_edge']} -> "
                    f"{tr['next_weld_path']}:{tr['next_first_edge']} | "
                    f"跳过，端点碰撞 start=({start_eval['arm_min']:+.4f},{start_eval['ee_min']:+.4f}) "
                    f"goal=({goal_eval['arm_min']:+.4f},{goal_eval['ee_min']:+.4f})"
                )
                continue

            is_state_valid = lambda q: check_state_valid_sdf(
                robot_id, joint_indices, fk_solver, sdf_layer, ee_link, local_tool_points, q
            )
            is_edge_valid = lambda qa, qb: check_edge_valid_sdf(
                robot_id,
                joint_indices,
                fk_solver,
                sdf_layer,
                ee_link,
                local_tool_points,
                qa,
                qb,
                resolution=RRT_EDGE_CHECK_RESOLUTION,
            )

            t_rrt = time.perf_counter()
            try:
                q_path, rrt_cost = rrt_connect_plan(
                    q_start,
                    q_goal,
                    lower_limits,
                    upper_limits,
                    is_state_valid,
                    is_edge_valid,
                    step_size=RRT_STEP_SIZE,
                    max_iter=RRT_MAX_ITER,
                    goal_bias=RRT_GOAL_BIAS,
                    goal_threshold=RRT_GOAL_THRESHOLD,
                    informed_bias=RRT_INFORMED_BIAS,
                    rng=rng,
                )
            except RuntimeError as exc:
                print(
                    f"[规划][{i:02d}/{len(transitions)}] "
                    f"{tr['current_weld_path']}:{tr['current_last_edge']} -> "
                    f"{tr['next_weld_path']}:{tr['next_first_edge']} | 失败: {exc}"
                )
                continue

            seg_time = time.perf_counter() - t_rrt
            total_planning_time += seg_time
            planned_total += 1

            q_dense = densify_trajectory_for_collision_check(q_path)
            seg_summary = summarize_trajectory_collision(
                robot_id, joint_indices, fk_solver, q_dense, sdf_layer, ee_link, local_tool_points
            )
            print(
                f"[规划][{i:02d}/{len(transitions)}] "
                f"{tr['current_weld_path']}:{tr['current_last_edge']} -> "
                f"{tr['next_weld_path']}:{tr['next_first_edge']} | "
                f"耗时={seg_time:.3f}s 路径长度={rrt_cost:.4f} "
                f"穿透={seg_summary['penetration_count']} 近距={seg_summary['near_count']}"
            )

            q_playback = densify_trajectory_for_collision_check(
                q_path, resolution=PLAYBACK_PATH_RESOLUTION
            )
            ee_playback = get_ee_path_world_positions(robot_id, joint_indices, q_playback, ee_link)
            successful_segment_visual_data.append({
                "start_pos": start_pos_adjusted,
                "goal_pos": goal_pos_adjusted,
                "ee_path": ee_playback,
                "label": f"{i:02d}",
            })
            successful_segments.append({
                "q_path": q_path,
                "q_playback": q_playback,
            })
    finally:
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    print(f"[规划] 成功段数: {planned_total}/{1 if play_index > 0 else len(transitions)}")
    if not successful_segments:
        print("[规划] 无可回放轨迹，结束。")
        if p.isConnected():
            p.disconnect()
        return

    merged_segments = []
    for i, seg in enumerate(successful_segments):
        q_seg = seg["q_path"]
        merged_segments.append(q_seg if i == 0 else q_seg[1:])
    Q_all = np.vstack(merged_segments)
    Q_all_dense = densify_trajectory_for_collision_check(Q_all)
    final_summary = summarize_trajectory_collision(
        robot_id, joint_indices, fk_solver, Q_all_dense, sdf_layer, ee_link, local_tool_points
    )

    print(
        f"[RRT-Connect] 拼接轨迹: "
        f"穿透={final_summary['penetration_count']} 近距={final_summary['near_count']} "
        f"arm_min={final_summary['arm_min_global']:+.4f} ee_min={final_summary['ee_min_global']:+.4f} "
        f"累计规划时间={total_planning_time:.3f}s"
    )
    check_trajectory_collisions_sdf(
        robot_id, joint_indices, fk_solver, Q_all_dense, sdf_layer, ee_link, local_tool_points,
        label="全段拼接轨迹(连续加密检查)"
    )

    save_replay_bundle(
        save_path,
        successful_segments,
        successful_segment_visual_data,
        metadata={
            "planned_total": planned_total,
            "requested_play_index": play_index,
            "playback_dt": PLAYBACK_DT,
            "playback_path_resolution": PLAYBACK_PATH_RESOLUTION,
        },
    )

    if gui:
        replay_saved_segments(robot_id, joint_indices, successful_segments, successful_segment_visual_data)

    if p.isConnected():
        p.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", action="store_true", help="使用 DIRECT 模式，不打开 GUI。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument("--play-index", type=int, default=0, help="指定回放路径编号：0=全部，1..N=第N条过渡路径")
    parser.add_argument("--save-path", type=str, default=DEFAULT_REPLAY_SAVE_PATH, help="规划结果保存路径")
    parser.add_argument("--replay-file", type=str, default=None, help="直接读取已保存结果并回放")
    args = parser.parse_args()
    main(
        gui=not args.direct,
        seed=args.seed,
        play_index=args.play_index,
        save_path=args.save_path,
        replay_file=args.replay_file,
    )
