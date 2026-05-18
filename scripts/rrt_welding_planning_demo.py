"""RRT-Connect welding path-planning demo in Isaac Sim.

This is a single-robot/single-workpiece demo that mirrors the structure of
``data_generation/path_planning/rrt_trajopt.py``:

1. Read one weld segment from ``weld_vectors.json``.
2. Convert the start/end xyz and pose normal into world-frame TCP targets.
3. Solve six-axis UR5e IK for the start and goal.
4. Run joint-space RRT-Connect.
5. Validate states/edges with Isaac Sim/PhysX scene collision queries.
6. Replay and record the planned trajectory by default.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
DEFAULT_JOB_DIR = REPO_ROOT / "data_generation/data/generated_jobs/job_000"
DEFAULT_URDF = REPO_ROOT / "source/weldRobot/weldRobot/assets/robot-model/ur5e_with_pen.urdf"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/rrt_welding_planning_demo.mp4"
DEFAULT_INITIAL_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.57,
    "elbow_joint": 1.57,
    "wrist_1_joint": -1.57,
    "wrist_2_joint": -1.57,
    "wrist_3_joint": 0.0,
}
PLANNING_JOINTS = tuple(DEFAULT_INITIAL_JOINT_POS)


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sim_welding_arm import import_robot_from_urdf, make_resolved_urdf  # noqa: E402
from sim_parallel_welding import add_scene_lighting, ensure_xform, move_prim_to_path, step_replicator  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an Isaac Sim RRT welding path-planning demo.")
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB_DIR, help="Generated job directory.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="UR5e welding-arm URDF.")
    parser.add_argument("--weld-index", type=int, default=0, help="Weld segment index from weld_vectors.json.")
    parser.add_argument("--seed", type=int, default=7, help="RRT random seed.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument("--record", dest="record", action="store_true", default=True, help="Record replay to MP4. Enabled by default.")
    parser.add_argument("--no-record", dest="record", action="store_false", help="Replay without writing frames or MP4.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="MP4 output path.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Temporary RGB frame directory.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep RGB frames after encoding.")
    parser.add_argument(
        "--allow-frames-only",
        action="store_true",
        help="Do not fail early when ffmpeg is missing; keep PNG frames instead of MP4.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing video/frames.")
    parser.add_argument("--fps", type=int, default=30, help="Recording FPS.")
    parser.add_argument("--width", type=int, default=1280, help="Recording width.")
    parser.add_argument("--height", type=int, default=720, help="Recording height.")
    parser.add_argument("--rt-subframes", type=int, default=8, help="Replicator render subframes per captured frame.")
    parser.add_argument("--workpiece-scale", type=float, default=0.001, help="STL and weld xyz scale, mm to m.")
    parser.add_argument("--workpiece-offset", type=float, nargs=3, default=[0.45, 0.0, 0.0], help="Workpiece offset in m.")
    parser.add_argument("--workpiece-z-offset", type=float, default=0.0025, help="Extra STL and weld z offset in m.")
    parser.add_argument("--tcp-normal-offset", type=float, default=0.035, help="Retreat distance along weld normal in m.")
    parser.add_argument("--endpoint-retreat-step", type=float, default=0.01, help="Additional endpoint retreat step along weld normal in m.")
    parser.add_argument("--endpoint-retreat-max-steps", type=int, default=8, help="Maximum endpoint retreat attempts if IK state collides.")
    parser.add_argument("--rrt-step-size", type=float, default=0.35, help="RRT joint-space step size.")
    parser.add_argument("--edge-resolution", type=float, default=0.08, help="Joint-space edge collision resolution.")
    parser.add_argument("--playback-resolution", type=float, default=0.025, help="Joint-space playback interpolation resolution.")
    parser.add_argument("--max-iter", type=int, default=2500, help="Maximum RRT-Connect iterations.")
    parser.add_argument("--goal-bias", type=float, default=0.20, help="Probability of sampling q_goal.")
    parser.add_argument("--collision-padding", type=float, default=0.015, help="AABB padding for PhysX overlap queries.")
    parser.add_argument("--num-idle-frames", type=int, default=30, help="Initial/final hold frames in recordings.")
    return parser.parse_args()


def prepare_recording_paths(args: argparse.Namespace) -> Path:
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.frames_dir is None:
        args.frames_dir = args.output.parent / f"{args.output.stem}_frames"
    args.frames_dir = args.frames_dir.resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite: {args.output}")
    if args.frames_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Frames directory already exists. Use --overwrite: {args.frames_dir}")
        shutil.rmtree(args.frames_dir)
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    return args.frames_dir


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found, so MP4 encoding cannot run. "
            f"RGB frames remain in: {frames_dir}"
        )
    frame_candidates = sorted(frames_dir.glob("*.png"))
    if not frame_candidates:
        frame_candidates = sorted(frames_dir.glob("**/*.png"))
        if frame_candidates:
            log(f"[demo] Replicator wrote nested PNG frames; normalizing {len(frame_candidates)} frames for ffmpeg.")
        else:
            raise RuntimeError(f"No PNG frames were written under: {frames_dir}")

    encode_dir = frames_dir / "_encode_sequence"
    if encode_dir.exists():
        shutil.rmtree(encode_dir)
    encode_dir.mkdir(parents=True)
    for index, source in enumerate(frame_candidates):
        target = encode_dir / f"frame_{index:06d}.png"
        try:
            target.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)

    if not list(encode_dir.glob("frame_*.png")):
        raise RuntimeError(f"No PNG frames were written under: {frames_dir}")

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(encode_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg finished but MP4 was not created: {output_path}")


def parse_binary_stl(data: bytes) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    if len(data) < 84:
        raise RuntimeError("Binary STL is too small.")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if 84 + triangle_count * 50 > len(data):
        raise RuntimeError("Binary STL size does not match its triangle count.")
    points: list[tuple[float, float, float]] = []
    face_counts: list[int] = []
    face_indices: list[int] = []
    offset = 84
    for _ in range(triangle_count):
        offset += 12
        for _ in range(3):
            points.append(struct.unpack_from("<fff", data, offset))
            face_indices.append(len(points) - 1)
            offset += 12
        face_counts.append(3)
        offset += 2
    return points, face_counts, face_indices


def parse_ascii_stl(text: str) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    vertices: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if len(vertices) % 3 != 0 or not vertices:
        raise RuntimeError("ASCII STL contains no complete triangle vertices.")
    return vertices, [3] * (len(vertices) // 3), list(range(len(vertices)))


def load_stl_mesh(stl_path: Path) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    data = stl_path.read_bytes()
    triangle_count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + triangle_count * 50:
        return parse_binary_stl(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return parse_binary_stl(data)
    return parse_ascii_stl(text) if text.lstrip().lower().startswith("solid") else parse_binary_stl(data)


def import_collision_stl(
    stage: Any,
    stl_path: Path,
    prim_path: str,
    scale: float,
    z_offset: float,
    local_offset: tuple[float, float, float],
) -> dict[str, Any]:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    points, face_counts, face_indices = load_stl_mesh(stl_path)
    scaled_points = [(p[0] * scale, p[1] * scale, p[2] * scale + z_offset) for p in points]
    min_point = tuple(min(point[axis] for point in scaled_points) for axis in range(3))
    max_point = tuple(max(point[axis] for point in scaled_points) for axis in range(3))
    size = tuple(max_point[axis] - min_point[axis] for axis in range(3))

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in scaled_points])
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr([Gf.Vec3f(*min_point), Gf.Vec3f(*max_point)])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.58, 0.40)])
    xformable = UsdGeom.Xformable(mesh.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*local_offset))

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr("none")

    material_path = f"{prim_path}_Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.72, 0.58, 0.40))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

    world_min = tuple(local_offset[axis] + min_point[axis] for axis in range(3))
    world_max = tuple(local_offset[axis] + max_point[axis] for axis in range(3))
    log(f"[demo] Workpiece bounds: min={world_min}, max={world_max}, size={size}")
    return {"prim_path": prim_path, "world_min": world_min, "world_max": world_max, "size": size}


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def transform_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = rpy_matrix(rpy)
    tf[:3, 3] = xyz
    return tf


@dataclass
class ChainJoint:
    name: str
    child: str
    joint_type: str
    axis: np.ndarray
    origin: np.ndarray
    lower: float
    upper: float


class URDFKinematics:
    def __init__(self, urdf_path: Path, base_link: str = "base_link", tcp_link: str = "tool0") -> None:
        root = ET.parse(urdf_path).getroot()
        child_to_joint: dict[str, tuple[str, ET.Element]] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            parent = joint.find("parent")
            if child is not None and parent is not None:
                child_to_joint[child.attrib["link"]] = (parent.attrib["link"], joint)

        chain_xml: list[ET.Element] = []
        current = tcp_link
        while current != base_link:
            if current not in child_to_joint:
                raise RuntimeError(f"Cannot build URDF chain from {base_link} to {tcp_link}; stopped at {current}")
            parent_link, joint = child_to_joint[current]
            chain_xml.append(joint)
            current = parent_link
        chain_xml.reverse()

        self.chain: list[ChainJoint] = []
        self.active_names: list[str] = []
        for joint in chain_xml:
            origin_xml = joint.find("origin")
            xyz = np.fromstring(origin_xml.attrib.get("xyz", "0 0 0"), sep=" ") if origin_xml is not None else np.zeros(3)
            rpy = np.fromstring(origin_xml.attrib.get("rpy", "0 0 0"), sep=" ") if origin_xml is not None else np.zeros(3)
            axis_xml = joint.find("axis")
            axis = np.fromstring(axis_xml.attrib.get("xyz", "0 0 1"), sep=" ") if axis_xml is not None else np.array([0, 0, 1])
            limit_xml = joint.find("limit")
            lower = float(limit_xml.attrib.get("lower", "-6.28318530718")) if limit_xml is not None else 0.0
            upper = float(limit_xml.attrib.get("upper", "6.28318530718")) if limit_xml is not None else 0.0
            item = ChainJoint(
                name=joint.attrib["name"],
                child=joint.find("child").attrib["link"],
                joint_type=joint.attrib.get("type", "fixed"),
                axis=axis.astype(float),
                origin=transform_matrix(xyz.astype(float), rpy.astype(float)),
                lower=lower,
                upper=upper,
            )
            self.chain.append(item)
            if item.joint_type in {"revolute", "continuous"}:
                self.active_names.append(item.name)

        self.planning_names = [name for name in self.active_names if name in PLANNING_JOINTS]
        if len(self.planning_names) != 6:
            raise RuntimeError(f"Expected 6 planning joints, got {self.planning_names}")
        self.name_to_q_index = {name: idx for idx, name in enumerate(self.planning_names)}
        self.lower = np.array([next(j.lower for j in self.chain if j.name == name) for name in self.planning_names])
        self.upper = np.array([next(j.upper for j in self.chain if j.name == name) for name in self.planning_names])

    def forward(self, q: np.ndarray) -> np.ndarray:
        tf = np.eye(4)
        for joint in self.chain:
            tf = tf @ joint.origin
            if joint.name in self.name_to_q_index:
                motion = np.eye(4)
                motion[:3, :3] = axis_angle_matrix(joint.axis, float(q[self.name_to_q_index[joint.name]]))
                tf = tf @ motion
        return tf

    def solve_ik(
        self,
        target_tf: np.ndarray,
        seeds: list[np.ndarray],
        max_iters: int = 240,
        pos_weight: float = 1.0,
        rot_weight: float = 0.35,
    ) -> np.ndarray:
        best_q = None
        best_err = float("inf")
        for seed in seeds:
            q = np.clip(seed.astype(float).copy(), self.lower, self.upper)
            damping = 5e-3
            for _ in range(max_iters):
                current = self.forward(q)
                pos_err = target_tf[:3, 3] - current[:3, 3]
                rot_err = 0.5 * (
                    np.cross(current[:3, 0], target_tf[:3, 0])
                    + np.cross(current[:3, 1], target_tf[:3, 1])
                    + np.cross(current[:3, 2], target_tf[:3, 2])
                )
                err = np.hstack([pos_weight * pos_err, rot_weight * rot_err])
                err_norm = float(np.linalg.norm(err))
                if err_norm < best_err:
                    best_err = err_norm
                    best_q = q.copy()
                if np.linalg.norm(pos_err) < 0.006 and np.linalg.norm(rot_err) < 0.08:
                    return q
                jac = np.zeros((6, len(q)))
                eps = 1e-5
                for j in range(len(q)):
                    q2 = q.copy()
                    q2[j] += eps
                    tf2 = self.forward(q2)
                    dp = (tf2[:3, 3] - current[:3, 3]) / eps
                    dr = 0.5 * (
                        np.cross(current[:3, 0], tf2[:3, 0])
                        + np.cross(current[:3, 1], tf2[:3, 1])
                        + np.cross(current[:3, 2], tf2[:3, 2])
                    ) / eps
                    jac[:, j] = np.hstack([pos_weight * dp, rot_weight * dr])
                lhs = jac @ jac.T + damping * np.eye(6)
                dq = jac.T @ np.linalg.solve(lhs, err)
                q = np.clip(q + np.clip(dq, -0.25, 0.25), self.lower, self.upper)
        raise RuntimeError(f"IK failed; best weighted error={best_err:.5f}, best_q={best_q}")


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def target_frame_from_weld(point: np.ndarray, normal: np.ndarray, tangent_hint: np.ndarray, tcp_offset: float) -> np.ndarray:
    normal = normalize(normal)
    z_axis = normalize(-normal)
    tangent = tangent_hint - np.dot(tangent_hint, z_axis) * z_axis
    if np.linalg.norm(tangent) < 1e-8:
        tangent = np.array([1.0, 0.0, 0.0]) - z_axis[0] * z_axis
    x_axis = normalize(tangent)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    tf = np.eye(4)
    tf[:3, 0] = x_axis
    tf[:3, 1] = y_axis
    tf[:3, 2] = z_axis
    tf[:3, 3] = point + normal * tcp_offset
    return tf


def matrix_to_quat_xyzw(rot: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return ((rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s, 0.25 * s)
    idx = int(np.argmax(np.diag(rot)))
    if idx == 0:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        return (0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[2, 1] - rot[1, 2]) / s)
    if idx == 1:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        return ((rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s, (rot[0, 2] - rot[2, 0]) / s)
    s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
    return ((rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s, (rot[1, 0] - rot[0, 1]) / s)


def load_weld_targets(job_dir: Path, weld_index: int, scale: float, offset: list[float], z_offset: float, tcp_offset: float):
    vector_path = job_dir / "weld_vectors.json"
    with vector_path.open("r", encoding="utf-8") as f:
        welds = json.load(f)["welds"]
    if weld_index < 0 or weld_index >= len(welds):
        raise IndexError(f"--weld-index {weld_index} out of range 0..{len(welds) - 1}")
    weld = welds[weld_index]
    start_xyz = np.array(weld["start"]["xyz"], dtype=float) * scale + np.array(offset, dtype=float)
    end_xyz = np.array(weld["end"]["xyz"], dtype=float) * scale + np.array(offset, dtype=float)
    start_xyz[2] += z_offset
    end_xyz[2] += z_offset
    start_normal = np.array(weld["start"]["pose"], dtype=float)
    end_normal = np.array(weld["end"]["pose"], dtype=float)
    tangent = normalize(end_xyz - start_xyz)
    return {
        "vector_path": vector_path,
        "start_tf": target_frame_from_weld(start_xyz, start_normal, tangent, tcp_offset),
        "goal_tf": target_frame_from_weld(end_xyz, end_normal, tangent, tcp_offset),
        "start_xyz": start_xyz,
        "end_xyz": end_xyz,
        "start_normal": start_normal,
        "end_normal": end_normal,
        "tangent": tangent,
    }


def retreated_target_tf(
    base_xyz: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    tcp_offset: float,
    retreat_step: float,
    retreat_index: int,
) -> np.ndarray:
    return target_frame_from_weld(base_xyz, normal, tangent, tcp_offset + retreat_step * retreat_index)


def interpolate_edge(q_from: np.ndarray, q_to: np.ndarray, resolution: float) -> np.ndarray:
    dist = float(np.linalg.norm(q_to - q_from))
    steps = max(1, int(math.ceil(dist / resolution)))
    return np.linspace(q_from, q_to, steps + 1)


def densify_path(path: np.ndarray, resolution: float) -> np.ndarray:
    dense = [path[0]]
    for i in range(len(path) - 1):
        dense.extend(interpolate_edge(path[i], path[i + 1], resolution)[1:])
    return np.array(dense)


def nearest_node_index(nodes: list[np.ndarray], q: np.ndarray) -> int:
    arr = np.array(nodes)
    return int(np.argmin(np.linalg.norm(arr - q, axis=1)))


def extend_tree(
    nodes: list[np.ndarray],
    parents: list[int],
    q_target: np.ndarray,
    step_size: float,
    is_state_valid: Callable[[np.ndarray], bool],
    is_edge_valid: Callable[[np.ndarray, np.ndarray], bool],
) -> tuple[int | None, str]:
    nearest = nearest_node_index(nodes, q_target)
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


def connect_tree(*args: Any) -> tuple[int | None, str]:
    last_idx = None
    while True:
        new_idx, status = extend_tree(*args)
        if new_idx is None:
            return last_idx, "trapped"
        last_idx = new_idx
        if status == "reached":
            return last_idx, "reached"


def reconstruct(nodes: list[np.ndarray], parents: list[int], idx: int) -> np.ndarray:
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
    is_state_valid: Callable[[np.ndarray], bool],
    is_edge_valid: Callable[[np.ndarray, np.ndarray], bool],
    step_size: float,
    max_iter: int,
    goal_bias: float,
    rng: np.random.Generator,
) -> np.ndarray:
    start_nodes, goal_nodes = [q_start.copy()], [q_goal.copy()]
    start_parents, goal_parents = [-1], [-1]
    for iteration in range(max_iter):
        q_rand = q_goal if rng.random() < goal_bias else rng.uniform(lower, upper)
        new_idx, _ = extend_tree(start_nodes, start_parents, q_rand, step_size, is_state_valid, is_edge_valid)
        if new_idx is not None:
            connect_idx, status = connect_tree(
                goal_nodes, goal_parents, start_nodes[new_idx], step_size, is_state_valid, is_edge_valid
            )
            if status == "reached" and connect_idx is not None:
                path_a = reconstruct(start_nodes, start_parents, new_idx)
                path_b = reconstruct(goal_nodes, goal_parents, connect_idx)
                path = np.vstack([path_a, path_b[::-1][1:]])
                log(f"[RRT-Connect] Found path at iter={iteration}, waypoints={len(path)}")
                return path
        start_nodes, goal_nodes = goal_nodes, start_nodes
        start_parents, goal_parents = goal_parents, start_parents
    raise RuntimeError(f"RRT-Connect failed after {max_iter} iterations.")


class IsaacCollisionChecker:
    def __init__(
        self,
        world: Any,
        stage: Any,
        robot: Any,
        robot_prim_path: str,
        workpiece_prim_path: str,
        dof_indices: list[int],
        padding: float,
    ) -> None:
        self.world = world
        self.stage = stage
        self.robot = robot
        self.robot_prim_path = robot_prim_path
        self.workpiece_prim_path = workpiece_prim_path
        self.dof_indices = dof_indices
        self.padding = padding
        self.query = self._get_scene_query()
        self.robot_collision_prims = self._collect_robot_collision_prims()
        self.last_collision_prim_path: str | None = None
        if not self.robot_collision_prims:
            raise RuntimeError(f"No robot collision prims found under {robot_prim_path}")
        log(f"[collision] Using {len(self.robot_collision_prims)} robot collision prims with PhysX overlap queries.")

    def _get_scene_query(self) -> Any:
        from omni.physx import get_physx_scene_query_interface

        return get_physx_scene_query_interface()

    def _collect_robot_collision_prims(self) -> list[Any]:
        from pxr import Usd, UsdPhysics

        root = self.stage.GetPrimAtPath(self.robot_prim_path)
        if not root.IsValid():
            raise RuntimeError(f"Robot root prim is invalid: {self.robot_prim_path}")

        all_prims = [prim for prim in Usd.PrimRange(root)]
        prims = []
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prims.append(prim)
        if prims:
            log(f"[collision] Found {len(prims)} prims with UsdPhysics.CollisionAPI.")
            return prims

        collision_meshes = []
        for prim in all_prims:
            path_lower = str(prim.GetPath()).lower()
            type_name = prim.GetTypeName()
            if type_name in {"Mesh", "Cube", "Sphere", "Capsule"} and (
                "collision" in path_lower or "collider" in path_lower
            ):
                collision_meshes.append(prim)
        if collision_meshes:
            log(
                "[collision] No UsdPhysics.CollisionAPI prims found; "
                f"using {len(collision_meshes)} collision-named geometry prims as bbox proxies."
            )
            return collision_meshes

        visual_meshes = []
        for prim in all_prims:
            path_lower = str(prim.GetPath()).lower()
            type_name = prim.GetTypeName()
            if type_name in {"Mesh", "Cube", "Sphere", "Capsule"} and (
                "/visual" in path_lower or "/visuals" in path_lower
            ):
                visual_meshes.append(prim)
        if visual_meshes:
            log(
                "[collision] No explicit collision geometry found; "
                f"using {len(visual_meshes)} visual mesh prims as bbox proxies."
            )
            return visual_meshes

        link_names = {
            "base_link",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "ee_link",
            "pen_link",
        }
        link_prims = [prim for prim in all_prims if prim.GetName() in link_names]
        if link_prims:
            log(
                "[collision] No explicit collision geometry found; "
                f"using {len(link_prims)} robot link prims as conservative bbox proxies."
            )
            return link_prims

        preview = [
            f"{prim.GetPath()} type={prim.GetTypeName()} apis={list(prim.GetAppliedSchemas())}"
            for prim in all_prims[:80]
        ]
        raise RuntimeError(
            f"No usable robot geometry prims found under {self.robot_prim_path}. "
            "First prims:\n" + "\n".join(preview)
        )

    def set_q(self, q: np.ndarray) -> None:
        full = np.array(self.robot.get_joint_positions(), dtype=float)
        for local_idx, dof_idx in enumerate(self.dof_indices):
            full[dof_idx] = q[local_idx]
        self.robot.set_joint_positions(full)
        try:
            self.robot.set_joint_velocities(np.zeros_like(full))
        except Exception:
            pass
        self.world.step(render=False)

    def _overlap_box_hits_workpiece(self, half_extent: tuple[float, float, float], center: tuple[float, float, float]) -> bool:
        hits: list[str] = []

        def report_hit(hit: Any) -> bool:
            fields = ("rigid_body", "collider", "collision", "prim_path")
            for field in fields:
                value = getattr(hit, field, "")
                if value and self.workpiece_prim_path in str(value):
                    hits.append(str(value))
                    return False
            text = str(hit)
            if self.workpiece_prim_path in text:
                hits.append(text)
                return False
            return True

        quat = (0.0, 0.0, 0.0, 1.0)
        try:
            self.query.overlap_box(half_extent, center, quat, report_hit, False)
        except TypeError:
            try:
                self.query.overlap_box(half_extent, center, quat, report_hit)
            except TypeError:
                self.query.overlap_box(half_extent, center, report_hit)
        return bool(hits)

    def is_state_valid(self, q: np.ndarray) -> bool:
        from pxr import Usd, UsdGeom

        self.set_q(q)
        self.last_collision_prim_path = None
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True)
        for prim in self.robot_collision_prims:
            bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            min_v = bbox.GetMin()
            max_v = bbox.GetMax()
            half = tuple(max((max_v[i] - min_v[i]) * 0.5 + self.padding, 0.002) for i in range(3))
            center = tuple((max_v[i] + min_v[i]) * 0.5 for i in range(3))
            if self._overlap_box_hits_workpiece(half, center):
                self.last_collision_prim_path = str(prim.GetPath())
                return False
        return True


def make_articulation(robot_prim_path: str) -> Any:
    try:
        from isaacsim.core.prims import SingleArticulation
    except ImportError:
        from omni.isaac.core.articulations import Articulation as SingleArticulation

    robot = SingleArticulation(prim_path=robot_prim_path, name="rrt_demo_robot")
    robot.initialize()
    return robot


def dof_indices_for(robot: Any, names: list[str]) -> list[int]:
    dof_names = list(robot.dof_names)
    missing = [name for name in names if name not in dof_names]
    if missing:
        raise RuntimeError(f"Imported articulation is missing joints {missing}; available={dof_names}")
    return [dof_names.index(name) for name in names]


def set_robot_q(robot: Any, dof_indices: list[int], q: np.ndarray) -> None:
    full = np.array(robot.get_joint_positions(), dtype=float)
    for local_idx, dof_idx in enumerate(dof_indices):
        full[dof_idx] = q[local_idx]
    robot.set_joint_positions(full)


def solve_valid_endpoint(
    label: str,
    kinematics: URDFKinematics,
    checker: IsaacCollisionChecker,
    base_xyz: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    seeds: list[np.ndarray],
    tcp_offset: float,
    retreat_step: float,
    max_retreat_steps: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    last_error: Exception | None = None
    for retreat_index in range(max_retreat_steps + 1):
        target_tf = retreated_target_tf(base_xyz, normal, tangent, tcp_offset, retreat_step, retreat_index)
        try:
            q = kinematics.solve_ik(target_tf, seeds)
        except RuntimeError as exc:
            last_error = exc
            continue
        if checker.is_state_valid(q):
            if retreat_index > 0:
                log(
                    f"[endpoint] {label} retreated by {retreat_step * retreat_index:.3f} m "
                    f"along weld normal to avoid collision."
                )
            return target_tf, q, retreat_index
        last_error = RuntimeError(
            f"{label} IK state collides at retreat_index={retreat_index}, "
            f"prim={checker.last_collision_prim_path}"
        )
    raise RuntimeError(
        f"Could not find a collision-free {label} endpoint after {max_retreat_steps} retreat steps. "
        f"Last error: {last_error}"
    )


def draw_target_markers(stage: Any, start: np.ndarray, goal: np.ndarray, path_points: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    for name, point, color in (
        ("Start", start, Gf.Vec3f(0.1, 0.85, 0.2)),
        ("Goal", goal, Gf.Vec3f(1.0, 0.15, 0.1)),
    ):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/Debug/{name}")
        sphere.CreateRadiusAttr(0.008)
        sphere.CreateDisplayColorAttr([color])
        UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*point))

    curve = UsdGeom.BasisCurves.Define(stage, "/World/Debug/TcpPath")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(path_points)])
    curve.CreatePointsAttr([Gf.Vec3f(*p) for p in path_points])
    curve.CreateWidthsAttr([0.006])
    curve.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.8, 1.0)])


def add_recording_camera(rep: Any, workpiece_info: dict[str, Any], args: argparse.Namespace):
    bmin = np.array(workpiece_info["world_min"], dtype=float)
    bmax = np.array(workpiece_info["world_max"], dtype=float)
    center = (bmin + bmax) * 0.5
    target = (float(center[0]), float(center[1]), float(center[2] + 0.18))
    eye = (float(center[0] + 1.15), float(center[1] - 1.35), float(center[2] + 0.82))
    log(f"[demo] Recording camera eye={eye}, look_at={target}")
    camera = rep.create.camera(position=eye, look_at=target, focal_length=32.0, focus_distance=2.5)
    return rep.create.render_product(camera, resolution=(args.width, args.height))


def import_single_robot(stage: Any, resolved_urdf: Path) -> str:
    target_path = "/World/UR5ePen"
    imported_path = import_robot_from_urdf(resolved_urdf, target_path, fix_base=True)
    if not stage.GetPrimAtPath(imported_path).IsValid() and stage.GetPrimAtPath("/ur5e_pen").IsValid():
        log("[demo] URDF importer ignored requested prim path; normalizing /ur5e_pen -> /World/UR5ePen")
        imported_path = "/ur5e_pen"
    robot_path = move_prim_to_path(stage, imported_path, target_path)
    log(f"[demo] Imported robot prim: {robot_path}")
    return robot_path


def main() -> None:
    args = parse_args()
    frames_dir = prepare_recording_paths(args) if args.record else None
    rng = np.random.default_rng(args.seed)

    if args.record and shutil.which("ffmpeg") is None:
        log(
            "[demo] WARNING: ffmpeg is not installed. Isaac will still record PNG frames; "
            "install ffmpeg on the server to encode MP4 automatically."
        )

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless, "enable_cameras": args.record})
    try:
        try:
            from isaacsim.core.api import World
        except ImportError:
            from omni.isaac.core import World

        from omni.usd import get_context
        if args.record:
            import omni.replicator.core as rep

        world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / args.fps, stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        stage = get_context().get_stage()
        ensure_xform(stage, "/World/Debug")
        add_scene_lighting(stage)

        resolved_urdf = make_resolved_urdf(args.urdf)
        kinematics = URDFKinematics(resolved_urdf)
        robot_prim_path = import_single_robot(stage, resolved_urdf)
        workpiece_info = import_collision_stl(
            stage,
            stl_path=args.job_dir / "workpiece.stl",
            prim_path="/World/Workpiece",
            scale=args.workpiece_scale,
            z_offset=args.workpiece_z_offset,
            local_offset=tuple(float(v) for v in args.workpiece_offset),
        )

        targets = load_weld_targets(
            args.job_dir,
            args.weld_index,
            args.workpiece_scale,
            args.workpiece_offset,
            args.workpiece_z_offset,
            args.tcp_normal_offset,
        )
        log(f"[demo] Loaded weld segment {args.weld_index} from {targets['vector_path']}")
        log(f"[demo] Raw weld start/end in world: {targets['start_xyz']} -> {targets['end_xyz']}")

        log("[demo] Resetting world and initializing articulation.")
        world.reset()
        robot = make_articulation(robot_prim_path)
        dof_indices = dof_indices_for(robot, kinematics.planning_names)
        log(f"[demo] Articulation ready: dofs={list(robot.dof_names)} planning_indices={dof_indices}")
        q_home = np.array([DEFAULT_INITIAL_JOINT_POS[name] for name in kinematics.planning_names], dtype=float)
        seeds = [
            q_home,
            q_home + np.array([0.25, -0.25, 0.20, 0.0, 0.2, 0.0]),
            q_home + np.array([-0.35, 0.20, -0.25, 0.2, -0.2, 0.3]),
            np.zeros(6),
        ]
        checker = IsaacCollisionChecker(
            world=world,
            stage=stage,
            robot=robot,
            robot_prim_path=robot_prim_path,
            workpiece_prim_path="/World/Workpiece",
            dof_indices=dof_indices,
            padding=args.collision_padding,
        )
        log("[demo] Solving collision-free IK endpoints.")
        start_tf, q_start, start_retreat_steps = solve_valid_endpoint(
            label="start",
            kinematics=kinematics,
            checker=checker,
            base_xyz=targets["start_xyz"],
            normal=targets["start_normal"],
            tangent=targets["tangent"],
            seeds=seeds,
            tcp_offset=args.tcp_normal_offset,
            retreat_step=args.endpoint_retreat_step,
            max_retreat_steps=args.endpoint_retreat_max_steps,
        )
        goal_tf, q_goal, goal_retreat_steps = solve_valid_endpoint(
            label="goal",
            kinematics=kinematics,
            checker=checker,
            base_xyz=targets["end_xyz"],
            normal=targets["end_normal"],
            tangent=targets["tangent"],
            seeds=[q_start, q_home, np.zeros(6)],
            tcp_offset=args.tcp_normal_offset,
            retreat_step=args.endpoint_retreat_step,
            max_retreat_steps=args.endpoint_retreat_max_steps,
        )
        targets["start_tf"] = start_tf
        targets["goal_tf"] = goal_tf
        log(f"[IK] q_start={np.round(q_start, 4)} retreat_steps={start_retreat_steps}")
        log(f"[IK] q_goal ={np.round(q_goal, 4)} retreat_steps={goal_retreat_steps}")

        def is_edge_valid(qa: np.ndarray, qb: np.ndarray) -> bool:
            for q in interpolate_edge(qa, qb, args.edge_resolution)[1:]:
                if not checker.is_state_valid(q):
                    return False
            return True

        t0 = time.perf_counter()
        q_path = rrt_connect_plan(
            q_start=q_start,
            q_goal=q_goal,
            lower=kinematics.lower,
            upper=kinematics.upper,
            is_state_valid=checker.is_state_valid,
            is_edge_valid=is_edge_valid,
            step_size=args.rrt_step_size,
            max_iter=args.max_iter,
            goal_bias=args.goal_bias,
            rng=rng,
        )
        plan_time = time.perf_counter() - t0
        q_playback = densify_path(q_path, args.playback_resolution)
        tcp_points = np.array([kinematics.forward(q)[:3, 3] for q in q_playback])
        draw_target_markers(stage, targets["start_tf"][:3, 3], targets["goal_tf"][:3, 3], tcp_points)
        log(f"[demo] Planned {len(q_path)} RRT waypoints, {len(q_playback)} playback points in {plan_time:.2f}s")

        writer = None
        if args.record:
            try:
                rep.orchestrator.set_capture_on_play(False)
            except Exception:
                pass
            render_product = add_recording_camera(rep, workpiece_info, args)
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=str(frames_dir), rgb=True)
            writer.attach([render_product])
            log(f"[demo] Recording frames to: {frames_dir}")

        def capture_step() -> None:
            world.step(render=True)
            if args.record:
                step_replicator(rep, args)

        set_robot_q(robot, dof_indices, q_playback[0])
        for _ in range(args.num_idle_frames):
            capture_step()
        for idx, q in enumerate(q_playback):
            set_robot_q(robot, dof_indices, q)
            capture_step()
            if args.record and (idx + 1) % max(args.fps, 1) == 0:
                log(f"[demo] Recorded playback frame {idx + 1}/{len(q_playback)}")
        for _ in range(args.num_idle_frames):
            capture_step()

        if args.record and writer is not None:
            rep.orchestrator.wait_until_complete()
            writer.detach()
            png_count = len(list(frames_dir.glob("*.png"))) + len(list(frames_dir.glob("**/*.png")))
            log(f"[demo] PNG files written under {frames_dir}: {png_count}")
            if png_count == 0:
                raise RuntimeError(
                    "Replicator did not write any PNG frames. Check that the script was run with "
                    "Isaac Sim's python.sh, --headless was used on the cloud server, and cameras are enabled."
                )

        if not args.record and not args.headless:
            log("[demo] Replay complete. Close Isaac Sim or press Ctrl+C to stop.")
            while simulation_app.is_running():
                world.step(render=True)
                time.sleep(0.0)

    except Exception:
        log("[demo] Fatal error during planning or recording:")
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

    if args.record and frames_dir is not None:
        try:
            encode_video(frames_dir, args.output, args.fps)
        except RuntimeError as exc:
            log(f"[demo] {exc}")
            log(f"[demo] Keeping frames for manual encoding: {frames_dir}")
            if not args.allow_frames_only:
                raise
            return
        if not args.keep_frames:
            shutil.rmtree(frames_dir)
        log(f"[demo] Video saved to: {args.output}")


if __name__ == "__main__":
    main()
