"""RRT-Connect welding path-planning demo in Isaac Sim.

This is a single-robot/single-workpiece demo that mirrors the structure of
``data_generation/path_planning/rrt_trajopt.py``:

1. Read one weld segment from ``weld_vectors.json``.
2. Convert the start/end xyz and pose normal into world-frame TCP targets.
3. Solve six-axis UR5e IK for the start and goal.
4. Run joint-space RRT-Connect to find a collision-free seed path.
5. Run TrajOpt-style smoothing on the seed path while preserving collision validity.
6. Replay and record the optimized trajectory by default.
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
from typing import Any

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
from planning_core import TrajOptConfig, densify_path, interpolate_edge, optimize_path, rrt_connect_plan_with_restarts  # noqa: E402
from sdf_trajopt import KinematicSDFCollisionEvaluator, SDFTrajOptConfig, run_sdf_trajopt  # noqa: E402
from workpiece_sdf import SDFBuildConfig, load_or_build_workpiece_sdf  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def configure_log_filters(args: argparse.Namespace) -> None:
    if not args.suppress_physx_warnings:
        return
    try:
        import carb
        import carb.logging
        import carb.settings

        settings = carb.settings.get_settings()
        settings.set("/log/channels/omni.physx.plugin", "error")
        settings.set("/log/channels/isaacsim.core.prims.impl.articulation", "error")
        carb.logging.refresh_logging_settings()
        log("[demo] Suppressed noisy PhysX/articulation warnings in console output.")
    except Exception as exc:
        log(f"[demo] Failed to configure log filters: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an Isaac Sim RRT welding path-planning demo.")
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB_DIR, help="Generated job directory.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="UR5e welding-arm URDF.")
    parser.add_argument(
        "--weld-index",
        type=int,
        default=0,
        help="Transition start weld index: plan from weld_index end to weld_index+1 start.",
    )
    parser.add_argument(
        "--transition-xyz-tol",
        type=float,
        default=1e-6,
        help="Skip planning when consecutive weld endpoint/startpoint xyz are already coincident within this tolerance.",
    )
    parser.add_argument(
        "--auto-first-transition",
        action="store_true",
        help="Scan consecutive weld pairs and plan only the first transition whose endpoint/startpoint xyz do not coincide.",
    )
    parser.add_argument(
        "--plan-all-transitions",
        action="store_true",
        help="Plan and optimize every valid transition from weld_index onward instead of stopping at the first successful one.",
    )
    parser.add_argument("--seed", type=int, default=7, help="RRT random seed.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument("--record", dest="record", action="store_true", default=True, help="Record replay to MP4. Enabled by default.")
    parser.add_argument("--no-record", dest="record", action="store_false", help="Replay without writing frames or MP4.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="MP4 output path.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Temporary RGB frame directory.")
    parser.add_argument("--failure-screenshot-dir", type=Path, default=None, help="Directory for a planning-failure PNG screenshot.")
    parser.add_argument("--ik-screenshot-dir", type=Path, default=None, help="Directory for pre-RRT start/goal IK PNG screenshots.")
    parser.add_argument(
        "--save-ik-screenshots",
        dest="save_ik_screenshots",
        action="store_true",
        default=False,
        help="Save start/goal IK static screenshots before RRT.",
    )
    parser.add_argument(
        "--no-ik-screenshots",
        dest="save_ik_screenshots",
        action="store_false",
        help="Do not save pre-RRT start/goal IK screenshots. Disabled by default.",
    )
    parser.add_argument("--encode-only", action="store_true", help="Only encode an existing frames directory to MP4.")
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
    parser.add_argument("--visual-ground-size", type=float, default=2.0, help="Size of the collidable visual ground plane.")
    parser.add_argument("--visual-ground-z", type=float, default=-0.002, help="Z height of the collidable visual ground plane.")
    parser.add_argument("--visual-ground-opacity", type=float, default=0.18, help="Visual opacity for the collidable ground plane.")
    parser.add_argument("--workpiece-scale", type=float, default=0.001, help="STL and weld xyz scale, mm to m.")
    parser.add_argument("--workpiece-offset", type=float, nargs=3, default=[0.5, 0.0, 0.0], help="Workpiece offset in m.")
    parser.add_argument("--workpiece-z-offset", type=float, default=0.0025, help="Extra STL and weld z offset in m.")
    parser.add_argument("--workpiece-opacity", type=float, default=1.0, help="Visual opacity for the workpiece mesh.")
    parser.add_argument("--tcp-normal-offset", type=float, default=0.035, help="Retreat distance along weld normal in m.")
    parser.add_argument("--endpoint-retreat-step", type=float, default=0.01, help="Additional endpoint retreat step along weld normal in m.")
    parser.add_argument("--endpoint-retreat-max-steps", type=int, default=4, help="Maximum endpoint retreat attempts if IK state collides.")
    parser.add_argument("--endpoint-random-seeds", type=int, default=16, help="Random IK seeds per endpoint retreat attempt.")
    parser.add_argument("--endpoint-ik-max-iters", type=int, default=160, help="Maximum IK iterations per seed during endpoint solving.")
    parser.add_argument("--endpoint-yaw-samples", type=int, default=12, help="TCP rotations around weld normal per endpoint retreat attempt.")
    parser.add_argument("--endpoint-ik-rot-weight", type=float, default=0.12, help="IK orientation weight for endpoint solving.")
    parser.add_argument(
        "--planning-anchor-max-extra-steps",
        type=int,
        default=6,
        help="Maximum additional retreat steps used to find free-space RRT anchor poses beyond the accepted weld endpoints.",
    )
    parser.add_argument(
        "--planning-anchor-escape-step",
        type=float,
        default=0.04,
        help="Joint-space probe step used to verify that an RRT anchor can expand away from its root.",
    )
    parser.add_argument("--rrt-step-size", type=float, default=0.08, help="RRT joint-space step size.")
    parser.add_argument("--edge-resolution", type=float, default=0.08, help="Joint-space edge collision resolution.")
    parser.add_argument("--playback-resolution", type=float, default=0.025, help="Joint-space playback interpolation resolution.")
    parser.add_argument("--max-iter", type=int, default=50000, help="Maximum RRT-Connect iterations per restart.")
    parser.add_argument("--rrt-restarts", type=int, default=4, help="Number of RRT-Connect restarts.")
    parser.add_argument("--goal-bias", type=float, default=0.45, help="Probability of sampling q_goal.")
    parser.add_argument("--trajopt", dest="trajopt", action="store_true", default=True, help="Run TrajOpt-style smoothing after RRT. Enabled by default.")
    parser.add_argument("--no-trajopt", dest="trajopt", action="store_false", help="Skip TrajOpt smoothing and replay the raw RRT path.")
    parser.add_argument("--trajopt-waypoints", type=int, default=12, help="Number of waypoints used by TrajOpt resampling.")
    parser.add_argument("--trajopt-max-waypoints", type=int, default=16, help="Upper bound on SDF TrajOpt resampling waypoints; 0 disables the cap.")
    parser.add_argument("--trajopt-maxiter", type=int, default=800, help="Maximum SLSQP iterations for TrajOpt.")
    parser.add_argument("--trajopt-smoothness-weight", type=float, default=5.0, help="Smoothness weight for TrajOpt.")
    parser.add_argument("--trajopt-path-length-weight", type=float, default=1.0, help="Path-length weight for TrajOpt.")
    parser.add_argument("--trajopt-seed-weight", type=float, default=0.05, help="Seed-adherence weight for TrajOpt.")
    parser.add_argument("--sdf-trajopt", dest="sdf_trajopt", action="store_true", default=True, help="Use cached workpiece SDF for trajectory optimization. Enabled by default.")
    parser.add_argument("--no-sdf-trajopt", dest="sdf_trajopt", action="store_false", help="Use the legacy non-SDF TrajOpt fallback.")
    parser.add_argument("--workpiece-sdf-path", type=Path, default=None, help="Optional path to the cached workpiece SDF .npz. Defaults to <job-dir>/workpiece_sdf.npz.")
    parser.add_argument("--rebuild-workpiece-sdf", action="store_true", help="Force rebuilding the cached workpiece SDF for the current job.")
    parser.add_argument("--workpiece-sdf-pitch", type=float, default=0.004, help="Voxel pitch in meters for cached workpiece SDF generation.")
    parser.add_argument("--workpiece-sdf-margin", type=float, default=0.03, help="Extra world-space margin in meters around the workpiece SDF grid.")
    parser.add_argument("--workpiece-sdf-voxelize-method", type=str, default="subdivide", choices=["subdivide", "ray"], help="Trimesh voxelization method for cached workpiece SDF generation.")
    parser.add_argument("--workpiece-sdf-voxelize-max-iter", type=int, default=64, help="Maximum subdivision iterations used by trimesh voxelization for cached workpiece SDF generation.")
    parser.add_argument("--sdf-collision-weight", type=float, default=2.0e8, help="Collision penalty weight for SDF TrajOpt.")
    parser.add_argument("--sdf-arm-safe-distance", type=float, default=0.01, help="Safe distance in meters for arm sample points in SDF TrajOpt.")
    parser.add_argument("--sdf-tool-safe-distance", type=float, default=0.000, help="Safe distance in meters for tool sample points in SDF TrajOpt.")
    parser.add_argument("--sdf-penetration-tol", type=float, default=-0.002, help="Allowed signed-distance penetration tolerance in meters for SDF TrajOpt.")
    parser.add_argument("--sdf-arm-step-size", type=float, default=0.02, help="Sampling resolution in meters along robot links for SDF evaluation.")
    parser.add_argument("--sdf-tool-step-size", type=float, default=0.01, help="Sampling resolution in meters along the tool segment for SDF evaluation.")
    parser.add_argument("--sdf-constraint-point-stride", type=int, default=12, help="Stride for sampled points used in SDF non-penetration constraints.")
    parser.add_argument(
        "--sdf-endpoint-relax-waypoints",
        type=int,
        default=2,
        help="Number of waypoints near each endpoint that use relaxed SDF safe-distance penalties during TrajOpt.",
    )
    parser.add_argument(
        "--sdf-endpoint-safe-distance-scale",
        type=float,
        default=0.0,
        help="Safe-distance penalty scale at the exact start/end waypoint for SDF TrajOpt; 0 means non-penetration only.",
    )
    parser.add_argument("--shortcut-iterations", type=int, default=600, help="Shortcut smoothing attempts per pass.")
    parser.add_argument("--shortcut-passes", type=int, default=6, help="Number of shortcut smoothing passes.")
    parser.add_argument("--average-passes", type=int, default=8, help="Number of local averaging smoothing passes.")
    parser.add_argument("--average-blend", type=float, default=0.4, help="Blend factor for local averaging smoothing.")
    parser.add_argument("--collision-padding", type=float, default=0.015, help="AABB padding for PhysX overlap queries.")
    parser.add_argument(
        "--robot-collision-approximation",
        type=str,
        default="sdf",
        choices=["sdf", "convexDecomposition", "convexHull"],
        help="Robot collision approximation for URDF collision meshes.",
    )
    parser.add_argument(
        "--robot-sdf-resolution",
        type=int,
        default=256,
        help="SDF resolution for robot mesh colliders when --robot-collision-approximation=sdf.",
    )
    parser.add_argument(
        "--robot-sdf-subgrid-resolution",
        type=int,
        default=6,
        help="Sparse SDF subgrid resolution for robot mesh colliders; 0 uses dense SDF.",
    )
    parser.add_argument("--include-tool-collision", dest="include_tool_collision", action="store_true", default=True, help="Use ee/pen collision geometry. Enabled by default.")
    parser.add_argument("--no-tool-collision", dest="include_tool_collision", action="store_false", help="Do not use ee/pen collision geometry.")
    parser.add_argument("--contact-settle-steps", type=int, default=1, help="Physics steps after setting q before reading contact reports.")
    parser.add_argument("--use-bbox-collision", action="store_true", help="Use legacy bbox overlap checks instead of PhysX contact reports.")
    parser.add_argument(
        "--show-collision-proxies",
        action="store_true",
        help="Display generated URDF collision STL proxies in the render.",
    )
    parser.add_argument("--num-idle-frames", type=int, default=30, help="Initial/final hold frames in recordings.")
    parser.add_argument(
        "--suppress-physx-warnings",
        dest="suppress_physx_warnings",
        action="store_true",
        default=True,
        help="Suppress noisy omni.physx.plugin warnings so demo logs stay readable. Enabled by default.",
    )
    parser.add_argument(
        "--show-physx-warnings",
        dest="suppress_physx_warnings",
        action="store_false",
        help="Do not suppress omni.physx.plugin warnings.",
    )
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
    frames_dir = frames_dir.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[demo] ffmpeg path: {ffmpeg}")

    frame_candidates = sorted(
        path for path in frames_dir.glob("*.png") if "_encode_sequence" not in path.parts
    )
    if not frame_candidates:
        frame_candidates = sorted(
            path for path in frames_dir.glob("**/*.png") if "_encode_sequence" not in path.parts
        )
        if frame_candidates:
            log(f"[demo] Replicator wrote nested PNG frames; normalizing {len(frame_candidates)} frames for ffmpeg.")
        else:
            raise RuntimeError(f"No PNG frames were written under: {frames_dir}")
    log(f"[demo] Encoding {len(frame_candidates)} PNG frames. First frame: {frame_candidates[0]}")

    encode_dir = frames_dir / "_encode_sequence"
    if encode_dir.exists():
        shutil.rmtree(encode_dir)
    encode_dir.mkdir(parents=True)
    for index, source in enumerate(frame_candidates):
        target = encode_dir / f"frame_{index:06d}.png"
        try:
            target.hardlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)

    if not list(encode_dir.glob("frame_*.png")):
        raise RuntimeError(f"No PNG frames were written under: {frames_dir}")

    commands = [
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
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(encode_dir / "frame_%06d.png"),
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            str(output_path),
        ],
    ]
    errors = []
    for command in commands:
        log("[demo] Running encoder: " + " ".join(command))
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode == 0:
            break
        errors.append(result.stdout[-4000:])
        log(f"[demo] Encoder failed with code {result.returncode}. Trying fallback codec if available.")
    else:
        raise RuntimeError("ffmpeg failed to encode MP4. Last ffmpeg output:\n" + "\n".join(errors[-1:]))

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg finished but MP4 was not created: {output_path}")
    log(f"[demo] Encoded MP4 size: {output_path.stat().st_size} bytes")


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


def parse_urdf_vec(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=float)
    vec = np.fromstring(value, sep=" ", dtype=float)
    if len(vec) != 3:
        return np.array(default, dtype=float)
    return vec


def safe_prim_name(name: str) -> str:
    cleaned = []
    for ch in name:
        cleaned.append(ch if ch.isalnum() or ch == "_" else "_")
    value = "".join(cleaned).strip("_")
    return value or "collision"


def configure_non_reflective_preview_surface(shader: Any, color: Any, roughness: float = 1.0, opacity: float | None = None) -> None:
    from pxr import Sdf

    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
    if opacity is not None:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))


def box_mesh(size: np.ndarray) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    sx, sy, sz = size * 0.5
    points = [
        (-sx, -sy, -sz),
        (sx, -sy, -sz),
        (sx, sy, -sz),
        (-sx, sy, -sz),
        (-sx, -sy, sz),
        (sx, -sy, sz),
        (sx, sy, sz),
        (-sx, sy, sz),
    ]
    face_counts = [4] * 6
    face_indices = [
        0, 1, 2, 3,
        4, 7, 6, 5,
        0, 4, 5, 1,
        1, 5, 6, 2,
        2, 6, 7, 3,
        3, 7, 4, 0,
    ]
    return points, face_counts, face_indices


def downsample_points(points: np.ndarray, spacing: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return np.empty((0, 3), dtype=float)
    spacing = max(float(spacing), 1e-6)
    buckets = np.rint(pts / spacing).astype(np.int64)
    _, unique_idx = np.unique(buckets, axis=0, return_index=True)
    unique_idx.sort()
    return pts[unique_idx]


def sample_box_surface_points(size: np.ndarray, spacing: float) -> np.ndarray:
    sx, sy, sz = np.asarray(size, dtype=float) * 0.5
    spacing = max(float(spacing), 1e-6)
    xs = np.arange(-sx, sx + spacing * 0.5, spacing, dtype=float)
    ys = np.arange(-sy, sy + spacing * 0.5, spacing, dtype=float)
    zs = np.arange(-sz, sz + spacing * 0.5, spacing, dtype=float)
    faces = []
    for z in (-sz, sz):
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        faces.append(np.column_stack([xx.reshape(-1), yy.reshape(-1), np.full(xx.size, z, dtype=float)]))
    for y in (-sy, sy):
        xx, zz = np.meshgrid(xs, zs, indexing="xy")
        faces.append(np.column_stack([xx.reshape(-1), np.full(xx.size, y, dtype=float), zz.reshape(-1)]))
    for x in (-sx, sx):
        yy, zz = np.meshgrid(ys, zs, indexing="xy")
        faces.append(np.column_stack([np.full(yy.size, x, dtype=float), yy.reshape(-1), zz.reshape(-1)]))
    return downsample_points(np.vstack(faces), spacing)


def find_descendant_by_name(stage: Any, root_path: str, name: str):
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return prim
    return None


def add_urdf_collision_stl_proxies(
    stage: Any,
    robot_prim_path: str,
    resolved_urdf: Path,
    include_tool_collision: bool,
    show_collision_proxies: bool,
    collision_approximation: str,
    sdf_resolution: int,
    sdf_subgrid_resolution: int,
) -> list[str]:
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    # The fixed base is mounted at the ground plane; excluding it avoids a
    # permanent base-ground contact from invalidating every sampled state.
    excluded_links = {"world", "base", "base_link"}
    if not include_tool_collision:
        excluded_links.update({"ee_link", "pen_link", "tool0"})

    tree = ET.parse(resolved_urdf)
    root = tree.getroot()
    proxy_paths: list[str] = []

    material_path = "/World/Debug/CollisionProxyMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    configure_non_reflective_preview_surface(shader, Gf.Vec3f(0.0, 0.55, 1.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        if link_name in excluded_links:
            continue
        link_prim = find_descendant_by_name(stage, robot_prim_path, link_name)
        if link_prim is None or not link_prim.IsValid():
            log(f"[collision] WARNING: cannot find imported link prim for URDF link {link_name}; skipping collision proxy.")
            continue

        proxy_root_path = f"{link_prim.GetPath()}/CollisionProxy"
        ensure_xform(stage, proxy_root_path)
        for index, collision in enumerate(link.findall("collision")):
            geometry = collision.find("geometry")
            if geometry is None:
                continue

            origin_xml = collision.find("origin")
            origin_xyz = parse_urdf_vec(origin_xml.attrib.get("xyz") if origin_xml is not None else None, (0.0, 0.0, 0.0))
            origin_rpy = parse_urdf_vec(origin_xml.attrib.get("rpy") if origin_xml is not None else None, (0.0, 0.0, 0.0))
            origin_rot = rpy_matrix(origin_rpy)

            mesh_xml = geometry.find("mesh")
            box_xml = geometry.find("box")
            source_name = f"{link_name}_{index}"
            if mesh_xml is not None:
                mesh_path = Path(mesh_xml.attrib["filename"]).expanduser()
                if not mesh_path.is_absolute():
                    mesh_path = (resolved_urdf.parent / mesh_path).resolve()
                mesh_scale = parse_urdf_vec(mesh_xml.attrib.get("scale"), (1.0, 1.0, 1.0))
                points, face_counts, face_indices = load_stl_mesh(mesh_path)
                local_points = []
                for point in points:
                    p = np.asarray(point, dtype=float) * mesh_scale
                    p = origin_rot @ p + origin_xyz
                    local_points.append(tuple(float(v) for v in p))
                source_name = mesh_path.stem
            elif box_xml is not None:
                size = parse_urdf_vec(box_xml.attrib.get("size"), (0.01, 0.01, 0.01))
                points, face_counts, face_indices = box_mesh(size)
                local_points = []
                for point in points:
                    p = origin_rot @ np.asarray(point, dtype=float) + origin_xyz
                    local_points.append(tuple(float(v) for v in p))
            else:
                continue

            proxy_path = f"{proxy_root_path}/{safe_prim_name(source_name)}_{index:02d}"
            mesh = UsdGeom.Mesh.Define(stage, proxy_path)
            mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in local_points])
            mesh.CreateFaceVertexCountsAttr(face_counts)
            mesh.CreateFaceVertexIndicesAttr(face_indices)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.55, 1.0)])
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            mesh_collision.CreateApproximationAttr(collision_approximation)
            if collision_approximation == "sdf":
                sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim())
                sdf_api.CreateSdfResolutionAttr(int(max(16, sdf_resolution)))
                if sdf_subgrid_resolution > 0:
                    sdf_api.CreateSdfSubgridResolutionAttr(int(sdf_subgrid_resolution))
            enable_contact_report(mesh.GetPrim())
            if not show_collision_proxies:
                mesh.CreatePurposeAttr("proxy")
            proxy_paths.append(proxy_path)

    log(
        f"[collision] Added {len(proxy_paths)} URDF collision STL/box proxies "
        f"(include_tool_collision={include_tool_collision}, visible={show_collision_proxies}, "
        f"approximation={collision_approximation})."
    )
    if proxy_paths:
        log("[collision] First collision proxies: " + ", ".join(proxy_paths[:5]))
    return proxy_paths


def import_collision_stl(
    stage: Any,
    stl_path: Path,
    prim_path: str,
    scale: float,
    z_offset: float,
    local_offset: tuple[float, float, float],
    opacity: float,
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
    mesh.CreateDisplayOpacityAttr([float(opacity)])
    xformable = UsdGeom.Xformable(mesh.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*local_offset))

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr("none")
    enable_contact_report(mesh.GetPrim())

    material_path = f"{prim_path}_Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    configure_non_reflective_preview_surface(shader, Gf.Vec3f(0.72, 0.58, 0.40), opacity=float(opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

    world_min = tuple(local_offset[axis] + min_point[axis] for axis in range(3))
    world_max = tuple(local_offset[axis] + max_point[axis] for axis in range(3))
    log(
        "[demo] Workpiece bounds: "
        f"min={world_min}, max={world_max}, size={size}, collision_approximation=none"
    )
    return {"prim_path": prim_path, "world_min": world_min, "world_max": world_max, "size": size}


def enable_contact_report(prim: Any, threshold: float = 0.0) -> None:
    try:
        from pxr import PhysxSchema

        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        attr = api.GetThresholdAttr()
        if attr:
            attr.Set(float(threshold))
        else:
            api.CreateThresholdAttr(float(threshold))
    except Exception:
        try:
            from pxr import PhysxSchema

            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            attr = getattr(api, "CreatePhysxContactReportThresholdAttr", None)
            if attr is not None:
                attr(float(threshold))
        except Exception:
            pass


def add_visual_ground(stage: Any, size: float, z: float, opacity: float) -> str:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    half = float(size) * 0.5
    prim_path = "/World/VisualGround"
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    points = [
        Gf.Vec3f(-half, -half, z),
        Gf.Vec3f(half, -half, z),
        Gf.Vec3f(half, half, z),
        Gf.Vec3f(-half, half, z),
    ]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.18, 0.18)])
    mesh.CreateDisplayOpacityAttr([float(opacity)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr("none")
    enable_contact_report(mesh.GetPrim())

    material_path = f"{prim_path}_Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    configure_non_reflective_preview_surface(shader, Gf.Vec3f(0.18, 0.18, 0.18), opacity=float(opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    log(f"[demo] Added collidable transparent ground: {prim_path}, size={size}, z={z}, opacity={opacity}")
    return prim_path


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
    def __init__(
        self,
        urdf_path: Path,
        base_link: str = "base_link",
        tcp_link: str = "tool0",
        include_tool_collision: bool = True,
    ) -> None:
        root = ET.parse(urdf_path).getroot()
        self.urdf_path = urdf_path
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
        self.base_link = base_link
        self.tcp_link = tcp_link
        self.include_tool_collision = include_tool_collision
        self.tool_collision_links = {"ee_link", "pen_link", "tool0"}
        self.link_collision_geometry = self._parse_link_collision_geometry(root)
        self._collision_points_cache: dict[tuple[str, str, float], np.ndarray] = {}

    def _parse_link_collision_geometry(self, root: ET.Element) -> dict[str, list[dict[str, Any]]]:
        collision_by_link: dict[str, list[dict[str, Any]]] = {}
        excluded_links = {"world", "base", "base_link"}
        if not self.include_tool_collision:
            excluded_links.update(self.tool_collision_links)
        for link in root.findall("link"):
            link_name = link.attrib.get("name", "")
            if link_name in excluded_links:
                continue
            geoms: list[dict[str, Any]] = []
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is None:
                    continue
                origin_xml = collision.find("origin")
                origin_xyz = parse_urdf_vec(origin_xml.attrib.get("xyz") if origin_xml is not None else None, (0.0, 0.0, 0.0))
                origin_rpy = parse_urdf_vec(origin_xml.attrib.get("rpy") if origin_xml is not None else None, (0.0, 0.0, 0.0))
                origin_rot = rpy_matrix(origin_rpy)
                mesh_xml = geometry.find("mesh")
                box_xml = geometry.find("box")
                if mesh_xml is not None:
                    mesh_path = Path(mesh_xml.attrib["filename"]).expanduser()
                    if not mesh_path.is_absolute():
                        mesh_path = (self.urdf_path.parent / mesh_path).resolve()
                    mesh_scale = parse_urdf_vec(mesh_xml.attrib.get("scale"), (1.0, 1.0, 1.0))
                    points, _, _ = load_stl_mesh(mesh_path)
                    raw_points = []
                    for point in points:
                        p = np.asarray(point, dtype=float) * mesh_scale
                        p = origin_rot @ p + origin_xyz
                        raw_points.append(p)
                    if raw_points:
                        geoms.append({"type": "mesh", "points": np.asarray(raw_points, dtype=float)})
                elif box_xml is not None:
                    size = parse_urdf_vec(box_xml.attrib.get("size"), (0.01, 0.01, 0.01))
                    geoms.append({"type": "box", "size": np.asarray(size, dtype=float), "origin_xyz": origin_xyz, "origin_rot": origin_rot})
            if geoms:
                collision_by_link[link_name] = geoms
        return collision_by_link

    def _sample_link_collision_geometry(self, link_name: str, spacing: float, tool_spacing: float) -> np.ndarray:
        cache_key = (
            link_name,
            "tool" if link_name in self.tool_collision_links else "arm",
            round(tool_spacing if link_name in self.tool_collision_links else spacing, 5),
        )
        cached = self._collision_points_cache.get(cache_key)
        if cached is not None:
            return cached
        geoms = self.link_collision_geometry.get(link_name, [])
        if not geoms:
            result = np.empty((0, 3), dtype=float)
            self._collision_points_cache[cache_key] = result
            return result
        local_spacing = tool_spacing if link_name in self.tool_collision_links else spacing
        sampled = []
        for geom in geoms:
            if geom["type"] == "mesh":
                sampled.append(downsample_points(geom["points"], local_spacing))
            elif geom["type"] == "box":
                points = sample_box_surface_points(geom["size"], local_spacing)
                transformed = (geom["origin_rot"] @ points.T).T + geom["origin_xyz"][None, :]
                sampled.append(transformed)
        result = downsample_points(np.vstack(sampled), local_spacing) if sampled else np.empty((0, 3), dtype=float)
        self._collision_points_cache[cache_key] = result
        return result

    def forward(self, q: np.ndarray) -> np.ndarray:
        tf = np.eye(4)
        for joint in self.chain:
            tf = tf @ joint.origin
            if joint.name in self.name_to_q_index:
                motion = np.eye(4)
                motion[:3, :3] = axis_angle_matrix(joint.axis, float(q[self.name_to_q_index[joint.name]]))
                tf = tf @ motion
        return tf

    def compute_link_transforms(self, q: np.ndarray) -> dict[str, np.ndarray]:
        tf = np.eye(4)
        transforms: dict[str, np.ndarray] = {self.base_link: tf.copy()}
        for joint in self.chain:
            tf = tf @ joint.origin
            if joint.name in self.name_to_q_index:
                motion = np.eye(4)
                motion[:3, :3] = axis_angle_matrix(joint.axis, float(q[self.name_to_q_index[joint.name]]))
                tf = tf @ motion
            transforms[joint.child] = tf.copy()
        return transforms

    def sample_collision_points(
        self,
        q: np.ndarray,
        arm_step_size: float = 0.02,
        tool_step_size: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        transforms = self.compute_link_transforms(q)
        if self.link_collision_geometry:
            arm_points = []
            tool_points = []
            for link_name, link_tf in transforms.items():
                local_points = self._sample_link_collision_geometry(link_name, arm_step_size, tool_step_size)
                if len(local_points) == 0:
                    continue
                world_points = (link_tf[:3, :3] @ local_points.T).T + link_tf[:3, 3][None, :]
                if link_name in self.tool_collision_links:
                    tool_points.append(world_points)
                else:
                    arm_points.append(world_points)
            arm_arr = np.vstack(arm_points) if arm_points else np.empty((0, 3), dtype=float)
            tool_arr = np.vstack(tool_points) if tool_points else np.empty((0, 3), dtype=float)
            if len(arm_arr) > 0 or len(tool_arr) > 0:
                return arm_arr, tool_arr

        chain_points = [transforms[self.base_link][:3, 3]]
        for joint in self.chain:
            chain_points.append(transforms[joint.child][:3, 3])

        arm_segments = []
        for p0, p1 in zip(chain_points[:-1], chain_points[1:]):
            segment_length = float(np.linalg.norm(p1 - p0))
            num_steps = max(1, int(np.ceil(segment_length / max(arm_step_size, 1e-6))))
            alphas = np.linspace(0.0, 1.0, num_steps + 1)[:, None]
            arm_segments.append((1.0 - alphas) * p0[None, :] + alphas * p1[None, :])
        arm_points = np.vstack(arm_segments) if arm_segments else np.empty((0, 3), dtype=float)

        tool_points = np.empty((0, 3), dtype=float)
        if "pen_link" in transforms and self.tcp_link in transforms:
            p0 = transforms["pen_link"][:3, 3]
            p1 = transforms[self.tcp_link][:3, 3]
            segment_length = float(np.linalg.norm(p1 - p0))
            num_steps = max(1, int(np.ceil(segment_length / max(tool_step_size, 1e-6))))
            alphas = np.linspace(0.0, 1.0, num_steps + 1)[:, None]
            tool_points = (1.0 - alphas) * p0[None, :] + alphas * p1[None, :]
        elif self.tcp_link in transforms:
            tool_points = transforms[self.tcp_link][:3, 3][None, :]
        return arm_points, tool_points

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


def fallback_pose_normal(tangent: np.ndarray) -> np.ndarray:
    tangent = normalize(np.asarray(tangent, dtype=float).reshape(3))
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(tangent, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    normal = np.cross(tangent, up)
    if np.linalg.norm(normal) < 1e-8:
        normal = np.array([1.0, 0.0, 0.0], dtype=float)
    return normalize(normal)


def resolve_pose_normal(
    pose_value: Any,
    label: str,
    tangent: np.ndarray,
    fallback_normal: np.ndarray | None = None,
) -> np.ndarray:
    if pose_value is not None:
        arr = np.asarray(pose_value, dtype=float).reshape(-1)
        if arr.size == 3 and np.linalg.norm(arr) > 1e-8:
            return normalize(arr)
        log(f"[demo] Invalid {label} pose_normal={pose_value!r}; using fallback.")
    else:
        log(f"[demo] Missing {label} pose_normal; using fallback.")
    if fallback_normal is not None and np.linalg.norm(fallback_normal) > 1e-8:
        return normalize(fallback_normal)
    return fallback_pose_normal(tangent)


def target_frame_from_weld(
    point: np.ndarray,
    normal: np.ndarray,
    tangent_hint: np.ndarray,
    tcp_offset: float,
    yaw_about_normal: float = 0.0,
) -> np.ndarray:
    normal = normalize(normal)
    z_axis = normalize(-normal)
    tangent = tangent_hint - np.dot(tangent_hint, z_axis) * z_axis
    if np.linalg.norm(tangent) < 1e-8:
        tangent = np.array([1.0, 0.0, 0.0]) - z_axis[0] * z_axis
    x_axis = normalize(tangent)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    if abs(yaw_about_normal) > 1e-12:
        c = math.cos(yaw_about_normal)
        s = math.sin(yaw_about_normal)
        x_rot = c * x_axis + s * y_axis
        y_rot = -s * x_axis + c * y_axis
        x_axis = normalize(x_rot)
        y_axis = normalize(y_rot)
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


def _world_xyz(xyz: list[float], scale: float, offset: list[float], z_offset: float) -> np.ndarray:
    point = np.array(xyz, dtype=float) * scale + np.array(offset, dtype=float)
    point[2] += z_offset
    return point


def load_weld_targets(
    job_dir: Path,
    weld_index: int,
    scale: float,
    offset: list[float],
    z_offset: float,
    tcp_offset: float,
    transition_xyz_tol: float,
):
    vector_path = job_dir / "weld_vectors.json"
    with vector_path.open("r", encoding="utf-8") as f:
        welds = json.load(f)["welds"]
    if weld_index < 0 or weld_index + 1 >= len(welds):
        raise IndexError(f"--weld-index {weld_index} out of range 0..{len(welds) - 2} for consecutive-weld transition planning")

    prev_weld = welds[weld_index]
    next_weld = welds[weld_index + 1]

    prev_start_xyz = _world_xyz(prev_weld["start"]["xyz"], scale, offset, z_offset)
    prev_end_xyz = _world_xyz(prev_weld["end"]["xyz"], scale, offset, z_offset)
    next_start_xyz = _world_xyz(next_weld["start"]["xyz"], scale, offset, z_offset)
    next_end_xyz = _world_xyz(next_weld["end"]["xyz"], scale, offset, z_offset)

    prev_tangent = normalize(prev_end_xyz - prev_start_xyz)
    next_tangent = normalize(next_end_xyz - next_start_xyz)

    start_xyz = prev_end_xyz
    goal_xyz = next_start_xyz
    transition_distance = float(np.linalg.norm(goal_xyz - start_xyz))
    if transition_distance <= transition_xyz_tol:
        return {
            "vector_path": vector_path,
            "skip_planning": True,
            "transition_distance": transition_distance,
            "start_xyz": start_xyz,
            "end_xyz": goal_xyz,
            "prev_weld_index": weld_index,
            "next_weld_index": weld_index + 1,
        }

    start_normal = resolve_pose_normal(prev_weld["end"].get("pose"), "previous weld end", prev_tangent)
    goal_normal = resolve_pose_normal(next_weld["start"].get("pose"), "next weld start", next_tangent)
    return {
        "vector_path": vector_path,
        "skip_planning": False,
        "transition_distance": transition_distance,
        "prev_weld_index": weld_index,
        "next_weld_index": weld_index + 1,
        "start_tf": target_frame_from_weld(start_xyz, start_normal, prev_tangent, tcp_offset),
        "goal_tf": target_frame_from_weld(goal_xyz, goal_normal, next_tangent, tcp_offset),
        "start_xyz": start_xyz,
        "end_xyz": goal_xyz,
        "start_normal": start_normal,
        "end_normal": goal_normal,
        "tangent": normalize(goal_xyz - start_xyz),
    }


def iter_transition_targets(
    job_dir: Path,
    weld_index: int,
    scale: float,
    offset: list[float],
    z_offset: float,
    tcp_offset: float,
    transition_xyz_tol: float,
    auto_first_transition: bool,
) -> dict[str, Any]:
    if not auto_first_transition:
        yield load_weld_targets(
            job_dir,
            weld_index,
            scale,
            offset,
            z_offset,
            tcp_offset,
            transition_xyz_tol,
        )
        return

    vector_path = job_dir / "weld_vectors.json"
    with vector_path.open("r", encoding="utf-8") as f:
        weld_count = len(json.load(f)["welds"])

    for candidate_index in range(max(0, weld_index), max(0, weld_count - 1)):
        targets = load_weld_targets(
            job_dir,
            candidate_index,
            scale,
            offset,
            z_offset,
            tcp_offset,
            transition_xyz_tol,
        )
        yield targets


def retreated_target_tf(
    base_xyz: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    tcp_offset: float,
    retreat_step: float,
    retreat_index: int,
    yaw_about_normal: float = 0.0,
) -> np.ndarray:
    return target_frame_from_weld(
        base_xyz,
        normal,
        tangent,
        tcp_offset + retreat_step * retreat_index,
        yaw_about_normal=yaw_about_normal,
    )


class IsaacCollisionChecker:
    def __init__(
        self,
        world: Any,
        stage: Any,
        robot: Any,
        robot_prim_path: str,
        workpiece_prim_path: str,
        ground_prim_path: str,
        dof_indices: list[int],
        padding: float,
        contact_settle_steps: int,
        use_bbox_collision: bool,
    ) -> None:
        self.world = world
        self.stage = stage
        self.robot = robot
        self.robot_prim_path = robot_prim_path
        self.workpiece_prim_path = workpiece_prim_path
        self.ground_prim_path = ground_prim_path
        self.environment_prim_paths = [workpiece_prim_path, ground_prim_path]
        self.dof_indices = dof_indices
        self.padding = padding
        self.contact_settle_steps = max(1, int(contact_settle_steps))
        self.use_bbox_collision = use_bbox_collision
        self.query = self._get_scene_query()
        self.sim_query = self._get_simulation_interface()
        self.robot_collision_prims = self._collect_robot_collision_prims()
        self.last_collision_prim_path: str | None = None
        self.last_collision_bbox: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self.last_contact_pair: tuple[str, str] | None = None
        if not self.robot_collision_prims:
            raise RuntimeError(f"No robot collision prims found under {robot_prim_path}")
        self.robot_collision_prim_paths = [str(prim.GetPath()) for prim in self.robot_collision_prims]
        self.robot_contact_link_paths = self._build_robot_contact_link_paths()
        self._enable_contact_reports()
        mode = "bbox overlap fallback" if self.use_bbox_collision else "PhysX contact reports"
        log(f"[collision] Using {len(self.robot_collision_prims)} robot collision prims with {mode}.")

    def _get_scene_query(self) -> Any:
        from omni.physx import get_physx_scene_query_interface

        return get_physx_scene_query_interface()

    def _get_simulation_interface(self) -> Any:
        from omni.physx import get_physx_simulation_interface

        return get_physx_simulation_interface()

    def _enable_contact_reports(self) -> None:
        for prim in self.robot_collision_prims:
            enable_contact_report(prim)
        for prim_path in self.environment_prim_paths:
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                enable_contact_report(prim)

    def _collect_robot_collision_prims(self) -> list[Any]:
        from pxr import Usd, UsdPhysics

        root = self.stage.GetPrimAtPath(self.robot_prim_path)
        if not root.IsValid():
            raise RuntimeError(f"Robot root prim is invalid: {self.robot_prim_path}")

        all_prims = [prim for prim in Usd.PrimRange(root)]
        proxy_prims = []
        for prim in all_prims:
            if prim.HasAPI(UsdPhysics.CollisionAPI) and "/CollisionProxy/" in str(prim.GetPath()):
                proxy_prims.append(prim)
        if proxy_prims:
            log(f"[collision] Found {len(proxy_prims)} generated CollisionProxy prims with UsdPhysics.CollisionAPI.")
            return proxy_prims

        prims = []
        ignored_link_names = {"base_link", "base", "world"}
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and not any(
                f"/{name}" in str(prim.GetPath()) for name in ignored_link_names
            ):
                prims.append(prim)
        if prims:
            log(f"[collision] Found {len(prims)} non-base prims with UsdPhysics.CollisionAPI.")
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

    def _build_robot_contact_link_paths(self) -> list[str]:
        link_paths = set()
        for proxy_path in self.robot_collision_prim_paths:
            if "/CollisionProxy/" in proxy_path:
                link_paths.add(proxy_path.split("/CollisionProxy/")[0])
            else:
                link_paths.add(proxy_path)
        paths = sorted(link_paths)
        log("[collision] Contact-filter robot links: " + ", ".join(paths[:12]))
        return paths

    def set_q(self, q: np.ndarray) -> None:
        q = coerce_joint_vector(q, len(self.dof_indices), label="collision-check joint vector")
        full = read_full_joint_positions(self.robot)
        for local_idx, dof_idx in enumerate(self.dof_indices):
            full[dof_idx] = q[local_idx]
        self.robot.set_joint_positions(full)
        try:
            self.robot.set_joint_velocities(np.zeros_like(full))
        except Exception:
            pass
        self.world.step(render=False)

    def _overlap_box_hits_environment(self, half_extent: tuple[float, float, float], center: tuple[float, float, float]) -> bool:
        hits: list[str] = []

        def report_hit(hit: Any) -> bool:
            fields = ("rigid_body", "collider", "collision", "prim_path")
            for field in fields:
                value = getattr(hit, field, "")
                if value and any(env_path in str(value) for env_path in self.environment_prim_paths):
                    hits.append(str(value))
                    return False
            text = str(hit)
            if any(env_path in text for env_path in self.environment_prim_paths):
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

    def _path_from_physx_id(self, value: Any) -> str:
        try:
            from pxr import PhysicsSchemaTools

            return str(PhysicsSchemaTools.intToSdfPath(int(value)))
        except Exception:
            return str(value)

    def _contact_header_paths(self, header: Any) -> list[str]:
        paths = []
        for field in ("actor0", "actor1", "collider0", "collider1", "rigid_body0", "rigid_body1"):
            if hasattr(header, field):
                paths.append(self._path_from_physx_id(getattr(header, field)))
        return paths

    def _is_robot_path(self, path: str) -> bool:
        if not path.startswith(self.robot_prim_path):
            return False
        if path in {f"{self.robot_prim_path}/base_link", f"{self.robot_prim_path}/base", f"{self.robot_prim_path}/world"}:
            return False
        for proxy_path in self.robot_collision_prim_paths:
            if path.startswith(proxy_path) or proxy_path.startswith(path):
                return True
        for link_path in self.robot_contact_link_paths:
            if path.startswith(link_path) or link_path.startswith(path):
                return True
        return False

    def _is_environment_path(self, path: str) -> bool:
        return any(path.startswith(env_path) for env_path in self.environment_prim_paths)

    def _has_robot_environment_contact(self) -> bool:
        try:
            contact_report = self.sim_query.get_contact_report()
        except Exception as exc:
            raise RuntimeError(f"Failed to read PhysX contact report: {exc}") from exc

        headers = contact_report
        if isinstance(contact_report, tuple):
            headers = []
            for item in contact_report:
                try:
                    first = next(iter(item))
                except Exception:
                    continue
                if any(hasattr(first, field) for field in ("actor0", "collider0", "rigid_body0")):
                    headers = item
                    break
        for header in headers:
            header_type = str(getattr(header, "type", ""))
            if "LOST" in header_type:
                continue
            paths = self._contact_header_paths(header)
            robot_paths = [path for path in paths if self._is_robot_path(path)]
            env_paths = [path for path in paths if self._is_environment_path(path)]
            if robot_paths and env_paths:
                self.last_contact_pair = (robot_paths[0], env_paths[0])
                self.last_collision_prim_path = robot_paths[0]
                return True
        return False

    def is_state_valid(self, q: np.ndarray) -> bool:
        from pxr import Usd, UsdGeom

        self.last_collision_prim_path = None
        self.last_collision_bbox = None
        self.last_contact_pair = None

        if not self.use_bbox_collision:
            for _ in range(self.contact_settle_steps):
                self.set_q(q)
                if self._has_robot_environment_contact():
                    return False
            return True

        self.set_q(q)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True)
        for prim in self.robot_collision_prims:
            bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            min_v = bbox.GetMin()
            max_v = bbox.GetMax()
            half = tuple(max((max_v[i] - min_v[i]) * 0.5 + self.padding, 0.002) for i in range(3))
            center = tuple((max_v[i] + min_v[i]) * 0.5 for i in range(3))
            if self._overlap_box_hits_environment(half, center):
                self.last_collision_prim_path = str(prim.GetPath())
                self.last_collision_bbox = (center, half)
                return False
        return True


def make_articulation(world: Any, robot_prim_path: str) -> Any:
    try:
        from isaacsim.core.prims import SingleArticulation
    except ImportError:
        from omni.isaac.core.articulations import Articulation as SingleArticulation

    robot = SingleArticulation(prim_path=robot_prim_path, name="rrt_demo_robot")
    try:
        robot = world.scene.add(robot)
    except Exception:
        pass
    return robot


def ensure_physics_sim_view(world: Any, warmup_steps: int = 2) -> None:
    try:
        world.play()
    except Exception:
        pass
    for _ in range(max(warmup_steps, 0)):
        world.step(render=True)


def refresh_articulation_view(world: Any, robot: Any, warmup_steps: int = 2) -> None:
    world.reset()
    robot.initialize()
    ensure_physics_sim_view(world, warmup_steps=warmup_steps)


def warm_up_articulation_state(world: Any, robot: Any, steps: int = 3) -> np.ndarray:
    refresh_articulation_view(world, robot, warmup_steps=steps)
    joint_positions = read_full_joint_positions(robot)
    log(f"[robot] Warm-up joint state shape={joint_positions.shape}, dofs={robot_dof_count(robot)}")
    return joint_positions


def dof_indices_for(robot: Any, names: list[str]) -> list[int]:
    dof_names = list(robot.dof_names)
    missing = [name for name in names if name not in dof_names]
    if missing:
        raise RuntimeError(f"Imported articulation is missing joints {missing}; available={dof_names}")
    return [dof_names.index(name) for name in names]


def robot_dof_count(robot: Any) -> int:
    dof_names = getattr(robot, "dof_names", None)
    if dof_names is not None:
        return len(list(dof_names))
    count = getattr(robot, "num_dof", None)
    if count is not None:
        return int(count)
    raise RuntimeError(f"Could not determine articulation dof count for robot type={type(robot).__name__}")


def coerce_joint_vector(q: Any, expected_len: int, label: str = "joint vector") -> np.ndarray:
    value = q
    while isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != expected_len:
        raise RuntimeError(
            f"Expected {label} of length {expected_len}, got shape={arr.shape}, size={arr.size}, "
            f"type={type(q).__name__}, value={q!r}"
        )
    return arr


def read_full_joint_positions(robot: Any) -> np.ndarray:
    raw = robot.get_joint_positions()
    expected_len = robot_dof_count(robot)
    value = raw
    while isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()

    if value is None:
        log(f"[robot] get_joint_positions returned None; initializing zero joint state of length {expected_len}.")
        return np.zeros(expected_len, dtype=float)

    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        log(
            f"[robot] get_joint_positions returned scalar-like value type={type(raw).__name__}, "
            f"shape={arr.shape}; initializing zero joint state of length {expected_len}."
        )
        return np.zeros(expected_len, dtype=float)

    arr = arr.reshape(-1)
    if arr.size != expected_len:
        log(
            f"[robot] get_joint_positions size mismatch: expected {expected_len}, got {arr.size}. "
            "Falling back to zero joint state."
        )
        return np.zeros(expected_len, dtype=float)
    return arr.copy()


def set_robot_q(robot: Any, dof_indices: list[int], q: np.ndarray) -> None:
    q = coerce_joint_vector(q, len(dof_indices), label="robot joint vector")
    full = read_full_joint_positions(robot)
    for local_idx, dof_idx in enumerate(dof_indices):
        full[dof_idx] = q[local_idx]
    robot.set_joint_positions(full)


def build_endpoint_seed_pool(
    base_seeds: list[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    random_count: int,
) -> list[np.ndarray]:
    seeds = [np.clip(seed.astype(float), lower, upper) for seed in base_seeds]
    midpoint = 0.5 * (lower + upper)
    seeds.append(np.clip(midpoint, lower, upper))
    for _ in range(max(0, random_count)):
        seeds.append(rng.uniform(lower, upper))
    return seeds


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
    yaw_samples: int,
    random_seeds: int,
    ik_rot_weight: float,
    ik_max_iters: int,
    rng: np.random.Generator,
    endpoint_accept_validator: Callable[[np.ndarray], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    last_error: Exception | None = None
    accept_validator = checker.is_state_valid if endpoint_accept_validator is None else endpoint_accept_validator
    for retreat_index in range(max_retreat_steps + 1):
        seed_pool = build_endpoint_seed_pool(
            seeds,
            kinematics.lower,
            kinematics.upper,
            rng,
            random_count=random_seeds,
        )
        log(
            f"[endpoint] Trying {label}: retreat_index={retreat_index}/{max_retreat_steps}, "
            f"retreated={retreat_step * retreat_index:.3f} m, seeds={len(seed_pool)}, ik_max_iters={ik_max_iters}"
        )
        target_tf = retreated_target_tf(
            base_xyz,
            normal,
            tangent,
            tcp_offset,
            retreat_step,
            retreat_index,
            yaw_about_normal=0.0,
        )
        try:
            q = kinematics.solve_ik(
                target_tf,
                seed_pool,
                max_iters=ik_max_iters,
                rot_weight=ik_rot_weight,
            )
        except RuntimeError as exc:
            last_error = exc
            continue
        if accept_validator(q):
            if retreat_index > 0:
                log(f"[endpoint] {label} retreated={retreat_step * retreat_index:.3f} m along pose_normal.")
            return target_tf, q, retreat_index
        last_error = RuntimeError(
            f"{label} IK state collides at retreat_index={retreat_index}, "
            f"prim={checker.last_collision_prim_path}, contact={checker.last_contact_pair}, "
            f"bbox={checker.last_collision_bbox}"
        )
    raise RuntimeError(
        f"Could not find a collision-free {label} endpoint after {max_retreat_steps} retreat steps. "
        f"Last error: {last_error}"
    )


def checker_edge_valid(checker: Any, qa: np.ndarray, qb: np.ndarray, resolution: float) -> bool:
    for q in interpolate_edge(qa, qb, resolution)[1:]:
        if not checker.is_state_valid(q):
            return False
    return True


def has_rrt_root_escape(
    q_root: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    checker: Any,
    edge_resolution: float,
    escape_step: float,
) -> bool:
    for dof_idx in range(len(q_root)):
        for sign in (-1.0, 1.0):
            q_probe = q_root.copy()
            q_probe[dof_idx] = float(np.clip(q_probe[dof_idx] + sign * escape_step, lower[dof_idx], upper[dof_idx]))
            if np.allclose(q_probe, q_root, atol=1e-9):
                continue
            if not checker.is_state_valid(q_probe):
                continue
            if checker_edge_valid(checker, q_root, q_probe, edge_resolution):
                return True
    return False


def solve_planning_anchor(
    label: str,
    kinematics: URDFKinematics,
    checker: Any,
    endpoint_q: np.ndarray,
    base_xyz: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    tcp_offset: float,
    retreat_step: float,
    accepted_retreat_index: int,
    max_extra_retreat_steps: int,
    ik_rot_weight: float,
    ik_max_iters: int,
    rng: np.random.Generator,
    edge_resolution: float,
    escape_step: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if checker.is_state_valid(endpoint_q) and has_rrt_root_escape(
        endpoint_q,
        kinematics.lower,
        kinematics.upper,
        checker,
        edge_resolution=edge_resolution,
        escape_step=escape_step,
    ):
        return (
            retreated_target_tf(base_xyz, normal, tangent, tcp_offset, retreat_step, accepted_retreat_index, yaw_about_normal=0.0),
            endpoint_q.copy(),
            accepted_retreat_index,
        )

    last_error: Exception | None = None
    total_max_index = accepted_retreat_index + max(0, int(max_extra_retreat_steps))
    for retreat_index in range(accepted_retreat_index + 1, total_max_index + 1):
        target_tf = retreated_target_tf(
            base_xyz,
            normal,
            tangent,
            tcp_offset,
            retreat_step,
            retreat_index,
            yaw_about_normal=0.0,
        )
        seed_pool = build_endpoint_seed_pool(
            [endpoint_q],
            kinematics.lower,
            kinematics.upper,
            rng,
            random_count=0,
        )
        try:
            q_anchor = kinematics.solve_ik(
                target_tf,
                seed_pool,
                max_iters=ik_max_iters,
                rot_weight=ik_rot_weight,
            )
        except RuntimeError as exc:
            last_error = exc
            continue
        if not checker.is_state_valid(q_anchor):
            last_error = RuntimeError(f"{label} anchor collides at retreat_index={retreat_index}")
            continue
        if not checker_edge_valid(checker, endpoint_q, q_anchor, edge_resolution):
            last_error = RuntimeError(f"{label} anchor edge back to endpoint collides at retreat_index={retreat_index}")
            continue
        if not has_rrt_root_escape(
            q_anchor,
            kinematics.lower,
            kinematics.upper,
            checker,
            edge_resolution=edge_resolution,
            escape_step=escape_step,
        ):
            last_error = RuntimeError(f"{label} anchor still cannot expand from root at retreat_index={retreat_index}")
            continue
        log(
            f"[endpoint] {label} planning anchor retreated to {retreat_step * retreat_index:.3f} m "
            f"(accepted endpoint was {retreat_step * accepted_retreat_index:.3f} m)."
        )
        return target_tf, q_anchor, retreat_index

    raise RuntimeError(
        f"Could not find an expandable RRT anchor for {label} after {max_extra_retreat_steps} extra retreat steps. "
        f"Last error: {last_error}"
    )


def draw_curve(
    stage: Any,
    prim_path: str,
    points: list[np.ndarray],
    color: Any,
    width: float,
) -> None:
    from pxr import Gf, UsdGeom

    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    curve.CreateWidthsAttr([width])
    curve.CreateDisplayColorAttr([color])


def draw_pose_vector(stage: Any, prim_path: str, origin: np.ndarray, vector: np.ndarray, color: Any, length: float) -> None:
    v = normalize(np.asarray(vector, dtype=float))
    start = np.asarray(origin, dtype=float)
    end = start + v * length
    draw_curve(stage, prim_path, [start, end], color, width=0.006)


def set_translate_op(xformable: Any, translation: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    translate_attr = xformable.GetPrim().GetAttribute("xformOp:translate")
    if translate_attr.IsValid():
        translate_attr.Set(Gf.Vec3d(*translation))
        return

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*translation))
            return
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))


def draw_target_markers(
    stage: Any,
    weld_start: np.ndarray,
    weld_goal: np.ndarray,
    start_normal: np.ndarray,
    goal_normal: np.ndarray,
    tcp_start: np.ndarray,
    tcp_goal: np.ndarray,
    path_points: np.ndarray,
) -> None:
    from pxr import Gf, UsdGeom

    for name, point, color in (
        ("WeldStart", weld_start, Gf.Vec3f(1.0, 0.0, 0.0)),
        ("WeldGoal", weld_goal, Gf.Vec3f(1.0, 0.0, 0.0)),
        ("TcpStart", tcp_start, Gf.Vec3f(0.2, 1.0, 0.8)),
        ("TcpGoal", tcp_goal, Gf.Vec3f(1.0, 0.55, 0.15)),
    ):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/Debug/{name}")
        sphere.CreateRadiusAttr(0.007 if name.startswith("Weld") else 0.005)
        sphere.CreateDisplayColorAttr([color])
        set_translate_op(UsdGeom.Xformable(sphere.GetPrim()), point)

    draw_pose_vector(
        stage,
        "/World/Debug/StartPoseVector",
        weld_start,
        start_normal,
        Gf.Vec3f(0.0, 1.0, 0.0),
        length=0.12,
    )
    draw_pose_vector(
        stage,
        "/World/Debug/GoalPoseVector",
        weld_goal,
        goal_normal,
        Gf.Vec3f(0.0, 1.0, 0.0),
        length=0.12,
    )
    draw_curve(stage, "/World/Debug/TcpPath", list(path_points), Gf.Vec3f(0.05, 0.8, 1.0), width=0.006)


@dataclass(frozen=True)
class FixedCameraRig:
    target: tuple[float, float, float]
    eye: tuple[float, float, float]
    focal_length: float = 32.0
    focus_distance: float = 2.5


@dataclass
class RecordingSession:
    rep: Any
    world: Any
    args: argparse.Namespace
    writer: Any
    camera_prim_path: str
    rig: FixedCameraRig


def workpiece_bounds(workpiece_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bmin = np.array(workpiece_info["world_min"], dtype=float)
    bmax = np.array(workpiece_info["world_max"], dtype=float)
    return bmin, bmax


def workpiece_center_and_span(workpiece_info: dict[str, Any]) -> tuple[np.ndarray, float]:
    bmin, bmax = workpiece_bounds(workpiece_info)
    center = (bmin + bmax) * 0.5
    span = float(max(np.max(bmax - bmin), 0.35))
    return center, span


def recording_camera_rig(workpiece_info: dict[str, Any]) -> FixedCameraRig:
    center, span = workpiece_center_and_span(workpiece_info)
    target = (float(center[0]), float(center[1]), float(center[2] + max(0.14, span * 0.22)))
    initial_eye = np.array(
        [
            float(center[0] + max(1.15, span * 3.3)),
            float(center[1] - max(1.35, span * 3.8)),
            float(center[2] + max(0.82, span * 1.9)),
        ],
        dtype=float,
    )
    center_xy = center[:2]
    rotated_eye_xy = center_xy - (initial_eye[:2] - center_xy)
    return FixedCameraRig(
        target=target,
        eye=(float(rotated_eye_xy[0]), float(rotated_eye_xy[1]), float(initial_eye[2])),
    )


def ensure_recording_camera(stage: Any, rig: FixedCameraRig, prim_path: str = "/World/RecordingCamera") -> str:
    from pxr import UsdGeom

    camera = UsdGeom.Camera.Define(stage, prim_path)
    camera.CreateFocalLengthAttr(float(rig.focal_length))
    camera.CreateFocusDistanceAttr(float(rig.focus_distance))
    return prim_path


def set_transform_matrix(xformable: Any, matrix: Any) -> None:
    from pxr import UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            op.Set(matrix)
            return
    xformable.MakeMatrixXform().Set(matrix)


def set_camera_pose(stage: Any, camera_prim_path: str, eye: tuple[float, float, float], target: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim.IsValid():
        raise RuntimeError(f"Camera prim does not exist: {camera_prim_path}")

    view = Gf.Matrix4d(1.0)
    view.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0))
    set_transform_matrix(UsdGeom.Xformable(camera_prim), view.GetInverse())


def create_basic_writer(rep: Any, output_dir: Path | str, render_products: list[Any]) -> Any:
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(output_dir), rgb=True)
    writer.attach(render_products)
    return writer


def start_recording_session(
    rep: Any,
    world: Any,
    stage: Any,
    workpiece_info: dict[str, Any],
    frames_dir: Path,
    args: argparse.Namespace,
) -> RecordingSession:
    try:
        rep.orchestrator.set_capture_on_play(False)
    except Exception:
        pass

    rig = recording_camera_rig(workpiece_info)
    camera_prim_path = ensure_recording_camera(stage, rig)
    log(f"[demo] Recording camera eye={rig.eye}, look_at={rig.target}, fixed_at=180deg")
    set_camera_pose(stage, camera_prim_path, rig.eye, rig.target)
    render_product = rep.create.render_product(camera_prim_path, resolution=(args.width, args.height))
    writer = create_basic_writer(rep, frames_dir, [render_product])
    return RecordingSession(
        rep=rep,
        world=world,
        args=args,
        writer=writer,
        camera_prim_path=camera_prim_path,
        rig=rig,
    )


def capture_recording_frame(session: RecordingSession, stage: Any) -> None:
    set_camera_pose(stage, session.camera_prim_path, session.rig.eye, session.rig.target)
    session.world.step(render=True)
    step_replicator(session.rep, session.args)


def finish_recording_session(session: RecordingSession, frames_dir: Path) -> None:
    session.rep.orchestrator.wait_until_complete()
    session.writer.detach()
    png_count = len(list(frames_dir.glob("*.png"))) + len(list(frames_dir.glob("**/*.png")))
    log(f"[demo] PNG files written under {frames_dir}: {png_count}")
    if png_count == 0:
        raise RuntimeError(
            "Replicator did not write any PNG frames. Check that the script was run with "
            "Isaac Sim's python.sh, --headless was used on the cloud server, and cameras are enabled."
        )


def zero_robot_velocities(robot: Any) -> None:
    try:
        velocities = np.zeros_like(np.array(robot.get_joint_velocities(), dtype=float))
        robot.set_joint_velocities(velocities)
    except Exception:
        pass


def write_ik_endpoint_screenshots(
    rep: Any,
    world: Any,
    stage: Any,
    robot: Any,
    dof_indices: list[int],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    workpiece_info: dict[str, Any],
    args: argparse.Namespace,
) -> list[Path]:
    screenshot_dir = (
        args.ik_screenshot_dir
        if args.ik_screenshot_dir is not None
        else args.output.resolve().parent / f"{args.output.resolve().stem}_ik_endpoints"
    )
    screenshot_dir = screenshot_dir.resolve()
    if screenshot_dir.exists():
        shutil.rmtree(screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    rig = recording_camera_rig(workpiece_info)
    camera_prim_path = ensure_recording_camera(stage, rig, prim_path="/World/IkScreenshotCamera")

    saved_paths: list[Path] = []
    for index, (label, q) in enumerate((("start_ik", q_start), ("goal_ik", q_goal))):
        label_dir = screenshot_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        ensure_physics_sim_view(world, warmup_steps=1)
        set_robot_q(robot, dof_indices, q)
        zero_robot_velocities(robot)
        world.step(render=True)

        set_camera_pose(stage, camera_prim_path, rig.eye, rig.target)
        render_product = rep.create.render_product(camera_prim_path, resolution=(args.width, args.height))
        writer = create_basic_writer(rep, label_dir, [render_product])
        world.step(render=True)
        step_replicator(rep, args)
        rep.orchestrator.wait_until_complete()
        writer.detach()

        pngs = sorted(label_dir.glob("*.png")) or sorted(label_dir.glob("**/*.png"))
        if not pngs:
            raise RuntimeError(f"Failed to write {label} IK screenshot under {label_dir}")
        saved_paths.append(pngs[0])

    log("[demo] Pre-RRT IK endpoint screenshots saved:")
    for path in saved_paths:
        log(f"  {path}")
    return saved_paths


def failure_camera_poses(workpiece_info: dict[str, Any]) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    center, span = workpiece_center_and_span(workpiece_info)
    distance = max(1.1, span * 4.0)
    target = (float(center[0]), float(center[1]), float(center[2] + 0.12))
    return [
        ("front", (float(center[0] + distance), float(center[1] - distance), float(center[2] + 0.75)), target),
        ("back", (float(center[0] - distance), float(center[1] + distance), float(center[2] + 0.75)), target),
        ("left", (float(center[0] - distance), float(center[1] - distance), float(center[2] + 0.75)), target),
        ("right", (float(center[0] + distance), float(center[1] + distance), float(center[2] + 0.75)), target),
    ]


def write_failure_screenshots(
    rep: Any,
    world: Any,
    stage: Any,
    workpiece_info: dict[str, Any],
    args: argparse.Namespace,
) -> list[Path]:
    screenshot_dir = (
        args.failure_screenshot_dir
        if args.failure_screenshot_dir is not None
        else args.output.resolve().parent / f"{args.output.resolve().stem}_failure_screenshot"
    )
    screenshot_dir = screenshot_dir.resolve()
    if screenshot_dir.exists():
        shutil.rmtree(screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        rep.orchestrator.set_capture_on_play(False)
    except Exception:
        pass
    rig = recording_camera_rig(workpiece_info)
    render_products = []
    for index, (view_name, eye, target) in enumerate(failure_camera_poses(workpiece_info)):
        log(f"[demo] Failure screenshot camera {view_name}: eye={eye}, look_at={target}")
        camera_prim_path = ensure_recording_camera(stage, rig, f"/World/FailureCamera_{index}")
        set_camera_pose(stage, camera_prim_path, eye, target)
        render_products.append(rep.create.render_product(camera_prim_path, resolution=(args.width, args.height)))

    writer = create_basic_writer(rep, screenshot_dir, render_products)
    world.step(render=True)
    step_replicator(rep, args)
    rep.orchestrator.wait_until_complete()
    writer.detach()
    pngs = sorted(screenshot_dir.glob("*.png")) or sorted(screenshot_dir.glob("**/*.png"))
    if not pngs:
        raise RuntimeError(f"Failed to write planning-failure screenshots under {screenshot_dir}")
    log("[demo] Planning-failure screenshots saved:")
    for path in pngs:
        log(f"  {path}")
    return pngs


def import_single_robot(stage: Any, resolved_urdf: Path) -> str:
    target_path = "/World/UR5ePen"
    imported_path = import_robot_from_urdf(resolved_urdf, target_path, fix_base=True)
    if not stage.GetPrimAtPath(imported_path).IsValid() and stage.GetPrimAtPath("/ur5e_pen").IsValid():
        log("[demo] URDF importer ignored requested prim path; normalizing /ur5e_pen -> /World/UR5ePen")
        imported_path = "/ur5e_pen"
    robot_path = move_prim_to_path(stage, imported_path, target_path)
    log(f"[demo] Imported robot prim: {robot_path}")
    return robot_path


def step_scene(world: Any, rep: Any | None, args: argparse.Namespace | None) -> None:
    world.step(render=True)
    if rep is not None and args is not None:
        step_replicator(rep, args)


def run_playback(
    world: Any,
    stage: Any,
    robot: Any,
    dof_indices: list[int],
    q_playback: np.ndarray,
    args: argparse.Namespace,
    recording_session: RecordingSession | None,
) -> None:
    ensure_physics_sim_view(world, warmup_steps=1)

    def capture_step() -> None:
        if recording_session is None:
            step_scene(world, rep=None, args=None)
            return
        capture_recording_frame(recording_session, stage)

    set_robot_q(robot, dof_indices, q_playback[0])
    for _ in range(args.num_idle_frames):
        capture_step()
    for idx, q in enumerate(q_playback):
        set_robot_q(robot, dof_indices, q)
        capture_step()
        if recording_session is not None and (idx + 1) % max(args.fps, 1) == 0:
            log(f"[demo] Recorded playback frame {idx + 1}/{len(q_playback)}")
    for _ in range(args.num_idle_frames):
        capture_step()


def run_playback_segments(
    world: Any,
    stage: Any,
    robot: Any,
    dof_indices: list[int],
    playback_segments: list[np.ndarray],
    args: argparse.Namespace,
    recording_session: RecordingSession | None,
) -> None:
    for segment_index, q_segment in enumerate(playback_segments, start=1):
        log(f"[demo] Replaying segment {segment_index}/{len(playback_segments)} with {len(q_segment)} points.")
        run_playback(world, stage, robot, dof_indices, q_segment, args, recording_session)


def main() -> None:
    args = parse_args()
    if args.encode_only:
        args.output = args.output.resolve()
        if args.frames_dir is None:
            args.frames_dir = args.output.parent / f"{args.output.stem}_frames"
        frames_dir = args.frames_dir.resolve()
        encode_video(frames_dir, args.output, args.fps)
        log(f"[demo] Video saved to: {args.output}")
        return

    frames_dir = prepare_recording_paths(args) if args.record else None
    rng = np.random.default_rng(args.seed)

    if args.record and shutil.which("ffmpeg") is None:
        log(
            "[demo] WARNING: ffmpeg is not installed. Isaac will still record PNG frames; "
            "install ffmpeg on the server to encode MP4 automatically."
        )
    needs_replicator = args.record or args.save_ik_screenshots

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless, "enable_cameras": needs_replicator})
    try:
        configure_log_filters(args)
        try:
            from isaacsim.core.api import World
        except ImportError:
            from omni.isaac.core import World

        from omni.usd import get_context
        if needs_replicator:
            import omni.replicator.core as rep

        world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / args.fps, stage_units_in_meters=1.0)
        stage = get_context().get_stage()
        ensure_xform(stage, "/World/Debug")
        ground_prim_path = add_visual_ground(
            stage,
            size=args.visual_ground_size,
            z=args.visual_ground_z,
            opacity=args.visual_ground_opacity,
        )
        add_scene_lighting(stage)

        resolved_urdf = make_resolved_urdf(args.urdf)
        kinematics = URDFKinematics(resolved_urdf, include_tool_collision=args.include_tool_collision)
        robot_prim_path = import_single_robot(stage, resolved_urdf)
        add_urdf_collision_stl_proxies(
            stage=stage,
            robot_prim_path=robot_prim_path,
            resolved_urdf=resolved_urdf,
            include_tool_collision=args.include_tool_collision,
            show_collision_proxies=args.show_collision_proxies,
            collision_approximation=args.robot_collision_approximation,
            sdf_resolution=args.robot_sdf_resolution,
            sdf_subgrid_resolution=args.robot_sdf_subgrid_resolution,
        )
        workpiece_info = import_collision_stl(
            stage,
            stl_path=args.job_dir / "workpiece.stl",
            prim_path="/World/Workpiece",
            scale=args.workpiece_scale,
            z_offset=args.workpiece_z_offset,
            local_offset=tuple(float(v) for v in args.workpiece_offset),
            opacity=args.workpiece_opacity,
        )
        sdf_layer = None
        sdf_npz_path = None
        if args.trajopt and args.sdf_trajopt:
            sdf_npz_path = (
                args.workpiece_sdf_path.resolve()
                if args.workpiece_sdf_path is not None
                else (args.job_dir / "workpiece_sdf.npz").resolve()
            )
            sdf_layer, sdf_npz_path = load_or_build_workpiece_sdf(
                stl_path=(args.job_dir / "workpiece.stl").resolve(),
                scale=args.workpiece_scale,
                z_offset=args.workpiece_z_offset,
                local_offset=tuple(float(v) for v in args.workpiece_offset),
                npz_path=sdf_npz_path,
                config=SDFBuildConfig(
                    voxel_pitch=args.workpiece_sdf_pitch,
                    margin=args.workpiece_sdf_margin,
                    voxelize_method=args.workpiece_sdf_voxelize_method,
                    voxelize_max_iter=args.workpiece_sdf_voxelize_max_iter,
                ),
                logger=log,
                rebuild=args.rebuild_workpiece_sdf,
            )
            log(f"[SDF] Using workpiece SDF cache: {sdf_npz_path}")

        log("[demo] Resetting world and initializing articulation.")
        world.reset()
        ensure_physics_sim_view(world, warmup_steps=2)
        robot = make_articulation(world, robot_prim_path)
        world.reset()
        robot.initialize()
        log(f"[robot] Articulation object type={type(robot).__name__}")
        warm_up_articulation_state(world, robot)
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
            ground_prim_path=ground_prim_path,
            dof_indices=dof_indices,
            padding=args.collision_padding,
            contact_settle_steps=args.contact_settle_steps,
            use_bbox_collision=args.use_bbox_collision,
        )
        endpoint_accept_validator = checker.is_state_valid
        if args.trajopt and args.sdf_trajopt and sdf_layer is not None:
            endpoint_sdf_evaluator = KinematicSDFCollisionEvaluator(
                kinematics=kinematics,
                sdf_layer=sdf_layer,
                config=SDFTrajOptConfig(
                    num_waypoints=args.trajopt_waypoints,
                    max_waypoints=args.trajopt_max_waypoints,
                    maxiter=args.trajopt_maxiter,
                    collision_weight=args.sdf_collision_weight,
                    smoothness_weight=args.trajopt_smoothness_weight,
                    path_length_weight=args.trajopt_path_length_weight,
                    arm_safe_distance=args.sdf_arm_safe_distance,
                    tool_safe_distance=args.sdf_tool_safe_distance,
                    penetration_tol=args.sdf_penetration_tol,
                    arm_step_size=args.sdf_arm_step_size,
                    tool_step_size=args.sdf_tool_step_size,
                    constraint_point_stride=args.sdf_constraint_point_stride,
                    dense_check_resolution=min(args.edge_resolution, args.playback_resolution),
                    endpoint_relax_waypoints=args.sdf_endpoint_relax_waypoints,
                    endpoint_safe_distance_scale=args.sdf_endpoint_safe_distance_scale,
                ),
            )
            endpoint_accept_validator = endpoint_sdf_evaluator.is_state_nonpenetrating
            log("[demo] Endpoint SDF check uses non-penetration only; near-contact without negative SDF will not trigger retreat.")
            log(
                "[demo] SDF TrajOpt endpoint relaxation: "
                f"waypoints={args.sdf_endpoint_relax_waypoints}, "
                f"endpoint_safe_distance_scale={args.sdf_endpoint_safe_distance_scale:.2f}"
            )
        planned_segments: list[dict[str, Any]] = []
        found_noncoincident_transition = False
        scan_all_transitions = args.auto_first_transition or args.plan_all_transitions
        for targets in iter_transition_targets(
            args.job_dir,
            args.weld_index,
            args.workpiece_scale,
            args.workpiece_offset,
            args.workpiece_z_offset,
            args.tcp_normal_offset,
            args.transition_xyz_tol,
            scan_all_transitions,
        ):
            log(
                f"[demo] Loaded weld transition {targets['prev_weld_index']} -> {targets['next_weld_index']} "
                f"from {targets['vector_path']}"
            )
            log(
                f"[demo] Transition world xyz: {targets['start_xyz']} -> {targets['end_xyz']} "
                f"(distance={targets['transition_distance']:.6f} m)"
            )
            if targets.get("skip_planning", False):
                log(
                    f"[demo] Consecutive weld endpoint/startpoint already coincide within "
                    f"{args.transition_xyz_tol:.2e} m; continuing to next transition."
                )
                continue

            found_noncoincident_transition = True
            log("[demo] Solving collision-free IK endpoints.")
            try:
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
                    yaw_samples=args.endpoint_yaw_samples,
                    random_seeds=args.endpoint_random_seeds,
                    ik_rot_weight=args.endpoint_ik_rot_weight,
                    ik_max_iters=args.endpoint_ik_max_iters,
                    rng=rng,
                    endpoint_accept_validator=endpoint_accept_validator,
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
                    yaw_samples=args.endpoint_yaw_samples,
                    random_seeds=args.endpoint_random_seeds,
                    ik_rot_weight=args.endpoint_ik_rot_weight,
                    ik_max_iters=args.endpoint_ik_max_iters,
                    rng=rng,
                    endpoint_accept_validator=endpoint_accept_validator,
                )
            except RuntimeError as exc:
                log(
                    f"[demo] Transition {targets['prev_weld_index']} -> {targets['next_weld_index']} "
                    f"has no collision-free IK endpoints; trying next transition. Last error: {exc}"
                )
                continue

            targets["start_tf"] = start_tf
            targets["goal_tf"] = goal_tf
            log(f"[IK] q_start shape={np.asarray(q_start).shape}, q_goal shape={np.asarray(q_goal).shape}")
            log(f"[IK] q_start={np.round(q_start, 4)} retreat_steps={start_retreat_steps}")
            log(f"[IK] q_goal ={np.round(q_goal, 4)} retreat_steps={goal_retreat_steps}")
            draw_target_markers(
                stage=stage,
                weld_start=targets["start_xyz"],
                weld_goal=targets["end_xyz"],
                start_normal=targets["start_normal"],
                goal_normal=targets["end_normal"],
                tcp_start=targets["start_tf"][:3, 3],
                tcp_goal=targets["goal_tf"][:3, 3],
                path_points=np.array([targets["start_tf"][:3, 3], targets["goal_tf"][:3, 3]]),
            )
            if args.save_ik_screenshots and not planned_segments:
                write_ik_endpoint_screenshots(
                    rep=rep,
                    world=world,
                    stage=stage,
                    robot=robot,
                    dof_indices=dof_indices,
                    q_start=q_start,
                    q_goal=q_goal,
                    workpiece_info=workpiece_info,
                    args=args,
                )
                refresh_articulation_view(world, robot, warmup_steps=2)
                set_robot_q(robot, dof_indices, q_start)
                zero_robot_velocities(robot)
                world.step(render=True)

            collision_check_resolution = min(args.edge_resolution, args.playback_resolution)
            def is_edge_valid(qa: np.ndarray, qb: np.ndarray) -> bool:
                return checker_edge_valid(checker, qa, qb, collision_check_resolution)

            try:
                rrt_start_tf, q_rrt_start, rrt_start_retreat_steps = solve_planning_anchor(
                    label="start",
                    kinematics=kinematics,
                    checker=checker,
                    endpoint_q=q_start,
                    base_xyz=targets["start_xyz"],
                    normal=targets["start_normal"],
                    tangent=targets["tangent"],
                    tcp_offset=args.tcp_normal_offset,
                    retreat_step=args.endpoint_retreat_step,
                    accepted_retreat_index=start_retreat_steps,
                    max_extra_retreat_steps=args.planning_anchor_max_extra_steps,
                    ik_rot_weight=args.endpoint_ik_rot_weight,
                    ik_max_iters=args.endpoint_ik_max_iters,
                    rng=rng,
                    edge_resolution=collision_check_resolution,
                    escape_step=args.planning_anchor_escape_step,
                )
                rrt_goal_tf, q_rrt_goal, rrt_goal_retreat_steps = solve_planning_anchor(
                    label="goal",
                    kinematics=kinematics,
                    checker=checker,
                    endpoint_q=q_goal,
                    base_xyz=targets["end_xyz"],
                    normal=targets["end_normal"],
                    tangent=targets["tangent"],
                    tcp_offset=args.tcp_normal_offset,
                    retreat_step=args.endpoint_retreat_step,
                    accepted_retreat_index=goal_retreat_steps,
                    max_extra_retreat_steps=args.planning_anchor_max_extra_steps,
                    ik_rot_weight=args.endpoint_ik_rot_weight,
                    ik_max_iters=args.endpoint_ik_max_iters,
                    rng=rng,
                    edge_resolution=collision_check_resolution,
                    escape_step=args.planning_anchor_escape_step,
                )
            except RuntimeError as exc:
                log(
                    f"[demo] Transition {targets['prev_weld_index']} -> {targets['next_weld_index']} "
                    f"has no expandable RRT anchors; trying next transition. Last error: {exc}"
                )
                continue
            log(
                f"[demo] RRT anchors ready: start_retreat_steps={rrt_start_retreat_steps}, "
                f"goal_retreat_steps={rrt_goal_retreat_steps}"
            )

            t0 = time.perf_counter()
            try:
                q_anchor_seed_path = rrt_connect_plan_with_restarts(
                    q_start=q_rrt_start,
                    q_goal=q_rrt_goal,
                    lower=kinematics.lower,
                    upper=kinematics.upper,
                    is_state_valid=checker.is_state_valid,
                    is_edge_valid=is_edge_valid,
                    step_size=args.rrt_step_size,
                    max_iter=args.max_iter,
                    restarts=args.rrt_restarts,
                    goal_bias=args.goal_bias,
                    rng=rng,
                    logger=log,
                )
            except RuntimeError as exc:
                log(
                    f"[demo] Transition {targets['prev_weld_index']} -> {targets['next_weld_index']} "
                    f"failed during RRT planning; trying next transition. Last error: {exc}"
                )
                continue
            q_seed_parts: list[np.ndarray] = [q_start]
            if not np.allclose(q_rrt_start, q_start, atol=1e-6):
                q_seed_parts.append(q_rrt_start)
            for q_mid in q_anchor_seed_path[1:-1]:
                q_seed_parts.append(q_mid)
            if not np.allclose(q_rrt_goal, q_goal, atol=1e-6):
                q_seed_parts.append(q_rrt_goal)
            q_seed_parts.append(q_goal)
            q_seed_path = np.vstack(q_seed_parts)
            plan_time = time.perf_counter() - t0
            q_plan = q_seed_path
            trajopt_success = False
            if args.trajopt:
                trajopt_runner = None
                if args.sdf_trajopt:
                    if sdf_layer is None:
                        raise RuntimeError("SDF TrajOpt requested but workpiece SDF failed to load.")
                    sdf_evaluator = KinematicSDFCollisionEvaluator(
                        kinematics=kinematics,
                        sdf_layer=sdf_layer,
                        config=SDFTrajOptConfig(
                            num_waypoints=args.trajopt_waypoints,
                            max_waypoints=args.trajopt_max_waypoints,
                            maxiter=args.trajopt_maxiter,
                            collision_weight=args.sdf_collision_weight,
                            smoothness_weight=args.trajopt_smoothness_weight,
                            path_length_weight=args.trajopt_path_length_weight,
                            arm_safe_distance=args.sdf_arm_safe_distance,
                            tool_safe_distance=args.sdf_tool_safe_distance,
                            penetration_tol=args.sdf_penetration_tol,
                            arm_step_size=args.sdf_arm_step_size,
                            tool_step_size=args.sdf_tool_step_size,
                            constraint_point_stride=args.sdf_constraint_point_stride,
                            dense_check_resolution=collision_check_resolution,
                            endpoint_relax_waypoints=args.sdf_endpoint_relax_waypoints,
                            endpoint_safe_distance_scale=args.sdf_endpoint_safe_distance_scale,
                        ),
                    )

                    def trajopt_runner(q_seed_current: np.ndarray, logger: Any) -> tuple[np.ndarray, bool]:
                        return run_sdf_trajopt(
                            q_seed=q_seed_current,
                            lower=kinematics.lower,
                            upper=kinematics.upper,
                            evaluator=sdf_evaluator,
                            logger=logger,
                        )

                q_plan, optimization_info = optimize_path(
                    q_seed=q_seed_path,
                    lower=kinematics.lower,
                    upper=kinematics.upper,
                    is_state_valid=checker.is_state_valid,
                    is_edge_valid=is_edge_valid,
                    config=TrajOptConfig(
                        num_waypoints=args.trajopt_waypoints,
                        maxiter=args.trajopt_maxiter,
                        smoothness_weight=args.trajopt_smoothness_weight,
                        path_length_weight=args.trajopt_path_length_weight,
                        seed_weight=args.trajopt_seed_weight,
                        constraint_edge_resolution=args.edge_resolution,
                        shortcut_iterations=args.shortcut_iterations,
                        shortcut_passes=args.shortcut_passes,
                        averaging_passes=args.average_passes,
                        averaging_blend=args.average_blend,
                        validation_resolution=collision_check_resolution,
                    ),
                    rng=rng,
                    logger=log,
                    trajopt_runner=trajopt_runner,
                )
                trajopt_success = bool(optimization_info.get("trajopt_success", False))
            q_playback = densify_path(q_plan, args.playback_resolution)
            playback_collision = next((q for q in q_playback if not checker.is_state_valid(q)), None)
            if playback_collision is not None:
                log(
                    f"[demo] Transition {targets['prev_weld_index']} -> {targets['next_weld_index']} "
                    f"rejected after optimization: playback collision at sample={np.round(playback_collision, 4)}"
                )
                continue
            tcp_points = np.array([kinematics.forward(q)[:3, 3] for q in q_playback])
            planned_segments.append(
                {
                    "targets": targets,
                    "q_seed_path": q_seed_path,
                    "q_plan": q_plan,
                    "q_playback": q_playback,
                    "tcp_points": tcp_points,
                    "trajopt_success": trajopt_success,
                    "plan_time": plan_time,
                    "collision_check_resolution": collision_check_resolution,
                }
            )
            log(
                f"[demo] Planned transition {targets['prev_weld_index']} -> {targets['next_weld_index']}: "
                f"{len(q_seed_path)} RRT waypoints, {len(q_plan)} optimized waypoints, "
                f"{len(q_playback)} playback points in {plan_time:.2f}s "
                f"(trajopt={'accepted' if trajopt_success else 'skipped'})"
            )
            if not args.plan_all_transitions:
                break

        if not planned_segments:
            if not scan_all_transitions:
                raise RuntimeError(
                    f"Selected transition {args.weld_index} -> {args.weld_index + 1} could not be planned."
                )
            if not found_noncoincident_transition:
                raise RuntimeError(
                    "No transition requires planning: all consecutive weld endpoint/startpoint pairs "
                    f"coincide within tolerance {args.transition_xyz_tol:.2e} m."
                )
            raise RuntimeError("No valid transition could be fully planned and optimized.")

        playback_segments = [segment["q_playback"] for segment in planned_segments]
        tcp_points = np.vstack([segment["tcp_points"] for segment in planned_segments])
        first_targets = planned_segments[0]["targets"]
        last_targets = planned_segments[-1]["targets"]
        draw_target_markers(
            stage=stage,
            weld_start=first_targets["start_xyz"],
            weld_goal=last_targets["end_xyz"],
            start_normal=first_targets["start_normal"],
            goal_normal=last_targets["end_normal"],
            tcp_start=first_targets["start_tf"][:3, 3],
            tcp_goal=last_targets["goal_tf"][:3, 3],
            path_points=tcp_points,
        )
        log(
            f"[demo] Planned {len(planned_segments)} transition segments, "
            f"{sum(len(segment) for segment in playback_segments)} total playback points."
        )

        recording_session = None
        if args.record and frames_dir is not None:
            recording_session = start_recording_session(
                rep=rep,
                world=world,
                stage=stage,
                workpiece_info=workpiece_info,
                frames_dir=frames_dir,
                args=args,
            )
            refresh_articulation_view(world, robot, warmup_steps=1)
            log(f"[demo] Recording frames to: {frames_dir}")

        run_playback_segments(world, stage, robot, dof_indices, playback_segments, args, recording_session)

        if recording_session is not None and frames_dir is not None:
            finish_recording_session(recording_session, frames_dir)

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
        except Exception as exc:
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
