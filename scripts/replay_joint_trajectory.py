"""Replay a robot joint trajectory with optional STL workpiece import and video capture."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sim_parallel_welding import (  # noqa: E402
    encode_video,
    ensure_xform,
    load_stl_mesh,
    move_prim_to_path,
    import_stl_as_mesh,
    prepare_recording_paths,
    set_xform_translation,
    step_replicator,
)
from sim_welding_arm import (  # noqa: E402
    DEFAULT_INITIAL_JOINT_POS,
    DEFAULT_URDF,
    add_camera_view,
    import_robot_from_urdf,
    make_resolved_urdf,
    set_initial_joint_positions,
)


DEFAULT_TRAJECTORY_JOINT_NAMES = list(DEFAULT_INITIAL_JOINT_POS.keys())
DEFAULT_START_KEY_CANDIDATES = (
    "q_start",
    "start_joint_positions",
    "start_joints",
    "start_q",
    "start",
    "robot_start",
    "joint_start",
)
DEFAULT_TRAJECTORY_KEY_CANDIDATES = (
    "q_playback",
    "q_plan",
    "q_rrt_playback",
    "q_seed_path",
    "joint_positions",
    "trajectory",
    "waypoints",
    "q",
    "positions",
)
DEFAULT_REFERENCE_KEY_CANDIDATES = (
    "q_playback",
    "q_plan",
    "q_rrt_playback",
    "q_seed_path",
    "ground_truth",
    "gt_joint_positions",
    "real_joint_positions",
    "reference_joint_positions",
    "expert_joint_positions",
    "target_joint_positions",
    "joint_positions",
    "trajectory",
    "waypoints",
    "q",
    "positions",
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a robot joint-angle trajectory in Isaac Sim with optional STL workpiece import."
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Joint trajectory file: json/csv/txt/npy/npz. Not required with --encode-only.",
    )
    parser.add_argument(
        "--pred-check-dir",
        type=Path,
        default=None,
        help="Directory containing pred_joint_horizon.npy, gt_joint_horizon.npy, and optional summary.json.",
    )
    parser.add_argument(
        "--trajectory-representation",
        choices=("auto", "absolute", "delta"),
        default="auto",
        help="Interpret trajectory waypoints as absolute joint angles or joint-angle deltas. "
        "In auto mode, all supported formats default to absolute.",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="UR5e welding-arm URDF.")
    parser.add_argument("--stl", type=Path, default=None, help="Optional STL workpiece path.")
    parser.add_argument("--robot-prim-path", default="/World/UR5ePen", help="USD prim path for the imported robot.")
    parser.add_argument("--workpiece-prim-path", default="/World/Workpiece", help="USD prim path for the imported STL workpiece.")
    parser.add_argument(
        "--reference-trajectory",
        type=Path,
        default=None,
        help="Optional reference/original joint trajectory file: json/csv/txt/npy/npz.",
    )
    parser.add_argument("--reference-npz", type=Path, default=None, help="Optional NPZ file containing the reference trajectory.")
    parser.add_argument("--reference-key", default=None, help="Key inside --reference-npz for the reference trajectory.")
    parser.add_argument(
        "--playback-mode",
        choices=("sequential", "overlay"),
        default="sequential",
        help="Replay predicted and reference trajectories sequentially on one robot, or overlaid on two robots.",
    )
    parser.add_argument(
        "--overlay-finish-behavior",
        choices=("hide", "hold"),
        default="hide",
        help="When one overlaid trajectory finishes before the other, hide it or hold its final pose.",
    )
    parser.add_argument(
        "--overlay-reference-visual",
        choices=("skeleton", "visual_mesh", "articulation"),
        default="visual_mesh",
        help="Render the overlaid reference as a pure visual FK skeleton, visual meshes, or a second Isaac articulation.",
    )
    parser.add_argument(
        "--overlay-visual-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Small extra visual/world offset for the overlaid reference robot to reduce z-fighting.",
    )
    parser.add_argument(
        "--reference-representation",
        choices=("auto", "absolute", "delta"),
        default="absolute",
        help="Interpret the reference trajectory as absolute joint angles or joint-angle deltas.",
    )
    parser.add_argument("--reference-robot-prim-path", default="/World/UR5ePenGroundTruth", help="USD prim path for the reference robot.")
    parser.add_argument(
        "--reference-workpiece-prim-path",
        default="/World/WorkpieceReference",
        help="Deprecated compatibility option; reference replay now uses the single shared workpiece.",
    )
    parser.add_argument(
        "--reference-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="World-space offset applied to the reference robot.",
    )
    parser.add_argument(
        "--reference-ghost-opacity",
        type=float,
        default=0.28,
        help="Display opacity for the reference/ground-truth robot ghost. Set to 1.0 for opaque.",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without the UI.")
    parser.add_argument("--floating", action="store_true", help="Do not fix the robot base to the world.")
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0, help="Physics timestep in seconds.")
    parser.add_argument("--rendering-dt", type=float, default=1.0 / 60.0, help="Rendering timestep in seconds.")
    parser.add_argument(
        "--render-only-replay",
        dest="render_only_replay",
        action="store_true",
        default=True,
        help="For replay frames, set joint positions and render without advancing physics.",
    )
    parser.add_argument(
        "--physics-replay",
        dest="render_only_replay",
        action="store_false",
        help="Use the older behavior that advances physics on every replay frame.",
    )
    parser.add_argument("--hold-steps", type=int, default=None, help="Simulation steps to hold each waypoint.")
    parser.add_argument(
        "--waypoint-substeps",
        type=int,
        default=None,
        help="Linear interpolation substeps inserted between adjacent waypoints to slow and smooth playback.",
    )
    parser.add_argument(
        "--segment-gap-steps",
        type=int,
        default=20,
        help="Extra simulation steps to hold between sequential predicted/ground-truth segments.",
    )
    parser.add_argument("--loop", type=int, default=1, help="Repeat count for the entire trajectory.")
    parser.add_argument("--fps", type=int, default=30, help="Recording frame rate.")
    parser.add_argument("--width", type=int, default=1280, help="Recording width.")
    parser.add_argument("--height", type=int, default=720, help="Recording height.")
    parser.add_argument("--record", action="store_true", help="Record RGB frames and encode MP4.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/replay_joint_trajectory.mp4", help="MP4 output path.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Directory used for RGB frame images.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep exported RGB frames after MP4 encoding.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing video and frame directory.")
    parser.add_argument("--encode-only", action="store_true", help="Encode an existing --frames-dir to --output without replaying.")
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=None,
        help="Optional ffmpeg executable path. Can also be set with WELDROBOT_FFMPEG.",
    )
    parser.add_argument("--rt-subframes", type=int, default=8, help="Replicator render subframes per captured frame.")
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        default=(2.0, -2.4, 1.4),
        metavar=("X", "Y", "Z"),
        help="Recording camera position in world coordinates.",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.45),
        metavar=("X", "Y", "Z"),
        help="Recording camera look-at target in world coordinates.",
    )
    parser.add_argument(
        "--camera-z-rotation-deg",
        type=float,
        default=180.0,
        help="Rotate the recording camera eye around the look-at target about the world Z axis.",
    )
    parser.add_argument(
        "--joint-names",
        nargs="+",
        default=None,
        help="Optional joint-name list for formats that do not store names.",
    )
    parser.add_argument(
        "--start-joint-positions",
        type=float,
        nargs="+",
        default=None,
        help="Optional absolute joint-angle start state used when replaying delta trajectories.",
    )
    parser.add_argument(
        "--start-npz",
        type=Path,
        default=None,
        help="Optional NPZ file containing the absolute start joint positions.",
    )
    parser.add_argument(
        "--start-key",
        default=None,
        help="Key inside --start-npz for the absolute start joint positions. "
        "If omitted, common key names are searched automatically.",
    )
    parser.add_argument(
        "--reference-start-joint-positions",
        type=float,
        nargs="+",
        default=None,
        help="Optional absolute start state used when the reference trajectory is stored as deltas.",
    )
    parser.add_argument(
        "--reference-start-npz",
        type=Path,
        default=None,
        help="Optional NPZ file containing the absolute reference start joint positions.",
    )
    parser.add_argument(
        "--reference-start-key",
        default=None,
        help="Key inside --reference-start-npz for the absolute reference start joint positions.",
    )
    parser.add_argument(
        "--workpiece-scale",
        type=float,
        default=0.001,
        help="Scale applied to STL vertices. Generated data is usually in mm; Isaac Sim scene is in m.",
    )
    parser.add_argument(
        "--workpiece-z-offset",
        type=float,
        default=0.0025,
        help="Additional Z offset applied to imported STL vertices after scaling.",
    )
    parser.add_argument(
        "--workpiece-offset",
        type=float,
        nargs=3,
        default=[0.5, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Local STL workpiece offset relative to the robot base, in meters.",
    )
    parser.add_argument("--debug-workpiece-box", action="store_true", help="Add debug axes at the STL workpiece center.")
    parser.add_argument(
        "--visualization-file",
        type=Path,
        default=None,
        help="Optional JSON/NPZ replay overlay containing start/end markers, vectors, and spline control points.",
    )
    parser.add_argument(
        "--visualization-frame",
        choices=("workpiece_mm", "world_m"),
        default="workpiece_mm",
        help="Coordinate frame for --visualization-file positions. workpiece_mm is transformed with workpiece scale/offset/z-offset.",
    )
    parser.add_argument("--control-point-radius", type=float, default=0.008, help="Replay overlay control-point sphere radius in meters.")
    parser.add_argument("--control-polyline-width", type=float, default=0.004, help="Replay overlay control polygon width in meters.")
    parser.add_argument("--debug-vector-length", type=float, default=0.12, help="Replay overlay start/end vector length in meters.")
    return parser.parse_args()


def _coerce_trajectory_array(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise RuntimeError(f"{label} must be a 2D numeric array, got shape={arr.shape}")
    return arr


def _normalize_joint_name_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"Joint names must be a list/tuple, got {type(value).__name__}")
    names = [str(item) for item in value]
    if not names:
        raise RuntimeError("Joint names list is empty.")
    return names


def _load_json_trajectory(path: Path) -> tuple[np.ndarray, list[str] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return _coerce_trajectory_array(payload, "JSON trajectory"), None
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported JSON trajectory structure: {type(payload).__name__}")

    joint_names = _normalize_joint_name_list(payload.get("joint_names"))
    for key in ("joint_positions", "trajectory", "waypoints", "q", "positions"):
        if key in payload:
            return _coerce_trajectory_array(payload[key], f"JSON field '{key}'"), joint_names
    raise RuntimeError(
        "JSON trajectory must be a list of waypoints or contain one of: "
        "'joint_positions', 'trajectory', 'waypoints', 'q', 'positions'."
    )


def _split_csv_row(row: list[str]) -> list[str]:
    if len(row) == 1:
        return [item for item in row[0].replace(",", " ").split() if item]
    return [item.strip() for item in row if item.strip()]


def _load_csv_trajectory(path: Path) -> tuple[np.ndarray, list[str] | None]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [_split_csv_row(row) for row in csv.reader(f) if row]

    rows = [row for row in rows if row]
    if not rows:
        raise RuntimeError(f"Trajectory file is empty: {path}")

    joint_names: list[str] | None = None
    try:
        first_numeric = [float(item) for item in rows[0]]
    except ValueError:
        joint_names = [str(item) for item in rows[0]]
        rows = rows[1:]
        if not rows:
            raise RuntimeError(f"Trajectory file contains only a header: {path}")
        first_numeric = [float(item) for item in rows[0]]

    data = [first_numeric]
    for row in rows[1:]:
        data.append([float(item) for item in row])

    return _coerce_trajectory_array(data, "CSV trajectory"), joint_names


def _load_npz_array(
    path: Path,
    preferred_key: str | None,
    candidate_keys: tuple[str, ...],
    label: str,
) -> tuple[np.ndarray, list[str] | None]:
    with np.load(path, allow_pickle=True) as data:
        joint_names = _normalize_joint_name_list(data["joint_names"]) if "joint_names" in data else None
        keys = [preferred_key] if preferred_key else list(candidate_keys)
        for key in keys:
            if key in data:
                return _coerce_trajectory_array(data[key], f"NPZ field '{key}'"), joint_names
        if len(data.files) == 1:
            only_key = data.files[0]
            return _coerce_trajectory_array(data[only_key], f"NPZ field '{only_key}'"), joint_names
        raise RuntimeError(
            f"{label} NPZ must contain one of: {keys}. Available keys={list(data.files)}"
        )


def _load_npz_trajectory(path: Path, preferred_key: str | None = None) -> tuple[np.ndarray, list[str] | None]:
    return _load_npz_array(path, preferred_key, DEFAULT_TRAJECTORY_KEY_CANDIDATES, "Trajectory")


def load_joint_trajectory(
    path: Path,
    cli_joint_names: list[str] | None,
    preferred_npz_key: str | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        waypoints, file_joint_names = _load_json_trajectory(path)
    elif suffix in {".csv", ".txt"}:
        waypoints, file_joint_names = _load_csv_trajectory(path)
    elif suffix == ".npy":
        waypoints, file_joint_names = _coerce_trajectory_array(np.load(path, allow_pickle=False), "NPY trajectory"), None
    elif suffix == ".npz":
        waypoints, file_joint_names = _load_npz_trajectory(path, preferred_npz_key)
    else:
        raise RuntimeError(f"Unsupported trajectory format '{suffix}'. Use json/csv/txt/npy/npz.")

    joint_names = cli_joint_names or file_joint_names
    if joint_names is not None and len(joint_names) != waypoints.shape[1]:
        raise RuntimeError(
            f"Joint-name count {len(joint_names)} does not match waypoint width {waypoints.shape[1]} "
            f"for trajectory {path}"
        )
    return waypoints, joint_names


def resolve_trajectory_representation(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return "absolute"


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _fallback_stl_from_summary(summary: dict[str, Any]) -> Path | None:
    stl_value = summary.get("stl_path")
    candidates: list[Path] = []
    if isinstance(stl_value, str) and stl_value:
        stl_path = Path(stl_value).expanduser()
        candidates.append(stl_path)
        match = re.search(r"(job_\d+)", stl_value)
        if match:
            job_name = match.group(1)
            candidates.extend(
                [
                    REPO_ROOT / "data_generation/data/generated_jobs" / job_name / "workpiece.stl",
                    REPO_ROOT / "data_generation/data/generated_jobs" / job_name / "workpiece_sim.stl",
                    REPO_ROOT / "data_generation/data/generated_jobs/simple_jobs" / job_name / "workpiece.stl",
                    REPO_ROOT / "data_generation/data/generated_jobs/simple_jobs" / job_name / "workpiece_sim.stl",
                ]
            )
    return _first_existing_path(candidates)


def apply_pred_check_defaults(args: argparse.Namespace) -> None:
    if args.pred_check_dir is None:
        if args.hold_steps is None:
            args.hold_steps = 1
        if args.waypoint_substeps is None:
            args.waypoint_substeps = 1
        return

    pred_check_dir = args.pred_check_dir.expanduser().resolve()
    if not pred_check_dir.is_dir():
        raise FileNotFoundError(f"--pred-check-dir does not exist or is not a directory: {pred_check_dir}")
    args.pred_check_dir = pred_check_dir

    if args.trajectory is None:
        pred_path = pred_check_dir / "pred_joint_horizon.npy"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing predicted trajectory: {pred_path}")
        args.trajectory = pred_path
        log(f"[weldRobot] Using predicted trajectory from pred-check dir: {pred_path}")

    if args.reference_trajectory is None and args.reference_npz is None:
        gt_path = pred_check_dir / "gt_joint_horizon.npy"
        if gt_path.is_file():
            args.reference_trajectory = gt_path
            log(f"[weldRobot] Using ground-truth trajectory from pred-check dir: {gt_path}")
        else:
            log(f"[weldRobot] No ground-truth trajectory found at: {gt_path}")

    if args.stl is None:
        summary_path = pred_check_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise RuntimeError(f"summary.json must contain a JSON object: {summary_path}")
            stl_path = _fallback_stl_from_summary(summary)
            if stl_path is not None:
                args.stl = stl_path
                log(f"[weldRobot] Using workpiece STL from pred-check summary/fallback: {stl_path}")
            else:
                log(f"[weldRobot] Could not resolve workpiece STL from: {summary_path}")

    if args.hold_steps is None:
        args.hold_steps = 1
    if args.waypoint_substeps is None:
        args.waypoint_substeps = 6
        log("[weldRobot] pred-check replay defaulting to waypoint_substeps=6 for slower, smoother playback")


def create_articulation(robot_prim_path: str):
    try:
        from isaacsim.core.prims import SingleArticulation
    except ImportError:
        from omni.isaac.core.articulations import Articulation as SingleArticulation

    articulation_name = "ur5e_pen_" + robot_prim_path.strip("/").replace("/", "_")
    robot = SingleArticulation(prim_path=robot_prim_path, name=articulation_name)
    robot.initialize()
    return robot


def robot_dof_count(robot: Any) -> int:
    dof_names = getattr(robot, "dof_names", None)
    if dof_names is not None:
        return len(list(dof_names))
    count = getattr(robot, "num_dof", None)
    if count is not None:
        return int(count)
    raise RuntimeError(f"Could not determine articulation dof count for robot type={type(robot).__name__}")


def read_full_joint_positions(robot: Any) -> np.ndarray:
    raw = robot.get_joint_positions()
    expected_len = robot_dof_count(robot)
    if raw is None:
        return np.zeros(expected_len, dtype=float)

    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        return np.zeros(expected_len, dtype=float)
    arr = arr.reshape(-1)
    if arr.size != expected_len:
        return np.zeros(expected_len, dtype=float)
    return arr.copy()


def resolve_dof_indices(robot: Any, joint_names: list[str] | None, trajectory_width: int) -> tuple[list[str], list[int]]:
    dof_names = list(robot.dof_names)
    if joint_names is None:
        if trajectory_width == len(DEFAULT_TRAJECTORY_JOINT_NAMES):
            joint_names = DEFAULT_TRAJECTORY_JOINT_NAMES
        elif trajectory_width == len(dof_names):
            joint_names = dof_names
        else:
            raise RuntimeError(
                f"Trajectory width={trajectory_width} cannot be mapped automatically. "
                f"Provide --joint-names explicitly. Robot DOFs={dof_names}"
            )

    missing = [name for name in joint_names if name not in dof_names]
    if missing:
        raise RuntimeError(f"Imported articulation is missing joints {missing}; available={dof_names}")
    return joint_names, [dof_names.index(name) for name in joint_names]


def load_start_array_from_npz(npz_path: Path, start_key: str | None) -> np.ndarray:
    npz_path = npz_path.expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Start NPZ file does not exist: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        candidate_keys = [start_key] if start_key else list(DEFAULT_START_KEY_CANDIDATES)
        key = next((name for name in candidate_keys if name in data), None)
        if key is None:
            raise RuntimeError(
                f"Could not find start joint field in {npz_path}. "
                f"Available keys={list(data.files)}; searched={candidate_keys}"
            )
        value = np.asarray(data[key], dtype=float).reshape(-1)
        return value


def current_joint_subset(robot: Any, dof_indices: list[int]) -> np.ndarray:
    full = read_full_joint_positions(robot)
    return np.asarray([full[dof_idx] for dof_idx in dof_indices], dtype=float)


def resolve_start_joint_positions(
    robot: Any,
    dof_indices: list[int],
    cli_start_joint_positions: list[float] | None,
    start_npz: Path | None,
    start_key: str | None,
) -> np.ndarray:
    if cli_start_joint_positions is None:
        if start_npz is None:
            return current_joint_subset(robot, dof_indices)
        start = load_start_array_from_npz(start_npz, start_key)
        if start.size != len(dof_indices):
            raise RuntimeError(
                f"Start joint count from {start_npz} is {start.size}, expected {len(dof_indices)}"
            )
        log(f"[weldRobot] Loaded absolute start joints from {start_npz.name}")
        return start

    start = np.asarray(cli_start_joint_positions, dtype=float).reshape(-1)
    if start.size != len(dof_indices):
        raise RuntimeError(
            f"--start-joint-positions expects {len(dof_indices)} values, got {start.size}"
        )
    return start


def build_absolute_trajectory(
    trajectory: np.ndarray,
    representation: str,
    start_joint_positions: np.ndarray,
    lower_limits: np.ndarray | None = None,
    upper_limits: np.ndarray | None = None,
) -> np.ndarray:
    if representation == "absolute":
        absolute = np.asarray(trajectory, dtype=float)
    elif representation == "delta":
        trajectory = np.asarray(trajectory, dtype=float)
        start_joint_positions = np.asarray(start_joint_positions, dtype=float).reshape(1, -1)
        absolute = np.cumsum(trajectory, axis=0) + start_joint_positions
    else:
        raise RuntimeError(f"Unsupported trajectory representation: {representation}")
    if lower_limits is not None and upper_limits is not None:
        absolute = np.clip(absolute, lower_limits.reshape(1, -1), upper_limits.reshape(1, -1))
    return absolute


def densify_trajectory(trajectory: np.ndarray, substeps: int) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=float)
    if substeps <= 1 or len(trajectory) <= 1:
        return trajectory.copy()

    dense_waypoints: list[np.ndarray] = []
    for idx in range(len(trajectory) - 1):
        start = trajectory[idx]
        end = trajectory[idx + 1]
        for substep_idx in range(substeps):
            alpha = float(substep_idx) / float(substeps)
            dense_waypoints.append((1.0 - alpha) * start + alpha * end)
    dense_waypoints.append(trajectory[-1].copy())
    return np.asarray(dense_waypoints, dtype=float)


def import_robot_instance(
    stage: Any,
    resolved_urdf: Path,
    requested_prim_path: str,
    fix_base: bool,
    world_offset: tuple[float, float, float] | None = None,
) -> str:
    imported_robot_path = import_robot_from_urdf(resolved_urdf, requested_prim_path, fix_base=fix_base)
    if not stage.GetPrimAtPath(imported_robot_path).IsValid() and stage.GetPrimAtPath("/ur5e_pen").IsValid():
        imported_robot_path = "/ur5e_pen"
    robot_prim_path = move_prim_to_path(stage, imported_robot_path, requested_prim_path)
    if world_offset is not None:
        set_xform_translation(stage.GetPrimAtPath(robot_prim_path), world_offset)
    return robot_prim_path


def apply_reference_ghost_material(stage: Any, robot_prim_path: str, opacity: float) -> None:
    opacity = max(0.0, min(1.0, float(opacity)))
    if opacity >= 0.999:
        return

    from pxr import Gf, Sdf, UsdGeom, UsdShade

    material_path = f"{robot_prim_path}/GroundTruthGhostMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.55, 0.12))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    mesh_count = 0
    root_prefix = robot_prim_path.rstrip("/") + "/"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path != robot_prim_path and not path.startswith(root_prefix):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.55, 0.12)])
        mesh.CreateDisplayOpacityAttr([opacity])
        UsdShade.MaterialBindingAPI(prim).Bind(material)
        mesh_count += 1

    log(
        f"[weldRobot] Applied ground-truth ghost material to {mesh_count} mesh prims "
        f"under {robot_prim_path}, opacity={opacity:.2f}"
    )


def disable_collisions_under_prim(stage: Any, root_prim_path: str) -> None:
    from pxr import UsdPhysics

    disabled = 0
    root_prefix = root_prim_path.rstrip("/") + "/"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path != root_prim_path and not path.startswith(root_prefix):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_api = UsdPhysics.CollisionAPI(prim)
        attr = collision_api.GetCollisionEnabledAttr()
        if attr.IsValid():
            attr.Set(False)
        else:
            collision_api.CreateCollisionEnabledAttr(False)
        disabled += 1
    log(f"[weldRobot] Disabled collisions on {disabled} prims under {root_prim_path}")


def set_prim_visibility(stage: Any, prim_path: str, visible: bool) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return
    imageable.MakeVisible() if visible else imageable.MakeInvisible()


def normalize_vector(vector: Any, label: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.size != 3:
        raise RuntimeError(f"{label} must contain exactly 3 values, got shape={arr.shape}")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise RuntimeError(f"{label} must be non-zero")
    return arr / norm


def coerce_optional_points(value: Any, label: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise RuntimeError(f"{label} must have shape (N, 3), got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{label} contains non-finite values")
    return arr


def transform_overlay_points(points: np.ndarray | None, args: argparse.Namespace) -> np.ndarray | None:
    if points is None:
        return None
    points = np.asarray(points, dtype=float)
    if args.visualization_frame == "world_m":
        return points.copy()
    transformed = points * float(args.workpiece_scale) + np.asarray(args.workpiece_offset, dtype=float)[None, :]
    transformed[:, 2] += float(args.workpiece_z_offset)
    return transformed


def transform_overlay_position(position: Any, args: argparse.Namespace, label: str) -> np.ndarray:
    arr = np.asarray(position, dtype=float).reshape(-1)
    if arr.size != 3:
        raise RuntimeError(f"{label} must contain exactly 3 values, got shape={arr.shape}")
    return transform_overlay_points(arr.reshape(1, 3), args)[0]


def transform_optional_overlay_position(position: Any, args: argparse.Namespace, label: str) -> np.ndarray | None:
    if position is None:
        return None
    return transform_overlay_position(position, args, label)


def _json_overlay_field(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def load_replay_visualization_overlay(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.visualization_file is None:
        return None
    path = args.visualization_file.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Visualization file does not exist: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Visualization JSON must be an object.")
        start = payload.get("start") or {}
        end = payload.get("end") or {}
        if not isinstance(start, dict) or not isinstance(end, dict):
            raise RuntimeError("Visualization JSON fields 'start' and 'end' must be objects when provided.")
        start_position = transform_optional_overlay_position(start.get("position"), args, "start.position")
        end_position = transform_optional_overlay_position(end.get("position"), args, "end.position")
        start_vector = normalize_vector(start.get("vector"), "start.vector") if start.get("vector") is not None else None
        end_vector = normalize_vector(end.get("vector"), "end.vector") if end.get("vector") is not None else None
        pred_control_points = coerce_optional_points(
            _json_overlay_field(payload, "pred_control_points", "predicted_control_points"),
            "pred_control_points",
        )
        gt_control_points = coerce_optional_points(
            _json_overlay_field(payload, "gt_control_points", "ground_truth_control_points", "true_control_points"),
            "gt_control_points",
        )
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            start_position = transform_optional_overlay_position(
                data["start_position"] if "start_position" in data else None,
                args,
                "start_position",
            )
            end_position = transform_optional_overlay_position(
                data["end_position"] if "end_position" in data else None,
                args,
                "end_position",
            )
            start_vector = normalize_vector(data["start_vector"], "start_vector") if "start_vector" in data else None
            end_vector = normalize_vector(data["end_vector"], "end_vector") if "end_vector" in data else None
            pred_control_points = coerce_optional_points(
                data["pred_control_points"] if "pred_control_points" in data else None,
                "pred_control_points",
            )
            gt_control_points = coerce_optional_points(
                data["gt_control_points"] if "gt_control_points" in data else None,
                "gt_control_points",
            )
    else:
        raise RuntimeError(f"Unsupported visualization file extension: {path.suffix}")

    overlay = {
        "path": path,
        "start_position": start_position,
        "end_position": end_position,
        "start_vector": start_vector,
        "end_vector": end_vector,
        "pred_control_points": transform_overlay_points(pred_control_points, args),
        "gt_control_points": transform_overlay_points(gt_control_points, args),
    }
    log(f"[weldRobot] Loaded replay visualization overlay: {path}")
    return overlay


def set_debug_translation(xformable: Any, translation: np.ndarray) -> None:
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


def set_transform_matrix(xformable: Any, matrix: Any) -> None:
    from pxr import UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            op.Set(matrix)
            return
    xformable.MakeMatrixXform().Set(matrix)


def numpy_to_gf_matrix(matrix: np.ndarray) -> Any:
    from pxr import Gf

    matrix = np.asarray(matrix, dtype=float)
    gf_matrix = Gf.Matrix4d(1.0)
    for row_idx in range(4):
        gf_matrix.SetRow(row_idx, Gf.Vec4d(*[float(v) for v in matrix[row_idx]]))
    return gf_matrix


def draw_debug_curve(stage: Any, prim_path: str, points: list[np.ndarray], color: Any, width: float) -> None:
    from pxr import Gf, UsdGeom

    if len(points) < 2:
        return
    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(*np.asarray(point, dtype=float)) for point in points])
    curve.CreateWidthsAttr([float(width)])
    curve.CreateDisplayColorAttr([color])


def draw_debug_sphere(stage: Any, prim_path: str, point: np.ndarray, color: Any, radius: float) -> None:
    from pxr import UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([color])
    set_debug_translation(UsdGeom.Xformable(sphere.GetPrim()), np.asarray(point, dtype=float))


def set_debug_sphere(stage: Any, prim_path: str, point: np.ndarray, color: Any, radius: float) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and prim.IsA(UsdGeom.Sphere):
        sphere = UsdGeom.Sphere(prim)
    else:
        sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([color])
    set_debug_translation(UsdGeom.Xformable(sphere.GetPrim()), np.asarray(point, dtype=float))


def draw_debug_vector(stage: Any, prim_path: str, origin: np.ndarray, vector: np.ndarray | None, color: Any, length: float) -> None:
    if vector is None:
        return
    start = np.asarray(origin, dtype=float)
    end = start + np.asarray(vector, dtype=float) * float(length)
    draw_debug_curve(stage, prim_path, [start, end], color, width=0.006)


def draw_control_polygon(stage: Any, prim_prefix: str, points: np.ndarray | None, color: Any, radius: float, width: float, dashed: bool) -> None:
    if points is None or len(points) == 0:
        return
    pts = [np.asarray(point, dtype=float) for point in points]
    for idx, point in enumerate(pts):
        draw_debug_sphere(stage, f"{prim_prefix}/Point_{idx:03d}", point, color, radius)
    if len(pts) < 2:
        return
    if dashed:
        for idx in range(len(pts) - 1):
            if idx % 2 == 0:
                draw_debug_curve(stage, f"{prim_prefix}/Dash_{idx:03d}", [pts[idx], pts[idx + 1]], color, width)
    else:
        draw_debug_curve(stage, f"{prim_prefix}/Polyline", pts, color, width)


def update_debug_curve(stage: Any, prim_path: str, points: np.ndarray, color: Any, width: float) -> None:
    from pxr import Gf, UsdGeom

    if len(points) < 2:
        return
    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(*np.asarray(point, dtype=float)) for point in points])
    curve.CreateWidthsAttr([float(width)])
    curve.CreateDisplayColorAttr([color])


def trajectory_q_by_name(waypoint: np.ndarray, joint_names: list[str]) -> dict[str, float]:
    return {name: float(waypoint[idx]) for idx, name in enumerate(joint_names)}


def update_reference_ghost_skeleton(
    stage: Any,
    kinematics: "ReplayURDFKinematics",
    waypoint: np.ndarray,
    joint_names: list[str],
    visible: bool,
) -> None:
    from pxr import Gf

    root_path = "/World/ReplayDebug/ReferenceGhostSkeleton"
    ensure_xform(stage, root_path)
    set_prim_visibility(stage, root_path, visible)
    if not visible:
        return

    points = kinematics.chain_points(trajectory_q_by_name(waypoint, joint_names))
    update_debug_curve(
        stage,
        f"{root_path}/Arm",
        points,
        Gf.Vec3f(1.0, 0.48, 0.05),
        width=0.012,
    )
    for idx, point in enumerate(points):
        set_debug_sphere(
            stage,
            f"{root_path}/Joint_{idx:02d}",
            point,
            Gf.Vec3f(1.0, 0.62, 0.16),
            radius=0.012 if idx in {0, len(points) - 1} else 0.008,
        )


def ensure_reference_visual_mesh(stage: Any, kinematics: "ReplayURDFKinematics", opacity: float) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    root_path = "/World/ReplayDebug/ReferenceVisualMesh"
    ensure_xform(stage, root_path)

    material_path = f"{root_path}/GhostMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.5, 0.08))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(max(0.0, min(1.0, float(opacity))))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(root_path)).Bind(material)

    mesh_count = 0
    for visual in kinematics.visual_links.values():
        if not visual.mesh_path.is_file():
            log(f"[weldRobot] WARNING: visual mesh does not exist for {visual.link_name}: {visual.mesh_path}")
            continue
        link_path = f"{root_path}/{visual.link_name}"
        points, face_counts, face_indices = load_stl_mesh(visual.mesh_path)
        scale = visual.mesh_scale.astype(float)
        scaled_points = [
            (point[0] * scale[0], point[1] * scale[1], point[2] * scale[2])
            for point in points
        ]
        if not scaled_points:
            continue
        min_point = tuple(min(point[axis] for point in scaled_points) for axis in range(3))
        max_point = tuple(max(point[axis] for point in scaled_points) for axis in range(3))
        mesh = UsdGeom.Mesh.Define(stage, link_path)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in scaled_points])
        mesh.CreateFaceVertexCountsAttr(face_counts)
        mesh.CreateFaceVertexIndicesAttr(face_indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateExtentAttr([Gf.Vec3f(*min_point), Gf.Vec3f(*max_point)])
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.5, 0.08)])
        mesh.CreateDisplayOpacityAttr([max(0.0, min(1.0, float(opacity)))])
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
        mesh_count += 1
    log(f"[weldRobot] Reference visual ghost meshes ready: {mesh_count}")


def update_reference_visual_mesh(
    stage: Any,
    kinematics: "ReplayURDFKinematics",
    waypoint: np.ndarray,
    joint_names: list[str],
    visible: bool,
    offset: np.ndarray,
) -> None:
    from pxr import UsdGeom

    root_path = "/World/ReplayDebug/ReferenceVisualMesh"
    ensure_xform(stage, root_path)
    set_prim_visibility(stage, root_path, visible)
    if not visible:
        return

    offset_tf = np.eye(4)
    offset_tf[:3, 3] = np.asarray(offset, dtype=float).reshape(3)
    link_transforms = kinematics.link_transforms(trajectory_q_by_name(waypoint, joint_names))
    for link_name, visual in kinematics.visual_links.items():
        if link_name not in link_transforms:
            continue
        prim = stage.GetPrimAtPath(f"{root_path}/{link_name}")
        if not prim.IsValid():
            continue
        world_tf = offset_tf @ link_transforms[link_name] @ visual.visual_origin
        set_transform_matrix(UsdGeom.Xformable(prim), numpy_to_gf_matrix(world_tf))


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def urdf_transform_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = rpy_matrix(rpy)
    tf[:3, 3] = np.asarray(xyz, dtype=float)
    return tf


@dataclass
class ReplayChainJoint:
    name: str
    child_link: str
    joint_type: str
    axis: np.ndarray
    origin: np.ndarray


@dataclass
class ReplayVisualLink:
    link_name: str
    mesh_path: Path
    visual_origin: np.ndarray
    mesh_scale: np.ndarray


class ReplayURDFKinematics:
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

        self.base_link = base_link
        self.visual_links = self._parse_visual_links(root, urdf_path)
        self.chain: list[ReplayChainJoint] = []
        self.active_names: list[str] = []
        for joint in chain_xml:
            origin_xml = joint.find("origin")
            xyz = np.fromstring(origin_xml.attrib.get("xyz", "0 0 0"), sep=" ") if origin_xml is not None else np.zeros(3)
            rpy = np.fromstring(origin_xml.attrib.get("rpy", "0 0 0"), sep=" ") if origin_xml is not None else np.zeros(3)
            axis_xml = joint.find("axis")
            axis = np.fromstring(axis_xml.attrib.get("xyz", "0 0 1"), sep=" ") if axis_xml is not None else np.array([0.0, 0.0, 1.0])
            child_xml = joint.find("child")
            item = ReplayChainJoint(
                name=joint.attrib["name"],
                child_link=child_xml.attrib["link"] if child_xml is not None else "",
                joint_type=joint.attrib.get("type", "fixed"),
                axis=axis.astype(float),
                origin=urdf_transform_matrix(xyz.astype(float), rpy.astype(float)),
            )
            self.chain.append(item)
            if item.joint_type in {"revolute", "continuous", "prismatic"}:
                self.active_names.append(item.name)

    @staticmethod
    def _parse_visual_links(root: ET.Element, urdf_path: Path) -> dict[str, ReplayVisualLink]:
        visual_links: dict[str, ReplayVisualLink] = {}
        for link in root.findall("link"):
            link_name = link.attrib.get("name")
            collision = link.find("collision")
            if not link_name or collision is None:
                continue
            mesh = collision.find("./geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.attrib.get("filename")
            if not filename:
                continue
            mesh_path = Path(filename)
            if not mesh_path.is_absolute():
                mesh_path = (urdf_path.parent / mesh_path).resolve()
            origin_xml = collision.find("origin")
            xyz = np.fromstring(origin_xml.attrib.get("xyz", "0 0 0"), sep=" ") if origin_xml is not None else np.zeros(3)
            rpy = np.fromstring(origin_xml.attrib.get("rpy", "0 0 0"), sep=" ") if origin_xml is not None else np.zeros(3)
            scale = np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ")
            if scale.size != 3:
                scale = np.ones(3, dtype=float)
            visual_links[link_name] = ReplayVisualLink(
                link_name=link_name,
                mesh_path=mesh_path,
                visual_origin=urdf_transform_matrix(xyz.astype(float), rpy.astype(float)),
                mesh_scale=scale.astype(float),
            )
        return visual_links

    def link_transforms(self, q_by_name: dict[str, float]) -> dict[str, np.ndarray]:
        tf = np.eye(4)
        transforms = {self.base_link: tf.copy()}
        for joint in self.chain:
            tf = tf @ joint.origin
            if joint.name in q_by_name:
                motion = np.eye(4)
                if joint.joint_type in {"revolute", "continuous"}:
                    motion[:3, :3] = axis_angle_matrix(joint.axis, float(q_by_name[joint.name]))
                elif joint.joint_type == "prismatic":
                    motion[:3, 3] = joint.axis * float(q_by_name[joint.name])
                tf = tf @ motion
            if joint.child_link:
                transforms[joint.child_link] = tf.copy()
        return transforms

    def forward(self, q_by_name: dict[str, float]) -> np.ndarray:
        tf = np.eye(4)
        for joint in self.chain:
            tf = tf @ joint.origin
            if joint.name not in q_by_name:
                continue
            motion = np.eye(4)
            if joint.joint_type in {"revolute", "continuous"}:
                motion[:3, :3] = axis_angle_matrix(joint.axis, float(q_by_name[joint.name]))
            elif joint.joint_type == "prismatic":
                motion[:3, 3] = joint.axis * float(q_by_name[joint.name])
            tf = tf @ motion
        return tf

    def chain_points(self, q_by_name: dict[str, float]) -> np.ndarray:
        tf = np.eye(4)
        points = [tf[:3, 3].copy()]
        for joint in self.chain:
            tf = tf @ joint.origin
            if joint.name in q_by_name:
                motion = np.eye(4)
                if joint.joint_type in {"revolute", "continuous"}:
                    motion[:3, :3] = axis_angle_matrix(joint.axis, float(q_by_name[joint.name]))
                elif joint.joint_type == "prismatic":
                    motion[:3, 3] = joint.axis * float(q_by_name[joint.name])
                tf = tf @ motion
            points.append(tf[:3, 3].copy())
        return np.asarray(points, dtype=float)


def joint_names_for_trajectory(joint_names: list[str] | None, width: int) -> list[str]:
    if joint_names is not None:
        return list(joint_names)
    if width == len(DEFAULT_TRAJECTORY_JOINT_NAMES):
        return list(DEFAULT_TRAJECTORY_JOINT_NAMES)
    raise RuntimeError(
        f"Cannot compute TCP path for trajectory width={width} without joint names. "
        "Provide --joint-names or use a 6-joint trajectory in the default UR5e order."
    )


def compute_tcp_path(
    kinematics: ReplayURDFKinematics,
    trajectory: np.ndarray,
    joint_names: list[str] | None,
) -> np.ndarray:
    names = joint_names_for_trajectory(joint_names, trajectory.shape[1])
    missing = [name for name in kinematics.active_names if name not in names]
    if missing:
        raise RuntimeError(f"Trajectory is missing joint(s) needed for TCP FK: {missing}")
    tcp_points = []
    for waypoint in np.asarray(trajectory, dtype=float):
        q_by_name = {name: float(waypoint[idx]) for idx, name in enumerate(names)}
        tcp_points.append(kinematics.forward(q_by_name)[:3, 3])
    return np.asarray(tcp_points, dtype=float)


def endpoint_vector_from_path(path: np.ndarray, at_start: bool) -> np.ndarray | None:
    if path is None or len(path) < 2:
        return None
    vec = path[1] - path[0] if at_start else path[-1] - path[-2]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return None
    return vec / norm


def merged_overlay(
    overlay: dict[str, Any] | None,
    pred_tcp_path: np.ndarray,
) -> dict[str, Any]:
    merged = dict(overlay or {})
    if pred_tcp_path is not None and len(pred_tcp_path) > 0:
        merged["start_position"] = merged.get("start_position")
        if merged["start_position"] is None:
            merged["start_position"] = pred_tcp_path[0]
        merged["end_position"] = merged.get("end_position")
        if merged["end_position"] is None:
            merged["end_position"] = pred_tcp_path[-1]
        if merged.get("start_vector") is None:
            merged["start_vector"] = endpoint_vector_from_path(pred_tcp_path, at_start=True)
        if merged.get("end_vector") is None:
            merged["end_vector"] = endpoint_vector_from_path(pred_tcp_path, at_start=False)
    return merged


def draw_tcp_path_overlay(
    stage: Any,
    tcp_path: np.ndarray | None,
    prim_prefix: str,
    color: Any,
    marker_radius: float,
    width: float,
) -> None:
    if tcp_path is None or len(tcp_path) == 0:
        return
    points = [np.asarray(point, dtype=float) for point in tcp_path]
    draw_debug_curve(stage, f"{prim_prefix}/Path", points, color, width)
    draw_debug_sphere(stage, f"{prim_prefix}/Start", points[0], color, marker_radius)
    draw_debug_sphere(stage, f"{prim_prefix}/End", points[-1], color, marker_radius)


def draw_replay_visualization_overlay(
    stage: Any,
    overlay: dict[str, Any] | None,
    args: argparse.Namespace,
    pred_tcp_path: np.ndarray | None = None,
    reference_tcp_path: np.ndarray | None = None,
) -> None:
    overlay = merged_overlay(overlay, pred_tcp_path)
    if overlay.get("start_position") is None or overlay.get("end_position") is None:
        return
    from pxr import Gf

    try:
        stage.RemovePrim("/World/ReplayDebug")
    except Exception:
        pass
    ensure_xform(stage, "/World/ReplayDebug")
    ensure_xform(stage, "/World/ReplayDebug/PredControl")
    ensure_xform(stage, "/World/ReplayDebug/GtControl")
    ensure_xform(stage, "/World/ReplayDebug/PredTcp")
    ensure_xform(stage, "/World/ReplayDebug/ReferenceTcp")

    start_position = np.asarray(overlay["start_position"], dtype=float)
    end_position = np.asarray(overlay["end_position"], dtype=float)
    draw_debug_sphere(stage, "/World/ReplayDebug/Start", start_position, Gf.Vec3f(0.1, 1.0, 0.1), radius=0.01)
    draw_debug_sphere(stage, "/World/ReplayDebug/End", end_position, Gf.Vec3f(1.0, 0.1, 0.1), radius=0.01)
    draw_debug_vector(
        stage,
        "/World/ReplayDebug/StartVector",
        start_position,
        overlay.get("start_vector"),
        Gf.Vec3f(0.0, 1.0, 0.0),
        args.debug_vector_length,
    )
    draw_debug_vector(
        stage,
        "/World/ReplayDebug/EndVector",
        end_position,
        overlay.get("end_vector"),
        Gf.Vec3f(0.0, 1.0, 0.0),
        args.debug_vector_length,
    )
    draw_control_polygon(
        stage,
        "/World/ReplayDebug/PredControl",
        overlay.get("pred_control_points"),
        Gf.Vec3f(1.0, 0.55, 0.05),
        args.control_point_radius,
        args.control_polyline_width,
        dashed=True,
    )
    draw_control_polygon(
        stage,
        "/World/ReplayDebug/GtControl",
        overlay.get("gt_control_points"),
        Gf.Vec3f(0.65, 0.25, 1.0),
        args.control_point_radius,
        args.control_polyline_width,
        dashed=False,
    )
    draw_tcp_path_overlay(
        stage,
        pred_tcp_path,
        "/World/ReplayDebug/PredTcp",
        Gf.Vec3f(0.0, 0.35, 1.0),
        marker_radius=0.006,
        width=0.005,
    )
    draw_tcp_path_overlay(
        stage,
        reference_tcp_path,
        "/World/ReplayDebug/ReferenceTcp",
        Gf.Vec3f(1.0, 0.48, 0.05),
        marker_radius=0.006,
        width=0.0035,
    )


def apply_joint_waypoint(robot: Any, dof_indices: list[int], waypoint: np.ndarray) -> None:
    if waypoint.shape != (len(dof_indices),):
        raise RuntimeError(
            f"Expected waypoint shape {(len(dof_indices),)}, got {waypoint.shape}"
        )
    full = read_full_joint_positions(robot)
    for local_idx, dof_idx in enumerate(dof_indices):
        full[dof_idx] = float(waypoint[local_idx])
    robot.set_joint_positions(full)
    try:
        velocities = robot.get_joint_velocities()
        if velocities is not None:
            velocities = np.asarray(velocities, dtype=float).reshape(-1)
            if velocities.size == full.size:
                velocities[:] = 0.0
                robot.set_joint_velocities(velocities)
    except Exception:
        pass


def rotate_point_about_world_z(point: Any, center: Any, angle_deg: float) -> tuple[float, float, float]:
    point_arr = np.asarray(point, dtype=float).reshape(3)
    center_arr = np.asarray(center, dtype=float).reshape(3)
    relative = point_arr - center_arr
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    rotated = np.array(
        [
            c * relative[0] - s * relative[1],
            s * relative[0] + c * relative[1],
            relative[2],
        ],
        dtype=float,
    )
    result = center_arr + rotated
    return (float(result[0]), float(result[1]), float(result[2]))


def add_recording_camera(rep: Any, args: argparse.Namespace):
    target = tuple(float(value) for value in args.camera_target)
    eye = rotate_point_about_world_z(args.camera_eye, target, args.camera_z_rotation_deg)
    target = tuple(float(value) for value in args.camera_target)
    log(
        f"[weldRobot] Recording camera eye={eye}, look_at={target}, "
        f"z_rotation_deg={float(args.camera_z_rotation_deg):.1f}"
    )
    camera = rep.create.camera(
        position=eye,
        look_at=target,
        focal_length=28.0,
        focus_distance=3.0,
        clipping_range=(0.01, 1000.0),
    )
    return rep.create.render_product(camera, resolution=(args.width, args.height))


def render_replay_frame(world: Any, render_only_replay: bool) -> None:
    if render_only_replay:
        render = getattr(world, "render", None)
        if callable(render):
            render()
            return
    world.step(render=True)


def main() -> None:
    args = parse_args()
    apply_pred_check_defaults(args)
    if args.hold_steps < 1:
        raise RuntimeError("--hold-steps must be >= 1")
    if args.waypoint_substeps < 1:
        raise RuntimeError("--waypoint-substeps must be >= 1")
    if args.segment_gap_steps < 0:
        raise RuntimeError("--segment-gap-steps must be >= 0")
    if args.loop < 1:
        raise RuntimeError("--loop must be >= 1")
    if args.encode_only:
        args.output = args.output.expanduser().resolve()
        if args.frames_dir is None:
            args.frames_dir = args.output.parent / f"{args.output.stem}_frames"
        frames_dir = args.frames_dir.expanduser().resolve()
        if not frames_dir.is_dir():
            raise RuntimeError(f"--encode-only requires an existing frame directory: {frames_dir}")
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists. Use --overwrite to replace it: {args.output}")
        encode_video(frames_dir, args.output, args.fps, args.ffmpeg)
        log(f"[weldRobot] Video saved to: {args.output}")
        return
    if args.trajectory is None:
        raise RuntimeError("--trajectory is required unless --encode-only is used.")

    trajectory_path = args.trajectory.expanduser().resolve()
    trajectory, requested_joint_names = load_joint_trajectory(trajectory_path, args.joint_names)
    trajectory_representation = resolve_trajectory_representation(trajectory_path, args.trajectory_representation)
    reference_trajectory = None
    reference_joint_names = None
    reference_trajectory_representation = args.reference_representation
    reference_path = args.reference_trajectory or args.reference_npz
    if reference_path is not None:
        reference_npz_path = reference_path.expanduser().resolve()
        reference_trajectory, reference_joint_names = load_joint_trajectory(
            reference_npz_path,
            args.joint_names,
            preferred_npz_key=args.reference_key,
        )
        reference_trajectory_representation = resolve_trajectory_representation(
            reference_npz_path, args.reference_representation
        )
    visualization_overlay = load_replay_visualization_overlay(args)
    frames_dir = prepare_recording_paths(args) if args.record else None

    if args.stl is not None:
        args.stl = args.stl.expanduser().resolve()
        if not args.stl.is_file():
            raise FileNotFoundError(f"STL file does not exist: {args.stl}")

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

        rendering_dt = 1.0 / args.fps if args.record else args.rendering_dt
        world = World(physics_dt=args.physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        resolved_urdf = make_resolved_urdf(args.urdf)
        stage = get_context().get_stage()
        ensure_xform(stage, "/World")
        robot_prim_path = import_robot_instance(
            stage=stage,
            resolved_urdf=resolved_urdf,
            requested_prim_path=args.robot_prim_path,
            fix_base=not args.floating,
        )
        set_prim_visibility(stage, robot_prim_path, True)
        reference_robot_prim_path = None
        use_reference_articulation = (
            reference_trajectory is not None
            and args.playback_mode == "overlay"
            and args.overlay_reference_visual == "articulation"
        )
        if use_reference_articulation:
            reference_world_offset = (
                np.asarray(args.reference_offset, dtype=float)
                + np.asarray(args.overlay_visual_offset, dtype=float)
            )
            reference_robot_prim_path = import_robot_instance(
                stage=stage,
                resolved_urdf=resolved_urdf,
                requested_prim_path=args.reference_robot_prim_path,
                fix_base=not args.floating,
                world_offset=tuple(float(v) for v in reference_world_offset),
            )
            apply_reference_ghost_material(stage, reference_robot_prim_path, args.reference_ghost_opacity)
            disable_collisions_under_prim(stage, reference_robot_prim_path)
            set_prim_visibility(stage, reference_robot_prim_path, True)
        world.reset()
        set_initial_joint_positions(robot_prim_path)
        if reference_robot_prim_path is not None:
            set_initial_joint_positions(reference_robot_prim_path)

        if args.stl is not None:
            import_stl_as_mesh(
                stage=stage,
                stl_path=args.stl,
                prim_path=args.workpiece_prim_path,
                scale=args.workpiece_scale,
                z_offset=args.workpiece_z_offset,
                local_offset=tuple(float(v) for v in args.workpiece_offset),
                debug_box=args.debug_workpiece_box,
            )

        world.step(render=False)
        set_initial_joint_positions(robot_prim_path)
        if reference_robot_prim_path is not None:
            set_initial_joint_positions(reference_robot_prim_path)

        robot = create_articulation(robot_prim_path)
        joint_names, dof_indices = resolve_dof_indices(robot, requested_joint_names, trajectory.shape[1])
        start_joint_positions = resolve_start_joint_positions(
            robot=robot,
            dof_indices=dof_indices,
            cli_start_joint_positions=args.start_joint_positions,
            start_npz=args.start_npz,
            start_key=args.start_key,
        )
        trajectory = build_absolute_trajectory(
            trajectory=trajectory,
            representation=trajectory_representation,
            start_joint_positions=start_joint_positions,
        )
        trajectory = densify_trajectory(trajectory, args.waypoint_substeps)
        reference_robot = None
        reference_dof_indices = None
        resolved_reference_joint_names = None
        if reference_trajectory is not None:
            if reference_robot_prim_path is not None:
                reference_robot = create_articulation(reference_robot_prim_path)
                resolved_reference_joint_names, reference_dof_indices = resolve_dof_indices(
                    reference_robot,
                    reference_joint_names or args.joint_names,
                    reference_trajectory.shape[1],
                )
                reference_start_joint_positions = resolve_start_joint_positions(
                    robot=reference_robot,
                    dof_indices=reference_dof_indices,
                    cli_start_joint_positions=args.reference_start_joint_positions,
                    start_npz=args.reference_start_npz,
                    start_key=args.reference_start_key,
                )
            else:
                resolved_reference_joint_names = joint_names
                reference_dof_indices = dof_indices
                reference_start_joint_positions = resolve_start_joint_positions(
                    robot=robot,
                    dof_indices=dof_indices,
                    cli_start_joint_positions=args.reference_start_joint_positions,
                    start_npz=args.reference_start_npz,
                    start_key=args.reference_start_key,
                )
            reference_trajectory = build_absolute_trajectory(
                trajectory=reference_trajectory,
                representation=reference_trajectory_representation,
                start_joint_positions=reference_start_joint_positions,
            )
            reference_trajectory = densify_trajectory(reference_trajectory, args.waypoint_substeps)

        replay_kinematics = ReplayURDFKinematics(resolved_urdf)
        pred_tcp_path = compute_tcp_path(replay_kinematics, trajectory, joint_names)
        reference_tcp_path = None
        if reference_trajectory is not None:
            reference_tcp_path = compute_tcp_path(
                replay_kinematics,
                reference_trajectory,
                resolved_reference_joint_names or reference_joint_names or args.joint_names,
            )
        draw_replay_visualization_overlay(
            stage,
            visualization_overlay,
            args,
            pred_tcp_path=pred_tcp_path,
            reference_tcp_path=reference_tcp_path,
        )
        if (
            reference_trajectory is not None
            and args.playback_mode == "overlay"
            and args.overlay_reference_visual == "visual_mesh"
        ):
            ensure_reference_visual_mesh(stage, replay_kinematics, args.reference_ghost_opacity)
        log(f"[weldRobot] Predicted TCP path points: {len(pred_tcp_path)}")
        if reference_tcp_path is not None:
            log(f"[weldRobot] Reference TCP path points: {len(reference_tcp_path)}")

        writer = None
        if args.record:
            try:
                rep.orchestrator.set_capture_on_play(False)
            except Exception:
                pass
            render_product = add_recording_camera(rep, args)
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=str(frames_dir), rgb=True)
            writer.attach([render_product])
        elif not args.headless:
            add_camera_view()

        playback_segments: list[tuple[str, np.ndarray]] = [("pred", trajectory)]
        if reference_trajectory is not None:
            if args.playback_mode == "overlay":
                total_waypoints = max(trajectory.shape[0], reference_trajectory.shape[0])
            else:
                total_waypoints = trajectory.shape[0] + reference_trajectory.shape[0]
                playback_segments.append(("gt", reference_trajectory))
        else:
            total_waypoints = trajectory.shape[0]

        if reference_trajectory is not None and args.playback_mode == "overlay":
            total_frames = total_waypoints * args.hold_steps * args.loop
        else:
            segment_gap_total = args.segment_gap_steps * max(len(playback_segments) - 1, 0)
            total_frames = (total_waypoints * args.hold_steps + segment_gap_total) * args.loop
        log(
            f"[weldRobot] Replay ready: pred_waypoints={trajectory.shape[0]}, hold_steps={args.hold_steps}, "
            f"waypoint_substeps={args.waypoint_substeps}, loop={args.loop}, total_steps={total_frames}, "
            f"joints={joint_names}, representation={trajectory_representation}, playback_mode={args.playback_mode}, "
            f"render_only_replay={args.render_only_replay}"
        )
        if trajectory_representation == "delta":
            log(f"[weldRobot] Delta trajectory start joint positions: {start_joint_positions.tolist()}")
        if reference_trajectory is not None:
            log(
                f"[weldRobot] Reference replay ready: waypoints={reference_trajectory.shape[0]}, "
                f"representation={reference_trajectory_representation}, offset={tuple(args.reference_offset)}"
            )
            if (
                args.playback_mode == "overlay"
                and args.overlay_reference_visual == "articulation"
                and np.allclose(np.asarray(args.reference_offset, dtype=float), 0.0)
            ):
                log(
                    "[weldRobot] Overlapped dual-robot replay: ghost collisions disabled. "
                    "Without this, overlapping collision bodies can cause visible twitching."
                )
            if args.playback_mode == "overlay":
                log(
                    f"[weldRobot] Overlay finish behavior={args.overlay_finish_behavior}. "
                    "Independent trajectory completion avoids forcing both robots to finish together."
                )
                log(
                    f"[weldRobot] Overlay reference visual={args.overlay_reference_visual}, "
                    f"visual offset={tuple(float(v) for v in args.overlay_visual_offset)}"
                )
            if args.playback_mode == "sequential":
                log(
                    "[weldRobot] Sequential replay uses one robot and interpolated waypoints. "
                    "This avoids dual-robot overlap and reduces large per-step joint jumps."
                )
        if args.stl is not None:
            log(
                f"[weldRobot] Imported STL {args.stl.name} with offset={tuple(args.workpiece_offset)} "
                f"scale={args.workpiece_scale} z_offset={args.workpiece_z_offset}"
            )

        frame_idx = 0
        for loop_idx in range(args.loop):
            if reference_trajectory is not None and args.playback_mode == "overlay":
                set_prim_visibility(stage, robot_prim_path, True)
                if reference_robot_prim_path is not None:
                    set_prim_visibility(stage, reference_robot_prim_path, True)
                if args.overlay_reference_visual == "skeleton":
                    set_prim_visibility(stage, "/World/ReplayDebug/ReferenceVisualMesh", False)
                    update_reference_ghost_skeleton(
                        stage,
                        replay_kinematics,
                        reference_trajectory[0],
                        resolved_reference_joint_names or joint_names,
                        visible=True,
                    )
                elif args.overlay_reference_visual == "visual_mesh":
                    set_prim_visibility(stage, "/World/ReplayDebug/ReferenceGhostSkeleton", False)
                    update_reference_visual_mesh(
                        stage,
                        replay_kinematics,
                        reference_trajectory[0],
                        resolved_reference_joint_names or joint_names,
                        visible=True,
                        offset=np.asarray(args.reference_offset, dtype=float)
                        + np.asarray(args.overlay_visual_offset, dtype=float),
                    )
                for waypoint_idx in range(total_waypoints):
                    pred_active = waypoint_idx < trajectory.shape[0]
                    gt_active = waypoint_idx < reference_trajectory.shape[0]
                    waypoint = trajectory[min(waypoint_idx, trajectory.shape[0] - 1)]
                    reference_waypoint = reference_trajectory[min(waypoint_idx, reference_trajectory.shape[0] - 1)]
                    for _ in range(args.hold_steps):
                        if pred_active or args.overlay_finish_behavior == "hold":
                            apply_joint_waypoint(robot, dof_indices, waypoint.reshape(-1))
                        if reference_robot is not None and (gt_active or args.overlay_finish_behavior == "hold"):
                            apply_joint_waypoint(reference_robot, reference_dof_indices, reference_waypoint.reshape(-1))
                        if args.overlay_reference_visual == "skeleton":
                            skeleton_visible = gt_active or args.overlay_finish_behavior == "hold"
                            update_reference_ghost_skeleton(
                                stage,
                                replay_kinematics,
                                reference_waypoint.reshape(-1),
                                resolved_reference_joint_names or joint_names,
                                visible=skeleton_visible,
                            )
                        elif args.overlay_reference_visual == "visual_mesh":
                            visual_mesh_visible = gt_active or args.overlay_finish_behavior == "hold"
                            update_reference_visual_mesh(
                                stage,
                                replay_kinematics,
                                reference_waypoint.reshape(-1),
                                resolved_reference_joint_names or joint_names,
                                visible=visual_mesh_visible,
                                offset=np.asarray(args.reference_offset, dtype=float)
                                + np.asarray(args.overlay_visual_offset, dtype=float),
                            )
                        if not pred_active and args.overlay_finish_behavior == "hide":
                            set_prim_visibility(stage, robot_prim_path, False)
                        if reference_robot is not None and not gt_active and args.overlay_finish_behavior == "hide":
                            set_prim_visibility(stage, reference_robot_prim_path, False)
                        render_replay_frame(world, args.render_only_replay)
                        if args.record:
                            step_replicator(rep, args)
                        frame_idx += 1
                        if args.record and frame_idx % max(args.fps, 1) == 0:
                            log(f"[weldRobot] Recorded {frame_idx}/{total_frames} frames")
                    if not args.record and (waypoint_idx + 1) % 50 == 0:
                        log(
                            f"[weldRobot] Replayed {waypoint_idx + 1}/{total_waypoints} overlay waypoints "
                            f"in loop {loop_idx + 1}/{args.loop}"
                        )
            else:
                sequential_index = 0
                for segment_idx, (segment_name, segment_traj) in enumerate(playback_segments):
                    for waypoint in segment_traj:
                        for _ in range(args.hold_steps):
                            apply_joint_waypoint(robot, dof_indices, np.asarray(waypoint, dtype=float).reshape(-1))
                            render_replay_frame(world, args.render_only_replay)
                            if args.record:
                                step_replicator(rep, args)
                            frame_idx += 1
                            if args.record and frame_idx % max(args.fps, 1) == 0:
                                log(f"[weldRobot] Recorded {frame_idx}/{total_frames} frames")
                        sequential_index += 1
                        if not args.record and sequential_index % 50 == 0:
                            log(
                                f"[weldRobot] Replayed {sequential_index}/{total_waypoints} sequential waypoints "
                                f"in loop {loop_idx + 1}/{args.loop}"
                            )
                    if segment_idx < len(playback_segments) - 1:
                        for _ in range(args.segment_gap_steps):
                            apply_joint_waypoint(robot, dof_indices, segment_traj[-1].reshape(-1))
                            render_replay_frame(world, args.render_only_replay)
                            if args.record:
                                step_replicator(rep, args)
                            frame_idx += 1
                            if args.record and frame_idx % max(args.fps, 1) == 0:
                                log(f"[weldRobot] Recorded {frame_idx}/{total_frames} frames")
                        log(f"[weldRobot] Finished sequential segment '{segment_name}', advancing to next segment")

        if args.record and writer is not None:
            rep.orchestrator.wait_until_complete()
            writer.detach()
            png_count = len(list(frames_dir.glob("*.png"))) + len(list(frames_dir.glob("**/*.png")))
            log(f"[weldRobot] PNG files written under {frames_dir}: {png_count}")

    except Exception:
        log("[weldRobot] Fatal error while building or recording the replay scene:")
        traceback.print_exc()
        raise

    finally:
        simulation_app.close()

    if args.record and frames_dir is not None:
        encode_video(frames_dir, args.output, args.fps, args.ffmpeg)
        if not args.keep_frames:
            import shutil

            shutil.rmtree(frames_dir)
        log(f"[weldRobot] Video saved to: {args.output}")


if __name__ == "__main__":
    main()
