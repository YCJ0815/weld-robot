"""Replay a robot joint trajectory with optional STL workpiece import and video capture."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sim_parallel_welding import (  # noqa: E402
    encode_video,
    move_prim_to_path,
    import_stl_as_mesh,
    prepare_recording_paths,
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
    "start_joint_positions",
    "start_joints",
    "q_start",
    "start_q",
    "start",
    "robot_start",
    "joint_start",
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a robot joint-angle trajectory in Isaac Sim with optional STL workpiece import."
    )
    parser.add_argument("--trajectory", type=Path, required=True, help="Joint trajectory file: json/csv/txt/npy/npz.")
    parser.add_argument(
        "--trajectory-representation",
        choices=("auto", "absolute", "delta"),
        default="auto",
        help="Interpret trajectory waypoints as absolute joint angles or joint-angle deltas. "
        "In auto mode, .npy/.npz default to delta and text formats default to absolute.",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="UR5e welding-arm URDF.")
    parser.add_argument("--stl", type=Path, default=None, help="Optional STL workpiece path.")
    parser.add_argument("--robot-prim-path", default="/World/UR5ePen", help="USD prim path for the imported robot.")
    parser.add_argument("--workpiece-prim-path", default="/World/Workpiece", help="USD prim path for the imported STL workpiece.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without the UI.")
    parser.add_argument("--floating", action="store_true", help="Do not fix the robot base to the world.")
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0, help="Physics timestep in seconds.")
    parser.add_argument("--rendering-dt", type=float, default=1.0 / 60.0, help="Rendering timestep in seconds.")
    parser.add_argument("--hold-steps", type=int, default=1, help="Simulation steps to hold each waypoint.")
    parser.add_argument("--loop", type=int, default=1, help="Repeat count for the entire trajectory.")
    parser.add_argument("--fps", type=int, default=30, help="Recording frame rate.")
    parser.add_argument("--width", type=int, default=1280, help="Recording width.")
    parser.add_argument("--height", type=int, default=720, help="Recording height.")
    parser.add_argument("--record", action="store_true", help="Record RGB frames and encode MP4.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/replay_joint_trajectory.mp4", help="MP4 output path.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Directory used for RGB frame images.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep exported RGB frames after MP4 encoding.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing video and frame directory.")
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
        help="Optional NPZ file containing the normalized start joint positions.",
    )
    parser.add_argument(
        "--start-key",
        default=None,
        help="Key inside --start-npz for the normalized start joint positions. "
        "If omitted, common key names are searched automatically.",
    )
    parser.add_argument(
        "--start-normalization",
        choices=("auto", "neg_one_to_one", "zero_to_one"),
        default="auto",
        help="Normalization range used by the start joint positions stored in --start-npz.",
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


def _load_npz_trajectory(path: Path) -> tuple[np.ndarray, list[str] | None]:
    with np.load(path, allow_pickle=True) as data:
        joint_names = _normalize_joint_name_list(data["joint_names"]) if "joint_names" in data else None
        for key in ("joint_positions", "trajectory", "waypoints", "q", "positions"):
            if key in data:
                return _coerce_trajectory_array(data[key], f"NPZ field '{key}'"), joint_names
        if len(data.files) == 1:
            only_key = data.files[0]
            return _coerce_trajectory_array(data[only_key], f"NPZ field '{only_key}'"), joint_names
        raise RuntimeError(
            "NPZ trajectory must contain one of: "
            "'joint_positions', 'trajectory', 'waypoints', 'q', 'positions'."
        )


def load_joint_trajectory(path: Path, cli_joint_names: list[str] | None) -> tuple[np.ndarray, list[str] | None]:
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
        waypoints, file_joint_names = _load_npz_trajectory(path)
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
    if path.suffix.lower() in {".npy", ".npz"}:
        return "delta"
    return "absolute"


def load_urdf_joint_limits(urdf_path: Path, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {urdf_path}")

    root = ET.parse(urdf_path).getroot()
    limits_by_name: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit_xml = joint.find("limit")
        if limit_xml is None:
            continue
        lower = float(limit_xml.attrib.get("lower", "-6.28318530718"))
        upper = float(limit_xml.attrib.get("upper", "6.28318530718"))
        limits_by_name[joint.attrib["name"]] = (lower, upper)

    missing = [name for name in joint_names if name not in limits_by_name]
    if missing:
        raise RuntimeError(f"Missing URDF joint limits for joints: {missing}")

    lower = np.asarray([limits_by_name[name][0] for name in joint_names], dtype=float)
    upper = np.asarray([limits_by_name[name][1] for name in joint_names], dtype=float)
    return lower, upper


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


def resolve_normalization_mode(values: np.ndarray, requested: str) -> str:
    if requested != "auto":
        return requested
    if np.all(values >= -1.000001) and np.all(values <= 1.000001):
        return "neg_one_to_one"
    if np.all(values >= -0.000001) and np.all(values <= 1.000001):
        return "zero_to_one"
    raise RuntimeError(
        "Could not infer start normalization range automatically. "
        "Use --start-normalization neg_one_to_one or --start-normalization zero_to_one."
    )


def denormalize_joint_positions(
    normalized: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    normalization: str,
) -> np.ndarray:
    if normalized.size != lower_limits.size:
        raise RuntimeError(
            f"Normalized start joint count {normalized.size} does not match joint count {lower_limits.size}"
        )
    if normalization == "neg_one_to_one":
        scaled = 0.5 * (normalized + 1.0)
    elif normalization == "zero_to_one":
        scaled = normalized
    else:
        raise RuntimeError(f"Unsupported normalization mode: {normalization}")
    return lower_limits + scaled * (upper_limits - lower_limits)


def current_joint_subset(robot: Any, dof_indices: list[int]) -> np.ndarray:
    full = read_full_joint_positions(robot)
    return np.asarray([full[dof_idx] for dof_idx in dof_indices], dtype=float)


def resolve_start_joint_positions(
    robot: Any,
    dof_indices: list[int],
    joint_names: list[str],
    urdf_path: Path,
    cli_start_joint_positions: list[float] | None,
    start_npz: Path | None,
    start_key: str | None,
    start_normalization: str,
) -> np.ndarray:
    if cli_start_joint_positions is None:
        if start_npz is None:
            return current_joint_subset(robot, dof_indices)
        normalized = load_start_array_from_npz(start_npz, start_key)
        lower_limits, upper_limits = load_urdf_joint_limits(urdf_path, joint_names)
        normalization_mode = resolve_normalization_mode(normalized, start_normalization)
        start = denormalize_joint_positions(normalized, lower_limits, upper_limits, normalization_mode)
        log(
            f"[weldRobot] Loaded normalized start joints from {start_npz.name} "
            f"using normalization={normalization_mode}"
        )
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


def add_recording_camera(rep: Any, args: argparse.Namespace):
    eye = tuple(float(value) for value in args.camera_eye)
    target = tuple(float(value) for value in args.camera_target)
    log(f"[weldRobot] Recording camera eye={eye}, look_at={target}")
    camera = rep.create.camera(
        position=eye,
        look_at=target,
        focal_length=28.0,
        focus_distance=3.0,
        clipping_range=(0.01, 1000.0),
    )
    return rep.create.render_product(camera, resolution=(args.width, args.height))


def main() -> None:
    args = parse_args()
    if args.hold_steps < 1:
        raise RuntimeError("--hold-steps must be >= 1")
    if args.loop < 1:
        raise RuntimeError("--loop must be >= 1")

    trajectory_path = args.trajectory.expanduser().resolve()
    trajectory, requested_joint_names = load_joint_trajectory(trajectory_path, args.joint_names)
    trajectory_representation = resolve_trajectory_representation(trajectory_path, args.trajectory_representation)
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
        imported_robot_path = import_robot_from_urdf(resolved_urdf, args.robot_prim_path, fix_base=not args.floating)
        if not stage.GetPrimAtPath(imported_robot_path).IsValid() and stage.GetPrimAtPath("/ur5e_pen").IsValid():
            imported_robot_path = "/ur5e_pen"
        robot_prim_path = move_prim_to_path(stage, imported_robot_path, args.robot_prim_path)
        world.reset()
        set_initial_joint_positions(robot_prim_path)

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

        robot = create_articulation(robot_prim_path)
        joint_names, dof_indices = resolve_dof_indices(robot, requested_joint_names, trajectory.shape[1])
        lower_limits, upper_limits = load_urdf_joint_limits(args.urdf, joint_names)
        start_joint_positions = resolve_start_joint_positions(
            robot=robot,
            dof_indices=dof_indices,
            joint_names=joint_names,
            urdf_path=args.urdf,
            cli_start_joint_positions=args.start_joint_positions,
            start_npz=args.start_npz,
            start_key=args.start_key,
            start_normalization=args.start_normalization,
        )
        trajectory = build_absolute_trajectory(
            trajectory=trajectory,
            representation=trajectory_representation,
            start_joint_positions=start_joint_positions,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )

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

        total_frames = trajectory.shape[0] * args.hold_steps * args.loop
        log(
            f"[weldRobot] Replay ready: waypoints={trajectory.shape[0]}, hold_steps={args.hold_steps}, "
            f"loop={args.loop}, total_steps={total_frames}, joints={joint_names}, "
            f"representation={trajectory_representation}"
        )
        if trajectory_representation == "delta":
            log(f"[weldRobot] Delta trajectory start joint positions: {start_joint_positions.tolist()}")
            log(
                f"[weldRobot] URDF joint limits: lower={lower_limits.tolist()} upper={upper_limits.tolist()}"
            )
        if args.stl is not None:
            log(
                f"[weldRobot] Imported STL {args.stl.name} with offset={tuple(args.workpiece_offset)} "
                f"scale={args.workpiece_scale} z_offset={args.workpiece_z_offset}"
            )

        frame_idx = 0
        for loop_idx in range(args.loop):
            for waypoint_idx, waypoint in enumerate(trajectory):
                apply_joint_waypoint(robot, dof_indices, waypoint.reshape(-1))
                for _ in range(args.hold_steps):
                    world.step(render=True)
                    if args.record:
                        step_replicator(rep, args)
                    frame_idx += 1
                    if args.record and frame_idx % max(args.fps, 1) == 0:
                        log(f"[weldRobot] Recorded {frame_idx}/{total_frames} frames")
                if not args.record and (waypoint_idx + 1) % 50 == 0:
                    log(
                        f"[weldRobot] Replayed {waypoint_idx + 1}/{trajectory.shape[0]} waypoints "
                        f"in loop {loop_idx + 1}/{args.loop}"
                    )

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
        encode_video(frames_dir, args.output, args.fps)
        if not args.keep_frames:
            import shutil

            shutil.rmtree(frames_dir)
        log(f"[weldRobot] Video saved to: {args.output}")


if __name__ == "__main__":
    main()
