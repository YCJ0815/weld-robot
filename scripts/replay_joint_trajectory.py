"""Replay a robot joint trajectory with optional STL workpiece import and video capture."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sim_parallel_welding import (  # noqa: E402
    encode_video,
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


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a robot joint-angle trajectory in Isaac Sim with optional STL workpiece import."
    )
    parser.add_argument("--trajectory", type=Path, required=True, help="Joint trajectory file: json/csv/txt/npy/npz.")
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

    trajectory, requested_joint_names = load_joint_trajectory(args.trajectory, args.joint_names)
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
        robot_prim_path = import_robot_from_urdf(resolved_urdf, args.robot_prim_path, fix_base=not args.floating)
        world.reset()
        set_initial_joint_positions(robot_prim_path)

        stage = get_context().get_stage()
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
        trajectory = np.asarray(trajectory, dtype=float)

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
            f"loop={args.loop}, total_steps={total_frames}, joints={joint_names}"
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
