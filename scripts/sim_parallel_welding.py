"""Launch parallel UR5e welding-arm scenes from generated job manifests.

The script consumes ``data_generation/data/generated_jobs/manifest.json``.
Each manifest job spawns one robot and one STL workpiece under an isolated
``/World/envs/env_XXX`` namespace.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
DEFAULT_MANIFEST = REPO_ROOT / "data_generation/data/generated_jobs/manifest.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sim_welding_arm import (  # noqa: E402
    DEFAULT_URDF,
    add_camera_view,
    import_robot_from_urdf,
    make_resolved_urdf,
    set_initial_joint_positions,
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import generated welding jobs into a parallel Isaac Sim scene."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Generated jobs manifest.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="UR5e welding-arm URDF.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without the UI.")
    parser.add_argument("--floating", action="store_true", help="Do not fix robot bases to the world.")
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0, help="Physics timestep in seconds.")
    parser.add_argument("--rendering-dt", type=float, default=1.0 / 60.0, help="Rendering timestep in seconds.")
    parser.add_argument("--num-steps", type=int, default=-1, help="Number of simulation steps; -1 runs forever.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Limit jobs imported from the manifest.")
    parser.add_argument("--record", action="store_true", help="Record RGB frames and optionally encode MP4.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/parallel_welding_scene.mp4", help="MP4 output path.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Directory used for RGB frame images.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep exported RGB frames after MP4 encoding.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing video and frame directory.")
    parser.add_argument("--fps", type=int, default=30, help="Recording frame rate.")
    parser.add_argument("--width", type=int, default=1280, help="Recording width.")
    parser.add_argument("--height", type=int, default=720, help="Recording height.")
    parser.add_argument("--rt-subframes", type=int, default=8, help="Replicator render subframes per captured frame.")
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Camera position in world coordinates.",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Camera look-at target in world coordinates.",
    )
    parser.add_argument(
        "--workpiece-scale",
        type=float,
        default=0.001,
        help="Scale applied to STL vertices. Generated data is in mm; Isaac Sim scene is in m.",
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
        default=[0.45, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Local workpiece offset relative to each robot/env origin, in meters. Manifest workpiece_offset overrides this when present.",
    )
    parser.add_argument("--debug-workpiece-box", action="store_true", help="Add a red debug cube at each workpiece center.")
    return parser.parse_args()


def prepare_recording_paths(args: argparse.Namespace) -> Path:
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.frames_dir is None:
        args.frames_dir = args.output.parent / f"{args.output.stem}_frames"
    args.frames_dir = args.frames_dir.resolve()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite to replace it: {args.output}")
    if args.frames_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Frames directory already exists. Use --overwrite to replace it: {args.frames_dir}")
        shutil.rmtree(args.frames_dir)

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    return args.frames_dir


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found. RGB frames were written, but MP4 encoding cannot run. "
            f"Inspect frames in: {frames_dir}"
        )

    frame_candidates = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("**/*.png"))
    if not frame_candidates:
        raise RuntimeError(f"No PNG frames were written in: {frames_dir}")

    if not sorted(frames_dir.glob("*.png")):
        raise RuntimeError(
            "PNG frames were written in nested Replicator directories, but MP4 encoding expects them directly under "
            f"{frames_dir}. Inspect nested files or set --frames-dir to a clean directory and rerun."
        )

    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(frames_dir / "*.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def resolve_job_path(manifest_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def load_manifest(manifest_path: Path, max_jobs: int | None) -> list[dict[str, Any]]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest does not exist: {manifest_path}\n"
            "Generate it first, for example:\n"
            "  python data_generation/src/main.py --count 4 "
            "--jobs-dir data_generation/data/generated_jobs"
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    raw_jobs = manifest.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise RuntimeError(f"Manifest has no jobs: {manifest_path}")

    manifest_dir = manifest_path.parent
    jobs: list[dict[str, Any]] = []
    for index, raw_job in enumerate(raw_jobs[:max_jobs]):
        if not isinstance(raw_job, dict):
            raise RuntimeError(f"Invalid job at index {index}: {raw_job!r}")

        job = dict(raw_job)
        job.setdefault("id", f"job_{index:03d}")
        workpiece_value = job.get("stl_asset") or job.get("workpiece_asset")
        path_value = job.get("path_json")
        if not workpiece_value:
            raise RuntimeError(f"Job {job['id']} has no STL/workpiece asset.")
        if not path_value:
            raise RuntimeError(f"Job {job['id']} has no path_json.")

        job["workpiece_asset_path"] = resolve_job_path(manifest_dir, workpiece_value)
        job["path_json_path"] = resolve_job_path(manifest_dir, path_value)
        if not job["workpiece_asset_path"].is_file():
            raise FileNotFoundError(f"Missing workpiece for {job['id']}: {job['workpiece_asset_path']}")
        if not job["path_json_path"].is_file():
            raise FileNotFoundError(f"Missing path JSON for {job['id']}: {job['path_json_path']}")
        jobs.append(job)
    return jobs


def ensure_xform(stage: Any, prim_path: str):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    return prim


def set_xform_translation(prim: Any, translation: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))


def add_scene_lighting(stage: Any) -> None:
    from pxr import Gf, UsdLux

    ensure_xform(stage, "/World/Lights")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(450.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    distant.CreateIntensityAttr(2500.0)
    distant.CreateAngleAttr(0.35)
    distant.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))


def parse_binary_stl(data: bytes) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    if len(data) < 84:
        raise RuntimeError("Binary STL is too small.")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if expected_size > len(data):
        raise RuntimeError("Binary STL size does not match its triangle count.")

    points: list[tuple[float, float, float]] = []
    face_counts: list[int] = []
    face_indices: list[int] = []
    offset = 84
    for _ in range(triangle_count):
        offset += 12
        for _vertex in range(3):
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
    face_counts = [3] * (len(vertices) // 3)
    face_indices = list(range(len(vertices)))
    return vertices, face_counts, face_indices


def load_stl_mesh(stl_path: Path) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    data = stl_path.read_bytes()
    triangle_count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + triangle_count * 50:
        return parse_binary_stl(data)

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return parse_binary_stl(data)
    if text.lstrip().lower().startswith("solid"):
        return parse_ascii_stl(text)
    return parse_binary_stl(data)


def add_debug_axes(stage: Any, prim_path: str, center: tuple[float, float, float], size: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    axis_length = max(max(size), 0.05) * 0.6
    thickness = max(axis_length * 0.03, 0.005)
    axes = (
        ("X", Gf.Vec3f(1.0, 0.05, 0.02), (axis_length, thickness, thickness), (axis_length / 2, 0.0, 0.0)),
        ("Y", Gf.Vec3f(0.05, 0.8, 0.05), (thickness, axis_length, thickness), (0.0, axis_length / 2, 0.0)),
        ("Z", Gf.Vec3f(0.05, 0.2, 1.0), (thickness, thickness, axis_length), (0.0, 0.0, axis_length / 2)),
    )
    for axis_name, color, scale, offset in axes:
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}_{axis_name}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([color])
        xformable = UsdGeom.Xformable(cube.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(
            Gf.Vec3d(center[0] + offset[0], center[1] + offset[1], center[2] + offset[2])
        )
        xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def import_stl_as_mesh(
    stage: Any,
    stl_path: Path,
    prim_path: str,
    scale: float,
    z_offset: float,
    local_offset: tuple[float, float, float],
    debug_box: bool,
) -> str:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    points, face_counts, face_indices = load_stl_mesh(stl_path)
    scaled_points = [
        (point[0] * scale, point[1] * scale, point[2] * scale + z_offset)
        for point in points
    ]
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
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.78, 0.62, 0.38)])
    xformable = UsdGeom.Xformable(mesh.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*local_offset))

    material_path = f"{prim_path}_Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.78, 0.62, 0.38))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

    if debug_box:
        center = tuple(local_offset[axis] + (min_point[axis] + max_point[axis]) / 2.0 for axis in range(3))
        add_debug_axes(stage, f"{prim_path}_DebugAxes", center, size)

    log(
        f"[weldRobot] STL bounds for {stl_path.name}: "
        f"local_min={min_point}, local_max={max_point}, size_m={size}, "
        f"scale={scale}, z_offset={z_offset}, local_offset={local_offset}"
    )
    return prim_path


def workpiece_local_offset(job: dict[str, Any], default_offset: list[float]) -> tuple[float, float, float]:
    offset = job.get("workpiece_offset", default_offset)
    if not (isinstance(offset, list) and len(offset) == 3):
        raise RuntimeError(f"Invalid workpiece_offset for {job['id']}: {offset!r}")
    return (float(offset[0]), float(offset[1]), float(offset[2]))


def spawn_job(
    stage: Any,
    job: dict[str, Any],
    index: int,
    resolved_urdf: Path,
    fix_base: bool,
    workpiece_scale: float,
    workpiece_z_offset: float,
    default_workpiece_offset: list[float],
    debug_workpiece_box: bool,
):
    env_path = f"/World/envs/env_{index:03d}"
    robot_path = f"{env_path}/Robot"
    workpiece_path = f"{env_path}/Workpiece"

    origin = job.get("origin")
    if not (isinstance(origin, list) and len(origin) == 3):
        origin = [float(index) * 2.0, 0.0, 0.0]
    origin_tuple = (float(origin[0]), float(origin[1]), float(origin[2]))

    env_prim = ensure_xform(stage, env_path)
    set_xform_translation(env_prim, origin_tuple)

    robot_prim_path = import_robot_from_urdf(resolved_urdf, robot_path, fix_base=fix_base)
    local_offset = workpiece_local_offset(job, default_workpiece_offset)
    workpiece_prim_path = import_stl_as_mesh(
        stage,
        stl_path=job["workpiece_asset_path"],
        prim_path=workpiece_path,
        scale=workpiece_scale,
        z_offset=workpiece_z_offset,
        local_offset=local_offset,
        debug_box=debug_workpiece_box,
    )
    return {
        "id": job["id"],
        "env_path": env_path,
        "robot_prim_path": robot_prim_path,
        "workpiece_prim_path": workpiece_prim_path,
        "workpiece_offset": local_offset,
        "path_json": str(job["path_json_path"]),
    }


def scene_camera_pose(jobs: list[dict[str, Any]], explicit_eye: list[float] | None, explicit_target: list[float] | None):
    origins = []
    for index, job in enumerate(jobs):
        origin = job.get("origin")
        if not (isinstance(origin, list) and len(origin) == 3):
            origin = [float(index) * 2.0, 0.0, 0.0]
        origins.append([float(origin[0]), float(origin[1]), float(origin[2])])

    center_x = sum(origin[0] for origin in origins) / len(origins)
    center_y = sum(origin[1] for origin in origins) / len(origins)
    center_z = sum(origin[2] for origin in origins) / len(origins)
    target = tuple(explicit_target) if explicit_target is not None else (center_x, center_y, center_z + 0.25)

    span_x = max(origin[0] for origin in origins) - min(origin[0] for origin in origins)
    distance = max(2.8, span_x + 2.5)
    eye = tuple(explicit_eye) if explicit_eye is not None else (center_x + distance, center_y - distance, center_z + 1.8)
    return eye, target


def add_recording_camera(rep: Any, jobs: list[dict[str, Any]], args: argparse.Namespace):
    eye, target = scene_camera_pose(jobs, args.camera_eye, args.camera_target)
    log(f"[weldRobot] Recording camera eye={eye}, look_at={target}")
    camera = rep.create.camera(
        position=eye,
        look_at=target,
        focal_length=28.0,
        focus_distance=5.0,
        clipping_range=(0.01, 1000.0),
    )
    return rep.create.render_product(camera, resolution=(args.width, args.height))


def step_replicator(rep: Any, args: argparse.Namespace) -> None:
    try:
        rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=1.0 / args.fps)
    except TypeError:
        try:
            rep.orchestrator.step(rt_subframes=args.rt_subframes)
        except TypeError:
            rep.orchestrator.step()


def main() -> None:
    args = parse_args()
    jobs = load_manifest(args.manifest, args.max_jobs)
    frames_dir = prepare_recording_paths(args) if args.record else None
    if args.record:
        log(f"[weldRobot] Prepared recording directory: {frames_dir}")

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
        stage = get_context().get_stage()
        ensure_xform(stage, "/World/envs")
        add_scene_lighting(stage)

        resolved_urdf = make_resolved_urdf(args.urdf)
        spawned = []
        for index, job in enumerate(jobs):
            spawned_job = spawn_job(
                stage=stage,
                job=job,
                index=index,
                resolved_urdf=resolved_urdf,
                fix_base=not args.floating,
                workpiece_scale=args.workpiece_scale,
                workpiece_z_offset=args.workpiece_z_offset,
                default_workpiece_offset=args.workpiece_offset,
                debug_workpiece_box=args.debug_workpiece_box,
            )
            spawned.append(spawned_job)
            log(
                f"[weldRobot] {spawned_job['id']}: "
                f"robot={spawned_job['robot_prim_path']} "
                f"workpiece={spawned_job['workpiece_prim_path']} "
                f"workpiece_offset={spawned_job['workpiece_offset']} "
                f"path={spawned_job['path_json']}"
            )

        world.reset()
        for spawned_job in spawned:
            set_initial_joint_positions(spawned_job["robot_prim_path"])

        if not args.headless:
            add_camera_view()

        writer = None
        if args.record:
            try:
                rep.orchestrator.set_capture_on_play(False)
            except Exception:
                pass
            render_product = add_recording_camera(rep, jobs, args)
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=str(frames_dir), rgb=True)
            writer.attach([render_product])

        log(f"[weldRobot] Parallel scene ready: {len(spawned)} robots and {len(spawned)} workpieces.")
        if args.record:
            log(f"[weldRobot] Recording frames to: {frames_dir}")
        else:
            log("[weldRobot] Press Ctrl+C in the terminal to stop.")
        step_count = 0
        while simulation_app.is_running() and (args.num_steps < 0 or step_count < args.num_steps):
            world.step(render=True)
            if args.record:
                step_replicator(rep, args)
            step_count += 1
            if args.record and step_count % max(args.fps, 1) == 0:
                log(f"[weldRobot] Recorded {step_count} frames")
            time.sleep(0.0)

        if args.record and writer is not None:
            rep.orchestrator.wait_until_complete()
            writer.detach()
            png_count = len(list(frames_dir.glob("*.png"))) + len(list(frames_dir.glob("**/*.png")))
            log(f"[weldRobot] PNG files written under {frames_dir}: {png_count}")

    except Exception:
        log("[weldRobot] Fatal error while building or recording the scene:")
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

    if args.record and frames_dir is not None:
        encode_video(frames_dir, args.output, args.fps)
        if not args.keep_frames:
            shutil.rmtree(frames_dir)
        log(f"[weldRobot] Video saved to: {args.output}")


if __name__ == "__main__":
    main()
