"""Record the UR5e welding-arm Isaac Sim scene to an MP4 video."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sim_welding_arm import (  # noqa: E402
    DEFAULT_URDF,
    import_robot_from_urdf,
    make_resolved_urdf,
    set_initial_joint_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a video of the UR5e welding-arm Isaac Sim scene.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Path to the UR5e-with-pen URDF file.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/welding_arm_scene.mp4", help="MP4 output path.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Directory used for temporary RGB frames.")
    parser.add_argument("--num-frames", type=int, default=180, help="Number of frames to record.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
    parser.add_argument("--width", type=int, default=1280, help="Render width.")
    parser.add_argument("--height", type=int, default=720, help="Render height.")
    parser.add_argument("--gui", action="store_true", help="Show Isaac Sim UI while recording.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep exported RGB frame images after encoding.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing video and frame directory.")
    return parser.parse_args()


def prepare_output_paths(args: argparse.Namespace) -> Path:
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


def add_recording_camera(rep, resolution: tuple[int, int]):
    camera_position = (2.0, -2.4, 1.4)
    camera_target = (0.0, 0.0, 0.45)
    print(f"[weldRobot] Recording camera eye={camera_position}, look_at={camera_target}")
    camera = rep.create.camera(
        position=camera_position,
        look_at=camera_target,
        focal_length=28.0,
        focus_distance=3.0,
    )
    return rep.create.render_product(camera, resolution=resolution)


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found. The RGB frames were written, but MP4 encoding cannot run. "
            f"Install ffmpeg or inspect frames in: {frames_dir}"
        )

    frame_candidates = sorted(frames_dir.glob("rgb_*.png")) or sorted(frames_dir.glob("*.png"))
    if not frame_candidates:
        raise RuntimeError(f"No PNG frames were written in: {frames_dir}")

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


def main() -> None:
    args = parse_args()
    frames_dir = prepare_output_paths(args)

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": not args.gui, "enable_cameras": True})

    try:
        try:
            from isaacsim.core.api import World
        except ImportError:
            from omni.isaac.core import World

        import omni.replicator.core as rep

        world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / args.fps, stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        resolved_urdf = make_resolved_urdf(args.urdf)
        robot_prim_path = import_robot_from_urdf(resolved_urdf, "/World/UR5ePen", fix_base=True)
        print(f"[weldRobot] Imported robot prim: {robot_prim_path}")

        world.reset()
        set_initial_joint_positions(robot_prim_path)

        render_product = add_recording_camera(rep, resolution=(args.width, args.height))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(frames_dir), rgb=True)
        writer.attach([render_product])

        print(f"[weldRobot] Recording {args.num_frames} frames to: {frames_dir}")
        for frame_idx in range(args.num_frames):
            world.step(render=True)
            rep.orchestrator.step()
            if (frame_idx + 1) % max(args.fps, 1) == 0:
                print(f"[weldRobot] Recorded {frame_idx + 1}/{args.num_frames} frames")

        rep.orchestrator.wait_until_complete()
        writer.detach()

    finally:
        simulation_app.close()

    encode_video(frames_dir, args.output, args.fps)
    if not args.keep_frames:
        shutil.rmtree(frames_dir)
    print(f"[weldRobot] Video saved to: {args.output}")


if __name__ == "__main__":
    main()
