from __future__ import annotations

import argparse
from pathlib import Path

from workpiece_sdf import SDFBuildConfig, load_or_build_workpiece_sdf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_ROOT = REPO_ROOT / "data_generation" / "data" / "generated_jobs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute cached workpiece SDF files for generated job folders.")
    parser.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT, help="Root directory containing job_* folders.")
    parser.add_argument("--job-glob", type=str, default="job_*", help="Glob pattern for selecting job directories.")
    parser.add_argument("--workpiece-scale", type=float, default=0.001, help="Scale applied to workpiece STL coordinates.")
    parser.add_argument("--workpiece-z-offset", type=float, default=0.0025, help="Extra Z offset in meters for the workpiece mesh.")
    parser.add_argument("--workpiece-offset", type=float, nargs=3, default=[0.45, 0.0, 0.0], help="World offset in meters for the workpiece mesh.")
    parser.add_argument("--sdf-pitch", type=float, default=0.004, help="Voxel pitch in meters for the cached workpiece SDF.")
    parser.add_argument("--sdf-margin", type=float, default=0.03, help="Extra world-space margin in meters around the SDF grid.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding existing workpiece_sdf.npz files.")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    args = parse_args()
    jobs_root = args.jobs_root.resolve()
    if not jobs_root.exists():
        raise FileNotFoundError(f"Jobs root does not exist: {jobs_root}")

    job_dirs = sorted(path for path in jobs_root.glob(args.job_glob) if path.is_dir())
    if not job_dirs:
        raise RuntimeError(f"No job directories matched {args.job_glob!r} under {jobs_root}")

    succeeded = 0
    skipped = 0
    for job_dir in job_dirs:
        stl_path = job_dir / "workpiece.stl"
        if not stl_path.exists():
            log(f"[SDF] Skipping {job_dir.name}: missing workpiece.stl")
            skipped += 1
            continue
        npz_path = job_dir / "workpiece_sdf.npz"
        load_or_build_workpiece_sdf(
            stl_path=stl_path,
            scale=args.workpiece_scale,
            z_offset=args.workpiece_z_offset,
            local_offset=tuple(float(v) for v in args.workpiece_offset),
            npz_path=npz_path,
            config=SDFBuildConfig(voxel_pitch=args.sdf_pitch, margin=args.sdf_margin),
            logger=log,
            rebuild=args.rebuild,
        )
        succeeded += 1

    log(f"[SDF] Completed precompute: succeeded={succeeded}, skipped={skipped}, jobs_root={jobs_root}")


if __name__ == "__main__":
    main()
