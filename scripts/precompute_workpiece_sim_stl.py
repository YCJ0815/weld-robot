from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_ROOT = REPO_ROOT / "data_generation" / "data" / "generated_jobs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fused display/simulation STL files from per-job STEP workpieces.")
    parser.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT, help="Root directory containing job_* folders.")
    parser.add_argument("--job-glob", type=str, default="job_*", help="Glob pattern for selecting job directories.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding existing workpiece_sim.stl files.")
    parser.add_argument("--embed-mm", type=float, default=0.05, help="Small downward embed for non-base parts before fusion, in mm.")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def build_simulation_stl_from_step(step_file: Path, output_stl: Path, embed_mm: float) -> Path:
    import cadquery as cq

    imported = cq.importers.importStep(str(step_file))
    shapes = list(imported.solids().vals())
    if not shapes:
        shapes = list(imported.vals())
    if not shapes:
        raise RuntimeError(f"STEP file contains no shapes: {step_file}")

    zmins = [float(shape.BoundingBox().zmin) for shape in shapes]
    base_index = int(min(range(len(zmins)), key=zmins.__getitem__))
    moved_shapes = []
    for index, shape in enumerate(shapes):
        if index == base_index or embed_mm <= 0.0:
            moved_shapes.append(shape)
        else:
            moved_shapes.append(shape.translate((0.0, 0.0, -float(embed_mm))))

    fused = moved_shapes[0]
    for shape in moved_shapes[1:]:
        fused = fused.fuse(shape)
    try:
        fused = fused.clean()
    except Exception:
        pass

    output_stl.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(cq.Workplane("XY").newObject([fused]), str(output_stl))
    return output_stl


def main() -> None:
    args = parse_args()
    jobs_root = args.jobs_root.resolve()
    job_dirs = sorted(path for path in jobs_root.glob(args.job_glob) if path.is_dir())
    if not job_dirs:
        raise RuntimeError(f"No job directories matched {args.job_glob!r} under {jobs_root}")

    built = 0
    skipped = 0
    for job_dir in job_dirs:
        step_path = job_dir / "workpiece.step"
        output_stl = job_dir / "workpiece_sim.stl"
        if not step_path.exists():
            log(f"[sim-stl] Skipping {job_dir.name}: missing workpiece.step")
            skipped += 1
            continue
        if output_stl.exists() and not args.rebuild:
            log(f"[sim-stl] Skipping {job_dir.name}: existing workpiece_sim.stl")
            skipped += 1
            continue
        build_simulation_stl_from_step(step_path, output_stl, embed_mm=args.embed_mm)
        log(f"[sim-stl] Built {output_stl}")
        built += 1

    log(f"[sim-stl] Completed: built={built}, skipped={skipped}, jobs_root={jobs_root}")


if __name__ == "__main__":
    main()
