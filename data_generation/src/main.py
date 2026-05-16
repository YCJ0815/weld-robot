from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "mode_generate" / "model_generation.py").exists():
            return candidate
    raise FileNotFoundError("Cannot find project root containing mode_generate/model_generation.py")


ROOT = _find_project_root()
MODEL_OUTPUT_DIR = ROOT / "data" / "model"
EXTRACT_OUTPUT_DIR = ROOT / "data" / "extract"
FINAL_OUTPUT_DIR = ROOT / "data" / "path"
VECTOR_OUTPUT_DIR = ROOT / "data" / "vector"
JOBS_OUTPUT_DIR = ROOT / "data" / "generated_jobs"
SEAM_EXTRACT_DIR = ROOT / "seam_extract"
DEFAULT_EXTRACT_PYTHON = Path("/Users/ycj/miniconda3/bin/python")
CACHE_DIR = ROOT / "data" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
root_path = str(ROOT)
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def _load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clear_output_dir(path: Path) -> None:
    resolved = path.resolve()
    if resolved in {ROOT.resolve(), ROOT.resolve().parent, Path.home().resolve()}:
        raise RuntimeError(f"Refusing to clear unsafe output directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    for child in resolved.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def relative_to_directory(path: Path, directory: Path) -> str:
    try:
        return path.resolve().relative_to(directory.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), directory.resolve())


def copy_if_exists(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def build_sim_path_json(vector_json: Path, output_json: Path) -> Path:
    with vector_json.open("r", encoding="utf-8") as f:
        vector_obj = json.load(f)

    welds = vector_obj.get("welds")
    if not isinstance(welds, list):
        raise RuntimeError(f"Invalid weld vector JSON structure: {vector_json}")

    segments: list[dict[str, Any]] = []
    waypoints: list[dict[str, Any]] = []
    for weld_index, weld in enumerate(welds):
        if not isinstance(weld, dict):
            continue
        start = weld.get("start") if isinstance(weld.get("start"), dict) else {}
        end = weld.get("end") if isinstance(weld.get("end"), dict) else {}
        start_xyz = start.get("xyz")
        end_xyz = end.get("xyz")
        if not (isinstance(start_xyz, list) and len(start_xyz) == 3):
            continue
        if not (isinstance(end_xyz, list) and len(end_xyz) == 3):
            continue

        start_waypoint = {
            "position": [float(value) for value in start_xyz],
            "normal": start.get("pose"),
            "role": "start",
            "weld_index": weld_index,
        }
        end_waypoint = {
            "position": [float(value) for value in end_xyz],
            "normal": end.get("pose"),
            "role": "end",
            "weld_index": weld_index,
        }
        segment = {
            "id": f"weld_{weld_index:04d}",
            "start": start_waypoint,
            "end": end_waypoint,
        }
        segments.append(segment)
        if not waypoints or waypoints[-1]["position"] != start_waypoint["position"]:
            waypoints.append(start_waypoint)
        waypoints.append(end_waypoint)

    sim_path = {
        "schema": "weld_robot.path.v1",
        "frame": "workpiece",
        "units": "mm",
        "orientation": "normal_vector",
        "source": relative_to_directory(vector_json, output_json.parent),
        "segments": segments,
        "waypoints": waypoints,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(sim_path, f, indent=2, ensure_ascii=False)
    return output_json


def load_compiler_module() -> Any:
    seam_path = str(SEAM_EXTRACT_DIR)
    if seam_path not in sys.path:
        sys.path.insert(0, seam_path)
    return _load_module("seam_extract_compiler_main", SEAM_EXTRACT_DIR / "compiler-main.py")


def load_sequence_module() -> Any:
    return _load_module("seam_extract_s5_build_welds_sequence", SEAM_EXTRACT_DIR / "s5_build_welds_sequence.py")


def load_pose_normals_module() -> Any:
    return _load_module("compute_point_pose_normals", ROOT / "compute_point_pose_normals.py")


def load_model_generation_module() -> Any:
    return _load_module("mode_generate_model_generation", ROOT / "mode_generate" / "model_generation.py")


def generate_models_in_subprocess(count: int, output_dir: Path) -> list[dict[str, str]]:
    code = r"""
import importlib.util
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
count = int(sys.argv[2])
output_dir = Path(sys.argv[3])
spec = importlib.util.spec_from_file_location(
    "mode_generate_model_generation",
    root / "mode_generate" / "model_generation.py",
)
if spec is None or spec.loader is None:
    raise ImportError("Cannot load mode_generate/model_generation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
generated = module.generate_batch(count=count, output_dir=str(output_dir))
print("__GENERATED_JSON__" + json.dumps(generated, ensure_ascii=False))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(count), str(output_dir)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    marker = "__GENERATED_JSON__"

    for line in stdout.splitlines():
        if not line.startswith(marker):
            print(line)

    if result.returncode != 0:
        raise RuntimeError(
            "Model generation subprocess failed.\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])

    raise RuntimeError(
        "Model generation subprocess did not return generated model paths.\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def get_extract_python() -> str:
    configured = os.environ.get("WELD_EXTRACT_PYTHON")
    if configured:
        return configured
    if DEFAULT_EXTRACT_PYTHON.exists():
        return str(DEFAULT_EXTRACT_PYTHON)
    return sys.executable


def extract_sequence_in_subprocess(
    step_file: Path,
    extract_dir: Path,
    final_dir: Path,
    vector_dir: Path,
    compute_pose_normals: bool,
    pose_normal_tol: float,
) -> Path:
    result = subprocess.run(
        [
            get_extract_python(),
            str(Path(__file__).resolve()),
            "--extract-worker",
            "--step-file",
            str(step_file),
            "--extract-dir",
            str(extract_dir),
            "--final-dir",
            str(final_dir),
            "--vector-dir",
            str(vector_dir),
            "--pose-normal-tol",
            str(pose_normal_tol),
            *(["--skip-pose-normals"] if not compute_pose_normals else []),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    marker = "__SEQUENCE_JSON__"

    for line in stdout.splitlines():
        if not line.startswith(marker):
            print(line)

    if result.returncode != 0:
        raise RuntimeError(
            "Weld extraction subprocess failed.\n"
            f"python: {get_extract_python()}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            return Path(line[len(marker):])

    raise RuntimeError(
        "Weld extraction subprocess did not return a sequence JSON path.\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def install_occ_import_alias_if_needed() -> None:
    try:
        import OCC  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    try:
        import occ
        import occ.core
        import occ.Display
        import occ.Extend
        import occ.Wrapper
    except ModuleNotFoundError:
        return

    setattr(occ, "Core", occ.core)
    sys.modules.setdefault("OCC", occ)
    sys.modules.setdefault("OCC.Core", occ.core)
    sys.modules.setdefault("OCC.Display", occ.Display)
    sys.modules.setdefault("OCC.Extend", occ.Extend)
    sys.modules.setdefault("OCC.Wrapper", occ.Wrapper)


def detect_breakpoints_headless(compiler: Any, geometry_graph: dict) -> dict:
    process_graph, candidate_ids = compiler.s0.detect_through_hole_edges_from_adjacent(
        geometry_graph,
        bspline_rms_tol=0.25,
        require_arc_center=False,
        bspline_min_angle_deg=8.0,
        bspline_max_radius=500.0,
        bspline_min_sagitta_abs=0.30,
        bspline_min_sagitta_ratio=0.02,
        detect_chamfer_holes=True,
        chamfer_min_angle_deg=92.0,
        chamfer_max_angle_deg=150.0,
        chamfer_max_length=50.0,
        debug=True,
    )
    connected_nodes = []
    for node_id, node_data in process_graph.get("nodes", {}).items():
        if node_data.get("process", {}).get("through_hole_edge_ids"):
            connected_nodes.append(str(node_id))

    for node_id in connected_nodes:
        node_process = (process_graph.get("nodes", {}).get(str(node_id), {}) or {}).get("process", {})
        hole_edge_ids = node_process.get("through_hole_edge_ids") or []
        for hole_edge_id in hole_edge_ids:
            info = compiler.s0.insert_breakpoint_for_node(
                geometry_graph,
                node_id,
                hole_edge_id=str(hole_edge_id),
                require_hole_edge_match=True,
                max_weld_length=100.0,
                verbose=True,
            )
            if info:
                break

    t_type_info = compiler.s0.detect_t_type_breakpoints(
        geometry_graph,
        candidate_ids,
        t_type_min_weld_length=5.0,
        t_type_max_weld_length=5000.0,
        t_type_extension_ratio=2.0,
        t_type_max_distance_to_weld=1000.0,
        debug=True,
    )
    geometry_graph["t_type_breakpoints"] = (
        t_type_info["t_type_breakpoints"] if t_type_info else []
    )
    return process_graph


def extract_welds_for_step(compiler: Any, step_file: Path, base_dir: Path) -> Path:
    from src.config import ProjectConfig

    stem = step_file.stem
    cfg = ProjectConfig(stp_head=stem, base_dir=str(base_dir))
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] loading STEP: {step_file}")
    shape = compiler.load_step_file(str(step_file))
    if shape is None:
        raise RuntimeError(f"Failed to load STEP file: {step_file}")

    compiler.extract_contact_boundaries_face_based(
        shape,
        cfg.contact_edges_file,
        bbox_tol=5.0,
        contact_tol=0.5,
        face_tol=0.8,
        min_edge_length=0.1,
        do_section_approx=True,
        profile=True,
        use_solid_grid_candidates=True,
        solid_grid_cell_size=None,
        face_grid_cell_size=None,
        bspline_detail=False,
        bspline_len_samples=24,
        bspline_vis_samples=24,
        geometry_graph_file=cfg.geometry_graph_file,
        graph_point_tol=0.2,
        graph_min_geom_edge_length=0.05,
        graph_include_weld_edges_in_adjacent=False,
        graph_store_all_geom_edges=False,
        graph_source_step=step_file.name,
    )

    geometry_graph = compiler.load_json(cfg.geometry_graph_file)
    process_graph = detect_breakpoints_headless(compiler, geometry_graph)
    compiler.save_json(cfg.geometry_graph_with_breakpoints_file, geometry_graph)
    compiler.save_json(cfg.process_graph_file, process_graph)

    final_obj, _up_axis = compiler.s1.build_final_json(
        geometry_graph,
        process_graph,
        l_push_on="B",
        u_wrap_distance_threshold=20.0,
        u_wrap_max_nearby_welds=2,
        vertical_weld_deg=20.0,
    )
    compiler.save_json(cfg.final_welds_file, final_obj)

    compiler.s2.update_final_welds_with_t_type_breakpoints(
        final_welds_json=cfg.final_welds_file,
        geometry_graph_with_breakpoints_json=cfg.geometry_graph_with_breakpoints_file,
        out_json=cfg.final_welds_with_junctions_file,
        max_remove_length=50.0,
    )
    return Path(cfg.final_welds_with_junctions_file)


def build_sequence_file(sequence_builder: Any, input_json: Path, output_json: Path) -> None:
    obj = sequence_builder.load_json(str(input_json))
    contact_edges = obj.get("contact_edges", {}) or {}
    raw_sequences = sequence_builder.build_weld_sequences(contact_edges)
    processed_sequences: list[list[str]] = []
    for sequence in raw_sequences:
        ordered, _ = sequence_builder.order_sequence_edges(contact_edges, sequence)
        t_split = sequence_builder.split_sequence_by_adjacent_t(ordered)
        for t_segment in t_split:
            sequence_builder.orient_sequence_edges(contact_edges, t_segment)
            processed_sequences.append(t_segment)

    obj["welds_sequence"] = sequence_builder.build_welds_sequence_dict(
        processed_sequences,
        contact_edges,
    )
    trajectories, seq_map = sequence_builder.build_welds_trajectory(
        processed_sequences,
        obj["welds_sequence"],
    )
    up_axis = obj.get("up_axis")
    if not isinstance(up_axis, list) or len(up_axis) != 3:
        up_axis = [0.0, 0.0, 1.0]
    sequence_builder.sort_trajectories(trajectories, obj["welds_sequence"], up_axis)
    basis = sequence_builder.sort_trajectory_list_by_plane(
        trajectories,
        obj["welds_sequence"],
        up_axis,
    )
    if basis is not None:
        sequence_builder.sort_sequences_within_trajectory(
            trajectories,
            obj["welds_sequence"],
            seq_map,
            contact_edges,
            basis,
        )
    sequence_builder.align_label1_sequences(
        trajectories,
        obj["welds_sequence"],
        seq_map,
        contact_edges,
        processed_sequences,
        up_axis,
    )

    welds_trajectory: dict[str, dict[str, Any]] = {}
    for i, trajectory in enumerate(trajectories, start=1):
        key = f"welds_trajectory_{i}"
        welds_trajectory[key] = {}
        category_map = trajectory.get("sequence_category", {}) or {}
        for j, sequence_id in enumerate(trajectory.get("sequence_ids", []), start=1):
            label = category_map.get(str(sequence_id), 3)
            welds_trajectory[key][f"sequence{j}"] = {
                "id": str(sequence_id),
                "label": label,
            }
        welds_trajectory[key]["solid_ids"] = trajectory.get("solid_ids")

    obj["welds_trajectory"] = welds_trajectory
    sequence_builder.save_json(str(output_json), obj)


def _sequence_edge_ids(sequence: dict[str, Any]) -> list[str]:
    edge_items: list[tuple[int, str]] = []
    for key, value in sequence.items():
        if not str(key).startswith("edge"):
            continue
        try:
            index = int(str(key)[4:])
        except ValueError:
            continue
        edge_items.append((index, str(value)))
    return [edge_id for _index, edge_id in sorted(edge_items)]


def _ordered_sequence_refs(final_obj: dict[str, Any]) -> list[dict[str, Any]]:
    welds_sequence = final_obj.get("welds_sequence") or {}
    welds_trajectory = final_obj.get("welds_trajectory") or {}
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    trajectory_items: list[tuple[int, dict[str, Any]]] = []
    for key, value in welds_trajectory.items():
        if not isinstance(value, dict):
            continue
        try:
            index = int(str(key).rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        trajectory_items.append((index, value))

    for _trajectory_index, trajectory in sorted(trajectory_items):
        sequence_items: list[tuple[int, str]] = []
        for key, value in trajectory.items():
            if not str(key).startswith("sequence") or not isinstance(value, dict):
                continue
            try:
                index = int(str(key)[8:])
            except ValueError:
                continue
            sequence_id = value.get("id")
            if sequence_id is not None:
                sequence_items.append((index, str(sequence_id)))
        for _sequence_index, sequence_id in sorted(sequence_items):
            if sequence_id not in seen:
                ordered.append(
                    {
                        "trajectory_order": _trajectory_index,
                        "sequence_order_in_trajectory": _sequence_index,
                        "sequence_id": sequence_id,
                    }
                )
                seen.add(sequence_id)

    def sequence_key_order(key: Any) -> tuple[int, str]:
        suffix = str(key).rsplit("_", 1)[-1]
        return (int(suffix) if suffix.isdigit() else 10**9, str(key))

    fallback_keys = sorted(welds_sequence, key=sequence_key_order)
    for key in fallback_keys:
        sequence_id = str(key).rsplit("_", 1)[-1]
        if sequence_id not in seen:
            ordered.append(
                {
                    "trajectory_order": None,
                    "sequence_order_in_trajectory": None,
                    "sequence_id": sequence_id,
                }
            )
            seen.add(sequence_id)
    return ordered


def _point_payload(points: dict[str, Any], point_ref: Any, fallback_xyz: Any) -> dict[str, Any]:
    point_id = None
    if isinstance(point_ref, dict):
        point_id = point_ref.get("point_id")
    point = points.get(str(point_id), {}) if point_id is not None else {}
    if not isinstance(point, dict):
        point = {}
    return {
        "xyz": point.get("xyz", fallback_xyz),
        "pose": point.get("pose_normal"),
    }


def export_ordered_weld_vectors(final_json: Path, output_json: Path) -> Path:
    with final_json.open("r", encoding="utf-8") as f:
        final_obj = json.load(f)

    contact_edges = final_obj.get("contact_edges") or {}
    welds_sequence = final_obj.get("welds_sequence") or {}
    points = final_obj.get("points") or {}
    if not isinstance(contact_edges, dict) or not isinstance(welds_sequence, dict):
        raise RuntimeError(f"Invalid final weld JSON structure: {final_json}")
    if not isinstance(points, dict):
        points = {}

    sequence_refs = _ordered_sequence_refs(final_obj)
    ordered_welds: list[dict[str, Any]] = []

    for sequence_ref in sequence_refs:
        sequence_id = str(sequence_ref["sequence_id"])
        sequence_key = f"welds_sequence_{sequence_id}"
        sequence = welds_sequence.get(sequence_key)
        if not isinstance(sequence, dict):
            continue

        for edge_id in _sequence_edge_ids(sequence):
            edge = contact_edges.get(str(edge_id))
            if not isinstance(edge, dict):
                continue
            edge_points = edge.get("points") if isinstance(edge.get("points"), list) else []
            start_ref = edge_points[0] if edge_points else None
            end_ref = edge_points[-1] if edge_points else None
            weld = {
                "start": _point_payload(points, start_ref, edge.get("start")),
                "end": _point_payload(points, end_ref, edge.get("end")),
            }
            ordered_welds.append(weld)
    out = {"welds": ordered_welds}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[vector] {final_json.name} -> {output_json}")
    return output_json


def build_simulation_jobs(
    generated_models: list[dict[str, str]],
    sequence_outputs: list[Path],
    vector_dir: Path,
    jobs_dir: Path,
    manifest_name: str,
    spacing: float,
) -> Path:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []

    for index, (model_info, sequence_output) in enumerate(zip(generated_models, sequence_outputs)):
        job_id = f"job_{index:03d}"
        job_dir = jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        sequence_output = sequence_output.resolve()
        stem = sequence_output.stem
        step_source = Path(model_info["step"]).resolve()
        stl_source = Path(model_info["stl"]).resolve()
        vector_source = (vector_dir / f"{stem}_weld_vectors.json").resolve()

        step_asset = copy_if_exists(step_source, job_dir / "workpiece.step")
        stl_asset = copy_if_exists(stl_source, job_dir / "workpiece.stl")
        raw_path_json = copy_if_exists(sequence_output, job_dir / "raw_weld_topology.json")
        vector_json = copy_if_exists(vector_source, job_dir / "weld_vectors.json")
        path_json = build_sim_path_json(vector_json, job_dir / "path.json") if vector_json is not None else raw_path_json

        workpiece_asset = stl_asset or step_asset
        if workpiece_asset is None:
            raise RuntimeError(f"No workpiece asset was produced for {job_id}: {model_info}")
        if path_json is None:
            raise RuntimeError(f"No path JSON was produced for {job_id}: {sequence_output}")

        job: dict[str, Any] = {
            "id": job_id,
            "source_stem": stem,
            "workpiece_asset": relative_to_directory(workpiece_asset, jobs_dir),
            "path_json": relative_to_directory(path_json, jobs_dir),
            "origin": [float(index) * float(spacing), 0.0, 0.0],
            "workpiece_offset": [0.45, 0.0, 0.0],
            "frame": "workpiece",
            "units": "mm",
        }
        if step_asset is not None:
            job["step_asset"] = relative_to_directory(step_asset, jobs_dir)
        if stl_asset is not None:
            job["stl_asset"] = relative_to_directory(stl_asset, jobs_dir)
        if vector_json is not None:
            job["vector_json"] = relative_to_directory(vector_json, jobs_dir)
        if raw_path_json is not None:
            job["raw_topology_json"] = relative_to_directory(raw_path_json, jobs_dir)
        jobs.append(job)

    manifest = {
        "schema": "weld_robot.simulation_jobs.v1",
        "description": "Generated workpiece/path jobs for Isaac Sim parallel welding simulation.",
        "path_resolution": "relative_to_manifest_directory",
        "jobs_dir": ".",
        "job_count": len(jobs),
        "default_frame": "workpiece",
        "default_units": "mm",
        "jobs": jobs,
    }
    manifest_path = jobs_dir / manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[manifest] saved {len(jobs)} jobs to {manifest_path}")
    return manifest_path


def run_pipeline(
    count: int,
    model_dir: Path,
    extract_dir: Path,
    final_dir: Path,
    vector_dir: Path,
    jobs_dir: Path,
    manifest_name: str,
    spacing: float,
    compute_pose_normals: bool,
    pose_normal_tol: float,
) -> tuple[list[Path], Path]:
    for output_dir in (model_dir, extract_dir, final_dir, vector_dir, jobs_dir):
        clear_output_dir(output_dir)

    try:
        generated = generate_models_in_subprocess(count=count, output_dir=model_dir)
    except RuntimeError:
        raise
    except ModuleNotFoundError as exc:
        if exc.name == "cadquery":
            raise RuntimeError(
                "Missing dependency: cadquery. Please run main.py with an environment that has cadquery installed."
            ) from exc
        raise

    outputs: list[Path] = []
    for item in generated:
        step_file = Path(item["step"]).resolve()
        sequence_output = extract_sequence_in_subprocess(
            step_file,
            extract_dir,
            final_dir,
            vector_dir,
            compute_pose_normals=compute_pose_normals,
            pose_normal_tol=pose_normal_tol,
        )
        outputs.append(sequence_output)

    manifest_path = build_simulation_jobs(
        generated_models=generated,
        sequence_outputs=outputs,
        vector_dir=vector_dir,
        jobs_dir=jobs_dir,
        manifest_name=manifest_name,
        spacing=spacing,
    )
    return outputs, manifest_path


def run_extract_worker(
    step_file: Path,
    extract_dir: Path,
    final_dir: Path,
    vector_dir: Path,
    compute_pose_normals: bool,
    pose_normal_tol: float,
) -> Path:
    install_occ_import_alias_if_needed()
    try:
        compiler = load_compiler_module()
    except ModuleNotFoundError as exc:
        if exc.name == "OCC":
            raise RuntimeError(
                "Missing dependency: OCC/pythonocc-core. The extraction stage requires pythonocc-core."
            ) from exc
        raise
    if hasattr(compiler, "load_runtime_dependencies"):
        try:
            compiler.load_runtime_dependencies()
        except ModuleNotFoundError as exc:
            if exc.name == "OCC":
                raise RuntimeError(
                    "Missing dependency: OCC/pythonocc-core. The extraction stage requires pythonocc-core."
                ) from exc
            raise

    sequence_builder = load_sequence_module()
    step_file = step_file.resolve()
    stem = step_file.stem
    model_extract_dir = extract_dir / stem
    sequence_input = extract_welds_for_step(compiler, step_file, model_extract_dir)
    sequence_output = final_dir / f"{stem}.json"
    final_dir.mkdir(parents=True, exist_ok=True)
    build_sequence_file(sequence_builder, sequence_input, sequence_output)
    if compute_pose_normals:
        pose_normals = load_pose_normals_module()
        pose_normals.compute_pose_normals(
            step_file=step_file,
            final_json=sequence_output,
            geometry_graph_path=model_extract_dir / f"{stem}_geometry_graph_with_breakpoints.json",
            process_graph_path=model_extract_dir / f"{stem}_process_graph.json",
            output_json=sequence_output,
            tol=pose_normal_tol,
        )
    export_ordered_weld_vectors(sequence_output, vector_dir / f"{stem}_weld_vectors.json")
    print(f"[save] {step_file.name} -> {sequence_output}")
    return sequence_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random STEP models, extract welds, sort weld sequences, and save one JSON per STEP."
    )
    parser.add_argument("--count", type=int, default=1, help="Number of random models to generate.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_OUTPUT_DIR, help="Generated STEP/STL output directory.")
    parser.add_argument("--extract-dir", type=Path, default=EXTRACT_OUTPUT_DIR, help="Intermediate extraction directory.")
    parser.add_argument("--final-dir", type=Path, default=FINAL_OUTPUT_DIR, help="Final sequence JSON output directory.")
    parser.add_argument("--vector-dir", type=Path, default=VECTOR_OUTPUT_DIR, help="Ordered weld endpoint/vector JSON output directory.")
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_OUTPUT_DIR, help="Isaac Sim job package output directory.")
    parser.add_argument("--manifest-name", default="manifest.json", help="Manifest filename written inside --jobs-dir.")
    parser.add_argument("--spacing", type=float, default=2.0, help="Default X spacing between jobs in Isaac Sim scene units.")
    parser.add_argument("--skip-pose-normals", action="store_true", help="Skip point pose normal computation.")
    parser.add_argument("--pose-normal-tol", type=float, default=1e-2, help="Point-to-face/arc tolerance for pose normals.")
    parser.add_argument("--extract-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--step-file", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.extract_worker:
            if args.step_file is None:
                raise RuntimeError("--extract-worker requires --step-file")
            output = run_extract_worker(
                step_file=args.step_file,
                extract_dir=args.extract_dir,
                final_dir=args.final_dir,
                vector_dir=args.vector_dir,
                compute_pose_normals=not args.skip_pose_normals,
                pose_normal_tol=args.pose_normal_tol,
            )
            print(f"__SEQUENCE_JSON__{output}")
            return
        outputs, manifest_path = run_pipeline(
            count=args.count,
            model_dir=args.model_dir,
            extract_dir=args.extract_dir,
            final_dir=args.final_dir,
            vector_dir=args.vector_dir,
            jobs_dir=args.jobs_dir,
            manifest_name=args.manifest_name,
            spacing=args.spacing,
            compute_pose_normals=not args.skip_pose_normals,
            pose_normal_tol=args.pose_normal_tol,
        )
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"[done] saved {len(outputs)} JSON files to {args.final_dir}")
    print(f"[done] Isaac Sim manifest: {manifest_path}")


if __name__ == "__main__":
    main()
