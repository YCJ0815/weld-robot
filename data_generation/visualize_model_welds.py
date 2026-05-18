from __future__ import annotations

import argparse
import os
import json
import struct
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
SEAM_EXTRACT_DIR = ROOT / "seam_extract"
CACHE_DIR = ROOT / "data" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

WELD_COLOR = "red"
WELD_RGB = (1.0, 0.05, 0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a STEP/STP model with extracted weld seams overlaid."
    )
    parser.add_argument(
        "--step",
        type=Path,
        default=None,
        help="Path to the STEP/STP model file. Default: auto-detect from data/model.",
    )
    parser.add_argument(
        "--stl",
        type=Path,
        default=None,
        help="Path to the STL model file for Matplotlib visualization. Default: same stem as --step with .stl.",
    )
    parser.add_argument(
        "--weld-json",
        type=Path,
        default=None,
        help="Path to contact_edges/final_welds/final sequence JSON. Default: auto-detect from model stem.",
    )
    parser.add_argument(
        "--weld-width",
        type=float,
        default=4.0,
        help="Displayed weld line width.",
    )
    parser.add_argument(
        "--point-radius",
        type=float,
        default=1.8,
        help="Endpoint/breakpoint marker radius. Use 0 to hide markers.",
    )
    parser.add_argument(
        "--show-endpoints",
        action="store_true",
        default=True,
        help="Show endpoint markers. Enabled by default.",
    )
    parser.add_argument(
        "--hide-endpoints",
        dest="show_endpoints",
        action="store_false",
        help="Hide endpoint markers.",
    )
    parser.add_argument(
        "--no-points",
        action="store_true",
        help="Hide endpoint and breakpoint markers.",
    )
    parser.add_argument(
        "--no-pose-normals",
        action="store_true",
        help="Hide point pose_normal vectors.",
    )
    parser.add_argument(
        "--normal-length",
        type=float,
        default=12.0,
        help="Displayed pose normal vector length in model units.",
    )
    parser.add_argument(
        "--normal-color",
        default="limegreen",
        help="Matplotlib color for pose normal vectors.",
    )
    parser.add_argument(
        "--show-pose-components",
        action="store_true",
        default=True,
        help="Show the three component vectors used to compose pose_normal. Enabled by default.",
    )
    parser.add_argument(
        "--no-pose-components",
        dest="show_pose_components",
        action="store_false",
        help="Hide the three component vectors used to compose pose_normal.",
    )
    parser.add_argument(
        "--component-length",
        type=float,
        default=10.0,
        help="Displayed length for pose component vectors in model units.",
    )
    parser.add_argument(
        "--normal-offset",
        type=float,
        default=1.0,
        help="Offset pose normal origin along its direction to keep vectors visible on opaque models.",
    )
    parser.add_argument(
        "--model-transparency",
        type=float,
        default=0.75,
        help="Model transparency, 0 is opaque and 1 is invisible.",
    )
    parser.add_argument(
        "--backend",
        choices=("matplotlib", "occ"),
        default="matplotlib",
        help="Visualization backend. matplotlib is safer; occ uses the interactive pythonocc viewer.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save a PNG when using the Matplotlib backend.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive Matplotlib window. Useful with --save.",
    )
    return parser.parse_args()


def normalize_model_stem(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "_final_welds_with_junctions_sequence",
        "_final_welds_with_junctions",
        "_final_welds_sequence",
        "_final_welds",
        "_contact_edges",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_model_file(stem: str) -> Path | None:
    model_dir = ROOT / "data" / "model"
    for suffix in (".step", ".stp", ".STEP", ".STP"):
        candidate = model_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def find_weld_json(stem: str) -> Path | None:
    candidates = [
        ROOT / "data" / "path" / f"{stem}.json",
        ROOT / "data" / "extract" / stem / f"{stem}_final_welds_with_junctions.json",
        ROOT / "data" / "extract" / stem / f"{stem}_final_welds.json",
        ROOT / "data" / "extract" / stem / f"{stem}_contact_edges.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_path_json_model_pairs() -> list[tuple[float, Path, Path]]:
    path_dir = ROOT / "data" / "path"
    pairs: list[tuple[float, Path, Path]] = []
    for weld_json in path_dir.glob("*.json"):
        stem = normalize_model_stem(weld_json)
        step_path = find_model_file(stem)
        if step_path is None:
            continue
        pairs.append((max(mtime_or_zero(step_path), mtime_or_zero(weld_json)), step_path, weld_json))
    return pairs


def mtime_or_zero(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    return path.stat().st_mtime


def autodetect_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.step is not None:
        step_path = args.step.resolve()
        stem = normalize_model_stem(step_path)
        weld_json_path = args.weld_json.resolve() if args.weld_json is not None else find_weld_json(stem)
        if weld_json_path is None:
            raise FileNotFoundError(f"Cannot auto-detect weld JSON for model stem: {stem}")
        return step_path, weld_json_path.resolve()

    if args.weld_json is not None:
        weld_json_path = args.weld_json.resolve()
        stem = normalize_model_stem(weld_json_path)
        step_path = find_model_file(stem)
        if step_path is None:
            raise FileNotFoundError(f"Cannot auto-detect STEP/STP model for weld JSON stem: {stem}")
        return step_path.resolve(), weld_json_path

    # Check for generated_jobs/job_000 first
    generated_jobs_dir = ROOT / "data" / "generated_jobs" / "job_003"
    if generated_jobs_dir.exists():
        workpiece_step = generated_jobs_dir / "workpiece.step"
        weld_vectors = generated_jobs_dir / "weld_vectors.json"
        if workpiece_step.exists() and weld_vectors.exists():
            print(f"[visualize] auto-detected from generated_jobs/job_000")
            print(f"[visualize] auto-detected model -> {workpiece_step}")
            print(f"[visualize] auto-detected weld vectors -> {weld_vectors}")
            return workpiece_step.resolve(), weld_vectors.resolve()

    pairs = find_path_json_model_pairs()
    if pairs:
        _, step_path, weld_json_path = max(pairs, key=lambda item: item[0])
        print(f"[visualize] auto-detected path JSON -> {weld_json_path}")
        print(f"[visualize] auto-detected model -> {step_path}")
        return step_path.resolve(), weld_json_path.resolve()

    model_dir = ROOT / "data" / "model"
    model_files = []
    for pattern in ("*.step", "*.stp", "*.STEP", "*.STP"):
        model_files.extend(model_dir.glob(pattern))

    extract_pairs: list[tuple[float, Path, Path]] = []
    for model_file in model_files:
        stem = normalize_model_stem(model_file)
        weld_json = find_weld_json(stem)
        if weld_json is None:
            continue
        extract_pairs.append((max(mtime_or_zero(model_file), mtime_or_zero(weld_json)), model_file, weld_json))

    if not extract_pairs:
        raise FileNotFoundError(
            "Cannot auto-detect matching model and weld JSON. "
            "Pass --step and/or --weld-json explicitly."
        )

    _, step_path, weld_json_path = max(extract_pairs, key=lambda item: item[0])
    print(f"[visualize] auto-detected model -> {step_path}")
    print(f"[visualize] auto-detected weld JSON -> {weld_json_path}")
    return step_path.resolve(), weld_json_path.resolve()


def install_occ_import_alias_if_needed() -> None:
    try:
        import OCC  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    try:
        import occ
        import occ.core
    except ModuleNotFoundError:
        return

    setattr(occ, "Core", occ.core)
    sys.modules.setdefault("OCC", occ)
    sys.modules.setdefault("OCC.Core", occ.core)


def ensure_seam_extract_on_path() -> None:
    seam_path = str(SEAM_EXTRACT_DIR)
    if seam_path not in sys.path:
        sys.path.insert(0, seam_path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def convert_weld_vectors_to_contact_edges(weld_vectors_data: dict[str, Any]) -> dict[str, Any]:
    """Convert weld_vectors.json format to contact_edges format for visualization."""
    contact_edges = {}
    welds = weld_vectors_data.get("welds", [])
    
    for idx, weld in enumerate(welds):
        start = weld.get("start", {})
        end = weld.get("end", {})
        
        start_xyz = start.get("xyz")
        end_xyz = end.get("xyz")
        start_pose = start.get("pose")
        end_pose = end.get("pose")
        
        if start_xyz and end_xyz:
            contact_edges[str(idx)] = {
                "start": start_xyz,
                "end": end_xyz,
                "samples": [start_xyz, end_xyz],
            }
    
    # Convert to the structure expected by the visualization code
    return {
        "contact_edges": contact_edges,
        "points": convert_weld_vectors_to_points(weld_vectors_data),
    }


def convert_weld_vectors_to_points(weld_vectors_data: dict[str, Any]) -> dict[str, Any]:
    """Convert weld_vectors to points format with pose_normal vectors."""
    points = {}
    welds = weld_vectors_data.get("welds", [])
    
    point_idx = 0
    for weld_idx, weld in enumerate(welds):
        start = weld.get("start", {})
        end = weld.get("end", {})
        
        # Process start point
        if start.get("xyz"):
            points[str(point_idx)] = {
                "xyz": start.get("xyz"),
                "pose_normal": start.get("pose"),
                "role": "endpoint",
                "weld_index": weld_idx,
                "point_type": "start",
            }
            point_idx += 1
        
        # Process end point
        if end.get("xyz"):
            points[str(point_idx)] = {
                "xyz": end.get("xyz"),
                "pose_normal": end.get("pose"),
                "role": "endpoint",
                "weld_index": weld_idx,
                "point_type": "end",
            }
            point_idx += 1
    
    return points


def as_xyz(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def iter_edge_samples(edge: dict[str, Any]) -> Iterable[list[list[float]]]:
    samples = edge.get("samples")
    if isinstance(samples, list):
        clean = [xyz for xyz in (as_xyz(p) for p in samples) if xyz is not None]
        if len(clean) >= 2:
            yield clean
            return

    start = as_xyz(edge.get("start"))
    end = as_xyz(edge.get("end"))
    if start and end:
        yield [start, end]


def polyline_midpoint(polyline: list[list[float]]) -> list[float]:
    if not polyline:
        return [0.0, 0.0, 0.0]
    if len(polyline) == 1:
        return polyline[0]

    lengths: list[float] = []
    total = 0.0
    for p0, p1 in zip(polyline, polyline[1:]):
        seg_len = sum((p1[i] - p0[i]) ** 2 for i in range(3)) ** 0.5
        lengths.append(seg_len)
        total += seg_len

    if total <= 1e-9:
        return polyline[len(polyline) // 2]

    target = total * 0.5
    acc = 0.0
    for p0, p1, seg_len in zip(polyline, polyline[1:], lengths):
        if acc + seg_len >= target:
            t = (target - acc) / max(seg_len, 1e-9)
            return [p0[i] + t * (p1[i] - p0[i]) for i in range(3)]
        acc += seg_len
    return polyline[-1]


def collect_marker_points(weld_data: dict[str, Any]) -> tuple[list[list[float]], list[list[float]]]:
    endpoints: list[list[float]] = []
    breakpoints: list[list[float]] = []

    points = weld_data.get("points")
    if isinstance(points, dict) and points:
        for point in points.values():
            if not isinstance(point, dict):
                continue
            xyz = as_xyz(point.get("xyz"))
            if xyz is None:
                continue
            if point.get("role") == "breakpoint":
                breakpoints.append(xyz)
            else:
                endpoints.append(xyz)
        return dedup_points(endpoints), dedup_points(breakpoints)

    for edge in (weld_data.get("contact_edges") or {}).values():
        if not isinstance(edge, dict):
            continue
        start = as_xyz(edge.get("start"))
        end = as_xyz(edge.get("end"))
        if start:
            endpoints.append(start)
        if end:
            endpoints.append(end)

    return dedup_points(endpoints), breakpoints


def collect_pose_normals(
    weld_data: dict[str, Any],
    role: str | None = None,
) -> list[tuple[list[float], list[float], dict[str, Any]]]:
    vectors: list[tuple[list[float], list[float], dict[str, Any]]] = []
    points = weld_data.get("points")
    if not isinstance(points, dict):
        return vectors

    for point_id, point in points.items():
        if not isinstance(point, dict):
            continue
        if role is not None and point.get("role") != role:
            continue
        xyz = as_xyz(point.get("xyz"))
        normal = as_xyz(point.get("pose_normal"))
        if xyz is None or normal is None:
            continue
        vectors.append((xyz, normal, {"point_id": str(point_id), **point}))
    return vectors


def collect_pose_component_vectors(
    weld_data: dict[str, Any],
    component_indices: list[int],
) -> dict[int, list[tuple[list[float], list[float], dict[str, Any]]]]:
    out: dict[int, list[tuple[list[float], list[float], dict[str, Any]]]] = {1: [], 2: [], 3: []}
    if not component_indices:
        return out

    points = weld_data.get("points")
    if not isinstance(points, dict):
        return out

    selected_set = set(component_indices)
    for point_id, point in points.items():
        if not isinstance(point, dict):
            continue
        origin = as_xyz(point.get("xyz"))
        if origin is None:
            continue
        comps = point.get("selected_face_normals")
        if not isinstance(comps, list):
            continue
        for idx, item in enumerate(comps[:3], start=1):
            if idx not in selected_set:
                continue
            if not isinstance(item, dict):
                continue
            vec = as_xyz(item.get("normal"))
            if vec is None:
                continue
            out[idx].append((origin, vec, {"point_id": str(point_id), **point, "component_index": idx, "component": item}))
    return out

def dedup_points(points: list[list[float]], ndigits: int = 6) -> list[list[float]]:
    seen: set[tuple[float, float, float]] = set()
    unique: list[list[float]] = []
    for point in points:
        key = tuple(round(coord, ndigits) for coord in point)
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
    return unique


def infer_stl_path(step_path: Path, stl_path: Path | None) -> Path | None:
    if stl_path is not None:
        return stl_path.resolve()

    candidate = step_path.with_suffix(".stl")
    if candidate.exists():
        return candidate.resolve()

    upper_candidate = step_path.with_suffix(".STL")
    if upper_candidate.exists():
        return upper_candidate.resolve()

    return None


def load_stl_triangles(path: Path) -> list[list[list[float]]]:
    data = path.read_bytes()
    if len(data) >= 84:
        tri_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + tri_count * 50
        if expected_size == len(data):
            triangles: list[list[list[float]]] = []
            offset = 84
            for _ in range(tri_count):
                offset += 12  # normal
                p0 = list(struct.unpack_from("<fff", data, offset))
                p1 = list(struct.unpack_from("<fff", data, offset + 12))
                p2 = list(struct.unpack_from("<fff", data, offset + 24))
                triangles.append([p0, p1, p2])
                offset += 38
            return triangles

    triangles = []
    vertices = []
    for raw_line in data.decode("utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
            if len(vertices) == 3:
                triangles.append(vertices)
                vertices = []
    return triangles


def weld_polylines(weld_data: dict[str, Any]) -> list[list[list[float]]]:
    polylines: list[list[list[float]]] = []
    for edge in (weld_data.get("contact_edges") or {}).values():
        if not isinstance(edge, dict):
            continue
        polylines.extend(iter_edge_samples(edge))
    return polylines


def weld_edge_records(weld_data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for edge_id, edge in (weld_data.get("contact_edges") or {}).items():
        if not isinstance(edge, dict):
            continue
        for polyline in iter_edge_samples(edge):
            records.append({
                "edge_id": str(edge_id),
                "polyline": polyline,
            })
    return records


def set_axes_equal(ax: Any, points: list[list[float]]) -> None:
    if not points:
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
    if max_range <= 1e-9:
        max_range = 1.0

    xmid = (xmin + xmax) * 0.5
    ymid = (ymin + ymax) * 0.5
    zmid = (zmin + zmax) * 0.5
    half = max_range * 0.5

    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    ax.set_zlim(zmid - half, zmid + half)


def visualize_with_matplotlib(args: argparse.Namespace, weld_data: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    step_path = args.step.resolve()
    stl_path = infer_stl_path(step_path, args.stl)
    triangles: list[list[list[float]]] = []
    if stl_path and stl_path.exists():
        triangles = load_stl_triangles(stl_path)
        print(f"[visualize] loaded {len(triangles)} STL triangles from {stl_path}")
    else:
        print("[visualize] STL not found; plotting welds without model surface.")

    weld_records = weld_edge_records(weld_data)
    endpoints, breakpoints = collect_marker_points(weld_data)
    endpoint_pose_normals = collect_pose_normals(weld_data, role="endpoint") if not args.no_pose_normals else []
    breakpoint_pose_normals = collect_pose_normals(weld_data, role="breakpoint") if not args.no_pose_normals else []
    component_indices = [1, 2, 3]
    show_pose_components = args.show_pose_components
    pose_components = (
        collect_pose_component_vectors(weld_data, component_indices)
        if show_pose_components
        else {1: [], 2: [], 3: []}
    )

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{step_path.name} + {args.weld_json.name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    all_points: list[list[float]] = []
    if triangles:
        mesh = Poly3DCollection(
            triangles,
            facecolor=(0.62, 0.66, 0.70, max(0.0, min(1.0, 1.0 - args.model_transparency))),
            edgecolor=(0.35, 0.38, 0.42, 0.12),
            linewidths=0.15,
        )
        ax.add_collection3d(mesh)
        for tri in triangles:
            all_points.extend(tri)

    for record in weld_records:
        polyline = record["polyline"]
        xs = [p[0] for p in polyline]
        ys = [p[1] for p in polyline]
        zs = [p[2] for p in polyline]
        ax.plot(
            xs,
            ys,
            zs,
            color=WELD_COLOR,
            linewidth=args.weld_width,
            solid_capstyle="round",
        )
        all_points.extend(polyline)

    if not args.no_points and args.point_radius > 0:
        if args.show_endpoints and endpoints:
            ax.scatter(
                [p[0] for p in endpoints],
                [p[1] for p in endpoints],
                [p[2] for p in endpoints],
                s=(args.point_radius * 8) ** 2,
                color="royalblue",
                depthshade=True,
                label="endpoints",
            )
            all_points.extend(endpoints)
        if breakpoints:
            ax.scatter(
                [p[0] for p in breakpoints],
                [p[1] for p in breakpoints],
                [p[2] for p in breakpoints],
                s=(args.point_radius * 10) ** 2,
                color="orange",
                marker="^",
                depthshade=True,
                label="breakpoints",
            )
            all_points.extend(breakpoints)

    if show_pose_components and component_indices:
        component_colors = {
            1: "deepskyblue",
            2: "gold",
            3: "magenta",
        }
        for idx in component_indices:
            vectors = pose_components.get(idx, [])
            if not vectors:
                continue
            normals = [item[1] for item in vectors]
            origins = [
                [
                    item[0][0] + item[1][0] * args.normal_offset,
                    item[0][1] + item[1][1] * args.normal_offset,
                    item[0][2] + item[1][2] * args.normal_offset,
                ]
                for item in vectors
            ]
            ax.quiver(
                [p[0] for p in origins],
                [p[1] for p in origins],
                [p[2] for p in origins],
                [n[0] for n in normals],
                [n[1] for n in normals],
                [n[2] for n in normals],
                length=args.component_length,
                normalize=True,
                color=component_colors[idx],
                linewidth=1.0,
                arrow_length_ratio=0.22,
                label=f"pose component {idx}",
            )
            for origin, normal in zip(origins, normals):
                all_points.append(origin)
                all_points.append(
                    [
                        origin[0] + normal[0] * args.component_length,
                        origin[1] + normal[1] * args.component_length,
                        origin[2] + normal[2] * args.component_length,
                    ]
                )

    if not args.no_pose_normals:
        if endpoint_pose_normals:
            normals = [item[1] for item in endpoint_pose_normals]
            origins = [
                [
                    item[0][0] + item[1][0] * args.normal_offset,
                    item[0][1] + item[1][1] * args.normal_offset,
                    item[0][2] + item[1][2] * args.normal_offset,
                ]
                for item in endpoint_pose_normals
            ]
            ax.quiver(
                [p[0] for p in origins],
                [p[1] for p in origins],
                [p[2] for p in origins],
                [n[0] for n in normals],
                [n[1] for n in normals],
                [n[2] for n in normals],
                length=args.normal_length,
                normalize=True,
                color=args.normal_color,
                linewidth=1.25,
                arrow_length_ratio=0.25,
                label="endpoint pose normals",
            )
            for origin, normal in zip(origins, normals):
                all_points.append(origin)
                all_points.append(
                    [
                        origin[0] + normal[0] * args.normal_length,
                        origin[1] + normal[1] * args.normal_length,
                        origin[2] + normal[2] * args.normal_length,
                    ]
                )
        if breakpoint_pose_normals:
            normals = [item[1] for item in breakpoint_pose_normals]
            origins = [
                [
                    item[0][0] + item[1][0] * args.normal_offset,
                    item[0][1] + item[1][1] * args.normal_offset,
                    item[0][2] + item[1][2] * args.normal_offset,
                ]
                for item in breakpoint_pose_normals
            ]
            ax.quiver(
                [p[0] for p in origins],
                [p[1] for p in origins],
                [p[2] for p in origins],
                [n[0] for n in normals],
                [n[1] for n in normals],
                [n[2] for n in normals],
                length=args.normal_length,
                normalize=True,
                color="darkorange",
                linewidth=1.35,
                arrow_length_ratio=0.25,
                label="breakpoint pose normals",
            )
            for origin, normal in zip(origins, normals):
                all_points.append(origin)
                all_points.append(
                    [
                        origin[0] + normal[0] * args.normal_length,
                        origin[1] + normal[1] * args.normal_length,
                        origin[2] + normal[2] * args.normal_length,
                    ]
                )
        if not endpoint_pose_normals and not breakpoint_pose_normals:
            print("[visualize] no points[*].pose_normal vectors found in weld JSON.")

    set_axes_equal(ax, all_points)
    ax.view_init(elev=24, azim=-55)
    has_components = any(pose_components.get(i) for i in (1, 2, 3))
    if (args.show_endpoints and endpoints) or breakpoints or endpoint_pose_normals or breakpoint_pose_normals or has_components:
        ax.legend()
    plt.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=220)
        print(f"[visualize] saved image -> {args.save}")

    print(
        "[visualize] displayed "
        f"{len(weld_records)} weld polylines "
        f"(endpoint_pose_normals={len(endpoint_pose_normals)}, "
        f"breakpoint_pose_normals={len(breakpoint_pose_normals)})"
    )
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def display_step_model(display: Any, step_path: Path, transparency: float) -> Any:
    from OCC.Display.OCCViewer import rgb_color
    from OCC.Extend.TopologyUtils import TopologyExplorer

    from src.geometry.step_loader import load_step_file

    shape = load_step_file(str(step_path))
    if shape is None:
        raise RuntimeError(f"Failed to load STEP file: {step_path}")

    explorer = TopologyExplorer(shape)
    solid_count = 0
    for solid in explorer.solids():
        display.DisplayShape(
            solid,
            color=rgb_color(0.62, 0.66, 0.70),
            transparency=transparency,
            update=False,
        )
        solid_count += 1

    print(f"[visualize] displayed {solid_count} solids from {step_path}")
    return shape


def display_weld_edges(
    display: Any,
    weld_data: dict[str, Any],
    line_width: float,
) -> int:
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.gp import gp_Pnt
    from OCC.Display.OCCViewer import rgb_color

    displayed = 0
    for record in weld_edge_records(weld_data):
        polyline = record["polyline"]
        color = rgb_color(*WELD_RGB)

        for p0, p1 in zip(polyline, polyline[1:]):
            occ_edge = BRepBuilderAPI_MakeEdge(
                gp_Pnt(*p0),
                gp_Pnt(*p1),
            ).Edge()
            display.DisplayShape(
                occ_edge,
                color=color,
                linewidth=line_width,
                update=False,
            )
            displayed += 1

    print(f"[visualize] displayed {displayed} weld segments")
    return displayed


def display_markers(
    display: Any,
    endpoints: list[list[float]],
    breakpoints: list[list[float]],
    radius: float,
    show_endpoints: bool,
) -> None:
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCC.Core.gp import gp_Pnt
    from OCC.Display.OCCViewer import rgb_color

    if show_endpoints:
        for point in endpoints:
            marker = BRepPrimAPI_MakeSphere(gp_Pnt(*point), radius).Shape()
            display.DisplayShape(
                marker,
                color=rgb_color(0.05, 0.25, 1.0),
                transparency=0.0,
                update=False,
            )

    for point in breakpoints:
        marker = BRepPrimAPI_MakeSphere(gp_Pnt(*point), radius * 1.3).Shape()
        display.DisplayShape(
            marker,
            color=rgb_color(1.0, 0.55, 0.0),
            transparency=0.0,
            update=False,
        )

    print(
        f"[visualize] displayed {len(endpoints) if show_endpoints else 0} endpoints, "
        f"{len(breakpoints)} breakpoints"
    )


def display_pose_normals(
    display: Any,
    pose_normals: list[tuple[list[float], list[float], dict[str, Any]]],
    length: float,
    offset: float,
    color_rgb: tuple[float, float, float] = (0.05, 0.80, 0.20),
    label: str = "pose_normal",
) -> None:
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.gp import gp_Pnt
    from OCC.Display.OCCViewer import rgb_color

    color = rgb_color(*color_rgb)
    displayed = 0
    for origin, normal, _meta in pose_normals:
        shifted_origin = [
            origin[0] + normal[0] * offset,
            origin[1] + normal[1] * offset,
            origin[2] + normal[2] * offset,
        ]
        end = [
            shifted_origin[0] + normal[0] * length,
            shifted_origin[1] + normal[1] * length,
            shifted_origin[2] + normal[2] * length,
        ]
        occ_edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*shifted_origin), gp_Pnt(*end)).Edge()
        display.DisplayShape(
            occ_edge,
            color=color,
            linewidth=2.0,
            update=False,
        )
        displayed += 1

    print(f"[visualize] displayed {displayed} {label} vectors")


def display_pose_component_normals(
    display: Any,
    pose_components: dict[int, list[tuple[list[float], list[float], dict[str, Any]]]],
    component_indices: list[int],
    length: float,
    offset: float,
) -> None:
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.gp import gp_Pnt
    from OCC.Display.OCCViewer import rgb_color

    color_map = {
        1: rgb_color(0.10, 0.75, 1.00),
        2: rgb_color(1.00, 0.85, 0.10),
        3: rgb_color(1.00, 0.10, 0.90),
    }
    total = 0
    for idx in component_indices:
        vectors = pose_components.get(idx, [])
        color = color_map[idx]
        for origin, normal, _meta in vectors:
            shifted_origin = [
                origin[0] + normal[0] * offset,
                origin[1] + normal[1] * offset,
                origin[2] + normal[2] * offset,
            ]
            end = [
                shifted_origin[0] + normal[0] * length,
                shifted_origin[1] + normal[1] * length,
                shifted_origin[2] + normal[2] * length,
            ]
            occ_edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*shifted_origin), gp_Pnt(*end)).Edge()
            display.DisplayShape(
                occ_edge,
                color=color,
                linewidth=1.5,
                update=False,
            )
            total += 1
    print(f"[visualize] displayed {total} pose component vectors")

def visualize_with_occ(args: argparse.Namespace, weld_data: dict[str, Any]) -> None:
    step_path = args.step.resolve()

    ensure_seam_extract_on_path()
    install_occ_import_alias_if_needed()

    from OCC.Display.SimpleGui import init_display

    display, start_display, _add_menu, _add_function_to_menu = init_display()

    display_step_model(display, step_path, args.model_transparency)
    display_weld_edges(display, weld_data, args.weld_width)

    if not args.no_points and args.point_radius > 0:
        endpoints, breakpoints = collect_marker_points(weld_data)
        display_markers(display, endpoints, breakpoints, args.point_radius, args.show_endpoints)

    if not args.no_pose_normals:
        endpoint_pose_normals = collect_pose_normals(weld_data, role="endpoint")
        breakpoint_pose_normals = collect_pose_normals(weld_data, role="breakpoint")
        if endpoint_pose_normals:
            display_pose_normals(
                display,
                endpoint_pose_normals,
                args.normal_length,
                args.normal_offset,
                color_rgb=(0.05, 0.80, 0.20),
                label="endpoint pose_normal",
            )
        if breakpoint_pose_normals:
            display_pose_normals(
                display,
                breakpoint_pose_normals,
                args.normal_length,
                args.normal_offset,
                color_rgb=(1.0, 0.45, 0.0),
                label="breakpoint pose_normal",
            )
        if not endpoint_pose_normals and not breakpoint_pose_normals:
            print("[visualize] no points[*].pose_normal vectors found in weld JSON.")

    show_pose_components = args.show_pose_components
    if show_pose_components:
        component_indices = [1, 2, 3]
        pose_components = collect_pose_component_vectors(weld_data, component_indices)
        if any(pose_components.get(i) for i in (1, 2, 3)):
            display_pose_component_normals(
                display,
                pose_components,
                component_indices,
                args.component_length,
                args.normal_offset,
            )
        else:
            print("[visualize] no pose component vectors found for selected indices.")

    display.FitAll()
    display.View_Iso()
    print(
        "[visualize] red lines=welds, green lines=pose_normal, "
        "orange lines=breakpoint pose_normal, blue spheres=endpoints, "
        "orange spheres=breakpoints"
    )
    start_display()


def main() -> None:
    args = parse_args()
    step_path, weld_json_path = autodetect_inputs(args)
    args.step = step_path
    args.weld_json = weld_json_path

    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")
    if not weld_json_path.exists():
        raise FileNotFoundError(f"Weld JSON not found: {weld_json_path}")

    weld_data = load_json(weld_json_path)
    
    # Convert weld_vectors.json format to contact_edges format if needed
    if "welds" in weld_data and "contact_edges" not in weld_data:
        print("[visualize] converting weld_vectors.json format to contact_edges format")
        weld_data = convert_weld_vectors_to_contact_edges(weld_data)
    
    if args.backend == "occ":
        visualize_with_occ(args, weld_data)
    else:
        visualize_with_matplotlib(args, weld_data)


if __name__ == "__main__":
    main()
