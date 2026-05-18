from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
SEAM_EXTRACT_DIR = ROOT / "seam_extract"
if str(SEAM_EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(SEAM_EXTRACT_DIR))


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


install_occ_import_alias_if_needed()

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # noqa: E402
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier  # noqa: E402
from OCC.Core.Bnd import Bnd_Box  # noqa: E402
from OCC.Core.BRepBndLib import brepbndlib_Add  # noqa: E402
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf  # noqa: E402
from OCC.Core.GeomAbs import GeomAbs_Plane  # noqa: E402
from OCC.Core.GeomLProp import GeomLProp_SLProps  # noqa: E402
from OCC.Core.gp import gp_Pnt, gp_Vec  # noqa: E402
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_REVERSED  # noqa: E402
from OCC.Core.TopExp import TopExp_Explorer  # noqa: E402
from OCC.Core.TopoDS import topods  # noqa: E402
from OCC.Extend.TopologyUtils import TopologyExplorer  # noqa: E402
from src.geometry.step_loader import load_step_file  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def as_xyz(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) == 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None


def unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return None
    return vec / norm


def vec_key(vec: np.ndarray, ndigits: int = 5) -> tuple[float, float, float]:
    u = unit(vec)
    if u is None:
        return (0.0, 0.0, 0.0)
    return tuple(round(float(x), ndigits) for x in u)


def infer_paths(final_json: Path) -> tuple[Path | None, Path | None]:
    stem = final_json.stem
    extract_dir = ROOT / "data" / "extract" / stem
    geometry_graph = extract_dir / f"{stem}_geometry_graph_with_breakpoints.json"
    if not geometry_graph.exists():
        geometry_graph = extract_dir / f"{stem}_geometry_graph.json"
    process_graph = extract_dir / f"{stem}_process_graph.json"
    return (
        geometry_graph if geometry_graph.exists() else None,
        process_graph if process_graph.exists() else None,
    )


def collect_final_points(final_obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    points = final_obj.get("points")
    if isinstance(points, dict) and points:
        return {str(pid): pdata for pid, pdata in points.items() if isinstance(pdata, dict)}

    collected: dict[str, dict[str, Any]] = {}
    for edge in (final_obj.get("contact_edges") or {}).values():
        if not isinstance(edge, dict):
            continue
        for endpoint_name in ("start", "end"):
            xyz = as_xyz(edge.get(endpoint_name))
            if xyz is None:
                continue
            pid = f"P{len(collected) + 1}"
            collected[pid] = {"id": pid, "xyz": xyz, "role": "endpoint"}
    return collected


def build_point_incidence(final_obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    incidence: dict[str, dict[str, Any]] = {}
    for edge_id, edge in (final_obj.get("contact_edges") or {}).items():
        if not isinstance(edge, dict):
            continue
        solid_ids = edge.get("solid_ids") if isinstance(edge.get("solid_ids"), list) else []
        for point_ref in edge.get("points") or []:
            if not isinstance(point_ref, dict):
                continue
            point_id = point_ref.get("point_id")
            if point_id is None:
                continue
            entry = incidence.setdefault(str(point_id), {"contact_edges": [], "solid_ids": set()})
            entry["contact_edges"].append(str(edge_id))
            for solid_id in solid_ids:
                try:
                    entry["solid_ids"].add(int(solid_id))
                except (TypeError, ValueError):
                    pass
    return incidence


def face_normals_at_point(shape: Any, xyz: list[float], tol: float) -> list[dict[str, Any]]:
    point = gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    faces = []
    if shape.ShapeType() == TopAbs_FACE:
        faces = [topods.Face(shape)]
    else:
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            faces.append(topods.Face(explorer.Current()))
            explorer.Next()

    normals: list[dict[str, Any]] = []
    for face_index, face in enumerate(faces, start=1):
        surface_adaptor = BRepAdaptor_Surface(face)
        surface_type = surface_adaptor.GetType()
        surface = surface_adaptor.Surface().Surface()
        projection = GeomAPI_ProjectPointOnSurf(point, surface)
        if projection.NbPoints() < 1:
            continue

        distance = float(projection.LowerDistance())
        if distance > tol * 20.0:
            continue

        u, v = projection.LowerDistanceParameters()
        umin, umax = surface_adaptor.FirstUParameter(), surface_adaptor.LastUParameter()
        vmin, vmax = surface_adaptor.FirstVParameter(), surface_adaptor.LastVParameter()
        if not (umin - tol <= u <= umax + tol and vmin - tol <= v <= vmax + tol):
            continue

        props = GeomLProp_SLProps(surface, u, v, 1, 1e-6)
        if not props.IsNormalDefined():
            continue

        normal_dir = props.Normal()
        normal_vec = gp_Vec(normal_dir.XYZ())
        if face.Orientation() == TopAbs_REVERSED:
            normal_vec.Reverse()
        normal = np.array([normal_vec.X(), normal_vec.Y(), normal_vec.Z()], dtype=float)
        normal_unit = unit(normal)
        if normal_unit is None:
            continue
        normals.append(
            {
                "face_index": face_index,
                "is_plane": surface_type == GeomAbs_Plane,
                "normal": [float(x) for x in normal_unit],
                "distance": distance,
            }
        )
    return normals


def collect_unique_normals(
    solids: list[Any],
    xyz: list[float],
    solid_ids: set[int],
    tol: float,
    classifiers: list[Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    ids = sorted(solid_ids) if solid_ids else list(range(1, len(solids) + 1))
    for solid_id in ids:
        if solid_id < 1 or solid_id > len(solids):
            continue
        for item in face_normals_at_point(solids[solid_id - 1], xyz, tol):
            normal = np.array(item["normal"], dtype=float)
            
            # 如果面被其他实体完全覆盖，法线延长线会立刻进入其他实体内部，将其过滤
            if classifiers is not None:
                test_pt = np.array(xyz, dtype=float) + normal * 0.1
                p_occ = gp_Pnt(float(test_pt[0]), float(test_pt[1]), float(test_pt[2]))
                is_covered = False
                for clf in classifiers:
                    clf.Perform(p_occ, 1e-5)
                    if clf.State() == TopAbs_IN:
                        is_covered = True
                        break
                if is_covered:
                    continue

            item = dict(item)
            item["solid_id"] = solid_id
            item["_key"] = vec_key(normal)
            candidates.append(item)

    by_key: dict[tuple[float, float, float], dict[str, Any]] = {}
    for item in candidates:
        key = item["_key"]
        old = by_key.get(key)
        if old is None or item["distance"] < old["distance"]:
            by_key[key] = item

    result = []
    for item in by_key.values():
        clean = {k: v for k, v in item.items() if k != "_key"}
        result.append(clean)
    result.sort(key=lambda x: (float(x["distance"]), int(x["solid_id"]), int(x["face_index"])))
    return result


def solid_bounds(shape: Any) -> tuple[float, float, float, float, float, float]:
    box = Bnd_Box()
    brepbndlib_Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax))


def classify_solids(solids: list[Any]) -> dict[int, dict[str, Any]]:
    infos: dict[int, dict[str, Any]] = {}
    for solid_id, solid in enumerate(solids, start=1):
        xmin, xmax, ymin, ymax, zmin, zmax = solid_bounds(solid)
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        infos[solid_id] = {
            "solid_id": solid_id,
            "bounds": [xmin, xmax, ymin, ymax, zmin, zmax],
            "center": [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5],
            "size": [dx, dy, dz],
            "role": "unknown",
        }

    if not infos:
        return infos

    plate_id = max(
        infos,
        key=lambda sid: (infos[sid]["size"][0] * infos[sid]["size"][1], -infos[sid]["size"][2]),
    )
    infos[plate_id]["role"] = "base_plate"

    rib_ids = [sid for sid in infos if sid != plate_id]
    if not rib_ids:
        return infos

    main_id = max(rib_ids, key=lambda sid: max(infos[sid]["size"][0], infos[sid]["size"][1]))
    infos[main_id]["role"] = "main_rib"
    main_size = infos[main_id]["size"]
    main_axis = "X" if main_size[0] >= main_size[1] else "Y"

    for sid in rib_ids:
        if sid == main_id:
            continue
        size = infos[sid]["size"]
        long_axis = "X" if size[0] >= size[1] else "Y"
        infos[sid]["role"] = "bridge_rib" if long_axis == main_axis else "side_rib"
    return infos


def role_for_incident_solids(solid_ids: set[int], solid_infos: dict[int, dict[str, Any]]) -> str | None:
    roles = [solid_infos.get(sid, {}).get("role") for sid in solid_ids]
    if "side_rib" in roles:
        return "side_rib"
    if "main_rib" in roles:
        return "main_rib"
    return None


def nearest_face_normal_on_solid(shape: Any, xyz: list[float], solid_id: int, classifiers: list[Any] | None = None) -> dict[str, Any] | None:
    point = gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    best: dict[str, Any] | None = None
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while explorer.More():
        face_index += 1
        face = topods.Face(explorer.Current())
        explorer.Next()

        surface_adaptor = BRepAdaptor_Surface(face)
        surface = surface_adaptor.Surface().Surface()
        projection = GeomAPI_ProjectPointOnSurf(point, surface)
        if projection.NbPoints() < 1:
            continue

        distance = float(projection.LowerDistance())
        u, v = projection.LowerDistanceParameters()
        umin, umax = surface_adaptor.FirstUParameter(), surface_adaptor.LastUParameter()
        vmin, vmax = surface_adaptor.FirstVParameter(), surface_adaptor.LastVParameter()
        u_clamped = max(float(umin), min(float(umax), float(u)))
        v_clamped = max(float(vmin), min(float(vmax), float(v)))

        props = GeomLProp_SLProps(surface, u_clamped, v_clamped, 1, 1e-6)
        if not props.IsNormalDefined():
            continue

        normal_dir = props.Normal()
        normal_vec = gp_Vec(normal_dir.XYZ())
        if face.Orientation() == TopAbs_REVERSED:
            normal_vec.Reverse()
        normal = unit(np.array([normal_vec.X(), normal_vec.Y(), normal_vec.Z()], dtype=float))
        if normal is None:
            continue

        if classifiers is not None:
            test_pt = np.array(xyz, dtype=float) + normal * 0.1
            p_occ = gp_Pnt(float(test_pt[0]), float(test_pt[1]), float(test_pt[2]))
            is_covered = False
            for clf in classifiers:
                clf.Perform(p_occ, 1e-5)
                if clf.State() == TopAbs_IN:
                    is_covered = True
                    break
            if is_covered:
                continue

        item = {
            "solid_id": solid_id,
            "face_index": face_index,
            "normal": [float(x) for x in normal],
            "distance": distance,
            "fallback": True,
        }
        if best is None or distance < float(best["distance"]):
            best = item
    return best


def select_lower_level_fallback_face(
    solids: list[Any],
    solid_infos: dict[int, dict[str, Any]],
    xyz: list[float],
    solid_ids: set[int],
    selected: list[dict[str, Any]],
    classifiers: list[Any] | None = None,
) -> dict[str, Any] | None:
    source_role = role_for_incident_solids(solid_ids, solid_infos)
    target_role = None
    if source_role == "main_rib":
        target_role = "side_rib"
    elif source_role == "side_rib":
        target_role = "bridge_rib"
    if target_role is None:
        return None

    selected_keys = {vec_key(np.array(item["normal"], dtype=float)) for item in selected}
    best: dict[str, Any] | None = None
    for sid, info in solid_infos.items():
        if info.get("role") != target_role or sid in solid_ids:
            continue
        item = nearest_face_normal_on_solid(solids[sid - 1], xyz, sid, classifiers=classifiers)
        if item is None:
            continue
        if vec_key(np.array(item["normal"], dtype=float)) in selected_keys:
            continue
        item["fallback_role"] = target_role
        item["fallback_source_role"] = source_role
        if best is None or float(item["distance"]) < float(best["distance"]):
            best = item
    return best


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    ua = unit(a)
    ub = unit(b)
    if ua is None or ub is None:
        return math.inf
    dot = max(-1.0, min(1.0, float(np.dot(ua, ub))))
    return float(math.acos(dot))


def point_on_arc(xyz: list[float], edge: dict[str, Any], tol: float) -> bool:
    center = as_xyz(edge.get("center"))
    start = as_xyz(edge.get("start"))
    end = as_xyz(edge.get("end"))
    radius = edge.get("radius")
    if center is None or start is None or end is None or radius is None:
        return False

    p = np.array(xyz, dtype=float)
    c = np.array(center, dtype=float)
    s = np.array(start, dtype=float)
    e = np.array(end, dtype=float)
    r = float(radius)

    vs = s - c
    ve = e - c
    normal = unit(np.cross(vs, ve))
    if normal is None:
        return False

    vp = p - c
    plane_distance = abs(float(np.dot(vp, normal)))
    vp_in_plane = vp - np.dot(vp, normal) * normal
    radius_error = abs(float(np.linalg.norm(vp_in_plane)) - r)
    if plane_distance > tol or radius_error > tol:
        return False

    total = angle_between(vs, ve)
    part_a = angle_between(vs, vp_in_plane)
    part_b = angle_between(vp_in_plane, ve)
    return abs((part_a + part_b) - total) <= max(1e-1, tol / max(r, 1e-6))


def load_through_hole_arcs(
    geometry_graph: dict[str, Any] | None,
    process_graph: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any]]]:
    if not geometry_graph:
        return []

    process_edges = (process_graph or {}).get("geom_edges", {}) if isinstance(process_graph, dict) else {}
    result = []
    for edge_id, edge in (geometry_graph.get("geom_edges") or {}).items():
        if not isinstance(edge, dict) or edge.get("type") != "arc":
            continue
        process = process_edges.get(str(edge_id), {}).get("process", {})
        if process.get("through_hole_candidate") is True:
            result.append((str(edge_id), edge))
    return result


def find_hole_arc_for_point(
    xyz: list[float],
    hole_arcs: list[tuple[str, dict[str, Any]]],
    tol: float,
) -> tuple[str, dict[str, Any]] | None:
    for edge_id, edge in hole_arcs:
        if point_on_arc(xyz, edge, tol):
            return edge_id, edge
    return None


def infer_up_axis(final_obj: dict[str, Any]) -> np.ndarray:
    up_axis = final_obj.get("up_axis")
    xyz = as_xyz(up_axis)
    if xyz is not None:
        u = unit(np.array(xyz, dtype=float))
        if u is not None:
            return u
    return np.array([0.0, 0.0, 1.0], dtype=float)


def select_regular_normals(normals: list[dict[str, Any]], up_axis: np.ndarray) -> list[dict[str, Any]]:
    """
    For a regular non-through-hole contact point, prefer one base face plus two vertical side faces.
    This matches the physical junction: base plate top face + two rib side faces.
    """
    if not normals:
        return []

    base_candidates = []
    side_candidates = []
    for item in normals:
        n = np.array(item["normal"], dtype=float)
        up_dot = float(np.dot(n, up_axis))
        abs_up_dot = abs(up_dot)
        if up_dot > 0.65:
            base_candidates.append((up_dot, float(item["distance"]), item))
        elif abs_up_dot < 0.45:
            side_candidates.append((abs_up_dot, float(item["distance"]), item))

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[float, float, float]] = set()

    if base_candidates:
        base = max(base_candidates, key=lambda x: (x[0], -x[1]))[2]
        selected.append(base)
        selected_keys.add(vec_key(np.array(base["normal"], dtype=float)))

    side_candidates.sort(key=lambda x: (x[0], x[1]))
    for _score, _distance, item in side_candidates:
        key = vec_key(np.array(item["normal"], dtype=float))
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= 3:
            return selected

    for item in normals:
        key = vec_key(np.array(item["normal"], dtype=float))
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= 3:
            break

    return selected


def select_hole_normals(normals: list[dict[str, Any]], up_axis: np.ndarray) -> list[dict[str, Any]]:
    if not normals:
        return []
    scored = []
    for item in normals:
        n = np.array(item["normal"], dtype=float)
        scored.append((float(np.dot(n, up_axis)), item))
    base = max(scored, key=lambda x: x[0])[1]
    base_key = vec_key(np.array(base["normal"], dtype=float))

    side_candidates = []
    for item in normals:
        n = np.array(item["normal"], dtype=float)
        if vec_key(n) == base_key:
            continue
        side_score = abs(float(np.dot(n, up_axis)))
        side_candidates.append((side_score, item))
    side_candidates.sort(key=lambda x: (x[0], float(x[1]["distance"])))
    return [base] + [item for _, item in side_candidates[:2]]


def select_hole_side_normals(
    solids: list[Any],
    solid_infos: dict[int, dict[str, Any]],
    xyz: list[float],
    incident_solid_ids: set[int],
    incident_edge_ids: list[str],
    contact_edges: dict[str, Any],
    normals: list[dict[str, Any]],
    up_axis: np.ndarray,
    classifiers: list[Any] | None = None,
) -> list[dict[str, Any]]:
    p = np.array(xyz, dtype=float)

    base_id = next((sid for sid, info in solid_infos.items() if info.get("role") == "base_plate"), None)
    if base_id is None or base_id < 1 or base_id > len(solids):
        return []

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[float, float, float]] = set()

    def nearest_item_from_normals(sid: int) -> dict[str, Any] | None:
        candidates = [item for item in normals if int(item.get("solid_id", -1)) == int(sid)]
        if not candidates:
            return None
        return min(candidates, key=lambda item: float(item.get("distance", float("inf"))))

    base_candidates = [item for item in normals if int(item.get("solid_id", -1)) == int(base_id)]
    if base_candidates:
        base = max(
            base_candidates,
            key=lambda item: (float(np.dot(np.array(item["normal"], dtype=float), up_axis)), -float(item["distance"])),
        )
    else:
        base = nearest_face_normal_on_solid(solids[base_id - 1], xyz, base_id, classifiers=classifiers)
    if base is None:
        return []
    base = dict(base)
    base["source"] = "hole_side_base_plate"
    selected.append(base)
    selected_keys.add(vec_key(np.array(base["normal"], dtype=float)))

    rib_roles = {"main_rib", "side_rib", "bridge_rib"}
    contact_rib_ids: list[int] = []
    for sid in sorted(incident_solid_ids):
        role = solid_infos.get(sid, {}).get("role")
        if role in rib_roles and 1 <= sid <= len(solids):
            contact_rib_ids.append(sid)

    contact_rib_ids.sort(
        key=lambda sid: float(
            np.linalg.norm(np.array(solid_infos.get(sid, {}).get("center", [0.0, 0.0, 0.0]), dtype=float) - p)
        )
    )

    selected_rib_ids = contact_rib_ids[:2]

    if len(contact_rib_ids) == 1:
        weld_vec_item = longest_incident_weld_direction(xyz, incident_edge_ids, contact_edges)
        if weld_vec_item is not None:
            selected.append(weld_vec_item)
            selected_keys.add(vec_key(np.array(weld_vec_item["normal"], dtype=float)))

    for sid in selected_rib_ids:
        item = nearest_item_from_normals(sid)
        if item is None:
            item = nearest_face_normal_on_solid(solids[sid - 1], xyz, sid, classifiers=classifiers)
        if item is None and classifiers is not None:
            item = nearest_face_normal_on_solid(solids[sid - 1], xyz, sid, classifiers=None)
        if item is None:
            continue
        role = solid_infos.get(sid, {}).get("role")
        key = vec_key(np.array(item["normal"], dtype=float))
        if key in selected_keys:
            continue
        item = dict(item)
        if sid in contact_rib_ids:
            item["source"] = "hole_contact_rib"
        else:
            item["source"] = "hole_upper_level_rib"
        item["solid_role"] = role
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= 3:
            break

    return selected if len(selected) >= 3 else []


def select_hole_non_base_plane_normals(
    solids: list[Any],
    solid_infos: dict[int, dict[str, Any]],
    xyz: list[float],
    incident_solid_ids: set[int],
    incident_edge_ids: list[str],
    contact_edges: dict[str, Any],
    normals: list[dict[str, Any]],
    up_axis: np.ndarray,
    classifiers: list[Any] | None = None,
) -> list[dict[str, Any]]:
    # 获取两条焊缝的方向向量
    p = np.array(xyz, dtype=float)
    edge_items: list[tuple[float, str, np.ndarray]] = []
    
    for edge_id in incident_edge_ids:
        edge = contact_edges.get(str(edge_id))
        if not isinstance(edge, dict):
            continue
        s_xyz = as_xyz(edge.get("start"))
        t_xyz = as_xyz(edge.get("end"))
        if s_xyz is None or t_xyz is None:
            continue
        s = np.array(s_xyz, dtype=float)
        t = np.array(t_xyz, dtype=float)
        raw_len = edge.get("length")
        length = float(raw_len) if isinstance(raw_len, (int, float)) else float(np.linalg.norm(t - s))
        # 找到离点最远的端点（作为"另一端"）
        other = t if float(np.linalg.norm(p - s)) <= float(np.linalg.norm(p - t)) else s
        if unit(other - p) is None:
            continue
        edge_items.append((length, str(edge_id), other))
    
    if len(edge_items) < 2:
        return []
    
    # 分别标记长焊缝和短焊缝
    longest = max(edge_items, key=lambda item: (item[0], item[1]))
    remaining = [item for item in edge_items if item[1] != longest[1]]
    if not remaining:
        return []
    shortest = min(remaining, key=lambda item: (item[0], item[1]))
    
    # 第1个向量：从当前点指向长焊缝的另一端
    long_dir = unit(longest[2] - p)
    # 第2个向量：从短焊缝的另一端指向当前点
    short_dir = unit(p - shortest[2])
    
    if long_dir is None or short_dir is None:
        return []
    
    # 第3个向量：获取最高优先级肋板上当前点实际接触的面的法向（从内向外）
    # 优先级：main_rib > side_rib > bridge_rib > unknown
    role_priority = {"main_rib": 3, "side_rib": 2, "bridge_rib": 1, "unknown": 0}
    
    highest_solid_id = None
    highest_priority = -1
    for sid in incident_solid_ids:
        role = solid_infos.get(sid, {}).get("role", "unknown")
        if role == "base_plate":
            continue  # 跳过基板
        priority = role_priority.get(role, 0)
        if priority > highest_priority:
            highest_priority = priority
            highest_solid_id = sid
    
    if highest_solid_id is None:
        return []
    
    # 从 normals 中查找当前点实际接触的最高优先级肋板的面
    # normals 列表中的所有面都是当前点实际接触的候选面
    plane_normal = None
    for item in normals:
        if int(item.get("solid_id", -1)) == int(highest_solid_id):
            plane_normal = dict(item)
            break
    
    if plane_normal is None:
        # 当前点未接触到最高优先级肋板，无法计算
        return []
    
    plane_normal["source"] = "hole_non_base_highest_rib_contact_face"
    plane_normal["solid_role"] = solid_infos.get(highest_solid_id, {}).get("role")
    
    plane_vec = np.array(plane_normal["normal"], dtype=float)
    # 确保法向指向肋板外侧：法向应该与从肋板中心指向点的方向同向
    rib_center = np.array(solid_infos.get(highest_solid_id, {}).get("center", [0.0, 0.0, 0.0]), dtype=float)
    rib_to_point = p - rib_center
    if float(np.dot(plane_vec, rib_to_point)) < 0.0:
        plane_vec = -plane_vec
        plane_normal["normal"] = [float(x) for x in plane_vec]
    
    return [
        {
            "solid_id": None,
            "face_index": None,
            "normal": [float(x) for x in long_dir],
            "distance": 0.0,
            "source": "hole_non_base_longest_weld_current_to_other",
            "edge_id": longest[1],
            "edge_length": longest[0],
        },
        {
            "solid_id": None,
            "face_index": None,
            "normal": [float(x) for x in short_dir],
            "distance": 0.0,
            "source": "hole_non_base_shortest_weld_other_to_current",
            "edge_id": shortest[1],
            "edge_length": shortest[0],
        },
        plane_normal,
    ]


def select_hole_base_weld_direction_normals(
    solids: list[Any],
    solid_infos: dict[int, dict[str, Any]],
    xyz: list[float],
    incident_solid_ids: set[int],
    incident_edge_ids: list[str],
    contact_edges: dict[str, Any],
    normals: list[dict[str, Any]],
    up_axis: np.ndarray,
    classifiers: list[Any] | None = None,
) -> list[dict[str, Any]]:
    base_id = next((sid for sid, info in solid_infos.items() if info.get("role") == "base_plate"), None)
    if base_id is None or base_id < 1 or base_id > len(solids):
        return []
    if base_id not in incident_solid_ids:
        return []

    base_candidates = [item for item in normals if int(item.get("solid_id", -1)) == int(base_id)]
    if base_candidates:
        base = max(
            base_candidates,
            key=lambda item: (float(np.dot(np.array(item["normal"], dtype=float), up_axis)), -float(item["distance"])),
        )
    else:
        base = nearest_face_normal_on_solid(solids[base_id - 1], xyz, base_id, classifiers=classifiers)
    if base is None:
        return []

    base = dict(base)
    base_vec = np.array(base["normal"], dtype=float)
    if float(np.dot(base_vec, up_axis)) < 0.0:
        base_vec = -base_vec
        base["normal"] = [float(x) for x in base_vec]
    base["source"] = "hole_base_base_up"

    p = np.array(xyz, dtype=float)
    edge_items: list[tuple[float, str, np.ndarray]] = []
    for edge_id in incident_edge_ids:
        edge = contact_edges.get(str(edge_id))
        if not isinstance(edge, dict):
            continue
        s_xyz = as_xyz(edge.get("start"))
        t_xyz = as_xyz(edge.get("end"))
        if s_xyz is None or t_xyz is None:
            continue
        s = np.array(s_xyz, dtype=float)
        t = np.array(t_xyz, dtype=float)
        raw_len = edge.get("length")
        length = float(raw_len) if isinstance(raw_len, (int, float)) else float(np.linalg.norm(t - s))
        other = t if float(np.linalg.norm(p - s)) <= float(np.linalg.norm(p - t)) else s
        if unit(other - p) is None:
            continue
        edge_items.append((length, str(edge_id), other))

    if len(edge_items) < 2:
        return []

    longest = max(edge_items, key=lambda item: (item[0], item[1]))
    remaining = [item for item in edge_items if item[1] != longest[1]]
    if not remaining:
        return []
    shortest = min(remaining, key=lambda item: (item[0], item[1]))

    long_dir = unit(longest[2] - p)
    short_dir = unit(p - shortest[2])
    if long_dir is None or short_dir is None:
        return []

    return [
        base,
        {
            "solid_id": None,
            "face_index": None,
            "normal": [float(x) for x in long_dir],
            "distance": 0.0,
            "source": "hole_base_longest_weld_current_to_other",
            "edge_id": longest[1],
            "edge_length": longest[0],
        },
        {
            "solid_id": None,
            "face_index": None,
            "normal": [float(x) for x in short_dir],
            "distance": 0.0,
            "source": "hole_base_shortest_weld_other_to_current",
            "edge_id": shortest[1],
            "edge_length": shortest[0],
        },
    ]


def select_contact_face_normals_for_solids(
    solids: list[Any],
    xyz: list[float],
    solid_ids: list[Any],
    normals: list[dict[str, Any]],
    source: str,
    classifiers: list[Any] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    def nearest_item_from_normals(sid: int) -> dict[str, Any] | None:
        candidates = [item for item in normals if int(item.get("solid_id", -1)) == sid]
        if not candidates:
            return None
        return min(candidates, key=lambda item: float(item.get("distance", float("inf"))))

    for raw_sid in solid_ids[:2]:
        try:
            sid = int(raw_sid)
        except (TypeError, ValueError):
            continue
        if sid < 1 or sid > len(solids):
            continue
        item = nearest_item_from_normals(sid)
        if item is None:
            item = nearest_face_normal_on_solid(solids[sid - 1], xyz, sid, classifiers=classifiers)
        if item is None:
            continue
        item = dict(item)
        item["source"] = source
        selected.append(item)

    return selected if len(selected) >= 2 else []


def longest_incident_weld_direction(
    xyz: list[float],
    incident_edge_ids: list[str],
    contact_edges: dict[str, Any],
) -> dict[str, Any] | None:
    p = np.array(xyz, dtype=float)

    def edge_length(edge: dict[str, Any], s: np.ndarray, t: np.ndarray) -> float:
        raw = edge.get("length")
        if isinstance(raw, (int, float)):
            return float(raw)
        return float(np.linalg.norm(t - s))

    best: dict[str, Any] | None = None
    best_len = -1.0
    for edge_id in incident_edge_ids:
        edge = contact_edges.get(str(edge_id))
        if not isinstance(edge, dict):
            continue
        s_xyz = as_xyz(edge.get("start"))
        t_xyz = as_xyz(edge.get("end"))
        if s_xyz is None or t_xyz is None:
            continue
        s = np.array(s_xyz, dtype=float)
        t = np.array(t_xyz, dtype=float)

        ds = float(np.linalg.norm(p - s))
        dt = float(np.linalg.norm(p - t))
        if ds <= dt:
            other = t
        else:
            other = s

        direction = unit(other - p)
        if direction is None:
            continue

        L = edge_length(edge, s, t)
        if L > best_len:
            best_len = L
            best = {
                "solid_id": None,
                "face_index": None,
                "distance": 0.0,
                "normal": [float(x) for x in direction],
                "source": "hole_longest_incident_weld_direction",
                "edge_id": str(edge_id),
                "edge_length": L,
            }
    return best


def longest_incident_weld(
    incident_edge_ids: list[str],
    contact_edges: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    best_id: str | None = None
    best_edge: dict[str, Any] | None = None
    best_len = -1.0
    for edge_id in incident_edge_ids:
        edge = contact_edges.get(str(edge_id))
        if not isinstance(edge, dict):
            continue
        raw_len = edge.get("length")
        if isinstance(raw_len, (int, float)):
            length = float(raw_len)
        else:
            s = as_xyz(edge.get("start"))
            t = as_xyz(edge.get("end"))
            if s is None or t is None:
                continue
            length = float(np.linalg.norm(np.array(t, dtype=float) - np.array(s, dtype=float)))
        if length > best_len:
            best_len = length
            best_id = str(edge_id)
            best_edge = edge
    if best_id is None or best_edge is None:
        return None
    return best_id, best_edge


def select_non_hole_non_base_three_normals(
    solids: list[Any],
    solid_infos: dict[int, dict[str, Any]],
    xyz: list[float],
    incident_solid_ids: set[int],
    incident_edge_ids: list[str],
    contact_edges: dict[str, Any],
    normals: list[dict[str, Any]],
    up_axis: np.ndarray,
    classifiers: list[Any] | None = None,
) -> list[dict[str, Any]]:
    base_id = next((sid for sid, info in solid_infos.items() if info.get("role") == "base_plate"), None)
    if base_id is None or base_id < 1 or base_id > len(solids):
        return []
    if base_id in incident_solid_ids:
        return []

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[float, float, float]] = set()

    for item in sorted(normals, key=lambda x: (float(x["distance"]), int(x["solid_id"]), int(x["face_index"]))):
        key = vec_key(np.array(item["normal"], dtype=float))
        if key in selected_keys:
            continue
        item = dict(item)
        item["source"] = "regular_non_base_point_plane"
        item["solid_role"] = solid_infos.get(int(item["solid_id"]), {}).get("role")
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= 3:
            break

    return selected if len(selected) >= 3 else []


def combine_normals(selected: list[dict[str, Any]]) -> list[float] | None:
    if not selected:
        return None
    acc = np.zeros(3, dtype=float)
    for item in selected:
        acc += np.array(item["normal"], dtype=float)
    out = unit(acc)
    if out is None:
        return None
    return [float(x) for x in out]


def regular_face_mix(selected: list[dict[str, Any]], up_axis: np.ndarray) -> dict[str, int]:
    base = 0
    side = 0
    other = 0
    for item in selected:
        n = np.array(item["normal"], dtype=float)
        up_dot = float(np.dot(n, up_axis))
        abs_up_dot = abs(up_dot)
        if up_dot > 0.65:
            base += 1
        elif abs_up_dot < 0.45:
            side += 1
        else:
            other += 1
    return {"base_faces": base, "side_faces": side, "other_faces": other}


def compute_pose_normals(
    step_file: Path,
    final_json: Path,
    geometry_graph_path: Path | None,
    process_graph_path: Path | None,
    output_json: Path,
    tol: float,
) -> None:
    final_obj = load_json(final_json)
    geometry_graph = load_json(geometry_graph_path) if geometry_graph_path else None
    process_graph = load_json(process_graph_path) if process_graph_path else None

    shape = load_step_file(str(step_file))
    if shape is None:
        raise RuntimeError(f"Failed to load STEP file: {step_file}")
    solids = list(TopologyExplorer(shape).solids())
    if not solids:
        raise RuntimeError(f"No solids found in STEP file: {step_file}")
    solid_infos = classify_solids(solids)
    
    # 预初始化实体分类器，用于剔除被遮挡或在内部的伪外部法线
    classifiers = [BRepClass3d_SolidClassifier(s) for s in solids]

    points = collect_final_points(final_obj)
    final_obj["points"] = points
    incidence = build_point_incidence(final_obj)
    hole_arcs = load_through_hole_arcs(geometry_graph, process_graph)
    up_axis = infer_up_axis(final_obj)

    valid_breakpoints: dict[str, dict[str, Any]] = {}
    weld_seams = final_obj.get("weld_seams", {}) or {}
    contact_edges = final_obj.get("contact_edges", {}) or {}
    
    for seam_id, seam in weld_seams.items():
        if not isinstance(seam, dict):
            continue
        edge_ids = seam.get("edge_ids", [])
        if not isinstance(edge_ids, list):
            continue
            
        point_counts: dict[str, int] = {}
        point_to_edge: dict[str, tuple[str, str]] = {}
        
        for eid in edge_ids:
            edge = contact_edges.get(str(eid))
            if not isinstance(edge, dict):
                continue
            pts = edge.get("points", [])
            if len(pts) != 2:
                continue
            
            p0_id = pts[0].get("point_id")
            p1_id = pts[-1].get("point_id")
            
            if p0_id and p1_id:
                point_counts[str(p0_id)] = point_counts.get(str(p0_id), 0) + 1
                point_to_edge[str(p0_id)] = (str(eid), str(p1_id))
                
                point_counts[str(p1_id)] = point_counts.get(str(p1_id), 0) + 1
                point_to_edge[str(p1_id)] = (str(eid), str(p0_id))

        for pid, count in point_counts.items():
            if count == 1:  # 只出现1次说明位于当前这整条焊缝的末端
                p_data = points.get(pid)
                if isinstance(p_data, dict) and p_data.get("role") == "breakpoint":
                    if pid not in valid_breakpoints:
                        edge_id, other_pid = point_to_edge[pid]
                        edge = contact_edges.get(edge_id, {})
                        valid_breakpoints[pid] = {
                            "other_id": other_pid,
                            "solid_ids": edge.get("solid_ids", [])
                        }

    summary = {"regular": 0, "hole_arc": 0, "fallback_lower_level_face": 0, "breakpoint": 0, "failed": 0, "skipped": 0}
    for point_id, point in points.items():
        role = point.get("role")
        bp_info = valid_breakpoints.get(str(point_id))
        incident = incidence.get(str(point_id), {})
        if role != "endpoint" and bp_info is None and not incident:
            summary["skipped"] += 1
            continue
        
        xyz = as_xyz(point.get("xyz"))
        if xyz is None:
            summary["failed"] += 1
            continue

        solid_ids = incident.get("solid_ids", set())
        normals = collect_unique_normals(solids, xyz, solid_ids, tol, classifiers=classifiers)
        arc_hit = find_hole_arc_for_point(xyz, hole_arcs, tol)

        if role == "breakpoint" and bp_info is not None:
            bp_solid_ids = bp_info["solid_ids"]
            selected = select_contact_face_normals_for_solids(
                solids,
                xyz,
                bp_solid_ids,
                normals,
                source="breakpoint_current_weld_contact_face",
                classifiers=classifiers,
            )
            
            # 添加焊缝方向向量：从当前断点指向焊缝另一端
            other_pid = bp_info.get("other_id")
            if other_pid:
                other_point = points.get(str(other_pid))
                if isinstance(other_point, dict):
                    other_xyz = as_xyz(other_point.get("xyz"))
                    if other_xyz is not None:
                        p = np.array(xyz, dtype=float)
                        other_p = np.array(other_xyz, dtype=float)
                        weld_dir = unit(other_p - p)
                        if weld_dir is not None:
                            selected.append({
                                "solid_id": None,
                                "face_index": None,
                                "normal": [float(x) for x in weld_dir],
                                "distance": 0.0,
                                "source": "breakpoint_weld_direction",
                            })
            
            method = "breakpoint_two_contact_faces_plus_weld_direction"
            fallback_item = None
            summary["breakpoint"] += 1

        elif role == "breakpoint" and incident.get("contact_edges"):
            # 在焊缝中间的断点（不在焊缝端点），不计算向量，直接跳过
            summary["skipped"] += 1
            continue

        elif arc_hit is not None:
            base_id = next((sid for sid, info in solid_infos.items() if info.get("role") == "base_plate"), None)
            is_non_base_point = base_id is not None and base_id not in solid_ids
            if is_non_base_point:
                selected = select_hole_non_base_plane_normals(
                    solids,
                    solid_infos,
                    xyz,
                    solid_ids,
                    incident.get("contact_edges", []),
                    final_obj.get("contact_edges") or {},
                    normals,
                    up_axis,
                    classifiers=classifiers,
                )
                method = "hole_arc_non_base_long_short_weld_plus_highest_plane"
            else:
                selected = select_hole_base_weld_direction_normals(
                    solids,
                    solid_infos,
                    xyz,
                    solid_ids,
                    incident.get("contact_edges", []),
                    final_obj.get("contact_edges") or {},
                    normals,
                    up_axis,
                    classifiers=classifiers,
                )
                method = "hole_arc_base_up_plus_long_short_weld_directions"
            if not selected and is_non_base_point:
                selected = select_hole_side_normals(
                    solids,
                    solid_infos,
                    xyz,
                    solid_ids,
                    incident.get("contact_edges", []),
                    final_obj.get("contact_edges") or {},
                    normals,
                    up_axis,
                    classifiers=classifiers,
                )
                method = "hole_arc_base_up_plus_contact_ribs"
            elif not selected:
                method = "hole_arc_base_insufficient_long_short_weld_directions"
            if not selected and is_non_base_point:
                selected = select_hole_normals(normals, up_axis)
                method = "hole_arc_base_up_plus_contact_ribs_fallback_face_normals"
            fallback_item = None
            summary["hole_arc"] += 1
        else:
            contact_edges = final_obj.get("contact_edges") or {}
            base_id = next((sid for sid, info in solid_infos.items() if info.get("role") == "base_plate"), None)
            is_non_base_point = base_id is not None and base_id not in solid_ids
            selected = select_non_hole_non_base_three_normals(
                solids,
                solid_infos,
                xyz,
                solid_ids,
                incident.get("contact_edges", []),
                contact_edges,
                normals,
                up_axis,
                classifiers=classifiers,
            )
            method = "regular_non_base_three_uncovered_point_planes"
            if not selected and not is_non_base_point:
                selected = select_regular_normals(normals, up_axis)
                method = "regular_three_contact_faces"
            fallback_item = None
            if len(selected) < 3 and not is_non_base_point:
                fallback_item = select_lower_level_fallback_face(
                    solids,
                    solid_infos,
                    xyz,
                    solid_ids,
                    selected,
                    classifiers=classifiers,
                )
                if fallback_item is not None:
                    selected = selected + [fallback_item]
                    method = f"regular_with_{fallback_item['fallback_role']}_fallback_face"
                    summary["fallback_lower_level_face"] += 1
            elif len(selected) < 3 and is_non_base_point:
                method = "regular_non_base_insufficient_uncovered_point_planes"
            summary["regular"] += 1

        pose_normal = combine_normals(selected)
        if pose_normal is None:
            summary["failed"] += 1

        point["pose_normal"] = pose_normal
        point["pose_normal_method"] = method
        point["pose_normal_used_fallback"] = fallback_item is not None
        point["regular_face_mix"] = regular_face_mix(selected, up_axis) if arc_hit is None else None
        point["hole_arc_side_face_normals"] = selected if arc_hit is not None else None
        point["pose_normal_fallback_face"] = (
            {
                "solid_id": fallback_item["solid_id"],
                "solid_role": fallback_item.get("fallback_role"),
                "source_role": fallback_item.get("fallback_source_role"),
                "face_index": fallback_item["face_index"],
                "normal": fallback_item["normal"],
                "distance": fallback_item["distance"],
            }
            if fallback_item is not None
            else None
        )
        point["on_through_hole_arc"] = arc_hit is not None
        point["through_hole_arc_id"] = arc_hit[0] if arc_hit else None
        point["incident_contact_edges"] = incident.get("contact_edges", [])
        point["incident_solid_ids"] = sorted(solid_ids) if isinstance(solid_ids, set) else []
        point["selected_face_normals"] = [
            {
                "solid_id": item.get("solid_id"),
                "face_index": item.get("face_index"),
                "normal": item.get("normal"),
                "distance": item.get("distance"),
                "is_plane": item.get("is_plane"),
                "source": item.get("source"),
                "edge_id": item.get("edge_id"),
                "edge_length": item.get("edge_length"),
                "fallback": bool(item.get("fallback")),
                "fallback_role": item.get("fallback_role"),
                "fallback_source_role": item.get("fallback_source_role"),
            }
            for item in selected
        ]
        point["candidate_face_normals"] = [
            {
                "solid_id": item["solid_id"],
                "face_index": item["face_index"],
                "normal": item["normal"],
                "distance": item["distance"],
                "is_plane": item.get("is_plane"),
            }
            for item in normals
        ]
        point["solid_role_inference"] = [
            {
                "solid_id": sid,
                "role": solid_infos.get(sid, {}).get("role"),
            }
            for sid in sorted(solid_ids)
        ]

    final_obj.setdefault("meta", {})
    final_obj["meta"]["point_pose_normals"] = {
        "step_file": str(step_file),
        "geometry_graph": str(geometry_graph_path) if geometry_graph_path else None,
        "process_graph": str(process_graph_path) if process_graph_path else None,
        "face_normal_definition": "outward_from_solid_interior",
        "tolerance": tol,
        "summary": summary,
        "solid_role_inference": [
            {
                "solid_id": sid,
                "role": info.get("role"),
                "bounds": info.get("bounds"),
                "size": info.get("size"),
            }
            for sid, info in sorted(solid_infos.items())
        ],
    }
    save_json(output_json, final_obj)
    print(
        f"[pose_normals] points={len(points)} regular={summary['regular']} "
        f"hole_arc={summary['hole_arc']} fallback={summary['fallback_lower_level_face']} "
        f"breakpoint={summary['breakpoint']} "
        f"skipped={summary['skipped']} failed={summary['failed']}"
    )
    print(f"[pose_normals] saved -> {output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute outward-face-composed pose normals for points in final weld JSON."
    )
    parser.add_argument("--step", type=Path, required=True, help="STEP/STP file used for OCC face normal queries.")
    parser.add_argument("--final-json", type=Path, required=True, help="Final weld/path JSON containing points.")
    parser.add_argument("--geometry-graph", type=Path, default=None, help="Optional *_geometry_graph_with_breakpoints.json.")
    parser.add_argument("--process-graph", type=Path, default=None, help="Optional *_process_graph.json.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON. Defaults to *_pose_normals.json.")
    parser.add_argument("--tol", type=float, default=1e-2, help="Point-to-face/arc tolerance in model units.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry_graph = args.geometry_graph
    process_graph = args.process_graph
    if geometry_graph is None or process_graph is None:
        inferred_geometry, inferred_process = infer_paths(args.final_json)
        geometry_graph = geometry_graph or inferred_geometry
        process_graph = process_graph or inferred_process

    output = args.output
    if output is None:
        output = args.final_json.with_name(f"{args.final_json.stem}_pose_normals.json")

    compute_pose_normals(
        step_file=args.step,
        final_json=args.final_json,
        geometry_graph_path=geometry_graph,
        process_graph_path=process_graph,
        output_json=output,
        tol=args.tol,
    )


if __name__ == "__main__":
    main()
