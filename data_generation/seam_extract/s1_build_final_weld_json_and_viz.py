from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# IO
# ----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------
# geometry helpers
# ----------------------------
def _safe_unit(v: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return None
    return v / n


def _tangent_from_points(a: List[float], b: List[float]) -> Optional[List[float]]:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    u = _safe_unit(vb - va)
    if u is None:
        return None
    return [float(u[0]), float(u[1]), float(u[2])]


def _segment_length(a: List[float], b: List[float]) -> float:
    return float(np.linalg.norm(np.array(b, float) - np.array(a, float)))


def _midpoint_polyline(samples: List[List[float]]) -> List[float]:
    """Simple midpoint by arclen along samples (for placing 'L' label)."""
    if not samples or len(samples) < 2:
        return samples[0] if samples else [0.0, 0.0, 0.0]
    pts = np.array(samples, dtype=float)
    seg = pts[1:] - pts[:-1]
    d = np.linalg.norm(seg, axis=1)
    total = float(np.sum(d))
    if total <= 1e-12:
        m = pts[len(pts)//2]
        return [float(m[0]), float(m[1]), float(m[2])]
    half = 0.5 * total
    acc = 0.0
    for i, di in enumerate(d):
        if acc + float(di) >= half:
            t = (half - acc) / max(float(di), 1e-12)
            p = pts[i] + t * (pts[i+1] - pts[i])
            return [float(p[0]), float(p[1]), float(p[2])]
        acc += float(di)
    last = pts[-1]
    return [float(last[0]), float(last[1]), float(last[2])]


def _count_nearby_welds(
    point: List[float],
    all_weld_edges: Dict[str, Any],
    current_weld_id: str,
    distance_threshold: float = 50.0
) -> int:
    """
    统计指定点附近distance_threshold范围内，除了当前焊缝外还有多少条其他焊缝。
    用于判断该点是否适合U包角策略（附近焊缝少则适合U包）。
    
    返回：附近其他焊缝的数量
    """
    pt = np.array(point, dtype=float)
    nearby_count = 0
    
    for wid, w in all_weld_edges.items():
        if wid == current_weld_id:
            continue
            
        start = w.get("start")
        end = w.get("end")
        if not start or not end:
            continue
        
        # 检查焊缝的起点和终点是否在阈值范围内
        start_dist = float(np.linalg.norm(np.array(start, dtype=float) - pt))
        end_dist = float(np.linalg.norm(np.array(end, dtype=float) - pt))
        
        if start_dist < distance_threshold or end_dist < distance_threshold:
            nearby_count += 1
    
    return nearby_count


def _infer_weld_normal(weld_edges: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    从所有焊缝的中点拟合一个主平面，返回该平面的法向量作为"上轴"。
    用于判断焊缝是否为立焊缝。
    
    返回：法向量（单位向量），如果拟合失败则返回None
    """
    # 收集所有焊缝的中点
    midpoints = []
    for w in weld_edges.values():
        start = w.get("start")
        end = w.get("end")
        if not start or not end:
            continue
        mid = [(start[0] + end[0]) * 0.5, 
               (start[1] + end[1]) * 0.5, 
               (start[2] + end[2]) * 0.5]
        midpoints.append(mid)
    
    if len(midpoints) < 3:
        return None
    
    pts = np.array(midpoints, dtype=float)
    
    # 使用PCA找主平面
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    
    # 协方差矩阵
    cov = np.cov(centered.T)
    
    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # 最小特征值对应的特征向量就是法向量
    normal = eigvecs[:, 0]  # 最小特征值的特征向量
    normal = normal / np.linalg.norm(normal)
    
    # 确保法向量朝上（Z分量为正）
    if normal[2] < 0:
        normal = -normal
    
    return normal


def _is_vertical_weld(
    start: List[float],
    end: List[float],
    up_axis: np.ndarray,
    vertical_deg: float = 20.0
) -> bool:
    """
    判断焊缝是否为立焊缝。
    
    参数：
        start: 焊缝起点
        end: 焊缝终点
        up_axis: 上轴方向（通常是底板法向）
        vertical_deg: 判断为立焊的角度阈值（默认20度）
    
    返回：True表示立焊缝，False表示非立焊缝
    """
    # 计算焊缝方向
    direction = np.array(end, dtype=float) - np.array(start, dtype=float)
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm < 1e-12:
        return False
    
    direction = direction / direction_norm
    
    # 计算焊缝方向与上轴的夹角余弦值
    cos_angle = abs(float(np.dot(direction, up_axis)))
    cos_angle = max(0.0, min(1.0, cos_angle))
    
    # 转换为角度
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    
    # 如果角度小于阈值，说明焊缝方向接近垂直于底板，即立焊缝
    return angle_deg <= vertical_deg


def _is_connected_to_vertical_weld(
    weld_id: str,
    endpoint: List[float],
    all_weld_edges: Dict[str, Any],
    up_axis: np.ndarray,
    vertical_deg: float = 20.0,
    connection_tolerance: float = 1e-6
) -> bool:
    """
    判断指定端点是否与立焊缝相连。
    
    参数：
        weld_id: 当前焊缝ID
        endpoint: 要检查的端点坐标
        all_weld_edges: 所有焊缝边
        up_axis: 上轴方向（底板法向）
        vertical_deg: 判断为立焊的角度阈值
        connection_tolerance: 端点连接判断的距离容差
    
    返回：True表示该端点与立焊缝相连
    """
    pt = np.array(endpoint, dtype=float)
    
    for other_wid, other_w in all_weld_edges.items():
        if other_wid == weld_id:
            continue
        
        other_start = other_w.get("start")
        other_end = other_w.get("end")
        if not other_start or not other_end:
            continue
        
        # 检查是否与当前端点相连
        start_dist = float(np.linalg.norm(np.array(other_start, dtype=float) - pt))
        end_dist = float(np.linalg.norm(np.array(other_end, dtype=float) - pt))
        
        is_connected = (start_dist < connection_tolerance or end_dist < connection_tolerance)
        
        if is_connected:
            # 检查相连的焊缝是否为立焊缝
            if _is_vertical_weld(other_start, other_end, up_axis, vertical_deg):
                return True
    
    return False


# ----------------------------
# point registry
# ----------------------------
def _pt_key(p: List[float], ndigits: int = 8) -> Tuple[float, float, float]:
    return (round(float(p[0]), ndigits), round(float(p[1]), ndigits), round(float(p[2]), ndigits))


class PointRegistry:
    """
    Goal:
      - If a point already exists in input graph nodes, reuse its node_id as point_id.
      - Only create new IDs (P1, P2, ...) for newly introduced points (e.g., new breakpoints).
    """
    def __init__(self):
        self._key_to_id: Dict[Tuple[float, float, float], str] = {}
        self.points: Dict[str, Dict[str, Any]] = {}
        self._cnt = 0

    def get_or_add(self, xyz: List[float], role: str, preferred_id: Optional[str] = None) -> str:
        k = _pt_key(xyz)
        pid = self._key_to_id.get(k)
        if pid is not None:
            return pid

        # Try reuse preferred_id if it's safe
        if preferred_id:
            preferred_id = str(preferred_id)
            if preferred_id in self.points:
                # If same id already used for different coordinate, fall back to a new P#
                prev = self.points[preferred_id].get("xyz")
                if isinstance(prev, list) and len(prev) == 3 and _pt_key(prev) == k:
                    pid = preferred_id
                else:
                    self._cnt += 1
                    pid = f"P{self._cnt}"
            else:
                pid = preferred_id
        else:
            self._cnt += 1
            pid = f"P{self._cnt}"

        self._key_to_id[k] = pid
        self.points[pid] = {
            "id": pid,
            "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            "role": role,  # "endpoint" | "breakpoint"
        }
        return pid


# ----------------------------
# build final json
# ----------------------------
def build_final_json(
    geometry_graph: Dict[str, Any],
    process_graph: Optional[Dict[str, Any]] = None,
    *,
    # kept for API compatibility; corner strategies are no longer inferred or used
    l_push_on: str = "B",
    u_wrap_distance_threshold: float = 50.0,
    u_wrap_max_nearby_welds: int = 0,
    vertical_weld_deg: float = 20.0,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    geometry_graph: your *_geometry_graph_with_breakpoints.json
    process_graph:  your *_process_graph.json (optional; used to reinforce breakpoint node roles)
    u_wrap_distance_threshold: 距离阈值，用于检测端点附近的焊缝（默认50.0）
    u_wrap_max_nearby_welds: 如果端点附近其他焊缝数量<=此值，则使用U包角策略（默认0，即附近无其他焊缝）
    vertical_weld_deg: 判断为立焊缝的角度阈值（默认20度）
    
    返回：(final_obj, up_axis)
    """
    weld_edges_raw = geometry_graph.get("weld_edges", {}) or {}
    weld_edges: Dict[str, Any] = {str(k): v for k, v in weld_edges_raw.items()}
    
    # 拟合底板法向量，用于判断立焊缝
    up_axis = _infer_weld_normal(weld_edges)
    if up_axis is None:
        # 如果拟合失败，使用默认Z轴
        up_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        print("[warning] 无法拟合底板法向量，使用默认Z轴")
    else:
        print(f"[info] 拟合底板法向量: [{up_axis[0]:.4f}, {up_axis[1]:.4f}, {up_axis[2]:.4f}]")

    # Prepare output skeleton
    out: Dict[str, Any] = {
        "solids": {},
        "contact_edges": {},
        "weld_seams": {},
        "global_safety": {
            "min_distance_to_base_plate": None,
            "min_distance_between_tool_and_rib": None,
        },
        "points": {},  # exported at the end
    }

    # Build coord->node_id map from geometry_graph nodes, so we can reuse existing ids.
    nodes = geometry_graph.get("nodes", {}) or {}
    coord2nid: Dict[Tuple[float, float, float], str] = {}
    for nid, n in nodes.items():
        if isinstance(n, dict) and isinstance(n.get("point"), list) and len(n["point"]) == 3:
            coord2nid[_pt_key(n["point"])] = str(nid)

    preg = PointRegistry()

    pg_nodes = (process_graph or {}).get("nodes", {}) if isinstance(process_graph, dict) else {}

    def is_breakpoint_node(nid: str) -> bool:
        info = pg_nodes.get(nid)
        return bool(isinstance(info, dict) and info.get("is_breakpoint") is True)

    def prefer_id_for_xyz(xyz: List[float]) -> Optional[str]:
        return coord2nid.get(_pt_key(xyz))

    # Helper to create one contact_edge record
    def add_contact_edge(
        edge_id: str,
        a: List[float],
        b: List[float],
        seg_samples: List[List[float]],
        corner_strategy: Optional[str],
        *,
        start_role: str = "endpoint",
        end_role: str = "endpoint",
        start_preferred_id: Optional[str] = None,
        end_preferred_id: Optional[str] = None,
        wtype: Any = None,
        preferred_normal: Any = None,
        solid_ids: Any = None,
        source_weld_id: str = "",
    ) -> None:
        spid = preg.get_or_add(a, start_role, preferred_id=start_preferred_id)
        epid = preg.get_or_add(b, end_role, preferred_id=end_preferred_id)

        tangent = _tangent_from_points(a, b)

        out["contact_edges"][edge_id] = {
            "type": wtype,
            "start": [float(a[0]), float(a[1]), float(a[2])],
            "end": [float(b[0]), float(b[1]), float(b[2])],
            "length": _segment_length(a, b),
            "solid_ids": solid_ids,
            "tangent": tangent,
            "preferred_normal": preferred_normal,

            # ordered along the edge
            "points": [
                {"point_id": spid, "role": start_role},
                {"point_id": epid, "role": end_role},
            ],

            "samples": seg_samples,
            "corner_strategy": corner_strategy,
            "source_weld_id": source_weld_id,
        }

    # 1) build contact_edges (splitting if breakpoint)
    for wid, w in weld_edges.items():
        wtype = w.get("type", None)
        start = w.get("start", None)
        end = w.get("end", None)
        if not start or not end:
            continue

        samples = w.get("samples", None)
        if not isinstance(samples, list) or len(samples) < 2:
            samples = [start, end]

        bp_node = w.get("breakpoint_node", None)
        bp_point = w.get("breakpoint_point", None)

        has_bp = bool(w.get("has_breakpoint") is True or bp_node or bp_point)
        if bp_node and is_breakpoint_node(str(bp_node)):
            has_bp = True

        # If has breakpoint but missing bp_point, try lookup from geometry_graph node
        if has_bp and (not bp_point) and isinstance(bp_node, str):
            n = (geometry_graph.get("nodes", {}) or {}).get(bp_node)
            if isinstance(n, dict) and n.get("point"):
                bp_point = n["point"]

        if not has_bp or not bp_point:
            # no split: keep edge id = wid, and REUSE point ids from input if possible
            add_contact_edge(
                edge_id=wid,
                a=start,
                b=end,
                seg_samples=samples,
                corner_strategy=None,
                start_role="endpoint",
                end_role="endpoint",
                start_preferred_id=prefer_id_for_xyz(start),
                end_preferred_id=prefer_id_for_xyz(end),
                wtype=wtype,
                preferred_normal=w.get("preferred_normal", None),
                solid_ids=w.get("solid_ids", None),
                source_weld_id=wid,
            )
        else:
            # split into A(start->bp) and B(bp->end)
            bp_point = [float(bp_point[0]), float(bp_point[1]), float(bp_point[2])]

            # breakpoint id: prefer bp_node if provided; else reuse existing node id by coord; else create new P#
            bp_preferred_id = None
            if bp_node:
                bp_preferred_id = str(bp_node)
            else:
                bp_preferred_id = prefer_id_for_xyz(bp_point)

            # Split samples around bp_point if it exists in samples; else 2-point segments
            pts = np.array(samples, dtype=float)
            bp = np.array(bp_point, dtype=float)
            dist = np.linalg.norm(pts - bp[None, :], axis=1)
            idx = int(np.argmin(dist)) if len(dist) > 0 else -1

            if idx < 0:
                sA = [start, bp_point]
                sB = [bp_point, end]
            else:
                sA = samples[: idx + 1]
                if _pt_key(sA[-1]) != _pt_key(bp_point):
                    sA.append(bp_point)
                sB = samples[idx:]
                if _pt_key(sB[0]) != _pt_key(bp_point):
                    sB = [bp_point] + sB
                if _pt_key(sB[-1]) != _pt_key(end):
                    sB.append(end)

            edge_id_A = f"{wid}_A"
            edge_id_B = f"{wid}_B"

            add_contact_edge(
                edge_id=edge_id_A,
                a=start,
                b=bp_point,
                seg_samples=sA,
                corner_strategy=None,
                start_role="endpoint",
                end_role="breakpoint",
                start_preferred_id=prefer_id_for_xyz(start),
                end_preferred_id=bp_preferred_id,
                wtype=wtype,
                preferred_normal=w.get("preferred_normal", None),
                solid_ids=w.get("solid_ids", None),
                source_weld_id=wid,
            )

            add_contact_edge(
                edge_id=edge_id_B,
                a=bp_point,
                b=end,
                seg_samples=sB,
                corner_strategy=None,
                start_role="breakpoint",
                end_role="endpoint",
                start_preferred_id=bp_preferred_id,
                end_preferred_id=prefer_id_for_xyz(end),
                wtype=wtype,
                preferred_normal=w.get("preferred_normal", None),
                solid_ids=w.get("solid_ids", None),
                source_weld_id=wid,
            )

    # 2) build weld_seams (1 seam per original weld_id; include split edges if any)
    for wid in weld_edges.keys():
        edge_ids: List[str] = []
        if f"{wid}_A" in out["contact_edges"]:
            edge_ids += [f"{wid}_A", f"{wid}_B"]
        elif wid in out["contact_edges"]:
            edge_ids += [wid]
        if not edge_ids:
            continue

        w = weld_edges[wid]
        out["weld_seams"][f"weld_{wid}"] = {
            "name": w.get("name", None),
            "edge_ids": edge_ids,
            "solid_pair": w.get("solid_ids", None),
            "preferred_solid_id": w.get("preferred_solid_id", None),
            "process": {
                "travel_speed": None,
                "torch_angle_deg": None,
                "work_angle_deg": None,
            },
            "safety": {
                "approach_clearance": None,
                "retract_clearance": None,
                "approach_along_solid_id": None,
                "retract_along_solid_id": None,
            },
        }

    # export points
    out["points"] = preg.points

    # save up_axis into the output json
    out["up_axis"] = [float(up_axis[0]), float(up_axis[1]), float(up_axis[2])]

    return out, up_axis


# ----------------------------
# Visualization
# ----------------------------
def visualize_final_json(final_obj: Dict[str, Any], up_axis: Optional[np.ndarray] = None, *, 
                        title: str = "Final Weld JSON Visualization", 
                        show_z_axis: bool = False) -> None:
    contact_edges = final_obj.get("contact_edges", {}) or {}
    points = final_obj.get("points", {}) or {}

    endpoint_xyz = []
    breakpoint_xyz = []

    for p in points.values():
        if not isinstance(p, dict):
            continue
        xyz = p.get("xyz")
        role = p.get("role")
        if not xyz or len(xyz) != 3:
            continue
        if role == "breakpoint":
            breakpoint_xyz.append(xyz)
        else:
            endpoint_xyz.append(xyz)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    import matplotlib.cm as cm
    keys = sorted(list(contact_edges.keys()), key=lambda x: (len(str(x)), str(x)))
    cmap = cm.get_cmap("tab20")

    all_xyz = []

    for i, eid in enumerate(keys):
        e = contact_edges[eid]
        samples = e.get("samples")
        if not isinstance(samples, list) or len(samples) < 2:
            samples = [e.get("start"), e.get("end")]
        if not samples or len(samples) < 2:
            continue

        xs = [p[0] for p in samples]
        ys = [p[1] for p in samples]
        zs = [p[2] for p in samples]

        color = cmap(i % 20)
        ax.plot(xs, ys, zs, linewidth=2.0, color=color)

        all_xyz += samples

    if endpoint_xyz:
        ex = [p[0] for p in endpoint_xyz]
        ey = [p[1] for p in endpoint_xyz]
        ez = [p[2] for p in endpoint_xyz]
        ax.scatter(ex, ey, ez, marker="o", s=18, label="Endpoints")

    if breakpoint_xyz:
        bx = [p[0] for p in breakpoint_xyz]
        by = [p[1] for p in breakpoint_xyz]
        bz = [p[2] for p in breakpoint_xyz]
        ax.scatter(bx, by, bz, marker="^", s=28, label="Breakpoints")

    # 可视化Z轴（底板法向量）
    if show_z_axis and up_axis is not None and all_xyz:
        pts = np.array(all_xyz, dtype=float)
        centroid = np.mean(pts, axis=0)
        
        # 计算场景大小
        xmin, ymin, zmin = pts.min(axis=0)
        xmax, ymax, zmax = pts.max(axis=0)
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
        
        # Z轴长度为场景大小的30%
        arrow_length = max_range * 0.3
        
        # 绘制Z轴箭头
        ax.quiver(
            centroid[0], centroid[1], centroid[2],
            up_axis[0], up_axis[1], up_axis[2],
            length=arrow_length, color='green', arrow_length_ratio=0.2, linewidth=3,
            label=f'Z-axis (up): [{up_axis[0]:.3f}, {up_axis[1]:.3f}, {up_axis[2]:.3f}]'
        )

    if all_xyz:
        pts = np.array(all_xyz, dtype=float)
        xmin, ymin, zmin = pts.min(axis=0)
        xmax, ymax, zmax = pts.max(axis=0)
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
        if max_range <= 1e-9:
            max_range = 1.0
        xm, ym, zm = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5
        ax.set_xlim(xm - 0.5 * max_range, xm + 0.5 * max_range)
        ax.set_ylim(ym - 0.5 * max_range, ym + 0.5 * max_range)
        ax.set_zlim(zm - 0.5 * max_range, zm + 0.5 * max_range)

    ax.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    step_head = "D018-F205B"
    geometry_graph = f"model/sub_assembly/D018-F205B/{step_head}_geometry_graph_with_breakpoints.json"
    process_graph = f"model/sub_assembly/D018-F205B/{step_head}_process_graph.json"
    out_final = f"model/sub_assembly/D018-F205B/{step_head}_final_welds.json"

    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry_graph_with_breakpoints", default=geometry_graph,
                    help="input geometry_graph_with_breakpoints.json")
    ap.add_argument("--process_graph", default=process_graph,
                    help="optional process_graph_simple.json (for through-hole half selection)")
    ap.add_argument("--out_json", default=out_final, help="output final_welds.json")
    ap.add_argument("--visualize", default=True, action="store_true",
                    help="visualize endpoints/breakpoints and weld segments")
    ap.add_argument("--l_push_on", default="B", choices=["A", "B"],
                    help="deprecated; kept for CLI compatibility and ignored")
    
    ap.add_argument("--u_wrap_distance_threshold", type=float, default=100.0,
                    help="deprecated; kept for CLI compatibility and ignored")
    ap.add_argument("--u_wrap_max_nearby_welds", type=int, default=2,
                    help="deprecated; kept for CLI compatibility and ignored")
    ap.add_argument("--vertical_weld_deg", type=float, default=20.0,
                    help="deprecated; kept for CLI compatibility and ignored")
    
    # 可视化参数
    ap.add_argument("--show_z_axis", action="store_true",
                    help="在可视化中显示Z轴（底板法向量）")

    args = ap.parse_args()

    gg = load_json(args.geometry_graph_with_breakpoints)
    pg = load_json(args.process_graph) if args.process_graph else None

    final_obj, up_axis = build_final_json(
        gg, pg, 
        l_push_on=args.l_push_on,
        u_wrap_distance_threshold=args.u_wrap_distance_threshold,
        u_wrap_max_nearby_welds=args.u_wrap_max_nearby_welds,
        vertical_weld_deg=args.vertical_weld_deg
    )

    save_json(args.out_json, final_obj)
    print(f"[save] final json -> {args.out_json}")
    print(f"[stats] contact_edges={len(final_obj.get('contact_edges', {}) or {})}  points={len(final_obj.get('points', {}) or {})}")
    
    if args.visualize:
        visualize_final_json(final_obj, up_axis, title=os.path.basename(args.out_json), show_z_axis=args.show_z_axis)


if __name__ == "__main__":
    main()
