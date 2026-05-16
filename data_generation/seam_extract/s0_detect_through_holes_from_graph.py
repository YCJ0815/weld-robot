# visualize_geometry_graph_with_through_holes.py
# ------------------------------------------------------------
# Visualize geometry_graph.json:
#   - All geom_edges: GREEN
#   - Through-hole candidate edges (detected from weld-end nodes adjacency):
#       ARC     -> candidate (optionally require center via flag)
#       BSPLINE -> circle-fit rms <= tol AND NOT "almost straight"
#                 (reject near-line bsplines via chord-deviation + angle + radius cap)
#     shown in RED
#
# Output:
#   - process_graph.json (process attributes only)
#
# Key Fix (for your "long red straight line" false positives):
#   For bspline we add anti-line filters:
#     1) min_sagitta: max distance of samples to chord(start-end) must be >= threshold
#     2) min_angle_deg: fitted sweep angle must be >= threshold
#     3) max_radius: fitted radius must be <= threshold (avoid "huge radius circle == line")
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import math
import argparse
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Basic helpers
# =========================
def _safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else 0.0


def _unit(v: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    n = _safe_norm(v, eps)
    if n <= eps:
        return None
    return v / n


def _angle_from_center(center: List[float], start: List[float], end: List[float]) -> float:
    c = np.array(center, dtype=float)
    s = np.array(start, dtype=float)
    e = np.array(end, dtype=float)
    v1 = s - c
    v2 = e - c
    n1 = _unit(v1)
    n2 = _unit(v2)
    if n1 is None or n2 is None:
        return 0.0
    dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    return float(math.acos(dot))


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to segment ab in 3D."""
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 <= 1e-18:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / ab2)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _max_chord_deviation(samples_xyz: List[List[float]], start: List[float], end: List[float]) -> float:
    """
    Sagitta-like measure: max distance from samples to the chord segment(start-end).
    This catches "almost straight" bsplines even if circle-fit rms is small.
    """
    if not samples_xyz:
        return 0.0
    a = np.array(start, dtype=float)
    b = np.array(end, dtype=float)
    maxd = 0.0
    for q in samples_xyz:
        p = np.array(q, dtype=float)
        d = _point_segment_distance(p, a, b)
        if d > maxd:
            maxd = d
    return float(maxd)


def _compute_edge_tangent_at_node(edge: Dict[str, Any], node_point: List[float], eps: float = 1) -> Optional[np.ndarray]:
    """
    计算边在指定节点处的切向量（单位向量）
    
    Args:
        edge: 边的数据字典
        node_point: 节点坐标
        eps: 判断节点是否在端点的容差
    
    Returns:
        切向量（单位向量），如果无法计算则返回None
    """
    etype = edge.get("type", "unknown")
    s = edge.get("start")
    e = edge.get("end")
    
    if not s or not e:
        return None
    
    node_pt = np.array(node_point, dtype=float)
    start_pt = np.array(s, dtype=float)
    end_pt = np.array(e, dtype=float)
    
    # 判断节点在哪个端点
    dist_to_start = float(np.linalg.norm(node_pt - start_pt))
    dist_to_end = float(np.linalg.norm(node_pt - end_pt))
    
    at_start = dist_to_start < eps
    at_end = dist_to_end < eps
    
    if not at_start and not at_end:
        # 节点不在端点，无法计算
        return None
    
    # 获取边的采样点
    if etype == "bspline" and edge.get("samples"):
        samples = edge["samples"]
    elif etype == "arc":
        c = edge.get("center")
        if c:
            samples = _arc_polyline(c, s, e, angle=edge.get("angle", None), n=64)
        else:
            samples = [s, e]
    else:
        samples = [s, e]
    
    if len(samples) < 2:
        return None
    
    # 计算切向量
    if at_start:
        # 节点在起点，切向量指向第二个点
        tangent = np.array(samples[1], dtype=float) - np.array(samples[0], dtype=float)
    else:
        # 节点在终点，切向量指向倒数第二个点
        tangent = np.array(samples[-1], dtype=float) - np.array(samples[-2], dtype=float)
    
    return _unit(tangent)


def _compute_angle_between_edges(edge1: Dict[str, Any], edge2: Dict[str, Any], 
                                  node_point: List[float]) -> Optional[float]:
    """
    计算两条边在共同节点处的夹角（弧度）
    
    Args:
        edge1: 第一条边
        edge2: 第二条边
        node_point: 共同节点坐标
    
    Returns:
        夹角（弧度，范围0到π），如果无法计算则返回None
    """
    t1 = _compute_edge_tangent_at_node(edge1, node_point)
    t2 = _compute_edge_tangent_at_node(edge2, node_point)
    
    if t1 is None or t2 is None:
        return None
    
    # 计算夹角（注意：这里计算的是两个切向量的夹角，需要考虑方向）
    # 由于切向量都是从节点出发的方向，我们需要反转其中一个
    dot = float(np.clip(np.dot(t1, -t2), -1.0, 1.0))
    angle = float(math.acos(dot))
    
    return angle


# =========================
# Circle fit for bspline
# =========================
def _fit_plane_pca(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = pts.mean(axis=0)
    X = pts - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1, :]

    n_hat = _unit(normal)
    if n_hat is None:
        n_hat = np.array([0.0, 0.0, 1.0], dtype=float)

    tmp = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(tmp, n_hat))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=float)

    u = _unit(np.cross(n_hat, tmp))
    if u is None:
        u = np.array([1.0, 0.0, 0.0], dtype=float)

    v = _unit(np.cross(n_hat, u))
    if v is None:
        v = np.array([0.0, 1.0, 0.0], dtype=float)

    return centroid, n_hat, np.stack([u, v], axis=0)


def _fit_circle_2d_kasa(xy: np.ndarray) -> Tuple[np.ndarray, float, float]:
    x = xy[:, 0]
    y = xy[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y

    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b2, c = float(sol[0]), float(sol[1]), float(sol[2])

    cx = a / 2.0
    cy = b2 / 2.0
    r2 = c + cx * cx + cy * cy
    if r2 < 0:
        r2 = 0.0
    r = math.sqrt(r2)

    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rms = float(np.sqrt(np.mean((d - r) ** 2))) if len(d) > 0 else float("inf")
    return np.array([cx, cy], dtype=float), r, rms


def _fit_circle_3d(samples_xyz: List[List[float]]) -> Optional[Dict[str, Any]]:
    if not samples_xyz or len(samples_xyz) < 4:
        return None

    pts = np.array(samples_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None

    centroid, n_hat, uv = _fit_plane_pca(pts)
    u = uv[0]
    v = uv[1]

    X = pts - centroid
    xy = np.stack([X @ u, X @ v], axis=1)

    c2, r, rms = _fit_circle_2d_kasa(xy)
    c3 = centroid + c2[0] * u + c2[1] * v

    # angle estimate using endpoint rays in 2D
    p0 = xy[0] - c2
    p1 = xy[-1] - c2
    n0 = _unit(p0)
    n1 = _unit(p1)
    if n0 is None or n1 is None:
        ang = 0.0
    else:
        dot = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
        ang = float(math.acos(dot))

    return {
        "center": [float(c3[0]), float(c3[1]), float(c3[2])],
        "radius": float(r),
        "rms": float(rms),
        "plane_normal": [float(n_hat[0]), float(n_hat[1]), float(n_hat[2])],
        "angle_est": float(ang),
    }


# =========================
# Arc polyline for visualization
# =========================
def _arc_polyline(center, start, end, angle=None, n=64):
    c = np.array(center, dtype=float)
    s = np.array(start, dtype=float)
    e = np.array(end, dtype=float)

    v1 = s - c
    v2 = e - c
    r1 = _safe_norm(v1)
    r2 = _safe_norm(v2)
    if r1 < 1e-12 or r2 < 1e-12:
        return [start, end]

    v1n = v1 / r1
    v2n = v2 / r2

    nrm = np.cross(v1n, v2n)
    nn = _safe_norm(nrm)
    if nn < 1e-12:
        return [start, end]
    nrm = nrm / nn

    b2 = np.cross(nrm, v1n)
    b2n = _unit(b2)
    if b2n is None:
        return [start, end]

    dot = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    theta_end = float(math.acos(dot))
    theta = theta_end

    if angle is not None:
        try:
            ang = float(angle)
            if ang > 1e-9:
                theta = min(ang, 2.0 * math.pi)
                if theta < theta_end - 1e-6:
                    theta = theta_end
        except Exception:
            pass

    r = 0.5 * (r1 + r2)
    ts = np.linspace(0.0, theta, max(2, int(n)))
    pts = []
    for t in ts:
        v = math.cos(t) * v1n + math.sin(t) * b2n
        p = c + r * v
        pts.append([float(p[0]), float(p[1]), float(p[2])])
    return pts


# =========================
# Detection (from adjacency only) with anti-line filters for bspline
# =========================
def detect_through_hole_edges_from_adjacent(
    geometry_graph: Dict[str, Any],
    *,
    bspline_rms_tol: float = 0.25,
    require_arc_center: bool = False,
    # --- anti-line filters (bspline only) ---
    bspline_min_angle_deg: float = 8.0,
    bspline_max_radius: float = 500.0,
    bspline_min_sagitta_abs: float = 0.30,
    bspline_min_sagitta_ratio: float = 0.02,
    # --- chamfer-type through-hole detection ---
    detect_chamfer_holes: bool = False,
    chamfer_min_angle_deg: float = 90.0,
    chamfer_max_angle_deg: float = 150.0,
    chamfer_max_length: float = 50.0,
    debug: bool = False,
) -> Tuple[Dict[str, Any], Set[str]]:
    """
    Returns:
      - process_graph (process attributes only)
      - candidate_edge_ids (set of str gids) for visualization
    """
    nodes_raw = geometry_graph.get("nodes", {}) or {}
    geom_edges_raw = geometry_graph.get("geom_edges", {}) or {}
    weld_edges_raw = geometry_graph.get("weld_edges", {}) or {}

    nodes: Dict[str, Any] = {str(k): v for k, v in nodes_raw.items()}
    geom_edges: Dict[str, Any] = {str(k): v for k, v in geom_edges_raw.items()}
    weld_edges: Dict[str, Any] = {str(k): v for k, v in weld_edges_raw.items()}

    process_graph: Dict[str, Any] = {
        "meta": {
            "input_geometry_graph_meta": geometry_graph.get("meta", {}),
            "params": {
                "bspline_rms_tol": bspline_rms_tol,
                "require_arc_center": require_arc_center,
                "bspline_min_angle_deg": bspline_min_angle_deg,
                "bspline_max_radius": bspline_max_radius,
                "bspline_min_sagitta_abs": bspline_min_sagitta_abs,
                "bspline_min_sagitta_ratio": bspline_min_sagitta_ratio,
                "detect_chamfer_holes": detect_chamfer_holes,
                "chamfer_min_angle_deg": chamfer_min_angle_deg,
                "chamfer_max_angle_deg": chamfer_max_angle_deg,
                "chamfer_max_length": chamfer_max_length,
            },
            "notes": "process attributes only. geometry is in geometry_graph.json",
        },
        "nodes": {},
        "geom_edges": {},
    }

    for nid in nodes.keys():
        process_graph["nodes"][nid] = {"process": {"through_hole_edge_ids": []}}

    for gid in geom_edges.keys():
        process_graph["geom_edges"][gid] = {
            "process": {
                "through_hole_candidate": False,
                "through_hole_attached_nodes": [],
                "through_hole_geom": {},
                "through_hole_score": 0.0,
            }
        }

    candidate_ids: Set[str] = set()

    dbg = {
        "cand_total": 0,
        "edge_not_found": 0,
        "type_reject": 0,
        "arc_reject_center": 0,
        "bspline_fit_fail": 0,
        "bspline_rms_reject": 0,
        "bspline_radius_reject": 0,
        "bspline_angle_reject": 0,
        "bspline_sagitta_reject": 0,
        "hit_arc": 0,
        "hit_bspline": 0,
        "hit_total": 0,
        "chamfer_checked": 0,
        "chamfer_no_weld": 0,
        "chamfer_length_reject": 0,
        "chamfer_angle_reject": 0,
        "chamfer_hit": 0,
    }

    min_ang = math.radians(float(bspline_min_angle_deg))

    for nid, ninfo in nodes.items():
        cand = ninfo.get("adjacent_geom_edges", []) or []
        cand_ids_local = [str(x) for x in cand]

        for gid in cand_ids_local:
            dbg["cand_total"] += 1
            g = geom_edges.get(gid)
            if not g:
                dbg["edge_not_found"] += 1
                continue

            etype = g.get("type", "unknown")
            if etype not in ("arc", "bspline"):
                dbg["type_reject"] += 1
                continue

            s = g.get("start")
            e = g.get("end")
            if not s or not e:
                continue

            if etype == "arc":
                c = g.get("center")
                if require_arc_center and not c:
                    dbg["arc_reject_center"] += 1
                    continue

                r = g.get("radius", None)
                try:
                    r = float(r) if r is not None else None
                except Exception:
                    r = None

                ang = g.get("angle", None)
                try:
                    ang = float(ang) if ang is not None else 0.0
                except Exception:
                    ang = 0.0
                if ang <= 1e-9 and c:
                    ang = _angle_from_center(c, s, e)

                score = 1.0 if (r is not None and r > 1e-9) else 0.8
                geom_info = {"type": "arc", "center": c, "radius": r, "angle": ang}

                dbg["hit_arc"] += 1
                dbg["hit_total"] += 1

            else:
                samples = g.get("samples", [])
                fit = _fit_circle_3d(samples)
                if fit is None:
                    dbg["bspline_fit_fail"] += 1
                    continue

                rms = float(fit["rms"])
                if rms > bspline_rms_tol:
                    dbg["bspline_rms_reject"] += 1
                    continue

                radius = float(fit["radius"])
                if radius <= 1e-9 or radius > float(bspline_max_radius):
                    dbg["bspline_radius_reject"] += 1
                    continue

                ang = float(fit["angle_est"])
                if ang < min_ang:
                    dbg["bspline_angle_reject"] += 1
                    continue

                # sagitta (distance to chord) filter: reject almost straight
                chord_len = float(np.linalg.norm(np.array(e, float) - np.array(s, float)))
                min_sag = max(float(bspline_min_sagitta_abs), float(bspline_min_sagitta_ratio) * chord_len)
                sag = _max_chord_deviation(samples, s, e)
                if sag < min_sag:
                    dbg["bspline_sagitta_reject"] += 1
                    continue

                score = float(max(0.0, 1.0 - rms / max(bspline_rms_tol, 1e-9)))

                geom_info = {
                    "type": "bspline_circle_fit",
                    "center": fit["center"],
                    "radius": radius,
                    "angle_est": ang,
                    "plane_normal": fit["plane_normal"],
                    "fit_rms": rms,
                    "chord_len": chord_len,
                    "sagitta": sag,
                    "min_sagitta_used": min_sag,
                }

                dbg["hit_bspline"] += 1
                dbg["hit_total"] += 1

            # mark candidate
            candidate_ids.add(gid)

            pe = process_graph["geom_edges"][gid]["process"]
            if (not pe["through_hole_candidate"]) or (score > float(pe["through_hole_score"])):
                pe["through_hole_candidate"] = True
                pe["through_hole_score"] = float(score)
                pe["through_hole_geom"] = geom_info

            if nid not in pe["through_hole_attached_nodes"]:
                pe["through_hole_attached_nodes"].append(nid)

            pn = process_graph["nodes"][nid]["process"]
            if gid not in pn["through_hole_edge_ids"]:
                pn["through_hole_edge_ids"].append(gid)

    # ========== Chamfer-type through-hole detection ==========
    if detect_chamfer_holes:
        chamfer_min_angle_rad = math.radians(float(chamfer_min_angle_deg))
        chamfer_max_angle_rad = math.radians(float(chamfer_max_angle_deg))

        # 预建坐标 -> 节点ID 查找表，用于反查倒角边另一端节点
        _pt_tol = 0.5  # 坐标匹配容差 (mm)
        _node_by_pt: Dict[Tuple, str] = {}
        for _nid, _ninfo in nodes.items():
            _pt = _ninfo.get("point")
            if _pt:
                _key = tuple(round(float(v) / _pt_tol) for v in _pt)
                _node_by_pt[_key] = _nid

        def _find_node_by_coord(coord, tol=_pt_tol) -> Optional[str]:
            """通过坐标查找最近节点ID，容差内返回节点ID，否则返回None。"""
            if not coord:
                return None
            key = tuple(round(float(v) / tol) for v in coord)
            return _node_by_pt.get(key)

        # 遍历所有节点，检查是否有焊缝与临边形成钝角
        for nid, ninfo in nodes.items():
            node_point = ninfo.get("point")
            if not node_point:
                continue
            
            # 获取该节点的焊缝和临边
            incident_welds = ninfo.get("incident_weld_edges", []) or []
            adjacent_geoms = ninfo.get("adjacent_geom_edges", []) or []
            
            incident_welds = [str(x) for x in incident_welds]
            adjacent_geoms = [str(x) for x in adjacent_geoms]
            
            if not incident_welds or not adjacent_geoms:
                continue
            
            # 检查每条临边
            for gid in adjacent_geoms:
                dbg["chamfer_checked"] += 1
                
                g = geom_edges.get(gid)
                if not g:
                    continue
                
                # 跳过已经被标记为圆弧型过焊孔的边
                if gid in candidate_ids:
                    continue
                
                # 检查边的长度（倒角型过焊孔通常较短）
                edge_length = g.get("length", None)
                if edge_length is None:
                    # 计算长度
                    s = g.get("start")
                    e = g.get("end")
                    if s and e:
                        edge_length = float(np.linalg.norm(np.array(e, float) - np.array(s, float)))
                    else:
                        continue
                else:
                    edge_length = float(edge_length)
                
                if edge_length > chamfer_max_length:
                    dbg["chamfer_length_reject"] += 1
                    continue
                
                # 检查该边与焊缝的夹角
                max_angle = 0.0
                has_obtuse_angle = False
                
                for wid in incident_welds:
                    w = weld_edges.get(wid)
                    if not w:
                        continue
                    
                    # 计算夹角
                    angle = _compute_angle_between_edges(g, w, node_point)
                    if angle is None:
                        continue
                    
                    if angle > max_angle:
                        max_angle = angle
                    
                    # 如果夹角在合理范围内（下界到上界之间）
                    if chamfer_min_angle_rad <= angle <= chamfer_max_angle_rad:
                        has_obtuse_angle = True
                
                if not has_obtuse_angle:
                    dbg["chamfer_angle_reject"] += 1
                    continue
                
                if not incident_welds:
                    dbg["chamfer_no_weld"] += 1
                    continue
                
                # 标记为倒角型过焊孔
                score = 0.7  # 倒角型的置信度略低于圆弧型
                geom_info = {
                    "type": "chamfer",
                    "length": edge_length,
                    "max_angle_deg": math.degrees(max_angle),
                    "max_angle_rad": max_angle,
                }
                
                candidate_ids.add(gid)
                
                pe = process_graph["geom_edges"][gid]["process"]
                if (not pe["through_hole_candidate"]) or (score > float(pe["through_hole_score"])):
                    pe["through_hole_candidate"] = True
                    pe["through_hole_score"] = float(score)
                    pe["through_hole_geom"] = geom_info
                
                if nid not in pe["through_hole_attached_nodes"]:
                    pe["through_hole_attached_nodes"].append(nid)
                
                pn = process_graph["nodes"][nid]["process"]
                if gid not in pn["through_hole_edge_ids"]:
                    pn["through_hole_edge_ids"].append(gid)

                # 反查倒角边的另一端节点，也加入 attached_nodes
                # （避免单侧打断点的情况）
                g_data = geom_edges.get(gid, {})
                g_start = g_data.get("start")
                g_end = g_data.get("end")
                cur_pt = np.array(node_point, dtype=float)
                for other_coord in (g_start, g_end):
                    if other_coord is None:
                        continue
                    other_pt = np.array(other_coord, dtype=float)
                    if float(np.linalg.norm(other_pt - cur_pt)) < _pt_tol * 2:
                        # 这一端就是当前节点，跳过
                        continue
                    other_nid = _find_node_by_coord(other_coord)
                    if other_nid and other_nid != nid:
                        if other_nid not in pe["through_hole_attached_nodes"]:
                            pe["through_hole_attached_nodes"].append(other_nid)
                        pn_other = process_graph["nodes"].get(other_nid, {}).get("process")
                        if pn_other is not None and gid not in pn_other["through_hole_edge_ids"]:
                            pn_other["through_hole_edge_ids"].append(gid)

                dbg["chamfer_hit"] += 1
                dbg["hit_total"] += 1

    if debug:
        print("========== detect debug (adjacent arc-like, anti-line bspline, chamfer) ==========")
        for k, v in dbg.items():
            print(f"{k:30s}: {v}")
        print(f"{'total_candidates':30s}: {len(candidate_ids)}")
        
        # 统计 candidate_ids 中各类型的数量
        arc_cand = 0
        bspline_cand = 0
        line_cand = 0
        other_cand = 0
        for gid in candidate_ids:
            g = geom_edges.get(str(gid))
            if g:
                et = g.get("type", "unknown")
                if et == "arc":
                    arc_cand += 1
                elif et == "bspline":
                    bspline_cand += 1
                elif et == "line":
                    line_cand += 1
                else:
                    other_cand += 1
        
        print(f"{'candidate_types':30s}: arc={arc_cand}, bspline={bspline_cand}, line={line_cand}, other={other_cand}")
        print("===================================================================================")

    return process_graph, candidate_ids


# =========================
# Breakpoint insertion helpers
# =========================
def _next_node_id(nodes: Dict[str, Any]) -> str:
    mx = 0
    for k in nodes.keys():
        if isinstance(k, str) and k.startswith("N"):
            try:
                mx = max(mx, int(k[1:]))
            except Exception:
                pass
    return f"N{mx + 1}"


def _ensure_weld_samples(w: Dict[str, Any], default_arc_n: int = 64) -> List[List[float]]:
    """
    Make a polyline representation for the weld edge so we can insert a midpoint.
    - line: [start, end]
    - arc: sampled points if center exists else [start, end]
    - bspline: use existing samples if present else [start, end]
    """
    et = w.get("type", "unknown")
    s = w.get("start")
    e = w.get("end")
    if not s or not e:
        return []

    if et == "bspline" and w.get("samples"):
        return list(w["samples"])

    if et == "arc":
        c = w.get("center")
        if c:
            pts = _arc_polyline(c, s, e, angle=w.get("angle", None), n=default_arc_n)
            return pts
        return [s, e]

    # line or unknown
    return [s, e]


def _polyline_midpoint_by_arclen(samples: List[List[float]]) -> Tuple[List[float], int]:
    """
    Return midpoint (by arc-length) and insert index (where to insert in samples).
    The returned index i means insert AFTER i-1 (i is the position in list).
    """
    if not samples or len(samples) < 2:
        return samples[0] if samples else [0.0, 0.0, 0.0], 0

    pts = np.array(samples, dtype=float)
    seg = pts[1:] - pts[:-1]
    d = np.linalg.norm(seg, axis=1)
    total = float(np.sum(d))
    if total <= 1e-12:
        mid = pts[len(pts)//2].tolist()
        return [float(mid[0]), float(mid[1]), float(mid[2])], len(samples)//2

    half = 0.5 * total
    acc = 0.0
    for i, di in enumerate(d):
        if acc + float(di) >= half:
            t = (half - acc) / max(float(di), 1e-12)
            p = pts[i] + t * (pts[i+1] - pts[i])
            # insert point between i and i+1 => index i+1
            return [float(p[0]), float(p[1]), float(p[2])], i + 1
        acc += float(di)

    # fallback end
    mid = pts[-1].tolist()
    return [float(mid[0]), float(mid[1]), float(mid[2])], len(samples)


def insert_breakpoint_for_node(
    geometry_graph: Dict[str, Any],
    nid: str,
    *,
    hole_edge_id: Optional[str] = None,     # 触发该节点的过焊孔边ID，用于精准定位目标焊缝
    require_hole_edge_match: bool = False,  # True 时：若 hole_edge_id 匹配失败则不回退 shortest_weld
    hole_edge_endpoint_tol: float = 1.0,    # 判断端点重合的距离容差
    weld_hole_max_collinear_deg: float = 30.0,  # 焊缝与过焊孔边夹角小于此值视为共线（延长线），排除
    breakpoint_strategy: str = "shortest_weld",  # "shortest_weld" or "all_welds"
    breakpoint_position: str = "midpoint",  # "midpoint" or "near_node"
    near_node_ratio: float = 0.3,  # if near_node, insert at this ratio from node
    min_weld_length: float = 3.0,           # 只在长度大于此值的焊缝上插入断点（过滤倒角短边）
    max_weld_length: float = float('inf'),  # 只在长度小于此值的焊缝上插入断点
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    For node nid (connected to through-hole):
      hole_edge_id: 如果传入，优先选择端点与该过焊孔边端点空间重合的焊缝，
                   比 shortest_weld 更精准。
            require_hole_edge_match: 当 hole_edge_id 传入但未匹配到合法焊缝时，
                                     若为 True 则直接跳过，不使用 shortest_weld 回退。
      weld_hole_max_collinear_deg: 焊缝与过焊孔边夹角小于此值时视为共线（焊缝在过焊孔延长线上），
                   排除该焊缝，避免断点打在延长线方向的焊缝上。默认 30°。
      Strategy options:
        - "shortest_weld": insert breakpoint only on shortest incident weld
        - "all_welds": insert breakpoint on all incident welds
      
      Position options:
        - "midpoint": insert at weld midpoint (arc-length)
        - "near_node": insert near the through-hole node (at near_node_ratio from node)
      
      Length threshold:
        - min_weld_length: only insert breakpoint if weld length > min_weld_length (filter chamfer edges)
        - max_weld_length: only insert breakpoint if weld length < max_weld_length
    
    Return info dict or None.
    """
    nodes = geometry_graph.get("nodes", {}) or {}
    weld_edges = geometry_graph.get("weld_edges", {}) or {}
    geom_edges = geometry_graph.get("geom_edges", {}) or {}

    ninfo = nodes.get(nid)
    if not ninfo:
        return None

    incident = ninfo.get("incident_weld_edges", []) or []
    incident = [str(x) for x in incident]
    if not incident:
        return None

    # Determine which welds to process
    target_welds = []

    # 优先策略：通过过焊孔边端点精准匹配焊缝
    if hole_edge_id is not None:
        hole_edge = geom_edges.get(str(hole_edge_id))
        if hole_edge:
            hole_start = hole_edge.get("start")
            hole_end = hole_edge.get("end")
            hole_pts = [np.array(p, dtype=float) for p in [hole_start, hole_end] if p]

            matched_welds = []
            for wid in incident:
                w = weld_edges.get(wid)
                if not w:
                    continue
                w_start = w.get("start")
                w_end = w.get("end")
                w_pts = [np.array(p, dtype=float) for p in [w_start, w_end] if p]
                # 检查焊缝任意端点是否与过焊孔边任意端点足够近
                matched = False
                for wp in w_pts:
                    for hp in hole_pts:
                        if float(np.linalg.norm(wp - hp)) < hole_edge_endpoint_tol:
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    matched_welds.append(wid)

            if matched_welds:
                # 过滤1：长度范围
                valid_welds = [
                    wid for wid in matched_welds
                    if min_weld_length <= float((weld_edges.get(wid) or {}).get("length", 0)) <= max_weld_length
                ]

                # 过滤2：角度过滤 —— 焊缝不能在过焊孔延长线上（夹角不能接近 180°）
                # 即：焊缝切向量与过焊孔边切向量的夹角必须 < weld_hole_max_collinear_deg
                angle_filtered = []
                node_pt = np.array(ninfo.get("point", [0, 0, 0]), dtype=float)
                for wid in valid_welds:
                    w = weld_edges.get(wid)
                    if not w:
                        continue
                    # 计算焊缝在节点处的切向量
                    t_weld = _compute_edge_tangent_at_node(w, node_pt.tolist())
                    # 计算过焊孔边在节点处的切向量
                    t_hole = _compute_edge_tangent_at_node(hole_edge, node_pt.tolist())
                    if t_weld is None or t_hole is None:
                        # 无法计算角度，保留
                        angle_filtered.append(wid)
                        continue
                    # 两者夹角（0~180°）
                    dot = float(np.clip(np.dot(t_weld, t_hole), -1.0, 1.0))
                    angle_deg = math.degrees(math.acos(abs(dot)))  # 取绝对值处理方向
                    # 夹角接近 0° 或 180° 说明共线（焊缝在过焊孔延长线上），排除
                    if angle_deg > weld_hole_max_collinear_deg:
                        angle_filtered.append(wid)
                        if verbose:
                            print(f"[breakpoint] weld {wid} angle={angle_deg:.1f}deg vs hole_edge -> pass")
                    else:
                        if verbose:
                            print(f"[breakpoint] weld {wid} angle={angle_deg:.1f}deg vs hole_edge -> collinear, skip")

                if angle_filtered:
                    angle_filtered.sort(key=lambda wid: float(
                        (weld_edges.get(wid) or {}).get("length", float("inf"))
                    ))
                    target_welds = [angle_filtered[0]]
                    if verbose:
                        print(f"[breakpoint] node={nid} hole_edge={hole_edge_id} -> "
                              f"matched weld={target_welds[0]} by endpoint proximity + angle filter")
                elif valid_welds:
                    if verbose:
                        print(f"[breakpoint] node={nid} hole_edge={hole_edge_id} -> "
                              f"all matched welds are collinear with hole_edge, falling back")
                else:
                    if verbose:
                        print(f"[breakpoint] node={nid} hole_edge={hole_edge_id} -> "
                              f"all matched welds filtered by length, falling back to shortest_weld")
            else:
                if verbose:
                    print(f"[breakpoint] node={nid} hole_edge={hole_edge_id} -> "
                          f"no endpoint-matched weld, falling back to shortest_weld")

    if hole_edge_id is not None and require_hole_edge_match and not target_welds:
        if verbose:
            print(
                f"[breakpoint] node={nid} hole_edge={hole_edge_id} -> "
                "strict hole-edge matching enabled, skip insertion"
            )
        return None

    # 回退策略
    if not target_welds:
        if breakpoint_strategy == "shortest_weld":
            best_wid = None
            best_len = float("inf")
            for wid in incident:
                w = weld_edges.get(wid)
                if not w:
                    continue
                try:
                    L = float(w.get("length", float("inf")))
                except Exception:
                    L = float("inf")
                # 回退策略也遵守长度范围
                if L < min_weld_length or L > max_weld_length:
                    continue
                if L < best_len:
                    best_len = L
                    best_wid = wid
            if best_wid:
                target_welds = [best_wid]
        
        elif breakpoint_strategy == "all_welds":
            target_welds = incident
        
        else:
            if verbose:
                print(f"[breakpoint] unknown strategy: {breakpoint_strategy}")
            return None
    
    if not target_welds:
        return None
    
    # Insert breakpoints on target welds
    results = []
    node_point = np.array(ninfo.get("point", [0, 0, 0]), dtype=float)
    skipped_count = 0
    
    for wid in target_welds:
        w = weld_edges.get(wid)
        if not w:
            continue
        
        # Check weld length threshold (already filtered in selection stage, double-check here)
        weld_len = float(w.get("length", 0))
        if weld_len < min_weld_length:
            if verbose:
                print(f"[breakpoint] weld {wid} (len={weld_len:.3f}) below min_length={min_weld_length:.3f}, skipping")
            skipped_count += 1
            continue
        if weld_len > max_weld_length:
            if verbose:
                print(f"[breakpoint] weld {wid} (len={weld_len:.3f}) exceeds max_length={max_weld_length:.3f}, skipping")
            skipped_count += 1
            continue
        
        # Skip if already has breakpoint
        if w.get("has_breakpoint", False):
            if verbose:
                print(f"[breakpoint] weld {wid} already has breakpoint, skipping")
            continue
        
        # build samples
        samples = _ensure_weld_samples(w, default_arc_n=64)
        if len(samples) < 2:
            continue
        
        # Determine breakpoint position
        if breakpoint_position == "midpoint":
            bp_xyz, insert_idx = _polyline_midpoint_by_arclen(samples)
        
        elif breakpoint_position == "near_node":
            # Insert near the through-hole node
            bp_xyz, insert_idx = _polyline_point_near_node(samples, node_point, near_node_ratio)
        
        else:
            if verbose:
                print(f"[breakpoint] unknown position: {breakpoint_position}")
            continue
        
        # Create new breakpoint node
        new_nid = _next_node_id(nodes)
        nodes[new_nid] = {
            "point": bp_xyz,
            "key": f"BREAK:{new_nid}",
            "incident_weld_edges": [wid],
            "adjacent_geom_edges": [],
            "process": {"is_breakpoint": True},
        }
        
        # Update weld edge
        samples2 = samples[:]
        samples2.insert(insert_idx, bp_xyz)
        
        w["samples"] = samples2
        w["breakpoint_node"] = new_nid
        w["breakpoint_point"] = bp_xyz
        w["has_breakpoint"] = True
        
        weld_edges[wid] = w
        
        if verbose:
            print(f"[breakpoint] node={nid} -> weld={wid} (len={weld_len:.3f}) "
                  f"strategy={breakpoint_strategy} position={breakpoint_position} "
                  f"insert_idx={insert_idx} new_node={new_nid}")
        
        results.append({
            "src_node": nid,
            "weld_id": wid,
            "weld_length": weld_len,
            "break_node": new_nid,
            "break_point": bp_xyz,
        })
    
    # Write back
    geometry_graph["nodes"] = nodes
    geometry_graph["weld_edges"] = weld_edges
    
    if skipped_count > 0 and verbose:
        print(f"[breakpoint] node={nid}: skipped {skipped_count} weld(s) due to length threshold")
    
    return results[0] if len(results) == 1 else {"multiple": results} if results else None


def _polyline_point_near_node(
    samples: List[List[float]], 
    node_point: np.ndarray, 
    ratio: float = 0.3
) -> Tuple[List[float], int]:
    """
    Find a point on the polyline near the given node.
    Returns point at 'ratio' of arc-length from the nearest endpoint to node.
    
    Args:
        samples: polyline samples
        node_point: the through-hole node position
        ratio: how far along the weld to insert (0.0 = at node, 0.5 = midpoint)
    
    Returns:
        (breakpoint_xyz, insert_index)
    """
    if not samples or len(samples) < 2:
        return samples[0] if samples else [0.0, 0.0, 0.0], 0
    
    pts = np.array(samples, dtype=float)
    
    # Find which endpoint is closer to node
    dist_start = float(np.linalg.norm(pts[0] - node_point))
    dist_end = float(np.linalg.norm(pts[-1] - node_point))
    
    # Calculate arc lengths
    seg = pts[1:] - pts[:-1]
    d = np.linalg.norm(seg, axis=1)
    total = float(np.sum(d))
    
    if total <= 1e-12:
        mid = pts[len(pts)//2].tolist()
        return [float(mid[0]), float(mid[1]), float(mid[2])], len(samples)//2
    
    # Determine target distance based on which end is closer
    if dist_start < dist_end:
        # Node is near start, insert at ratio from start
        target_dist = ratio * total
        reverse = False
    else:
        # Node is near end, insert at ratio from end
        target_dist = (1.0 - ratio) * total
        reverse = False
    
    # Find insertion point
    acc = 0.0
    for i, di in enumerate(d):
        if acc + float(di) >= target_dist:
            t = (target_dist - acc) / max(float(di), 1e-12)
            p = pts[i] + t * (pts[i+1] - pts[i])
            return [float(p[0]), float(p[1]), float(p[2])], i + 1
        acc += float(di)
    
    # Fallback
    mid = pts[-1].tolist()
    return [float(mid[0]), float(mid[1]), float(mid[2])], len(samples)


# =========================
# T-type breakpoint detection
# =========================
def _line_line_intersection_3d(
    p1: np.ndarray, d1: np.ndarray,
    p2: np.ndarray, d2: np.ndarray,
    eps: float = 1e-6,
    max_dist: float = 5.0,
) -> Optional[Tuple[np.ndarray, float, float, float]]:
    """
    Find closest approach point of two RAYS in 3D.
    Ray 1: p1 + t * d1  (t >= 0)
    Ray 2: p2 + s * d2  (s >= 0)

    Returns (midpoint_of_closest_approach, skew_distance, t, s) if rays are not parallel,
    else None. The caller must check t >= 0 and s >= 0 to ensure the intersection
    is in the forward direction of both rays.
    """
    d1_norm = _safe_norm(d1)
    d2_norm = _safe_norm(d2)
    if d1_norm < 1e-9 or d2_norm < 1e-9:
        return None

    d1 = d1 / d1_norm
    d2 = d2 / d2_norm

    w = p1 - p2

    a = float(np.dot(d1, d1))  # always 1.0
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))  # always 1.0
    d = float(np.dot(d1, w))
    e = float(np.dot(d2, w))

    denom = a * c - b * b  # 1 - cos^2(theta)
    if abs(denom) < eps:
        # Lines are parallel
        return None

    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom

    pt1 = p1 + t * d1
    pt2 = p2 + s * d2

    skew_dist = float(np.linalg.norm(pt1 - pt2))
    midpoint = (pt1 + pt2) / 2.0
    return (midpoint, skew_dist, float(t), float(s))


def _line_segment_intersection_3d(
    p1: np.ndarray, d1: np.ndarray,
    seg_start: np.ndarray, seg_end: np.ndarray,
    eps: float = 1e-6
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Find intersection of a line with a line segment in 3D.
    Line: p1 + t * d1 (t can be any value)
    Segment: seg_start + s * (seg_end - seg_start), s in [0, 1]
    
    Returns (intersection_point, parameter_s) if intersection exists, else None.
    """
    seg_dir = seg_end - seg_start
    seg_len = _safe_norm(seg_dir)
    if seg_len < eps:
        return None
    
    seg_dir = seg_dir / seg_len
    
    d1_norm = _safe_norm(d1)
    if d1_norm < eps:
        return None
    d1 = d1 / d1_norm
    
    w = seg_start - p1
    
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, seg_dir))
    c = float(np.dot(seg_dir, seg_dir))
    d = float(np.dot(d1, w))
    e = float(np.dot(seg_dir, w))
    
    denom = a * c - b * b
    if abs(denom) < eps:
        return None
    
    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    
    # Check if s is in [0, 1]
    if s < -eps or s > 1.0 + eps:
        return None
    
    pt_line = p1 + t * d1
    pt_seg = seg_start + s * seg_dir
    
    dist = float(np.linalg.norm(pt_line - pt_seg))
    if dist < eps:
        return (pt_seg, max(0.0, min(1.0, s)))
    
    return None


def _are_coplanar(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
    eps: float = 0.5
) -> bool:
    """
    Check if 4 points are approximately coplanar.
    Uses the volume of tetrahedron formed by the 4 points.
    """
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p1
    
    volume = abs(float(np.dot(v1, np.cross(v2, v3))))
    
    # Normalize by edge lengths
    l1 = _safe_norm(v1)
    l2 = _safe_norm(v2)
    l3 = _safe_norm(v3)
    
    if l1 < 1e-9 or l2 < 1e-9 or l3 < 1e-9:
        return True
    
    normalized_volume = volume / (l1 * l2 * l3)
    
    return normalized_volume < eps


def detect_t_type_breakpoints(
    geometry_graph: Dict[str, Any],
    candidate_edge_ids: Set[str],
    *,
    t_type_min_weld_length: float = 5.0,
    t_type_max_weld_length: float = 500.0,
    t_type_extension_ratio: float = 2.0,
    t_type_max_distance_to_weld: float = 10.0,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Detect T-type breakpoints on through-hole edges.
    
    Simplified algorithm - only checks weld length threshold:
    For each through-hole edge:
      1. Find two incident welds A, B at the two endpoints
      2. Check if both welds meet length requirements
      3. Extend A, B towards the hole interior
      4. Find intersection of extended lines
      5. Find closest third weld C to intersection point
      6. If intersection exists and C is nearby, mark as T-type breakpoint
    
    Args:
        geometry_graph: the geometry graph
        candidate_edge_ids: set of through-hole edge IDs
        t_type_min_weld_length: minimum weld length (mm)
        t_type_max_weld_length: maximum weld length (mm)
        t_type_extension_ratio: how far to extend welds (ratio of hole edge length)
        t_type_max_distance_to_weld: max distance from intersection to weld C
        debug: print debug info
    
    Returns:
        Dict with T-type breakpoint info
    """
    nodes = geometry_graph.get("nodes", {}) or {}
    geom_edges = geometry_graph.get("geom_edges", {}) or {}
    weld_edges = geometry_graph.get("weld_edges", {}) or {}
    
    t_type_info = {
        "t_type_breakpoints": [],
        "debug": {
            "total_hole_edges": len(candidate_edge_ids),
            "checked": 0,
            "no_two_welds": 0,
            "weld_length_reject": 0,
            "no_intersection": 0,
            "found": 0,
        }
    }
    
    for hole_gid in candidate_edge_ids:
        hole_edge = geom_edges.get(str(hole_gid))
        if not hole_edge:
            continue
        
        t_type_info["debug"]["checked"] += 1
        
        hole_start = hole_edge.get("start")
        hole_end = hole_edge.get("end")
        if not hole_start or not hole_end:
            continue
        
        hole_start_pt = np.array(hole_start, dtype=float)
        hole_end_pt = np.array(hole_end, dtype=float)
        hole_dir = hole_end_pt - hole_start_pt
        hole_len = _safe_norm(hole_dir)
        if hole_len < 1e-9:
            continue
        hole_dir = hole_dir / hole_len
        
        if debug:
            print(f"\n[T-type] ===== hole_edge={hole_gid} =====")
            print(f"  hole_start={hole_start}, hole_end={hole_end}")
            print(f"  hole_len={hole_len:.3f}")
        
        # Find nodes at hole edge endpoints
        nodes_at_hole = []
        for nid, ninfo in nodes.items():
            npt = ninfo.get("point")
            if not npt:
                continue
            npt_arr = np.array(npt, dtype=float)
            if float(np.linalg.norm(npt_arr - hole_start_pt)) < 1.0:
                nodes_at_hole.append((nid, "start"))
            elif float(np.linalg.norm(npt_arr - hole_end_pt)) < 1.0:
                nodes_at_hole.append((nid, "end"))
        
        if debug:
            print(f"  nodes_at_hole: {nodes_at_hole}")
        
        if len(nodes_at_hole) < 2:
            t_type_info["debug"]["no_two_welds"] += 1
            if debug:
                print(f"  -> SKIP: less than 2 nodes at hole endpoints")
            continue
        
        # Get welds at each endpoint
        welds_at_start = []
        welds_at_end = []
        
        for nid, endpoint in nodes_at_hole:
            ninfo = nodes.get(nid)
            if not ninfo:
                continue
            incident_welds = ninfo.get("incident_weld_edges", []) or []
            
            if endpoint == "start":
                welds_at_start.extend([str(w) for w in incident_welds])
            else:
                welds_at_end.extend([str(w) for w in incident_welds])
        
        if debug:
            print(f"  welds_at_start: {welds_at_start}")
            print(f"  welds_at_end: {welds_at_end}")
        
        if not welds_at_start or not welds_at_end:
            t_type_info["debug"]["no_two_welds"] += 1
            if debug:
                print(f"  -> SKIP: no welds at one or both endpoints")
            continue
        
        # ==== 新逻辑：每端只取最长焊缝，用采样点计算切线 ====
        
        def _pick_longest_weld_at_endpoint(weld_ids, hole_pt, endpoint_tol=1.0):
            """
            从 weld_ids 中找到与 hole_pt 相连（端点在容差内）且最长的焊缝。
            返回 (weld_id, weld_data, at_hole_pt, tangent_dir) 或 None。
            tangent_dir 是从 at_hole_pt 出发、沿焊缝方向的单位切向量（用采样点计算）。
            """
            best_wid = None
            best_len = -1.0
            best_at_hole = None
            best_tangent = None
            best_w = None
            
            for wid in weld_ids:
                w = weld_edges.get(wid)
                if not w:
                    continue
                ws = w.get("start")
                we = w.get("end")
                if not ws or not we:
                    continue
                
                ws_pt = np.array(ws, dtype=float)
                we_pt = np.array(we, dtype=float)
                hp = np.array(hole_pt, dtype=float)
                
                dist_s = float(np.linalg.norm(ws_pt - hp))
                dist_e = float(np.linalg.norm(we_pt - hp))
                
                if dist_s < endpoint_tol:
                    at_hole = ws_pt
                    at_hole_idx = 0       # 焊缝起点在孔边
                elif dist_e < endpoint_tol:
                    at_hole = we_pt
                    at_hole_idx = -1      # 焊缝终点在孔边
                else:
                    continue  # 该焊缝不连接到这个端点
                
                w_len = float(w.get("length", 0))
                if w_len < t_type_min_weld_length or w_len > t_type_max_weld_length:
                    continue
                
                # 用采样点计算切线：
                # 找采样点中最靠近孔边端点的那一端，
                # 取端点起的前 n_tangent 个采样点做累积方向，提高稳定性
                samples = w.get("samples", [])
                if not samples or len(samples) < 2:
                    # 无采样点，fallback 到 start/end
                    # 注意：这里 tangent 统一定义为“从孔端点指向焊缝内部”
                    if at_hole_idx == 0:
                        at_hole = ws_pt
                        tangent = we_pt - ws_pt
                        tangent_sample_far = we_pt
                    else:
                        at_hole = we_pt
                        tangent = ws_pt - we_pt
                        tangent_sample_far = ws_pt
                else:
                    pts_arr = [np.array(p, dtype=float) for p in samples]
                    dist_sample_0 = float(np.linalg.norm(pts_arr[0] - hp))
                    dist_sample_n = float(np.linalg.norm(pts_arr[-1] - hp))

                    if dist_sample_0 < dist_sample_n:
                        # 孔端点在 samples 起点这一侧
                        at_hole = pts_arr[0]
                        # 只用“端点第一段”作为切线，保证和可视化出来的焊缝末端严格共线
                        tangent = pts_arr[1] - pts_arr[0]
                        tangent_sample_far = pts_arr[1]
                    else:
                        # 孔端点在 samples 终点这一侧
                        at_hole = pts_arr[-1]
                        # 只用“端点第一段”作为切线，注意方向仍然定义为“从孔端点指向焊缝内部”
                        tangent = pts_arr[-2] - pts_arr[-1]
                        tangent_sample_far = pts_arr[-2]


                
                t_norm = _safe_norm(tangent)
                if t_norm < 1e-9:
                    continue
                tangent = tangent / t_norm

                print(f"\n[DEBUG weld {wid}]")
                print(f"  hole_pt      = {hp}")
                print(f"  weld start   = {ws_pt}")
                print(f"  weld end     = {we_pt}")
                print(f"  dist(start,hole) = {dist_s:.3f}")
                print(f"  dist(end,hole)   = {dist_e:.3f}")

                if samples and len(samples) >= 2:
                    print(f"  sample[0]    = {pts_arr[0]}")
                    print(f"  sample[-1]   = {pts_arr[-1]}")
                    print(f"  dist(sample0,hole) = {dist_sample_0:.3f}")
                    print(f"  dist(sampleN,hole) = {dist_sample_n:.3f}")
                    if dist_sample_0 < dist_sample_n:
                        print(f"  chosen at_hole = sample[0]")
                        print(f"  chosen sample_far = sample[1] = {pts_arr[1]}")
                        print(f"  tangent(into_weld) = {pts_arr[1] - pts_arr[0]}")
                        print(f"  tangent(extension) = {pts_arr[0] - pts_arr[1]}")
                    else:
                        print(f"  chosen at_hole = sample[-1]")
                        print(f"  chosen sample_far = sample[-2] = {pts_arr[-2]}")
                        print(f"  tangent(into_weld) = {pts_arr[-2] - pts_arr[-1]}")
                        print(f"  tangent(extension) = {pts_arr[-1] - pts_arr[-2]}")
                else:
                    print("  no samples")

                
                if w_len > best_len:
                    best_len = w_len
                    best_wid = wid
                    best_at_hole = at_hole
                    best_tangent = tangent
                    best_tangent_sample_far = tangent_sample_far
                    best_w = w
            
            if best_wid is None:
                return None
            return (best_wid, best_w, best_at_hole, best_tangent, best_tangent_sample_far)
        
        # 每端取最长焊缝
        result_a = _pick_longest_weld_at_endpoint(welds_at_start, hole_start)
        result_b = _pick_longest_weld_at_endpoint(welds_at_end, hole_end)
        
        if result_a is None or result_b is None:
            t_type_info["debug"]["weld_length_reject"] += 1
            if debug:
                if result_a is None:
                    print(f"  -> SKIP: no valid weld at hole_start (after length filter)")
                else:
                    print(f"  -> SKIP: no valid weld at hole_end (after length filter)")
            continue
        
        weld_a_id, weld_a, weld_a_at_hole, weld_a_dir, weld_a_sample_far = result_a
        weld_b_id, weld_b, weld_b_at_hole, weld_b_dir, weld_b_sample_far = result_b
        
        if weld_a_id == weld_b_id:
            if debug:
                print(f"  -> SKIP: same weld at both endpoints")
            continue
        
        if debug:
            print(f"  weld_a={weld_a_id} (len={weld_a.get('length', 0):.3f}) at_hole={weld_a_at_hole} dir={weld_a_dir}")
            print(f"  weld_b={weld_b_id} (len={weld_b.get('length', 0):.3f}) at_hole={weld_b_at_hole} dir={weld_b_dir}")
        
        # 延长线
        extension_len = float(t_type_extension_ratio) * hole_len
        weld_a_ext_end = weld_a_at_hole - extension_len * weld_a_dir
        weld_b_ext_end = weld_b_at_hole - extension_len * weld_b_dir    

        weld_a_sample_ext = weld_a_at_hole - (weld_a_sample_far - weld_a_at_hole)
        weld_b_sample_ext = weld_b_at_hole - (weld_b_sample_far - weld_b_at_hole)


        
        if debug:
            print(f"  extension_len={extension_len:.3f}")
            print(f"  weld_a: {weld_a_at_hole} -> {weld_a_ext_end}")
            print(f"  weld_b: {weld_b_at_hole} -> {weld_b_ext_end}")
        

        # 求两条“可视化延长线”的最近点
        max_skew_dist = float(t_type_max_distance_to_weld)

        ray_a_p0 = np.array(weld_a_at_hole, dtype=float)
        ray_a_p1 = np.array(weld_a_ext_end, dtype=float)
        ray_b_p0 = np.array(weld_b_at_hole, dtype=float)
        ray_b_p1 = np.array(weld_b_ext_end, dtype=float)

        ray_a_dir = ray_a_p1 - ray_a_p0
        ray_b_dir = ray_b_p1 - ray_b_p0

        result_inter = _line_line_intersection_3d(
            ray_a_p0, ray_a_dir,
            ray_b_p0, ray_b_dir,
        )

        
        if result_inter is None:
            t_type_info["debug"]["no_intersection"] += 1
            if debug:
                print(f"  -> REJECT: lines are parallel")
            continue
        
        intersection_pt, skew_dist, t_param, s_param = result_inter
        
        if debug:
            print(f"  -> closest approach at {intersection_pt}, skew_dist={skew_dist:.3f} (max={max_skew_dist:.3f})")
            print(f"     t_param(A)={t_param:.3f}, s_param(B)={s_param:.3f}")
        
        if skew_dist > max_skew_dist:
            t_type_info["debug"]["no_intersection"] += 1
            if debug:
                print(f"  -> REJECT: skew_dist too large")
            continue
        
        # 注意：这里 t/s 的正方向仍然是“从孔端点指向焊缝内部”的方向
        # 所以这一步检查的是：最近点是否落在焊缝内部射线前方
        # 真正画图的“端点外延线”在可视化中使用的是 -weld_dir

        # 现在 t/s 的正方向就是“端点外延线方向”
        if t_param < 0 or s_param < 0:
            t_type_info["debug"]["no_intersection"] += 1
            if debug:
                print(f"  -> REJECT: intersection behind extension ray origin (t={t_param:.3f}, s={s_param:.3f})")
            continue

        
        # 检查交点在延长范围内
        dist_a = float(np.linalg.norm(intersection_pt - weld_a_at_hole))
        dist_b = float(np.linalg.norm(intersection_pt - weld_b_at_hole))
        
        if debug:
            print(f"  dist_a={dist_a:.3f}, dist_b={dist_b:.3f}, extension_len={extension_len:.3f}")
        
        if dist_a > extension_len or dist_b > extension_len:
            t_type_info["debug"]["no_intersection"] += 1
            if debug:
                print(f"  -> REJECT: intersection outside extension range")
            continue
        
        # 查找最近的第三条焊缝 C
        closest_weld_c = None
        closest_distance = float(t_type_max_distance_to_weld)
        
        if debug:
            print(f"  searching for weld C within {t_type_max_distance_to_weld:.3f}...")
        
        for wid, w in weld_edges.items():
            if wid in [weld_a_id, weld_b_id]:
                continue
            w_start = w.get("start")
            w_end = w.get("end")
            if not w_start or not w_end:
                continue
            w_start_pt = np.array(w_start, dtype=float)
            w_end_pt = np.array(w_end, dtype=float)
            seg_dir = w_end_pt - w_start_pt
            seg_len = _safe_norm(seg_dir)
            if seg_len < 1e-9:
                closest_pt = (w_start_pt + w_end_pt) / 2.0
            else:
                seg_dir = seg_dir / seg_len
                w_vec = intersection_pt - w_start_pt
                t_param = float(np.dot(w_vec, seg_dir))
                t_param = max(0.0, min(seg_len, t_param))
                closest_pt = w_start_pt + t_param * seg_dir
            dist = float(np.linalg.norm(intersection_pt - closest_pt))
            if dist < closest_distance:
                closest_distance = dist
                closest_weld_c = wid
                if debug:
                    print(f"    weld {wid}: dist={dist:.3f} (NEW BEST)")
            elif debug and dist < t_type_max_distance_to_weld * 2:
                print(f"    weld {wid}: dist={dist:.3f}")
        
        if closest_weld_c is None:
            t_type_info["debug"]["no_intersection"] += 1
            if debug:
                print(f"  -> REJECT: no weld C within {t_type_max_distance_to_weld}")
            continue
        
        if debug:
            print(f"  -> ACCEPTED: weld_c={closest_weld_c} dist={closest_distance:.3f}")
        
        t_type_info["t_type_breakpoints"].append({
            "hole_edge_id": str(hole_gid),
            "weld_a_id": weld_a_id,
            "weld_b_id": weld_b_id,
            "weld_c_id": closest_weld_c,
            "intersection_point": [float(intersection_pt[0]), float(intersection_pt[1]), float(intersection_pt[2])],
            "distance_to_weld_c": float(closest_distance),
            "skew_dist": float(skew_dist),
            "weld_a_at_hole": [float(weld_a_at_hole[0]), float(weld_a_at_hole[1]), float(weld_a_at_hole[2])],
            "weld_a_ext_end": [float(weld_a_ext_end[0]), float(weld_a_ext_end[1]), float(weld_a_ext_end[2])],
            "weld_a_sample_far": [float(weld_a_sample_far[0]), float(weld_a_sample_far[1]), float(weld_a_sample_far[2])],
            "weld_b_at_hole": [float(weld_b_at_hole[0]), float(weld_b_at_hole[1]), float(weld_b_at_hole[2])],
            "weld_b_ext_end": [float(weld_b_ext_end[0]), float(weld_b_ext_end[1]), float(weld_b_ext_end[2])],
            "weld_b_sample_far": [float(weld_b_sample_far[0]), float(weld_b_sample_far[1]), float(weld_b_sample_far[2])],
            "weld_a_sample_ext": [float(weld_a_sample_ext[0]), float(weld_a_sample_ext[1]), float(weld_a_sample_ext[2])],
            "weld_b_sample_ext": [float(weld_b_sample_ext[0]), float(weld_b_sample_ext[1]), float(weld_b_sample_ext[2])],
            "weld_a_away": [float((weld_a_at_hole + weld_a_dir)[0]), float((weld_a_at_hole + weld_a_dir)[1]), float((weld_a_at_hole + weld_a_dir)[2])],
            "weld_b_away": [float((weld_b_at_hole + weld_b_dir)[0]), float((weld_b_at_hole + weld_b_dir)[1]), float((weld_b_at_hole + weld_b_dir)[2])],
        })
        t_type_info["debug"]["found"] += 1
        
        if debug:
            print(f"  [SUCCESS] hole={hole_gid} weld_a={weld_a_id} weld_b={weld_b_id} weld_c={closest_weld_c}")
    
    if debug:
        print("\n========== T-type breakpoint detection debug ==========")
        for k, v in t_type_info["debug"].items():
            print(f"{k:30s}: {v}")
        print(f"{'total_found':30s}: {t_type_info['debug']['found']}")
        print("=======================================================\n")
    
    return t_type_info


# =========================
# Visualization: multiple modes support
# =========================
def visualize_geometry_graph(
    geometry_graph: Dict[str, Any],
    candidate_edge_ids: Set[str],
    *,
    viz_mode: str = "holes_weld_breakpoints",  # "geom_holes", "weld_breakpoints", "holes_weld_breakpoints", "holes_weld_all_breakpoints", "all", "adjacent_debug"
    node_size: float = 6.0,
    bp_size: float = 15.0,
    t_type_size: float = 12.0,
    lw_geom: float = 1.0,
    lw_weld: float = 2.0,
    lw_hole: float = 2.5,
    lw_adjacent: float = 1.5,
    show_adjacent_geoms: bool = False,
    t_type_breakpoints: Optional[List[Dict[str, Any]]] = None,
):
    """
    可视化模式:
      1. geom_holes: 所有几何边(绿色) + 过焊孔边(红色)
      2. weld_breakpoints: 焊缝(蓝色) + 断点(洋红色)
      3. holes_weld_breakpoints: 过焊孔边(红色) + 焊缝(蓝色) + 断点(洋红色)
      4. holes_weld_all_breakpoints: 过焊孔边(红色) + 焊缝(蓝色) + 普通断点(洋红色) + T型断点(青色星)
      5. all: 所有边(绿色) + 焊缝(蓝色) + 过焊孔边(红色) + 断点(洋红色) + T型断点(青色星)
      6. adjacent_debug: 焊缝(蓝色) + 过焊孔边(红色) + 所有几何边(橙色虚线) + 断点(洋红色)
    
    参数:
      show_adjacent_geoms: 是否显示所有几何边（橙色虚线，排除过焊孔边）
    """
    nodes = geometry_graph.get("nodes", {}) or {}
    geom_edges_raw = geometry_graph.get("geom_edges", {}) or {}
    weld_edges_raw = geometry_graph.get("weld_edges", {}) or {}
    
    geom_edges: Dict[str, Any] = {str(k): v for k, v in geom_edges_raw.items()}
    weld_edges: Dict[str, Any] = {str(k): v for k, v in weld_edges_raw.items()}

    # 根据模式设置显示选项
    show_geom = viz_mode in ["geom_holes", "all"]
    show_weld = viz_mode in ["weld_breakpoints", "holes_weld_breakpoints", "holes_weld_all_breakpoints", "all", "adjacent_debug"]
    show_holes = viz_mode in ["geom_holes", "holes_weld_breakpoints", "holes_weld_all_breakpoints", "all", "adjacent_debug"]
    show_breakpoints = viz_mode in ["weld_breakpoints", "holes_weld_breakpoints", "holes_weld_all_breakpoints", "all", "adjacent_debug"]
    show_t_type = viz_mode in ["holes_weld_all_breakpoints", "all"]
    
    # adjacent_debug 模式或显式开启时显示临边
    show_adjacent = show_adjacent_geoms or (viz_mode == "adjacent_debug")

    # 设置标题
    title_parts = []
    if show_geom:
        title_parts.append("GREEN=geom_edges")
    if show_weld:
        title_parts.append("BLUE=welds")
    if show_holes:
        title_parts.append("RED=through-holes")
    if show_adjacent:
        title_parts.append("ORANGE=all_geom_edges(excl_holes)")
    if show_breakpoints:
        title_parts.append("MAGENTA=breakpoints")
    
    title = " | ".join(title_parts) if title_parts else "Visualization"

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Mode: {viz_mode}\n{title}")

    all_x, all_y, all_z = [], [], []

    # 1. 绘制所有几何边 (绿色，细线)
    if show_geom:
        geom_count = 0
        for _, g in geom_edges.items():
            et = g.get("type", "unknown")
            s = g.get("start"); e = g.get("end")
            if not s or not e:
                continue

            if et == "bspline" and g.get("samples"):
                pts = g["samples"]
            elif et == "arc":
                c = g.get("center")
                if c:
                    pts = _arc_polyline(c, s, e, angle=g.get("angle", None), n=64)
                else:
                    pts = [s, e]
            else:
                pts = [s, e]

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            
            if geom_count == 0:
                ax.plot(xs, ys, zs, color="g", linewidth=float(lw_geom), alpha=0.5, label="Geom edges")
            else:
                ax.plot(xs, ys, zs, color="g", linewidth=float(lw_geom), alpha=0.5)
            
            all_x += xs; all_y += ys; all_z += zs
            geom_count += 1
        
        if geom_count > 0:
            print(f"[viz] geom_edges: {geom_count} (green)")

    # 2. 绘制焊缝 (蓝色，中等粗细)
    if show_weld:
        weld_count = 0
        for _, w in weld_edges.items():
            et = w.get("type", "unknown")
            s = w.get("start"); e = w.get("end")
            if not s or not e:
                continue

            if et == "bspline" and w.get("samples"):
                pts = w["samples"]
            elif et == "arc":
                c = w.get("center")
                if c:
                    pts = _arc_polyline(c, s, e, angle=w.get("angle", None), n=64)
                else:
                    pts = [s, e]
            else:
                pts = [s, e]

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            
            if weld_count == 0:
                ax.plot(xs, ys, zs, color="b", linewidth=float(lw_weld), alpha=0.7, label="Welds")
            else:
                ax.plot(xs, ys, zs, color="b", linewidth=float(lw_weld), alpha=0.7)
            
            all_x += xs; all_y += ys; all_z += zs
            weld_count += 1
        
        if weld_count > 0:
            print(f"[viz] weld_edges: {weld_count} (blue)")

    # 3. 绘制过焊孔边 (红色，粗线)
    if show_holes:
        hole_count = 0
        arc_count = 0
        bspline_count = 0
        chamfer_count = 0
        
        for gid in candidate_edge_ids:
            g = geom_edges.get(str(gid))
            if not g:
                continue

            et = g.get("type", "unknown")
            s = g.get("start"); e = g.get("end")
            if not s or not e:
                continue

            if et == "bspline" and g.get("samples"):
                pts = g["samples"]
                bspline_count += 1
            elif et == "arc":
                c = g.get("center")
                if c:
                    pts = _arc_polyline(c, s, e, angle=g.get("angle", None), n=64)
                else:
                    pts = [s, e]
                arc_count += 1
            else:
                # line or other types (chamfer holes)
                pts = [s, e]
                chamfer_count += 1

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            
            if hole_count == 0:
                ax.plot(xs, ys, zs, color="r", linewidth=float(lw_hole), alpha=0.9, label="Through-holes")
            else:
                ax.plot(xs, ys, zs, color="r", linewidth=float(lw_hole), alpha=0.9)
            
            all_x += xs; all_y += ys; all_z += zs
            hole_count += 1
        
        if hole_count > 0:
            print(f"[viz] through-hole edges: {hole_count} (red) - arc:{arc_count}, bspline:{bspline_count}, chamfer/line:{chamfer_count}")

    # 3.5. 绘制临边 (橙色虚线) - 显示所有 geom_edges（排除过焊孔边）
    if show_adjacent:
        # 直接使用所有 geom_edges，而不是只从节点的 adjacent_geom_edges 收集
        all_geom_gids = set(geom_edges.keys())
        
        # 排除已经显示为过焊孔的边
        adjacent_gids = all_geom_gids - candidate_edge_ids
        
        adj_count = 0
        for gid in adjacent_gids:
            g = geom_edges.get(str(gid))
            if not g:
                continue

            et = g.get("type", "unknown")
            s = g.get("start"); e = g.get("end")
            if not s or not e:
                continue

            if et == "bspline" and g.get("samples"):
                pts = g["samples"]
            elif et == "arc":
                c = g.get("center")
                if c:
                    pts = _arc_polyline(c, s, e, angle=g.get("angle", None), n=64)
                else:
                    pts = [s, e]
            else:
                pts = [s, e]

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            
            if adj_count == 0:
                ax.plot(xs, ys, zs, color="orange", linewidth=float(lw_adjacent), 
                       linestyle="--", alpha=0.6, label="All geom_edges")
            else:
                ax.plot(xs, ys, zs, color="orange", linewidth=float(lw_adjacent), 
                       linestyle="--", alpha=0.6)
            
            all_x += xs; all_y += ys; all_z += zs
            adj_count += 1
        
        if adj_count > 0:
            print(f"[viz] all_geom_edges (excluding holes): {adj_count} (orange dashed)")
            print(f"[viz] total_geom_edges_in_json: {len(geom_edges)}")
            print(f"[viz] through_hole_edges: {len(candidate_edge_ids)}")

    # 4. 绘制断点 (洋红色三角形)
    if show_breakpoints:
        bp_xs, bp_ys, bp_zs = [], [], []
        for nid, ninfo in nodes.items():
            if ninfo.get("process", {}).get("is_breakpoint", False):
                p = ninfo.get("point")
                if p:
                    bp_xs.append(p[0]); bp_ys.append(p[1]); bp_zs.append(p[2])
        
        if bp_xs:
            ax.scatter(bp_xs, bp_ys, bp_zs, s=float(bp_size), c="magenta", marker="^", 
                      label="Breakpoints", alpha=0.9, edgecolors='darkred', linewidths=1.5)
            all_x += bp_xs; all_y += bp_ys; all_z += bp_zs
            print(f"[viz] breakpoints: {len(bp_xs)} (magenta)")

    # 5. 绘制T型断点及调试信息
    if show_t_type and t_type_breakpoints:
        t_xs, t_ys, t_zs = [], [], []
        
        for idx, t_info in enumerate(t_type_breakpoints):
            pt = t_info.get("intersection_point")
            if not pt:
                continue
            t_xs.append(pt[0]); t_ys.append(pt[1]); t_zs.append(pt[2])
            all_x.append(pt[0]); all_y.append(pt[1]); all_z.append(pt[2])
            
            # 高亮 weld_a (黄色粗线)
            weld_a_id = t_info.get("weld_a_id")
            weld_b_id = t_info.get("weld_b_id")
            
            for wid, color, label_prefix in [
                (weld_a_id, "yellow", "weld_A"),
                (weld_b_id, "lime",   "weld_B"),
            ]:
                if not wid:
                    continue
                w = weld_edges.get(str(wid))
                if not w:
                    continue
                ws = w.get("start"); we = w.get("end")
                if not ws or not we:
                    continue
                et = w.get("type", "unknown")
                if et == "bspline" and w.get("samples"):
                    wpts = w["samples"]
                elif et == "arc":
                    wc = w.get("center")
                    wpts = _arc_polyline(wc, ws, we, angle=w.get("angle"), n=64) if wc else [ws, we]
                else:
                    wpts = [ws, we]
                wx = [p[0] for p in wpts]
                wy = [p[1] for p in wpts]
                wz = [p[2] for p in wpts]
                lbl = f"{label_prefix}[{idx}]" if idx == 0 else None
                ax.plot(wx, wy, wz, color=color, linewidth=4.0, alpha=1.0,
                        label=lbl, zorder=5)
                all_x += wx; all_y += wy; all_z += wz
            
            # 绘制延长线 weld_a (黄色虚线)
            a_at = t_info.get("weld_a_at_hole")
            a_ext = t_info.get("weld_a_ext_end")
            a_samp = t_info.get("weld_a_sample_ext", t_info.get("weld_a_sample_far"))
            b_at = t_info.get("weld_b_at_hole")
            b_ext = t_info.get("weld_b_ext_end")
            b_samp = t_info.get("weld_b_sample_ext", t_info.get("weld_b_sample_far"))

            
            if a_at and a_ext:
                ax.plot([a_at[0], a_ext[0]], [a_at[1], a_ext[1]], [a_at[2], a_ext[2]],
                        color="yellow", linewidth=1.5, linestyle="--", alpha=0.8,
                        label="ext_A[0]" if idx == 0 else None)
                all_x += [a_at[0], a_ext[0]]; all_y += [a_at[1], a_ext[1]]; all_z += [a_at[2], a_ext[2]]
            
            # 绘制切线参考点：孔边端点(大圆) + 采样远端点(方块) + 连线(黄色实线)
            if a_at and a_samp:
                ax.scatter([a_at[0]], [a_at[1]], [a_at[2]], s=60, c="yellow", marker="o",
                           alpha=1.0, zorder=7, edgecolors='black', linewidths=1)
                ax.scatter([a_samp[0]], [a_samp[1]], [a_samp[2]], s=60, c="yellow", marker="s",
                           alpha=1.0, zorder=7, edgecolors='black', linewidths=1,
                           label="samp_A[0]" if idx == 0 else None)
                ax.plot([a_at[0], a_samp[0]], [a_at[1], a_samp[1]], [a_at[2], a_samp[2]],
                        color="yellow", linewidth=2.0, linestyle="-", alpha=1.0)
            
            # 绘制延长线 weld_b (绿色虚线)
            if b_at and b_ext:
                ax.plot([b_at[0], b_ext[0]], [b_at[1], b_ext[1]], [b_at[2], b_ext[2]],
                        color="lime", linewidth=1.5, linestyle="--", alpha=0.8,
                        label="ext_B[0]" if idx == 0 else None)
                all_x += [b_at[0], b_ext[0]]; all_y += [b_at[1], b_ext[1]]; all_z += [b_at[2], b_ext[2]]
            
            # 绘制切线参考点：孔边端点(大圆) + 采样远端点(方块) + 连线(绿色实线)
            if b_at and b_samp:
                ax.scatter([b_at[0]], [b_at[1]], [b_at[2]], s=60, c="lime", marker="o",
                           alpha=1.0, zorder=7, edgecolors='black', linewidths=1)
                ax.scatter([b_samp[0]], [b_samp[1]], [b_samp[2]], s=60, c="lime", marker="s",
                           alpha=1.0, zorder=7, edgecolors='black', linewidths=1,
                           label="samp_B[0]" if idx == 0 else None)
                ax.plot([b_at[0], b_samp[0]], [b_at[1], b_samp[1]], [b_at[2], b_samp[2]],
                        color="lime", linewidth=2.0, linestyle="-", alpha=1.0)
            
            # 过焊孔边 a_at -> b_at（白色点线）
            if a_at and b_at:
                ax.plot([a_at[0], b_at[0]], [a_at[1], b_at[1]], [a_at[2], b_at[2]],
                        color="white", linewidth=1.5, linestyle=":", alpha=0.7,
                        label="hole_edge[0]" if idx == 0 else None)
            
            # 从孔边两端点到交点的连线（验证交点确实在延长线上）
            if a_at and pt:
                ax.plot([a_at[0], pt[0]], [a_at[1], pt[1]], [a_at[2], pt[2]],
                        color="yellow", linewidth=1.0, linestyle=":", alpha=0.5)
            if b_at and pt:
                ax.plot([b_at[0], pt[0]], [b_at[1], pt[1]], [b_at[2], pt[2]],
                        color="lime", linewidth=1.0, linestyle=":", alpha=0.5)
        
        if t_xs:
            ax.scatter(t_xs, t_ys, t_zs, s=float(t_type_size) * 3, c="cyan", marker="*",
                      label="T-type intersection", alpha=0.95, edgecolors='darkblue', linewidths=1.5,
                      zorder=6)
            print(f"[viz] T-type breakpoints: {len(t_xs)} (cyan star, weld_A=yellow, weld_B=lime, ext=dashed)")

    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right', fontsize=10)

    # 自动调整坐标轴
    if all_x:
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        zmin, zmax = min(all_z), max(all_z)
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
        if max_range <= 0:
            max_range = 1.0
        xm = 0.5 * (xmin + xmax)
        ym = 0.5 * (ymin + ymax)
        zm = 0.5 * (zmin + zmax)
        ax.set_xlim(xm - 0.5 * max_range, xm + 0.5 * max_range)
        ax.set_ylim(ym - 0.5 * max_range, ym + 0.5 * max_range)
        ax.set_zlim(zm - 0.5 * max_range, zm + 0.5 * max_range)

    plt.tight_layout()
    plt.show()


# =========================
# I/O
# =========================
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


# =========================
# CLI
# =========================
def main():
    stp_head = "D018-F205B"
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry_graph", default=f"model/sub_assembly/{stp_head}/{stp_head}_geometry_graph.json", help="input geometry_graph.json")
    ap.add_argument("--out_geometry_graph", default=f"model/sub_assembly/{stp_head}/{stp_head}_geometry_graph_with_breakpoints.json", help="output geometry_graph with breakpoints")
    ap.add_argument("--process_graph", default=f"model/sub_assembly/{stp_head}/{stp_head}_process_graph.json", help="output process_graph.json")

    ap.add_argument("--require_arc_center", action="store_true")
    ap.add_argument("--debug", default=True, action="store_true")

    # bspline filters (strict defaults from accurate script)
    ap.add_argument("--bspline_rms_tol", type=float, default=0.25)
    ap.add_argument("--bspline_min_angle_deg", type=float, default=8.0)
    ap.add_argument("--bspline_max_radius", type=float, default=500.0)
    ap.add_argument("--bspline_min_sagitta_abs", type=float, default=0.30)
    ap.add_argument("--bspline_min_sagitta_ratio", type=float, default=0.02)

    # chamfer-type through-hole detection
    ap.add_argument("--detect_chamfer_holes", default=True, action="store_true", help="检测倒角型过焊孔（焊缝与临边形成钝角）")
    ap.add_argument("--chamfer_min_angle_deg", type=float, default=92.0, help="倒角型过焊孔最小夹角（度）")
    ap.add_argument("--chamfer_max_angle_deg", type=float, default=150.0, help="倒角型过焊孔最大夹角（度），排除接近平行的普通边")
    ap.add_argument("--chamfer_max_length", type=float, default=50.0, help="倒角型过焊孔最大长度")

    # breakpoint insertion
    ap.add_argument("--insert_breakpoints", default=True, action="store_true", help="insert breakpoints for through-hole nodes")
    ap.add_argument("--max_weld_length", type=float, default=float('100'), help="只在长度小于此值的焊缝上插入断点（默认无限制）")

    # T-type breakpoint detection
    ap.add_argument("--detect_t_type", default=True, action="store_true", help="检测T型断点")
    ap.add_argument("--t_type_min_weld_length", type=float, default=5.0, help="T型断点焊缝最小长度(mm)")
    ap.add_argument("--t_type_max_weld_length", type=float, default=5000.0, help="T型断点焊缝最大长度(mm)")
    ap.add_argument("--t_type_extension_ratio", type=float, default=2.0, help="T型断点延长比例")
    ap.add_argument("--t_type_max_distance_to_weld", type=float, default=1000.0, help="T型断点到焊缝最大距离")

    # visualization
    ap.add_argument("--no_visualize", action="store_true", help="不显示可视化")
    ap.add_argument("--viz_mode", type=str, default="holes_weld_all_breakpoints", 
                    choices=["geom_holes", "weld_breakpoints", "holes_weld_breakpoints", "holes_weld_all_breakpoints", "all", "adjacent_debug"],
                    help="可视化模式: geom_holes=几何边+过焊孔, weld_breakpoints=焊缝+断点, holes_weld_breakpoints=过焊孔+焊缝+断点, holes_weld_all_breakpoints=过焊孔+焊缝+断点+T型断点, all=全部显示, adjacent_debug=焊缝+过焊孔+所有几何边+断点")
    ap.add_argument("--show_adjacent_geoms", action="store_true", help="显示所有几何边（橙色虚线，排除过焊孔边）")
    ap.add_argument("--bp_size", type=float, default=8.0, help="断点标记尺寸")
    ap.add_argument("--t_type_size", type=float, default=12.0, help="T型断点标记尺寸")
    ap.add_argument("--lw_geom", type=float, default=1.0, help="几何边线宽")
    ap.add_argument("--lw_weld", type=float, default=2.0, help="焊缝线宽")
    ap.add_argument("--lw_hole", type=float, default=2.5, help="过焊孔边线宽")
    ap.add_argument("--lw_adjacent", type=float, default=1.5, help="临边线宽")

    args = ap.parse_args()

    gg = load_json(args.geometry_graph)

    # Step 1: Detect through-hole edges (accurate algorithm)
    pg, cand_ids = detect_through_hole_edges_from_adjacent(
        gg,
        bspline_rms_tol=args.bspline_rms_tol,
        require_arc_center=args.require_arc_center,
        bspline_min_angle_deg=args.bspline_min_angle_deg,
        bspline_max_radius=args.bspline_max_radius,
        bspline_min_sagitta_abs=args.bspline_min_sagitta_abs,
        bspline_min_sagitta_ratio=args.bspline_min_sagitta_ratio,
        detect_chamfer_holes=args.detect_chamfer_holes,
        chamfer_min_angle_deg=args.chamfer_min_angle_deg,
        chamfer_max_angle_deg=args.chamfer_max_angle_deg,
        chamfer_max_length=args.chamfer_max_length,
        debug=args.debug,
    )

    # Collect nodes connected to through-hole edges
    connected_nodes = []
    for nid, ndata in pg.get("nodes", {}).items():
        if ndata.get("process", {}).get("through_hole_edge_ids"):
            connected_nodes.append(str(nid))

    print(f"[result] detected {len(cand_ids)} through-hole candidate edges")
    print(f"[result] {len(connected_nodes)} nodes connected to through-hole edges")

    # Step 2: Insert breakpoints if requested
    if args.insert_breakpoints:
        print(f"\n[breakpoint] inserting breakpoints for {len(connected_nodes)} nodes...")
        if args.max_weld_length < float('inf'):
            print(f"[breakpoint] length threshold: only insert on welds with length < {args.max_weld_length}")
        inserted = 0
        for nid in connected_nodes:
            node_proc = (pg.get("nodes", {}).get(str(nid), {}) or {}).get("process", {})
            hole_edge_ids = node_proc.get("through_hole_edge_ids") or []
            info = None
            for hole_edge_id in hole_edge_ids:
                info = insert_breakpoint_for_node(
                    gg,
                    nid,
                    hole_edge_id=str(hole_edge_id),
                    require_hole_edge_match=True,
                    max_weld_length=args.max_weld_length,
                    verbose=True,
                )
                if info:
                    break
            if info:
                inserted += 1
        print(f"[breakpoint] inserted {inserted} breakpoints")

        # Save updated geometry_graph with breakpoints
        save_json(args.out_geometry_graph, gg)
        print(f"[save] geometry_graph with breakpoints -> {args.out_geometry_graph}")

    # Step 3: Detect T-type breakpoints if requested
    t_type_info = None
    if args.detect_t_type:
        print(f"\n[T-type] detecting T-type breakpoints...")
        t_type_info = detect_t_type_breakpoints(
            gg,
            cand_ids,
            t_type_min_weld_length=args.t_type_min_weld_length,
            t_type_max_weld_length=args.t_type_max_weld_length,
            t_type_extension_ratio=args.t_type_extension_ratio,
            t_type_max_distance_to_weld=args.t_type_max_distance_to_weld,
            debug=args.debug,
        )
        if t_type_info:
            n_found = t_type_info['debug']['found']
            print(f"[T-type] found {n_found} T-type breakpoints")
            # Write t_type_breakpoints into geometry_graph so downstream scripts can consume it
            gg["t_type_breakpoints"] = t_type_info["t_type_breakpoints"]
            save_json(args.out_geometry_graph, gg)
            print(f"[save] geometry_graph with t_type_breakpoints -> {args.out_geometry_graph}")

    # Save process_graph
    save_json(args.process_graph, pg)
    print(f"[save] process_graph -> {args.process_graph}")

    # Step 4: Visualize
    if not args.no_visualize:
        t_type_breakpoints = t_type_info["t_type_breakpoints"] if t_type_info else None
        visualize_geometry_graph(
            gg,
            cand_ids,
            viz_mode=args.viz_mode,
            show_adjacent_geoms=args.show_adjacent_geoms,
            bp_size=args.bp_size,
            t_type_size=args.t_type_size,
            lw_geom=args.lw_geom,
            lw_weld=args.lw_weld,
            lw_hole=args.lw_hole,
            lw_adjacent=args.lw_adjacent,
            t_type_breakpoints=t_type_breakpoints,
        )


if __name__ == "__main__":
    main()
