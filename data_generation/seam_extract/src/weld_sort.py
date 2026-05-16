# -*- coding: utf-8 -*-
import os
import json
import math
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np

from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.gp import gp_Pnt


Vec3 = Tuple[float, float, float]


# -----------------------
# basic vector utils
# -----------------------
def _unit(v: Vec3) -> Vec3:
    n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0]/n, v[1]/n, v[2]/n)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _midpoint(s: List[float], e: List[float]) -> List[float]:
    return [(s[0]+e[0])*0.5, (s[1]+e[1])*0.5, (s[2]+e[2])*0.5]


def _normalize_np(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def _proj_to_plane_vec(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - float(np.dot(v, n)) * n


# -----------------------
# points from weld edges
# -----------------------
def _collect_weld_midpoints(edges: Dict[str, Dict[str, Any]]) -> np.ndarray:
    pts = []
    for _, info in edges.items():
        s = info.get("start")
        e = info.get("end")
        if not s or not e:
            continue
        pts.append(_midpoint(s, e))
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(pts, dtype=float)


# -----------------------
# solid centroid (for normal sign)
# -----------------------
def _solid_centroid(solid) -> Optional[gp_Pnt]:
    try:
        props = GProp_GProps()
        brepgprop_VolumeProperties(solid, props)
        return props.CentreOfMass()
    except Exception:
        return None


def _orient_normal_by_solids(shape, n: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """
    让法向尽量“朝向 solid 主要分布的一侧”
    """
    centroids = []
    for solid in TopologyExplorer(shape).solids():
        c = _solid_centroid(solid)
        if c is not None:
            centroids.append(np.array([float(c.X()), float(c.Y()), float(c.Z())], float))
    if not centroids:
        return n

    eps = 1e-6
    pos = 0
    neg = 0
    for c in centroids:
        sd = float(np.dot((c - origin), n))
        if sd >= -eps:
            pos += 1
        else:
            neg += 1
    if neg > pos:
        return -n
    return n


# -----------------------
# RANSAC plane from points
# plane: n·x = d  (n unit)
# -----------------------
def _plane_from_3pts(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return None
    n = n / nn
    d = float(np.dot(n, p1))
    return n, d


def _plane_point_dist(n: np.ndarray, d: float, pts: np.ndarray) -> np.ndarray:
    # |n·x - d|
    return np.abs(pts @ n - d)


def _inlier_spread_on_plane(pts: np.ndarray, n: np.ndarray) -> float:
    """
    用 inlier 点投影到平面后的二维“扩展度”做 tie-break：
    spread 越大越像底板（焊缝覆盖范围更大）
    """
    if pts.shape[0] < 3:
        return 0.0
    # 构造平面内正交基 (u,v)
    tmp = np.array([1.0, 0.0, 0.0], float)
    if abs(float(np.dot(tmp, n))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], float)
    u = _normalize_np(_proj_to_plane_vec(tmp, n))
    v = _normalize_np(np.cross(n, u))
    # 2D coords
    xy = np.stack([pts @ u, pts @ v], axis=1)
    cov = np.cov(xy.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    # 用面积感知：sqrt(lambda1*lambda2)
    return float(math.sqrt(float(eigvals[-1] * eigvals[-2])))


def ransac_dominant_plane(
    pts: np.ndarray,
    dist_tol: float = 1.0,
    n_iter: int = 3000,
    seed: int = 7
) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray, Dict[str, Any]]:
    """
    返回：best_n, best_d, inlier_mask, debug
    """
    debug: Dict[str, Any] = {
        "mode": "ransac_weld_points",
        "dist_tol": dist_tol,
        "n_iter": n_iter,
        "points": int(pts.shape[0]),
    }

    if pts.shape[0] < 3:
        debug["status"] = "too_few_points"
        return None, None, np.zeros((pts.shape[0],), dtype=bool), debug

    rng = np.random.default_rng(seed)
    best_inliers = -1
    best_spread = -1.0
    best_n = None
    best_d = None
    best_mask = None

    idxs = np.arange(pts.shape[0])

    for _ in range(n_iter):
        i1, i2, i3 = rng.choice(idxs, size=3, replace=False)
        res = _plane_from_3pts(pts[i1], pts[i2], pts[i3])
        if res is None:
            continue
        n, d = res
        dist = _plane_point_dist(n, d, pts)
        mask = dist <= dist_tol
        cnt = int(np.sum(mask))
        if cnt < 3:
            continue

        # tie-break：inlier 多优先；再比 spread
        if cnt > best_inliers:
            best_inliers = cnt
            best_n = n
            best_d = d
            best_mask = mask
            best_spread = _inlier_spread_on_plane(pts[mask], n)
        elif cnt == best_inliers:
            spread = _inlier_spread_on_plane(pts[mask], n)
            if spread > best_spread:
                best_inliers = cnt
                best_n = n
                best_d = d
                best_mask = mask
                best_spread = spread

    if best_n is None or best_mask is None:
        debug["status"] = "failed"
        return None, None, np.zeros((pts.shape[0],), dtype=bool), debug

    debug.update({
        "status": "ok",
        "best_inliers": int(best_inliers),
        "inlier_ratio": float(best_inliers / max(1, pts.shape[0])),
        "best_normal": [float(best_n[0]), float(best_n[1]), float(best_n[2])],
        "best_d": float(best_d),
        "best_spread": float(best_spread),
    })
    return best_n, best_d, best_mask, debug


# -----------------------
# build base_frame from dominant plane
# -----------------------
def infer_base_frame_from_welds_ransac(
    shape,
    edges: Dict[str, Dict[str, Any]],
    dist_tol: float = 1.0,
    n_iter: int = 3000
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pts = _collect_weld_midpoints(edges)

    n, d, mask, dbg = ransac_dominant_plane(pts, dist_tol=dist_tol, n_iter=n_iter)
    if n is None:
        # fallback: global Z
        z = np.array([0.0, 0.0, 1.0], float)
        origin = np.array([0.0, 0.0, 0.0], float)
        x = np.array([1.0, 0.0, 0.0], float)
        y = np.array([0.0, 1.0, 0.0], float)
        frame = {
            "origin": origin.tolist(),
            "x_axis": x.tolist(),
            "y_axis": y.tolist(),
            "z_axis": z.tolist(),
        }
        dbg["fallback"] = "use_global_Z"
        return frame, dbg

    inlier_pts = pts[mask]
    # origin：inlier 质心投影到平面
    c = np.mean(inlier_pts, axis=0)
    # 投影：c_proj = c - (n·c - d) n
    c_proj = c - (float(np.dot(n, c)) - float(d)) * n

    # normal 符号用 solid 分布决定（可选）
    n = _orient_normal_by_solids(shape, n, c_proj)

    z = _normalize_np(n)

    # x_axis：inlier 在平面内 PCA 主方向
    V = inlier_pts - c_proj
    Vp = np.stack([_proj_to_plane_vec(v, z) for v in V], axis=0)
    # 去掉太小的
    Vp = Vp[np.linalg.norm(Vp, axis=1) > 1e-6]
    if Vp.shape[0] >= 3:
        C = np.cov(Vp.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        x = eigvecs[:, int(np.argmax(eigvals))]
        x = _normalize_np(_proj_to_plane_vec(x, z))
    else:
        tmp = np.array([1.0, 0.0, 0.0], float)
        if abs(float(np.dot(tmp, z))) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], float)
        x = _normalize_np(_proj_to_plane_vec(tmp, z))

    y = _normalize_np(np.cross(z, x))
    x = _normalize_np(np.cross(y, z))

    frame = {
        "origin": c_proj.tolist(),
        "x_axis": x.tolist(),
        "y_axis": y.tolist(),
        "z_axis": z.tolist(),
    }
    dbg["base_frame"] = frame
    return frame, dbg


# -----------------------
# weld orientation class
# -----------------------
def weld_orientation_class(
    start: List[float],
    end: List[float],
    up_axis: Vec3,
    vertical_deg: float = 20.0,
    horizontal_deg: float = 70.0
) -> str:
    d = _sub((end[0], end[1], end[2]), (start[0], start[1], start[2]))
    du = _unit(d)
    uu = _unit(up_axis)

    c = abs(_dot(du, uu))
    c = max(0.0, min(1.0, c))
    angle = math.degrees(math.acos(c))

    if angle <= vertical_deg:
        return "vertical"
    if angle >= horizontal_deg:
        return "horizontal"
    return "other"


# -----------------------
# main: sort and export
# -----------------------
def sort_welds_and_export(
    shape,
    input_edges_json: str,
    output_order_json: str,
    up_axis: Union[Vec3, str] = "auto",
    vertical_deg: float = 20.0,
    horizontal_deg: float = 70.0,
    base_plane_weld_tol: float = 1.0,
    ransac_iter: int = 3000,
    priority_mode: str = "vertical_first",   # ✅ 新增：vertical_first / horizontal_first
    in_class_mode: str = "length_desc"       # ✅ 可选：length_desc / length_asc / id_asc
) -> None:
    if not os.path.exists(input_edges_json):
        raise FileNotFoundError(f"input json not found: {input_edges_json}")

    with open(input_edges_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges: Dict[str, Dict[str, Any]] = data.get("contact_edges", {})
    if not edges:
        raise ValueError("input json has no contact_edges")

    base_frame = None
    base_debug = None

    if isinstance(up_axis, str) and up_axis.lower() == "auto":
        base_frame, base_debug = infer_base_frame_from_welds_ransac(
            shape=shape,
            edges=edges,
            dist_tol=base_plane_weld_tol,
            n_iter=ransac_iter
        )
        up_axis_vec = tuple(base_frame["z_axis"])  # type: ignore
        axis_mode = "auto_ransac_weld_plane"
    else:
        up_axis_vec = up_axis  # type: ignore
        axis_mode = "manual"

    # ✅ 优先级可配置
    if priority_mode == "horizontal_first":
        pri_map = {"horizontal": 0, "vertical": 1, "other": 2}
        priority_list = ["horizontal", "vertical", "other"]
    else:
        pri_map = {"vertical": 0, "horizontal": 1, "other": 2}
        priority_list = ["vertical", "horizontal", "other"]

    # ✅ 同类内部排序规则可配置
    def in_class_key(length: float, eid_int: int):
        if in_class_mode == "length_asc":
            return (length, eid_int)
        if in_class_mode == "id_asc":
            return (eid_int,)
        # default: length_desc
        return (-length, eid_int)

    items = []
    for eid, info in edges.items():
        s = info.get("start")
        e = info.get("end")
        if not s or not e:
            continue

        ori = weld_orientation_class(
            s, e,
            up_axis=up_axis_vec,
            vertical_deg=vertical_deg,
            horizontal_deg=horizontal_deg
        )

        length = float(info.get("length", 0.0))
        pri = pri_map.get(ori, 2)

        eid_int = int(eid)
        items.append((pri, *in_class_key(length, eid_int), eid_int, str(eid), ori, length))

    items.sort()

    weld_order = []
    for idx, item in enumerate(items, start=1):
        eid = item[-3]
        ori = item[-2]
        length = item[-1]
        weld_order.append({
            "order": idx,
            "edge_id": eid,
            "orientation": ori,
            "length": length
        })

    out = {
        "input": os.path.basename(input_edges_json),
        "up_axis": {
            "mode": axis_mode,
            "vector": [float(up_axis_vec[0]), float(up_axis_vec[1]), float(up_axis_vec[2])]
        },
        "rules": {
            "priority_mode": priority_mode,
            "priority": priority_list,
            "in_class_mode": in_class_mode,
            "vertical_deg": vertical_deg,
            "horizontal_deg": horizontal_deg,
            "base_plane_weld_tol": base_plane_weld_tol,
            "ransac_iter": ransac_iter
        },
        "base_frame": base_frame,
        "base_plane_debug": base_debug,
        "weld_order": weld_order
    }

    os.makedirs(os.path.dirname(output_order_json), exist_ok=True)
    with open(output_order_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    print(f"[weld_sort] priority_mode={priority_mode}, in_class_mode={in_class_mode}, export -> {output_order_json} (count={len(weld_order)})")
