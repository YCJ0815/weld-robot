# -*- coding: utf-8 -*-
"""
s4_final_json_transform.py

读取 final_welds_with_junctions.json，将 up_axis 旋转对齐到世界 Z 轴 [0, 0, 1]，
并将 JSON 中所有三维坐标/向量同步施加该旋转变换，输出新的 JSON 文件。

变换规则：
  - 构造旋转矩阵 R，使得 R @ up_axis = [0, 0, 1]
  - 对以下字段施加旋转（点坐标）：
      contact_edges[*].start, end, samples[*]
      points[*].xyz
  - 对以下字段施加旋转（方向向量，不平移）：
      contact_edges[*].tangent, preferred_normal
  - up_axis 变换后写为 [0.0, 0.0, 1.0]
  - solids / weld_seams / global_safety 无坐标，原样保留

用法：
    python s4_final_json_transform.py input.json output.json
    python s4_final_json_transform.py input.json          # 自动输出为 input_z_aligned.json
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ============================================================
# IO helpers
# ============================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# 旋转矩阵构造
# ============================================================

def rotation_matrix_from_vec_to_z(up_axis: List[float]) -> np.ndarray:
    """
    构造旋转矩阵 R（3x3），使得：
        R @ up_axis_unit = [0, 0, 1]

    采用 Rodrigues 旋转公式：
        旋转轴 k = up_axis x z_hat（归一化）
        旋转角 theta = acos(up_axis · z_hat)

    特殊情况：
        - up_axis 已经是 [0,0,1]：返回单位矩阵
        - up_axis 是 [0,0,-1]：绕 X 轴旋转 180°
    """
    u = np.array(up_axis, dtype=float)
    norm = np.linalg.norm(u)
    if norm < 1e-12:
        raise ValueError(f"up_axis is a zero vector: {up_axis}")
    u = u / norm

    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(u, z), -1.0, 1.0))

    # 已对齐
    if abs(dot - 1.0) < 1e-10:
        return np.eye(3)

    # 反向：绕 X 轴旋转 180°
    if abs(dot + 1.0) < 1e-10:
        return np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])

    # 通用 Rodrigues
    k = np.cross(u, z)
    k = k / np.linalg.norm(k)          # 旋转轴（单位向量）
    theta = float(np.arccos(dot))       # 旋转角

    K = np.array([
        [ 0.0,  -k[2],  k[1]],
        [ k[2],  0.0,  -k[0]],
        [-k[1],  k[0],  0.0 ],
    ])  # k 的反对称矩阵

    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R


# ============================================================
# 变换辅助函数
# ============================================================

def xf_point(R: np.ndarray, pt: List[float]) -> List[float]:
    """旋转一个三维点（坐标）"""
    p = R @ np.array(pt, dtype=float)
    return [float(p[0]), float(p[1]), float(p[2])]


def xf_vec(R: np.ndarray, v: List[float]) -> List[float]:
    """旋转一个三维方向向量（不平移）"""
    return xf_point(R, v)


def xf_samples(R: np.ndarray, samples: List[List[float]]) -> List[List[float]]:
    """旋转采样点列表"""
    return [xf_point(R, s) for s in samples]


# ============================================================
# 主变换逻辑
# ============================================================

def transform_final_json(data: Dict[str, Any], R: np.ndarray) -> Dict[str, Any]:
    """
    对 final_welds_with_junctions JSON 的所有坐标/向量字段施加旋转 R。
    原始数据不被修改，返回新的 dict。
    """
    import copy
    out = copy.deepcopy(data)

    # --- contact_edges ---
    for eid, edge in out.get("contact_edges", {}).items():
        if isinstance(edge.get("start"), list):
            edge["start"] = xf_point(R, edge["start"])
        if isinstance(edge.get("end"), list):
            edge["end"] = xf_point(R, edge["end"])
        if isinstance(edge.get("tangent"), list):
            edge["tangent"] = xf_vec(R, edge["tangent"])
        if isinstance(edge.get("preferred_normal"), list):
            edge["preferred_normal"] = xf_vec(R, edge["preferred_normal"])
        if isinstance(edge.get("samples"), list):
            edge["samples"] = xf_samples(R, edge["samples"])

    # --- points ---
    for pid, pdata in out.get("points", {}).items():
        if isinstance(pdata, dict) and isinstance(pdata.get("xyz"), list):
            pdata["xyz"] = xf_point(R, pdata["xyz"])

    # --- up_axis -> [0, 0, 1] ---
    out["up_axis"] = [0.0, 0.0, 1.0]

    return out


# ============================================================
# 可视化（参照 s3_visualize_final_json.py）
# ============================================================

def _midpoint_polyline(samples: List[List[float]]) -> List[float]:
    """按弧长取采样点列表的中点，用于放置标签。"""
    if not samples or len(samples) < 2:
        return samples[0] if samples else [0.0, 0.0, 0.0]
    pts = np.array(samples, dtype=float)
    seg = pts[1:] - pts[:-1]
    d = np.linalg.norm(seg, axis=1)
    total = float(np.sum(d))
    if total <= 1e-12:
        m = pts[len(pts) // 2]
        return [float(m[0]), float(m[1]), float(m[2])]
    half = 0.5 * total
    acc = 0.0
    for i, di in enumerate(d):
        if acc + float(di) >= half:
            t = (half - acc) / max(float(di), 1e-12)
            p = pts[i] + t * (pts[i + 1] - pts[i])
            return [float(p[0]), float(p[1]), float(p[2])]
        acc += float(di)
    last = pts[-1]
    return [float(last[0]), float(last[1]), float(last[2])]


def _as_xyz(v: Any) -> Optional[List[float]]:
    if isinstance(v, list) and len(v) == 3:
        try:
            return [float(v[0]), float(v[1]), float(v[2])]
        except Exception:
            return None
    return None


def visualize_transformed(
    final_obj: Dict[str, Any],
    *,
    title: str = "Final Welds (Z-aligned)",
    show_ids: bool = False,
    show_L: bool = True,
    up_axis_length: float = 100.0,
) -> None:
    """
    可视化变换后的焊缝 JSON，与 s3_visualize_final_json.visualize_final_welds 逻辑一致。
    变换后 up_axis = [0,0,1]，箭头应垂直朝上。
    """
    contact_edges: Dict[str, Any] = final_obj.get("contact_edges", {}) or {}
    points_table: Dict[str, Any] = final_obj.get("points", {}) or {}

    # 收集端点 / 断点坐标
    endpoint_xyz: List[List[float]] = []
    breakpoint_xyz: List[List[float]] = []

    if isinstance(points_table, dict) and len(points_table) > 0:
        for _, p in points_table.items():
            if not isinstance(p, dict):
                continue
            xyz = _as_xyz(p.get("xyz"))
            if not xyz:
                continue
            if p.get("role") == "breakpoint":
                breakpoint_xyz.append(xyz)
            else:
                endpoint_xyz.append(xyz)
    else:
        for e in contact_edges.values():
            if not isinstance(e, dict):
                continue
            s = _as_xyz(e.get("start"))
            t = _as_xyz(e.get("end"))
            pts = e.get("points")
            if isinstance(pts, list) and len(pts) == 2 and s and t:
                r0 = pts[0].get("role") if isinstance(pts[0], dict) else "endpoint"
                r1 = pts[1].get("role") if isinstance(pts[1], dict) else "endpoint"
                (breakpoint_xyz if r0 == "breakpoint" else endpoint_xyz).append(s)
                (breakpoint_xyz if r1 == "breakpoint" else endpoint_xyz).append(t)
            else:
                if s:
                    endpoint_xyz.append(s)
                if t:
                    endpoint_xyz.append(t)

    # 去重
    def _dedup(pts_list: List[List[float]], nd: int = 8) -> List[List[float]]:
        seen: set = set()
        out = []
        for p in pts_list:
            k = (round(p[0], nd), round(p[1], nd), round(p[2], nd))
            if k not in seen:
                seen.add(k)
                out.append(p)
        return out

    endpoint_xyz = _dedup(endpoint_xyz)
    breakpoint_xyz = _dedup(breakpoint_xyz)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    keys = sorted(contact_edges.keys(), key=lambda x: (len(str(x)), str(x)))
    cmap = cm.get_cmap("tab20")
    all_xyz: List[List[float]] = []

    for i, eid in enumerate(keys):
        e = contact_edges.get(eid, {})
        if not isinstance(e, dict):
            continue
        samples = e.get("samples")
        if not isinstance(samples, list) or len(samples) < 2:
            s = _as_xyz(e.get("start"))
            t = _as_xyz(e.get("end"))
            if s and t:
                samples = [s, t]
            else:
                continue
        clean = [_as_xyz(p) for p in samples]
        clean = [p for p in clean if p is not None]
        if len(clean) < 2:
            continue

        xs = [p[0] for p in clean]
        ys = [p[1] for p in clean]
        zs = [p[2] for p in clean]
        color = cmap(i % 20)
        ax.plot(xs, ys, zs, linewidth=2.0, color=color)
        all_xyz.extend(clean)

        if show_L and e.get("corner_strategy") == "L_push":
            m = _midpoint_polyline(clean)
            ax.text(m[0], m[1], m[2], "L", fontsize=10)

        if show_ids:
            m = _midpoint_polyline(clean)
            ax.text(m[0], m[1], m[2], str(eid), fontsize=7)

    # 端点（圆点）
    if endpoint_xyz:
        ax.scatter([p[0] for p in endpoint_xyz],
                   [p[1] for p in endpoint_xyz],
                   [p[2] for p in endpoint_xyz],
                   marker="o", s=20, label="Endpoints")

    # 断点（三角）
    if breakpoint_xyz:
        ax.scatter([p[0] for p in breakpoint_xyz],
                   [p[1] for p in breakpoint_xyz],
                   [p[2] for p in breakpoint_xyz],
                   marker="^", s=32, color="orange", label="Breakpoints")

    # up_axis 箭头（变换后应垂直朝上）
    if all_xyz:
        up_axis_raw = final_obj.get("up_axis")
        if isinstance(up_axis_raw, list) and len(up_axis_raw) == 3:
            try:
                ua = np.array(up_axis_raw, dtype=float)
                ua_norm = float(np.linalg.norm(ua))
                if ua_norm > 1e-12:
                    ua = ua / ua_norm
                    pts_arr = np.array(all_xyz, dtype=float)
                    centroid = np.mean(pts_arr, axis=0)
                    ax.quiver(
                        centroid[0], centroid[1], centroid[2],
                        ua[0] * up_axis_length,
                        ua[1] * up_axis_length,
                        ua[2] * up_axis_length,
                        color="green", arrow_length_ratio=0.15, linewidth=3,
                        label=f"up_axis: [{ua[0]:.3f},{ua[1]:.3f},{ua[2]:.3f}]",
                    )
            except Exception:
                pass

        # 等比例坐标轴
        pts_arr = np.array(all_xyz, dtype=float)
        xmin, ymin, zmin = pts_arr.min(axis=0)
        xmax, ymax, zmax = pts_arr.max(axis=0)
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
        if max_range <= 1e-9:
            max_range = 1.0
        xm = (xmin + xmax) * 0.5
        ym = (ymin + ymax) * 0.5
        zm = (zmin + zmax) * 0.5
        ax.set_xlim(xm - 0.5 * max_range, xm + 0.5 * max_range)
        ax.set_ylim(ym - 0.5 * max_range, ym + 0.5 * max_range)
        ax.set_zlim(zm - 0.5 * max_range, zm + 0.5 * max_range)

    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 入口
# ============================================================

def process_file(input_path: str, output_path: Optional[str] = None) -> str:
    """
    读取 input_path，变换坐标系，保存到 output_path。
    返回实际输出路径。
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_z_aligned{ext}"

    print(f"[s4] Loading: {input_path}")
    data = load_json(input_path)

    up_axis = data.get("up_axis")
    if not isinstance(up_axis, list) or len(up_axis) != 3:
        raise ValueError(f"Invalid or missing 'up_axis' in {input_path}: {up_axis}")

    print(f"[s4] up_axis (original): {up_axis}")
    R = rotation_matrix_from_vec_to_z(up_axis)
    print(f"[s4] Rotation matrix R:\n{R}")

    # 验证：R @ up_axis 应该 ≈ [0, 0, 1]
    check = R @ np.array(up_axis, dtype=float)
    check = check / np.linalg.norm(check)
    print(f"[s4] Verification R @ up_axis = {check.tolist()} (should be [0,0,1])")

    print(f"[s4] Transforming all coordinates...")
    n_edges = len(data.get("contact_edges", {}))
    n_points = len(data.get("points", {}))
    print(f"[s4]   contact_edges: {n_edges}, points: {n_points}")

    out = transform_final_json(data, R)

    save_json(output_path, out)
    print(f"[s4] Saved -> {output_path}")
    return output_path


if __name__ == "__main__":
    stp_head = "D018-F205B"
    parser = argparse.ArgumentParser(
        description="Transform final_welds_with_junctions.json: rotate up_axis to world Z."
    )
    parser.add_argument("--input", default=f"model/sub_assembly/{stp_head}/{stp_head}_final_welds_with_junctions.json",
                        help="Path to input JSON file")
    parser.add_argument(
        "--output", nargs="?", default=f"model/sub_assembly/{stp_head}/{stp_head}_final_welds_with_junctions_transformed.json",
        help="Path to output JSON file (default: input_z_aligned.json)"
    )
    parser.add_argument(
        "--no_viz", action="store_true",
        help="Skip visualization after transform"
    )
    parser.add_argument("--show_ids", action="store_true", help="Show edge IDs on plot")
    parser.add_argument("--no_L", action="store_true", help="Disable L-push labels on plot")
    args = parser.parse_args()

    out_path = process_file(args.input, args.output)

    if not args.no_viz:
        print(f"[s4] Visualizing transformed result: {out_path}")
        transformed = load_json(out_path)
        visualize_transformed(
            transformed,
            title=f"Z-aligned: {os.path.basename(out_path)}",
            show_ids=args.show_ids,
            show_L=not args.no_L,
        )

