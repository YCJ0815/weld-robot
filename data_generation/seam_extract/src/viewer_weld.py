# -*- coding: utf-8 -*-
import os
import json
import math
import random
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# 通用绘图工具
# -----------------------
def _set_axes_equal(ax, xs, ys, zs):
    if not xs:
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
    if max_range <= 1e-12:
        max_range = 1.0
    xm = 0.5 * (xmin + xmax)
    ym = 0.5 * (ymin + ymax)
    zm = 0.5 * (zmin + zmax)
    ax.set_xlim(xm - 0.5 * max_range, xm + 0.5 * max_range)
    ax.set_ylim(ym - 0.5 * max_range, ym + 0.5 * max_range)
    ax.set_zlim(zm - 0.5 * max_range, zm + 0.5 * max_range)


def _midpoint(s, e):
    return [(s[0] + e[0]) * 0.5, (s[1] + e[1]) * 0.5, (s[2] + e[2]) * 0.5]


# -----------------------
# 坐标系变换：world -> base_frame local
# -----------------------
def _apply_frame(pt: List[float], frame: Dict[str, Any]) -> List[float]:
    """
    world -> local
    local = [x_axis; y_axis; z_axis] · (pt - origin)
    """
    o = np.array(frame["origin"], float)
    x = np.array(frame["x_axis"], float)
    y = np.array(frame["y_axis"], float)
    z = np.array(frame["z_axis"], float)
    p = np.array(pt, float) - o
    return [float(np.dot(p, x)), float(np.dot(p, y)), float(np.dot(p, z))]


def _polyline_apply_frame(pts: List[List[float]], frame: Dict[str, Any]) -> List[List[float]]:
    return [_apply_frame(p, frame) for p in pts]


def _maybe_localize_pts(pts: List[List[float]], frame: Optional[Dict[str, Any]]) -> List[List[float]]:
    if frame is None:
        return pts
    return _polyline_apply_frame(pts, frame)


def _maybe_localize_pt(pt: List[float], frame: Optional[Dict[str, Any]]) -> List[float]:
    if frame is None:
        return pt
    return _apply_frame(pt, frame)


# -----------------------
# edge polyline
# -----------------------
def _edge_polyline_from_info(info: Dict[str, Any], fallback_segments: int = 24) -> Optional[List[List[float]]]:
    """
    把一条 edge_info 转成可画的 polyline 点列。
    支持 line/arc/bspline。
    - line: [start,end]
    - bspline: 使用 samples（若无则退化为 chord）
    - arc: 用 center/radius/angle/start/end 做插值（若缺字段则退化为 chord）
    """
    etype = info.get("type", "unknown")
    s = info.get("start")
    e = info.get("end")
    if not s or not e:
        return None

    if etype == "line":
        return [s, e]

    if etype == "bspline":
        samples = info.get("samples", None)
        if samples and len(samples) >= 2:
            return samples
        return [s, e]

    if etype == "arc":
        c = info.get("center")
        r = info.get("radius")
        ang = info.get("angle")
        if c is None or r is None:
            return [s, e]

        c = np.array(c, float)
        s_np = np.array(s, float)
        e_np = np.array(e, float)
        v1 = s_np - c
        v2 = e_np - c
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return [s, e]
        v1n = v1 / n1
        v2n = v2 / n2

        n = np.cross(v1n, v2n)
        if np.linalg.norm(n) < 1e-12:
            return [s, e]
        n = n / np.linalg.norm(n)

        b2 = np.cross(n, v1n)
        if np.linalg.norm(b2) < 1e-12:
            return [s, e]
        b2 = b2 / np.linalg.norm(b2)

        x2 = float(np.dot(v2n, v1n))
        y2 = float(np.dot(v2n, b2))
        theta2 = math.atan2(y2, x2)

        total_angle = float(ang) if ang is not None and abs(float(ang)) > 1e-6 else theta2
        seg = max(12, int(abs(total_angle) / (math.pi / 36)))
        ts = np.linspace(0.0, total_angle, seg)

        pts = []
        rr = float(r)
        for t in ts:
            v = math.cos(t) * v1n + math.sin(t) * b2
            p = (c + rr * v).tolist()
            pts.append([float(p[0]), float(p[1]), float(p[2])])
        return pts

    return [s, e]


def _load_edges(edges_json: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(edges_json):
        raise FileNotFoundError(edges_json)
    with open(edges_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    edges = data.get("contact_edges", {})
    if not edges:
        return {}
    return edges


# -----------------------
# 1) 可视化：焊缝排序（支持 base_frame local）
# -----------------------
def visualize_weld_order(
    order_json: str,
    edges_json: str,
    label_top_n: int = 60,
    use_local_frame: bool = True,
    show_legend: bool = True
):
    """
    读 weld_order.json + edges.json
    - 颜色区分 vertical/horizontal/other
    - 标注前 label_top_n 条的“真实顺序 order”
    - 若 use_local_frame=True 且 order_json 中有 base_frame：
        把所有点转换到 local 坐标后再画（底板=XY, local Z≈0）
    """
    if not os.path.exists(order_json):
        raise FileNotFoundError(order_json)

    edges = _load_edges(edges_json)
    with open(order_json, "r", encoding="utf-8") as f:
        od = json.load(f)

    order = od.get("weld_order", [])
    if not order:
        print("[vis_order] empty weld_order")
        return

    base_frame = od.get("base_frame", None)
    frame = base_frame if (use_local_frame and isinstance(base_frame, dict)) else None

    # 颜色策略（固定，便于记忆）
    color_map = {
        "vertical": (1.0, 0.2, 0.2),      # red
        "horizontal": (0.2, 0.2, 1.0),    # blue
        "other": (0.2, 0.8, 0.2),         # green
    }

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    title = "Weld Order (color = label)"
    if frame is not None:
        title += " [LOCAL: base_frame]"
    else:
        title += " [WORLD]"
    ax.set_title(title)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    xs_all, ys_all, zs_all = [], [], []
    counts = {"vertical": 0, "horizontal": 0, "other": 0}
    missing = 0

    for idx, item in enumerate(order, start=1):
        eid = str(item.get("edge_id"))
        ori = item.get("orientation", "other")
        info = edges.get(eid)
        if not info:
            missing += 1
            continue

        pts = _edge_polyline_from_info(info)
        if not pts or len(pts) < 2:
            continue

        # ✅ 转到 local frame（如果有）
        pts = _maybe_localize_pts(pts, frame)

        if ori not in counts:
            ori = "other"
        counts[ori] += 1

        c = color_map.get(ori, color_map["other"])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        ax.plot(xs, ys, zs, color=c, linewidth=1.8)

        xs_all.extend(xs); ys_all.extend(ys); zs_all.extend(zs)

        # ✅ 标注真实焊接顺序：item["order"]
        if idx <= label_top_n:
            s = info.get("start"); e = info.get("end")
            if s and e:
                m = _midpoint(s, e)
                m = _maybe_localize_pt(m, frame)
                label = item.get("order", idx)
                ax.text(m[0], m[1], m[2], str(label), fontsize=8)

    _set_axes_equal(ax, xs_all, ys_all, zs_all)

    if show_legend:
        import matplotlib.lines as mlines
        handles = [
            mlines.Line2D([], [], color=color_map["vertical"], label=f"vertical ({counts['vertical']})"),
            mlines.Line2D([], [], color=color_map["horizontal"], label=f"horizontal ({counts['horizontal']})"),
            mlines.Line2D([], [], color=color_map["other"], label=f"other ({counts['other']})"),
        ]
        ax.legend(handles=handles, loc="upper left")

    plt.tight_layout()
    plt.show()
    print(f"[vis_order] frame={'LOCAL' if frame is not None else 'WORLD'}, counts={counts}, missing_edges={missing}, labeled_top={label_top_n}")


# -----------------------
# 2) 可视化：板厚映射（支持 base_frame local）
# -----------------------
def visualize_weld_thickness(
    thickness_json: str,
    edges_json: str,
    order_json_for_frame: Optional[str] = None,
    use_local_frame: bool = True,
    show_hist: bool = True
):
    """
    读 weld_thickness.json + edges.json
    - edge_thickness 用 colormap 映射到焊缝颜色
    - 可选输出 thickness histogram
    - 若给了 order_json_for_frame 且 use_local_frame=True：
        用其中 base_frame 把点转到 local 再画
    """
    if not os.path.exists(thickness_json):
        raise FileNotFoundError(thickness_json)

    edges = _load_edges(edges_json)
    with open(thickness_json, "r", encoding="utf-8") as f:
        td = json.load(f)

    edge_thk = td.get("edge_thickness", {})
    if not edge_thk:
        print("[vis_thickness] empty edge_thickness")
        return

    frame = None
    if use_local_frame and order_json_for_frame and os.path.exists(order_json_for_frame):
        with open(order_json_for_frame, "r", encoding="utf-8") as f:
            od = json.load(f)
        bf = od.get("base_frame", None)
        if isinstance(bf, dict):
            frame = bf

    vals = [float(v) for v in edge_thk.values() if float(v) > 0.0]
    if not vals:
        print("[vis_thickness] all thickness are 0")
        return
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    cmap = plt.get_cmap("viridis")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    title = "Weld Thickness Map (color = thickness)"
    title += " [LOCAL]" if frame is not None else " [WORLD]"
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    xs_all, ys_all, zs_all = [], [], []
    plotted = 0
    for eid, info in edges.items():
        t = float(edge_thk.get(str(eid), 0.0))
        if t <= 0.0:
            continue
        pts = _edge_polyline_from_info(info)
        if not pts or len(pts) < 2:
            continue

        pts = _maybe_localize_pts(pts, frame)

        a = (t - vmin) / (vmax - vmin)
        c = cmap(a)

        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        ax.plot(xs, ys, zs, color=c, linewidth=1.6)
        xs_all.extend(xs); ys_all.extend(ys); zs_all.extend(zs)
        plotted += 1

    _set_axes_equal(ax, xs_all, ys_all, zs_all)
    plt.tight_layout()
    plt.show()

    print(f"[vis_thickness] plotted {plotted} welds, thickness in [{vmin:.6g}, {vmax:.6g}], frame={'LOCAL' if frame is not None else 'WORLD'}")

    if show_hist:
        plt.figure()
        plt.title("Thickness Histogram (non-zero)")
        plt.hist(vals, bins=30)
        plt.xlabel("Thickness")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


# -----------------------
# 3) 可视化：路径连接 + 相交打断（支持 base_frame local）
# -----------------------
def visualize_weld_paths(
    paths_json: str,
    order_json_for_frame: Optional[str] = None,
    use_local_frame: bool = True,
    show_virtual_edges: bool = True,
    show_intersection_points: bool = True,
    label_path_id: bool = True
):
    """
    读 weld_paths.json
    - 每条 path 用不同颜色
    - 若 show_virtual_edges=True：画 virtual_edges（相交打断后的段）
    - 若 show_intersection_points=True：把所有 virtual_edges 的端点画出来
    - 若给了 order_json_for_frame 且 use_local_frame=True：
        用其中 base_frame 把点转到 local 再画
    """
    if not os.path.exists(paths_json):
        raise FileNotFoundError(paths_json)

    frame = None
    if use_local_frame and order_json_for_frame and os.path.exists(order_json_for_frame):
        with open(order_json_for_frame, "r", encoding="utf-8") as f:
            od = json.load(f)
        bf = od.get("base_frame", None)
        if isinstance(bf, dict):
            frame = bf

    with open(paths_json, "r", encoding="utf-8") as f:
        pd = json.load(f)

    v_edges: Dict[str, Dict[str, Any]] = pd.get("virtual_edges", {})
    paths: List[Dict[str, Any]] = pd.get("paths", [])
    if not paths:
        print("[vis_paths] empty paths")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    title = "Weld Paths (connected) + Split Diagnostics"
    title += " [LOCAL]" if frame is not None else " [WORLD]"
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    xs_all, ys_all, zs_all = [], [], []

    # path 着色
    path_colors = {}
    for p in paths:
        pid = p.get("path_id", 0)
        path_colors[pid] = (random.random(), random.random(), random.random())

    # A) 画 path 的 polyline 点列
    for p in paths:
        pid = p.get("path_id", 0)
        pts = p.get("points", [])
        if not pts or len(pts) < 2:
            continue

        pts = _maybe_localize_pts(pts, frame)

        c = path_colors.get(pid, (0.6, 0.6, 0.6))
        xs = [q[0] for q in pts]; ys = [q[1] for q in pts]; zs = [q[2] for q in pts]
        ax.plot(xs, ys, zs, color=c, linewidth=2.6)
        xs_all.extend(xs); ys_all.extend(ys); zs_all.extend(zs)

        if label_path_id:
            mid = pts[len(pts) // 2]
            ax.text(mid[0], mid[1], mid[2], f"P{pid}", fontsize=9)

    # B) virtual_edges
    if show_virtual_edges and v_edges:
        for _, info in v_edges.items():
            s = info.get("start"); e = info.get("end")
            if not s or not e:
                continue
            seg = [s, e]
            seg = _maybe_localize_pts(seg, frame)
            xs = [seg[0][0], seg[1][0]]
            ys = [seg[0][1], seg[1][1]]
            zs = [seg[0][2], seg[1][2]]
            ax.plot(xs, ys, zs, linewidth=1.0)
            xs_all.extend(xs); ys_all.extend(ys); zs_all.extend(zs)

    # C) 端点点云
    if show_intersection_points and v_edges:
        pts = []
        for _, info in v_edges.items():
            if info.get("start") and info.get("end"):
                pts.append(info["start"])
                pts.append(info["end"])
        if pts:
            pts = _maybe_localize_pts(pts, frame)
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
            ax.scatter(xs, ys, zs, s=8)
            xs_all.extend(xs); ys_all.extend(ys); zs_all.extend(zs)

    _set_axes_equal(ax, xs_all, ys_all, zs_all)
    plt.tight_layout()
    plt.show()

    print(f"[vis_paths] paths={len(paths)}, virtual_edges={len(v_edges)}, frame={'LOCAL' if frame is not None else 'WORLD'}")


# -----------------------
# 4) 可视化：包角结构识别结果（支持 base_frame local）
# -----------------------
def visualize_wrap_corner_candidates(
    wrap_json: str,
    edges_json: str,
    order_json_for_frame: Optional[str] = None,
    use_local_frame: bool = True,
    show_only_first: int = 30
):
    """
    读 wrap_corner.json + edges.json
    - 把每个 candidate 的 front/back 两条边画出来
    - 标注 wrap_mode：continuous_wrap / push_corner
    - 若给了 order_json_for_frame 且 use_local_frame=True：
        用其中 base_frame 把点转到 local 再画
    """
    if not os.path.exists(wrap_json):
        raise FileNotFoundError(wrap_json)

    frame = None
    if use_local_frame and order_json_for_frame and os.path.exists(order_json_for_frame):
        with open(order_json_for_frame, "r", encoding="utf-8") as f:
            od = json.load(f)
        bf = od.get("base_frame", None)
        if isinstance(bf, dict):
            frame = bf

    edges = _load_edges(edges_json)
    with open(wrap_json, "r", encoding="utf-8") as f:
        wd = json.load(f)

    cands = wd.get("wrap_corner_candidates", [])
    if not cands:
        print("[vis_wrap] no wrap_corner_candidates")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    title = "Wrap Corner Candidates (front/back pairs)"
    title += " [LOCAL]" if frame is not None else " [WORLD]"
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    xs_all, ys_all, zs_all = [], [], []
    nshow = min(show_only_first, len(cands))

    for k in range(nshow):
        c = cands[k]
        e1 = str(c.get("front_edge_id"))
        e2 = str(c.get("back_edge_id"))
        mode = c.get("wrap_mode", "unknown")

        col = (0.2, 0.85, 0.2) if mode == "continuous_wrap" else (1.0, 0.55, 0.1)

        for eid in (e1, e2):
            info = edges.get(eid)
            if not info:
                continue
            pts = _edge_polyline_from_info(info)
            if not pts or len(pts) < 2:
                continue

            pts = _maybe_localize_pts(pts, frame)

            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
            ax.plot(xs, ys, zs, color=col, linewidth=3.0)
            xs_all.extend(xs); ys_all.extend(ys); zs_all.extend(zs)

            s = info.get("start"); e = info.get("end")
            if s and e:
                m = _midpoint(s, e)
                m = _maybe_localize_pt(m, frame)
                ax.text(m[0], m[1], m[2], f"{mode}", fontsize=8)

    _set_axes_equal(ax, xs_all, ys_all, zs_all)
    plt.tight_layout()
    plt.show()

    print(f"[vis_wrap] show {nshow}/{len(cands)} candidates, frame={'LOCAL' if frame is not None else 'WORLD'}")


# -----------------------
# 5) 可视化：焊缝打断（X交叉 / T结点）
# -----------------------
def visualize_weld_splits(
    split_json: str,
    original_edges_json: Optional[str] = None,
    order_json_for_frame: Optional[str] = None,
    use_local_frame: bool = True,
    show_original: bool = True,
    show_virtual: bool = True,
    show_junctions: bool = True,
    junction_size: int = 5,
    label_junctions: bool = False,
    show_only_first_virtual: int = 0
):
    """
    读 weld_split.json（由 split_welds_at_junctions 输出）
    可叠加显示：原始 contact_edges + 打断后的 virtual_edges + junction 点。

    参数：
      split_json: *_split_edges.json
      original_edges_json: 原始 contact_edges.json（可选，用来叠加“拆分前”）
      order_json_for_frame: weld_order.json（可选，取 base_frame 显示 local）
      use_local_frame: 是否将点转到 local 坐标
      show_original: 是否画原始焊缝（灰色）
      show_virtual: 是否画打断后的焊缝段（按 source_edge_id 着色）
      show_junctions: 是否画 junction 点（X=红，T=橙）
      show_only_first_virtual: 若 >0，只画前 N 段 virtual_edges（用于大模型调试）
    """
    if not os.path.exists(split_json):
        raise FileNotFoundError(split_json)

    # frame
    frame = None
    if use_local_frame and order_json_for_frame and os.path.exists(order_json_for_frame):
        with open(order_json_for_frame, "r", encoding="utf-8") as f:
            od = json.load(f)
        bf = od.get("base_frame", None)
        if isinstance(bf, dict):
            frame = bf

    with open(split_json, "r", encoding="utf-8") as f:
        sd = json.load(f)

    v_edges: Dict[str, Dict[str, Any]] = sd.get("virtual_edges", {})
    junctions: List[Dict[str, Any]] = sd.get("junctions", [])

    if not v_edges:
        print("[vis_split] virtual_edges empty")
        return

    # 可选叠加原始焊缝
    original_edges = {}
    if original_edges_json and os.path.exists(original_edges_json):
        original_edges = _load_edges(original_edges_json)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    title = "Weld Split Visualization (original vs virtual + junctions)"
    title += " [LOCAL]" if frame is not None else " [WORLD]"
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    xs_all, ys_all, zs_all = [], [], []

    # -----------------------------------
    # A) 原始焊缝（灰色细线）作为背景参考
    # -----------------------------------
    if show_original and original_edges:
        for eid, info in original_edges.items():
            pts = _edge_polyline_from_info(info)
            if not pts or len(pts) < 2:
                continue
            pts = _maybe_localize_pts(pts, frame)

            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
            ax.plot(xs, ys, zs, color=(0.6, 0.6, 0.6), linewidth=0.8, alpha=0.7)

            xs_all.extend(xs); ys_all.extend(ys); zs_all.extend(zs)

    # -----------------------------------
    # B) virtual_edges（打断后的段）
    # 颜色策略：同一个 source_edge_id 同色
    # -----------------------------------
    if show_virtual:
        # 为每个 source_edge_id 分配稳定颜色
        src_color: Dict[str, Tuple[float, float, float]] = {}

        def _color_for_src(src_id: str) -> Tuple[float, float, float]:
            if src_id in src_color:
                return src_color[src_id]
            # 随机但稳定：用 hash 映射到 [0,1)
            h = abs(hash(src_id)) % 10_000_000
            r = ((h * 37) % 1000) / 1000.0
            g = ((h * 57) % 1000) / 1000.0
            b = ((h * 77) % 1000) / 1000.0
            src_color[src_id] = (r, g, b)
            return src_color[src_id]

        keys = list(v_edges.keys())
        if show_only_first_virtual > 0:
            keys = keys[:show_only_first_virtual]

        for vid in keys:
            info = v_edges[vid]
            pts = info.get("samples", None)
            if not pts or len(pts) < 2:
                # fallback start-end
                s = info.get("start"); e = info.get("end")
                if not s or not e:
                    continue
                pts = [s, e]

            pts = _maybe_localize_pts(pts, frame)

            src = str(info.get("source_edge_id", "0"))
            c = _color_for_src(src)
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
            ax.plot(xs, ys, zs, color=c, linewidth=1.0, alpha=0.95)

            # 标注 virtual id（可选：如果你想看每段编号，可改成 ax.text）
            xs_all.extend(xs); ys_all.extend(ys); zs_all.extend(zs)

    # -----------------------------------
    # C) junction 点：每个断点不同颜色
    # -----------------------------------
    if show_junctions and junctions:
        # 为每个 junction 生成唯一颜色（使用 HSV 色彩空间）
        import colorsys
        
        junction_points = []
        junction_colors = []
        junction_markers = []
        
        # ✅ 计算场景尺度，用于智能偏移标签
        if xs_all and ys_all and zs_all:
            scene_range = max(max(xs_all) - min(xs_all), max(ys_all) - min(ys_all), max(zs_all) - min(zs_all))
        else:
            scene_range = 100.0
        label_offset = scene_range * 0.015  # 标签偏移为场景尺寸的 1.5%
        
        for idx, j in enumerate(junctions):
            p = j.get("point")
            if not p or len(p) != 3:
                continue
            p_loc = _maybe_localize_pt([float(p[0]), float(p[1]), float(p[2])], frame)
            
            # 为每个 junction 生成不同的颜色（HSV 色彩空间均匀分布）
            hue = (idx * 0.618033988749895) % 1.0  # 黄金角度，确保颜色分布均匀
            saturation = 0.8 + (idx % 3) * 0.1  # 0.8, 0.9, 1.0 循环
            value = 0.9 + (idx % 2) * 0.1  # 0.9, 1.0 循环
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            
            junction_points.append(p_loc)
            junction_colors.append(rgb)
            
            # 根据类型选择标记
            if j.get("type") == "X":
                junction_markers.append("o")
            else:
                junction_markers.append("^")
            
            # ✅ 智能偏移标签位置，避免重叠
            # 使用螺旋偏移模式：每个标签在不同方向
            angle = (idx * 137.5) % 360  # 黄金角度（度数）
            rad = math.radians(angle)
            offset_x = label_offset * math.cos(rad)
            offset_y = label_offset * math.sin(rad)
            offset_z = label_offset * (0.5 if idx % 2 == 0 else -0.3)  # Z 方向交替偏移
            
            label_x = p_loc[0] + offset_x
            label_y = p_loc[1] + offset_y
            label_z = p_loc[2] + offset_z
            
            # ✅ 显示断点编号（带引导线效果的标签）
            # ax.text(label_x, label_y, label_z, str(idx), 
            #        fontsize=5, fontweight='bold', 
            #        color=rgb,  # 使用与断点相同的颜色
            #        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, 
            #                 edgecolor=rgb, linewidth=1.5),
            #        ha='center', va='center')
            
            # ✅ 画一条细线连接标签和断点（引导线）
            # ax.plot([p_loc[0], label_x], [p_loc[1], label_y], [p_loc[2], label_z], 
            #        color=rgb, linewidth=0.8, alpha=0.6, linestyle='--')
            
            # 可选：显示详细标签（类型和模式）
            if label_junctions:
                mode = j.get("mode", "?")
                typ = j.get("type", "?")
                label_text = f"{typ}-{mode}" if mode != "?" else typ
                # 在编号下方显示详细信息
                ax.text(label_x, label_y, label_z - label_offset * 0.8, label_text, 
                       fontsize=7, color='darkblue', style='italic', alpha=0.8)
        
        # 绘制每个 junction（每个点单独绘制以使用不同颜色）
        for pt, color, marker in zip(junction_points, junction_colors, junction_markers):
            ax.scatter([pt[0]], [pt[1]], [pt[2]], s=junction_size, c=[color], marker=marker, 
                      edgecolors='black', linewidths=0.8, zorder=10)
            xs_all.append(pt[0]); ys_all.append(pt[1]); zs_all.append(pt[2])
        
        # 添加图例说明
        if junction_points:
            import matplotlib.lines as mlines
            legend_x = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', 
                                    markersize=6, markeredgewidth=0.8, markeredgecolor='black',
                                    label=f'X junction ({sum(1 for m in junction_markers if m == "o")})')
            legend_t = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', 
                                    markersize=6, markeredgewidth=0.8, markeredgecolor='black',
                                    label=f'T junction ({sum(1 for m in junction_markers if m == "^")})')
            ax.legend(handles=[legend_x, legend_t], loc="upper left")

    _set_axes_equal(ax, xs_all, ys_all, zs_all)
    plt.tight_layout()
    plt.show()

    print(f"[vis_split] virtual_edges={len(v_edges)}, junctions={len(junctions)}, "
          f"original={'yes' if original_edges else 'no'}, frame={'LOCAL' if frame is not None else 'WORLD'}")
