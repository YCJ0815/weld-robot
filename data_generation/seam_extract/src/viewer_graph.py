# visualize_geometry_graph.py
# ------------------------------------------------------------
# Visualize geometry_graph.json:
#   - nodes (weld endpoints)
#   - weld_edges (from contact_edges)
#   - geom_edges (topology edges in solids, indexed for adjacency)
# Support focusing on a node and its adjacent edges for debugging.
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import math
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# basic helpers
# ---------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else 0.0


def _unit(v: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    n = _safe_norm(v, eps)
    if n <= eps:
        return None
    return v / n


def _arc_polyline(
    center: List[float],
    start: List[float],
    end: List[float],
    *,
    angle: Optional[float] = None,
    n: int = 64,
) -> List[List[float]]:
    """
    Generate polyline points for an arc in 3D using center/start/end.
    If 'angle' is provided, it is treated as swept angle magnitude (>= endpoint angle).
    """
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


def _set_equal_axes(ax, xs, ys, zs):
    if not xs:
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
    if max_range <= 0:
        max_range = 1.0
    xm = 0.5 * (xmin + xmax)
    ym = 0.5 * (ymin + ymax)
    zm = 0.5 * (zmin + zmax)
    ax.set_xlim(xm - 0.5 * max_range, xm + 0.5 * max_range)
    ax.set_ylim(ym - 0.5 * max_range, ym + 0.5 * max_range)
    ax.set_zlim(zm - 0.5 * max_range, zm + 0.5 * max_range)


# ---------------------------
# stats
# ---------------------------
def print_graph_stats(gg: Dict[str, Any]) -> None:
    nodes = gg.get("nodes", {}) or {}
    weld_edges = gg.get("weld_edges", {}) or {}
    geom_edges = gg.get("geom_edges", {}) or {}

    print("========== geometry_graph stats ==========")
    print(f"nodes      : {len(nodes)}")
    print(f"weld_edges : {len(weld_edges)}")
    print(f"geom_edges : {len(geom_edges)}")

    # geom edge type counts + length range
    type_cnt = {}
    lens = []
    for _, g in geom_edges.items():
        t = g.get("type", "unknown")
        type_cnt[t] = type_cnt.get(t, 0) + 1
        try:
            lens.append(float(g.get("length", 0.0)))
        except Exception:
            pass
    print("geom_edges type counts:", type_cnt)
    if lens:
        print(f"geom_edges length range: min={min(lens):.4f}, max={max(lens):.4f}")

    # node degrees
    degs = []
    for _, n in nodes.items():
        adj = n.get("adjacent_geom_edges", []) or []
        degs.append(len(adj))
    if degs:
        degs_sorted = sorted(degs)
        print(f"node adjacent_geom_edges degree: min={degs_sorted[0]}, med={degs_sorted[len(degs_sorted)//2]}, max={degs_sorted[-1]}")
    print("==========================================")


# ---------------------------
# visualization
# ---------------------------
def visualize_geometry_graph(
    gg: Dict[str, Any],
    *,
    show_nodes: bool = True,
    show_weld_edges: bool = True,
    show_geom_edges: bool = True,
    max_weld_edges: int = 2000,
    max_geom_edges: int = 3000,
    focus_node: Optional[str] = None,
    only_adjacent: bool = False,
    min_len: float = 0.0,
    types: Optional[List[str]] = None,
):
    nodes = gg.get("nodes", {}) or {}
    weld_edges = gg.get("weld_edges", {}) or {}
    geom_edges = gg.get("geom_edges", {}) or {}

    # determine which geom edges to draw
    geom_ids_to_draw: List[str]
    weld_ids_to_draw: List[str]

    if focus_node:
        if focus_node not in nodes:
            print(f"[warn] focus_node={focus_node} not in nodes. Available example: {next(iter(nodes.keys()), None)}")
            geom_ids_to_draw = []
            weld_ids_to_draw = []
        else:
            ninfo = nodes[focus_node]
            geom_ids_to_draw = list(ninfo.get("adjacent_geom_edges", []) or [])
            weld_ids_to_draw = list(ninfo.get("incident_weld_edges", []) or [])
            if not only_adjacent:
                # if not only_adjacent, we still draw all weld edges by default, but user can limit via max_weld_edges
                pass
    else:
        geom_ids_to_draw = list(geom_edges.keys())
        weld_ids_to_draw = list(weld_edges.keys())

    # apply filters
    if types:
        types_set = set(types)
    else:
        types_set = None

    def edge_passes(info: Dict[str, Any]) -> bool:
        try:
            L = float(info.get("length", 0.0))
        except Exception:
            L = 0.0
        if L < min_len:
            return False
        if types_set is not None:
            if info.get("type") not in types_set:
                return False
        return True

    # build figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    title = "geometry_graph"
    if focus_node:
        title += f" | focus_node={focus_node}"
    ax.set_title(title)

    all_x, all_y, all_z = [], [], []

    # nodes
    if show_nodes:
        xs, ys, zs = [], [], []
        if focus_node and only_adjacent:
            # show only focus node for clean debug
            p = nodes[focus_node].get("point")
            if p:
                xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        else:
            for _, ninfo in nodes.items():
                p = ninfo.get("point")
                if not p:
                    continue
                xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        if xs:
            ax.scatter(xs, ys, zs, s=6.0)
            all_x += xs; all_y += ys; all_z += zs

    # weld edges (thicker)
    if show_weld_edges:
        cnt = 0
        for wid in weld_ids_to_draw:
            if cnt >= max_weld_edges:
                break
            w = weld_edges.get(str(wid)) or weld_edges.get(wid)
            if not w:
                continue
            if not edge_passes(w):
                continue
            s = w.get("start")
            e = w.get("end")
            if not s or not e:
                continue

            etype = w.get("type", "unknown")
            if etype == "bspline" and w.get("samples"):
                pts = w["samples"]
            elif etype == "arc" and w.get("center"):
                pts = _arc_polyline(w["center"], s, e, angle=w.get("angle", None), n=64)
            else:
                pts = [s, e]

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            ax.plot(xs, ys, zs, linewidth=2.0)
            all_x += xs; all_y += ys; all_z += zs
            cnt += 1

        if cnt > 0:
            print(f"[draw] weld_edges drawn: {cnt} (limit {max_weld_edges})")

    # geom edges (thinner)
    if show_geom_edges:
        cnt = 0
        # if focus_node + only_adjacent, draw only those adjacent (already set)
        # if focus_node but not only_adjacent: still draw only adjacent geom edges by default to avoid clutter
        if focus_node and not only_adjacent:
            geom_ids = geom_ids_to_draw
        else:
            geom_ids = geom_ids_to_draw

        for gid in geom_ids:
            if cnt >= max_geom_edges:
                break
            g = geom_edges.get(str(gid)) or geom_edges.get(gid)
            if not g:
                continue
            if not edge_passes(g):
                continue

            s = g.get("start")
            e = g.get("end")
            if not s or not e:
                continue

            etype = g.get("type", "unknown")
            if etype == "bspline" and g.get("samples"):
                pts = g["samples"]
            elif etype == "arc" and g.get("center"):
                pts = _arc_polyline(g["center"], s, e, angle=g.get("angle", None), n=64)
            else:
                pts = [s, e]

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            ax.plot(xs, ys, zs, linewidth=0.8)
            all_x += xs; all_y += ys; all_z += zs
            cnt += 1

        if cnt > 0:
            print(f"[draw] geom_edges drawn: {cnt} (limit {max_geom_edges})")

    _set_equal_axes(ax, all_x, all_y, all_z)
    plt.tight_layout()
    plt.show()


def main():
    stp_head = "U251-F75G"
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry_graph", default=f"model/sub_assembly/{stp_head}_geometry_graph.json", help="Path to geometry_graph.json")

    # show options
    ap.add_argument("--hide_nodes", default=True,action="store_true", help="Do not draw nodes")
    ap.add_argument("--hide_weld", action="store_true", help="Do not draw weld_edges")
    ap.add_argument("--hide_geom", action="store_true", help="Do not draw geom_edges")

    ap.add_argument("--max_weld_edges", type=int, default=2000, help="Limit weld edges drawn")
    ap.add_argument("--max_geom_edges", type=int, default=3000, help="Limit geom edges drawn")

    # debug focus
    ap.add_argument("--focus_node", default=None, help="Focus on a node id like N12")
    ap.add_argument("--only_adjacent", action="store_true", help="If focusing node: only show that node + its edges")

    # filters
    ap.add_argument("--min_len", type=float, default=0.0, help="Min edge length to draw")
    ap.add_argument("--types", default=None, help='Comma list of types to draw, e.g. "arc,bspline"')

    args = ap.parse_args()

    gg = load_json(args.geometry_graph)
    print_graph_stats(gg)

    types = None
    if args.types:
        types = [t.strip() for t in args.types.split(",") if t.strip()]

    visualize_geometry_graph(
        gg,
        show_nodes=(not args.hide_nodes),
        show_weld_edges=(not args.hide_weld),
        show_geom_edges=(not args.hide_geom),
        max_weld_edges=args.max_weld_edges,
        max_geom_edges=args.max_geom_edges,
        focus_node=args.focus_node,
        only_adjacent=args.only_adjacent,
        min_len=args.min_len,
        types=types,
    )


if __name__ == "__main__":
    main()
