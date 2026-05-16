from __future__ import annotations

import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _midpoint_polyline(samples: List[List[float]]) -> List[float]:
    """Midpoint by arclength along samples (for placing 'L' label)."""
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


def visualize_final_welds(
    final_obj: Dict[str, Any],
    *,
    title: str = "Final Welds Visualization",
    show_ids: bool = False,
    show_L: bool = True,
    show_up_axis: bool = True,
    up_axis_length: float = 100.0,
) -> None:
    contact_edges: Dict[str, Any] = final_obj.get("contact_edges", {}) or {}
    points_table: Dict[str, Any] = final_obj.get("points", {}) or {}

    # --- collect endpoint/breakpoint coordinates ---
    endpoint_xyz: List[List[float]] = []
    breakpoint_xyz: List[List[float]] = []

    # Preferred: use points table if present
    if isinstance(points_table, dict) and len(points_table) > 0:
        for _, p in points_table.items():
            if not isinstance(p, dict):
                continue
            xyz = _as_xyz(p.get("xyz"))
            role = p.get("role")
            if not xyz:
                continue
            if role == "breakpoint":
                breakpoint_xyz.append(xyz)
            else:
                endpoint_xyz.append(xyz)
    else:
        # Fallback: infer from each edge's start/end + its "points" roles if present
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

    # Deduplicate points a bit (by rounding)
    def dedup(pts: List[List[float]], ndigits: int = 8) -> List[List[float]]:
        seen = set()
        out = []
        for p in pts:
            k = (round(p[0], ndigits), round(p[1], ndigits), round(p[2], ndigits))
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        return out

    endpoint_xyz = dedup(endpoint_xyz)
    breakpoint_xyz = dedup(breakpoint_xyz)

    # --- plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    import matplotlib.cm as cm

    keys = sorted(list(contact_edges.keys()), key=lambda x: (len(str(x)), str(x)))
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

        # sanitize samples
        clean = []
        for p in samples:
            xyz = _as_xyz(p)
            if xyz:
                clean.append(xyz)
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

    # endpoints: circles
    if endpoint_xyz:
        ex = [p[0] for p in endpoint_xyz]
        ey = [p[1] for p in endpoint_xyz]
        ez = [p[2] for p in endpoint_xyz]
        ax.scatter(ex, ey, ez, marker="o", s=20)

    # breakpoints: triangles
    if breakpoint_xyz:
        bx = [p[0] for p in breakpoint_xyz]
        by = [p[1] for p in breakpoint_xyz]
        bz = [p[2] for p in breakpoint_xyz]
        ax.scatter(bx, by, bz, marker="^", s=32)

    # visualize up_axis from final_obj
    if show_up_axis and all_xyz:
        up_axis_raw = final_obj.get("up_axis")
        if isinstance(up_axis_raw, list) and len(up_axis_raw) == 3:
            try:
                ua = np.array([float(up_axis_raw[0]), float(up_axis_raw[1]), float(up_axis_raw[2])], dtype=float)
                ua_norm = float(np.linalg.norm(ua))
                if ua_norm > 1e-12:
                    ua = ua / ua_norm
                    pts_arr = np.array(all_xyz, dtype=float)
                    centroid = np.mean(pts_arr, axis=0)
                    ax.quiver(
                        centroid[0], centroid[1], centroid[2],
                        ua[0] * up_axis_length, ua[1] * up_axis_length, ua[2] * up_axis_length,
                        color='green', arrow_length_ratio=0.15, linewidth=3,
                        label=f'up_axis: [{ua[0]:.3f}, {ua[1]:.3f}, {ua[2]:.3f}]'
                    )
                    ax.legend()
            except Exception:
                pass

    # auto-fit axis
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

    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--final_json",
        default="model/sub_assembly/D018-F205B/D018-F205B_final_welds_with_junctions.json",
        help="Path to *_final_welds_with_junctions.json or *_final_welds.json",
    )
    ap.add_argument("--title", default=None, help="Plot title")
    ap.add_argument("--show_ids", default=True,action="store_true", help="Show edge ids near each segment")
    ap.add_argument("--no_L", default=False, action="store_true", help="Disable 'L' label rendering")
    args = ap.parse_args()

    import os
    path = args.final_json
    # Fallback: if _with_junctions not found, try _final_welds
    if not os.path.exists(path):
        fallback = path.replace("_final_welds_with_junctions.json", "_final_welds.json")
        if os.path.exists(fallback):
            print(f"Warning: {path} not found, falling back to {fallback}")
            path = fallback
        else:
            print(f"Error: {path} not found.")
            return

    obj = load_json(path)
    title = args.title if args.title else path
    visualize_final_welds(obj, title=title, show_ids=args.show_ids, show_L=(not args.no_L))


if __name__ == "__main__":
    main()
