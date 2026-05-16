# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


# ============================================================
# basic helpers
# ============================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def _pt_key(p: List[float], ndigits: int = 8) -> Tuple[float, float, float]:
    return (round(float(p[0]), ndigits), round(float(p[1]), ndigits), round(float(p[2]), ndigits))

def _segment_length(a: List[float], b: List[float]) -> float:
    return float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))

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


# ============================================================
# polyline split utilities (world 3D)
# ============================================================

def polyline_cumlen(pl: np.ndarray) -> np.ndarray:
    if pl.shape[0] < 2:
        return np.zeros((pl.shape[0],), float)
    d = np.linalg.norm(pl[1:] - pl[:-1], axis=1)
    return np.concatenate(([0.0], np.cumsum(d)))

def point_at_s_world(pl: np.ndarray, cum: np.ndarray, s_global: float) -> np.ndarray:
    L = float(cum[-1]) if float(cum[-1]) > 1e-12 else 1.0
    a = float(_clamp(s_global, 0.0, 1.0)) * L
    i = int(np.searchsorted(cum, a, side="right") - 1)
    i = max(0, min(i, pl.shape[0] - 2))
    seg_len = float(cum[i + 1] - cum[i])
    if seg_len < 1e-12:
        return pl[i].copy()
    t = (a - float(cum[i])) / seg_len
    t = _clamp(float(t), 0.0, 1.0)
    return pl[i] + t * (pl[i + 1] - pl[i])

def insert_splits_on_polyline_world(
    pts: np.ndarray, split_s: List[float], merge_tol: float = 1e-6
) -> List[np.ndarray]:
    """
    Split 3D polyline by s_global list in [0,1].
    Return list of segments (each is (M,3) array).
    """
    if pts.shape[0] < 2:
        return []

    ss = sorted([float(_clamp(s, 0.0, 1.0)) for s in split_s])

    merged: List[float] = []
    for s in ss:
        if not merged or abs(s - merged[-1]) > merge_tol:
            merged.append(s)
        else:
            merged[-1] = 0.5 * (merged[-1] + s)

    merged = [0.0] + [s for s in merged if 0.0 < s < 1.0] + [1.0]

    cum = polyline_cumlen(pts)
    L = float(cum[-1]) if float(cum[-1]) > 1e-12 else 1.0

    out: List[np.ndarray] = []
    for k in range(len(merged) - 1):
        s0, s1 = merged[k], merged[k + 1]
        if s1 - s0 <= 1e-10:
            continue

        a0, a1 = s0 * L, s1 * L
        i0 = int(np.searchsorted(cum, a0, side="right") - 1)
        i1 = int(np.searchsorted(cum, a1, side="left"))
        i0 = max(0, min(i0, pts.shape[0] - 2))
        i1 = max(1, min(i1, pts.shape[0] - 1))

        seg_pts = [point_at_s_world(pts, cum, s0)]
        for ii in range(i0 + 1, i1):
            aa = float(cum[ii])
            if a0 + 1e-9 < aa < a1 - 1e-9:
                seg_pts.append(pts[ii])
        seg_pts.append(point_at_s_world(pts, cum, s1))

        seg = np.asarray(seg_pts, float)
        if seg.shape[0] >= 2 and _norm(seg[0] - seg[-1]) > 1e-9:
            out.append(seg)

    return out


# ============================================================
# Point registry that reuses existing ids (final_welds points),
# and optionally reuses geometry_graph node ids by coord.
# ============================================================

class PointRegistry:
    def __init__(self, existing_points: Dict[str, Any], coord2nid: Dict[Tuple[float,float,float], str]):
        self.points: Dict[str, Dict[str, Any]] = {}
        self.coord2pid: Dict[Tuple[float,float,float], str] = {}
        self.coord2nid = coord2nid

        # load existing
        max_pnum = 0
        for pid, p in (existing_points or {}).items():
            if not isinstance(p, dict):
                continue
            xyz = p.get("xyz")
            role = p.get("role", "endpoint")
            if isinstance(xyz, list) and len(xyz) == 3:
                k = _pt_key(xyz)
                self.points[str(pid)] = {"id": str(pid), "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])], "role": role}
                self.coord2pid[k] = str(pid)
                m = re.match(r"^P(\d+)$", str(pid))
                if m:
                    max_pnum = max(max_pnum, int(m.group(1)))
        self._next_p = max_pnum + 1

    def _new_pid(self) -> str:
        pid = f"P{self._next_p}"
        self._next_p += 1
        return pid

    def get_or_add(self, xyz: List[float], role: str, *, prefer_geom_node: bool = True) -> str:
        k = _pt_key(xyz)
        if k in self.coord2pid:
            pid = self.coord2pid[k]
            # do not downgrade existing role; but if existing is endpoint and we now want breakpoint, upgrade it
            if role == "breakpoint" and self.points.get(pid, {}).get("role") != "breakpoint":
                self.points[pid]["role"] = "breakpoint"
            return pid

        # try reuse geometry_graph node id if available and not conflicting
        if prefer_geom_node:
            nid = self.coord2nid.get(k)
            if nid and nid not in self.points:
                pid = str(nid)
                self.points[pid] = {"id": pid, "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])], "role": role}
                self.coord2pid[k] = pid
                return pid

        # else create new P#
        pid = self._new_pid()
        self.points[pid] = {"id": pid, "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])], "role": role}
        self.coord2pid[k] = pid
        return pid


# ============================================================
# Geometry graph coord->node_id (optional)
# ============================================================

def build_coord2nid_from_geometry_graph(path: Optional[str]) -> Dict[Tuple[float,float,float], str]:
    if not path or (not os.path.exists(path)):
        return {}
    gd = load_json(path)
    nodes = gd.get("nodes", {}) or {}
    out: Dict[Tuple[float,float,float], str] = {}
    for nid, n in nodes.items():
        if not isinstance(n, dict):
            continue
        p = n.get("point")
        if isinstance(p, list) and len(p) == 3:
            out[_pt_key(p)] = str(nid)
    return out


# ============================================================
# Update final_welds with junction breakpoints
# ============================================================

def update_final_welds_with_junctions(
    final_welds_json: str,
    split_result_json: str,
    out_json: str,
    geometry_graph_json: Optional[str] = None,
    *,
    only_modes: Optional[List[str]] = None,     # e.g. ["EXT_LINE", "BRANCH_EXT_TO_THROUGH"]
    add_corner_strategy: Optional[str] = None,
    skip_near_end_eps: float = 1e-5,
    interference_breakpoint: bool = True,     # ✅ NEW: 干涉断点模式
    keep_source_weld_id: bool = False,        # ✅ NEW: 是否输出 source_weld_id
    max_remove_length: Optional[float] = None,  # ✅ NEW: 删除断点间边的最大长度阈值（mm）
) -> None:
    fw = load_json(final_welds_json)
    sr = load_json(split_result_json)

    contact_edges: Dict[str, Any] = fw.get("contact_edges", {}) or {}
    weld_seams: Dict[str, Any] = fw.get("weld_seams", {}) or {}
    fw_points: Dict[str, Any] = fw.get("points", {}) or {}

    coord2nid = build_coord2nid_from_geometry_graph(geometry_graph_json)
    preg = PointRegistry(fw_points, coord2nid)

    # group split points by through_edge
    junctions = sr.get("junctions", []) or []
    splits_by_edge: Dict[str, List[Tuple[float, List[float]]]] = {}  # eid -> [(s_global, xyz), ...]

    for j in junctions:
        if not isinstance(j, dict):
            continue
        mode = j.get("mode")
        if only_modes and mode not in only_modes:
            continue
        eid = str(j.get("through_edge", ""))
        if not eid or eid not in contact_edges:
            continue
        s_global = None
        # split_result里没直接存 s_global；但它存了 world point（proj_w），我们用它去投影回 samples
        pw = j.get("point")
        if not (isinstance(pw, list) and len(pw) == 3):
            continue

        # compute s_global by projecting to edge polyline using samples
        e = contact_edges[eid]
        samples = e.get("samples")
        if not (isinstance(samples, list) and len(samples) >= 2):
            # fallback to start/end
            s = e.get("start")
            t = e.get("end")
            if not (isinstance(s, list) and isinstance(t, list)):
                continue
            samples = [s, t]

        pl = np.asarray(samples, float)
        if pl.shape[0] < 2:
            continue
        cum = polyline_cumlen(pl)
        L = float(cum[-1]) if float(cum[-1]) > 1e-12 else 1.0

        p = np.asarray(pw, float)
        # find closest point along polyline (3D) by scanning segments
        best_d = 1e18
        best_s_abs = None
        for i in range(pl.shape[0] - 1):
            a = pl[i]
            b = pl[i + 1]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-12:
                q = a
                u = 0.0
            else:
                u = float(np.dot(p - a, ab) / denom)
                u = _clamp(u, 0.0, 1.0)
                q = a + u * ab
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best_d = d
                s_abs = float(cum[i] + u * (cum[i + 1] - cum[i]))
                best_s_abs = s_abs

        if best_s_abs is None:
            continue
        s_global = float(best_s_abs / L)

        if not (skip_near_end_eps < s_global < 1.0 - skip_near_end_eps):
            continue

        splits_by_edge.setdefault(eid, []).append((s_global, [float(pw[0]), float(pw[1]), float(pw[2])]))

    if not splits_by_edge:
        # nothing to do; still write copy
        fw["points"] = preg.points
        save_json(out_json, fw)
        print(f"[update] no junction splits found; saved copy -> {out_json}")
        return

    # Helper: replace an edge_id in seams
    def replace_edge_in_seams(old_eid: str, new_eids: List[str]) -> None:
        for sid, seam in weld_seams.items():
            if not isinstance(seam, dict):
                continue
            ids = seam.get("edge_ids")
            if not isinstance(ids, list):
                continue
            if old_eid not in ids:
                continue
            new_list = []
            for x in ids:
                if x == old_eid:
                    new_list.extend(new_eids)
                else:
                    new_list.append(x)
            seam["edge_ids"] = new_list

    # Perform split on each affected edge
    for eid, lst in splits_by_edge.items():
        # sort & dedup s_global
        ss = sorted([float(s) for s, _pw in lst])
        merged = []
        for s in ss:
            if not merged or abs(s - merged[-1]) > 1e-6:
                merged.append(s)
            else:
                merged[-1] = 0.5 * (merged[-1] + s)

        e0 = contact_edges[eid]
        samples = e0.get("samples")
        if not (isinstance(samples, list) and len(samples) >= 2):
            samples = [e0.get("start"), e0.get("end")]
        plw = np.asarray(samples, float)

        segs = insert_splits_on_polyline_world(plw, merged, merge_tol=1e-6)
        if len(segs) <= 1:
            continue

        # build point ids: endpoints keep existing if present
        # try read existing points list:
        old_pts = e0.get("points", [])
        old_start_pid = None
        old_end_pid = None
        if isinstance(old_pts, list) and len(old_pts) >= 2:
            old_start_pid = old_pts[0].get("point_id") if isinstance(old_pts[0], dict) else None
            old_end_pid = old_pts[-1].get("point_id") if isinstance(old_pts[-1], dict) else None

        # if no pid, register from xyz as endpoint
        start_xyz = [float(segs[0][0][0]), float(segs[0][0][1]), float(segs[0][0][2])]
        end_xyz = [float(segs[-1][-1][0]), float(segs[-1][-1][1]), float(segs[-1][-1][2])]

        if old_start_pid and old_start_pid in preg.points:
            spid = old_start_pid
        else:
            spid = preg.get_or_add(start_xyz, "endpoint")

        if old_end_pid and old_end_pid in preg.points:
            epid = old_end_pid
        else:
            epid = preg.get_or_add(end_xyz, "endpoint")

        # create breakpoint point ids for internal segment joints
        joint_pids: List[str] = []
        for k in range(1, len(segs)):
            jxyz = [float(segs[k][0][0]), float(segs[k][0][1]), float(segs[k][0][2])]
            jpid = preg.get_or_add(jxyz, "breakpoint", prefer_geom_node=True)
            joint_pids.append(jpid)

        # build new edges ids and records
        new_edge_ids: List[str] = []
        base = eid
        # remove old edge
        del contact_edges[eid]

        # prepare metadata copied
        wtype = e0.get("type")
        preferred_normal = e0.get("preferred_normal", None)
        solid_ids = e0.get("solid_ids", None)
        source_weld_id = e0.get("source_weld_id", eid)  # keep trace

        # point chain: spid, joint_pids..., epid
        chain = [spid] + joint_pids + [epid]

                # point chain: spid, joint_pids..., epid
        chain = [spid] + joint_pids + [epid]

        def _role_of_pid(pid: str) -> str:
            pinfo = preg.points.get(pid, {})
            r = pinfo.get("role", "endpoint")
            return "breakpoint" if r == "breakpoint" else "endpoint"

        for i, seg in enumerate(segs):
            a = seg[0].tolist()
            b = seg[-1].tolist()

            pid0 = chain[i]
            pid1 = chain[i + 1]
            r0 = _role_of_pid(pid0)
            r1 = _role_of_pid(pid1)

            # ✅ NEW: 删除两端都是断点的焊缝段（带长度阈值）
            if r0 == "breakpoint" and r1 == "breakpoint":
                seg_length = float(np.sum(np.linalg.norm(seg[1:] - seg[:-1], axis=1)))
                if max_remove_length is None or seg_length <= max_remove_length:
                    continue

            neid = f"{base}_J{i+1}"
            new_edge_ids.append(neid)

            rec = {
                "type": wtype,
                "start": [float(a[0]), float(a[1]), float(a[2])],
                "end": [float(b[0]), float(b[1]), float(b[2])],
                "length": float(np.sum(np.linalg.norm(seg[1:] - seg[:-1], axis=1))),
                "solid_ids": solid_ids,
                "tangent": _tangent_from_points(
                    [float(a[0]), float(a[1]), float(a[2])],
                    [float(b[0]), float(b[1]), float(b[2])]
                ),
                "preferred_normal": preferred_normal,
                "points": [
                    {"point_id": pid0, "role": r0},
                    {"point_id": pid1, "role": r1},
                ],
                "samples": [[float(p[0]), float(p[1]), float(p[2])] for p in seg.tolist()],
            }

            # corner_strategy is deprecated; only preserve this optional hook for old callers.
            if not interference_breakpoint and add_corner_strategy is not None:
                rec["corner_strategy"] = add_corner_strategy

            # ✅ NEW: 默认不写 source_weld_id（除非你需要追溯）
            if keep_source_weld_id:
                rec["source_weld_id"] = source_weld_id

            contact_edges[neid] = rec


        # update seams edge_ids replacing old with new
        replace_edge_in_seams(eid, new_edge_ids)

    fw["points"] = preg.points
    fw["contact_edges"] = contact_edges
    fw["weld_seams"] = weld_seams

    save_json(out_json, fw)
    print(f"[update] updated final_welds with junction breakpoints -> {out_json}")


# ============================================================
# T-type breakpoint processing (from geometry_graph t_type_breakpoints)
# ============================================================

def update_final_welds_with_t_type_breakpoints(
    final_welds_json: str,
    geometry_graph_with_breakpoints_json: str,
    out_json: str,
    *,
    max_remove_length: Optional[float] = 30.0,
    keep_source_weld_id: bool = False,
) -> None:
    """
    从 geometry_graph_with_breakpoints.json 中读取 t_type_breakpoints，
    将每个 T型断点的 weld_c_id 对应焊缝在 intersection_point 处打断，
    然后删除两端都是断点且长度 <= max_remove_length 的焊缝段。

    Args:
        final_welds_json: s1 输出的 *_final_welds.json
        geometry_graph_with_breakpoints_json: s0 输出的含 t_type_breakpoints 字段的 json
        out_json: 输出文件路径
        max_remove_length: 删除两端都是断点的焊缝段的最大长度（mm），None 表示全删
        keep_source_weld_id: 是否在输出边中保留 source_weld_id 字段
    """
    fw = load_json(final_welds_json)
    gg = load_json(geometry_graph_with_breakpoints_json)

    contact_edges: Dict[str, Any] = fw.get("contact_edges", {}) or {}
    weld_seams: Dict[str, Any] = fw.get("weld_seams", {}) or {}
    fw_points: Dict[str, Any] = fw.get("points", {}) or {}

    coord2nid = build_coord2nid_from_geometry_graph(geometry_graph_with_breakpoints_json)
    preg = PointRegistry(fw_points, coord2nid)

    t_type_breakpoints: List[Dict[str, Any]] = gg.get("t_type_breakpoints", []) or []

    if not t_type_breakpoints:
        fw["points"] = preg.points
        save_json(out_json, fw)
        print(f"[t_type] no t_type_breakpoints found in geometry_graph; saved copy -> {out_json}")
        return

    print(f"[t_type] processing {len(t_type_breakpoints)} T-type breakpoints...")

    # Build splits_by_edge: contact_edge_id -> [(s_global, xyz), ...]
    # weld_c_id in t_type_breakpoints corresponds to weld_edges keys in geometry_graph.
    # In final_welds contact_edges, the edge may be stored as weld_c_id, weld_c_id_A, weld_c_id_B, etc.
    # We look for the contact_edge whose source_weld_id == weld_c_id and that contains the intersection_point.
    splits_by_edge: Dict[str, List[Tuple[float, List[float]]]] = {}

    def _find_contact_edge_for_weld(weld_c_id: str, pt: List[float]) -> Optional[str]:
        """Find the contact_edge id that corresponds to weld_c_id and is closest to pt."""
        best_eid = None
        best_dist = 1e18
        p = np.asarray(pt, float)
        for eid, e in contact_edges.items():
            # Match by source_weld_id or direct id
            src = e.get("source_weld_id", eid)
            # eid could be "wid", "wid_A", "wid_B" etc. — check prefix
            if src != weld_c_id and not eid.startswith(weld_c_id):
                continue
            # Compute distance from pt to this edge's polyline
            samples = e.get("samples")
            if not (isinstance(samples, list) and len(samples) >= 2):
                s = e.get("start")
                t = e.get("end")
                if not (isinstance(s, list) and isinstance(t, list)):
                    continue
                samples = [s, t]
            pl = np.asarray(samples, float)
            for i in range(pl.shape[0] - 1):
                ab = pl[i + 1] - pl[i]
                denom = float(np.dot(ab, ab))
                if denom < 1e-12:
                    q = pl[i]
                else:
                    u = float(_clamp(float(np.dot(p - pl[i], ab) / denom), 0.0, 1.0))
                    q = pl[i] + u * ab
                d = float(np.linalg.norm(p - q))
                if d < best_dist:
                    best_dist = d
                    best_eid = eid
        return best_eid if best_dist < 50.0 else None  # 50mm tolerance

    def _compute_s_global(eid: str, pt: List[float]) -> Optional[float]:
        """Project pt onto contact_edge eid polyline, return s_global in (0,1)."""
        e = contact_edges.get(eid)
        if not e:
            return None
        samples = e.get("samples")
        if not (isinstance(samples, list) and len(samples) >= 2):
            s = e.get("start")
            t = e.get("end")
            if not (isinstance(s, list) and isinstance(t, list)):
                return None
            samples = [s, t]
        pl = np.asarray(samples, float)
        cum = polyline_cumlen(pl)
        L = float(cum[-1]) if float(cum[-1]) > 1e-12 else 1.0
        p = np.asarray(pt, float)
        best_d = 1e18
        best_s_abs = None
        for i in range(pl.shape[0] - 1):
            ab = pl[i + 1] - pl[i]
            denom = float(np.dot(ab, ab))
            if denom < 1e-12:
                u = 0.0
                q = pl[i]
            else:
                u = float(_clamp(float(np.dot(p - pl[i], ab) / denom), 0.0, 1.0))
                q = pl[i] + u * ab
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best_d = d
                best_s_abs = float(cum[i] + u * (cum[i + 1] - cum[i]))
        if best_s_abs is None:
            return None
        s_global = float(best_s_abs / L)
        skip_eps = 1e-5
        if not (skip_eps < s_global < 1.0 - skip_eps):
            return None
        return s_global

    for t_info in t_type_breakpoints:
        weld_c_id = str(t_info.get("weld_c_id", ""))
        inter_pt = t_info.get("intersection_point")
        if not weld_c_id or not (isinstance(inter_pt, list) and len(inter_pt) == 3):
            continue

        eid = _find_contact_edge_for_weld(weld_c_id, inter_pt)
        if not eid:
            print(f"[t_type] WARNING: cannot find contact_edge for weld_c={weld_c_id}, skipping")
            continue

        s_global = _compute_s_global(eid, inter_pt)
        if s_global is None:
            print(f"[t_type] WARNING: intersection_point is too close to endpoint of {eid}, skipping")
            continue

        splits_by_edge.setdefault(eid, []).append((s_global, [float(inter_pt[0]), float(inter_pt[1]), float(inter_pt[2])]))
        print(f"[t_type] weld_c={weld_c_id} -> contact_edge={eid} s_global={s_global:.4f}")

    if not splits_by_edge:
        fw["points"] = preg.points
        save_json(out_json, fw)
        print(f"[t_type] no valid splits computed; saved copy -> {out_json}")
        return

    # Helper: replace edge_id in seams
    def replace_edge_in_seams(old_eid: str, new_eids: List[str]) -> None:
        for sid, seam in weld_seams.items():
            if not isinstance(seam, dict):
                continue
            ids = seam.get("edge_ids")
            if not isinstance(ids, list) or old_eid not in ids:
                continue
            new_list = []
            for x in ids:
                if x == old_eid:
                    new_list.extend(new_eids)
                else:
                    new_list.append(x)
            seam["edge_ids"] = new_list

    # Perform splits
    for eid, lst in splits_by_edge.items():
        ss = sorted([float(s) for s, _ in lst])
        merged: List[float] = []
        for s in ss:
            if not merged or abs(s - merged[-1]) > 1e-6:
                merged.append(s)
            else:
                merged[-1] = 0.5 * (merged[-1] + s)

        e0 = contact_edges.get(eid)
        if not e0:
            continue
        samples = e0.get("samples")
        if not (isinstance(samples, list) and len(samples) >= 2):
            samples = [e0.get("start"), e0.get("end")]
        plw = np.asarray(samples, float)

        segs = insert_splits_on_polyline_world(plw, merged, merge_tol=1e-6)
        if len(segs) <= 1:
            continue

        # Point chain
        old_pts = e0.get("points", [])
        old_start_pid = old_pts[0].get("point_id") if isinstance(old_pts, list) and len(old_pts) >= 2 and isinstance(old_pts[0], dict) else None
        old_end_pid = old_pts[-1].get("point_id") if isinstance(old_pts, list) and len(old_pts) >= 2 and isinstance(old_pts[-1], dict) else None

        start_xyz = segs[0][0].tolist()
        end_xyz = segs[-1][-1].tolist()
        spid = old_start_pid if (old_start_pid and old_start_pid in preg.points) else preg.get_or_add(start_xyz, "endpoint")
        epid = old_end_pid if (old_end_pid and old_end_pid in preg.points) else preg.get_or_add(end_xyz, "endpoint")

        joint_pids: List[str] = []
        for k in range(1, len(segs)):
            jxyz = segs[k][0].tolist()
            jpid = preg.get_or_add(jxyz, "breakpoint", prefer_geom_node=True)
            joint_pids.append(jpid)

        chain = [spid] + joint_pids + [epid]

        def _role_of_pid(pid: str) -> str:
            return "breakpoint" if preg.points.get(pid, {}).get("role") == "breakpoint" else "endpoint"

        wtype = e0.get("type")
        preferred_normal = e0.get("preferred_normal", None)
        solid_ids = e0.get("solid_ids", None)
        source_weld_id = e0.get("source_weld_id", eid)

        del contact_edges[eid]
        new_edge_ids: List[str] = []

        for i, seg in enumerate(segs):
            a = seg[0].tolist()
            b = seg[-1].tolist()
            pid0 = chain[i]
            pid1 = chain[i + 1]
            r0 = _role_of_pid(pid0)
            r1 = _role_of_pid(pid1)

            # Delete segment between two breakpoints if within length threshold
            if r0 == "breakpoint" and r1 == "breakpoint":
                seg_length = float(np.sum(np.linalg.norm(seg[1:] - seg[:-1], axis=1)))
                if max_remove_length is None or seg_length <= max_remove_length:
                    print(f"[t_type] removing segment {eid}_T{i+1} (len={seg_length:.2f}mm, between two breakpoints)")
                    continue

            neid = f"{eid}_T{i + 1}"
            new_edge_ids.append(neid)

            rec: Dict[str, Any] = {
                "type": wtype,
                "start": [float(a[0]), float(a[1]), float(a[2])],
                "end": [float(b[0]), float(b[1]), float(b[2])],
                "length": float(np.sum(np.linalg.norm(seg[1:] - seg[:-1], axis=1))),
                "solid_ids": solid_ids,
                "tangent": _tangent_from_points(
                    [float(a[0]), float(a[1]), float(a[2])],
                    [float(b[0]), float(b[1]), float(b[2])]
                ),
                "preferred_normal": preferred_normal,
                "points": [
                    {"point_id": pid0, "role": r0},
                    {"point_id": pid1, "role": r1},
                ],
                "samples": [[float(p[0]), float(p[1]), float(p[2])] for p in seg.tolist()],
            }
            if keep_source_weld_id:
                rec["source_weld_id"] = source_weld_id

            contact_edges[neid] = rec

        replace_edge_in_seams(eid, new_edge_ids)

    fw["points"] = preg.points
    fw["contact_edges"] = contact_edges
    fw["weld_seams"] = weld_seams

    save_json(out_json, fw)
    print(f"[t_type] updated final_welds with T-type breakpoints -> {out_json}")


# ============================================================
# CLI
# ============================================================

def main():
    step_head = "D018-F205B"
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_welds", default=f"model/sub_assembly/D018-F205B/{step_head}_final_welds.json", help="input *_final_welds.json")
    ap.add_argument("--split_result", default=f"model/sub_assembly/D018-F205B/{step_head}_split_edges.json", help="input split output json (has junctions)")
    ap.add_argument("--out_final", default=f"model/sub_assembly/D018-F205B/{step_head}_final_welds_with_junctions.json", help="output *_final_welds_with_junctions.json")
    ap.add_argument("--geometry_graph", default=None, help="optional *_geometry_graph_with_breakpoints.json (reuse node ids by coord)")
    ap.add_argument("--only_modes", default="", help="comma list of junction modes to apply (blank=all)")
    ap.add_argument("--max_remove_length", type=float, default=30, help="max length (mm) for removing breakpoint-to-breakpoint edges (default: no limit)")
    args = ap.parse_args()

    only_modes = [x.strip() for x in args.only_modes.split(",") if x.strip()] if args.only_modes else None

    update_final_welds_with_junctions(
        final_welds_json=args.final_welds,
        split_result_json=args.split_result,
        out_json=args.out_final,
        geometry_graph_json=args.geometry_graph,
        only_modes=only_modes,
        add_corner_strategy=None,
        max_remove_length=args.max_remove_length,
    )

if __name__ == "__main__":
    main()
