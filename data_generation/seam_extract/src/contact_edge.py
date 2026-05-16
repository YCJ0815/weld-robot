# contact_edge.py (optimized + profiled, minimal-invasive)
from __future__ import annotations

import os
import json
import math
import random
import time
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict, Counter
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt

from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Face, TopoDS_Edge
from OCC.Core.TopoDS import topods

from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Line, GeomAbs_Circle, GeomAbs_BSplineCurve
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Pln
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopAbs import TopAbs_REVERSED


# ============================================================
# Profiling / counters (minimal intrusive)
# ============================================================
class _Profiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timers = defaultdict(float)
        self.counts = Counter()
        self.kv = {}

    @contextmanager
    def t(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timers[name] += (time.perf_counter() - t0)

    def inc(self, name: str, n: int = 1):
        if self.enabled:
            self.counts[name] += n

    def set(self, key: str, value: Any):
        if self.enabled:
            self.kv[key] = value

    def report(self, topk: int = 50):
        if not self.enabled:
            return

        def _fmt(s: float) -> str:
            if s >= 1.0:
                return f"{s:.3f}s"
            return f"{s*1000.0:.2f}ms"

        print("\n================ PROFILE REPORT ================")
        if self.kv:
            print("[meta]")
            for k, v in self.kv.items():
                print(f"  - {k}: {v}")

        print("\n[counts]")
        for k, v in self.counts.most_common(topk):
            print(f"  - {k}: {v}")

        print("\n[timers]")
        for k, v in sorted(self.timers.items(), key=lambda x: x[1], reverse=True)[:topk]:
            print(f"  - {k}: {_fmt(v)}")

        # Handy ratios
        pairs = self.counts.get("solid_pairs_total", 0)
        cand = self.counts.get("solid_pairs_candidates", 0)
        sec = self.counts.get("section_called", 0)
        sec_empty = self.counts.get("section_empty", 0)
        fbA = self.counts.get("fallback_face_face_used", 0)
        fbB = self.counts.get("fallback_plane_near_used", 0)

        if pairs > 0:
            print("\n[ratios]")
            if cand:
                print(f"  - candidate_pairs / total_pairs: {cand}/{pairs} = {cand/pairs:.3%}")
            if sec:
                print(f"  - section_empty / section_called: {sec_empty}/{sec} = {sec_empty/max(sec,1):.3%}")
            if cand:
                print(f"  - fallback_face_face_used / candidate_pairs: {fbA}/{cand} = {fbA/max(cand,1):.3%}")
                print(f"  - fallback_plane_near_used / candidate_pairs: {fbB}/{cand} = {fbB/max(cand,1):.3%}")

        print("================================================\n")


# ============================================================
# Basic topology helpers
# ============================================================
def _shape_edges(shape: TopoDS_Shape):
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        yield topods.Edge(exp.Current())
        exp.Next()


def _shape_faces(shape: TopoDS_Shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        yield topods.Face(exp.Current())
        exp.Next()


# ============================================================
# Bounding box helpers
# ============================================================
def bbox_of_shape(shape: TopoDS_Shape) -> Tuple[np.ndarray, np.ndarray]:
    b = Bnd_Box()
    brepbndlib_Add(shape, b)
    xmin, ymin, zmin, xmax, ymax, zmax = b.Get()
    return np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])


def bbox_distance(b1_min, b1_max, b2_min, b2_max) -> float:
    """Min gap between two AABBs (0 if overlapping)."""
    dx = max(0.0, max(b1_min[0] - b2_max[0], b2_min[0] - b1_max[0]))
    dy = max(0.0, max(b1_min[1] - b2_max[1], b2_min[1] - b1_max[1]))
    dz = max(0.0, max(b1_min[2] - b2_max[2], b2_min[2] - b1_max[2]))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _expand_bbox(bmin: np.ndarray, bmax: np.ndarray, d: float) -> Tuple[np.ndarray, np.ndarray]:
    dd = float(d)
    return bmin - dd, bmax + dd


# ============================================================
# Cheap-ish distance wrappers (centralize + profile)
# ============================================================
def _dist_shape_shape(a: TopoDS_Shape, b: TopoDS_Shape, prof: _Profiler, tag: str) -> float:
    prof.inc(f"{tag}_called")
    with prof.t(f"{tag}_time"):
        dss = BRepExtrema_DistShapeShape(a, b)
        dss.Perform()
        if dss.IsDone():
            return float(dss.Value())
        return float("inf")


# ============================================================
# Edge classification (lightweight by default)
# ============================================================
def classify_edge(
    edge: TopoDS_Edge,
    *,
    bspline_detail: bool = False,
    bspline_len_samples: int = 24,
    bspline_vis_samples: int = 24,
) -> dict:
    """
    Minimal-invasive optimization:
      - default bspline_detail=False: do NOT extract poles/knots/weights/mults
      - still returns: type/start/end/length
      - for bspline: returns 'samples' for visualization (reduced), adjustable
    """
    cad = BRepAdaptor_Curve(edge)
    ctype = cad.GetType()
    first_param = cad.FirstParameter()
    last_param = cad.LastParameter()

    # Line
    if ctype == GeomAbs_Line:
        p1 = cad.Value(first_param)
        p2 = cad.Value(last_param)
        length = math.dist((p1.X(), p1.Y(), p1.Z()), (p2.X(), p2.Y(), p2.Z()))
        return {
            "type": "line",
            "start": [p1.X(), p1.Y(), p1.Z()],
            "end": [p2.X(), p2.Y(), p2.Z()],
            "length": length,
        }

    # Circle arc
    if ctype == GeomAbs_Circle:
        circ = cad.Circle()
        center = circ.Location()
        radius = circ.Radius()
        u1 = first_param
        u2 = last_param
        p1 = cad.Value(u1)
        p2 = cad.Value(u2)
        angle = abs(u2 - u1)
        direction = "clockwise" if u2 < u1 else "counterclockwise"
        length = float(radius) * float(angle)
        return {
            "type": "arc",
            "start": [p1.X(), p1.Y(), p1.Z()],
            "end": [p2.X(), p2.Y(), p2.Z()],
            "center": [center.X(), center.Y(), center.Z()],
            "radius": float(radius),
            "angle": float(angle),
            "direction": direction,
            "length": float(length),
        }

    # B-spline
    if ctype == GeomAbs_BSplineCurve:
        # --- length by sampling (adjustable) ---
        # NOTE: Using proper OCC length tools is possible, but this keeps changes minimal.
        u1, u2 = first_param, last_param
        if not (math.isfinite(u1) and math.isfinite(u2)):
            return {"type": "unknown", "length": 0.0}

        # length samples
        nlen = max(8, int(bspline_len_samples))
        t_vals = np.linspace(u1, u2, nlen)
        pts = [cad.Value(float(t)) for t in t_vals]
        length = 0.0
        for i in range(1, len(pts)):
            p_prev, p_curr = pts[i - 1], pts[i]
            length += math.sqrt(
                (p_curr.X() - p_prev.X()) ** 2
                + (p_curr.Y() - p_prev.Y()) ** 2
                + (p_curr.Z() - p_prev.Z()) ** 2
            )

        # vis samples (for json visual)
        nvis = max(6, int(bspline_vis_samples))
        t_vis = np.linspace(u1, u2, nvis)
        samples = []
        for t in t_vis:
            p = cad.Value(float(t))
            samples.append([p.X(), p.Y(), p.Z()])

        p_start = cad.Value(u1)
        p_end = cad.Value(u2)

        out = {
            "type": "bspline",
            "param_range": [float(u1), float(u2)],
            "start": [p_start.X(), p_start.Y(), p_start.Z()],
            "end": [p_end.X(), p_end.Y(), p_end.Z()],
            "length": float(length),
            "samples": samples,
        }

        if bspline_detail:
            # heavy details only if requested
            try:
                geom_bspline = cad.BSpline()
                degree = geom_bspline.Degree()
                num_poles = geom_bspline.NbPoles()
                num_knots = geom_bspline.NbKnots()
                is_periodic = bool(geom_bspline.IsPeriodic())
                is_rational = bool(geom_bspline.IsRational())

                poles = []
                for i in range(1, num_poles + 1):
                    p = geom_bspline.Pole(i)
                    poles.append([p.X(), p.Y(), p.Z()])

                weights = [geom_bspline.Weight(i) for i in range(1, num_poles + 1)] if is_rational else [1.0] * num_poles
                knots = [geom_bspline.Knot(i) for i in range(1, num_knots + 1)]
                mults = [geom_bspline.Multiplicity(i) for i in range(1, num_knots + 1)]

                out.update({
                    "degree": int(degree),
                    "is_periodic": is_periodic,
                    "is_rational": is_rational,
                    "poles": poles,
                    "weights": weights,
                    "knots": knots,
                    "multiplicities": mults,
                })
            except Exception:
                pass

        return out

    return {"type": "unknown", "length": 0.0}


# ============================================================
# Canonical keys for de-dup
# ============================================================
def canonical_line_key(start, end, ndigits: int = 6):
    s = tuple(round(float(v), ndigits) for v in start)
    e = tuple(round(float(v), ndigits) for v in end)
    return ("line", s, e) if s <= e else ("line", e, s)


def canonical_arc_key(center, radius, start, end, ndigits: int = 6):
    c = tuple(round(float(v), ndigits) for v in center)
    r = round(float(radius), ndigits)
    s = tuple(round(float(v), ndigits) for v in start)
    e = tuple(round(float(v), ndigits) for v in end)
    return ("arc", c, r, s, e) if s <= e else ("arc", c, r, e, s)


def canonical_bspline_key(samples, ndigits: int = 6):
    """
    Approx key for B-spline: start + mid + end samples.
    """
    if not samples or len(samples) < 2:
        return ("bspline",)
    s = tuple(round(float(v), ndigits) for v in samples[0])
    e = tuple(round(float(v), ndigits) for v in samples[-1])
    mid = samples[len(samples) // 2]
    m = tuple(round(float(v), ndigits) for v in mid)
    return ("bspline", s, m, e) if s <= e else ("bspline", e, m, s)


# ============================================================
# Plane coincidence (fallback)
# ============================================================
def point_to_plane_distance(point: gp_Pnt, plane: gp_Pln) -> float:
    o = plane.Location()
    n = plane.Axis().Direction()
    v = gp_Vec(o, point)
    return abs(v.Dot(gp_Vec(n.X(), n.Y(), n.Z())))


def are_planes_almost_coincident(face_a: TopoDS_Face,
                                 face_b: TopoDS_Face,
                                 ang_tol: float = 1e-3,
                                 dist_tol: float = 1e-3) -> bool:
    surf_a = BRepAdaptor_Surface(face_a)
    surf_b = BRepAdaptor_Surface(face_b)
    if surf_a.GetType() != GeomAbs_Plane or surf_b.GetType() != GeomAbs_Plane:
        return False

    pln_a = surf_a.Plane()
    pln_b = surf_b.Plane()

    na = pln_a.Axis().Direction()
    nb = pln_b.Axis().Direction()
    dot = max(min(na.Dot(nb), 1.0), -1.0)
    theta = math.acos(abs(dot))
    if theta > ang_tol:
        return False

    def face_center(face: TopoDS_Face) -> gp_Pnt:
        ad = BRepAdaptor_Surface(face)
        umin, umax = ad.FirstUParameter(), ad.LastUParameter()
        vmin, vmax = ad.FirstVParameter(), ad.LastVParameter()
        return ad.Value(0.5 * (umin + umax), 0.5 * (vmin + vmax))

    pc_a = face_center(face_a)
    pc_b = face_center(face_b)
    d1 = point_to_plane_distance(pc_a, pln_b)
    d2 = point_to_plane_distance(pc_b, pln_a)
    return (d1 < dist_tol) and (d2 < dist_tol)


# ============================================================
# Spatial grid (AABB) to reduce O(N^2) pairs
# ============================================================
def _cell_index(x: float, cell: float) -> int:
    return int(math.floor(float(x) / float(cell)))


def _aabb_cells(bmin: np.ndarray, bmax: np.ndarray, cell: float) -> List[Tuple[int, int, int]]:
    ix0, iy0, iz0 = _cell_index(bmin[0], cell), _cell_index(bmin[1], cell), _cell_index(bmin[2], cell)
    ix1, iy1, iz1 = _cell_index(bmax[0], cell), _cell_index(bmax[1], cell), _cell_index(bmax[2], cell)
    out = []
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            for iz in range(iz0, iz1 + 1):
                out.append((ix, iy, iz))
    return out


def _build_solid_candidate_pairs(
    solid_bboxes: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cell_size: float,
    bbox_gate: float,
    prof: _Profiler,
) -> List[Tuple[int, int]]:
    """
    Build candidate solid pairs using uniform grid:
      - put each solid's expanded AABB into cells
      - only consider pairs that co-occur in any cell
      - still apply bbox_distance <= bbox_gate quick check
    Returns pairs as (i, j) using indices in solids list (0-based).
    """
    prof.set("grid_cell_size", float(cell_size))
    prof.set("bbox_gate", float(bbox_gate))

    grid: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)

    # insert expanded bboxes to reduce misses
    for i, (bmin, bmax) in enumerate(solid_bboxes):
        ebmin, ebmax = _expand_bbox(bmin, bmax, bbox_gate)
        for c in _aabb_cells(ebmin, ebmax, cell_size):
            grid[c].append(i)

    # generate candidate pairs
    pair_set = set()
    for _, ids in grid.items():
        if len(ids) < 2:
            continue
        ids_sorted = sorted(ids)
        for a in range(len(ids_sorted)):
            for b in range(a + 1, len(ids_sorted)):
                i = ids_sorted[a]
                j = ids_sorted[b]
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                pair_set.add((i, j))

    prof.inc("solid_pairs_candidates", len(pair_set))

    # bbox gate filter (cheap)
    filtered = []
    with prof.t("solid_pair_bbox_filter_time"):
        for i, j in pair_set:
            bmin_i, bmax_i = solid_bboxes[i]
            bmin_j, bmax_j = solid_bboxes[j]
            if bbox_distance(bmin_i, bmax_i, bmin_j, bmax_j) <= bbox_gate:
                filtered.append((i, j))
    prof.inc("solid_pairs_after_bbox_gate", len(filtered))
    return filtered


# ============================================================
# Face grid for fallback
# ============================================================
def _build_face_grid(
    faces: List[TopoDS_Face],
    *,
    bbox_gate: float,
    cell_size: float,
) -> Tuple[Dict[Tuple[int, int, int], List[int]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Returns (grid, face_bboxes) where grid maps cell -> face_indices.
    Faces are inserted using expanded AABBs (by bbox_gate).
    """
    face_bboxes = [bbox_of_shape(f) for f in faces]
    grid: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for idx, (bmin, bmax) in enumerate(face_bboxes):
        ebmin, ebmax = _expand_bbox(bmin, bmax, bbox_gate)
        for c in _aabb_cells(ebmin, ebmax, cell_size):
            grid[c].append(idx)
    return grid, face_bboxes


def _query_face_candidates(
    grid: Dict[Tuple[int, int, int], List[int]],
    fb_bboxes: List[Tuple[np.ndarray, np.ndarray]],
    fa_bbox: Tuple[np.ndarray, np.ndarray],
    *,
    bbox_gate: float,
    cell_size: float,
) -> List[int]:
    """
    For a face AABB, find candidate face indices from grid and filter by bbox_distance <= bbox_gate.
    """
    bmin_a, bmax_a = fa_bbox
    ebmin, ebmax = _expand_bbox(bmin_a, bmax_a, bbox_gate)
    cand = set()
    for c in _aabb_cells(ebmin, ebmax, cell_size):
        for idx in grid.get(c, []):
            cand.add(idx)
    # bbox filter
    out = []
    for idx in cand:
        bmin_b, bmax_b = fb_bboxes[idx]
        if bbox_distance(bmin_a, bmax_a, bmin_b, bmax_b) <= bbox_gate:
            out.append(idx)
    return out


# ============================================================
# Nearness checks (edge-face)
# ============================================================
def _edge_is_near_face_by_distance(
    edge: TopoDS_Edge,
    face: TopoDS_Face,
    tol: float,
    prof: _Profiler,
) -> bool:
    """
    Minimal-invasive replacement for projection sampling:
      - uses DistShapeShape(edge, face) which is usually cheaper than many projections
    """
    d = _dist_shape_shape(edge, face, prof, tag="dist_edge_face")
    return d <= float(tol)


def _collect_near_edges_between_faces(
    fa: TopoDS_Face,
    fb: TopoDS_Face,
    tol: float,
    prof: _Profiler,
) -> List[TopoDS_Edge]:
    """Collect boundary edges from fa and fb that are near the other face (both directions)."""
    near_edges: List[TopoDS_Edge] = []
    topo_a = TopologyExplorer(fa)
    topo_b = TopologyExplorer(fb)

    for e in topo_a.edges():
        if _edge_is_near_face_by_distance(e, fb, tol, prof):
            near_edges.append(e)

    for e in topo_b.edges():
        if _edge_is_near_face_by_distance(e, fa, tol, prof):
            near_edges.append(e)

    return near_edges


# ============================================================
# Preferred Normal Calculation
# ============================================================
def _get_shape_normals_at_point(shape: TopoDS_Shape, p: gp_Pnt, tol: float = 1e-3) -> List[gp_Vec]:
    """
    Find faces of shape close to p, and return their normals at p (projection).
    Adjusts for face orientation (points outward for solids).
    """
    normals = []
    candidates = []
    
    if shape.ShapeType() == TopAbs_FACE:
        candidates = [topods.Face(shape)]
    else:
        # For solids, we iterate all faces. 
        # Optimization: could filter by bbox, but solid usually has few faces.
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            candidates.append(topods.Face(exp.Current()))
            exp.Next()
            
    for f in candidates:
        bas = BRepAdaptor_Surface(f)
        surf = bas.Surface().Surface()
        
        # Project point to underlying geometry surface
        proj = GeomAPI_ProjectPointOnSurf(p, surf)
        if proj.NbPoints() < 1:
            continue
            
        dist = proj.LowerDistance()
        # Coarse filter on distance to surface geometry
        if dist > tol * 20.0:
            continue
            
        # Check UV bounds to ensure we are "on" the face
        u, v = proj.LowerDistanceParameters()
        umin, umax = bas.FirstUParameter(), bas.LastUParameter()
        vmin, vmax = bas.FirstVParameter(), bas.LastVParameter()
        
        # Relaxed bound check (allow small tolerance)
        if not (umin - tol <= u <= umax + tol and vmin - tol <= v <= vmax + tol):
            continue
            
        # Refined distance check could go here, but geometry distance is usually good enough
        # if UV is within bounds.
        
        # Calculate normal properties
        props = GeomLProp_SLProps(surf, u, v, 1, 1e-6)
        if props.IsNormalDefined():
            n_dir = props.Normal()
            n = gp_Vec(n_dir.XYZ())
            # Flip if face orientation is reversed relative to surface
            if f.Orientation() == TopAbs_REVERSED:
                n.Reverse()
            normals.append(n)
            
    return normals


def _compute_preferred_normal(edge: TopoDS_Edge, shape_a: TopoDS_Shape, shape_b: TopoDS_Shape) -> Optional[List[float]]:
    """
    Calculate the preferred welding normal based on the two shapes forming the edge.
    Strategy: Average of normals of adjacent faces, projected to be perpendicular to tangent.
    """
    try:
        # 1. Sample midpoint
        bc = BRepAdaptor_Curve(edge)
        u_start = bc.FirstParameter()
        u_end = bc.LastParameter()
        
        if not (math.isfinite(u_start) and math.isfinite(u_end)):
            return None
            
        u_mid = (u_start + u_end) * 0.5
        p_mid = bc.Value(u_mid)
        
        # 2. Tangent at midpoint
        p_dummy = gp_Pnt()
        tan_vec = gp_Vec()
        bc.D1(u_mid, p_dummy, tan_vec)
        if tan_vec.Magnitude() < 1e-12:
            return None
        tan_vec.Normalize()
        
        # 3. Get normals from both shapes
        # Use a slightly larger tolerance for detection as Section edges might be slightly off
        tol = 1e-2 
        
        normals_a = _get_shape_normals_at_point(shape_a, p_mid, tol)
        normals_b = _get_shape_normals_at_point(shape_b, p_mid, tol)
        
        if not normals_a or not normals_b:
            return None
            
        # Average normals for A (handle singularities like corners)
        na = gp_Vec(0,0,0)
        for n in normals_a:
            na.Add(n)
        if na.Magnitude() > 1e-12:
            na.Normalize()
        else:
            return None
            
        # Average normals for B
        nb = gp_Vec(0,0,0)
        for n in normals_b:
            nb.Add(n)
        if nb.Magnitude() > 1e-12:
            nb.Normalize()
        else:
            return None
            
        # 4. Combine: Bisector direction
        n_pref = na.Added(nb)
        
        # If normals are opposite (e.g. butt weld on same plane), sum is zero.
        # In that case, the normal should be perpendicular to the surface (which is na).
        # Or if it's a lap joint.
        if n_pref.Magnitude() < 1e-6:
            n_pref = na 
            
        # 5. Project to plane perpendicular to tangent (ensure orthogonality)
        # n_final = n_pref - (n_pref . tan) * tan
        dot = n_pref.Dot(tan_vec)
        proj = tan_vec.Multiplied(dot)
        n_final = n_pref.Subtracted(proj)
        
        if n_final.Magnitude() < 1e-12:
            return None
            
        n_final.Normalize()
        
        return [n_final.X(), n_final.Y(), n_final.Z()]
        
    except Exception as e:
        print(f"Warning: _compute_preferred_normal failed: {e}")
        return None


# ============================================================
# Geometry graph helpers (kept, but classify_edge is lightweight)
# ============================================================
def _quantize(v: float, tol: float) -> float:
    if tol <= 0:
        return float(v)
    return round(float(v) / tol) * tol


def _point_key_xyz(xyz, point_tol: float = 0.2, ndigits: int = 6) -> str:
    xq = round(_quantize(float(xyz[0]), point_tol), ndigits)
    yq = round(_quantize(float(xyz[1]), point_tol), ndigits)
    zq = round(_quantize(float(xyz[2]), point_tol), ndigits)
    return f"P:{xq:.{ndigits}f},{yq:.{ndigits}f},{zq:.{ndigits}f}"


def _key_to_jsonable(key_tuple):
    if isinstance(key_tuple, tuple):
        return [_key_to_jsonable(x) for x in key_tuple]
    if isinstance(key_tuple, list):
        return [_key_to_jsonable(x) for x in key_tuple]
    return key_tuple


def _key_to_str(key_tuple) -> str:
    return json.dumps(_key_to_jsonable(key_tuple), separators=(",", ":"), ensure_ascii=False)


def build_geometry_graph_from_contact_edges(
    compound: TopoDS_Shape,
    contact_edges: Dict[int, dict],
    *,
    output_json: str,
    point_tol: float = 0.2,
    key_ndigits: int = 6,
    min_geom_edge_length: float = 0.0,
    include_weld_edges_in_adjacent: bool = False,
    store_all_geom_edges: bool = False,
    source_step: str = "",
    prof: Optional[_Profiler] = None,
) -> None:
    if prof is None:
        prof = _Profiler(enabled=False)

    with prof.t("geometry_graph_total_time"):
        # ---- A) nodes + weld_edges ----
        nodes: Dict[str, dict] = {}
        pkey_to_nid: Dict[str, str] = {}
        nid_counter = 1

        weld_edges: Dict[str, dict] = {}
        weld_key_str_set: set[str] = set()
        weld_endpoint_pkeys: set[str] = set()

        def get_or_create_node(p_xyz):
            nonlocal nid_counter
            pk = _point_key_xyz(p_xyz, point_tol=point_tol, ndigits=key_ndigits)
            if pk in pkey_to_nid:
                return pkey_to_nid[pk]
            nid = f"N{nid_counter}"
            nid_counter += 1
            pkey_to_nid[pk] = nid
            nodes[nid] = {
                "point": [
                    round(_quantize(float(p_xyz[0]), point_tol), key_ndigits),
                    round(_quantize(float(p_xyz[1]), point_tol), key_ndigits),
                    round(_quantize(float(p_xyz[2]), point_tol), key_ndigits),
                ],
                "key": pk,
                "incident_weld_edges": [],
                "adjacent_geom_edges": [],
            }
            return nid

        for wid, info in contact_edges.items():
            etype = info.get("type", "unknown")
            if etype not in ("line", "arc", "bspline"):
                continue
            s = info.get("start")
            e = info.get("end")
            if not s or not e:
                continue

            n_start = get_or_create_node(s)
            n_end = get_or_create_node(e)
            nodes[n_start]["incident_weld_edges"].append(str(wid))
            nodes[n_end]["incident_weld_edges"].append(str(wid))

            pks = _point_key_xyz(s, point_tol=point_tol, ndigits=key_ndigits)
            pke = _point_key_xyz(e, point_tol=point_tol, ndigits=key_ndigits)
            weld_endpoint_pkeys.add(pks)
            weld_endpoint_pkeys.add(pke)

            if etype == "line":
                k = canonical_line_key(s, e, ndigits=key_ndigits)
            elif etype == "arc":
                k = canonical_arc_key(info.get("center"), float(info.get("radius", 0.0)), s, e, ndigits=key_ndigits)
            else:
                k = canonical_bspline_key(info.get("samples", []), ndigits=key_ndigits)
            weld_key_str_set.add(_key_to_str(k))

            weld_edges[str(wid)] = dict(info)
            weld_edges[str(wid)].update({
                "start_node": n_start,
                "end_node": n_end,
                "adjacent_at_start": [],
                "adjacent_at_end": [],
            })

        # ---- B) traverse solids topo edges ----
        topo = TopologyExplorer(compound)
        solids = list(topo.solids())
        if not solids:
            raise ValueError("compound has no solids")

        geom_edges: Dict[str, dict] = {}
        geom_id_counter = 1

        solid_point_to_geom_edges: Dict[int, Any] = { (i + 1): defaultdict(set) for i in range(len(solids)) }
        geom_key_str_by_gid: Dict[str, str] = {}

        for i, solid in enumerate(solids):
            solid_id = i + 1
            for edge in _shape_edges(solid):
                # lightweight classify (do not pull heavy bspline data)
                info = classify_edge(
                    edge,
                    bspline_detail=False,
                    bspline_len_samples=16,
                    bspline_vis_samples=9,  # enough for canonical key + adjacency, cheaper
                )
                etype = info.get("type", "unknown")
                if etype not in ("line", "arc", "bspline"):
                    continue

                length = float(info.get("length", 0.0))
                if length < float(min_geom_edge_length):
                    continue

                s = info.get("start")
                e = info.get("end")
                if not s or not e:
                    continue

                pks = _point_key_xyz(s, point_tol=point_tol, ndigits=key_ndigits)
                pke = _point_key_xyz(e, point_tol=point_tol, ndigits=key_ndigits)

                if (not store_all_geom_edges) and (pks not in weld_endpoint_pkeys) and (pke not in weld_endpoint_pkeys):
                    continue

                if etype == "line":
                    ck = canonical_line_key(s, e, ndigits=key_ndigits)
                elif etype == "arc":
                    ck = canonical_arc_key(info.get("center"), float(info.get("radius", 0.0)), s, e, ndigits=key_ndigits)
                else:
                    ck = canonical_bspline_key(info.get("samples", []), ndigits=key_ndigits)

                gid = f"G{geom_id_counter}"
                geom_id_counter += 1

                gobj: Dict[str, Any] = {
                    "type": etype,
                    "solid_id": solid_id,
                    "length": length,
                    "start": s,
                    "end": e,
                    "canonical_key": _key_to_jsonable(ck),
                }

                if etype == "arc":
                    gobj.update({
                        "center": info.get("center"),
                        "radius": info.get("radius"),
                        "angle": info.get("angle"),
                        "direction": info.get("direction"),
                    })
                elif etype == "bspline":
                    # keep only samples
                    gobj.update({"samples": info.get("samples", [])})

                geom_edges[gid] = gobj
                solid_point_to_geom_edges[solid_id][pks].add(gid)
                solid_point_to_geom_edges[solid_id][pke].add(gid)
                geom_key_str_by_gid[gid] = json.dumps(gobj["canonical_key"], separators=(",", ":"), ensure_ascii=False)

        # ---- C) fill adjacency ----
        for wid, winfo in weld_edges.items():
            sids = winfo.get("solid_ids", []) or []
            s = winfo.get("start")
            e = winfo.get("end")
            if not s or not e:
                continue

            pks = _point_key_xyz(s, point_tol=point_tol, ndigits=key_ndigits)
            pke = _point_key_xyz(e, point_tol=point_tol, ndigits=key_ndigits)

            adj_s = set()
            adj_e = set()

            for sid in sids:
                try:
                    sid_int = int(sid)
                except Exception:
                    continue
                adj_s |= set(solid_point_to_geom_edges.get(sid_int, {}).get(pks, set()))
                adj_e |= set(solid_point_to_geom_edges.get(sid_int, {}).get(pke, set()))

            if not include_weld_edges_in_adjacent:
                adj_s = {gid for gid in adj_s if geom_key_str_by_gid.get(gid, "") not in weld_key_str_set}
                adj_e = {gid for gid in adj_e if geom_key_str_by_gid.get(gid, "") not in weld_key_str_set}

            winfo["adjacent_at_start"] = sorted(adj_s)
            winfo["adjacent_at_end"] = sorted(adj_e)

            ns = winfo.get("start_node")
            ne = winfo.get("end_node")
            if ns in nodes:
                nodes[ns]["adjacent_geom_edges"] = sorted(set(nodes[ns]["adjacent_geom_edges"]) | adj_s)
            if ne in nodes:
                nodes[ne]["adjacent_geom_edges"] = sorted(set(nodes[ne]["adjacent_geom_edges"]) | adj_e)

        index_point_to_geom_edges = {ninfo["key"]: list(ninfo.get("adjacent_geom_edges", [])) for ninfo in nodes.values()}
        index_point_to_nodes = {ninfo["key"]: nid for nid, ninfo in nodes.items()}

        out_obj = {
            "meta": {
                "source_step": source_step,
                "point_tol": point_tol,
                "key_ndigits": key_ndigits,
                "min_geom_edge_length": min_geom_edge_length,
                "include_weld_edges_in_adjacent": include_weld_edges_in_adjacent,
                "store_all_geom_edges": store_all_geom_edges,
                "notes": "nodes are weld endpoints; adjacent geom edges are topo edges adjacent to node points.",
            },
            "nodes": nodes,
            "weld_edges": weld_edges,
            "geom_edges": geom_edges,
            "index": {
                "point_to_geom_edges": index_point_to_geom_edges,
                "point_to_nodes": index_point_to_nodes,
            }
        }

        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=4, ensure_ascii=False)

        print(f"[geometry_graph] nodes={len(nodes)} weld_edges={len(weld_edges)} geom_edges={len(geom_edges)} -> {output_json}")


# ============================================================
# Contact boundary extraction (main)
# ============================================================
def extract_contact_boundaries_face_based(
    compound: TopoDS_Shape,
    output_file: str,
    bbox_tol: float = 1e-2,
    min_edge_length: float = 1e-3,
    do_section_approx: bool = True,
    contact_tol: float = 1e-4,
    face_tol: float = 2e-4,

    # ---- optional: geometry graph output ----
    geometry_graph_file: str | None = None,
    graph_point_tol: float = 0.2,
    graph_key_ndigits: int = 6,
    graph_min_geom_edge_length: float = 0.0,
    graph_include_weld_edges_in_adjacent: bool = False,
    graph_store_all_geom_edges: bool = False,
    graph_source_step: str = "",

    # ---- NEW: profiling / behavior knobs (safe defaults) ----
    profile: bool = True,
    use_solid_grid_candidates: bool = True,
    solid_grid_cell_size: Optional[float] = None,   # None => auto
    face_grid_cell_size: Optional[float] = None,    # None => auto

    # ---- NEW: bspline sampling knobs ----
    bspline_detail: bool = False,       # keep False for speed
    bspline_len_samples: int = 24,      # detection length sampling
    bspline_vis_samples: int = 24,      # json visualization samples
) -> None:
    prof = _Profiler(enabled=bool(profile))

    with prof.t("total_extract_time"):
        topo = TopologyExplorer(compound)
        solids: List[TopoDS_Solid] = list(topo.solids())
        nb_solids = len(solids)
        prof.set("nb_solids", nb_solids)
        prof.set("bbox_tol", bbox_tol)
        prof.set("contact_tol", contact_tol)
        prof.set("face_tol", face_tol)
        prof.set("min_edge_length", min_edge_length)
        prof.set("do_section_approx", bool(do_section_approx))
        prof.set("bspline_detail", bool(bspline_detail))
        prof.set("bspline_len_samples", int(bspline_len_samples))
        prof.set("bspline_vis_samples", int(bspline_vis_samples))

        if nb_solids < 2:
            print("实体数量小于 2，无法提取接触边界线。")
            return

        # solid bbox cache
        with prof.t("solid_bbox_time"):
            solid_bboxes = [bbox_of_shape(s) for s in solids]

        # helpful debug info: face/edge counts per solid (sampled)
        with prof.t("solid_topo_stats_time"):
            face_counts = []
            edge_counts = []
            for s in solids:
                # counting via explorer is cheap-ish
                fc = sum(1 for _ in _shape_faces(s))
                ec = sum(1 for _ in _shape_edges(s))
                face_counts.append(fc)
                edge_counts.append(ec)
            prof.set("solid_faces_min/avg/max", (min(face_counts), sum(face_counts)/len(face_counts), max(face_counts)))
            prof.set("solid_edges_min/avg/max", (min(edge_counts), sum(edge_counts)/len(edge_counts), max(edge_counts)))

        contact_edges: Dict[int, dict] = {}
        edge_id_counter = 1
        seen_keys: Dict[tuple, int] = {}

        print(f"共有 {nb_solids} 个实体，开始基于 solid-solid Section 的接触线提取（profile={profile}）。")

        # ---- Candidate solid pairs ----
        prof.inc("solid_pairs_total", nb_solids * (nb_solids - 1) // 2)

        bbox_gate = max(float(bbox_tol), 5.0 * float(contact_tol))
        if solid_grid_cell_size is None:
            # auto: a bit larger than gate to reduce false candidates
            solid_grid_cell_size = max(bbox_gate * 2.5, 1e-3)
        if face_grid_cell_size is None:
            face_grid_cell_size = max(bbox_gate * 2.0, 1e-3)

        if use_solid_grid_candidates:
            with prof.t("build_solid_candidate_pairs_time"):
                candidate_pairs = _build_solid_candidate_pairs(
                    solid_bboxes,
                    cell_size=float(solid_grid_cell_size),
                    bbox_gate=bbox_gate,
                    prof=prof,
                )
        else:
            # fallback to original N^2
            candidate_pairs = [(i, j) for i in range(nb_solids) for j in range(i + 1, nb_solids)]
            prof.inc("solid_pairs_candidates", len(candidate_pairs))

        print(f"[pairs] total={prof.counts['solid_pairs_total']} candidates={prof.counts['solid_pairs_candidates']} after_bbox_gate={prof.counts.get('solid_pairs_after_bbox_gate', 'n/a')}")

        # ---- Loop candidate pairs ----
        for i, j in candidate_pairs:
            si = solids[i]
            sj = solids[j]
            si_id = i + 1
            sj_id = j + 1

            bmin_i, bmax_i = solid_bboxes[i]
            bmin_j, bmax_j = solid_bboxes[j]
            dist_bb = bbox_distance(bmin_i, bmax_i, bmin_j, bmax_j)

            # If grid-based: most pairs already within bbox_gate; still keep safety
            if dist_bb > bbox_gate:
                # expensive distance check only when needed
                dmin = _dist_shape_shape(si, sj, prof, tag="dist_solid_solid")
                if dmin > contact_tol:
                    prof.inc("pair_rejected_by_solid_distance")
                    continue

            print(f"\n实体对 ({si_id}, {sj_id}) bbox gap={dist_bb:.4e} -> try Section...")

            # 1) solid-solid section
            sec_edges_with_source: List[Tuple[TopoDS_Edge, TopoDS_Shape, TopoDS_Shape]] = []
            with prof.t("section_total_time"):
                try:
                    prof.inc("section_called")
                    sec = BRepAlgoAPI_Section(si, sj)
                    try:
                        sec.SetFuzzyValue(contact_tol)
                    except Exception:
                        pass
                    sec.ComputePCurveOn1(False)
                    sec.ComputePCurveOn2(False)
                    sec.Approximation(bool(do_section_approx))
                    sec.Build()
                    if not sec.IsDone():
                        print("  * Section 失败")
                        prof.inc("section_failed")
                        continue
                    sec_shape = sec.Shape()
                    # Store with source solids
                    for e in _shape_edges(sec_shape):
                        sec_edges_with_source.append((e, si, sj))
                except Exception as e:
                    print(f"  * Section 异常：{e}")
                    prof.inc("section_exception")
                    continue

            # 2) fallback A: section empty -> face-face nearness (with face grid)
            if not sec_edges_with_source:
                prof.inc("section_empty")
                # solid distance gate (do it ONCE here, not twice)
                dmin = _dist_shape_shape(si, sj, prof, tag="dist_solid_solid_after_empty_section")
                if dmin > contact_tol:
                    print(f" * Section 无交线且实体最小距离 {dmin:.3e} > contact_tol，跳过")
                    prof.inc("pair_rejected_after_empty_section_by_distance")
                    continue

                print(" * Section 无交线，启用 face-face 贴近回退提取（face grid）...")
                prof.inc("fallback_face_face_used")

                topo_si = TopologyExplorer(si)
                topo_sj = TopologyExplorer(sj)

                faces_i = list(topo_si.faces())
                faces_j = list(topo_sj.faces())

                # Build face grid for sj (one-time per pair)
                with prof.t("fallbackA_build_face_grid_time"):
                    grid_j, face_bboxes_j = _build_face_grid(
                        faces_j,
                        bbox_gate=bbox_gate,
                        cell_size=float(face_grid_cell_size),
                    )
                    face_bboxes_i = [bbox_of_shape(fa) for fa in faces_i]

                # fallback_edges: List[TopoDS_Edge] = [] # Removed
                with prof.t("fallbackA_face_scan_time"):
                    for idx_a, fa in enumerate(faces_i):
                        bmin_a, bmax_a = face_bboxes_i[idx_a]
                        cand_b = _query_face_candidates(
                            grid_j,
                            face_bboxes_j,
                            (bmin_a, bmax_a),
                            bbox_gate=bbox_gate,
                            cell_size=float(face_grid_cell_size),
                        )
                        prof.inc("fallbackA_faceA_count", 1)
                        prof.inc("fallbackA_faceB_candidates_total", len(cand_b))

                        for idx_b in cand_b:
                            fb = faces_j[idx_b]
                            # face-face distance (still expensive, but now much fewer pairs)
                            d_face = _dist_shape_shape(fa, fb, prof, tag="dist_face_face")
                            if d_face > contact_tol:
                                continue

                            # collect near edges using edge-face distance (not projection sampling)
                            found_edges = _collect_near_edges_between_faces(fa, fb, tol=face_tol, prof=prof)
                            for fe in found_edges:
                                sec_edges_with_source.append((fe, fa, fb))

                if not sec_edges_with_source:
                    print(" * face-face 回退也未找到边")
                    prof.inc("fallbackA_no_edges")
                    continue

            pair_edges = 0

            # 3) classify + length filter + de-dup + record
            with prof.t("pair_edges_classify_time"):
                for e, shape_a, shape_b in sec_edges_with_source:
                    info = classify_edge(
                        e,
                        bspline_detail=bspline_detail,
                        bspline_len_samples=bspline_len_samples,
                        bspline_vis_samples=bspline_vis_samples,
                    )
                    etype = info.get("type", "unknown")
                    prof.inc(f"edge_type_{etype}")
                    if etype not in ("line", "arc", "bspline"):
                        continue

                    length = float(info.get("length", 0.0))
                    if length < min_edge_length:
                        prof.inc("edge_rejected_by_min_length")
                        continue
                    
                    # Compute Preferred Normal
                    pref_norm = _compute_preferred_normal(e, shape_a, shape_b)
                    info["preferred_normal"] = pref_norm

                    if etype == "line":
                        key = canonical_line_key(info["start"], info["end"])
                    elif etype == "arc":
                        key = canonical_arc_key(info["center"], info["radius"], info["start"], info["end"])
                    else:
                        key = canonical_bspline_key(info.get("samples", []))

                    if key in seen_keys:
                        old_id = seen_keys[key]
                        old = contact_edges.get(old_id)
                        if old:
                            sset = set(old.get("solid_ids", []))
                            sset.update([si_id, sj_id])
                            old["solid_ids"] = sorted(list(sset))
                        prof.inc("edge_dedup_hit")
                        continue

                    seen_keys[key] = edge_id_counter

                    edge_data: Dict[str, Any] = {
                        "type": etype,
                        "start": info.get("start"),
                        "end": info.get("end"),
                        "length": length,
                        "solid_ids": [si_id, sj_id],
                        "preferred_normal": info.get("preferred_normal"),
                    }

                    if etype == "arc":
                        edge_data.update({
                            "center": info.get("center"),
                            "radius": info.get("radius"),
                            "angle": info.get("angle"),
                            "direction": info.get("direction"),
                        })

                    if etype == "bspline":
                        # keep samples for visualization
                        edge_data.update({
                            "param_range": info.get("param_range"),
                            "samples": info.get("samples", []),
                        })
                        # optional heavy details included only if bspline_detail=True
                        for k in ("degree", "is_periodic", "is_rational", "poles", "weights", "knots", "multiplicities"):
                            if k in info:
                                edge_data[k] = info[k]

                    contact_edges[edge_id_counter] = edge_data
                    edge_id_counter += 1
                    pair_edges += 1
                    prof.inc("edges_kept_total")

            # 4) fallback B: planar coincidence nearness (kept, but still heavy; now counts + timing)
            if pair_edges == 0:
                prof.inc("fallback_plane_near_used")
                fallback_face_tol = max(0.5, bbox_tol * 50.0)

                topo_si = TopologyExplorer(si)
                topo_sj = TopologyExplorer(sj)

                faces_i = list(topo_si.faces())
                faces_j = list(topo_sj.faces())

                fb_edges = 0
                with prof.t("fallbackB_time"):
                    for fa in faces_i:
                        for fb in faces_j:
                            if not are_planes_almost_coincident(fa, fb, ang_tol=1e-3, dist_tol=fallback_face_tol):
                                continue

                            near_edges = _collect_near_edges_between_faces(fa, fb, tol=fallback_face_tol, prof=prof)
                            for e in near_edges:
                                info = classify_edge(
                                    e,
                                    bspline_detail=bspline_detail,
                                    bspline_len_samples=bspline_len_samples,
                                    bspline_vis_samples=bspline_vis_samples,
                                )
                                etype = info.get("type", "unknown")
                                if etype not in ("line", "arc", "bspline"):
                                    continue

                                length = float(info.get("length", 0.0))
                                if length < min_edge_length:
                                    continue

                                if etype == "line":
                                    key = canonical_line_key(info["start"], info["end"])
                                elif etype == "arc":
                                    key = canonical_arc_key(info["center"], info["radius"], info["start"], info["end"])
                                else:
                                    key = canonical_bspline_key(info.get("samples", []))

                                if key in seen_keys:
                                    old_id = seen_keys[key]
                                    old = contact_edges.get(old_id)
                                    if old:
                                        sset = set(old.get("solid_ids", []))
                                        sset.update([si_id, sj_id])
                                        old["solid_ids"] = sorted(list(sset))
                                    continue

                                seen_keys[key] = edge_id_counter

                                edge_data: Dict[str, Any] = {
                                    "type": etype,
                                    "start": info.get("start"),
                                    "end": info.get("end"),
                                    "length": length,
                                    "solid_ids": [si_id, sj_id],
                                    "source": "face_near_fallback",
                                }

                                if etype == "arc":
                                    edge_data.update({
                                        "center": info.get("center"),
                                        "radius": info.get("radius"),
                                        "angle": info.get("angle"),
                                        "direction": info.get("direction"),
                                    })

                                if etype == "bspline":
                                    edge_data.update({
                                        "param_range": info.get("param_range"),
                                        "samples": info.get("samples", []),
                                    })
                                    for k in ("degree", "is_periodic", "is_rational", "poles", "weights", "knots", "multiplicities"):
                                        if k in info:
                                            edge_data[k] = info[k]

                                contact_edges[edge_id_counter] = edge_data
                                edge_id_counter += 1
                                fb_edges += 1
                                prof.inc("edges_kept_total")

                if fb_edges > 0:
                    pair_edges += fb_edges
                    print(f"  * fallback(face_near) 补到 {fb_edges} 条")
                else:
                    prof.inc("fallbackB_no_edges")

            print(f"实体对 ({si_id}, {sj_id}) candidates={len(sec_edges_with_source)} kept={pair_edges}.")

        # write contact edges json
        with prof.t("write_contact_edges_json_time"):
            out_obj = {"contact_edges": {str(k): v for k, v in contact_edges.items()}}
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(out_obj, f, indent=4, ensure_ascii=False)

        print(f"\n总共提取到 {len(contact_edges)} 条接触线。JSON 已写入：{output_file}")

        # write geometry graph json (optional)
        if geometry_graph_file:
            try:
                build_geometry_graph_from_contact_edges(
                    compound=compound,
                    contact_edges=contact_edges,
                    output_json=geometry_graph_file,
                    point_tol=graph_point_tol,
                    key_ndigits=graph_key_ndigits,
                    min_geom_edge_length=graph_min_geom_edge_length,
                    include_weld_edges_in_adjacent=graph_include_weld_edges_in_adjacent,
                    store_all_geom_edges=graph_store_all_geom_edges,
                    source_step=graph_source_step,
                    prof=prof,
                )
            except Exception as e:
                print(f"[geometry_graph] failed: {e}")
                prof.inc("geometry_graph_failed")

    # dump profiling summary at end
    prof.report()


# ============================================================
# Visualization (compatible with your previous behavior)
# ============================================================
def visualize_contact_edges_from_json(json_file: str, only_pair=None, min_length=0.0):
    if not os.path.exists(json_file):
        print(f"错误：JSON 文件 {json_file} 不存在。")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges = data.get("contact_edges", {})
    if edges is None:
        print("JSON 缺少 contact_edges 字段。")
        return
    if len(edges) == 0:
        print("contact_edges 为空（0 条）。")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Contact Boundaries (Face-based Detection)")

    pair_color = {}
    all_x, all_y, all_z = [], [], []

    for edge_id, info in edges.items():
        etype = info.get("type", "unknown")
        sids = info.get("solid_ids", [])
        if only_pair is not None:
            if tuple(sorted(sids)) != tuple(sorted(only_pair)):
                continue

        if float(info.get("length", 0.0)) < min_length:
            continue

        key = tuple(sorted(sids)) if sids else ("none",)
        if key not in pair_color:
            pair_color[key] = (random.random(), random.random(), random.random())
        c = pair_color[key]

        if etype == "line":
            s = info.get("start")
            e = info.get("end")
            if not s or not e:
                continue
            xs = [s[0], e[0]]
            ys = [s[1], e[1]]
            zs = [s[2], e[2]]
            ax.plot(xs, ys, zs, color=c, linewidth=0.5)
            all_x.extend(xs); all_y.extend(ys); all_z.extend(zs)

        elif etype == "arc":
            s = np.array(info.get("start"), float)
            e = np.array(info.get("end"), float)
            center = np.array(info.get("center"), float)
            r = float(info.get("radius", 0.0))
            angle = float(info.get("angle", 0.0))
            if r <= 0:
                continue

            v1 = s - center
            v2 = e - center
            if np.linalg.norm(v1) < 1e-12 or np.linalg.norm(v2) < 1e-12:
                xs = [s[0], e[0]]; ys = [s[1], e[1]]; zs = [s[2], e[2]]
                ax.plot(xs, ys, zs, color=c, linewidth=0.5)
                all_x.extend(xs); all_y.extend(ys); all_z.extend(zs)
                continue

            v1n = v1 / np.linalg.norm(v1)
            v2n = v2 / np.linalg.norm(v2)
            n = np.cross(v1n, v2n)
            if np.linalg.norm(n) < 1e-12:
                xs = [s[0], e[0]]; ys = [s[1], e[1]]; zs = [s[2], e[2]]
                ax.plot(xs, ys, zs, color=c, linewidth=0.5)
                all_x.extend(xs); all_y.extend(ys); all_z.extend(zs)
                continue
            n /= np.linalg.norm(n)
            b2 = np.cross(n, v1n)
            b2 /= np.linalg.norm(b2)

            x2 = np.dot(v2n, v1n)
            y2 = np.dot(v2n, b2)
            theta2 = math.atan2(y2, x2)

            total_angle = angle if abs(angle) > 1e-6 else theta2
            num_seg = max(12, int(abs(total_angle) / (math.pi / 36)))
            ts = np.linspace(0.0, total_angle, num_seg)

            xs, ys, zs = [], [], []
            for t in ts:
                v = math.cos(t) * v1n + math.sin(t) * b2
                p = center + r * v
                xs.append(p[0]); ys.append(p[1]); zs.append(p[2])

            ax.plot(xs, ys, zs, color=c, linewidth=0.5)
            all_x.extend(xs); all_y.extend(ys); all_z.extend(zs)

        elif etype == "bspline":
            samples = info.get("samples", [])
            if not samples or len(samples) < 2:
                continue
            xs = [p[0] for p in samples]
            ys = [p[1] for p in samples]
            zs = [p[2] for p in samples]
            ax.plot(xs, ys, zs, color=c, linewidth=0.5)
            all_x.extend(xs); all_y.extend(ys); all_z.extend(zs)

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
    print("接触边界线可视化完成。")