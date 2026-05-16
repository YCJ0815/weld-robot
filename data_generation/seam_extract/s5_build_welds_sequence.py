import json
import os
import math
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# INPUT_JSON = "model/sub_assembly/e4db9188/e4db9188_final_welds_with_junctions.json"
# OUTPUT_JSON = "model/sub_assembly/e4db9188/e4db9188_final_welds_with_junctions_sequence.json"

INPUT_JSON = "model/sub_assembly/model2/model2_final_welds_with_junctions.json"
OUTPUT_JSON = "model/sub_assembly/model2/model2_final_welds_with_junctions_sequence.json"




POINT_KEY_DIGITS = 1

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _as_xyz(v: Any) -> List[float] | None:
    if isinstance(v, list) and len(v) == 3:
        try:
            return [float(v[0]), float(v[1]), float(v[2])]
        except Exception:
            return None
    return None

def point_key(xyz: List[float], ndigits: int = POINT_KEY_DIGITS) -> Tuple[float, float, float]:
    return (
        round(float(xyz[0]), ndigits),
        round(float(xyz[1]), ndigits),
        round(float(xyz[2]), ndigits),
    )

def _sort_key(edge_id: Any) -> Tuple[int, str]:
    s = str(edge_id)
    return (len(s), s)

def _edge_samples(edge: Dict[str, Any]) -> List[List[float]]:
    samples = edge.get("samples")
    if isinstance(samples, list) and len(samples) >= 2:
        pts = []
        for p in samples:
            xyz = _as_xyz(p)
            if xyz:
                pts.append(xyz)
        if len(pts) >= 2:
            return pts
    s = _as_xyz(edge.get("start"))
    e = _as_xyz(edge.get("end"))
    if s and e:
        return [s, e]
    return []


def _edge_keys(edge: Dict[str, Any]) -> Tuple[Tuple[float, float, float] | None, Tuple[float, float, float] | None]:
    keys = []
    for p in (edge.get("start"), edge.get("end")):
        xyz = _as_xyz(p)
        if xyz:
            keys.append(point_key(xyz))
    if len(keys) == 2:
        return keys[0], keys[1]
    if len(keys) == 1:
        return keys[0], keys[0]
    return None, None


def _edge_strategy(edge: Dict[str, Any]) -> str | None:
    return None


def _edge_length(edge: Dict[str, Any]) -> float:
    try:
        return float(edge.get("length", 0.0))
    except Exception:
        return 0.0


def order_sequence_edges(contact_edges: Dict[str, Any], edge_ids: List[str]) -> Tuple[List[str], bool]:
    if not edge_ids:
        return [], False

    edge_set = set(edge_ids)
    edges_by_key: Dict[Tuple[float, float, float], List[str]] = {}
    edge_end_keys: Dict[str, Tuple[Tuple[float, float, float] | None, Tuple[float, float, float] | None]] = {}

    for edge_id in edge_ids:
        edge = contact_edges.get(edge_id, {})
        if not isinstance(edge, dict):
            continue
        k1, k2 = _edge_keys(edge)
        edge_end_keys[edge_id] = (k1, k2)
        if k1 is not None:
            edges_by_key.setdefault(k1, []).append(edge_id)
        if k2 is not None and k2 != k1:
            edges_by_key.setdefault(k2, []).append(edge_id)

    degrees: Dict[Tuple[float, float, float], int] = {}
    for k, eids in edges_by_key.items():
        deg = 0
        for eid in eids:
            if eid in edge_set:
                deg += 1
        degrees[k] = deg

    end_keys = [k for k, d in degrees.items() if d == 1]
    is_closed = len(end_keys) == 0

    def traverse(start_edge: str, start_key: Tuple[float, float, float] | None) -> List[str]:
        order: List[str] = []
        visited = set()
        current_edge = start_edge
        current_key = start_key
        while True:
            if current_edge in visited:
                break
            order.append(current_edge)
            visited.add(current_edge)
            k1, k2 = edge_end_keys.get(current_edge, (None, None))
            next_key = k2 if k1 == current_key else k1
            if next_key is None:
                break
            candidates = [e for e in edges_by_key.get(next_key, []) if e in edge_set and e not in visited]
            if not candidates:
                break
            candidates = sorted(candidates, key=_sort_key)
            current_edge = candidates[0]
            current_key = next_key
        remaining = [e for e in edge_ids if e not in visited]
        if remaining:
            order.extend(sorted(remaining, key=_sort_key))
        return order

    if not is_closed and end_keys:
        start_key = sorted(end_keys, key=lambda k: (k[0], k[1], k[2]))[0]
        start_edges = [e for e in edges_by_key.get(start_key, []) if e in edge_set]
        start_edges = sorted(start_edges, key=_sort_key)
        if start_edges:
            return traverse(start_edges[0], start_key), False

    start_edge = sorted(edge_ids, key=lambda e: (_edge_length(contact_edges.get(e, {})), _sort_key(e)))[0]

    k1, k2 = edge_end_keys.get(start_edge, (None, None))
    order1 = traverse(start_edge, k1)
    order2 = traverse(start_edge, k2)

    def score(order: List[str]) -> int:
        if not order:
            return 0
        return 0

    if score(order2) > score(order1):
        return order2, True
    return order1, True


def split_sequence_by_adjacent_corner(order: List[str], contact_edges: Dict[str, Any]) -> List[List[str]]:
    return [order] if order else []


def split_sequence_by_adjacent_t(order: List[str]) -> List[List[str]]:
    if len(order) < 2:
        return [order]
    segments: List[List[str]] = []
    start = 0
    for i in range(len(order) - 1):
        if "T" in str(order[i]) and "T" in str(order[i + 1]):
            segments.append(order[start:i + 1])
            start = i + 1
    segments.append(order[start:])
    return [s for s in segments if s]


def _swap_edge_direction(edge: Dict[str, Any]) -> None:
    if not isinstance(edge, dict):
        return
    if "start" in edge and "end" in edge:
        edge["start"], edge["end"] = edge["end"], edge["start"]
    if "start_point" in edge and "end_point" in edge:
        edge["start_point"], edge["end_point"] = edge["end_point"], edge["start_point"]
    t = edge.get("tangent")
    if isinstance(t, list) and len(t) == 3:
        edge["tangent"] = [-float(t[0]), -float(t[1]), -float(t[2])]
    if "angle" in edge and edge.get("angle") is not None:
        try:
            edge["angle"] = -float(edge["angle"])
        except Exception:
            pass
    pts = edge.get("points")
    if isinstance(pts, list):
        edge["points"] = list(reversed(pts))
    samples = edge.get("samples")
    if isinstance(samples, list):
        edge["samples"] = list(reversed(samples))


def orient_sequence_edges(contact_edges: Dict[str, Any], edge_ids: List[str]) -> None:
    if len(edge_ids) < 2:
        return
    first = edge_ids[0]
    second = edge_ids[1]
    f_edge = contact_edges.get(first, {})
    s_edge = contact_edges.get(second, {})
    f_start, f_end = _edge_keys(f_edge)
    s_start, s_end = _edge_keys(s_edge)
    if f_end is None or s_start is None:
        return
    if f_end == s_start or f_end == s_end:
        pass
    elif f_start == s_start or f_start == s_end:
        _swap_edge_direction(f_edge)

    for i in range(len(edge_ids) - 1):
        curr = edge_ids[i]
        nxt = edge_ids[i + 1]
        curr_edge = contact_edges.get(curr, {})
        next_edge = contact_edges.get(nxt, {})
        curr_start, curr_end = _edge_keys(curr_edge)
        next_start, next_end = _edge_keys(next_edge)
        if curr_end is None or next_start is None:
            continue
        if curr_end == next_start:
            continue
        if curr_end == next_end:
            _swap_edge_direction(next_edge)


def _sequence_start_end(contact_edges: Dict[str, Any], seq: List[str]):
    if not seq:
        return None, None, None
    first_edge = contact_edges.get(seq[0], {})
    last_edge = contact_edges.get(seq[-1], {})
    start_pt = _as_xyz(first_edge.get("start")) if isinstance(first_edge, dict) else None
    end_pt = _as_xyz(last_edge.get("end")) if isinstance(last_edge, dict) else None
    if start_pt and end_pt:
        normal = [end_pt[0] - start_pt[0], end_pt[1] - start_pt[1], end_pt[2] - start_pt[2]]
        return start_pt, end_pt, normal
    return start_pt, end_pt, None


def build_weld_sequences(contact_edges: Dict[str, Any]) -> List[List[str]]:
    point_to_edges: Dict[Tuple[float, float, float], List[str]] = {}
    for edge_id, edge in contact_edges.items():
        if not isinstance(edge, dict):
            continue
        for p in (edge.get("start"), edge.get("end")):
            xyz = _as_xyz(p)
            if not xyz:
                continue
            k = point_key(xyz)
            point_to_edges.setdefault(k, []).append(str(edge_id))

    unvisited = set(str(k) for k in contact_edges.keys())
    sequences: List[List[str]] = []

    while unvisited:
        start = min(unvisited, key=_sort_key)
        stack = [start]
        unvisited.remove(start)
        component: List[str] = []
        while stack:
            edge_id = stack.pop()
            component.append(edge_id)
            edge = contact_edges.get(edge_id, {})
            if not isinstance(edge, dict):
                continue
            keys = []
            for p in (edge.get("start"), edge.get("end")):
                xyz = _as_xyz(p)
                if xyz:
                    keys.append(point_key(xyz))
            for k in keys:
                for neighbor in point_to_edges.get(k, []):
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        stack.append(neighbor)
        sequences.append(sorted(component, key=_sort_key))

    return sequences

def build_welds_sequence_dict(sequences: List[List[str]], contact_edges: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for i, seq in enumerate(sequences, start=1):
        key = f"welds_sequence_{i}"
        result[key] = {}
        for j, edge_id in enumerate(seq, start=1):
            result[key][f"edge{j}"] = str(edge_id)
        start_pt, end_pt, normal = _sequence_start_end(contact_edges, seq)
        result[key]["start_point"] = start_pt
        result[key]["end_point"] = end_pt
        result[key]["normal"] = normal
        solid_ids = None
        if seq:
            first_edge = contact_edges.get(seq[0], {})
            if isinstance(first_edge, dict):
                solid_ids = first_edge.get("solid_ids")
        result[key]["solid_ids"] = solid_ids
    return result

def visualize_sequences(contact_edges: Dict[str, Any], sequences: List[List[str]], title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    cmap = cm.get_cmap("hsv", max(1, len(sequences)))
    xs_all: List[float] = []
    ys_all: List[float] = []
    zs_all: List[float] = []
    start_points: List[List[float]] = []
    end_points: List[List[float]] = []

    for i, seq in enumerate(sequences):
        color = cmap(i)
        group_points: List[List[float]] = []
        for edge_id in seq:
            edge = contact_edges.get(edge_id, {})
            if not isinstance(edge, dict):
                continue
            samples = _edge_samples(edge)
            if len(samples) < 2:
                continue
            xs = [p[0] for p in samples]
            ys = [p[1] for p in samples]
            zs = [p[2] for p in samples]
            ax.plot(xs, ys, zs, linewidth=2.0, color=color)
            xs_all.extend(xs)
            ys_all.extend(ys)
            zs_all.extend(zs)
            group_points.extend(samples)
        if group_points:
            cx = sum(p[0] for p in group_points) / len(group_points)
            cy = sum(p[1] for p in group_points) / len(group_points)
            cz = sum(p[2] for p in group_points) / len(group_points)
            ax.text(cx, cy, cz, f"{i + 1}", fontsize=10, color="black")
        sp, ep, _ = _sequence_start_end(contact_edges, seq)
        if sp:
            start_points.append(sp)
        if ep:
            end_points.append(ep)

    if start_points:
        xs = [p[0] for p in start_points]
        ys = [p[1] for p in start_points]
        zs = [p[2] for p in start_points]
        ax.scatter(xs, ys, zs, c="red", marker="o", s=30)
    if end_points:
        xs = [p[0] for p in end_points]
        ys = [p[1] for p in end_points]
        zs = [p[2] for p in end_points]
        ax.scatter(xs, ys, zs, c="green", marker="^", s=30)

    if xs_all and ys_all and zs_all:
        x_min, x_max = min(xs_all), max(xs_all)
        y_min, y_max = min(ys_all), max(ys_all)
        z_min, z_max = min(zs_all), max(zs_all)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        try:
            ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
        except Exception:
            pass

    plt.show()


def _sequence_has_t(seq: List[str]) -> bool:
    for eid in seq:
        if "T" in str(eid):
            return True
    return False


def _solid_ids_key(solid_ids: Any) -> Tuple[Any, ...] | None:
    if isinstance(solid_ids, list) and solid_ids:
        return tuple(solid_ids)
    return None


def build_welds_trajectory(
    sequences: List[List[str]],
    welds_sequence: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    seq_entries = []
    seq_map: Dict[str, List[str]] = {}
    for i, seq in enumerate(sequences, start=1):
        key = f"welds_sequence_{i}"
        solid_ids = None
        if isinstance(welds_sequence.get(key), dict):
            solid_ids = welds_sequence[key].get("solid_ids")
        seq_id = str(i)
        seq_entries.append({"id": seq_id, "edges": seq, "solid_ids": solid_ids})
        seq_map[seq_id] = seq

    assigned = set()
    trajectories: List[Dict[str, Any]] = []

    groups: Dict[Tuple[Any, ...] | None, List[Dict[str, Any]]] = {}
    for e in seq_entries:
        if _sequence_has_t(e["edges"]):
            groups.setdefault(_solid_ids_key(e["solid_ids"]), []).append(e)

    for solid_key, entries in groups.items():
        seq_ids = [e["id"] for e in entries]
        for e in entries:
            assigned.add(e["id"])
        for e in seq_entries:
            if e["id"] in assigned:
                continue
            if _solid_ids_key(e["solid_ids"]) == solid_key:
                seq_ids.append(e["id"])
                assigned.add(e["id"])
        trajectories.append({
            "sequence_ids": seq_ids,
            "initial_sequence_ids": list(seq_ids),
            "solid_ids": []
        })

    solid_counts: Dict[Any, int] = {}
    traj_solids: List[List[Any]] = []
    for traj in trajectories:
        solids: List[Any] = []
        for sid in traj["sequence_ids"]:
            entry = next((e for e in seq_entries if e["id"] == sid), None)
            if not entry:
                continue
            sids = entry.get("solid_ids")
            if isinstance(sids, list):
                for v in sids:
                    if v not in solids:
                        solids.append(v)
        traj_solids.append(solids)
        for v in solids:
            solid_counts[v] = solid_counts.get(v, 0) + 1

    repeated = {v for v, c in solid_counts.items() if c > 1}
    initial_seeds_per_traj: List[set] = []
    initial_seeds_all: set = set()
    for idx, traj in enumerate(trajectories):
        solids = traj_solids[idx] if idx < len(traj_solids) else []
        seeds = [v for v in solids if solid_counts.get(v, 0) == 1]
        traj["solid_ids"] = seeds
        seed_set = set(seeds)
        initial_seeds_per_traj.append(seed_set)
        initial_seeds_all.update(seed_set)

    changed = True
    while changed:
        changed = False
        for e in seq_entries:
            if e["id"] in assigned:
                continue
            sids = e.get("solid_ids")
            if not isinstance(sids, list) or not sids:
                continue
            for t_idx, traj in enumerate(trajectories):
                if not traj["solid_ids"]:
                    continue
                if any(v in traj["solid_ids"] for v in sids):
                    traj["sequence_ids"].append(e["id"])
                    assigned.add(e["id"])
                    forbidden = initial_seeds_all - initial_seeds_per_traj[t_idx]
                    for v in sids:
                        if v not in traj["solid_ids"] and v not in repeated and v not in forbidden:
                            traj["solid_ids"].append(v)
                    changed = True
                    break

    for e in seq_entries:
        if e["id"] in assigned:
            continue
        sids = e.get("solid_ids")
        solid_list = sids if isinstance(sids, list) else []
        trajectories.append({
            "sequence_ids": [e["id"]],
            "initial_sequence_ids": [e["id"]],
            "solid_ids": list(solid_list)
        })
        assigned.add(e["id"])

    return trajectories, seq_map


def _normalize_vec(v: List[float]) -> Tuple[float, float, float] | None:
    if not isinstance(v, list) or len(v) != 3:
        return None
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    n = math.sqrt(x * x + y * y + z * z)
    if n <= 1e-12:
        return None
    return (x / n, y / n, z / n)


def _angle_deg(a: List[float], b: List[float]) -> float | None:
    ua = _normalize_vec(a)
    ub = _normalize_vec(b)
    if ua is None or ub is None:
        return None
    dot = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2]))
    return math.degrees(math.acos(dot))


def sort_trajectories(
    trajectories: List[Dict[str, Any]],
    welds_sequence: Dict[str, Dict[str, Any]],
    up_axis: List[float],
    angle_eps_deg: float = 15.0
) -> None:
    for traj in trajectories:
        seq_ids = list(traj.get("sequence_ids", []))
        initial_ids = set(traj.get("initial_sequence_ids", []))
        cat_map: Dict[str, int] = {}
        front: List[str] = []
        tail: List[str] = []
        middle: List[str] = []
        for sid in seq_ids:
            wkey = f"welds_sequence_{sid}"
            normal = None
            if isinstance(welds_sequence.get(wkey), dict):
                normal = welds_sequence[wkey].get("normal")
            angle = _angle_deg(normal, up_axis) if normal is not None else None
            is_parallel = False
            if angle is not None:
                if angle <= angle_eps_deg or abs(180.0 - angle) <= angle_eps_deg:
                    is_parallel = True
            if is_parallel:
                front.append(sid)
                cat_map[str(sid)] = 1
            elif str(sid) in initial_ids:
                tail.append(sid)
                cat_map[str(sid)] = 2
            else:
                middle.append(sid)
                cat_map[str(sid)] = 3
        traj["sequence_ids"] = front + middle + tail
        traj["sequence_category"] = cat_map


def _collect_sequence_points(seq_ids: List[str], welds_sequence: Dict[str, Dict[str, Any]]) -> List[List[float]]:
    pts: List[List[float]] = []
    for sid in seq_ids:
        wkey = f"welds_sequence_{sid}"
        w = welds_sequence.get(wkey, {}) if isinstance(welds_sequence.get(wkey), dict) else {}
        sp = w.get("start_point")
        ep = w.get("end_point")
        if isinstance(sp, list) and len(sp) == 3:
            pts.append([float(sp[0]), float(sp[1]), float(sp[2])])
        if isinstance(ep, list) and len(ep) == 3:
            pts.append([float(ep[0]), float(ep[1]), float(ep[2])])
    return pts


def sort_trajectory_list_by_plane(
    trajectories: List[Dict[str, Any]],
    welds_sequence: Dict[str, Dict[str, Any]],
    up_axis: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    all_pts: List[List[float]] = []
    for traj in trajectories:
        cat_map = traj.get("sequence_category", {}) or {}
        label1_ids = [sid for sid in traj.get("sequence_ids", []) if cat_map.get(str(sid)) == 1]
        all_pts.extend(_collect_sequence_points(label1_ids, welds_sequence))
    if len(all_pts) < 3:
        return None

    pts = np.asarray(all_pts, dtype=float)
    centroid = np.mean(pts, axis=0)
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    up = np.asarray(up_axis, dtype=float)
    if float(np.dot(normal, up)) < 0.0:
        normal = -normal

    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(normal, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(normal, ref)
    un = float(np.linalg.norm(u))
    if un < 1e-12:
        return None
    u = u / un
    v = np.cross(normal, u)

    for traj in trajectories:
        cat_map = traj.get("sequence_category", {}) or {}
        label1_ids = [sid for sid in traj.get("sequence_ids", []) if cat_map.get(str(sid)) == 1]
        pts_traj = _collect_sequence_points(label1_ids, welds_sequence)
        if not pts_traj:
            pts_traj = _collect_sequence_points(list(traj.get("sequence_ids", [])), welds_sequence)
        if not pts_traj:
            traj["_angle"] = 0.0
            continue
        p = np.mean(np.asarray(pts_traj, dtype=float), axis=0) - centroid
        x = float(np.dot(p, u))
        y = float(np.dot(p, v))
        traj["_angle"] = math.atan2(y, x)

    trajectories.sort(key=lambda t: t.get("_angle", 0.0), reverse=True)
    for traj in trajectories:
        if "_angle" in traj:
            del traj["_angle"]

    return u, v, centroid


def _sequence_center_for_sort(
    seq_id: str,
    welds_sequence: Dict[str, Dict[str, Any]],
    seq_map: Dict[str, List[str]],
    contact_edges: Dict[str, Any]
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    wkey = f"welds_sequence_{seq_id}"
    w = welds_sequence.get(wkey, {}) if isinstance(welds_sequence.get(wkey), dict) else {}
    sp = w.get("start_point")
    ep = w.get("end_point")
    main = None
    if isinstance(sp, list) and isinstance(ep, list) and len(sp) == 3 and len(ep) == 3:
        main = np.array(
            [(float(sp[0]) + float(ep[0])) * 0.5,
             (float(sp[1]) + float(ep[1])) * 0.5,
             (float(sp[2]) + float(ep[2])) * 0.5],
            dtype=float
        )
    tie = None
    seq = seq_map.get(str(seq_id), [])
    if seq:
        edge = contact_edges.get(seq[0], {})
        if isinstance(edge, dict):
            es = edge.get("start")
            ee = edge.get("end")
            if isinstance(es, list) and isinstance(ee, list) and len(es) == 3 and len(ee) == 3:
                tie = np.array(
                    [(float(es[0]) + float(ee[0])) * 0.5,
                     (float(es[1]) + float(ee[1])) * 0.5,
                     (float(es[2]) + float(ee[2])) * 0.5],
                    dtype=float
                )
    return main, tie


def sort_sequences_within_trajectory(
    trajectories: List[Dict[str, Any]],
    welds_sequence: Dict[str, Dict[str, Any]],
    seq_map: Dict[str, List[str]],
    contact_edges: Dict[str, Any],
    basis: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    u, v, centroid = basis
    for traj in trajectories:
        cat_map = traj.get("sequence_category", {}) or {}
        ordered: List[str] = []
        for label in [1, 3, 2]:
            ids = [str(sid) for sid in traj.get("sequence_ids", []) if cat_map.get(str(sid), 3) == label]
            if not ids:
                continue
            def key_fn(sid: str):
                main, tie = _sequence_center_for_sort(sid, welds_sequence, seq_map, contact_edges)
                if main is None:
                    main = np.zeros(3, dtype=float)
                p = main - centroid
                ang = math.atan2(float(np.dot(p, v)), float(np.dot(p, u)))
                if tie is None:
                    tang = ang
                else:
                    tp = tie - centroid
                    tang = math.atan2(float(np.dot(tp, v)), float(np.dot(tp, u)))
                return (ang, tang)
            ids.sort(key=key_fn, reverse=True)
            ordered.extend(ids)
        traj["sequence_ids"] = ordered


def _rebuild_welds_sequence_entry(
    welds_sequence: Dict[str, Dict[str, Any]],
    seq_id: str,
    seq: List[str],
    contact_edges: Dict[str, Any]
) -> None:
    key = f"welds_sequence_{seq_id}"
    entry = welds_sequence.get(key)
    if not isinstance(entry, dict):
        return
    solid_ids = entry.get("solid_ids")
    start_pt, end_pt, normal = _sequence_start_end(contact_edges, seq)
    new_entry: Dict[str, Any] = {}
    for j, edge_id in enumerate(seq, start=1):
        new_entry[f"edge{j}"] = str(edge_id)
    new_entry["start_point"] = start_pt
    new_entry["end_point"] = end_pt
    new_entry["normal"] = normal
    new_entry["solid_ids"] = solid_ids
    welds_sequence[key] = new_entry


def align_label1_sequences(
    trajectories: List[Dict[str, Any]],
    welds_sequence: Dict[str, Dict[str, Any]],
    seq_map: Dict[str, List[str]],
    contact_edges: Dict[str, Any],
    processed_sequences: List[List[str]],
    up_axis: List[float]
) -> None:
    ua = _normalize_vec(up_axis)
    if ua is None:
        return
    for traj in trajectories:
        cat_map = traj.get("sequence_category", {}) or {}
        for sid in traj.get("sequence_ids", []):
            if cat_map.get(str(sid)) != 1:
                continue
            wkey = f"welds_sequence_{sid}"
            w = welds_sequence.get(wkey)
            if not isinstance(w, dict):
                continue
            normal = w.get("normal")
            un = _normalize_vec(normal) if isinstance(normal, list) else None
            if un is None:
                continue
            dot = un[0] * ua[0] + un[1] * ua[1] + un[2] * ua[2]
            if dot >= 0.0:
                continue
            seq = seq_map.get(str(sid), [])
            if not seq:
                continue
            seq = list(reversed(seq))
            for eid in seq:
                edge = contact_edges.get(eid)
                if isinstance(edge, dict):
                    _swap_edge_direction(edge)
            seq_map[str(sid)] = seq
            try:
                idx = int(sid) - 1
                if 0 <= idx < len(processed_sequences):
                    processed_sequences[idx] = seq
            except Exception:
                pass
            _rebuild_welds_sequence_entry(welds_sequence, str(sid), seq, contact_edges)


def _shade_color(color: Tuple[float, float, float, float] | Tuple[float, float, float], lighten: float):
    if len(color) == 4:
        r, g, b, a = color
    else:
        r, g, b = color
        a = 1.0
    r = r + (1.0 - r) * lighten
    g = g + (1.0 - g) * lighten
    b = b + (1.0 - b) * lighten
    return (r, g, b, a)


def visualize_trajectories(
    contact_edges: Dict[str, Any],
    seq_map: Dict[str, List[str]],
    trajectories: List[Dict[str, Any]],
    title: str
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    cmap = cm.get_cmap("tab20", max(1, len(trajectories)))
    xs_all: List[float] = []
    ys_all: List[float] = []
    zs_all: List[float] = []

    for traj in trajectories:
        for sid in traj.get("sequence_ids", []):
            seq = seq_map.get(str(sid), [])
            for edge_id in seq:
                edge = contact_edges.get(edge_id, {})
                if not isinstance(edge, dict):
                    continue
                samples = _edge_samples(edge)
                if len(samples) < 2:
                    continue
                xs_all.extend([p[0] for p in samples])
                ys_all.extend([p[1] for p in samples])
                zs_all.extend([p[2] for p in samples])

    x_min = y_min = z_min = 0.0
    x_max = y_max = z_max = 1.0
    if xs_all and ys_all and zs_all:
        x_min, x_max = min(xs_all), max(xs_all)
        y_min, y_max = min(ys_all), max(ys_all)
        z_min, z_max = min(zs_all), max(zs_all)

    def setup_axes():
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        try:
            ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
        except Exception:
            pass

    plt.ion()
    setup_axes()
    for i, traj in enumerate(trajectories, start=1):
        base_color = cmap(i - 1)
        cat_map = traj.get("sequence_category", {}) or {}
        for s_idx, sid in enumerate(traj.get("sequence_ids", []), start=1):
            label = int(cat_map.get(str(sid), 3))
            if label == 1:
                color = _shade_color(base_color, 0.0)
            elif label == 3:
                color = _shade_color(base_color, 0.35)
            else:
                color = _shade_color(base_color, 0.7)

            seq = seq_map.get(str(sid), [])
            group_points: List[List[float]] = []
            for edge_id in seq:
                edge = contact_edges.get(edge_id, {})
                if not isinstance(edge, dict):
                    continue
                samples = _edge_samples(edge)
                if len(samples) < 2:
                    continue
                xs = [p[0] for p in samples]
                ys = [p[1] for p in samples]
                zs = [p[2] for p in samples]
                ax.plot(xs, ys, zs, linewidth=2.0, color=color)
                group_points.extend(samples)
            if group_points:
                cx = sum(p[0] for p in group_points) / len(group_points)
                cy = sum(p[1] for p in group_points) / len(group_points)
                cz = sum(p[2] for p in group_points) / len(group_points)
                ax.text(cx, cy, cz, f"{i}-{s_idx}", fontsize=9, color="black")
            plt.pause(1.0)

    plt.ioff()
    plt.show()


def main() -> None:
    obj = load_json(INPUT_JSON)
    contact_edges = obj.get("contact_edges", {}) or {}
    raw_sequences = build_weld_sequences(contact_edges)
    processed_sequences: List[List[str]] = []
    for seq in raw_sequences:
        ordered, _ = order_sequence_edges(contact_edges, seq)
        t_split = split_sequence_by_adjacent_t(ordered)
        for t in t_split:
            orient_sequence_edges(contact_edges, t)
            processed_sequences.append(t)
    obj["welds_sequence"] = build_welds_sequence_dict(processed_sequences, contact_edges)
    trajectories, seq_map = build_welds_trajectory(
        processed_sequences,
        obj["welds_sequence"]
    )
    up_axis = obj.get("up_axis")
    if not isinstance(up_axis, list) or len(up_axis) != 3:
        up_axis = [0.0, 0.0, 1.0]
    sort_trajectories(trajectories, obj["welds_sequence"], up_axis)
    basis = sort_trajectory_list_by_plane(trajectories, obj["welds_sequence"], up_axis)
    if basis is not None:
        sort_sequences_within_trajectory(
            trajectories,
            obj["welds_sequence"],
            seq_map,
            contact_edges,
            basis
        )
    align_label1_sequences(
        trajectories,
        obj["welds_sequence"],
        seq_map,
        contact_edges,
        processed_sequences,
        up_axis
    )

    welds_trajectory: Dict[str, Dict[str, Any]] = {}
    for i, traj in enumerate(trajectories, start=1):
        key = f"welds_trajectory_{i}"
        welds_trajectory[key] = {}
        cat_map = traj.get("sequence_category", {}) or {}
        for j, sid in enumerate(traj.get("sequence_ids", []), start=1):
            label = cat_map.get(str(sid), 3)
            welds_trajectory[key][f"sequence{j}"] = {
                "id": str(sid),
                "label": label
            }
        welds_trajectory[key]["solid_ids"] = traj.get("solid_ids")

    obj["welds_trajectory"] = welds_trajectory
    save_json(OUTPUT_JSON, obj)
    visualize_sequences(contact_edges, processed_sequences, title=OUTPUT_JSON)
    visualize_trajectories(contact_edges, seq_map, trajectories, title=f"{OUTPUT_JSON} | trajectory")

if __name__ == "__main__":
    main()
