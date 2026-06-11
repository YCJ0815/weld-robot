import argparse
import math
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import cadquery as cq

MODULE_DIR = Path(__file__).resolve().parent
module_dir_str = str(MODULE_DIR)
if module_dir_str not in sys.path:
    sys.path.insert(0, module_dir_str)

from model_generation import (
    MODEL_GEN_PARAMS,
    _box_on_plate,
    _rib_on_plate_with_end_corner_holes,
    _resolve_output_dir,
    _side_rib_on_plate_with_corner_hole,
)


MAIN_RIB_CFG = MODEL_GEN_PARAMS["main_rib"]
SIDE_RIB_CFG = MODEL_GEN_PARAMS["side_rib"]


def _loosen_min(value, factor=0.8):
    return value * factor


def _loosen_max(value, factor=1.25):
    return value * factor


def _ratio_min(value, delta=0.15):
    return max(0.15, value - delta)


def _height_max_capped(value, factor=1.25, cap=95.0):
    return min(cap, value * factor)


def _length_ratio_max_capped(plate_dim, max_length_mm):
    return min(1.0, max_length_mm / plate_dim)


# 简单结构沿用 model_generation.py 的尺寸约束来源，
# 但适当放宽厚度/高度/长度比例，提升几何变化范围。
SIMPLE_STRUCTURE_PARAMS = {
    "plate": {
        **deepcopy(MODEL_GEN_PARAMS["plate"]),
        "edge_clearance_min": 50.0,
    },
    "single_rib": {
        "samples": 10,
        "orientation_choices": MAIN_RIB_CFG["orientation_choices"],
        "thickness_min": _loosen_min(MAIN_RIB_CFG["thickness_min"]),
        "thickness_max": _loosen_max(MAIN_RIB_CFG["thickness_max"]),
        "height_min": _loosen_min(MAIN_RIB_CFG["height_min"]),
        "height_max": _height_max_capped(MAIN_RIB_CFG["height_max"]),
        "length_ratio_min": _ratio_min(MAIN_RIB_CFG["length_ratio_min"], delta=0.45),
        "length_ratio_max": _length_ratio_max_capped(MODEL_GEN_PARAMS["plate"]["length"], 350.0),
        "placement_attempts": 1000,
        "rotation_angle_min": 0.0,
        "rotation_angle_max": 180.0,
    },
    "parallel_ribs": {
        "rib_counts": [1, 2, 3],
        "samples_per_count": 10,
        "orientation_choices": MAIN_RIB_CFG["orientation_choices"],
        "thickness_min": _loosen_min(MAIN_RIB_CFG["thickness_min"]),
        "thickness_max": _loosen_max(MAIN_RIB_CFG["thickness_max"]),
        "height_min": _loosen_min(MAIN_RIB_CFG["height_min"]),
        "height_max": _height_max_capped(MAIN_RIB_CFG["height_max"]),
        "length_ratio_min": _ratio_min(MAIN_RIB_CFG["length_ratio_min"], delta=0.45),
        "length_ratio_max": _length_ratio_max_capped(MODEL_GEN_PARAMS["plate"]["length"], 350.0),
        "min_gap": SIDE_RIB_CFG["min_spacing"] * 0.4,
        "placement_attempts": 1000,
        "rotation_angle_min": 0.0,
        "rotation_angle_max": 180.0,
    },
    "t_ribs": {
        "samples": 10,
        "orientation_choices": MAIN_RIB_CFG["orientation_choices"],
        "main_thickness_min": _loosen_min(MAIN_RIB_CFG["thickness_min"]),
        "main_thickness_max": _loosen_max(MAIN_RIB_CFG["thickness_max"]),
        "main_height_min": _loosen_min(MAIN_RIB_CFG["height_min"]),
        "main_height_max": _height_max_capped(MAIN_RIB_CFG["height_max"]),
        "main_length_ratio_min": _ratio_min(MAIN_RIB_CFG["length_ratio_min"], delta=0.35),
        "main_length_ratio_max": _length_ratio_max_capped(MODEL_GEN_PARAMS["plate"]["length"], 350.0),
        "side_thickness_min": _loosen_min(SIDE_RIB_CFG["thickness_min"]),
        "side_thickness_max": _loosen_max(SIDE_RIB_CFG["thickness_max"]),
        "side_height_min": _loosen_min(SIDE_RIB_CFG["height_min"]),
        "side_length_ratio_min": 0.18,
        "side_length_ratio_max": 0.82,
        "corner_hole_radius": SIDE_RIB_CFG["corner_hole_radius"],
        "placement_attempts": 1000,
        "rotation_angle_min": 0.0,
        "rotation_angle_max": 180.0,
    },
    "sandwich_ribs": {
        "samples": 10,
        "orientation_choices": MAIN_RIB_CFG["orientation_choices"],
        "main_thickness_min": _loosen_min(MAIN_RIB_CFG["thickness_min"]),
        "main_thickness_max": _loosen_max(MAIN_RIB_CFG["thickness_max"]),
        "main_height_min": _loosen_min(MAIN_RIB_CFG["height_min"]),
        "main_height_max": _height_max_capped(MAIN_RIB_CFG["height_max"]),
        "main_length_ratio_min": _ratio_min(MAIN_RIB_CFG["length_ratio_min"], delta=0.35),
        "main_length_ratio_max": _length_ratio_max_capped(MODEL_GEN_PARAMS["plate"]["length"], 350.0),
        "middle_thickness_min": _loosen_min(SIDE_RIB_CFG["thickness_min"]),
        "middle_thickness_max": _loosen_max(SIDE_RIB_CFG["thickness_max"]),
        "middle_height_min": _loosen_min(SIDE_RIB_CFG["height_min"]),
        "middle_length_ratio_min": 0.18,
        "middle_length_ratio_max": 0.72,
        "corner_hole_radius": SIDE_RIB_CFG["corner_hole_radius"],
        "placement_attempts": 1000,
        "rotation_angle_min": 0.0,
        "rotation_angle_max": 180.0,
    },
    "grid_ribs": {
        "samples": 10,
        "orientation_choices": MAIN_RIB_CFG["orientation_choices"],
        "main_thickness_min": _loosen_min(MAIN_RIB_CFG["thickness_min"]),
        "main_thickness_max": _loosen_max(MAIN_RIB_CFG["thickness_max"]),
        "main_height_min": _loosen_min(MAIN_RIB_CFG["height_min"]),
        "main_height_max": _height_max_capped(MAIN_RIB_CFG["height_max"]),
        "main_length_ratio_min": _ratio_min(MAIN_RIB_CFG["length_ratio_min"], delta=0.35),
        "main_length_ratio_max": _length_ratio_max_capped(MODEL_GEN_PARAMS["plate"]["length"], 350.0),
        "middle_thickness_min": _loosen_min(SIDE_RIB_CFG["thickness_min"]),
        "middle_thickness_max": _loosen_max(SIDE_RIB_CFG["thickness_max"]),
        "middle_height_min": _loosen_min(SIDE_RIB_CFG["height_min"]),
        "middle_length_ratio_min": 0.18,
        "middle_length_ratio_max": 0.72,
        "middle_min_gap": SIDE_RIB_CFG["min_spacing"] * 0.4,
        "corner_hole_radius": SIDE_RIB_CFG["corner_hole_radius"],
        "placement_attempts": 1000,
        "rotation_angle_min": 0.0,
        "rotation_angle_max": 180.0,
    },
    "batch": {
        "output_dir": "data/generated_jobs/simple_jobs",
    },
}


@dataclass
class RibSpec:
    width: float
    depth: float
    height: float
    center_x: float
    center_y: float
    hole_type: str = "none"
    hole_radius: float = 0.0
    hole_center_x: float | None = None
    hole_center_y: float | None = None
    hole_centers: list[tuple[float, float]] = field(default_factory=list)
    cutter_axis: str | None = None

    @property
    def footprint(self):
        return self.width, self.depth, self.center_x, self.center_y


def _make_plate_parts(plate_cfg):
    plate_l = plate_cfg["length"]
    plate_w = plate_cfg["width"]
    plate_h = plate_cfg["height"]
    return [cq.Workplane("XY").box(plate_l, plate_w, plate_h)], plate_h / 2


def _axis_dims(plate_l, plate_w, orientation):
    if orientation == "X":
        return plate_l, plate_w
    return plate_w, plate_l


def _parallel_rib(length, thickness, height, center_axis, center_perp, orientation):
    if orientation == "X":
        return RibSpec(length, thickness, height, center_axis, center_perp)
    return RibSpec(thickness, length, height, center_perp, center_axis)


def _cross_rib(length, thickness, height, center_axis, center_perp, orientation):
    if orientation == "X":
        return RibSpec(thickness, length, height, center_axis, center_perp)
    return RibSpec(length, thickness, height, center_perp, center_axis)


def _rotate_xy(x, y, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def _rotated_rect_corners(width, depth, center_x, center_y, angle_deg):
    corners = []
    for local_x in (-width / 2, width / 2):
        for local_y in (-depth / 2, depth / 2):
            rot_x, rot_y = _rotate_xy(local_x, local_y, angle_deg)
            corners.append((center_x + rot_x, center_y + rot_y))
    return corners


def _ribs_fit_after_rotation(ribs, plate_l, plate_w, angle_deg, edge_clearance_min=0.0, tol=1e-7):
    x_min = -plate_l / 2 + edge_clearance_min
    x_max = plate_l / 2 - edge_clearance_min
    y_min = -plate_w / 2 + edge_clearance_min
    y_max = plate_w / 2 - edge_clearance_min
    for rib in ribs:
        rot_center_x, rot_center_y = _rotate_xy(rib.center_x, rib.center_y, angle_deg)
        for corner_x, corner_y in _rotated_rect_corners(
            rib.width,
            rib.depth,
            rot_center_x,
            rot_center_y,
            angle_deg,
        ):
            if not (x_min - tol <= corner_x <= x_max + tol):
                return False
            if not (y_min - tol <= corner_y <= y_max + tol):
                return False
    return True


def _rotate_ribs(parts, angle_deg):
    if abs(angle_deg) < 1e-9:
        return
    for index in range(1, len(parts)):
        parts[index] = parts[index].rotate((0, 0, 0), (0, 0, 1), angle_deg)


def _add_rib(parts, rib, plate_top_z):
    if rib.hole_type == "corner":
        _side_rib_on_plate_with_corner_hole(
            parts,
            rib.width,
            rib.depth,
            rib.height,
            rib.center_x,
            rib.center_y,
            plate_top_z,
            hole_radius=rib.hole_radius,
            hole_center_x=rib.hole_center_x,
            hole_center_y=rib.hole_center_y,
            cutter_axis=rib.cutter_axis,
        )
        return

    if rib.hole_type == "end":
        _rib_on_plate_with_end_corner_holes(
            parts,
            rib.width,
            rib.depth,
            rib.height,
            rib.center_x,
            rib.center_y,
            plate_top_z,
            hole_radius=rib.hole_radius,
            hole_centers=rib.hole_centers,
            cutter_axis=rib.cutter_axis,
        )
        return

    _box_on_plate(parts, rib.width, rib.depth, rib.height, rib.center_x, rib.center_y, plate_top_z)


def _assert_no_solid_intersections(parts, tol=1e-7):
    solids = [part.val() for part in parts]
    for i, solid_i in enumerate(solids):
        for j in range(i + 1, len(solids)):
            common = solid_i.intersect(solids[j])
            if common.Volume() > tol:
                raise ValueError(
                    f"Generated entities {i + 1} and {j + 1} intersect with positive volume; "
                    "only contact is allowed."
                )


def _export_parts(parts, step_filename, stl_filename):
    _assert_no_solid_intersections(parts)
    compound = cq.Compound.makeCompound([part.val() for part in parts])
    cq.exporters.export(compound, step_filename)
    cq.exporters.export(compound, stl_filename)


def _sample_value(rng, cfg, prefix, name):
    return rng.uniform(cfg[f"{prefix}_{name}_min"], cfg[f"{prefix}_{name}_max"])


def _sample_length_and_center(rng, axis_dim, ratio_min, ratio_max):
    length = rng.uniform(axis_dim * ratio_min, axis_dim * ratio_max)
    center = rng.uniform(-axis_dim / 2 + length / 2, axis_dim / 2 - length / 2)
    return length, center


def _sample_rotation(rng, cfg):
    return rng.uniform(cfg["rotation_angle_min"], cfg["rotation_angle_max"])


def _sample_non_overlapping_centers(rng, count, plate_dim, thicknesses, min_gap, attempts):
    intervals = []
    centers = []
    for thickness in thicknesses:
        low = -plate_dim / 2 + thickness / 2
        high = plate_dim / 2 - thickness / 2
        for _ in range(attempts):
            center = rng.uniform(low, high)
            interval = (center - thickness / 2, center + thickness / 2)
            if all(interval[1] + min_gap <= old[0] or old[1] + min_gap <= interval[0] for old in intervals):
                intervals.append(interval)
                centers.append(center)
                break
        else:
            return None
    return centers


def _sample_non_overlapping_centers_in_range(rng, count, low, high, thicknesses, min_gap, attempts):
    intervals = []
    centers = []
    for thickness in thicknesses:
        center_low = low + thickness / 2
        center_high = high - thickness / 2
        if center_low > center_high:
            return None

        for _ in range(attempts):
            center = rng.uniform(center_low, center_high)
            interval = (center - thickness / 2, center + thickness / 2)
            if all(interval[1] + min_gap <= old[0] or old[1] + min_gap <= interval[0] for old in intervals):
                intervals.append(interval)
                centers.append(center)
                break
        else:
            return None
    return centers


def _sample_layout(rng, cfg, plate_l, plate_w, edge_clearance_min, build_ribs):
    safe_plate_l = plate_l - 2 * edge_clearance_min
    safe_plate_w = plate_w - 2 * edge_clearance_min
    if safe_plate_l <= 0 or safe_plate_w <= 0:
        raise ValueError(
            "Invalid plate clearance constraints: "
            f"plate=({plate_l}, {plate_w}), edge_clearance_min={edge_clearance_min}."
        )

    for _ in range(cfg["placement_attempts"]):
        orientation = rng.choice(cfg["orientation_choices"])
        angle_deg = _sample_rotation(rng, cfg)
        ribs = build_ribs(rng, cfg, orientation, *_axis_dims(safe_plate_l, safe_plate_w, orientation))
        if ribs and _ribs_fit_after_rotation(
            ribs,
            plate_l,
            plate_w,
            angle_deg,
            edge_clearance_min=edge_clearance_min,
        ):
            return orientation, angle_deg, ribs
    raise ValueError(
        "Unable to sample a valid layout within the plate boundary: "
        f"plate=({plate_l}, {plate_w}), safe=({safe_plate_l}, {safe_plate_w}), "
        f"edge_clearance_min={edge_clearance_min}."
    )


def _write_model(output_dir, base_name, ribs, angle_deg, plate_cfg):
    output_dir = _resolve_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    parts, plate_top_z = _make_plate_parts(plate_cfg)
    for rib in ribs:
        _add_rib(parts, rib, plate_top_z)
    _rotate_ribs(parts, angle_deg)

    step_filename = os.path.join(output_dir, f"{base_name}.step")
    stl_filename = os.path.join(output_dir, f"{base_name}.stl")
    _export_parts(parts, step_filename, stl_filename)
    print(f"Generated: {step_filename}")
    print(f"Generated: {stl_filename}")
    return step_filename, stl_filename


def _parallel_ribs_layout(rng, cfg, orientation, axis_dim, perp_dim, rib_count):
    thicknesses = [rng.uniform(cfg["thickness_min"], cfg["thickness_max"]) for _ in range(rib_count)]
    heights = [rng.uniform(cfg["height_min"], cfg["height_max"]) for _ in range(rib_count)]
    center_perps = _sample_non_overlapping_centers(
        rng,
        rib_count,
        perp_dim,
        thicknesses,
        cfg["min_gap"],
        cfg["placement_attempts"],
    )
    if center_perps is None:
        return None

    ribs = []
    for thickness, height, center_perp in zip(thicknesses, heights, center_perps):
        length, center_axis = _sample_length_and_center(
            rng,
            axis_dim,
            cfg["length_ratio_min"],
            cfg["length_ratio_max"],
        )
        ribs.append(_parallel_rib(length, thickness, height, center_axis, center_perp, orientation))
    return ribs


def _single_rib_layout(rng, cfg, orientation, axis_dim, perp_dim):
    thickness = rng.uniform(cfg["thickness_min"], cfg["thickness_max"])
    height = rng.uniform(cfg["height_min"], cfg["height_max"])
    length, center_axis = _sample_length_and_center(
        rng,
        axis_dim,
        cfg["length_ratio_min"],
        cfg["length_ratio_max"],
    )
    center_perp = rng.uniform(-perp_dim / 2 + thickness / 2, perp_dim / 2 - thickness / 2)
    return [_parallel_rib(length, thickness, height, center_axis, center_perp, orientation)]


def generate_single_rib_plate(index, output_dir=None, params=SIMPLE_STRUCTURE_PARAMS, rng=None):
    rng = rng or random
    plate_cfg = params["plate"]
    cfg = params["single_rib"]
    output_dir = output_dir or params["batch"]["output_dir"]
    plate_l = plate_cfg["length"]
    plate_w = plate_cfg["width"]
    edge_clearance_min = plate_cfg.get("edge_clearance_min", 0.0)

    orientation, angle_deg, ribs = _sample_layout(
        rng,
        cfg,
        plate_l,
        plate_w,
        edge_clearance_min,
        _single_rib_layout,
    )
    step_filename, stl_filename = _write_model(
        output_dir,
        f"simple_single_rib_{index:03d}_{orientation.lower()}_rot{angle_deg:.1f}",
        ribs,
        angle_deg,
        plate_cfg,
    )
    return {
        "structure_type": "single_rib",
        "rib_count": 1,
        "orientation": orientation,
        "rotation_angle_deg": angle_deg,
        "step": step_filename,
        "stl": stl_filename,
    }


def generate_single_rib_batch(samples=None, output_dir=None, params=SIMPLE_STRUCTURE_PARAMS, seed=None):
    cfg = params["single_rib"]
    samples = samples or cfg["samples"]
    output_dir = output_dir or params["batch"]["output_dir"]

    rng = random.Random(seed) if seed is not None else random
    return [
        generate_single_rib_plate(index, output_dir=output_dir, params=params, rng=rng)
        for index in range(samples)
    ]


def generate_parallel_rib_plate(index, rib_count, output_dir=None, params=SIMPLE_STRUCTURE_PARAMS, rng=None):
    rng = rng or random
    plate_cfg = params["plate"]
    cfg = params["parallel_ribs"]
    output_dir = output_dir or params["batch"]["output_dir"]
    plate_l = plate_cfg["length"]
    plate_w = plate_cfg["width"]
    edge_clearance_min = plate_cfg.get("edge_clearance_min", 0.0)

    orientation, angle_deg, ribs = _sample_layout(
        rng,
        cfg,
        plate_l,
        plate_w,
        edge_clearance_min,
        lambda *args: _parallel_ribs_layout(*args, rib_count=rib_count),
    )
    step_filename, stl_filename = _write_model(
        output_dir,
        f"simple_parallel_ribs_{rib_count}_{index:03d}_{orientation.lower()}_rot{angle_deg:.1f}",
        ribs,
        angle_deg,
        plate_cfg,
    )
    return {
        "structure_type": "parallel_ribs",
        "rib_count": rib_count,
        "orientation": orientation,
        "rotation_angle_deg": angle_deg,
        "step": step_filename,
        "stl": stl_filename,
    }


def generate_parallel_rib_batch(
    rib_counts=None,
    samples_per_count=None,
    output_dir=None,
    params=SIMPLE_STRUCTURE_PARAMS,
    seed=None,
):
    cfg = params["parallel_ribs"]
    rib_counts = rib_counts or cfg["rib_counts"]
    samples_per_count = samples_per_count or cfg["samples_per_count"]
    output_dir = output_dir or params["batch"]["output_dir"]

    rng = random.Random(seed) if seed is not None else random
    return [
        generate_parallel_rib_plate(index, rib_count, output_dir=output_dir, params=params, rng=rng)
        for rib_count in rib_counts
        for index in range(samples_per_count)
    ]


def _t_ribs_layout(rng, cfg, orientation, axis_dim, perp_dim):
    main_thickness = _sample_value(rng, cfg, "main", "thickness")
    side_thickness = _sample_value(rng, cfg, "side", "thickness")
    main_height = _sample_value(rng, cfg, "main", "height")
    side_height = rng.uniform(cfg["side_height_min"], main_height)
    main_len, main_center_axis = _sample_length_and_center(
        rng,
        axis_dim,
        cfg["main_length_ratio_min"],
        cfg["main_length_ratio_max"],
    )

    main_center_perp = rng.uniform(-perp_dim / 2 + main_thickness / 2, perp_dim / 2 - main_thickness / 2)
    side_sign = rng.choice([-1, 1])
    if side_sign == -1:
        available_side_len = main_center_perp - main_thickness / 2 + perp_dim / 2
    else:
        available_side_len = perp_dim / 2 - main_center_perp - main_thickness / 2

    min_side_len = perp_dim * cfg["side_length_ratio_min"]
    max_side_len = min(perp_dim * cfg["side_length_ratio_max"], available_side_len)
    if max_side_len < min_side_len:
        return None

    side_len = rng.uniform(min_side_len, max_side_len)
    intersection_axis = rng.uniform(
        main_center_axis - main_len / 2 + side_thickness / 2,
        main_center_axis + main_len / 2 - side_thickness / 2,
    )
    side_center_perp = main_center_perp + side_sign * (main_thickness / 2 + side_len / 2)
    hole_center_perp = main_center_perp + side_sign * (main_thickness / 2)

    main = _parallel_rib(main_len, main_thickness, main_height, main_center_axis, main_center_perp, orientation)
    side = _cross_rib(side_len, side_thickness, side_height, intersection_axis, side_center_perp, orientation)
    side.hole_type = "corner"
    side.hole_radius = cfg["corner_hole_radius"]
    side.cutter_axis = "X" if orientation == "X" else "Y"
    if orientation == "X":
        side.hole_center_x = intersection_axis
        side.hole_center_y = hole_center_perp
    else:
        side.hole_center_x = hole_center_perp
        side.hole_center_y = intersection_axis
    return [main, side]


def generate_t_rib_plate(index, output_dir=None, params=SIMPLE_STRUCTURE_PARAMS, rng=None):
    rng = rng or random
    plate_cfg = params["plate"]
    cfg = params["t_ribs"]
    output_dir = output_dir or params["batch"]["output_dir"]
    plate_l = plate_cfg["length"]
    plate_w = plate_cfg["width"]
    edge_clearance_min = plate_cfg.get("edge_clearance_min", 0.0)

    orientation, angle_deg, ribs = _sample_layout(rng, cfg, plate_l, plate_w, edge_clearance_min, _t_ribs_layout)
    step_filename, stl_filename = _write_model(
        output_dir,
        f"simple_t_ribs_{index:03d}_{orientation.lower()}_rot{angle_deg:.1f}",
        ribs,
        angle_deg,
        plate_cfg,
    )
    return {
        "structure_type": "t_ribs",
        "orientation": orientation,
        "rotation_angle_deg": angle_deg,
        "step": step_filename,
        "stl": stl_filename,
    }


def generate_t_rib_batch(samples=None, output_dir=None, params=SIMPLE_STRUCTURE_PARAMS, seed=None):
    cfg = params["t_ribs"]
    samples = samples or cfg["samples"]
    output_dir = output_dir or params["batch"]["output_dir"]

    rng = random.Random(seed) if seed is not None else random
    return [
        generate_t_rib_plate(index, output_dir=output_dir, params=params, rng=rng)
        for index in range(samples)
    ]


def _main_pair_with_middle_layout(rng, cfg, orientation, axis_dim, perp_dim):
    main_1 = {
        "length": None,
        "thickness": _sample_value(rng, cfg, "main", "thickness"),
        "height": _sample_value(rng, cfg, "main", "height"),
    }
    main_2 = {
        "length": None,
        "thickness": _sample_value(rng, cfg, "main", "thickness"),
        "height": _sample_value(rng, cfg, "main", "height"),
    }
    middle = {
        "thickness": _sample_value(rng, cfg, "middle", "thickness"),
        "height": rng.uniform(cfg["middle_height_min"], min(main_1["height"], main_2["height"])),
    }

    main_1["length"], main_1["center_axis"] = _sample_length_and_center(
        rng,
        axis_dim,
        cfg["main_length_ratio_min"],
        cfg["main_length_ratio_max"],
    )
    main_2["length"], main_2["center_axis"] = _sample_length_and_center(
        rng,
        axis_dim,
        cfg["main_length_ratio_min"],
        cfg["main_length_ratio_max"],
    )

    overlap_start = max(
        main_1["center_axis"] - main_1["length"] / 2,
        main_2["center_axis"] - main_2["length"] / 2,
    )
    overlap_end = min(
        main_1["center_axis"] + main_1["length"] / 2,
        main_2["center_axis"] + main_2["length"] / 2,
    )
    if overlap_end - overlap_start < middle["thickness"]:
        return None

    middle["center_axis"] = rng.uniform(overlap_start + middle["thickness"] / 2, overlap_end - middle["thickness"] / 2)
    middle["length"] = rng.uniform(
        perp_dim * cfg["middle_length_ratio_min"],
        perp_dim * cfg["middle_length_ratio_max"],
    )

    required_perp = middle["length"] + main_1["thickness"] / 2 + main_2["thickness"] / 2
    if required_perp > perp_dim:
        return None

    middle["center_perp"] = rng.uniform(-perp_dim / 2 + required_perp / 2, perp_dim / 2 - required_perp / 2)
    main_1["center_perp"] = middle["center_perp"] - middle["length"] / 2 - main_1["thickness"] / 2
    main_2["center_perp"] = middle["center_perp"] + middle["length"] / 2 + main_2["thickness"] / 2
    middle["hole_perp_1"] = main_1["center_perp"] + main_1["thickness"] / 2
    middle["hole_perp_2"] = main_2["center_perp"] - main_2["thickness"] / 2

    ribs = [
        _parallel_rib(main_1["length"], main_1["thickness"], main_1["height"], main_1["center_axis"], main_1["center_perp"], orientation),
        _parallel_rib(main_2["length"], main_2["thickness"], main_2["height"], main_2["center_axis"], main_2["center_perp"], orientation),
        _cross_rib(middle["length"], middle["thickness"], middle["height"], middle["center_axis"], middle["center_perp"], orientation),
    ]
    ribs[-1].hole_type = "end"
    ribs[-1].hole_radius = cfg["corner_hole_radius"]
    ribs[-1].cutter_axis = "X" if orientation == "X" else "Y"
    if orientation == "X":
        ribs[-1].hole_centers = [
            (middle["center_axis"], middle["hole_perp_1"]),
            (middle["center_axis"], middle["hole_perp_2"]),
        ]
    else:
        ribs[-1].hole_centers = [
            (middle["hole_perp_1"], middle["center_axis"]),
            (middle["hole_perp_2"], middle["center_axis"]),
        ]
    return ribs


def generate_sandwich_rib_plate(index, output_dir=None, params=SIMPLE_STRUCTURE_PARAMS, rng=None):
    rng = rng or random
    plate_cfg = params["plate"]
    cfg = params["sandwich_ribs"]
    output_dir = output_dir or params["batch"]["output_dir"]
    plate_l = plate_cfg["length"]
    plate_w = plate_cfg["width"]
    edge_clearance_min = plate_cfg.get("edge_clearance_min", 0.0)

    orientation, angle_deg, ribs = _sample_layout(
        rng,
        cfg,
        plate_l,
        plate_w,
        edge_clearance_min,
        _main_pair_with_middle_layout,
    )
    step_filename, stl_filename = _write_model(
        output_dir,
        f"simple_sandwich_ribs_{index:03d}_{orientation.lower()}_rot{angle_deg:.1f}",
        ribs,
        angle_deg,
        plate_cfg,
    )
    return {
        "structure_type": "sandwich_ribs",
        "orientation": orientation,
        "rotation_angle_deg": angle_deg,
        "step": step_filename,
        "stl": stl_filename,
    }


def generate_sandwich_rib_batch(samples=None, output_dir=None, params=SIMPLE_STRUCTURE_PARAMS, seed=None):
    cfg = params["sandwich_ribs"]
    samples = samples or cfg["samples"]
    output_dir = output_dir or params["batch"]["output_dir"]

    rng = random.Random(seed) if seed is not None else random
    return [
        generate_sandwich_rib_plate(index, output_dir=output_dir, params=params, rng=rng)
        for index in range(samples)
    ]


def _main_pair_with_two_cross_ribs_layout(rng, cfg, orientation, axis_dim, perp_dim):
    main_1 = {
        "length": None,
        "thickness": _sample_value(rng, cfg, "main", "thickness"),
        "height": _sample_value(rng, cfg, "main", "height"),
    }
    main_2 = {
        "length": None,
        "thickness": _sample_value(rng, cfg, "main", "thickness"),
        "height": _sample_value(rng, cfg, "main", "height"),
    }
    middle_thicknesses = [
        _sample_value(rng, cfg, "middle", "thickness"),
        _sample_value(rng, cfg, "middle", "thickness"),
    ]
    middle_heights = [
        rng.uniform(cfg["middle_height_min"], min(main_1["height"], main_2["height"])),
        rng.uniform(cfg["middle_height_min"], min(main_1["height"], main_2["height"])),
    ]

    main_1["length"], main_1["center_axis"] = _sample_length_and_center(
        rng,
        axis_dim,
        cfg["main_length_ratio_min"],
        cfg["main_length_ratio_max"],
    )
    main_2["length"], main_2["center_axis"] = _sample_length_and_center(
        rng,
        axis_dim,
        cfg["main_length_ratio_min"],
        cfg["main_length_ratio_max"],
    )

    overlap_start = max(
        main_1["center_axis"] - main_1["length"] / 2,
        main_2["center_axis"] - main_2["length"] / 2,
    )
    overlap_end = min(
        main_1["center_axis"] + main_1["length"] / 2,
        main_2["center_axis"] + main_2["length"] / 2,
    )
    middle_centers = _sample_non_overlapping_centers_in_range(
        rng,
        2,
        overlap_start,
        overlap_end,
        middle_thicknesses,
        cfg["middle_min_gap"],
        cfg["placement_attempts"],
    )
    if middle_centers is None:
        return None

    middle_length = rng.uniform(
        perp_dim * cfg["middle_length_ratio_min"],
        perp_dim * cfg["middle_length_ratio_max"],
    )
    required_perp = middle_length + main_1["thickness"] / 2 + main_2["thickness"] / 2
    if required_perp > perp_dim:
        return None

    middle_center_perp = rng.uniform(-perp_dim / 2 + required_perp / 2, perp_dim / 2 - required_perp / 2)
    main_1["center_perp"] = middle_center_perp - middle_length / 2 - main_1["thickness"] / 2
    main_2["center_perp"] = middle_center_perp + middle_length / 2 + main_2["thickness"] / 2
    hole_perp_1 = main_1["center_perp"] + main_1["thickness"] / 2
    hole_perp_2 = main_2["center_perp"] - main_2["thickness"] / 2

    ribs = [
        _parallel_rib(
            main_1["length"],
            main_1["thickness"],
            main_1["height"],
            main_1["center_axis"],
            main_1["center_perp"],
            orientation,
        ),
        _parallel_rib(
            main_2["length"],
            main_2["thickness"],
            main_2["height"],
            main_2["center_axis"],
            main_2["center_perp"],
            orientation,
        ),
    ]

    for center_axis, thickness, height in zip(middle_centers, middle_thicknesses, middle_heights):
        rib = _cross_rib(middle_length, thickness, height, center_axis, middle_center_perp, orientation)
        rib.hole_type = "end"
        rib.hole_radius = cfg["corner_hole_radius"]
        rib.cutter_axis = "X" if orientation == "X" else "Y"
        if orientation == "X":
            rib.hole_centers = [
                (center_axis, hole_perp_1),
                (center_axis, hole_perp_2),
            ]
        else:
            rib.hole_centers = [
                (hole_perp_1, center_axis),
                (hole_perp_2, center_axis),
            ]
        ribs.append(rib)
    return ribs


def generate_grid_rib_plate(index, output_dir=None, params=SIMPLE_STRUCTURE_PARAMS, rng=None):
    rng = rng or random
    plate_cfg = params["plate"]
    cfg = params["grid_ribs"]
    output_dir = output_dir or params["batch"]["output_dir"]
    plate_l = plate_cfg["length"]
    plate_w = plate_cfg["width"]
    edge_clearance_min = plate_cfg.get("edge_clearance_min", 0.0)

    orientation, angle_deg, ribs = _sample_layout(
        rng,
        cfg,
        plate_l,
        plate_w,
        edge_clearance_min,
        _main_pair_with_two_cross_ribs_layout,
    )
    step_filename, stl_filename = _write_model(
        output_dir,
        f"simple_grid_ribs_{index:03d}_{orientation.lower()}_rot{angle_deg:.1f}",
        ribs,
        angle_deg,
        plate_cfg,
    )
    return {
        "structure_type": "grid_ribs",
        "orientation": orientation,
        "rotation_angle_deg": angle_deg,
        "step": step_filename,
        "stl": stl_filename,
    }


def generate_grid_rib_batch(samples=None, output_dir=None, params=SIMPLE_STRUCTURE_PARAMS, seed=None):
    cfg = params["grid_ribs"]
    samples = samples or cfg["samples"]
    output_dir = output_dir or params["batch"]["output_dir"]

    rng = random.Random(seed) if seed is not None else random
    return [
        generate_grid_rib_plate(index, output_dir=output_dir, params=params, rng=rng)
        for index in range(samples)
    ]


def _parallel_sample_counts(samples_per_type, rib_counts):
    if samples_per_type <= 0:
        return {rib_count: 0 for rib_count in rib_counts}

    base = samples_per_type // len(rib_counts)
    remainder = samples_per_type % len(rib_counts)
    counts = {}
    for index, rib_count in enumerate(rib_counts):
        counts[rib_count] = base + (1 if index < remainder else 0)
    return counts


def generate_all_structure_batches(
    samples_per_type=30,
    output_dir=None,
    params=SIMPLE_STRUCTURE_PARAMS,
    seed=None,
    start_index=0,
    structure_types=None,
):
    output_dir = output_dir or params["batch"]["output_dir"]
    rng = random.Random(seed) if seed is not None else random
    generated = []
    structure_types = structure_types or ["parallel_ribs", "t_ribs", "sandwich_ribs", "grid_ribs"]

    current_index = start_index

    if "single_rib" in structure_types:
        for _ in range(samples_per_type):
            generated.append(
                generate_single_rib_plate(
                    current_index,
                    output_dir=output_dir,
                    params=params,
                    rng=rng,
                )
            )
            current_index += 1

    if "parallel_ribs" in structure_types:
        parallel_cfg = params["parallel_ribs"]
        for rib_count, sample_count in _parallel_sample_counts(
            samples_per_type,
            parallel_cfg["rib_counts"],
        ).items():
            for _ in range(sample_count):
                generated.append(
                    generate_parallel_rib_plate(
                        current_index,
                        rib_count,
                        output_dir=output_dir,
                        params=params,
                        rng=rng,
                    )
                )
                current_index += 1

    if "t_ribs" in structure_types:
        for _ in range(samples_per_type):
            generated.append(
                generate_t_rib_plate(
                    current_index,
                    output_dir=output_dir,
                    params=params,
                    rng=rng,
                )
            )
            current_index += 1

    if "sandwich_ribs" in structure_types:
        for _ in range(samples_per_type):
            generated.append(
                generate_sandwich_rib_plate(
                    current_index,
                    output_dir=output_dir,
                    params=params,
                    rng=rng,
                )
            )
            current_index += 1

    if "grid_ribs" in structure_types:
        for _ in range(samples_per_type):
            generated.append(
                generate_grid_rib_plate(
                    current_index,
                    output_dir=output_dir,
                    params=params,
                    rng=rng,
                )
            )
            current_index += 1
    return generated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate simple welded workpieces with configurable structure types."
    )
    parser.add_argument(
        "--structure",
        choices=["single_rib", "parallel_ribs", "t_ribs", "sandwich_ribs", "grid_ribs", "all"],
        default="parallel_ribs",
        help="Simple structure type to generate.",
    )
    parser.add_argument(
        "--rib-counts",
        nargs="+",
        type=int,
        default=None,
        help="Rib counts to generate. Default: 1 2 3.",
    )
    parser.add_argument(
        "--samples-per-count",
        type=int,
        default=None,
        help="Number of samples generated for each structure setting. Default: 10.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Relative paths are resolved under data_generation/.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.structure in ("single_rib", "all"):
        generate_single_rib_batch(
            samples=args.samples_per_count,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    if args.structure in ("parallel_ribs", "all"):
        generate_parallel_rib_batch(
            rib_counts=args.rib_counts,
            samples_per_count=args.samples_per_count,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    if args.structure in ("t_ribs", "all"):
        generate_t_rib_batch(
            samples=args.samples_per_count,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    if args.structure in ("sandwich_ribs", "all"):
        generate_sandwich_rib_batch(
            samples=args.samples_per_count,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    if args.structure in ("grid_ribs", "all"):
        generate_grid_rib_batch(
            samples=args.samples_per_count,
            output_dir=args.output_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
