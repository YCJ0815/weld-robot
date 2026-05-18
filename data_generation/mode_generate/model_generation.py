import cadquery as cq
import random
import os
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SCALE_FACTOR = 2.0

# 统一参数配置区：仅需在这里调整即可影响模型生成行为。
MODEL_GEN_PARAMS = {
    "plate": {
        "length": 200,  # 底板长度（X 向尺寸，mm）
        "width": 200,  # 底板宽度（Y 向尺寸，mm）
        "height": 5,  # 底板厚度（Z 向尺寸，mm）
    },
    "main_rib": {
        "orientation_choices": ["X", "Y"],  # 主肋板可选朝向
        "thickness_min": 2,  # 主肋板最小厚度（mm）
        "thickness_max": 6,  # 主肋板最大厚度（mm）
        "height_min": 30,  # 主肋板最小高度（mm）
        "height_max": 60,  # 主肋板最大高度（mm）
        "length_ratio_min": 0.8,  # 主肋板长度相对底板边长的最小比例
        "length_ratio_max": 1.0,  # 主肋板长度相对底板边长的最大比例
        "dual_side_center_band_divisor": 6,  # 主肋板位于中间区域时两侧都生成副肋板：[-尺寸/divisor, +尺寸/divisor]
    },
    "side_rib": {
        "min_spacing": 35,  # 副肋板之间的最小净间距阈值（mm）
        "min_space_for_rib": 10,  # 某一侧允许放置副肋板的最小可用空间（mm）
        "thickness_min": 2,  # 副肋板最小厚度（mm）
        "thickness_max": 4,  # 副肋板最大厚度（mm）
        "height_min": 25,  # 副肋板最小高度（mm）
        "length_min": 60,  # 副肋板最小长度（沿垂直于主肋板方向，mm）
        "corner_hole_radius": 5,  # 副肋板-主肋板-底板交界处扇形孔半径（mm）
        "num_ribs_min": 3,  # 每侧尝试生成副肋板的最小数量
        "num_ribs_max": 5,  # 每侧尝试生成副肋板的最大数量
        "placement_attempts": 100,  # 单根副肋板的最大重试次数
    },
    "bridge_rib": {
        "min_connector_gap": 10,  # 连接肋板最小长度/相邻判定间距（mm）
        "thickness_min": 2,  # 连接肋板最小厚度（mm）
        "thickness_max": 4,  # 连接肋板最大厚度（mm）
        "height_min": 20,  # 连接肋板最小高度（mm）
        "corner_hole_radius": 5,  # 连接肋板-副肋板-底板接触处扇形孔半径（mm）
        "min_clearance_from_main": 50,  # 连接肋板相对主肋板的最小离开距离（mm）
        "interior_connect_probability": 0.65,  # 内部连接肋板生成概率
        "open_rib_probability": 0.6,  # 端部开放肋板生成概率
        "open_rib_min_extension": 10,  # 开放肋板最小外伸长度（mm）
        "open_rib_max_extension": 35,  # 开放肋板最大外伸长度（mm）
    },
    "batch": {
        "count": 10,  # 批量生成模型数量
        "output_dir": "data/model",  # 导出目录（相对当前 data_generation 项目根目录）
    },
}


_SCALABLE_PARAM_KEYS = {
    "length",
    "width",
    "height",
    "thickness_min",
    "thickness_max",
    "height_min",
    "height_max",
    "min_spacing",
    "min_space_for_rib",
    "length_min",
    "corner_hole_radius",
    "min_connector_gap",
    "min_clearance_from_main",
    "open_rib_min_extension",
    "open_rib_max_extension",
}


def _scaled_model_params(params, scale_factor):
    scaled = deepcopy(params)
    for section in scaled.values():
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if key in _SCALABLE_PARAM_KEYS:
                section[key] = value * scale_factor
    return scaled


MODEL_GEN_PARAMS = _scaled_model_params(MODEL_GEN_PARAMS, MODEL_SCALE_FACTOR)

def _box_on_plate(parts, width, depth, height, center_x, center_y, plate_top_z):
    part = (
        cq.Workplane("XY")
        .box(width, depth, height, centered=(True, True, False))
        .translate((center_x, center_y, plate_top_z))
    )
    parts.append(part)
    return part


def _resolve_output_dir(output_dir):
    output_dir = os.fspath(output_dir)
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(PROJECT_ROOT, output_dir)


def _cylinder_cutter(radius, height, start_point, direction):
    solid = cq.Solid.makeCylinder(
        radius,
        height,
        pnt=cq.Vector(*start_point),
        dir=cq.Vector(*direction),
    )
    return cq.Workplane("XY").add(solid)


def _side_rib_on_plate_with_corner_hole(
    parts,
    width,
    depth,
    height,
    center_x,
    center_y,
    plate_top_z,
    *,
    hole_radius,
    hole_center_x,
    hole_center_y,
    cutter_axis,
):
    part = (
        cq.Workplane("XY")
        .box(width, depth, height, centered=(True, True, False))
        .translate((center_x, center_y, plate_top_z))
    )

    if hole_radius and hole_radius > 0:
        margin = max(width, depth, hole_radius) + 1.0
        if cutter_axis == "X":
            cutter = _cylinder_cutter(
                hole_radius,
                width + 2 * margin,
                (center_x - width / 2 - margin, hole_center_y, plate_top_z),
                (1, 0, 0),
            )
        elif cutter_axis == "Y":
            cutter = _cylinder_cutter(
                hole_radius,
                depth + 2 * margin,
                (hole_center_x, center_y - depth / 2 - margin, plate_top_z),
                (0, 1, 0),
            )
        else:
            raise ValueError(f"Unsupported side-rib corner-hole cutter axis: {cutter_axis}")
        part = part.cut(cutter)

    parts.append(part)
    return part


def _rib_on_plate_with_end_corner_holes(
    parts,
    width,
    depth,
    height,
    center_x,
    center_y,
    plate_top_z,
    *,
    hole_radius,
    hole_centers,
    cutter_axis,
):
    part = (
        cq.Workplane("XY")
        .box(width, depth, height, centered=(True, True, False))
        .translate((center_x, center_y, plate_top_z))
    )

    if hole_radius and hole_radius > 0:
        margin = max(width, depth, hole_radius) + 1.0
        for hole_center_x, hole_center_y in hole_centers:
            if cutter_axis == "X":
                cutter = _cylinder_cutter(
                    hole_radius,
                    width + 2 * margin,
                    (center_x - width / 2 - margin, hole_center_y, plate_top_z),
                    (1, 0, 0),
                )
            elif cutter_axis == "Y":
                cutter = _cylinder_cutter(
                    hole_radius,
                    depth + 2 * margin,
                    (hole_center_x, center_y - depth / 2 - margin, plate_top_z),
                    (0, 1, 0),
                )
            else:
                raise ValueError(f"Unsupported rib corner-hole cutter axis: {cutter_axis}")
            part = part.cut(cutter)

    parts.append(part)
    return part


def _part_bounds(part):
    bbox = part.val().BoundingBox()
    return (
        float(bbox.xmin),
        float(bbox.xmax),
        float(bbox.ymin),
        float(bbox.ymax),
        float(bbox.zmin),
        float(bbox.zmax),
    )


def _has_positive_volume_overlap(bounds_a, bounds_b, tol=1e-7):
    ax0, ax1, ay0, ay1, az0, az1 = bounds_a
    bx0, bx1, by0, by1, bz0, bz1 = bounds_b
    overlap_x = min(ax1, bx1) - max(ax0, bx0)
    overlap_y = min(ay1, by1) - max(ay0, by0)
    overlap_z = min(az1, bz1) - max(az0, bz0)
    return overlap_x > tol and overlap_y > tol and overlap_z > tol


def _assert_no_entity_intersections(parts):
    bounds = [_part_bounds(part) for part in parts]
    for i, bounds_i in enumerate(bounds):
        for j in range(i + 1, len(bounds)):
            if _has_positive_volume_overlap(bounds_i, bounds[j]):
                raise ValueError(
                    f"Generated entities {i + 1} and {j + 1} intersect with positive volume; "
                    "only contact is allowed."
                )


def _export_parts(parts, step_filename, stl_filename):
    _assert_no_entity_intersections(parts)
    compound = cq.Compound.makeCompound([part.val() for part in parts])
    cq.exporters.export(compound, step_filename)
    cq.exporters.export(compound, stl_filename)


def _add_parallel_bridge_ribs(
    model,
    parts,
    plate_top_z,
    placed_ribs,
    main_len,
    main_thickness,
    main_pos_perp,
    side,
    is_x_oriented_main_rib,
    bridge_cfg,
):
    """
    随机生成与主肋板平行的连接肋板。
    连接肋板的两端必须分别与两块相邻副肋板相连，
    且其相对主肋板的位置大于 50 mm，但不固定在副肋板最外侧。
    对最外侧副肋板，还可以额外生成只与该副肋板相连的开放肋板。
    """
    min_connector_gap = bridge_cfg["min_connector_gap"]
    min_connector_thickness = bridge_cfg["thickness_min"]
    max_connector_thickness = bridge_cfg["thickness_max"]
    min_connector_height = bridge_cfg["height_min"]
    connector_corner_hole_radius = bridge_cfg.get("corner_hole_radius", 5)
    min_connector_clearance_from_main = bridge_cfg["min_clearance_from_main"]
    interior_connect_probability = bridge_cfg["interior_connect_probability"]
    open_rib_probability = bridge_cfg["open_rib_probability"]
    open_rib_min_extension = bridge_cfg["open_rib_min_extension"]
    open_rib_max_extension = bridge_cfg["open_rib_max_extension"]

    if not placed_ribs:
        return model

    sorted_ribs = sorted(placed_ribs, key=lambda rib: rib["start"])
    axis_min = -main_len / 2
    axis_max = main_len / 2

    def _add_connector(
        connector_center_parallel,
        connector_len,
        connector_thickness,
        connector_height,
        anchor_length,
        contact_positions_parallel,
    ):
        max_clearance = anchor_length - connector_thickness
        if max_clearance <= min_connector_clearance_from_main:
            return model

        connector_clearance = random.uniform(
            min_connector_clearance_from_main,
            max_clearance,
        )
        pos_perp = main_pos_perp + side * (
            main_thickness / 2 + connector_clearance + connector_thickness / 2
        )

        if is_x_oriented_main_rib:
            _rib_on_plate_with_end_corner_holes(
                parts,
                connector_len,
                connector_thickness,
                connector_height,
                connector_center_parallel,
                pos_perp,
                plate_top_z,
                hole_radius=connector_corner_hole_radius,
                hole_centers=[(pos_parallel, pos_perp) for pos_parallel in contact_positions_parallel],
                cutter_axis="Y",
            )
            return (
                model.workplaneFromTagged("top_face")
                .center(connector_center_parallel, pos_perp)
                .box(connector_len, connector_thickness, connector_height, centered=(True, True, False), combine=False)
            )
        _rib_on_plate_with_end_corner_holes(
            parts,
            connector_thickness,
            connector_len,
            connector_height,
            pos_perp,
            connector_center_parallel,
            plate_top_z,
            hole_radius=connector_corner_hole_radius,
            hole_centers=[(pos_perp, pos_parallel) for pos_parallel in contact_positions_parallel],
            cutter_axis="X",
        )
        return (
            model.workplaneFromTagged("top_face")
            .center(pos_perp, connector_center_parallel)
            .box(connector_thickness, connector_len, connector_height, centered=(True, True, False), combine=False)
        )

    for left_rib, right_rib in zip(sorted_ribs, sorted_ribs[1:]):
        gap = right_rib["start"] - left_rib["end"]
        if gap < min_connector_gap or random.random() > interior_connect_probability:
            continue

        connector_thickness = min(
            random.uniform(min_connector_thickness, max_connector_thickness),
            min(left_rib["length"], right_rib["length"]) - 1
        )
        if connector_thickness <= 0:
            continue

        # 连接肋板只填充两根副肋板之间的净空，端面分别与左右副肋板接触，不插入副肋板内部。
        connector_start = left_rib["end"]
        connector_end = right_rib["start"]
        connector_len = connector_end - connector_start
        pos_parallel = (connector_start + connector_end) / 2
        connector_height = random.uniform(min_connector_height, min(left_rib["height"], right_rib["height"]))
        anchor_length = min(left_rib["length"], right_rib["length"])
        model = _add_connector(
            pos_parallel,
            connector_len,
            connector_thickness,
            connector_height,
            anchor_length,
            [connector_start, connector_end],
        )

    def _add_open_rib(edge_rib, open_side):
        connector_thickness = min(
            random.uniform(min_connector_thickness, max_connector_thickness),
            edge_rib["length"] - 1
        )
        if connector_thickness <= 0:
            return model

        if open_side == "left":
            max_extension = min(open_rib_max_extension, edge_rib["start"] - axis_min)
        else:
            max_extension = min(open_rib_max_extension, axis_max - edge_rib["end"])

        if max_extension <= open_rib_min_extension:
            return model

        extension_len = random.uniform(open_rib_min_extension, max_extension)
        if open_side == "left":
            connector_end = edge_rib["start"]
            connector_start = connector_end - extension_len
        else:
            connector_start = edge_rib["end"]
            connector_end = connector_start + extension_len

        connector_len = connector_end - connector_start
        if connector_len < min_connector_gap:
            return model

        connector_height = random.uniform(min_connector_height, edge_rib["height"])
        return _add_connector(
            (connector_start + connector_end) / 2,
            connector_len,
            connector_thickness,
            connector_height,
            edge_rib["length"],
            [connector_end if open_side == "left" else connector_start],
        )

    if sorted_ribs and random.random() <= open_rib_probability:
        model = _add_open_rib(sorted_ribs[0], "left")

    if sorted_ribs and random.random() <= open_rib_probability:
        model = _add_open_rib(sorted_ribs[-1], "right")

    return model

def _add_side_ribs(
    model,
    parts,
    plate_top_z,
    main_len,
    main_height,
    main_thickness,
    main_pos_perp,
    plate_dim_perp,
    side,
    is_x_oriented_main_rib,
    side_cfg,
    bridge_cfg,
):
    """
    在主肋板的一侧随机生成多个垂直的副肋板。

    :param model: 当前的 CadQuery 模型
    :param main_len: 主肋板的长度
    :param main_height: 主肋板的高度
    :param main_thickness: 主肋板的厚度
    :param main_pos_perp: 主肋板在垂直轴上的位置
    :param plate_dim_perp: 底板在垂直于主肋板方向上的尺寸
    :param side: -1 表示负方向一侧, 1 表示正方向一侧
    :param is_x_oriented_main_rib: 主肋板是否为 X-向
    :return: (更新后的模型, 成功添加的肋板数量)
    """
    min_rib_spacing = side_cfg["min_spacing"]
    min_space_for_rib = side_cfg["min_space_for_rib"]
    min_rib_thickness = side_cfg["thickness_min"]
    max_rib_thickness = side_cfg["thickness_max"]
    min_rib_height = side_cfg["height_min"]
    min_rib_length = side_cfg["length_min"]
    corner_hole_radius = side_cfg.get("corner_hole_radius", 5)
    num_ribs_min = side_cfg["num_ribs_min"]
    num_ribs_max = side_cfg["num_ribs_max"]
    placement_attempts = side_cfg["placement_attempts"]

    # 计算该侧的可用空间
    if side == -1:
        available_space = (main_pos_perp - main_thickness / 2) - (-plate_dim_perp / 2)
    else:  # side == 1
        available_space = (plate_dim_perp / 2) - (main_pos_perp + main_thickness / 2)

    if available_space < min_space_for_rib:
        return model, 0

    max_ribs_fit = int((main_len + min_rib_spacing) // (min_rib_thickness + min_rib_spacing))
    if max_ribs_fit <= 0:
        return model, 0

    num_ribs_to_try = random.randint(num_ribs_min, max(num_ribs_min, min(num_ribs_max, max_ribs_fit)))
    placed_ribs_bounds = []  # 存储已放置肋板在平行于主肋板轴上的 (start, end) 坐标
    placed_ribs = []
    ribs_added = 0

    for _ in range(num_ribs_to_try):
        for _ in range(placement_attempts):  # 多尝试几次，提高生成多个短肋板的成功率
            sec_thickness = random.uniform(min_rib_thickness, max_rib_thickness)
            sec_height = random.uniform(min_rib_height, main_height)
            sec_len = random.uniform(min_rib_length, available_space)  # 副肋板的长度
            pos_parallel = random.uniform(-main_len / 2 + sec_thickness / 2, main_len / 2 - sec_thickness / 2)

            # 碰撞与间距检查
            new_rib_start = pos_parallel - sec_thickness / 2
            new_rib_end = pos_parallel + sec_thickness / 2
            is_too_close = any(
                new_rib_start < existing_end + min_rib_spacing
                and new_rib_end > existing_start - min_rib_spacing
                for existing_start, existing_end in placed_ribs_bounds
            )

            if not is_too_close:
                pos_perp = main_pos_perp + side * (main_thickness / 2 + sec_len / 2)
                hole_pos_perp = main_pos_perp + side * (main_thickness / 2)
                if is_x_oriented_main_rib:
                    _side_rib_on_plate_with_corner_hole(
                        parts,
                        sec_thickness,
                        sec_len,
                        sec_height,
                        pos_parallel,
                        pos_perp,
                        plate_top_z,
                        hole_radius=corner_hole_radius,
                        hole_center_x=pos_parallel,
                        hole_center_y=hole_pos_perp,
                        cutter_axis="X",
                    )
                    model = model.workplaneFromTagged("top_face").center(pos_parallel, pos_perp).box(sec_thickness, sec_len, sec_height, centered=(True, True, False), combine=False)
                else:
                    _side_rib_on_plate_with_corner_hole(
                        parts,
                        sec_len,
                        sec_thickness,
                        sec_height,
                        pos_perp,
                        pos_parallel,
                        plate_top_z,
                        hole_radius=corner_hole_radius,
                        hole_center_x=hole_pos_perp,
                        hole_center_y=pos_parallel,
                        cutter_axis="Y",
                    )
                    model = model.workplaneFromTagged("top_face").center(pos_perp, pos_parallel).box(sec_len, sec_thickness, sec_height, centered=(True, True, False), combine=False)
                placed_ribs_bounds.append((new_rib_start, new_rib_end))
                placed_ribs.append({
                    "start": new_rib_start,
                    "end": new_rib_end,
                    "length": sec_len,
                    "height": sec_height,
                })
                ribs_added += 1
                break  # 成功放置，继续尝试下一个

    model = _add_parallel_bridge_ribs(
        model,
        parts,
        plate_top_z,
        placed_ribs,
        main_len,
        main_thickness,
        main_pos_perp,
        side,
        is_x_oriented_main_rib,
        bridge_cfg,
    )
    return model, ribs_added

def generate_random_plate(index, output_dir=None, params=MODEL_GEN_PARAMS):
    plate_cfg = params["plate"]
    main_cfg = params["main_rib"]
    side_cfg = params["side_rib"]
    bridge_cfg = params["bridge_rib"]

    if output_dir is None:
        output_dir = params["batch"]["output_dir"]
    output_dir = _resolve_output_dir(output_dir)

    # 1. 底板基本尺寸
    plate_l = plate_cfg["length"]
    plate_w = plate_cfg["width"]
    plate_h = plate_cfg["height"]
    
    # 2. 创建底板并标记顶面
    model = cq.Workplane("XY").box(plate_l, plate_w, plate_h).faces(">Z").tag("top_face")
    parts = [cq.Workplane("XY").box(plate_l, plate_w, plate_h)]
    plate_top_z = plate_h / 2
    
    # 3. 随机生成主肋板（基本占满整个底板长度）
    orientation = random.choice(main_cfg["orientation_choices"])
    main_thickness = random.uniform(main_cfg["thickness_min"], main_cfg["thickness_max"])
    main_height = random.uniform(main_cfg["height_min"], main_cfg["height_max"])
    
    sides_with_ribs = 0

    if orientation == 'X':
        # X 方向为主肋板
        main_len = random.uniform(plate_l * main_cfg["length_ratio_min"], plate_l * main_cfg["length_ratio_max"])
        main_w, main_d = main_len, main_thickness
        pos_x = 0  # X 方向居中
        pos_y = random.uniform(-plate_w/2 + main_thickness/2, plate_w/2 - main_thickness/2)
        
        model = (
            model.workplaneFromTagged("top_face")
            .center(pos_x, pos_y)
            .box(main_w, main_d, main_height, centered=(True, True, False), combine=False)
        )
        _box_on_plate(parts, main_w, main_d, main_height, pos_x, pos_y, plate_top_z)
        
        # 主肋板位于底板宽度中间 1/3-2/3 区域时，两侧都生成；否则仅在空间更大的一侧生成
        center_band = plate_w / main_cfg["dual_side_center_band_divisor"]
        if -center_band <= pos_y <= center_band:
            model, num_added_1 = _add_side_ribs(model, parts, plate_top_z, main_len, main_height, main_thickness, pos_y, plate_w, -1, True, side_cfg, bridge_cfg)
            if num_added_1 > 0:
                sides_with_ribs += 1
            model, num_added_2 = _add_side_ribs(model, parts, plate_top_z, main_len, main_height, main_thickness, pos_y, plate_w, 1, True, side_cfg, bridge_cfg)
            if num_added_2 > 0:
                sides_with_ribs += 1
        elif pos_y < -center_band:
            model, num_added = _add_side_ribs(model, parts, plate_top_z, main_len, main_height, main_thickness, pos_y, plate_w, 1, True, side_cfg, bridge_cfg)
            if num_added > 0:
                sides_with_ribs += 1
        else:
            model, num_added = _add_side_ribs(model, parts, plate_top_z, main_len, main_height, main_thickness, pos_y, plate_w, -1, True, side_cfg, bridge_cfg)
            if num_added > 0:
                sides_with_ribs += 1
    else:
        # Y 方向为主肋板
        main_len = random.uniform(plate_w * main_cfg["length_ratio_min"], plate_w * main_cfg["length_ratio_max"])
        main_w, main_d = main_thickness, main_len
        pos_x = random.uniform(-plate_l/2 + main_thickness/2, plate_l/2 - main_thickness/2)
        pos_y = 0  # Y 方向居中
        
        model = (
            model.workplaneFromTagged("top_face")
            .center(pos_x, pos_y)
            .box(main_w, main_d, main_height, centered=(True, True, False), combine=False)
        )
        _box_on_plate(parts, main_w, main_d, main_height, pos_x, pos_y, plate_top_z)
        
        # 主肋板位于底板长度中间 1/3-2/3 区域时，两侧都生成；否则仅在空间更大的一侧生成
        center_band = plate_l / main_cfg["dual_side_center_band_divisor"]
        if -center_band <= pos_x <= center_band:
            model, num_added_1 = _add_side_ribs(model, parts, plate_top_z, main_len, main_height, main_thickness, pos_x, plate_l, -1, False, side_cfg, bridge_cfg)
            if num_added_1 > 0:
                sides_with_ribs += 1
            model, num_added_2 = _add_side_ribs(model, parts, plate_top_z, main_len, main_height, main_thickness, pos_x, plate_l, 1, False, side_cfg, bridge_cfg)
            if num_added_2 > 0:
                sides_with_ribs += 1
        elif pos_x < -center_band:
            model, num_added = _add_side_ribs(model, parts, plate_top_z, main_len, main_height, main_thickness, pos_x, plate_l, 1, False, side_cfg, bridge_cfg)
            if num_added > 0:
                sides_with_ribs += 1
        else:
            model, num_added = _add_side_ribs(model, parts, plate_top_z, main_len, main_height, main_thickness, pos_x, plate_l, -1, False, side_cfg, bridge_cfg)
            if num_added > 0:
                sides_with_ribs += 1

    # 4. 确保目录存在并导出
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    side_rib_suffix = "_double_side_ribs" if sides_with_ribs == 2 else ""
    base_filename = os.path.join(output_dir, f"random_plate_{index}{side_rib_suffix}")
    step_filename = f"{base_filename}.step"
    stl_filename = f"{base_filename}.stl"

    _export_parts(parts, step_filename, stl_filename)
    print(f"已生成: {step_filename}")
    print(f"已生成: {stl_filename}")
    return {
        "step": step_filename,
        "stl": stl_filename,
    }

def generate_batch(count=None, output_dir=None, params=MODEL_GEN_PARAMS):
    if count is None:
        count = params["batch"]["count"]
    if output_dir is None:
        output_dir = params["batch"]["output_dir"]
    output_dir = _resolve_output_dir(output_dir)

    generated = []
    for j in range(count):
        generated.append(generate_random_plate(j, output_dir=output_dir, params=params))
    return generated


if __name__ == "__main__":
    generate_batch()
