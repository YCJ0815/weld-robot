"""
通用工具模块

提供几何计算、IO、数学辅助等功能
"""

from .geometry import (
    vector_unit,
    vector_dot,
    vector_cross,
    vector_norm,
    point_distance,
    segment_length,
    point_to_segment_distance,
    point_to_plane_distance,
    project_point_to_segment,
    project_point_to_plane,
    apply_transform,
    build_local_frame,
)

from .io import (
    load_json,
    save_json,
    ensure_dir,
)

from .math_helpers import (
    clamp,
    lerp,
    point_key,
    polyline_length,
    polyline_cumulative_length,
    point_at_parameter,
)

__all__ = [
    # geometry
    "vector_unit",
    "vector_dot",
    "vector_cross",
    "vector_norm",
    "point_distance",
    "segment_length",
    "point_to_segment_distance",
    "point_to_plane_distance",
    "project_point_to_segment",
    "project_point_to_plane",
    "apply_transform",
    "build_local_frame",
    # io
    "load_json",
    "save_json",
    "ensure_dir",
    # math
    "clamp",
    "lerp",
    "point_key",
    "polyline_length",
    "polyline_cumulative_length",
    "point_at_parameter",
]

