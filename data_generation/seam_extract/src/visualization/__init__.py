"""
可视化模块

提供 OCC 3D 查看器、几何图可视化、焊缝可视化等功能
"""

from .occ_viewer import visualize_step_solids
from .plot_helpers import set_axes_equal, build_color_map

__all__ = [
    "visualize_step_solids",
    "set_axes_equal",
    "build_color_map",
]

