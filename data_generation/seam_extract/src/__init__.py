"""
Swiss Weldseam Compiler - 焊缝工艺路径规划系统

主要模块：
- geometry: 几何处理（STEP加载、接触边界提取、几何图构建）
- process: 工艺处理（焊缝排序、分割、断点插入）
- visualization: 可视化（OCC查看器、图形可视化）
- ui: 用户界面（对话框、菜单）
- utils: 通用工具（几何计算、IO、数学函数）
- export: 导出模块（JSON生成、格式转换）
"""

__version__ = "2.0.0"
__author__ = "Swiss Weldseam Team"

# 向后兼容：保留旧的导入路径
from .config import ProjectConfig

# 便捷导入
from .utils.io import load_json, save_json
from .utils.geometry import (
    vector_unit,
    vector_dot,
    vector_cross,
    point_distance,
    segment_length,
)

__all__ = [
    "ProjectConfig",
    "load_json",
    "save_json",
    "vector_unit",
    "vector_dot",
    "vector_cross",
    "point_distance",
    "segment_length",
]

