"""
几何处理模块

提供 STEP 文件加载、接触边界提取、几何图构建等功能
"""

from .step_loader import load_step_file, visualize_step_solids

__all__ = [
    "load_step_file",
    "visualize_step_solids",
]

