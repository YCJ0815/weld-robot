"""
向后兼容层

保持旧的导入路径可用，同时引导用户使用新的模块结构
"""
from __future__ import annotations

import warnings


def _deprecation_warning(old_path: str, new_path: str):
    """显示弃用警告"""
    warnings.warn(
        f"'{old_path}' is deprecated. Please use '{new_path}' instead.",
        DeprecationWarning,
        stacklevel=3
    )


# ============================================================
# 兼容旧的 project_config.py
# ============================================================
from .config import ProjectConfig

# ============================================================
# 兼容旧的 load_step_file.py
# ============================================================
def load_step_file(file_name: str):
    """
    向后兼容的 STEP 文件加载函数
    
    已弃用：请使用 src.geometry.step_loader.load_step_file
    """
    _deprecation_warning(
        "src.load_step_file",
        "src.geometry.step_loader.load_step_file"
    )
    from .geometry.step_loader import load_step_file as new_load
    return new_load(file_name)


# ============================================================
# 兼容旧的 visual_step_solids.py
# ============================================================
def visual_step_solids(shape, display):
    """
    向后兼容的 STEP 可视化函数
    
    已弃用：请使用 src.visualization.occ_viewer.visualize_step_solids
    """
    _deprecation_warning(
        "src.visual_step_solids",
        "src.visualization.occ_viewer.visualize_step_solids"
    )
    from .visualization.occ_viewer import visualize_step_solids
    return visualize_step_solids(shape, display)


# ============================================================
# 兼容旧的 ui_weld_sort_options.py
# ============================================================
def get_weld_sort_options(parent=None, default_priority="vertical_first", default_in_class="length_desc"):
    """
    向后兼容的焊缝排序选项对话框
    
    已弃用：请使用 src.ui.dialogs.get_weld_sort_options
    """
    _deprecation_warning(
        "src.ui_weld_sort_options.get_weld_sort_options",
        "src.ui.dialogs.get_weld_sort_options"
    )
    from .ui.dialogs import get_weld_sort_options as new_get
    return new_get(parent, default_priority, default_in_class)


# ============================================================
# 兼容旧的 exit_application.py
# ============================================================
def exit_application(event=None):
    """
    向后兼容的退出函数
    
    已弃用：直接使用 sys.exit() 或 display.close()
    """
    import sys
    sys.exit()


# ============================================================
# 导出所有兼容函数
# ============================================================
__all__ = [
    "ProjectConfig",
    "load_step_file",
    "visual_step_solids",
    "get_weld_sort_options",
    "exit_application",
]

