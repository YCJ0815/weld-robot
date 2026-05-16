"""
对话框模块

提供各种参数设置对话框
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

try:
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel,
        QComboBox, QPushButton, QGroupBox
    )
    from PyQt5.QtCore import Qt
    HAS_QT = True
except ImportError:
    HAS_QT = False


def get_weld_sort_options(
    parent=None,
    default_priority: str = "vertical_first",
    default_in_class: str = "length_desc"
) -> Tuple[Dict[str, Any], bool]:
    """
    显示焊缝排序选项对话框
    
    Args:
        parent: 父窗口（可选）
        default_priority: 默认优先级模式
        default_in_class: 默认类内排序模式
        
    Returns:
        (options_dict, ok)
        - options_dict: {"priority_mode": str, "in_class_mode": str}
        - ok: 用户是否点击了确定
    """
    if not HAS_QT:
        # 如果没有 PyQt5，返回默认值
        print("[dialog] PyQt5 not available, using defaults")
        return {
            "priority_mode": default_priority,
            "in_class_mode": default_in_class
        }, True
    
    dialog = QDialog(parent)
    dialog.setWindowTitle("焊缝排序选项")
    dialog.setMinimumWidth(400)
    
    layout = QVBoxLayout()
    
    # 优先级模式
    priority_group = QGroupBox("优先级模式")
    priority_layout = QVBoxLayout()
    
    priority_label = QLabel("选择焊接位置优先级：")
    priority_combo = QComboBox()
    priority_combo.addItems([
        "vertical_first (优先立焊)",
        "horizontal_first (优先横焊)",
        "overhead_first (优先仰焊)",
        "flat_first (优先平焊)"
    ])
    
    # 设置默认值
    priority_map = {
        "vertical_first": 0,
        "horizontal_first": 1,
        "overhead_first": 2,
        "flat_first": 3
    }
    priority_combo.setCurrentIndex(priority_map.get(default_priority, 0))
    
    priority_layout.addWidget(priority_label)
    priority_layout.addWidget(priority_combo)
    priority_group.setLayout(priority_layout)
    
    # 类内排序模式
    in_class_group = QGroupBox("类内排序")
    in_class_layout = QVBoxLayout()
    
    in_class_label = QLabel("同类焊缝的排序方式：")
    in_class_combo = QComboBox()
    in_class_combo.addItems([
        "length_desc (长度降序)",
        "length_asc (长度升序)",
        "spatial (空间顺序)"
    ])
    
    # 设置默认值
    in_class_map = {
        "length_desc": 0,
        "length_asc": 1,
        "spatial": 2
    }
    in_class_combo.setCurrentIndex(in_class_map.get(default_in_class, 0))
    
    in_class_layout.addWidget(in_class_label)
    in_class_layout.addWidget(in_class_combo)
    in_class_group.setLayout(in_class_layout)
    
    # 按钮
    button_layout = QHBoxLayout()
    ok_button = QPushButton("确定")
    cancel_button = QPushButton("取消")
    
    ok_button.clicked.connect(dialog.accept)
    cancel_button.clicked.connect(dialog.reject)
    
    button_layout.addStretch()
    button_layout.addWidget(ok_button)
    button_layout.addWidget(cancel_button)
    
    # 组装布局
    layout.addWidget(priority_group)
    layout.addWidget(in_class_group)
    layout.addLayout(button_layout)
    
    dialog.setLayout(layout)
    
    # 显示对话框
    result = dialog.exec_()
    
    if result == QDialog.Accepted:
        # 解析选择
        priority_text = priority_combo.currentText()
        in_class_text = in_class_combo.currentText()
        
        priority_mode = priority_text.split()[0]
        in_class_mode = in_class_text.split()[0]
        
        return {
            "priority_mode": priority_mode,
            "in_class_mode": in_class_mode
        }, True
    else:
        return {
            "priority_mode": default_priority,
            "in_class_mode": default_in_class
        }, False

