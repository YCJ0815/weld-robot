"""
STEP 文件加载器

提供 STEP 文件的加载和基本可视化功能
"""
from __future__ import annotations

import os
from typing import Optional

from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopoDS import TopoDS_Shape


def load_step_file(file_path: str) -> Optional[TopoDS_Shape]:
    """
    加载 STEP 文件
    
    Args:
        file_path: STEP 文件路径
        
    Returns:
        TopoDS_Shape 对象，加载失败返回 None
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"STEP file not found: {file_path}")
    
    print(f"[load_step] Loading: {file_path}")
    
    try:
        shape = read_step_file(file_path)
        if shape is None:
            print(f"[load_step] Failed to load: {file_path}")
            return None
        
        print(f"[load_step] Successfully loaded: {file_path}")
        return shape
        
    except Exception as e:
        print(f"[load_step] Error loading {file_path}: {e}")
        return None


def visualize_step_solids(shape: TopoDS_Shape, display) -> None:
    """
    在 OCC 显示器中可视化 STEP 实体
    
    Args:
        shape: TopoDS_Shape 对象
        display: OCC Display 对象
    """
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Display.OCCViewer import rgb_color
    import random
    
    if shape is None:
        print("[visualize] No shape to display")
        return
    
    explorer = TopologyExplorer(shape)
    
    # 为每个实体分配随机颜色
    for i, solid in enumerate(explorer.solids()):
        # 生成柔和的随机颜色
        r = random.uniform(0.3, 0.9)
        g = random.uniform(0.3, 0.9)
        b = random.uniform(0.3, 0.9)
        
        color = rgb_color(r, g, b)
        display.DisplayShape(solid, color=color, transparency=0.3, update=False)
    
    display.FitAll()
    display.View_Iso()
    
    print(f"[visualize] Displayed {i+1} solids")


def get_solid_count(shape: TopoDS_Shape) -> int:
    """
    获取 STEP 模型中的实体数量
    
    Args:
        shape: TopoDS_Shape 对象
        
    Returns:
        实体数量
    """
    from OCC.Extend.TopologyUtils import TopologyExplorer
    
    if shape is None:
        return 0
    
    explorer = TopologyExplorer(shape)
    return len(list(explorer.solids()))


def get_face_count(shape: TopoDS_Shape) -> int:
    """
    获取 STEP 模型中的面数量
    
    Args:
        shape: TopoDS_Shape 对象
        
    Returns:
        面数量
    """
    from OCC.Extend.TopologyUtils import TopologyExplorer
    
    if shape is None:
        return 0
    
    explorer = TopologyExplorer(shape)
    return len(list(explorer.faces()))


def get_edge_count(shape: TopoDS_Shape) -> int:
    """
    获取 STEP 模型中的边数量
    
    Args:
        shape: TopoDS_Shape 对象
        
    Returns:
        边数量
    """
    from OCC.Extend.TopologyUtils import TopologyExplorer
    
    if shape is None:
        return 0
    
    explorer = TopologyExplorer(shape)
    return len(list(explorer.edges()))


def print_step_info(shape: TopoDS_Shape) -> None:
    """
    打印 STEP 模型的基本信息
    
    Args:
        shape: TopoDS_Shape 对象
    """
    if shape is None:
        print("[step_info] No shape provided")
        return
    
    print("\n" + "="*50)
    print("STEP Model Information")
    print("="*50)
    print(f"Solids: {get_solid_count(shape)}")
    print(f"Faces:  {get_face_count(shape)}")
    print(f"Edges:  {get_edge_count(shape)}")
    print("="*50 + "\n")

