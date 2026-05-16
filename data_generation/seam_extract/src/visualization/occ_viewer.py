"""
OCC 3D 查看器

提供基于 pythonocc 的 3D 可视化功能
"""
from __future__ import annotations

import random
from typing import Optional

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Display.OCCViewer import rgb_color


def visualize_step_solids(shape: TopoDS_Shape, display) -> None:
    """
    在 OCC 显示器中可视化 STEP 实体
    
    为每个实体分配随机颜色，半透明显示
    
    Args:
        shape: TopoDS_Shape 对象
        display: OCC Display 对象
    """
    if shape is None:
        print("[visualize] No shape to display")
        return
    
    explorer = TopologyExplorer(shape)
    
    solid_count = 0
    for solid in explorer.solids():
        # 生成柔和的随机颜色
        r = random.uniform(0.3, 0.9)
        g = random.uniform(0.3, 0.9)
        b = random.uniform(0.3, 0.9)
        
        color = rgb_color(r, g, b)
        display.DisplayShape(solid, color=color, transparency=0.3, update=False)
        solid_count += 1
    
    display.FitAll()
    display.View_Iso()
    
    print(f"[visualize] Displayed {solid_count} solids")

