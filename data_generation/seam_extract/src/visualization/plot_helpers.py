"""
Matplotlib 绘图辅助函数

提供 3D 绘图的常用工具
"""
from __future__ import annotations

from typing import List, Dict, Any
import matplotlib.cm as cm


def set_axes_equal(ax, xs: List[float], ys: List[float], zs: List[float]) -> None:
    """
    设置 3D 坐标轴等比例显示
    
    Args:
        ax: matplotlib 3D axes 对象
        xs, ys, zs: 所有点的坐标列表
    """
    if not xs:
        return
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
    if max_range <= 1e-12:
        max_range = 1.0
    
    xm = 0.5 * (xmin + xmax)
    ym = 0.5 * (ymin + ymax)
    zm = 0.5 * (zmin + zmax)
    
    ax.set_xlim(xm - 0.5 * max_range, xm + 0.5 * max_range)
    ax.set_ylim(ym - 0.5 * max_range, ym + 0.5 * max_range)
    ax.set_zlim(zm - 0.5 * max_range, zm + 0.5 * max_range)


def build_color_map(keys: List[str], cmap_name: str = "tab20") -> Dict[str, tuple]:
    """
    为一组键构建稳定的颜色映射
    
    Args:
        keys: 键列表（如焊缝ID）
        cmap_name: matplotlib colormap 名称
        
    Returns:
        键到颜色的映射字典
    """
    cmap = cm.get_cmap(cmap_name)
    sorted_keys = sorted(keys, key=lambda x: (len(str(x)), str(x)))
    
    color_map = {}
    for i, key in enumerate(sorted_keys):
        # tab20 有 20 种颜色，超过则循环
        color_map[key] = cmap(i % 20)
    
    return color_map


def plot_polyline_3d(
    ax,
    points: List[List[float]],
    *,
    color: str = "blue",
    linewidth: float = 1.0,
    label: str = None,
    alpha: float = 1.0
) -> None:
    """
    在 3D 坐标系中绘制折线
    
    Args:
        ax: matplotlib 3D axes 对象
        points: 点列表 [[x,y,z], ...]
        color: 线条颜色
        linewidth: 线宽
        label: 标签（用于图例）
        alpha: 透明度
    """
    if not points or len(points) < 2:
        return
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    
    ax.plot(xs, ys, zs, color=color, linewidth=linewidth, label=label, alpha=alpha)


def plot_points_3d(
    ax,
    points: List[List[float]],
    *,
    color: str = "red",
    marker: str = "o",
    size: float = 20,
    label: str = None,
    alpha: float = 1.0
) -> None:
    """
    在 3D 坐标系中绘制点
    
    Args:
        ax: matplotlib 3D axes 对象
        points: 点列表 [[x,y,z], ...]
        color: 点颜色
        marker: 标记样式
        size: 点大小
        label: 标签（用于图例）
        alpha: 透明度
    """
    if not points:
        return
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    
    ax.scatter(xs, ys, zs, c=color, marker=marker, s=size, label=label, alpha=alpha)


def add_text_3d(
    ax,
    position: List[float],
    text: str,
    *,
    fontsize: int = 10,
    color: str = "black"
) -> None:
    """
    在 3D 坐标系中添加文本标注
    
    Args:
        ax: matplotlib 3D axes 对象
        position: 文本位置 [x, y, z]
        text: 文本内容
        fontsize: 字体大小
        color: 文本颜色
    """
    ax.text(position[0], position[1], position[2], text, fontsize=fontsize, color=color)

