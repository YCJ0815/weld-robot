"""
数学辅助函数

提供常用的数学计算功能
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np


def clamp(x: float, min_val: float, max_val: float) -> float:
    """
    将值限制在指定范围内
    
    Args:
        x: 输入值
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        限制后的值
    """
    return max(min_val, min(max_val, x))


def lerp(a: float, b: float, t: float) -> float:
    """
    线性插值
    
    Args:
        a: 起始值
        b: 结束值
        t: 插值参数 [0, 1]
        
    Returns:
        插值结果 a + t * (b - a)
    """
    return a + t * (b - a)


def point_key(point: List[float], ndigits: int = 8) -> Tuple[float, float, float]:
    """
    生成点的哈希键（用于去重）
    
    Args:
        point: 3D点 [x, y, z]
        ndigits: 保留小数位数
        
    Returns:
        (x, y, z) 元组，四舍五入到指定精度
    """
    return (
        round(float(point[0]), ndigits),
        round(float(point[1]), ndigits),
        round(float(point[2]), ndigits)
    )


def polyline_length(points: List[List[float]]) -> float:
    """
    计算折线的总长度
    
    Args:
        points: 点列表 [[x,y,z], ...]
        
    Returns:
        总长度
    """
    if len(points) < 2:
        return 0.0
    
    pts = np.array(points, dtype=float)
    segments = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    
    return float(np.sum(lengths))


def polyline_cumulative_length(points: List[List[float]]) -> np.ndarray:
    """
    计算折线的累积长度
    
    Args:
        points: 点列表 [[x,y,z], ...]
        
    Returns:
        累积长度数组，长度与 points 相同
        第一个元素为 0，最后一个元素为总长度
    """
    if len(points) < 2:
        return np.zeros(len(points), dtype=float)
    
    pts = np.array(points, dtype=float)
    segments = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    
    return np.concatenate(([0.0], np.cumsum(lengths)))


def point_at_parameter(
    points: List[List[float]],
    t: float,
    cumulative_lengths: np.ndarray = None
) -> List[float]:
    """
    在折线上按参数 t ∈ [0, 1] 获取点
    
    Args:
        points: 点列表 [[x,y,z], ...]
        t: 参数，0 表示起点，1 表示终点
        cumulative_lengths: 预计算的累积长度（可选，提高性能）
        
    Returns:
        插值点 [x, y, z]
    """
    if len(points) < 2:
        return points[0] if points else [0.0, 0.0, 0.0]
    
    t = clamp(t, 0.0, 1.0)
    
    if cumulative_lengths is None:
        cumulative_lengths = polyline_cumulative_length(points)
    
    total_length = float(cumulative_lengths[-1])
    if total_length <= 1e-12:
        # 退化为点
        return points[0]
    
    target_length = t * total_length
    
    # 找到目标长度所在的线段
    idx = int(np.searchsorted(cumulative_lengths, target_length, side="right") - 1)
    idx = max(0, min(idx, len(points) - 2))
    
    seg_start_length = float(cumulative_lengths[idx])
    seg_end_length = float(cumulative_lengths[idx + 1])
    seg_length = seg_end_length - seg_start_length
    
    if seg_length <= 1e-12:
        return points[idx]
    
    # 在线段内插值
    local_t = (target_length - seg_start_length) / seg_length
    local_t = clamp(local_t, 0.0, 1.0)
    
    p0 = np.array(points[idx], dtype=float)
    p1 = np.array(points[idx + 1], dtype=float)
    
    result = p0 + local_t * (p1 - p0)
    return [float(result[0]), float(result[1]), float(result[2])]


def angle_between_vectors(v1: List[float], v2: List[float]) -> float:
    """
    计算两个向量之间的夹角（弧度）
    
    Args:
        v1, v2: 向量
        
    Returns:
        夹角 [0, π]
    """
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    
    cos_angle = float(np.dot(a, b) / (norm_a * norm_b))
    cos_angle = clamp(cos_angle, -1.0, 1.0)
    
    return float(np.arccos(cos_angle))


def is_parallel(v1: List[float], v2: List[float], tol: float = 1e-6) -> bool:
    """
    判断两个向量是否平行
    
    Args:
        v1, v2: 向量
        tol: 容差（sin值）
        
    Returns:
        True 如果平行
    """
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return True
    
    cross = np.cross(a, b)
    sin_angle = float(np.linalg.norm(cross) / (norm_a * norm_b))
    
    return sin_angle < tol


def is_perpendicular(v1: List[float], v2: List[float], tol: float = 1e-6) -> bool:
    """
    判断两个向量是否垂直
    
    Args:
        v1, v2: 向量
        tol: 容差（cos值）
        
    Returns:
        True 如果垂直
    """
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return False
    
    cos_angle = abs(float(np.dot(a, b) / (norm_a * norm_b)))
    
    return cos_angle < tol

