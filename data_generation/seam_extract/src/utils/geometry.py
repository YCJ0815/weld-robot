"""
几何计算工具函数

提供向量运算、距离计算、投影等基础几何功能
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np


Vec3 = List[float]
Point3 = List[float]


# ============================================================
# 向量运算
# ============================================================

def vector_norm(v: Vec3, eps: float = 1e-12) -> float:
    """计算向量的模长"""
    return float(np.linalg.norm(np.array(v, dtype=float)))


def vector_unit(v: Vec3, eps: float = 1e-12) -> Optional[Vec3]:
    """
    单位化向量
    
    Args:
        v: 输入向量 [x, y, z]
        eps: 最小长度阈值
        
    Returns:
        单位向量，如果长度小于eps则返回None
    """
    arr = np.array(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= eps:
        return None
    u = arr / n
    return [float(u[0]), float(u[1]), float(u[2])]


def vector_dot(a: Vec3, b: Vec3) -> float:
    """向量点积"""
    return float(np.dot(np.array(a, dtype=float), np.array(b, dtype=float)))


def vector_cross(a: Vec3, b: Vec3) -> Vec3:
    """向量叉积"""
    c = np.cross(np.array(a, dtype=float), np.array(b, dtype=float))
    return [float(c[0]), float(c[1]), float(c[2])]


def vector_subtract(a: Point3, b: Point3) -> Vec3:
    """向量减法: a - b"""
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    return [float(diff[0]), float(diff[1]), float(diff[2])]


def vector_add(a: Vec3, b: Vec3) -> Vec3:
    """向量加法: a + b"""
    sum_vec = np.array(a, dtype=float) + np.array(b, dtype=float)
    return [float(sum_vec[0]), float(sum_vec[1]), float(sum_vec[2])]


def vector_scale(v: Vec3, s: float) -> Vec3:
    """向量缩放: v * s"""
    scaled = np.array(v, dtype=float) * float(s)
    return [float(scaled[0]), float(scaled[1]), float(scaled[2])]


# ============================================================
# 距离计算
# ============================================================

def point_distance(a: Point3, b: Point3) -> float:
    """计算两点之间的欧氏距离"""
    return float(np.linalg.norm(np.array(b, dtype=float) - np.array(a, dtype=float)))


def segment_length(start: Point3, end: Point3) -> float:
    """计算线段长度（point_distance的别名）"""
    return point_distance(start, end)


def point_to_segment_distance(
    p: Point3, 
    a: Point3, 
    b: Point3
) -> Tuple[float, Point3, float]:
    """
    计算点到线段的最短距离
    
    Args:
        p: 查询点
        a: 线段起点
        b: 线段终点
        
    Returns:
        (t, closest_point, distance)
        - t: 参数 [0,1]，表示最近点在线段上的位置
        - closest_point: 线段上的最近点
        - distance: 点到线段的距离
    """
    pa = np.array(a, dtype=float)
    pb = np.array(b, dtype=float)
    pp = np.array(p, dtype=float)
    
    ab = pb - pa
    denom = float(np.dot(ab, ab))
    
    if denom < 1e-12:
        # 退化为点
        return 0.0, [float(pa[0]), float(pa[1]), float(pa[2])], float(np.linalg.norm(pp - pa))
    
    t = float(np.dot(pp - pa, ab) / denom)
    t = max(0.0, min(1.0, t))  # clamp to [0,1]
    
    closest = pa + t * ab
    dist = float(np.linalg.norm(pp - closest))
    
    return t, [float(closest[0]), float(closest[1]), float(closest[2])], dist


def point_to_plane_distance(
    p: Point3,
    plane_origin: Point3,
    plane_normal: Vec3
) -> float:
    """
    计算点到平面的有符号距离
    
    Args:
        p: 查询点
        plane_origin: 平面上的一点
        plane_normal: 平面法向量（应为单位向量）
        
    Returns:
        有符号距离（正值表示在法向量方向）
    """
    pp = np.array(p, dtype=float)
    po = np.array(plane_origin, dtype=float)
    pn = np.array(plane_normal, dtype=float)
    
    return float(np.dot(pp - po, pn))


# ============================================================
# 投影计算
# ============================================================

def project_point_to_segment(
    p: Point3,
    a: Point3,
    b: Point3
) -> Tuple[float, Point3]:
    """
    将点投影到线段上
    
    Returns:
        (t, projected_point)
        - t: 参数 [0,1]
        - projected_point: 投影点
    """
    t, proj, _ = point_to_segment_distance(p, a, b)
    return t, proj


def project_point_to_plane(
    p: Point3,
    plane_origin: Point3,
    plane_normal: Vec3
) -> Point3:
    """
    将点投影到平面上
    
    Args:
        p: 查询点
        plane_origin: 平面上的一点
        plane_normal: 平面法向量（应为单位向量）
        
    Returns:
        投影点
    """
    pp = np.array(p, dtype=float)
    po = np.array(plane_origin, dtype=float)
    pn = np.array(plane_normal, dtype=float)
    
    dist = float(np.dot(pp - po, pn))
    proj = pp - dist * pn
    
    return [float(proj[0]), float(proj[1]), float(proj[2])]


def project_vector_to_plane(v: Vec3, plane_normal: Vec3) -> Vec3:
    """
    将向量投影到平面上（去除法向分量）
    
    Args:
        v: 输入向量
        plane_normal: 平面法向量（应为单位向量）
        
    Returns:
        投影后的向量
    """
    vv = np.array(v, dtype=float)
    pn = np.array(plane_normal, dtype=float)
    
    proj = vv - float(np.dot(vv, pn)) * pn
    return [float(proj[0]), float(proj[1]), float(proj[2])]


# ============================================================
# 坐标变换
# ============================================================

def apply_transform(
    point: Point3,
    origin: Point3,
    x_axis: Vec3,
    y_axis: Vec3,
    z_axis: Vec3
) -> Point3:
    """
    将点从世界坐标系变换到局部坐标系
    
    local = [x_axis; y_axis; z_axis] · (point - origin)
    
    Args:
        point: 世界坐标系中的点
        origin: 局部坐标系原点
        x_axis, y_axis, z_axis: 局部坐标系的三个轴（应为单位向量）
        
    Returns:
        局部坐标系中的点
    """
    pp = np.array(point, dtype=float)
    po = np.array(origin, dtype=float)
    px = np.array(x_axis, dtype=float)
    py = np.array(y_axis, dtype=float)
    pz = np.array(z_axis, dtype=float)
    
    rel = pp - po
    local = np.array([
        float(np.dot(rel, px)),
        float(np.dot(rel, py)),
        float(np.dot(rel, pz))
    ], dtype=float)
    
    return [float(local[0]), float(local[1]), float(local[2])]


def build_local_frame(
    origin: Point3,
    z_axis: Vec3,
    reference_up: Optional[Vec3] = None
) -> Tuple[Point3, Vec3, Vec3, Vec3]:
    """
    构建局部坐标系
    
    Args:
        origin: 坐标系原点
        z_axis: Z轴方向（会被单位化）
        reference_up: 参考向上方向（用于确定X轴），默认为[0,0,1]
        
    Returns:
        (origin, x_axis, y_axis, z_axis)
    """
    if reference_up is None:
        reference_up = [0.0, 0.0, 1.0]
    
    z = np.array(z_axis, dtype=float)
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    
    up = np.array(reference_up, dtype=float)
    
    # X = up × Z
    x = np.cross(up, z)
    x_norm = float(np.linalg.norm(x))
    if x_norm < 1e-6:
        # up 和 z 平行，选择另一个参考方向
        if abs(z[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            up = np.array([1.0, 0.0, 0.0], dtype=float)
        x = np.cross(up, z)
        x_norm = float(np.linalg.norm(x))
    
    x = x / max(x_norm, 1e-12)
    
    # Y = Z × X
    y = np.cross(z, x)
    
    return (
        origin,
        [float(x[0]), float(x[1]), float(x[2])],
        [float(y[0]), float(y[1]), float(y[2])],
        [float(z[0]), float(z[1]), float(z[2])]
    )


# ============================================================
# 其他几何工具
# ============================================================

def midpoint(a: Point3, b: Point3) -> Point3:
    """计算两点的中点"""
    mid = (np.array(a, dtype=float) + np.array(b, dtype=float)) * 0.5
    return [float(mid[0]), float(mid[1]), float(mid[2])]


def bbox_overlap(
    min1: Point3, max1: Point3,
    min2: Point3, max2: Point3
) -> bool:
    """检查两个轴对齐包围盒是否重叠"""
    mn1 = np.array(min1, dtype=float)
    mx1 = np.array(max1, dtype=float)
    mn2 = np.array(min2, dtype=float)
    mx2 = np.array(max2, dtype=float)
    
    return bool(np.all(mx1 >= mn2) and np.all(mx2 >= mn1))


def compute_bbox(points: List[Point3]) -> Tuple[Point3, Point3]:
    """
    计算点集的轴对齐包围盒
    
    Returns:
        (min_point, max_point)
    """
    if not points:
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    
    pts = np.array(points, dtype=float)
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    
    return (
        [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])],
        [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])]
    )

