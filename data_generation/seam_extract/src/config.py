"""
项目配置管理模块

提供统一的文件路径管理和配置参数
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class ProjectConfig:
    """
    集中管理一个子装配体的所有文件路径
    
    使用方式:
        cfg = ProjectConfig(stp_head="GL5")
        # 自动使用 model/sub_assembly/GL5 作为基础目录
        shape = load_step_file(cfg.step_file)
    """

    stp_head: str
    base_dir: Optional[str] = None

    def __post_init__(self):
        """初始化后自动设置 base_dir"""
        if self.base_dir is None:
            self.base_dir = f"model/sub_assembly/{self.stp_head}"

    @property
    def step_file(self) -> str:
        """STEP 模型文件路径（自动检测 .stp 或 .step 后缀）"""
        # 优先检查 .step 后缀
        step_path = f"{self.base_dir}/{self.stp_head}.step"
        if os.path.exists(step_path):
            return step_path
        # 回退到 .stp 后缀
        return f"{self.base_dir}/{self.stp_head}.stp"

    @property
    def geometry_file(self) -> str:
        """几何信息 JSON（已废弃，保留兼容）"""
        return f"{self.base_dir}/{self.stp_head}_geometry.json"

    @property
    def contact_edges_file(self) -> str:
        """接触边界（焊缝）JSON"""
        return f"{self.base_dir}/{self.stp_head}_contact_edges.json"

    @property
    def filtered_edges_file(self) -> str:
        """过滤后的边界 JSON"""
        return f"{self.base_dir}/{self.stp_head}_filter_edges.json"

    @property
    def weld_order_file(self) -> str:
        """焊缝排序结果 JSON"""
        return f"{self.base_dir}/{self.stp_head}_weld_order.json"

    @property
    def split_edges_file(self) -> str:
        """焊缝分割结果 JSON（包含交叉断点）"""
        return f"{self.base_dir}/{self.stp_head}_split_edges.json"

    @property
    def through_holes_file(self) -> str:
        """通孔检测结果 JSON"""
        return f"{self.base_dir}/{self.stp_head}_through_holes.json"

    @property
    def geometry_graph_file(self) -> str:
        """几何图 JSON"""
        return f"{self.base_dir}/{self.stp_head}_geometry_graph.json"

    @property
    def geometry_graph_with_breakpoints_file(self) -> str:
        """带断点的几何图 JSON"""
        return f"{self.base_dir}/{self.stp_head}_geometry_graph_with_breakpoints.json"

    @property
    def process_graph_file(self) -> str:
        """工艺图 JSON"""
        return f"{self.base_dir}/{self.stp_head}_process_graph.json"

    @property
    def final_welds_file(self) -> str:
        """最终焊缝 JSON（标准交付格式）"""
        return f"{self.base_dir}/{self.stp_head}_final_welds.json"

    @property
    def final_welds_with_junctions_file(self) -> str:
        """最终焊缝 JSON（包含交叉断点）"""
        return f"{self.base_dir}/{self.stp_head}_final_welds_with_junctions.json"


@dataclass
class AlgorithmConfig:
    """算法参数配置"""
    
    # 接触边界提取参数
    bbox_tol: float = 5.0
    contact_tol: float = 0.5
    face_tol: float = 0.8
    min_edge_length: float = 0.1
    
    # 几何图构建参数
    graph_point_tol: float = 0.2
    graph_min_geom_edge_length: float = 0.05
    
    # 焊缝排序参数
    base_plane_weld_tol: float = 2.0
    ransac_iter: int = 4000
    
    # 焊缝分割参数
    seed_snap_tol_xy: float = 160.0
    seed_spacing_xy: float = 40.0
    snap_to_through_tol_xy: float = 80.0
    parallel_sin_tol: float = 0.06
    
    # 通孔检测参数
    through_hole_min_radius: float = 2.0
    through_hole_max_radius: float = 50.0
    
    # 断点插入参数
    breakpoint_merge_tol: float = 1e-6
    skip_near_end_eps: float = 1e-5


# 默认配置实例
DEFAULT_CONFIG = AlgorithmConfig()

