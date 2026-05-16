"""
文件 IO 工具

提供 JSON、STEP 文件的读写功能
"""
from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional


def load_json(path: str) -> Dict[str, Any]:
    """
    加载 JSON 文件
    
    Args:
        path: JSON 文件路径
        
    Returns:
        解析后的字典对象
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 格式错误
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any], *, indent: int = 2) -> None:
    """
    保存 JSON 文件
    
    Args:
        path: 保存路径
        data: 要保存的数据
        indent: 缩进空格数（默认2）
    """
    ensure_dir(os.path.dirname(path))
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def load_step_file(path: str):
    """
    加载 STEP 文件
    
    Args:
        path: STEP 文件路径
        
    Returns:
        TopoDS_Shape 对象
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"STEP file not found: {path}")
    
    from OCC.Extend.DataExchange import read_step_file
    
    shape = read_step_file(path)
    if shape is None:
        raise ValueError(f"Failed to load STEP file: {path}")
    
    return shape


def file_exists(path: str) -> bool:
    """检查文件是否存在"""
    return os.path.exists(path)


def get_file_size(path: str) -> int:
    """
    获取文件大小（字节）
    
    Returns:
        文件大小，文件不存在返回 0
    """
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path)


def list_json_files(directory: str, pattern: Optional[str] = None) -> list[str]:
    """
    列出目录下的所有 JSON 文件
    
    Args:
        directory: 目录路径
        pattern: 文件名模式（可选），如 "*_geometry_graph.json"
        
    Returns:
        JSON 文件路径列表
    """
    if not os.path.exists(directory):
        return []
    
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            if pattern is None or pattern in filename:
                files.append(os.path.join(directory, filename))
    
    return sorted(files)

