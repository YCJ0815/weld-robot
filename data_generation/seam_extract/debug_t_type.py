#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T型断点检测调试脚本
运行此脚本以启用详细的调试输出，帮助诊断T型断点检测问题
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from s0_detect_through_holes_from_graph import (
    load_json, save_json,
    detect_through_hole_edges_from_adjacent,
    detect_t_type_breakpoints,
    insert_breakpoint_for_node,
    visualize_geometry_graph
)

def main():
    stp_head = "D018-F205B"
    
    # 文件路径
    geometry_graph_path = f"model/sub_assembly/{stp_head}/{stp_head}_geometry_graph.json"
    out_geometry_graph_path = f"model/sub_assembly/{stp_head}/{stp_head}_geometry_graph_with_breakpoints.json"
    process_graph_path = f"model/sub_assembly/{stp_head}/{stp_head}_process_graph.json"
    
    print("[INFO] Loading geometry graph...")
    gg = load_json(geometry_graph_path)
    
    # Step 1: 检测过焊孔边
    print("\n[STEP 1] Detecting through-hole edges...")
    pg, cand_ids = detect_through_hole_edges_from_adjacent(
        gg,
        bspline_rms_tol=0.25,
        require_arc_center=False,
        bspline_min_angle_deg=8.0,
        bspline_max_radius=500.0,
        bspline_min_sagitta_abs=0.30,
        bspline_min_sagitta_ratio=0.02,
        detect_chamfer_holes=True,
        chamfer_min_angle_deg=92.0,
        chamfer_max_angle_deg=150.0,
        chamfer_max_length=50.0,
        debug=True,
    )
    
    print(f"\n[RESULT] Detected {len(cand_ids)} through-hole edges")
    
    # 收集连接到过焊孔的节点
    connected_nodes = []
    for nid, ndata in pg.get("nodes", {}).items():
        if ndata.get("process", {}).get("through_hole_edge_ids"):
            connected_nodes.append(str(nid))
    
    print(f"[RESULT] {len(connected_nodes)} nodes connected to through-hole edges")
    
    # Step 2: 插入断点
    print("\n[STEP 2] Inserting breakpoints...")
    inserted = 0
    for nid in connected_nodes:
        node_proc = (pg.get("nodes", {}).get(str(nid), {}) or {}).get("process", {})
        hole_edge_ids = node_proc.get("through_hole_edge_ids") or []
        info = None
        for hole_edge_id in hole_edge_ids:
            info = insert_breakpoint_for_node(
                gg,
                nid,
                hole_edge_id=str(hole_edge_id),
                require_hole_edge_match=True,
                max_weld_length=100.0,
                verbose=False,
            )
            if info:
                break
        if info:
            inserted += 1
    
    print(f"[RESULT] Inserted {inserted} breakpoints")
    
    # 保存更新后的几何图
    save_json(out_geometry_graph_path, gg)
    print(f"[SAVE] Geometry graph with breakpoints -> {out_geometry_graph_path}")
    
    # Step 3: 检测T型断点（启用详细调试）
    print("\n[STEP 3] Detecting T-type breakpoints (with detailed debug output)...")
    print("=" * 80)
    
    t_type_info = detect_t_type_breakpoints(
        gg,
        cand_ids,
        t_type_angle_min_deg=15.0,
        t_type_angle_max_deg=165.0,
        t_type_coplanar_tol=2.0,
        t_type_extension_ratio=2.0,
        t_type_max_distance_to_weld=10.0,
        debug=True,  # 启用详细调试
    )
    
    print("=" * 80)
    print(f"\n[RESULT] Found {t_type_info['debug']['found']} T-type breakpoints")
    
    # 打印详细的调试统计
    print("\n[DEBUG STATISTICS]")
    for k, v in t_type_info["debug"].items():
        print(f"  {k:30s}: {v}")
    
    # 保存处理图
    save_json(process_graph_path, pg)
    print(f"\n[SAVE] Process graph -> {process_graph_path}")
    
    # Step 4: 可视化
    print("\n[STEP 4] Visualizing results...")
    t_type_breakpoints = t_type_info["t_type_breakpoints"] if t_type_info else None
    
    visualize_geometry_graph(
        gg,
        cand_ids,
        viz_mode="all",
        show_adjacent_geoms=False,
        bp_size=8.0,
        t_type_size=12.0,
        lw_geom=1.0,
        lw_weld=2.0,
        lw_hole=2.5,
        lw_adjacent=1.5,
        t_type_breakpoints=t_type_breakpoints,
    )
    
    print("\n[DONE] T-type breakpoint detection completed!")

if __name__ == "__main__":
    main()

