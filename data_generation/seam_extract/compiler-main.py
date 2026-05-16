from __future__ import print_function

import sys
import os

from src.config import ProjectConfig
from src.geometry.step_loader import load_step_file
from src.visualization.occ_viewer import visualize_step_solids
# from src.ui.dialogs import get_weld_sort_options

from src.contact_edge import (
    extract_contact_boundaries_face_based,
    visualize_contact_edges_from_json,
)  
# from src.weld_sort import sort_welds_and_export


from src.viewer_weld import visualize_weld_order, visualize_weld_splits


import s0_detect_through_holes_from_graph as s0    
import s1_build_final_weld_json_and_viz as s1
import s2_merge_split_breakpoints_into_final as s2
import s3_visualize_final_json as s3
from src.utils import load_json, save_json  
from s2_merge_split_breakpoints_into_final import load_json, save_json
  

CFG = ProjectConfig(stp_head="model2")

display = None
start_display = None
add_menu = None
add_function_to_menu = None
shape = None


def ensure_display():
    global display, start_display, add_menu, add_function_to_menu
    if display is None:
        from OCC.Display.SimpleGui import init_display
        display, start_display, add_menu, add_function_to_menu = init_display()
    return display, start_display, add_menu, add_function_to_menu


def ensure_shape():
    global shape
    if shape is None:
        shape = load_step_file(CFG.step_file)
    return shape

# 焊缝排序（优先立焊、优先横焊），记住上次选择
# _last_priority_mode = "vertical_first"
# _last_in_class_mode = "length_desc"


# def export_weld_order_with_options():
#     global _last_priority_mode, _last_in_class_mode

#     opts, ok = get_weld_sort_options(
#         parent=None,  # 如果你有主窗口对象可传进来；没有也没关系
#         default_priority=_last_priority_mode,
#         default_in_class=_last_in_class_mode
#     )
#     if not ok:
#         print("[weld_sort] canceled.")
#         return

#     _last_priority_mode = opts["priority_mode"]
#     _last_in_class_mode = opts["in_class_mode"]

#     sort_welds_and_export(
#         shape=shape,
#         input_edges_json=CFG.contact_edges_file,
#         output_order_json=CFG.weld_order_file,
#         up_axis="auto",
#         base_plane_weld_tol=2.0,  
#         ransac_iter=4000,
#         priority_mode=_last_priority_mode,   
#         in_class_mode=_last_in_class_mode    
#     )



def export_split_edges_via_hole_nodes():
    """使用新方法（基于过焊孔节点）拆分焊缝"""
    gg_path = CFG.geometry_graph_file
    if not os.path.exists(gg_path):
        print(f"Error: {gg_path} not found. Please run 'Export contact boundaries' first.")
        return
    

# 可视化
def show_weld_splits():
    visualize_weld_splits(
        split_json=CFG.split_edges_file,
        original_edges_json=CFG.contact_edges_file,   # 叠加拆分前（灰色）
        order_json_for_frame=CFG.weld_order_file,       # 用 base_frame 显示 LOCAL
        use_local_frame=True,
        show_original=True,
        show_virtual=True,
        show_junctions=True,
        label_junctions=False
    )

def show_weld_order():
    visualize_weld_order(
        CFG.weld_order_file,
        CFG.filtered_edges_file,
        label_top_n=60,
    )

def visual_step():
    current_display, _, _, _ = ensure_display()
    visualize_step_solids(ensure_shape(), current_display)


def export_contact_edges():
    current_shape = ensure_shape()
    extract_contact_boundaries_face_based(
        current_shape,
        CFG.contact_edges_file,
        bbox_tol=5.0,
        contact_tol=0.5,
        face_tol=0.8,
        min_edge_length=0.1,
        do_section_approx=True,
        profile=True,
        use_solid_grid_candidates=True,
        solid_grid_cell_size=None,   # 先用自动
        face_grid_cell_size=None,    # 先用自动
        bspline_detail=False,
        bspline_len_samples=24,
        bspline_vis_samples=24,
        geometry_graph_file=CFG.geometry_graph_file,
        graph_point_tol=0.2,
        graph_min_geom_edge_length=0.05,
        graph_include_weld_edges_in_adjacent=False,
        graph_store_all_geom_edges=False,
        graph_source_step=f"{CFG.stp_head}.stp",
    )


def show_contact_edges():
    visualize_contact_edges_from_json(CFG.contact_edges_file)


# ============================================================
# New Wrapper Functions for Added Scripts
# ============================================================

def run_detect_through_holes():
    print("Running s0: Detect Through Holes...")
    gg_path = CFG.geometry_graph_file
    if not os.path.exists(gg_path):
        print(f"Error: {gg_path} not found. Please run 'Export contact boundaries' first.")
        return
    
    geometry_graph = load_json(gg_path)
    
    # 1. Detect through-hole edges
    process_graph, candidate_ids = s0.detect_through_hole_edges_from_adjacent(
        geometry_graph,
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

    # Collect nodes connected to through-hole edges
    connected_nodes = []
    for nid, ndata in process_graph.get("nodes", {}).items():
        if ndata.get("process", {}).get("through_hole_edge_ids"):
            connected_nodes.append(str(nid))

    print(f"[result] detected {len(candidate_ids)} through-hole candidate edges")
    print(f"[result] {len(connected_nodes)} nodes connected to through-hole edges")

    # 2. Insert Breakpoints
    print(f"\n[breakpoint] inserting breakpoints for {len(connected_nodes)} nodes...")
    print(f"[breakpoint] length threshold: only insert on welds with length < 100.0")
    inserted = 0
    for nid in connected_nodes:
        node_proc = (process_graph.get("nodes", {}).get(str(nid), {}) or {}).get("process", {})
        hole_edge_ids = node_proc.get("through_hole_edge_ids") or []
        info = None
        for hole_edge_id in hole_edge_ids:
            info = s0.insert_breakpoint_for_node(
                geometry_graph,
                nid,
                hole_edge_id=str(hole_edge_id),
                require_hole_edge_match=True,
                max_weld_length=100.0,
                verbose=True,
            )
            if info:
                break
        if info:
            inserted += 1
    print(f"[breakpoint] inserted {inserted} breakpoints")

    # 3. Detect T-type breakpoints
    print(f"\n[T-type] detecting T-type breakpoints...")
    t_type_info = s0.detect_t_type_breakpoints(
        geometry_graph,
        candidate_ids,
        t_type_min_weld_length=5.0,
        t_type_max_weld_length=5000.0,
        t_type_extension_ratio=2.0,
        t_type_max_distance_to_weld=1000.0,
        debug=True,
    )
    # Write t_type_breakpoints into geometry_graph so s2 can consume it (always write, even if empty)
    if t_type_info:
        n_found = t_type_info['debug']['found']
        print(f"[T-type] found {n_found} T-type breakpoints")
        geometry_graph["t_type_breakpoints"] = t_type_info["t_type_breakpoints"]
    else:
        geometry_graph["t_type_breakpoints"] = []

    # 4. Save
    save_json(CFG.geometry_graph_with_breakpoints_file, geometry_graph)
    save_json(CFG.process_graph_file, process_graph)
    print(f"[save] geometry_graph with breakpoints -> {CFG.geometry_graph_with_breakpoints_file}")
    print(f"[save] process_graph -> {CFG.process_graph_file}")

    # 5. Visualize
    t_type_breakpoints = t_type_info["t_type_breakpoints"] if t_type_info else None
    s0.visualize_geometry_graph(
        geometry_graph,
        candidate_ids,
        viz_mode="holes_weld_all_breakpoints",
        show_adjacent_geoms=False,
        bp_size=8.0,
        t_type_size=12.0,
        lw_geom=1.0,
        lw_weld=2.0,
        lw_hole=2.5,
        lw_adjacent=1.5,
        t_type_breakpoints=t_type_breakpoints,
    )

def run_build_final_json():
    print("Running s1: Build Final JSON...")
    gg_path = CFG.geometry_graph_with_breakpoints_file
    pg_path = CFG.process_graph_file
    
    if not os.path.exists(gg_path):
        print(f"Error: {gg_path} not found. Please run 'Detect Through Holes' first.")
        return
        
    gg = load_json(gg_path)
    pg = load_json(pg_path) if os.path.exists(pg_path) else None
    
    final_obj, up_axis = s1.build_final_json(
        gg, pg,
        l_push_on="B",
        u_wrap_distance_threshold=100.0,
        u_wrap_max_nearby_welds=2,
        vertical_weld_deg=20.0
    )
    
    save_json(CFG.final_welds_file, final_obj)
    print(f"Saved to {CFG.final_welds_file}")
    
    s1.visualize_final_json(final_obj, up_axis, title="Final Welds", show_z_axis=False)

def run_merge_split_breakpoints():
    print("Running s2: Merge T-type Breakpoints into Final Welds...")
    # Inputs: final_welds (s1 output), geometry_graph_with_breakpoints (s0 output, contains t_type_breakpoints)

    if not os.path.exists(CFG.final_welds_file):
        print(f"Error: {CFG.final_welds_file} not found. Please run 'Build Final JSON' first.")
        return
    if not os.path.exists(CFG.geometry_graph_with_breakpoints_file):
        print(f"Error: {CFG.geometry_graph_with_breakpoints_file} not found. Please run 'Detect Through Holes' first.")
        return

    s2.update_final_welds_with_t_type_breakpoints(
        final_welds_json=CFG.final_welds_file,
        geometry_graph_with_breakpoints_json=CFG.geometry_graph_with_breakpoints_file,
        out_json=CFG.final_welds_with_junctions_file,
        max_remove_length=50.0,
    )
    print(f"Saved to {CFG.final_welds_with_junctions_file}")

def run_visualize_final_json():
    print("Running s3: Visualize Final JSON...")
    path = CFG.final_welds_with_junctions_file
    if not os.path.exists(path):
        print(f"Warning: {path} not found, falling back to {CFG.final_welds_file}")
        path = CFG.final_welds_file
        
    if not os.path.exists(path):
        print("Error: No final weld file found to visualize.")
        return

    data = load_json(path)
    s3.visualize_final_welds(data, title=path, show_ids=False, show_L=True)

def run_process_weld_data():
    print("Running s4: Process Weld Data...")
    input_file = CFG.final_welds_with_junctions_file
    output_file = f"{CFG.base_dir}/{CFG.stp_head}_merged_information_generated.json"
    
    # Construct reference file path to maintain visualization consistency with standalone script
    reference_file = f"{CFG.base_dir}/{CFG.stp_head}_merged_information.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run 'Merge Split Breakpoints' first.")
        return
    
    # Check if reference file exists and use it for visualization if possible
    viz_target = reference_file if os.path.exists(reference_file) else None
    
    if viz_target:
        print(f"Visualizing reference file to match standalone script behavior: {viz_target}")
    else:
        print(f"Reference file not found. Visualizing generated file: {output_file}")
        
    s4.process_file(input_file, output_file, viz_path=viz_target)


if __name__ == "__main__":
    add_menu('STEP Viewer')
    add_function_to_menu('STEP Viewer', visual_step)

    add_menu('Export contact boundaries')
    add_function_to_menu("Export contact boundaries", export_contact_edges)
    add_function_to_menu("Export contact boundaries", show_contact_edges)

    # # 焊缝排序（优先立焊、优先横焊）
    # add_menu("Weld Order")
    # add_function_to_menu("Weld Order", export_weld_order_with_options)

    # # 可视化
    # add_menu("Weld Visual Debug")
    # add_function_to_menu("Weld Visual Debug", show_weld_splits) 
    
    # New Weld Processing Menu
    add_menu("Weld Processing")
    add_function_to_menu("Weld Processing", run_detect_through_holes)   # s0
    add_function_to_menu("Weld Processing", run_build_final_json)       # s1
    add_function_to_menu("Weld Processing", run_merge_split_breakpoints)# s2
    add_function_to_menu("Weld Processing", run_visualize_final_json)   # s3
    # add_menu("Weld Sorting")
    # add_function_to_menu("Weld Sorting", run_process_weld_data)      # s4

    add_menu('Exit')
    add_function_to_menu('Exit', lambda: sys.exit())


    start_display()
