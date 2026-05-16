from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Optional
import os
import shutil
import uuid
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

from src.config import ProjectConfig
from src.geometry.step_loader import load_step_file
from src.contact_edge import extract_contact_boundaries_face_based
from src.utils import load_json, save_json
from src.visualization.plot_helpers import set_axes_equal

import s0_detect_through_holes_from_graph as s0
import s1_build_final_weld_json_and_viz as s1
import s2_merge_split_breakpoints_into_final as s2

app = FastAPI(title="Weld Seam Compiler API", version="3.0.0")


def visualize_and_save(final_obj: dict, up_axis, output_path: str) -> bool:
    """
    将最终焊缝 JSON 可视化并保存为静态图片。
    对应 compiler-main.py 中的 run_visualize_final_json (s3)。
    """
    try:
        contact_edges = final_obj.get("contact_edges", {}) or {}
        points = final_obj.get("points", {}) or {}

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Final Welds Visualization", fontsize=14)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        cmap = plt.get_cmap("tab20")
        xs_all, ys_all, zs_all = [], [], []
        endpoint_xyz = []
        breakpoint_xyz = []

        for p in points.values():
            if isinstance(p, dict):
                xyz = p.get("xyz")
                if xyz and len(xyz) == 3:
                    if p.get("role") == "breakpoint":
                        breakpoint_xyz.append(xyz)
                    else:
                        endpoint_xyz.append(xyz)

        for idx, (eid, edge) in enumerate(contact_edges.items()):
            color = cmap(idx % 20)
            samples = edge.get("samples")
            if not isinstance(samples, list) or len(samples) < 2:
                s = edge.get("start")
                t = edge.get("end")
                if not (s and t):
                    continue
                samples = [s, t]
            xs = [p[0] for p in samples]
            ys = [p[1] for p in samples]
            zs = [p[2] for p in samples]
            ax.plot(xs, ys, zs, color=color, linewidth=2.0, alpha=0.8)
            xs_all.extend(xs)
            ys_all.extend(ys)
            zs_all.extend(zs)

            strategy = edge.get("corner_strategy")
            if strategy in ("L_push", "U_wrap"):
                label_pos = [(samples[0][i] + samples[-1][i]) * 0.5 for i in range(3)]
                label_text = "L" if strategy == "L_push" else "U"
                label_color = "red" if strategy == "L_push" else "blue"
                ax.text(
                    label_pos[0], label_pos[1], label_pos[2],
                    label_text, fontsize=9, color=label_color, weight="bold"
                )

        if endpoint_xyz:
            ex = [p[0] for p in endpoint_xyz]
            ey = [p[1] for p in endpoint_xyz]
            ez = [p[2] for p in endpoint_xyz]
            ax.scatter(ex, ey, ez, marker="o", s=18, color="green",
                       label="Endpoints", alpha=0.7)

        if breakpoint_xyz:
            bx = [p[0] for p in breakpoint_xyz]
            by = [p[1] for p in breakpoint_xyz]
            bz = [p[2] for p in breakpoint_xyz]
            ax.scatter(bx, by, bz, marker="^", s=28, color="orange",
                       label="Breakpoints", alpha=0.7)

        if up_axis is not None and xs_all:
            pts = np.array(list(zip(xs_all, ys_all, zs_all)), dtype=float)
            centroid = pts.mean(axis=0)
            max_range = max(
                pts[:, 0].max() - pts[:, 0].min(),
                pts[:, 1].max() - pts[:, 1].min(),
                pts[:, 2].max() - pts[:, 2].min(),
            )
            ax.quiver(
                centroid[0], centroid[1], centroid[2],
                up_axis[0], up_axis[1], up_axis[2],
                length=max_range * 0.3, color="green",
                arrow_length_ratio=0.2, linewidth=3,
                label=f"Z-axis: [{up_axis[0]:.3f}, {up_axis[1]:.3f}, {up_axis[2]:.3f}]",
            )

        set_axes_equal(ax, xs_all, ys_all, zs_all)
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Visualization saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
        return False


def process_step_file(model_id: str, enable_visualization: bool = False):
    """
    完整焊缝处理流程，与 compiler-main.py 中的菜单功能严格对应：

      Step 1  export_contact_edges          -> contact_edges_file + geometry_graph_file
      Step 2  run_detect_through_holes (s0) -> geometry_graph_with_breakpoints + process_graph
      Step 3  run_build_final_json      (s1) -> final_welds_file
      Step 4  run_merge_split_breakpoints (s2) -> final_welds_with_junctions_file  <- 最终交付

    可选 Step 5: 生成可视化图片（对应 run_visualize_final_json / s3）
    """
    CFG = ProjectConfig(stp_head=model_id)
    os.makedirs(CFG.base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: 提取接触边界（对应 export_contact_edges）
    # ------------------------------------------------------------------
    print(f"[1/4] Loading STEP file: {CFG.step_file}")
    shape = load_step_file(CFG.step_file)

    print(f"[2/4] Extracting contact boundaries...")
    extract_contact_boundaries_face_based(
        shape,
        CFG.contact_edges_file,
        bbox_tol=5.0,
        contact_tol=0.5,
        face_tol=0.8,
        min_edge_length=0.1,
        do_section_approx=True,
        profile=True,
        use_solid_grid_candidates=True,
        solid_grid_cell_size=None,
        face_grid_cell_size=None,
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

    # ------------------------------------------------------------------
    # Step 2: 检测过焊孔 + 插入断点 + T型断点（对应 run_detect_through_holes / s0）
    # ------------------------------------------------------------------
    print(f"[3/4] Detecting through holes and inserting breakpoints (s0)...")
    gg_path = CFG.geometry_graph_file
    if not os.path.exists(gg_path):
        raise HTTPException(
            status_code=500,
            detail=f"{gg_path} not found after contact edge extraction."
        )

    geometry_graph = load_json(gg_path)

    # 2a. 检测过焊孔候选边
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

    # 2b. 收集与过焊孔边相连的节点
    connected_nodes = []
    for nid, ndata in process_graph.get("nodes", {}).items():
        if ndata.get("process", {}).get("through_hole_edge_ids"):
            connected_nodes.append(str(nid))

    print(f"[s0] detected {len(candidate_ids)} through-hole candidate edges")
    print(f"[s0] {len(connected_nodes)} nodes connected to through-hole edges")

    # 2c. 插入断点（与 compiler-main.py run_detect_through_holes 保持一致）
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
    print(f"[s0] inserted {inserted} breakpoints")

    # 2d. 检测 T型断点
    print(f"[s0] detecting T-type breakpoints...")
    t_type_info = s0.detect_t_type_breakpoints(
        geometry_graph,
        candidate_ids,
        t_type_min_weld_length=5.0,
        t_type_max_weld_length=5000.0,
        t_type_extension_ratio=2.0,
        t_type_max_distance_to_weld=1000.0,
        debug=True,
    )
    # 无论是否找到，都写入字段，保证 s2 能消费
    if t_type_info:
        n_found = t_type_info["debug"]["found"]
        print(f"[s0] found {n_found} T-type breakpoints")
        geometry_graph["t_type_breakpoints"] = t_type_info["t_type_breakpoints"]
    else:
        geometry_graph["t_type_breakpoints"] = []

    # 2e. 保存
    save_json(CFG.geometry_graph_with_breakpoints_file, geometry_graph)
    save_json(CFG.process_graph_file, process_graph)
    print(f"[s0] saved -> {CFG.geometry_graph_with_breakpoints_file}")
    print(f"[s0] saved -> {CFG.process_graph_file}")

    # ------------------------------------------------------------------
    # Step 3: 构建最终焊缝 JSON（对应 run_build_final_json / s1）
    # ------------------------------------------------------------------
    print(f"[3/4] Building final weld JSON (s1)...")
    gg = load_json(CFG.geometry_graph_with_breakpoints_file)
    pg = load_json(CFG.process_graph_file) if os.path.exists(CFG.process_graph_file) else None

    final_obj, up_axis = s1.build_final_json(
        gg, pg,
        l_push_on="B",
        u_wrap_distance_threshold=100.0,
        u_wrap_max_nearby_welds=2,
        vertical_weld_deg=20.0,
    )
    save_json(CFG.final_welds_file, final_obj)
    print(f"[s1] saved -> {CFG.final_welds_file}")

    # ------------------------------------------------------------------
    # Step 4: 合并 T型断点（对应 run_merge_split_breakpoints / s2）
    # ------------------------------------------------------------------
    print(f"[4/4] Merging T-type breakpoints into final welds (s2)...")
    s2.update_final_welds_with_t_type_breakpoints(
        final_welds_json=CFG.final_welds_file,
        geometry_graph_with_breakpoints_json=CFG.geometry_graph_with_breakpoints_file,
        out_json=CFG.final_welds_with_junctions_file,
        max_remove_length=50.0,
    )
    print(f"[s2] saved -> {CFG.final_welds_with_junctions_file}")

    # 读取最终交付文件（final_welds_with_junctions，与 run_visualize_final_json 一致）
    # 若不存在则回退到 final_welds
    output_path = CFG.final_welds_with_junctions_file
    if not os.path.exists(output_path):
        print(f"Warning: {output_path} not found, falling back to {CFG.final_welds_file}")
        output_path = CFG.final_welds_file

    result_data = load_json(output_path)

    # ------------------------------------------------------------------
    # Step 5（可选）: 生成可视化图片（对应 run_visualize_final_json / s3）
    # ------------------------------------------------------------------
    viz_file = None
    if enable_visualization:
        print(f"[viz] Generating visualization...")
        viz_file = f"{CFG.base_dir}/{CFG.stp_head}_visualization.png"
        success = visualize_and_save(result_data, up_axis, viz_file)
        if not success:
            viz_file = None

    print(f"Processing complete! Output: {output_path}")
    return output_path, viz_file, result_data


@app.post("/process")
async def process_weld(
    file: UploadFile = File(...),
    enable_visualization: bool = False,
):
    """
    上传 STEP 文件并执行完整焊缝处理流程，返回最终的焊缝 JSON。

    处理步骤与 compiler-main.py 菜单一一对应：
      1. 提取接触边界 (export_contact_edges)
      2. 检测过焊孔 + 插入断点 + T型断点 (run_detect_through_holes / s0)
      3. 构建最终焊缝 JSON (run_build_final_json / s1)
      4. 合并 T型断点 (run_merge_split_breakpoints / s2)

    参数:
    - file: STEP 文件 (.stp 或 .step)
    - enable_visualization: 是否生成可视化图片（默认 False）

    返回:
    - JSON 格式的处理结果（final_welds_with_junctions）
    """
    if not (file.filename.endswith(".stp") or file.filename.endswith(".step")):
        raise HTTPException(status_code=400,
                            detail="Only .stp or .step files are supported")

    model_id = str(uuid.uuid4())[:8]
    model_dir = Path(f"model/sub_assembly/{model_id}")
    model_dir.mkdir(parents=True, exist_ok=True)

    file_extension = ".step" if file.filename.endswith(".step") else ".stp"
    step_file_path = model_dir / f"{model_id}{file_extension}"

    try:
        with open(step_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_path, viz_file, result_data = process_step_file(
            model_id, enable_visualization
        )

        response = {
            "status": "success",
            "message": "Processing completed successfully",
            "model_id": model_id,
            "output_file": output_path,
            "result": result_data,
        }
        if viz_file:
            response["visualization_file"] = viz_file

        return response

    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
