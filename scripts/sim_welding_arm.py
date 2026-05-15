"""Launch a minimal Isaac Sim scene for the UR5e welding arm.

The robot is imported from the URDF asset in this repository.  The URDF uses
``package://urdf-pen`` mesh references, so this script creates a temporary URDF
with those references resolved to local mesh files before handing it to Isaac
Sim's URDF importer.
"""

from __future__ import annotations

import argparse
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "source/weldRobot/weldRobot/assets/robot-model/ur5e_with_pen.urdf"
DEFAULT_INITIAL_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.57,
    "elbow_joint": 1.57,
    "wrist_1_joint": -1.57,
    "wrist_2_joint": -1.57,
    "wrist_3_joint": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal Isaac Sim UR5e welding-arm scene.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Path to the UR5e-with-pen URDF file.")
    parser.add_argument("--robot-prim-path", default="/World/UR5ePen", help="USD prim path for the imported robot.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without the UI.")
    parser.add_argument("--floating", action="store_true", help="Do not fix the robot base to the world.")
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0, help="Physics timestep in seconds.")
    parser.add_argument("--rendering-dt", type=float, default=1.0 / 60.0, help="Rendering timestep in seconds.")
    return parser.parse_args()


def make_resolved_urdf(source_urdf: Path) -> Path:
    """Create a temporary URDF whose package mesh paths point at this repo."""
    if not source_urdf.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {source_urdf}")

    robot_model_dir = source_urdf.parent.resolve()
    tree = ET.parse(source_urdf)
    root = tree.getroot()

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and filename.startswith("package://urdf-pen/"):
            local_path = robot_model_dir / filename.removeprefix("package://urdf-pen/")
            mesh.set("filename", str(local_path))

    temp_dir = Path(tempfile.mkdtemp(prefix="ur5e_pen_urdf_"))
    resolved_urdf = temp_dir / source_urdf.name
    tree.write(resolved_urdf, encoding="utf-8", xml_declaration=True)
    return resolved_urdf


def enable_extension(extension_name: str) -> bool:
    """Enable an Omniverse extension if it is available."""
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    extension_id = None
    if hasattr(manager, "get_enabled_extension_id"):
        extension_id = manager.get_enabled_extension_id(extension_name)
    if extension_id:
        return True

    if hasattr(manager, "get_extension_id_by_module"):
        extension_id = manager.get_extension_id_by_module(extension_name)
    if not extension_id and hasattr(manager, "get_extension_id_by_name"):
        extension_id = manager.get_extension_id_by_name(extension_name)
    if not extension_id:
        return False

    manager.set_extension_enabled_immediate(extension_id, True)
    return True


def acquire_urdf_module():
    """Load the URDF importer module across Isaac Sim naming variants."""
    for extension_name in ("isaacsim.asset.importer.urdf", "omni.importer.urdf", "omni.isaac.urdf"):
        enable_extension(extension_name)

    try:
        from isaacsim.asset.importer.urdf import _urdf

        return _urdf
    except ImportError:
        from omni.importer.urdf import _urdf

        return _urdf


def configure_urdf_import(_urdf, fix_base: bool):
    import_config = _urdf.ImportConfig()
    options = {
        "merge_fixed_joints": False,
        "fix_base": fix_base,
        "import_inertia_tensor": True,
        "convex_decomp": False,
        "self_collision": False,
        "distance_scale": 1.0,
        "make_default_prim": False,
        "default_drive_strength": 400.0,
        "default_position_drive_damping": 40.0,
    }
    for name, value in options.items():
        setter = getattr(import_config, f"set_{name}", None)
        if setter is not None:
            setter(value)
        elif hasattr(import_config, name):
            setattr(import_config, name, value)
    return import_config


def import_robot_from_urdf(urdf_path: Path, prim_path: str, fix_base: bool) -> str:
    """Import the URDF and return the robot prim path."""
    _urdf = acquire_urdf_module()
    import_config = configure_urdf_import(_urdf, fix_base=fix_base)
    urdf_interface = _urdf.acquire_urdf_interface()

    root_path = str(urdf_path.parent)
    file_name = urdf_path.name
    parsed_robot = urdf_interface.parse_urdf(root_path, file_name, import_config)
    if isinstance(parsed_robot, tuple):
        success, parsed_robot = parsed_robot
        if not success:
            raise RuntimeError(f"Isaac Sim failed to parse URDF: {urdf_path}")

    imported_prim_path = urdf_interface.import_robot(root_path, file_name, parsed_robot, import_config, "")
    if isinstance(imported_prim_path, tuple):
        imported_prim_path = imported_prim_path[-1]
    if not imported_prim_path:
        imported_prim_path = prim_path
    return str(imported_prim_path)


def set_initial_joint_positions(robot_prim_path: str) -> None:
    try:
        from isaacsim.core.prims import SingleArticulation
    except ImportError:
        from omni.isaac.core.articulations import Articulation as SingleArticulation

    robot = SingleArticulation(prim_path=robot_prim_path, name="ur5e_pen")
    robot.initialize()

    joint_names = list(robot.dof_names)
    joint_positions = robot.get_joint_positions()
    for joint_name, target_pos in DEFAULT_INITIAL_JOINT_POS.items():
        if joint_name in joint_names:
            joint_positions[joint_names.index(joint_name)] = target_pos
    robot.set_joint_positions(joint_positions)


def add_camera_view() -> None:
    import omni.kit.commands
    from pxr import Gf, UsdGeom

    omni.kit.commands.execute(
        "CreatePrimWithDefaultXform",
        prim_type="Camera",
        prim_path="/World/Camera",
        attributes={
            "focusDistance": 3.0,
            "focalLength": 28.0,
        },
    )

    from omni.usd import get_context

    stage = get_context().get_stage()
    camera = stage.GetPrimAtPath("/World/Camera")
    UsdGeom.XformCommonAPI(camera).SetTranslate(Gf.Vec3d(1.8, -2.2, 1.4))
    UsdGeom.XformCommonAPI(camera).SetRotate(
        Gf.Vec3f(60.0, 0.0, 40.0), UsdGeom.XformCommonAPI.RotationOrderXYZ
    )

    viewport_ext = "omni.kit.viewport.utility"
    if enable_extension(viewport_ext):
        try:
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().camera_path = "/World/Camera"
        except Exception:
            pass


def main() -> None:
    args = parse_args()

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    try:
        try:
            from isaacsim.core.api import World
        except ImportError:
            from omni.isaac.core import World

        world = World(physics_dt=args.physics_dt, rendering_dt=args.rendering_dt, stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        resolved_urdf = make_resolved_urdf(args.urdf)
        robot_prim_path = import_robot_from_urdf(resolved_urdf, args.robot_prim_path, fix_base=not args.floating)

        world.reset()
        set_initial_joint_positions(robot_prim_path)
        add_camera_view()

        print(f"[weldRobot] Isaac Sim scene ready: {robot_prim_path}")
        print("[weldRobot] Press Ctrl+C in the terminal to stop.")
        while simulation_app.is_running():
            world.step(render=True)
            time.sleep(0.0)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
