from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, map_coordinates


Logger = Callable[[str], None]


@dataclass(frozen=True)
class SDFBuildConfig:
    voxel_pitch: float = 0.004
    margin: float = 0.03
    voxelize_method: str = "subdivide"
    voxelize_max_iter: int = 64


class SDFCollisionLayer:
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.sdf_grid = np.asarray(data["sdf"], dtype=float)
        self.x = np.asarray(data["x"], dtype=float)
        self.y = np.asarray(data["y"], dtype=float)
        self.z = np.asarray(data["z"], dtype=float)
        self.origin = np.array([self.x[0], self.y[0], self.z[0]], dtype=float)
        self.spacing = np.array(
            [
                self.x[1] - self.x[0] if len(self.x) > 1 else 1.0,
                self.y[1] - self.y[0] if len(self.y) > 1 else 1.0,
                self.z[1] - self.z[0] if len(self.z) > 1 else 1.0,
            ],
            dtype=float,
        )
        median_val = float(np.median(self.sdf_grid))
        if median_val < 0.0:
            self.sdf_grid = -self.sdf_grid

    def get_distances(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return np.array([], dtype=float)
        indices = ((pts - self.origin[None, :]) / self.spacing[None, :]).T
        outside_value = float(np.max(self.sdf_grid)) + float(np.max(np.abs(self.spacing))) * 2.0
        return map_coordinates(
            self.sdf_grid,
            indices,
            order=1,
            mode="constant",
            cval=outside_value,
        )


def _world_transform_mesh(
    stl_path: Path,
    scale: float,
    z_offset: float,
    local_offset: tuple[float, float, float],
) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    mesh.apply_translation(
        np.array(
            [
                float(local_offset[0]),
                float(local_offset[1]),
                float(local_offset[2] + z_offset),
            ],
            dtype=float,
        )
    )
    return mesh


def build_sdf_from_workpiece_mesh(
    stl_path: Path,
    scale: float,
    z_offset: float,
    local_offset: tuple[float, float, float],
    config: SDFBuildConfig,
    npz_path: Path,
    logger: Logger | None = None,
) -> Path:
    pitch = float(config.voxel_pitch)
    margin = float(config.margin)
    if pitch <= 0.0:
        raise ValueError(f"voxel_pitch must be positive, got {pitch}")

    mesh = _world_transform_mesh(stl_path, scale, z_offset, local_offset)
    bounds = np.asarray(mesh.bounds, dtype=float)
    world_min = bounds[0] - margin
    world_max = bounds[1] + margin

    try:
        filled = mesh.voxelized(
            pitch,
            method=config.voxelize_method,
            max_iter=int(max(1, config.voxelize_max_iter)),
        ).fill()
    except ValueError as exc:
        raise RuntimeError(
            f"SDF voxelization failed for {stl_path} with pitch={pitch:.6f} m, "
            f"method={config.voxelize_method}, max_iter={config.voxelize_max_iter}. "
            "Try increasing voxelize_max_iter or using a coarser voxel pitch."
        ) from exc
    occupied_points = np.asarray(filled.points, dtype=float)
    if occupied_points.size == 0:
        raise RuntimeError(f"Voxelization produced an empty occupancy set for {stl_path}")

    x = np.arange(world_min[0], world_max[0] + pitch * 0.5, pitch, dtype=float)
    y = np.arange(world_min[1], world_max[1] + pitch * 0.5, pitch, dtype=float)
    z = np.arange(world_min[2], world_max[2] + pitch * 0.5, pitch, dtype=float)
    grid_shape = (len(x), len(y), len(z))
    occupancy = np.zeros(grid_shape, dtype=bool)

    indices = np.rint((occupied_points - np.array([x[0], y[0], z[0]], dtype=float)) / pitch).astype(int)
    valid_mask = np.all((indices >= 0) & (indices < np.array(grid_shape, dtype=int)[None, :]), axis=1)
    indices = indices[valid_mask]
    occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    outside = distance_transform_edt(~occupancy, sampling=(pitch, pitch, pitch))
    inside = distance_transform_edt(occupancy, sampling=(pitch, pitch, pitch))
    sdf = outside.astype(float)
    sdf[occupancy] = -inside[occupancy]

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, sdf=sdf, x=x, y=y, z=z)
    if logger is not None:
        logger(
            f"[SDF] Built workpiece SDF at {npz_path} "
            f"(shape={grid_shape}, pitch={pitch:.4f} m, margin={margin:.4f} m, "
            f"method={config.voxelize_method}, max_iter={config.voxelize_max_iter})"
        )
    return npz_path


def load_or_build_workpiece_sdf(
    stl_path: Path,
    scale: float,
    z_offset: float,
    local_offset: tuple[float, float, float],
    npz_path: Path,
    config: SDFBuildConfig,
    logger: Logger | None = None,
    rebuild: bool = False,
) -> tuple[SDFCollisionLayer, Path]:
    if rebuild or not npz_path.exists():
        build_sdf_from_workpiece_mesh(
            stl_path=stl_path,
            scale=scale,
            z_offset=z_offset,
            local_offset=local_offset,
            config=config,
            npz_path=npz_path,
            logger=logger,
        )
    else:
        if logger is not None:
            logger(f"[SDF] Loading cached workpiece SDF from {npz_path}")
    return SDFCollisionLayer(npz_path), npz_path
