from __future__ import annotations

import hashlib
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
    lateral_margin: float = 0.05
    bottom_margin: float = 0.03
    top_margin: float = 0.10
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


def _as_scalar(data: np.lib.npyio.NpzFile, key: str, default: float | None = None) -> float | None:
    if key not in data:
        return default
    value = np.asarray(data[key], dtype=float)
    if value.size == 0:
        return default
    return float(value.reshape(-1)[0])


def _as_vec3(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    value = np.asarray(data[key], dtype=float).reshape(-1)
    if value.size != 3:
        return None
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_sdf_matches(
    npz_path: Path,
    stl_path: Path,
    scale: float,
    z_offset: float,
    local_offset: tuple[float, float, float],
    config: SDFBuildConfig,
) -> tuple[bool, str]:
    try:
        data = np.load(npz_path)
    except Exception as exc:
        return False, f"failed to read cached SDF metadata ({exc})"

    cached_scale = _as_scalar(data, "workpiece_scale")
    cached_z_offset = _as_scalar(data, "workpiece_z_offset")
    cached_pitch = _as_scalar(data, "voxel_pitch")
    cached_lateral_margin = _as_scalar(data, "lateral_margin")
    cached_bottom_margin = _as_scalar(data, "bottom_margin")
    legacy_margin = _as_scalar(data, "margin")
    if cached_lateral_margin is None:
        cached_lateral_margin = legacy_margin
    if cached_bottom_margin is None:
        cached_bottom_margin = legacy_margin
    cached_top_margin = _as_scalar(data, "top_margin", legacy_margin)
    cached_offset = _as_vec3(data, "workpiece_offset")
    cached_sha = str(data["workpiece_source_sha256"].reshape(-1)[0]) if "workpiece_source_sha256" in data else None

    current_sha = _file_sha256(stl_path)
    if cached_sha is None or cached_sha != current_sha:
        return False, "workpiece mesh content changed or cache has no source hash"
    if cached_scale is None or not np.isclose(cached_scale, float(scale), atol=1e-12):
        return False, f"scale mismatch: cached={cached_scale}, requested={scale}"
    if cached_z_offset is None or not np.isclose(cached_z_offset, float(z_offset), atol=1e-12):
        return False, f"z_offset mismatch: cached={cached_z_offset}, requested={z_offset}"
    if cached_pitch is None or not np.isclose(cached_pitch, float(config.voxel_pitch), atol=1e-12):
        return False, f"voxel_pitch mismatch: cached={cached_pitch}, requested={config.voxel_pitch}"
    if cached_lateral_margin is None or not np.isclose(cached_lateral_margin, float(config.lateral_margin), atol=1e-12):
        return False, f"lateral_margin mismatch: cached={cached_lateral_margin}, requested={config.lateral_margin}"
    if cached_bottom_margin is None or not np.isclose(cached_bottom_margin, float(config.bottom_margin), atol=1e-12):
        return False, f"bottom_margin mismatch: cached={cached_bottom_margin}, requested={config.bottom_margin}"
    if cached_top_margin is None or not np.isclose(cached_top_margin, float(config.top_margin), atol=1e-12):
        return False, f"top_margin mismatch: cached={cached_top_margin}, requested={config.top_margin}"
    requested_offset = np.asarray(local_offset, dtype=float).reshape(3)
    if cached_offset is None or not np.allclose(cached_offset, requested_offset, atol=1e-12):
        return False, f"workpiece_offset mismatch: cached={cached_offset}, requested={requested_offset}"
    return True, "metadata matches"


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
    lateral_margin = float(config.lateral_margin)
    bottom_margin = float(config.bottom_margin)
    top_margin = float(config.top_margin)
    if pitch <= 0.0:
        raise ValueError(f"voxel_pitch must be positive, got {pitch}")
    if lateral_margin < 0.0:
        raise ValueError(f"lateral_margin must be non-negative, got {lateral_margin}")
    if bottom_margin < 0.0:
        raise ValueError(f"bottom_margin must be non-negative, got {bottom_margin}")
    if top_margin < 0.0:
        raise ValueError(f"top_margin must be non-negative, got {top_margin}")

    mesh = _world_transform_mesh(stl_path, scale, z_offset, local_offset)
    bounds = np.asarray(mesh.bounds, dtype=float)
    lower_padding = np.array([lateral_margin, lateral_margin, bottom_margin], dtype=float)
    upper_padding = np.array([lateral_margin, lateral_margin, top_margin], dtype=float)
    world_min = bounds[0] - lower_padding
    world_max = bounds[1] + upper_padding

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
    np.savez_compressed(
        npz_path,
        sdf=sdf,
        x=x,
        y=y,
        z=z,
        workpiece_scale=np.array([float(scale)], dtype=float),
        workpiece_z_offset=np.array([float(z_offset)], dtype=float),
        workpiece_offset=np.asarray(local_offset, dtype=float).reshape(3),
        workpiece_source_sha256=np.array([_file_sha256(stl_path)]),
        workpiece_source_path=np.array([str(stl_path.resolve())]),
        voxel_pitch=np.array([pitch], dtype=float),
        lateral_margin=np.array([lateral_margin], dtype=float),
        bottom_margin=np.array([bottom_margin], dtype=float),
        top_margin=np.array([top_margin], dtype=float),
    )
    if logger is not None:
        logger(
            f"[SDF] Built workpiece SDF at {npz_path} "
            f"(shape={grid_shape}, pitch={pitch:.4f} m, lateral_margin={lateral_margin:.4f} m, "
            f"bottom_margin={bottom_margin:.4f} m, top_margin={top_margin:.4f} m, "
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
    needs_rebuild = bool(rebuild) or not npz_path.exists()
    if not needs_rebuild:
        metadata_ok, reason = _cached_sdf_matches(
            npz_path=npz_path,
            stl_path=stl_path,
            scale=scale,
            z_offset=z_offset,
            local_offset=local_offset,
            config=config,
        )
        if not metadata_ok:
            needs_rebuild = True
            if logger is not None:
                logger(
                    f"[SDF] Cached workpiece SDF at {npz_path} does not match current workpiece transform/config; "
                    f"rebuilding ({reason})."
                )

    if needs_rebuild:
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
