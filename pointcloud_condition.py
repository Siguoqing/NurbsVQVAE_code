from __future__ import annotations

import os
from typing import Optional

import numpy as np


def pc_normalize(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.astype(np.float32, copy=False)

    points = points.astype(np.float32, copy=False)
    centroid = np.mean(points, axis=0, keepdims=True)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    if np.isfinite(scale) and scale > 1e-8:
        points = points / scale
    return points


def sample_or_repeat_points(points: np.ndarray, num_points: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    if len(points) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    if len(points) >= num_points:
        indices = rng.choice(len(points), size=num_points, replace=False)
    else:
        indices = rng.choice(len(points), size=num_points, replace=True)
    return points[indices].astype(np.float32, copy=False)


def preprocess_point_cloud_array(array: np.ndarray, num_points: int, normalize: bool = True) -> np.ndarray:
    points = np.asarray(array, dtype=np.float32)

    if points.ndim == 4 and points.shape[-1] >= 3:
        points = points.reshape(-1, points.shape[-1])
    elif points.ndim == 3 and points.shape[-1] >= 3:
        points = points.reshape(-1, points.shape[-1])
    elif points.ndim != 2 or points.shape[-1] < 3:
        raise ValueError(f"Unsupported point-cloud shape: {points.shape}")

    points = points[:, :3]
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    if normalize:
        points = pc_normalize(points)

    return sample_or_repeat_points(points, num_points=num_points)


def resolve_point_cloud_path(point_cloud_dir: str, file_name: str) -> Optional[str]:
    candidates = [
        os.path.join(point_cloud_dir, f"{file_name}.npy"),
        os.path.join(point_cloud_dir, file_name, "point_cloud.npy"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def load_and_preprocess_point_cloud(source_path: str, num_points: int, normalize: bool = True) -> np.ndarray:
    array = np.load(source_path, allow_pickle=True)
    return preprocess_point_cloud_array(array, num_points=num_points, normalize=normalize)
