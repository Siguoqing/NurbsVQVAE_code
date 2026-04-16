from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NurbsDatasetSummary:
    split: str
    num_files: int
    num_items: int
    num_faces: int
    num_edges: int
    invalid_faces: int
    invalid_edges: int
    failed_files: int


class NurbsVQDataset(Dataset):
    """Flat dataset of 4x4 NURBS face grids and stretched edge controls."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        split_ratio: float = 0.95,
        seed: int = 42,
        max_files: Optional[int] = None,
        use_type_flag: bool = True,
        include_faces: bool = True,
        include_edges: bool = True,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if not include_faces and not include_edges:
            raise ValueError("At least one of include_faces/include_edges must be True")

        self.data_dir = data_dir
        self.split = split
        self.use_type_flag = use_type_flag
        self.include_faces = include_faces
        self.include_edges = include_edges
        self.items: List[Tuple[str, np.ndarray]] = []

        all_pkl_files = self._collect_pkl_files(data_dir)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_pkl_files)

        if max_files is not None:
            all_pkl_files = all_pkl_files[: max(0, int(max_files))]

        split_idx = int(len(all_pkl_files) * split_ratio)
        data_paths = all_pkl_files[:split_idx] if split == "train" else all_pkl_files[split_idx:]

        num_faces = 0
        num_edges = 0
        invalid_faces = 0
        invalid_edges = 0
        failed_files = 0

        for path in data_paths:
            try:
                with open(path, "rb") as f:
                    cad = pickle.load(f)
            except Exception:
                failed_files += 1
                continue

            if self.include_faces:
                added, invalid = self._append_controls(
                    cad=cad,
                    candidate_keys=("face_ctrs_wcs_norm", "face_ctrs"),
                    item_type="face",
                    expected_rows=16,
                )
                num_faces += added
                invalid_faces += invalid

            if self.include_edges:
                added, invalid = self._append_controls(
                    cad=cad,
                    candidate_keys=("edge_ctrs_wcs_norm", "edge_ctrs"),
                    item_type="edge",
                    expected_rows=4,
                )
                num_edges += added
                invalid_edges += invalid

        self.summary = NurbsDatasetSummary(
            split=split,
            num_files=len(data_paths),
            num_items=len(self.items),
            num_faces=num_faces,
            num_edges=num_edges,
            invalid_faces=invalid_faces,
            invalid_edges=invalid_edges,
            failed_files=failed_files,
        )

    @staticmethod
    def _collect_pkl_files(data_dir: str) -> List[str]:
        all_pkl_files: List[str] = []
        for root, _, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith(".pkl"):
                    all_pkl_files.append(os.path.join(root, filename))
        all_pkl_files.sort()
        return all_pkl_files

    def _append_controls(
        self,
        cad: dict,
        candidate_keys: Sequence[str],
        item_type: str,
        expected_rows: int,
    ) -> Tuple[int, int]:
        raw = None
        for key in candidate_keys:
            raw = cad.get(key)
            if raw is not None:
                break

        if raw is None:
            return 0, 0

        try:
            array = np.asarray(raw, dtype=np.float32)
        except Exception:
            return 0, 0

        if array.ndim != 3 or array.shape[-2:] != (expected_rows, 3):
            return 0, 0

        added = 0
        invalid = 0
        for controls in array:
            if np.isfinite(controls).all():
                self.items.append((item_type, controls.astype(np.float32, copy=False)))
                added += 1
            else:
                invalid += 1
        return added, invalid

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item_type, item_data = self.items[idx]

        if item_type == "face":
            grid = item_data.reshape(4, 4, 3)
            flag = np.zeros((4, 4, 1), dtype=np.float32)
        else:
            grid = np.tile(item_data, (4, 1)).reshape(4, 4, 3)
            flag = np.ones((4, 4, 1), dtype=np.float32)

        sample = np.concatenate([grid, flag], axis=-1) if self.use_type_flag else grid
        return torch.from_numpy(sample.copy()).float()


def build_nurbs_train_val_datasets(
    data_dir: str,
    split_ratio: float = 0.95,
    seed: int = 42,
    max_files: Optional[int] = None,
    use_type_flag: bool = True,
    include_faces: bool = True,
    include_edges: bool = True,
) -> Tuple[NurbsVQDataset, NurbsVQDataset]:
    common_kwargs = {
        "data_dir": data_dir,
        "split_ratio": split_ratio,
        "seed": seed,
        "max_files": max_files,
        "use_type_flag": use_type_flag,
        "include_faces": include_faces,
        "include_edges": include_edges,
    }
    train_dataset = NurbsVQDataset(split="train", **common_kwargs)
    val_dataset = NurbsVQDataset(split="val", **common_kwargs)
    return train_dataset, val_dataset


def format_dataset_summary(summary: NurbsDatasetSummary) -> List[str]:
    return [
        f"{summary.split} dataset: {summary.num_items} items from {summary.num_files} files",
        f"  faces={summary.num_faces}, edges={summary.num_edges}",
        f"  invalid_faces={summary.invalid_faces}, invalid_edges={summary.invalid_edges}, failed_files={summary.failed_files}",
    ]
