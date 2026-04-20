#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils_nurbs import rotate_axis
from vqvae_eval_utils import build_model_input, load_model_from_checkpoint


def load_vqvae_with_kwargs(ckpt_path: str, device):
    loaded = load_model_from_checkpoint(ckpt_path, device)
    if isinstance(loaded, (tuple, list)) and len(loaded) >= 2:
        return loaded[0], loaded[1]
    raise ValueError("load_model_from_checkpoint must return at least (model, model_kwargs)")


def extract_vq_indices(quantize_output):
    """Handle the quantizer tuple layout used by trainer_nurbs.VQVAE."""
    if torch.is_tensor(quantize_output):
        return quantize_output
    if not isinstance(quantize_output, (tuple, list)):
        raise TypeError(f"Unsupported quantize output type: {type(quantize_output)}")

    # Existing project code does: _, _, indices = model.quantize(h); indices[2].reshape(...)
    if len(quantize_output) >= 3:
        candidate = quantize_output[2]
        if torch.is_tensor(candidate):
            return candidate
        if isinstance(candidate, (tuple, list)):
            for item in reversed(candidate):
                if torch.is_tensor(item):
                    return item

    for item in reversed(quantize_output):
        if torch.is_tensor(item):
            return item
        if isinstance(item, (tuple, list)):
            for sub_item in reversed(item):
                if torch.is_tensor(sub_item):
                    return sub_item

    raise TypeError("Could not find tensor VQ indices in quantize output")


def normalize_step_name(path_or_name: str) -> str:
    base = os.path.basename(str(path_or_name).strip())
    for suffix in (".step", ".stp", ".pkl", ".npy"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def load_step_to_model_id(records_json: str | None) -> dict[str, str]:
    if not records_json:
        return {}
    path = Path(records_json)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    mapping: dict[str, str] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        step_file = record.get("step_file")
        if not step_file:
            continue
        model_dir = record.get("model_dir")
        model_id = record.get("model_id")
        if not model_id and model_dir:
            model_id = os.path.basename(os.path.normpath(model_dir))
        if model_id:
            mapping[normalize_step_name(step_file)] = model_id
    return mapping


def load_step_to_model_id_from_root(root: str | None) -> dict[str, str]:
    if not root:
        return {}
    root_path = Path(root)
    if not root_path.exists():
        return {}

    mapping: dict[str, str] = {}
    for model_dir in sorted(root_path.glob("model_*")):
        if not model_dir.is_dir():
            continue
        step_files = sorted(model_dir.glob("*.step")) + sorted(model_dir.glob("*.stp"))
        for step_file in step_files:
            mapping[normalize_step_name(step_file.name)] = model_dir.name
    return mapping


def dfs_face_ordering_from_core(edge_face_pairs, num_faces: int):
    nbrs = [set() for _ in range(num_faces)]
    for f1, f2 in edge_face_pairs:
        if 0 <= f1 < num_faces and 0 <= f2 < num_faces and f1 != f2:
            nbrs[f1].add(f2)
            nbrs[f2].add(f1)
    deg = [len(n) for n in nbrs]
    visited = [False] * num_faces
    face_order: list[int] = []
    seeds = sorted(range(num_faces), key=lambda x: (-deg[x], x))

    def dfs(u: int):
        visited[u] = True
        face_order.append(u)
        neighbors = [v for v in nbrs[u] if not visited[v]]
        neighbors.sort(key=lambda x: (deg[x], x))
        for v in neighbors:
            if not visited[v]:
                dfs(v)

    for seed in seeds:
        if not visited[seed]:
            dfs(seed)

    return face_order, {face_id: pos for pos, face_id in enumerate(face_order)}


def lexicographic_edge_ordering(edge_face_pairs):
    edge_sort_info = []
    for edge_idx, pair in enumerate(edge_face_pairs):
        if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
            continue
        f1, f2 = int(pair[0]), int(pair[1])
        edge_sort_info.append(((max(f1, f2), min(f1, f2)), edge_idx, (f1, f2)))
    edge_sort_info.sort()
    return [item[1] for item in edge_sort_info], [item[2] for item in edge_sort_info]


def bbox_to_corners(bboxes: np.ndarray) -> np.ndarray:
    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 6)
    corners = []
    for bbox in bboxes:
        mn = bbox[:3]
        mx = bbox[3:]
        corners.append(
            np.array(
                [
                    [mn[0], mn[1], mn[2]],
                    [mn[0], mn[1], mx[2]],
                    [mn[0], mx[1], mn[2]],
                    [mn[0], mx[1], mx[2]],
                    [mx[0], mn[1], mn[2]],
                    [mx[0], mn[1], mx[2]],
                    [mx[0], mx[1], mn[2]],
                    [mx[0], mx[1], mx[2]],
                ],
                dtype=np.float32,
            )
        )
    return np.asarray(corners, dtype=np.float32)


def corners_to_bbox(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 8, 3)
    mn = corners.min(axis=1)
    mx = corners.max(axis=1)
    return np.concatenate([mn, mx], axis=1).astype(np.float32)


def rotate_bbox_z(bboxes: np.ndarray, angle: float) -> np.ndarray:
    corners = bbox_to_corners(bboxes)
    rotated = rotate_axis(corners, angle, "z", normalized=False)
    return corners_to_bbox(rotated)


def quantize_bbox(bbox: np.ndarray, num_tokens: int) -> np.ndarray:
    bbox = np.asarray(bbox, dtype=np.float32)
    normalized = np.clip((bbox + 1.0) / 2.0, 0.0, 1.0)
    return np.rint(normalized * (num_tokens - 1)).astype(np.int64)


class NurbsARV2Preprocessor:
    def __init__(self, data_list: str, args):
        self.args = args
        with open(data_list, "rb") as f:
            ds = pickle.load(f)
        self.train_paths = list(ds["train"])
        self.val_paths = list(ds.get("val", []))
        if not self.val_paths:
            split_idx = int(len(self.train_paths) * 0.95)
            self.val_paths = self.train_paths[split_idx:]
            self.train_paths = self.train_paths[:split_idx]

        self.step_to_model_id = load_step_to_model_id(args.records_json)
        if args.pointcloud_root:
            self.step_to_model_id.update(load_step_to_model_id_from_root(args.pointcloud_root))

        self.face_index_size = int(args.max_face)
        self.quantization_size = int(args.quantization_size)
        self.bbox_index_size = int(args.bbox_index_size)
        self.special_token_size = 3

        self.face_index_offset = 0
        self.quantization_offset = self.face_index_offset + self.face_index_size
        self.bbox_token_offset = self.quantization_offset + self.quantization_size
        self.START_TOKEN = self.bbox_token_offset + self.bbox_index_size
        self.SEP_TOKEN = self.START_TOKEN + 1
        self.END_TOKEN = self.START_TOKEN + 2
        self.PAD_TOKEN = self.START_TOKEN + 3

        self.face_vq_tokens = 4
        self.edge_vq_tokens = 4
        self.bbox_tokens_per_element = 6
        self.face_block = self.bbox_tokens_per_element + self.face_vq_tokens + 1
        self.edge_block = 2 + self.bbox_tokens_per_element + self.edge_vq_tokens
        # Keep the old convention: vocab_size excludes PAD, dataset uses PAD=vocab_size.
        self.vocab_size = self.PAD_TOKEN

        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.model_kwargs = load_vqvae_with_kwargs(args.ckpt_path, self.device)
        self.model.eval()
        print("V2 preprocessor: VQ-VAE checkpoint loaded.")

        self.group_cache: list[tuple[str, dict]] = []
        self.stats = {
            "missing_controls": 0,
            "missing_bbox_or_topology": 0,
            "bad_shape": 0,
            "too_large": 0,
            "encode_failed": 0,
            "missing_model_id": 0,
        }
        self.error_examples: dict[str, list[str]] = {key: [] for key in self.stats}
        self._process_all_data()

    def _process_all_data(self):
        max_cad_count = self.args.max_cad_count
        train_count = 0
        for path in tqdm(self.train_paths, desc="Processing train"):
            if max_cad_count is not None and train_count >= max_cad_count:
                break
            group = self._process_single_cad(path, "train")
            if group:
                self.group_cache.append(("train", group))
                train_count += 1

        val_count = 0
        for path in tqdm(self.val_paths, desc="Processing val"):
            if max_cad_count is not None and val_count >= max_cad_count:
                break
            group = self._process_single_cad(path, "val")
            if group:
                self.group_cache.append(("val", group))
                val_count += 1

    def _encode_controls(self, controls: np.ndarray, item_type: str) -> list[list[int]]:
        if len(controls) == 0:
            return []
        tensors = []
        for control in controls:
            x_in, _ = build_model_input(control, item_type, self.model_kwargs["in_channels"])
            tensors.append(x_in)
        x = torch.cat(tensors, dim=0).to(self.device)
        with torch.no_grad():
            h = self.model.encoder(x)
            h = self.model.quant_conv(h)
            indices = extract_vq_indices(self.model.quantize(h))
            indices = indices.reshape(len(controls), -1)
        if indices.shape[1] != 4:
            raise ValueError(f"Expected 4 VQ tokens per {item_type}, got {indices.shape[1]}")
        return indices.detach().cpu().long().tolist()

    def _model_id_for_path(self, path: str) -> str | None:
        step_name = normalize_step_name(path)
        model_id = self.step_to_model_id.get(step_name)
        if model_id:
            return model_id
        self.stats["missing_model_id"] += 1
        self._record_example("missing_model_id", path)
        return None

    def _record_example(self, key: str, message: str) -> None:
        examples = self.error_examples.setdefault(key, [])
        if len(examples) < 5:
            examples.append(str(message))

    def _encode_single_cad(
        self,
        face_ctrs: np.ndarray,
        edge_ctrs: np.ndarray,
        surf_bbox_wcs: np.ndarray,
        edge_bbox_wcs: np.ndarray,
        edge_face_adj: np.ndarray,
    ):
        num_face = len(face_ctrs)
        num_edge = len(edge_ctrs)
        edge_face_pairs = []
        for adj in edge_face_adj:
            if len(adj) >= 2:
                edge_face_pairs.append((int(adj[0]), int(adj[1])))
        if len(edge_face_pairs) != num_edge:
            raise ValueError(f"edgeFace_adj count {len(edge_face_pairs)} != num_edge {num_edge}")

        face_order, face_position_map = dfs_face_ordering_from_core(edge_face_pairs, num_face)
        face_ctrs = face_ctrs[face_order]
        surf_bbox_wcs = surf_bbox_wcs[face_order]

        remapped_pairs = []
        for f1, f2 in edge_face_pairs:
            remapped_pairs.append((face_position_map.get(f1, f1), face_position_map.get(f2, f2)))

        edge_order, ordered_edge_face_pairs = lexicographic_edge_ordering(remapped_pairs)
        edge_ctrs = edge_ctrs[edge_order]
        edge_bbox_wcs = edge_bbox_wcs[edge_order]

        face_tokens = self._encode_controls(face_ctrs.reshape(num_face, 16, 3), "face")
        edge_tokens = self._encode_controls(edge_ctrs.reshape(num_edge, 4, 3), "edge")

        r = random.randint(0, self.face_index_size - 1)
        face_index_map = {i: (i + r) % self.face_index_size for i in range(num_face)}

        tokens = [self.START_TOKEN]
        for i in range(num_face):
            tokens.extend(int(self.bbox_token_offset + x) for x in quantize_bbox(surf_bbox_wcs[i], self.bbox_index_size))
            tokens.extend(int(self.quantization_offset + x) for x in face_tokens[i])
            tokens.append(self.face_index_offset + face_index_map[i])

        tokens.append(self.SEP_TOKEN)
        for edge_idx, (src, dst) in enumerate(ordered_edge_face_pairs):
            tokens.append(self.face_index_offset + face_index_map[src])
            tokens.append(self.face_index_offset + face_index_map[dst])
            tokens.extend(int(self.bbox_token_offset + x) for x in quantize_bbox(edge_bbox_wcs[edge_idx], self.bbox_index_size))
            tokens.extend(int(self.quantization_offset + x) for x in edge_tokens[edge_idx])

        tokens.append(self.END_TOKEN)
        return tokens, [1] * len(tokens)

    def _process_single_cad(self, path: str, split: str):
        try:
            with open(path, "rb") as f:
                cad = pickle.load(f)

            face_ctrs = np.asarray(cad.get("face_ctrs"), dtype=np.float32)
            edge_ctrs = np.asarray(cad.get("edge_ctrs"), dtype=np.float32)
            surf_bbox_wcs = np.asarray(cad.get("surf_bbox_wcs"), dtype=np.float32)
            edge_bbox_wcs = np.asarray(cad.get("edge_bbox_wcs"), dtype=np.float32)
            edge_face_adj = np.asarray(cad.get("edgeFace_adj"), dtype=np.int64)

            if face_ctrs.size == 0 or edge_ctrs.size == 0:
                self.stats["missing_controls"] += 1
                self._record_example("missing_controls", path)
                return None
            if surf_bbox_wcs.size == 0 or edge_bbox_wcs.size == 0 or edge_face_adj.size == 0:
                self.stats["missing_bbox_or_topology"] += 1
                self._record_example("missing_bbox_or_topology", path)
                return None
            if not (np.isfinite(face_ctrs).all() and np.isfinite(edge_ctrs).all()):
                self.stats["bad_shape"] += 1
                self._record_example("bad_shape", f"{path}: non-finite controls")
                return None

            try:
                face_ctrs = face_ctrs.reshape(-1, 16, 3)
                edge_ctrs = edge_ctrs.reshape(-1, 4, 3)
                surf_bbox_wcs = surf_bbox_wcs.reshape(-1, 6)
                edge_bbox_wcs = edge_bbox_wcs.reshape(-1, 6)
                edge_face_adj = edge_face_adj.reshape(-1, 2)
            except Exception as exc:
                self.stats["bad_shape"] += 1
                self._record_example(
                    "bad_shape",
                    f"{path}: face={face_ctrs.shape}, edge={edge_ctrs.shape}, "
                    f"surf_bbox={surf_bbox_wcs.shape}, edge_bbox={edge_bbox_wcs.shape}, "
                    f"edgeFace={edge_face_adj.shape}, error={exc}",
                )
                return None

            if len(face_ctrs) > self.args.max_face or len(edge_ctrs) > self.args.max_edge:
                self.stats["too_large"] += 1
                self._record_example("too_large", f"{path}: faces={len(face_ctrs)}, edges={len(edge_ctrs)}")
                return None
            if len(surf_bbox_wcs) != len(face_ctrs) or len(edge_bbox_wcs) != len(edge_ctrs):
                self.stats["bad_shape"] += 1
                self._record_example(
                    "bad_shape",
                    f"{path}: faces={len(face_ctrs)}, edges={len(edge_ctrs)}, "
                    f"surf_bbox={len(surf_bbox_wcs)}, edge_bbox={len(edge_bbox_wcs)}",
                )
                return None

            rotation_angles = [0, 90, 180, 270] if (split == "train" and self.args.aug) else [0]
            model_id = self._model_id_for_path(path) or normalize_step_name(path)

            group = {
                "name": model_id,
                "file_name": model_id,
                "source_file": path,
                "step_name": normalize_step_name(path),
                "original": None,
                "augmented": [],
            }

            for rot in rotation_angles:
                cur_face_ctrs = face_ctrs.copy()
                cur_edge_ctrs = edge_ctrs.copy()
                cur_surf_bbox = surf_bbox_wcs.copy()
                cur_edge_bbox = edge_bbox_wcs.copy()
                if rot != 0:
                    cur_face_ctrs = rotate_axis(cur_face_ctrs, rot, "z", normalized=False)
                    cur_edge_ctrs = rotate_axis(cur_edge_ctrs, rot, "z", normalized=False)
                    cur_surf_bbox = rotate_bbox_z(cur_surf_bbox, rot)
                    cur_edge_bbox = rotate_bbox_z(cur_edge_bbox, rot)

                tokens, attn = self._encode_single_cad(
                    cur_face_ctrs, cur_edge_ctrs, cur_surf_bbox, cur_edge_bbox, edge_face_adj
                )
                item = {"input_ids": tokens, "attention_mask": attn}
                if rot == 0:
                    group["original"] = item
                else:
                    group["augmented"].append(item)

            if group["original"] is None:
                return None
            if not group["augmented"]:
                group.pop("augmented", None)
            return group
        except Exception as exc:
            self.stats["encode_failed"] += 1
            self._record_example("encode_failed", f"{path}: {type(exc).__name__}: {exc}")
            return None

    def metadata(self):
        return {
            "vocab_size": self.vocab_size,
            "special_token_size": self.special_token_size,
            "face_index_size": self.face_index_size,
            "quantization_size": self.quantization_size,
            "bbox_index_size": self.bbox_index_size,
            "bbox_size": self.bbox_index_size,
            "face_index_offset": self.face_index_offset,
            "quantization_offset": self.quantization_offset,
            "se_token_offset": self.quantization_offset,
            "bbox_token_offset": self.bbox_token_offset,
            "se_codebook_size": self.quantization_size,
            "face_block": self.face_block,
            "edge_block": self.edge_block,
            "face_vq_tokens": self.face_vq_tokens,
            "edge_vq_tokens": self.edge_vq_tokens,
            "se_tokens_per_element": self.face_vq_tokens,
            "bbox_tokens_per_element": self.bbox_tokens_per_element,
            "protocol": "nurbs_v2_bbox_vq",
            "special_tokens": {
                "START_TOKEN": self.START_TOKEN,
                "SEP_TOKEN": self.SEP_TOKEN,
                "END_TOKEN": self.END_TOKEN,
                "PAD_TOKEN": self.PAD_TOKEN,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Build NURBS AR v2 sequences with bbox + VQ tokens.")
    parser.add_argument("--data_list", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/checkpoint/se/abc/8192,4096,128,64,false,1e-4,0,p/deepcad_nurbs_vqvae_best.pt",
    )
    parser.add_argument("--records_json", type=str, default=None)
    parser.add_argument(
        "--pointcloud_root",
        type=str,
        default=None,
        help="Optional ABC_Dataset_NEW root. Used to map parsed STEP names back to model_XXXXX point-cloud dirs.",
    )
    parser.add_argument("--max_face", type=int, default=50)
    parser.add_argument("--max_edge", type=int, default=124)
    parser.add_argument("--quantization_size", type=int, default=1024)
    parser.add_argument("--bbox_index_size", type=int, default=2048)
    parser.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_cad_count", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    processor = NurbsARV2Preprocessor(args.data_list, args)

    train_groups, val_groups = [], []
    for split, group in processor.group_cache:
        if split == "train":
            train_groups.append(group)
        elif split == "val":
            val_groups.append(group)

    output_data = {
        "train": train_groups,
        "val": val_groups,
        **processor.metadata(),
        "preprocess_stats": processor.stats,
        "preprocess_error_examples": processor.error_examples,
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(output_data, f)

    print("\nDone: NURBS v2 sequences saved")
    print(f"output_file: {out_path}")
    print(f"train: {len(train_groups)}")
    print(f"val: {len(val_groups)}")
    print(f"vocab(no PAD): {processor.vocab_size}")
    print(f"PAD: {processor.PAD_TOKEN}")
    print(f"face_block: {processor.face_block}, edge_block: {processor.edge_block}")
    print(f"stats: {processor.stats}")
    print(f"error_examples: {processor.error_examples}")


if __name__ == "__main__":
    main()
