import argparse
import glob
import json
import os
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from vqvae_eval_utils import (
    load_controls,
    load_model_from_checkpoint,
    pick_candidate_keys,
    reconstruct_controls,
)


DEFAULT_DATA_DIR = "/mnt/docker_dir/lijiahao/NurbsVQVAE_code/furniture_parsed/ABC_Dataset_NEW"

def _collect_split_files(data_dir: str, split: str, split_ratio: float, seed: int) -> List[str]:
    all_pkls = glob.glob(os.path.join(data_dir, "**", "*.pkl"), recursive=True)
    all_pkls.sort()
    rng = np.random.default_rng(seed)
    rng.shuffle(all_pkls)

    split_idx = int(len(all_pkls) * split_ratio)
    if split == "train":
        return all_pkls[:split_idx]
    if split == "val":
        return all_pkls[split_idx:]
    if split == "all":
        return all_pkls
    raise ValueError(f"Unsupported split: {split}")


def _collect_split_files_from_list(data_list: str, split: str) -> List[str]:
    with open(data_list, "rb") as f:
        split_data = pickle.load(f)

    if split == "all":
        files = []
        for key in ("train", "val", "test"):
            files.extend(split_data.get(key, []))
        return files
    return list(split_data.get(split, []))

def _gather_eval_items(file_paths: Sequence[str], item_type: str, coord_key: str, max_items: Optional[int], seed: int):
    expected_rows = 16 if item_type == "face" else 4
    candidate_keys = pick_candidate_keys(coord_key, item_type)
    items = []
    skipped_files = 0

    for path in tqdm(file_paths, desc=f"Collecting {item_type} items"):
        try:
            with open(path, "rb") as f:
                cad = pickle.load(f)
        except Exception:
            skipped_files += 1
            continue

        used_key, controls = load_controls(cad, candidate_keys, expected_rows)
        if controls is None:
            continue

        for local_idx, control in enumerate(controls):
            items.append({"file": path, "coord_key": used_key, "index": local_idx, "controls": control})

    if max_items is not None and len(items) > max_items:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(items), size=max_items, replace=False)
        items = [items[int(idx)] for idx in selected]

    return items, skipped_files

def _compute_metrics(diff: torch.Tensor, item_type: str) -> Dict[str, float]:
    face_boundary_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
        dtype=torch.bool,
        device=diff.device,
    ).view(1, 1, 4, 4)
    edge_endpoint_mask = torch.tensor(
        [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]],
        dtype=torch.bool,
        device=diff.device,
    ).view(1, 1, 4, 4)

    out = {
        "mse": (diff ** 2).mean().item(),
        "max_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "boundary_max_error": 0.0,
        "endpoint_max_error": 0.0,
    }
    if item_type == "face":
        out["boundary_max_error"] = diff.masked_fill(~face_boundary_mask, 0.0).max().item()
    else:
        out["endpoint_max_error"] = diff.masked_fill(~edge_endpoint_mask, 0.0).max().item()
    return out


def _evaluate_items(model, model_kwargs, items: Sequence[Dict], item_type: str, device):
    records = []
    for item in tqdm(items, desc=f"Evaluating {item_type} items"):
        x_in, _, _, diff_grid = reconstruct_controls(model, model_kwargs, item["controls"], item_type, device)
        diff = torch.from_numpy(diff_grid).to(device).permute(2, 0, 1).unsqueeze(0)

        metrics = _compute_metrics(diff, item_type)
        bbox_min = np.min(item["controls"], axis=0)
        bbox_max = np.max(item["controls"], axis=0)
        bbox_size = float(np.linalg.norm(bbox_max - bbox_min))

        record = {
            "file": item["file"],
            "coord_key": item["coord_key"],
            "index": item["index"],
            "bbox_size": bbox_size,
            "rel_error_pct": (metrics["max_error"] / (bbox_size + 1e-6)) * 100.0,
        }
        record.update(metrics)
        records.append(record)
    return records


def _print_metric_block(title: str, values: np.ndarray):
    print(f"  {title:<18} mean={values.mean():.6f} median={np.median(values):.6f} p95={np.percentile(values, 95):.6f}")


def _serialize_record(record: Dict) -> Dict:
    serialized = {}
    for key, value in record.items():
        if isinstance(value, np.generic):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return serialized


def _dump_worst_records(item_type: str, records: Sequence[Dict], worst_key: str, output_dir: str, top_k: int):
    if not records:
        return

    os.makedirs(output_dir, exist_ok=True)
    sorted_records = sorted(records, key=lambda r: r[worst_key], reverse=True)
    payload = {
        "item_type": item_type,
        "sort_key": worst_key,
        "count": min(top_k, len(sorted_records)),
        "records": [_serialize_record(record) for record in sorted_records[:top_k]],
    }
    save_path = os.path.join(output_dir, f"worst_{item_type}_cases.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Saved worst {item_type} cases to: {save_path}")


def _summarize_records(item_type: str, records: Sequence[Dict], output_dir: Optional[str] = None, top_k: int = 50):
    if not records:
        print(f"\n{item_type} report: no valid items evaluated.")
        return

    max_errors = np.array([r["max_error"] for r in records], dtype=np.float64)
    mean_abs_errors = np.array([r["mean_abs_error"] for r in records], dtype=np.float64)
    mses = np.array([r["mse"] for r in records], dtype=np.float64)
    rel_errors = np.array([r["rel_error_pct"] for r in records], dtype=np.float64)

    print("\n" + "=" * 72)
    print(f"{item_type.upper()} reconstruction report ({len(records)} items)")
    print("=" * 72)
    _print_metric_block("Max Error", max_errors)
    _print_metric_block("Mean Abs Error", mean_abs_errors)
    _print_metric_block("MSE", mses)
    _print_metric_block("Relative Error %", rel_errors)

    if item_type == "face":
        boundary_max_errors = np.array([r["boundary_max_error"] for r in records], dtype=np.float64)
        _print_metric_block("Boundary Max", boundary_max_errors)
        worst = max(records, key=lambda r: r["boundary_max_error"])
        print("-" * 72)
        print("  Worst by Boundary Max:")
        print(f"    file={worst['file']}")
        print(f"    face_idx={worst['index']}")
        print(f"    boundary_max={worst['boundary_max_error']:.6f}")
        print(f"    max_error={worst['max_error']:.6f}")
        print(f"    rel_error={worst['rel_error_pct']:.3f} %")
        if output_dir:
            _dump_worst_records("face", records, "boundary_max_error", output_dir, top_k)
    else:
        endpoint_max_errors = np.array([r["endpoint_max_error"] for r in records], dtype=np.float64)
        _print_metric_block("Endpoint Max", endpoint_max_errors)
        worst = max(records, key=lambda r: r["endpoint_max_error"])
        print("-" * 72)
        print("  Worst by Endpoint Max:")
        print(f"    file={worst['file']}")
        print(f"    edge_idx={worst['index']}")
        print(f"    endpoint_max={worst['endpoint_max_error']:.6f}")
        print(f"    max_error={worst['max_error']:.6f}")
        print(f"    rel_error={worst['rel_error_pct']:.3f} %")
        if output_dir:
            _dump_worst_records("edge", records, "endpoint_max_error", output_dir, top_k)


def verify_model(args):
    print("=" * 72)
    print("Starting NURBS VQ-VAE verification")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.ckpt_path):
        print(f"Checkpoint not found: {args.ckpt_path}")
        return

    model, model_kwargs = load_model_from_checkpoint(args.ckpt_path, device)
    print(f"Loaded checkpoint: {args.ckpt_path}")
    print(
        f"Model config: in_channels={model_kwargs['in_channels']}, "
        f"block_out_channels={model_kwargs['block_out_channels']}, "
        f"latent_channels={model_kwargs['latent_channels']}, "
        f"vq_embed_dim={model_kwargs['vq_embed_dim']}, "
        f"num_vq_embeddings={model_kwargs['num_vq_embeddings']}"
    )

    if args.data_list:
        split_files = _collect_split_files_from_list(args.data_list, args.split)
    else:
        split_files = _collect_split_files(args.data_dir, args.split, args.split_ratio, args.seed)
    print(f"Using split={args.split}, files={len(split_files)}, coord_key={args.coord_key}")

    face_items, face_skipped = _gather_eval_items(split_files, "face", args.coord_key, args.max_face_items, args.seed)
    edge_items, edge_skipped = _gather_eval_items(split_files, "edge", args.coord_key, args.max_edge_items, args.seed + 1)

    print(
        f"Collected face_items={len(face_items)} (skipped_files={face_skipped}), "
        f"edge_items={len(edge_items)} (skipped_files={edge_skipped})"
    )

    face_records = _evaluate_items(model, model_kwargs, face_items, "face", device)
    edge_records = _evaluate_items(model, model_kwargs, edge_items, "edge", device)

    _summarize_records("face", face_records, output_dir=args.output_dir, top_k=args.top_k)
    _summarize_records("edge", edge_records, output_dir=args.output_dir, top_k=args.top_k)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify NURBS VQ-VAE face and edge reconstruction quality")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints_vqvae/restart_baseline/deepcad_nurbs_vqvae_best.pt",
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--data_list", type=str, default="", help="Optional split pickle with explicit file lists")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--split_ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coord_key", choices=["auto", "face_ctrs_wcs_norm", "face_ctrs"], default="auto")
    parser.add_argument("--max_face_items", type=int, default=2000)
    parser.add_argument("--max_edge_items", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="verify_reports")
    parser.add_argument("--top_k", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    verify_model(parse_args())
