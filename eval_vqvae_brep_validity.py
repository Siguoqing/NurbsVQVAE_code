#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils import (
    check_brep_validity,
    construct_brep,
    create_bspline_curve,
    create_bspline_surface,
    joint_optimize,
    sample_bspline_curve,
    sample_bspline_surface,
)
from vqvae_eval_utils import load_model_from_checkpoint, reconstruct_controls


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_split_files(split_file: str, split: str) -> list[str]:
    data = load_pickle(split_file)
    if split == "all":
        out = []
        for key in ("train", "val", "test"):
            out.extend(data.get(key, []))
        return out
    return list(data.get(split, []))


def get_array(cad: dict, keys: tuple[str, ...]):
    for key in keys:
        value = cad.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim > 0 and arr.size > 0 and np.isfinite(arr).all():
            return arr, key
    return None, None


def normalize_face_edge_adj(face_edge_adj, num_faces: int) -> list[list[int]]:
    if face_edge_adj is None:
        return [[] for _ in range(num_faces)]
    out = []
    for item in list(face_edge_adj):
        arr = np.asarray(item).reshape(-1)
        out.append([int(x) for x in arr.tolist()])
    while len(out) < num_faces:
        out.append([])
    return out[:num_faces]


def normalize_edge_vertex_adj(edge_vertex_adj, num_edges: int) -> np.ndarray:
    if edge_vertex_adj is None:
        return np.zeros((num_edges, 2), dtype=np.int32)
    arr = np.asarray(edge_vertex_adj, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((num_edges, 2), dtype=np.int32)
    return arr[:, :2]


def controls_to_sampled_ncs(face_ctrs: np.ndarray, edge_ctrs: np.ndarray):
    surf_ncs = []
    for face_ctr in face_ctrs:
        surface = create_bspline_surface(np.asarray(face_ctr, dtype=np.float64))
        surf_ncs.append(sample_bspline_surface(surface, num_u=32, num_v=32))

    edge_ncs = []
    for edge_ctr in edge_ctrs:
        curve = create_bspline_curve(np.asarray(edge_ctr, dtype=np.float64))
        edge_ncs.append(sample_bspline_curve(curve, num_points=32))

    return np.asarray(surf_ncs, dtype=np.float64), np.asarray(edge_ncs, dtype=np.float64)


def reconstruct_all_controls(model, model_kwargs, controls: np.ndarray, item_type: str, device):
    recon = []
    max_errors = []
    mean_abs_errors = []
    for control in controls:
        _, target_grid, recon_grid, diff_grid = reconstruct_controls(
            model, model_kwargs, control, item_type, device
        )
        if item_type == "face":
            recon_control = recon_grid.reshape(16, 3)
        else:
            # Edge controls were tiled to a 4x4 grid during VQ-VAE training.
            recon_control = recon_grid.mean(axis=0).reshape(4, 3)
        recon.append(recon_control.astype(np.float32))
        max_errors.append(float(np.max(diff_grid)))
        mean_abs_errors.append(float(np.mean(np.abs(diff_grid))))
    return np.asarray(recon, dtype=np.float32), {
        f"{item_type}_max_error_mean": float(np.mean(max_errors)) if max_errors else None,
        f"{item_type}_max_error_p95": float(np.percentile(max_errors, 95)) if max_errors else None,
        f"{item_type}_mean_abs_error_mean": float(np.mean(mean_abs_errors)) if mean_abs_errors else None,
    }


def evaluate_file(path: str, model, model_kwargs, device):
    cad = load_pickle(path)
    face_ctrs, face_key = get_array(cad, ("face_ctrs_wcs_norm", "face_ctrs"))
    edge_ctrs, edge_key = get_array(cad, ("edge_ctrs_wcs_norm", "edge_ctrs"))
    surf_bbox_wcs, surf_bbox_key = get_array(cad, ("surf_bbox_wcs",))
    corner_unique, corner_key = get_array(cad, ("corner_unique", "corner_wcs"))

    if face_ctrs is None or edge_ctrs is None:
        return {
            "file": path,
            "status": "missing_controls",
            "face_key": face_key,
            "edge_key": edge_key,
        }
    if surf_bbox_wcs is None or corner_unique is None or cad.get("faceEdge_adj") is None or cad.get("edgeCorner_adj") is None:
        return {
            "file": path,
            "status": "missing_topology_or_bbox",
            "face_key": face_key,
            "edge_key": edge_key,
            "surf_bbox_key": surf_bbox_key,
            "corner_key": corner_key,
        }

    try:
        face_ctrs = face_ctrs.reshape(-1, 16, 3)
        edge_ctrs = edge_ctrs.reshape(-1, 4, 3)
    except Exception as exc:
        return {"file": path, "status": "bad_control_shape", "error": str(exc)}

    if len(face_ctrs) == 0 or len(edge_ctrs) == 0:
        return {"file": path, "status": "empty_controls"}

    face_edge_adj = normalize_face_edge_adj(cad.get("faceEdge_adj"), len(face_ctrs))
    edge_vertex_adj = normalize_edge_vertex_adj(cad.get("edgeCorner_adj"), len(edge_ctrs))
    surf_bbox_wcs = np.asarray(surf_bbox_wcs, dtype=np.float64).reshape(-1, 6)
    corner_unique = np.asarray(corner_unique, dtype=np.float64).reshape(-1, 3)

    try:
        recon_face_ctrs, face_metrics = reconstruct_all_controls(
            model, model_kwargs, face_ctrs, "face", device
        )
        recon_edge_ctrs, edge_metrics = reconstruct_all_controls(
            model, model_kwargs, edge_ctrs, "edge", device
        )
        surf_ncs, edge_ncs = controls_to_sampled_ncs(recon_face_ctrs, recon_edge_ctrs)
    except Exception as exc:
        return {
            "file": path,
            "status": "vqvae_or_sampling_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "num_faces": len(face_ctrs),
            "num_edges": len(edge_ctrs),
        }

    solid = None
    error = None
    try:
        surf_wcs, edge_wcs = joint_optimize(
            surf_ncs,
            edge_ncs,
            surf_bbox_wcs,
            corner_unique,
            edge_vertex_adj,
            face_edge_adj,
            len(edge_ncs),
            len(surf_ncs),
        )
        solid = construct_brep(surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    is_valid = False
    if solid is not None:
        try:
            is_valid = bool(check_brep_validity(solid))
        except Exception as exc:
            error = f"validity_check_failed: {type(exc).__name__}: {exc}"

    record = {
        "file": path,
        "status": "ok" if solid is not None else "brep_failed",
        "brep_valid": is_valid,
        "solid_returned": solid is not None,
        "error": error,
        "face_key": face_key,
        "edge_key": edge_key,
        "surf_bbox_key": surf_bbox_key,
        "corner_key": corner_key,
        "num_faces": int(len(face_ctrs)),
        "num_edges": int(len(edge_ctrs)),
        "min_face_degree": int(min((len(x) for x in face_edge_adj), default=0)),
        "max_face_degree": int(max((len(x) for x in face_edge_adj), default=0)),
    }
    record.update(face_metrics)
    record.update(edge_metrics)
    return record


def summarize(records: list[dict]) -> dict:
    total = len(records)
    eligible = [
        r for r in records
        if r.get("status") not in {"missing_controls", "missing_topology_or_bbox", "bad_control_shape", "empty_controls"}
    ]
    solid = sum(1 for r in records if r.get("solid_returned"))
    valid = sum(1 for r in records if r.get("brep_valid"))
    eligible_solid = sum(1 for r in eligible if r.get("solid_returned"))
    eligible_valid = sum(1 for r in eligible if r.get("brep_valid"))
    status_counts = {}
    for record in records:
        status = record.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "total": total,
        "solid_returned": solid,
        "brep_valid": valid,
        "solid_rate": solid / total if total else 0.0,
        "valid_rate": valid / total if total else 0.0,
        "eligible": len(eligible),
        "eligible_solid_returned": eligible_solid,
        "eligible_brep_valid": eligible_valid,
        "eligible_solid_rate": eligible_solid / len(eligible) if eligible else 0.0,
        "eligible_valid_rate": eligible_valid / len(eligible) if eligible else 0.0,
        "status_counts": status_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BREP validity after NURBS VQ-VAE reconstruction.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="result/eval_vqvae_brep_validity")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_split_files(args.split_file, args.split)
    files = sorted(files)
    if args.max_samples and len(files) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(files), size=args.max_samples, replace=False)
        files = [files[int(idx)] for idx in indices]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, model_kwargs = load_model_from_checkpoint(args.ckpt_path, device)

    records = []
    for path in tqdm(files, desc="VQ-VAE -> BREP"):
        records.append(evaluate_file(path, model, model_kwargs, device))

    summary = summarize(records)
    summary.update({
        "ckpt_path": args.ckpt_path,
        "split_file": args.split_file,
        "split": args.split,
        "max_samples": args.max_samples,
        "evaluated_files": len(files),
    })

    with open(output_dir / "records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("VQ-VAE -> BREP Validity Summary")
    print("=" * 60)
    print(f"evaluated_files: {summary['evaluated_files']}")
    print(f"eligible_files: {summary['eligible']}")
    print(f"solid_returned: {summary['solid_returned']}")
    print(f"brep_valid: {summary['brep_valid']}")
    print(f"solid_rate: {summary['solid_rate']:.4f}")
    print(f"valid_rate: {summary['valid_rate']:.4f}")
    print(f"eligible_solid_rate: {summary['eligible_solid_rate']:.4f}")
    print(f"eligible_valid_rate: {summary['eligible_valid_rate']:.4f}")
    print(f"status_counts: {summary['status_counts']}")
    print("-" * 60)
    print(f"records_json: {output_dir / 'records.json'}")
    print(f"summary_json: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
