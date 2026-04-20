#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def inspect_point_cloud(path: Path) -> dict:
    result = {
        "model_id": path.parent.name if path.name == "point_cloud.npy" else path.stem,
        "path": str(path),
        "exists": path.exists(),
        "shape": None,
        "dtype": None,
        "total_points": 0,
        "valid_points": 0,
        "invalid_points": 0,
        "valid_ratio": 0.0,
        "status": "missing",
    }

    if not path.exists():
        return result

    try:
        array = np.load(path, allow_pickle=True)
    except Exception as exc:
        result["status"] = "load_failed"
        result["error"] = str(exc)
        return result

    result["shape"] = list(array.shape)
    result["dtype"] = str(array.dtype)

    if array.ndim < 2 or array.shape[-1] < 3:
        result["status"] = "bad_shape"
        return result

    points = np.asarray(array, dtype=np.float32).reshape(-1, array.shape[-1])[:, :3]
    finite_mask = np.isfinite(points).all(axis=1)

    total_points = int(points.shape[0])
    valid_points = int(finite_mask.sum())
    invalid_points = total_points - valid_points

    result["total_points"] = total_points
    result["valid_points"] = valid_points
    result["invalid_points"] = invalid_points
    result["valid_ratio"] = float(valid_points / total_points) if total_points > 0 else 0.0

    if total_points == 0:
        result["status"] = "empty"
    elif valid_points == 0:
        result["status"] = "all_invalid"
    elif invalid_points > 0:
        result["status"] = "partially_invalid"
    else:
        result["status"] = "ok"

    return result


def collect_point_cloud_paths(root: Path) -> list[Path]:
    direct_npy = [p for p in root.glob("*.npy") if p.name != "face_labels.npy"]
    nested_npy = list(root.glob("*/point_cloud.npy"))
    all_paths = sorted(set(direct_npy + nested_npy))
    return all_paths


def summarize(records: list[dict]) -> dict:
    summary = {
        "total_files": len(records),
        "ok": 0,
        "partially_invalid": 0,
        "all_invalid": 0,
        "empty": 0,
        "bad_shape": 0,
        "load_failed": 0,
        "missing": 0,
        "avg_valid_ratio": 0.0,
    }

    if not records:
        return summary

    valid_ratio_sum = 0.0
    for record in records:
        status = record["status"]
        if status in summary:
            summary[status] += 1
        valid_ratio_sum += record.get("valid_ratio", 0.0)

    summary["avg_valid_ratio"] = valid_ratio_sum / len(records)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Inspect raw point-cloud quality")
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/ABC_Dataset_NEW",
        help="Root directory containing model folders",
    )
    parser.add_argument("--output_dir", type=str, default="pointcloud_quality_report")
    parser.add_argument("--top_k", type=int, default=50, help="Number of bad cases to dump in text files")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = collect_point_cloud_paths(root)
    records = [inspect_point_cloud(path) for path in paths]
    summary = summarize(records)

    bad_records = [r for r in records if r["status"] != "ok"]
    all_invalid_records = [r for r in records if r["status"] == "all_invalid"]

    records_json = output_dir / "pointcloud_records.json"
    summary_json = output_dir / "pointcloud_summary.json"
    bad_txt = output_dir / "bad_pointcloud_models.txt"
    all_invalid_txt = output_dir / "all_invalid_models.txt"

    with open(records_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(bad_txt, "w", encoding="utf-8") as f:
        for record in bad_records[: args.top_k]:
            f.write(f"{record['model_id']}\t{record['status']}\t{record['valid_points']}/{record['total_points']}\n")

    with open(all_invalid_txt, "w", encoding="utf-8") as f:
        for record in all_invalid_records[: args.top_k]:
            f.write(f"{record['model_id']}\t{record['path']}\n")

    print("=" * 60)
    print("Point Cloud Quality Summary")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 60)
    print(f"records_json: {records_json}")
    print(f"summary_json: {summary_json}")
    print(f"bad_txt: {bad_txt}")
    print(f"all_invalid_txt: {all_invalid_txt}")


if __name__ == "__main__":
    main()
