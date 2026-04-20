#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path


def normalize_model_name(name: str) -> str:
    base = os.path.basename(str(name).strip())
    for suffix in [".step", ".stp", ".npy", ".pkl"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def load_ok_step_names(records_json: Path) -> set[str]:
    with open(records_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    ok_names: set[str] = set()
    for record in records:
        if record.get("status") != "ok":
            continue

        path_value = record.get("path")
        if path_value:
            model_dir = Path(path_value).parent
        else:
            model_dir = Path(record.get("model_dir", ""))

        step_candidates = []
        if model_dir.exists():
            step_candidates = sorted(model_dir.glob("*.step")) + sorted(model_dir.glob("*.stp"))

        step_file = record.get("step_file")
        if step_file:
            ok_names.add(normalize_model_name(step_file))
        elif step_candidates:
            ok_names.add(normalize_model_name(step_candidates[0].name))
        elif record.get("model_id"):
            ok_names.add(normalize_model_name(record["model_id"]))

    return ok_names


def collect_parsed_pkls(parsed_dir: Path) -> list[str]:
    return sorted(str(path) for path in parsed_dir.rglob("*.pkl"))


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a train/val split from all parsed NURBS pkls that have valid point clouds."
    )
    parser.add_argument(
        "--parsed_dir",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/ABC_Dataset_NEW_parsed",
        help="Directory containing parsed NURBS .pkl files.",
    )
    parser.add_argument(
        "--records_json",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/pointcloud_quality_report_full/pointcloud_records.json",
        help="Point-cloud quality records from inspect_pointcloud_quality.py.",
    )
    parser.add_argument(
        "--output_split_file",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_data_split_6bit_allparsed_pc_clean_split.pkl",
        help="Output pickle with train/val parsed pkl paths.",
    )
    parser.add_argument(
        "--output_stats_json",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_data_split_6bit_allparsed_pc_clean_stats.json",
        help="Output json with count statistics.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    parsed_paths = collect_parsed_pkls(Path(args.parsed_dir))
    ok_step_names = load_ok_step_names(Path(args.records_json))

    matched = []
    missing_pointcloud = []
    for path in parsed_paths:
        name = normalize_model_name(path)
        if name in ok_step_names:
            matched.append(path)
        else:
            missing_pointcloud.append(path)

    rng = random.Random(args.seed)
    rng.shuffle(matched)

    val_count = int(round(len(matched) * args.val_ratio))
    val_paths = matched[:val_count]
    train_paths = matched[val_count:]

    split = {
        "train": train_paths,
        "val": val_paths,
    }
    stats = {
        "parsed_total": len(parsed_paths),
        "ok_pointcloud_step_names": len(ok_step_names),
        "matched": len(matched),
        "missing_or_bad_pointcloud": len(missing_pointcloud),
        "train": len(train_paths),
        "val": len(val_paths),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "parsed_dir": args.parsed_dir,
        "records_json": args.records_json,
    }

    save_pickle(Path(args.output_split_file), split)
    Path(args.output_stats_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Full Parsed Point-Cloud Split Built")
    print("=" * 60)
    print(f"parsed_total: {stats['parsed_total']}")
    print(f"ok_pointcloud_step_names: {stats['ok_pointcloud_step_names']}")
    print(f"matched: {stats['matched']}")
    print(f"missing_or_bad_pointcloud: {stats['missing_or_bad_pointcloud']}")
    print(f"train: {stats['train']}")
    print(f"val: {stats['val']}")
    print("-" * 60)
    print(f"output_split_file: {args.output_split_file}")
    print(f"output_stats_json: {args.output_stats_json}")


if __name__ == "__main__":
    main()
