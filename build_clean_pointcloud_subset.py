#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
from pathlib import Path


def normalize_model_name(name: str) -> str:
    base = os.path.basename(str(name).strip())
    for suffix in [".step", ".stp", ".npy"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if base.endswith(".pkl"):
        base = base[:-4]
    return base


def load_ok_model_ids(records_json: Path) -> set[str]:
    with open(records_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    ok_ids = set()
    for record in records:
        if record.get("status") != "ok":
            continue

        pc_path = Path(record["path"])
        model_dir = pc_path.parent
        step_candidates = sorted(model_dir.glob("*.step")) + sorted(model_dir.glob("*.stp"))

        if step_candidates:
            ok_ids.add(normalize_model_name(step_candidates[0].name))
        else:
            ok_ids.add(normalize_model_name(record["model_id"]))

    return ok_ids


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def resolve_split_names(sequence_groups, split_data, split_key: str):
    names_from_split = split_data.get(split_key) if split_data is not None else None
    if names_from_split:
        return list(names_from_split)

    names = []
    for idx, group in enumerate(sequence_groups):
        model_name = group.get("name") or group.get("file_name") or f"model_{idx:06d}"
        names.append(model_name)
    return names


def is_split_only_pickle(data) -> bool:
    if not isinstance(data, dict):
        return False
    train_items = data.get("train")
    if not isinstance(train_items, list) or len(train_items) == 0:
        return False
    return isinstance(train_items[0], str)


def filter_split_only_data(split_data, ok_model_ids: set[str], min_val: int):
    output_split = copy.deepcopy(split_data)
    stats = {}

    for split_key in ["train", "val"]:
        split_names = split_data.get(split_key, [])
        normalized_names = [normalize_model_name(name) for name in split_names]
        filtered_names = [
            split_names[idx]
            for idx, name in enumerate(normalized_names)
            if name in ok_model_ids
        ]
        output_split[split_key] = filtered_names
        stats[split_key] = {
            "original": len(split_names),
            "kept": len(filtered_names),
        }

    if len(output_split["val"]) < min_val and len(output_split["train"]) > min_val:
        need = min_val - len(output_split["val"])
        move_count = min(need, max(0, len(output_split["train"]) - min_val))
        if move_count > 0:
            output_split["val"].extend(output_split["train"][:move_count])
            output_split["train"] = output_split["train"][move_count:]
            stats["rebalanced"] = {"moved_train_to_val": move_count}

    stats["final_train"] = len(output_split["train"])
    stats["final_val"] = len(output_split["val"])
    stats["total"] = stats["final_train"] + stats["final_val"]
    return output_split, stats


def filter_split_and_sequence(sequence_data, split_data, ok_model_ids: set[str], min_val: int):
    output_sequence = dict(sequence_data)
    output_split = copy.deepcopy(split_data) if split_data is not None else {}
    stats = {}

    for split_key in ["train", "val"]:
        sequence_groups = sequence_data.get(split_key, [])
        split_names = resolve_split_names(sequence_groups, split_data, split_key)

        normalized_names = [normalize_model_name(name) for name in split_names]
        kept_indices = [idx for idx, name in enumerate(normalized_names) if name in ok_model_ids]

        if len(sequence_groups) != len(split_names):
            raise ValueError(
                f"Length mismatch for split '{split_key}': "
                f"sequence has {len(sequence_groups)}, split has {len(split_names)}. "
                f"This script currently expects aligned ordering."
            )

        filtered_groups = []
        filtered_names = []
        for idx in kept_indices:
            group = copy.deepcopy(sequence_groups[idx])
            model_name = normalized_names[idx]
            group["name"] = model_name
            filtered_groups.append(group)
            filtered_names.append(split_names[idx])

        output_sequence[split_key] = filtered_groups
        output_split[split_key] = filtered_names
        stats[split_key] = {
            "original": len(split_names),
            "kept": len(filtered_names),
        }

    if len(output_split["val"]) < min_val and len(output_split["train"]) > min_val:
        need = min_val - len(output_split["val"])
        move_count = min(need, max(0, len(output_split["train"]) - min_val))
        if move_count > 0:
            output_split["val"].extend(output_split["train"][:move_count])
            output_sequence["val"].extend(output_sequence["train"][:move_count])
            output_split["train"] = output_split["train"][move_count:]
            output_sequence["train"] = output_sequence["train"][move_count:]
            stats["rebalanced"] = {"moved_train_to_val": move_count}

    stats["final_train"] = len(output_split["train"])
    stats["final_val"] = len(output_split["val"])
    stats["total"] = stats["final_train"] + stats["final_val"]
    return output_sequence, output_split, stats


def main():
    parser = argparse.ArgumentParser(description="Build a clean AR subset using only samples with valid point clouds.")
    parser.add_argument(
        "--sequence_file",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_data_split_6bit_complete.pkl",
        help="Sequence pickle used by train_ar.py",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="",
        help="Optional split pickle aligned with the sequence file",
    )
    parser.add_argument(
        "--records_json",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/pointcloud_quality_report/pointcloud_records.json",
        help="Point-cloud quality report generated by inspect_pointcloud_quality.py",
    )
    parser.add_argument(
        "--output_sequence_file",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_data_split_6bit_complete_pc_clean.pkl",
        help="Filtered output sequence pickle. If input is split-only, this will not be written.",
    )
    parser.add_argument(
        "--output_split_file",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_data_split_6bit_complete_pc_clean_split.pkl",
        help="Filtered output split pickle",
    )
    parser.add_argument(
        "--output_stats_json",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_data_split_6bit_complete_pc_clean_stats.json",
        help="Statistics json for the filtered subset",
    )
    parser.add_argument(
        "--min_val",
        type=int,
        default=32,
        help="If filtered val is too small, move a few clean samples from train to val.",
    )
    args = parser.parse_args()

    ok_model_ids = load_ok_model_ids(Path(args.records_json))
    sequence_data = load_pickle(Path(args.sequence_file))
    split_data = load_pickle(Path(args.split_file)) if args.split_file else None

    wrote_sequence = False
    if is_split_only_pickle(sequence_data):
        output_split, stats = filter_split_only_data(
            split_data=sequence_data,
            ok_model_ids=ok_model_ids,
            min_val=args.min_val,
        )
    else:
        output_sequence, output_split, stats = filter_split_and_sequence(
            sequence_data=sequence_data,
            split_data=split_data,
            ok_model_ids=ok_model_ids,
            min_val=args.min_val,
        )
        save_pickle(Path(args.output_sequence_file), output_sequence)
        wrote_sequence = True

    save_pickle(Path(args.output_split_file), output_split)

    Path(args.output_stats_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Clean Point-Cloud Subset Built")
    print("=" * 60)
    print(f"ok_model_ids: {len(ok_model_ids)}")
    for split_key in ["train", "val"]:
        split_stats = stats[split_key]
        print(f"{split_key}: {split_stats['kept']} / {split_stats['original']}")
    if "rebalanced" in stats:
        print(f"rebalanced: {stats['rebalanced']}")
    print(f"final_train: {stats['final_train']}")
    print(f"final_val: {stats['final_val']}")
    print(f"total: {stats['total']}")
    print("-" * 60)
    if wrote_sequence:
        print(f"output_sequence_file: {args.output_sequence_file}")
    else:
        print("output_sequence_file: <not written, input is split-only>")
    print(f"output_split_file: {args.output_split_file}")
    print(f"output_stats_json: {args.output_stats_json}")


if __name__ == "__main__":
    main()
