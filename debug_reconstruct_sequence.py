#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from generate_cond import summarize_sequence
from utils import check_brep_validity, parse_sequence_to_cad_data_nurbs, reconstruct_cad_from_sequence_nurbs


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sequence_group(sequence_file: str, split: str, index: int) -> dict:
    with open(sequence_file, "rb") as f:
        data = pickle.load(f)
    groups = data[split]
    if index < 0 or index >= len(groups):
        raise IndexError(f"index {index} out of range for split={split}, len={len(groups)}")
    return groups[index], data


def get_original_sequence(group: dict) -> list[int]:
    seq = group["original"]["input_ids"]
    if isinstance(seq, torch.Tensor):
        seq = seq.cpu().tolist()
    return [int(x) for x in seq]


def save_debug(output_dir: Path, prefix: str, sequence: list[int], vocab_config: dict, cad_data: dict, error: str | None):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_sequence(sequence, vocab_config)
    summary["error"] = error
    summary["parsed_faces"] = len(cad_data.get("face_ctrs", []))
    summary["parsed_edges"] = len(cad_data.get("edge_ctrs", []))
    summary["parsed_edge_face_pairs"] = len(cad_data.get("edgeFace_adj", []))

    with open(output_dir / f"{prefix}_debug.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "sequence": sequence}, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        output_dir / f"{prefix}_parsed.npz",
        face_ctrs=np.asarray(cad_data.get("face_ctrs", []), dtype=np.float32),
        edge_ctrs=np.asarray(cad_data.get("edge_ctrs", []), dtype=np.float32),
        edgeFace_adj=np.asarray(cad_data.get("edgeFace_adj", []), dtype=np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug parse/reconstruct on an existing AR token sequence.")
    parser.add_argument("--sequence_file", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", default="result/debug_reconstruct_sequence")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    vocab_config = load_config(args.config)
    group, sequence_data = load_sequence_group(args.sequence_file, args.split, args.index)

    # Prefer sequence-file metadata when present; config.json may omit debug-only fields.
    vocab_config.setdefault("face_block", sequence_data.get("face_block", 5))
    vocab_config.setdefault("edge_block", sequence_data.get("edge_block", 6))

    sequence = get_original_sequence(group)
    name = group.get("name") or group.get("file_name") or f"{args.split}_{args.index:06d}"
    prefix = Path(str(name)).stem

    cad_data = parse_sequence_to_cad_data_nurbs(sequence, vocab_config, device=device, verbose=False)
    error = None
    solid = None
    try:
        solid = reconstruct_cad_from_sequence_nurbs(sequence, vocab_config, device=device, verbose=False)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    is_valid = False
    if solid is not None:
        try:
            is_valid = bool(check_brep_validity(solid))
        except Exception as exc:
            error = f"validity_check_failed: {type(exc).__name__}: {exc}"
    elif error is None:
        error = "reconstruct returned None"

    save_debug(Path(args.output_dir), prefix, sequence, vocab_config, cad_data, error)

    summary = summarize_sequence(sequence, vocab_config)
    print("=" * 60)
    print("Sequence Reconstruction Debug")
    print("=" * 60)
    print(f"sequence_file: {args.sequence_file}")
    print(f"split/index: {args.split}/{args.index}")
    print(f"name: {name}")
    print(f"length: {summary['length']}")
    print(f"num_faces_protocol: {summary.get('num_faces_protocol')}")
    print(f"num_edges_protocol: {summary.get('num_edges_protocol')}")
    print(f"parsed_faces: {len(cad_data.get('face_ctrs', []))}")
    print(f"parsed_edges: {len(cad_data.get('edge_ctrs', []))}")
    print(f"parsed_edge_face_pairs: {len(cad_data.get('edgeFace_adj', []))}")
    print(f"solid_returned: {solid is not None}")
    print(f"brep_valid: {is_valid}")
    print(f"error: {error}")
    print("-" * 60)
    print(f"debug_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
