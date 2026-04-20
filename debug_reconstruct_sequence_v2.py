#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils import (
    check_brep_validity,
    compute_bbox_center_and_size,
    construct_brep,
    create_bspline_curve,
    create_bspline_surface,
    joint_optimize,
    sample_bspline_curve,
    sample_bspline_surface,
)
from vqvae_eval_utils import load_model_from_checkpoint


def load_vqvae_model(ckpt_path: str, device):
    loaded = load_model_from_checkpoint(ckpt_path, device)
    if isinstance(loaded, (tuple, list)) and len(loaded) >= 1:
        return loaded[0]
    raise ValueError("load_model_from_checkpoint must return a model")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def dequantize_bbox(indices: np.ndarray, num_tokens: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.float64)
    normalized = indices / float(num_tokens - 1)
    return normalized * 2.0 - 1.0


def embedding_weight(model):
    if hasattr(model.quantize, "embedding"):
        return model.quantize.embedding.weight
    if hasattr(model.quantize, "embed"):
        return model.quantize.embed.weight
    raise AttributeError("Cannot find VQ embedding weight on model.quantize")


def decode_tokens_to_controls(model, token_rows: list[list[int]], item_type: str, device) -> np.ndarray:
    if not token_rows:
        return np.zeros((0, 16 if item_type == "face" else 4, 3), dtype=np.float32)

    token_tensor = torch.tensor(token_rows, dtype=torch.long, device=device)
    if token_tensor.ndim != 2 or token_tensor.shape[1] != 4:
        raise ValueError(f"Expected token shape [B, 4], got {tuple(token_tensor.shape)}")

    token_tensor = token_tensor.reshape(token_tensor.shape[0], 2, 2)
    with torch.no_grad():
        quantized = F.embedding(token_tensor, embedding_weight(model)).permute(0, 3, 1, 2)
        decoded = model.decoder(model.post_quant_conv(quantized))
    grid = decoded[:, :3].permute(0, 2, 3, 1).detach().cpu().numpy()

    if item_type == "face":
        return grid.reshape(len(token_rows), 16, 3).astype(np.float32)
    if item_type == "edge":
        return grid.mean(axis=1).reshape(len(token_rows), 4, 3).astype(np.float32)
    raise ValueError(f"Unknown item_type: {item_type}")


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


def parse_sequence_v2(sequence: list[int], vocab: dict, model, device):
    face_index_offset = vocab["face_index_offset"]
    face_index_size = vocab["face_index_size"]
    quantization_offset = vocab.get("quantization_offset", vocab.get("se_token_offset"))
    quantization_size = vocab.get("quantization_size", vocab.get("se_codebook_size"))
    bbox_token_offset = vocab["bbox_token_offset"]
    bbox_index_size = vocab.get("bbox_index_size", vocab.get("bbox_size", 2048))
    bbox_tokens = vocab.get("bbox_tokens_per_element", 6)
    vq_tokens = vocab.get("se_tokens_per_element", vocab.get("face_vq_tokens", 4))
    special = vocab.get("special_tokens", {})
    start_token = special.get("START_TOKEN", vocab.get("START_TOKEN"))
    sep_token = special.get("SEP_TOKEN", vocab.get("SEP_TOKEN"))
    end_token = special.get("END_TOKEN", vocab.get("END_TOKEN"))

    i = 0
    if i < len(sequence) and sequence[i] == start_token:
        i += 1

    face_token_rows: list[list[int]] = []
    face_bbox_rows: list[list[int]] = []
    face_ids: list[int] = []
    while i < len(sequence) and sequence[i] != sep_token:
        if i + bbox_tokens + vq_tokens >= len(sequence):
            break
        bbox_row = []
        for _ in range(bbox_tokens):
            token = sequence[i]
            if not (bbox_token_offset <= token < bbox_token_offset + bbox_index_size):
                raise ValueError(f"Bad face bbox token {token} at position {i}")
            bbox_row.append(token - bbox_token_offset)
            i += 1

        vq_row = []
        for _ in range(vq_tokens):
            token = sequence[i]
            if not (quantization_offset <= token < quantization_offset + quantization_size):
                raise ValueError(f"Bad face VQ token {token} at position {i}")
            vq_row.append(token - quantization_offset)
            i += 1

        token = sequence[i]
        if not (face_index_offset <= token < face_index_offset + face_index_size):
            raise ValueError(f"Bad face index token {token} at position {i}")
        face_ids.append(token - face_index_offset)
        face_bbox_rows.append(bbox_row)
        face_token_rows.append(vq_row)
        i += 1

    if i < len(sequence) and sequence[i] == sep_token:
        i += 1

    edge_token_rows: list[list[int]] = []
    edge_bbox_rows: list[list[int]] = []
    edge_pairs_raw: list[tuple[int, int]] = []
    while i < len(sequence) and sequence[i] != end_token:
        if i + 1 >= len(sequence):
            break
        src_token = sequence[i]
        dst_token = sequence[i + 1]
        if not (
            face_index_offset <= src_token < face_index_offset + face_index_size
            and face_index_offset <= dst_token < face_index_offset + face_index_size
        ):
            raise ValueError(f"Bad edge face pair at position {i}: {src_token}, {dst_token}")
        edge_pairs_raw.append((src_token - face_index_offset, dst_token - face_index_offset))
        i += 2

        bbox_row = []
        for _ in range(bbox_tokens):
            token = sequence[i]
            if not (bbox_token_offset <= token < bbox_token_offset + bbox_index_size):
                raise ValueError(f"Bad edge bbox token {token} at position {i}")
            bbox_row.append(token - bbox_token_offset)
            i += 1

        vq_row = []
        for _ in range(vq_tokens):
            token = sequence[i]
            if not (quantization_offset <= token < quantization_offset + quantization_size):
                raise ValueError(f"Bad edge VQ token {token} at position {i}")
            vq_row.append(token - quantization_offset)
            i += 1
        edge_bbox_rows.append(bbox_row)
        edge_token_rows.append(vq_row)

    face_id_to_idx = {face_id: idx for idx, face_id in enumerate(face_ids)}
    face_edge_adj = [[] for _ in face_ids]
    remapped_pairs = []
    for edge_idx, (src, dst) in enumerate(edge_pairs_raw):
        if src not in face_id_to_idx or dst not in face_id_to_idx:
            continue
        src_idx = face_id_to_idx[src]
        dst_idx = face_id_to_idx[dst]
        remapped_pairs.append((src_idx, dst_idx))
        face_edge_adj[src_idx].append(edge_idx)
        face_edge_adj[dst_idx].append(edge_idx)

    face_ctrs = decode_tokens_to_controls(model, face_token_rows, "face", device)
    edge_ctrs = decode_tokens_to_controls(model, edge_token_rows, "edge", device)
    surf_ncs, edge_ncs = controls_to_sampled_ncs(face_ctrs, edge_ctrs)
    surf_bbox_wcs = dequantize_bbox(np.asarray(face_bbox_rows, dtype=np.int64), bbox_index_size)
    edge_bbox_wcs = dequantize_bbox(np.asarray(edge_bbox_rows, dtype=np.int64), bbox_index_size)

    return {
        "surf_ncs": surf_ncs,
        "edge_ncs": edge_ncs,
        "surf_bbox_wcs": surf_bbox_wcs,
        "edge_bbox_wcs": edge_bbox_wcs,
        "faceEdge_adj": face_edge_adj,
        "edge_face_pairs": remapped_pairs,
        "face_ids": face_ids,
    }


def infer_vertices(edge_ncs: np.ndarray, edge_bbox_wcs: np.ndarray, face_edge_adj: list[list[int]]):
    edge_v_bbox = []
    for edge_idx, ncs_curve in enumerate(edge_ncs):
        bbox = edge_bbox_wcs[edge_idx]
        center, size = compute_bbox_center_and_size(bbox[:3], bbox[3:])
        wcs_curve = ncs_curve * (size / 2.0) + center
        edge_v_bbox.append(wcs_curve[[0, -1]])
    edge_v_bbox = np.asarray(edge_v_bbox, dtype=np.float64)

    total_vertices = len(edge_ncs) * 2
    parent = list(range(total_vertices))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    face_merged_groups = []
    for edge_indices in face_edge_adj:
        face_vertices = []
        for edge_idx in edge_indices:
            if not (0 <= edge_idx < len(edge_v_bbox)):
                continue
            for vertex_pos_idx in (0, 1):
                global_vertex_id = edge_idx * 2 + vertex_pos_idx
                face_vertices.append((global_vertex_id, edge_v_bbox[edge_idx, vertex_pos_idx]))

        merged = set()
        face_groups = []
        n_vertices = len(face_vertices)
        while len(merged) < n_vertices:
            min_dist = float("inf")
            min_i, min_j = -1, -1
            for i in range(n_vertices):
                if i in merged:
                    continue
                for j in range(i + 1, n_vertices):
                    if j in merged:
                        continue
                    edge_i = face_vertices[i][0] // 2
                    edge_j = face_vertices[j][0] // 2
                    if edge_i == edge_j:
                        continue
                    dist = np.linalg.norm(face_vertices[i][1] - face_vertices[j][1])
                    if dist < min_dist:
                        min_dist = dist
                        min_i, min_j = i, j
            if min_i < 0 or min_j < 0:
                break
            vid1 = face_vertices[min_i][0]
            vid2 = face_vertices[min_j][0]
            union(vid1, vid2)
            face_groups.append([vid1, vid2])
            merged.add(min_i)
            merged.add(min_j)
        face_merged_groups.append(face_groups)

    for i in range(len(face_merged_groups)):
        for j in range(i + 1, len(face_merged_groups)):
            for group1 in face_merged_groups[i]:
                for group2 in face_merged_groups[j]:
                    if set(group1) & set(group2):
                        for v1 in group1:
                            for v2 in group2:
                                union(v1, v2)

    final_groups = {}
    for vid in range(total_vertices):
        final_groups.setdefault(find(vid), []).append(vid)

    unique_vertices = []
    vertex_mapping = [-1] * total_vertices
    for group in final_groups.values():
        positions = []
        for vertex_id in group:
            edge_idx = vertex_id // 2
            vertex_pos_idx = vertex_id % 2
            positions.append(edge_v_bbox[edge_idx, vertex_pos_idx])
        unique_idx = len(unique_vertices)
        unique_vertices.append(np.mean(positions, axis=0))
        for vertex_id in group:
            vertex_mapping[vertex_id] = unique_idx

    edge_vertex_adj = np.zeros((len(edge_ncs), 2), dtype=np.int32)
    for edge_idx in range(len(edge_ncs)):
        edge_vertex_adj[edge_idx, 0] = vertex_mapping[edge_idx * 2]
        edge_vertex_adj[edge_idx, 1] = vertex_mapping[edge_idx * 2 + 1]

    return np.asarray(unique_vertices, dtype=np.float64), edge_vertex_adj


def reconstruct_one(sequence: list[int], vocab: dict, model, device):
    parsed = parse_sequence_v2(sequence, vocab, model, device)
    unique_vertices, edge_vertex_adj = infer_vertices(
        parsed["edge_ncs"], parsed["edge_bbox_wcs"], parsed["faceEdge_adj"]
    )
    surf_wcs, edge_wcs = joint_optimize(
        parsed["surf_ncs"],
        parsed["edge_ncs"],
        parsed["surf_bbox_wcs"],
        unique_vertices,
        edge_vertex_adj,
        parsed["faceEdge_adj"],
        len(parsed["edge_ncs"]),
        len(parsed["surf_ncs"]),
    )
    solid = construct_brep(surf_wcs, edge_wcs, parsed["faceEdge_adj"], edge_vertex_adj)
    valid = bool(check_brep_validity(solid)) if solid is not None else False
    return solid, valid, parsed, unique_vertices, edge_vertex_adj


def sequence_vocab(data: dict) -> dict:
    keys = [
        "face_index_offset",
        "face_index_size",
        "quantization_offset",
        "quantization_size",
        "bbox_token_offset",
        "bbox_index_size",
        "bbox_size",
        "se_token_offset",
        "se_codebook_size",
        "se_tokens_per_element",
        "bbox_tokens_per_element",
        "face_vq_tokens",
        "START_TOKEN",
        "SEP_TOKEN",
        "END_TOKEN",
        "special_tokens",
    ]
    return {key: data[key] for key in keys if key in data}


def debug_sample(data: dict, split: str, index: int, model, device, output_dir: Path, save_step: bool):
    group = data[split][index]
    sequence = group["original"]["input_ids"]
    if torch.is_tensor(sequence):
        sequence = sequence.detach().cpu().tolist()
    result = {
        "split": split,
        "index": index,
        "name": group.get("name"),
        "source_file": group.get("source_file"),
        "length": len(sequence),
    }
    try:
        solid, valid, parsed, unique_vertices, edge_vertex_adj = reconstruct_one(
            sequence, sequence_vocab(data), model, device
        )
        result.update(
            {
                "solid_returned": solid is not None,
                "brep_valid": valid,
                "num_faces": int(len(parsed["surf_ncs"])),
                "num_edges": int(len(parsed["edge_ncs"])),
                "num_vertices": int(len(unique_vertices)),
                "min_face_degree": int(min((len(x) for x in parsed["faceEdge_adj"]), default=0)),
                "max_face_degree": int(max((len(x) for x in parsed["faceEdge_adj"]), default=0)),
                "error": None,
            }
        )
        if save_step and solid is not None:
            from OCC.Extend.DataExchange import write_step_file

            step_path = output_dir / f"{split}_{index:06d}_{group.get('name', 'sample')}.step"
            write_step_file(solid, str(step_path))
            result["step_file"] = str(step_path)
    except Exception as exc:
        result.update(
            {
                "solid_returned": False,
                "brep_valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description="Debug BREP reconstruction from NURBS v2 AR sequences.")
    parser.add_argument("--sequence_file", type=str, required=True)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/checkpoint/se/abc/8192,4096,128,64,false,1e-4,0,p/deepcad_nurbs_vqvae_best.pt",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_step", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_vqvae_model(args.ckpt_path, device)
    model.eval()
    data = load_pickle(args.sequence_file)

    records = []
    end = min(len(data[args.split]), args.index + args.max_samples)
    for idx in range(args.index, end):
        records.append(debug_sample(data, args.split, idx, model, device, output_dir, args.save_step))

    summary = {
        "sequence_file": args.sequence_file,
        "split": args.split,
        "index": args.index,
        "max_samples": args.max_samples,
        "evaluated": len(records),
        "solid_returned": sum(1 for r in records if r.get("solid_returned")),
        "brep_valid": sum(1 for r in records if r.get("brep_valid")),
        "errors": {},
    }
    for record in records:
        if record.get("error"):
            summary["errors"][record["error"]] = summary["errors"].get(record["error"], 0) + 1
    summary["solid_rate"] = summary["solid_returned"] / len(records) if records else 0.0
    summary["valid_rate"] = summary["brep_valid"] / len(records) if records else 0.0

    with open(output_dir / "records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("NURBS v2 Sequence Reconstruction Debug")
    print("=" * 60)
    for key in ("evaluated", "solid_returned", "brep_valid", "solid_rate", "valid_rate"):
        print(f"{key}: {summary[key]}")
    if records and len(records) == 1:
        record = records[0]
        print(f"name: {record.get('name')}")
        print(f"num_faces: {record.get('num_faces')}")
        print(f"num_edges: {record.get('num_edges')}")
        print(f"error: {record.get('error')}")
    print("-" * 60)
    print(f"records_json: {output_dir / 'records.json'}")
    print(f"summary_json: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
