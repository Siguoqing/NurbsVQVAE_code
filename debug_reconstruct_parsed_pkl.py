#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from utils import (
    check_brep_validity,
    construct_brep,
    create_bspline_curve,
    create_bspline_surface,
    sample_bspline_curve,
    sample_bspline_surface,
)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def choose_path_from_split(split_file: str, split: str, index: int) -> str:
    data = load_pickle(split_file)
    paths = data[split]
    if index < 0 or index >= len(paths):
        raise IndexError(f"index {index} out of range for split={split}, len={len(paths)}")
    return paths[index]


def edge_pairs_from_adj(edge_face_adj) -> list[tuple[int, int]]:
    pairs = []
    if edge_face_adj is None:
        return pairs
    for item in edge_face_adj:
        if len(item) >= 2:
            pairs.append((int(item[0]), int(item[1])))
    return pairs


def get_array(cad: dict, keys: tuple[str, ...]):
    for key in keys:
        value = cad.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.size > 0:
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


def make_face_edge_adj(num_faces: int, edge_face_pairs: list[tuple[int, int]]) -> list[list[int]]:
    face_adj = [[] for _ in range(num_faces)]
    for edge_idx, (f1, f2) in enumerate(edge_face_pairs):
        if 0 <= f1 < num_faces:
            face_adj[f1].append(edge_idx)
        if 0 <= f2 < num_faces and f2 != f1:
            face_adj[f2].append(edge_idx)
    return face_adj


def make_edge_vertex_adj(edge_wcs: np.ndarray, face_edge_adj: list[list[int]]) -> np.ndarray:
    edge_v_bbox = np.asarray([edge[[0, -1]] for edge in edge_wcs], dtype=np.float64)
    total_vertices = len(edge_wcs) * 2
    parent = list(range(total_vertices))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for edge_indices in face_edge_adj:
        face_vertices = []
        for edge_idx in edge_indices:
            for vertex_pos_idx in (0, 1):
                vertex_id = edge_idx * 2 + vertex_pos_idx
                face_vertices.append((vertex_id, edge_v_bbox[edge_idx, vertex_pos_idx]))

        used = set()
        for _ in range(len(face_vertices) // 2):
            best = None
            best_dist = float("inf")
            for i, (vid_i, pos_i) in enumerate(face_vertices):
                if i in used:
                    continue
                for j, (vid_j, pos_j) in enumerate(face_vertices):
                    if j <= i or j in used:
                        continue
                    if vid_i // 2 == vid_j // 2:
                        continue
                    dist = float(np.linalg.norm(pos_i - pos_j))
                    if dist < best_dist:
                        best = (i, j, vid_i, vid_j)
                        best_dist = dist
            if best is None:
                break
            i, j, vid_i, vid_j = best
            union(vid_i, vid_j)
            used.add(i)
            used.add(j)

    groups = {}
    for vertex_id in range(total_vertices):
        groups.setdefault(find(vertex_id), []).append(vertex_id)

    vertex_mapping = [-1] * total_vertices
    for unique_idx, group in enumerate(groups.values()):
        for vertex_id in group:
            vertex_mapping[vertex_id] = unique_idx

    edge_vertex_adj = np.zeros((len(edge_wcs), 2), dtype=np.int32)
    for edge_idx in range(len(edge_wcs)):
        edge_vertex_adj[edge_idx, 0] = vertex_mapping[edge_idx * 2]
        edge_vertex_adj[edge_idx, 1] = vertex_mapping[edge_idx * 2 + 1]
    return edge_vertex_adj


def reconstruct_from_controls(face_ctrs: np.ndarray, edge_ctrs: np.ndarray, edge_face_pairs: list[tuple[int, int]]):
    surf_wcs = []
    for face_ctr in face_ctrs:
        surface = create_bspline_surface(face_ctr)
        surf_wcs.append(sample_bspline_surface(surface, num_u=32, num_v=32))

    edge_wcs = []
    for edge_ctr in edge_ctrs:
        curve = create_bspline_curve(edge_ctr)
        edge_wcs.append(sample_bspline_curve(curve, num_points=32))

    surf_wcs = np.asarray(surf_wcs, dtype=np.float64)
    edge_wcs = np.asarray(edge_wcs, dtype=np.float64)
    face_edge_adj = make_face_edge_adj(len(surf_wcs), edge_face_pairs)
    edge_vertex_adj = make_edge_vertex_adj(edge_wcs, face_edge_adj)
    return construct_brep(surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj), surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj


def reconstruct_from_sampled_wcs(cad: dict):
    surf_wcs, surf_key = get_array(cad, ("surf_wcs", "face_wcs", "face_pnts"))
    edge_wcs, edge_key = get_array(cad, ("edge_wcs", "edge_pnts"))
    if surf_wcs is None or edge_wcs is None:
        raise KeyError("Cannot find sampled WCS arrays: expected surf_wcs/edge_wcs.")

    face_edge_raw = cad.get("faceEdge_adj")
    edge_vertex_raw = cad.get("edgeCorner_adj")
    face_edge_adj = normalize_face_edge_adj(face_edge_raw, len(surf_wcs))
    edge_vertex_adj = normalize_edge_vertex_adj(edge_vertex_raw, len(edge_wcs))

    solid = construct_brep(
        np.asarray(surf_wcs, dtype=np.float64),
        np.asarray(edge_wcs, dtype=np.float64),
        face_edge_adj,
        edge_vertex_adj,
    )
    meta = {
        "geometry_mode": "sampled_wcs",
        "surf_key": surf_key,
        "edge_key": edge_key,
        "has_faceEdge_adj": face_edge_raw is not None,
        "has_edgeCorner_adj": edge_vertex_raw is not None,
    }
    return solid, np.asarray(surf_wcs), np.asarray(edge_wcs), face_edge_adj, edge_vertex_adj, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug BREP reconstruction directly from a parsed NURBS pkl.")
    parser.add_argument("--pkl", default="")
    parser.add_argument("--split_file", default="")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", default="result/debug_reconstruct_parsed_pkl")
    args = parser.parse_args()

    if args.pkl:
        pkl_path = args.pkl
    elif args.split_file:
        pkl_path = choose_path_from_split(args.split_file, args.split, args.index)
    else:
        raise ValueError("Provide either --pkl or --split_file.")

    cad = load_pickle(pkl_path)
    face_ctrs, face_ctrs_key = get_array(cad, ("face_ctrs", "face_ctrs_wcs_norm"))
    edge_ctrs, edge_ctrs_key = get_array(cad, ("edge_ctrs", "edge_ctrs_wcs_norm"))
    edge_face_pairs = edge_pairs_from_adj(cad.get("edgeFace_adj"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    error = None
    solid = None
    surf_wcs = np.zeros((0, 32, 32, 3), dtype=np.float32)
    edge_wcs = np.zeros((0, 32, 3), dtype=np.float32)
    face_edge_adj = []
    edge_vertex_adj = np.zeros((0, 2), dtype=np.int32)
    meta = {"available_keys": sorted([str(k) for k in cad.keys()])}
    try:
        if "surf_wcs" in cad and "edge_wcs" in cad:
            solid, surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj, mode_meta = reconstruct_from_sampled_wcs(cad)
            meta.update(mode_meta)
        elif face_ctrs is not None and edge_ctrs is not None:
            solid, surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj = reconstruct_from_controls(
                np.asarray(face_ctrs, dtype=np.float32),
                np.asarray(edge_ctrs, dtype=np.float32),
                edge_face_pairs,
            )
            meta.update({
                "geometry_mode": "controls",
                "face_ctrs_key": face_ctrs_key,
                "edge_ctrs_key": edge_ctrs_key,
            })
        else:
            raise KeyError("Cannot find face/edge geometry arrays.")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    is_valid = False
    if solid is not None:
        try:
            is_valid = bool(check_brep_validity(solid))
        except Exception as exc:
            error = f"validity_check_failed: {type(exc).__name__}: {exc}"

    summary = {
        "pkl_path": pkl_path,
        **meta,
        "num_faces": int(len(surf_wcs)) if len(surf_wcs) else int(len(face_ctrs)) if face_ctrs is not None else 0,
        "num_edges": int(len(edge_wcs)) if len(edge_wcs) else int(len(edge_ctrs)) if edge_ctrs is not None else 0,
        "num_edge_face_pairs": int(len(edge_face_pairs)),
        "solid_returned": solid is not None,
        "brep_valid": is_valid,
        "error": error,
        "face_edge_degrees": [len(x) for x in face_edge_adj],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        output_dir / "arrays.npz",
        face_ctrs=np.asarray(face_ctrs if face_ctrs is not None else [], dtype=np.float32),
        edge_ctrs=np.asarray(edge_ctrs if edge_ctrs is not None else [], dtype=np.float32),
        edgeFace_adj=np.asarray(edge_face_pairs, dtype=np.int32),
        surf_wcs=np.asarray(surf_wcs, dtype=np.float32),
        edge_wcs=np.asarray(edge_wcs, dtype=np.float32),
        EdgeVertexAdj=np.asarray(edge_vertex_adj, dtype=np.int32),
    )

    print("=" * 60)
    print("Parsed PKL Reconstruction Debug")
    print("=" * 60)
    for key, value in summary.items():
        if key != "face_edge_degrees":
            print(f"{key}: {value}")
    print(f"min/max face degree: {min(summary['face_edge_degrees'], default=0)} / {max(summary['face_edge_degrees'], default=0)}")
    print("-" * 60)
    print(f"debug_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
