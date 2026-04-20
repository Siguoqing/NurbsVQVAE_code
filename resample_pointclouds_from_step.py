#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
from pathlib import Path

import numpy as np
from tqdm import tqdm

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods


def load_step_shape(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"Null shape from STEP: {step_path}")
    return shape


def triangle_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def get_triangulation_node(triangulation, node_idx: int):
    if hasattr(triangulation, "Node"):
        return triangulation.Node(node_idx)
    nodes = triangulation.Nodes()
    return nodes.Value(node_idx)


def get_triangulation_triangle(triangulation, tri_idx: int):
    if hasattr(triangulation, "Triangle"):
        return triangulation.Triangle(tri_idx)
    triangles = triangulation.Triangles()
    return triangles.Value(tri_idx)


def sample_points_on_triangles(vertices: np.ndarray, triangles: np.ndarray, counts: np.ndarray, rng) -> np.ndarray:
    sampled_points = []
    for tri_idx, count in enumerate(counts):
        if count <= 0:
            continue
        i0, i1, i2 = triangles[tri_idx]
        a = vertices[i0]
        b = vertices[i1]
        c = vertices[i2]

        r1 = rng.random(count).astype(np.float32, copy=False)
        r2 = rng.random(count).astype(np.float32, copy=False)
        sqrt_r1 = np.sqrt(r1)
        points = (
            (1.0 - sqrt_r1)[:, None] * a
            + (sqrt_r1 * (1.0 - r2))[:, None] * b
            + (sqrt_r1 * r2)[:, None] * c
        )
        sampled_points.append(points.astype(np.float32, copy=False))

    if not sampled_points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(sampled_points, axis=0)


def extract_face_triangles(shape, linear_deflection: float, angular_deflection: float):
    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesher.Perform()

    face_data = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0

    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        explorer.Next()

        if triangulation is None:
            face_idx += 1
            continue

        trsf = location.Transformation()
        vertices = []
        for node_idx in range(1, triangulation.NbNodes() + 1):
            pnt = get_triangulation_node(triangulation, node_idx).Transformed(trsf)
            vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

        vertices = np.asarray(vertices, dtype=np.float32)
        if len(vertices) == 0:
            face_idx += 1
            continue

        triangles = []
        areas = []
        for tri_idx in range(1, triangulation.NbTriangles() + 1):
            triangle = get_triangulation_triangle(triangulation, tri_idx)
            i1, i2, i3 = triangle.Get()
            tri = np.array([i1 - 1, i2 - 1, i3 - 1], dtype=np.int32)
            area = triangle_area(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]])
            if area <= 1e-12 or not np.isfinite(area):
                continue
            triangles.append(tri)
            areas.append(area)

        if triangles:
            face_data.append(
                {
                    "face_index": face_idx,
                    "vertices": vertices,
                    "triangles": np.asarray(triangles, dtype=np.int32),
                    "areas": np.asarray(areas, dtype=np.float64),
                }
            )

        face_idx += 1

    return face_data


def sample_point_cloud_from_shape(shape, num_points: int, linear_deflection: float, angular_deflection: float, rng):
    face_data = extract_face_triangles(shape, linear_deflection, angular_deflection)
    if not face_data:
        raise RuntimeError("No triangulated faces found")

    all_face_areas = np.array([face["areas"].sum() for face in face_data], dtype=np.float64)
    total_area = float(all_face_areas.sum())
    if total_area <= 0 or not np.isfinite(total_area):
        raise RuntimeError("Invalid total mesh area")

    face_probs = all_face_areas / total_area
    face_counts = rng.multinomial(num_points, face_probs)

    point_chunks = []
    label_chunks = []
    for face, face_count in zip(face_data, face_counts):
        if face_count <= 0:
            continue
        tri_probs = face["areas"] / face["areas"].sum()
        tri_counts = rng.multinomial(face_count, tri_probs)
        face_points = sample_points_on_triangles(face["vertices"], face["triangles"], tri_counts, rng)
        if len(face_points) == 0:
            continue
        point_chunks.append(face_points)
        label_chunks.append(np.full((len(face_points),), face["face_index"], dtype=np.int32))

    if not point_chunks:
        raise RuntimeError("Failed to sample any point from triangulation")

    points = np.concatenate(point_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)

    if len(points) != num_points:
        if len(points) > num_points:
            select = rng.choice(len(points), size=num_points, replace=False)
        else:
            select = rng.choice(len(points), size=num_points, replace=True)
        points = points[select]
        labels = labels[select]

    return points.astype(np.float32, copy=False), labels.astype(np.int32, copy=False)


def process_model_dir(model_dir: Path, num_points: int, linear_deflection: float, angular_deflection: float, seed: int, overwrite: bool):
    point_path = model_dir / "point_cloud.npy"
    label_path = model_dir / "face_labels.npy"

    if not overwrite and point_path.exists() and label_path.exists():
        return {"model_dir": str(model_dir), "status": "skipped_existing"}

    step_candidates = sorted(model_dir.glob("*.step")) + sorted(model_dir.glob("*.stp"))
    if not step_candidates:
        return {"model_dir": str(model_dir), "status": "missing_step"}

    step_path = step_candidates[0]
    rng = np.random.default_rng(seed)

    try:
        shape = load_step_shape(step_path)
        points, labels = sample_point_cloud_from_shape(
            shape=shape,
            num_points=num_points,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            rng=rng,
        )
        np.save(point_path, points)
        np.save(label_path, labels)
        return {
            "model_dir": str(model_dir),
            "status": "ok",
            "step_file": step_path.name,
            "num_points": int(len(points)),
            "num_faces_hit": int(len(np.unique(labels))),
        }
    except Exception as exc:
        return {
            "model_dir": str(model_dir),
            "status": "failed",
            "step_file": step_path.name,
            "error": str(exc),
        }


def _process_model_dir_worker(result_queue, model_dir_str: str, num_points: int, linear_deflection: float, angular_deflection: float, seed: int, overwrite: bool):
    result = process_model_dir(
        model_dir=Path(model_dir_str),
        num_points=num_points,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        seed=seed,
        overwrite=overwrite,
    )
    result_queue.put(result)


def process_model_dir_isolated(model_dir: Path, num_points: int, linear_deflection: float, angular_deflection: float, seed: int, overwrite: bool, timeout: int):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_process_model_dir_worker,
        args=(
            result_queue,
            str(model_dir),
            num_points,
            linear_deflection,
            angular_deflection,
            seed,
            overwrite,
        ),
    )
    proc.start()
    proc.join(timeout=timeout if timeout > 0 else None)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "model_dir": str(model_dir),
            "status": "failed",
            "error": f"timeout after {timeout}s",
        }

    if proc.exitcode != 0:
        return {
            "model_dir": str(model_dir),
            "status": "failed",
            "error": f"worker crashed with exit code {proc.exitcode}",
        }

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {
            "model_dir": str(model_dir),
            "status": "failed",
            "error": "worker exited without returning a result",
        }


def collect_model_dirs(root: Path, single_model: str | None, limit: int | None):
    if single_model:
        model_dirs = [root / single_model]
    else:
        model_dirs = sorted(path for path in root.glob("model_*") if path.is_dir())
    if limit is not None:
        model_dirs = model_dirs[:limit]
    return model_dirs


def main():
    parser = argparse.ArgumentParser(description="Resample point clouds from STEP files and save point_cloud.npy + face_labels.npy.")
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/ABC_Dataset_NEW",
        help="Root directory containing model_xxxxx folders",
    )
    parser.add_argument("--model", type=str, default=None, help="Optional single model directory name, e.g. model_00001")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N model directories")
    parser.add_argument("--num_points", type=int, default=4096, help="Number of sampled points per STEP model")
    parser.add_argument("--linear_deflection", type=float, default=0.1, help="Meshing linear deflection")
    parser.add_argument("--angular_deflection", type=float, default=0.5, help="Meshing angular deflection")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing point_cloud.npy and face_labels.npy")
    parser.add_argument("--timeout", type=int, default=120, help="Per-model timeout in seconds when running isolated workers")
    parser.add_argument("--no_isolation", action="store_true", help="Disable per-model subprocess isolation")
    parser.add_argument(
        "--report_json",
        type=str,
        default="",
        help="Optional json report path",
    )
    args = parser.parse_args()

    root = Path(args.root)
    model_dirs = collect_model_dirs(root, args.model, args.limit)
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found under: {root}")

    results = []
    ok_count = 0
    fail_count = 0
    skip_count = 0

    for idx, model_dir in enumerate(tqdm(model_dirs, desc="Resampling STEP point clouds", unit="model")):
        if args.no_isolation:
            result = process_model_dir(
                model_dir=model_dir,
                num_points=args.num_points,
                linear_deflection=args.linear_deflection,
                angular_deflection=args.angular_deflection,
                seed=args.seed + idx,
                overwrite=args.overwrite,
            )
        else:
            result = process_model_dir_isolated(
                model_dir=model_dir,
                num_points=args.num_points,
                linear_deflection=args.linear_deflection,
                angular_deflection=args.angular_deflection,
                seed=args.seed + idx,
                overwrite=args.overwrite,
                timeout=args.timeout,
            )
        results.append(result)
        status = result["status"]
        if status == "ok":
            ok_count += 1
        elif status == "failed":
            fail_count += 1
        elif status == "skipped_existing":
            skip_count += 1

    summary = {
        "root": str(root),
        "processed": len(model_dirs),
        "ok": ok_count,
        "failed": fail_count,
        "skipped_existing": skip_count,
        "num_points": args.num_points,
        "linear_deflection": args.linear_deflection,
        "angular_deflection": args.angular_deflection,
        "isolation": not args.no_isolation,
        "timeout": args.timeout,
    }

    print("=" * 60)
    print("STEP Point Cloud Resampling Summary")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
        print(f"report_json: {report_path}")


if __name__ == "__main__":
    main()
