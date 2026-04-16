#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np

from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt, TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def ensure_array(a):
    return np.array(a) if not isinstance(a, np.ndarray) else a


def is_empty_value(value):
    """
    Check if a value is empty (None, empty array, empty list, etc.)
    
    Args:
        value: Value to check
    
    Returns:
        True if value is empty, False otherwise
    """
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def build_vertFace_adj(data: dict) -> list:
    verts = data.get("corner_unique", data.get("vert_wcs"))
    if verts is None:
        raise KeyError("Neither 'corner_unique' nor 'vert_wcs' found in data.")
    nv = int(ensure_array(verts).shape[0])

    edge_v = data.get("edgeCorner_adj", data.get("edgeVert_adj"))
    if edge_v is None:
        raise KeyError("Neither 'edgeCorner_adj' nor 'edgeVert_adj' found in data.")
    edge_v = ensure_array(edge_v)
    if edge_v.ndim != 2 or edge_v.shape[1] != 2:
        raise ValueError(f"edge-vertex adjacency shape invalid: {edge_v.shape}")

    edge_face = data.get("edgeFace_adj", None)
    if edge_face is None:
        raise KeyError("'edgeFace_adj' not found in data.")
    edge_face = ensure_array(edge_face)
    if edge_face.ndim < 2 or edge_face.shape[0] != edge_v.shape[0]:
        raise ValueError("edgeFace_adj and edgeCorner_adj length mismatch.")

    vertex_edges = [[] for _ in range(nv)]
    for e_id, (v1, v2) in enumerate(edge_v):
        vertex_edges[v1].append(e_id)
        vertex_edges[v2].append(e_id)

    vertFace_adj = []
    for v_id in range(nv):
        e_ids = vertex_edges[v_id]
        faces = []
        for e in e_ids:
            faces.extend(np.array(edge_face[e]).reshape(-1).tolist())
        faces = sorted(set(int(x) for x in faces if int(x) >= 0))
        vertFace_adj.append(faces)

    return vertFace_adj


def fit_bspline_controls(data: dict, verbose: bool = False):
    face_ncs = data.get("surf_ncs", data.get("face_ncs"))
    face_ctrs = None
    if face_ncs is not None:
        face_ncs = ensure_array(face_ncs)
        try:
            f_ctrs = []
            for points in face_ncs:  # (32, 32, 3)
                nu, nv = points.shape[0], points.shape[1]
                arr = TColgp_Array2OfPnt(1, nu, 1, nv)
                for ui in range(1, nu + 1):
                    for vi in range(1, nv + 1):
                        x, y, z = map(float, points[ui - 1, vi - 1])
                        arr.SetValue(ui, vi, gp_Pnt(x, y, z))
                approx = GeomAPI_PointsToBSplineSurface(arr, 3, 3, GeomAbs_C2, 5e-2).Surface()
                nu_p, nv_p = approx.NbUPoles(), approx.NbVPoles()
                if nu_p != 4 or nv_p != 4:
                    raise RuntimeError(f"Surface poles not 4x4, got {nu_p}x{nv_p}.")
                poles = approx.Poles()
                ctr = np.zeros((nu_p * nv_p, 3), dtype=np.float32)
                k = 0
                for u in range(1, nu_p + 1):
                    for v in range(1, nv_p + 1):
                        p = poles.Value(u, v)
                        ctr[k] = [p.X(), p.Y(), p.Z()]
                        k += 1
                f_ctrs.append(ctr)
            face_ctrs = np.stack(f_ctrs, axis=0)  # (nf, 16, 3)
        except Exception as e:
            if verbose:
                print("[fit] face_ctrs failed:", e)
            face_ctrs = None

    edge_ncs = data.get("edge_ncs", None)
    edge_ctrs = None
    if edge_ncs is not None:
        edge_ncs = ensure_array(edge_ncs)
        try:
            e_ctrs = []
            for points in edge_ncs:  # (32, 3)
                nu = points.shape[0]
                arr = TColgp_Array1OfPnt(1, nu)
                for i in range(1, nu + 1):
                    x, y, z = map(float, points[i - 1])
                    arr.SetValue(i, gp_Pnt(x, y, z))
                approx = None
                for tol in (5e-3, 8e-3, 5e-2):
                    try:
                        approx = GeomAPI_PointsToBSpline(arr, 3, 3, GeomAbs_C2, tol).Curve()
                        break
                    except Exception:
                        approx = None
                if approx is None:
                    raise RuntimeError("Edge bspline fitting failed under all tolerances.")
                npoles = approx.NbPoles()
                if npoles != 4:
                    raise RuntimeError(f"Edge poles not 4, got {npoles}.")
                poles = approx.Poles()
                ctr = np.zeros((npoles, 3), dtype=np.float32)
                for i in range(1, npoles + 1):
                    p = poles.Value(i)
                    ctr[i - 1] = [p.X(), p.Y(), p.Z()]
                e_ctrs.append(ctr)
            edge_ctrs = np.stack(e_ctrs, axis=0)  # (ne, 4, 3)
        except Exception as e:
            if verbose:
                print("[fit] edge_ctrs failed:", e)
            edge_ctrs = None

    # # 整体归一化 face_ctrs 和 edge_ctrs
    # if face_ctrs is not None or edge_ctrs is not None:
    #     # 收集所有控制点用于计算全局归一化参数
    #     all_points = []
    #     if face_ctrs is not None:
    #         all_points.append(face_ctrs.reshape(-1, 3))
    #     if edge_ctrs is not None:
    #         all_points.append(edge_ctrs.reshape(-1, 3))
        
    #     if len(all_points) > 0:
    #         total_points = np.concatenate(all_points, axis=0)
    #         min_vals = np.min(total_points, axis=0)
    #         max_vals = np.max(total_points, axis=0)
    #         global_offset = min_vals + (max_vals - min_vals) / 2
    #         global_scale = np.max(max_vals - min_vals)
            
    #         if global_scale > 1e-10:  # 避免除零
    #             # 归一化 face_ctrs
    #             if face_ctrs is not None:
    #                 face_ctrs = (face_ctrs - global_offset[np.newaxis, np.newaxis, :]) / (global_scale * 0.5)
                
    #             # 归一化 edge_ctrs
    #             if edge_ctrs is not None:
    #                 edge_ctrs = (edge_ctrs - global_offset[np.newaxis, np.newaxis, :]) / (global_scale * 0.5)
    #         elif verbose:
    #             print("[fit] Warning: global_scale too small, skipping normalization")

    return face_ctrs, edge_ctrs


def count_fef_adj(face_edge):
    num_faces = len(face_edge)
    fef_adj = np.zeros((num_faces, num_faces), dtype=int)
    face_edge_sets = [set(fe) for fe in face_edge]
    for i in range(num_faces):
        for j in range(i + 1, num_faces):
            common_elements = face_edge_sets[i].intersection(face_edge_sets[j])
            common_count = len(common_elements)
            fef_adj[i, j] = common_count
            fef_adj[j, i] = common_count
    return fef_adj


def parse_args():
    ap = argparse.ArgumentParser(description="Process pkl files from train/val split")
    ap.add_argument("--input-list", type=str, default="data/extracted_close_samples_10k.pkl", 
                    help="Path to pkl file containing train/val split")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory to save summary results")
    ap.add_argument("--verbose", action="store_true", help="Print detailed error messages")
    return ap.parse_args()


def load_pkl_paths(input_list: str):
    """
    Load pkl file paths from train and val splits.
    
    Args:
        input_list: Path to pkl file containing {'train': [...], 'val': [...], ...}
    
    Returns:
        List of paths to pkl files (train + val)
    """
    with open(input_list, "rb") as f:
        data = pickle.load(f)
    
    pkl_paths = []
    if 'train' in data:
        pkl_paths.extend(data['train'])
    if 'val' in data:
        pkl_paths.extend(data['val'])
    
    return pkl_paths


def main():
    args = parse_args()

    required_keys = {"face_ctrs", "edge_ctrs", "fef_adj", "vertFace_adj"}
    pkl_paths = load_pkl_paths(args.input_list)
    print(f"Found {len(pkl_paths)} pkl files to process (train + val)")

    success_files, skipped_files, failed_files = [], [], []

    for pkl_path in tqdm(pkl_paths, desc="Processing PKLs", ncols=100):
        try:
            data = load_pickle(pkl_path)

            # 已经处理过的，直接跳过（检查键是否存在且非空）
            all_keys_exist = required_keys.issubset(data.keys())
            if all_keys_exist:
                # 检查所有必需键的值是否非空
                all_values_valid = all(
                    not is_empty_value(data[key]) for key in required_keys
                )
                if all_values_valid:
                    skipped_files.append(pkl_path)
                    continue

            # 1) 构建拓扑信息
            try:
                fef_adj = count_fef_adj(data["faceEdge_adj"])
                data["fef_adj"] = fef_adj
                data["vertFace_adj"] = build_vertFace_adj(data)
            except Exception as e:
                if args.verbose:
                    print(f"[vertFace_adj failed] {pkl_path}: {e}")
                failed_files.append(pkl_path)
                continue

            # 2) 曲线/曲面拟合
            face_ctrs, edge_ctrs = fit_bspline_controls(data, verbose=args.verbose)
            
            # 检查拟合是否成功：两者都必须非空
            if face_ctrs is None or edge_ctrs is None:
                if args.verbose:
                    reason = []
                    if face_ctrs is None:
                        reason.append("face_ctrs")
                    if edge_ctrs is None:
                        reason.append("edge_ctrs")
                    print(f"[fit failed] {pkl_path}: {' and '.join(reason)} 拟合失败")
                failed_files.append(pkl_path)
                continue
            
            # 检查拟合结果是否为空数组
            if is_empty_value(face_ctrs) or is_empty_value(edge_ctrs):
                if args.verbose:
                    reason = []
                    if is_empty_value(face_ctrs):
                        reason.append("face_ctrs")
                    if is_empty_value(edge_ctrs):
                        reason.append("edge_ctrs")
                    print(f"[fit empty] {pkl_path}: {' and '.join(reason)} 为空")
                failed_files.append(pkl_path)
                continue
            
            # 拟合成功，保存到数据中
            data["face_ctrs"] = face_ctrs
            data["edge_ctrs"] = edge_ctrs

            # 3) 保存更新后的数据
            save_pickle(pkl_path, data)
            success_files.append(pkl_path)

        except Exception as e:
            if args.verbose:
                print(f"[error] {pkl_path}: {e}")
            failed_files.append(pkl_path)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.input_list))
    os.makedirs(out_dir, exist_ok=True)

    save_pickle(os.path.join(out_dir, "deepcad_successful_files.pkl"), success_files)
    save_pickle(os.path.join(out_dir, "deepcad_skipped_files.pkl"), skipped_files)
    save_pickle(os.path.join(out_dir, "deepcad_failed_files.pkl"), failed_files)

    print("\n== Done ==")
    print(f"✅ 成功处理: {len(success_files)}")
    print(f"⚙️ 已跳过: {len(skipped_files)} (已包含所有目标键)")
    print(f"❌ 失败: {len(failed_files)}")
    print(f"结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
