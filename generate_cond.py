#!/usr/bin/env python3
import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import random
import shutil
import tempfile
import signal
import multiprocessing
import traceback
import platform
import glob
from tqdm import tqdm
from typing import Optional, List, Dict, Any

from model import LLaMA3Config, LLaMA3ARModel
from pointcloud_condition import load_and_preprocess_point_cloud
from utils import (
    check_brep_validity,
    compute_bbox_center_and_size,
    construct_brep,
    create_bspline_curve,
    create_bspline_surface,
    joint_optimize,
    parse_sequence_to_cad_data_nurbs,
    reconstruct_cad_from_sequence_nurbs,
    sample_bspline_curve,
    sample_bspline_surface,
)
from vqvae_eval_utils import load_model_from_checkpoint


def load_vqvae_model(ckpt_path: str, device):
    loaded = load_model_from_checkpoint(ckpt_path, device)
    if isinstance(loaded, (tuple, list)) and len(loaded) >= 1:
        return loaded[0]
    raise ValueError("load_model_from_checkpoint must return a model")


# ================= [底层安全写入与并行处理] =================
def _global_write_worker(temp_step_path, final_step_path, final_stl_path, result_file_path):
    def write_result(status, error_msg=None):
        try:
            with open(result_file_path, 'w') as f:
                json.dump({'status': status, 'error': str(error_msg) if error_msg else None}, f)
        except Exception as e:
            print(f"[Worker] 无法写入结果文件: {e}")

    try:
        try:
            from OCC.Extend.DataExchange import write_stl_file, read_step_file
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.IFSelect import IFSelect_RetDone
        except ImportError as e:
            write_result('error', f'OpenCASCADE 模块导入失败: {e}')
            return

        shape = None
        try:
            shape = read_step_file(temp_step_path)
        except Exception:
            pass

        if shape is None:
            step_reader = STEPControl_Reader()
            status = step_reader.ReadFile(temp_step_path)
            if status != IFSelect_RetDone:
                write_result('error', f'无法读取临时 STEP 文件: {temp_step_path}')
                return
            step_reader.TransferRoots()
            shape = step_reader.OneShape()

        if shape is None or shape.IsNull():
            write_result('error', '从 STEP 文件读取的 Shape 为空')
            return

        try:
            write_stl_file(shape, final_stl_path, linear_deflection=0.001, angular_deflection=0.5)
        except Exception as e:
            write_result('stl_failed', f'STL 写入失败: {e}')
            return

        shutil.copy2(temp_step_path, final_step_path)
        write_result('success', None)

    except Exception as e:
        write_result('error', f"子进程发生未捕获异常: {e}\n{traceback.format_exc()}")


def timeout_handler(signum, frame):
    raise TimeoutError("操作超时")


def write_files_safe(solid, step_path, stl_path, write_timeout=30):
    fd, temp_step_path = tempfile.mkstemp(suffix=".step", prefix="cad_temp_")
    os.close(fd)
    fd_res, result_file_path = tempfile.mkstemp(suffix=".json", prefix="cad_res_")
    os.close(fd_res)

    is_windows = platform.system() == 'Windows'

    try:
        try:
            if not is_windows:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)

            from OCC.Extend.DataExchange import write_step_file
            write_step_file(solid, temp_step_path)

            if not is_windows:
                signal.alarm(0)
        except TimeoutError:
            return 'timeout', "主进程导出临时 STEP 超时"
        except Exception as e:
            if not is_windows: signal.alarm(0)
            return 'error', f"主进程导出 STEP 失败: {e}"

        p = multiprocessing.Process(
            target=_global_write_worker,
            args=(temp_step_path, step_path, stl_path, result_file_path)
        )
        p.start()
        p.join(timeout=write_timeout)

        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
            if p.is_alive(): p.kill()
            return 'timeout', f"子进程写入 STL 超时 ({write_timeout}s)"

        if os.path.exists(result_file_path) and os.path.getsize(result_file_path) > 0:
            with open(result_file_path, 'r') as f:
                res = json.load(f)
                return res.get('status', 'error'), res.get('error', None)
        else:
            return 'error', "子进程未生成结果文件"

    finally:
        for path in [temp_step_path, result_file_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


# ================= [资源加载与序列展平] =================
def load_resources(model_path: str, device: torch.device, config_path: str = "config.json"):
    checkpoint = torch.load(model_path, map_location='cpu')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_json = json.load(f)

    if uses_v2_protocol(config_json):
        config_json.setdefault("bbox_tokens_per_element", 6)
        config_json.setdefault("face_vq_tokens", 4)
        config_json.setdefault("edge_vq_tokens", 4)
        config_json.setdefault("se_tokens_per_element", 4)
        config_json.setdefault("face_block", config_json["bbox_tokens_per_element"] + config_json["face_vq_tokens"] + 1)
        config_json.setdefault("edge_block", 2 + config_json["bbox_tokens_per_element"] + config_json["edge_vq_tokens"])

    state_dict = checkpoint.get('model_state_dict', {})

    llama_config = LLaMA3Config(
        vocab_size=config_json['vocab_size'],
        d_model=config_json['d_model'],
        n_layers=config_json['n_layers'],
        n_heads=config_json['n_heads'],
        n_kv_heads=config_json['n_kv_heads'],
        dim_feedforward=config_json['dim_feedforward'],
        dropout=config_json['dropout'],
        pad_token_id=config_json['PAD_TOKEN'],
        num_components=config_json.get('num_components', 1),
        point_prefix_tokens=config_json.get('point_prefix_tokens', 8),
    )

    model = LLaMA3ARModel(llama_config).to(device)
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('module.', '').replace('model.', '')
        new_state_dict[clean_k] = v

    model_state = model.state_dict()
    compatible_state = {
        key: value for key, value in new_state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(compatible_state, strict=False)
    model.eval().float()
    skipped_keys = sorted(set(new_state_dict.keys()) - set(compatible_state.keys()))
    if skipped_keys:
        print(f"⚠️ 推理加载时跳过了 {len(skipped_keys)} 个不兼容权重")
        critical_prefixes = ("token_embedding", "lm_head", "prefix_projector")
        critical_skips = [key for key in skipped_keys if key.startswith(critical_prefixes)]
        if critical_skips:
            print("⚠️ 其中包含关键权重，说明 config 与训练时可能不一致：")
            for key in critical_skips[:10]:
                print(f"   - {key}")
    print("🧠 模型与 PointNet++ 前缀编码器 加载完成")
    return model, config_json


def flatten_points(point_sequence: List[List[int]], vocab_config: Dict) -> List[int]:
    flat_seq = []
    if not point_sequence: return flat_seq
    SEP = vocab_config['SEP_TOKEN']
    END = vocab_config['END_TOKEN']

    flat_seq.append(point_sequence[0][0])
    sep_idx = next((i for i, p in enumerate(point_sequence) if p[0] == SEP), None)
    if sep_idx is None: raise ValueError("SEP_TOKEN not found in sequence.")

    FACE_COORD_POINTS = 16
    i = 1
    while i < sep_idx:
        remaining = sep_idx - i
        needed = FACE_COORD_POINTS + 1
        if remaining < needed: break
        for j in range(FACE_COORD_POINTS):
            flat_seq.extend(point_sequence[i + j])
        flat_seq.append(point_sequence[i + FACE_COORD_POINTS][0])
        i += needed

    flat_seq.append(point_sequence[sep_idx][0])

    EDGE_COORD_POINTS = 4
    i = sep_idx + 1
    n_points = len(point_sequence)
    while i < n_points:
        pt = point_sequence[i]
        if pt[0] == END:
            flat_seq.append(pt[0])
            break
        remaining = n_points - i
        needed = 2 + EDGE_COORD_POINTS
        if remaining < needed: break
        flat_seq.append(point_sequence[i][0])
        flat_seq.append(point_sequence[i + 1][0])
        for j in range(EDGE_COORD_POINTS):
            flat_seq.extend(point_sequence[i + 2 + j])
        i += needed

    return flat_seq


# ================= [条件生成核心逻辑] =================
def load_condition_point_cloud(npy_path, device, num_points=2048):
    """
    点云加载器 (正式激活版)
    """
    if os.path.exists(npy_path):
        try:
            # 真实加载咱们刚才生成的 .npy 数据
            pc_data = load_and_preprocess_point_cloud(npy_path, num_points=num_points, normalize=True)
            # 转为 Tensor，并增加一个 Batch 维度 -> (1, K, N, 3)
            pc_tensor = torch.tensor(pc_data, dtype=torch.float32, device=device).unsqueeze(0)
            return pc_tensor
        except Exception as e:
            print(f"\n⚠️ 加载 {npy_path} 失败: {e}")

    # 兜底用的假数据
    print(f"\n⚠️ 找不到文件，使用占位噪声: {npy_path}")
    dummy_pc = torch.randn(1, num_points, 3, device=device)
    return dummy_pc


def generate_cond_sequence(model, vocab_config, device, point_clouds, max_length=2048, **kwargs):
    """
    调用模型底层的 generate_conditional 进行有条件的自回归推演
    """
    constrained = kwargs.pop("constrained", False)
    if constrained and uses_v2_protocol(vocab_config):
        return generate_cond_sequence_v2_constrained(
            model=model,
            vocab_config=vocab_config,
            device=device,
            point_clouds=point_clouds,
            max_length=max_length,
            **kwargs,
        )

    with torch.no_grad():
        output_ids = model.generate_conditional(
            point_clouds=point_clouds,
            max_new_tokens=max_length,
            eos_token_id=vocab_config['END_TOKEN'],
            special_token_offset=vocab_config['START_TOKEN'],
            **kwargs
        )
    return output_ids[0].cpu().tolist()


def apply_top_k_top_p(next_token_logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k > 0:
        k = min(top_k, next_token_logits.shape[-1])
        indices_to_remove = next_token_logits < torch.topk(next_token_logits, k)[0][..., -1, None]
        next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))
    return next_token_logits


def allowed_v2_tokens(sequence: List[int], vocab_config: Dict, max_faces: Optional[int] = None) -> List[int]:
    face_index_offset = vocab_config.get("face_index_offset", 0)
    face_index_size = vocab_config.get("face_index_size", 50)
    quantization_offset = vocab_config.get("quantization_offset", 50)
    quantization_size = vocab_config.get("quantization_size", 1024)
    bbox_token_offset = vocab_config["bbox_token_offset"]
    bbox_index_size = vocab_config["bbox_index_size"]
    bbox_tokens = vocab_config.get("bbox_tokens_per_element", 6)
    vq_tokens = vocab_config.get("se_tokens_per_element", vocab_config.get("face_vq_tokens", 4))
    face_block = vocab_config.get("face_block", bbox_tokens + vq_tokens + 1)
    edge_block = vocab_config.get("edge_block", 2 + bbox_tokens + vq_tokens)
    start_token = vocab_config["START_TOKEN"]
    sep_token = vocab_config["SEP_TOKEN"]
    end_token = vocab_config["END_TOKEN"]

    if not sequence:
        return [start_token]

    if sep_token not in sequence:
        face_len = len(sequence) - (1 if sequence[0] == start_token else 0)
        pos = face_len % face_block
        num_faces = face_len // face_block
        face_ids = [
            int(sequence[(1 if sequence[0] == start_token else 0) + i * face_block + face_block - 1])
            for i in range(num_faces)
            if (1 if sequence[0] == start_token else 0) + i * face_block + face_block - 1 < len(sequence)
        ]
        unused_face_ids = [
            face_index_offset + i
            for i in range(face_index_size)
            if face_index_offset + i not in set(face_ids)
        ]

        if pos == 0 and num_faces > 0:
            if max_faces is not None and num_faces >= max_faces:
                return [sep_token]
            return list(range(bbox_token_offset, bbox_token_offset + bbox_index_size)) + [sep_token]
        if pos < bbox_tokens:
            return list(range(bbox_token_offset, bbox_token_offset + bbox_index_size))
        if pos < bbox_tokens + vq_tokens:
            return list(range(quantization_offset, quantization_offset + quantization_size))
        return unused_face_ids or [sep_token]

    end_idx = sequence.index(end_token) if end_token in sequence else len(sequence)
    sep_idx = sequence.index(sep_token)
    face_start = 1 if sequence and sequence[0] == start_token else 0
    face_seq = sequence[face_start:sep_idx]
    face_ids = [
        int(face_seq[i + face_block - 1])
        for i in range(0, len(face_seq), face_block)
        if i + face_block - 1 < len(face_seq)
    ]
    valid_face_ids = [
        token for token in face_ids
        if face_index_offset <= token < face_index_offset + face_index_size
    ]
    if len(valid_face_ids) < 2:
        return [end_token]

    edge_len = end_idx - sep_idx - 1
    pos = edge_len % edge_block
    if pos == 0 and edge_len > 0:
        return valid_face_ids + [end_token]
    if pos < 2:
        return valid_face_ids
    if pos < 2 + bbox_tokens:
        return list(range(bbox_token_offset, bbox_token_offset + bbox_index_size))
    return list(range(quantization_offset, quantization_offset + quantization_size))


def generate_cond_sequence_v2_constrained(
    model,
    vocab_config,
    device,
    point_clouds,
    max_length=2048,
    temperature=1.0,
    top_p=0.9,
    top_k=50,
    min_faces=1,
    max_faces=None,
):
    model.eval()
    batch_size = point_clouds.shape[0]
    if batch_size != 1:
        raise ValueError("v2 constrained decoding currently supports batch_size=1")

    start_token = vocab_config["START_TOKEN"]
    end_token = vocab_config["END_TOKEN"]
    generated_ids = torch.tensor([[start_token]], dtype=torch.long, device=device)

    with torch.no_grad():
        prefix_embeds = model.prefix_projector(point_clouds)
        k_prefix = prefix_embeds.shape[1]

        for _ in range(max_length):
            text_embeds = model.token_embedding(generated_ids)
            inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
            total_len = inputs_embeds.shape[1]

            causal_mask = torch.full((total_len, total_len), float("-inf"), device=device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            if k_prefix > 0:
                causal_mask[:k_prefix, :k_prefix] = 0.0
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            position_ids = torch.arange(0, total_len, dtype=torch.long, device=device).unsqueeze(0)

            hidden_states = inputs_embeds
            for layer in model.layers:
                hidden_states, _ = layer(
                    hidden_states, position_ids=position_ids, attention_mask=attention_mask
                )
            hidden_states = model.norm(hidden_states)
            logits = model.lm_head(hidden_states)

            seq_list = generated_ids[0].detach().cpu().tolist()
            allowed = allowed_v2_tokens(seq_list, vocab_config, max_faces=max_faces)

            # Do not allow SEP before the requested minimum face count.
            sep_token = vocab_config["SEP_TOKEN"]
            if sep_token in allowed and min_faces > 1:
                face_block = vocab_config.get("face_block", 11)
                face_len = len(seq_list) - 1
                if face_len // face_block < min_faces:
                    allowed = [token for token in allowed if token != sep_token]

            allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
            next_token_logits = torch.full_like(logits[:, -1, :], -float("inf"))
            next_token_logits[:, allowed_tensor] = logits[:, -1, allowed_tensor]
            next_token_logits = next_token_logits / max(float(temperature), 1e-6)
            next_token_logits = apply_top_k_top_p(next_token_logits, top_k=top_k, top_p=top_p)

            if torch.isinf(next_token_logits).all():
                next_token = torch.tensor([[allowed[0]]], dtype=torch.long, device=device)
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if int(next_token.item()) == end_token:
                break

    return generated_ids[0].detach().cpu().tolist()


def uses_v2_protocol(vocab_config: Dict) -> bool:
    return "bbox_token_offset" in vocab_config and "bbox_index_size" in vocab_config


def dequantize_bbox(indices: np.ndarray, num_tokens: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.float64)
    return (indices / float(num_tokens - 1)) * 2.0 - 1.0


def vq_embedding_weight(model):
    if hasattr(model.quantize, "embedding"):
        return model.quantize.embedding.weight
    if hasattr(model.quantize, "embed"):
        return model.quantize.embed.weight
    raise AttributeError("Cannot find VQ embedding weight on model.quantize")


def decode_v2_tokens_to_controls(vqvae_model, token_rows: List[List[int]], item_type: str, device) -> np.ndarray:
    if not token_rows:
        rows = 16 if item_type == "face" else 4
        return np.zeros((0, rows, 3), dtype=np.float32)

    token_tensor = torch.tensor(token_rows, dtype=torch.long, device=device)
    token_tensor = token_tensor.reshape(token_tensor.shape[0], 2, 2)
    with torch.no_grad():
        quantized = F.embedding(token_tensor, vq_embedding_weight(vqvae_model)).permute(0, 3, 1, 2)
        decoded = vqvae_model.decoder(vqvae_model.post_quant_conv(quantized))
    grid = decoded[:, :3].permute(0, 2, 3, 1).detach().cpu().numpy()

    if item_type == "face":
        return grid.reshape(len(token_rows), 16, 3).astype(np.float32)
    if item_type == "edge":
        return grid.mean(axis=1).reshape(len(token_rows), 4, 3).astype(np.float32)
    raise ValueError(f"Unknown item_type: {item_type}")


def v2_controls_to_sampled_ncs(face_ctrs: np.ndarray, edge_ctrs: np.ndarray):
    surf_ncs = []
    for face_ctr in face_ctrs:
        surface = create_bspline_surface(np.asarray(face_ctr, dtype=np.float64))
        surf_ncs.append(sample_bspline_surface(surface, num_u=32, num_v=32))

    edge_ncs = []
    for edge_ctr in edge_ctrs:
        curve = create_bspline_curve(np.asarray(edge_ctr, dtype=np.float64))
        edge_ncs.append(sample_bspline_curve(curve, num_points=32))

    return np.asarray(surf_ncs, dtype=np.float64), np.asarray(edge_ncs, dtype=np.float64)


def parse_sequence_to_cad_data_v2(sequence: List[int], vocab_config: Dict, vqvae_model, device):
    face_index_offset = vocab_config.get("face_index_offset", 0)
    face_index_size = vocab_config.get("face_index_size", 50)
    quantization_offset = vocab_config.get("quantization_offset", vocab_config.get("se_token_offset", 50))
    quantization_size = vocab_config.get("quantization_size", vocab_config.get("se_codebook_size", 1024))
    bbox_token_offset = vocab_config["bbox_token_offset"]
    bbox_index_size = vocab_config.get("bbox_index_size", vocab_config.get("bbox_size", 2048))
    bbox_tokens = vocab_config.get("bbox_tokens_per_element", 6)
    vq_tokens = vocab_config.get("se_tokens_per_element", vocab_config.get("face_vq_tokens", 4))

    start_token = vocab_config.get("START_TOKEN")
    sep_token = vocab_config.get("SEP_TOKEN")
    end_token = vocab_config.get("END_TOKEN")

    i = 0
    if i < len(sequence) and sequence[i] == start_token:
        i += 1

    face_token_rows, face_bbox_rows, face_ids = [], [], []
    while i < len(sequence) and sequence[i] != sep_token:
        if i + bbox_tokens + vq_tokens >= len(sequence):
            break
        bbox_row = []
        for _ in range(bbox_tokens):
            token = int(sequence[i])
            if not (bbox_token_offset <= token < bbox_token_offset + bbox_index_size):
                raise ValueError(f"Bad face bbox token {token} at position {i}")
            bbox_row.append(token - bbox_token_offset)
            i += 1

        vq_row = []
        for _ in range(vq_tokens):
            token = int(sequence[i])
            if not (quantization_offset <= token < quantization_offset + quantization_size):
                raise ValueError(f"Bad face VQ token {token} at position {i}")
            vq_row.append(token - quantization_offset)
            i += 1

        token = int(sequence[i])
        if not (face_index_offset <= token < face_index_offset + face_index_size):
            raise ValueError(f"Bad face index token {token} at position {i}")
        face_ids.append(token - face_index_offset)
        face_bbox_rows.append(bbox_row)
        face_token_rows.append(vq_row)
        i += 1

    if i < len(sequence) and sequence[i] == sep_token:
        i += 1

    edge_token_rows, edge_bbox_rows, edge_pairs_raw = [], [], []
    while i < len(sequence) and sequence[i] != end_token:
        if i + 1 >= len(sequence):
            break
        src_token = int(sequence[i])
        dst_token = int(sequence[i + 1])
        if not (
            face_index_offset <= src_token < face_index_offset + face_index_size
            and face_index_offset <= dst_token < face_index_offset + face_index_size
        ):
            raise ValueError(f"Bad edge face pair at position {i}: {src_token}, {dst_token}")
        edge_pairs_raw.append((src_token - face_index_offset, dst_token - face_index_offset))
        i += 2

        bbox_row = []
        for _ in range(bbox_tokens):
            token = int(sequence[i])
            if not (bbox_token_offset <= token < bbox_token_offset + bbox_index_size):
                raise ValueError(f"Bad edge bbox token {token} at position {i}")
            bbox_row.append(token - bbox_token_offset)
            i += 1

        vq_row = []
        for _ in range(vq_tokens):
            token = int(sequence[i])
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

    face_ctrs = decode_v2_tokens_to_controls(vqvae_model, face_token_rows, "face", device)
    edge_ctrs = decode_v2_tokens_to_controls(vqvae_model, edge_token_rows, "edge", device)
    surf_ncs, edge_ncs = v2_controls_to_sampled_ncs(face_ctrs, edge_ctrs)

    return {
        "surf_ncs": surf_ncs,
        "edge_ncs": edge_ncs,
        "surf_bbox_wcs": dequantize_bbox(np.asarray(face_bbox_rows, dtype=np.int64), bbox_index_size),
        "edge_bbox_wcs": dequantize_bbox(np.asarray(edge_bbox_rows, dtype=np.int64), bbox_index_size),
        "edgeFace_adj": remapped_pairs,
        "faceEdge_adj": face_edge_adj,
        "face_ctrs": face_ctrs,
        "edge_ctrs": edge_ctrs,
        "face_ids": face_ids,
    }


def infer_vertices_v2(edge_ncs: np.ndarray, edge_bbox_wcs: np.ndarray, face_edge_adj: List[List[int]]):
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
            for a in range(n_vertices):
                if a in merged:
                    continue
                for b in range(a + 1, n_vertices):
                    if b in merged:
                        continue
                    edge_a = face_vertices[a][0] // 2
                    edge_b = face_vertices[b][0] // 2
                    if edge_a == edge_b:
                        continue
                    dist = np.linalg.norm(face_vertices[a][1] - face_vertices[b][1])
                    if dist < min_dist:
                        min_dist = dist
                        min_i, min_j = a, b
            if min_i < 0 or min_j < 0:
                break
            vid1 = face_vertices[min_i][0]
            vid2 = face_vertices[min_j][0]
            union(vid1, vid2)
            face_groups.append([vid1, vid2])
            merged.add(min_i)
            merged.add(min_j)
        face_merged_groups.append(face_groups)

    for a in range(len(face_merged_groups)):
        for b in range(a + 1, len(face_merged_groups)):
            for group1 in face_merged_groups[a]:
                for group2 in face_merged_groups[b]:
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


def reconstruct_cad_from_sequence_v2(sequence: List[int], vocab_config: Dict, vqvae_model, device):
    cad_data = parse_sequence_to_cad_data_v2(sequence, vocab_config, vqvae_model, device)
    if len(cad_data["surf_ncs"]) == 0 or len(cad_data["edge_ncs"]) == 0:
        return None

    unique_vertices, edge_vertex_adj = infer_vertices_v2(
        cad_data["edge_ncs"], cad_data["edge_bbox_wcs"], cad_data["faceEdge_adj"]
    )
    surf_wcs, edge_wcs = joint_optimize(
        cad_data["surf_ncs"],
        cad_data["edge_ncs"],
        cad_data["surf_bbox_wcs"],
        unique_vertices,
        edge_vertex_adj,
        cad_data["faceEdge_adj"],
        len(cad_data["edge_ncs"]),
        len(cad_data["surf_ncs"]),
    )
    return construct_brep(surf_wcs, edge_wcs, cad_data["faceEdge_adj"], edge_vertex_adj)


def summarize_sequence(sequence, vocab_config):
    start_token = vocab_config["START_TOKEN"]
    sep_token = vocab_config["SEP_TOKEN"]
    end_token = vocab_config["END_TOKEN"]
    face_index_offset = vocab_config.get("face_index_offset", 0)
    quantization_offset = vocab_config.get("quantization_offset", 50)
    face_index_size = vocab_config.get("face_index_size", 50)
    quantization_size = vocab_config.get("quantization_size", 1024)
    bbox_token_offset = vocab_config.get("bbox_token_offset")
    bbox_index_size = vocab_config.get("bbox_index_size", 0)
    bbox_tokens = vocab_config.get("bbox_tokens_per_element", 6)
    vq_tokens = vocab_config.get("se_tokens_per_element", vocab_config.get("face_vq_tokens", 4))
    if uses_v2_protocol(vocab_config):
        face_block = int(vocab_config.get("face_block", bbox_tokens + vq_tokens + 1))
        edge_block = int(vocab_config.get("edge_block", 2 + bbox_tokens + vq_tokens))
    else:
        face_block = int(vocab_config.get("face_block", 5))
        edge_block = int(vocab_config.get("edge_block", 6))
    summary = {
        "length": len(sequence),
        "starts_with_start": bool(sequence and sequence[0] == start_token),
        "has_sep": sep_token in sequence,
        "has_end": end_token in sequence,
        "sep_index": sequence.index(sep_token) if sep_token in sequence else None,
        "end_index": sequence.index(end_token) if end_token in sequence else None,
        "face_block": face_block,
        "edge_block": edge_block,
    }
    if summary["has_sep"]:
        face_start = 1 if summary["starts_with_start"] else 0
        face_len = summary["sep_index"] - face_start
        edge_end = summary["end_index"] if summary["has_end"] else len(sequence)
        edge_len = edge_end - summary["sep_index"] - 1
        summary.update({
            "face_token_len": face_len,
            "edge_token_len": edge_len,
            "face_block_aligned": face_len % face_block == 0,
            "edge_block_aligned": edge_len % edge_block == 0,
            "num_faces_protocol": face_len // face_block if face_len >= 0 else 0,
            "num_edges_protocol": edge_len // edge_block if edge_len >= 0 else 0,
        })
        if face_len >= 0 and face_len % face_block == 0:
            face_seq = sequence[face_start:summary["sep_index"]]
            face_blocks = [
                face_seq[i:i + face_block]
                for i in range(0, len(face_seq), face_block)
            ]
            face_ids = [block[-1] - face_index_offset for block in face_blocks]
            summary["face_ids"] = [int(x) for x in face_ids]
            summary["unique_face_ids"] = sorted({int(x) for x in face_ids})
            summary["duplicate_face_ids"] = len(face_ids) - len(set(face_ids))
            if bbox_token_offset is not None:
                summary["invalid_face_bbox_blocks"] = int(sum(
                    any(
                        token < bbox_token_offset or token >= bbox_token_offset + bbox_index_size
                        for token in block[:bbox_tokens]
                    )
                    for block in face_blocks
                ))
                face_vq_slices = [block[bbox_tokens:bbox_tokens + vq_tokens] for block in face_blocks]
            else:
                face_vq_slices = [block[:-1] for block in face_blocks]
            summary["invalid_face_vq_blocks"] = int(sum(
                any(
                    token < quantization_offset or token >= quantization_offset + quantization_size
                    for token in vq_slice
                )
                for vq_slice in face_vq_slices
            ))
        if edge_len >= 0 and edge_len % edge_block == 0:
            edge_seq = sequence[summary["sep_index"] + 1:edge_end]
            edge_blocks = [
                edge_seq[i:i + edge_block]
                for i in range(0, len(edge_seq), edge_block)
            ]
            edge_face_pairs = [
                (block[0] - face_index_offset, block[1] - face_index_offset)
                for block in edge_blocks
            ]
            valid_face_ids = set(summary.get("unique_face_ids", []))
            invalid_edge_refs = [
                [int(a), int(b)]
                for a, b in edge_face_pairs
                if a not in valid_face_ids or b not in valid_face_ids
            ]
            duplicate_edge_pairs = len(edge_face_pairs) - len({
                tuple(sorted((int(a), int(b)))) for a, b in edge_face_pairs
            })
            face_degrees = {str(int(face_id)): 0 for face_id in valid_face_ids}
            for a, b in edge_face_pairs:
                if a in valid_face_ids and b in valid_face_ids:
                    face_degrees[str(int(a))] = face_degrees.get(str(int(a)), 0) + 1
                    face_degrees[str(int(b))] = face_degrees.get(str(int(b)), 0) + 1
            summary["edge_face_pairs_protocol"] = [[int(a), int(b)] for a, b in edge_face_pairs]
            summary["invalid_edge_refs"] = invalid_edge_refs
            summary["num_invalid_edge_refs"] = len(invalid_edge_refs)
            summary["duplicate_undirected_edge_pairs"] = int(duplicate_edge_pairs)
            summary["valid_face_degrees_from_protocol_edges"] = face_degrees
            summary["invalid_edge_face_index_tokens"] = int(sum(
                a < 0 or a >= face_index_size or b < 0 or b >= face_index_size
                for a, b in edge_face_pairs
            ))
            if bbox_token_offset is not None:
                summary["invalid_edge_bbox_blocks"] = int(sum(
                    any(
                        token < bbox_token_offset or token >= bbox_token_offset + bbox_index_size
                        for token in block[2:2 + bbox_tokens]
                    )
                    for block in edge_blocks
                ))
                edge_vq_slices = [block[2 + bbox_tokens:2 + bbox_tokens + vq_tokens] for block in edge_blocks]
            else:
                edge_vq_slices = [block[2:] for block in edge_blocks]
            summary["invalid_edge_vq_blocks"] = int(sum(
                any(
                    token < quantization_offset or token >= quantization_offset + quantization_size
                    for token in vq_slice
                )
                for vq_slice in edge_vq_slices
            ))
    return summary


def save_generation_debug(output_dir, file_prefix, raw_seq, vocab_config, device, error, vqvae_model=None):
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    summary = summarize_sequence(raw_seq, vocab_config)
    summary["error"] = str(error) if error else None

    cad_data = None
    try:
        if uses_v2_protocol(vocab_config):
            if vqvae_model is None:
                raise ValueError("v2 debug parse requires vqvae_model")
            cad_data = parse_sequence_to_cad_data_v2(raw_seq, vocab_config, vqvae_model, device)
            summary["parsed_faces"] = len(cad_data.get("face_ctrs", []))
            summary["parsed_edges"] = len(cad_data.get("edge_ctrs", []))
            summary["parsed_edge_face_pairs"] = len(cad_data.get("edgeFace_adj", []))
        else:
            cad_data = parse_sequence_to_cad_data_nurbs(
                raw_seq, vocab_config, device=device, verbose=False
            )
            summary["parsed_faces"] = len(cad_data.get("face_ctrs", []))
            summary["parsed_edges"] = len(cad_data.get("edge_ctrs", []))
            summary["parsed_edge_face_pairs"] = len(cad_data.get("edgeFace_adj", []))
    except Exception as parse_error:
        summary["parse_error"] = str(parse_error)

    json_path = os.path.join(debug_dir, f"{file_prefix}_debug.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "sequence": [int(x) for x in raw_seq]},
            f,
            indent=2,
            ensure_ascii=False,
        )

    if cad_data is not None:
        npz_path = os.path.join(debug_dir, f"{file_prefix}_parsed.npz")
        np.savez_compressed(
            npz_path,
            face_ctrs=np.asarray(cad_data.get("face_ctrs", []), dtype=np.float32),
            edge_ctrs=np.asarray(cad_data.get("edge_ctrs", []), dtype=np.float32),
            edgeFace_adj=np.asarray(cad_data.get("edgeFace_adj", []), dtype=np.int32),
        )


def process_single_cond_sample(model, vocab_config, device, output_dir, file_prefix, point_clouds, args, vqvae_model=None):
    res = {'sequence': None, 'solid': None, 'is_valid': False, 'saved': False, 'error': None}

    # 1. 看着点云，生成序列
    try:
        raw_seq = generate_cond_sequence(
            model, vocab_config, device,
            point_clouds=point_clouds,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            constrained=args.constrained_decoding,
            min_faces=args.min_faces,
            max_faces=args.max_faces,
        )
    except Exception as e:
        res['error'] = f"条件序列生成异常: {e}"
        return res

    if not raw_seq:
        res['error'] = "生成序列为空"
        return res
    res['sequence'] = raw_seq

    # 2. 直接重建 (LLaMA 输出的已经是标准的 1D 序列，无需展平！)
    try:
        if uses_v2_protocol(vocab_config):
            if vqvae_model is None:
                raise ValueError("v2 generation requires --vqvae_ckpt")
            solid = reconstruct_cad_from_sequence_v2(
                sequence=raw_seq,
                vocab_config=vocab_config,
                vqvae_model=vqvae_model,
                device=device,
            )
        else:
            solid = reconstruct_cad_from_sequence_nurbs(
                sequence=raw_seq, vocab_info=vocab_config, device=device, verbose=False
            )
    except Exception as e:
        res['error'] = f"重建过程异常: {e}"
        save_generation_debug(output_dir, file_prefix, raw_seq, vocab_config, device, res['error'], vqvae_model)
        return res

    if solid is None:
        res['error'] = "BREP 重建返回 None"
        save_generation_debug(output_dir, file_prefix, raw_seq, vocab_config, device, res['error'], vqvae_model)
        return res
    res['solid'] = solid

    # 3. 验证有效性
    try:
        res['is_valid'] = check_brep_validity(solid)
    except Exception as e:
        res['is_valid'] = False

    # 4. 保存为 STEP/STL
    step_path = os.path.join(output_dir, f"{file_prefix}_generated.step")
    stl_path = os.path.join(output_dir, f"{file_prefix}_generated.stl")
    try:
        status, err = write_files_safe(solid, step_path, stl_path, args.timeout)
    except Exception as e:
        res['error'] = f"文件保存异常: {e}"
        return res

    if status == 'success':
        res['saved'] = True
    else:
        res['error'] = f"文件保存失败 ({status}): {err}"

    return res


def main():
    parser = argparse.ArgumentParser(description="Conditional Point-to-CAD Generation")
    parser.add_argument("--ar_model", type=str, default="checkpoints/your_model_path.pt", help="你的自回归模型权重路径")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--output_dir", type=str, default="result/generated_cad_cond/")

    # 关键参数：测试点云存放的目录
    parser.add_argument("--test_pc_dir", type=str, default="data/test_pointclouds/", help="存放测试 .npy 点云的目录")

    parser.add_argument("--max_length", type=int, default=1597)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--point_cloud_npoints", type=int, default=2048)
    parser.add_argument("--constrained_decoding", action="store_true", help="Use v2 grammar-constrained decoding.")
    parser.add_argument("--min_faces", type=int, default=1, help="Minimum face blocks before SEP is allowed in constrained decoding.")
    parser.add_argument("--max_faces", type=int, default=None, help="Maximum face blocks before SEP is forced in constrained decoding.")
    parser.add_argument(
        "--vqvae_ckpt",
        type=str,
        default="/mnt/docker_dir/lijiahao/NurbsVQVAE_code/checkpoint/se/abc/8192,4096,128,64,false,1e-4,0,p/deepcad_nurbs_vqvae_best.pt",
        help="NURBS VQ-VAE checkpoint used to decode v2 VQ tokens.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model, vocab_config = load_resources(args.ar_model, device, args.config)
    vqvae_model = None
    if uses_v2_protocol(vocab_config):
        vqvae_model = load_vqvae_model(args.vqvae_ckpt, device)
        vqvae_model.eval()
        print("VQ-VAE decoder loaded for v2 sequence reconstruction")

    # 自动扫描目录下所有的测试点云
    test_files = [
        path for path in glob.glob(os.path.join(args.test_pc_dir, "*.npy"))
        if os.path.basename(path) != "face_labels.npy"
    ]
    test_files.extend(glob.glob(os.path.join(args.test_pc_dir, "*", "point_cloud.npy")))
    test_files = sorted(set(test_files))
    if len(test_files) == 0:
        print(f"⚠️ 警告：在 {args.test_pc_dir} 下没有找到 .npy 文件！将使用 2 个占位数据测试流水线...")
        test_files = ["dummy_pc_001.npy", "dummy_pc_002.npy"]

    stats = {'total': len(test_files), 'saved_success': 0, 'brep_valid': 0, 'brep_invalid': 0, 'save_failed': 0}
    pbar = tqdm(total=len(test_files), desc="Cond-Generating", unit="obj")

    for pc_file in test_files:
        file_prefix = (
            os.path.basename(os.path.dirname(pc_file))
            if os.path.basename(pc_file) == "point_cloud.npy"
            else os.path.splitext(os.path.basename(pc_file))[0]
        )

        # 加载条件
        point_clouds = load_condition_point_cloud(pc_file, device, num_points=args.point_cloud_npoints)

        # 进行条件生成
        res = process_single_cond_sample(
            model, vocab_config, device,
            args.output_dir,
            file_prefix,
            point_clouds,
            args,
            vqvae_model
        )

        if res['saved']:
            stats['saved_success'] += 1
            if res['is_valid']:
                stats['brep_valid'] += 1
            else:
                stats['brep_invalid'] += 1
        else:
            stats['save_failed'] += 1
            # 【增加这一行】：一旦失败，立刻在屏幕上打印出具体的死亡原因！
            print(f"\n❌ [{file_prefix}] 失败原因: {res['error']}")

        pbar.update(1)
        pbar.set_postfix({'Valid': stats['brep_valid'], 'Fail': stats['save_failed']})

    pbar.close()

    # 打印最终统计
    summary_text = (
        f"\n{'=' * 60}\n"
        f"条件生成统计 (模型: {os.path.basename(args.ar_model)}):\n"
        f"  总测试点云:     {stats['total']}\n"
        f"  成功保存文件:   {stats['saved_success']}\n"
        f"    ├─ BREP 有效:  {stats['brep_valid']}\n"
        f"    └─ BREP 无效:  {stats['brep_invalid']}\n"
        f"  保存失败:       {stats['save_failed']}\n"
        f"{'=' * 60}"
    )
    print(summary_text)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
