#!/usr/bin/env python3
import os
import json
import time
import torch
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
from utils import check_brep_validity, reconstruct_cad_from_sequence_nurbs


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
        num_components=config_json['num_components']
    )

    model = LLaMA3ARModel(llama_config).to(device)
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('module.', '').replace('model.', '')
        new_state_dict[clean_k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval().float()
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
def load_condition_point_cloud(npy_path, device):
    """
    点云加载器 (正式激活版)
    """
    if os.path.exists(npy_path):
        try:
            # 真实加载咱们刚才生成的 .npy 数据
            pc_data = np.load(npy_path, allow_pickle=True)
            # 转为 Tensor，并增加一个 Batch 维度 -> (1, K, N, 3)
            pc_tensor = torch.tensor(pc_data, dtype=torch.float32, device=device).unsqueeze(0)
            return pc_tensor
        except Exception as e:
            print(f"\n⚠️ 加载 {npy_path} 失败: {e}")

    # 兜底用的假数据
    print(f"\n⚠️ 找不到文件，使用占位噪声: {npy_path}")
    dummy_pc = torch.randn(1, 10, 512, 3, device=device)
    return dummy_pc


def generate_cond_sequence(model, vocab_config, device, point_clouds, max_length=2048, **kwargs):
    """
    调用模型底层的 generate_conditional 进行有条件的自回归推演
    """
    with torch.no_grad():
        output_ids = model.generate_conditional(
            point_clouds=point_clouds,
            max_new_tokens=max_length,
            eos_token_id=vocab_config['END_TOKEN'],
            special_token_offset=vocab_config['START_TOKEN'],
            **kwargs
        )
    return output_ids[0].cpu().tolist()


def process_single_cond_sample(model, vocab_config, device, output_dir, file_prefix, point_clouds, args):
    res = {'sequence': None, 'solid': None, 'is_valid': False, 'saved': False, 'error': None}

    # 1. 看着点云，生成序列
    try:
        raw_seq = generate_cond_sequence(
            model, vocab_config, device,
            point_clouds=point_clouds,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
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
        solid = reconstruct_cad_from_sequence_nurbs(
            sequence=raw_seq, vocab_info=vocab_config, device=device, verbose=False
        )
    except Exception as e:
        res['error'] = f"重建过程异常: {e}"
        return res

    if solid is None:
        res['error'] = "BREP 重建返回 None"
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
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model, vocab_config = load_resources(args.ar_model, device, args.config)

    # 自动扫描目录下所有的测试点云
    test_files = glob.glob(os.path.join(args.test_pc_dir, "*.npy"))
    if len(test_files) == 0:
        print(f"⚠️ 警告：在 {args.test_pc_dir} 下没有找到 .npy 文件！将使用 2 个占位数据测试流水线...")
        test_files = ["dummy_pc_001.npy", "dummy_pc_002.npy"]

    stats = {'total': len(test_files), 'saved_success': 0, 'brep_valid': 0, 'brep_invalid': 0, 'save_failed': 0}
    pbar = tqdm(total=len(test_files), desc="Cond-Generating", unit="obj")

    for pc_file in test_files:
        file_prefix = os.path.splitext(os.path.basename(pc_file))[0]

        # 加载条件
        point_clouds = load_condition_point_cloud(pc_file, device)

        # 进行条件生成
        res = process_single_cond_sample(
            model, vocab_config, device,
            args.output_dir,
            file_prefix,
            point_clouds,
            args
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