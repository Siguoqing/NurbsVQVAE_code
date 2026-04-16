#!/usr/bin/env python3
import os
import json
import pickle
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
from tqdm import tqdm
from typing import Optional, List, Dict, Any
from model import LLaMA3Config, LLaMA3ARModel
from utils import check_brep_validity, reconstruct_cad_from_sequence_nurbs

def _global_write_worker(temp_step_path, final_step_path, final_stl_path, result_file_path):
    """
    子进程工作函数：读取临时 STEP -> 写 STL -> 移动 STEP 到目标位置
    """
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

        # 1. 读取 STEP
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

        # 2. 写入 STL
        try:
            write_stl_file(shape, final_stl_path, linear_deflection=0.001, angular_deflection=0.5)
        except Exception as e:
            write_result('stl_failed', f'STL 写入失败: {e}')
            return

        # 3. 复制 STEP
        shutil.copy2(temp_step_path, final_step_path)
        write_result('success', None)

    except Exception as e:
        write_result('error', f"子进程发生未捕获异常: {e}\n{traceback.format_exc()}")

def timeout_handler(signum, frame):
    raise TimeoutError("操作超时")

def write_files_safe(solid, step_path, stl_path, write_timeout=30):
    """
    安全的文件写入函数：主进程写临时 STEP，子进程转 STL
    """
    fd, temp_step_path = tempfile.mkstemp(suffix=".step", prefix="cad_temp_")
    os.close(fd)
    fd_res, result_file_path = tempfile.mkstemp(suffix=".json", prefix="cad_res_")
    os.close(fd_res)
    
    is_windows = platform.system() == 'Windows'

    try:
        # 阶段 1: 主进程导出 STEP
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

        # 阶段 2: 子进程转 STL
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

def load_resources(model_path: str, device: torch.device, config_path: str = "config.json"):
    # 1. 加载 Checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 2. 从 config.json 读取 vocab 配置和模型超参
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

    # 4. 初始化并加载权重 (LLaMA3ARModel 需导入)
    model = LLaMA3ARModel(llama_config).to(device)
    
    # 清理 key 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('module.', '').replace('model.', '')
        new_state_dict[clean_k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval().float()
    print("模型加载完成")

    return model, config_json

def flatten_points(point_sequence: List[List[int]], vocab_config: Dict) -> List[int]:
    """
    将 Point-wise 序列 (List[x,y,z]) 展平为一维 token 流，处理 Primary-Null 结构
    """
    flat_seq = []
    if not point_sequence:
        return flat_seq
    
    SEP = vocab_config['SEP_TOKEN']
    END = vocab_config['END_TOKEN']

    # 0. START
    flat_seq.append(point_sequence[0][0]) # START_TOKEN

    # 1. Find SEP
    sep_idx = next((i for i, p in enumerate(point_sequence) if p[0] == SEP), None)
    if sep_idx is None:
        raise ValueError("SEP_TOKEN not found in sequence.")

    # 2. Faces (16 coords + 1 index = 17 points)
    FACE_COORD_POINTS = 16
    i = 1
    while i < sep_idx:
        remaining = sep_idx - i
        needed = FACE_COORD_POINTS + 1
        if remaining < needed: break # 不完整丢弃

        # 坐标点 (x,y,z) 直接展开
        for j in range(FACE_COORD_POINTS):
            flat_seq.extend(point_sequence[i + j])
        
        # 面索引点只取 x
        flat_seq.append(point_sequence[i + FACE_COORD_POINTS][0])
        i += needed

    # 3. SEP
    flat_seq.append(point_sequence[sep_idx][0])

    # 4. Edges
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

        # 2 个索引点只取 x
        flat_seq.append(point_sequence[i][0])
        flat_seq.append(point_sequence[i + 1][0])

        # 4 个坐标点展开
        for j in range(EDGE_COORD_POINTS):
            flat_seq.extend(point_sequence[i + 2 + j])
        
        i += needed

    return flat_seq

def generate_raw_sequence(model, vocab_config, device, max_length=2048, **kwargs):
    """
    执行自回归生成
    """
    # 构造 Prompt [START, PAD, PAD]
    start_point = [vocab_config['START_TOKEN'], vocab_config['PAD_TOKEN'], vocab_config['PAD_TOKEN']]
    prompt = torch.tensor([start_point], dtype=torch.long, device=device).unsqueeze(0) # (1, 1, 3)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prompt,
            max_length=max_length,
            pad_token_id=vocab_config['PAD_TOKEN'],
            eos_token_id=vocab_config['END_TOKEN'],
            special_token_offset=vocab_config['START_TOKEN'],
            quantization_offset=vocab_config['quantization_offset'],
            **kwargs
        )
    
    return output_ids[0].cpu().tolist()

def process_single_sample(model, vocab_config, device, output_dir, file_prefix, args):
    """
    生成单个样本的完整流程：生成 -> 展平 -> 重建 -> 保存
    """
    res = {'sequence': None, 'solid': None, 'is_valid': False, 'saved': False, 'error': None}
    
    # 1. 生成
    try:
        raw_seq = generate_raw_sequence(
            model, vocab_config, device, 
            max_length=args.max_length, 
            temperature=args.temperature, 
            top_p=args.top_p, 
            top_k=args.top_k
        )
    except Exception as e:
        res['error'] = f"序列生成异常: {e}"
        traceback.print_exc()
        return res

    if not raw_seq:
        res['error'] = "生成序列为空"
        return res
    res['sequence'] = raw_seq

    # 2. 展平与重建
    try:
        flat_seq = flatten_points(raw_seq, vocab_config)
    except Exception as e:
        res['error'] = f"序列展平失败: {e}"
        traceback.print_exc()
        return res
    
    try:
        solid = reconstruct_cad_from_sequence_nurbs(
            sequence=flat_seq, vocab_info=vocab_config, device=device, verbose=False
        )
    except Exception as e:
        res['error'] = f"BREP 重建异常: {e}"
        traceback.print_exc()
        return res

    if solid is None:
        res['error'] = "BREP 重建返回 None"
        return res
    res['solid'] = solid

    # 3. 验证
    try:
        res['is_valid'] = check_brep_validity(solid)
    except Exception as e:
        res['is_valid'] = False
        res['error'] = f"BREP 验证异常: {e}"
        traceback.print_exc()
        return res

    # 4. 保存
    timestamp = f"{int(time.time())}_{random.randint(1000,9999)}"
    step_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.step")
    stl_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.stl")
    try:
        status, err = write_files_safe(solid, step_path, stl_path, args.timeout)
    except Exception as e:
        res['error'] = f"文件保存异常: {e}"
        traceback.print_exc()
        return res
    
    if status == 'success':
        res['saved'] = True
    else:
        res['error'] = f"文件保存失败 ({status}): {err}"
    
    return res

def main():
    parser = argparse.ArgumentParser(description="Simplified Point-wise CAD Generation")
    parser.add_argument("--ar_model", type=str, default="checkpoints/llama3_cw_pad_cascade_fusion_v2/64_384_5e-4_1.0/epoch_70.pt")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--output_dir", type=str, default="result/generated_brep/llama3_cw_pad_cascade_fusion_v2/64_384_5e-4_1.0/1_0.9_0")
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--max_length", type=int, default=1597)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    # 初始化设置
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model, vocab_config = load_resources(args.ar_model, device, args.config)

    # 批量生成循环
    stats = {
        'total_attempts': 0,     # 总尝试次数
        'saved_success': 0,      # 成功保存文件的数量
        'brep_valid': 0,         # BREP 有效的数量
        'brep_invalid': 0,       # BREP 无效的数量（保存成功但验证失败）
        'save_failed': 0         # 保存失败的数量
    }
    pbar = tqdm(total=args.num_samples, desc="Generating", unit="obj")
    
    while stats['saved_success'] < args.num_samples:
        stats['total_attempts'] += 1
        
        # 防止无限循环
        if stats['total_attempts'] > args.num_samples * 20 and stats['saved_success'] == 0:
            print("错误: 连续失败次数过多，停止运行。")
            break

        res = process_single_sample(
            model, vocab_config, device, 
            args.output_dir, 
            f"gen_{stats['saved_success']:04d}", 
            args
        )

        if res['saved']:
            stats['saved_success'] += 1
            if res['is_valid']:
                stats['brep_valid'] += 1
            else:
                stats['brep_invalid'] += 1
            pbar.update(1)
        else:
            stats['save_failed'] += 1
        
        pbar.set_postfix({
            'Attempts': stats['total_attempts'],
            'Valid': stats['brep_valid'],
            'Invalid': stats['brep_invalid'],
            'SaveFail': stats['save_failed'],
            'LastErr': (res['error'][:30] + '...') if res['error'] and len(res['error']) > 30 else (res['error'] or '')
        })

    pbar.close()
    
    # 最终统计输出
    # ================= [修改开始] =================
    # 1. 构建统计信息的字符串 (这样既可以打印，也可以写入文件)
    summary_lines = []
    summary_lines.append(f"\n{'=' * 60}")
    summary_lines.append(f"生成统计 (模型: {os.path.basename(args.ar_model)}):")
    summary_lines.append(f"  总尝试次数:     {stats['total_attempts']}")

    # 成功的详情
    valid_pct = (stats['brep_valid'] / stats['saved_success'] * 100) if stats['saved_success'] > 0 else 0
    invalid_pct = (stats['brep_invalid'] / stats['saved_success'] * 100) if stats['saved_success'] > 0 else 0

    summary_lines.append(f"  成功保存文件:   {stats['saved_success']}")
    if stats['saved_success'] > 0:
        summary_lines.append(f"    ├─ BREP 有效:  {stats['brep_valid']} ({valid_pct:.1f}%)")
        summary_lines.append(f"    └─ BREP 无效:  {stats['brep_invalid']} ({invalid_pct:.1f}%)")
    else:
        summary_lines.append(f"    ├─ BREP 有效:  0 (0.0%)")
        summary_lines.append(f"    └─ BREP 无效:  0 (0.0%)")

    summary_lines.append(f"  保存失败:       {stats['save_failed']}")

    if stats['total_attempts'] > 0:
        success_rate = stats['saved_success'] / stats['total_attempts'] * 100
        valid_rate = stats['brep_valid'] / stats['total_attempts'] * 100
        summary_lines.append(f"  成功率 (Saved): {success_rate:.1f}%")
        summary_lines.append(f"  有效率 (Valid): {valid_rate:.1f}%")  # <--- 您最关心的指标
    summary_lines.append(f"{'=' * 60}")

    summary_text = "\n".join(summary_lines)

    # 2. 打印到屏幕 (满足您“跑的时候看详细日志”的需求)
    print(summary_text)

    # 3. 自动保存到文件 (满足您“只存最后结果”的需求)
    # 文件会保存在 output_dir 下，名为 eval_summary.txt
    summary_path = os.path.join(args.output_dir, "eval_summary.txt")
    try:
        with open(summary_path, "w") as f:
            f.write(summary_text)
        print(f"\n[提示] 最终统计结果已独立保存至: {summary_path}")
    except Exception as e:
        print(f"\n[警告] 无法保存统计结果文件: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()