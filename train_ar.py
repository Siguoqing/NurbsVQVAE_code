#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.distributed as dist
import traceback
from utils import get_ar_args
from trainer import ARTrainer

# 【新增】同时导入两种 Dataset，实现双轨制兼容
from dataset import NurbsARData, ConditionalCADDataset


def setup_distributed():
    """
    检查分布式环境变量。如果存在，则初始化进程组。
    返回 (local_rank, world_size, rank)，如果非分布式环境则返回 (None, None, None)。
    """
    if 'RANK' not in os.environ:
        return None, None, None  # 判断为非 DDP 模式

    # 从环境变量中获取分布式训练参数
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    # 为当前进程设置对应的 GPU 设备
    torch.cuda.set_device(local_rank)

    # 初始化进程组，nccl 是 NVIDIA GPU 推荐的后端
    dist.init_process_group(backend='nccl', init_method='env://')

    return local_rank, world_size, rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def single_gpu_train(args):
    """
    单GPU训练模式 - LLaMA3架构
    默认使用 cuda:0，或 CUDA_VISIBLE_DEVICES 指定的第一个设备
    """
    print("\n" + "=" * 70)
    print("LLaMA3 CAD自回归模型 - 单GPU训练模式")
    print("=" * 70)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"\n设备信息:")
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  显存: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("\n[警告] 未检测到CUDA设备，将使用CPU训练（速度较慢）")

    # 【核心修改：动态判断加载哪种数据集】
    print(f"\n加载数据集...")
    if getattr(args, 'point_cloud_dir', None):
        print(f"✨ 检测到点云目录参数，启动【条件生成 (Point2CAD)】数据集加载器！")
        train_dataset = ConditionalCADDataset(
            sequence_file=args.sequence_file,
            point_cloud_dir=args.point_cloud_dir,
            data_list_file=getattr(args, 'data_list_file', None),
            validate=False,
            args=args
        )
        val_dataset = ConditionalCADDataset(
            sequence_file=args.sequence_file,
            point_cloud_dir=args.point_cloud_dir,
            data_list_file=getattr(args, 'data_list_file', None),
            validate=True,
            args=args
        )
    else:
        print(f"💡 未提供点云目录参数，启动【无条件生成 (标准自回归)】数据集加载器！")
        train_dataset = NurbsARData(sequence_file=args.sequence_file, validate=False, args=args)
        val_dataset = NurbsARData(sequence_file=args.sequence_file, validate=True, args=args)

    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_dataset):,} 样本")
    print(f"  验证集: {len(val_dataset):,} 样本")
    print(f"  词汇表大小: {train_dataset.vocab_size:,}")

    print(f"\n词汇表结构:")
    print(f"  面索引 ({train_dataset.face_index_size}个): "
          f"{train_dataset.face_index_offset} ~ {train_dataset.face_index_offset + train_dataset.face_index_size - 1}")
    print(f"  量化tokens ({train_dataset.quantization_size}个): "
          f"{train_dataset.quantization_offset} ~ {train_dataset.quantization_offset + train_dataset.quantization_size - 1}")
    if getattr(train_dataset, "bbox_token_offset", None) is not None and getattr(train_dataset, "bbox_index_size", 0):
        print(f"  BBox tokens ({train_dataset.bbox_index_size}个): "
              f"{train_dataset.bbox_token_offset} ~ {train_dataset.bbox_token_offset + train_dataset.bbox_index_size - 1}")
    print(f"  特殊Tokens ({train_dataset.special_token_size}个): "
          f"START={train_dataset.START_TOKEN}, SEP={train_dataset.SEP_TOKEN}, "
          f"END={train_dataset.END_TOKEN}, PAD={train_dataset.PAD_TOKEN}")
    print(f"  Block大小: Face={train_dataset.face_block}, Edge={train_dataset.edge_block}")

    # GRPO 配置（可选）
    grpo_config = None
    if hasattr(args, 'grpo_enabled') and args.grpo_enabled:
        grpo_config = {
            'enabled': True,
            'grpo_ratio': getattr(args, 'grpo_ratio', 0.5),  # 用于 GRPO 的样本占比
            'group_size': getattr(args, 'grpo_group_size', 4),
            'reward_scale': getattr(args, 'reward_scale', 1.0),
            'kl_penalty': getattr(args, 'kl_penalty', 0.0),  # 默认0，使用SFT Loss约束模型
            'sft_weight': getattr(args, 'sft_weight', 1.0),  # SFT Loss 权重（λ）
            'use_brep_reward': True
        }

    # 创建训练器
    trainer = ARTrainer(train_dataset, val_dataset, args, device=device, multi_gpu=False, grpo_config=grpo_config)

    resumed_from = getattr(trainer, "start_epoch", 1) - 1
    if resumed_from >= 1:
        print(f"\n检测到断点: 已完成 {resumed_from} 个 epoch，将从 epoch {resumed_from + 1} 继续。")

    # 开始训练
    trainer.train()


def multi_gpu_train(args, local_rank, world_size, rank):
    """
    DDP多GPU训练模式 - LLaMA3架构
    此函数由torchrun启动的每个进程分别执行
    """
    device = torch.device(f'cuda:{local_rank}')

    # 只在主进程(rank=0)打印信息
    if rank == 0:
        print("\n" + "=" * 70)
        print("LLaMA3 CAD自回归模型 - DDP多GPU训练模式")
        print("=" * 70)
        print(f"\n集群配置:")
        print(f"  进程总数: {world_size}")
        print(f"  每进程GPU: 1")
        print(f"  总GPU数: {world_size}")

        # 显示每个GPU信息
        for i in range(world_size):
            if torch.cuda.is_available() and i < torch.cuda.device_count():
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 【核心修改：动态判断加载哪种数据集】
    if rank == 0:
        print(f"\n加载数据集...")

    if getattr(args, 'point_cloud_dir', None):
        if rank == 0:
            print(f"✨ 检测到点云目录参数，启动【条件生成 (Point2CAD)】数据集加载器！")
        train_dataset = ConditionalCADDataset(
            sequence_file=args.sequence_file,
            point_cloud_dir=args.point_cloud_dir,
            data_list_file=getattr(args, 'data_list_file', None),
            validate=False,
            args=args
        )
        val_dataset = ConditionalCADDataset(
            sequence_file=args.sequence_file,
            point_cloud_dir=args.point_cloud_dir,
            data_list_file=getattr(args, 'data_list_file', None),
            validate=True,
            args=args
        )
    else:
        if rank == 0:
            print(f"💡 未提供点云目录参数，启动【无条件生成 (标准自回归)】数据集加载器！")
        train_dataset = NurbsARData(sequence_file=args.sequence_file, validate=False, args=args)
        val_dataset = NurbsARData(sequence_file=args.sequence_file, validate=True, args=args)

    if rank == 0:
        print(f"\n数据集统计:")
        print(f"  训练集: {len(train_dataset):,} 样本")
        print(f"  验证集: {len(val_dataset):,} 样本")
        print(f"  词汇表大小: {train_dataset.vocab_size:,}")
        print(f"  每GPU batch size: {args.batch_size}")
        print(f"  全局batch size: {args.batch_size * world_size}")

        print(f"\n词汇表结构:")
        print(f"  面索引 ({train_dataset.face_index_size}个): "
              f"{train_dataset.face_index_offset} ~ {train_dataset.face_index_offset + train_dataset.face_index_size - 1}")
        print(f"  量化tokens ({train_dataset.quantization_size}个): "
              f"{train_dataset.quantization_offset} ~ {train_dataset.quantization_offset + train_dataset.quantization_size - 1}")
        if getattr(train_dataset, "bbox_token_offset", None) is not None and getattr(train_dataset, "bbox_index_size", 0):
            print(f"  BBox tokens ({train_dataset.bbox_index_size}个): "
                  f"{train_dataset.bbox_token_offset} ~ {train_dataset.bbox_token_offset + train_dataset.bbox_index_size - 1}")
        print(f"  特殊Tokens ({train_dataset.special_token_size}个): "
              f"START={train_dataset.START_TOKEN}, SEP={train_dataset.SEP_TOKEN}, "
              f"END={train_dataset.END_TOKEN}, PAD={train_dataset.PAD_TOKEN}")
        print(f"  Block大小: Face={train_dataset.face_block}, Edge={train_dataset.edge_block}")

    # GRPO 配置（可选）
    grpo_config = None
    if hasattr(args, 'grpo_enabled') and args.grpo_enabled:
        grpo_config = {
            'enabled': True,
            'grpo_ratio': getattr(args, 'grpo_ratio', 0.5),  # 用于 GRPO 的样本占比
            'group_size': getattr(args, 'grpo_group_size', 4),
            'reward_scale': getattr(args, 'reward_scale', 1.0),
            'kl_penalty': getattr(args, 'kl_penalty', 0.0),  # 默认0，使用SFT Loss约束模型
            'sft_weight': getattr(args, 'sft_weight', 1.0),  # SFT Loss 权重（λ）
            'use_brep_reward': True
        }

    # 创建训练器（内部会处理DDP封装）
    trainer = ARTrainer(train_dataset, val_dataset, args, device=device, multi_gpu=True, grpo_config=grpo_config)

    if rank == 0:
        resumed_from = getattr(trainer, "start_epoch", 1) - 1
        if resumed_from >= 1:
            print(f"\n检测到断点: 已完成 {resumed_from} 个 epoch，将从 epoch {resumed_from + 1} 继续。")

    # 开始训练
    trainer.train()


def main():
    try:
        # 解析命令行参数
        args = get_ar_args()

        # 【核心新增：动态参数劫持】
        # 为了不破坏你原有的 utils.py，我们在这里动态捕获条件生成独有的参数
        parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument('--point_cloud_dir', type=str, default=None,
                            help='存放 .npy 点云的目录 (如果设置了，即开启条件生成)')
        parser.add_argument('--data_list_file', type=str, default=None, help='用于匹配点云和序列的 txt/pkl 列表文件')
        extra_args, _ = parser.parse_known_args(sys.argv[1:])

        # 将参数融合到全局 args 中
        if not hasattr(args, 'point_cloud_dir'):
            args.point_cloud_dir = extra_args.point_cloud_dir
        if not hasattr(args, 'data_list_file'):
            args.data_list_file = extra_args.data_list_file

        # 检查是否处于 DDP 环境
        local_rank, world_size, rank = setup_distributed()

        if local_rank is None:
            # 如果不是 DDP 环境，则执行单卡训练
            single_gpu_train(args)
        else:
            # 如果是 DDP 环境，则执行多卡训练
            multi_gpu_train(args, local_rank, world_size, rank)

    except KeyboardInterrupt:
        print("训练被用户手动中断。")
    except Exception as e:
        print(f"训练过程中发生未捕获的异常: {e}")
        traceback.print_exc()
    finally:
        # 确保在程序退出前清理分布式环境，避免僵尸进程
        cleanup_distributed()


if __name__ == "__main__":
    main()
