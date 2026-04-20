from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from nurbs_dataset import build_nurbs_train_val_datasets, format_dataset_summary
from trainer_nurbs import NurbsVQVAETrainer


DEFAULT_DATA_DIR = "/mnt/docker_dir/lijiahao/NurbsVQVAE_code/furniture_parsed/ABC_Dataset_NEW"


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a clean NURBS VQ-VAE baseline")

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing CAD .pkl files")
    parser.add_argument("--data_list", type=str, default="", help="Optional split pickle with explicit train/val file lists")
    parser.add_argument("--save_dir", type=str, default="checkpoints_vqvae/restart_baseline")
    parser.add_argument("--tb_log_dir", type=str, default="logs/nurbs_vqvae/restart_baseline")
    parser.add_argument("--weight", type=str, default="", help="Checkpoint path to resume from")

    parser.add_argument("--split_ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--use_type_flag", type=str2bool, default=True)
    parser.add_argument("--include_faces", type=str2bool, default=True)
    parser.add_argument("--include_edges", type=str2bool, default=True)
    parser.add_argument("--aug", type=str2bool, default=True, help="Enable train-time rotation augmentation")

    parser.add_argument("--batch_size", type=int, default=1536, help="Per-GPU batch size under DDP")
    parser.add_argument("--train_nepoch", type=int, default=500)
    parser.add_argument("--test_nepoch", type=int, default=1)
    parser.add_argument("--save_nepoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--best_metric", choices=["recon_loss", "total_loss", "precision_score"], default="recon_loss")

    parser.add_argument("--quantization_size", type=int, default=1024)
    parser.add_argument("--model_down_blocks", type=int, default=2)
    parser.add_argument("--base_channel_dim", type=int, default=64)
    parser.add_argument("--latent_channels", type=int, default=128)
    parser.add_argument("--vq_embed_dim", type=int, default=32)
    parser.add_argument("--vq_loss_weight", type=float, default=1.0)
    parser.add_argument("--vq_distance", choices=["cos", "l2"], default="cos")
    parser.add_argument("--vq_anchor", choices=["probrandom", "closest", "random"], default="probrandom")
    parser.add_argument("--vq_first_batch", type=str2bool, default=False)
    parser.add_argument("--vq_contras_loss", type=str2bool, default=True)
    parser.add_argument("--vq_beta", type=float, default=None)

    parser.add_argument("--recon_l1_weight", type=float, default=0.0)
    parser.add_argument("--face_boundary_weight", type=float, default=0.0)
    parser.add_argument("--face_corner_weight", type=float, default=0.0)
    parser.add_argument("--edge_endpoint_weight", type=float, default=0.0)
    parser.add_argument("--max_error_weight", type=float, default=0.0)
    parser.add_argument("--boundary_max_error_weight", type=float, default=0.0)

    return parser


def print_args(args):
    print("=" * 60)
    print("NURBS VQ-VAE Restart Configuration")
    print("=" * 60)
    for key in sorted(vars(args)):
        print(f"{key}: {getattr(args, key)}")
    print("=" * 60)


def main(argv=None):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = build_parser()
    args = parser.parse_args(argv)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    multi_gpu = world_size > 1
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        print_args(args)

    train_dataset, val_dataset = build_nurbs_train_val_datasets(
        data_dir=args.data_dir,
        data_list=args.data_list or None,
        split_ratio=args.split_ratio,
        seed=args.seed,
        max_files=args.max_files,
        use_type_flag=args.use_type_flag,
        include_faces=args.include_faces,
        include_edges=args.include_edges,
        aug=args.aug,
    )

    if rank == 0:
        for line in format_dataset_summary(train_dataset.summary):
            print(line)
        for line in format_dataset_summary(val_dataset.summary):
            print(line)
        if len(train_dataset) == 0:
            raise RuntimeError("Training dataset is empty. Check data_dir and selected keys.")

    trainer = NurbsVQVAETrainer(args, train_dataset, val_dataset, multi_gpu=multi_gpu)

    try:
        trainer.fit()
    finally:
        trainer.close_writer()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
