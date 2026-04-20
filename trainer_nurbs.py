from __future__ import annotations

import math
import os
import time
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
try:
    from diffusers import VQModel as DiffusersVQModel
except ModuleNotFoundError:
    DiffusersVQModel = None
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from quantise import VectorQuantiser


def _make_group_norm(num_channels: int, max_groups: int) -> nn.GroupNorm:
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=max(1, num_groups), num_channels=num_channels)


def _make_activation(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn == "silu":
        return nn.SiLU()
    if act_fn == "relu":
        return nn.ReLU(inplace=True)
    if act_fn == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation function: {act_fn}")


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_fn: str, norm_num_groups: int):
        super().__init__()
        self.norm1 = _make_group_norm(in_channels, norm_num_groups)
        self.act1 = _make_activation(act_fn)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _make_group_norm(out_channels, norm_num_groups)
        self.act2 = _make_activation(act_fn)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        add_downsample: bool,
        act_fn: str,
        norm_num_groups: int,
    ):
        super().__init__()
        resnets = []
        current_channels = in_channels
        for _ in range(num_layers):
            resnets.append(ResnetBlock2D(current_channels, out_channels, act_fn, norm_num_groups))
            current_channels = out_channels
        self.resnets = nn.ModuleList(resnets)
        self.downsampler = (
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if add_downsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        return self.downsampler(x)


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        add_upsample: bool,
        act_fn: str,
        norm_num_groups: int,
    ):
        super().__init__()
        resnets = []
        current_channels = in_channels
        for _ in range(num_layers):
            resnets.append(ResnetBlock2D(current_channels, out_channels, act_fn, norm_num_groups))
            current_channels = out_channels
        self.resnets = nn.ModuleList(resnets)
        self.upsampler = (
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if add_upsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        return self.upsampler(x)


class Encoder2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        block_out_channels,
        layers_per_block: int,
        act_fn: str,
        norm_num_groups: int,
        latent_channels: int,
    ):
        super().__init__()
        first_channels = int(block_out_channels[0])
        self.conv_in = nn.Conv2d(in_channels, first_channels, kernel_size=3, padding=1)

        down_blocks = []
        current_channels = first_channels
        for idx, out_channels in enumerate(block_out_channels):
            down_blocks.append(
                DownEncoderBlock2D(
                    in_channels=current_channels,
                    out_channels=int(out_channels),
                    num_layers=layers_per_block,
                    add_downsample=idx < len(block_out_channels) - 1,
                    act_fn=act_fn,
                    norm_num_groups=norm_num_groups,
                )
            )
            current_channels = int(out_channels)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.mid = ResnetBlock2D(current_channels, current_channels, act_fn, norm_num_groups)
        self.norm_out = _make_group_norm(current_channels, norm_num_groups)
        self.act = _make_activation(act_fn)
        self.conv_out = nn.Conv2d(current_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid(x)
        x = self.conv_out(self.act(self.norm_out(x)))
        return x


class Decoder2D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        block_out_channels,
        layers_per_block: int,
        act_fn: str,
        norm_num_groups: int,
        latent_channels: int,
    ):
        super().__init__()
        reversed_channels = list(reversed([int(ch) for ch in block_out_channels]))
        self.conv_in = nn.Conv2d(latent_channels, reversed_channels[0], kernel_size=3, padding=1)

        up_blocks = []
        current_channels = reversed_channels[0]
        for idx, next_channels in enumerate(reversed_channels):
            up_blocks.append(
                UpDecoderBlock2D(
                    in_channels=current_channels,
                    out_channels=int(next_channels),
                    num_layers=layers_per_block,
                    add_upsample=idx < len(reversed_channels) - 1,
                    act_fn=act_fn,
                    norm_num_groups=norm_num_groups,
                )
            )
            current_channels = int(next_channels)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.mid = ResnetBlock2D(current_channels, current_channels, act_fn, norm_num_groups)
        self.norm_out = _make_group_norm(current_channels, norm_num_groups)
        self.act = _make_activation(act_fn)
        self.conv_out = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.mid(x)
        x = self.conv_out(self.act(self.norm_out(x)))
        return x


class LocalVQVAE(nn.Module):
    """Fallback 2D VQ-VAE used only when diffusers is unavailable locally."""

    def __init__(self, *args, **kwargs):
        vq_distance = kwargs.pop("vq_distance", "cos")
        vq_anchor = kwargs.pop("vq_anchor", "probrandom")
        vq_first_batch = kwargs.pop("vq_first_batch", False)
        vq_contras_loss = kwargs.pop("vq_contras_loss", True)
        vq_beta = kwargs.pop("vq_beta", None)
        in_channels = kwargs.pop("in_channels")
        out_channels = kwargs.pop("out_channels")
        block_out_channels = kwargs.pop("block_out_channels")
        layers_per_block = kwargs.pop("layers_per_block", 2)
        act_fn = kwargs.pop("act_fn", "silu")
        latent_channels = kwargs.pop("latent_channels")
        vq_embed_dim = kwargs.pop("vq_embed_dim")
        num_vq_embeddings = kwargs.pop("num_vq_embeddings")
        norm_num_groups = kwargs.pop("norm_num_groups", 32)
        kwargs.pop("down_block_types", None)
        kwargs.pop("up_block_types", None)
        kwargs.pop("sample_size", None)
        super().__init__()

        self.config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "block_out_channels": list(block_out_channels),
            "layers_per_block": layers_per_block,
            "act_fn": act_fn,
            "latent_channels": latent_channels,
            "vq_embed_dim": vq_embed_dim,
            "num_vq_embeddings": num_vq_embeddings,
            "norm_num_groups": norm_num_groups,
        }

        self.encoder = Encoder2D(
            in_channels=in_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            latent_channels=latent_channels,
        )
        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, kernel_size=1)
        self.quantize = VectorQuantiser(
            num_embed=num_vq_embeddings,
            embed_dim=vq_embed_dim,
            beta=0.25 if vq_beta is None else vq_beta,
            distance=vq_distance,
            anchor=vq_anchor,
            first_batch=vq_first_batch,
            contras_loss=vq_contras_loss,
        )
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, kernel_size=1)
        self.decoder = Decoder2D(
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            latent_channels=latent_channels,
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant_out, vq_loss, indices = self.quantize(h)
        recon = self.decoder(self.post_quant_conv(quant_out))
        return recon, vq_loss, indices


if DiffusersVQModel is not None:
    class VQVAE(DiffusersVQModel):
        """Preferred implementation aligned with the senior's original framework."""

        def __init__(self, *args, **kwargs):
            vq_distance = kwargs.pop("vq_distance", "cos")
            vq_anchor = kwargs.pop("vq_anchor", "probrandom")
            vq_first_batch = kwargs.pop("vq_first_batch", False)
            vq_contras_loss = kwargs.pop("vq_contras_loss", True)
            vq_beta = kwargs.pop("vq_beta", None)
            super().__init__(*args, **kwargs)

            old_quant = self.quantize
            self.quantize = VectorQuantiser(
                num_embed=old_quant.n_e,
                embed_dim=old_quant.vq_embed_dim,
                beta=old_quant.beta if vq_beta is None else vq_beta,
                distance=vq_distance,
                anchor=vq_anchor,
                first_batch=vq_first_batch,
                contras_loss=vq_contras_loss,
            )
            self.quantize.embedding.weight.data.copy_(old_quant.embedding.weight.data)
else:
    VQVAE = LocalVQVAE


class NurbsVQVAETrainer:
    """Clean restart-friendly VQ-VAE trainer for 4x4 NURBS control grids."""

    def __init__(self, args, train_dataset, val_dataset, multi_gpu: bool = False, collate_fn=None):
        self.args = args
        self.iters = 0
        self.epoch = 1
        self.save_dir = args.save_dir
        self.multi_gpu = multi_gpu

        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = bool(multi_gpu and self.world_size > 1)
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))

        self.device = self._init_device()
        self.is_main_process = self.rank == 0

        self.use_type_flag = bool(getattr(args, "use_type_flag", True))
        self.in_channels = 4 if self.use_type_flag else 3
        self.out_channels = 3

        self.learning_rate = float(getattr(args, "learning_rate", 1e-4))
        self.weight_decay = float(getattr(args, "weight_decay", 1e-6))
        self.beta1 = float(getattr(args, "beta1", 0.9))
        self.beta2 = float(getattr(args, "beta2", 0.999))
        self.max_grad_norm = float(getattr(args, "max_grad_norm", 1.0))
        self.vq_loss_weight = float(getattr(args, "vq_loss_weight", 1.0))
        self.best_metric_name = getattr(args, "best_metric", "recon_loss")

        self.quantization_size = int(getattr(args, "quantization_size", 1024))
        self.model_down_blocks = int(getattr(args, "model_down_blocks", 2))
        self.base_channel_dim = int(getattr(args, "base_channel_dim", 64))
        self.latent_channels = int(getattr(args, "latent_channels", 128))
        self.vq_embed_dim = int(getattr(args, "vq_embed_dim", 32))

        if self.model_down_blocks < 1:
            raise ValueError(f"model_down_blocks must be >= 1, got {self.model_down_blocks}")

        self.block_out_channels = tuple(
            self.base_channel_dim * (2 ** idx) for idx in range(self.model_down_blocks)
        )

        self.recon_l1_weight = float(getattr(args, "recon_l1_weight", 0.0))
        self.face_boundary_weight = float(getattr(args, "face_boundary_weight", 0.0))
        self.face_corner_weight = float(getattr(args, "face_corner_weight", 0.0))
        self.edge_endpoint_weight = float(getattr(args, "edge_endpoint_weight", 0.0))
        self.max_error_weight = float(getattr(args, "max_error_weight", 0.0))
        self.boundary_max_error_weight = float(getattr(args, "boundary_max_error_weight", 0.0))

        self.vq_distance = getattr(args, "vq_distance", "cos")
        self.vq_anchor = getattr(args, "vq_anchor", "probrandom")
        self.vq_first_batch = bool(getattr(args, "vq_first_batch", False))
        self.vq_contras_loss = bool(getattr(args, "vq_contras_loss", True))
        self.vq_beta = getattr(args, "vq_beta", None)

        self.model_config = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "down_block_types": ["DownEncoderBlock2D"] * self.model_down_blocks,
            "up_block_types": ["UpDecoderBlock2D"] * self.model_down_blocks,
            "block_out_channels": list(self.block_out_channels),
            "layers_per_block": 2,
            "act_fn": "silu",
            "latent_channels": self.latent_channels,
            "vq_embed_dim": self.vq_embed_dim,
            "num_vq_embeddings": self.quantization_size,
            "norm_num_groups": 32,
            "sample_size": 4,
            "vq_distance": self.vq_distance,
            "vq_anchor": self.vq_anchor,
            "vq_first_batch": self.vq_first_batch,
            "vq_contras_loss": self.vq_contras_loss,
            "vq_beta": self.vq_beta,
        }

        self.face_boundary_mask = torch.tensor(
            [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 4, 4)
        self.face_corner_mask = torch.tensor(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 4, 4)
        self.edge_endpoint_mask = torch.tensor(
            [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 4, 4)

        self.writer = None
        if self.is_main_process:
            os.makedirs(args.tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=args.tb_log_dir)

        self.train_sampler, self.val_sampler = self._build_samplers(train_dataset, val_dataset)
        self.train_dataloader = self._build_dataloader(train_dataset, train=True, collate_fn=collate_fn)
        self.val_dataloader = self._build_dataloader(val_dataset, train=False, collate_fn=collate_fn)

        self.model = VQVAE(**self.model_config).to(self.device)
        if self.distributed:
            device_ids = [self.local_rank] if self.device.type == "cuda" else None
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=device_ids)

        self.codebook_size = self.unwrap_model().quantize.embedding.num_embeddings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=1e-8,
        )

        self.amp_enabled = bool(getattr(args, "amp", True)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self.best_val_loss = float("inf")
        self.best_val_score = float("inf")
        self.best_path = None

        if getattr(args, "weight", "") and os.path.exists(args.weight):
            self._load_checkpoint(args.weight)
        elif self.is_main_process:
            print("Starting NURBS VQ-VAE training from scratch")

        if self.is_main_process:
            print(
                "VQ-VAE config: "
                f"blocks={self.model_down_blocks}, "
                f"channels={list(self.block_out_channels)}, "
                f"latent_channels={self.latent_channels}, "
                f"vq_embed_dim={self.vq_embed_dim}, "
                f"codebook={self.quantization_size}"
            )
            backend_name = "diffusers.VQModel" if DiffusersVQModel is not None else "local-fallback"
            print(f"VQ-VAE backbone: {backend_name}")

    def _init_device(self) -> torch.device:
        if self.distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
            dist.init_process_group(backend=backend)

        if torch.cuda.is_available():
            if self.distributed:
                torch.cuda.set_device(self.local_rank)
                return torch.device(f"cuda:{self.local_rank}")
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_samplers(self, train_dataset, val_dataset):
        if not self.distributed:
            return None, None

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        return train_sampler, val_sampler

    def _build_dataloader(self, dataset, train: bool, collate_fn=None) -> DataLoader:
        sampler = self.train_sampler if train else self.val_sampler
        shuffle = bool(train and sampler is None)
        num_workers = int(getattr(self.args, "num_workers", 4))
        batch_size = int(getattr(self.args, "batch_size", 1024))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=train,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
        )

    def unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _autocast_context(self, enabled: bool):
        return torch.cuda.amp.autocast(enabled=enabled and self.amp_enabled)

    def _extract_target_coords(self, batch_data: torch.Tensor) -> torch.Tensor:
        return batch_data[:, :3, :, :] if self.use_type_flag and batch_data.shape[1] >= 4 else batch_data

    def _split_face_edge_samples(self, batch_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch_data.shape[0]
        if self.use_type_flag and batch_data.shape[1] >= 4:
            edge_mask = batch_data[:, 3:4, :, :].mean(dim=(1, 2, 3)) > 0.5
        else:
            edge_mask = torch.zeros(batch_size, dtype=torch.bool, device=batch_data.device)
        face_mask = ~edge_mask
        return face_mask, edge_mask

    def _masked_group_mean(
        self, abs_diff: torch.Tensor, sample_mask: torch.Tensor, spatial_mask: torch.Tensor
    ) -> torch.Tensor:
        if sample_mask.numel() == 0 or not sample_mask.any():
            return abs_diff.new_tensor(0.0)
        subset = abs_diff[sample_mask]
        denom = sample_mask.sum().float() * spatial_mask.sum() * subset.shape[1]
        return (subset * spatial_mask).sum() / denom.clamp_min(1.0)

    def _masked_group_max(
        self, abs_diff: torch.Tensor, sample_mask: torch.Tensor, spatial_mask: torch.Tensor
    ) -> torch.Tensor:
        if sample_mask.numel() == 0 or not sample_mask.any():
            return abs_diff.new_tensor(0.0)
        subset = abs_diff[sample_mask]
        return (subset * spatial_mask).amax(dim=(1, 2, 3)).mean()

    def _compute_recon_loss(self, recon: torch.Tensor, batch_data: torch.Tensor):
        target = self._extract_target_coords(batch_data)
        abs_diff = torch.abs(recon - target)

        mse_loss = F.mse_loss(recon, target)
        l1_loss = abs_diff.mean()
        max_abs_error = abs_diff.amax(dim=(1, 2, 3)).mean()

        face_mask, edge_mask = self._split_face_edge_samples(batch_data)
        face_boundary_l1 = self._masked_group_mean(abs_diff, face_mask, self.face_boundary_mask)
        face_corner_l1 = self._masked_group_mean(abs_diff, face_mask, self.face_corner_mask)
        edge_endpoint_l1 = self._masked_group_mean(abs_diff, edge_mask, self.edge_endpoint_mask)

        face_boundary_max = self._masked_group_max(abs_diff, face_mask, self.face_boundary_mask)
        edge_endpoint_max = self._masked_group_max(abs_diff, edge_mask, self.edge_endpoint_mask)
        boundary_max_error = torch.maximum(face_boundary_max, edge_endpoint_max)

        recon_loss = mse_loss

        metrics = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "max_abs_error": max_abs_error,
            "face_boundary_l1": face_boundary_l1,
            "face_corner_l1": face_corner_l1,
            "edge_endpoint_l1": edge_endpoint_l1,
            "boundary_max_error": boundary_max_error,
        }
        return recon_loss, metrics

    def _reduce_metric_dict(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if not self.distributed:
            return metrics
        keys = list(metrics.keys())
        values = torch.tensor([float(metrics[key]) for key in keys], device=self.device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return {key: values[idx].item() for idx, key in enumerate(keys)}

    def _barrier(self):
        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def fit(self):
        while self.epoch <= self.args.train_nepoch:
            self.train_one_epoch()
            finished_epoch = self.epoch - 1

            if finished_epoch % self.args.test_nepoch == 0:
                self.validate()

            if finished_epoch % self.args.save_nepoch == 0:
                self.save_model(save_epoch=finished_epoch)

    def train_one_epoch(self):
        self.model.train()
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(self.epoch)
        if self.val_sampler is not None and hasattr(self.val_sampler, "set_epoch"):
            self.val_sampler.set_epoch(self.epoch)

        progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.is_main_process)
        if self.is_main_process:
            progress_bar.set_description(f"Epoch {self.epoch}")

        epoch_sums = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "vq_loss": 0.0,
            "mse_loss": 0.0,
            "l1_loss": 0.0,
            "max_abs_error": 0.0,
            "boundary_max_error": 0.0,
            "usage_rate": 0.0,
            "perplexity": 0.0,
            "batch_count": 0.0,
        }

        start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(self.train_dataloader):
            try:
                batch_data = batch_data.to(self.device, non_blocking=True).permute(0, 3, 1, 2)
                model = self.unwrap_model()

                with self._autocast_context(enabled=True):
                    h = model.encoder(batch_data)
                    h = model.quant_conv(h)

                with self._autocast_context(enabled=False):
                    quant_out, vq_loss, indices = model.quantize(h.float())

                with self._autocast_context(enabled=True):
                    recon = model.decoder(model.post_quant_conv(quant_out.to(dtype=h.dtype)))
                    recon_loss, recon_metrics = self._compute_recon_loss(recon, batch_data)
                    total_loss = recon_loss + self.vq_loss_weight * vq_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue

                self.scaler.scale(total_loss).backward()
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                perplexity, usage_rate = self._extract_codebook_stats(indices)

                epoch_sums["total_loss"] += total_loss.item()
                epoch_sums["recon_loss"] += recon_loss.item()
                epoch_sums["vq_loss"] += vq_loss.item()
                epoch_sums["mse_loss"] += recon_metrics["mse_loss"].item()
                epoch_sums["l1_loss"] += recon_metrics["l1_loss"].item()
                epoch_sums["max_abs_error"] += recon_metrics["max_abs_error"].item()
                epoch_sums["boundary_max_error"] += recon_metrics["boundary_max_error"].item()
                epoch_sums["usage_rate"] += usage_rate
                epoch_sums["perplexity"] += perplexity
                epoch_sums["batch_count"] += 1.0
                self.iters += 1

                if self.is_main_process:
                    denom = max(epoch_sums["batch_count"], 1.0)
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss": f"{epoch_sums['total_loss'] / denom:.6f}",
                            "recon": f"{epoch_sums['recon_loss'] / denom:.6f}",
                            "vq": f"{epoch_sums['vq_loss'] / denom:.6f}",
                            "bmax": f"{epoch_sums['boundary_max_error'] / denom:.6f}",
                            "usage": f"{epoch_sums['usage_rate'] / denom:.2f}",
                        }
                    )
            except Exception as exc:
                if "out of memory" in str(exc).lower():
                    self.optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if self.is_main_process:
                    print(f"Skipping failed train batch {batch_idx}: {exc}")
                continue

        progress_bar.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        reduced = self._reduce_metric_dict(epoch_sums)
        batch_count = max(reduced["batch_count"], 1.0)
        epoch_time = time.time() - start_time

        if self.is_main_process:
            print(f"Epoch [{self.epoch}/{self.args.train_nepoch}] Time: {epoch_time:.2f}s")
            print(
                f"  Train Loss: total={reduced['total_loss'] / batch_count:.6f} "
                f"recon={reduced['recon_loss'] / batch_count:.6f} "
                f"vq={reduced['vq_loss'] / batch_count:.6f}"
            )
            print(
                f"  Train Metrics: mse={reduced['mse_loss'] / batch_count:.6f} "
                f"l1={reduced['l1_loss'] / batch_count:.6f} "
                f"max={reduced['max_abs_error'] / batch_count:.6f} "
                f"boundary_max={reduced['boundary_max_error'] / batch_count:.6f}"
            )

            if self.writer is not None:
                self.writer.add_scalar("train/total_loss", reduced["total_loss"] / batch_count, self.epoch)
                self.writer.add_scalar("train/recon_loss", reduced["recon_loss"] / batch_count, self.epoch)
                self.writer.add_scalar("train/vq_loss", reduced["vq_loss"] / batch_count, self.epoch)
                self.writer.add_scalar("train/mse_loss", reduced["mse_loss"] / batch_count, self.epoch)
                self.writer.add_scalar("train/l1_loss", reduced["l1_loss"] / batch_count, self.epoch)
                self.writer.add_scalar(
                    "train/max_abs_error", reduced["max_abs_error"] / batch_count, self.epoch
                )
                self.writer.add_scalar(
                    "train/boundary_max_error",
                    reduced["boundary_max_error"] / batch_count,
                    self.epoch,
                )
                self.writer.add_scalar("train/codebook_usage", reduced["usage_rate"] / batch_count, self.epoch)
                self.writer.add_scalar("train/perplexity", reduced["perplexity"] / batch_count, self.epoch)
                self.writer.add_scalar(
                    "train/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    self.epoch,
                )

        self.epoch += 1
        self._barrier()

    def _extract_codebook_stats(self, indices) -> Tuple[float, float]:
        if not isinstance(indices, tuple) or len(indices) == 0:
            return 0.0, 0.0

        perplexity = indices[0]
        if hasattr(perplexity, "item"):
            perplexity = float(perplexity.item())
        else:
            perplexity = float(perplexity)

        encoding_indices = indices[2] if len(indices) > 2 else indices[1]
        used_codes = torch.unique(encoding_indices).numel()
        usage_rate = float(used_codes) / float(self.codebook_size)
        return perplexity, usage_rate

    def validate(self):
        self.model.eval()
        val_sums = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "vq_loss": 0.0,
            "mse_loss": 0.0,
            "l1_loss": 0.0,
            "max_abs_error": 0.0,
            "boundary_max_error": 0.0,
            "perplexity": 0.0,
            "batch_count": 0.0,
        }
        all_code_counts = torch.zeros(self.codebook_size, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_dataloader):
                try:
                    batch_data = batch_data.to(self.device, non_blocking=True).permute(0, 3, 1, 2)
                    model = self.unwrap_model()

                    with self._autocast_context(enabled=True):
                        h = model.encoder(batch_data)
                        h = model.quant_conv(h)
                    with self._autocast_context(enabled=False):
                        quant_out, vq_loss, indices = model.quantize(h.float())
                    with self._autocast_context(enabled=True):
                        recon = model.decoder(model.post_quant_conv(quant_out.to(dtype=h.dtype)))
                        recon_loss, recon_metrics = self._compute_recon_loss(recon, batch_data)
                        total_loss = recon_loss + self.vq_loss_weight * vq_loss

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue

                    perplexity, _ = self._extract_codebook_stats(indices)
                    indices_tensor = indices[2] if len(indices) > 2 else indices[1]
                    all_code_counts += torch.bincount(
                        indices_tensor.view(-1),
                        minlength=self.codebook_size,
                    ).float()

                    val_sums["total_loss"] += total_loss.item()
                    val_sums["recon_loss"] += recon_loss.item()
                    val_sums["vq_loss"] += vq_loss.item()
                    val_sums["mse_loss"] += recon_metrics["mse_loss"].item()
                    val_sums["l1_loss"] += recon_metrics["l1_loss"].item()
                    val_sums["max_abs_error"] += recon_metrics["max_abs_error"].item()
                    val_sums["boundary_max_error"] += recon_metrics["boundary_max_error"].item()
                    val_sums["perplexity"] += perplexity
                    val_sums["batch_count"] += 1.0
                except Exception as exc:
                    if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if self.is_main_process:
                        print(f"Skipping failed val batch {batch_idx}: {exc}")
                    continue

        if self.distributed:
            dist.all_reduce(all_code_counts, op=dist.ReduceOp.SUM)
        reduced = self._reduce_metric_dict(val_sums)
        batch_count = max(reduced["batch_count"], 1.0)
        avg_usage = float((all_code_counts > 0).sum().item()) / float(self.codebook_size)
        precision_score = (
            reduced["boundary_max_error"] / batch_count
            + 0.5 * reduced["max_abs_error"] / batch_count
            + 0.1 * reduced["l1_loss"] / batch_count
        )

        if self.distributed:
            usage_tensor = torch.tensor(avg_usage, device=self.device)
            dist.all_reduce(usage_tensor, op=dist.ReduceOp.SUM)
            avg_usage = usage_tensor.item() / float(self.world_size)

        avg_recon_loss = reduced["recon_loss"] / batch_count
        avg_total_loss = reduced["total_loss"] / batch_count
        candidate_score = {
            "recon_loss": avg_recon_loss,
            "total_loss": avg_total_loss,
            "precision_score": precision_score,
        }[self.best_metric_name]

        val_epoch = self.epoch - 1
        is_best = False
        if avg_recon_loss < self.best_val_loss:
            self.best_val_loss = avg_recon_loss
        if candidate_score < self.best_val_score:
            self.best_val_score = candidate_score
            is_best = True
            self.best_path = "deepcad_nurbs_vqvae_best.pt"

        if self.is_main_process:
            print(
                f"Validation: total={avg_total_loss:.6f} recon={avg_recon_loss:.6f} "
                f"vq={reduced['vq_loss'] / batch_count:.6f}"
            )
            print(
                f"Validation metrics: mse={reduced['mse_loss'] / batch_count:.6f} "
                f"l1={reduced['l1_loss'] / batch_count:.6f} "
                f"max={reduced['max_abs_error'] / batch_count:.6f} "
                f"boundary_max={reduced['boundary_max_error'] / batch_count:.6f} "
                f"precision_score={precision_score:.6f} "
                f"usage={avg_usage:.4f}"
            )
            print(f"Best checkpoint metric ({self.best_metric_name}): {self.best_val_score:.6f}")

            if self.writer is not None:
                self.writer.add_scalar("val/total_loss", avg_total_loss, val_epoch)
                self.writer.add_scalar("val/recon_loss", avg_recon_loss, val_epoch)
                self.writer.add_scalar("val/vq_loss", reduced["vq_loss"] / batch_count, val_epoch)
                self.writer.add_scalar("val/mse_loss", reduced["mse_loss"] / batch_count, val_epoch)
                self.writer.add_scalar("val/l1_loss", reduced["l1_loss"] / batch_count, val_epoch)
                self.writer.add_scalar("val/max_abs_error", reduced["max_abs_error"] / batch_count, val_epoch)
                self.writer.add_scalar(
                    "val/boundary_max_error",
                    reduced["boundary_max_error"] / batch_count,
                    val_epoch,
                )
                self.writer.add_scalar("val/precision_score", precision_score, val_epoch)
                self.writer.add_scalar("val/codebook_usage", avg_usage, val_epoch)
                self.writer.add_scalar("val/perplexity", reduced["perplexity"] / batch_count, val_epoch)
                self.writer.add_scalar("val/best_recon_loss", self.best_val_loss, val_epoch)
                self.writer.add_scalar("val/best_metric", self.best_val_score, val_epoch)

        if is_best:
            self.save_model(is_best=True, save_epoch=val_epoch)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._barrier()
        return avg_total_loss

    def test_val(self):
        return self.validate()

    def save_model(self, is_best: bool = False, save_epoch: int | None = None):
        if not self.is_main_process:
            self._barrier()
            return

        epoch_to_save = self.epoch - 1 if save_epoch is None else save_epoch
        os.makedirs(self.save_dir, exist_ok=True)

        model_to_save = self.unwrap_model()
        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": epoch_to_save + 1,
            "iters": self.iters,
            "best_val_loss": self.best_val_loss,
            "best_val_score": self.best_val_score,
            "best_path": self.best_path,
            "current_learning_rate": self.optimizer.param_groups[0]["lr"],
            "multi_gpu": self.multi_gpu,
            "args": self._serializable_args(),
            "model_config": self.model_config,
        }

        save_path = os.path.join(self.save_dir, f"nurbs_vqvae_epoch_{epoch_to_save}.pt")
        torch.save(checkpoint, save_path)

        if is_best:
            best_save_path = os.path.join(self.save_dir, "deepcad_nurbs_vqvae_best.pt")
            torch.save(checkpoint, best_save_path)
            print(f"Saved new best checkpoint to: {best_save_path}")

        self._barrier()

    def _serializable_args(self):
        serializable = {}
        for key, value in vars(self.args).items():
            if isinstance(value, (int, float, str, bool, type(None))):
                serializable[key] = value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(item, (int, float, str, bool, type(None))) for item in value
            ):
                serializable[key] = list(value)
        return serializable

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        clean_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("module.") and not hasattr(self.model, "module"):
                new_key = key[7:]
            elif not key.startswith("module.") and hasattr(self.model, "module"):
                new_key = f"module.{key}"
            clean_state_dict[new_key] = value

        self.model.load_state_dict(clean_state_dict, strict=False)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.epoch = int(checkpoint.get("epoch", self.epoch))
        self.iters = int(checkpoint.get("iters", self.iters))
        self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
        self.best_val_score = float(checkpoint.get("best_val_score", self.best_val_score))
        self.best_path = checkpoint.get("best_path", self.best_path)

        if self.is_main_process:
            print(f"Resumed training from checkpoint: {checkpoint_path}")

    def close_writer(self):
        if self.writer is not None:
            self.writer.close()
