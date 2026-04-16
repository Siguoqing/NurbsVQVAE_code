from __future__ import annotations

import pickle
from typing import Sequence

import numpy as np
import torch

from trainer_nurbs import VQVAE


def clean_state_dict(checkpoint):
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def infer_model_kwargs(clean_state_dict_, checkpoint_args, checkpoint_model_config):
    if checkpoint_model_config:
        model_kwargs = dict(checkpoint_model_config)
        model_kwargs["vq_distance"] = checkpoint_args.get("vq_distance", model_kwargs.get("vq_distance", "cos"))
        model_kwargs["vq_anchor"] = checkpoint_args.get("vq_anchor", model_kwargs.get("vq_anchor", "probrandom"))
        model_kwargs["vq_first_batch"] = bool(
            checkpoint_args.get("vq_first_batch", model_kwargs.get("vq_first_batch", False))
        )
        model_kwargs["vq_contras_loss"] = bool(
            checkpoint_args.get("vq_contras_loss", model_kwargs.get("vq_contras_loss", True))
        )
        model_kwargs["vq_beta"] = checkpoint_args.get("vq_beta", model_kwargs.get("vq_beta", None))
        return model_kwargs

    quant_weight = clean_state_dict_.get("quantize.embedding.weight")
    if quant_weight is not None:
        num_vq_embeddings, vq_embed_dim = quant_weight.shape
    else:
        num_vq_embeddings = int(checkpoint_args.get("quantization_size", 4096))
        vq_embed_dim = int(checkpoint_args.get("vq_embed_dim", 64))

    encoder_weight = clean_state_dict_.get("encoder.conv_in.weight")
    if encoder_weight is not None:
        in_channels = int(encoder_weight.shape[1])
    else:
        use_type_flag = bool(checkpoint_args.get("use_type_flag", True))
        in_channels = 4 if use_type_flag else 3

    quant_conv_weight = clean_state_dict_.get("quant_conv.weight")
    if quant_conv_weight is not None:
        latent_channels = int(quant_conv_weight.shape[1])
    else:
        latent_channels = int(checkpoint_args.get("latent_channels", 128))

    block_indices = sorted(
        {
            int(parts[2])
            for key in clean_state_dict_.keys()
            if key.startswith("encoder.down_blocks.")
            for parts in [key.split(".")]
            if len(parts) > 2 and parts[2].isdigit()
        }
    )
    block_out_channels = []
    for block_idx in block_indices:
        weight = clean_state_dict_.get(f"encoder.down_blocks.{block_idx}.resnets.0.conv1.weight")
        if weight is not None:
            block_out_channels.append(int(weight.shape[0]))

    if not block_out_channels:
        num_blocks = int(checkpoint_args.get("model_down_blocks", 1))
        base_channel_dim = int(checkpoint_args.get("base_channel_dim", 64))
        block_out_channels = [base_channel_dim * (2 ** idx) for idx in range(num_blocks)]

    return {
        "in_channels": in_channels,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D"] * len(block_out_channels),
        "up_block_types": ["UpDecoderBlock2D"] * len(block_out_channels),
        "block_out_channels": block_out_channels,
        "layers_per_block": 2,
        "act_fn": "silu",
        "latent_channels": latent_channels,
        "vq_embed_dim": int(vq_embed_dim),
        "num_vq_embeddings": int(num_vq_embeddings),
        "norm_num_groups": 32,
        "sample_size": 4,
        "vq_distance": checkpoint_args.get("vq_distance", "cos"),
        "vq_anchor": checkpoint_args.get("vq_anchor", "probrandom"),
        "vq_first_batch": bool(checkpoint_args.get("vq_first_batch", False)),
        "vq_contras_loss": bool(checkpoint_args.get("vq_contras_loss", True)),
        "vq_beta": checkpoint_args.get("vq_beta", None),
    }


def load_model_from_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_model_config = checkpoint.get("model_config", {})
    clean_dict = clean_state_dict(checkpoint)
    model_kwargs = infer_model_kwargs(clean_dict, checkpoint_args, checkpoint_model_config)

    model = VQVAE(**model_kwargs).to(device)
    model.load_state_dict(clean_dict, strict=True)
    model.eval()
    return model, model_kwargs


def pick_candidate_keys(coord_key: str, item_type: str):
    if item_type == "face":
        return ("face_ctrs_wcs_norm", "face_ctrs") if coord_key == "auto" else (coord_key,)
    if coord_key == "auto":
        return ("edge_ctrs_wcs_norm", "edge_ctrs")
    if coord_key == "face_ctrs_wcs_norm":
        return ("edge_ctrs_wcs_norm",)
    if coord_key == "face_ctrs":
        return ("edge_ctrs",)
    return (coord_key,)


def load_controls(cad: dict, candidate_keys: Sequence[str], expected_rows: int):
    for key in candidate_keys:
        raw = cad.get(key)
        if raw is None:
            continue
        try:
            array = np.asarray(raw, dtype=np.float32).reshape(-1, expected_rows, 3)
        except Exception:
            continue
        if len(array) == 0:
            continue
        array = array[np.isfinite(array).all(axis=(1, 2))]
        if len(array) > 0:
            return key, array
    return None, None


def load_controls_from_file(file_path: str, item_type: str, coord_key: str):
    expected_rows = 16 if item_type == "face" else 4
    candidate_keys = pick_candidate_keys(coord_key, item_type)
    with open(file_path, "rb") as f:
        cad = pickle.load(f)

    used_key, controls = load_controls(cad, candidate_keys, expected_rows)
    if controls is None:
        raise ValueError(f"Could not load {item_type} controls from {file_path}")
    return used_key, controls


def build_model_input(controls: np.ndarray, item_type: str, in_channels: int):
    if item_type == "face":
        grid = controls.reshape(4, 4, 3)
        flag = np.zeros((4, 4, 1), dtype=np.float32)
    else:
        grid = np.tile(controls, (4, 1)).reshape(4, 4, 3)
        flag = np.ones((4, 4, 1), dtype=np.float32)
    model_input = np.concatenate([grid, flag], axis=-1) if in_channels == 4 else grid
    tensor = torch.from_numpy(model_input).float().unsqueeze(0).permute(0, 3, 1, 2)
    return tensor, grid


def reconstruct_controls(model, model_kwargs, controls: np.ndarray, item_type: str, device):
    x_in, target_grid = build_model_input(controls, item_type, model_kwargs["in_channels"])
    x_in = x_in.to(device)
    with torch.no_grad():
        h = model.encoder(x_in)
        h = model.quant_conv(h)
        quant_out, _, _ = model.quantize(h)
        x_recon = model.decoder(model.post_quant_conv(quant_out))
    recon_grid = x_recon[0, :3].permute(1, 2, 0).cpu().numpy()
    diff_grid = np.abs(recon_grid - target_grid)
    return x_in, target_grid, recon_grid, diff_grid
