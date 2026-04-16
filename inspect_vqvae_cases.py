import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vqvae_eval_utils import load_controls_from_file, load_model_from_checkpoint, reconstruct_controls


def _save_npz(case_dir: Path, item_type: str, record: dict, target_grid, recon_grid, diff_grid):
    save_path = case_dir / "case_data.npz"
    np.savez_compressed(
        save_path,
        item_type=item_type,
        source_file=record["file"],
        item_index=record["index"],
        target_grid=target_grid,
        recon_grid=recon_grid,
        diff_grid=diff_grid,
    )
    return save_path


def _maybe_plot(case_dir: Path, item_type: str, target_grid, recon_grid, diff_grid):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig = plt.figure(figsize=(14, 4))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    def plot_grid(ax, grid, title):
        pts = grid.reshape(-1, 3)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18)
        if item_type == "face":
            for i in range(4):
                ax.plot(grid[i, :, 0], grid[i, :, 1], grid[i, :, 2], linewidth=1.0)
                ax.plot(grid[:, i, 0], grid[:, i, 1], grid[:, i, 2], linewidth=1.0)
        else:
            curve = grid[0]
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], linewidth=1.5)
        ax.set_title(title)

    plot_grid(ax1, target_grid, "Target")
    plot_grid(ax2, recon_grid, "Reconstruction")

    diff_pts = diff_grid.reshape(-1, 3)
    ax3.scatter(diff_pts[:, 0], diff_pts[:, 1], diff_pts[:, 2], s=18, c=diff_pts.mean(axis=1), cmap="magma")
    ax3.set_title("Abs Diff")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    fig.tight_layout()
    save_path = case_dir / "case_plot.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def inspect_cases(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_kwargs = load_model_from_checkpoint(args.ckpt_path, device)

    with open(args.cases_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    item_type = payload["item_type"]
    records = payload["records"][: args.num_cases]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(records)} {item_type} cases from {args.cases_json}")
    print(f"Saving inspection outputs to {output_dir}")

    for case_rank, record in enumerate(records, start=1):
        coord_key = record.get("coord_key", args.coord_key)
        used_key, controls_all = load_controls_from_file(record["file"], item_type, coord_key)
        controls = controls_all[int(record["index"])]
        _, target_grid, recon_grid, diff_grid = reconstruct_controls(model, model_kwargs, controls, item_type, device)

        slug = f"{item_type}_{case_rank:03d}_idx_{record['index']}"
        case_dir = output_dir / slug
        case_dir.mkdir(parents=True, exist_ok=True)

        npz_path = _save_npz(case_dir, item_type, record, target_grid, recon_grid, diff_grid)
        plot_path = _maybe_plot(case_dir, item_type, target_grid, recon_grid, diff_grid)

        summary = {
            "file": record["file"],
            "coord_key": used_key,
            "index": record["index"],
            "max_error": record.get("max_error"),
            "boundary_max_error": record.get("boundary_max_error"),
            "endpoint_max_error": record.get("endpoint_max_error"),
            "mean_abs_error": record.get("mean_abs_error"),
            "rel_error_pct": record.get("rel_error_pct"),
            "bbox_size": record.get("bbox_size"),
            "npz_path": str(npz_path),
            "plot_path": str(plot_path) if plot_path else None,
        }
        with open(case_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"[{case_rank}/{len(records)}] saved {case_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect worst VQ-VAE face/edge cases")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--cases_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="inspect_reports")
    parser.add_argument("--num_cases", type=int, default=10)
    parser.add_argument("--coord_key", choices=["auto", "face_ctrs_wcs_norm", "face_ctrs"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    inspect_cases(parse_args())
