# NURBS VQ-VAE

这个仓库包含当前使用的 NURBS VQ-VAE 训练与验证流程，输入对象为 4x4 NURBS face 控制点网格和 edge 控制点。

## 环境说明

- 默认在 Linux + CUDA 环境下训练和验证。
- 当前脚本里的默认数据目录为：

```bash
/mnt/docker_dir/lijiahao/NurbsVQVAE_code/furniture_parsed/ABC_Dataset_NEW
```

## 训练

### 双卡训练

在 2 张 GPU 上启动训练：

```bash
torchrun --nproc_per_node=2 main_nurbs.py
```

### 小规模 smoke test

正式训练前，可以先限制文件数快速检查流程是否正常：

```bash
torchrun --nproc_per_node=2 main_nurbs.py --max_files 2000
```

### 单卡快速启动检查

如果只是想确认脚本能正常启动：

```bash
python main_nurbs.py --max_files 2000 --batch_size 512
```

## 验证

### 重建效果验证

在验证集上评估 VQ-VAE 的 face 和 edge 重建效果：

```bash
python verify_vqvae.py \
  --ckpt_path checkpoints_vqvae/restart_baseline/deepcad_nurbs_vqvae_best.pt \
  --split val \
  --max_face_items 5000 \
  --max_edge_items 5000 \
  --output_dir verify_reports \
  --top_k 100
```

这条命令会输出：

- face 重建指标
- edge 重建指标
- `verify_reports/worst_face_cases.json`
- `verify_reports/worst_edge_cases.json`

### 快速验证 smoke test

```bash
python verify_vqvae.py \
  --ckpt_path checkpoints_vqvae/restart_baseline/deepcad_nurbs_vqvae_best.pt \
  --split val \
  --max_face_items 500 \
  --max_edge_items 500
```

## Worst Case 检查

导出最差 face 样本的详细重建结果：

```bash
python inspect_vqvae_cases.py \
  --ckpt_path checkpoints_vqvae/restart_baseline/deepcad_nurbs_vqvae_best.pt \
  --cases_json verify_reports/worst_face_cases.json \
  --output_dir inspect_reports/worst_faces \
  --num_cases 10
```

导出最差 edge 样本的详细重建结果：

```bash
python inspect_vqvae_cases.py \
  --ckpt_path checkpoints_vqvae/restart_baseline/deepcad_nurbs_vqvae_best.pt \
  --cases_json verify_reports/worst_edge_cases.json \
  --output_dir inspect_reports/worst_edges \
  --num_cases 10
```

每个 case 目录下通常会包含：

- `summary.json`
- `case_data.npz`
- `case_plot.png`，如果环境中安装了 `matplotlib`
