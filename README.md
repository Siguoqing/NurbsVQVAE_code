# NURBS VQ-VAE + 点云条件 CAD 生成

本仓库当前用于研究基于 NURBS VQ-VAE token 的点云条件 CAD 生成。

当前主线流程是：

```text
STEP / parsed NURBS
-> NURBS VQ-VAE primitive tokenizer
-> v2 bbox-aware AR sequence
-> point-cloud-conditioned AR model
-> generated STEP / STL
```

## 当前阶段结论

- VQ-VAE 已训练完成，并且 face / edge NURBS 控制点重建可用。
- 原始点云大部分不可用，已从 STEP 文件重新采样生成点云。
- 旧 AR token 协议已替换为带 bbox 的 v2 协议。
- GT v2 序列可以较高比例重建为 valid BREP。
- 点云条件 AR 已经可以稳定 DDP 训练。
- 当前模型已经可以从点云条件生成 valid BREP STEP 文件。

当前阶段关键结果：

```text
VQ-VAE -> BREP validity，全量 val:
  evaluated_files: 668
  eligible_files: 426
  solid_returned: 420
  brep_valid: 385
  eligible_valid_rate: 0.9038

GT v2 sequence -> BREP，val 前 50:
  solid_returned: 49 / 50
  brep_valid: 46 / 50
  valid_rate: 0.92

Point2CAD 普通采样 baseline，val414:
  saved STEP/STL: 232 / 414
  valid BREP: 195 / 414
  valid among saved: 195 / 232

Point2CAD constrained decoding，val414:
  saved STEP/STL: 227 / 414
  valid BREP: 203 / 414
  valid among saved: 203 / 227
```

## 数据目录

当前服务器实验使用的数据路径：

```text
/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/ABC_Dataset_NEW
/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/ABC_Dataset_NEW_parsed
```

原始 CAD 模型目录结构：

```text
model_XXXXX/
  *.step
  point_cloud.npy
  face_labels.npy
```

解析后的 NURBS `.pkl` 文件位于：

```text
data/ABC_Dataset_NEW_parsed
```

当前全量 parsed + clean point-cloud split：

```text
data/furniture_data_split_6bit_allparsed_pc_clean_split.pkl
```

当前 v2 AR 序列文件：

```text
data/furniture_nurbs_sequences_allparsed_pc_clean_v2.pkl
```

## Token 协议

旧协议只保存 VQ token 和 face index：

```text
face block = 4 个 VQ token + 1 个 face index
edge block = 2 个 face index + 4 个 VQ token
```

这个协议缺少 bbox / WCS 位置信息，导致即使 GT token 序列也难以稳定重建 BREP。

当前使用 v2 协议：

```text
Face block = 6 个 bbox token + 4 个 face VQ token + 1 个 face index
Edge block = 2 个 face index + 6 个 bbox token + 4 个 edge VQ token
```

词表布局：

```text
face index: 0 ~ 49
VQ token:   50 ~ 1073
bbox token: 1074 ~ 3121
START:      3122
SEP:        3123
END:        3124
PAD:        3125
```

当前 `config.json` 应与上述布局保持一致。

## 点云重采样

原始点云大部分为无效值，因此使用 STEP 文件重新采样：

```bash
python resample_pointclouds_from_step.py \
  --root /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/ABC_Dataset_NEW \
  --num_points 4096 \
  --overwrite \
  --timeout 120 \
  --report_json /mnt/docker_dir/lijiahao/NurbsVQVAE_code/resample_full_report.json
```

重采样结果：

```text
processed: 18449
ok: 18395
failed: 54
```

点云质量检查：

```bash
python inspect_pointcloud_quality.py \
  --root /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/ABC_Dataset_NEW \
  --output_dir /mnt/docker_dir/lijiahao/NurbsVQVAE_code/pointcloud_quality_report_full
```

质量检查结果：

```text
total_files: 18449
ok: 18406
all_invalid: 43
avg_valid_ratio: 0.9977
```

## 构建 clean split

从所有 parsed NURBS 文件中筛选有可用点云的样本：

```bash
python build_full_parsed_pointcloud_split.py \
  --parsed_dir /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/ABC_Dataset_NEW_parsed \
  --records_json /mnt/docker_dir/lijiahao/NurbsVQVAE_code/pointcloud_quality_report_full/pointcloud_records.json \
  --output_split_file /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_data_split_6bit_allparsed_pc_clean_split.pkl \
  --output_stats_json /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_data_split_6bit_allparsed_pc_clean_stats.json \
  --val_ratio 0.05
```

结果：

```text
parsed_total: 13373
matched: 13363
train: 12695
val: 668
```

## 构建 v2 AR 序列

生成带 bbox 的 v2 AR 序列：

```bash
python 2sequence_nurbs_v2.py \
  --data_list /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_data_split_6bit_allparsed_pc_clean_split.pkl \
  --records_json /mnt/docker_dir/lijiahao/NurbsVQVAE_code/pointcloud_quality_report_full/pointcloud_records.json \
  --pointcloud_root /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/ABC_Dataset_NEW \
  --ckpt_path /mnt/docker_dir/lijiahao/NurbsVQVAE_code/checkpoint/se/abc/8192,4096,128,64,false,1e-4,0,p/deepcad_nurbs_vqvae_best.pt \
  --output_file /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_nurbs_sequences_allparsed_pc_clean_v2.pkl \
  --device cuda:0
```

结果：

```text
train: 7594
val: 414
bad_shape: 4998
too_large: 357
encode_failed: 0
missing_model_id: 0
```

说明：

- `bad_shape` 主要是 fitted NURBS 控制点中存在 NaN / Inf。
- `too_large` 表示超过当前 `max_face=50` 或 `max_edge=124`。
- `encode_failed=0` 说明 VQ-VAE 编码链路正常。
- `missing_model_id=0` 说明 parsed NURBS 与点云目录匹配正常。

## 验证 v2 序列可重建性

训练 AR 前，先验证 GT v2 序列能否重建 BREP：

```bash
python debug_reconstruct_sequence_v2.py \
  --sequence_file /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_nurbs_sequences_allparsed_pc_clean_v2.pkl \
  --ckpt_path /mnt/docker_dir/lijiahao/NurbsVQVAE_code/checkpoint/se/abc/8192,4096,128,64,false,1e-4,0,p/deepcad_nurbs_vqvae_best.pt \
  --split val \
  --index 0 \
  --max_samples 50 \
  --output_dir /mnt/docker_dir/lijiahao/NurbsVQVAE_code/result/debug_reconstruct_sequence_v2_val50 \
  --gpu 0
```

当前结果：

```text
evaluated: 50
solid_returned: 49
brep_valid: 46
valid_rate: 0.92
```

这说明 v2 token 协议本身是可重建的，旧协议失败主要是因为缺少 bbox / WCS 信息。

## 训练点云条件 AR

当前稳定训练配置：

```bash
torchrun --nproc_per_node=2 train_ar.py \
  --sequence_file /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_nurbs_sequences_allparsed_pc_clean_v2.pkl \
  --point_cloud_dir /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/ABC_Dataset_NEW \
  --data_list_file /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_data_split_6bit_allparsed_pc_clean_split.pkl \
  --save_dir /mnt/docker_dir/lijiahao/NurbsVQVAE_code/checkpoints/ar_pc_allparsed_clean_v2_bbox_ddp_seq2048_bs2 \
  --tb_log_dir /mnt/docker_dir/lijiahao/NurbsVQVAE_code/logs/ar_pc_allparsed_clean_v2_bbox_ddp_seq2048_bs2 \
  --max_seq_len 2048 \
  --batch_size 2 \
  --train_nepoch 1000 \
  --test_nepoch 1 \
  --save_nepoch 20 \
  --learning_rate 1e-4 \
  --point_cloud_npoints 2048 \
  --point_prefix_tokens 8 \
  --length_bucket_mult 2
```

训练说明：

- `max_seq_len=2048` 可以吃满当前 v2 序列。
- 每卡 `batch_size=4` 在长序列下不稳定。
- 每卡 `batch_size=2` 配合 length bucket 可以稳定训练。
- `LengthBucketBatchSampler` 会把相近长度的序列组成 batch，减少 padding 浪费。

当前训练大约收敛到：

```text
Validation CE: 0.63 ~ 0.65
Perplexity:    1.89 ~ 1.91
```

注意：CE / PPL 与最终 BREP valid rate 不完全一致。后续 checkpoint 选择应加入生成 valid rate，而不是只看 CE。

## 条件生成

创建 val 点云评估目录：

```bash
python - <<'PY'
import os, pickle

seq_file = "/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/furniture_nurbs_sequences_allparsed_pc_clean_v2.pkl"
pc_root = "/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/ABC_Dataset_NEW"
out_dir = "/mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/pc_val_v2"

os.makedirs(out_dir, exist_ok=True)

with open(seq_file, "rb") as f:
    data = pickle.load(f)

for group in data["val"]:
    name = group.get("name") or group.get("file_name")
    if not name:
        continue
    src = os.path.join(pc_root, name)
    dst = os.path.join(out_dir, name)
    if os.path.isdir(src):
        if os.path.lexists(dst):
            os.unlink(dst)
        os.symlink(src, dst)
PY
```

运行 constrained v2 条件生成：

```bash
python generate_cond.py \
  --ar_model /mnt/docker_dir/lijiahao/NurbsVQVAE_code/checkpoints/ar_pc_allparsed_clean_v2_bbox_ddp_seq2048_bs2/deepcad_ar_point_best_model.pt \
  --config /mnt/docker_dir/lijiahao/NurbsVQVAE_code/config.json \
  --vqvae_ckpt /mnt/docker_dir/lijiahao/NurbsVQVAE_code/checkpoint/se/abc/8192,4096,128,64,false,1e-4,0,p/deepcad_nurbs_vqvae_best.pt \
  --test_pc_dir /mnt/docker_dir/lijiahao/NurbsVQVAE_code/data/pc_val_v2 \
  --output_dir /mnt/docker_dir/lijiahao/NurbsVQVAE_code/result/generated_cad_cond_v2_best_val_constrained_t07_final \
  --gpu 0 \
  --point_cloud_npoints 2048 \
  --max_length 2048 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 50 \
  --timeout 60 \
  --constrained_decoding \
  --min_faces 2 \
  --max_faces 50
```

当前 val414 constrained generation 结果：

```text
总测试点云:     414
成功保存文件:   227
BREP 有效:      203
BREP 无效:      24
保存失败:       187
```

## Constrained Decoding

`generate_cond.py` 支持 v2 grammar-constrained decoding：

```bash
--constrained_decoding
```

它会根据当前生成位置限制 token 类型：

```text
Face block:
  bbox bbox bbox bbox bbox bbox
  vq vq vq vq
  face_index

Edge block:
  face_index face_index
  bbox bbox bbox bbox bbox bbox
  vq vq vq vq
```

同时 edge 的 face index 只能引用已经生成过的 face id。

这个约束主要用于消除以下格式错误：

```text
Bad face bbox token
Bad face VQ token
Bad edge bbox token
Bad edge VQ token
SEP / END 出现在 block 中间
edge 引用不存在的 face
```

constrained decoding 不能完全解决几何拓扑问题。当前剩余失败主要是：

```text
BRepBuilderAPI_MakeWire
BRepBuilderAPI_MakeSolid_Add
zero-size array
```

这些属于 edge loop 闭合、wire 顺序、shell 构造等 BREP 拓扑/几何问题。

## 重要脚本

- `resample_pointclouds_from_step.py`：从 STEP 文件重新采样点云。
- `inspect_pointcloud_quality.py`：检查点云有效性。
- `build_full_parsed_pointcloud_split.py`：构建 parsed NURBS + point cloud clean split。
- `2sequence_nurbs_v2.py`：构建带 bbox 的 v2 AR 序列。
- `debug_reconstruct_sequence_v2.py`：从 GT v2 序列重建 BREP。
- `eval_vqvae_brep_validity.py`：评估 VQ-VAE 解码结果的 BREP valid rate。
- `train_ar.py`：训练 AR / 点云条件 AR。
- `generate_cond.py`：点云条件生成并导出 STEP / STL。
- `pointcloud_condition.py`：点云读取、归一化、采样。

## 当前已知问题

- 部分 parsed PKL 的 `face_ctrs` / `edge_ctrs` 含 NaN / Inf，会被过滤。
- 超过 `max_face=50` 或 `max_edge=124` 的样本暂时被过滤。
- CE / PPL 不能完全代表生成 valid rate。
- 当前模型对复杂拓扑仍不稳定，保存失败仍较多。
- constrained decoding 只保证 token grammar，不保证 edge loop 闭合或 solid 水密。

## 后续建议

- 对 `MakeWire` 失败样本做 edge loop / wire 顺序后处理。
- 增加固定小规模生成验证集，用 valid rate 选择 checkpoint。
- 尝试基于 BREP validity 的 GRPO / reward fine-tuning。
- 在 decoding 阶段加入更强的拓扑约束。
- 分析成功/失败样本的 face count、edge count、sequence length 和几何类型。
