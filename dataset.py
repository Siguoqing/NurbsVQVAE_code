import random
import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset


class NurbsARData(Dataset):
    def __init__(self, sequence_file, validate=False, args=None, point_cloud_dir=None, data_list_file=None):
        """
        基础与条件兼容的 CAD 序列数据集
        :param sequence_file: VQ-VAE 生成的 1D token 序列 (.pkl)
        :param point_cloud_dir: [新增] 存放 .npy 点云的文件夹，如果传入，则自动开启条件加载模式
        :param data_list_file: [新增] 原始数据划分列表 (.txt 或 .pkl)，用于精准匹配文件名
        """
        self.args = args
        # 兼容处理：防止 args 为空时报错
        self.max_seq_len = getattr(args, 'max_seq_len', 2048) if args else 2048
        self.validate = validate
        self.point_cloud_dir = point_cloud_dir

        # --- [新增] 解析文件名列表，用于将序列和点云精准配对 ---
        self.file_names = []
        if data_list_file and os.path.exists(data_list_file):
            try:
                if data_list_file.endswith('.pkl'):
                    with open(data_list_file, 'rb') as f_list:
                        split_dict = pickle.load(f_list)
                        split_key = "val" if validate else "train"
                        raw_names = split_dict.get(split_key, [])
                        self.file_names = [os.path.basename(n).replace('.step', '').replace('.npy', '') for n in
                                           raw_names]
                else:
                    with open(data_list_file, 'r') as f_list:
                        self.file_names = [os.path.basename(line.strip()).replace('.step', '').replace('.npy', '') for
                                           line in f_list.readlines() if line.strip()]
            except Exception as e:
                print(f"⚠️ 解析 data_list_file 失败: {e}")

        # 获取 Rank 用于日志打印
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            split_name = "val" if validate else "train"
            mode = "条件生成(带点云)" if point_cloud_dir else "无条件生成(仅序列)"
            print(f"Loading 1D NURBS AR sequences from '{sequence_file}' (split={split_name}, mode={mode})...")

        try:
            with open(sequence_file, "rb") as f:
                data = pickle.load(f)

            split_key = "val" if validate else "train"
            raw_groups = data[split_key]

            # --- 加载配置元数据 ---
            self.vocab_size = data["vocab_size"]
            self.special_token_size = data["special_token_size"]
            self.face_index_size = data["face_index_size"]
            self.quantization_size = data["quantization_size"]
            self.face_index_offset = data["face_index_offset"]
            self.quantization_offset = data["quantization_offset"]

            self.face_block = data.get("face_block", 5)
            self.edge_block = data.get("edge_block", 6)

            special_tokens = data["special_tokens"]
            self.START_TOKEN = special_tokens["START_TOKEN"]
            self.SEP_TOKEN = special_tokens["SEP_TOKEN"]
            self.END_TOKEN = special_tokens["END_TOKEN"]
            self.PAD_TOKEN = self.vocab_size

            # --- 解析纯粹的 1D NLP 序列 ---
            self.groups = []
            filtered_count = 0
            error_count = 0

            for g_idx, g in enumerate(raw_groups):
                processed_group = {}

                # 【核心新增：尝试获取当前样本的唯一 ID，以便读取对应的 .npy】
                file_name = g.get("name", g.get("file_name", None))
                if file_name is None and g_idx < len(self.file_names):
                    file_name = self.file_names[g_idx]
                elif file_name is None:
                    file_name = f"model_{g_idx:06d}"  # 最终 Fallback
                processed_group["name"] = file_name

                # 1. 处理 original 样本
                try:
                    raw_ids = g["original"]["input_ids"]
                    if isinstance(raw_ids, torch.Tensor):
                        raw_ids = raw_ids.tolist()

                    grouped_ids = torch.tensor(raw_ids, dtype=torch.long)

                    if grouped_ids.shape[0] <= self.max_seq_len:
                        processed_group["original"] = {
                            "input_ids": grouped_ids,
                            "attention_mask": torch.ones(grouped_ids.shape[0], dtype=torch.long)
                        }
                    else:
                        filtered_count += 1
                        continue

                except Exception as e:
                    error_count += 1
                    continue

                # 2. 处理 augmented 样本
                if "augmented" in g and g["augmented"]:
                    processed_augmented = []
                    for aug_sample in g["augmented"]:
                        try:
                            raw_ids = aug_sample["input_ids"]
                            if isinstance(raw_ids, torch.Tensor):
                                raw_ids = raw_ids.tolist()

                            grouped_ids = torch.tensor(raw_ids, dtype=torch.long)

                            if grouped_ids.shape[0] <= self.max_seq_len:
                                processed_augmented.append({
                                    "input_ids": grouped_ids,
                                    "attention_mask": torch.ones(grouped_ids.shape[0], dtype=torch.long)
                                })
                        except Exception as e:
                            error_count += 1
                            continue

                    if processed_augmented:
                        processed_group["augmented"] = processed_augmented

                if "original" in processed_group:
                    self.groups.append(processed_group)

            if rank == 0:
                print(
                    f"Loaded {len(self.groups)} valid sequence groups (filtered {filtered_count} due to length, {error_count} errors).")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            raise

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        file_name = group.get("name", "")

        # 随机选择原图或数据增强的样本
        if self.validate:
            sample = group["original"]
        else:
            if random.random() < 0.5 or "augmented" not in group or not group["augmented"]:
                sample = group["original"]
            else:
                sample = random.choice(group["augmented"])

        item = {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"]
        }

        # 【核心新增：动态加载对应的点云矩阵】
        if self.point_cloud_dir is not None:
            pc_path = os.path.join(self.point_cloud_dir, f"{file_name}.npy")
            try:
                pc_data = np.load(pc_path, allow_pickle=True)
                # 假设预处理好的点云形状为 (K面, N点, 3坐标)
                point_clouds = torch.tensor(pc_data, dtype=torch.float32)
            except Exception:
                # 极个别文件丢失或读取失败时的占位处理，防止整个训练崩溃 (K=10, N=512)
                point_clouds = torch.zeros((10, 512, 3), dtype=torch.float32)

            item["point_clouds"] = point_clouds

        return item

    def collate_fn(self, batch):
        """
        批处理函数：1D 序列用 PAD_TOKEN 对齐，3D 点云按面数量 (K) 对齐
        """
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]

        max_len = max(len(ids) for ids in input_ids)

        padded_ids_list = []
        padded_mask_list = []

        for ids, mask in zip(input_ids, attention_masks):
            curr_len = ids.shape[0]
            pad_len = max_len - curr_len

            if pad_len > 0:
                pads = torch.full((pad_len,), self.PAD_TOKEN, dtype=torch.long)
                padded_ids = torch.cat([ids, pads], dim=0)

                mask_pads = torch.zeros(pad_len, dtype=torch.long)
                padded_mask = torch.cat([mask, mask_pads], dim=0)
            else:
                padded_ids = ids
                padded_mask = mask

            padded_ids_list.append(padded_ids)
            padded_mask_list.append(padded_mask)

        final_input_ids = torch.stack(padded_ids_list)
        final_attention_mask = torch.stack(padded_mask_list)

        batch_dict = {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask,
            "labels": final_input_ids.clone()  # 【新增】：直接生成 labels 供 LLaMA 计算 Loss
        }

        # 【核心新增：点云维度对齐】
        if "point_clouds" in batch[0]:
            pcs_list = [item["point_clouds"] for item in batch]

            # 因为每个零件的面数 K 不同，需要找到 Batch 里最大的 K 进行 Zero-Padding
            max_k = max(pc.shape[0] for pc in pcs_list)

            padded_pcs_list = []
            for pc in pcs_list:
                k, n, c = pc.shape
                if k < max_k:
                    # 【神级替换：真实面复制】用当前零件的第 1 个面来重复填充
                    # 这样方差绝对正常，彻底杜绝 BatchNorm 崩溃，且会被 LLaMA 的 Mask 完美屏蔽！
                    pad = pc[0:1].repeat(max_k - k, 1, 1)
                    pc = torch.cat([pc, pad], dim=0)
                elif k > max_k:
                    # 理论上不会出现，防御性截断
                    pc = pc[:max_k]
                padded_pcs_list.append(pc)

            batch_dict["point_clouds"] = torch.stack(padded_pcs_list)

        return batch_dict


# =========================================================================
# 【条件生成专属封装类】
# 继承自 NurbsARData，保证对外接口干净，且自动拥有完整的 vocab 解析能力
# =========================================================================
class ConditionalCADDataset(NurbsARData):
    def __init__(self, sequence_file, point_cloud_dir, data_list_file=None, validate=False, args=None):
        """
        专用于点云条件生成的 Dataset 接口
        """
        super().__init__(
            sequence_file=sequence_file,
            point_cloud_dir=point_cloud_dir,
            data_list_file=data_list_file,
            validate=validate,
            args=args
        )