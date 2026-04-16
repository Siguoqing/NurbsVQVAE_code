import os
import numpy as np
import torch
import pickle
import argparse
from tqdm import tqdm
import random

# 注意这里：如果是重命名后的 utils，请确保能正确导入 rotate_axis
from utils_nurbs import rotate_axis
from trainer_nurbs import VQVAE  # 导入我们精简好的纯净神经网络


def dfs_face_ordering_from_core(edge_face_pairs, num_faces):
    """
    面的 DFS 排序：从连接数最多的核心面开始遍历
    """
    nbrs = [set() for _ in range(num_faces)]
    for f1, f2 in edge_face_pairs:
        if 0 <= f1 < num_faces and 0 <= f2 < num_faces and f1 != f2:
            nbrs[f1].add(f2);
            nbrs[f2].add(f1)
    deg = [len(n) for n in nbrs]

    visited = [False] * num_faces
    face_order = []

    seeds = sorted(range(num_faces), key=lambda x: (-deg[x], x))

    def dfs(u):
        visited[u] = True
        face_order.append(u)
        unvisited_neighbors = [v for v in nbrs[u] if not visited[v]]
        unvisited_neighbors.sort(key=lambda x: (deg[x], x))
        for v in unvisited_neighbors:
            if not visited[v]:
                dfs(v)

    for s in seeds:
        if not visited[s]:
            dfs(s)

    face_position_map = {f: i for i, f in enumerate(face_order)}
    return face_order, face_position_map


def lexicographic_edge_ordering(edge_face_pairs):
    """
    边的字典序排序
    """
    edge_sort_info = []
    for eidx, pair in enumerate(edge_face_pairs):
        if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
            continue
        f1, f2 = pair[0], pair[1]
        max_idx = max(f1, f2)
        min_idx = min(f1, f2)
        sort_key = (max_idx, min_idx)
        edge_sort_info.append((sort_key, eidx, pair))

    edge_sort_info.sort()
    edge_order = [item[1] for item in edge_sort_info]
    ordered_edge_face_pairs = [item[2] for item in edge_sort_info]

    return edge_order, ordered_edge_face_pairs


# ==============================
#    加载神经网络的超级转换器
# ==============================
class NurbsARDataPreprocessor:
    def __init__(self, data_list, args):
        self.data_list = data_list
        self.args = args

        # 1. 读取数据列表
        with open(data_list, 'rb') as f:
            ds = pickle.load(f)
        self.train_paths = ds['train']
        self.val_paths = ds.get('val', [])

        # 如果没有划分 val，自动切分 5% 给验证集
        if not self.val_paths:
            split_idx = int(len(self.train_paths) * 0.95)
            self.val_paths = self.train_paths[split_idx:]
            self.train_paths = self.train_paths[:split_idx]

        # 2. Vocabulary / offsets 配置
        self.face_index_size = args.max_face if hasattr(args, 'max_face') else 50
        self.quantization_size = 1024  # 我们专属的 VQ-VAE 密码本大小
        self.special_token_size = 3

        # 【核心修正】统一采用 AI 输出的真实 Token 数量 (4个特征词汇)
        self.face_block = 5  # 4个face_tokens + 1个面索引 = 5
        self.edge_block = 6  # 2个面索引 + 4个edge_tokens = 6

        self.face_index_offset = 0
        self.quantization_offset = self.face_index_offset + self.face_index_size

        self.vocab_size = (self.face_index_size + self.quantization_size + self.special_token_size)
        special_token_offset = self.quantization_offset + self.quantization_size
        self.START_TOKEN = special_token_offset
        self.SEP_TOKEN = special_token_offset + 1
        self.END_TOKEN = special_token_offset + 2

        # ---------------------------------------------------------
        # 3. 初始化并加载我们炼好的 VQ-VAE 终极模型！
        # ---------------------------------------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VQVAE(
            in_channels=4, out_channels=3,
            down_block_types=['DownEncoderBlock2D'] * 2,
            up_block_types=['UpDecoderBlock2D'] * 2,
            block_out_channels=[64, 128], layers_per_block=2,
            act_fn='silu', latent_channels=128,
            vq_embed_dim=3, num_vq_embeddings=1024,
            norm_num_groups=32, sample_size=4
        ).to(self.device)

        # ⚠️ 请确保此路径指向你刚才炼出来的那个 _best.pt 文件
        ckpt_path = "/mnt/docker_dir/lijiahao/NurbsVQVAE_code/checkpoint/se/abc/8192,4096,128,64,false,1e-4,0,p/deepcad_nurbs_vqvae_best.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到权重文件：{ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        print("✅ 预处理器：VQ-VAE 密码本加载完成！")

        self.group_cache = []
        self._process_all_data()

    def _process_all_data(self):
        max_cad_count = getattr(self.args, 'max_cad_count', None)

        train_count = 0
        for path in tqdm(self.train_paths, desc="Processing train"):
            if max_cad_count is not None and train_count >= max_cad_count:
                break
            cad_group = self._process_single_cad(path, 'train')
            if cad_group:
                self.group_cache.append(('train', cad_group))
                train_count += 1

        val_count = 0
        for path in tqdm(self.val_paths, desc="Processing val"):
            if max_cad_count is not None and val_count >= max_cad_count:
                break
            cad_group = self._process_single_cad(path, 'val')
            if cad_group:
                self.group_cache.append(('val', cad_group))
                val_count += 1

    def _encode_single_cad(self, face_ctrs, edge_ctrs, edgeFace_adj):
        num_face = len(face_ctrs)
        num_edge = len(edge_ctrs)

        # 构建并排序拓扑
        edge_face_pairs = []
        if len(edgeFace_adj) > 0:
            for edge_adj in edgeFace_adj:
                if len(edge_adj) >= 2:
                    face1_idx, face2_idx = edge_adj[0], edge_adj[1]
                    edge_face_pairs.append((face1_idx, face2_idx))

        face_order, face_position_map = dfs_face_ordering_from_core(edge_face_pairs, num_face)
        face_ctrs = face_ctrs[face_order]

        updated_edge_face_pairs = []
        for f1, f2 in edge_face_pairs:
            new_f1 = face_position_map.get(f1, f1)
            new_f2 = face_position_map.get(f2, f2)
            updated_edge_face_pairs.append((new_f1, new_f2))
        edge_face_pairs = updated_edge_face_pairs

        edge_order, ordered_edge_face_pairs = lexicographic_edge_ordering(edge_face_pairs)
        edge_ctrs = edge_ctrs[edge_order]

        N = self.face_index_size
        r = random.randint(0, N - 1) if N > 0 else 0
        face_index_map = {i: (i + r) % N for i in range(num_face)} if N > 0 else {i: i for i in range(num_face)}

        # ---------------------------------------------------------
        # 【核心修正】批量送入神经网络获取真实压缩 Token
        # ---------------------------------------------------------
        face_tokens_list = []
        if num_face > 0:
            face_grids = face_ctrs.reshape(num_face, 4, 4, 3)
            face_flags = np.zeros((num_face, 4, 4, 1), dtype=np.float32)
            face_inputs = np.concatenate([face_grids, face_flags], axis=-1)
            x_in = torch.FloatTensor(face_inputs).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                h = self.model.encoder(x_in)
                h = self.model.quant_conv(h)
                _, _, indices = self.model.quantize(h)
                # 获取属于这些面的 4 个整数词汇
                face_tokens_list = indices[2].reshape(num_face, 4).cpu().tolist()

        ordered_edge_tokens_list = []
        if num_edge > 0:
            edge_grids = np.tile(edge_ctrs, (1, 4, 1)).reshape(num_edge, 4, 4, 3)
            edge_flags = np.ones((num_edge, 4, 4, 1), dtype=np.float32)
            edge_inputs = np.concatenate([edge_grids, edge_flags], axis=-1)
            x_in = torch.FloatTensor(edge_inputs).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                h = self.model.encoder(x_in)
                h = self.model.quant_conv(h)
                _, _, indices = self.model.quantize(h)
                # 获取属于这些边的 4 个整数词汇
                ordered_edge_tokens_list = indices[2].reshape(num_edge, 4).cpu().tolist()

        # 构建一维自回归文本序列
        tokens, attention_mask = [], []
        tokens.append(self.START_TOKEN)
        attention_mask.append(1)

        # --- Faces 部分 ---
        for i in range(num_face):
            if i < len(face_tokens_list):
                for face_token in face_tokens_list[i]:
                    tokens.append(self.quantization_offset + int(face_token))
                    attention_mask.append(1)
            tokens.append(self.face_index_offset + face_index_map[i])
            attention_mask.append(1)

        tokens.append(self.SEP_TOKEN)
        attention_mask.append(1)

        # --- Edges 部分 ---
        for k, (face_pair) in enumerate(ordered_edge_face_pairs):
            src, dst = face_pair
            tokens.append(self.face_index_offset + face_index_map[src])
            attention_mask.append(1)
            tokens.append(self.face_index_offset + face_index_map[dst])
            attention_mask.append(1)

            if k < len(ordered_edge_tokens_list):
                for edge_token in ordered_edge_tokens_list[k]:
                    tokens.append(self.quantization_offset + int(edge_token))
                    attention_mask.append(1)

        tokens.append(self.END_TOKEN)
        attention_mask.append(1)

        return tokens, attention_mask

    def _process_single_cad(self, path, split='train'):
        try:
            with open(path, 'rb') as f:
                cad = pickle.load(f)

            face_ctrs = np.array(cad.get('face_ctrs'), dtype=np.float32)
            edge_ctrs = np.array(cad.get('edge_ctrs'), dtype=np.float32)
            edgeFace_adj = cad.get('edgeFace_adj', [])

            # 过滤脏数据，防止污染语言模型
            if face_ctrs.size == 0 or edge_ctrs.size == 0: return None
            if np.isnan(face_ctrs).any() or np.isnan(edge_ctrs).any(): return None

            max_face = self.args.max_face if hasattr(self.args, 'max_face') else 50
            max_edge = self.args.max_edge if hasattr(self.args, 'max_edge') else 124
            if len(face_ctrs) > max_face or len(edge_ctrs) > max_edge: return None

            rotation_angles = [0, 90, 180, 270] if (
                        split == 'train' and hasattr(self.args, 'aug') and self.args.aug) else [0]

            if split == 'train':
                group = {'original': None, 'augmented': []}
                for rot in rotation_angles:
                    current_face_ctrs = face_ctrs.copy()
                    current_edge_ctrs = edge_ctrs.copy()
                    if rot != 0:
                        current_face_ctrs = rotate_axis(current_face_ctrs, rot, 'z', normalized=False)
                        current_edge_ctrs = rotate_axis(current_edge_ctrs, rot, 'z', normalized=False)

                    tokens, attn = self._encode_single_cad(current_face_ctrs, current_edge_ctrs, edgeFace_adj)
                    item = {'input_ids': tokens, 'attention_mask': attn}
                    if rot == 0:
                        group['original'] = item
                    else:
                        group['augmented'].append(item)
                if group['original'] is None: return None
                return group
            else:
                tokens, attn = self._encode_single_cad(face_ctrs, edge_ctrs, edgeFace_adj)
                item = {'input_ids': tokens, 'attention_mask': attn}
                group = {'original': item}
                return group

        except Exception as e:
            return None


def main():
    parser = argparse.ArgumentParser()
    # 强制将读取目录指向包含了 14 万个有效文件的名单！
    parser.add_argument('--data_list', type=str, default='process_list.pkl', help='Path to pkl with train paths')
    parser.add_argument('--output_file', type=str, default='data/deepcad_nurbs_sequences_vqvae.pkl',
                        help='Output pickle file')
    parser.add_argument('--max_face', type=int, default=50)
    parser.add_argument('--max_edge', type=int, default=124)
    parser.add_argument('--aug', default=True, type=bool, help='Whether to save rotation augmentation')

    # 设定为 None，表示全火力跑完所有数据！
    parser.add_argument('--max_cad_count', type=int, default=None)

    args = parser.parse_args()

    processor = NurbsARDataPreprocessor(args.data_list, args)

    train_groups, val_groups = [], []
    for split, group in processor.group_cache:
        if split == 'train':
            train_groups.append(group)
        elif split == 'val':
            val_groups.append(group)

    output_data = {
        'train': train_groups,
        'val': val_groups,
        'vocab_size': processor.vocab_size,
        'special_token_size': processor.special_token_size,
        'face_index_size': processor.face_index_size,
        'quantization_size': processor.quantization_size,
        'face_index_offset': processor.face_index_offset,
        'quantization_offset': processor.quantization_offset,
        'face_block': processor.face_block,
        'edge_block': processor.edge_block,
        'special_tokens': {
            'START_TOKEN': processor.START_TOKEN,
            'SEP_TOKEN': processor.SEP_TOKEN,
            'END_TOKEN': processor.END_TOKEN,
        }
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n🎉 [大功告成] 所有序列已保存 -> {args.output_file}")
    print(f"  🔥 训练集: {len(train_groups)} 个模型 | 验证集: {len(val_groups)} 个模型")


if __name__ == "__main__":
    main()