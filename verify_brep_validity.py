import os
import torch
import pickle
import numpy as np
import glob

# 导入师兄的原版 VQModel 架构 (必须与 trainer_nurbs.py 保持完全一致)
from trainer_nurbs import VQVAE


def get_boundaries(grid):
    """提取 4x4 控制点网格的 4 条边界曲线 (每条线由 4 个控制点组成)"""
    b1 = grid[0, :, :]  # 上边界
    b2 = grid[3, :, :]  # 下边界
    b3 = grid[:, 0, :]  # 左边界
    b4 = grid[:, 3, :]  # 右边界
    return [b1, b2, b3, b4]


def verify_brep_validity():
    print("=" * 70)
    print("🔬 正在启动 B-rep 拓扑有效性 (Watertight) 检测 [Diffusers 架构版]...")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 1. 严格按照 trainer_nurbs.py 的参数初始化模型
    # ==========================================
    model = VQVAE(
        in_channels=4,  # use_type_flag=True
        out_channels=3,
        down_block_types=['DownEncoderBlock2D'] * 2,
        up_block_types=['UpDecoderBlock2D'] * 2,
        block_out_channels=[64, 128],
        layers_per_block=2,
        act_fn='silu',
        latent_channels=128,
        vq_embed_dim=3,  # 核心：3维极致量化
        num_vq_embeddings=4096,  # 核心：4096大词表
        norm_num_groups=32,
        sample_size=4
    ).to(device)

    # 加载最佳权重
    ckpt_path = "checkpoints_vqvae/deepcad_nurbs_vqvae_best.pt"

    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到权重文件: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # 清理多卡训练可能带来的前缀
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()
    print(f"✅ 成功加载最佳权重: {ckpt_path}")

    # ==========================================
    # 2. 随机加载一个面数 >= 2 的 CAD 文件
    # ==========================================
    data_dir = '/mnt/docker_dir/lijiahao/NurbsVQVAE_code/furniture_parsed/ABC_Dataset_NEW'
    all_pkls = glob.glob(os.path.join(data_dir, '**/*.pkl'), recursive=True)

    faces = []
    test_pkl = None
    for _ in range(50):
        test_pkl = np.random.choice(all_pkls)
        with open(test_pkl, 'rb') as f:
            cad = pickle.load(f)
        faces = np.array(cad.get('face_ctrs', [])).reshape(-1, 16, 3)
        if len(faces) >= 2:
            break

    if len(faces) < 2:
        print("⚠️ 未能随机到包含多个面的实体，请重新运行脚本。")
        return

    print(f"📂 测试文件: {test_pkl}")
    print(f"📊 包含 {len(faces)} 个面")

    # ==========================================
    # 阶段 A：提取原始拓扑连接关系 (寻找重合边)
    # ==========================================
    orig_faces_grid = faces.reshape(-1, 4, 4, 3)
    original_boundaries = []

    for i, face in enumerate(orig_faces_grid):
        bounds = get_boundaries(face)
        for j, b in enumerate(bounds):
            original_boundaries.append((i, j, b))

    connections = []
    for idx1 in range(len(original_boundaries)):
        for idx2 in range(idx1 + 1, len(original_boundaries)):
            f1, b1, curve1 = original_boundaries[idx1]
            f2, b2, curve2 = original_boundaries[idx2]
            if f1 == f2: continue

            dist_fwd = np.max(np.linalg.norm(curve1 - curve2, axis=-1))
            dist_rev = np.max(np.linalg.norm(curve1 - curve2[::-1], axis=-1))

            if min(dist_fwd, dist_rev) < 1e-4:
                connections.append((f1, b1, f2, b2, dist_fwd < dist_rev))

    if not connections:
        print("⚠️ 该模型似乎没有共享边界。请重新运行脚本。")
        return

    print(f"🔗 在原始 CAD 中检测到 {len(connections)} 处拓扑缝合连接。")

    # ==========================================
    # 阶段 B：经过 VQ-VAE 压缩并还原 (走 Diffusers 流程)
    # ==========================================
    flags = np.zeros((len(faces), 4, 4, 1), dtype=np.float32)
    face_input = np.concatenate([orig_faces_grid, flags], axis=-1)
    # (B, H, W, C) -> (B, C, H, W)
    x_in = torch.FloatTensor(face_input).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        h = model.encoder(x_in)
        h = model.quant_conv(h)
        quant_out, _, _ = model.quantize(h)
        x_recon = model.decoder(model.post_quant_conv(quant_out))

    # 转回 numpy: (B, 3, 4, 4) -> (B, 4, 4, 3)
    recon_faces_grid = x_recon.permute(0, 2, 3, 1).cpu().numpy()

    # ==========================================
    # 阶段 C：计算“拓扑撕裂误差” (Crack Error)
    # ==========================================
    crack_errors = []
    for f1, b1, f2, b2, is_fwd in connections:
        r_curve1 = get_boundaries(recon_faces_grid[f1])[b1]
        r_curve2 = get_boundaries(recon_faces_grid[f2])[b2]

        if is_fwd:
            gap = np.max(np.linalg.norm(r_curve1 - r_curve2, axis=-1))
        else:
            gap = np.max(np.linalg.norm(r_curve1 - r_curve2[::-1], axis=-1))
        crack_errors.append(gap)

    max_crack = max(crack_errors)
    mean_crack = np.mean(crack_errors)

    # 工业缝合标准 (Crack < 0.005)
    valid_ratio = sum(1 for e in crack_errors if e <= 0.005) / len(crack_errors) * 100

    print("\n🟩 【B-rep 还原后拓扑有效性报告】")
    print(f"  🌊 平均撕裂缝隙 (Mean Crack): {mean_crack:.6f}")
    print(f"  🔥 最大撕裂缝隙 (Max Crack):  {max_crack:.6f}")
    print(f"  ✅ 工业有效性比例 (Crack < 0.005): {valid_ratio:.2f}% 的缝合边依然保持有效贴合")

    if valid_ratio > 90:
        print("\n🏆 结论: 极好！模型不仅形状还原度高，且极大地保留了拓扑水密性！")
    else:
        print(
            "\n⚠️ 结论: 警告！正如师兄所料，独立量化在 3 维瓶颈下必然会导致部分边界撕裂。这需要在后续 LLaMA 生成时用代码强行求均值缝合。")


if __name__ == '__main__':
    verify_brep_validity()