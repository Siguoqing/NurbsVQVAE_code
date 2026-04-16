import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# 导入咱们刚改好的模型
from model import LLaMA3ARModel, LLaMA3Config


# ==========================================
# 1. 构造“假数据引擎” (模拟未来真实的数据流)
# ==========================================
class DummyConditionalCADDataset(Dataset):
    def __init__(self, num_samples=100, max_faces=12, points_per_face=512, seq_len=256):
        """
        模拟真实数据集:
        - max_faces (K): 一个零件最多 12 个面
        - points_per_face (N): 每个面采样 512 个点
        - seq_len: 大模型生成的 CAD 序列长度为 256
        """
        self.num_samples = num_samples
        self.max_faces = max_faces
        self.points_per_face = points_per_face
        self.seq_len = seq_len
        self.vocab_size = 4054

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. 伪造的面级别点云: (K, N, 3)
        pcs = torch.randn(self.max_faces, self.points_per_face, 3)

        # 2. 伪造的 CAD 离散序列: (Seq_Len)
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))

        # 3. 标签 (自回归训练中，labels 和 input_ids 是一样的，模型内部会做 shift)
        labels = ids.clone()

        return {
            "point_clouds": pcs,
            "input_ids": ids,
            "labels": labels
        }


# ==========================================
# 2. 初始化环境与模型
# ==========================================
def main():
    print("🚀 正在初始化 4090 训练环境...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")

    # 初始化配置和模型 (这里用轻量级配置测试)
    config = LLaMA3Config(d_model=256, n_layers=4, n_heads=8)
    model = LLaMA3ARModel(config).to(device)

    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 准备 DataLoader (Batch Size 设为 16 探探底)
    dataset = DummyConditionalCADDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print("\n🔥 开始模拟训练回路 (Epoch 1)...")
    model.train()

    for step, batch in enumerate(dataloader):
        # 将数据推入显存
        point_clouds = batch["point_clouds"].to(device)  # (B, K, N, 3)
        input_ids = batch["input_ids"].to(device)  # (B, Seq_Len)
        labels = batch["labels"].to(device)  # (B, Seq_Len)

        # 梯度清零
        optimizer.zero_grad()

        # 极其核心的一步：前向传播！
        # 模型会先用 PointNet++ 提取点云特征，再拼接到文字序列前
        outputs = model(
            input_ids=input_ids,
            point_clouds=point_clouds,
            labels=labels
        )

        # 获取 Loss (Prefix 部分的 Loss 已经被 -100 自动屏蔽了)
        loss = outputs["loss"]

        # 反向传播与权重更新
        loss.backward()
        optimizer.step()

        print(
            f"  -> Step {step + 1} | Loss: {loss.item():.4f} | 显存占用: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")

        # 测试跑通 3 个 Batch 就足够说明框架完美了
        if step == 2:
            break

    print("\n🎉 恭喜！条件生成的闭环系统彻底打通！")


if __name__ == "__main__":
    main()