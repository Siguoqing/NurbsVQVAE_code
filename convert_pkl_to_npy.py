import os
import glob
import pickle
import numpy as np

# 设置你的路径
pkl_dir = "/mnt/docker_dir/lijiahao/brepgen/data/deepcad_parsed/0000/"  # 你刚找到的 pkl 文件夹
out_npy_dir = "data/test_pointclouds/"  # 存放测试 npy 的文件夹
os.makedirs(out_npy_dir, exist_ok=True)

# 挑选前 20 个文件进行盲盒测试
pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))[:20]

for pkl_file in pkl_files:
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # 提取模型中所有面的全局坐标系点云 (surf_wcs)
        surf_points = data['surf_wcs']

        # 将可能为 (K, 32, 32, 3) 的网格点展平为 (K, 1024, 3)
        surf_points = surf_points.reshape(surf_points.shape[0], -1, 3)

        # ⭐️ 关键对齐：降采样到 512 个点
        # (因为 generate_cond.py 里咱们假设的 N 是 512，请根据你实际训练的维度修改)
        if surf_points.shape[1] > 512:
            # 随机打乱并抽取 512 个点
            idx = np.random.choice(surf_points.shape[1], 512, replace=False)
            surf_points = surf_points[:, idx, :]
        elif surf_points.shape[1] < 512:
            # 如果点不够，补零或者重复采样 (这通常不会发生，因为原数据是 32x32=1024)
            pass

        out_name = os.path.basename(pkl_file).replace(".pkl", ".npy")
        out_path = os.path.join(out_npy_dir, out_name)

        # 必须存为 float32
        np.save(out_path, surf_points.astype(np.float32))
        print(f"✅ 已生成测试点云: {out_name} | 矩阵形状: {surf_points.shape}")

    except Exception as e:
        print(f"❌ 解析 {pkl_file} 失败: {e}")