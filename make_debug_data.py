import pickle

# 1. 你的 SFT 数据集路径 (正确源头)
SOURCE_FILE = "/mnt/sdb/siguoqing/NurbsRL_code/data/deepcad_nurbs_sequences_v1.pkl"
# 2. 输出的调试数据集路径
OUTPUT_FILE = "/mnt/sdb/siguoqing/NurbsRL_code/data/deepcad_nurbs_sequences_debug.pkl"

print(f"Loading {SOURCE_FILE}...")
with open(SOURCE_FILE, 'rb') as f:
    data = pickle.load(f)

# 3. 只取前 200 个做调试 (保证词表 metadata 完全继承)
original_train = data['train']
data['train'] = original_train[:200]
data['val'] = data['val'][:50]  # 验证集也切小点

print(f"Original Size: {len(original_train)}")
print(f"Debug Size: {len(data['train'])}")

# 4. 保存
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(data, f)
print(f"Saved to {OUTPUT_FILE}")