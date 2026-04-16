import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from diffusers import VQModel
# from chamferdist import ChamferDistance
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.Geom import Geom_BSplineSurface, Geom_BSplineCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire, ShapeAnalysis_Shell, ShapeAnalysis_FreeBounds
from OCC.Core.ShapeExtend import ShapeExtend_WireData
#from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Wire, TopoDS_Shell, TopoDS_Edge, TopoDS_Solid, topods_Shell, topods_Wire, topods_Face, topods_Edge
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Wire, TopoDS_Shell, TopoDS_Edge, TopoDS_Solid, topods

# 【新增】兼容性补丁：为新版 pythonocc 创建旧版函数别名
def topods_Shell(shape): return topods.Shell(shape)
def topods_Wire(shape): return topods.Wire(shape)
def topods_Face(shape): return topods.Face(shape)
def topods_Edge(shape): return topods.Edge(shape)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_SHELL, TopAbs_EDGE
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
import OCC.Core.BRep
from occwl.io import Solid, load_step
class ChamferDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source, target, bidirectional=True, reverse=False):
        """
        Computes the Chamfer distance between source and target point clouds.
        source: (B, N, 3)
        target: (B, M, 3)
        """
        # x: (B, N, 1, 3)
        x = source.unsqueeze(2)
        # y: (B, 1, M, 3)
        y = target.unsqueeze(1)
        
        # Squared L2 distance
        # dist: (B, N, M)
        dist = torch.sum((x - y) ** 2, dim=-1)

        if reverse:
            # Target to Source
            min_dist_t2s, _ = torch.min(dist, dim=1) # (B, M)
            loss = torch.mean(min_dist_t2s, dim=1) # (B,)
        else:
            # Source to Target
            min_dist_s2t, _ = torch.min(dist, dim=2) # (B, N)
            loss_s2t = torch.mean(min_dist_s2t, dim=1)
            loss = loss_s2t

            if bidirectional:
                min_dist_t2s, _ = torch.min(dist, dim=1) # (B, M)
                loss_t2s = torch.mean(min_dist_t2s, dim=1)
                loss = loss + loss_t2s
        
        return loss.mean()

import shutup; shutup.please()

### 参数加载 ###

def get_ar_args():
    """获取LLaMA3自回归模型的训练参数"""
    parser = argparse.ArgumentParser(description='LLaMA3 CAD自回归模型训练')
    
    # === 数据参数 ===
    parser.add_argument('--sequence_file', type=str, default='data/deepcad_nurbs_sequences_test.pkl', help='预处理的序列数据路径')
    parser.add_argument('--max_face', type=int, default=50, help='最大面数')
    parser.add_argument('--max_edge', type=int, default=124, help='最大边数')
    parser.add_argument('--dataset_type', type=str, choices=['furniture', 'deepcad', 'abc'], default='deepcad', help='数据集类型')
    parser.add_argument('--max_seq_len', type=int, default=1597, help='最大序列长度')
    
    # === 训练基础参数 ===
    parser.add_argument('--batch_size', type=int, default=3, help='每张GPU的batch size')
    parser.add_argument('--train_nepoch', type=int, default=10000, help='训练轮数')
    parser.add_argument('--test_nepoch', type=int, default=1, help='验证频率（每N个epoch验证一次）')
    parser.add_argument('--save_nepoch', type=int, default=5, help='保存频率（每N个epoch保存一次）')
    
    # === 优化器参数（针对19M小模型优化）===
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率（小模型推荐6e-4~1e-3，大模型3e-4）')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减（小模型推荐0.01，大模型0.1）')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值（0表示不裁剪，推荐小模型不裁剪）')
    
    # === 模型架构参数 ===
    parser.add_argument('--d_model', type=int, default=512, help='模型维度（从LLaMA3保持384）')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数（保持8）')
    parser.add_argument('--n_kv_heads', type=int, default=4, help='KV头数（GQA，n_heads/2=4）')
    parser.add_argument('--n_layers', type=int, default=8, help='Transformer层数（保持8）')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='FFN维度（4*d_model=1024）')
    
    # === 训练优化开关 ===
    parser.add_argument('--use_amp', type=bool, default=True, help='使用混合精度训练（已启用，提速50%）')
    parser.add_argument('--gradient_checkpointing', type=bool, default=False, help='使用梯度检查点节省显存')
    parser.add_argument('--compile_model', type=bool, default=False, help='使用torch.compile加速（PyTorch 2.0+，提速10-30%）')
    
    # === 保存和日志参数 ===
    parser.add_argument('--weight', type=str, default='checkpoints/llama3_cw_pad_cascade_fusion1/16_512_1e-4_1.0/epoch_56.pt', help='checkpoint路径（留空从头训练）')
    parser.add_argument('--save_dir', type=str, default="checkpoints/llama3_cw_pad_cascade_fusion_v2_rl/64_256_1e-3_1.0", help='保存目录名')
    parser.add_argument('--tb_log_dir', type=str, default="logs/llama3_cw_pad_cascade_fusion_v2_rl/64_256_1e-3_1.0", help='TensorBoard日志目录')
    
    # === GRPO 强化学习参数 ===
    parser.add_argument('--grpo_enabled', type=bool, default=False, help='启用 GRPO 强化学习训练')
    parser.add_argument('--grpo_ratio', type=float, default=0.5, help='用于 GRPO 的样本占比（0.0-1.0）')
    parser.add_argument('--grpo_group_size', type=int, default=4, help='每个 group 生成的序列数')
    parser.add_argument('--reward_scale', type=float, default=1.0, help='奖励缩放因子')
    parser.add_argument('--kl_penalty', type=float, default=0.0, help='KL 散度惩罚系数（默认0，使用SFT Loss约束）')
    parser.add_argument('--sft_weight', type=float, default=1.0, help='SFT Loss 权重（λ）')
    
    args, _ = parser.parse_known_args()
    
    # === 自动设置LLaMA3固定参数（用户不需要调整） ===
    args.dropout = 0.1  # LLaMA3不使用dropout
    args.rope_theta = 500000.0  # LLaMA3标准RoPE频率
    args.rms_norm_eps = 1e-5  # RMSNorm标准epsilon
    args.beta1 = 0.9  # AdamW标准beta1
    args.beta2 = 0.95  # LLaMA3推荐beta2
    
    # 验证参数合理性
    assert args.d_model % args.n_heads == 0, \
        f"d_model ({args.d_model}) 必须能被 n_heads ({args.n_heads}) 整除"
    assert args.n_heads % args.n_kv_heads == 0, \
        f"n_heads ({args.n_heads}) 必须能被 n_kv_heads ({args.n_kv_heads}) 整除"
    
    return args

### eval ###
def compute_brep_score(step_file_path) -> float:
    """
    计算 B-rep 模型的几何质量分数 (0.0 ~ 1.0)
    
    分数阶梯:
    - 0.0: 读取失败 / 非法文件
    - 0.1: 读取成功
    - 0.3: 包含且仅包含一个实体 (Solid Count = 1)
    - 0.6: 线条顺序正确 (Wire Order OK)
    - 0.8: 无自交 (No Self-Intersection)
    - 1.0: 闭合且水密 (Closed & Watertight)
    """
    if isinstance(step_file_path, str):
        # Read the STEP file
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(step_file_path)

        if status != IFSelect_RetDone:
            # print("Error: Unable to read STEP file")
            return 0.0

        step_reader.TransferRoot()
        shape = step_reader.Shape()

    elif isinstance(step_file_path, TopoDS_Solid):
        shape = step_file_path

    else:
        return 0.0

    # Level 1: Read Success
    score = 0.1

    # Initialize check results
    wire_order_ok = True
    wire_self_intersection_ok = True
    shell_bad_edges_ok = True
    brep_closed_ok = True  # Initialize closed BRep check
    solid_one_ok = True

    # 1. Check if BRep has more than one solid
    if isinstance(step_file_path, str):
        try:
            cad_solid = load_step(step_file_path)
            if len(cad_solid) != 1:
                solid_one_ok = False
        except Exception:
            return score # 读取后续失败，保持 0.1
    
    if not solid_one_ok:
        return score # 0.1
    
    # Level 2: Single Solid
    score = 0.3

    # 2. Check all wires
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods_Wire(wire_explorer.Current())

            # Create a ShapeFix_Wire object
            wire_fixer = ShapeFix_Wire(wire, face, 0.01)
            wire_fixer.Load(wire)
            wire_fixer.SetFace(face)
            wire_fixer.SetPrecision(0.01)
            wire_fixer.SetMaxTolerance(1)
            wire_fixer.SetMinTolerance(0.0001)

            # Fix the wire
            wire_fixer.Perform()
            fixed_wire = wire_fixer.Wire()

            # Analyze the fixed wire
            wire_analysis = ShapeAnalysis_Wire(fixed_wire, face, 0.01)
            wire_analysis.Load(fixed_wire)
            wire_analysis.SetPrecision(0.01)
            wire_analysis.SetSurface(BRep_Tool.Surface(face))

            # 1. Check wire edge order
            order_status = wire_analysis.CheckOrder()
            if order_status != 0:  # 0 means no error
                wire_order_ok = False

            # 2. Check wire self-intersection
            if wire_analysis.CheckSelfIntersection():
                wire_self_intersection_ok = False

            wire_explorer.Next()
        face_explorer.Next()

    if not wire_order_ok:
        return score # 0.3
    
    # Level 3: Wire Order OK
    score = 0.6
    
    if not wire_self_intersection_ok:
        return score # 0.6
    
    # Level 4: No Self-Intersection
    score = 0.8

    # 3. Check for bad edges in shells
    shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while shell_explorer.More():
        shell = topods_Shell(shell_explorer.Current())
        shell_analysis = ShapeAnalysis_Shell()
        shell_analysis.LoadShells(shell)

        if shell_analysis.HasBadEdges():
            shell_bad_edges_ok = False

        shell_explorer.Next()

    # 4. Check if BRep is closed (no free edges)
    free_bounds = ShapeAnalysis_FreeBounds(shape)
    free_edges = free_bounds.GetOpenWires()
    edge_explorer = TopExp_Explorer(free_edges, TopAbs_EDGE)
    num_free_edges = 0
    while edge_explorer.More():
        num_free_edges += 1
        edge_explorer.Next()
    if num_free_edges > 0:
        brep_closed_ok = False

    if shell_bad_edges_ok and brep_closed_ok:
        score = 1.0 # Level 5: Watertight
        
    return score

def check_brep_validity(shape: TopoDS_Shape) -> bool:
    """保留旧接口以防兼容性问题，但建议使用 compute_brep_score"""
    return compute_brep_score(shape) == 1.0

### 数据处理 ###
def get_bbox(pnts):
    """
    Get the tighest fitting 3D (axis-aligned) bounding box giving a set of points
    """
    bbox_corners = []
    for point_cloud in pnts:
        # Find the minimum and maximum coordinates along each axis
        min_x = np.min(point_cloud[:, 0])
        max_x = np.max(point_cloud[:, 0])

        min_y = np.min(point_cloud[:, 1])
        max_y = np.max(point_cloud[:, 1])

        min_z = np.min(point_cloud[:, 2])
        max_z = np.max(point_cloud[:, 2])

        # Create the 3D bounding box using the min and max values
        min_point = np.array([min_x, min_y, min_z])
        max_point = np.array([max_x, max_y, max_z])
        bbox_corners.append([min_point, max_point])
    return np.array(bbox_corners)

def rotate_axis(pnts, angle_degrees, axis, normalized=False):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, ..., 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)
    
    # Convert points to homogeneous coordinates
    shape = list(np.shape(pnts))
    shape[-1] = 1
    pnts_homogeneous = np.concatenate((pnts, np.ones(shape)), axis=-1)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
            [0, np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Apply rotation
    rotated_pnts_homogeneous = np.dot(pnts_homogeneous, rotation_matrix.T)
    rotated_pnts = rotated_pnts_homogeneous[...,:3]

    # Scale the point cloud to fit within the -1 to 1 cube
    if normalized:
        max_abs_coord = np.max(np.abs(rotated_pnts))
        rotated_pnts = rotated_pnts / max_abs_coord

    return rotated_pnts

def bbox_corners(bboxes):
    """
    Given the bottom-left and top-right corners of the bbox
    Return all eight corners 
    """
    bboxes_all_corners = []
    for bbox in bboxes:
        bottom_left, top_right = bbox[:3], bbox[3:]
        # Bottom 4 corners
        bottom_front_left = bottom_left
        bottom_front_right = (top_right[0], bottom_left[1], bottom_left[2])
        bottom_back_left = (bottom_left[0], top_right[1], bottom_left[2])
        bottom_back_right = (top_right[0], top_right[1], bottom_left[2])

        # Top 4 corners
        top_front_left = (bottom_left[0], bottom_left[1], top_right[2])
        top_front_right = (top_right[0], bottom_left[1], top_right[2])
        top_back_left = (bottom_left[0], top_right[1], top_right[2])
        top_back_right = top_right

        # Combine all coordinates
        all_corners = [
            bottom_front_left,
            bottom_front_right,
            bottom_back_left,
            bottom_back_right,
            top_front_left,
            top_front_right,
            top_back_left,
            top_back_right,
        ]
        bboxes_all_corners.append(np.vstack(all_corners))
    bboxes_all_corners = np.array(bboxes_all_corners)
    return bboxes_all_corners

def compute_bbox_center_and_size(min_corner, max_corner):
    # Calculate the center
    center_x = (min_corner[0] + max_corner[0]) / 2
    center_y = (min_corner[1] + max_corner[1]) / 2
    center_z = (min_corner[2] + max_corner[2]) / 2
    center = np.array([center_x, center_y, center_z])
    # Calculate the size
    size_x = max_corner[0] - min_corner[0]
    size_y = max_corner[1] - min_corner[1]
    size_z = max_corner[2] - min_corner[2]
    size = max(size_x, size_y, size_z)
    return center, size

def rotate_point_cloud(point_cloud, angle_degrees, axis):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_radians), -np.sin(angle_radians)],
                                    [0, np.sin(angle_radians), np.cos(angle_radians)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Center the point cloud
    center = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - center

    # Apply rotation
    rotated_point_cloud = np.dot(centered_point_cloud, rotation_matrix.T)

    # Translate back to original position
    rotated_point_cloud += center

    # Find the maximum absolute coordinate value
    max_abs_coord = np.max(np.abs(rotated_point_cloud))

    # Scale the point cloud to fit within the -1 to 1 cube
    normalized_point_cloud = rotated_point_cloud / max_abs_coord

    return normalized_point_cloud

def keep_largelist(int_lists):
    # Initialize a list to store the largest integer lists
    largest_int_lists = []

    # Convert each list to a set for efficient comparison
    sets = [set(lst) for lst in int_lists]

    # Iterate through the sets and check if they are subsets of others
    for i, s1 in enumerate(sets):
        is_subset = False
        for j, s2 in enumerate(sets):
            if i != j and s1.issubset(s2) and s1 != s2:
                is_subset = True
                break
        if not is_subset:
            largest_int_lists.append(list(s1))

    # Initialize a set to keep track of seen tuples
    seen_tuples = set()

    # Initialize a list to store unique integer lists
    unique_int_lists = []

    # Iterate through the input list
    for int_list in largest_int_lists:
        # Convert the list to a tuple for hashing
        int_tuple = tuple(sorted(int_list))

        # Check if the tuple is not in the set of seen tuples
        if int_tuple not in seen_tuples:
            # Add the tuple to the set of seen tuples
            seen_tuples.add(int_tuple)

            # Add the original list to the list of unique integer lists
            unique_int_lists.append(int_list)

    return unique_int_lists

def quantize_se(bbox_coords, num_tokens=2048):
    """
    将范围在 [-1, 1] 内的边界框坐标量化为整数索引。
    使用较高的 num_tokens (1024) 来为Bbox保留更高精度。
    """
    normalized_coords = (bbox_coords + 1) / 2.0
    if isinstance(bbox_coords, torch.Tensor):
        normalized_coords = torch.clip(normalized_coords, 0, 1)
        scaled_coords = normalized_coords * (num_tokens - 1)
        quantized_indices = torch.round(scaled_coords).long()
    else: # 假设是 NumPy 数组
        normalized_coords = np.clip(normalized_coords, 0, 1)
        scaled_coords = normalized_coords * (num_tokens - 1)
        quantized_indices = np.round(scaled_coords).astype(int)
    return quantized_indices

def dequantize_se(indices, num_tokens=2048):
    """
    将整数索引反量化为范围在 [-1, 1] 内的边界框坐标。
    """
    if isinstance(indices, torch.Tensor):
        float_indices = indices.float()
    else:
        float_indices = indices.astype(float)
    normalized_coords = float_indices / (num_tokens - 1)
    bbox_coords = normalized_coords * 2.0 - 1.0
    return bbox_coords


def decode_tokens_to_ncs(tokens, vqvae_model, data_type='face', tokens_per_element=4, device="cpu"):
    """将token索引解码为NCS数据"""
    if len(tokens) == 0:
        return []
    
    with torch.no_grad():
        token_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
        batch_size, seq_len = token_tensor.shape
        
        feat_h = feat_w = int(np.sqrt(tokens_per_element))
        # print(f"Using feature map size: {feat_h}x{feat_w} for {seq_len} tokens")
        
        token_indices_reshaped = token_tensor.reshape(batch_size, feat_h, feat_w)
        
        if hasattr(vqvae_model.quantize, 'embedding'):
            embedding_weight = vqvae_model.quantize.embedding.weight
        elif hasattr(vqvae_model.quantize, 'embed'):
            embedding_weight = vqvae_model.quantize.embed.weight
        else:
            print("Error: Cannot find embedding weight in quantizer")
            return []
        
        quantized_features = torch.nn.functional.embedding(token_indices_reshaped, embedding_weight)
        quantized_features = quantized_features.permute(0, 3, 1, 2)
        
        decoded = vqvae_model.decoder(vqvae_model.post_quant_conv(quantized_features))
        
        if data_type == 'face' or data_type == 'edge':
            return convert_vqvae_output_to_ncs(decoded, data_type)
        else:
            # 非面/边的情形不在此函数处理
            return []

def parse_sequence_to_cad_data_nurbs(sequence, vocab_info, device="cpu", verbose=False):
    """
    解析由 _flatten_points 生成的 1D Token 序列为 NURBS CAD 数据。
    
    序列结构 (由 _flatten_points 定义):
    1. [START]
    2. Face Block (直到 SEP): N 个面，每个面由 [48个坐标Token, 1个索引Token] 组成 (步长49)
    3. [SEP]
    4. Edge Block (直到 END): M 条边，每条边由 [2个索引Token, 12个坐标Token] 组成 (步长14)
    5. [END]
    """
    
    # 1. 获取词表配置
    face_index_offset = vocab_info['face_index_offset']
    quantization_offset = vocab_info['quantization_offset']
    quantization_size = vocab_info['quantization_size']
    
    START_TOKEN = vocab_info['START_TOKEN']
    SEP_TOKEN = vocab_info['SEP_TOKEN']
    END_TOKEN = vocab_info['END_TOKEN']
    
    # 转换为 list 以便操作 (如果是 Tensor)
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.cpu().tolist()
    
    seq_len = len(sequence)
    if seq_len == 0:
        return {'face_ctrs': [], 'edge_ctrs': [], 'edgeFace_adj': [], 'graph_edges': ([], [])}

    # 2. 定位关键分隔符位置
    try:
        # 查找 SEP 的位置
        sep_idx = sequence.index(SEP_TOKEN)
    except ValueError:
        if verbose: print("Error: SEP_TOKEN not found in sequence.")
        return {'face_ctrs': [], 'edge_ctrs': [], 'edgeFace_adj': [], 'graph_edges': ([], [])}
    
    # 查找 END 的位置 (如果没有 END，则读取到最后)
    try:
        end_idx = sequence.index(END_TOKEN)
    except ValueError:
        end_idx = seq_len

    # ==========================================
    # Part 1: 解析 Faces
    # 区间: START(0) + 1  ->  SEP
    # 结构: [Coords(48), Index(1)] -> stride 49
    # ==========================================
    
    # 确定 Face 数据段的起止
    face_start = 1 if sequence[0] == START_TOKEN else 0
    face_end = sep_idx
    
    face_seq = sequence[face_start:face_end]
    num_face_tokens = len(face_seq)
    FACE_STRIDE = 49 # 48 coords + 1 index
    
    face_ctrs = []
    # 查表法：用于将「全局面索引」（经过 cyclic offset 后的值）映射回
    # 当前序列中的「局部面编号」(0 ~ num_face-1)
    global_to_local_face_idx = None
    
    if num_face_tokens > 0 and num_face_tokens % FACE_STRIDE == 0:
        # 使用 numpy 进行快速切片处理
        face_arr = np.array(face_seq, dtype=np.int32)
        # Reshape 为 (N_faces, 49)
        face_arr = face_arr.reshape(-1, FACE_STRIDE)
        
        # 提取坐标部分: 前48个是坐标
        # 减去 quantization_offset 还原为 0~(quantization_size-1) 的量化索引
        coords_tokens = face_arr[:, :48] - quantization_offset
        
        # 提取索引部分: 第49个是面索引 (已经在 2sequence_nurbs.py 中进行了 cyclic offset)
        # 这里减去 face_index_offset 得到「全局面索引」g_i
        indices_tokens = face_arr[:, 48] - face_index_offset  # 形状: (num_face,)
        
        # 校验合法性 (简单的范围检查，防止坏数据导致 crash)
        if np.any(coords_tokens < 0) or np.any(coords_tokens >= quantization_size):
            if verbose: print("Warning: Invalid coordinate tokens detected in Faces.")
            # 可以选择在这里截断或继续
        
        # 反量化: (N, 48) -> (N, 16, 3)
        dequantized = dequantize_se(coords_tokens, num_tokens=quantization_size)
        face_ctrs = dequantized.reshape(-1, 16, 3)

        # 构建查找表：全局面索引 g_i -> 本序列中的局部面编号 i (0 ~ num_face-1)
        # 这样就可以在解析 Edge 段时，把边引用的全局索引还原到当前序列的面编号空间，
        # 自动抵消 2sequence_nurbs.py 中的 face_index_map (cyclic offset) 影响。
        global_to_local_face_idx = {
            int(g_idx): int(i) for i, g_idx in enumerate(indices_tokens)
        }
        if verbose:
            print(f"[NURBS解析] Face 段解析得到 {len(face_ctrs)} 个面，构建了 {len(global_to_local_face_idx)} 个全局→局部面索引映射。")
        
    elif num_face_tokens > 0:
        if verbose: print(f"Warning: Face sequence length {num_face_tokens} is not a multiple of {FACE_STRIDE}.")

    # ==========================================
    # Part 2: 解析 Edges
    # 区间: SEP + 1  ->  END
    # 结构: [Index1(1), Index2(1), Coords(12)] -> stride 14
    # ==========================================
    
    edge_start = sep_idx + 1
    edge_end = end_idx
    
    edge_seq = sequence[edge_start:edge_end]
    num_edge_tokens = len(edge_seq)
    EDGE_STRIDE = 14 # 2 indices + 12 coords
    
    edge_ctrs = []
    edge_face_pairs = []
    
    if num_edge_tokens > 0 and num_edge_tokens % EDGE_STRIDE == 0:
        edge_arr = np.array(edge_seq, dtype=np.int32)
        # Reshape 为 (N_edges, 14)
        edge_arr = edge_arr.reshape(-1, EDGE_STRIDE)
        
        # 提取索引部分: 前2个是面索引 (同样经过了 face_index_map 的 cyclic offset)
        # 这里先减去 face_index_offset 得到「全局面索引」g1, g2
        adj_tokens_global = edge_arr[:, :2] - face_index_offset  # 形状: (N_edges, 2)
        
        # 提取坐标部分: 后12个是坐标
        coords_tokens = edge_arr[:, 2:] - quantization_offset
        
        # 校验坐标范围
        if np.any(coords_tokens < 0) or np.any(coords_tokens >= quantization_size):
            if verbose: print("Warning: Invalid coordinate tokens detected in Edges.")
        
        # 使用查表法，将「全局面索引」映射回当前序列中的「局部面编号」
        edge_face_pairs = []
        if global_to_local_face_idx is not None and len(global_to_local_face_idx) > 0:
            num_faces_local = len(global_to_local_face_idx)
            for pair in adj_tokens_global:
                g1, g2 = int(pair[0]), int(pair[1])
                if (g1 in global_to_local_face_idx) and (g2 in global_to_local_face_idx):
                    f1 = global_to_local_face_idx[g1]
                    f2 = global_to_local_face_idx[g2]
                    edge_face_pairs.append((f1, f2))
                else:
                    if verbose:
                        print(
                            f"Warning: Edge references invalid face indices "
                            f"({g1}, {g2}) w.r.t {num_faces_local} parsed faces，跳过该边。"
                        )
        else:
            # 理论上不会触发：有 Edge 就应当有 Face；这里作为兜底逻辑，保持原始行为
            if verbose:
                print("Warning: global_to_local_face_idx 为空，Edge 段暂时按原始索引解析（可能存在越界风险）。")
            edge_face_pairs = [tuple(pair) for pair in adj_tokens_global]
        
        # 反量化: (N, 12) -> (N, 4, 3)
        dequantized = dequantize_se(coords_tokens, num_tokens=quantization_size)
        edge_ctrs = dequantized.reshape(-1, 4, 3)
        
    elif num_edge_tokens > 0:
        if verbose: print(f"Warning: Edge sequence length {num_edge_tokens} is not a multiple of {EDGE_STRIDE}.")

    # ==========================================
    # Return
    # ==========================================
    
    if verbose:
        print(f"Parsed CAD Data: {len(face_ctrs)} Faces, {len(edge_ctrs)} Edges")

    return {
        'face_ctrs': face_ctrs,      # List or Numpy array (Nf, 16, 3)
        'edge_ctrs': edge_ctrs,      # List or Numpy array (Ne, 4, 3)
        'edgeFace_adj': edge_face_pairs, # List of (idx1, idx2)
        'graph_edges': ([p[0] for p in edge_face_pairs], [p[1] for p in edge_face_pairs]) if edge_face_pairs else ([], [])
    }

def check_nurbs_format(sequence: List[int], vocab_info: Dict) -> float:
    """
    检查序列是否符合 NURBS 格式规范，返回 0.0 ~ 1.0 的格式分
    格式要求: [START] [Faces...] [SEP] [Edges...] [END]
    Faces Block: 必须是 49 的倍数
    Edges Block: 必须是 14 的倍数
    """
    START_TOKEN = vocab_info.get('START_TOKEN')
    SEP_TOKEN = vocab_info.get('SEP_TOKEN')
    END_TOKEN = vocab_info.get('END_TOKEN')
    
    if SEP_TOKEN is None:
        return 0.0
    
    # 1. 检查是否存在 SEP
    if SEP_TOKEN not in sequence:
        return 0.1 # 严重错误：没有 SEP
    
    sep_idx = sequence.index(SEP_TOKEN)
    
    # 2. 检查 Faces Block 长度
    # Start -> SEP
    face_start_idx = 1 if (sequence and sequence[0] == START_TOKEN) else 0
    face_block = sequence[face_start_idx : sep_idx]
    len_faces = len(face_block)
    
    FACE_STRIDE = 49
    if len_faces == 0 or len_faces % FACE_STRIDE != 0:
        # Face 块长度不对
        return 0.3
        
    # 3. 检查 Edges Block 长度
    # SEP + 1 -> END (or End of Sequence)
    edge_start_idx = sep_idx + 1
    if END_TOKEN in sequence:
        end_idx = sequence.index(END_TOKEN)
        edge_block = sequence[edge_start_idx : end_idx]
    else:
        edge_block = sequence[edge_start_idx:]
        
    len_edges = len(edge_block)
    EDGE_STRIDE = 14
    
    if len_edges > 0 and len_edges % EDGE_STRIDE != 0:
        # Edge 块长度不对
        return 0.5
        
    # 完美格式
    return 1.0

def parse_sequence_to_cad_data(sequence, vocab_info, se_vqvae_model, device="cpu", scale_factor=1.0):
    """
    解析自回归序列为CAD数据，适配新的无顶点序列格式
    新格式：[START] bbox_tokens face_tokens face_index ... [SEP] face_index face_index bbox_tokens edge_tokens ... [END]
    
    Args:
        sequence: 输入的token序列
        vocab_info: 词汇表信息字典
        se_vqvae_model: 面/边VQ-VAE模型
        bbox_vqvae_model: 边界框VQ-VAE模型
        device: 计算设备
        scale_factor: 缩放因子
        
    Returns:
        CAD数据字典 (不包含顶点数据)
    """
    face_index_offset = vocab_info['face_index_offset']
    se_token_offset = vocab_info['se_token_offset']
    bbox_token_offset = vocab_info['bbox_token_offset']
    se_codebook_size = vocab_info['se_codebook_size']
    bbox_index_size = vocab_info['bbox_index_size']
    
    START_TOKEN = vocab_info['START_TOKEN']
    SEP_TOKEN = vocab_info['SEP_TOKEN']
    END_TOKEN = vocab_info['END_TOKEN']
    
    se_tokens_per_element = vocab_info['se_tokens_per_element']
    bbox_tokens_per_element = vocab_info['bbox_tokens_per_element']
    
    i = 0
    faces, face_bboxes, edges, edge_bboxes, edge_face_pairs = [], [], [], [], []
    
    if i < len(sequence) and sequence[i] == START_TOKEN:
        i += 1
    
    # Part 1: Parse faces - Format: bbox_tokens face_tokens face_index
    while i < len(sequence) and sequence[i] != SEP_TOKEN:
        bbox_tokens = []
        for _ in range(bbox_tokens_per_element):
            if i < len(sequence) and bbox_token_offset <= sequence[i] < bbox_token_offset + bbox_index_size:
                bbox_tokens.append(sequence[i] - bbox_token_offset)
                i += 1
            else: break
        
        face_tokens = []
        for _ in range(se_tokens_per_element):
            if i < len(sequence) and se_token_offset <= sequence[i] < se_token_offset + se_codebook_size:
                face_tokens.append(sequence[i] - se_token_offset)
                i += 1
            else: break
        
        if i < len(sequence) and face_index_offset <= sequence[i] < face_index_offset + vocab_info['face_index_size']:
            face_idx = sequence[i] - face_index_offset
            i += 1
            if len(bbox_tokens) == bbox_tokens_per_element: face_bboxes.append(bbox_tokens)
            else: print(f"警告：面{face_idx}的边界框token数量不匹配，期望{bbox_tokens_per_element}，实际{len(bbox_tokens)}")
            if len(face_tokens) == se_tokens_per_element: faces.append(face_tokens)
            else: print(f"警告：面{face_idx}的特征token数量不匹配，期望{se_tokens_per_element}，实际{len(face_tokens)}")
        else:
            if i < len(sequence): print(f"警告：在位置{i}处期望面索引token，但找到{sequence[i]}")
            i += 1
    
    if i < len(sequence) and sequence[i] == SEP_TOKEN:
        i += 1
    
    # Part 2: Parse edges - Format: face_index face_index bbox_tokens edge_tokens
    while i < len(sequence) and sequence[i] != END_TOKEN:
        if i + 1 < len(sequence) and \
           face_index_offset <= sequence[i] < face_index_offset + vocab_info['face_index_size'] and \
           face_index_offset <= sequence[i+1] < face_index_offset + vocab_info['face_index_size']:
            
            src_face = sequence[i] - face_index_offset
            dst_face = sequence[i+1] - face_index_offset
            i += 2
            edge_face_pairs.append((src_face, dst_face))
            
            bbox_tokens = []
            for _ in range(bbox_tokens_per_element):
                if i < len(sequence) and bbox_token_offset <= sequence[i] < bbox_token_offset + bbox_index_size:
                    bbox_tokens.append(sequence[i] - bbox_token_offset)
                    i += 1
                else: break
            if len(bbox_tokens) == bbox_tokens_per_element: edge_bboxes.append(bbox_tokens)
            else: print(f"警告：边{len(edge_bboxes)}的边界框token数量不匹配，期望{bbox_tokens_per_element}，实际{len(bbox_tokens)}")
            
            # REMOVED: Block for parsing vertex tokens is completely removed.
            
            edge_tokens = []
            for _ in range(se_tokens_per_element):
                if i < len(sequence) and se_token_offset <= sequence[i] < se_token_offset + se_codebook_size:
                    edge_tokens.append(sequence[i] - se_token_offset)
                    i += 1
                else: break
            if len(edge_tokens) == se_tokens_per_element: edges.append(edge_tokens)
            else: print(f"警告：边{len(edges)}的特征token数量不匹配，期望{se_tokens_per_element}，实际{len(edge_tokens)}")
        else:
            if i < len(sequence): print(f"警告：期望源面/目标面索引，但在位置{i}找到token {sequence[i]}")
            i += 1
            
    # print(f"解码数据: {len(faces)}个面, {len(edges)}个边, {len(face_bboxes)}个面边界框, {len(edge_bboxes)}个边边界框")
    
    surf_ncs = decode_tokens_to_ncs(faces, se_vqvae_model, 'face', se_tokens_per_element, device) if faces else []
    edge_ncs = decode_tokens_to_ncs(edges, se_vqvae_model, 'edge', se_tokens_per_element, device) if edges else []
    surf_bbox_wcs = dequantize_se(np.array(face_bboxes), num_tokens=bbox_index_size).tolist() if face_bboxes else []
    edge_bbox_wcs = dequantize_se(np.array(edge_bboxes), num_tokens=bbox_index_size).tolist() if edge_bboxes else []
    
    if scale_factor != 1.0:
        surf_bbox_wcs = [bbox / scale_factor for bbox in surf_bbox_wcs]
        edge_bbox_wcs = [bbox / scale_factor for bbox in edge_bbox_wcs]
    
    return {
        'surf_ncs': surf_ncs,
        'edge_ncs': edge_ncs,
        'surf_bbox_wcs': surf_bbox_wcs,
        'edge_bbox_wcs': edge_bbox_wcs,
        'edgeFace_adj': edge_face_pairs,
        'graph_edges': ( [p[0] for p in edge_face_pairs], [p[1] for p in edge_face_pairs] )
    }

def reconstruct_cad_from_sequence_nurbs(sequence, vocab_info, device="cpu", verbose=True):
    """
    从自回归序列重建CAD模型（NURBS格式，不使用VQ-VAE）
    参考 reconstruct_cad_from_sequence 的实现方式，但使用 NURBS 数据
    
    Args:
        sequence: 输入的token序列（已展平的token列表）
        vocab_info: 词汇表信息字典
        device: 计算设备
        verbose: 是否打印详细信息
        
    Returns:
        OpenCASCADE Solid对象或None
    """
    if verbose:
        print(f"[BREP重建] 输入序列长度: {len(sequence)} tokens")
    
    cad_data = parse_sequence_to_cad_data_nurbs(
        sequence, vocab_info, device, verbose
    )
    
    face_ctrs = np.array(cad_data['face_ctrs'])  # (Nf, 16, 3)
    edge_ctrs = np.array(cad_data['edge_ctrs'])  # (Ne, 4, 3)
    graph_edges = cad_data['graph_edges']
    edge_face_pairs = cad_data['edgeFace_adj']
    
    if len(face_ctrs) == 0 or len(edge_ctrs) == 0:
        if verbose:
            print("错误：面或边数据为空")
        return None
    
    if len(edge_ctrs) != len(edge_face_pairs):
        if verbose:
            print(f"错误：边的数量 ({len(edge_ctrs)}) 与边-面对应关系数量 ({len(edge_face_pairs)}) 不匹配")
        return None
    
    surf_wcs = []
    for face_ctr in face_ctrs:
        bspline_surface = create_bspline_surface(face_ctr)
        sampled_points = sample_bspline_surface(bspline_surface, num_u=32, num_v=32)
        surf_wcs.append(sampled_points)
    
    edge_wcs = []
    for edge_ctr in edge_ctrs:
        bspline_curve = create_bspline_curve(edge_ctr)
        sampled_points = sample_bspline_curve(bspline_curve, num_points=32)
        edge_wcs.append(sampled_points)
    
    surf_wcs = np.array(surf_wcs)  # (Nf, 32, 32, 3)
    edge_wcs = np.array(edge_wcs)  # (Ne, 32, 3)
    
    src_nodes, dst_nodes = graph_edges
    FaceEdgeAdj = []
    face_adj = [[] for _ in range(len(surf_wcs))]
    for edge_idx, (node1, node2) in enumerate(zip(graph_edges[0], graph_edges[1])):
        face_adj[node1].append(edge_idx)
        face_adj[node2].append(edge_idx)
    FaceEdgeAdj.extend(face_adj)
    
    edgeV_bbox = []
    for edge_idx in range(len(edge_wcs)):
        bbox_start_end = edge_wcs[edge_idx][[0, -1]]  # shape: [2, 3]
        edgeV_bbox.append(bbox_start_end)
    edgeV_bbox = np.array(edgeV_bbox)  # shape: [num_edges, 2, 3]

    total_vertices = len(edge_wcs) * 2
    parent = list(range(total_vertices))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    face_merged_groups = []
    for face_idx, edge_indices in enumerate(FaceEdgeAdj):
        if len(edge_indices) == 0:
            continue
            
        face_vertices = []
        for edge_idx in edge_indices:
            for vertex_pos_idx in [0, 1]:
                global_vertex_id = edge_idx * 2 + vertex_pos_idx
                position = edgeV_bbox[edge_idx, vertex_pos_idx]
                face_vertices.append((global_vertex_id, position))
        
        n_vertices = len(face_vertices)
        distance_matrix = np.zeros((n_vertices, n_vertices))
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                vid1, pos1 = face_vertices[i]
                vid2, pos2 = face_vertices[j]
                distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(pos1 - pos2)
        
        merged = set()
        face_groups = []
        while len(merged) < n_vertices:
            min_dist = float('inf')
            min_i, min_j = -1, -1
            
            for i in range(n_vertices):
                if i in merged:
                    continue
                for j in range(i+1, n_vertices):
                    if j in merged:
                        continue
                    vid_i = face_vertices[i][0]
                    vid_j = face_vertices[j][0]
                    edge_i = vid_i // 2
                    edge_j = vid_j // 2
                    if edge_i == edge_j:
                        continue
                    if distance_matrix[i, j] < min_dist:
                        min_dist = distance_matrix[i, j]
                        min_i, min_j = i, j
            
            if min_i == -1 or min_j == -1:
                break
            
            vid1, _ = face_vertices[min_i]
            vid2, _ = face_vertices[min_j]
            union(vid1, vid2)
            face_groups.append([vid1, vid2])
            merged.add(min_i)
            merged.add(min_j)
        
        face_merged_groups.append(face_groups)
    
    for i in range(len(face_merged_groups)):
        for j in range(i+1, len(face_merged_groups)):
            face1_groups = face_merged_groups[i]
            face2_groups = face_merged_groups[j]
            for group1 in face1_groups:
                for group2 in face2_groups:
                    common_vertices = set(group1) & set(group2)
                    if common_vertices:
                        for v1 in group1:
                            for v2 in group2:
                                union(v1, v2)
    
    final_groups = {}
    for vid in range(total_vertices):
        root = find(vid)
        if root not in final_groups:
            final_groups[root] = []
        final_groups[root].append(vid)
    
    unique_vertices = []
    vertex_mapping = [-1] * total_vertices
    for root, group in final_groups.items():
        group_positions = []
        for vertex_id in group:
            edge_idx = vertex_id // 2
            vertex_pos_idx = vertex_id % 2
            if edge_idx < len(edgeV_bbox):
                group_positions.append(edgeV_bbox[edge_idx, vertex_pos_idx])
        
        if group_positions:
            avg_position = np.mean(group_positions, axis=0)
            unique_vertex_idx = len(unique_vertices)
            unique_vertices.append(avg_position)
            for vertex_id in group:
                vertex_mapping[vertex_id] = unique_vertex_idx
    
    unique_vertices = np.array(unique_vertices)
    
    EdgeVertexAdj = np.zeros((len(edge_wcs), 2), dtype=int)
    for edge_idx in range(len(edge_wcs)):
        start_global_id = edge_idx * 2
        end_global_id = edge_idx * 2 + 1
        if start_global_id < len(vertex_mapping) and end_global_id < len(vertex_mapping):
            start_vertex_idx = vertex_mapping[start_global_id]
            end_vertex_idx = vertex_mapping[end_global_id]
            if start_vertex_idx >= 0 and end_vertex_idx >= 0:
                EdgeVertexAdj[edge_idx, 0] = start_vertex_idx
                EdgeVertexAdj[edge_idx, 1] = end_vertex_idx
    
    solid = construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj)
    return solid


def reconstruct_cad_from_sequence(sequence, vocab_info, se_vqvae_model, device="cpu", scale_factor=1.0, verbose=True):
    """从自回归序列重建CAD模型（无显式顶点数据）"""
    # print("=== Starting CAD Reconstruction from Sequence (No Vertex Data) ===")
    
    try:
        cad_data = parse_sequence_to_cad_data(
            sequence, vocab_info, se_vqvae_model, device, scale_factor
        )
        
        surf_ncs_vqvae = np.array(cad_data['surf_ncs'])
        edge_ncs_vqvae = np.array(cad_data['edge_ncs'])
        surf_bbox_vqvae = np.array(cad_data['surf_bbox_wcs'])
        edge_bbox_vqvae = np.array(cad_data['edge_bbox_wcs'])
        graph_edges = cad_data['graph_edges']

        if len(edge_bbox_vqvae) != len(edge_ncs_vqvae):
            print(f"边的边界框数量 ({len(edge_bbox_vqvae)}) 与NCS数据数量 ({len(edge_ncs_vqvae)}) 不匹配。无法继续。")
            return None

        # # 1. 构建面边连接关系
        # src_nodes, dst_nodes = graph_edges
        # FaceEdgeAdj = []
        # face_adj = [[] for _ in range(len(surf_ncs_vqvae))]
        # for edge_idx, (node1, node2) in enumerate(zip(graph_edges[0], graph_edges[1])):
        #     face_adj[node1].append(edge_idx)
        #     face_adj[node2].append(edge_idx)
        # FaceEdgeAdj.extend(face_adj)
        
        # 1. 智能地反向推导出索引映射
        src_nodes, dst_nodes = graph_edges
        all_face_ids_in_sequence = sorted(list(set(src_nodes) | set(dst_nodes)))
        num_actual_faces = len(surf_ncs_vqvae)

        # 2. 检查推导出的ID数量与实际解码的面数量是否一致
        if len(all_face_ids_in_sequence) != num_actual_faces:
            print(f"警告：推导出的唯一面ID数量 ({len(all_face_ids_in_sequence)}) 与实际解码的面数量 ({num_actual_faces}) 不匹配。")
            # 如果数量不匹配，创建一个截断的映射
            if len(all_face_ids_in_sequence) > num_actual_faces:
                face_id_to_idx_map = {all_face_ids_in_sequence[i]: i for i in range(num_actual_faces)}
            else:
                 face_id_to_idx_map = {face_id: i for i, face_id in enumerate(all_face_ids_in_sequence)}
        else:
            # 数量匹配，创建完整映射
            face_id_to_idx_map = {face_id: i for i, face_id in enumerate(all_face_ids_in_sequence)}

        # 3. 使用映射安全地构建面-边邻接关系
        FaceEdgeAdj = []
        face_adj = [[] for _ in range(num_actual_faces)]
        
        for edge_idx, (node1_id, node2_id) in enumerate(zip(src_nodes, dst_nodes)):
            if node1_id in face_id_to_idx_map and node2_id in face_id_to_idx_map:
                internal_idx1 = face_id_to_idx_map[node1_id]
                internal_idx2 = face_id_to_idx_map[node2_id]
                
                face_adj[internal_idx1].append(edge_idx)
                face_adj[internal_idx2].append(edge_idx)
            else:
                if verbose:
                    print(f"警告：边 {edge_idx} 引用了未知的面索引 ({node1_id}, {node2_id})。此边将被忽略。")
        
        # 4. 将构建好的邻接列表赋值
        FaceEdgeAdj.extend(face_adj)

        # 将NCS边数据转换为WCS坐标
        edge_wcs_list = []
        edgeV_bbox = []
        
        for edge_idx in range(len(edge_ncs_vqvae)):
            # try:
            # 从[6]格式的边界框中提取min/max点
            bbox = edge_bbox_vqvae[edge_idx]  # shape: [6]
            min_point = bbox[:3]  # [min_x, min_y, min_z]
            max_point = bbox[3:]  # [max_x, max_y, max_z]
            
            # 计算边界框的中心和大小
            bcenter, bsize = compute_bbox_center_and_size(min_point, max_point)
            
            # 将归一化的NCS曲线转换到WCS坐标系
            ncs_curve = edge_ncs_vqvae[edge_idx]  # shape: [32, 3], 归一化到(-1,1)
            wcs_curve = ncs_curve * (bsize / 2) + bcenter
            edge_wcs_list.append(wcs_curve)
            
            # 提取起点和终点作为边界框顶点
            bbox_start_end = wcs_curve[[0, -1]]  # shape: [2, 3]
            edgeV_bbox.append(bbox_start_end)
                
            # except Exception as e:
            #     # 使用vertex_wcs作为后备
            #     edgeV_bbox.append(vertex_vqvae[edge_idx])
            #     # 生成一个简单的直线作为后备
            #     start, end = vertex_vqvae[edge_idx]
            #     edge_wcs_list.append(np.linspace(start, end, 32))
        
        edgeV_bbox = np.array(edgeV_bbox)  # shape: [num_edges, 2, 3]
        # edge_wcs = np.array(edge_wcs_list)  # shape: [num_edges, 32, 3]

        # Step 1: 改进的基于拓扑的顶点检测
        try:
            # 为每个顶点分配全局ID: edge_idx * 2 + vertex_pos_idx (0或1)
            total_vertices = len(edge_ncs_vqvae) * 2
            
            # 使用并查集管理顶点合并
            parent = list(range(total_vertices))
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
            
            if verbose:
                print("  Step 1.1: 处理面内顶点合并...")
            # 阶段1：面内顶点检测和合并
            face_merged_groups = []  # 存储每个面内合并后的顶点组
            
            for face_idx, edge_indices in enumerate(FaceEdgeAdj):
                if len(edge_indices) == 0:
                    continue
                    
                # 收集该面内所有顶点的全局ID和位置
                face_vertices = []  # [(global_vertex_id, position), ...]
                for edge_idx in edge_indices:
                    for vertex_pos_idx in [0, 1]:  # 起点和终点
                        global_vertex_id = edge_idx * 2 + vertex_pos_idx
                        position = edgeV_bbox[edge_idx, vertex_pos_idx]
                        face_vertices.append((global_vertex_id, position))
                
                # 计算所有顶点对之间的距离
                n_vertices = len(face_vertices)
                distance_matrix = np.zeros((n_vertices, n_vertices))
                for i in range(n_vertices):
                    for j in range(i+1, n_vertices):
                        vid1, pos1 = face_vertices[i]
                        vid2, pos2 = face_vertices[j]
                        
                        # 计算几何距离
                        distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(pos1 - pos2)
                
                # 贪心算法：每次选择最短的边进行合并
                merged = set()  # 已经合并的顶点ID
                face_groups = []  # 存储该面内的顶点合并组
                
                while len(merged) < n_vertices:
                    # 找出未合并顶点中距离最小的一对
                    min_dist = float('inf')
                    min_i, min_j = -1, -1
                    
                    for i in range(n_vertices):
                        if i in merged:
                            continue
                        
                        for j in range(i+1, n_vertices):
                            if j in merged:
                                continue
                            
                            # 检查是否来自同一条边
                            vid_i = face_vertices[i][0]
                            vid_j = face_vertices[j][0]
                            edge_i = vid_i // 2
                            edge_j = vid_j // 2
                            
                            # 如果是同一条边的两个端点，跳过
                            if edge_i == edge_j:
                                continue
                            
                            if distance_matrix[i, j] < min_dist:
                                min_dist = distance_matrix[i, j]
                                min_i, min_j = i, j
                    
                    # 如果找不到可合并的顶点对，退出循环
                    if min_i == -1 or min_j == -1:
                        break
                    
                    # 合并这两个顶点
                    vid1, _ = face_vertices[min_i]
                    vid2, _ = face_vertices[min_j]
                    union(vid1, vid2)
                    
                    # 记录该面内的合并组
                    face_groups.append([vid1, vid2])
                    
                    # 标记为已合并
                    merged.add(min_i)
                    merged.add(min_j)
                
                # 将该面的合并组添加到总列表
                face_merged_groups.append(face_groups)
            
            # 统计面内合并结果
            face_groups = {}
            for vid in range(total_vertices):
                root = find(vid)
                if root not in face_groups:
                    face_groups[root] = []
                face_groups[root].append(vid)
            
            merged_groups = [group for group in face_groups.values() if len(group) > 1]
            if verbose:
                print(f"  面内合并结果: {len(merged_groups)} 个合并组")
            
            if verbose:
                print("  Step 1.2: 处理面间顶点合并...")
            # 阶段2：面间顶点合并 - 检查面内已合并的顶点组是否可以进一步合并
            
            # 对于每对面，检查它们的合并组是否有共同顶点
            for i in range(len(face_merged_groups)):
                for j in range(i+1, len(face_merged_groups)):
                    face1_groups = face_merged_groups[i]
                    face2_groups = face_merged_groups[j]
                    
                    # 检查两个面的合并组是否有交集
                    for group1 in face1_groups:
                        for group2 in face2_groups:
                            # 检查两个组是否有共同顶点
                            common_vertices = set(group1) & set(group2)
                            if common_vertices:
                                # 如果有共同顶点，合并这两个组的所有顶点
                                for v1 in group1:
                                    for v2 in group2:
                                        union(v1, v2)
            
            # 统计面间合并结果
            final_groups = {}
            for vid in range(total_vertices):
                root = find(vid)
                if root not in final_groups:
                    final_groups[root] = []
                final_groups[root].append(vid)
            
            merged_final_groups = [group for group in final_groups.values() if len(group) > 1]
            if verbose:
                print(f"  最终合并结果: {len(merged_final_groups)} 个合并组")
            
            # 生成唯一顶点和映射
            unique_vertices = []
            vertex_mapping = [-1] * total_vertices
            
            # 处理所有顶点组（包括合并的和未合并的）
            for root, group in final_groups.items():
                # 计算组内顶点的平均位置
                group_positions = []
                for vertex_id in group:
                    edge_idx = vertex_id // 2
                    vertex_pos_idx = vertex_id % 2
                    if edge_idx < len(edgeV_bbox):
                        group_positions.append(edgeV_bbox[edge_idx, vertex_pos_idx])
                
                if group_positions:
                    avg_position = np.mean(group_positions, axis=0)
                    unique_vertex_idx = len(unique_vertices)
                    unique_vertices.append(avg_position)
                    
                    # 更新映射
                    for vertex_id in group:
                        vertex_mapping[vertex_id] = unique_vertex_idx
            
            unique_vertices = np.array(unique_vertices)
            
            # 构建EdgeVertexAdj
            EdgeVertexAdj = np.zeros((len(edge_ncs_vqvae), 2), dtype=int)
            for edge_idx in range(len(edge_ncs_vqvae)):
                start_global_id = edge_idx * 2
                end_global_id = edge_idx * 2 + 1
                
                # 确保索引在有效范围内
                if start_global_id < len(vertex_mapping) and end_global_id < len(vertex_mapping):
                    start_vertex_idx = vertex_mapping[start_global_id]
                    end_vertex_idx = vertex_mapping[end_global_id]
                    
                    # 确保映射有效
                    if start_vertex_idx >= 0 and end_vertex_idx >= 0:
                        EdgeVertexAdj[edge_idx, 0] = start_vertex_idx
                        EdgeVertexAdj[edge_idx, 1] = end_vertex_idx
                    else:
                        if verbose:
                            print(f"警告: 边 {edge_idx} 的顶点映射无效 ({start_vertex_idx}, {end_vertex_idx})")
                else:
                    if verbose:
                        print(f"警告: 边 {edge_idx} 的全局顶点ID超出范围")
            
            if verbose:
                print(f"找到 {len(unique_vertices)} 个唯一顶点，从 {total_vertices} 个原始顶点中")
            
            # 验证结果的合理性
            for i, adj in enumerate(EdgeVertexAdj):
                if adj[0] == adj[1]:
                    if verbose:
                        print(f"警告: 边 {i} 的起点和终点是同一个顶点 {adj[0]}")
            
        except Exception as e:
            import traceback
            print(f'顶点检测失败: {e}')
            traceback.print_exc()
            return None

        try:
            # print("Step 2: Joint Optimization...")
            surf_wcs, edge_wcs = joint_optimize(surf_ncs_vqvae, edge_ncs_vqvae, surf_bbox_vqvae, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, len(edge_ncs_vqvae), len(surf_ncs_vqvae))
        except Exception as e:
            import traceback
            print(f'联合优化失败: {e}'); traceback.print_exc(); return None
        
        # print("Step 3: Building B-rep...")
        solid = construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj)
        # print("B-rep construction completed")
        
        return solid
        
    except Exception as e:
        import traceback
        print(f"Error during reconstruction: {e}"); traceback.print_exc(); return None

def convert_vqvae_output_to_ncs(reconstructed_tensor, data_type='face'):
    """将VQVAE输出转换为NCS格式"""
    # reconstructed_tensor: (batch, 3, 32, 32)
    
    if data_type == 'face':
        # 转换面数据: (batch, 3, 32, 32) -> (batch, 32, 32, 3)
        faces_ncs = []
        for i in range(reconstructed_tensor.shape[0]):
            face_data = reconstructed_tensor[i].permute(1, 2, 0).cpu().numpy()  # (32, 32, 3)
            faces_ncs.append(face_data)
        return faces_ncs
        
    elif data_type == 'edge':
        # 转换边数据: (batch, 3, 32, 32) -> (batch, 32, 3)
        edges_ncs = []
        for i in range(reconstructed_tensor.shape[0]):
            edge_data = reconstructed_tensor[i].permute(1, 2, 0).cpu().numpy()  # (32, 32, 3)
            # 计算所有行的平均值作为边的采样点
            edge_curve = np.mean(edge_data, axis=1)  # (32, 3)
            edges_ncs.append(edge_curve)
        return edges_ncs
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

def create_bspline_curve(ctrs):

    assert ctrs.shape[0] == 4

    poles = TColgp_Array1OfPnt(1, 4)
    for i, ctr in enumerate(ctrs, 1):
        poles.SetValue(i, gp_Pnt(*ctr))

    n_knots = 2
    knots = TColStd_Array1OfReal(1, n_knots)
    knots.SetValue(1, 0.0)
    knots.SetValue(2, 1.0)

    mults = TColStd_Array1OfInteger(1, n_knots)
    mults.SetValue(1, 4)
    mults.SetValue(2, 4)

    bspline_curve = Geom_BSplineCurve(poles, knots, mults, 3)

    return bspline_curve

def create_bspline_surface(ctrs):

    assert ctrs.shape[0] == 16

    poles = TColgp_Array2OfPnt(1, 4, 1, 4)
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            poles.SetValue(i + 1, j + 1, gp_Pnt(*ctrs[idx]))

    u_knots = TColStd_Array1OfReal(1, 2)
    v_knots = TColStd_Array1OfReal(1, 2)

    u_knots.SetValue(1, 0.0)
    u_knots.SetValue(2, 1.0)
    v_knots.SetValue(1, 0.0)
    v_knots.SetValue(2, 1.0)

    u_mults = TColStd_Array1OfInteger(1, 2)
    v_mults = TColStd_Array1OfInteger(1, 2)

    u_mults.SetValue(1, 4)
    u_mults.SetValue(2, 4)
    v_mults.SetValue(1, 4)
    v_mults.SetValue(2, 4)

    bspline_surface = Geom_BSplineSurface(poles, u_knots, v_knots, u_mults, v_mults, 3, 3)

    return bspline_surface

def sample_bspline_curve(bspline_curve, num_points=32):
    u_start, u_end = bspline_curve.FirstParameter(), bspline_curve.LastParameter()
    u_range = np.linspace(u_start, u_end, num_points)

    points = np.zeros((num_points, 3))

    for i, u in enumerate(u_range):
        pnt = bspline_curve.Value(u)
        points[i] = [pnt.X(), pnt.Y(), pnt.Z()]

    return points    # 32*3

def sample_bspline_surface(bspline_surface, num_u=32, num_v=32):
    u_start, u_end, v_start, v_end = bspline_surface.Bounds()
    u_range = np.linspace(u_start, u_end, num_u)
    v_range = np.linspace(v_start, v_end, num_v)

    points = np.zeros((num_u, num_v, 3))

    for i, u in enumerate(u_range):
        for j, v in enumerate(v_range):
            pnt = bspline_surface.Value(u, v)
            points[i, j] = [pnt.X(), pnt.Y(), pnt.Z()]

    return points      # 32*32*3

### brep 后处理 ###

def compute_bbox_center_and_size(min_corner, max_corner):
    # Calculate the center
    center_x = (min_corner[0] + max_corner[0]) / 2
    center_y = (min_corner[1] + max_corner[1]) / 2
    center_z = (min_corner[2] + max_corner[2]) / 2
    center = np.array([center_x, center_y, center_z])
    # Calculate the size
    size_x = max_corner[0] - min_corner[0]
    size_y = max_corner[1] - min_corner[1]
    size_z = max_corner[2] - min_corner[2]
    size = max(size_x, size_y, size_z)
    return center, size

class STModel(nn.Module):
    def __init__(self, num_edge, num_surf):
        super().__init__()
        self.edge_t = nn.Parameter(torch.zeros((num_edge, 3)))
        self.surf_st = nn.Parameter(torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).repeat(num_surf, 1))

def get_bbox_minmax(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return (min_point, max_point)

def detect_shared_vertex(bbox_edges_global, face_edges_global, FaceEdgeAdj):
    """
    检测共享顶点并返回唯一顶点、顶点映射和边-顶点邻接关系
    
    参数:
        bbox_edges_global: 全局边界框边坐标 [num_edges, 2, 3]
        face_edges_global: 全局顶点坐标 [num_edges, 2, 3]
        FaceEdgeAdj: 面-边邻接关系 [num_faces, edges_per_face]
    
    返回:
        unique_vertices: 唯一顶点数组 (num_vertices, 3)
        new_vertex_dict: 新顶点到旧顶点索引的映射 {new_id: [old_id1, old_id2, ...]}
        EdgeVertexAdj: 边-顶点邻接矩阵 (num_edges, 2)
    """
    # 计算每个面的边数和顶点偏移
    edge_counts = [len(edges) for edges in FaceEdgeAdj]
    edge_id_offset = 2 * np.concatenate([[0], np.cumsum(edge_counts)[:-1]])
    
    valid = True
    
    # 第一步：面内顶点合并（基于edge2loop函数）
    print("第一步：面内顶点合并...")
    used_vertex = []
    face_sep_merges = []
    
    for face_idx, edge_indices in enumerate(FaceEdgeAdj):
        if len(edge_indices) == 0:
            print(f'面 {face_idx} [SKIP] - 没有边')
            continue
            
        face_start_id = edge_id_offset[face_idx]
        
        # 收集该面的边界框边和实际边
        face_bbox_edges = bbox_edges_global[edge_indices]  # shape: [num_edges_in_face, 2, 3]
        face_actual_edges = face_edges_global[edge_indices]  # shape: [num_edges_in_face, 2, 3]
        
        print(f'面 {face_idx} - 边数: {len(edge_indices)}, 边索引: {edge_indices}')
        
        # 尝试使用边界框坐标进行闭环检测
        merged_vertex_id = edge2loop(face_bbox_edges)            
        # 无论是否有合并关系都继续处理
        merged_vertex_id = face_start_id + merged_vertex_id
        face_sep_merges.append(merged_vertex_id)
        used_vertex.append(face_bbox_edges)
        print(f'面 {face_idx} [PASS] - 使用边界框坐标，合并 {len(merged_vertex_id)} 个顶点')
        continue
        
        # 尝试使用原始几何坐标进行闭环检测
        merged_vertex_id = edge2loop(face_actual_edges)
        # 无论是否有合并关系都继续处理
        merged_vertex_id = face_start_id + merged_vertex_id
        face_sep_merges.append(merged_vertex_id)
        used_vertex.append(face_actual_edges)
        print(f'面 {face_idx} [PASS] - 使用几何坐标，合并 {len(merged_vertex_id)} 个顶点')
        continue
        
        print(f'面 {face_idx} [FAILED] - 未找到任何顶点合并关系')
        print(f'  边界框边形状: {face_bbox_edges.shape}')
        print(f'  实际边形状: {face_actual_edges.shape}')
        valid = False
        break
    
    # Invalid
    if not valid:
        raise RuntimeError("面内顶点合并失败")
    
    # 第二步：面间顶点合并（使用并查集）
    print("第二步：面间顶点合并（并查集）...")
    
    # 初始化并查集
    total_vertices = len(bbox_edges_global) * 2  # 总顶点数
    print(f"  总顶点数: {total_vertices}")
    print(f"  总边数: {len(bbox_edges_global)}")
    print(f"  各面的face_start_id: {edge_id_offset}")
    
    parent = list(range(total_vertices))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 处理每个面内的顶点合并
    for face_idx, face_merge in enumerate(face_sep_merges):
        print(f"  处理面 {face_idx} 的顶点合并: {face_merge}")
        
        # edge2loop返回的是需要合并的顶点ID数组，我们需要构建实际的合并关系
        if len(face_merge) > 0:
            # 将需要合并的顶点进行并查集操作
            # 对于每个需要合并的顶点，将其与面内其他需要合并的顶点进行合并
            merge_vertices = list(face_merge)
            if len(merge_vertices) >= 2:
                # 将第一个顶点作为根节点，其他顶点都合并到这个根节点
                root = find(merge_vertices[0])
                for vertex_id in merge_vertices[1:]:
                    union(root, vertex_id)
                print(f"    面 {face_idx} 合并了 {len(merge_vertices)} 个顶点")
            else:
                print(f"    面 {face_idx} 没有足够的顶点进行合并")
        else:
            print(f"    面 {face_idx} 没有需要合并的顶点")
    
    # 收集所有顶点组
    vertex_groups = {}
    for vid in range(total_vertices):
        root = find(vid)
        if root not in vertex_groups:
            vertex_groups[root] = []
        vertex_groups[root].append(vid)
    
    # 转换为列表格式
    total_ids = list(vertex_groups.values())
    
    # 过滤掉只有一个顶点的组（没有合并的顶点）
    total_ids = [group for group in total_ids if len(group) > 1]
    
    print(f"  面间合并结果: {len(total_ids)} 个合并组")
    for i, group in enumerate(total_ids):
        print(f"    合并组 {i}: {group}")
    
    # 生成唯一顶点和映射关系
    print("生成唯一顶点和映射关系...")
    
    # 收集所有顶点坐标
    total_pnts = np.vstack(used_vertex)
    total_pnts = total_pnts.reshape(len(total_pnts), 2, 3)
    total_pnts_flatten = total_pnts.reshape(-1, 3)
    
    # 创建唯一顶点
    unique_vertices = []
    new_vertex_dict = {}
    
    # 处理合并的顶点组
    for new_id, old_ids in enumerate(total_ids):
        # 计算合并组的平均位置
        points = total_pnts_flatten[np.array(old_ids)]
        avg_position = points.mean(axis=0)
        unique_vertices.append(avg_position)
        new_vertex_dict[new_id] = old_ids
    
    # 处理未合并的顶点（单独成组）
    single_vertices = set(range(total_vertices))
    for group in total_ids:
        for vid in group:
            single_vertices.discard(vid)
    
    for vid in single_vertices:
        new_id = len(unique_vertices)
        unique_vertices.append(total_pnts_flatten[vid])
        new_vertex_dict[new_id] = [vid]
    
    unique_vertices = np.array(unique_vertices)
    
    print(f"  生成 {len(unique_vertices)} 个唯一顶点")
    
    # 构建EdgeVertexAdj
    print("构建边-顶点邻接关系...")
    total_edges = len(bbox_edges_global)
    EdgeVertexAdj = np.zeros((total_edges, 2), dtype=int)
    
    # 创建旧顶点索引到新顶点索引的全局映射
    global_vertex_map = {}
    for new_id, old_ids in new_vertex_dict.items():
        for old_id in old_ids:
            global_vertex_map[old_id] = new_id
    
    # 填充EdgeVertexAdj
    for edge_idx in range(total_edges):
        vertex_start = edge_idx * 2
        vertex_end = edge_idx * 2 + 1
            
            # 获取顶点对应新索引
        EdgeVertexAdj[edge_idx, 0] = global_vertex_map.get(vertex_start, -1)
        EdgeVertexAdj[edge_idx, 1] = global_vertex_map.get(vertex_end, -1)
            
        if EdgeVertexAdj[edge_idx, 0] == -1 or EdgeVertexAdj[edge_idx, 1] == -1:
            print(f"警告: 边 {edge_idx} 存在未映射的顶点")
    
    print("顶点合并完成!")
    return unique_vertices, new_vertex_dict, EdgeVertexAdj

def edge2loop(face_edges):
    face_edges_flatten = face_edges.reshape(-1,3)     
    # connect end points by closest distance
    merged_vertex_id = []
    
    print(f"  edge2loop: 处理 {len(face_edges)} 条边")
    
    for edge_idx, startend in enumerate(face_edges):
        self_id = [2*edge_idx, 2*edge_idx+1]
        # left endpoint 
        distance = np.linalg.norm(face_edges_flatten - startend[0], axis=1)
        min_id = list(np.argsort(distance))
        min_id_noself = [x for x in min_id if x not in self_id]
        if len(min_id_noself) > 0:
            merged_vertex_id.append(min_id_noself[0])
            print(f"    边 {edge_idx} 起点 {2*edge_idx} 与顶点 {min_id_noself[0]} 合并")
        else:
            print(f"    边 {edge_idx} 起点 {2*edge_idx} 未找到合并对象")
            
        # right endpoint
        distance = np.linalg.norm(face_edges_flatten - startend[1], axis=1)
        min_id = list(np.argsort(distance))
        min_id_noself = [x for x in min_id if x not in self_id]
        if len(min_id_noself) > 0:
            merged_vertex_id.append(min_id_noself[0])
            print(f"    边 {edge_idx} 终点 {2*edge_idx+1} 与顶点 {min_id_noself[0]} 合并")
        else:
            print(f"    边 {edge_idx} 终点 {2*edge_idx+1} 未找到合并对象")

    if len(merged_vertex_id) == 0:
        print(f"  edge2loop: 未找到任何合并关系")
        return np.array([])
    
    merged_vertex_id = np.unique(np.array(merged_vertex_id))
    print(f"  edge2loop: 找到 {len(merged_vertex_id)} 个唯一合并顶点: {merged_vertex_id}")
    return merged_vertex_id

def joint_optimize(surf_ncs, edge_ncs, surfPos, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf):
    """
    Jointly optimize the face/edge/vertex based on topology
    """
    loss_func = ChamferDistance()

    model = STModel(num_edge, num_surf)
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-08,
    )

    # Optimize edges (directly compute)
    edge_ncs_se = edge_ncs[:,[0,-1]]
    edge_vertex_se = unique_vertices[EdgeVertexAdj]

    edge_wcs = []
    # print('Joint Optimization...')
    for wcs, ncs_se, vertex_se in zip(edge_ncs, edge_ncs_se, edge_vertex_se):
        # scale
        scale_target = np.linalg.norm(vertex_se[0] - vertex_se[1])
        scale_ncs = np.linalg.norm(ncs_se[0] - ncs_se[1])
        edge_scale = scale_target / scale_ncs

        edge_updated = wcs*edge_scale
        edge_se = ncs_se*edge_scale  

        # offset
        offset = (vertex_se - edge_se)
        offset_rev = (vertex_se - edge_se[::-1])

        # swap start / end if necessary 
        offset_error = np.abs(offset[0] - offset[1]).mean()
        offset_rev_error =np.abs(offset_rev[0] - offset_rev[1]).mean()
        if offset_rev_error < offset_error:
            edge_updated = edge_updated[::-1]
            offset = offset_rev
    
        edge_updated = edge_updated + offset.mean(0)[np.newaxis,np.newaxis,:]
        edge_wcs.append(edge_updated)

    edge_wcs = np.vstack(edge_wcs)

    # Replace start/end points with corner, and backprop change along curve
    for index in range(len(edge_wcs)):
        start_vec = edge_vertex_se[index,0] - edge_wcs[index, 0]
        end_vec = edge_vertex_se[index,1] - edge_wcs[index, -1]
        weight = np.tile((np.arange(32)/31)[:,np.newaxis], (1,3))
        weighted_vec = np.tile(start_vec[np.newaxis,:],(32,1))*(1-weight) + np.tile(end_vec,(32,1))*weight
        edge_wcs[index] += weighted_vec            

    # Optimize surfaces 
    face_edges = []
    for adj in FaceEdgeAdj:
        all_pnts = edge_wcs[adj]
        face_edges.append(torch.FloatTensor(all_pnts).cuda())

    # Initialize surface in wcs based on surface pos
    surf_wcs_init = [] 
    bbox_threshold_min = []
    bbox_threshold_max = []   
    for edges_perface, ncs, bbox in zip(face_edges, surf_ncs, surfPos):
        surf_center, surf_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
        edges_perface_flat = edges_perface.reshape(-1, 3).detach().cpu().numpy()
        min_point, max_point = get_bbox_minmax(edges_perface_flat)
        edge_center, edge_scale = compute_bbox_center_and_size(min_point, max_point)
        bbox_threshold_min.append(min_point)
        bbox_threshold_max.append(max_point)

        # increase surface size if does not fully cover the wire bbox 
        if surf_scale < edge_scale:
            surf_scale = 1.05*edge_scale
    
        wcs = ncs * (surf_scale/2) + surf_center
        surf_wcs_init.append(wcs)

    surf_wcs_init = np.stack(surf_wcs_init)

    # optimize the surface offset
    surf = torch.FloatTensor(surf_wcs_init).cuda()
    for iters in range(200):
        surf_scale = model.surf_st[:,0].reshape(-1,1,1,1)
        surf_offset = model.surf_st[:,1:].reshape(-1,1,1,3)
        surf_updated = surf + surf_offset 
        
        surf_loss = 0
        for surf_pnt, edge_pnts in zip(surf_updated, face_edges):
            surf_pnt = surf_pnt.reshape(-1,3)
            edge_pnts = edge_pnts.reshape(-1,3).detach()
            surf_loss += loss_func(surf_pnt.unsqueeze(0), edge_pnts.unsqueeze(0), bidirectional=False, reverse=True) 
        surf_loss /= len(surf_updated) 

        optimizer.zero_grad()
        (surf_loss).backward()
        optimizer.step()

        # print(f'Iter {iters} surf:{surf_loss:.5f}') 

    surf_wcs = surf_updated.detach().cpu().numpy()

    return (surf_wcs, edge_wcs)

def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)

def fix_wires(face, debug=False):
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        if debug:
            wire_checker = ShapeAnalysis_Wire(wire, face, 0.01)
            print(f"Check order 3d {wire_checker.CheckOrder()}")
            print(f"Check 3d gaps {wire_checker.CheckGaps3d()}")
            print(f"Check closed {wire_checker.CheckClosed()}")
            print(f"Check connected {wire_checker.CheckConnected()}")
        wire_fixer = ShapeFix_Wire(wire, face, 0.01)

        # wire_fixer.SetClosedWireMode(True)
        # wire_fixer.SetFixConnectedMode(True)
        # wire_fixer.SetFixSeamMode(True)

        assert wire_fixer.IsReady()
        ok = wire_fixer.Perform()
        # assert ok

def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face

def get_bbox_norm(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return np.linalg.norm(max_point - min_point)

def construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj):
    """
    Fit parametric surfaces / curves and trim into B-rep
    """
    # print('Building the B-rep...')
    # Fit surface bspline
    recon_faces = []  
    for points in surf_wcs:
        num_u_points, num_v_points = 32, 32
        uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
        for u_index in range(1,num_u_points+1):
            for v_index in range(1,num_v_points+1):
                pt = points[u_index-1, v_index-1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)
        approx_face =  GeomAPI_PointsToBSplineSurface(uv_points_array, 3, 8, GeomAbs_C2, 5e-2).Surface() 
        recon_faces.append(approx_face)

    recon_edges = []
    for points in edge_wcs:
        num_u_points = 32
        u_points_array = TColgp_Array1OfPnt(1, num_u_points)
        for u_index in range(1,num_u_points+1):
            pt = points[u_index-1]
            point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            u_points_array.SetValue(u_index, point_2d)
        try:
            approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-3).Curve()  
        except Exception as e:
            print('high precision failed, trying mid precision...')
            try:
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 8e-3).Curve()  
            except Exception as e:
                print('mid precision failed, trying low precision...')
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-2).Curve()
        recon_edges.append(approx_edge)

    # Create edges from the curve list
    edge_list = []
    for curve in recon_edges:
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        edge_list.append(edge)

    # Cut surface by wire 
    post_faces = []
    post_edges = []
    for idx,(surface, edge_incides) in enumerate(zip(recon_faces, FaceEdgeAdj)):
        corner_indices = EdgeVertexAdj[edge_incides]
        
        # ordered loop
        loops = []
        ordered = [0]
        seen_corners = [corner_indices[0,0], corner_indices[0,1]]
        next_index = corner_indices[0,1]

        while len(ordered)<len(corner_indices):
            while True:
                next_row = [idx for idx, edge in enumerate(corner_indices) if next_index in edge and idx not in ordered]
                if len(next_row) == 0:
                    break
                ordered += next_row
                next_index = list(set(corner_indices[next_row][0]) - set(seen_corners))
                if len(next_index)==0:break
                else: next_index = next_index[0]
                seen_corners += [corner_indices[next_row][0][0], corner_indices[next_row][0][1]]
            
            cur_len = int(np.array([len(x) for x in loops]).sum()) # add to inner / outer loops
            loops.append(ordered[cur_len:])
            
            # Swith to next loop
            next_corner =  list(set(np.arange(len(corner_indices))) - set(ordered))
            if len(next_corner)==0:break
            else: next_corner = next_corner[0]
            next_index = corner_indices[next_corner][0]
            ordered += [next_corner]
            seen_corners += [corner_indices[next_corner][0], corner_indices[next_corner][1]]
            next_index = corner_indices[next_corner][1]

        # Determine the outer loop by bounding box length (?)
        bbox_spans = [get_bbox_norm(edge_wcs[x].reshape(-1,3)) for x in loops]
        
        # Create wire from ordered edges
        _edge_incides_ = [edge_incides[x] for x in ordered]
        edge_post = [edge_list[x] for x in _edge_incides_]
        post_edges += edge_post

        out_idx = np.argmax(np.array(bbox_spans))
        inner_idx = list(set(np.arange(len(loops))) - set([out_idx]))

        # Outer wire
        wire_builder = BRepBuilderAPI_MakeWire()
        for edge_idx in loops[out_idx]:
            wire_builder.Add(edge_list[edge_incides[edge_idx]])
        outer_wire = wire_builder.Wire()

        # Inner wires
        inner_wires = []
        for idx in inner_idx:
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_idx in loops[idx]:
                wire_builder.Add(edge_list[edge_incides[edge_idx]])
            inner_wires.append(wire_builder.Wire())
    
        # Cut by wires
        face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire)
        for wire in inner_wires:
            face_builder.Add(wire)
        face_occ = face_builder.Shape()
        fix_wires(face_occ)
        add_pcurves_to_edges(face_occ)
        fix_wires(face_occ)
        face_occ = fix_face(face_occ)
        post_faces.append(face_occ)

    # Sew faces into solid 
    sewing = BRepBuilderAPI_Sewing()
    sewing.SetTolerance(1e-3)  # 设置容差为1e-3
    for face in post_faces:
        sewing.Add(face)
        
    # Perform the sewing operation
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    # Make a solid from the shell
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()

    return solid

### 模型检查 ###
def check_watertight(solid):
    """
    检查CAD模型是否水密
    
    Args:
        solid: CAD实体
        
    Returns:
        bool: 是否水密
        dict: 检查详情
    """
    try:
        # 基本拓扑检查
        analyzer = BRepCheck_Analyzer(solid)
        is_valid = analyzer.IsValid(solid)
        
        if not is_valid:
            return False, {"error": "Invalid solid topology"}
        
        # 检查是否为闭合实体
        topo_exp = TopologyExplorer(solid, ignore_orientation=False)
        
        # 检查面数量
        num_faces = topo_exp.number_of_faces()
        num_edges = topo_exp.number_of_edges()
        num_vertices = topo_exp.number_of_vertices()
        
        if num_faces == 0:
            return False, {"error": "No faces found"}
        
        # 检查边界边（自由边）
        free_edges = []
        for edge in topo_exp.edges():
            faces_from_edge = list(topo_exp.faces_from_edge(edge))
            if len(faces_from_edge) != 2:
                free_edges.append(edge)
        
        is_watertight = len(free_edges) == 0
        
        details = {
            "is_valid": is_valid,
            "num_faces": num_faces,
            "num_edges": num_edges,
            "num_vertices": num_vertices,
            "free_edges": len(free_edges),
            "is_watertight": is_watertight
        }
        
        return is_watertight, details
        
    except Exception as e:
        return False, {"error": f"Exception during check: {str(e)}"}
    
def visualize_solid(solid):
    """可视化CAD模型"""
    try:
        # 检查传入的 solid 是否有效
        if solid is None:
            print("无效的固体对象，无法显示")
            return
        
        from OCC.Display.SimpleGui import init_display
        display, start_display, add_menu, add_function_to_menu = init_display()
        
        # 尝试显示固体
        display.DisplayShape(solid, update=True)
        display.FitAll()
        start_display()
        
    except Exception as e:
        print(f"可视化失败: {e}")
        
        # 尝试获取并打印固体信息
        try:
            from OCC.Extend.TopologyUtils import TopologyExplorer
            topo = TopologyExplorer(solid)
            print(f"固体信息: {topo.number_of_faces()} 个面，{topo.number_of_edges()} 条边")
        except Exception as e2:
            print(f"获取固体信息失败: {e2}")
            print("无法显示固体信息")

### CAD序列分析函数 ###

def token_to_symbol(token: int, START_TOKEN: int, SEP_TOKEN: int, END_TOKEN: int, PAD_TOKEN: int,
                   face_index_offset: int, face_index_size: int, se_token_offset: int, 
                   bbox_token_offset: int, se_codebook_size: int, bbox_index_size: int) -> str:
    """将token转换为可读的符号表示"""
    if token == START_TOKEN:
        return "START"
    elif token == SEP_TOKEN:
        return "SEP"
    elif token == END_TOKEN:
        return "END"
    elif token == PAD_TOKEN:
        return "PAD"
    elif face_index_offset <= token < face_index_offset + face_index_size:
        return f"F{token - face_index_offset}"
    elif se_token_offset <= token < se_token_offset + se_codebook_size:
        return f"SE{token - se_token_offset}"
    elif bbox_token_offset <= token < bbox_token_offset + bbox_index_size:
        return f"BB{token - bbox_token_offset}"
    else:
        return f"T{token}"


def analyze_face_section(face_tokens: List[int], bbox_tokens_per_element: int, se_tokens_per_element: int,
                        face_index_offset: int, bbox_token_offset: int, se_token_offset: int, 
                        se_codebook_size: int, bbox_index_size: int) -> Tuple[int, int]:
    """
    分析面部分的结构正确性
    面的结构: [bbox_token(6个)] + [se_token(4个)] + [面索引] + ...
    
    Args:
        face_tokens: 面部分的token序列
        bbox_tokens_per_element: 每个元素的bbox token数量
        se_tokens_per_element: 每个元素的se token数量  
        face_index_offset: 面索引偏移量
        bbox_token_offset: bbox token偏移量
        se_token_offset: se token偏移量
        se_codebook_size: SE码本大小
        bbox_index_size: BBox索引大小
        
    Returns:
        (正确位置数, 总位置数)
    """
    if len(face_tokens) == 0:
        return 0, 0
    
    correct_positions = 0
    total_positions = len(face_tokens)
    
    # 每个面的结构长度
    face_structure_len = bbox_tokens_per_element + se_tokens_per_element + 1  # +1 for face index
    
    face_index_expected = face_index_offset  # 面索引应该从face_index_offset开始递增
    
    i = 0
    while i < len(face_tokens):
        if i + face_structure_len > len(face_tokens):
            # 剩余token不足一个完整面结构
            break
        
        # 检查bbox tokens (前6个位置)
        for j in range(bbox_tokens_per_element):
            token = face_tokens[i + j]
            if bbox_token_offset <= token < bbox_token_offset + bbox_index_size:
                correct_positions += 1
        
        # 检查se tokens (第7-10个位置)
        for j in range(se_tokens_per_element):
            pos = i + bbox_tokens_per_element + j
            token = face_tokens[pos]
            if se_token_offset <= token < se_token_offset + se_codebook_size:
                correct_positions += 1
        
        # 检查面索引 (第9个位置)
        face_idx_pos = i + bbox_tokens_per_element + se_tokens_per_element
        if face_idx_pos < len(face_tokens):
            token = face_tokens[face_idx_pos]
            # 面索引必须等于期望值 (升序排布)
            if token == face_index_expected:
                correct_positions += 1
            face_index_expected += 1
        
        i += face_structure_len
    
    return correct_positions, total_positions


def analyze_edge_section(edge_tokens: List[int], bbox_tokens_per_element: int, se_tokens_per_element: int,
                        face_index_offset: int, face_index_size: int, bbox_token_offset: int,
                        se_token_offset: int, se_codebook_size: int, bbox_index_size: int) -> Tuple[int, int]:
    """
    分析边部分的结构正确性（已修正为无顶点格式）
    边的结构: 面索引1 面索引2 bbox_tokens edge_tokens

    Args:
        edge_tokens: 边部分的token序列
        bbox_tokens_per_element: 每个元素的bbox token数量
        se_tokens_per_element: 每个元素的se token数量
        face_index_offset: 面索引偏移量
        face_index_size: 面索引大小
        bbox_token_offset: bbox token偏移量
        se_token_offset: se token偏移量
        se_codebook_size: SE码本大小
        bbox_index_size: BBox索引大小

    Returns:
        (正确位置数, 总位置数)
    """
    if len(edge_tokens) == 0:
        return 0, 0

    correct_positions = 0
    total_positions = len(edge_tokens)

    # [修正] 每个边的结构长度 (2个面索引 + bbox_tokens + edge_tokens)
    edge_structure_len = 2 + bbox_tokens_per_element + se_tokens_per_element

    i = 0
    while i < len(edge_tokens):
        if i + edge_structure_len > len(edge_tokens):
            # 剩余token不足一个完整边结构
            break

        # 检查两个面索引 (前2个位置)
        for j in range(2):
            face_idx_pos = i + j
            token = edge_tokens[face_idx_pos]
            if face_index_offset <= token < face_index_offset + face_index_size:
                correct_positions += 1

        # 检查bbox tokens (第3个位置开始)
        for j in range(bbox_tokens_per_element):
            bbox_pos = i + 2 + j
            token = edge_tokens[bbox_pos]
            if bbox_token_offset <= token < bbox_token_offset + bbox_index_size:
                correct_positions += 1

        # [移除] 对vertex_tokens的检查已被完全移除

        # [修正] 检查edge tokens (se_tokens) 的位置
        for j in range(se_tokens_per_element):
            edge_pos = i + 2 + bbox_tokens_per_element + j
            token = edge_tokens[edge_pos]
            if se_token_offset <= token < se_token_offset + se_codebook_size:
                correct_positions += 1

        i += edge_structure_len

    return correct_positions, total_positions


def analyze_cad_structure(tokens: List[int], START_TOKEN: int, SEP_TOKEN: int, END_TOKEN: int,
                         bbox_tokens_per_element: int, se_tokens_per_element: int,
                         face_index_offset: int, face_index_size: int, bbox_token_offset: int,
                         se_token_offset: int, se_codebook_size: int, bbox_index_size: int) -> float:
    """
    分析CAD序列的结构正确性
    
    Args:
        tokens: token序列
        START_TOKEN: 开始token
        SEP_TOKEN: 分隔token
        END_TOKEN: 结束token
        bbox_tokens_per_element: 每个元素的bbox token数量
        se_tokens_per_element: 每个元素的se token数量
        face_index_offset: 面索引偏移量
        face_index_size: 面索引大小
        bbox_token_offset: bbox token偏移量
        se_token_offset: se token偏移量
        se_codebook_size: SE码本大小
        bbox_index_size: BBox索引大小
        
    Returns:
        结构正确性分数 (0.0-1.0)
    """
    if len(tokens) < 3:  # 至少需要START, SEP, END
        return 0.0
    
    try:
        # 检查START/SEP/END token的唯一性
        start_count = tokens.count(START_TOKEN)
        sep_count = tokens.count(SEP_TOKEN)
        end_count = tokens.count(END_TOKEN)
        
        if start_count != 1 or sep_count != 1 or end_count != 1:
            return 0.0
        
        # 找到关键位置
        start_pos = tokens.index(START_TOKEN)
        sep_pos = tokens.index(SEP_TOKEN)
        end_pos = tokens.index(END_TOKEN)
        
        # 基础结构检查
        if start_pos != 0 or start_pos >= sep_pos or sep_pos >= end_pos:
            return 0.0
        
        total_positions = 0
        correct_positions = 0
        
        # 面部分分析 (START到SEP之间)
        face_section = tokens[start_pos + 1:sep_pos]
        face_correct, face_total = analyze_face_section(
            face_section, bbox_tokens_per_element, se_tokens_per_element,
            face_index_offset, bbox_token_offset, se_token_offset, se_codebook_size, bbox_index_size
        )
        correct_positions += face_correct
        total_positions += face_total
        
        # 边部分分析 (SEP到END之间)
        edge_section = tokens[sep_pos + 1:end_pos]
        edge_correct, edge_total = analyze_edge_section(
            edge_section, bbox_tokens_per_element, se_tokens_per_element,
            face_index_offset, face_index_size, bbox_token_offset, se_token_offset, se_codebook_size, bbox_index_size
        )
        correct_positions += edge_correct
        total_positions += edge_total
        
        # 计算百分比
        if total_positions == 0:
            return 0.0
        
        return correct_positions / total_positions
        
    except Exception as e:
        print(f"结构分析出错: {e}")
        return 0.0


def analyze_sequence(tokens: List[int], max_display_tokens: int = 50,
                    START_TOKEN: int = None, SEP_TOKEN: int = None, END_TOKEN: int = None, PAD_TOKEN: int = None,
                    face_index_offset: int = None, face_index_size: int = None,
                    se_token_offset: int = None, bbox_token_offset: int = None, 
                    se_codebook_size: int = None, bbox_index_size: int = None,
                    bbox_tokens_per_element: int = None, se_tokens_per_element: int = None, 
                    token_to_symbol_func=None) -> Dict:
    """
    分析生成的序列结构正确性
    
    Args:
        tokens: token序列
        max_display_tokens: 最多显示的token数量
        START_TOKEN: 开始token
        SEP_TOKEN: 分隔token
        END_TOKEN: 结束token
        PAD_TOKEN: 填充token
        face_index_offset: 面索引偏移量
        face_index_size: 面索引大小
        se_token_offset: se token偏移量
        bbox_token_offset: bbox token偏移量
        se_codebook_size: SE码本大小
        bbox_index_size: BBox索引大小
        bbox_tokens_per_element: 每个元素的bbox token数量
        se_tokens_per_element: 每个元素的se token数量
        token_to_symbol_func: token转符号的函数
        
    Returns:
        分析结果字典
    """
    results = {
        'structure_score': 0.0,
        'diversity_score': 0.0,
        'token_types': {}
    }
    
    if not tokens or len(tokens) == 0:
        results['format_message'] = "序列为空"
        return results
    
    # 基本统计
    token_types = {
        'start': tokens.count(START_TOKEN),
        'sep': tokens.count(SEP_TOKEN),
        'end': tokens.count(END_TOKEN),
        'face_index': sum(1 for t in tokens if face_index_offset <= t < face_index_offset + face_index_size),
        'se_token': sum(1 for t in tokens if se_token_offset <= t < se_token_offset + se_codebook_size),
        'bbox_token': sum(1 for t in tokens if bbox_token_offset <= t < bbox_token_offset + bbox_index_size)
    }
    results['token_types'] = token_types
    
    # Token多样性分析
    if len(tokens) > 0:
        unique_tokens = len(set(tokens))
        results['diversity_score'] = unique_tokens / len(tokens)
    
    # 结构正确性分析
    structure_score = analyze_cad_structure(
        tokens, START_TOKEN, SEP_TOKEN, END_TOKEN,
        bbox_tokens_per_element, se_tokens_per_element,
        face_index_offset, face_index_size, bbox_token_offset,
        se_token_offset, se_codebook_size, bbox_index_size
    )
    results['structure_score'] = structure_score
    
    # 显示token序列（限制数量）
    if max_display_tokens > 0 and token_to_symbol_func:
        display_tokens = tokens[:max_display_tokens]
        print(f"Token序列分析 (显示前{min(max_display_tokens, len(tokens))}个):")
        
        for pos, token in enumerate(display_tokens):
            symbol = token_to_symbol_func(token)
            print(f"{pos:2d}: {token:4d} {symbol:>8s}")
    
    return results
