import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass

# 导入刚刚修改好的 PointNet++ 主干网络
from pointnet2_cls_ssg import get_model as PointNet2Backbone


@dataclass
class LLaMA3Config:
    vocab_size: int = 4054
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.0
    max_seq_len: int = 4189
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5

    pad_token_id: int = 4053
    eos_token_id: int = 4052
    quantization_offset: int = 50
    face_index_offset: int = 0
    special_token_offset: int = 4050
    point_prefix_tokens: int = 8
    num_components: int = 1  # 彻底变为 1D

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.pad_token_id >= self.vocab_size:
            self.vocab_size = self.pad_token_id + 100


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len + 1024)
        cos = F.embedding(position_ids, self.cos_cached)
        sin = F.embedding(position_ids, self.sin_cached)
        return cos.unsqueeze(1), sin.unsqueeze(1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: LLaMA3Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta)

    def forward(self, x, position_ids, attention_mask=None, past_key_value=None, use_cache=False):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present_key_value = (k, v) if use_cache else None
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_weights = self.dropout(attn_weights.to(q.dtype))
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output), present_key_value


class SwiGLU(nn.Module):
    def __init__(self, config: LLaMA3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.dim_feedforward, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.dim_feedforward, bias=False)
        self.down_proj = nn.Linear(config.dim_feedforward, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class LLaMA3Block(nn.Module):
    def __init__(self, config: LLaMA3Config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, position_ids, attention_mask=None, past_key_value=None, use_cache=False):
        h_norm = self.attention_norm(x)
        h, present_key_value = self.attention(h_norm, position_ids, attention_mask, past_key_value, use_cache)
        x = x + self.dropout(h)
        h_norm = self.ffn_norm(x)
        h = self.feed_forward(h_norm)
        x = x + self.dropout(h)
        return x, present_key_value


# 点云投影器：负责点云降维和特征维度对齐
class PointCloudPrefixProjector(nn.Module):
    def __init__(self, ar_d_model: int, prefix_tokens: int):
        super().__init__()
        self.prefix_tokens = prefix_tokens
        # 实例化 PointNet++
        self.pointnet2 = PointNet2Backbone(normal_channel=False)

        # 降维映射网络
        self.projector = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, ar_d_model * prefix_tokens)
        )

    def forward(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """
        :param face_point_clouds: (Batch, K, N, 3) K是面数量, N是点云数量
        :return: prefix_embeds: (Batch, K, d_model)
        """
        if point_clouds.dim() == 4:
            batch_size, _, _, channels = point_clouds.shape
            point_clouds = point_clouds.reshape(batch_size, -1, channels)
        elif point_clouds.dim() == 3:
            batch_size = point_clouds.shape[0]
        else:
            raise ValueError(f"Unsupported point-cloud shape: {tuple(point_clouds.shape)}")
        # 折叠维度骗过 PointNet
        flat_pcs = point_clouds[..., :3]
        # PointNet 要求通道在中间 (Batch, 3, N)
        flat_pcs = flat_pcs.transpose(1, 2).contiguous()

        # 提特征并映射
        pn2_features = self.pointnet2(flat_pcs)
        projected = self.projector(pn2_features)

        # 解开折叠
        prefix_embeds = projected.view(batch_size, self.prefix_tokens, -1)
        return prefix_embeds


class LLaMA3ARModel(nn.Module):
    def __init__(self, config: LLaMA3Config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # 注册点云投影器
        self.prefix_projector = PointCloudPrefixProjector(
            config.d_model,
            prefix_tokens=config.point_prefix_tokens,
        )

        self.layers = nn.ModuleList([LLaMA3Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            input_ids: torch.Tensor,
            point_clouds: Optional[torch.Tensor] = None,  # 允许输入点云作为条件
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, List]]:

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # --- 1. 获取 Embedding 并且做跨模态拼接 ---
        text_embeds = self.token_embedding(input_ids)

        if point_clouds is not None:
            # 提取点云 Prefix
            prefix_embeds = self.prefix_projector(point_clouds)  # (B, K, d_model)
            K = prefix_embeds.shape[1]

            # 拼接到文本的 Embedding 之前！
            hidden_states = torch.cat([prefix_embeds, text_embeds], dim=1)  # 最终长度: K + seq_len
            total_len = K + seq_len
        else:
            K = 0
            hidden_states = text_embeds
            total_len = seq_len

        # --- 2. 重新处理 Position IDs ---
        if position_ids is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_length, past_length + total_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # --- 3. 重写 Attention Mask 兼容 Prefix ---
        if attention_mask is None:
            # 生成对于整体长度 (K + seq_len) 的下三角因果 Mask
            causal_mask = torch.full((total_len, total_len), float("-inf"), device=device)
            causal_mask = torch.triu(causal_mask, diagonal=1)

            # 如果有 Prefix，让 Prefix 内部的 Token 能够互相双向看到 (提升局部感知)
            if K > 0:
                causal_mask[:K, :K] = 0.0

            # 处理 Padding
            is_pad = (input_ids == self.config.pad_token_id)
            if K > 0:
                # Prefix 部分是没有 Pad 的，前面拼接一排 False
                prefix_pad = torch.zeros((batch_size, K), dtype=torch.bool, device=device)
                is_pad = torch.cat([prefix_pad, is_pad], dim=1)

            padding_mask = torch.zeros((batch_size, 1, 1, total_len), device=device)
            padding_mask = padding_mask.masked_fill(is_pad.unsqueeze(1).unsqueeze(2), float("-inf"))

            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0) + padding_mask
        else:
            if attention_mask.dim() == 2:
                # 【新增逻辑】：如果有点云前缀，强行在 attention_mask 前面拼接全 1 的有效掩码
                if K > 0:
                    prefix_mask = torch.ones((batch_size, K), dtype=attention_mask.dtype, device=device)
                    attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

                head_dtype = self.lm_head.weight.dtype
                expanded_mask = attention_mask[:, None, None, :].to(dtype=head_dtype)
                expanded_mask = (1.0 - expanded_mask) * torch.finfo(head_dtype).min
                if total_len > 1:
                    causal_mask = torch.full((total_len, total_len), float("-inf"), device=device)
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                    if K > 0: causal_mask[:K, :K] = 0.0  # 保持双向
                    attention_mask = expanded_mask + causal_mask.unsqueeze(0).unsqueeze(0)
                else:
                    attention_mask = expanded_mask

        # --- 4. 标准 LLaMA 运算 ---
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, layer_present_kv = layer(
                hidden_states, position_ids=position_ids,
                attention_mask=attention_mask, past_key_value=past_kv, use_cache=use_cache
            )
            if use_cache:
                present_key_values.append(layer_present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        output = {
            'logits': logits,
            'past_key_values': present_key_values,
            'last_hidden_state': hidden_states,
            'prefix_length': K,
        }

        # --- 5. 屏蔽点云部分的 Loss ---
        if labels is not None:
            if K > 0:
                # 用 -100 填充 Prefix 部分的 Label，确保模型只计算预测 CAD 序列的 Loss
                prefix_labels = torch.full((batch_size, K), -100, dtype=labels.dtype, device=device)
                full_labels = torch.cat([prefix_labels, labels], dim=1)
            else:
                full_labels = labels

            # 标准的自回归位移损失计算
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            output['loss'] = loss

        return output

    def compute_all_log_probs(self, input_ids: torch.Tensor, point_clouds: Optional[torch.Tensor] = None,
                              temperature: float = 1.0) -> torch.Tensor:
        outputs = self(input_ids, point_clouds=point_clouds, use_cache=False)
        # 如果有 prefix，截取最后 seq_len 个 logits
        if point_clouds is not None:
            K = outputs['prefix_length']
            logits = outputs['logits'][:, K:-1, :]
        else:
            logits = outputs['logits'][:, :-1, :]

        targets = input_ids[:, 1:]

        logits = torch.clamp(logits, min=-50.0, max=50.0)
        log_probs_all = F.log_softmax(logits / temperature, dim=-1)
        log_probs = log_probs_all.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        return log_probs

    @torch.no_grad()
    def generate_conditional(
            self,
            point_clouds: torch.Tensor,
            max_new_tokens: int = 1597,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.9,
            eos_token_id: Optional[int] = None,
            special_token_offset: int = 4050
    ) -> torch.Tensor:
        """
        条件生成推理引擎：看着点云，一个词一个词地生成 CAD 序列
        """
        self.eval()
        device = point_clouds.device
        batch_size = point_clouds.shape[0]

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        # 1. 提取点云的 Prefix 特征，并在整个生成过程中保持固定
        prefix_embeds = self.prefix_projector(point_clouds)  # (Batch, K, d_model)
        K = prefix_embeds.shape[1]

        # 2. 初始化生成序列（放入 START TOKEN 作为第一个词）
        generated_ids = torch.full((batch_size, 1), special_token_offset, dtype=torch.long, device=device)

        # 3. 循环自回归生成
        for step in range(max_new_tokens):
            text_embeds = self.token_embedding(generated_ids)

            # 拼接: [点云特征] + [已生成文本]
            inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
            total_len = inputs_embeds.shape[1]

            # 构造因果掩码 (Prefix 内部互见，文本部分因果可见)
            causal_mask = torch.full((total_len, total_len), float("-inf"), device=device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            if K > 0:
                causal_mask[:K, :K] = 0.0
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            # 位置编码
            position_ids = torch.arange(0, total_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size,
                                                                                                           -1)

            # 前向传播
            hidden_states = inputs_embeds
            for layer in self.layers:
                hidden_states, _ = layer(
                    hidden_states, position_ids=position_ids, attention_mask=attention_mask
                )

            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)

            # 4. 取最后一步的预测结果
            next_token_logits = logits[:, -1, :] / temperature

            # 5. Top-K / Top-P 采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 6. 把新词拼接到序列中
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # 7. 如果生成了结束符 (END_TOKEN)，提前结束
            if (next_token == eos_token_id).all():
                break

        return generated_ids
