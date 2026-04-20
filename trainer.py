import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from typing import List, Dict, Optional, Tuple


class LengthBucketBatchSampler(Sampler):
    """
    Build batches from similarly sized token sequences.

    This prevents one very long v2 CAD sequence from forcing many short
    sequences in the same batch to pad up to a huge attention length.
    """

    def __init__(
        self,
        lengths,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        bucket_mult: int = 16,
        seed: int = 42,
    ):
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.bucket_mult = max(int(bucket_mult), 1)
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not (0 <= self.rank < self.num_replicas):
            raise ValueError("rank must be in [0, num_replicas)")

        global_batch = self.batch_size * self.num_replicas
        self.num_batches = len(self.lengths) // global_batch if self.drop_last else math.ceil(len(self.lengths) / global_batch)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        sorted_indices = sorted(range(len(self.lengths)), key=lambda idx: self.lengths[idx])
        global_batch = self.batch_size * self.num_replicas
        bucket_size = max(global_batch * self.bucket_mult, global_batch)
        buckets = [sorted_indices[i:i + bucket_size] for i in range(0, len(sorted_indices), bucket_size)]

        bucket_order = torch.randperm(len(buckets), generator=generator).tolist() if self.shuffle else list(range(len(buckets)))
        global_batches = []
        for bucket_idx in bucket_order:
            bucket = list(buckets[bucket_idx])
            if self.shuffle:
                perm = torch.randperm(len(bucket), generator=generator).tolist()
                bucket = [bucket[i] for i in perm]

            for start_idx in range(0, len(bucket), global_batch):
                chunk = bucket[start_idx:start_idx + global_batch]
                if len(chunk) < global_batch:
                    if self.drop_last:
                        continue
                    chunk = chunk + chunk[:global_batch - len(chunk)]
                global_batches.append(chunk)

        if self.shuffle:
            order = torch.randperm(len(global_batches), generator=generator).tolist()
            global_batches = [global_batches[i] for i in order]

        for chunk in global_batches[:self.num_batches]:
            start_idx = self.rank * self.batch_size
            yield chunk[start_idx:start_idx + self.batch_size]


class ARTrainer:
    """LLaMA3自回归CAD生成模型训练器 - 兼容无条件生成与点云条件生成"""

    def __init__(self, train_dataset, val_dataset, args, device=None, multi_gpu=False,
                 grpo_config: Optional[Dict] = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Args
        self.batch_size = args.batch_size
        self.train_nepoch = args.train_nepoch
        self.test_nepoch = args.test_nepoch
        self.save_nepoch = args.save_nepoch
        self.dataset_type = args.dataset_type
        self.save_dir = args.save_dir
        self.tb_log_dir = args.tb_log_dir
        self.weight_path = args.weight
        self.max_seq_len = args.max_seq_len
        self.args = args
        self.multi_gpu = multi_gpu
        self.rank = int(os.environ.get("RANK", 0))

        # GRPO 配置（可选）
        self.grpo_config = grpo_config or {}
        self.grpo_enabled = self.grpo_config.get('enabled', False)
        self.grpo_ratio = self.grpo_config.get('grpo_ratio', 0.5)  # GRPO 训练占比
        self.grpo_group_size = self.grpo_config.get('group_size', 4)  # 每个 group 生成的序列数
        self.reward_scale = self.grpo_config.get('reward_scale', 1.0)  # 奖励缩放因子
        self.kl_penalty = self.grpo_config.get('kl_penalty', 0.0)  # KL 散度惩罚系数
        self.sft_weight = self.grpo_config.get('sft_weight', 1.0)  # SFT Loss 权重
        self.use_brep_reward = self.grpo_config.get('use_brep_reward', True)
        self.reward_cache = {}

        # Model Params
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        # Arch Params
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_layers = args.n_layers
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout
        self.rope_theta = args.rope_theta
        self.rms_norm_eps = args.rms_norm_eps

        # Optimization
        self.use_amp = args.use_amp
        self.gradient_checkpointing = args.gradient_checkpointing
        self.compile_model = args.compile_model

        # Token Config
        self.vocab_size = train_dataset.vocab_size
        self.PAD_TOKEN = train_dataset.PAD_TOKEN
        self.START_TOKEN = train_dataset.START_TOKEN
        self.END_TOKEN = getattr(train_dataset, 'END_TOKEN', None)
        self.face_index_offset = train_dataset.face_index_offset
        self.quantization_offset = train_dataset.quantization_offset

        # State
        self.epoch = 1
        self.start_epoch = 1
        self.global_step = 0
        self.iters = 0
        self.best_loss = float('inf')
        self.best_path = None

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dir Setup
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_log_dir) if self.rank == 0 else None

        self.log_txt_path = os.path.join(self.save_dir, "clean_train_log.txt")
        if self.rank == 0:
            with open(self.log_txt_path, "a") as f:
                import datetime
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'=' * 20} New Training Session: {now_str} {'=' * 20}\n")
            print(f"  [日志] 精简版日志将保存在: {self.log_txt_path}")

        # Dataloader Logic
        if self.multi_gpu and torch.cuda.device_count() > 1 and dist.is_available() and dist.is_initialized():
            num_workers = 0
            effective_batch_size = self.batch_size
            train_sampler = None
            train_batch_sampler = None
            if hasattr(train_dataset, "sequence_lengths") and len(getattr(train_dataset, "sequence_lengths", [])) == len(train_dataset):
                train_batch_sampler = LengthBucketBatchSampler(
                    train_dataset.sequence_lengths,
                    batch_size=effective_batch_size,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                    drop_last=True,
                    bucket_mult=getattr(self.args, "length_bucket_mult", 16),
                )
            else:
                train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(),
                                                   shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(),
                                            shuffle=False)
            train_shuffle = False
            if self.rank == 0:
                sampler_name = "LengthBucketBatchSampler" if train_batch_sampler is not None else "DistributedSampler"
                print(
                    f"Distributed: World={dist.get_world_size()}, Batch/GPU={effective_batch_size}, "
                    f"Workers={num_workers}, Sampler={sampler_name}")
        else:
            num_workers = 0
            effective_batch_size = self.batch_size
            train_sampler = None
            train_batch_sampler = None
            val_sampler = None
            train_shuffle = True

        if train_batch_sampler is not None:
            self.train_dataloader = DataLoader(
                train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers,
                collate_fn=train_dataset.collate_fn, pin_memory=False
            )
            train_sampler = train_batch_sampler
        else:
            self.train_dataloader = DataLoader(
                train_dataset, batch_size=effective_batch_size, shuffle=train_shuffle,
                sampler=train_sampler, drop_last=True, num_workers=num_workers,
                collate_fn=train_dataset.collate_fn, pin_memory=False
            )
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=effective_batch_size, shuffle=False,
            sampler=val_sampler, drop_last=False, num_workers=num_workers,
            collate_fn=val_dataset.collate_fn, pin_memory=False
        )
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        self._init_model()
        self._init_optimizer()

        self.amp_enabled = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        if self.weight_path and os.path.exists(self.weight_path):
            self._load_checkpoint(self.weight_path)

        if self.rank == 0:
            print(f"Ready to train for {self.train_nepoch} epochs.")

        self._init_ref_model()

    def _init_ref_model(self):
        if not self.grpo_enabled:
            self.ref_model = None
            return
        if self.rank == 0:
            print(f"\n[Ref Model] Initializing Reference Model for KL Divergence...")
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(raw_model, '_orig_mod'):
            raw_model = raw_model._orig_mod
        import copy
        self.ref_model = copy.deepcopy(raw_model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.to(self.device)

    def _init_model(self):
        from model import LLaMA3ARModel, LLaMA3Config
        model_vocab_size = self.vocab_size + 1
        config = LLaMA3Config(
            vocab_size=model_vocab_size, d_model=self.d_model, n_layers=self.n_layers,
            n_heads=self.n_heads, n_kv_heads=self.n_kv_heads, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, max_seq_len=self.max_seq_len, rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps, pad_token_id=self.PAD_TOKEN,
            quantization_offset=self.quantization_offset, face_index_offset=self.face_index_offset,
            num_components=1,
            point_prefix_tokens=getattr(self.args, 'point_prefix_tokens', 8),
        )
        self.model = LLaMA3ARModel(config).to(self.device)
        if self.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        if self.compile_model:
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                pass
        if self.multi_gpu and dist.is_available() and dist.is_initialized():
            from torch.nn.parallel import DistributedDataParallel as DDP
            local_rank = int(os.environ['LOCAL_RANK'])
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank,
                             find_unused_parameters=False)

    def _init_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay, eps=1e-8
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR
        total_steps = self.train_nepoch * len(self.train_dataloader)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=self.learning_rate * 0.1)

    def _log_to_file(self, msg):
        if self.rank == 0 and hasattr(self, 'log_txt_path'):
            with open(self.log_txt_path, "a") as f:
                f.write(f"{msg}\n")

    def train_one_epoch(self):
        self.model.train()
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch} Training", disable=(self.rank != 0))

        global_loss_num = torch.tensor(0.0, device=self.device)
        global_loss_den = torch.tensor(0.0, device=self.device)

        total_grpo_loss = 0.0
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        total_reward = 0.0
        total_combined_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].contiguous().to(self.device)
            attention_mask = batch['attention_mask'].contiguous().to(self.device)

            # 【核心修改 1】：动态抓取点云特征（如果存在的话）
            point_clouds = batch.get('point_clouds', None)
            if point_clouds is not None:
                point_clouds = point_clouds.contiguous().to(self.device)

            batch_size = input_ids.shape[0]

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                labels = input_ids.clone()
                # 屏蔽 Pad Token 的 Loss
                labels.masked_fill_(attention_mask == 0, -100)

                # 【核心修改 2】：前向传播时，送入 point_clouds
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    point_clouds=point_clouds,
                    labels=labels
                )
                sft_loss = outputs.get('loss')

            grpo_loss = torch.tensor(0.0, device=self.device)
            grpo_stats = None

            if self.grpo_enabled:
                num_grpo_samples = max(1, min(int(batch_size * self.grpo_ratio), 2))
                grpo_indices = torch.randperm(batch_size, device=self.device)[:num_grpo_samples]
                try:
                    # 如果后续开启条件生成+GRPO，将点云一并传入
                    pc_grpo = point_clouds[grpo_indices] if point_clouds is not None else None
                    grpo_stats = self._compute_grpo_loss_for_batch(
                        input_ids[grpo_indices],
                        point_clouds=pc_grpo,
                        batch_size=num_grpo_samples
                    )
                    if grpo_stats is not None:
                        grpo_loss = grpo_stats['grpo_loss']
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                    grpo_loss = torch.tensor(0.0, device=self.device)

            total_loss = grpo_loss + self.sft_weight * sft_loss

            self.scaler.scale(total_loss).backward()
            if self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            self.iters += 1

            with torch.no_grad():
                valid_tokens = (labels != -100).sum().to(torch.float32)
                global_loss_num += sft_loss.detach() * valid_tokens
                global_loss_den += valid_tokens
                total_combined_loss += total_loss.item()
                if grpo_stats is not None:
                    total_grpo_loss += grpo_stats['grpo_loss']
                    total_policy_loss += grpo_stats['policy_loss']
                    total_kl_loss += grpo_stats['kl_loss']
                    total_reward += grpo_stats['reward']

            num_batches += 1

            if self.rank == 0:
                postfix = {'Total': f'{total_loss.item():.4f}', 'SFT': f'{sft_loss.item():.4f}',
                           'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'}
                if grpo_stats is not None:
                    postfix['GRPO'] = f'{grpo_stats["grpo_loss"]:.4f}'
                    postfix['R'] = f'{grpo_stats["reward"]:.4f}'
                progress_bar.set_postfix(postfix)

                if self.global_step % 50 == 0:
                    log_str = f"Ep {self.epoch} | Step {self.global_step} | Total={total_loss.item():.4f} | SFT={sft_loss.item():.4f}"
                    if grpo_stats: log_str += f" | GRPO={grpo_stats['grpo_loss']:.4f} | R={grpo_stats['reward']:.4f}"
                    self._log_to_file(log_str)

        global_loss_num = self._all_reduce_sum(global_loss_num)
        global_loss_den = self._all_reduce_sum(global_loss_den.clamp(min=1.0))
        global_avg_ce = (global_loss_num / global_loss_den).item() if global_loss_den > 0 else 0.0

        if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar('Train/CE_Loss', global_avg_ce, self.epoch)
            self.writer.add_scalar('Train/Total_Loss', total_combined_loss / max(num_batches, 1), self.epoch)
        return global_avg_ce

    def validate(self):
        self.model.eval()
        val_bar = tqdm(self.val_dataloader, desc="Validating", disable=(self.rank != 0))
        global_loss_num = torch.tensor(0.0, device=self.device)
        global_loss_den = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_bar):
                input_ids = batch['input_ids'].contiguous().to(self.device)
                attention_mask = batch['attention_mask'].contiguous().to(self.device)

                # 【核心修改 3】：验证时也同样挂载点云特征
                point_clouds = batch.get('point_clouds', None)
                if point_clouds is not None:
                    point_clouds = point_clouds.contiguous().to(self.device)

                # 【终极修复】验证时强行关闭 FP16，使用纯 FP32 运行，彻底杜绝 65504 溢出！
                with torch.cuda.amp.autocast(enabled=False):
                    labels = input_ids.clone()
                    labels.masked_fill_(attention_mask == 0, -100)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        point_clouds=point_clouds,
                        labels=labels
                    )
                    ce_loss = outputs.get('loss')

                valid_tokens = (labels != -100).sum().to(torch.float32)

                # 【验证集防 NaN 拦截器】
                if torch.isnan(ce_loss):
                    ce_loss = torch.tensor(0.0, device=self.device)
                    valid_tokens = torch.tensor(0.0, device=self.device)

                global_loss_num += ce_loss.detach() * valid_tokens
                global_loss_den += valid_tokens

                if self.rank == 0:
                    val_bar.set_postfix({'CE': f'{ce_loss.item():.4f}'})

        global_loss_num = self._all_reduce_sum(global_loss_num)
        global_loss_den = self._all_reduce_sum(global_loss_den.clamp(min=1.0))
        global_avg_ce = (global_loss_num / global_loss_den).item()
        perplexity = math.exp(min(global_avg_ce, 20.0))

        if self.rank == 0:
            print(f"Validation - CE Loss: {global_avg_ce:.6f}, Perplexity: {perplexity:.2f}")
        return global_avg_ce, perplexity

    def save_checkpoint(self, is_best=False):
        if 'RANK' in os.environ and int(os.environ['RANK']) != 0: return
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch, 'iters': self.iters, 'global_step': self.global_step,
            'best_loss': self.best_loss, 'best_path': self.best_path,
        }
        save_path = os.path.join(self.save_dir, f'epoch_{self.epoch}.pt')
        torch.save(checkpoint, save_path)
        if is_best:
            best_path = os.path.join(self.save_dir, f'{self.args.dataset_type}_ar_point_best_model.pt')
            torch.save(checkpoint, best_path)
            self.best_path = best_path
        return save_path

    def _load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
                new_state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
                model_state = model_to_load.state_dict()
                compatible_state = {
                    key: value for key, value in new_state.items()
                    if key in model_state and model_state[key].shape == value.shape
                }
                model_to_load.load_state_dict(compatible_state, strict=False)

            self.start_epoch = 1
            self.epoch = 1
            self.global_step = 0
            self.best_loss = float('inf')
        except Exception as e:
            self.start_epoch = 1
            self.epoch = 1

    def train(self):
        start_epoch = max(self.start_epoch, 1)
        for epoch in range(start_epoch, self.train_nepoch + 1):
            self.epoch = epoch
            if self.train_sampler: self.train_sampler.set_epoch(epoch)

            self.train_one_epoch()

            save_model = False
            is_best = False
            if epoch % self.test_nepoch == 0:
                val_loss, _ = self.validate()
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                    save_model = True

            if epoch % self.save_nepoch == 0: save_model = True
            if save_model: self.save_checkpoint(is_best)

    def _all_reduce_sum(self, tensor):
        if self.multi_gpu and dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    # ==================== GRPO / 1D 推理生成逻辑 (全面重写) ====================

    def compute_reward(self, generated_sequences: List[List[int]], vocab_config: Dict) -> torch.Tensor:
        batch_size = len(generated_sequences)
        rewards = torch.zeros(batch_size, device=self.device)
        from utils import reconstruct_cad_from_sequence_nurbs, check_nurbs_format, compute_brep_score

        for i, flat_seq in enumerate(generated_sequences):
            try:
                fmt_score = check_nurbs_format(flat_seq, vocab_config)
                if fmt_score < 0.8:
                    rewards[i] = fmt_score * self.reward_scale
                    continue
                solid = reconstruct_cad_from_sequence_nurbs(sequence=flat_seq, vocab_info=vocab_config,
                                                            device=self.device, verbose=False)
                if solid is not None:
                    geo_score = compute_brep_score(solid)
                    rewards[i] = (fmt_score + geo_score) * self.reward_scale
                else:
                    rewards[i] = fmt_score * self.reward_scale
            except:
                rewards[i] = 0.0
        return rewards

    def compute_relative_rewards(self, rewards: torch.Tensor, group_indices: List[int]) -> torch.Tensor:
        relative_rewards = torch.zeros_like(rewards)
        unique_groups = list(set(group_indices))
        for group_id in unique_groups:
            group_mask = torch.tensor([idx == group_id for idx in group_indices], device=self.device)
            group_rewards = rewards[group_mask]
            if len(group_rewards) > 0:
                mean, std = group_rewards.mean(), group_rewards.std()
                if std < 1e-8:
                    relative_rewards[group_mask] = group_rewards - mean
                else:
                    relative_rewards[group_mask] = (group_rewards - mean) / std
        return relative_rewards

    def _safe_sample_token(self, logits, temperature, top_p, top_k):
        if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, nan=-50.0)
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        probs = F.softmax(logits / temperature, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)

        if top_k > 0:
            v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
            probs[probs < v[:, [-1]]] = 0.0
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)

        sum_probs = probs.sum(dim=-1, keepdim=True)
        probs = probs / (sum_probs + 1e-10)

        zero_mask = (sum_probs < 1e-5).squeeze(-1)
        if zero_mask.any(): probs[zero_mask] = 1.0 / probs.size(-1)

        try:
            token = torch.multinomial(probs, 1)
        except RuntimeError:
            token = torch.zeros((probs.size(0), 1), dtype=torch.long, device=probs.device)
        return token

    def generate_without_grad(self, prompt: torch.Tensor, max_length: int = 2048,
                              temperature: float = 1.0, top_p: float = 0.9, top_k: int = 0) -> torch.Tensor:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()

        all_input_ids = prompt.clone()
        past_key_values = None
        model_inputs = prompt
        seq_len = prompt.shape[1]

        with torch.no_grad():
            while seq_len < max_length:
                outputs = model(model_inputs, use_cache=True, past_key_values=past_key_values)
                past_key_values = outputs['past_key_values']

                h = outputs['logits'][:, -1:, :]
                token = self._safe_sample_token(h.squeeze(1), temperature, top_p, top_k)

                all_input_ids = torch.cat([all_input_ids, token], dim=1)
                model_inputs = token
                seq_len += 1

                if self.END_TOKEN is not None and (token == self.END_TOKEN).any():
                    pass

        return all_input_ids

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int = 2048,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.9, **kwargs) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        model = self.model.module if hasattr(self.model, 'module') else self.model
        eos_id = getattr(model.config, 'eos_token_id', None)

        all_input_ids = input_ids.clone()
        past_key_values = None
        model_inputs = input_ids

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        print_step = max(1, max_length // 20)

        while seq_len < max_length:
            if seq_len % print_step == 0:
                print(f"  Generating: {seq_len}/{max_length} tokens", end='\r', flush=True)

            outputs = model(model_inputs, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs['past_key_values']

            h = outputs['logits'][:, -1:, :]
            token = self._safe_sample_token(h.squeeze(1), temperature, top_p, top_k)

            all_input_ids = torch.cat([all_input_ids, token], dim=1)
            model_inputs = token
            seq_len += 1

            if eos_id is not None:
                is_eos = (token[:, 0] == eos_id)
                unfinished_sequences = unfinished_sequences.mul((~is_eos).long())
                if unfinished_sequences.max() == 0:
                    break

        print(f"\n  Done. Generated {seq_len} tokens.")
        return all_input_ids

    def compute_log_probs_for_sequences(self, sequences: torch.Tensor, point_clouds: Optional[torch.Tensor] = None,
                                        temperature: float = 1.0) -> torch.Tensor:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.train()
        if sequences.device != self.device: sequences = sequences.to(self.device)
        return model.compute_all_log_probs(sequences, point_clouds=point_clouds, temperature=temperature)

    def _compute_grpo_loss_for_batch(self, input_ids: torch.Tensor, point_clouds: Optional[torch.Tensor] = None,
                                     batch_size: int = 4):
        vocab_config = {
            'START_TOKEN': self.START_TOKEN, 'PAD_TOKEN': self.PAD_TOKEN, 'END_TOKEN': self.END_TOKEN,
            'SEP_TOKEN': getattr(self.train_dataset, 'SEP_TOKEN', None),
            'quantization_offset': self.quantization_offset, 'face_index_offset': self.face_index_offset,
            'quantization_size': getattr(self.train_dataset, 'quantization_size', 256)
        }

        try:
            prompts = input_ids[:, :1].repeat_interleave(self.grpo_group_size, dim=0)
            with torch.no_grad():
                sequences_tensor = self.generate_without_grad(prompt=prompts, max_length=self.max_seq_len)

            all_sequences = sequences_tensor.cpu().tolist()
            group_indices = torch.arange(batch_size, device=self.device).repeat_interleave(
                self.grpo_group_size).tolist()

            if len(all_sequences) == 0: return None

            rewards = self.compute_reward(all_sequences, vocab_config)
            relative_rewards = self.compute_relative_rewards(rewards, group_indices)

            sequences_tensor = sequences_tensor.to(self.device).detach()

            # 【核心修改 4】：在 GRPO 概率计算中兼容条件生成
            pc_repeat = point_clouds.repeat_interleave(self.grpo_group_size,
                                                       dim=0) if point_clouds is not None else None
            log_probs_tensor = self.compute_log_probs_for_sequences(sequences=sequences_tensor, point_clouds=pc_repeat,
                                                                    temperature=1.0)

            ref_log_probs_tensor = None
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_log_probs_tensor = self.ref_model.compute_all_log_probs(input_ids=sequences_tensor,
                                                                                point_clouds=pc_repeat, temperature=1.0)

            grpo_loss, policy_loss, kl_loss = self.compute_grpo_loss(
                sequences=all_sequences, log_probs=log_probs_tensor,
                relative_rewards=relative_rewards, reference_log_probs=ref_log_probs_tensor
            )

            return {
                'grpo_loss': grpo_loss, 'policy_loss': policy_loss, 'kl_loss': kl_loss,
                'reward': rewards.mean().item(), 'relative_reward': relative_rewards.mean().item()
            }
        except Exception as e:
            return None

    def compute_grpo_loss(self, sequences: List[List[int]], log_probs: torch.Tensor,
                          relative_rewards: torch.Tensor, reference_log_probs: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_log_probs = log_probs.sum(dim=1)
        policy_loss = -(seq_log_probs * relative_rewards).mean()

        kl_loss = torch.tensor(0.0, device=self.device)
        if reference_log_probs is not None:
            ref_seq_log_probs = reference_log_probs.sum(dim=1)
            kl_div = (seq_log_probs - ref_seq_log_probs).mean()
            kl_loss = self.kl_penalty * kl_div

        total_loss = policy_loss + kl_loss
        return total_loss, policy_loss, kl_loss
