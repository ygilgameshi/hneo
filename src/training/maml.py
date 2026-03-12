"""
MAML (Model-Agnostic Meta-Learning) 实现

核心思想：
- 内循环（Inner Loop）：在support set上快速适应
- 外循环（Outer Loop）：在query set上评估，更新meta参数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import copy
from tqdm import tqdm
import numpy as np


class MAMLTrainer:
    """
    MAML训练器

    实现Model-Agnostic Meta-Learning算法

    Args:
        model: 要训练的模型
        inner_lr: 内循环学习率（任务适应学习率）
        meta_lr: 外循环学习率（元学习率）
        inner_steps: 内循环更新步数
        first_order: 是否使用First-order MAML（更快但性能略降）

    Example:
        >>> model = ImmuneAppPhase1(n_tasks=50)
        >>> trainer = MAMLTrainer(model, inner_lr=0.01, meta_lr=0.001)
        >>> metrics = trainer.meta_train_step(episode_data, graph_data)
    """

    def __init__(self,
                 model,
                 inner_lr=0.01,
                 meta_lr=0.001,
                 inner_steps=5,
                 first_order=False):

        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order

        # Meta优化器（用于更新元参数）
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)

        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()

        print(f"MAML Trainer initialized:")
        print(f"  Inner LR: {inner_lr}")
        print(f"  Meta LR: {meta_lr}")
        print(f"  Inner steps: {inner_steps}")
        print(f"  First-order: {first_order}")

    def inner_loop(self, support_data, graph_data):
        """
        内循环：在support set上快速适应

        使用改进的实现：直接操作模型参数的克隆

        Args:
            support_data: dict with keys: peptide, peptide_len, task_idx, label
            graph_data: dict with keys: edge_index, edge_weight

        Returns:
            adapted_params: 适应后的参数字典
            support_loss: support set上的平均loss
        """
        # 克隆当前参数
        fast_weights = OrderedDict()
        for name, param in self.model.named_parameters():
            fast_weights[name] = param.clone()

        # 内循环更新
        for step in range(self.inner_steps):
            # 前向传播（使用快速权重）
            logits = self._functional_forward(fast_weights, support_data, graph_data)

            # 计算loss
            loss = self.criterion(logits.squeeze(), support_data['label'].float())

            # 计算梯度
            grads = torch.autograd.grad(
                loss,
                fast_weights.values(),
                create_graph=not self.first_order,
                allow_unused=True
            )

            # 更新快速权重
            fast_weights = OrderedDict(
                (name, param - self.inner_lr * grad if grad is not None else param)
                for (name, param), grad in zip(fast_weights.items(), grads)
            )

        # 返回适应后的参数和最后的loss
        return fast_weights, loss.item()

    def outer_loop(self, query_data, adapted_params, graph_data):
        """
        外循环：在query set上评估适应后的模型

        Args:
            query_data: dict
            adapted_params: 适应后的参数
            graph_data: dict

        Returns:
            query_loss: query set上的loss
        """
        # 使用适应后的参数在query set上前向传播
        logits = self._functional_forward(adapted_params, query_data, graph_data)

        # 计算loss
        query_loss = self.criterion(logits.squeeze(), query_data['label'].float())

        return query_loss

    def meta_train_step(self, episode_data, graph_data):
        """
        完整的meta训练步骤（一个episode）

        步骤：
        1. 内循环：在support set上适应
        2. 外循环：在query set上评估
        3. 反向传播：更新meta参数

        Args:
            episode_data: dict with 'support' and 'query'
            graph_data: dict with 'edge_index' and 'edge_weight'

        Returns:
            metrics: dict with losses
        """
        support_data = episode_data['support']
        query_data = episode_data['query']

        # 清零meta梯度
        self.meta_optimizer.zero_grad()

        # 内循环：适应
        adapted_params, support_loss = self.inner_loop(support_data, graph_data)

        # 外循环：评估
        query_loss = self.outer_loop(query_data, adapted_params, graph_data)

        # 反向传播meta loss（更新元参数）
        query_loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 更新meta参数
        self.meta_optimizer.step()

        return {
            'support_loss': support_loss,
            'query_loss': query_loss.item(),
        }

    def _functional_forward(self, params, data, graph_data):
        """
        函数式前向传播：使用指定的参数

        这是MAML的核心：不修改模型的实际参数，
        而是用临时参数进行前向传播

        Args:
            params: OrderedDict of parameters
            data: input data
            graph_data: graph data

        Returns:
            logits
        """
        # ===== 方法：临时替换参数 =====
        # 保存原始参数
        original_params = OrderedDict()
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()

        # 替换为新参数
        for name, param in params.items():
            self._set_param_data(self.model, name, param)

        # 前向传播
        logits = self.model(
            data['peptide'],
            data['peptide_len'],
            data['task_idx'],
            graph_data['edge_index'],
            graph_data['edge_weight']
        )

        # 恢复原始参数
        for name, param in original_params.items():
            self._set_param_data(self.model, name, param)

        return logits

    @staticmethod
    def _set_param_data(model, name, param_data):
        """
        设置模型参数的data（不改变Parameter对象）

        这是关键：我们只修改Parameter的data，
        而不是替换整个Parameter对象
        """
        name_parts = name.split('.')
        module = model

        # 导航到目标模块
        for part in name_parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        # 获取参数对象
        param_name = name_parts[-1]
        if param_name.isdigit():
            param = module[int(param_name)]
        else:
            param = getattr(module, param_name)

        # 只修改data，不修改Parameter对象
        if isinstance(param, nn.Parameter):
            param.data = param_data.data
        else:
            # 如果是buffer或其他，直接赋值
            if param_name.isdigit():
                module[int(param_name)] = param_data
            else:
                setattr(module, param_name, param_data)


class MAMLDataLoader:
    """
    MAML的Episode数据加载器

    将标准的batch数据转换为MAML的episode格式：
    - Support set: K个样本/task，用于快速适应
    - Query set: Q个样本/task，用于评估

    Args:
        dataloader: 标准PyTorch DataLoader
        n_way: 每个episode包含几个tasks
        k_shot: 每个task的support样本数
        q_query: 每个task的query样本数

    Example:
        >>> standard_loader = DataLoader(dataset, batch_size=64)
        >>> maml_loader = MAMLDataLoader(standard_loader, n_way=5, k_shot=10, q_query=10)
        >>> for episode in maml_loader:
        ...     support = episode['support']
        ...     query = episode['query']
    """

    def __init__(self, dataloader, n_way=5, k_shot=10, q_query=10):
        self.dataloader = dataloader
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        self.samples_per_episode = n_way * (k_shot + q_query)

    def __iter__(self):
        """生成episodes"""
        batch_buffer = []

        for batch in self.dataloader:
            batch_buffer.append(batch)

            # 计算累积的样本数
            total_samples = sum(len(b['label']) for b in batch_buffer)

            # 当样本数足够时，构造一个episode
            if total_samples >= self.samples_per_episode:
                episode = self._construct_episode(batch_buffer)

                if episode is not None:
                    yield episode

                batch_buffer = []

        # 处理剩余数据
        if len(batch_buffer) > 0:
            episode = self._construct_episode(batch_buffer)
            if episode is not None:
                yield episode

    def _construct_episode(self, batch_buffer):
        """
        从累积的batches构造一个episode

        步骤：
        1. 合并所有batch
        2. 按task分组
        3. 随机选择n_way个tasks
        4. 每个task采样k_shot + q_query个样本
        5. 分为support和query
        """
        # 合并所有batch
        all_data = {
            'peptide': torch.cat([b['peptide'] for b in batch_buffer]),
            'peptide_len': torch.cat([b['peptide_len'] for b in batch_buffer]),
            'task_idx': torch.cat([b['task_idx'] for b in batch_buffer]),
            'label': torch.cat([b['label'] for b in batch_buffer]),
        }

        # 按task分组
        unique_tasks = torch.unique(all_data['task_idx'])

        # 如果task数不够，返回None
        if len(unique_tasks) < self.n_way:
            return None

        # 随机选择n_way个tasks
        selected_tasks = unique_tasks[torch.randperm(len(unique_tasks))[:self.n_way]]

        # 为每个task采样support和query
        support_indices = []
        query_indices = []

        for task in selected_tasks:
            # 找到该task的所有样本
            task_mask = all_data['task_idx'] == task
            task_indices = torch.where(task_mask)[0]

            # 样本数不够，跳过
            if len(task_indices) < (self.k_shot + self.q_query):
                continue

            # 随机采样
            perm = torch.randperm(len(task_indices))
            support_indices.extend(task_indices[perm[:self.k_shot]].tolist())
            query_indices.extend(
                task_indices[perm[self.k_shot:self.k_shot + self.q_query]].tolist()
            )

        # 如果采样失败，返回None
        if len(support_indices) == 0 or len(query_indices) == 0:
            return None

        # 构造support和query sets
        support_data = {
            'peptide': all_data['peptide'][support_indices],
            'peptide_len': all_data['peptide_len'][support_indices],
            'task_idx': all_data['task_idx'][support_indices],
            'label': all_data['label'][support_indices],
        }

        query_data = {
            'peptide': all_data['peptide'][query_indices],
            'peptide_len': all_data['peptide_len'][query_indices],
            'task_idx': all_data['task_idx'][query_indices],
            'label': all_data['label'][query_indices],
        }

        return {
            'support': support_data,
            'query': query_data,
        }

    def __len__(self):
        """估算episode数量"""
        return len(self.dataloader) // (self.samples_per_episode // 32)


class MAMLTrainerPhase2:
    """
    Phase 2 MAML训练器：支持HLA×Tissue任务

    包装你现有的MAMLTrainer，添加Phase 2功能

    Args:
        base_trainer: 你现有的MAMLTrainer实例
        tasks: 任务字典 {task_name: {'data': df, 'type': 'phase1'/'phase2', ...}}
        graph_wrapper: TaskGraphWrapper实例
        device: torch device
        max_len: 肽段最大长度（默认15）
    """

    def __init__(self, base_trainer, tasks, graph_wrapper, device, max_len=15):
        self.base_trainer = base_trainer
        self.tasks = tasks
        self.graph_wrapper = graph_wrapper
        self.device = device
        self.max_len = max_len

        # 氨基酸编码字典（与dataset.py保持一致）
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.aa_to_idx['X'] = len(self.aa_to_idx)  # padding token = 20

        # 分离Phase 1和Phase 2任务
        self.phase1_tasks = [name for name, info in tasks.items() if info['type'] == 'phase1']
        self.phase2_tasks = [name for name, info in tasks.items() if info['type'] == 'phase2']

        # 计算任务权重（平方根平滑）
        self.task_weights = self._compute_task_weights()

        print(f"\n=== MAMLTrainerPhase2 初始化 ===")
        print(f"Phase 1 任务: {len(self.phase1_tasks)}")
        print(f"Phase 2 任务: {len(self.phase2_tasks)}")
        print(f"总任务数: {len(self.tasks)}")

    def _compute_task_weights(self):
        """计算任务采样权重（平方根平滑避免大任务主导）"""
        weights = {}
        for task_name, task_info in self.tasks.items():
            n_samples = task_info['n_samples']
            weights[task_name] = np.sqrt(n_samples)  # 平方根平滑

        # 归一化
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        return weights

    def encode_peptide(self, peptide):
        """
        编码肽段序列（与dataset.py完全一致）

        Args:
            peptide: 肽段序列字符串

        Returns:
            tensor: (max_len,) 整数编码
        """
        # 转为索引
        indices = [self.aa_to_idx.get(aa, self.aa_to_idx['X']) for aa in peptide]

        # Padding
        if len(indices) < self.max_len:
            indices += [self.aa_to_idx['X']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return torch.LongTensor(indices)

    def _prepare_episode(self, task_names, k_shot, q_query):
        """
        准备一个episode的数据（support set和query set）

        这是完全适配你代码的版本！

        Args:
            task_names: 要采样的任务名称列表
            k_shot: 每个任务的support样本数
            q_query: 每个任务的query样本数

        Returns:
            episode_data: {'support': {...}, 'query': {...}}
        """
        support_peptides = []
        support_peptide_lens = []
        support_task_idxs = []
        support_labels = []

        query_peptides = []
        query_peptide_lens = []
        query_task_idxs = []
        query_labels = []

        # 为每个任务采样数据
        for task_idx, task_name in enumerate(task_names):
            task_data = self.tasks[task_name]['data']

            # 如果数据不足，跳过
            if len(task_data) < k_shot + q_query:
                continue

            # 随机采样
            sampled = task_data.sample(n=k_shot + q_query, replace=False)

            # 分割为support和query
            support_df = sampled.iloc[:k_shot]
            query_df = sampled.iloc[k_shot:k_shot + q_query]

            # 处理support set
            for _, row in support_df.iterrows():
                peptide_encoded = self.encode_peptide(row['peptide'])
                peptide_len = len(row['peptide'])

                support_peptides.append(peptide_encoded)
                support_peptide_lens.append(peptide_len)
                support_task_idxs.append(task_idx)  # 使用局部任务索引
                support_labels.append(int(row['label']))

            # 处理query set
            for _, row in query_df.iterrows():
                peptide_encoded = self.encode_peptide(row['peptide'])
                peptide_len = len(row['peptide'])

                query_peptides.append(peptide_encoded)
                query_peptide_lens.append(peptide_len)
                query_task_idxs.append(task_idx)
                query_labels.append(int(row['label']))

        # 转换为tensors并移动到device
        support_data = {
            'peptide': torch.stack(support_peptides).to(self.device),
            'peptide_len': torch.LongTensor(support_peptide_lens).to(self.device),
            'task_idx': torch.LongTensor(support_task_idxs).to(self.device),
            'label': torch.LongTensor(support_labels).to(self.device)
        }

        query_data = {
            'peptide': torch.stack(query_peptides).to(self.device),
            'peptide_len': torch.LongTensor(query_peptide_lens).to(self.device),
            'task_idx': torch.LongTensor(query_task_idxs).to(self.device),
            'label': torch.LongTensor(query_labels).to(self.device)
        }

        return {'support': support_data, 'query': query_data}

    def sample_batch_tasks(self, n_way, mode='mixed', phase2_ratio=0.5):
        """
        采样一批任务

        Args:
            n_way: 采样任务数量
            mode: 'phase1', 'phase2', 'mixed'
            phase2_ratio: mixed模式下Phase 2任务的比例

        Returns:
            task_names: 采样的任务名称列表
        """
        if mode == 'phase1':
            # 只采样Phase 1任务
            available_tasks = self.phase1_tasks
        elif mode == 'phase2':
            # 只采样Phase 2任务
            available_tasks = self.phase2_tasks
        else:  # mixed
            # 混合采样
            n_phase2 = int(n_way * phase2_ratio)
            n_phase1 = n_way - n_phase2

            # 采样Phase 1任务
            if n_phase1 > 0 and len(self.phase1_tasks) > 0:
                weights_p1 = [self.task_weights[t] for t in self.phase1_tasks]
                weights_p1 = np.array(weights_p1) / sum(weights_p1)
                phase1_sampled = np.random.choice(
                    self.phase1_tasks,
                    size=min(n_phase1, len(self.phase1_tasks)),
                    replace=False,
                    p=weights_p1
                ).tolist()
            else:
                phase1_sampled = []

            # 采样Phase 2任务
            if n_phase2 > 0 and len(self.phase2_tasks) > 0:
                weights_p2 = [self.task_weights[t] for t in self.phase2_tasks]
                weights_p2 = np.array(weights_p2) / sum(weights_p2)
                phase2_sampled = np.random.choice(
                    self.phase2_tasks,
                    size=min(n_phase2, len(self.phase2_tasks)),
                    replace=False,
                    p=weights_p2
                ).tolist()
            else:
                phase2_sampled = []

            return phase1_sampled + phase2_sampled

        # phase1或phase2模式：加权采样
        if len(available_tasks) == 0:
            raise ValueError(f"No tasks available for mode={mode}")

        weights = [self.task_weights[t] for t in available_tasks]
        weights = np.array(weights) / sum(weights)

        n_sample = min(n_way, len(available_tasks))
        sampled = np.random.choice(
            available_tasks,
            size=n_sample,
            replace=False,
            p=weights
        )

        return sampled.tolist()

    def progressive_train(self, phase1_epochs, phase2_epochs, finetune_epochs,
                          n_way=5, k_shot=10, q_query=10, episodes_per_epoch=100):
        """
        渐进式训练：Phase 1 → Phase 2 → 混合Fine-tuning

        Args:
            phase1_epochs: Phase 1预训练轮数
            phase2_epochs: Phase 2引入轮数
            finetune_epochs: 混合fine-tuning轮数
            n_way: 每个episode的任务数
            k_shot: 每个任务的support样本数
            q_query: 每个任务的query样本数
            episodes_per_epoch: 每轮的episode数量

        Returns:
            history: 训练历史记录
        """
        history = {
            'phase1': {'support_loss': [], 'query_loss': []},
            'phase2': {'support_loss': [], 'query_loss': []},
            'finetune': {'support_loss': [], 'query_loss': []}
        }

        # ========== Stage 1: Phase 1预训练 ==========
        print("\n" + "=" * 60)
        print("Stage 1: Phase 1预训练（仅HLA任务）")
        print("=" * 60)

        for epoch in range(phase1_epochs):
            epoch_support_losses = []
            epoch_query_losses = []

            pbar = tqdm(range(episodes_per_epoch), desc=f"Phase1 Epoch {epoch + 1}/{phase1_epochs}")
            for _ in pbar:
                # 采样Phase 1任务
                task_names = self.sample_batch_tasks(n_way, mode='phase1')

                # 准备episode数据
                episode_data = self._prepare_episode(task_names, k_shot, q_query)

                # 获取graph数据
                graph_data = self._get_graph_data()

                # MAML训练步骤
                metrics = self.base_trainer.meta_train_step(episode_data, graph_data)

                epoch_support_losses.append(metrics['support_loss'])
                epoch_query_losses.append(metrics['query_loss'])

                pbar.set_postfix({
                    'support_loss': f"{np.mean(epoch_support_losses):.4f}",
                    'query_loss': f"{np.mean(epoch_query_losses):.4f}"
                })

            history['phase1']['support_loss'].append(np.mean(epoch_support_losses))
            history['phase1']['query_loss'].append(np.mean(epoch_query_losses))

            print(f"Epoch {epoch + 1}: Support Loss={np.mean(epoch_support_losses):.4f}, "
                  f"Query Loss={np.mean(epoch_query_losses):.4f}")

        # ========== Stage 2: 渐进引入Phase 2 ==========
        print("\n" + "=" * 60)
        print("Stage 2: 渐进引入Phase 2（HLA×Tissue任务）")
        print("=" * 60)

        for epoch in range(phase2_epochs):
            # 逐渐增加Phase 2比例：0% → 100%
            phase2_ratio = (epoch + 1) / phase2_epochs

            epoch_support_losses = []
            epoch_query_losses = []

            pbar = tqdm(range(episodes_per_epoch),
                        desc=f"Phase2 Epoch {epoch + 1}/{phase2_epochs} (P2 ratio={phase2_ratio:.2f})")
            for _ in pbar:
                # 混合采样
                task_names = self.sample_batch_tasks(n_way, mode='mixed',
                                                     phase2_ratio=phase2_ratio)

                episode_data = self._prepare_episode(task_names, k_shot, q_query)
                graph_data = self._get_graph_data()

                metrics = self.base_trainer.meta_train_step(episode_data, graph_data)

                epoch_support_losses.append(metrics['support_loss'])
                epoch_query_losses.append(metrics['query_loss'])

                pbar.set_postfix({
                    'support_loss': f"{np.mean(epoch_support_losses):.4f}",
                    'query_loss': f"{np.mean(epoch_query_losses):.4f}"
                })

            history['phase2']['support_loss'].append(np.mean(epoch_support_losses))
            history['phase2']['query_loss'].append(np.mean(epoch_query_losses))

            print(f"Epoch {epoch + 1}: Support Loss={np.mean(epoch_support_losses):.4f}, "
                  f"Query Loss={np.mean(epoch_query_losses):.4f}")

        # ========== Stage 3: 混合Fine-tuning ==========
        print("\n" + "=" * 60)
        print("Stage 3: 混合Fine-tuning（50-50采样）")
        print("=" * 60)

        for epoch in range(finetune_epochs):
            epoch_support_losses = []
            epoch_query_losses = []

            pbar = tqdm(range(episodes_per_epoch),
                        desc=f"Finetune Epoch {epoch + 1}/{finetune_epochs}")
            for _ in pbar:
                # 50-50混合采样
                task_names = self.sample_batch_tasks(n_way, mode='mixed',
                                                     phase2_ratio=0.5)

                episode_data = self._prepare_episode(task_names, k_shot, q_query)
                graph_data = self._get_graph_data()

                metrics = self.base_trainer.meta_train_step(episode_data, graph_data)

                epoch_support_losses.append(metrics['support_loss'])
                epoch_query_losses.append(metrics['query_loss'])

                pbar.set_postfix({
                    'support_loss': f"{np.mean(epoch_support_losses):.4f}",
                    'query_loss': f"{np.mean(epoch_query_losses):.4f}"
                })

            history['finetune']['support_loss'].append(np.mean(epoch_support_losses))
            history['finetune']['query_loss'].append(np.mean(epoch_query_losses))

            print(f"Epoch {epoch + 1}: Support Loss={np.mean(epoch_support_losses):.4f}, "
                  f"Query Loss={np.mean(epoch_query_losses):.4f}")

        print("\n" + "=" * 60)
        print("✓ 渐进式训练完成！")
        print("=" * 60)

        return history

    def _get_graph_data(self):
        """获取graph数据（与你的代码接口一致）"""
        return {
            'edge_index': self.graph_wrapper.edge_index,
            'edge_weight': self.graph_wrapper.edge_weight
        }


# 测试代码
if __name__ == "__main__":
    print("Testing MAML components...")
    print("✓ MAML module loaded successfully!")
    print("  Use with actual model for full testing")