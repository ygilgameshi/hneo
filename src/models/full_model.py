"""
Phase 1完整模型

组合所有组件:
- PeptideEncoder: 编码肽段
- TaskGNN: 学习task embeddings
- TaskConditionedPredictor: task-conditioned预测
"""

import torch
import torch.nn as nn
from .peptide_encoder import PeptideEncoder
from .task_gnn import TaskGNN
from .predictor import TaskConditionedPredictor


class ImmuneAppPhase1(nn.Module):
    """
    Phase 1 完整模型: HLA-specific Presentation预测

    架构:
        Peptide → PeptideEncoder → peptide_features
                                        ↓
        Task Graph → TaskGNN → task_embeddings
                                        ↓
        (peptide_features, task_embedding) → Predictor → logits

    Args:
        n_tasks: 任务数量
        vocab_size: 氨基酸词汇表大小
        peptide_*: PeptideEncoder参数
        task_*: TaskGNN参数
        predictor_*: TaskConditionedPredictor参数

    Example:
        >>> model = ImmuneAppPhase1(n_tasks=50)
        >>> peptide = torch.randint(0, 20, (16, 15))
        >>> peptide_len = torch.randint(8, 15, (16,))
        >>> task_idx = torch.randint(0, 50, (16,))
        >>> edge_index = torch.tensor([[0, 1], [1, 0]])
        >>> edge_weight = torch.tensor([0.8, 0.8])
        >>> logits = model(peptide, peptide_len, task_idx, edge_index, edge_weight)
        >>> print(logits.shape)  # (16, 1)
    """

    def __init__(self,
                 n_tasks,
                 # PeptideEncoder参数
                 vocab_size=21,
                 peptide_embed_dim=32,
                 peptide_hidden_dim=128,
                 peptide_output_dim=64,
                 # TaskGNN参数
                 task_node_dim=64,
                 task_hidden_dim=128,
                 task_output_dim=64,
                 task_n_layers=3,
                 task_gnn_type='gcn',
                 # Predictor参数
                 predictor_hidden_dim=128,
                 predictor_n_layers=3,
                 # 通用参数
                 dropout=0.1):
        super(ImmuneAppPhase1, self).__init__()

        self.n_tasks = n_tasks

        # 1. 肽段编码器
        self.peptide_encoder = PeptideEncoder(
            vocab_size=vocab_size,
            embed_dim=peptide_embed_dim,
            hidden_dim=peptide_hidden_dim,
            output_dim=peptide_output_dim,
            dropout=dropout
        )

        # 2. Task GNN
        self.task_gnn = TaskGNN(
            n_tasks=n_tasks,
            node_feat_dim=task_node_dim,
            hidden_dim=task_hidden_dim,
            output_dim=task_output_dim,
            n_layers=task_n_layers,
            gnn_type=task_gnn_type,
            dropout=dropout
        )

        # 3. Task-conditioned预测器
        self.predictor = TaskConditionedPredictor(
            peptide_dim=peptide_output_dim,
            task_embed_dim=task_output_dim,
            hidden_dim=predictor_hidden_dim,
            n_layers=predictor_n_layers,
            dropout=dropout
        )

    def forward(self, peptide, peptide_len, task_idx, edge_index, edge_weight):
        """
        完整前向传播

        Args:
            peptide: (batch_size, max_len) 肽段序列
            peptide_len: (batch_size,) 肽段长度
            task_idx: (batch_size,) 任务索引
            edge_index: (2, num_edges) Task Graph边
            edge_weight: (num_edges,) Task Graph边权重

        Returns:
            logits: (batch_size, 1) 预测logits
        """
        # 1. 编码肽段
        peptide_features = self.peptide_encoder(peptide, peptide_len)

        # 2. 获取所有task embeddings
        all_task_embeddings = self.task_gnn(edge_index, edge_weight)

        # 3. 选择对应的task embeddings
        task_embeddings = all_task_embeddings[task_idx]

        # 4. Task-conditioned预测
        logits = self.predictor(peptide_features, task_embeddings)

        return logits

    def predict(self, peptide, peptide_len, task_idx, edge_index, edge_weight):
        """
        预测概率

        Returns:
            probs: (batch_size, 1) 概率值 [0, 1]
        """
        logits = self.forward(peptide, peptide_len, task_idx, edge_index, edge_weight)
        probs = torch.sigmoid(logits)
        return probs

    def get_parameters_count(self):
        """返回模型参数统计"""
        encoder_params = sum(p.numel() for p in self.peptide_encoder.parameters())
        gnn_params = sum(p.numel() for p in self.task_gnn.parameters())
        predictor_params = sum(p.numel() for p in self.predictor.parameters())
        total_params = encoder_params + gnn_params + predictor_params

        return {
            'peptide_encoder': encoder_params,
            'task_gnn': gnn_params,
            'predictor': predictor_params,
            'total': total_params
        }


# 测试代码
if __name__ == "__main__":
    print("Testing ImmuneAppPhase1...")

    # 模型参数
    n_tasks = 20
    batch_size = 16
    max_len = 15

    # 创建模型
    model = ImmuneAppPhase1(
        n_tasks=n_tasks,
        vocab_size=21,
        peptide_output_dim=64,
        task_output_dim=64
    )

    # 打印参数统计
    params_count = model.get_parameters_count()
    print(f"\n✓ Model created:")
    for component, count in params_count.items():
        print(f"  {component}: {count:,}")

    # 模拟输入
    peptide = torch.randint(0, 20, (batch_size, max_len))
    peptide_len = torch.randint(8, 15, (batch_size,))
    task_idx = torch.randint(0, n_tasks, (batch_size,))

    # 模拟Task Graph
    n_edges = 40
    edges = torch.randint(0, n_tasks, (2, n_edges))
    edge_weight = torch.rand(n_edges)

    # 前向传播
    logits = model(peptide, peptide_len, task_idx, edges, edge_weight)
    probs = model.predict(peptide, peptide_len, task_idx, edges, edge_weight)

    print(f"\n✓ Forward pass:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probs shape: {probs.shape}")
    print(f"  Sample logits: {logits[:3].squeeze()}")
    print(f"  Sample probs: {probs[:3].squeeze()}")

    print("\n✓ ImmuneAppPhase1 test passed!")