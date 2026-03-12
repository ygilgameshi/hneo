"""
Mode-Aware ImmuneApp Model

支持三种模式:
- Mode 1 (HLA_ONLY): HLA-specific presentation prediction
- Mode 2 (HLA_TISSUE): HLA×Tissue-specific prediction with tissue embedding
- Mode 3 (HYBRID): 混合模式 (接口预留)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.peptide_encoder import PeptideEncoder
from src.models.task_gnn import TaskGNN
from src.models.predictor import TaskConditionedPredictor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config.mode_config import ModeConfig, TrainingMode


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer

    用于Mode 2将tissue信息融合到task embedding中

    FiLM公式:
        output = gamma(condition) * features + beta(condition)

    Args:
        feature_dim: 特征维度
        condition_dim: 条件维度 (tissue embedding dim)
    """

    def __init__(self, feature_dim, condition_dim):
        super().__init__()

        # 生成gamma和beta
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, features, condition):
        """
        Args:
            features: (batch_size, feature_dim) task embeddings
            condition: (batch_size, condition_dim) tissue embeddings

        Returns:
            modulated: (batch_size, feature_dim) FiLM调制后的features
        """
        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)

        return gamma * features + beta


class ImmuneAppModel(nn.Module):
    """
    Mode-Aware ImmuneApp Model

    根据mode配置自动调整模型架构:
    - Mode 1: Peptide Encoder + Task GNN + Predictor
    - Mode 2: Peptide Encoder + Task GNN + Tissue Embedder + FiLM + Predictor
    - Mode 3: 待定

    Args:
        mode_config: ModeConfig实例
        n_tasks: 任务数量
        n_tissues: 组织数量 (Mode 2需要)
        vocab_size: 氨基酸词汇表大小
        peptide_*: PeptideEncoder参数
        task_*: TaskGNN参数
        tissue_*: Tissue embedder参数 (Mode 2)
        predictor_*: Predictor参数
        dropout: Dropout比例

    Example:
        >>> config = create_mode1_config()
        >>> model = ImmuneAppModel(config, n_tasks=50)
        >>> # Mode 1 forward
        # >>> batch = {'peptide': ..., 'peptide_len': ..., 'task_idx': ...}
        >>> logits = model(batch, graph_data)

        >>> config2 = create_mode2_config()
        >>> model2 = ImmuneAppModel(config2, n_tasks=100, n_tissues=10)
        >>> # Mode 2 forward
        # >>> batch2 = {'peptide': ..., 'task_idx': ..., 'tissue_idx': ...}
        >>> logits2 = model2(batch2, graph_data)
    """

    def __init__(self,
                 mode_config: ModeConfig,
                 n_tasks: int,
                 n_tissues: int = None,
                 # PeptideEncoder参数
                 vocab_size: int = 21,
                 peptide_embed_dim: int = 32,
                 peptide_hidden_dim: int = 128,
                 peptide_output_dim: int = 64,
                 # TaskGNN参数
                 task_node_dim: int = 64,
                 task_hidden_dim: int = 128,
                 task_output_dim: int = 64,
                 task_n_layers: int = 3,
                 task_gnn_type: str = 'gcn',
                 # Tissue Embedder参数 (Mode 2)
                 tissue_embed_dim: int = 32,
                 # Predictor参数
                 predictor_hidden_dim: int = 128,
                 predictor_n_layers: int = 3,
                 # 通用参数
                 dropout: float = 0.1):

        super().__init__()

        self.mode = mode_config.mode
        self.mode_config = mode_config
        self.n_tasks = n_tasks
        self.n_tissues = n_tissues

        # ========== 共享组件 (所有mode) ==========

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

        # ========== Mode 2专用组件 ==========

        if self.mode == TrainingMode.HLA_TISSUE:
            if n_tissues is None:
                raise ValueError("Mode 2 (HLA_TISSUE) requires n_tissues parameter")

            # Tissue embedding layer
            self.tissue_embedder = nn.Embedding(
                n_tissues,
                tissue_embed_dim,
                padding_idx=None
            )

            # FiLM layer: 用tissue信息调制task embedding
            self.tissue_film = FiLMLayer(
                feature_dim=task_output_dim,
                condition_dim=tissue_embed_dim
            )

            print(f"  Mode 2 components initialized:")
            print(f"    Tissue embedder: {n_tissues} tissues -> {tissue_embed_dim}D")
            print(f"    FiLM layer: {task_output_dim}D (task) × {tissue_embed_dim}D (tissue)")

        # ========== Mode 3组件 (接口预留) ==========

        if self.mode == TrainingMode.HYBRID:
            # 混合模式的特殊组件
            # TODO: 实现混合模式逻辑
            print(f"  Mode 3 (Hybrid) interface reserved")

        # ========== 预测器 (所有mode共享) ==========

        # 3. Task-conditioned预测器
        self.predictor = TaskConditionedPredictor(
            peptide_dim=peptide_output_dim,
            task_embed_dim=task_output_dim,
            hidden_dim=predictor_hidden_dim,
            n_layers=predictor_n_layers,
            dropout=dropout
        )

    def forward(self, batch, graph_data):
        """
        前向传播

        Args:
            batch: dict with keys:
                - 'peptide': (batch_size, max_len)
                - 'peptide_len': (batch_size,)
                - 'task_idx': (batch_size,)
                - 'tissue_idx': (batch_size,) [Mode 2 only]
            graph_data: dict with keys:
                - 'edge_index': (2, num_edges)
                - 'edge_weight': (num_edges,)

        Returns:
            logits: (batch_size, 1) 预测logits
        """
        # 1. 编码peptide
        peptide_features = self.peptide_encoder(
            batch['peptide'],
            batch['peptide_len']
        )

        # 2. 获取所有task embeddings通过GNN
        all_task_embeddings = self.task_gnn(
            graph_data['edge_index'],
            graph_data['edge_weight']
        )

        # 3. 选择当前batch的task embeddings
        task_embeddings = all_task_embeddings[batch['task_idx'].squeeze()]

        # 4. Mode 2: 融合tissue信息
        if self.mode == TrainingMode.HLA_TISSUE:
            # 获取tissue embeddings
            tissue_embeddings = self.tissue_embedder(batch['tissue_idx'].squeeze())

            # 用FiLM融合: task_emb = gamma * task_emb + beta
            task_embeddings = self.tissue_film(task_embeddings, tissue_embeddings)

        # 5. Task-conditioned预测
        logits = self.predictor(peptide_features, task_embeddings)

        return logits

    def predict(self, batch, graph_data):
        """
        预测概率

        Returns:
            probs: (batch_size, 1) 概率值 [0, 1]
        """
        logits = self.forward(batch, graph_data)
        probs = torch.sigmoid(logits)
        return probs

    def get_parameters_count(self):
        """返回模型参数统计"""
        params = {
            'peptide_encoder': sum(p.numel() for p in self.peptide_encoder.parameters()),
            'task_gnn': sum(p.numel() for p in self.task_gnn.parameters()),
            'predictor': sum(p.numel() for p in self.predictor.parameters()),
        }

        # Mode 2额外参数
        if self.mode == TrainingMode.HLA_TISSUE:
            params['tissue_embedder'] = sum(p.numel() for p in self.tissue_embedder.parameters())
            params['tissue_film'] = sum(p.numel() for p in self.tissue_film.parameters())

        params['total'] = sum(params.values())

        return params


# 向后兼容: 保留旧的类名
ImmuneAppPhase1 = ImmuneAppModel


if __name__ == "__main__":
    print("="*80)
    print("Testing ImmuneAppModel")
    print("="*80)

    from src.config.mode_config import create_mode1_config, create_mode2_config

    n_tasks = 20
    batch_size = 16
    max_len = 15

    # ========== Mode 1测试 ==========
    print("\n" + "="*80)
    print("Testing Mode 1 (HLA_ONLY)")
    print("="*80)

    config1 = create_mode1_config()
    model1 = ImmuneAppModel(
        mode_config=config1,
        n_tasks=n_tasks,
        vocab_size=21,
        peptide_output_dim=64,
        task_output_dim=64
    )

    # 打印参数统计
    params1 = model1.get_parameters_count()
    print(f"\n✓ Mode 1 Model created:")
    for component, count in params1.items():
        print(f"  {component}: {count:,}")

    # 模拟Mode 1输入
    batch1 = {
        'peptide': torch.randint(0, 20, (batch_size, max_len)),
        'peptide_len': torch.randint(8, 15, (batch_size,)),
        'task_idx': torch.randint(0, n_tasks, (batch_size,))
    }

    graph_data = {
        'edge_index': torch.randint(0, n_tasks, (2, 40)),
        'edge_weight': torch.rand(40)
    }

    # 前向传播
    logits1 = model1(batch1, graph_data)
    probs1 = model1.predict(batch1, graph_data)

    print(f"\n✓ Mode 1 Forward pass:")
    print(f"  Input peptide shape: {batch1['peptide'].shape}")
    print(f"  Output logits shape: {logits1.shape}")
    print(f"  Output probs shape: {probs1.shape}")
    print(f"  Sample probs: {probs1[:3].squeeze()}")

    # ========== Mode 2测试 ==========
    print("\n" + "="*80)
    print("Testing Mode 2 (HLA_TISSUE)")
    print("="*80)

    config2 = create_mode2_config()
    n_tissues = 10

    model2 = ImmuneAppModel(
        mode_config=config2,
        n_tasks=n_tasks,
        n_tissues=n_tissues,
        vocab_size=21,
        peptide_output_dim=64,
        task_output_dim=64,
        tissue_embed_dim=32
    )

    # 打印参数统计
    params2 = model2.get_parameters_count()
    print(f"\n✓ Mode 2 Model created:")
    for component, count in params2.items():
        print(f"  {component}: {count:,}")

    # 模拟Mode 2输入 (包含tissue_idx)
    batch2 = {
        'peptide': torch.randint(0, 20, (batch_size, max_len)),
        'peptide_len': torch.randint(8, 15, (batch_size,)),
        'task_idx': torch.randint(0, n_tasks, (batch_size,)),
        'tissue_idx': torch.randint(0, n_tissues, (batch_size,))  # Mode 2专属
    }

    # 前向传播
    logits2 = model2(batch2, graph_data)
    probs2 = model2.predict(batch2, graph_data)

    print(f"\n✓ Mode 2 Forward pass:")
    print(f"  Input peptide shape: {batch2['peptide'].shape}")
    print(f"  Input tissue_idx shape: {batch2['tissue_idx'].shape}")
    print(f"  Output logits shape: {logits2.shape}")
    print(f"  Output probs shape: {probs2.shape}")
    print(f"  Sample probs: {probs2[:3].squeeze()}")

    # ========== 参数对比 ==========
    print("\n" + "="*80)
    print("Parameter Comparison")
    print("="*80)

    print(f"\nMode 1 total: {params1['total']:,}")
    print(f"Mode 2 total: {params2['total']:,}")
    print(f"Mode 2 overhead: {params2['total'] - params1['total']:,} (+{(params2['total']/params1['total']-1)*100:.1f}%)")

    print("\n✓ All tests passed!")