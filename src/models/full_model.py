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
    def __init__(self,
                 mode_config: ModeConfig,
                 n_tasks: int,
                 n_tissues: int = None,
                 # ── 新增消融 flags ──────────────────
                 use_gnn: bool = True,   # False = −GNN 消融
                 use_film: bool = True,  # False = −FiLM 消融
                 # ────────────────────────────────────
                 vocab_size: int = 21,
                 peptide_embed_dim: int = 32,
                 peptide_hidden_dim: int = 128,
                 peptide_output_dim: int = 64,
                 task_node_dim: int = 64,
                 task_hidden_dim: int = 128,
                 task_output_dim: int = 64,
                 task_n_layers: int = 3,
                 task_gnn_type: str = 'gcn',
                 tissue_embed_dim: int = 32,
                 predictor_hidden_dim: int = 128,
                 predictor_n_layers: int = 3,
                 dropout: float = 0.1):

        super().__init__()
        self.mode = mode_config.mode
        self.mode_config = mode_config
        self.n_tasks = n_tasks
        self.n_tissues = n_tissues
        self.use_gnn = use_gnn    # ← 新增
        self.use_film = use_film  # ← 新增

        # ── Peptide Encoder（所有变体共享，不变）──
        self.peptide_encoder = PeptideEncoder(
            vocab_size=vocab_size,
            embed_dim=peptide_embed_dim,
            hidden_dim=peptide_hidden_dim,
            output_dim=peptide_output_dim,
            dropout=dropout
        )

        # ── Task GNN（始终初始化，use_gnn=False 时跳过传播）──
        self.task_gnn = TaskGNN(
            n_tasks=n_tasks,
            node_feat_dim=task_node_dim,
            hidden_dim=task_hidden_dim,
            output_dim=task_output_dim,
            n_layers=task_n_layers,
            gnn_type=task_gnn_type,
            dropout=dropout
        )

        # ── Mode 2 专用：Tissue Embedding + FiLM ──
        if self.mode == TrainingMode.HLA_TISSUE:
            if n_tissues is None:
                raise ValueError("Mode 2 requires n_tissues")
            self.tissue_embedder = nn.Embedding(n_tissues, tissue_embed_dim)
            if self.use_film:
                self.tissue_film = FiLMLayer(
                    feature_dim=task_output_dim,
                    condition_dim=tissue_embed_dim
                )
            else:
                # −FiLM 消融：用线性层把 tissue embedding 投影后直接加到 task emb
                self.tissue_concat_proj = nn.Linear(
                    task_output_dim + tissue_embed_dim, task_output_dim
                )

        # ── Predictor（所有变体共享，不变）──
        self.predictor = TaskConditionedPredictor(
            peptide_dim=peptide_output_dim,
            task_embed_dim=task_output_dim,
            hidden_dim=predictor_hidden_dim,
            n_layers=predictor_n_layers,
            dropout=dropout
        )

        ablation_info = []
        if not use_gnn:  ablation_info.append("−GNN")
        if not use_film: ablation_info.append("−FiLM")
        if ablation_info:
            print(f"  [Ablation] {', '.join(ablation_info)}")



    def forward(self, batch, graph_data):
        # 1. Peptide encoding（不变）
        peptide_features = self.peptide_encoder(
            batch['peptide'], batch['peptide_len']
        )

        # 2. Task embedding
        if self.use_gnn:
            # 正常路径：GNN 在图上传播
            all_task_emb = self.task_gnn(
                graph_data['edge_index'],
                graph_data['edge_weight']
            )
        else:
            # −GNN 消融：直接用 node embedding 权重，不做图传播
            all_task_emb = self.task_gnn.node_embedding.weight

        task_embeddings = all_task_emb[batch['task_idx'].squeeze()]

        # 3. Tissue conditioning（Mode 2）
        if self.mode == TrainingMode.HLA_TISSUE:
            tissue_emb = self.tissue_embedder(batch['tissue_idx'].squeeze())
            if self.use_film:
                # 正常路径：FiLM 调制
                task_embeddings = self.tissue_film(task_embeddings, tissue_emb)
            else:
                # −FiLM 消融：加性融合（additive）
                task_embeddings = self.tissue_concat_proj(
                    torch.cat([task_embeddings, tissue_emb], dim=-1)
                )

        # 4. Prediction
        return self.predictor(peptide_features, task_embeddings)

    def get_parameters_count(self):
        params = {
            'peptide_encoder': sum(p.numel() for p in self.peptide_encoder.parameters()),
            'task_gnn': sum(p.numel() for p in self.task_gnn.parameters()),
            'predictor': sum(p.numel() for p in self.predictor.parameters()),
        }
        if self.mode == TrainingMode.HLA_TISSUE:
            params['tissue_embedder'] = sum(p.numel() for p in self.tissue_embedder.parameters())
            if self.use_film:
                params['tissue_film'] = sum(p.numel() for p in self.tissue_film.parameters())
            else:
                params['tissue_concat_proj'] = sum(
                    p.numel() for p in self.tissue_concat_proj.parameters()
                )

        params['total'] = sum(params.values())
        return params

    def predict(self, batch, graph_data):
        """
        预测概率

        Returns:
            probs: (batch_size, 1) 概率值 [0, 1]
        """
        logits = self.forward(batch, graph_data)
        probs = torch.sigmoid(logits)
        return probs

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