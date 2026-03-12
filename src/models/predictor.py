"""
Task-conditioned预测器

使用FiLM (Feature-wise Linear Modulation) 机制，
根据task embedding条件化肽段特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskConditionedPredictor(nn.Module):
    """
    Task-conditioned预测器

    使用FiLM机制：task embedding → (gamma, beta) → 调制肽段特征

    FiLM公式:
        h' = gamma(z_task) ⊙ h + beta(z_task)

    其中:
        h: 肽段特征
        z_task: task embedding
        gamma, beta: 由task embedding生成的调制参数

    Args:
        peptide_dim: 肽段特征维度
        task_embed_dim: task嵌入维度
        hidden_dim: 隐藏层维度
        n_layers: 层数
        dropout: Dropout比例

    Example:
        >>> predictor = TaskConditionedPredictor(peptide_dim=64, task_embed_dim=64)
        >>> peptide_features = torch.randn(16, 64)
        >>> task_embeddings = torch.randn(16, 64)
        >>> logits = predictor(peptide_features, task_embeddings)
        >>> print(logits.shape)  # (16, 1)
    """

    def __init__(self,
                 peptide_dim=64,
                 task_embed_dim=64,
                 hidden_dim=128,
                 n_layers=3,
                 dropout=0.1):
        super(TaskConditionedPredictor, self).__init__()

        self.peptide_dim = peptide_dim
        self.task_embed_dim = task_embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # FiLM参数生成器: task embedding → (gamma, beta)
        self.film_generators = nn.ModuleList()
        for _ in range(n_layers):
            # 每层生成 gamma 和 beta (所以是 hidden_dim * 2)
            self.film_generators.append(
                nn.Sequential(
                    nn.Linear(task_embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * 2)
                )
            )

        # 主网络层
        self.layers = nn.ModuleList()
        in_dim = peptide_dim
        for i in range(n_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        # 输出层
        self.output = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, peptide_features, task_embedding):
        """
        前向传播

        Args:
            peptide_features: (batch_size, peptide_dim) 肽段特征
            task_embedding: (batch_size, task_embed_dim) task嵌入

        Returns:
            logits: (batch_size, 1) 预测logits
        """
        x = peptide_features

        # 逐层FiLM调制
        for i in range(self.n_layers):
            # 1. 生成FiLM参数
            film_params = self.film_generators[i](task_embedding)
            gamma, beta = torch.chunk(film_params, 2, dim=1)  # 分割成gamma和beta

            # 2. 主网络层
            x = self.layers[i](x)

            # 3. FiLM调制
            x = gamma * x + beta

            # 4. 激活和正则化
            x = F.relu(x)
            x = self.dropout(x)

        # 输出层
        logits = self.output(x)

        return logits

    def predict_proba(self, peptide_features, task_embedding):
        """
        预测概率

        Returns:
            probs: (batch_size, 1) 概率值 [0, 1]
        """
        logits = self.forward(peptide_features, task_embedding)
        probs = torch.sigmoid(logits)
        return probs


# 测试代码
if __name__ == "__main__":
    print("Testing TaskConditionedPredictor...")

    # 创建预测器
    predictor = TaskConditionedPredictor(
        peptide_dim=64,
        task_embed_dim=64,
        hidden_dim=128,
        n_layers=3
    )

    print(f"✓ Created predictor with {sum(p.numel() for p in predictor.parameters()):,} parameters")

    # 模拟输入
    batch_size = 16
    peptide_features = torch.randn(batch_size, 64)
    task_embeddings = torch.randn(batch_size, 64)

    # 前向传播
    logits = predictor(peptide_features, task_embeddings)
    probs = predictor.predict_proba(peptide_features, task_embeddings)

    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Probs shape: {probs.shape}")
    print(f"✓ Sample logits: {logits[:3].squeeze()}")
    print(f"✓ Sample probs: {probs[:3].squeeze()}")

    print("\n✓ TaskConditionedPredictor test passed!")