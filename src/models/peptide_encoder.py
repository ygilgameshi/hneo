"""
肽段序列编码器

使用 Embedding + CNN + LSTM 架构编码可变长度的肽段序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeptideEncoder(nn.Module):
    """
    肽段序列编码器

    架构:
        Input: 肽段序列 (batch_size, max_len)
        ↓
        Embedding (amino acid → vector)
        ↓
        CNN (提取局部特征)
        ↓
        Bi-LSTM (提取序列依赖)
        ↓
        Output: 肽段表示 (batch_size, output_dim)

    Args:
        vocab_size: 氨基酸词汇表大小 (默认21: 20种氨基酸 + padding)
        embed_dim: 氨基酸嵌入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出表示维度
        dropout: Dropout比例

    Example:
        >>> encoder = PeptideEncoder(vocab_size=21, output_dim=64)
        >>> peptide = torch.randint(0, 20, (16, 15))  # batch_size=16, max_len=15
        >>> peptide_len = torch.randint(8, 15, (16,))
        >>> features = encoder(peptide, peptide_len)
        >>> print(features.shape)  # (16, 64)
    """

    def __init__(self,
                 vocab_size=21,
                 embed_dim=32,
                 hidden_dim=128,
                 output_dim=64,
                 dropout=0.1):
        super(PeptideEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Embedding层 (padding_idx=20 对应 'X')
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)

        # CNN层 (提取n-gram特征)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Bi-LSTM层 (提取长程依赖)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,  # 双向，所以hidden_dim/2
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )

        # 输出投影层
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, peptide, peptide_len=None):
        """
        前向传播

        Args:
            peptide: (batch_size, max_len) 肽段序列索引
            peptide_len: (batch_size,) 实际长度 (可选)

        Returns:
            features: (batch_size, output_dim) 肽段表示
        """
        batch_size = peptide.size(0)

        # 1. Embedding
        x = self.embedding(peptide)  # (batch, max_len, embed_dim)

        # 2. CNN
        x = x.transpose(1, 2)  # (batch, embed_dim, max_len)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = x.transpose(1, 2)  # (batch, max_len, hidden_dim)

        # 3. LSTM
        if peptide_len is not None:
            # Pack sequence (处理变长序列)
            x = nn.utils.rnn.pack_padded_sequence(
                x,
                peptide_len.cpu().reshape(-1),
                batch_first=True,
                enforce_sorted=False
            )

        x, (h_n, c_n) = self.lstm(x)

        # h_n: (num_layers * num_directions, batch, hidden_dim/2)
        # 取最后一层的正向和反向hidden state
        h_forward = h_n[-2, :, :]  # (batch, hidden_dim/2)
        h_backward = h_n[-1, :, :]  # (batch, hidden_dim/2)
        h = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_dim)

        # 4. 输出投影
        features = self.fc(h)  # (batch, output_dim)
        features = self.dropout(features)

        return features

    def get_embedding_weights(self):
        """返回氨基酸嵌入权重 (用于可视化)"""
        return self.embedding.weight.data


# 测试代码
if __name__ == "__main__":
    print("Testing PeptideEncoder...")

    # 创建编码器
    encoder = PeptideEncoder(
        vocab_size=21,
        embed_dim=32,
        hidden_dim=128,
        output_dim=64
    )

    # 模拟输入
    batch_size = 16
    max_len = 15

    peptide = torch.randint(0, 20, (batch_size, max_len))
    peptide_len = torch.randint(8, 15, (batch_size,))

    # 前向传播
    features = encoder(peptide, peptide_len)

    print(f"✓ Input shape: {peptide.shape}")
    print(f"✓ Output shape: {features.shape}")
    print(f"✓ Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # 测试不提供长度信息
    features_no_len = encoder(peptide)
    print(f"✓ Without length info: {features_no_len.shape}")

    print("\n✓ PeptideEncoder test passed!")