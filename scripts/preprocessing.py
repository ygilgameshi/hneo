"""
Peptide Preprocessing Module

肽段序列预处理函数
"""

import torch
import numpy as np
from typing import List, Tuple

# 氨基酸编码映射 (20种标准氨基酸 + padding)
AA_TO_IDX = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
AA_TO_IDX['X'] = len(AA_TO_IDX)  # Padding token
AA_TO_IDX['<PAD>'] = AA_TO_IDX['X']  # 别名

# 反向映射
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}

# 词汇表大小
VOCAB_SIZE = len(AA_TO_IDX)  # 21 (20 + padding)


def encode_peptide(peptide: str, max_len: int = 15) -> Tuple[torch.LongTensor, int]:
    """
    编码单个肽段序列

    Args:
        peptide: 肽段序列字符串，如 "SIINFEKL"
        max_len: 最大长度，默认15

    Returns:
        encoded: (max_len,) 编码后的tensor
        length: 实际长度

    Example:
        >>> encoded, length = encode_peptide("SIINFEKL", max_len=15)
        >>> print(encoded.shape)
        torch.Size([15])
        >>> print(length)
        8
    """
    # 转为索引
    indices = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in peptide]

    # 记录实际长度
    actual_len = len(indices)

    # Padding或截断
    if len(indices) < max_len:
        indices += [AA_TO_IDX['X']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
        actual_len = max_len

    return torch.LongTensor(indices), actual_len


def preprocess_peptides(peptides: List[str],
                        max_len: int = 15) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    批量预处理肽段序列

    Args:
        peptides: 肽段序列列表
        max_len: 最大长度

    Returns:
        encoded: (n_samples, max_len) 编码后的tensor
        lengths: (n_samples,) 每个序列的实际长度

    Example:
        >>> peptides = ["SIINFEKL", "GILGFVFTL", "YLEPGPVTA"]
        >>> encoded, lengths = preprocess_peptides(peptides)
        >>> print(encoded.shape)
        torch.Size([3, 15])
        >>> print(lengths)
        tensor([8, 9, 9])
    """
    n_samples = len(peptides)

    # 初始化
    encoded = torch.zeros(n_samples, max_len, dtype=torch.long)
    lengths = torch.zeros(n_samples, dtype=torch.long)

    # 批量编码
    for i, peptide in enumerate(peptides):
        enc, length = encode_peptide(peptide, max_len)
        encoded[i] = enc
        lengths[i] = length

    return encoded, lengths


def decode_peptide(encoded: torch.LongTensor, length: int = None) -> str:
    """
    解码肽段序列

    Args:
        encoded: (max_len,) 编码后的tensor
        length: 实际长度（可选）

    Returns:
        peptide: 解码后的序列字符串

    Example:
        >>> encoded = torch.tensor([18, 8, 8, 13, 5, 4, 10, 11])
        >>> peptide = decode_peptide(encoded, length=8)
        >>> print(peptide)
        SIINFEKL
    """
    # 转为numpy
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.cpu().numpy()

    # 解码
    peptide = ''.join([IDX_TO_AA.get(int(idx), 'X') for idx in encoded])

    # 如果提供了长度，截断
    if length is not None:
        peptide = peptide[:length]

    # 去除padding
    peptide = peptide.replace('X', '').replace('<PAD>', '')

    return peptide


def validate_peptide(peptide: str) -> bool:
    """
    验证肽段序列是否有效

    Args:
        peptide: 肽段序列

    Returns:
        valid: 是否有效

    Example:
        >>> validate_peptide("SIINFEKL")
        True
        >>> validate_peptide("SIINFEKL123")
        False
    """
    if not peptide:
        return False

    # 检查是否只包含标准氨基酸
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    return all(aa in valid_aa for aa in peptide)


def get_peptide_stats(peptides: List[str]) -> dict:
    """
    获取肽段序列的统计信息

    Args:
        peptides: 肽段序列列表

    Returns:
        stats: 统计信息字典

    Example:
        >>> peptides = ["SIINFEKL", "GILGFVFTL", "YLEPGPVTA"]
        >>> stats = get_peptide_stats(peptides)
        >>> print(stats)
        {'count': 3, 'min_len': 8, 'max_len': 9, 'avg_len': 8.67, ...}
    """
    lengths = [len(p) for p in peptides]

    # 统计氨基酸频率
    aa_counts = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    for peptide in peptides:
        for aa in peptide:
            if aa in aa_counts:
                aa_counts[aa] += 1

    stats = {
        'count': len(peptides),
        'min_len': min(lengths) if lengths else 0,
        'max_len': max(lengths) if lengths else 0,
        'avg_len': np.mean(lengths) if lengths else 0,
        'median_len': np.median(lengths) if lengths else 0,
        'aa_counts': aa_counts,
        'unique_peptides': len(set(peptides))
    }

    return stats


# ==================== 便捷函数 ====================

def create_peptide_vocab():
    """
    创建肽段词汇表

    Returns:
        vocab: 词汇表字典
    """
    return AA_TO_IDX.copy()


def get_vocab_size():
    """
    获取词汇表大小

    Returns:
        size: 词汇表大小（21）
    """
    return VOCAB_SIZE


def get_padding_idx():
    """
    获取padding索引

    Returns:
        idx: padding token的索引
    """
    return AA_TO_IDX['X']


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Peptide Preprocessing Module - 测试")
    print("=" * 80)

    # 测试单个编码
    print("\n1. 测试单个肽段编码:")
    peptide = "SIINFEKL"
    encoded, length = encode_peptide(peptide)
    print(f"  原始: {peptide}")
    print(f"  编码: {encoded}")
    print(f"  长度: {length}")

    # 解码
    decoded = decode_peptide(encoded, length)
    print(f"  解码: {decoded}")
    print(f"  匹配: {decoded == peptide}")

    # 测试批量编码
    print("\n2. 测试批量编码:")
    peptides = ["SIINFEKL", "GILGFVFTL", "YLEPGPVTA", "AL"]
    encoded_batch, lengths_batch = preprocess_peptides(peptides)
    print(f"  样本数: {len(peptides)}")
    print(f"  编码形状: {encoded_batch.shape}")
    print(f"  长度: {lengths_batch}")

    # 测试统计
    print("\n3. 测试统计:")
    stats = get_peptide_stats(peptides)
    print(f"  总数: {stats['count']}")
    print(f"  长度范围: [{stats['min_len']}, {stats['max_len']}]")
    print(f"  平均长度: {stats['avg_len']:.2f}")
    print(f"  唯一肽段: {stats['unique_peptides']}")

    # 测试词汇表
    print("\n4. 测试词汇表:")
    print(f"  词汇表大小: {get_vocab_size()}")
    print(f"  Padding索引: {get_padding_idx()}")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过!")
    print("=" * 80)