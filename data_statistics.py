import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json


def analyze_cleaned_data(data_dir='cleaned_data'):
    """
    分析清洗后的数据分布
    """
    print("=" * 80)
    print("IEDB数据统计分析")
    print("=" * 80)

    # 读取所有cleaned文件
    all_files = sorted(Path(data_dir).glob('cleaned_*_chunk_*.csv'))
    print(f"\n找到 {len(all_files)} 个数据文件")

    # 初始化统计
    all_peptides = []
    all_hlas = []
    all_tissues = []
    peptide_lengths = []

    total_samples = 0

    # 逐文件读取（避免内存爆炸）
    for file in all_files[:5]:  # 先只看前5个文件，快速分析
        print(f"  读取: {file.name}")
        df = pd.read_csv(file)

        total_samples += len(df)

        all_peptides.extend(df['Peptide'].tolist())
        all_hlas.extend(df['MHC_Restriction_Name'].tolist())
        all_tissues.extend(df['Inferred_Tissue'].tolist())
        peptide_lengths.extend(df['Peptide_Length'].tolist())

    print(f"\n读取样本数: {total_samples:,}")

    # === 1. HLA分布 ===
    print(f"\n{'=' * 80}")
    print("HLA分布统计")
    print(f"{'=' * 80}")

    hla_counts = Counter(all_hlas)
    n_unique_hlas = len(hla_counts)

    print(f"独特HLA数量: {n_unique_hlas}")
    print(f"\nTop 20 HLA:")
    for i, (hla, count) in enumerate(hla_counts.most_common(20), 1):
        pct = count / total_samples * 100
        print(f"  {i:2d}. {hla:20s}: {count:7,} ({pct:5.2f}%)")

    # === 2. 肽段长度分布 ===
    print(f"\n{'=' * 80}")
    print("肽段长度分布")
    print(f"{'=' * 80}")

    length_counts = Counter(peptide_lengths)
    for length in sorted(length_counts.keys()):
        count = length_counts[length]
        pct = count / total_samples * 100
        print(f"  {int(length):2d}-mer: {count:7,} ({pct:5.2f}%)")

    # === 3. 组织分布 ===
    print(f"\n{'=' * 80}")
    print("组织分布统计")
    print(f"{'=' * 80}")

    tissue_counts = Counter(all_tissues)
    n_unique_tissues = len(tissue_counts)

    print(f"独特组织数量: {n_unique_tissues}")
    print(f"\nTop 15 组织:")
    for i, (tissue, count) in enumerate(tissue_counts.most_common(15), 1):
        pct = count / total_samples * 100
        print(f"  {i:2d}. {tissue:20s}: {count:7,} ({pct:5.2f}%)")

    # === 4. 独特肽段 ===
    print(f"\n{'=' * 80}")
    print("肽段统计")
    print(f"{'=' * 80}")

    n_unique_peptides = len(set(all_peptides))
    print(f"独特肽段数量: {n_unique_peptides:,}")
    print(f"总样本数: {total_samples:,}")
    print(f"平均每个肽段出现次数: {total_samples / n_unique_peptides:.2f}")

    # === 5. 保存统计到JSON ===
    stats = {
        'total_samples': total_samples,
        'n_unique_hlas': n_unique_hlas,
        'n_unique_peptides': n_unique_peptides,
        'n_unique_tissues': n_unique_tissues,
        'top_20_hlas': {hla: count for hla, count in hla_counts.most_common(20)},
        'length_distribution': {int(k): v for k, v in length_counts.items()},
        'top_15_tissues': {tissue: count for tissue, count in tissue_counts.most_common(15)},
    }

    stats_file = Path(data_dir) / 'data_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n统计已保存: {stats_file}")

    return stats, hla_counts, length_counts, tissue_counts


if __name__ == "__main__":
    stats, hla_counts, length_counts, tissue_counts = analyze_cleaned_data()