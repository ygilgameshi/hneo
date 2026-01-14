import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import json


def build_phase1_dataset(
        positive_data_dir='cleaned_data',
        negative_file='negative_samples/phase1_negatives.csv',
        output_dir='phase1_dataset',
        test_size=0.15,
        val_size=0.15,
        min_samples_per_hla=10,
        random_seed=42
):
    """
    构建Phase 1训练集
    """
    print("=" * 80)
    print("构建Phase 1数据集")
    print("=" * 80)

    np.random.seed(random_seed)

    # === 1. 加载正样本 ===
    print("\n加载正样本...")

    positive_files = sorted(Path(positive_data_dir).glob('cleaned_*_chunk_*.csv'))

    df_positives_list = []
    for file in positive_files:
        # ★ 修复：只读取需要的列，并指定类型
        df = pd.read_csv(
            file,
            usecols=['Peptide', 'Peptide_Length', 'MHC_Restriction_Name'],
            dtype={
                'Peptide': str,
                'MHC_Restriction_Name': str,
                'Peptide_Length': 'Int64'
            }
        )

        df['Label'] = 1
        df['Source'] = 'iedb'
        df_positives_list.append(df)

    df_positives = pd.concat(df_positives_list, ignore_index=True)

    print(f"  正样本: {len(df_positives):,}")

    # === 2. 加载负样本 ===
    print("\n加载负样本...")

    # ★ 同样的修复
    df_negatives = pd.read_csv(
        negative_file,
        dtype={
            'Peptide': str,
            'MHC_Restriction_Name': str,
            'Peptide_Length': 'Int64',
            'Label': int,
            'Source': str
        }
    )

    print(f"  负样本: {len(df_negatives):,}")
    # === 3. 合并 ===
    print("\n合并正负样本...")

    df_all = pd.concat([df_positives, df_negatives], ignore_index=True)

    print(f"  总样本: {len(df_all):,}")
    print(f"  正样本: {(df_all['Label'] == 1).sum():,} ({(df_all['Label'] == 1).sum() / len(df_all) * 100:.2f}%)")
    print(f"  负样本: {(df_all['Label'] == 0).sum():,} ({(df_all['Label'] == 0).sum() / len(df_all) * 100:.2f}%)")

    # === 4. 过滤样本数太少的HLA ===
    print(f"\n过滤样本数少于{min_samples_per_hla}的HLA...")

    hla_counts = df_all['MHC_Restriction_Name'].value_counts()

    # 找出样本数足够的HLA
    valid_hlas = hla_counts[hla_counts >= min_samples_per_hla].index.tolist()

    print(f"  原始HLA数: {len(hla_counts)}")
    print(f"  过滤后HLA数: {len(valid_hlas)}")

    # 只保留valid HLA的样本
    df_filtered = df_all[df_all['MHC_Restriction_Name'].isin(valid_hlas)].copy()

    removed_samples = len(df_all) - len(df_filtered)
    print(f"  移除样本数: {removed_samples:,} ({removed_samples / len(df_all) * 100:.2f}%)")
    print(f"  保留样本数: {len(df_filtered):,}")

    # 更新数据
    df_all = df_filtered

    # === 5. 统计HLA分布 ===
    print("\nHLA分布（Top 10）:")
    hla_counts = df_all['MHC_Restriction_Name'].value_counts()
    for i, (hla, count) in enumerate(hla_counts.head(10).items(), 1):
        n_pos = ((df_all['MHC_Restriction_Name'] == hla) & (df_all['Label'] == 1)).sum()
        n_neg = ((df_all['MHC_Restriction_Name'] == hla) & (df_all['Label'] == 0)).sum()
        print(f"  {i:2d}. {hla}: {count:,} (pos: {n_pos:,}, neg: {n_neg:,})")

    # === 6. 检查是否所有HLA都有足够样本进行split ===
    print(f"\n检查HLA样本分布...")

    min_samples_for_split = int(1 / min(test_size, val_size)) + 1
    problematic_hlas = []

    for hla in valid_hlas:
        count = (df_all['MHC_Restriction_Name'] == hla).sum()
        if count < min_samples_for_split:
            problematic_hlas.append((hla, count))

    if problematic_hlas:
        print(f"  ⚠ 发现{len(problematic_hlas)}个HLA样本数不足以进行{test_size:.0%}/{val_size:.0%}划分:")
        for hla, count in problematic_hlas[:5]:
            print(f"    {hla}: {count} 样本")

        # 进一步过滤
        print(f"\n  进一步过滤，要求每个HLA至少{min_samples_for_split}个样本...")

        sufficient_hlas = [hla for hla in valid_hlas
                           if (df_all['MHC_Restriction_Name'] == hla).sum() >= min_samples_for_split]

        df_all = df_all[df_all['MHC_Restriction_Name'].isin(sufficient_hlas)].copy()

        print(f"  最终HLA数: {len(sufficient_hlas)}")
        print(f"  最终样本数: {len(df_all):,}")
    else:
        print(f"  ✓ 所有HLA都有足够样本")

    # === 7. 划分数据集（Stratified by HLA） ===
    print(f"\n划分数据集...")
    print(f"  Train: {100 - test_size * 100 - val_size * 100:.0f}%")
    print(f"  Val: {val_size * 100:.0f}%")
    print(f"  Test: {test_size * 100:.0f}%")

    try:
        # 先划分train和temp（val+test）
        df_train, df_temp = train_test_split(
            df_all,
            test_size=(test_size + val_size),
            stratify=df_all['MHC_Restriction_Name'],
            random_state=random_seed
        )

        # 再从temp中划分val和test
        df_val, df_test = train_test_split(
            df_temp,
            test_size=test_size / (test_size + val_size),
            stratify=df_temp['MHC_Restriction_Name'],
            random_state=random_seed
        )

        print(f"\n✓ 划分成功")

    except ValueError as e:
        print(f"\n✗ Stratified split失败: {e}")
        print(f"\n尝试使用随机划分（不保证HLA分布）...")

        # 降级方案：随机划分（不stratify）
        df_train, df_temp = train_test_split(
            df_all,
            test_size=(test_size + val_size),
            random_state=random_seed
        )

        df_val, df_test = train_test_split(
            df_temp,
            test_size=test_size / (test_size + val_size),
            random_state=random_seed
        )

        print(f"✓ 使用随机划分完成")

    print(f"\n数据集大小:")
    print(f"  Train: {len(df_train):,} (pos: {(df_train['Label'] == 1).sum():,})")
    print(f"  Val: {len(df_val):,} (pos: {(df_val['Label'] == 1).sum():,})")
    print(f"  Test: {len(df_test):,} (pos: {(df_test['Label'] == 1).sum():,})")

    # === 8. 保存 ===
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    df_train.to_csv(output_dir / 'train.csv', index=False)
    df_val.to_csv(output_dir / 'val.csv', index=False)
    df_test.to_csv(output_dir / 'test.csv', index=False)

    print(f"\n✓ 数据集已保存到: {output_dir}")

    # === 9. 保存HLA列表（用于后续任务定义） ===
    # 使用训练集的HLA列表
    all_hlas = df_train['MHC_Restriction_Name'].unique().tolist()

    # 主要HLA（样本数>100的）
    major_hlas = []
    for hla in all_hlas:
        n_samples = (df_train['MHC_Restriction_Name'] == hla).sum()
        if n_samples >= 100:
            major_hlas.append(hla)

    print(f"\n主要HLA（训练集样本数>100）: {len(major_hlas)}")

    # 显示前10个
    print(f"\nTop 10 主要HLA:")
    major_hla_counts = df_train[df_train['MHC_Restriction_Name'].isin(major_hlas)][
        'MHC_Restriction_Name'].value_counts()
    for i, (hla, count) in enumerate(major_hla_counts.head(10).items(), 1):
        print(f"  {i:2d}. {hla}: {count:,}")

    hla_info = {
        'all_hlas': all_hlas,
        'major_hlas': major_hlas,
        'n_all': len(all_hlas),
        'n_major': len(major_hlas),
        'min_samples_per_hla': min_samples_per_hla,
    }

    with open(output_dir / 'hla_list.json', 'w') as f:
        json.dump(hla_info, f, indent=2)

    print(f"\n✓ HLA列表已保存: {output_dir / 'hla_list.json'}")

    # === 10. 保存统计信息 ===
    stats = {
        'total_samples': len(df_all),
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test),
        'n_hlas': len(all_hlas),
        'n_major_hlas': len(major_hlas),
        'positive_ratio': float((df_all['Label'] == 1).sum() / len(df_all)),
    }

    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    return df_train, df_val, df_test, major_hlas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build Phase 1 dataset')
    parser.add_argument('--positive_dir', type=str, default='cleaned_data',
                        help='Directory with cleaned positive samples')
    parser.add_argument('--negative_file', type=str,
                        default='negative_samples/phase1_negatives.csv',
                        help='Negative samples CSV file')
    parser.add_argument('--output_dir', type=str, default='phase1_dataset',
                        help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.15,
                        help='Validation set size')
    parser.add_argument('--min_samples', type=int, default=10,
                        help='Minimum samples per HLA to keep')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    df_train, df_val, df_test, major_hlas = build_phase1_dataset(
        positive_data_dir=args.positive_dir,
        negative_file=args.negative_file,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        min_samples_per_hla=args.min_samples,
        random_seed=args.random_seed
    )

    print("\n" + "=" * 80)
    print("✓ 数据集构建完成！")
    print("=" * 80)