"""
数据预处理脚本 - 适配Mode系统

处理分块的CSV文件，准备用于Mode 1/2训练
"""

import pandas as pd
from pathlib import Path
import numpy as np
import json
import argparse


def load_chunked_positives(positive_data_dir='data/cleaned_data', tissue_source='Host'):
    """
    加载所有chunk的正样本

    Args:
        positive_data_dir: 数据目录
        tissue_source: Tissue列名 (尝试读取)

    Returns:
        df_positives: 合并后的正样本DataFrame
    """
    print("="*80)
    print("Step 1: Loading Positive Samples")
    print("="*80)

    positive_files = sorted(Path(positive_data_dir).glob('cleaned_*_chunk_*.csv'))

    if not positive_files:
        raise FileNotFoundError(f"No chunk files found in {positive_data_dir}")

    print(f"\n✓ Found {len(positive_files)} chunk files")

    # 先读取第一个文件检查可用列
    first_file = positive_files[0]
    first_df = pd.read_csv(first_file, nrows=0)
    available_columns = first_df.columns.tolist()

    print(f"\n检测到的列: {available_columns}")

    # 确定要读取的列
    required_cols = ['Peptide', 'MHC_Restriction_Name']
    optional_cols = ['Peptide_Length', tissue_source, 'Host', 'Inferred_Tissue']

    cols_to_read = required_cols.copy()
    for col in optional_cols:
        if col in available_columns:
            cols_to_read.append(col)

    print(f"将要读取的列: {cols_to_read}")

    # 检查tissue列
    has_tissue = False
    tissue_col_name = None
    for possible_tissue_col in [tissue_source, 'Host', 'Inferred_Tissue']:
        if possible_tissue_col in available_columns:
            has_tissue = True
            tissue_col_name = possible_tissue_col
            print(f"\n✓ 检测到Tissue列: {tissue_col_name}")
            break

    if not has_tissue:
        print(f"\n⚠ 未检测到Tissue列")
        print(f"  尝试过的列名: {tissue_source}, Host, Inferred_Tissue")
        print(f"  → 只能使用 Mode 1 (HLA-only)")

    # 读取所有文件
    df_list = []
    for file in positive_files:
        df = pd.read_csv(file, usecols=cols_to_read)
        df_list.append(df)
        print(f"  {file.name}: {len(df):,} samples")

    df_positives = pd.concat(df_list, ignore_index=True)

    # 添加Label和Source
    df_positives['Label'] = 1
    df_positives['Source'] = 'iedb'

    print(f"\n✓ Total positive samples: {len(df_positives):,}")

    # 如果有tissue列,显示tissue统计
    if has_tissue and tissue_col_name in df_positives.columns:
        tissue_counts = df_positives[tissue_col_name].value_counts()
        n_tissues = len(tissue_counts)

        print(f"\n✓ Tissue信息统计:")
        print(f"  Tissue列名: {tissue_col_name}")
        print(f"  Unique tissues: {n_tissues}")

        if n_tissues <= 10:
            print(f"\n  Tissue分布:")
            for tissue, count in tissue_counts.items():
                pct = count / len(df_positives) * 100
                print(f"    {tissue}: {count:,} ({pct:.1f}%)")
        else:
            print(f"\n  Tissue分布 (Top 10):")
            for tissue, count in tissue_counts.head(10).items():
                pct = count / len(df_positives) * 100
                print(f"    {tissue}: {count:,} ({pct:.1f}%)")
            print(f"    ... 还有 {n_tissues-10} 个tissues")

        # 判断tissue质量
        unknown_count = df_positives[tissue_col_name].isna().sum()
        unknown_pct = unknown_count / len(df_positives) * 100

        if unknown_pct > 50:
            print(f"\n  ⚠ 警告: {unknown_pct:.1f}% 的样本tissue缺失")
            print(f"  → Tissue信息不充分,建议使用 Mode 1")
        elif n_tissues == 1:
            print(f"\n  ⚠ 警告: 只有1个tissue")
            print(f"  → 无法区分tissue差异,建议使用 Mode 1")
        elif n_tissues < 5:
            print(f"\n  ⚠ 注意: Tissue类型较少 (只有{n_tissues}个)")
            print(f"  → Mode 2的tissue信息可能不够丰富")
        else:
            print(f"\n  ✓ Tissue信息充分 (有{n_tissues}个不同tissues)")
            print(f"  → 可以使用 Mode 2")

    return df_positives


def prepare_for_mode_system(
    df_positives,
    negative_file=None,
    output_file='data/processed_data.tsv',
    add_tissue=False,
    tissue_source='Host'
):
    """
    准备用于Mode系统的数据

    Args:
        df_positives: 正样本DataFrame
        negative_file: 负样本文件路径 (可选)
        output_file: 输出文件
        add_tissue: 是否添加tissue信息 (Mode 2需要)
        tissue_source: Tissue列名

    Returns:
        df: 处理后的DataFrame
    """
    print("\n" + "="*80)
    print("Step 2: Preparing for Mode System")
    print("="*80)

    # ========== 1. 处理负样本 (如果提供) ==========
    if negative_file and Path(negative_file).exists():
        print(f"\n加载负样本: {negative_file}")

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

        print(f"  Negative samples: {len(df_negatives):,}")

        # 合并正负样本
        df = pd.concat([df_positives, df_negatives], ignore_index=True)

        print(f"\n✓ Combined:")
        print(f"  Total: {len(df):,}")
        print(f"  Positive: {(df['Label']==1).sum():,} ({(df['Label']==1).sum()/len(df)*100:.1f}%)")
        print(f"  Negative: {(df['Label']==0).sum():,} ({(df['Label']==0).sum()/len(df)*100:.1f}%)")
    else:
        print("\n⚠ No negative file provided, using positives only")
        print("  (Negatives will be generated by NegativeSampler later)")
        df = df_positives.copy()

    # ========== 2. 重命名列为标准格式 ==========
    print("\n重命名列为标准格式...")

    # Mode系统使用的标准列名
    df = df.rename(columns={
        'MHC_Restriction_Name': 'hla',
        'Peptide': 'peptide',
        'Label': 'label'
    })

    # 删除Peptide_Length (会在训练时自动计算)
    if 'Peptide_Length' in df.columns:
        df = df.drop(columns=['Peptide_Length'])

    print("  ✓ Renamed to: hla, peptide, label")

    # ========== 3. Mode 2: 添加tissue信息 ==========
    if add_tissue:
        print(f"\n{'='*80}")
        print(f"Mode 2: Tissue信息处理")
        print(f"{'='*80}")

        # 检查是否有tissue列
        possible_tissue_cols = [tissue_source, 'Host', 'Inferred_Tissue', 'tissue']
        found_tissue_col = None

        for col in possible_tissue_cols:
            if col in df.columns:
                found_tissue_col = col
                break

        if found_tissue_col is None:
            print(f"\n❌ 错误: 数据中没有tissue信息!")
            print(f"  尝试过的列名: {possible_tissue_cols}")
            print(f"\n💡 建议:")
            print(f"  1. 检查原始数据是否包含tissue列")
            print(f"  2. 或者使用 Mode 1 (HLA-only):")
            print(f"     python scripts/preprocess_data.py --mode mode1 ...")
            raise ValueError("No tissue column found in data")

        # 重命名为标准tissue列
        if found_tissue_col != 'tissue':
            df = df.rename(columns={found_tissue_col: 'tissue'})
            print(f"\n✓ 使用 '{found_tissue_col}' 作为tissue列")

        # 填充缺失值
        missing_count = df['tissue'].isna().sum()
        if missing_count > 0:
            print(f"\n  处理缺失值: {missing_count:,} ({missing_count/len(df)*100:.1f}%)")
            df['tissue'] = df['tissue'].fillna('Unknown')

        # Tissue统计
        tissue_counts = df['tissue'].value_counts()
        n_tissues = len(tissue_counts)

        print(f"\n✓ Tissue分布 (共{n_tissues}个):")
        if n_tissues <= 10:
            for tissue, count in tissue_counts.items():
                print(f"    {tissue}: {count:,} ({count/len(df)*100:.1f}%)")
        else:
            for tissue, count in tissue_counts.head(10).items():
                print(f"    {tissue}: {count:,} ({count/len(df)*100:.1f}%)")
            print(f"    ... 还有 {n_tissues-10} 个tissues")

        # ========== 重要诊断 ==========
        print(f"\n{'='*80}")
        print(f"Mode 2 适用性诊断")
        print(f"{'='*80}")

        if n_tissues == 1 and 'Unknown' in tissue_counts.index:
            print(f"\n❌ 诊断结果: 不适合使用 Mode 2")
            print(f"  原因: 所有样本的tissue都是'Unknown'")
            print(f"  → 数据中实际没有tissue信息")
            print(f"\n💡 强烈建议:")
            print(f"  重新运行预处理,使用 Mode 1:")
            print(f"  python scripts/preprocess_data.py \\")
            print(f"      --positive_dir {args.positive_dir} \\")
            print(f"      --output_file data/mode1_data.tsv \\")
            print(f"      --mode mode1")
            print(f"\n  然后训练:")
            print(f"  python scripts/train_mode1.py \\")
            print(f"      --data_file data/mode1_data.tsv \\")
            print(f"      --output_dir output/mode1_standard \\")
            print(f"      --n_epochs 50")

            # 询问是否继续
            print(f"\n⚠ 警告: 继续使用Mode 2将无法学到tissue特异性")
            print(f"  按 Ctrl+C 终止, 或等待5秒自动继续...")
            import time
            time.sleep(5)

        elif n_tissues <= 3:
            print(f"\n⚠ 诊断结果: Mode 2可用,但tissue信息有限")
            print(f"  Tissue类型数: {n_tissues}")
            print(f"  → Tissue区分能力有限")
            print(f"\n💡 建议:")
            print(f"  可以尝试 Mode 2, 但效果可能不如 Mode 1")

        else:
            print(f"\n✓ 诊断结果: 适合使用 Mode 2")
            print(f"  Tissue类型数: {n_tissues}")
            print(f"  → 有足够的tissue多样性")

            # 检查HLA×Tissue组合
            n_combinations = df[['hla', 'tissue']].drop_duplicates().shape[0]
            print(f"  HLA×Tissue组合数: {n_combinations}")

            combo_sizes = df.groupby(['hla', 'tissue']).size()
            min_size = combo_sizes.min()
            median_size = combo_sizes.median()

            print(f"  组合样本数: min={min_size}, median={median_size:.0f}")

            if min_size < 5:
                print(f"\n  ⚠ 某些组合样本数较少 (<5)")
                print(f"  → 训练时建议: --min_samples 5")
            else:
                print(f"\n  ✓ 数据充足")
                print(f"  → 训练时建议: --min_samples 10")

    # ========== 4. 清理数据 ==========
    print("\n清理数据...")

    # 删除重复
    before = len(df)
    df = df.drop_duplicates(subset=['peptide', 'hla'])
    after = len(df)
    if before != after:
        print(f"  ✓ Removed {before-after:,} duplicates")

    # 删除空值
    before = len(df)
    df = df.dropna(subset=['peptide', 'hla', 'label'])
    after = len(df)
    if before != after:
        print(f"  ✓ Removed {before-after:,} rows with missing values")

    # 确保label是整数
    df['label'] = df['label'].astype(int)

    # ========== 5. 保存 ==========
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 保存为TSV (Mode训练脚本的标准格式)
    df.to_csv(output_file, sep='\t', index=False)

    print(f"\n✓ Saved to: {output_file}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Samples: {len(df):,}")

    # 统计信息
    stats = {
        'total_samples': len(df),
        'positive_samples': int((df['label']==1).sum()),
        'negative_samples': int((df['label']==0).sum()),
        'n_hlas': int(df['hla'].nunique()),
        'columns': df.columns.tolist()
    }

    if add_tissue:
        stats['n_tissues'] = int(df['tissue'].nunique())

    # 保存统计
    stats_file = output_file.parent / f'{output_file.stem}_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Stats saved to: {stats_file}")

    return df


def main(args):
    """主流程"""

    print("="*80)
    print("Data Preprocessing for Mode System")
    print("="*80)
    print(f"  Positive dir: {args.positive_dir}")
    print(f"  Negative file: {args.negative_file}")
    print(f"  Output file: {args.output_file}")
    print(f"  Mode: {args.mode}")
    print("="*80)

    # 1. 加载正样本
    df_positives = load_chunked_positives(args.positive_dir, args.tissue_source)

    # 2. 准备数据
    add_tissue = (args.mode == 'mode2')

    df = prepare_for_mode_system(
        df_positives,
        negative_file=args.negative_file,
        output_file=args.output_file,
        add_tissue=add_tissue,
        tissue_source=args.tissue_source
    )

    # 3. 打印HLA统计
    print("\n" + "="*80)
    print("HLA Distribution (Top 20)")
    print("="*80)

    hla_counts = df['hla'].value_counts()
    for i, (hla, count) in enumerate(hla_counts.head(20).items(), 1):
        n_pos = ((df['hla']==hla) & (df['label']==1)).sum()
        n_neg = ((df['hla']==hla) & (df['label']==0)).sum()
        print(f"  {i:2d}. {hla:20s}: {count:6,} (pos: {n_pos:5,}, neg: {n_neg:5,})")

    # 4. 下一步提示
    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print(f"\n✓ Data preprocessing completed!")
    print(f"\n现在可以运行训练脚本:")

    if args.mode == 'mode1':
        print(f"\n  # Mode 1 - Standard训练")
        print(f"  python scripts/train_mode1.py \\")
        print(f"      --data_file {args.output_file} \\")
        print(f"      --output_dir output/mode1_standard \\")
        print(f"      --n_epochs 50")

        print(f"\n  # Mode 1 - MAML训练")
        print(f"  python scripts/train_mode1.py \\")
        print(f"      --data_file {args.output_file} \\")
        print(f"      --output_dir output/mode1_maml \\")
        print(f"      --use_maml \\")
        print(f"      --n_epochs 50")
    else:
        print(f"\n  # Mode 2 - Standard训练")
        print(f"  python scripts/train_mode2.py \\")
        print(f"      --data_file {args.output_file} \\")
        print(f"      --tissue_source {args.tissue_source} \\")
        print(f"      --output_dir output/mode2_standard \\")
        print(f"      --n_epochs 50")

        print(f"\n  # Mode 2 - MAML训练")
        print(f"  python scripts/train_mode2.py \\")
        print(f"      --data_file {args.output_file} \\")
        print(f"      --tissue_source {args.tissue_source} \\")
        print(f"      --output_dir output/mode2_maml \\")
        print(f"      --use_tissue_aware_negatives \\")
        print(f"      --use_maml \\")
        print(f"      --n_epochs 50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess chunked data for Mode system'
    )

    # 输入输出
    parser.add_argument('--positive_dir', type=str,
                        default='data/cleaned_data',
                        help='Directory with cleaned_*_chunk_*.csv files')
    parser.add_argument('--negative_file', type=str,
                        default=None,
                        help='Negative samples file (optional, can use NegativeSampler later)')
    parser.add_argument('--output_file', type=str,
                        default='data/processed_data.tsv',
                        help='Output TSV file')

    # Mode选择
    parser.add_argument('--mode', type=str,
                        choices=['mode1', 'mode2'],
                        default='mode1',
                        help='Mode 1 (HLA-only) or Mode 2 (HLA×Tissue)')
    parser.add_argument('--tissue_source', type=str,
                        default='Host',
                        help='Tissue column name (for Mode 2)')

    args = parser.parse_args()

    main(args)