"""
测试集去污染脚本

问题：NetMHCpan-4.1和MHCflurry-2.0的训练数据来自IEDB，
      你的正样本也来自IEDB，直接对比是不公平的。

解决方案：从你的测试集中剔除与baseline训练集重叠的正样本，
          只保留baseline从未见过的数据。

用法：
    python decontaminate_testset.py \
        --test_file  output/task_balanced_v2/data_splits/test.tsv \
        --output_dir output/clean_testset

支持两种去重策略（可同时使用）：
    1. MHCflurry训练数据去重（自动下载）
    2. NetMHCpan训练数据去重（需要手动提供文件）
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


# ============================================================
# NetMHCpan-4.1 训练集来源说明
# ============================================================
# NetMHCpan-4.1 于 2020 年发布（Reynisson et al., Nucleic Acids Res.）
# 训练数据截止日期：约 2019 年底
# 来源：
#   - IEDB MS eluted ligand 数据
#   - Abelin et al. 2017 (Immunity)
#   - Sarkizova et al. 2020 (Nature Genetics)
#   - 多个公开的单等位基因 MS 数据集
# ============================================================

# ============================================================
# MHCflurry-2.0 训练集来源说明
# ============================================================
# MHCflurry-2.0 于 2020 年发布（O'Donnell et al., Cell Systems）
# 训练数据截止日期：约 2019 年底
# 来源：
#   - IEDB MS eluted ligand 数据（与 NetMHCpan 高度重叠）
#   - Abelin et al. 2017
#   - Sarkizova et al. 2020
# ============================================================


def load_mhcflurry_train_peptides():
    """
    加载MHCflurry的训练肽段集合
    MHCflurry开源，训练数据可以直接获取
    """
    print("\n📦 Loading MHCflurry training peptides...")

    try:
        # 方法1：从已安装的MHCflurry获取训练数据路径
        from mhcflurry.downloads import get_path
        import os

        # MHCflurry训练数据文件
        train_data_candidates = [
            "models_class1_presentation/train_data.csv.bz2",
            "models_class1_presentation/train_data.csv",
            "models_class1/train_data.csv.bz2",
            "models_class1/train_data.csv",
        ]

        for candidate in train_data_candidates:
            try:
                path = get_path(*candidate.split("/", 1))
                if Path(path).exists():
                    print(f"  Found training data at: {path}")
                    df = pd.read_csv(path)
                    peptides = set(df['peptide'].str.upper().tolist())
                    print(f"  ✓ Loaded {len(peptides):,} unique MHCflurry training peptides")
                    return peptides
            except Exception:
                continue

        # 方法2：从mhcflurry内置数据获取
        from mhcflurry import Class1PresentationPredictor
        predictor = Class1PresentationPredictor.load()

        if hasattr(predictor, 'affinity_predictor'):
            ap = predictor.affinity_predictor
            if hasattr(ap, 'metadata_dataframes'):
                for name, df in ap.metadata_dataframes.items():
                    if 'peptide' in df.columns:
                        peptides = set(df['peptide'].str.upper().tolist())
                        print(f"  ✓ Loaded {len(peptides):,} peptides from metadata ({name})")
                        return peptides

    except ImportError:
        print("  ⚠️  MHCflurry not installed")
    except Exception as e:
        print(f"  ⚠️  Could not load MHCflurry training data automatically: {e}")

    # 方法3：使用公开的MHCflurry训练数据URL
    print("\n  Trying to download MHCflurry training data from GitHub...")
    try:
        import urllib.request
        import io

        # MHCflurry公开的训练数据
        url = "https://raw.githubusercontent.com/openvax/mhcflurry/master/downloads-generation/models_class1_presentation/train_data.csv.bz2"
        print(f"  Downloading from: {url}")
        response = urllib.request.urlopen(url, timeout=30)
        df = pd.read_csv(io.BytesIO(response.read()), compression='bz2')
        peptides = set(df['peptide'].str.upper().tolist())
        print(f"  ✓ Downloaded {len(peptides):,} MHCflurry training peptides")
        return peptides

    except Exception as e:
        print(f"  ⚠️  Download failed: {e}")

    print("  ⚠️  Could not load MHCflurry training peptides, returning empty set")
    return set()


def load_netmhcpan_train_peptides(netmhcpan_train_file=None):
    """
    加载NetMHCpan-4.1的训练肽段集合

    NetMHCpan不完全开源，但训练数据可以从作者提供的Supplementary获取。
    如果没有文件，返回空集合（仅用MHCflurry去重）。

    Args:
        netmhcpan_train_file: NetMHCpan训练数据文件路径（可选）
    """
    if netmhcpan_train_file is None:
        print("\n⚠️  No NetMHCpan training file provided.")
        print("   NetMHCpan-4.1 training data can be obtained from:")
        print("   https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/")
        print("   (Download the training data from Supplementary)")
        print("   Skipping NetMHCpan decontamination...")
        return set()

    print(f"\n📦 Loading NetMHCpan training peptides from: {netmhcpan_train_file}")

    try:
        path = Path(netmhcpan_train_file)

        if path.suffix in ['.csv', '.tsv']:
            sep = '\t' if path.suffix == '.tsv' else ','
            df = pd.read_csv(path, sep=sep)

            # 寻找peptide列
            peptide_col = None
            for col in ['peptide', 'Peptide', 'sequence', 'Sequence', 'pep']:
                if col in df.columns:
                    peptide_col = col
                    break

            if peptide_col:
                peptides = set(df[peptide_col].str.upper().dropna().tolist())
                print(f"  ✓ Loaded {len(peptides):,} NetMHCpan training peptides")
                return peptides

        # FASTA格式
        elif path.suffix in ['.fasta', '.fa']:
            peptides = set()
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('>'):
                        peptides.add(line.upper())
            print(f"  ✓ Loaded {len(peptides):,} NetMHCpan training peptides (FASTA)")
            return peptides

        # 纯文本，每行一个peptide
        else:
            peptides = set()
            with open(path) as f:
                for line in f:
                    p = line.strip().upper()
                    if p and p.isalpha():
                        peptides.add(p)
            print(f"  ✓ Loaded {len(peptides):,} NetMHCpan training peptides (text)")
            return peptides

    except Exception as e:
        print(f"  ⚠️  Error loading NetMHCpan training data: {e}")
        return set()


def decontaminate(test_file, output_dir,
                  netmhcpan_train_file=None,
                  peptide_col='Peptide',
                  hla_col='MHC_Restriction_Name',
                  label_col='Label',
                  sep='\t'):
    """
    主去重函数

    Args:
        test_file:            测试集文件路径
        output_dir:           输出目录
        netmhcpan_train_file: NetMHCpan训练集文件（可选）
        peptide_col:          peptide列名
        hla_col:              HLA列名
        label_col:            label列名
        sep:                  文件分隔符
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TEST SET DECONTAMINATION")
    print("=" * 80)

    # ========== 1. 加载测试集 ==========
    print(f"\n📂 Loading test set: {test_file}")
    test_df = pd.read_csv(test_file, sep=sep)

    # 标准化列名
    col_map = {}
    if peptide_col in test_df.columns:
        col_map[peptide_col] = 'peptide'
    if hla_col in test_df.columns:
        col_map[hla_col] = 'hla'
    if label_col in test_df.columns:
        col_map[label_col] = 'label'

    test_df = test_df.rename(columns=col_map)

    # 统一大写peptide
    test_df['peptide_upper'] = test_df['peptide'].str.upper()

    n_total       = len(test_df)
    n_positive    = (test_df['label'] == 1).sum()
    n_negative    = (test_df['label'] == 0).sum()
    n_hlas        = test_df['hla'].nunique()

    print(f"\n  Original test set:")
    print(f"    Total samples:    {n_total:,}")
    print(f"    Positive:         {n_positive:,}  ({n_positive/n_total*100:.1f}%)")
    print(f"    Negative:         {n_negative:,}  ({n_negative/n_total*100:.1f}%)")
    print(f"    HLA alleles:      {n_hlas}")
    print(f"    Unique peptides:  {test_df['peptide_upper'].nunique():,}")

    # ========== 2. 加载baseline训练集 ==========
    mhcflurry_peptides  = load_mhcflurry_train_peptides()
    netmhcpan_peptides  = load_netmhcpan_train_peptides(netmhcpan_train_file)

    # 合并所有baseline训练肽段
    all_baseline_peptides = mhcflurry_peptides | netmhcpan_peptides
    print(f"\n📊 Baseline training peptides summary:")
    print(f"    MHCflurry-2.0:    {len(mhcflurry_peptides):,}")
    print(f"    NetMHCpan-4.1:    {len(netmhcpan_peptides):,}")
    print(f"    Union (total):    {len(all_baseline_peptides):,}")

    # ========== 3. 分析重叠 ==========
    print("\n" + "=" * 80)
    print("📈 OVERLAP ANALYSIS")
    print("=" * 80)

    pos_mask     = test_df['label'] == 1
    test_pos_pep = set(test_df.loc[pos_mask, 'peptide_upper'])

    overlap_mhcflurry  = test_pos_pep & mhcflurry_peptides
    overlap_netmhcpan  = test_pos_pep & netmhcpan_peptides
    overlap_union      = test_pos_pep & all_baseline_peptides

    print(f"\n  Positive peptide overlap with baseline training sets:")
    print(f"    vs MHCflurry-2.0: {len(overlap_mhcflurry):,} / {len(test_pos_pep):,}"
          f"  ({len(overlap_mhcflurry)/len(test_pos_pep)*100:.1f}%)")
    print(f"    vs NetMHCpan-4.1: {len(overlap_netmhcpan):,} / {len(test_pos_pep):,}"
          f"  ({len(overlap_netmhcpan)/len(test_pos_pep)*100:.1f}%)")
    print(f"    vs Either:        {len(overlap_union):,} / {len(test_pos_pep):,}"
          f"  ({len(overlap_union)/len(test_pos_pep)*100:.1f}%)")

    # ========== 4. 去重（剔除重叠正样本） ==========
    print("\n" + "=" * 80)
    print("🧹 REMOVING OVERLAPPING POSITIVE SAMPLES")
    print("=" * 80)

    # 标记重叠样本
    overlap_flag = pos_mask & test_df['peptide_upper'].isin(all_baseline_peptides)
    clean_df     = test_df[~overlap_flag].copy()

    # 去掉辅助列
    clean_df = clean_df.drop(columns=['peptide_upper'])

    # 还原原始列名（如果做过重命名）
    reverse_map = {v: k for k, v in col_map.items()}
    clean_df = clean_df.rename(columns=reverse_map)

    n_removed      = overlap_flag.sum()
    n_clean_total  = len(clean_df)
    n_clean_pos    = (clean_df[label_col] == 1).sum() if label_col in clean_df.columns \
                     else (clean_df['label'] == 1).sum()
    n_clean_neg    = n_clean_total - n_clean_pos

    print(f"\n  Removed:  {n_removed:,} overlapping positive samples")
    print(f"\n  Clean test set:")
    print(f"    Total samples:    {n_clean_total:,}  (was {n_total:,})")
    print(f"    Positive:         {n_clean_pos:,}  (was {n_positive:,})")
    print(f"    Negative:         {n_clean_neg:,}  (was {n_negative:,})")
    print(f"    HLA alleles:      {clean_df[hla_col].nunique() if hla_col in clean_df.columns else 'N/A'}")

    # ========== 5. 检查去重后HLA覆盖 ==========
    print("\n" + "=" * 80)
    print("🔍 HLA COVERAGE AFTER DECONTAMINATION")
    print("=" * 80)

    hla_col_final = hla_col if hla_col in clean_df.columns else 'hla'
    label_col_final = label_col if label_col in clean_df.columns else 'label'

    hla_stats = clean_df.groupby(hla_col_final)[label_col_final].agg(
        total='count',
        positive='sum'
    ).reset_index()
    hla_stats['negative'] = hla_stats['total'] - hla_stats['positive']
    hla_stats = hla_stats.sort_values('positive', ascending=False)

    dropped_hlas = hla_stats[hla_stats['positive'] == 0]
    kept_hlas    = hla_stats[hla_stats['positive'] > 0]

    print(f"\n  HLAs with ≥1 positive after decontamination: {len(kept_hlas)}")
    print(f"  HLAs with 0 positives (all removed):         {len(dropped_hlas)}")

    if len(dropped_hlas) > 0:
        print(f"\n  ⚠️  HLAs losing all positives (will be excluded):")
        for _, row in dropped_hlas.iterrows():
            print(f"      {row[hla_col_final]}")

        # 同时剔除这些HLA的负样本（没有正样本的HLA不能评估）
        valid_hlas   = set(kept_hlas[hla_col_final])
        clean_df     = clean_df[clean_df[hla_col_final].isin(valid_hlas)].copy()
        n_clean_total = len(clean_df)
        n_clean_pos   = (clean_df[label_col_final] == 1).sum()
        n_clean_neg   = n_clean_total - n_clean_pos
        print(f"\n  After also removing HLAs with no positives:")
        print(f"    Total samples:    {n_clean_total:,}")
        print(f"    Positive:         {n_clean_pos:,}")
        print(f"    Negative:         {n_clean_neg:,}")
        print(f"    HLA alleles:      {clean_df[hla_col_final].nunique()}")

    # ========== 6. 保存文件 ==========
    print("\n" + "=" * 80)
    print("💾 SAVING")
    print("=" * 80)

    # 保存干净的测试集
    clean_file = output_dir / 'test_clean.tsv'
    clean_df.to_csv(clean_file, sep='\t', index=False)
    print(f"\n  ✓ Clean test set:  {clean_file}")

    # 保存被剔除的样本（用于审查）
    removed_df = test_df[overlap_flag].drop(columns=['peptide_upper'])
    removed_df = removed_df.rename(columns=reverse_map)
    removed_file = output_dir / 'removed_overlapping.tsv'
    removed_df.to_csv(removed_file, sep='\t', index=False)
    print(f"  ✓ Removed samples: {removed_file}")

    # 保存统计报告
    report_file = output_dir / 'decontamination_report.txt'
    with open(report_file, 'w') as f:
        f.write("TEST SET DECONTAMINATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("Original test set:\n")
        f.write(f"  Total:     {n_total:,}\n")
        f.write(f"  Positive:  {n_positive:,}\n")
        f.write(f"  Negative:  {n_negative:,}\n")
        f.write(f"  HLAs:      {n_hlas}\n\n")
        f.write("Baseline training set overlap (positive peptides):\n")
        f.write(f"  MHCflurry-2.0:  {len(overlap_mhcflurry):,}"
                f" ({len(overlap_mhcflurry)/max(len(test_pos_pep),1)*100:.1f}%)\n")
        f.write(f"  NetMHCpan-4.1:  {len(overlap_netmhcpan):,}"
                f" ({len(overlap_netmhcpan)/max(len(test_pos_pep),1)*100:.1f}%)\n")
        f.write(f"  Union:          {len(overlap_union):,}"
                f" ({len(overlap_union)/max(len(test_pos_pep),1)*100:.1f}%)\n\n")
        f.write("Clean test set:\n")
        f.write(f"  Total:     {n_clean_total:,}\n")
        f.write(f"  Positive:  {n_clean_pos:,}\n")
        f.write(f"  Negative:  {n_clean_neg:,}\n")
        f.write(f"  HLAs:      {clean_df[hla_col_final].nunique()}\n\n")
        f.write("Files:\n")
        f.write(f"  Clean test set:   test_clean.tsv\n")
        f.write(f"  Removed samples:  removed_overlapping.tsv\n")
    print(f"  ✓ Report:          {report_file}")

    print("\n" + "=" * 80)
    print("✓ Decontamination complete!")
    print("=" * 80)
    print(f"\n  ➡️  Use '{clean_file}' for fair baseline comparison.")
    print(f"  ➡️  Removed {n_removed:,} overlapping positive samples"
          f" ({n_removed/max(n_positive,1)*100:.1f}% of original positives).\n")

    return clean_df


def main():
    parser = argparse.ArgumentParser(
        description='Remove test set overlap with NetMHCpan/MHCflurry training data'
    )
    parser.add_argument('--test_file', type=str, required=True,
                        help='Test set file (TSV)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--netmhcpan_train_file', type=str, default=None,
                        help='NetMHCpan training peptides file (optional)')
    parser.add_argument('--peptide_col', type=str, default='Peptide',
                        help='Peptide column name (default: Peptide)')
    parser.add_argument('--hla_col', type=str, default='MHC_Restriction_Name',
                        help='HLA column name (default: MHC_Restriction_Name)')
    parser.add_argument('--label_col', type=str, default='Label',
                        help='Label column name (default: Label)')
    parser.add_argument('--sep', type=str, default='\t',
                        help='File separator (default: tab)')

    args = parser.parse_args()

    decontaminate(
        test_file=args.test_file,
        output_dir=args.output_dir,
        netmhcpan_train_file=args.netmhcpan_train_file,
        peptide_col=args.peptide_col,
        hla_col=args.hla_col,
        label_col=args.label_col,
        sep=args.sep,
    )


if __name__ == '__main__':
    main()
