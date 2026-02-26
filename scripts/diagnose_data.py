"""
快速数据诊断脚本

检查你的chunk数据包含哪些信息,推荐使用哪个Mode
"""

import pandas as pd
from pathlib import Path
import argparse


def quick_diagnose(data_dir='./data/cleaned_data'):
    """快速诊断数据"""
    
    print("="*80)
    print("快速数据诊断")
    print("="*80)
    print(f"数据目录: {data_dir}")
    
    # 找到chunk文件
    chunk_files = sorted(Path(data_dir).glob('cleaned_*_chunk_*.csv'))
    
    if not chunk_files:
        print(f"\n❌ 未找到chunk文件")
        print(f"   请确认路径: {data_dir}")
        return
    
    print(f"\n✓ 找到 {len(chunk_files)} 个chunk文件")
    
    # 读取第一个文件检查列
    first_file = chunk_files[0]
    print(f"\n检查文件: {first_file.name}")
    
    df_sample = pd.read_csv(first_file, nrows=100)
    
    print(f"\n{'='*80}")
    print(f"数据列检测")
    print(f"{'='*80}")
    print(f"\n可用的列: {df_sample.columns.tolist()}")
    
    # 检查必需列
    print(f"\n必需列检查:")
    required = ['Peptide', 'MHC_Restriction_Name']
    for col in required:
        if col in df_sample.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ❌ {col} (缺失)")
    
    # 检查tissue列
    print(f"\n组织(Tissue)信息检查:")
    tissue_candidates = ['Host', 'Inferred_Tissue', 'Tissue', 'tissue']
    found_tissue = []
    
    for col in tissue_candidates:
        if col in df_sample.columns:
            found_tissue.append(col)
            
            # 统计该列
            non_na = df_sample[col].notna().sum()
            unique = df_sample[col].nunique()
            
            print(f"  ✓ {col}")
            print(f"    非空样本: {non_na}/100")
            print(f"    唯一值数: {unique}")
            
            if unique <= 5:
                print(f"    值: {df_sample[col].unique().tolist()}")
    
    if not found_tissue:
        print(f"  ❌ 未找到tissue列")
        print(f"     尝试过: {tissue_candidates}")
    
    # 显示前几行
    print(f"\n{'='*80}")
    print(f"数据预览 (前5行)")
    print(f"{'='*80}")
    
    display_cols = ['Peptide', 'MHC_Restriction_Name'] + found_tissue
    display_cols = [c for c in display_cols if c in df_sample.columns]
    
    print(df_sample[display_cols].head())
    
    # 给出建议
    print(f"\n{'='*80}")
    print(f"诊断结果与建议")
    print(f"{'='*80}")
    
    if not found_tissue:
        print(f"\n❌ 数据中没有tissue信息")
        print(f"\n💡 推荐方案: Mode 1 (HLA-only)")
        print(f"\n命令:")
        print(f"  # 1. 预处理数据")
        print(f"  python scripts/preprocess_data.py \\")
        print(f"      --positive_dir {data_dir} \\")
        print(f"      --output_file data/mode1_data.tsv \\")
        print(f"      --mode mode1")
        print(f"\n  # 2. 训练")
        print(f"  python scripts/train_mode1.py \\")
        print(f"      --data_file data/mode1_data.tsv \\")
        print(f"      --output_dir output/mode1_standard \\")
        print(f"      --n_epochs 50")
    
    elif len(found_tissue) > 0:
        tissue_col = found_tissue[0]
        
        # 读取更多数据统计tissue
        print(f"\n分析Tissue信息 (读取全部chunk)...")
        
        all_tissues = []
        total_samples = 0
        
        for i, file in enumerate(chunk_files[:3], 1):  # 只读前3个快速检查
            df = pd.read_csv(file, usecols=[tissue_col])
            all_tissues.extend(df[tissue_col].dropna().tolist())
            total_samples += len(df)
            print(f"  处理中... {i}/{min(3, len(chunk_files))}")
        
        from collections import Counter
        tissue_dist = Counter(all_tissues)
        n_tissues = len(tissue_dist)
        
        print(f"\n✓ Tissue统计 (基于前3个chunk):")
        print(f"  样本总数: {total_samples:,}")
        print(f"  Tissue类型数: {n_tissues}")
        
        if n_tissues <= 10:
            for tissue, count in tissue_dist.most_common(10):
                print(f"    {tissue}: {count:,}")
        else:
            for tissue, count in tissue_dist.most_common(5):
                print(f"    {tissue}: {count:,}")
            print(f"    ... 还有 {n_tissues-5} 个")
        
        # 判断
        if n_tissues == 1:
            print(f"\n⚠ 只有1个tissue")
            print(f"\n💡 推荐方案: Mode 1 (HLA-only)")
            print(f"   原因: 只有1个tissue无法学习tissue差异")
        
        elif n_tissues <= 3:
            print(f"\n⚠ Tissue类型较少 ({n_tissues}个)")
            print(f"\n💡 推荐方案: 优先使用 Mode 1")
            print(f"   Mode 2可用,但tissue信息有限")
        
        else:
            print(f"\n✓ Tissue信息充足 ({n_tissues}个)")
            print(f"\n💡 推荐方案: 可以使用 Mode 2")
        
        # 给出两种方案的命令
        print(f"\n{'='*80}")
        print(f"方案 A: Mode 1 (HLA-only, 推荐先试)")
        print(f"{'='*80}")
        print(f"  python scripts/preprocess_data.py \\")
        print(f"      --positive_dir {data_dir} \\")
        print(f"      --output_file data/mode1_data.tsv \\")
        print(f"      --mode mode1")
        print(f"\n  python scripts/train_mode1.py \\")
        print(f"      --data_file data/mode1_data.tsv \\")
        print(f"      --output_dir output/mode1_standard \\")
        print(f"      --n_epochs 50")
        
        if n_tissues > 3:
            print(f"\n{'='*80}")
            print(f"方案 B: Mode 2 (HLA×Tissue)")
            print(f"{'='*80}")
            print(f"  python scripts/preprocess_data.py \\")
            print(f"      --positive_dir {data_dir} \\")
            print(f"      --output_file data/mode2_data.tsv \\")
            print(f"      --mode mode2 \\")
            print(f"      --tissue_source {tissue_col}")
            print(f"\n  python scripts/train_mode2.py \\")
            print(f"      --data_file data/mode2_data.tsv \\")
            print(f"      --tissue_source {tissue_col} \\")
            print(f"      --output_dir output/mode2_standard \\")
            print(f"      --n_epochs 50")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='快速诊断chunk数据')
    # parser.add_argument('--data_dir', type=str,
    #                     default='./data/cleaned_data',
    #                     help='Chunk数据目录')
    #
    # args = parser.parse_args()
    #
    # quick_diagnose(args.data_dir)
    import pandas as pd

    print(f"pandas: {pd.__version__}")