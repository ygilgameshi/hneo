#!/usr/bin/env python3
"""
从完整的HLA FASTA文件中提取数据中实际使用的HLA序列

用法:
    python scripts/extract_used_hla.py \
        --data_file data/mode2_data.tsv \
        --fasta_file hla_prot.fasta \
        --output_file configs/hla_sequences.json
"""

import argparse
import json
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Set


def parse_fasta(fasta_file: str) -> Dict[str, str]:
    """
    解析FASTA文件
    
    Args:
        fasta_file: FASTA文件路径
        
    Returns:
        {allele_name: sequence}
    """
    print(f"\n📂 解析FASTA文件: {fasta_file}")
    
    sequences = {}
    current_allele = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('>'):
                # 保存上一个序列
                if current_allele and current_seq:
                    sequences[current_allele] = ''.join(current_seq)
                
                # 解析header
                # 格式: >HLA:HLA00001 A*01:01:01:01 365 bp
                parts = line[1:].split()
                
                # 提取allele名称（第二个字段）
                if len(parts) >= 2:
                    allele = parts[1]
                    
                    # 标准化名称为 HLA-A*01:01 格式
                    # 去掉后面的字段级分辨率，只保留前2个field
                    match = re.match(r'([ABC]\*\d+:\d+)', allele)
                    if match:
                        # 添加HLA-前缀
                        current_allele = f"HLA-{match.group(1)}"
                    else:
                        current_allele = f"HLA-{allele}"
                    
                    current_seq = []
                else:
                    current_allele = None
            
            elif current_allele:
                # 累加序列
                current_seq.append(line)
        
        # 保存最后一个序列
        if current_allele and current_seq:
            sequences[current_allele] = ''.join(current_seq)
    
    print(f"  ✓ 解析了 {len(sequences):,} 个唯一的HLA序列（2-field分辨率）")
    
    return sequences


def get_hlas_from_data(data_file: str, hla_column: str = 'MHC_Restriction_Name') -> Set[str]:
    """
    从数据文件中获取所有使用的HLA
    
    Args:
        data_file: 数据文件路径
        hla_column: HLA列名
        
    Returns:
        Set of HLA names
    """
    print(f"\n📊 分析数据文件: {data_file}")
    
    df = pd.read_csv(data_file, sep='\t')
    
    # 尝试不同的列名
    possible_columns = [hla_column, 'hla', 'HLA', 'MHC', 'mhc', 'allele']
    
    hla_col = None
    for col in possible_columns:
        if col in df.columns:
            hla_col = col
            break
    
    if hla_col is None:
        print(f"  ❌ 找不到HLA列，可用列: {list(df.columns)}")
        return set()
    
    hlas = set(df[hla_col].unique())
    
    print(f"  ✓ 找到 {len(hlas)} 个唯一的HLA")
    print(f"  ✓ 总样本数: {len(df):,}")
    
    return hlas


def extract_used_sequences(all_sequences: Dict[str, str], 
                          used_hlas: Set[str]) -> Dict[str, str]:
    """
    提取数据中使用的HLA序列
    
    Args:
        all_sequences: 所有HLA序列
        used_hlas: 数据中使用的HLA
        
    Returns:
        提取的序列字典
    """
    print(f"\n🔍 提取使用的HLA序列...")
    
    extracted = {}
    missing = []
    
    for hla in used_hlas:
        if hla in all_sequences:
            extracted[hla] = all_sequences[hla]
        else:
            missing.append(hla)
    
    print(f"  ✓ 提取了 {len(extracted)} 个HLA序列")
    
    if missing:
        print(f"  ⚠️  {len(missing)} 个HLA在FASTA中未找到:")
        for hla in sorted(missing)[:20]:  # 只显示前20个
            print(f"     - {hla}")
        if len(missing) > 20:
            print(f"     ... 还有 {len(missing) - 20} 个")
    
    coverage = len(extracted) / len(used_hlas) * 100 if used_hlas else 0
    print(f"\n📈 覆盖率: {coverage:.1f}% ({len(extracted)}/{len(used_hlas)})")
    
    return extracted


def save_sequences(sequences: Dict[str, str], output_file: str):
    """保存序列到JSON文件"""
    print(f"\n💾 保存到: {output_file}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(sequences, f, indent=2)
    
    # 统计
    hla_a = sum(1 for k in sequences if k.startswith('HLA-A'))
    hla_b = sum(1 for k in sequences if k.startswith('HLA-B'))
    hla_c = sum(1 for k in sequences if k.startswith('HLA-C'))
    
    print(f"\n✅ 保存成功!")
    print(f"  文件: {output_file}")
    print(f"  总数: {len(sequences)}")
    print(f"  分布:")
    print(f"    HLA-A: {hla_a}")
    print(f"    HLA-B: {hla_b}")
    print(f"    HLA-C: {hla_c}")
    
    # 示例
    print(f"\n📋 示例序列:")
    for hla, seq in list(sequences.items())[:3]:
        print(f"  {hla}: {seq[:50]}... (长度: {len(seq)})")


def main(args):
    print("="*80)
    print("从完整FASTA中提取数据使用的HLA序列")
    print("="*80)
    
    # Step 1: 获取数据中使用的HLA
    used_hlas = get_hlas_from_data(args.data_file, args.hla_column)
    
    if not used_hlas:
        print("\n❌ 未找到HLA数据!")
        return
    
    # Step 2: 解析FASTA文件
    all_sequences = parse_fasta(args.fasta_file)
    
    if not all_sequences:
        print("\n❌ FASTA文件解析失败!")
        return
    
    # Step 3: 提取使用的序列
    extracted_sequences = extract_used_sequences(all_sequences, used_hlas)
    
    if not extracted_sequences:
        print("\n❌ 没有提取到任何序列!")
        return
    
    # Step 4: 保存
    save_sequences(extracted_sequences, args.output_file)
    
    print("\n" + "="*80)
    print("✓ 完成!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='从完整FASTA中提取数据使用的HLA序列'
    )
    
    parser.add_argument('--data_file', type=str, required=True,
                       help='数据文件路径 (TSV)')
    
    parser.add_argument('--fasta_file', type=str, required=True,
                       help='完整的HLA FASTA文件路径')
    
    parser.add_argument('--output_file', type=str, 
                       default='configs/hla_sequences.json',
                       help='输出JSON文件路径')
    
    parser.add_argument('--hla_column', type=str, 
                       default='MHC_Restriction_Name',
                       help='数据文件中HLA列的名称')
    
    args = parser.parse_args()
    
    main(args)
