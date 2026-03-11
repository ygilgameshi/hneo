#!/usr/bin/env python3
"""
解析IPD-IMGT/HLA FASTA文件并转换为JSON格式

用法:
    python scripts/parse_hla_fasta.py fasta/A_prot.fasta fasta/B_prot.fasta fasta/C_prot.fasta
    
或者:
    python scripts/parse_hla_fasta.py hla_prot.fasta
"""

import json
import re
from pathlib import Path
from typing import Dict
import sys

def parse_fasta(fasta_file: str) -> Dict[str, str]:
    """
    解析FASTA文件
    
    Args:
        fasta_file: FASTA文件路径
        
    Returns:
        {allele_name: sequence}
    """
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
                    # 去掉后面的字段级分辨率
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
    
    return sequences


def merge_sequences(sequences_dict: Dict[str, str]) -> Dict[str, str]:
    """
    合并序列，去重
    
    对于相同的allele名称（如HLA-A*01:01），可能有多个4-field版本
    我们保留第一个遇到的
    """
    merged = {}
    
    for allele, seq in sequences_dict.items():
        # 如果这个allele还没有，添加
        if allele not in merged:
            merged[allele] = seq
    
    return merged


def main(fasta_files: list, output_file: str = "configs/hla_sequences.json"):
    """
    主函数
    
    Args:
        fasta_files: FASTA文件列表
        output_file: 输出JSON文件路径
    """
    print("="*80)
    print("解析IPD-IMGT/HLA FASTA文件")
    print("="*80)
    
    all_sequences = {}
    
    for fasta_file in fasta_files:
        print(f"\n解析: {fasta_file}")
        
        if not Path(fasta_file).exists():
            print(f"  ❌ 文件不存在!")
            continue
        
        sequences = parse_fasta(fasta_file)
        print(f"  ✓ 解析了 {len(sequences)} 个序列")
        
        # 合并
        all_sequences.update(sequences)
    
    # 去重
    merged = merge_sequences(all_sequences)
    
    print(f"\n总计:")
    print(f"  原始序列: {len(all_sequences)}")
    print(f"  去重后: {len(merged)}")
    
    # 统计
    hla_a = sum(1 for k in merged if k.startswith('HLA-A'))
    hla_b = sum(1 for k in merged if k.startswith('HLA-B'))
    hla_c = sum(1 for k in merged if k.startswith('HLA-C'))
    
    print(f"\n分布:")
    print(f"  HLA-A: {hla_a}")
    print(f"  HLA-B: {hla_b}")
    print(f"  HLA-C: {hla_c}")
    
    # 示例
    print(f"\n示例序列:")
    for allele, seq in list(merged.items())[:3]:
        print(f"  {allele}: {seq[:50]}... (长度: {len(seq)})")
    
    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"\n✓ 已保存到: {output_path}")
    print(f"  共 {len(merged)} 个HLA序列")
    
    print("\n" + "="*80)
    print("✓ 完成!")
    print("="*80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:")
        print("  python parse_hla_fasta.py fasta/A_prot.fasta fasta/B_prot.fasta fasta/C_prot.fasta")
        print("  或")
        print("  python parse_hla_fasta.py hla_prot.fasta")
        sys.exit(1)
    
    fasta_files = sys.argv[1:]
    main(fasta_files)
