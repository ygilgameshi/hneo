"""
Phase 2: HLA×Tissue 任务创建器

从数据集创建:
1. Phase 1 tasks: HLA-only (baseline)
2. Phase 2 tasks: HLA×Tissue combinations
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def create_phase2_tasks(data_df, 
                        phase='combined',
                        min_samples_phase1=20,
                        min_samples_phase2=10,
                        exclude_unknown_tissue=True):
    """
    创建Phase 2任务（HLA × Tissue组合）
    
    Args:
        data_df: DataFrame with columns ['peptide', 'hla', 'tissue', 'label']
        phase: 'phase1' (HLA-only), 'phase2' (HLA×Tissue), 'combined' (both)
        min_samples_phase1: Phase 1任务最小样本数
        min_samples_phase2: Phase 2任务最小样本数（更低阈值）
        exclude_unknown_tissue: 是否排除Unknown组织
        
    Returns:
        dict: {task_name: {'data': df, 'type': 'phase1'|'phase2', 
                           'hla': str, 'tissue': str|None, ...}}
    """
    
    tasks = {}
    
    # ========== Phase 1: HLA-only tasks (baseline) ==========
    if phase in ['phase1', 'combined']:
        print(f"\n创建Phase 1任务 (HLA-only, min_samples={min_samples_phase1})...")
        
        for hla in data_df['hla'].unique():
            hla_data = data_df[data_df['hla'] == hla]
            
            if len(hla_data) < min_samples_phase1:
                continue
            
            task_name = f"HLA_{hla}"
            
            tasks[task_name] = {
                'data': hla_data.copy(),
                'type': 'phase1',
                'hla': hla,
                'tissue': None,
                'n_samples': len(hla_data),
                'n_positive': (hla_data['label'] == 1).sum(),
                'n_negative': (hla_data['label'] == 0).sum(),
            }
        
        print(f"  ✓ 创建了 {len([t for t in tasks.values() if t['type']=='phase1'])} 个Phase 1任务")
    
    # ========== Phase 2: HLA×Tissue tasks ==========
    if phase in ['phase2', 'combined']:
        print(f"\n创建Phase 2任务 (HLA×Tissue, min_samples={min_samples_phase2})...")
        
        # 过滤数据
        phase2_data = data_df.copy()
        if exclude_unknown_tissue:
            phase2_data = phase2_data[phase2_data['tissue'] != 'Unknown']
        
        # 创建HLA×Tissue组合任务
        for (hla, tissue), group in phase2_data.groupby(['hla', 'tissue']):
            if len(group) < min_samples_phase2:
                continue
            
            task_name = f"HLA_{hla}_Tissue_{tissue}"
            
            tasks[task_name] = {
                'data': group.copy(),
                'type': 'phase2',
                'hla': hla,
                'tissue': tissue,
                'n_samples': len(group),
                'n_positive': (group['label'] == 1).sum(),
                'n_negative': (group['label'] == 0).sum(),
            }
        
        print(f"  ✓ 创建了 {len([t for t in tasks.values() if t['type']=='phase2'])} 个Phase 2任务")
    
    # ========== 统计信息 ==========
    print(f"\n任务统计:")
    phase1_tasks = [t for t in tasks.values() if t['type'] == 'phase1']
    phase2_tasks = [t for t in tasks.values() if t['type'] == 'phase2']
    
    print(f"  Phase 1 (HLA-only): {len(phase1_tasks)} tasks")
    print(f"  Phase 2 (HLA×Tissue): {len(phase2_tasks)} tasks")
    print(f"  总任务数: {len(tasks)}")
    
    if phase1_tasks:
        phase1_samples = [t['n_samples'] for t in phase1_tasks]
        print(f"\n  Phase 1样本分布:")
        print(f"    平均: {np.mean(phase1_samples):.1f}")
        print(f"    中位数: {np.median(phase1_samples):.1f}")
        print(f"    范围: [{min(phase1_samples)}, {max(phase1_samples)}]")
    
    if phase2_tasks:
        phase2_samples = [t['n_samples'] for t in phase2_tasks]
        print(f"\n  Phase 2样本分布:")
        print(f"    平均: {np.mean(phase2_samples):.1f}")
        print(f"    中位数: {np.median(phase2_samples):.1f}")
        print(f"    范围: [{min(phase2_samples)}, {max(phase2_samples)}]")
    
    return tasks


def visualize_task_distribution(tasks, output_file='task_distribution_phase2.png'):
    """
    可视化任务分布
    """
    phase1_tasks = {k: v for k, v in tasks.items() if v['type'] == 'phase1'}
    phase2_tasks = {k: v for k, v in tasks.items() if v['type'] == 'phase2'}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 任务数量对比
    ax = axes[0, 0]
    counts = [len(phase1_tasks), len(phase2_tasks)]
    ax.bar(['Phase 1\n(HLA-only)', 'Phase 2\n(HLA×Tissue)'], counts, 
           color=['#3498db', '#e74c3c'])
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Task Count by Phase')
    for i, v in enumerate(counts):
        ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # 2. 样本数分布
    ax = axes[0, 1]
    if phase1_tasks:
        phase1_samples = [t['n_samples'] for t in phase1_tasks.values()]
        ax.hist(phase1_samples, bins=20, alpha=0.6, label='Phase 1', color='#3498db')
    if phase2_tasks:
        phase2_samples = [t['n_samples'] for t in phase2_tasks.values()]
        ax.hist(phase2_samples, bins=20, alpha=0.6, label='Phase 2', color='#e74c3c')
    ax.set_xlabel('Samples per Task')
    ax.set_ylabel('Frequency')
    ax.set_title('Sample Distribution')
    ax.legend()
    
    # 3. Phase 2: HLA分布
    ax = axes[1, 0]
    if phase2_tasks:
        hla_counts = defaultdict(int)
        for task in phase2_tasks.values():
            hla_counts[task['hla']] += 1
        
        top_hlas = sorted(hla_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ax.barh([h[0] for h in top_hlas], [h[1] for h in top_hlas], color='#2ecc71')
        ax.set_xlabel('Number of Tissue Combinations')
        ax.set_title('Top 10 HLAs in Phase 2')
    
    # 4. Phase 2: Tissue分布
    ax = axes[1, 1]
    if phase2_tasks:
        tissue_counts = defaultdict(int)
        for task in phase2_tasks.values():
            tissue_counts[task['tissue']] += 1
        
        tissues = sorted(tissue_counts.items(), key=lambda x: x[1], reverse=True)
        ax.barh([t[0] for t in tissues], [t[1] for t in tissues], color='#f39c12')
        ax.set_xlabel('Number of HLA Combinations')
        ax.set_title('Tissue Distribution in Phase 2')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 任务分布图已保存: {output_file}")
    plt.close()


# 测试代码
if __name__ == "__main__":
    # 模拟数据测试
    print("测试任务创建...")
    
    # 创建模拟数据
    np.random.seed(42)
    hlas = [f'HLA-A*{i:02d}:01' for i in range(1, 11)]
    tissues = ['Lymphoid', 'Blood', 'Lung', 'Liver', 'Brain']
    
    data = []
    for hla in hlas:
        for tissue in tissues:
            n = np.random.randint(5, 50)
            for _ in range(n):
                data.append({
                    'peptide': 'TESTPEPTIDE',
                    'hla': hla,
                    'tissue': tissue,
                    'label': np.random.choice([0, 1])
                })
    
    df = pd.DataFrame(data)
    
    # 测试任务创建
    tasks = create_phase2_tasks(df, phase='combined', 
                                min_samples_phase1=20, 
                                min_samples_phase2=10)
    
    # 可视化
    visualize_task_distribution(tasks)
    
    print("\n✓ 任务创建模块测试通过！")
