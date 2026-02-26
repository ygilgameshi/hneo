#!/usr/bin/env python3
"""
Task-Balanced Dataset with Adaptive Sampling

分层采样策略，避免数据浪费和过度过采样
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.preprocessing import preprocess_peptides

# !/usr/bin/env python3
"""
改进的Task-Balanced Dataset - 基于正样本数量

关键改进：
- 采样策略基于正样本数量，而不是总样本数
- 这样不同的negative_ratio下，正样本过采样倍数一致
"""



def compute_adaptive_samples_v2(n_positive, negative_ratio, strategy='adaptive'):
    """
    基于正样本数量计算目标采样数（改进版）

    关键改进：
    - 只看正样本数量，不受negative_ratio影响
    - 确保不同negative_ratio下，正样本过采样倍数一致

    Args:
        n_positive: 正样本数量
        negative_ratio: 负样本比例
        strategy: 'adaptive', 'moderate', 'aggressive'

    Returns:
        (pos_target, neg_target, total_target): 正样本目标、负样本目标、总样本目标

    Example:
        >>> compute_adaptive_samples_v2(10, 10, 'adaptive')
        (200, 2000, 2200)  # 10个正样本过采样到200

        >>> compute_adaptive_samples_v2(10, 25, 'adaptive')
        (200, 5000, 5200)  # 同样过采样到200，但负样本更多
    """
    if strategy == 'adaptive':
        # 根据正样本数量决定目标
        if n_positive < 50:
            # 极小tasks: 过采样到100-200
            pos_target = min(200, max(100, n_positive * 4))
        elif n_positive < 200:
            # 小tasks: 过采样到200-400
            pos_target = min(400, n_positive * 2)
        elif n_positive < 1000:
            # 中tasks: 轻度过采样
            pos_target = min(800, int(n_positive * 1.2))
        elif n_positive < 5000:
            # 大tasks: 适度使用
            pos_target = min(n_positive, 2000)
        else:
            # 超大tasks: 降采样
            pos_target = min(n_positive, 3000)

    elif strategy == 'moderate':
        # 温和策略: 更保守的过采样
        if n_positive < 50:
            pos_target = min(150, n_positive * 3)
        elif n_positive < 200:
            pos_target = min(300, int(n_positive * 1.5))
        elif n_positive < 1000:
            pos_target = min(n_positive, 600)
        else:
            pos_target = min(n_positive, 2000)

    elif strategy == 'aggressive':
        # 激进策略: 更强的过采样
        if n_positive < 50:
            pos_target = min(300, n_positive * 6)
        elif n_positive < 200:
            pos_target = min(500, n_positive * 3)
        elif n_positive < 1000:
            pos_target = min(1000, n_positive * 2)
        else:
            pos_target = min(n_positive, 2000)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 计算负样本目标（根据negative_ratio）
    neg_target = int(pos_target * negative_ratio)

    # 总样本数
    total_target = pos_target + neg_target

    return pos_target, neg_target, total_target


class AdaptiveTaskBalancedDataset(Dataset):
    """
    改进的Task-Balanced Dataset

    关键改进：
    - 基于正样本数量决定采样策略
    - 不受negative_ratio影响
    - 不同negative_ratio下，正样本过采样倍数一致

    Args:
        task_datasets: Dict[task_id, DataFrame] 包含peptide, hla, tissue, label列
        task_manager: TaskManager实例
        mode_config: ModeConfig实例
        graph_wrapper: TaskGraphWrapper实例
        sampling_strategy: 'adaptive', 'moderate', 'aggressive'
        negative_ratio: 负样本比例（用于计算采样数）
        random_seed: 随机种子
    """

    def __init__(self,
                 task_datasets,
                 task_manager,
                 mode_config,
                 graph_wrapper,
                 sampling_strategy='adaptive',
                 negative_ratio=25,  # 新增参数
                 random_seed=42):

        self.task_manager = task_manager
        self.mode_config = mode_config
        self.graph_wrapper = graph_wrapper
        self.sampling_strategy = sampling_strategy
        self.negative_ratio = negative_ratio
        self.random_seed = random_seed

        # 统计信息
        self.n_tasks = len(task_datasets)
        self.original_total = sum(len(df) for df in task_datasets.values())

        # 分析正负样本分布
        total_pos = sum((df['label'] == 1).sum() for df in task_datasets.values())
        total_neg = sum((df['label'] == 0).sum() for df in task_datasets.values())

        print(f"\n{'=' * 80}")
        print(f"Adaptive Task-Balanced Sampling (v2 - Position-aware)")
        print(f"{'=' * 80}")
        print(f"  Strategy: {sampling_strategy}")
        print(f"  Negative ratio: {negative_ratio}:1")
        print(f"  Total tasks: {self.n_tasks}")
        print(f"  Original samples: {self.original_total:,}")
        print(f"    Positive: {total_pos:,}")
        print(f"    Negative: {total_neg:,}")
        print(f"    Ratio: {total_neg / total_pos:.1f}:1")

        # 分析正样本分布
        pos_sizes = np.array([(df['label'] == 1).sum() for df in task_datasets.values()])
        print(f"\n  Positive sample distribution:")
        print(f"    Min: {pos_sizes.min()}")
        print(f"    25%: {int(np.percentile(pos_sizes, 25))}")
        print(f"    Median: {int(np.median(pos_sizes))}")
        print(f"    75%: {int(np.percentile(pos_sizes, 75))}")
        print(f"    Max: {pos_sizes.max()}")

        # 为每个task计算采样数
        balanced_dfs = []

        # 统计
        tiny_tasks = 0  # <10 pos
        small_tasks = 0  # 10-50 pos
        medium_tasks = 0  # 50-200 pos
        large_tasks = 0  # 200+ pos

        total_oversampled = 0
        total_undersampled = 0

        sampling_info = []

        for task_id, task_df in task_datasets.items():
            # 分离正负样本
            pos_df = task_df[task_df['label'] == 1]
            neg_df = task_df[task_df['label'] == 0]

            n_pos_original = len(pos_df)
            n_neg_original = len(neg_df)

            # 计算目标采样数（基于正样本数量）
            pos_target, neg_target, total_target = compute_adaptive_samples_v2(
                n_pos_original,
                negative_ratio,
                sampling_strategy
            )

            # 分类统计
            if n_pos_original < 10:
                tiny_tasks += 1
            elif n_pos_original < 50:
                small_tasks += 1
            elif n_pos_original < 200:
                medium_tasks += 1
            else:
                large_tasks += 1

            # 采样正样本
            if n_pos_original >= pos_target:
                pos_sampled = pos_df.sample(n=pos_target, replace=False, random_state=random_seed)
                if n_pos_original > pos_target:
                    total_undersampled += 1
            else:
                pos_sampled = pos_df.sample(n=pos_target, replace=True, random_state=random_seed)
                total_oversampled += 1

            # 采样负样本
            if n_neg_original >= neg_target:
                neg_sampled = neg_df.sample(n=neg_target, replace=False, random_state=random_seed)
            else:
                neg_sampled = neg_df.sample(n=neg_target, replace=True, random_state=random_seed)

            # 合并正负样本
            sampled = pd.concat([pos_sampled, neg_sampled], ignore_index=True)

            # 记录采样信息
            sampling_info.append({
                'task_id': task_id,
                'pos_original': n_pos_original,
                'pos_sampled': pos_target,
                'pos_ratio': pos_target / n_pos_original if n_pos_original > 0 else 0,
                'total_original': len(task_df),
                'total_sampled': len(sampled)
            })

            # 添加task_id
            sampled = sampled.copy()
            sampled['task_id'] = task_id
            balanced_dfs.append(sampled)

        # 打印统计
        print(f"\n  Task categories (by positive samples):")
        print(f"    Tiny (<10):      {tiny_tasks:3d} tasks")
        print(f"    Small (10-50):   {small_tasks:3d} tasks")
        print(f"    Medium (50-200): {medium_tasks:3d} tasks")
        print(f"    Large (>200):    {large_tasks:3d} tasks")

        print(f"\n  Sampling operations:")
        print(f"    Oversampled:  {total_oversampled:3d} tasks ({total_oversampled / self.n_tasks * 100:.1f}%)")
        print(f"    Undersampled: {total_undersampled:3d} tasks ({total_undersampled / self.n_tasks * 100:.1f}%)")

        # 合并并shuffle
        self.df = pd.concat(balanced_dfs, ignore_index=True)
        self.df = self.df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # 计算最终统计
        final_pos = (self.df['label'] == 1).sum()
        final_neg = (self.df['label'] == 0).sum()
        data_utilization = len(self.df) / self.original_total * 100

        print(f"\n  Data statistics:")
        print(f"    Original total: {self.original_total:,}")
        print(f"    After sampling: {len(self.df):,}")
        print(f"      Positive: {final_pos:,}")
        print(f"      Negative: {final_neg:,}")
        print(f"      Ratio: {final_neg / final_pos:.1f}:1")
        print(f"    Data utilization: {data_utilization:.1f}%")

        # 显示采样示例
        sampling_df = pd.DataFrame(sampling_info)
        sampling_df = sampling_df.sort_values('pos_original')

        print(f"\n  Positive sample resampling (smallest 5 tasks):")
        print(f"    {'Original':>8s} → {'Sampled':>8s}  {'Ratio':>6s}")
        for _, row in sampling_df.head(5).iterrows():
            print(f"    {row['pos_original']:8.0f} → {row['pos_sampled']:8.0f}  {row['pos_ratio']:6.2f}x")

        print(f"\n  Positive sample resampling (largest 5 tasks):")
        print(f"    {'Original':>8s} → {'Sampled':>8s}  {'Ratio':>6s}")
        for _, row in sampling_df.tail(5).iterrows():
            print(f"    {row['pos_original']:8.0f} → {row['pos_sampled']:8.0f}  {row['pos_ratio']:6.2f}x")

        print(f"{'=' * 80}\n")

        # 预处理peptide
        print(f"  Preprocessing peptide sequences...")
        self.peptide_encoded, self.peptide_lengths = preprocess_peptides(
            self.df['peptide'].tolist(),
            max_len=15
        )
        print(f"  ✓ Preprocessed {len(self.df):,} peptides")

        # Task indices
        task_to_idx = {tid: idx for idx, tid in enumerate(task_manager.get_all_tasks().keys())}
        self.task_indices = torch.LongTensor(
            self.df['task_id'].map(task_to_idx).values
        )

        # Tissue indices (Mode 2)
        if hasattr(mode_config, 'mode') and 'TISSUE' in str(mode_config.mode):
            unique_tissues = sorted(self.df['tissue'].unique())
            self.tissue_to_idx = {tissue: idx for idx, tissue in enumerate(unique_tissues)}
            self.tissue_indices = torch.LongTensor(
                self.df['tissue'].map(self.tissue_to_idx).values
            )
        else:
            self.tissue_indices = None

        # Labels
        self.labels = torch.FloatTensor(self.df['label'].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """返回一个样本"""
        result = {
            'peptide': self.peptide_encoded[idx],
            'peptide_len': self.peptide_lengths[idx],
            'task_idx': self.task_indices[idx],
            'label': self.labels[idx]
        }

        if self.tissue_indices is not None:
            result['tissue_idx'] = self.tissue_indices[idx]

        return result


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Adaptive Task-Balanced Dataset - 测试")
    print("=" * 80)

    # 创建模拟数据 - 覆盖各种大小的tasks
    print("\n创建模拟数据...")

    task_datasets = {}

    # Tiny task (30个样本)
    task_datasets['tiny'] = pd.DataFrame({
        'peptide': ['SIINFEKL'] * 30,
        'hla': ['HLA-A*02:01'] * 30,
        'tissue': ['Liver'] * 30,
        'label': [1] * 30
    })

    # Small task (150个样本)
    task_datasets['small'] = pd.DataFrame({
        'peptide': ['GILGFVFTL'] * 150,
        'hla': ['HLA-B*07:02'] * 150,
        'tissue': ['Kidney'] * 150,
        'label': [1] * 150
    })

    # Medium task (500个样本)
    task_datasets['medium'] = pd.DataFrame({
        'peptide': ['YLEPGPVTA'] * 500,
        'hla': ['HLA-A*01:01'] * 500,
        'tissue': ['Lung'] * 500,
        'label': [1] * 500
    })

    # Large task (3000个样本)
    task_datasets['large'] = pd.DataFrame({
        'peptide': ['LLWNGPMAV'] * 3000,
        'hla': ['HLA-B*57:01'] * 3000,
        'tissue': ['Spleen'] * 3000,
        'label': [1] * 3000
    })

    # XLarge task (10000个样本)
    task_datasets['xlarge'] = pd.DataFrame({
        'peptide': ['NLVPMVATV'] * 10000,
        'hla': ['HLA-A*02:01'] * 10000,
        'tissue': ['Blood'] * 10000,
        'label': [1] * 10000
    })

    print(f"  Tiny: {len(task_datasets['tiny'])} 样本")
    print(f"  Small: {len(task_datasets['small'])} 样本")
    print(f"  Medium: {len(task_datasets['medium'])} 样本")
    print(f"  Large: {len(task_datasets['large'])} 样本")
    print(f"  XLarge: {len(task_datasets['xlarge'])} 样本")
    print(f"  Total: {sum(len(df) for df in task_datasets.values())} 样本")


    # 创建mock对象
    class MockTaskManager:
        def get_all_tasks(self):
            return {k: None for k in task_datasets.keys()}


    class MockConfig:
        mode = 'HLA_TISSUE'


    class MockGraphWrapper:
        pass


    # 测试不同策略
    for strategy in ['adaptive', 'moderate', 'aggressive']:
        print(f"\n{'=' * 80}")
        print(f"Testing strategy: {strategy}")
        print(f"{'=' * 80}")

        dataset = AdaptiveTaskBalancedDataset(
            task_datasets=task_datasets,
            task_manager=MockTaskManager(),
            mode_config=MockConfig(),
            graph_wrapper=MockGraphWrapper(),
            sampling_strategy=strategy
        )

        print(f"\n✓ Dataset创建成功")
        print(f"  总样本数: {len(dataset):,}")

        # 测试__getitem__
        sample = dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过!")
    print("=" * 80)