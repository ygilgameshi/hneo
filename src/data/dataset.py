
"""
Mode-Aware Dataset

支持三种模式的PyTorch Dataset:
- Mode 1 (HLA_ONLY): HLA-specific presentation
- Mode 2 (HLA_TISSUE): HLA×Tissue-specific presentation
- Mode 3 (HYBRID): 混合模式 (接口预留)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import json

# 导入配置
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.config.mode_config import ModeConfig, TrainingMode


class ModeAwareDataset(Dataset):
    """
    Mode-aware Dataset

    根据mode配置自动处理不同的数据格式:
    - Mode 1: 返回 (peptide, task_idx, label)
    - Mode 2: 返回 (peptide, task_idx, tissue_idx, label)

    Args:
        task_datasets: Dict[task_id, DataFrame] 每个任务的数据
        task_manager: TaskManager实例
        mode_config: ModeConfig实例
        graph_wrapper: TaskGraphWrapper实例 (提供task_to_idx映射)
        max_len: 肽段最大长度

    Example:

    """

    def __init__(self,
                 task_datasets: Dict[str, pd.DataFrame],
                 task_manager,
                 mode_config: ModeConfig,
                 graph_wrapper,
                 max_len: int = 15):

        self.mode = mode_config.mode
        self.mode_config = mode_config
        self.task_manager = task_manager
        self.graph_wrapper = graph_wrapper
        self.max_len = max_len

        # 氨基酸编码映射
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.aa_to_idx['X'] = len(self.aa_to_idx)  # padding token

        # 构建样本列表
        self.samples = []
        self._build_samples(task_datasets)

        # Mode 2需要构建tissue映射
        if self.mode == TrainingMode.HLA_TISSUE:
            self._build_tissue_mappings()

        print(f"\n✓ ModeAwareDataset created:")
        print(f"  Mode: {mode_config.task_type_name}")
        print(f"  Total samples: {len(self.samples):,}")
        print(f"  Tasks: {len(task_manager.get_all_tasks())}")
        if self.mode == TrainingMode.HLA_TISSUE:
            print(f"  Tissues: {len(self.tissue_to_idx)}")

    def _build_samples(self, task_datasets):
        """构建样本列表"""
        for task_id, task_df in task_datasets.items():
            # 调试: 检查DataFrame的列
            if len(self.samples) == 0:  # 只在第一个task时打印
                print(f"\n  Debug: DataFrame columns for first task ({task_id}):")
                print(f"    {list(task_df.columns)}")
                if len(task_df) > 0:
                    print(f"  Sample row:")
                    print(f"    {dict(task_df.iloc[0])}")

            task = self.task_manager.get_task(task_id)
            task_idx = self.graph_wrapper.task_to_idx[task_id]

            for idx, row in task_df.iterrows():
                sample = {
                    'peptide': row['peptide'],
                    'hla': row['hla'],
                    'label': int(row['label']),
                    'task_id': task_id,
                    'task_idx': task_idx
                }

                # Mode 2需要tissue信息
                if self.mode == TrainingMode.HLA_TISSUE:
                    if 'tissue' not in row:
                        print(f"\n  ⚠ WARNING: 'tissue' column missing in task {task_id}")
                        print(f"    Available columns: {list(task_df.columns)}")
                        print(f"    Row data: {dict(row)}")
                        raise KeyError(f"'tissue' column not found in DataFrame for task {task_id}")
                    sample['tissue'] = row['tissue']

                self.samples.append(sample)

    def _build_tissue_mappings(self):
        """构建tissue映射 (Mode 2)"""
        unique_tissues = sorted(set(s['tissue'] for s in self.samples))
        self.tissue_to_idx = {tissue: idx for idx, tissue in enumerate(unique_tissues)}
        self.idx_to_tissue = {idx: tissue for tissue, idx in self.tissue_to_idx.items()}
        self.n_tissues = len(unique_tissues)

    def encode_peptide(self, peptide: str) -> torch.LongTensor:
        """
        编码肽段序列

        Args:
            peptide: 肽段序列字符串

        Returns:
            tensor: (max_len,) 整数编码
        """
        # 转为索引
        indices = [self.aa_to_idx.get(aa, self.aa_to_idx['X']) for aa in peptide]

        # Padding
        if len(indices) < self.max_len:
            indices += [self.aa_to_idx['X']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return torch.LongTensor(indices)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回一个样本

        Returns:
            dict:
                Mode 1: {'peptide', 'peptide_len', 'task_idx', 'label'}
                Mode 2: {'peptide', 'peptide_len', 'task_idx', 'tissue_idx', 'label'}
        """
        sample = self.samples[idx]

        # 编码peptide
        peptide_encoded = self.encode_peptide(sample['peptide'])
        peptide_len = min(len(sample['peptide']), self.max_len)

        # 基础返回值 (Mode 1和Mode 2共享)
        result = {
            'peptide': peptide_encoded,
            'peptide_len': torch.LongTensor([peptide_len]),
            'task_idx': torch.LongTensor([sample['task_idx']]),
            'label': torch.FloatTensor([sample['label']])
        }

        # Mode 2额外返回tissue_idx
        if self.mode == TrainingMode.HLA_TISSUE:
            tissue_idx = self.tissue_to_idx[sample['tissue']]
            result['tissue_idx'] = torch.LongTensor([tissue_idx])

        return result


def collate_fn_mode_aware(batch):
    """
    自定义collate函数,支持Mode 1和Mode 2

    Args:
        batch: List of samples from ModeAwareDataset

    Returns:
        dict: Batched tensors
    """
    # 提取所有字段
    peptides = torch.stack([item['peptide'] for item in batch])

    # 标量tensor使用stack (不是cat)
    peptide_lens = torch.stack([item['peptide_len'] for item in batch])
    task_idxs = torch.stack([item['task_idx'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # === 确保是1D tensor ===
    # pack_padded_sequence要求lengths是1D
    if peptide_lens.dim() > 1:
        peptide_lens = peptide_lens.squeeze()
    if task_idxs.dim() > 1:
        task_idxs = task_idxs.squeeze()
    if labels.dim() > 1:
        labels = labels.squeeze()

    result = {
        'peptide': peptides,
        'peptide_len': peptide_lens,
        'task_idx': task_idxs,
        'label': labels
    }

    # Mode 2包含tissue_idx
    if 'tissue_idx' in batch[0]:
        tissue_idxs = torch.stack([item['tissue_idx'] for item in batch])
        if tissue_idxs.dim() > 1:
            tissue_idxs = tissue_idxs.squeeze()
        result['tissue_idx'] = tissue_idxs

    return result


def create_mode_aware_dataloaders(
        train_datasets: Dict[str, pd.DataFrame],
        val_datasets: Dict[str, pd.DataFrame],
        test_datasets: Dict[str, pd.DataFrame],
        task_manager,
        mode_config: ModeConfig,
        graph_wrapper,
        batch_size: int = 32,
        num_workers: int = 4,
        max_len: int = 15
):
    """
    创建Mode-aware DataLoaders

    Args:
        train/val/test_datasets: 各个任务的数据集
        task_manager: TaskManager实例
        mode_config: ModeConfig实例
        graph_wrapper: TaskGraphWrapper实例
        batch_size: batch大小
        num_workers: 数据加载进程数
        max_len: 肽段最大长度

    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"\n{'=' * 80}")
    print(f"Creating Mode-Aware DataLoaders")
    print(f"{'=' * 80}")

    # 创建datasets
    train_dataset = ModeAwareDataset(
        train_datasets, task_manager, mode_config, graph_wrapper, max_len
    )

    val_dataset = ModeAwareDataset(
        val_datasets, task_manager, mode_config, graph_wrapper, max_len
    )

    test_dataset = ModeAwareDataset(
        test_datasets, task_manager, mode_config, graph_wrapper, max_len
    )

    # 创建dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_mode_aware,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_mode_aware,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_mode_aware,
        pin_memory=True
    )

    print(f"\n✓ DataLoaders created:")
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset):,} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset):,} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# 向后兼容: 保留旧的PeptideHLADataset类名
# 但功能已被ModeAwareDataset替代
PeptideHLADataset = ModeAwareDataset
create_dataloaders = create_mode_aware_dataloaders

if __name__ == "__main__":
    print("=" * 80)
    print("Testing ModeAwareDataset")
    print("=" * 80)

    # 创建模拟数据
    from src.config.mode_config import create_mode1_config, create_mode2_config
    from src.data.task_definition import Task, TaskManager
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    # Mode 1测试
    print("\n" + "=" * 80)
    print("Testing Mode 1 (HLA_ONLY)")
    print("=" * 80)

    config1 = create_mode1_config()

    # 创建模拟task_manager
    manager1 = TaskManager(mode='hla_only')
    task1 = Task.create_mode1_task(
        hla='HLA-A*02:01',
        data=pd.DataFrame({'label': [1] * 10})
    )
    manager1.add_task(task1)

    # 创建模拟task_datasets
    task_datasets1 = {
        task1.task_id: pd.DataFrame({
            'peptide': ['TESTPEP' + str(i) for i in range(20)],
            'hla': ['HLA-A*02:01'] * 20,
            'label': [1] * 10 + [0] * 10
        })
    }


    # 创建模拟graph_wrapper
    class MockGraphWrapper:
        def __init__(self):
            self.task_to_idx = {task1.task_id: 0}
            self.n_tasks = 1


    graph_wrapper1 = MockGraphWrapper()

    # 创建dataset
    dataset1 = ModeAwareDataset(
        task_datasets1,
        manager1,
        config1,
        graph_wrapper1
    )

    # 测试__getitem__
    sample1 = dataset1[0]
    print(f"\n✓ Mode 1 Sample:")
    for key, value in sample1.items():
        print(f"  {key}: {value.shape if isinstance(value, torch.Tensor) else value}")

    # Mode 2测试
    print("\n" + "=" * 80)
    print("Testing Mode 2 (HLA_TISSUE)")
    print("=" * 80)

    config2 = create_mode2_config()

    manager2 = TaskManager(mode='hla_tissue')
    task2 = Task.create_mode2_task(
        hla='HLA-A*02:01',
        tissue='Liver',
        data=pd.DataFrame({'label': [1] * 10})
    )
    manager2.add_task(task2)

    task_datasets2 = {
        task2.task_id: pd.DataFrame({
            'peptide': ['TESTPEP' + str(i) for i in range(20)],
            'hla': ['HLA-A*02:01'] * 20,
            'tissue': ['Liver'] * 20,
            'label': [1] * 10 + [0] * 10
        })
    }


    class MockGraphWrapper2:
        def __init__(self):
            self.task_to_idx = {task2.task_id: 0}
            self.n_tasks = 1


    graph_wrapper2 = MockGraphWrapper2()

    dataset2 = ModeAwareDataset(
        task_datasets2,
        manager2,
        config2,
        graph_wrapper2
    )

    sample2 = dataset2[0]
    print(f"\n✓ Mode 2 Sample:")
    for key, value in sample2.items():
        print(f"  {key}: {value.shape if isinstance(value, torch.Tensor) else value}")

    # 测试collate_fn
    print("\n" + "=" * 80)
    print("Testing collate_fn")
    print("=" * 80)

    batch1 = collate_fn_mode_aware([dataset1[i] for i in range(4)])
    print(f"\n✓ Mode 1 Batch:")
    for key, value in batch1.items():
        print(f"  {key}: {value.shape}")

    batch2 = collate_fn_mode_aware([dataset2[i] for i in range(4)])
    print(f"\n✓ Mode 2 Batch:")
    for key, value in batch2.items():
        print(f"  {key}: {value.shape}")

    print("\n✓ All tests passed!")