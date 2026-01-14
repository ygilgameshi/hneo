"""
PyTorch Dataset 和 DataLoader

用于加载训练/验证/测试数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json


class PeptideHLADataset(Dataset):
    """
    Peptide-HLA 配对数据集

    Args:
        csv_file: train.csv / val.csv / test.csv
        task_mapping_file: task_mapping.json
        mode: 'train' / 'val' / 'test'
        max_len: 肽段最大长度（padding到此长度）

    Example:
        >>> dataset = PeptideHLADataset('data/phase1_dataset/train.csv',
        ...                             'data/phase1_dataset/task_mapping.json')
        >>> sample = dataset[0]
        >>> print(sample.keys())
    """

    def __init__(self, csv_file, task_mapping_file, mode='train', max_len=15):
        self.df = pd.read_csv(csv_file)
        self.mode = mode
        self.max_len = max_len

        # 加载 task 映射
        with open(task_mapping_file) as f:
            task_mapping = json.load(f)

        self.task_to_idx = task_mapping['task_to_idx']
        self.tasks = task_mapping['tasks']

        # 氨基酸编码
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.aa_to_idx['X'] = len(self.aa_to_idx)  # padding token

        print(f"{mode.upper()} Dataset: {len(self.df):,} samples")

    def __len__(self):
        return len(self.df)

    def encode_peptide(self, peptide):
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

    def __getitem__(self, idx):
        """
        返回一个样本

        Returns:
            dict: {
                'peptide': tensor (max_len,),
                'peptide_len': int,
                'task_idx': int,
                'label': int (0 or 1)
            }
        """
        row = self.df.iloc[idx]

        peptide = row['Peptide']
        peptide_len = int(row['Peptide_Length'])
        hla = row['MHC_Restriction_Name']
        label = int(row['Label'])

        # 编码肽段
        peptide_encoded = self.encode_peptide(peptide)

        # 获取 task 索引
        task_idx = self.task_to_idx.get(hla, -1)

        # 如果 task 不在映射中，跳过（返回默认值）
        if task_idx == -1:
            # 使用第一个 task 作为默认
            task_idx = 0

        return {
            'peptide': peptide_encoded,
            'peptide_len': torch.LongTensor([peptide_len]),
            'task_idx': torch.LongTensor([task_idx]),
            'label': torch.LongTensor([label]),
        }


def collate_fn(batch):
    """
    自定义 collate 函数

    将 list of dicts 转换为 dict of tensors
    """
    peptides = torch.stack([item['peptide'] for item in batch])
    peptide_lens = torch.cat([item['peptide_len'] for item in batch])
    task_idxs = torch.cat([item['task_idx'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])

    return {
        'peptide': peptides,
        'peptide_len': peptide_lens,
        'task_idx': task_idxs,
        'label': labels,
    }


def create_dataloaders(
        dataset_dir='data/phase1_dataset',
        batch_size=32,
        num_workers=4,
        meta_learning=False
):
    """
    创建数据加载器

    Args:
        dataset_dir: 数据集目录
        batch_size: batch 大小
        num_workers: 数据加载的进程数
        meta_learning: 是否用于 meta-learning（目前未使用）

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_dir = Path(dataset_dir)
    task_mapping_file = dataset_dir / 'task_mapping.json'

    # 创建 datasets
    train_dataset = PeptideHLADataset(
        dataset_dir / 'train.csv',
        task_mapping_file,
        mode='train'
    )

    val_dataset = PeptideHLADataset(
        dataset_dir / 'val.csv',
        task_mapping_file,
        mode='val'
    )

    test_dataset = PeptideHLADataset(
        dataset_dir / 'test.csv',
        task_mapping_file,
        mode='test'
    )

    # 创建 dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ===== 新增：支持组织信息的Dataset =====
class ImmuneAppDatasetWithTissue(Dataset):
    """
    Phase 2: 支持组织信息的数据集
    兼容你的TSV格式数据
    """

    def __init__(self, data_path, tissue_source='Inferred_Tissue',
                 normalize_tissue=True, filter_unknown=False):
        print(f"\n加载数据: {data_path}")
        self.data = pd.read_csv(data_path, sep='\t')

        # 重命名列以匹配内部使用
        self.data = self.data.rename(columns={
            'MHC_Restriction_Name': 'hla',
            'Peptide': 'peptide',
            tissue_source: 'tissue',
            'Label': 'label'
        })

        # 处理组织信息
        self.data['tissue'] = self.data['tissue'].fillna('Unknown')

        if normalize_tissue:
            self.data['tissue'] = self.data['tissue'].apply(
                self._normalize_tissue_name
            )

        if filter_unknown:
            before = len(self.data)
            self.data = self.data[self.data['tissue'] != 'Unknown']
            print(f"  过滤Unknown: {before} -> {len(self.data)}")

        self._print_statistics()

    def _normalize_tissue_name(self, tissue):
        """标准化组织名称"""
        if pd.isna(tissue) or str(tissue).strip() == '':
            return 'Unknown'

        tissue = str(tissue).strip().lower()

        tissue_mapping = {
            'lymphoid': 'Lymphoid',
            'blood': 'Blood',
            'pbmc': 'Blood',
            'lung': 'Lung',
            'liver': 'Liver',
            'brain': 'Brain',
            'skin': 'Skin',
            'breast': 'Breast',
            'colon': 'Colon',
            'kidney': 'Kidney',
            'pancreas': 'Pancreas',
            'stomach': 'Stomach',
        }

        for key, value in tissue_mapping.items():
            if key in tissue:
                return value

        return tissue.capitalize()

    def _print_statistics(self):
        """打印统计信息"""
        print(f"\n数据集统计:")
        print(f"  总样本: {len(self.data)}")
        print(f"  正样本: {(self.data['label'] == 1).sum()}")
        print(f"  HLA类型: {self.data['hla'].nunique()}")
        print(f"  组织类型: {self.data['tissue'].nunique()}")

        print(f"\n  各组织样本数:")
        for tissue, count in self.data['tissue'].value_counts().items():
            pos = ((self.data['tissue'] == tissue) &
                   (self.data['label'] == 1)).sum()
            print(f"    {tissue:15s}: {count:5d} (正:{pos:4d})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'peptide': row['peptide'],
            'hla': row['hla'],
            'tissue': row['tissue'],
            'label': row['label']
        }

    def get_dataframe(self):
        """返回DataFrame用于任务创建"""
        return self.data[['peptide', 'hla', 'tissue', 'label']].copy()


# 测试代码
if __name__ == "__main__":
    print("Testing PeptideHLADataset...")

    # 需要实际的数据文件来测试
    # 这里只打印提示
    print("✓ Dataset module loaded successfully!")
    print("  Use with actual data files for full testing")