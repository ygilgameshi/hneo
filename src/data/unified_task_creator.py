"""
Mode-Aware Task Creator

统一的任务创建接口,支持三种模式:
- Mode 1: HLA-only tasks
- Mode 2: HLA×Tissue tasks
- Mode 3: Hybrid tasks (接口预留)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

from .task_definition import Task, TaskManager
from ..config.mode_config import ModeConfig, TrainingMode


class UnifiedTaskCreator:
    """
    统一任务创建器
    
    根据ModeConfig自动选择正确的任务创建逻辑
    """
    
    def __init__(self, config: ModeConfig):
        """
        初始化
        
        Args:
            config: 模式配置
        """
        self.config = config
        self.mode = config.mode
        
        print(f"\n{'='*80}")
        print(f"Task Creator - {config.task_type_name}")
        print(f"{'='*80}")
    
    def create_tasks(self, 
                     data_df: pd.DataFrame,
                     save_dir: Optional[Path] = None) -> TaskManager:
        """
        创建任务
        
        Args:
            data_df: 数据DataFrame,需包含['peptide', 'hla', 'label']
                    Mode 2还需要'tissue'列
            save_dir: 保存目录
        
        Returns:
            TaskManager with created tasks
        """
        # 根据mode调用不同的创建逻辑
        if self.mode == TrainingMode.HLA_ONLY:
            manager = self._create_mode1_tasks(data_df)
        
        elif self.mode == TrainingMode.HLA_TISSUE:
            manager = self._create_mode2_tasks(data_df)
        
        elif self.mode == TrainingMode.HYBRID:
            manager = self._create_mode3_tasks(data_df)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # 打印摘要
        manager.print_summary()
        
        # 保存
        if save_dir is not None:
            self._save_tasks(manager, save_dir)
        
        return manager
    
    def _create_mode1_tasks(self, data_df: pd.DataFrame) -> TaskManager:
        """
        创建Mode 1任务 (HLA-only)
        
        每个HLA分型是一个task
        """
        print(f"\n创建Mode 1任务 (min_samples={self.config.min_samples})...")
        
        manager = TaskManager(mode='hla_only')
        
        created_count = 0
        filtered_count = 0
        
        for hla in data_df['hla'].unique():
            hla_data = data_df[data_df['hla'] == hla]
            
            # 样本数过滤
            if len(hla_data) < self.config.min_samples:
                filtered_count += 1
                continue
            
            # 创建task
            task = Task.create_mode1_task(hla=hla, data=hla_data)
            manager.add_task(task)
            created_count += 1
        
        print(f"  ✓ 创建了 {created_count} 个任务")
        print(f"  × 过滤了 {filtered_count} 个任务 (样本数<{self.config.min_samples})")
        
        return manager
    
    def _create_mode2_tasks(self, data_df: pd.DataFrame) -> TaskManager:
        """
        创建Mode 2任务 (HLA×Tissue)
        
        每个HLA×Tissue组合是一个task
        """
        print(f"\n创建Mode 2任务 (min_samples={self.config.min_samples})...")
        
        # 验证必要列
        if 'tissue' not in data_df.columns:
            raise ValueError("Mode 2 requires 'tissue' column in data")
        
        # 过滤Unknown tissue
        if self.config.exclude_unknown_tissue:
            data_df = data_df[data_df['tissue'] != 'Unknown'].copy()
            print(f"  → 已排除 Unknown tissue")
        
        manager = TaskManager(mode='hla_tissue')
        
        created_count = 0
        filtered_count = 0
        
        # 按HLA×Tissue分组创建任务
        for (hla, tissue), group in data_df.groupby(['hla', 'tissue']):
            # 样本数过滤
            if len(group) < self.config.min_samples:
                filtered_count += 1
                continue
            
            # 创建task
            task = Task.create_mode2_task(hla=hla, tissue=tissue, data=group)
            manager.add_task(task)
            created_count += 1
        
        print(f"  ✓ 创建了 {created_count} 个任务")
        print(f"  × 过滤了 {filtered_count} 个任务 (样本数<{self.config.min_samples})")
        
        return manager
    
    def _create_mode3_tasks(self, data_df: pd.DataFrame) -> TaskManager:
        """
        创建Mode 3任务 (Hybrid) - 接口预留
        
        可能的策略:
        - Joint: 同时包含Mode 1和Mode 2任务
        - Sequential: 先Mode 1后Mode 2
        - Dynamic: 动态切换
        """
        print(f"\n创建Mode 3任务 (Hybrid)...")
        print(f"  Strategy: {self.config.hybrid_strategy}")
        
        if self.config.hybrid_strategy == 'joint':
            # 同时创建Mode 1和Mode 2任务
            # 这里只是接口示例,具体实现待定
            manager = TaskManager(mode='hybrid')
            print(f"  ✓ Hybrid task creation (to be implemented)")
            
        else:
            raise NotImplementedError(f"Hybrid strategy '{self.config.hybrid_strategy}' not implemented")
        
        return manager
    
    def _save_tasks(self, manager: TaskManager, save_dir: Path):
        """保存任务定义"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存任务列表
        with open(save_dir / 'tasks.json', 'w') as f:
            json.dump(manager.to_dict(), f, indent=2)
        
        # 保存配置
        with open(save_dir / 'task_config.json', 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"\n✓ 任务定义已保存到: {save_dir}")
    
    @staticmethod
    def load_tasks(save_dir: Path) -> Tuple[TaskManager, ModeConfig]:
        """
        加载任务定义
        
        Returns:
            (TaskManager, ModeConfig)
        """
        save_dir = Path(save_dir)
        
        # 加载任务
        with open(save_dir / 'tasks.json') as f:
            task_data = json.load(f)
        manager = TaskManager.from_dict(task_data)
        
        # 加载配置
        with open(save_dir / 'task_config.json') as f:
            config_data = json.load(f)
        
        # 重建config (简化版)
        from ..config.mode_config import TrainingMode
        config = ModeConfig(
            mode=TrainingMode(config_data['mode']),
            min_samples=config_data['min_samples'],
            negative_ratio=config_data['negative_ratio'],
        )
        
        print(f"\n✓ 已加载任务: {manager.get_statistics()['n_tasks']} tasks")
        
        return manager, config


def create_tasks_for_mode(mode: str,
                         data_df: pd.DataFrame,
                         min_samples: int = None,
                         tissue_source: str = 'Host',
                         save_dir: Optional[Path] = None,
                         **kwargs) -> TaskManager:
    """
    便捷函数:为指定模式创建任务
    
    Args:
        mode: 'mode1'/'hla_only', 'mode2'/'hla_tissue', 'mode3'/'hybrid'
        data_df: 数据
        min_samples: 最小样本数
        tissue_source: Mode 2的tissue列名
        save_dir: 保存目录
        **kwargs: 其他配置参数
    
    Returns:
        TaskManager
    """
    from ..config.mode_config import TrainingMode, ModeConfig
    
    # 解析mode
    training_mode = TrainingMode.from_string(mode)
    
    # 创建配置
    if training_mode == TrainingMode.HLA_ONLY:
        config = ModeConfig(
            mode=training_mode,
            min_samples=min_samples or 20,
            **kwargs
        )
    
    elif training_mode == TrainingMode.HLA_TISSUE:
        config = ModeConfig(
            mode=training_mode,
            min_samples=min_samples or 10,
            tissue_source=tissue_source,
            **kwargs
        )
    
    elif training_mode == TrainingMode.HYBRID:
        config = ModeConfig(
            mode=training_mode,
            hybrid_strategy=kwargs.get('hybrid_strategy', 'joint'),
            **kwargs
        )
    
    # 创建任务
    creator = UnifiedTaskCreator(config)
    manager = creator.create_tasks(data_df, save_dir=save_dir)
    
    return manager


if __name__ == "__main__":
    # 测试任务创建
    print("=" * 80)
    print("Unified Task Creator Test")
    print("=" * 80)
    
    # 创建模拟数据
    np.random.seed(42)
    hlas = [f'HLA-A*{i:02d}:01' for i in range(1, 6)]
    tissues = ['Lymphoid', 'Blood', 'Lung', 'Liver']
    
    data = []
    for hla in hlas:
        for tissue in tissues:
            n = np.random.randint(5, 30)
            for _ in range(n):
                data.append({
                    'peptide': 'TEST' + ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), 6)),
                    'hla': hla,
                    'tissue': tissue,
                    'label': np.random.choice([0, 1])
                })
    
    df = pd.DataFrame(data)
    print(f"\n测试数据: {len(df)} samples, {df['hla'].nunique()} HLAs, {df['tissue'].nunique()} tissues")
    
    # 测试Mode 1
    print("\n" + "=" * 80)
    print("Testing Mode 1")
    print("=" * 80)
    manager1 = create_tasks_for_mode('mode1', df, min_samples=15)
    
    # 测试Mode 2
    print("\n" + "=" * 80)
    print("Testing Mode 2")
    print("=" * 80)
    manager2 = create_tasks_for_mode('mode2', df, min_samples=10)
    
    print("\n✓ All tests passed!")
