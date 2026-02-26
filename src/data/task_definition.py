"""
统一Task定义

支持三种模式的任务定义:
- Mode 1: 基于HLA的任务
- Mode 2: 基于HLA×Tissue组合的任务  
- Mode 3: 混合任务 (接口预留)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class Task:
    """
    统一的Task定义
    
    核心设计:
    - Mode 1: task_id = "HLA_{hla}", tissue = None
    - Mode 2: task_id = "HLA_{hla}_Tissue_{tissue}", tissue有值
    - Mode 3: 待定
    """
    task_id: str                    # 唯一标识
    mode: str                       # 'hla_only', 'hla_tissue', 'hybrid'
    hla: str                        # HLA分型
    tissue: Optional[str] = None    # 组织类型(Mode 2)
    
    # 数据统计
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    
    # 任务级别的元信息
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """验证task定义合理性"""
        if self.metadata is None:
            self.metadata = {}
            
        # 验证mode和tissue的一致性
        if self.mode == 'hla_only' and self.tissue is not None:
            raise ValueError("Mode 1 (hla_only) should not have tissue")
        
        if self.mode == 'hla_tissue' and self.tissue is None:
            raise ValueError("Mode 2 (hla_tissue) requires tissue")
    
    @property
    def is_mode1(self) -> bool:
        """是否为Mode 1任务"""
        return self.mode == 'hla_only'
    
    @property
    def is_mode2(self) -> bool:
        """是否为Mode 2任务"""
        return self.mode == 'hla_tissue'
    
    @property
    def is_hybrid(self) -> bool:
        """是否为Hybrid任务"""
        return self.mode == 'hybrid'
    
    @property
    def task_context(self) -> Dict[str, Any]:
        """
        返回任务context,用于负样本生成和模型推理
        
        Mode 1: {'hla': 'HLA-A*02:01'}
        Mode 2: {'hla': 'HLA-A*02:01', 'tissue': 'Liver'}
        """
        context = {'hla': self.hla}
        if self.tissue is not None:
            context['tissue'] = self.tissue
        return context
    
    def to_dict(self) -> dict:
        """转为字典"""
        return {
            'task_id': self.task_id,
            'mode': self.mode,
            'hla': self.hla,
            'tissue': self.tissue,
            'n_samples': int(self.n_samples) if self.n_samples is not None else None,
            'n_positive': int(self.n_positive) if self.n_positive is not None else None,
            'n_negative': int(self.n_negative) if self.n_negative is not None else None,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        """从字典创建"""
        return cls(**data)

    @staticmethod
    def create_mode1_task(hla: str, data: pd.DataFrame) -> 'Task':
        """创建Mode 1任务"""
        task_id = f"HLA_{hla}"

        return Task(
            task_id=task_id,
            mode='hla_only',
            hla=hla,
            tissue=None,
            n_samples=len(data),
            n_positive=(data['label'] == 1).sum() if 'label' in data.columns else 0,
            n_negative=(data['label'] == 0).sum() if 'label' in data.columns else 0,
        )

    @staticmethod
    def create_mode2_task(hla: str, tissue: str, data: pd.DataFrame) -> 'Task':
        """创建Mode 2任务"""
        task_id = f"HLA_{hla}_Tissue_{tissue}"

        return Task(
            task_id=task_id,
            mode='hla_tissue',
            hla=hla,
            tissue=tissue,
            n_samples=len(data),
            n_positive=(data['label'] == 1).sum() if 'label' in data.columns else 0,
            n_negative=(data['label'] == 0).sum() if 'label' in data.columns else 0,
        )

    def __repr__(self) -> str:
        if self.is_mode1:
            return f"Task(mode=1, hla={self.hla}, n={self.n_samples})"
        elif self.is_mode2:
            return f"Task(mode=2, hla={self.hla}, tissue={self.tissue}, n={self.n_samples})"
        else:
            return f"Task(mode=hybrid, hla={self.hla}, n={self.n_samples})"


class TaskManager:
    """
    任务管理器

    统一管理三种模式的任务集合
    """
    def __init__(self, mode: str = 'hla_only'):
        self.mode = mode
        self.tasks: Dict[str, Task] = {}

    def add_task(self, task: Task):
        """添加任务"""
        if task.mode != self.mode:
            raise ValueError(f"Task mode {task.mode} doesn't match manager mode {self.mode}")
        self.tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, Task]:
        """获取所有任务"""
        return self.tasks

    def get_task_by_hla(self, hla: str) -> list:
        """获取指定HLA的所有任务"""
        return [t for t in self.tasks.values() if t.hla == hla]

    def get_task_by_tissue(self, tissue: str) -> list:
        """获取指定tissue的所有任务 (Mode 2)"""
        if self.mode != 'hla_tissue':
            return []
        return [t for t in self.tasks.values() if t.tissue == tissue]

    def get_statistics(self) -> dict:
        """获取任务统计信息"""
        tasks_list = list(self.tasks.values())

        stats = {
            'mode': self.mode,
            'n_tasks': int(len(tasks_list)),
            'total_samples': int(sum(t.n_samples for t in tasks_list)),
            'total_positive': int(sum(t.n_positive for t in tasks_list)),
            'total_negative': int(sum(t.n_negative for t in tasks_list)),
        }

        if tasks_list:
            stats['avg_samples_per_task'] = float(stats['total_samples'] / len(tasks_list))
            stats['min_samples'] = int(min(t.n_samples for t in tasks_list))
            stats['max_samples'] = int(max(t.n_samples for t in tasks_list))

        if self.mode == 'hla_only':
            stats['n_hlas'] = int(len(set(t.hla for t in tasks_list)))
        elif self.mode == 'hla_tissue':
            stats['n_hlas'] = int(len(set(t.hla for t in tasks_list)))
            stats['n_tissues'] = int(len(set(t.tissue for t in tasks_list)))
            stats['n_combinations'] = int(len(tasks_list))

        return stats

    def print_summary(self):
        """打印任务摘要"""
        stats = self.get_statistics()

        print(f"\n{'='*80}")
        print(f"Task Manager Summary - Mode: {self.mode.upper()}")
        print(f"{'='*80}")

        print(f"\n任务数量: {stats['n_tasks']}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"  正样本: {stats['total_positive']}")
        print(f"  负样本: {stats['total_negative']}")

        if 'avg_samples_per_task' in stats:
            print(f"\n每任务样本数:")
            print(f"  平均: {stats['avg_samples_per_task']:.1f}")
            print(f"  范围: [{stats['min_samples']}, {stats['max_samples']}]")

        if self.mode == 'hla_only':
            print(f"\nHLA数量: {stats['n_hlas']}")
        elif self.mode == 'hla_tissue':
            print(f"\nHLA数量: {stats['n_hlas']}")
            print(f"Tissue数量: {stats['n_tissues']}")
            print(f"HLA×Tissue组合: {stats['n_combinations']}")

    def to_dict(self) -> dict:
        """转为字典"""
        return {
            'mode': self.mode,
            'tasks': {tid: task.to_dict() for tid, task in self.tasks.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TaskManager':
        """从字典创建"""
        manager = cls(mode=data['mode'])
        for task_dict in data['tasks'].values():
            manager.add_task(Task.from_dict(task_dict))
        return manager


if __name__ == "__main__":
    # 测试Task定义
    print("=" * 80)
    print("Task Definition Test")
    print("=" * 80)

    # Mode 1 Task
    task1 = Task.create_mode1_task(
        hla='HLA-A*02:01',
        data=pd.DataFrame({'label': [1, 0, 1, 1, 0]})
    )
    print(f"\n✓ Mode 1 Task: {task1}")
    print(f"  Task ID: {task1.task_id}")
    print(f"  Context: {task1.task_context}")

    # Mode 2 Task
    task2 = Task.create_mode2_task(
        hla='HLA-A*02:01',
        tissue='Liver',
        data=pd.DataFrame({'label': [1, 0, 1]})
    )
    print(f"\n✓ Mode 2 Task: {task2}")
    print(f"  Task ID: {task2.task_id}")
    print(f"  Context: {task2.task_context}")

    # Task Manager
    manager = TaskManager(mode='hla_tissue')
    manager.add_task(task2)
    manager.print_summary()