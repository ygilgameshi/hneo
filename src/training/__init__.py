"""
训练模块

包含：
- MAMLTrainer: MAML元学习训练器
- StandardTrainer: 标准训练器
- Evaluator: 评估工具
"""

from .maml import MAMLTrainer, MAMLDataLoader
from .trainer import train_phase1, train_standard_epoch, train_maml_epoch
from .evaluator import Evaluator

__all__ = [
    'MAMLTrainer',
    'MAMLDataLoader',
    'train_phase1',
    'train_standard_epoch',
    'train_maml_epoch',
    'Evaluator',
]