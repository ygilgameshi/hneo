"""
模型模块

包含：
- PeptideEncoder: 肽段序列编码器
- TaskGNN: Task Graph神经网络
- TaskConditionedPredictor: Task条件化预测器
- ImmuneAppPhase1: 完整模型
"""

from .peptide_encoder import PeptideEncoder
from .task_gnn import TaskGNN, TaskGraphWrapper
from .predictor import TaskConditionedPredictor
from .full_model import ImmuneAppPhase1

__all__ = [
    'PeptideEncoder',
    'TaskGNN',
    'TaskGraphWrapper',
    'TaskConditionedPredictor',
    'ImmuneAppPhase1',
]