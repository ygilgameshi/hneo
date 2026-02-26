# ImmuneApp Multi-Mode System

## 📋 概述

ImmuneApp现在支持三种运行模式(Mode),而非之前的Phase概念:

- **Mode 1 (HLA_ONLY)**: 仅基于HLA分型的任务定义,建立baseline
- **Mode 2 (HLA_TISSUE)**: HLA×Tissue组合任务,预测tissue-specific presentation
- **Mode 3 (HYBRID)**: 混合模式,联合训练或动态切换 (接口已预留)

## 🎯 核心设计理念

### Mode vs Phase的区别

| 概念 | 定义 | 关键特征 |
|------|------|---------|
| **Phase** (旧) | 训练阶段 | Phase 1→Phase 2是时间顺序的训练流程 |
| **Mode** (新) | 运行模式 | Mode 1/2/3是并行的功能模式,可独立选择 |

### Mode 2的负样本生成哲学

**核心问题**: 如何为HLA×Tissue任务生成合理的负样本?

**解决方案**:
- 负样本peptide本身**不带固有的tissue属性**
- 负样本在特定**task context**(HLA×Tissue)下被评估
- 相当于提问:"这个peptide在Liver的HLA-A*02:01上能被呈递吗?"

```python
# Mode 1: 只有HLA context
task_context = {'hla': 'HLA-A*02:01'}

# Mode 2: HLA + Tissue context  
task_context = {'hla': 'HLA-A*02:01', 'tissue': 'Liver'}

# 负样本peptide在不同context下评估:
# "PEPTIDEABC在HLA-A*02:01 (Mode 1)能呈递吗?" 
# "PEPTIDEABC在Liver的HLA-A*02:01 (Mode 2)能呈递吗?"
```

## 🏗️ 架构设计

### 核心模块

```
src/
├── config/
│   └── mode_config.py          # Mode配置定义
├── data/
│   ├── task_definition.py      # 统一Task定义
│   ├── unified_task_creator.py # Mode-aware任务创建
│   ├── negative_sampler.py     # Mode-aware负样本生成
│   └── dataset.py              # 数据集(需更新)
├── models/
│   └── full_model.py           # 模型(需更新支持Mode)
├── training/
│   └── trainer.py              # 训练器(需更新)
└── graph/
    └── task_graph.py           # Task Graph(需更新)
```

### 数据流

```
1. 原始数据 (IEDB)
   ↓
2. ModeConfig指定运行模式
   ↓  
3. UnifiedTaskCreator创建tasks
   ↓
4. NegativeSampler生成负样本
   ↓
5. TaskGraphBuilder构建task graph
   ↓
6. Model训练和推理
```

## 🚀 使用方法

### Mode 1: HLA-only (Baseline)

```python
from src.config.mode_config import create_mode1_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.negative_sampler import NegativeSampler

# 1. 创建配置
config = create_mode1_config(
    min_samples=20,
    negative_ratio=20
)

# 2. 创建任务
creator = UnifiedTaskCreator(config)
task_manager = creator.create_tasks(
    data_df=train_df,
    save_dir='data/mode1_tasks'
)

# 3. 生成负样本
sampler = NegativeSampler(config, train_df)
task_datasets = sampler.generate_negatives_for_all_tasks(task_manager)

# 4. 训练 (使用现有trainer,需更新接口)
# train_model(task_manager, task_datasets, config)
```

### Mode 2: HLA×Tissue

```python
from src.config.mode_config import create_mode2_config

# 1. 创建配置
config = create_mode2_config(
    min_samples=10,              # Mode 2阈值更低
    negative_ratio=20,
    tissue_source='Host',        # tissue列名
    exclude_unknown_tissue=True,
    use_tissue_aware_negatives=True  # 使用tissue-aware hard negatives
)

# 2-4. 同Mode 1
creator = UnifiedTaskCreator(config)
task_manager = creator.create_tasks(data_df=train_df)
sampler = NegativeSampler(config, train_df)
task_datasets = sampler.generate_negatives_for_all_tasks(task_manager)
```

### Mode 3: Hybrid (接口预留)

```python
from src.config.mode_config import create_mode3_config

# 创建配置
config = create_mode3_config(
    hybrid_strategy='joint',  # 'joint', 'sequential', 'dynamic'
    mode1_weight=0.5,
    mode2_weight=0.5
)

# 具体实现待定
```

### 便捷函数

```python
from src.data.unified_task_creator import create_tasks_for_mode

# 快速创建Mode 1任务
manager1 = create_tasks_for_mode('mode1', train_df, min_samples=20)

# 快速创建Mode 2任务
manager2 = create_tasks_for_mode('mode2', train_df, 
                                 min_samples=10, 
                                 tissue_source='Host')
```

## 📊 负样本生成策略

### Mode 1: HLA-only

```
Random peptides (70%)     → Easy negatives
  └─ 从全部peptide池随机采样
  
Cross-HLA positives (30%) → Hard negatives  
  └─ 其他HLA的阳性样本
```

### Mode 2: HLA×Tissue

```
Random peptides (50%)                    → Easy negatives
  └─ 通用负样本,适用于任何tissue
  
Cross-HLA within tissue (25%)            → Hard negatives
  └─ 同tissue其他HLA的阳性
  └─ 例: Liver的HLA-B*07:02阳性 作为 Liver的HLA-A*02:01负样本
  
Cross-tissue within HLA (25%)            → Hard negatives
  └─ 同HLA其他tissue的阳性  
  └─ 例: Lung的HLA-A*02:01阳性 作为 Liver的HLA-A*02:01负样本
```

**关键**: 所有负样本peptide在特定task context下评估,peptide本身不带tissue标签

## 🔧 待完成的模块更新

### 1. Dataset类更新

需要让Dataset支持Mode参数:

```python
# src/data/dataset.py

class ModeAwareDataset(Dataset):
    def __init__(self, task_datasets, task_manager, mode_config):
        self.mode = mode_config.mode
        # ...
        
    def __getitem__(self, idx):
        # 根据mode返回不同格式的data
        if self.mode == TrainingMode.HLA_ONLY:
            return {'peptide': ..., 'hla': ..., 'label': ...}
        elif self.mode == TrainingMode.HLA_TISSUE:
            return {'peptide': ..., 'hla': ..., 'tissue': ..., 'label': ...}
```

### 2. Task Graph更新

需要支持HLA×Tissue节点:

```python
# src/graph/task_graph.py

class ModeAwareTaskGraphBuilder:
    def __init__(self, task_manager, mode_config):
        self.mode = mode_config.mode
        
    def build_similarity_matrix(self):
        if self.mode == TrainingMode.HLA_ONLY:
            # 基于HLA序列相似性
            return self._compute_hla_similarity()
        elif self.mode == TrainingMode.HLA_TISSUE:
            # 基于HLA相似性 + tissue共现
            return self._compute_hla_tissue_similarity()
```

### 3. Model更新

需要让模型接受mode参数:

```python
# src/models/full_model.py

class ImmuneAppModel(nn.Module):
    def __init__(self, mode_config, ...):
        self.mode = mode_config.mode
        
        # Mode 2需要tissue embedding
        if self.mode == TrainingMode.HLA_TISSUE:
            self.tissue_embedder = TissueEmbedder(...)
        
    def forward(self, peptide, task_idx, graph_data):
        # 根据mode选择不同的forward路径
        if self.mode == TrainingMode.HLA_ONLY:
            task_emb = self.task_gnn(task_idx, graph_data)
        elif self.mode == TrainingMode.HLA_TISSUE:
            task_emb = self.task_gnn_with_tissue(task_idx, graph_data)
```

### 4. Trainer更新

统一训练接口:

```python
# src/training/trainer.py

def train_model(task_manager, task_datasets, mode_config, **kwargs):
    """统一训练函数,支持所有mode"""
    
    if mode_config.mode == TrainingMode.HLA_ONLY:
        return _train_mode1(...)
    elif mode_config.mode == TrainingMode.HLA_TISSUE:
        return _train_mode2(...)
    elif mode_config.mode == TrainingMode.HYBRID:
        return _train_mode3(...)
```

## 🧪 测试

每个模块都包含测试代码:

```bash
# 测试配置
python -m src.config.mode_config

# 测试Task定义
python -m src.data.task_definition

# 测试任务创建
python -m src.data.unified_task_creator

# 测试负样本生成
python -m src.data.negative_sampler
```

## 📝 示例脚本

创建完整的训练脚本示例:

```python
# scripts/train_mode1.py
from src.config.mode_config import create_mode1_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.negative_sampler import NegativeSampler
import pandas as pd

# 加载数据
train_df = pd.read_csv('data/train.csv')

# Mode 1训练
config = create_mode1_config(min_samples=20, negative_ratio=20)
creator = UnifiedTaskCreator(config)
task_manager = creator.create_tasks(train_df, save_dir='data/mode1_tasks')

sampler = NegativeSampler(config, train_df)
task_datasets = sampler.generate_negatives_for_all_tasks(task_manager)

# TODO: 集成到trainer
# model = train_model(task_manager, task_datasets, config)
```

## 🎯 优势总结

### 相比原有Phase系统:

1. **清晰的概念分离**: Mode是功能选择,不是时间顺序
2. **统一的代码架构**: 三种模式共享大部分代码
3. **灵活性**: 可以独立运行任何mode,不需要先跑Mode 1
4. **可扩展性**: Mode 3接口已预留,未来容易扩展
5. **向后兼容**: Mode 1保持原有Phase 1的功能

### 负样本生成的生物学合理性:

- Mode 1: 纯binding prediction,不考虑生物学背景
- Mode 2: 在特定tissue context下评估,更接近真实生物学场景
- 负样本不需要显式tissue标签,通过task context隐式传递

## 📚 接下来的步骤

1. ✅ 创建Mode配置系统
2. ✅ 创建统一Task定义
3. ✅ 创建任务创建器
4. ✅ 创建负样本生成器
5. ⏳ 更新Dataset类
6. ⏳ 更新Task Graph构建
7. ⏳ 更新Model架构
8. ⏳ 更新Trainer
9. ⏳ 创建端到端训练脚本
10. ⏳ 测试和验证

---

**作者**: xty116  
**日期**: 2025-01-16  
**版本**: 1.0
