# Phase → Mode 迁移指南

## 📋 变更概述

### 核心概念变化

```
旧系统 (Phase-based):
├── Phase 1: HLA-only任务,建立baseline
├── Phase 2: HLA×Tissue任务,渐进式训练
└── Phase概念暗示时间顺序: 必须先Phase 1后Phase 2

新系统 (Mode-based):
├── Mode 1: HLA-only,独立运行
├── Mode 2: HLA×Tissue,独立运行  
├── Mode 3: Hybrid,混合模式(接口预留)
└── Mode概念表示功能选择: 可以独立选择任意mode
```

### 为什么要迁移?

1. **概念更清晰**: Mode表示功能差异,不是时间顺序
2. **更灵活**: 可以直接训练Mode 2,不需要先跑Mode 1
3. **更易扩展**: Mode 3等新模式更容易添加
4. **代码更统一**: 三种模式共享核心代码框架

## 🔄 代码迁移映射

### 1. 配置系统

**旧代码**:
```python
# Phase 1
use_phase2 = False
phase1_epochs = 50

# Phase 2  
use_phase2 = True
phase2_epochs = 30
tissue_source = 'Host'
```

**新代码**:
```python
from src.config.mode_config import create_mode1_config, create_mode2_config

# Mode 1
config = create_mode1_config(
    min_samples=20,
    negative_ratio=20
)

# Mode 2
config = create_mode2_config(
    min_samples=10,
    tissue_source='Host',
    use_tissue_aware_negatives=True
)
```

### 2. 任务创建

**旧代码**:
```python
# data/task_creator.py - create_phase2_tasks()
tasks = create_phase2_tasks(
    data_df,
    phase='phase1',  # or 'phase2', 'combined'
    min_samples_phase1=20,
    min_samples_phase2=10
)
```

**新代码**:
```python
from src.data.unified_task_creator import UnifiedTaskCreator

# Mode 1
creator = UnifiedTaskCreator(config)
task_manager = creator.create_tasks(data_df)

# 或使用便捷函数
from src.data.unified_task_creator import create_tasks_for_mode
task_manager = create_tasks_for_mode('mode1', data_df, min_samples=20)
```

### 3. Task定义

**旧代码**:
```python
# task字典格式不统一
task = {
    'data': df,
    'type': 'phase1',  # or 'phase2'
    'hla': 'HLA-A*02:01',
    'tissue': None,  # phase1没有, phase2有
    'n_samples': 100,
    ...
}
```

**新代码**:
```python
from src.data.task_definition import Task

# Mode 1
task = Task.create_mode1_task(hla='HLA-A*02:01', data=df)

# Mode 2  
task = Task.create_mode2_task(
    hla='HLA-A*02:01',
    tissue='Liver',
    data=df
)

# 统一接口
print(task.task_id)        # 唯一标识
print(task.mode)           # 'hla_only' or 'hla_tissue'
print(task.task_context)   # {'hla': ..., 'tissue': ...}
```

### 4. 负样本生成

**旧代码**:
```python
# 负样本生成分散在不同地方
# Phase 1: 在dataset中生成
# Phase 2: 在task creator中生成
# 逻辑不统一
```

**新代码**:
```python
from src.data.negative_sampler import NegativeSampler

# 统一接口
sampler = NegativeSampler(config, all_data)

# 为单个任务生成
negatives = sampler.generate_negatives(task, positives)

# 为所有任务生成  
task_datasets = sampler.generate_negatives_for_all_tasks(task_manager)
```

### 5. 训练流程

**旧代码**:
```python
# src/training/trainer.py
def train_phase1(use_phase2=False, ...):
    if use_phase2:
        # Phase 2特殊逻辑
        trainer = MAMLTrainerPhase2(...)
        history = trainer.progressive_train(
            phase1_epochs=50,
            phase2_epochs=30,
            ...
        )
    else:
        # Phase 1逻辑
        trainer = MAMLTrainer(...)
        for epoch in range(n_epochs):
            ...
```

**新代码** (待实现):
```python
# src/training/trainer.py
def train_model(task_manager, task_datasets, mode_config, **kwargs):
    """统一训练函数"""
    
    if mode_config.mode == TrainingMode.HLA_ONLY:
        return _train_mode1(task_manager, task_datasets, mode_config, **kwargs)
    
    elif mode_config.mode == TrainingMode.HLA_TISSUE:
        return _train_mode2(task_manager, task_datasets, mode_config, **kwargs)
    
    elif mode_config.mode == TrainingMode.HYBRID:
        return _train_mode3(task_manager, task_datasets, mode_config, **kwargs)
```

## 📂 文件组织变化

### 新增文件

```
src/
├── config/
│   └── mode_config.py          ✨ 新增: Mode配置
├── data/
│   ├── task_definition.py      ✨ 新增: 统一Task定义
│   ├── unified_task_creator.py ✨ 新增: Mode-aware任务创建
│   └── negative_sampler.py     ✨ 新增: Mode-aware负样本生成
└── docs/
    ├── MODE_SYSTEM.md          ✨ 新增: Mode系统文档
    └── MIGRATION_GUIDE.md      ✨ 新增: 迁移指南(本文件)
```

### 需要更新的文件

```
src/
├── data/
│   ├── dataset.py              🔄 需更新: 支持Mode参数
│   └── task_creator.py         ⚠️  将被弃用,使用unified_task_creator
├── models/
│   └── full_model.py           🔄 需更新: Mode-aware model
├── training/
│   ├── trainer.py              🔄 需更新: 统一训练接口
│   └── maml.py                 🔄 需更新: Mode-aware MAML
└── graph/
    └── task_graph.py           🔄 需更新: 支持HLA×Tissue节点
```

### 保持不变的文件

```
src/
├── models/
│   ├── peptide_encoder.py      ✅ 不变
│   ├── task_gnn.py             ✅ 不变(可能小幅增强)
│   └── predictor.py            ✅ 不变
└── utils/
    ├── metrics.py              ✅ 不变
    └── visualization.py        ✅ 不变
```

## 🔧 迁移步骤

### Step 1: 测试新模块 ✅

```bash
# 测试配置系统
python -m src.config.mode_config

# 测试Task定义
python -m src.data.task_definition

# 测试任务创建
python -m src.data.unified_task_creator

# 测试负样本生成
python -m src.data.negative_sampler
```

### Step 2: 更新Dataset类 ⏳

```python
# src/data/dataset.py

from ..config.mode_config import ModeConfig, TrainingMode

class ModeAwareDataset(Dataset):
    """支持Mode的Dataset"""
    
    def __init__(self, task_datasets, task_manager, mode_config, ...):
        self.mode = mode_config.mode
        self.task_manager = task_manager
        # ...
        
    def __getitem__(self, idx):
        # Mode 1: 返回 (peptide, hla, label)
        # Mode 2: 返回 (peptide, hla, tissue, label)
        # 根据mode动态调整返回格式
        ...
```

### Step 3: 更新Task Graph ⏳

```python
# src/graph/task_graph.py

class ModeAwareTaskGraphBuilder:
    def __init__(self, task_manager, mode_config):
        self.mode = mode_config.mode
        
    def build_similarity_matrix(self):
        if self.mode == TrainingMode.HLA_ONLY:
            # 基于HLA序列相似性
            return self._hla_similarity()
            
        elif self.mode == TrainingMode.HLA_TISSUE:
            # HLA相似性 + tissue共现
            hla_sim = self._hla_similarity()
            tissue_sim = self._tissue_cooccurrence()
            return 0.7 * hla_sim + 0.3 * tissue_sim
```

### Step 4: 更新Model ⏳

```python
# src/models/full_model.py

class ImmuneAppModel(nn.Module):
    def __init__(self, mode_config, ...):
        super().__init__()
        self.mode = mode_config.mode
        
        # 共享组件
        self.peptide_encoder = PeptideEncoder(...)
        self.task_gnn = TaskGNN(...)
        
        # Mode 2专用: Tissue embedder
        if self.mode == TrainingMode.HLA_TISSUE:
            self.tissue_embedder = nn.Embedding(n_tissues, tissue_dim)
            self.tissue_attention = nn.MultiheadAttention(...)
        
        self.predictor = TaskConditionedPredictor(...)
    
    def forward(self, batch, graph_data):
        # Encode peptide
        peptide_emb = self.peptide_encoder(batch['peptide'], ...)
        
        # Get task embedding
        if self.mode == TrainingMode.HLA_ONLY:
            task_emb = self.task_gnn(batch['task_idx'], graph_data)
            
        elif self.mode == TrainingMode.HLA_TISSUE:
            # Task GNN + Tissue信息
            hla_emb = self.task_gnn(batch['task_idx'], graph_data)
            tissue_emb = self.tissue_embedder(batch['tissue_idx'])
            
            # Combine (例如: FiLM)
            task_emb = self._combine_hla_tissue(hla_emb, tissue_emb)
        
        # Predict
        logits = self.predictor(peptide_emb, task_emb)
        return logits
```

### Step 5: 更新Trainer ⏳

```python
# src/training/trainer.py

def train_model(task_manager, task_datasets, mode_config, **train_args):
    """统一训练入口"""
    
    # 1. 创建model
    model = ImmuneAppModel(mode_config, ...)
    
    # 2. 创建dataset
    train_dataset = ModeAwareDataset(
        task_datasets, 
        task_manager, 
        mode_config
    )
    
    # 3. 创建task graph
    graph_builder = ModeAwareTaskGraphBuilder(task_manager, mode_config)
    graph = graph_builder.build_and_save()
    
    # 4. 训练
    if mode_config.mode == TrainingMode.HLA_ONLY:
        trainer = StandardTrainer(model, train_dataset, ...)
        
    elif mode_config.mode == TrainingMode.HLA_TISSUE:
        trainer = TissueAwareTrainer(model, train_dataset, ...)
    
    history = trainer.train(n_epochs=train_args['n_epochs'])
    
    return model, history
```

### Step 6: 创建端到端脚本 ⏳

```python
# scripts/train_mode1.py

from src.config.mode_config import create_mode1_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.negative_sampler import NegativeSampler
from src.training.trainer import train_model
import pandas as pd

def main():
    # 1. 加载数据
    train_df = pd.read_csv('data/processed/train.csv')
    
    # 2. 创建Mode 1配置
    config = create_mode1_config(
        min_samples=20,
        negative_ratio=20
    )
    
    # 3. 创建任务
    creator = UnifiedTaskCreator(config)
    task_manager = creator.create_tasks(
        train_df,
        save_dir='data/mode1_tasks'
    )
    
    # 4. 生成负样本
    sampler = NegativeSampler(config, train_df)
    task_datasets = sampler.generate_negatives_for_all_tasks(task_manager)
    
    # 5. 训练
    model, history = train_model(
        task_manager,
        task_datasets,
        config,
        n_epochs=50,
        batch_size=32,
        use_maml=True
    )
    
    # 6. 保存
    torch.save(model.state_dict(), 'models/mode1_best.pt')
    
    print("✓ Training completed!")

if __name__ == "__main__":
    main()
```

## 🎯 核心改进总结

### 1. 架构层面

| 方面 | Phase系统 | Mode系统 |
|------|----------|---------|
| 概念清晰度 | Phase暗示时间顺序 | Mode表示功能选择 |
| 代码复用 | Phase 1/2代码分离 | 统一框架,高复用 |
| 扩展性 | 添加Phase困难 | 添加Mode容易 |
| 灵活性 | 必须Phase 1→2 | 独立运行任意Mode |

### 2. 负样本生成

**Mode 2的关键创新**:
```python
# 负样本peptide不带tissue标签
negative_peptide = "TESTPEPTIDE"

# 但在task context下评估
task_context = {
    'hla': 'HLA-A*02:01',
    'tissue': 'Liver'
}

# 提问: "TESTPEPTIDE在Liver的HLA-A*02:01上能呈递吗?"
# 答案: 否 (negative sample)
```

### 3. Task定义

```python
# 统一的Task类
task = Task(
    task_id="HLA_A*02:01_Tissue_Liver",
    mode='hla_tissue',
    hla='HLA-A*02:01',
    tissue='Liver',
    n_samples=150,
    ...
)

# 统一的接口
task.task_context  # → {'hla': ..., 'tissue': ...}
task.is_mode1      # → False
task.is_mode2      # → True
```

## 📚 参考文档

- [MODE_SYSTEM.md](MODE_SYSTEM.md): Mode系统完整文档
- [src/config/mode_config.py](../src/config/mode_config.py): 配置系统
- [src/data/task_definition.py](../src/data/task_definition.py): Task定义
- [src/data/unified_task_creator.py](../src/data/unified_task_creator.py): 任务创建
- [src/data/negative_sampler.py](../src/data/negative_sampler.py): 负样本生成

## ❓ 常见问题

### Q1: 旧的Phase代码还能用吗?

可以,但建议逐步迁移。旧代码在`data/task_creator.py`中,标记为deprecated。

### Q2: Mode 2必须先跑Mode 1吗?

不需要!这是Mode系统的核心优势。可以直接训练Mode 2。

### Q3: Mode 3什么时候实现?

接口已预留。具体实现取决于Mode 1和Mode 2的性能对比。

### Q4: 负样本生成的tissue-aware策略合理吗?

合理。Mode 2的负样本在特定tissue context下评估,不需要显式tissue标签。这符合生物学实际:peptide能否被呈递取决于HLA×Tissue的综合环境。

### Q5: 如何选择Mode?

- 建立baseline → Mode 1
- Tissue-specific prediction → Mode 2  
- 需要两者结合 → Mode 3 (未来)

---

**更新日期**: 2025-01-16  
**作者**: xty116
