"""
训练循环

包含：
- train_phase1: 完整训练流程
- train_standard_epoch: 标准训练一个epoch
- train_maml_epoch: MAML训练一个epoch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from ..models import ImmuneAppPhase1
from ..models.task_gnn import TaskGraphWrapper
from ..data.dataset import create_dataloaders
from .maml import MAMLTrainer, MAMLDataLoader
from .evaluator import Evaluator


# ============= 添加这个辅助函数 =============
def convert_to_serializable(obj):
    """
    递归转换对象为JSON可序列化的类型

    处理：
    - numpy数值类型 → Python原生类型
    - numpy数组 → Python列表
    - 字典和列表 → 递归处理
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}

    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]

    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                          np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)

    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, np.bool_):
        return bool(obj)

    else:
        return obj


# =========================================


def train_phase1(
        dataset_dir='data/phase1_dataset',
        graph_dir='data/phase1_dataset',
        output_dir='models/phase1_models',
        use_maml=True,
        n_epochs=50,
        batch_size=32,
        inner_lr=0.01,
        meta_lr=0.001,
        inner_steps=5,
        n_way=5,
        k_shot=10,
        q_query=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        resume_from=None
):
    """
    Phase 1完整训练流程

    Args:
        dataset_dir: 数据集目录
        graph_dir: Task Graph目录
        output_dir: 模型保存目录
        use_maml: 是否使用MAML（否则使用标准训练）
        n_epochs: 训练轮数
        batch_size: batch大小
        inner_lr: MAML内循环学习率
        meta_lr: 元学习率/标准学习率
        inner_steps: MAML内循环步数
        n_way, k_shot, q_query: MAML episode参数
        device: 设备
        resume_from: 从checkpoint恢复

    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    print("=" * 80)
    print(f"Phase 1 训练 {'(MAML)' if use_maml else '(Standard)'}")
    print("=" * 80)
    print(f"Device: {device}")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 1. 加载数据 ===
    print("\n" + "=" * 80)
    print("加载数据...")
    print("=" * 80)

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        meta_learning=False
    )

    # 加载Task Graph
    graph_wrapper = TaskGraphWrapper(
        graph_file=Path(graph_dir) / 'task_graph.pkl',
        similarity_matrix_file=Path(graph_dir) / 'similarity_matrix.npy'
    )
    graph_wrapper.to(device)

    # === 2. 创建模型 ===
    print("\n" + "=" * 80)
    print("创建模型...")
    print("=" * 80)

    model = ImmuneAppPhase1(
        n_tasks=graph_wrapper.n_tasks,
        vocab_size=21,
        peptide_output_dim=64,
        task_output_dim=64,
        predictor_hidden_dim=128,
        predictor_n_layers=3,
        dropout=0.1
    ).to(device)

    params_count = model.get_parameters_count()
    print(f"\nModel parameters:")
    for component, count in params_count.items():
        print(f"  {component}: {count:,}")

    # === 3. 创建训练器 ===
    print("\n" + "=" * 80)
    print("创建训练器...")
    print("=" * 80)

    start_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': [], 'val_auprc': []}
    best_val_auroc = 0.0

    if use_maml:
        trainer = MAMLTrainer(
            model=model,
            inner_lr=inner_lr,
            meta_lr=meta_lr,
            inner_steps=inner_steps,
            first_order=False
        )

        # 包装DataLoader为MAML格式
        maml_train_loader = MAMLDataLoader(
            train_loader,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query
        )

        optimizer = None
        criterion = None
    else:
        trainer = None
        maml_train_loader = None
        optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        criterion = nn.BCEWithLogitsLoss()

    # 创建评估器
    evaluator = Evaluator(criterion=nn.BCEWithLogitsLoss())

    # === 4. 从checkpoint恢复 ===
    if resume_from and Path(resume_from).exists():
        print(f"\n从checkpoint恢复: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)
        best_val_auroc = checkpoint.get('best_val_auroc', 0.0)

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"✓ 恢复成功！从epoch {start_epoch}继续训练")

    # === 5. 训练循环 ===
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)

    for epoch in range(start_epoch, n_epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"{'=' * 80}")

        # 训练
        if use_maml:
            train_metrics = train_maml_epoch(
                trainer, maml_train_loader, graph_wrapper, device
            )
        else:
            train_metrics = train_standard_epoch(
                model, train_loader, optimizer, criterion, graph_wrapper, device
            )

        # 验证
        val_metrics = evaluator.evaluate(
            model, val_loader, graph_wrapper, device
        )

        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_auprc'].append(val_metrics['auprc'])

        # 打印指标
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
        if use_maml:
            print(f"  Support Loss: {train_metrics.get('support_loss', 0):.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val AUROC: {val_metrics['auroc']:.4f}")
        print(f"Val AUPRC: {val_metrics['auprc']:.4f}")
        print(f"Val PPV@100: {val_metrics['ppv']:.4f}")

        # 保存最佳模型
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auroc': val_metrics['auroc'],
                'val_auprc': val_metrics['auprc'],
                'history': history,
            }, output_dir / 'best_model.pt')

            print(f"✓ 保存最佳模型 (AUROC: {best_val_auroc:.4f})")

        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'best_val_auroc': best_val_auroc,
                'history': history,
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')

            print(f"✓ 保存checkpoint: epoch_{epoch + 1}.pt")

    # === 6. 最终测试 ===
    print(f"\n{'=' * 80}")
    print("最终测试")
    print(f"{'=' * 80}")

    # 加载最佳模型
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluator.evaluate(model, test_loader, graph_wrapper, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  PPV@100: {test_metrics['ppv']:.4f}")

    # ============= 修改这里：保存结果前转换类型 =============
    # 过滤掉不能序列化的大型数组（ROC/PR曲线数据）
    test_metrics_filtered = {
        k: v for k, v in test_metrics.items()
        if k not in ['roc_curve', 'pr_curve', 'confusion_matrix']
    }

    # 转换为可序列化类型
    test_metrics_clean = convert_to_serializable(test_metrics_filtered)
    history_clean = convert_to_serializable(history)

    results = {
        'history': history_clean,
        'test_metrics': test_metrics_clean,
        'config': {
            'use_maml': use_maml,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'inner_lr': inner_lr if use_maml else None,
            'meta_lr': meta_lr,
            'inner_steps': inner_steps if use_maml else None,
        }
    }

    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    # ====================================================

    print(f"\n✓ 训练完成！结果保存在: {output_dir}")

    return model, history

def train_standard_epoch(model, dataloader, optimizer, criterion, graph_wrapper, device):
    """
    标准训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        graph_wrapper: Task Graph包装器
        device: 设备

    Returns:
        metrics: dict with 'loss'
    """
    model.train()
    epoch_losses = []

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        # 移动数据到device
        peptide = batch['peptide'].to(device)
        peptide_len = batch['peptide_len'].to(device)
        task_idx = batch['task_idx'].to(device)
        label = batch['label'].float().to(device)

        # 前向传播
        logits = model(
            peptide, peptide_len, task_idx,
            graph_wrapper.edge_index,
            graph_wrapper.edge_weight
        )

        # 计算loss
        loss = criterion(logits.squeeze(), label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 记录
        epoch_losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return {'loss': np.mean(epoch_losses)}


def train_maml_epoch(trainer, dataloader, graph_wrapper, device):
    """
    MAML训练一个epoch

    Args:
        trainer: MAMLTrainer
        dataloader: MAMLDataLoader
        graph_wrapper: Task Graph包装器
        device: 设备

    Returns:
        metrics: dict with 'loss', 'support_loss'
    """
    trainer.model.train()

    epoch_support_losses = []
    epoch_query_losses = []

    graph_data = {
        'edge_index': graph_wrapper.edge_index,
        'edge_weight': graph_wrapper.edge_weight
    }

    pbar = tqdm(dataloader, desc="MAML Training")

    for episode in pbar:
        # 移动数据到device
        support_data = {k: v.to(device) for k, v in episode['support'].items()}
        query_data = {k: v.to(device) for k, v in episode['query'].items()}

        episode_data = {'support': support_data, 'query': query_data}

        # Meta训练步骤
        metrics = trainer.meta_train_step(episode_data, graph_data)

        epoch_support_losses.append(metrics['support_loss'])
        epoch_query_losses.append(metrics['query_loss'])

        pbar.set_postfix({
            'query_loss': f'{metrics["query_loss"]:.4f}',
            'support_loss': f'{metrics["support_loss"]:.4f}'
        })

    return {
        'loss': np.mean(epoch_query_losses),
        'support_loss': np.mean(epoch_support_losses),
    }


# 主程序入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Phase 1 Model')
    parser.add_argument('--dataset_dir', type=str, default='data/phase1_dataset')
    parser.add_argument('--output_dir', type=str, default='models/phase1_models')
    parser.add_argument('--use_maml', action='store_true', help='Use MAML training')
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--meta_lr', type=float, default=0.001)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--resume_from', type=str, default=None)

    args = parser.parse_args()

    model, history = train_phase1(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        use_maml=args.use_maml,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        resume_from=args.resume_from
    )