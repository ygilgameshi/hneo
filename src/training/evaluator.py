"""
评估工具

用于评估模型性能
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)


class Evaluator:
    """
    模型评估器

    计算各种评估指标：
    - Loss
    - AUROC (Area Under ROC Curve)
    - AUPRC (Area Under Precision-Recall Curve)
    - PPV@k (Positive Predictive Value at top k)

    Args:
        criterion: 损失函数

    Example:
        >>> evaluator = Evaluator()
        >>> metrics = evaluator.evaluate(model, dataloader, graph_wrapper, device)
        >>> print(metrics['auroc'])
    """

    def __init__(self, criterion=None):
        self.criterion = criterion or nn.BCEWithLogitsLoss()

    def evaluate(self, model, dataloader, graph_wrapper, device):
        """
        完整评估

        Args:
            model: 要评估的模型
            dataloader: 数据加载器
            graph_wrapper: Task Graph包装器
            device: 设备

        Returns:
            metrics: dict包含所有评估指标
        """
        model.eval()

        all_labels = []
        all_probs = []
        all_losses = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
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
                loss = self.criterion(logits.squeeze(), label)
                all_losses.append(loss.item())

                # 预测概率
                probs = torch.sigmoid(logits).squeeze()

                # 收集结果
                all_labels.extend(label.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 转为numpy数组
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算指标
        metrics = self.compute_metrics(all_labels, all_probs, all_losses)

        return metrics

    def compute_metrics(self, labels, probs, losses):
        """
        计算所有评估指标

        Args:
            labels: 真实标签 (numpy array)
            probs: 预测概率 (numpy array)
            losses: 损失列表

        Returns:
            metrics: dict
        """
        metrics = {}

        # Loss
        metrics['loss'] = np.mean(losses)

        # AUROC
        if len(np.unique(labels)) > 1:
            metrics['auroc'] = roc_auc_score(labels, probs)
        else:
            metrics['auroc'] = 0.0

        # AUPRC
        metrics['auprc'] = average_precision_score(labels, probs)

        # PPV@k (k=100)
        metrics['ppv'] = self.compute_ppv_at_k(labels, probs, k=100)

        # 额外指标
        fpr, tpr, thresholds = roc_curve(labels, probs)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['roc_thresholds'] = thresholds

        precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['pr_thresholds'] = pr_thresholds

        return metrics

    @staticmethod
    def compute_ppv_at_k(labels, probs, k=100):
        """
        计算PPV@k (Positive Predictive Value at top k)

        PPV@k = (top k中真阳性的数量) / k

        Args:
            labels: 真实标签
            probs: 预测概率
            k: top-k

        Returns:
            ppv: float
        """
        if len(labels) < k:
            k = len(labels)

        # 获取top-k的索引
        top_k_indices = np.argsort(probs)[-k:]

        # 计算PPV
        ppv = labels[top_k_indices].mean()

        return ppv

    def evaluate_by_task(self, model, dataloader, graph_wrapper, device):
        """
        按任务分层评估

        Returns:
            task_metrics: dict, key是task_idx，value是该task的metrics
        """
        model.eval()

        # 按task收集结果
        task_results = {}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating by task", leave=False):
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

                probs = torch.sigmoid(logits).squeeze()

                # 按task分组
                for i in range(len(task_idx)):
                    tid = task_idx[i].item()

                    if tid not in task_results:
                        task_results[tid] = {'labels': [], 'probs': []}

                    task_results[tid]['labels'].append(label[i].item())
                    task_results[tid]['probs'].append(probs[i].item())

        # 计算每个task的指标
        task_metrics = {}

        for tid, results in task_results.items():
            labels = np.array(results['labels'])
            probs = np.array(results['probs'])

            if len(np.unique(labels)) > 1:
                auroc = roc_auc_score(labels, probs)
            else:
                auroc = 0.0

            auprc = average_precision_score(labels, probs)

            task_metrics[tid] = {
                'n_samples': len(labels),
                'n_positive': labels.sum(),
                'auroc': auroc,
                'auprc': auprc,
            }

        return task_metrics


# 测试代码
if __name__ == "__main__":
    print("Testing Evaluator...")

    # 模拟数据
    labels = np.random.randint(0, 2, 1000)
    probs = np.random.rand(1000)
    losses = np.random.rand(100)

    evaluator = Evaluator()
    metrics = evaluator.compute_metrics(labels, probs, losses)

    print(f"✓ Metrics computed:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")

    print("\n✓ Evaluator test passed!")