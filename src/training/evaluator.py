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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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


"""
Phase 2评估器 - 完全适配你的代码

"""




def calculate_ppv_at_k(labels, scores, k_percent=2):
    """
    计算PPV@k% (Positive Predictive Value at top k%)

    这是ImmuneApp和MHLAPre论文中的关键指标！

    Args:
        labels: 真实标签 (numpy array)
        scores: 预测分数 (numpy array)
        k_percent: top k% (默认2%)

    Returns:
        ppv: PPV@k%值
    """
    n_total = len(labels)
    k = max(1, int(n_total * k_percent / 100))

    # 获取top-k的索引
    top_k_indices = np.argsort(scores)[-k:]

    # 计算PPV
    ppv = labels[top_k_indices].mean()

    return ppv


def evaluate_phase2(model, test_dataset, graph_wrapper, device,
                    stratify_by_tissue=True, output_dir=None):
    """
    Phase 2完整评估：多维度分析

    完全适配你的代码！直接使用你的ImmuneAppDatasetWithTissue

    Args:
        model: 训练好的模型
        test_dataset: ImmuneAppDatasetWithTissue实例
        graph_wrapper: TaskGraphWrapper实例
        device: torch device
        stratify_by_tissue: 是否按组织分层评估
        output_dir: 输出目录（保存可视化结果）

    Returns:
        results: 评估结果字典
    """
    model.eval()

    # 获取测试数据DataFrame
    test_df = test_dataset.get_dataframe()

    # 氨基酸编码（与dataset.py保持一致）
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    aa_to_idx['X'] = 20  # padding token
    max_len = 15

    def encode_peptide(peptide):
        """编码肽段"""
        indices = [aa_to_idx.get(aa, aa_to_idx['X']) for aa in peptide]
        if len(indices) < max_len:
            indices += [aa_to_idx['X']] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return torch.LongTensor(indices)

    # ========== 收集所有预测结果 ==========
    print("\n" + "=" * 60)
    print("Phase 2评估：预测测试集")
    print("=" * 60)

    y_true_all = []
    y_pred_all = []
    hla_list = []
    tissue_list = []

    # 创建HLA到task_idx的映射
    hla_to_task_idx = {}
    for idx, task_name in enumerate(graph_wrapper.task_names):
        hla_to_task_idx[task_name] = idx

    with torch.no_grad():
        for idx in tqdm(range(len(test_df)), desc="预测中"):
            row = test_df.iloc[idx]

            # 编码peptide
            peptide = encode_peptide(row['peptide']).unsqueeze(0).to(device)
            peptide_len = torch.LongTensor([len(row['peptide'])]).to(device)

            # 获取task_idx
            hla = row['hla']
            task_idx_value = hla_to_task_idx.get(hla, 0)  # 默认值0
            task_idx = torch.LongTensor([task_idx_value]).to(device)

            # 模型预测
            logits = model(
                peptide,
                peptide_len,
                task_idx,
                graph_wrapper.edge_index,
                graph_wrapper.edge_weight
            )

            # 转换为概率
            prob = torch.sigmoid(logits).item()

            # 收集结果
            y_true_all.append(int(row['label']))
            y_pred_all.append(prob)
            hla_list.append(hla)
            tissue_list.append(row['tissue'])

    # 转换为numpy数组
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # ========== 1. 总体评估 ==========
    print("\n" + "=" * 60)
    print("1. 总体评估")
    print("=" * 60)

    overall_auroc = roc_auc_score(y_true_all, y_pred_all)
    overall_auprc = average_precision_score(y_true_all, y_pred_all)
    overall_ppv2 = calculate_ppv_at_k(y_true_all, y_pred_all, k_percent=2)
    overall_ppv5 = calculate_ppv_at_k(y_true_all, y_pred_all, k_percent=5)

    print(f"AUROC: {overall_auroc:.4f}")
    print(f"AUPRC: {overall_auprc:.4f}")
    print(f"PPV@2%: {overall_ppv2:.4f}")
    print(f"PPV@5%: {overall_ppv5:.4f}")

    results = {
        'overall': {
            'auroc': overall_auroc,
            'auprc': overall_auprc,
            'ppv_2': overall_ppv2,
            'ppv_5': overall_ppv5,
            'n_samples': len(y_true_all),
            'n_positive': int(y_true_all.sum())
        }
    }

    # ========== 2. 按HLA分层评估 ==========
    print("\n" + "=" * 60)
    print("2. 按HLA分层评估")
    print("=" * 60)

    unique_hlas = list(set(hla_list))
    hla_results = {}

    for hla in unique_hlas:
        mask = np.array([h == hla for h in hla_list])
        if mask.sum() < 10:  # 样本太少，跳过
            continue

        y_true_hla = y_true_all[mask]
        y_pred_hla = y_pred_all[mask]

        if len(np.unique(y_true_hla)) < 2:  # 只有一个类别
            continue

        auroc = roc_auc_score(y_true_hla, y_pred_hla)
        auprc = average_precision_score(y_true_hla, y_pred_hla)
        ppv2 = calculate_ppv_at_k(y_true_hla, y_pred_hla, k_percent=2)

        hla_results[hla] = {
            'auroc': auroc,
            'auprc': auprc,
            'ppv_2': ppv2,
            'n_samples': int(mask.sum()),
            'n_positive': int(y_true_hla.sum())
        }

        print(f"{hla:15s}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, "
              f"PPV@2%={ppv2:.4f}, n={mask.sum()}")

    results['by_hla'] = hla_results

    # ========== 3. 按Tissue分层评估（如果启用）==========
    if stratify_by_tissue:
        print("\n" + "=" * 60)
        print("3. 按Tissue分层评估")
        print("=" * 60)

        unique_tissues = list(set(tissue_list))
        tissue_results = {}

        for tissue in unique_tissues:
            mask = np.array([t == tissue for t in tissue_list])
            if mask.sum() < 10:
                continue

            y_true_tissue = y_true_all[mask]
            y_pred_tissue = y_pred_all[mask]

            if len(np.unique(y_true_tissue)) < 2:
                continue

            auroc = roc_auc_score(y_true_tissue, y_pred_tissue)
            auprc = average_precision_score(y_true_tissue, y_pred_tissue)
            ppv2 = calculate_ppv_at_k(y_true_tissue, y_pred_tissue, k_percent=2)

            tissue_results[tissue] = {
                'auroc': auroc,
                'auprc': auprc,
                'ppv_2': ppv2,
                'n_samples': int(mask.sum()),
                'n_positive': int(y_true_tissue.sum())
            }

            print(f"{tissue:15s}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, "
                  f"PPV@2%={ppv2:.4f}, n={mask.sum()}")

        results['by_tissue'] = tissue_results

        # 计算tissue间AUROC方差（验证tissue signal）
        if len(tissue_results) > 1:
            tissue_aurocs = [r['auroc'] for r in tissue_results.values()]
            tissue_variance = np.var(tissue_aurocs)
            print(f"\nTissue AUROC方差: {tissue_variance:.4f}")
            if tissue_variance > 0.05:
                print("✓ Tissue信号显著（方差>0.05）")
            else:
                print("⚠ Tissue信号较弱（方差<=0.05）")

    # ========== 4. 按HLA×Tissue组合评估 ==========
    print("\n" + "=" * 60)
    print("4. 按HLA×Tissue组合评估")
    print("=" * 60)

    combination_results = {}

    for hla in unique_hlas:
        for tissue in unique_tissues:
            mask = np.array([(h == hla and t == tissue)
                             for h, t in zip(hla_list, tissue_list)])

            if mask.sum() < 5:  # 样本太少
                continue

            y_true_comb = y_true_all[mask]
            y_pred_comb = y_pred_all[mask]

            if len(np.unique(y_true_comb)) < 2:
                continue

            auroc = roc_auc_score(y_true_comb, y_pred_comb)
            auprc = average_precision_score(y_true_comb, y_pred_comb)

            combination_results[f"{hla}×{tissue}"] = {
                'auroc': auroc,
                'auprc': auprc,
                'n_samples': int(mask.sum()),
                'n_positive': int(y_true_comb.sum())
            }

    print(f"找到 {len(combination_results)} 个有效的HLA×Tissue组合")

    # 显示前10个性能最好的组合
    sorted_combs = sorted(combination_results.items(),
                          key=lambda x: x[1]['auroc'], reverse=True)
    print("\nTop 10 组合（按AUROC排序）:")
    for comb_name, metrics in sorted_combs[:10]:
        print(f"{comb_name:30s}: AUROC={metrics['auroc']:.4f}, "
              f"AUPRC={metrics['auprc']:.4f}, n={metrics['n_samples']}")

    results['by_combination'] = combination_results

    # ========== 5. 可视化（如果指定输出目录）==========
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("生成可视化结果...")
        print("=" * 60)

        _visualize_phase2_results(results, output_dir)

        print(f"✓ 可视化结果已保存到: {output_dir}")

    return results


def _visualize_phase2_results(results, output_dir):
    """生成Phase 2评估可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Phase 2 Evaluation Results', fontsize=16, fontweight='bold')

    # ========== 1. 总体指标柱状图 ==========
    ax = axes[0, 0]
    metrics = results['overall']
    metric_names = ['AUROC', 'AUPRC', 'PPV@2%', 'PPV@5%']
    metric_values = [metrics['auroc'], metrics['auprc'],
                     metrics['ppv_2'], metrics['ppv_5']]

    bars = ax.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Performance', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    # ========== 2. 按HLA的AUROC分布 ==========
    ax = axes[0, 1]
    if 'by_hla' in results and len(results['by_hla']) > 0:
        hla_aurocs = [r['auroc'] for r in results['by_hla'].values()]
        ax.hist(hla_aurocs, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(results['overall']['auroc'], color='red', linestyle='--',
                   linewidth=2, label=f"Overall: {results['overall']['auroc']:.3f}")
        ax.set_xlabel('AUROC', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('AUROC Distribution by HLA', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # ========== 3. 按Tissue的AUROC比较 ==========
    ax = axes[1, 0]
    if 'by_tissue' in results and len(results['by_tissue']) > 0:
        tissues = list(results['by_tissue'].keys())
        aurocs = [results['by_tissue'][t]['auroc'] for t in tissues]

        # 按AUROC排序
        sorted_indices = np.argsort(aurocs)
        tissues = [tissues[i] for i in sorted_indices]
        aurocs = [aurocs[i] for i in sorted_indices]

        bars = ax.barh(tissues, aurocs, color='#2ecc71', alpha=0.7)
        ax.axvline(results['overall']['auroc'], color='red', linestyle='--',
                   linewidth=2, label=f"Overall: {results['overall']['auroc']:.3f}")
        ax.set_xlabel('AUROC', fontsize=12)
        ax.set_ylabel('Tissue', fontsize=12)
        ax.set_title('AUROC by Tissue', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

    # ========== 4. HLA×Tissue组合热图（Top 20）==========
    ax = axes[1, 1]
    if 'by_combination' in results and len(results['by_combination']) > 0:
        # 选择样本数最多的前20个组合
        sorted_combs = sorted(results['by_combination'].items(),
                              key=lambda x: x[1]['n_samples'], reverse=True)[:20]

        comb_names = [c[0] for c in sorted_combs]
        comb_aurocs = [c[1]['auroc'] for c in sorted_combs]

        # 创建热图数据
        colors = plt.cm.RdYlGn(np.array(comb_aurocs))

        ax.barh(range(len(comb_names)), comb_aurocs, color=colors, alpha=0.8)
        ax.set_yticks(range(len(comb_names)))
        ax.set_yticklabels([name.replace('×', '\n×\n') for name in comb_names],
                           fontsize=8)
        ax.set_xlabel('AUROC', fontsize=12)
        ax.set_title('Top 20 HLA×Tissue Combinations', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 保存: phase2_evaluation_results.png")




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