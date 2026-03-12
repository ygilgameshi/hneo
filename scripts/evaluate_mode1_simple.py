"""
Mode 1 Per-Tissue评估脚本 - 完全基于实际代码

用法：
python scripts/evaluate_mode1_tissue_final.py \
    --mode1_output_dir output/task_balanced_HLA_v3_10_256 \
    --mode2_test_data output/task_balanced_v2/data_splits/test.tsv \
    --tissue_col Host \
    --output_dir output/mode1_per_tissue_eval
"""

import pandas as pd
import torch
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

from src.config.mode_config import create_mode1_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.enhanced_negative_sampler import EnhancedNegativeSampler
from src.graph.task_graph import TaskGraphWrapper
from src.data.dataset import ModeAwareDataset, collate_fn_mode_aware
from torch.utils.data import DataLoader
from src.models.full_model import ImmuneAppModel


def evaluate_per_tissue(model, test_loader, task_manager, test_task_datasets,
                        tissue_col, graph_wrapper, device):
    """评估per-tissue性能"""
    model.eval()

    # 准备graph data
    graph_data = {
        'edge_index': graph_wrapper.edge_index.to(device),
        'edge_weight': graph_wrapper.edge_weight.to(device)
    }

    # ✅ 创建task_idx到task_id的映射
    all_tasks = task_manager.get_all_tasks()
    task_idx_to_id = {idx: task_id for idx, (task_id, task) in enumerate(all_tasks.items())}

    # 收集所有预测
    all_predictions = {}

    with torch.no_grad():
        for batch in test_loader:
            # 移动到设备
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # 模型预测
            logits = model(batch, graph_data)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = batch['label'].cpu().numpy()
            task_idxs = batch['task_idx'].cpu().numpy()

            # ✅ 按task_id（而不是task_idx）组织预测结果
            for i in range(len(labels)):
                task_idx = task_idxs[i]
                # 将task_idx转换为task_id
                task_id = task_idx_to_id.get(task_idx)
                if task_id is None:
                    continue

                if task_id not in all_predictions:
                    all_predictions[task_id] = {
                        'labels': [],
                        'probs': []
                    }
                all_predictions[task_id]['labels'].append(labels[i])
                all_predictions[task_id]['probs'].append(probs[i])

    # 计算per-tissue性能
    per_tissue_records = []

    # ✅ 调试：检查第一个task的数据
    if len(test_task_datasets) > 0:
        first_task_id = list(test_task_datasets.keys())[0]
        first_df = test_task_datasets[first_task_id]
        print(f"\n  [DEBUG] First task_id: {first_task_id}")
        print(f"  [DEBUG] First task_id in all_predictions: {first_task_id in all_predictions}")
        print(f"  [DEBUG] Number of tasks with predictions: {len(all_predictions)}")
        print(f"  [DEBUG] Tissue column '{tissue_col}' exists: {tissue_col in first_df.columns}")
        if tissue_col in first_df.columns:
            pos_df = first_df[first_df['label'] == 1]
            print(f"  [DEBUG] Positive samples: {len(pos_df)}")
            print(f"  [DEBUG] Non-null tissues: {pos_df[tissue_col].notna().sum()}")
            print(f"  [DEBUG] Unique tissues: {pos_df[tissue_col].dropna().unique()[:5]}")

    for task_id, task_df in test_task_datasets.items():

        # 获取该task的预测
        if task_id not in all_predictions:
            continue

        # ✅ 重新获取task对象
        task = all_tasks[task_id]

        y_true = np.array(all_predictions[task_id]['labels'])
        y_prob = np.array(all_predictions[task_id]['probs'])
        y_pred = (y_prob > 0.5).astype(int)

        # 只看正样本（有tissue信息）
        pos_samples = task_df[task_df['label'] == 1]

        if len(pos_samples) > 0 and tissue_col in pos_samples.columns:
            # 按tissue分组
            for tissue in pos_samples[tissue_col].dropna().unique():
                tissue_samples = pos_samples[pos_samples[tissue_col] == tissue]
                n_pos = len(tissue_samples)

                # 计算该task的性能
                n_pos_pred = np.sum(y_true)
                n_neg_pred = len(y_true) - n_pos_pred

                metrics = {
                    'tissue': tissue,
                    'hla': task.hla,
                    'n_positive': n_pos,
                    'task_n_samples': len(y_true),
                    'task_n_positive': n_pos_pred
                }

                # 计算AUROC/AUPRC
                if n_pos_pred > 0 and n_neg_pred > 0:
                    try:
                        metrics['auroc'] = roc_auc_score(y_true, y_prob)
                        metrics['auprc'] = average_precision_score(y_true, y_prob)
                    except:
                        metrics['auroc'] = np.nan
                        metrics['auprc'] = np.nan
                else:
                    metrics['auroc'] = np.nan
                    metrics['auprc'] = np.nan

                # 计算F1
                try:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='binary', zero_division=0
                    )
                    metrics['precision'] = precision
                    metrics['recall'] = recall
                    metrics['f1'] = f1
                except:
                    metrics['precision'] = np.nan
                    metrics['recall'] = np.nan
                    metrics['f1'] = np.nan

                per_tissue_records.append(metrics)

    # 转换为DataFrame并按tissue聚合
    results_df = pd.DataFrame(per_tissue_records)

    if len(results_df) > 0:
        tissue_summary = results_df.groupby('tissue').apply(
            lambda x: pd.Series({
                'n_positive': x['n_positive'].sum(),
                'auroc': np.average(x['auroc'].dropna(),
                                   weights=x.loc[x['auroc'].notna(), 'n_positive'])
                         if x['auroc'].notna().any() else np.nan,
                'f1': np.average(x['f1'].dropna(),
                                weights=x.loc[x['f1'].notna(), 'n_positive'])
                      if x['f1'].notna().any() else np.nan,
                'auprc': np.average(x['auprc'].dropna(),
                                   weights=x.loc[x['auprc'].notna(), 'n_positive'])
                         if x['auprc'].notna().any() else np.nan
            })
        ).reset_index()

        tissue_summary = tissue_summary.sort_values('auroc', ascending=False, na_position='last')
        return tissue_summary, results_df, all_predictions  # ← 返回all_predictions
    else:
        return pd.DataFrame(), pd.DataFrame(), all_predictions  # ← 返回all_predictions


def print_detailed_statistics(all_predictions, test_task_datasets, task_manager):
    """
    打印详细统计信息（新增的四个部分）

    Args:
        all_predictions: {task_id: {'labels': [], 'probs': []}}
        test_task_datasets: {task_id: DataFrame}
        task_manager: TaskManager实例
    """

    # 计算per-task指标
    task_metrics = []
    all_tasks = task_manager.get_all_tasks()

    for task_id in all_predictions.keys():
        if task_id not in test_task_datasets:
            continue

        y_true = np.array(all_predictions[task_id]['labels'])
        y_prob = np.array(all_predictions[task_id]['probs'])

        task = all_tasks[task_id]
        n_samples = len(y_true)
        n_positive = np.sum(y_true)
        n_negative = n_samples - n_positive

        # 计算AUROC
        auroc = np.nan
        auprc = np.nan
        if n_positive > 0 and n_negative > 0:
            try:
                auroc = roc_auc_score(y_true, y_prob)
                auprc = average_precision_score(y_true, y_prob)
            except:
                pass

        task_metrics.append({
            'task_id': task_id,
            'hla': task.hla,
            'n_samples': n_samples,
            'n_positive': n_positive,
            'auroc': auroc,
            'auprc': auprc
        })

    df = pd.DataFrame(task_metrics)

    # ========== 1. 📊 总体统计 ==========
    print("\n" + "="*80)
    print("📊 OVERALL STATISTICS")
    print("="*80)

    valid_aurocs = df['auroc'].dropna()
    valid_auprcs = df['auprc'].dropna()

    if len(valid_aurocs) > 0:
        print(f"\n  Per-Task Statistics:")
        print(f"    Total tasks:      {len(df)}")
        print(f"    Valid tasks:      {len(valid_aurocs)}")
        print(f"    Mean AUROC:       {valid_aurocs.mean():.4f} ± {valid_aurocs.std():.4f}")
        print(f"    Median AUROC:     {valid_aurocs.median():.4f}")
        print(f"    Min AUROC:        {valid_aurocs.min():.4f}")
        print(f"    Max AUROC:        {valid_aurocs.max():.4f}")
        print(f"    Range:            {valid_aurocs.max() - valid_aurocs.min():.4f}")

        if len(valid_auprcs) > 0:
            print(f"    Mean AUPRC:       {valid_auprcs.mean():.4f} ± {valid_auprcs.std():.4f}")

    # ========== 2. 📈 按样本数分层 ==========
    print("\n" + "="*80)
    print("📈 PERFORMANCE BY SAMPLE SIZE")
    print("="*80)

    # 定义bins
    bins = [0, 50, 100, 500, 1000, float('inf')]
    labels = ['<50', '50-100', '100-500', '500-1K', '>1K']
    df['sample_bin'] = pd.cut(df['n_samples'], bins=bins, labels=labels)

    print(f"\n  {'Size Range':<15} {'N Tasks':<10} {'Mean AUROC':<12} {'Mean AUPRC':<12}")
    print("  " + "-" * 50)

    for bin_label in labels:
        bin_data = df[df['sample_bin'] == bin_label]
        if len(bin_data) > 0:
            valid_bin_aurocs = bin_data['auroc'].dropna()
            valid_bin_auprcs = bin_data['auprc'].dropna()

            auroc_str = f"{valid_bin_aurocs.mean():.4f}" if len(valid_bin_aurocs) > 0 else "N/A"
            auprc_str = f"{valid_bin_auprcs.mean():.4f}" if len(valid_bin_auprcs) > 0 else "N/A"

            print(f"  {bin_label:<15} {len(bin_data):<10} {auroc_str:<12} {auprc_str:<12}")

    # ========== 3. 🏆 Top 20 Tasks ==========
    print("\n" + "="*80)
    print("🏆 TOP 20 TASKS BY AUROC")
    print("="*80)

    top_20 = df.sort_values('auroc', ascending=False, na_position='last').head(20)

    print(f"\n  {'Rank':<6} {'Task ID':<30} {'HLA':<15} {'AUROC':<10} {'N Samples':<10}")
    print("  " + "-" * 75)

    for rank, (_, row) in enumerate(top_20.iterrows(), 1):
        auroc_str = f"{row['auroc']:.4f}" if not np.isnan(row['auroc']) else "N/A"
        print(f"  {rank:<6} {row['task_id']:<30} {row['hla']:<15} "
              f"{auroc_str:<10} {row['n_samples']:<10}")

    # ========== 4. ⚠️ Bottom 20 Tasks ==========
    print("\n" + "="*80)
    print("⚠️  BOTTOM 20 TASKS BY AUROC")
    print("="*80)

    # 只取有效AUROC
    df_valid = df[df['auroc'].notna()].copy()

    if len(df_valid) >= 20:
        bottom_20 = df_valid.sort_values('auroc', ascending=True).head(20)

        print(f"\n  {'Rank':<6} {'Task ID':<30} {'HLA':<15} {'AUROC':<10} {'N Samples':<10}")
        print("  " + "-" * 75)

        for rank, (_, row) in enumerate(bottom_20.iterrows(), 1):
            print(f"  {rank:<6} {row['task_id']:<30} {row['hla']:<15} "
                  f"{row['auroc']:<10.4f} {row['n_samples']:<10}")
    else:
        print(f"\n  Not enough valid tasks (only {len(df_valid)} tasks with valid AUROC)")

    print("\n" + "="*80)


def visualize_results(all_predictions, test_task_datasets, task_manager, output_dir):
    """
    生成可视化结果并保存图片

    Args:
        all_predictions: {task_id: {'labels': [], 'probs': []}}
        test_task_datasets: {task_id: DataFrame}
        task_manager: TaskManager实例
        output_dir: 输出目录
    """

    # 计算per-task指标
    task_metrics = []
    all_tasks = task_manager.get_all_tasks()

    for task_id in all_predictions.keys():
        if task_id not in test_task_datasets:
            continue

        y_true = np.array(all_predictions[task_id]['labels'])
        y_prob = np.array(all_predictions[task_id]['probs'])

        task = all_tasks[task_id]
        n_samples = len(y_true)
        n_positive = np.sum(y_true)
        n_negative = n_samples - n_positive

        # 计算AUROC
        auroc = np.nan
        auprc = np.nan
        if n_positive > 0 and n_negative > 0:
            try:
                auroc = roc_auc_score(y_true, y_prob)
                auprc = average_precision_score(y_true, y_prob)
            except:
                pass

        task_metrics.append({
            'task_id': task_id,
            'hla': task.hla,
            'n_samples': n_samples,
            'n_positive': n_positive,
            'auroc': auroc,
            'auprc': auprc
        })

    df = pd.DataFrame(task_metrics)

    # 设置样式
    plt.style.use('default')
    sns.set_palette("husl")

    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 1 Evaluation Results', fontsize=16, fontweight='bold', y=0.995)

    # ========== 1. AUROC分布直方图 ==========
    ax = axes[0, 0]
    valid_aurocs = df['auroc'].dropna()

    if len(valid_aurocs) > 0:
        ax.hist(valid_aurocs, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(valid_aurocs.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_aurocs.mean():.3f}')
        ax.axvline(valid_aurocs.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {valid_aurocs.median():.3f}')
        ax.set_xlabel('AUROC', fontsize=12)
        ax.set_ylabel('Number of Tasks', fontsize=12)
        ax.set_title('AUROC Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # ========== 2. 按样本数分层的性能 ==========
    ax = axes[0, 1]

    bins = [0, 50, 100, 500, 1000, float('inf')]
    labels = ['<50', '50-100', '100-500', '500-1K', '>1K']
    df['sample_bin'] = pd.cut(df['n_samples'], bins=bins, labels=labels)

    bin_stats = []
    for bin_label in labels:
        bin_data = df[df['sample_bin'] == bin_label]
        if len(bin_data) > 0:
            valid_bin_aurocs = bin_data['auroc'].dropna()
            if len(valid_bin_aurocs) > 0:
                bin_stats.append({
                    'bin': bin_label,
                    'mean_auroc': valid_bin_aurocs.mean(),
                    'n_tasks': len(bin_data)
                })

    if bin_stats:
        bin_df = pd.DataFrame(bin_stats)
        bars = ax.bar(bin_df['bin'], bin_df['mean_auroc'], color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Sample Size Range', fontsize=12)
        ax.set_ylabel('Mean AUROC', fontsize=12)
        ax.set_title('Performance by Sample Size', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, n_tasks in zip(bars, bin_df['n_tasks']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}\n(n={n_tasks})',
                    ha='center', va='bottom', fontsize=9)

    # ========== 3. Top 15 HLAs性能 ==========
    ax = axes[1, 0]

    # 按HLA分组
    hla_stats = df.groupby('hla').agg({
        'auroc': lambda x: x.dropna().mean() if len(x.dropna()) > 0 else np.nan,
        'n_samples': 'sum'
    }).reset_index()

    hla_stats = hla_stats.dropna(subset=['auroc'])
    hla_stats = hla_stats.sort_values('auroc', ascending=True).tail(15)

    if len(hla_stats) > 0:
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(hla_stats)))
        bars = ax.barh(range(len(hla_stats)), hla_stats['auroc'], color=colors, alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(hla_stats)))
        ax.set_yticklabels(hla_stats['hla'], fontsize=9)
        ax.set_xlabel('AUROC', fontsize=12)
        ax.set_title('Top 15 HLAs by AUROC', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

        # 添加数值标签
        for i, (bar, n_samples) in enumerate(zip(bars, hla_stats['n_samples'])):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {width:.3f}',
                    ha='left', va='center', fontsize=8)

    # ========== 4. 样本数 vs AUROC 散点图 ==========
    ax = axes[1, 1]

    valid_df = df[df['auroc'].notna()].copy()

    if len(valid_df) > 0:
        scatter = ax.scatter(valid_df['n_samples'], valid_df['auroc'],
                           c=valid_df['auroc'], cmap='RdYlGn',
                           s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Number of Samples (log scale)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title('Sample Size vs AUROC', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('AUROC', fontsize=10)

        # 添加趋势线
        if len(valid_df) > 2:
            z = np.polyfit(np.log10(valid_df['n_samples']), valid_df['auroc'], 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(valid_df['n_samples'].min()),
                                 np.log10(valid_df['n_samples'].max()), 100)
            ax.plot(x_trend, p(np.log10(x_trend)), "r--", alpha=0.5, linewidth=2, label='Trend')
            ax.legend()

    plt.tight_layout()

    # 保存图片
    output_path = output_dir / 'evaluation_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode1_output_dir', type=str, required=True,
                        help='Mode 1 output directory')
    parser.add_argument('--mode2_test_data', type=str, required=True,
                        help='Path to Mode 2 test.tsv')
    parser.add_argument('--tissue_col', type=str, default='Host',
                        help='Tissue column name')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--negative_ratio', type=int, default=10,
                        help='Negative ratio')

    args = parser.parse_args()

    print("="*80)
    print("Mode 1 Per-Tissue Evaluation")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode1_dir = Path(args.mode1_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Device: {device}")
    print(f"  Mode 1 dir: {mode1_dir}")

    # 1. 加载测试数据
    print(f"\n📂 Loading test data...")
    test_df = pd.read_csv(args.mode2_test_data, sep='\t')

    if 'MHC_Restriction_Name' in test_df.columns:
        test_df = test_df.rename(columns={
            'MHC_Restriction_Name': 'hla',
            'Peptide': 'peptide',
            'Label': 'label'
        })

    print(f"✓ Loaded {len(test_df):,} samples")
    print(f"  HLAs in test set: {test_df['hla'].nunique()}")

    # ✅ 自动检测实际的tissue列名
    actual_tissue_col = None
    if args.tissue_col in test_df.columns:
        actual_tissue_col = args.tissue_col
        print(f"  ✓ Tissue column: '{actual_tissue_col}'")
    elif 'tissue' in test_df.columns:
        actual_tissue_col = 'tissue'
        print(f"  ⚠️  Tissue column '{args.tissue_col}' not found, using 'tissue' instead")
    elif 'Host' in test_df.columns:
        actual_tissue_col = 'Host'
        print(f"  ⚠️  Tissue column '{args.tissue_col}' not found, using 'Host' instead")

    if actual_tissue_col:
        print(f"  Tissues: {test_df[actual_tissue_col].nunique()}")
    else:
        print(f"  ⚠️  No tissue column found!")

    # 2. 创建配置（评估时不筛选tasks）
    print(f"\n⚙️  Creating config...")
    config = create_mode1_config(
        negative_ratio=args.negative_ratio,
        min_samples=1  # ✅ 评估时不筛选，保留所有tasks
    )

    # 3. 创建tasks
    print(f"\n📋 Creating tasks...")
    creator = UnifiedTaskCreator(config)
    task_manager = creator.create_tasks(test_df)
    n_tasks = len(task_manager.get_all_tasks())

    print(f"✓ Created {n_tasks} tasks")

    # 5. 加载task graph（提前加载以获取训练时的HLA）
    print(f"\n📐 Loading task graph...")
    task_graph_dir = mode1_dir / 'task_graph'

    if not task_graph_dir.exists():
        raise FileNotFoundError(f"Task graph not found: {task_graph_dir}")

    graph_wrapper = TaskGraphWrapper(task_graph_dir)
    graph_wrapper.to(device)

    print(f"✓ Loaded task graph: {graph_wrapper.n_tasks} nodes")

    # 从graph中获取训练时的HLA列表
    # 如果graph_wrapper没有task_names属性，从task_manager获取并过滤
    trained_hlas = set()
    try:
        trained_hlas = set(graph_wrapper.task_names)
        print(f"  Trained HLAs from graph: {len(trained_hlas)}")
    except AttributeError:
        # 如果没有task_names，假设所有当前的tasks都在训练中
        print(f"  ⚠️  Could not get HLA list from graph, using all current tasks")
        trained_hlas = set(task.hla for task in task_manager.get_all_tasks().values())

    # 过滤测试集，只保留训练时见过的HLA
    original_len = len(test_df)
    test_df = test_df[test_df['hla'].isin(trained_hlas)].copy()
    filtered_out = original_len - len(test_df)

    if filtered_out > 0:
        print(f"  ⚠️  Filtered out {filtered_out:,} samples from {test_df['hla'].nunique()} unseen HLAs")
        print(f"  ✓ Kept {len(test_df):,} samples from {test_df['hla'].nunique()} trained HLAs")

    # 4. 生成测试负样本（现在所有HLA都是训练时见过的）
    print(f"\n🔄 Generating test negatives...")
    test_sampler = EnhancedNegativeSampler(config, test_df)
    test_task_datasets = {}

    for task_id, task in task_manager.get_all_tasks().items():
        task_test_df = test_df[test_df['hla'] == task.hla].copy()

        if len(task_test_df) > 0:
            positive_peptides = list(task_test_df['peptide'])
            negative_peptides = test_sampler.generate_negatives_for_task(
                task, positive_peptides
            )

            neg_df = pd.DataFrame({
                'peptide': negative_peptides,
                'hla': task.hla,
                'label': 0
            })

            # ✅ 保留tissue列（负样本设为None）
            if actual_tissue_col and actual_tissue_col in task_test_df.columns:
                neg_df[actual_tissue_col] = None

            task_test_combined = pd.concat([task_test_df, neg_df], ignore_index=True)
            test_task_datasets[task_id] = task_test_combined

    print(f"✓ Generated negatives for {len(test_task_datasets)} tasks")

    # 6. 创建DataLoader（彻底过滤未知HLA）
    print(f"\n📦 Creating test DataLoader...")

    # ✅ 从graph_wrapper获取实际的task映射
    if hasattr(graph_wrapper, 'task_to_idx'):
        valid_task_ids = set(graph_wrapper.task_to_idx.keys())
        print(f"  Valid task IDs from graph: {len(valid_task_ids)}")
    else:
        # 如果没有task_to_idx，尝试从G获取
        valid_task_ids = set(f"HLA_{graph_wrapper.G.nodes[i]['hla']}"
                            for i in range(graph_wrapper.n_tasks))
        print(f"  Valid task IDs from graph nodes: {len(valid_task_ids)}")

    # 过滤掉graph中不存在的tasks
    filtered_test_datasets = {}
    skipped_tasks = []

    for task_id, task_df in test_task_datasets.items():
        task = task_manager.get_all_tasks()[task_id]
        task_key = f"HLA_{task.hla}"

        if task_key in valid_task_ids:
            filtered_test_datasets[task_id] = task_df
        else:
            skipped_tasks.append(task.hla)

    if skipped_tasks:
        print(f"  ⚠️  Skipped {len(skipped_tasks)} tasks not in graph:")
        for hla in skipped_tasks[:3]:
            print(f"    - {hla}")
        if len(skipped_tasks) > 3:
            print(f"    ... and {len(skipped_tasks)-3} more")

    print(f"  ✓ Using {len(filtered_test_datasets)} tasks")

    test_dataset = ModeAwareDataset(
        filtered_test_datasets,  # ✅ 使用过滤后的
        task_manager,
        config,
        graph_wrapper
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_mode_aware
    )

    # 7. 加载模型（使用训练时的task数量）
    print(f"\n🔧 Loading model...")
    checkpoint_path = mode1_dir / 'best_model.pt'

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ✅ 使用训练时的task数量（从graph_wrapper获取）
    n_tasks_in_model = graph_wrapper.n_tasks  # ← 这是训练时的task数量

    print(f"  Task count in checkpoint: {n_tasks_in_model}")
    print(f"  Task count in test set: {n_tasks}")

    # ✅ 使用训练时的task数量创建模型
    model_kwargs = {
        'mode_config': config,
        'n_tasks': n_tasks_in_model  # ← 使用训练时的数量
    }

    model = ImmuneAppModel(**model_kwargs).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded")

    # 打印模型参数
    params = model.get_parameters_count()
    print(f"\n  Model parameters:")
    for component, count in params.items():
        print(f"    {component}: {count:,}")

    # 8. 评估
    print(f"\n{'='*80}")
    print("Evaluating per-tissue performance...")
    print(f"{'='*80}")

    # ✅ 详细调试
    print(f"\n  [DEBUG] Number of filtered tasks: {len(filtered_test_datasets)}")
    print(f"  [DEBUG] Using tissue column: '{actual_tissue_col}'")

    if len(filtered_test_datasets) > 0:
        first_task_id = list(filtered_test_datasets.keys())[0]
        first_task = task_manager.get_all_tasks()[first_task_id]
        first_df = filtered_test_datasets[first_task_id]
        print(f"  [DEBUG] First task: {first_task.hla}")
        print(f"  [DEBUG] DataFrame shape: {first_df.shape}")
        print(f"  [DEBUG] Columns: {list(first_df.columns)}")

        if actual_tissue_col and actual_tissue_col in first_df.columns:
            pos_df = first_df[first_df['label'] == 1]
            print(f"  [DEBUG] Positive samples: {len(pos_df)}")
            print(f"  [DEBUG] Tissue non-null: {pos_df[actual_tissue_col].notna().sum()}")
            print(f"  [DEBUG] Sample tissues: {pos_df[actual_tissue_col].dropna().unique()[:3]}")
        else:
            print(f"  [DEBUG] ❌ Tissue column '{actual_tissue_col}' NOT FOUND!")

    tissue_summary, detailed_results, all_predictions = evaluate_per_tissue(
        model, test_loader, task_manager, filtered_test_datasets,
        actual_tissue_col if actual_tissue_col else args.tissue_col,  # ✅ 使用实际列名
        graph_wrapper, device
    )

    # ========== 新增：打印详细统计 ==========
    print_detailed_statistics(all_predictions, filtered_test_datasets, task_manager)

    # ========== 新增：生成可视化 ==========
    visualize_results(all_predictions, filtered_test_datasets, task_manager, output_dir)

    # 9. 保存结果
    if len(tissue_summary) > 0:
        print(f"\n💾 Saving results...")

        tissue_file = output_dir / 'per_tissue_results.csv'
        tissue_summary.to_csv(tissue_file, index=False)
        print(f"✓ Saved: {tissue_file}")

        detailed_file = output_dir / 'detailed_tissue_results.csv'
        detailed_results.to_csv(detailed_file, index=False)
        print(f"✓ Saved: {detailed_file}")

        # 打印结果
        print(f"\n{'='*80}")
        print("Per-Tissue Performance Summary")
        print(f"{'='*80}")
        print(tissue_summary.to_string(index=False))
        print(f"{'='*80}")

        print(f"\n✓ Evaluation complete!")
        print(f"\n📁 Results: {output_dir}")
    else:
        print(f"\n⚠️  No results. Check tissue column '{args.tissue_col}'.")


if __name__ == '__main__':
    main()