"""
Mode 2 (HLA×Tissue) 评估脚本

用法：
python scripts/evaluate_mode2_simple.py \
    --mode2_output_dir output/mode2_experiment \
    --test_data output/task_balanced_v2/data_splits/test.tsv \
    --tissue_col Host \
    --output_dir output/mode2_eval
"""

import pandas as pd
import torch
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.mode_config import create_mode2_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.enhanced_negative_sampler import EnhancedNegativeSampler
from src.graph.task_graph import TaskGraphWrapper
from src.data.dataset import ModeAwareDataset, collate_fn_mode_aware
from torch.utils.data import DataLoader
from src.models.full_model import ImmuneAppModel


def evaluate_per_task(model, test_loader, task_manager, test_task_datasets,
                      graph_wrapper, device):
    """评估per-task (HLA×Tissue) 性能，返回all_predictions"""
    model.eval()

    graph_data = {
        'edge_index': graph_wrapper.edge_index.to(device),
        'edge_weight': graph_wrapper.edge_weight.to(device)
    }

    all_tasks = task_manager.get_all_tasks()
    task_idx_to_id = {idx: task_id for idx, (task_id, task) in enumerate(all_tasks.items())}

    all_predictions = {}

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            logits = model(batch, graph_data)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = batch['label'].cpu().numpy()
            task_idxs = batch['task_idx'].cpu().numpy()

            for i in range(len(labels)):
                task_id = task_idx_to_id.get(task_idxs[i])
                if task_id is None:
                    continue
                if task_id not in all_predictions:
                    all_predictions[task_id] = {'labels': [], 'probs': []}
                all_predictions[task_id]['labels'].append(labels[i])
                all_predictions[task_id]['probs'].append(probs[i])

    # 计算per-task指标
    per_task_records = []
    for task_id, task_df in test_task_datasets.items():
        if task_id not in all_predictions:
            continue

        task = all_tasks[task_id]
        y_true = np.array(all_predictions[task_id]['labels'])
        y_prob = np.array(all_predictions[task_id]['probs'])
        y_pred = (y_prob > 0.5).astype(int)

        n_pos = int(np.sum(y_true))
        n_neg = len(y_true) - n_pos

        metrics = {
            'task_id':  task_id,
            'hla':      task.hla,
            'tissue':   task.tissue,
            'n_samples': len(y_true),
            'n_positive': n_pos,
        }

        if n_pos > 0 and n_neg > 0:
            try:
                metrics['auroc'] = roc_auc_score(y_true, y_prob)
                metrics['auprc'] = average_precision_score(y_true, y_prob)
            except:
                metrics['auroc'] = np.nan
                metrics['auprc'] = np.nan
        else:
            metrics['auroc'] = np.nan
            metrics['auprc'] = np.nan

        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall']    = recall
            metrics['f1']        = f1
        except:
            metrics['precision'] = np.nan
            metrics['recall']    = np.nan
            metrics['f1']        = np.nan

        per_task_records.append(metrics)

    results_df = pd.DataFrame(per_task_records)
    return results_df, all_predictions


def print_detailed_statistics(results_df):
    """打印四个统计部分（与mode1保持一致）"""

    # ========== 1. 📊 总体统计 ==========
    print("\n" + "="*80)
    print("📊 OVERALL STATISTICS")
    print("="*80)

    valid_aurocs = results_df['auroc'].dropna()
    valid_auprcs = results_df['auprc'].dropna()

    if len(valid_aurocs) > 0:
        print(f"\n  Per-Task (HLA×Tissue) Statistics:")
        print(f"    Total tasks:      {len(results_df)}")
        print(f"    Valid tasks:      {len(valid_aurocs)}")
        print(f"    Mean AUROC:       {valid_aurocs.mean():.4f} ± {valid_aurocs.std():.4f}")
        print(f"    Median AUROC:     {valid_aurocs.median():.4f}")
        print(f"    Min AUROC:        {valid_aurocs.min():.4f}")
        print(f"    Max AUROC:        {valid_aurocs.max():.4f}")
        print(f"    Range:            {valid_aurocs.max() - valid_aurocs.min():.4f}")
        if len(valid_auprcs) > 0:
            print(f"    Mean AUPRC:       {valid_auprcs.mean():.4f} ± {valid_auprcs.std():.4f}")

    # Mode 2特有：按tissue汇总
    print(f"\n  Per-Tissue Summary (weighted average AUROC):")
    print(f"  {'Tissue':<25} {'N Tasks':<10} {'Mean AUROC':<12} {'Mean AUPRC':<12}")
    print("  " + "-"*60)

    tissue_group = results_df[results_df['auroc'].notna()].groupby('tissue')
    tissue_summary = tissue_group.apply(
        lambda x: pd.Series({
            'n_tasks':    len(x),
            'mean_auroc': np.average(x['auroc'], weights=x['n_positive']),
            'mean_auprc': np.average(x['auprc'].fillna(0), weights=x['n_positive']),
        })
    ).reset_index().sort_values('mean_auroc', ascending=False)

    for _, row in tissue_summary.iterrows():
        print(f"  {row['tissue']:<25} {int(row['n_tasks']):<10} "
              f"{row['mean_auroc']:<12.4f} {row['mean_auprc']:<12.4f}")

    # ========== 2. 📈 按样本数分层 ==========
    print("\n" + "="*80)
    print("📈 PERFORMANCE BY SAMPLE SIZE")
    print("="*80)

    bins   = [0, 50, 100, 500, 1000, float('inf')]
    labels = ['<50', '50-100', '100-500', '500-1K', '>1K']
    results_df = results_df.copy()
    results_df['sample_bin'] = pd.cut(results_df['n_samples'], bins=bins, labels=labels)

    print(f"\n  {'Size Range':<15} {'N Tasks':<10} {'Mean AUROC':<12} {'Mean AUPRC':<12}")
    print("  " + "-"*50)

    for bin_label in labels:
        bin_data = results_df[results_df['sample_bin'] == bin_label]
        if len(bin_data) > 0:
            va = bin_data['auroc'].dropna()
            vp = bin_data['auprc'].dropna()
            auroc_str = f"{va.mean():.4f}" if len(va) > 0 else "N/A"
            auprc_str = f"{vp.mean():.4f}" if len(vp) > 0 else "N/A"
            print(f"  {bin_label:<15} {len(bin_data):<10} {auroc_str:<12} {auprc_str:<12}")

    # ========== 3. 🏆 Top 20 Tasks ==========
    print("\n" + "="*80)
    print("🏆 TOP 20 TASKS BY AUROC")
    print("="*80)

    top_20 = results_df.sort_values('auroc', ascending=False, na_position='last').head(20)

    print(f"\n  {'Rank':<6} {'Task ID':<40} {'HLA':<15} {'Tissue':<20} {'AUROC':<10} {'N':<8}")
    print("  " + "-"*100)

    for rank, (_, row) in enumerate(top_20.iterrows(), 1):
        auroc_str = f"{row['auroc']:.4f}" if not np.isnan(row['auroc']) else "N/A"
        print(f"  {rank:<6} {str(row['task_id']):<40} {row['hla']:<15} "
              f"{str(row['tissue']):<20} {auroc_str:<10} {row['n_samples']:<8}")

    # ========== 4. ⚠️ Bottom 20 Tasks ==========
    print("\n" + "="*80)
    print("⚠️  BOTTOM 20 TASKS BY AUROC")
    print("="*80)

    df_valid = results_df[results_df['auroc'].notna()].copy()

    if len(df_valid) >= 20:
        bottom_20 = df_valid.sort_values('auroc', ascending=True).head(20)

        print(f"\n  {'Rank':<6} {'Task ID':<40} {'HLA':<15} {'Tissue':<20} {'AUROC':<10} {'N':<8}")
        print("  " + "-"*100)

        for rank, (_, row) in enumerate(bottom_20.iterrows(), 1):
            print(f"  {rank:<6} {str(row['task_id']):<40} {row['hla']:<15} "
                  f"{str(row['tissue']):<20} {row['auroc']:<10.4f} {row['n_samples']:<8}")
    else:
        print(f"\n  Not enough valid tasks (only {len(df_valid)})")

    print("\n" + "="*80)

    return results_df


def visualize_results(results_df, output_dir):
    """生成可视化图表并保存"""

    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 2 (HLA×Tissue) Evaluation Results', fontsize=16, fontweight='bold')

    # ========== 1. AUROC分布直方图 ==========
    ax = axes[0, 0]
    valid_aurocs = results_df['auroc'].dropna()

    if len(valid_aurocs) > 0:
        ax.hist(valid_aurocs, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(valid_aurocs.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_aurocs.mean():.3f}')
        ax.axvline(valid_aurocs.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {valid_aurocs.median():.3f}')
        ax.set_xlabel('AUROC', fontsize=12)
        ax.set_ylabel('Number of Tasks', fontsize=12)
        ax.set_title('AUROC Distribution (per HLA×Tissue Task)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # ========== 2. Per-Tissue AUROC箱线图 ==========
    ax = axes[0, 1]
    valid_df = results_df[results_df['auroc'].notna()].copy()

    if len(valid_df) > 0:
        # 按tissue的中位数AUROC排序
        tissue_order = (valid_df.groupby('tissue')['auroc']
                        .median()
                        .sort_values(ascending=False)
                        .index.tolist())

        tissue_data = [valid_df[valid_df['tissue'] == t]['auroc'].values
                       for t in tissue_order]

        bp = ax.boxplot(tissue_data, labels=tissue_order, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2))

        colors = plt.cm.RdYlGn(np.linspace(0.9, 0.3, len(tissue_order)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(tissue_order, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title('AUROC by Tissue', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

    # ========== 3. 按样本数分层的Mean AUROC ==========
    ax = axes[1, 0]
    bins   = [0, 50, 100, 500, 1000, float('inf')]
    labels = ['<50', '50-100', '100-500', '500-1K', '>1K']
    results_df['sample_bin'] = pd.cut(results_df['n_samples'], bins=bins, labels=labels)

    bin_stats = []
    for bin_label in labels:
        bin_data = results_df[results_df['sample_bin'] == bin_label]
        va = bin_data['auroc'].dropna()
        if len(va) > 0:
            bin_stats.append({'bin': bin_label, 'mean_auroc': va.mean(), 'n_tasks': len(bin_data)})

    if bin_stats:
        bin_df = pd.DataFrame(bin_stats)
        bars = ax.bar(bin_df['bin'], bin_df['mean_auroc'],
                      color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Sample Size Range', fontsize=12)
        ax.set_ylabel('Mean AUROC', fontsize=12)
        ax.set_title('Performance by Sample Size', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        for bar, row in zip(bars, bin_df.itertuples()):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                    f'{h:.3f}\n(n={row.n_tasks})',
                    ha='center', va='bottom', fontsize=9)

    # ========== 4. 样本数 vs AUROC 散点图（按Tissue着色）==========
    ax = axes[1, 1]
    valid_df = results_df[results_df['auroc'].notna()].copy()

    if len(valid_df) > 0:
        tissues = valid_df['tissue'].unique()
        palette = plt.cm.tab20(np.linspace(0, 1, len(tissues)))
        tissue_color = {t: palette[i] for i, t in enumerate(tissues)}

        for tissue in tissues:
            td = valid_df[valid_df['tissue'] == tissue]
            ax.scatter(td['n_samples'], td['auroc'],
                       c=[tissue_color[tissue]],
                       label=tissue, s=80, alpha=0.6,
                       edgecolors='black', linewidth=0.4)

        ax.set_xlabel('Number of Samples (log scale)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title('Sample Size vs AUROC (colored by Tissue)', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # 趋势线
        if len(valid_df) > 2:
            z = np.polyfit(np.log10(valid_df['n_samples']), valid_df['auroc'], 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(valid_df['n_samples'].min()),
                                  np.log10(valid_df['n_samples'].max()), 100)
            ax.plot(x_trend, p(np.log10(x_trend)), 'r--', alpha=0.6, linewidth=2,
                    label='Trend')

        # 图例：最多显示10个tissue
        handles, lbls = ax.get_legend_handles_labels()
        n_legend = min(10, len(handles))
        if 'Trend' in lbls:
            trend_idx = lbls.index('Trend')
            ax.legend([handles[trend_idx]] + handles[:n_legend],
                      ['Trend'] + lbls[:n_legend],
                      fontsize=7, loc='lower right')
        else:
            ax.legend(handles[:n_legend], lbls[:n_legend], fontsize=7, loc='lower right')

    plt.tight_layout()

    fig_path = output_dir / 'evaluation_results.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Visualization saved to: {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode2_output_dir', type=str, required=True,
                        help='Mode 2 output directory (contains best_model.pt and task_graph/)')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test.tsv')
    parser.add_argument('--tissue_col', type=str, default='Host',
                        help='Tissue column name (default: Host)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--negative_ratio', type=int, default=10,
                        help='Negative ratio (default: 10)')
    parser.add_argument('--filter_unknown_tissue', action='store_true',
                        help='Filter samples with Unknown tissue')

    args = parser.parse_args()

    print("="*80)
    print("Mode 2 (HLA×Tissue) Evaluation")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode2_dir = Path(args.mode2_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Device:      {device}")
    print(f"  Mode 2 dir:  {mode2_dir}")

    # ========== 1. 加载测试数据 ==========
    print(f"\n📂 Loading test data...")
    test_df = pd.read_csv(args.test_data, sep='\t')

    # 标准化列名
    rename_map = {}
    if 'MHC_Restriction_Name' in test_df.columns:
        rename_map['MHC_Restriction_Name'] = 'hla'
    if 'Peptide' in test_df.columns:
        rename_map['Peptide'] = 'peptide'
    if 'Label' in test_df.columns:
        rename_map['Label'] = 'label'
    test_df = test_df.rename(columns=rename_map)

    # 检测tissue列
    actual_tissue_col = None
    for candidate in [args.tissue_col, 'tissue', 'Host', 'Inferred_Tissue']:
        if candidate in test_df.columns:
            actual_tissue_col = candidate
            break

    if actual_tissue_col is None:
        raise ValueError(f"No tissue column found! Tried: {args.tissue_col}, tissue, Host, Inferred_Tissue")

    print(f"✓ Loaded {len(test_df):,} samples")
    print(f"  HLAs:    {test_df['hla'].nunique()}")
    print(f"  Tissues: {test_df[actual_tissue_col].nunique()}  (column: '{actual_tissue_col}')")
    print(f"  HLA×Tissue combos: {test_df[['hla', actual_tissue_col]].drop_duplicates().shape[0]}")

    # 重命名tissue列为统一的'tissue'
    if actual_tissue_col != 'tissue':
        test_df = test_df.rename(columns={actual_tissue_col: 'tissue'})

    # 填充缺失tissue
    test_df['tissue'] = test_df['tissue'].fillna('Unknown')

    # 可选：过滤Unknown tissue
    if args.filter_unknown_tissue:
        before = len(test_df)
        test_df = test_df[test_df['tissue'] != 'Unknown'].copy()
        print(f"  Filtered Unknown tissue: {before:,} → {len(test_df):,}")

    # ========== 2. 创建Mode2配置 ==========
    print(f"\n⚙️  Creating Mode 2 config...")
    config = create_mode2_config(
        negative_ratio=args.negative_ratio,
        min_samples=1,           # 评估时不筛选，保留所有tasks
        tissue_source='tissue',
    )

    # ========== 3. 创建HLA×Tissue Tasks ==========
    print(f"\n📋 Creating HLA×Tissue tasks...")
    creator = UnifiedTaskCreator(config)
    task_manager = creator.create_tasks(test_df)
    all_tasks = task_manager.get_all_tasks()

    hlas    = set(t.hla    for t in all_tasks.values())
    tissues = set(t.tissue for t in all_tasks.values())
    print(f"✓ Created {len(all_tasks)} tasks")
    print(f"  Unique HLAs:    {len(hlas)}")
    print(f"  Unique Tissues: {len(tissues)}")

    # ========== 4. 加载Task Graph ==========
    print(f"\n📐 Loading task graph...")
    task_graph_dir = mode2_dir / 'task_graph'
    if not task_graph_dir.exists():
        raise FileNotFoundError(f"Task graph not found: {task_graph_dir}")

    graph_wrapper = TaskGraphWrapper(task_graph_dir)
    graph_wrapper.to(device)
    print(f"✓ Loaded task graph: {graph_wrapper.n_tasks} nodes")

    # 获取训练时见过的task keys
    trained_task_keys = set()
    try:
        trained_task_keys = set(graph_wrapper.task_names)
        print(f"  Trained task keys from graph: {len(trained_task_keys)}")
    except AttributeError:
        print(f"  ⚠️  Could not get task_names from graph, using all current tasks")
        trained_task_keys = set(f"HLA_{t.hla}_TISSUE_{t.tissue}"
                                for t in all_tasks.values())

    # ========== 5. 生成测试负样本 ==========
    print(f"\n🔄 Generating test negatives...")

    # 过滤graph中存在的tasks
    if hasattr(graph_wrapper, 'task_to_idx'):
        valid_task_ids = set(graph_wrapper.task_to_idx.keys())
    else:
        valid_task_ids = trained_task_keys

    test_task_datasets = {}
    skipped_tasks = []
    test_sampler = EnhancedNegativeSampler(config, test_df)

    for task_id, task in all_tasks.items():
        # 检查是否在graph中
        task_key = f"HLA_{task.hla}_TISSUE_{task.tissue}"
        if task_key not in valid_task_ids:
            skipped_tasks.append(f"{task.hla}×{task.tissue}")
            continue

        # 筛选该task对应的正样本
        task_test_df = test_df[
            (test_df['hla'] == task.hla) &
            (test_df['tissue'] == task.tissue)
        ].copy()

        if len(task_test_df) == 0:
            continue

        positive_peptides = list(task_test_df['peptide'])
        negative_peptides = test_sampler.generate_negatives_for_task(
            task, positive_peptides
        )

        neg_df = pd.DataFrame({
            'peptide': negative_peptides,
            'hla':     task.hla,
            'tissue':  task.tissue,
            'label':   0
        })

        task_test_combined = pd.concat([task_test_df, neg_df], ignore_index=True)
        test_task_datasets[task_id] = task_test_combined

    if skipped_tasks:
        print(f"  ⚠️  Skipped {len(skipped_tasks)} tasks not in graph:")
        for t in skipped_tasks[:5]:
            print(f"      {t}")
        if len(skipped_tasks) > 5:
            print(f"      ... and {len(skipped_tasks)-5} more")

    print(f"✓ Generated negatives for {len(test_task_datasets)} tasks")

    # ========== 6. 创建DataLoader ==========
    print(f"\n📦 Creating test DataLoader...")

    test_dataset = ModeAwareDataset(
        test_task_datasets,
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

    # ========== 7. 加载模型 ==========
    print(f"\n🔧 Loading model...")
    checkpoint_path = mode2_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    n_tasks_in_model = graph_wrapper.n_tasks

    model = ImmuneAppModel(
        mode_config=config,
        n_tasks=n_tasks_in_model
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded  (n_tasks={n_tasks_in_model})")

    params = model.get_parameters_count()
    print(f"\n  Model parameters:")
    for component, count in params.items():
        print(f"    {component}: {count:,}")

    # ========== 8. 评估 ==========
    print(f"\n{'='*80}")
    print("Evaluating per HLA×Tissue task...")
    print(f"{'='*80}")

    results_df, all_predictions = evaluate_per_task(
        model, test_loader, task_manager,
        test_task_datasets, graph_wrapper, device
    )

    # ========== 9. 打印统计 ==========
    results_df = print_detailed_statistics(results_df)

    # ========== 10. 可视化 ==========
    print(f"\n📊 Generating visualizations...")
    visualize_results(results_df, output_dir)

    # ========== 11. 保存结果 ==========
    print(f"\n💾 Saving results...")

    # Per-task完整结果
    task_file = output_dir / 'per_task_results.csv'
    results_df.to_csv(task_file, index=False)
    print(f"✓ Per-task results:   {task_file}")

    # Per-tissue汇总
    valid_df = results_df[results_df['auroc'].notna()].copy()
    if len(valid_df) > 0:
        tissue_summary = valid_df.groupby('tissue').apply(
            lambda x: pd.Series({
                'n_tasks':    len(x),
                'n_positive': x['n_positive'].sum(),
                'mean_auroc': np.average(x['auroc'], weights=x['n_positive']),
                'mean_auprc': np.average(x['auprc'].fillna(0), weights=x['n_positive']),
                'mean_f1':    np.average(x['f1'].fillna(0),    weights=x['n_positive']),
            })
        ).reset_index().sort_values('mean_auroc', ascending=False)

        tissue_file = output_dir / 'per_tissue_summary.csv'
        tissue_summary.to_csv(tissue_file, index=False)
        print(f"✓ Per-tissue summary: {tissue_file}")

        # 打印汇总表
        print(f"\n{'='*80}")
        print("Per-Tissue Performance Summary")
        print(f"{'='*80}")
        print(tissue_summary.to_string(index=False))
        print(f"{'='*80}")

    print(f"\n✓ Evaluation complete!")
    print(f"\n📁 Results: {output_dir}")


if __name__ == '__main__':
    main()
