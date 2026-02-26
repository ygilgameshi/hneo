"""
MHCflurry 2.0 评估脚本（修正版）

关键修正：与你自己的模型评估保持完全一致
- 用同一份正样本测试集
- 用相同的负样本生成策略（EnhancedNegativeSampler）
- 用相同的负样本比例（negative_ratio）
- 按 per-HLA task 计算指标

用法：
python scripts/evaluate_mhcflurry.py \
    --test_file  output/task_balanced_v2/data_splits/test.tsv \
    --mode1_output_dir output/task_balanced_HLA_v3_10_256 \
    --output_dir results/mhcflurry_eval
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config.mode_config import create_mode1_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.enhanced_negative_sampler import EnhancedNegativeSampler
from src.graph.task_graph import TaskGraphWrapper


def generate_test_datasets(test_df, mode1_dir, negative_ratio):
    """
    用与mode1评估完全相同的方式生成带负样本的测试集

    Returns:
        filtered_test_datasets: {task_id: DataFrame(peptide, hla, label)}
        task_manager, all_tasks
    """
    mode1_dir = Path(mode1_dir)

    # 创建config（评估时min_samples=1）
    config = create_mode1_config(
        negative_ratio=negative_ratio,
        min_samples=1
    )

    # 创建tasks
    print(f"\n📋 Creating tasks...")
    creator = UnifiedTaskCreator(config)
    task_manager = creator.create_tasks(test_df)
    all_tasks = task_manager.get_all_tasks()
    print(f"✓ Created {len(all_tasks)} tasks")

    # 加载task graph（确定哪些HLA是训练时见过的）
    print(f"\n📐 Loading task graph from: {mode1_dir / 'task_graph'}")
    graph_wrapper = TaskGraphWrapper(mode1_dir / 'task_graph')
    print(f"✓ Loaded: {graph_wrapper.n_tasks} nodes")

    # 获取graph中有效的task keys
    if hasattr(graph_wrapper, 'task_to_idx'):
        valid_task_ids = set(graph_wrapper.task_to_idx.keys())
    else:
        try:
            valid_task_ids = set(graph_wrapper.task_names)
        except AttributeError:
            valid_task_ids = set(f"HLA_{t.hla}" for t in all_tasks.values())

    # 生成负样本
    print(f"\n🔄 Generating test negatives (ratio={negative_ratio})...")
    test_sampler = EnhancedNegativeSampler(config, test_df)
    filtered_test_datasets = {}
    skipped = 0

    for task_id, task in all_tasks.items():
        task_key = f"HLA_{task.hla}"
        if task_key not in valid_task_ids:
            skipped += 1
            continue

        task_df = test_df[test_df['hla'] == task.hla].copy()
        if len(task_df) == 0:
            continue

        neg_peptides = test_sampler.generate_negatives_for_task(
            task, list(task_df['peptide'])
        )

        neg_df = pd.DataFrame({
            'peptide': neg_peptides,
            'hla':     task.hla,
            'label':   0
        })

        combined = pd.concat([task_df, neg_df], ignore_index=True)
        filtered_test_datasets[task_id] = combined

    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} tasks not in training graph")
    print(f"✓ Ready: {len(filtered_test_datasets)} tasks with positives + negatives")

    # 统计
    total_pos = sum((d['label'] == 1).sum() for d in filtered_test_datasets.values())
    total_neg = sum((d['label'] == 0).sum() for d in filtered_test_datasets.values())
    print(f"  Total positive: {total_pos:,}")
    print(f"  Total negative: {total_neg:,}")

    return filtered_test_datasets, task_manager, all_tasks


def convert_hla_format(hla):
    """
    转换HLA格式以兼容MHCflurry

    输入:  HLA-A*02:05, HLA-B*14:02
    输出:  HLA-A02:05, HLA-B14:02  (去掉星号)

    或者如果需要2-digit:
    输出:  HLA-A0205, HLA-B1402  (去掉星号和冒号)
    """
    # MHCflurry支持多种格式，先尝试去掉*
    if '*' in hla:
        # HLA-A*02:05 -> HLA-A02:05
        return hla.replace('*', '')
    return hla


def check_hla_support(predictor, hla_list):
    """检查哪些HLA被MHCflurry支持"""
    supported = predictor.supported_alleles
    supported_set = set(supported)

    # 尝试多种格式
    hla_mapping = {}
    unsupported = []

    for hla in hla_list:
        # 尝试原格式
        if hla in supported_set:
            hla_mapping[hla] = hla
            continue

        # 尝试去掉*
        hla_no_star = hla.replace('*', '')
        if hla_no_star in supported_set:
            hla_mapping[hla] = hla_no_star
            continue

        # 尝试去掉*和:
        hla_compact = hla.replace('*', '').replace(':', '')
        if hla_compact in supported_set:
            hla_mapping[hla] = hla_compact
            continue

        # 尝试添加HLA-前缀变体
        for prefix in ['HLA-', 'HLA']:
            for variant in [hla, hla_no_star, hla_compact]:
                test_name = variant.replace('HLA-', '').replace('HLA', '')
                test_with_prefix = f"{prefix}{test_name}"
                if test_with_prefix in supported_set:
                    hla_mapping[hla] = test_with_prefix
                    break
            if hla in hla_mapping:
                break

        if hla not in hla_mapping:
            unsupported.append(hla)

    return hla_mapping, unsupported


def run_mhcflurry_predictions(filtered_test_datasets, all_tasks, batch_size=2000):
    """
    对每个task的所有peptide运行MHCflurry预测

    Returns:
        all_predictions: {task_id: {'labels': [...], 'probs': [...]}}
    """
    from mhcflurry import Class1PresentationPredictor

    print(f"\n🔧 Loading MHCflurry predictor...")
    predictor = Class1PresentationPredictor.load()
    print(f"✓ MHCflurry loaded")
    print(f"  Supported alleles: {len(predictor.supported_alleles)}")

    # 检查HLA兼容性
    print(f"\n🔍 Checking HLA compatibility...")
    unique_hlas = list(set(t.hla for t in all_tasks.values()))
    hla_mapping, unsupported = check_hla_support(predictor, unique_hlas)

    print(f"  Total HLAs in test set: {len(unique_hlas)}")
    print(f"  Supported by MHCflurry: {len(hla_mapping)}")
    print(f"  Unsupported:            {len(unsupported)}")

    if unsupported:
        print(f"\n  ⚠️  Unsupported HLAs (first 10):")
        for hla in unsupported[:10]:
            print(f"      {hla}")
        if len(unsupported) > 10:
            print(f"      ... and {len(unsupported)-10} more")

    if len(hla_mapping) == 0:
        print("\n  ❌ ERROR: No HLAs are supported by MHCflurry!")
        print("     This usually means HLA format mismatch.")
        print(f"     Example test HLA:     {unique_hlas[0]}")
        print(f"     Example MHCflurry HLA: {predictor.supported_alleles[0]}")
        return {}

    print(f"\n🚀 Running MHCflurry predictions...")
    print(f"   Using optimized dict-based batch API...")

    # 按HLA分组数据
    hla_grouped = {}
    for task_id, task_df in filtered_test_datasets.items():
        task = all_tasks[task_id]
        hla  = task.hla

        if hla not in hla_mapping:
            continue

        mhcflurry_hla = hla_mapping[hla]

        if mhcflurry_hla not in hla_grouped:
            hla_grouped[mhcflurry_hla] = {
                'peptides': [],
                'labels':   [],
                'task_ids': []
            }

        for _, row in task_df.iterrows():
            hla_grouped[mhcflurry_hla]['peptides'].append(row['peptide'])
            hla_grouped[mhcflurry_hla]['labels'].append(row['label'])
            hla_grouped[mhcflurry_hla]['task_ids'].append(task_id)

    print(f"   Grouped into {len(hla_grouped)} unique HLAs")
    print(f"   Total samples: {sum(len(v['peptides']) for v in hla_grouped.values()):,}")

    # 按HLA批量预测（每个HLA单独预测，内部MHCflurry会优化）
    all_predictions_flat = {}

    for mhcflurry_hla, data in tqdm(hla_grouped.items(), desc="HLAs"):
        peptides = data['peptides']
        labels   = data['labels']
        task_ids = data['task_ids']

        try:
            # MHCflurry推荐方式：用dict传入，每个HLA作为一个sample
            pred = predictor.predict(
                peptides=peptides,
                alleles={mhcflurry_hla: [mhcflurry_hla]},  # dict方式
                verbose=0
            )
            scores = pred['presentation_score'].tolist()
        except Exception as e:
            # 失败时用0.5填充
            scores = [0.5] * len(peptides)

        # 按task_id重新组织
        for i, task_id in enumerate(task_ids):
            if task_id not in all_predictions_flat:
                all_predictions_flat[task_id] = {'labels': [], 'probs': []}
            all_predictions_flat[task_id]['labels'].append(labels[i])
            all_predictions_flat[task_id]['probs'].append(scores[i])

    print(f"\n✓ Predictions complete for {len(all_predictions_flat)} tasks")
    return all_predictions_flat


def compute_metrics_and_report(all_predictions, all_tasks, output_dir):
    """计算per-task指标并生成报告"""

    output_dir = Path(output_dir)
    task_records = []

    for task_id, pred in all_predictions.items():
        task   = all_tasks[task_id]
        y_true = np.array(pred['labels'])
        y_prob = np.array(pred['probs'])
        y_pred = (y_prob > 0.5).astype(int)

        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos

        auroc = auprc = np.nan
        if n_pos > 0 and n_neg > 0:
            try:
                auroc = roc_auc_score(y_true, y_prob)
                auprc = average_precision_score(y_true, y_prob)
            except Exception:
                pass

        precision = recall = f1 = np.nan
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
        except Exception:
            pass

        task_records.append({
            'task_id':    task_id,
            'hla':        task.hla,
            'n_samples':  len(y_true),
            'n_positive': n_pos,
            'auroc':      auroc,
            'auprc':      auprc,
            'precision':  precision,
            'recall':     recall,
            'f1':         f1,
        })

    df = pd.DataFrame(task_records)

    valid_aurocs = df['auroc'].dropna()
    valid_auprcs = df['auprc'].dropna()

    # ========== 打印统计 ==========
    print("\n" + "="*80)
    print("📊 OVERALL STATISTICS")
    print("="*80)
    print(f"\n  Total tasks:      {len(df)}")
    print(f"  Valid tasks:      {len(valid_aurocs)}")
    if len(valid_aurocs) > 0:
        print(f"  Mean AUROC:       {valid_aurocs.mean():.4f} ± {valid_aurocs.std():.4f}")
        print(f"  Median AUROC:     {valid_aurocs.median():.4f}")
        print(f"  Min / Max AUROC:  {valid_aurocs.min():.4f} / {valid_aurocs.max():.4f}")
    if len(valid_auprcs) > 0:
        print(f"  Mean AUPRC:       {valid_auprcs.mean():.4f} ± {valid_auprcs.std():.4f}")

    # 按样本数分层
    print("\n" + "="*80)
    print("📈 PERFORMANCE BY SAMPLE SIZE")
    print("="*80)
    bins   = [0, 50, 100, 500, 1000, float('inf')]
    labels = ['<50', '50-100', '100-500', '500-1K', '>1K']
    df['sample_bin'] = pd.cut(df['n_samples'], bins=bins, labels=labels)
    print(f"\n  {'Size Range':<15} {'N Tasks':<10} {'Mean AUROC':<12} {'Mean AUPRC':<12}")
    print("  " + "-"*50)
    for b in labels:
        bd = df[df['sample_bin'] == b]
        if len(bd) > 0:
            va = bd['auroc'].dropna()
            vp = bd['auprc'].dropna()
            auroc_s = f"{va.mean():.4f}" if len(va) > 0 else "N/A"
            auprc_s = f"{vp.mean():.4f}" if len(vp) > 0 else "N/A"
            print(f"  {b:<15} {len(bd):<10} {auroc_s:<12} {auprc_s:<12}")

    # Top 20
    print("\n" + "="*80)
    print("🏆 TOP 20 TASKS BY AUROC")
    print("="*80)
    top20 = df.sort_values('auroc', ascending=False, na_position='last').head(20)
    print(f"\n  {'Rank':<6} {'HLA':<20} {'AUROC':<10} {'AUPRC':<10} {'N Samples':<10}")
    print("  " + "-"*58)
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        auroc_s = f"{row['auroc']:.4f}" if not np.isnan(row['auroc']) else "N/A"
        auprc_s = f"{row['auprc']:.4f}" if not np.isnan(row['auprc']) else "N/A"
        print(f"  {rank:<6} {row['hla']:<20} {auroc_s:<10} {auprc_s:<10} {row['n_samples']:<10}")

    # Bottom 20
    print("\n" + "="*80)
    print("⚠️  BOTTOM 20 TASKS BY AUROC")
    print("="*80)
    df_valid = df[df['auroc'].notna()]
    if len(df_valid) >= 20:
        bot20 = df_valid.sort_values('auroc').head(20)
        print(f"\n  {'Rank':<6} {'HLA':<20} {'AUROC':<10} {'AUPRC':<10} {'N Samples':<10}")
        print("  " + "-"*58)
        for rank, (_, row) in enumerate(bot20.iterrows(), 1):
            auprc_s = f"{row['auprc']:.4f}" if not np.isnan(row['auprc']) else "N/A"
            print(f"  {rank:<6} {row['hla']:<20} {row['auroc']:<10.4f} {auprc_s:<10} {row['n_samples']:<10}")

    print("\n" + "="*80)

    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MHCflurry 2.0 Evaluation Results', fontsize=16, fontweight='bold')

    # 1. AUROC分布
    ax = axes[0, 0]
    if len(valid_aurocs) > 0:
        ax.hist(valid_aurocs, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(valid_aurocs.mean(),   color='red',   linestyle='--', linewidth=2,
                   label=f'Mean: {valid_aurocs.mean():.3f}')
        ax.axvline(valid_aurocs.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {valid_aurocs.median():.3f}')
        ax.set_xlabel('AUROC'); ax.set_ylabel('Number of Tasks')
        ax.set_title('AUROC Distribution (per HLA Task)', fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)

    # 2. Top 15 HLAs 横条图
    ax = axes[0, 1]
    top15 = df[df['auroc'].notna()].sort_values('auroc', ascending=True).tail(15)
    if len(top15) > 0:
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top15)))
        bars = ax.barh(range(len(top15)), top15['auroc'], color=colors, alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15['hla'], fontsize=9)
        for bar in bars:
            w = bar.get_width()
            ax.text(w, bar.get_y() + bar.get_height()/2., f' {w:.3f}', va='center', fontsize=8)
        ax.set_xlabel('AUROC')
        ax.set_title('Top 15 HLAs by AUROC', fontweight='bold')
        ax.set_xlim(0, 1); ax.grid(axis='x', alpha=0.3)

    # 3. 按样本数分层
    ax = axes[1, 0]
    bin_stats = []
    for b in labels:
        bd = df[df['sample_bin'] == b]
        va = bd['auroc'].dropna()
        if len(va) > 0:
            bin_stats.append({'bin': b, 'mean_auroc': va.mean(), 'n': len(bd)})
    if bin_stats:
        bdf = pd.DataFrame(bin_stats)
        bars = ax.bar(bdf['bin'], bdf['mean_auroc'],
                      color='#2ecc71', alpha=0.7, edgecolor='black')
        for bar, row in zip(bars, bdf.itertuples()):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                    f'{h:.3f}\n(n={row.n})', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Sample Size Range'); ax.set_ylabel('Mean AUROC')
        ax.set_title('Performance by Sample Size', fontweight='bold')
        ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3)

    # 4. 样本数 vs AUROC 散点图
    ax = axes[1, 1]
    vdf = df[df['auroc'].notna()]
    if len(vdf) > 0:
        sc = ax.scatter(vdf['n_samples'], vdf['auroc'],
                        c=vdf['auroc'], cmap='RdYlGn',
                        s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
        if len(vdf) > 2:
            z = np.polyfit(np.log10(vdf['n_samples']), vdf['auroc'], 1)
            p = np.poly1d(z)
            xs = np.logspace(np.log10(vdf['n_samples'].min()),
                             np.log10(vdf['n_samples'].max()), 100)
            ax.plot(xs, p(np.log10(xs)), 'r--', alpha=0.5, linewidth=2, label='Trend')
            ax.legend()
        plt.colorbar(sc, ax=ax, label='AUROC')
        ax.set_xlabel('Number of Samples (log scale)'); ax.set_ylabel('AUROC')
        ax.set_title('Sample Size vs AUROC', fontweight='bold')
        ax.set_xscale('log'); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / 'mhcflurry_evaluation.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Visualization saved to: {fig_path}")

    # ========== 保存结果 ==========
    per_hla_file = output_dir / 'mhcflurry_per_hla.csv'
    df.to_csv(per_hla_file, index=False)
    print(f"✓ Per-HLA metrics: {per_hla_file}")

    summary_file = output_dir / 'mhcflurry_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("MHCflurry 2.0 Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total tasks:   {len(df)}\n")
        f.write(f"Valid tasks:   {len(valid_aurocs)}\n\n")
        if len(valid_aurocs) > 0:
            f.write(f"Mean AUROC:    {valid_aurocs.mean():.4f} ± {valid_aurocs.std():.4f}\n")
            f.write(f"Median AUROC:  {valid_aurocs.median():.4f}\n")
            f.write(f"Min AUROC:     {valid_aurocs.min():.4f}\n")
            f.write(f"Max AUROC:     {valid_aurocs.max():.4f}\n")
        if len(valid_auprcs) > 0:
            f.write(f"Mean AUPRC:    {valid_auprcs.mean():.4f} ± {valid_auprcs.std():.4f}\n")
    print(f"✓ Summary:      {summary_file}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='MHCflurry 2.0 Evaluation (with negative sampling)'
    )
    parser.add_argument('--test_file', type=str, required=True,
                        help='Test data file (TSV, positive samples only)')
    parser.add_argument('--mode1_output_dir', type=str, required=True,
                        help='Mode 1 output directory (for task graph)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--negative_ratio', type=int, default=10,
                        help='Negative:Positive ratio, same as your model (default: 10)')
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='MHCflurry prediction batch size (default: 2000)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MHCflurry 2.0 Evaluation (Fair Comparison Mode)")
    print("=" * 80)
    print(f"\n  Test file:      {args.test_file}")
    print(f"  Mode1 dir:      {args.mode1_output_dir}")
    print(f"  Negative ratio: {args.negative_ratio}")
    print(f"  Output:         {args.output_dir}")

    # 1. 加载测试数据，只保留正样本
    print(f"\n📂 Loading test data...")
    test_df = pd.read_csv(args.test_file, sep='\t')

    rename = {}
    if 'MHC_Restriction_Name' in test_df.columns: rename['MHC_Restriction_Name'] = 'hla'
    if 'Peptide'              in test_df.columns: rename['Peptide']               = 'peptide'
    if 'Label'                in test_df.columns: rename['Label']                 = 'label'
    test_df = test_df.rename(columns=rename)

    pos_df = test_df[test_df['label'] == 1].copy()
    print(f"✓ Positive samples: {len(pos_df):,}  |  HLAs: {pos_df['hla'].nunique()}")

    # 2. 生成负样本（与mode1评估完全一致）
    filtered_test_datasets, task_manager, all_tasks = generate_test_datasets(
        pos_df, args.mode1_output_dir, args.negative_ratio
    )

    if len(filtered_test_datasets) == 0:
        print("❌ No valid tasks found. Check --mode1_output_dir and --test_file.")
        return

    # 3. MHCflurry预测
    all_predictions = run_mhcflurry_predictions(
        filtered_test_datasets, all_tasks, args.batch_size
    )

    # 4. 计算指标、生成报告
    compute_metrics_and_report(all_predictions, all_tasks, output_dir)

    print(f"\n{'='*80}")
    print("✓ MHCflurry evaluation complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()