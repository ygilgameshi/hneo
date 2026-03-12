"""
MixMHCpred 2.2 评估脚本

MixMHCpred 是命令行工具，在多个基准测试中表现最佳

安装：
cd /path/to/your/workspace
git clone https://github.com/GfellerLab/MixMHCpred.git

用法：
python scripts/evaluate_mixmhcpred.py \
    --test_file        output/task_balanced_v2/data_splits/test.tsv \
    --mode1_output_dir output/task_balanced_HLA_v3_10_256 \
    --mixmhcpred_dir   /path/to/MixMHCpred \
    --output_dir       results/mixmhcpred_eval \
    --negative_ratio   10
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import subprocess
import tempfile
import os

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config.mode_config import create_mode1_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.enhanced_negative_sampler import EnhancedNegativeSampler
from src.graph.task_graph import TaskGraphWrapper


def convert_hla_to_mixmhcpred_format(hla):
    """
    转换HLA格式为MixMHCpred格式
    
    输入:  HLA-A*02:01, HLA-B*14:02
    输出:  A0201, B1402  (去掉HLA-和*和:)
    """
    hla = hla.replace('HLA-', '').replace('*', '').replace(':', '')
    return hla


def load_test_datasets_from_cache(data_file, mode1_dir,
                                  cache_dir='data/negative_samples',
                                  negative_ratio=10, min_samples=10):
    """
    从 NegativeSampleCache（.pkl）直接加载 test 负样本，跳过生成步骤。

    缓存文件命名格式：{data_stem}_mode2_{hash}_test.pkl
    对应 data/negative_samples/ 目录下训练时保存的缓存。

    Args:
        data_file:       原始数据文件路径（与训练时一致，用于还原 cache key）
        mode1_dir:       Mode 1 输出目录（加载 task graph 过滤无效 task）
        cache_dir:       负样本缓存目录（默认 data/negative_samples）
        negative_ratio:  负样本比例（与训练时一致，影响 cache key）
        min_samples:     最小样本数（与训练时一致，影响 cache key）

    Returns:
        filtered_test_datasets: {task_id: DataFrame}
        task_manager:           TaskManager 实例（基于 test 正样本构建）
        all_tasks:              {task_id: Task}
    """
    from src.data.negative_cache import NegativeSampleCache
    from src.config.mode_config import create_mode1_config
    from src.data.unified_task_creator import UnifiedTaskCreator

    mode1_dir = Path(mode1_dir)

    # ── 构建与训练时相同的 cache_config ──
    cache_config = {
        'negative_ratio':           negative_ratio,
        'use_tissue_aware_negatives': False,   # 已废弃，保持 False 与 key 一致
        'min_samples':              min_samples,
        'tissue_source':            'Host',
    }

    cache = NegativeSampleCache(cache_dir)

    if not cache.exists(data_file, 'mode1', cache_config):
        print(f"  ❌ 缓存不存在，请检查:")
        print(f"       data_file:      {data_file}")
        print(f"       cache_dir:      {cache_dir}")
        print(f"       negative_ratio: {negative_ratio}")
        print(f"       min_samples:    {min_samples}")
        print(f"  提示: 缓存 key 由这些参数生成，需与训练时完全一致。")
        return None, None, None

    # ── 加载缓存（只需要 test 部分）──
    result = cache.load(data_file, 'mode1', cache_config)
    if result is None:
        print("  ❌ 缓存加载失败")
        return None, None, None

    _, _, test_task_datasets = result
    print(f"\n✓ 从缓存加载 test 负样本: {len(test_task_datasets)} tasks")

    # ── 统计 ──
    total_pos = sum((df['label'] == 1).sum() for df in test_task_datasets.values())
    total_neg = sum((df['label'] == 0).sum() for df in test_task_datasets.values())
    print(f"  Positive: {total_pos:,}  |  Negative: {total_neg:,}")

    # ── 从 test 正样本构建 task_manager ──
    all_pos = pd.concat(
        [df[df['label'] == 1] for df in test_task_datasets.values()],
        ignore_index=True
    )
    config = create_mode1_config(negative_ratio=negative_ratio, min_samples=1)
    creator = UnifiedTaskCreator(config)
    task_manager = creator.create_tasks(all_pos)
    all_tasks = task_manager.get_all_tasks()

    # ── 加载 task graph，过滤不在图中的 task ──
    print(f"\n📐 Loading task graph...")
    graph_wrapper = TaskGraphWrapper(mode1_dir / 'task_graph')
    print(f"✓ Loaded: {graph_wrapper.n_tasks} nodes")

    if hasattr(graph_wrapper, 'task_to_idx'):
        valid_task_ids = set(graph_wrapper.task_to_idx.keys())
    else:
        try:
            valid_task_ids = set(graph_wrapper.task_names)
        except AttributeError:
            valid_task_ids = set(f"HLA_{t.hla}" for t in all_tasks.values())

    filtered_test_datasets = {}
    skipped = 0
    for task_id, task_df in test_task_datasets.items():
        # task_id 格式与 all_tasks 对齐
        task = all_tasks.get(task_id)
        if task is None:
            skipped += 1
            continue
        task_key = f"HLA_{task.hla}"
        if task_key not in valid_task_ids:
            skipped += 1
            continue
        filtered_test_datasets[task_id] = task_df

    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} tasks not in training graph")

    print(f"✓ Ready: {len(filtered_test_datasets)} tasks")
    return filtered_test_datasets, task_manager, all_tasks


def generate_test_datasets(test_df, mode1_dir, negative_ratio):
    """生成带负样本的测试集（与MHCflurry相同）"""
    mode1_dir = Path(mode1_dir)

    config = create_mode1_config(
        negative_ratio=negative_ratio,
        min_samples=1
    )

    print(f"\n📋 Creating tasks...")
    creator = UnifiedTaskCreator(config)
    task_manager = creator.create_tasks(test_df)
    all_tasks = task_manager.get_all_tasks()
    print(f"✓ Created {len(all_tasks)} tasks")

    print(f"\n📐 Loading task graph...")
    graph_wrapper = TaskGraphWrapper(mode1_dir / 'task_graph')
    print(f"✓ Loaded: {graph_wrapper.n_tasks} nodes")

    if hasattr(graph_wrapper, 'task_to_idx'):
        valid_task_ids = set(graph_wrapper.task_to_idx.keys())
    else:
        try:
            valid_task_ids = set(graph_wrapper.task_names)
        except AttributeError:
            valid_task_ids = set(f"HLA_{t.hla}" for t in all_tasks.values())

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

    total_pos = sum((d['label'] == 1).sum() for d in filtered_test_datasets.values())
    total_neg = sum((d['label'] == 0).sum() for d in filtered_test_datasets.values())
    print(f"✓ Ready: {len(filtered_test_datasets)} tasks")
    print(f"  Total positive: {total_pos:,}")
    print(f"  Total negative: {total_neg:,}")

    return filtered_test_datasets, task_manager, all_tasks


def run_mixmhcpred_predictions(filtered_test_datasets, all_tasks, mixmhcpred_dir):
    """
    调用MixMHCpred命令行工具进行预测

    MixMHCpred 命令格式：
    ./MixMHCpred -i input.txt -o output.txt -a A0201,B0702,C0303

    输入文件格式：每行一个peptide
    输出文件格式：peptide, allele, %Rank_best, Score_best, ...
    """
    mixmhcpred_exe = Path(mixmhcpred_dir) / 'MixMHCpred'

    if not mixmhcpred_exe.exists():
        raise FileNotFoundError(
            f"MixMHCpred executable not found: {mixmhcpred_exe}\n"
            f"Please install: git clone https://github.com/GfellerLab/MixMHCpred.git"
        )

    print(f"\n🚀 Running MixMHCpred predictions...")
    print(f"   Executable: {mixmhcpred_exe}")

    all_predictions = {}
    failed_hlas = []

    for task_id, task_df in tqdm(filtered_test_datasets.items(), desc="Tasks"):
        task     = all_tasks[task_id]
        hla      = task.hla
        peptides = task_df['peptide'].tolist()
        labels   = task_df['label'].tolist()

        mixmhc_hla = convert_hla_to_mixmhcpred_format(hla)

        # MixMHCpred只支持长度8-14的peptides，过滤其他长度
        valid_indices = [i for i, pep in enumerate(peptides) if 8 <= len(pep) <= 14]
        invalid_count = len(peptides) - len(valid_indices)

        if len(valid_indices) == 0:
            # 所有peptides都不符合长度要求，用0.5填充
            scores = np.array([0.5] * len(peptides))
            all_predictions[task_id] = {
                'labels': labels,
                'probs':  scores.tolist()
            }
            if invalid_count > 0:
                failed_hlas.append((hla, f"All {len(peptides)} peptides have invalid length (not 8-14)"))
            continue

        # 只保留有效长度的peptides
        valid_peptides = [peptides[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]

        # 使用用户临时目录（避免/tmp权限问题）
        import time
        timestamp = int(time.time() * 1000000)
        # 使用环境变量TMPDIR或者当前用户的home目录
        tmpdir = os.environ.get('TMPDIR', os.path.expanduser('~/tmp'))
        os.makedirs(tmpdir, exist_ok=True)

        # 清理task_id中的特殊字符
        safe_task_id = task_id.replace('*', '_').replace(':', '_').replace('/', '_')
        temp_input = os.path.join(tmpdir, f"mixmhc_input_{timestamp}_{safe_task_id}.txt")
        temp_output = os.path.join(tmpdir, f"mixmhc_output_{timestamp}_{safe_task_id}.txt")

        # 写入有效的peptides
        with open(temp_input, 'w') as f:
            for pep in valid_peptides:
                f.write(f"{pep}\n")

        try:
            # 调用MixMHCpred
            cmd = [
                str(mixmhcpred_exe),
                '-i', temp_input,
                '-o', temp_output,
                '-a', mixmhc_hla
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(mixmhcpred_exe.parent)  # 在MixMHCpred目录下运行
            )

            # 打印第一个任务的详细信息用于调试
            if task_id == list(filtered_test_datasets.keys())[0]:
                print(f"\n  🔍 Debug info for first task ({hla}):")
                print(f"      Command: {' '.join(cmd)}")
                print(f"      Return code: {result.returncode}")
                if result.stdout:
                    print(f"      STDOUT: {result.stdout[:300]}")
                if result.stderr:
                    print(f"      STDERR: {result.stderr[:300]}")
                print(f"      Output file exists: {os.path.exists(temp_output)}")
                if os.path.exists(temp_output):
                    print(f"      Output file size: {os.path.getsize(temp_output)} bytes")

            if result.returncode != 0:
                raise Exception(f"MixMHCpred failed (code {result.returncode})")

            # 检查输出文件是否存在
            if not os.path.exists(temp_output):
                raise Exception(f"Output file not created: {temp_output}")

            # 读取预测结果
            # MixMHCpred输出格式：
            # 开头有多行 # 注释
            # 然后是 tab 分隔的数据：Peptide  Score_bestAllele  BestAllele  %Rank_bestAllele  ...
            result_df = pd.read_csv(temp_output, sep='\t', comment='#')

            # 打印第一个任务的输出格式
            if task_id == list(filtered_test_datasets.keys())[0]:
                print(f"      Output columns: {list(result_df.columns)}")
                print(f"      First few rows:\n{result_df.head(3)}")

            # MixMHCpred返回Score（范围可能是负数到正数）
            # 从测试输出看，列名是：Score_bestAllele, %Rank_bestAllele, Score_A0201
            if 'Score_bestAllele' in result_df.columns:
                raw_scores = result_df['Score_bestAllele'].values
                # Score范围约-2到+2，需要归一化到0-1
                # 使用sigmoid函数: 1 / (1 + exp(-score))
                scores = 1 / (1 + np.exp(-raw_scores))
            elif 'Score_best' in result_df.columns:
                raw_scores = result_df['Score_best'].values
                scores = 1 / (1 + np.exp(-raw_scores))
            elif 'Score' in result_df.columns:
                raw_scores = result_df['Score'].values
                scores = 1 / (1 + np.exp(-raw_scores))
            else:
                # 如果没有Score列，用%Rank转换（%Rank越小越好）
                rank_col = '%Rank_bestAllele' if '%Rank_bestAllele' in result_df.columns else '%Rank_best'
                rank_values = result_df[rank_col].values
                scores = 1 - (rank_values / 100.0)  # 转换为0-1分数

            scores = np.clip(scores, 0, 1)

            # 如果有peptides被过滤了（长度不符合8-14），需要为它们填充0.5
            if invalid_count > 0:
                full_scores = np.full(len(peptides), 0.5)
                full_scores[valid_indices] = scores
                scores = full_scores

        except Exception as e:
            # 预测失败，用0.5填充
            failed_hlas.append((hla, str(e)))
            scores = np.array([0.5] * len(peptides))

        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_input):
                    os.unlink(temp_input)
                if os.path.exists(temp_output):
                    os.unlink(temp_output)
            except:
                pass

        all_predictions[task_id] = {
            'labels': labels,
            'probs':  scores.tolist()
        }

    if failed_hlas:
        print(f"\n  ⚠️  Failed predictions for {len(failed_hlas)} HLAs (first 5):")
        for hla, err in failed_hlas[:5]:
            print(f"      {hla}: {err[:80]}")

    print(f"\n✓ Predictions complete for {len(all_predictions)} tasks")
    return all_predictions


def compute_metrics_and_report(all_predictions, all_tasks, output_dir):
    """计算指标并生成报告（与MHCflurry相同）"""

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

    # 打印统计
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

    print("\n" + "="*80)

    # 可视化（与MHCflurry相同）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MixMHCpred 2.2 Evaluation Results', fontsize=16, fontweight='bold')

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

    # 2. Top 15 HLAs
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

    # 4. 样本数 vs AUROC
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
    fig_path = output_dir / 'mixmhcpred_evaluation.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Visualization saved to: {fig_path}")

    # 保存结果
    per_hla_file = output_dir / 'mixmhcpred_per_hla.csv'
    df.to_csv(per_hla_file, index=False)
    print(f"✓ Per-HLA metrics: {per_hla_file}")

    summary_file = output_dir / 'mixmhcpred_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("MixMHCpred 2.2 Evaluation Summary\n")
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
        description='MixMHCpred 2.2 Evaluation'
    )
    parser.add_argument('--test_file', type=str, required=True,
                        help='Test data file (TSV, positive samples)')
    parser.add_argument('--mode1_output_dir', type=str, required=True,
                        help='Mode 1 output directory (for task graph)')
    parser.add_argument('--mode2_output_dir', type=str, default=None,
                        help='Mode 2 output directory（cache key 还原用，与训练时的 --mode2_output_dir 一致）')
    parser.add_argument('--mixmhcpred_dir', type=str, required=True,
                        help='MixMHCpred installation directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--negative_ratio', type=int, default=10,
                        help='Negative:Positive ratio (default: 10)')

    # ── 负样本来源（三选一）──
    neg_group = parser.add_mutually_exclusive_group()
    neg_group.add_argument('--use_negative_cache', action='store_true',
                           help='从 NegativeSampleCache .pkl 直接加载负样本（最快，'
                                '需与训练时用同一 data_file / negative_ratio）')
    neg_group.add_argument('--test_with_negatives', type=str, default=None,
                           help='已含负样本的 TSV 文件或目录，直接加载跳过生成')

    parser.add_argument('--cache_dir', type=str, default='data/negative_samples',
                        help='NegativeSampleCache 目录（默认 data/negative_samples）')
    parser.add_argument('--min_samples', type=int, default=10,
                        help='训练时的 min_samples，用于还原 cache key（默认 10）')
    parser.add_argument('--all_data_file', type=str, default=None,
                        help='完整数据集 TSV（生成模式下建议提供，用于全量 exclude set）')
    parser.add_argument('--proteome_file', type=str, default='data/human_proteome.fasta',
                        help='蛋白质组 FASTA 文件（生成模式下需要）')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MixMHCpred 2.2 Evaluation")
    print("=" * 80)
    # 判断负样本来源模式
    if args.use_negative_cache:
        neg_mode = f"cache  ({args.cache_dir})"
    elif args.test_with_negatives:
        neg_mode = f"prebuilt TSV  ({args.test_with_negatives})"
    else:
        neg_mode = "generate (realtime)"

    print(f"\n  Test file:       {args.test_file}")
    print(f"  Mode1 dir:       {args.mode1_output_dir}")
    print(f"  MixMHCpred dir:  {args.mixmhcpred_dir}")
    print(f"  Negative ratio:  {args.negative_ratio}")
    print(f"  Negative source: {neg_mode}")
    print(f"  Output:          {args.output_dir}")

    # ── Step 1: 加载正样本（始终需要，用于 task 定义） ──
    print(f"\n📂 Loading test data (positive samples)...")
    test_df = pd.read_csv(args.test_file, sep='\t')

    rename = {}
    if 'MHC_Restriction_Name' in test_df.columns: rename['MHC_Restriction_Name'] = 'hla'
    if 'Peptide'              in test_df.columns: rename['Peptide']               = 'peptide'
    if 'Label'                in test_df.columns: rename['Label']                 = 'label'
    test_df = test_df.rename(columns=rename)

    pos_df = test_df[test_df['label'] == 1].copy()
    print(f"✓ Positive samples: {len(pos_df):,}  |  HLAs: {pos_df['hla'].nunique()}")

    # ── Step 2: 获取带负样本的测试集 ──
    if args.use_negative_cache:
        # ── 模式 A：从 NegativeSampleCache .pkl 加载（最快）──
        print(f"\n🚀 Loading from NegativeSampleCache...")
        cache_data_file = args.mode2_output_dir if args.mode2_output_dir else args.test_file
        filtered_test_datasets, task_manager, all_tasks = load_test_datasets_from_cache(
            data_file=cache_data_file,     # 与训练时的 --mode2_output_dir 一致
            mode1_dir=args.mode1_output_dir,
            cache_dir=args.cache_dir,
            negative_ratio=args.negative_ratio,
            min_samples=args.min_samples,
        )
        if filtered_test_datasets is None:
            print("❌ 缓存加载失败，请改用 --test_with_negatives 或去掉 --use_negative_cache 实时生成。")
            return

    elif args.test_with_negatives:
        # ── 模式 B：从已有 TSV 文件/目录加载 ──
        from src.data.unified_task_creator import UnifiedTaskCreator
        from src.config.mode_config import create_mode1_config

        print(f"\n📂 Loading pre-built negatives from: {args.test_with_negatives}")
        combined_df = pd.read_csv(args.test_with_negatives, sep='\t').rename(columns=rename)
        config = create_mode1_config(negative_ratio=args.negative_ratio, min_samples=1)
        task_manager = UnifiedTaskCreator(config).create_tasks(
            combined_df[combined_df['label'] == 1]
        )
        all_tasks = task_manager.get_all_tasks()

        # 按 task 分组
        graph_wrapper = TaskGraphWrapper(Path(args.mode1_output_dir) / 'task_graph')
        valid_ids = (set(graph_wrapper.task_to_idx.keys())
                     if hasattr(graph_wrapper, 'task_to_idx')
                     else set(f"HLA_{t.hla}" for t in all_tasks.values()))

        filtered_test_datasets = {}
        for task_id, task in all_tasks.items():
            if f"HLA_{task.hla}" not in valid_ids:
                continue
            task_df = combined_df[combined_df['hla'] == task.hla].copy()
            if len(task_df):
                filtered_test_datasets[task_id] = task_df
        print(f"✓ Loaded {len(filtered_test_datasets)} tasks")

    else:
        # ── 模式 C：实时生成负样本 ──
        all_df = None
        if args.all_data_file:
            print(f"\n📂 Loading full dataset for exclude set...")
            all_df = pd.read_csv(args.all_data_file, sep='\t').rename(columns=rename)
            print(f"✓ {len(all_df):,} rows  |  Positive: {(all_df['label']==1).sum():,}")

        protein_sequences = None
        from src.data.enhanced_negative_sampler import load_protein_sequences
        proteome_path = Path(args.proteome_file)
        if proteome_path.exists():
            print(f"\n📂 Loading proteome sequences...")
            protein_sequences = load_protein_sequences(str(proteome_path))
        else:
            print(f"  ⚠️  Proteome file not found: {proteome_path}")

        filtered_test_datasets, task_manager, all_tasks = generate_test_datasets(
            pos_df, args.mode1_output_dir, args.negative_ratio,
            all_df=all_df, protein_sequences=protein_sequences
        )

    if len(filtered_test_datasets) == 0:
        print("❌ No valid tasks found.")
        return

    # 3. MixMHCpred预测
    all_predictions = run_mixmhcpred_predictions(
        filtered_test_datasets, all_tasks, args.mixmhcpred_dir
    )

    if len(all_predictions) == 0:
        print("❌ No predictions generated.")
        return

    # 4. 计算指标和生成报告
    compute_metrics_and_report(all_predictions, all_tasks, output_dir)

    print(f"\n{'='*80}")
    print("✓ MixMHCpred evaluation complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()