#!/usr/bin/env python3
"""
NetMHCpan-4.2c Evaluation Script
使用 NegativeSampleCache 缓存加载测试集，确保与 HistoNeo 训练时完全一致的负样本

用法（小鼠）：
    python scripts/evaluate_netmhcpan.py \
        --test_file        output/mouse_mode2/data_splits/test.tsv \
        --mode1_output_dir output/mouse_mode1_on_mode2_splits \
        --mode2_data_file  output/mouse_mode2/data_splits/test.tsv \
        --output_dir       results/mouse_netmhcpan_eval \
        --netmhcpan_bin    /gpfs/work/aac/yuntianhou20/HNeo/netMHCpan-4.2/netMHCpan \
        --use_negative_cache \
        --cache_dir        data/negative_samples \
        --negative_ratio   10 \
        --min_samples      1

用法（人类）：
    python scripts/evaluate_netmhcpan.py \
        --test_file        output/0227_mode2_proteome/data_splits/test.tsv \
        --mode1_output_dir output/mode1_ablation \
        --mode2_data_file  output/0227_mode2_proteome/data_splits/test.tsv \
        --output_dir       results/human_netmhcpan_eval \
        --netmhcpan_bin    /gpfs/work/aac/yuntianhou20/HNeo/netMHCpan-4.2/netMHCpan \
        --use_negative_cache \
        --cache_dir        data/negative_samples \
        --negative_ratio   10 \
        --min_samples      10
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data.negative_cache import NegativeSampleCache

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MHC2_PREFIXES = ("H2-IA", "H2-IE", "H-2IA", "H-2IE")


# ── HLA 名称 ───────────────────────────────────────────────────

# 小鼠 MHC-I allele 名称映射（HistoNeo格式 → NetMHCpan allelenames格式）
# allelenames 文件使用 H-2-Kb 格式（两个连字符）
MOUSE_ALLELE_MAP = {
    "H2-Kb":  "H-2-Kb",
    "H2-Db":  "H-2-Db",
    "H2-Kd":  "H-2-Kd",
    "H2-Dd":  "H-2-Dd",
    "H2-Kk":  "H-2-Kk",
    "H2-Kq":  "H-2-Kq",
    "H2-Dq":  "H-2-Dq",
    "H2-Ld":  "H-2-Ld",
    "H2-Lq":  "H-2-Lq",
    "H2-Q1":  "H-2-Qa1",
    "H2-Q2":  "H-2-Qa2",
    "H2-Qa1": "H-2-Qa1",
    "H2-Qa2": "H-2-Qa2",
}

def to_netmhcpan_allele(hla: str):
    """
    H2-Kb  → H-2Kb   (查映射表)
    H2-IA* → None    (MHC-II，不支持)
    HLA-A*02:01 → HLA-A02:01
    """
    if any(hla.startswith(p) for p in MHC2_PREFIXES):
        return None
    if hla in MOUSE_ALLELE_MAP:
        return MOUSE_ALLELE_MAP[hla]
    if hla.startswith("HLA-"):
        return hla.replace("*", "")
    # H2- 前缀通用转换: H2-Xx → H-2Xx
    if hla.startswith("H2-"):
        return "H-2" + hla[3:]
    return hla


# ── 缓存加载（与 evaluate_mixmhcpred.py 完全一致）───────────────

def load_test_from_cache(mode2_data_file, mode1_output_dir,
                         cache_dir, negative_ratio, min_samples):
    cache_config = {
        'negative_ratio':            negative_ratio,
        'use_tissue_aware_negatives': False,
        'min_samples':               min_samples,
        'tissue_source':             'Host',
    }
    cache = NegativeSampleCache(cache_dir)

    if not cache.exists(mode2_data_file, 'mode1', cache_config):
        print(f"\n  ❌ 缓存不存在，请检查以下参数是否与训练时一致:")
        print(f"       mode2_data_file: {mode2_data_file}")
        print(f"       cache_dir:       {cache_dir}")
        print(f"       negative_ratio:  {negative_ratio}")
        print(f"       min_samples:     {min_samples}")
        cache_path = Path(cache_dir)
        if cache_path.exists():
            pkls = list(cache_path.glob("*_test.pkl"))
            if pkls:
                print(f"\n  现有缓存文件（参考）:")
                for p in sorted(pkls)[:8]:
                    print(f"    {p.name}")
        return None

    result = cache.load(mode2_data_file, 'mode1', cache_config)
    if result is None:
        print("  ❌ 缓存加载失败")
        return None

    _, _, test_task_datasets = result
    total_pos = sum((df['label'] == 1).sum() for df in test_task_datasets.values())
    total_neg = sum((df['label'] == 0).sum() for df in test_task_datasets.values())
    print(f"  ✓ 缓存加载: {len(test_task_datasets)} tasks  "
          f"(pos={total_pos:,}, neg={total_neg:,})")
    return test_task_datasets


# ── NetMHCpan 调用 ──────────────────────────────────────────────

def run_netmhcpan(peptides, hla_nm, netmhcpan_bin, tmp_dir, batch_size=500):
    if not peptides:
        return pd.DataFrame()

    all_records = []
    for i in range(0, len(peptides), batch_size):
        batch = peptides[i: i + batch_size]
        pep_file = tmp_dir / f"batch_{i}.txt"
        pep_file.write_text("\n".join(batch) + "\n")

        cmd = [netmhcpan_bin, "-a", hla_nm, "-p", "-f", str(pep_file), "-BA"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            continue
        except FileNotFoundError:
            print(f"  ❌ NetMHCpan 不存在: {netmhcpan_bin}")
            sys.exit(1)

        if res.returncode == 0:
            all_records.extend(parse_output(res.stdout, batch))

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.sort_values("pred_score", ascending=False)
    df = df.drop_duplicates(subset=["peptide"], keep="first")
    return df


def parse_output(output, peptides_input):
    records = []
    header_found = False
    col_map = {}

    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if ("Peptide" in line or "peptide" in line) and \
           any(k in line for k in ["EL-score", "Score_EL", "EL_score", "Score"]):
            tokens = line.split()
            header_found = True
            col_map = {t: i for i, t in enumerate(tokens)}
            col_map.update({t.lower(): i for i, t in enumerate(tokens)})
            continue

        if not header_found:
            continue
        if line.startswith(("---", "===", "Protein")):
            continue

        tokens = line.split()
        if len(tokens) < 4:
            continue

        pep_idx = col_map.get("Peptide", col_map.get("peptide", 2))
        try:
            peptide = tokens[pep_idx].upper()
        except IndexError:
            continue

        if not (8 <= len(peptide) <= 15) or not all(a in VALID_AA for a in peptide):
            continue

        el_score = None
        for key in ["Score_EL", "EL-score", "EL_score", "score_el"]:
            idx = col_map.get(key, col_map.get(key.lower()))
            if idx is not None and idx < len(tokens):
                try:
                    el_score = float(tokens[idx]); break
                except ValueError:
                    continue

        el_rank = None
        for key in ["EL_Rank", "%Rank_EL", "EL-rank"]:
            idx = col_map.get(key, col_map.get(key.lower()))
            if idx is not None and idx < len(tokens):
                try:
                    el_rank = float(tokens[idx]); break
                except ValueError:
                    continue

        if el_score is not None:
            pred_score = el_score
        elif el_rank is not None:
            pred_score = max(0.0, 1.0 - el_rank / 100.0)
        else:
            pred_score = 0.0

        records.append({"peptide": peptide, "el_score": el_score,
                        "el_rank": el_rank, "pred_score": pred_score})

    # fallback
    if not records:
        pep_set = {p.upper() for p in peptides_input}
        for line in output.split("\n"):
            tokens = line.split()
            for i, tok in enumerate(tokens):
                if tok.upper() in pep_set:
                    nums = []
                    for t in tokens[i + 1:]:
                        try: nums.append(float(t))
                        except ValueError: break
                    score = nums[0] if nums else 0.0
                    records.append({"peptide": tok.upper(), "el_score": score,
                                    "el_rank": None, "pred_score": score})
                    break
    return records


# ── 主评估 ──────────────────────────────────────────────────────

def evaluate_netmhcpan(test_file, mode1_output_dir, mode2_data_file,
                       output_dir, netmhcpan_bin,
                       cache_dir='data/negative_samples',
                       negative_ratio=10, min_samples=10, batch_size=500):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载缓存测试集
    print(f"\n📂 Loading test datasets from NegativeSampleCache...")
    test_task_datasets = load_test_from_cache(
        mode2_data_file, mode1_output_dir, cache_dir, negative_ratio, min_samples)
    if test_task_datasets is None:
        return pd.DataFrame()

    # 2. 按 HLA 分组（始终从 df['hla'] 列取，不用 task_id 字符串）
    hla_task_map = {}
    for task_id, df in test_task_datasets.items():
        if 'hla' in df.columns and len(df) > 0:
            hla = df['hla'].iloc[0]
        else:
            # fallback: 从 task_id 里去掉 HLA_ 前缀
            hla = str(task_id).replace('HLA_', '')
        hla_task_map.setdefault(hla, []).append(task_id)

    supported = [h for h in hla_task_map if to_netmhcpan_allele(h) is not None]
    skipped   = [h for h in hla_task_map if to_netmhcpan_allele(h) is None]
    print(f"\n  MHC-I (supported): {len(supported)}  |  MHC-II (skipped): {len(skipped)}")
    if skipped:
        print(f"  Skipped: {skipped}")

    # 3. 评估
    task_records = []
    tmp_dir = output_dir / "tmp_netmhcpan"
    tmp_dir.mkdir(exist_ok=True)

    for hla in skipped:
        for task_id in hla_task_map[hla]:
            df = test_task_datasets[task_id]
            task_records.append({
                'hla': hla, 'task_id': task_id,
                'hla_netmhcpan': 'unsupported_mhc2',
                'n_samples': len(df), 'n_positive': int((df['label']==1).sum()),
                'auroc': np.nan, 'auprc': np.nan,
                'precision': np.nan, 'recall': np.nan, 'f1': np.nan,
            })

    for hla in tqdm(supported, desc="Evaluating HLA alleles"):
        hla_nm = to_netmhcpan_allele(hla)
        for task_id in hla_task_map[hla]:
            eval_df = test_task_datasets[task_id].copy()
            n_pos   = int((eval_df['label']==1).sum())
            eval_df = eval_df[
                eval_df['peptide'].str.len().between(8,15) &
                eval_df['peptide'].apply(lambda p: all(a in VALID_AA for a in str(p).upper()))
            ].copy()
            eval_df['peptide'] = eval_df['peptide'].str.upper()

            pred_df = run_netmhcpan(eval_df['peptide'].tolist(),
                                    hla_nm, netmhcpan_bin, tmp_dir, batch_size)

            if pred_df.empty:
                print(f"  ⚠ {hla} ({task_id}): 无输出，跳过")
                task_records.append({
                    'hla': hla, 'task_id': task_id, 'hla_netmhcpan': hla_nm,
                    'n_samples': len(eval_df), 'n_positive': n_pos,
                    'auroc': np.nan, 'auprc': np.nan,
                    'precision': np.nan, 'recall': np.nan, 'f1': np.nan,
                })
                continue

            eval_df = eval_df.merge(pred_df[['peptide','pred_score']], on='peptide', how='left')
            eval_df['pred_score'] = eval_df['pred_score'].fillna(0.0)
            y_true = eval_df['label'].values
            y_pred = eval_df['pred_score'].values

            auroc = auprc = precision = recall = f1 = np.nan
            try:
                if len(np.unique(y_true)) > 1:
                    auroc     = roc_auc_score(y_true, y_pred)
                    auprc     = average_precision_score(y_true, y_pred)
                    y_bin     = (y_pred >= 0.5).astype(int)
                    precision = precision_score(y_true, y_bin, zero_division=0)
                    recall    = recall_score(y_true, y_bin, zero_division=0)
                    f1        = f1_score(y_true, y_bin, zero_division=0)
            except Exception as e:
                print(f"  ⚠ {hla}: 指标计算失败 — {e}")

            task_records.append({
                'hla': hla, 'task_id': task_id, 'hla_netmhcpan': hla_nm,
                'n_samples': len(eval_df), 'n_positive': n_pos,
                'auroc': auroc, 'auprc': auprc,
                'precision': precision, 'recall': recall, 'f1': f1,
            })

    df = pd.DataFrame(task_records)

    # 保存 task 级别原始结果（含每个 HLA×Tissue 任务的 AUROC）
    task_file = output_dir / 'netmhcpan_per_task.csv'
    df.to_csv(task_file, index=False)
    print(f'✓ Per-task results: {task_file}')

    # 4a. 补充 tissue 列（从缓存数据里取）
    tissue_map = {}
    for task_id, tdf in test_task_datasets.items():
        if 'tissue' in tdf.columns and len(tdf) > 0:
            tissue_map[task_id] = tdf['tissue'].iloc[0]
    df['tissue'] = df['task_id'].map(tissue_map).fillna('unknown')

    # 4b. Per-tissue 汇总（与 evaluate_mode2_simple.py 完全一致）
    valid_df = df[df['auroc'].notna()].copy()
    if len(valid_df) > 0 and 'tissue' in valid_df.columns:
        tissue_summary = valid_df.groupby('tissue').apply(
            lambda x: pd.Series({
                'n_tasks':    len(x),
                'n_positive': x['n_positive'].sum(),
                'mean_auroc': np.average(x['auroc'], weights=x['n_positive']),
                'mean_auprc': np.average(x['auprc'].fillna(0), weights=x['n_positive']),
            })
        ).reset_index().sort_values('mean_auroc', ascending=False)

        print("\n" + "="*80)
        print("🧬 PER-TISSUE SUMMARY (NetMHCpan, MHC-I only)")
        print("="*80)
        print(f"\n{'Tissue':<30} {'N Tasks':<10} {'Mean AUROC':<12} {'Mean AUPRC':<12}")
        print("  " + "-"*65)
        for _, row in tissue_summary.iterrows():
            print(f"  {row['tissue']:<30} {int(row['n_tasks']):<10} "
                  f"{row['mean_auroc']:<12.4f} {row['mean_auprc']:<12.4f}")

        tissue_file = output_dir / 'netmhcpan_per_tissue.csv'
        tissue_summary.to_csv(tissue_file, index=False)
        print(f"\n✓ Per-tissue summary: {tissue_file}")

    # 4. 按 HLA 聚合（只对有有效 AUROC 的 task 做 mean）
    valid_tasks = df[df['auroc'].notna()]
    print(f'\n  Task-level: {len(df)} total, {len(valid_tasks)} with valid AUROC')
    if len(valid_tasks) > 0:
        print(f'  Valid task HLAs: {valid_tasks["hla"].unique().tolist()}')

    agg_valid = valid_tasks.groupby('hla').agg(
        auroc = ('auroc', 'mean'),
        auprc = ('auprc', 'mean'),
        f1    = ('f1',    'mean'),
    ).reset_index()

    agg_size = df.groupby('hla').agg(
        n_samples  = ('n_samples',  'sum'),
        n_positive = ('n_positive', 'sum'),
    ).reset_index()

    agg = agg_size.merge(agg_valid, on='hla', how='left')

    valid_aurocs = agg['auroc'].dropna()
    valid_auprcs = agg['auprc'].dropna()

    # 5. 打印统计
    print("\n" + "="*80)
    print("📊 OVERALL STATISTICS")
    print("="*80)
    print(f"\n  Total alleles:    {len(agg)}")
    print(f"  Valid alleles:    {len(valid_aurocs)}")
    print(f"  Skipped (MHC-II):{len(skipped)}")
    if len(valid_aurocs) > 0:
        print(f"  Mean AUROC:       {valid_aurocs.mean():.4f} ± {valid_aurocs.std():.4f}")
        print(f"  Median AUROC:     {valid_aurocs.median():.4f}")
        print(f"  Min / Max AUROC:  {valid_aurocs.min():.4f} / {valid_aurocs.max():.4f}")
    if len(valid_auprcs) > 0:
        print(f"  Mean AUPRC:       {valid_auprcs.mean():.4f} ± {valid_auprcs.std():.4f}")

    # 样本量分层
    print("\n" + "="*80)
    print("📈 PERFORMANCE BY SAMPLE SIZE")
    print("="*80)
    bins   = [0, 50, 100, 500, 1000, float('inf')]
    labels = ['<50','50-100','100-500','500-1K','>1K']
    agg['sample_bin'] = pd.cut(agg['n_samples'], bins=bins, labels=labels)
    print(f"\n  {'Size Range':<15} {'N Alleles':<12} {'Mean AUROC':<12} {'Mean AUPRC':<12}")
    print("  " + "-"*52)
    for b in labels:
        bd = agg[agg['sample_bin']==b]
        if len(bd) > 0:
            va = bd['auroc'].dropna(); vp = bd['auprc'].dropna()
            auroc_s = f"{va.mean():.4f}" if len(va) > 0 else "N/A"
            auprc_s = f"{vp.mean():.4f}" if len(vp) > 0 else "N/A"
            print(f"  {b:<15} {len(bd):<12} {auroc_s:<12} {auprc_s:<12}")

    # Top 20
    print("\n" + "="*80)
    print("🏆 TOP 20 ALLELES BY AUROC")
    print("="*80)
    top20 = agg[agg['auroc'].notna()].sort_values('auroc', ascending=False).head(20)
    print(f"\n  {'Rank':<6} {'HLA':<20} {'AUROC':<10} {'AUPRC':<10} {'N Samples':<10}")
    print("  " + "-"*58)
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        auroc_s = f"{row['auroc']:.4f}" if pd.notna(row['auroc']) else "N/A"
        auprc_s = f"{row['auprc']:.4f}" if pd.notna(row['auprc']) else "N/A"
        print(f"  {rank:<6} {row['hla']:<20} {auroc_s:<10} {auprc_s:<10} {int(row['n_samples']):<10}")
    print("\n" + "="*80)

    # 6. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    fig.suptitle('NetMHCpan-4.2c Evaluation Results', fontsize=16, fontweight='bold')

    ax = axes[0,0]
    if len(valid_aurocs) > 0:
        ax.hist(valid_aurocs, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(valid_aurocs.mean(),   color='red',   linestyle='--', linewidth=2,
                   label=f'Mean: {valid_aurocs.mean():.3f}')
        ax.axvline(valid_aurocs.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {valid_aurocs.median():.3f}')
        ax.set_xlabel('AUROC'); ax.set_ylabel('Number of Alleles')
        ax.set_title('AUROC Distribution (per HLA Task)', fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)

    ax = axes[0,1]
    top15 = agg[agg['auroc'].notna()].sort_values('auroc', ascending=True).tail(15)
    if len(top15) > 0:
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top15)))
        bars = ax.barh(range(len(top15)), top15['auroc'],
                       color=colors, alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15['hla'], fontsize=9)
        for bar in bars:
            w = bar.get_width()
            ax.text(w, bar.get_y()+bar.get_height()/2, f' {w:.3f}', va='center', fontsize=8)
        ax.set_xlabel('AUROC')
        ax.set_title('Top 15 HLAs by AUROC', fontweight='bold')
        ax.set_xlim(0,1); ax.grid(axis='x', alpha=0.3)

    ax = axes[1,0]
    bin_means, bin_labels_plot = [], []
    for b in labels:
        bd = agg[agg['sample_bin']==b]; va = bd['auroc'].dropna()
        if len(va) > 0:
            bin_means.append(va.mean())
            bin_labels_plot.append(f"{b}\n(n={len(bd)})")
    if bin_means:
        bars = ax.bar(range(len(bin_means)), bin_means,
                      color='#2ecc71', alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, bin_means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(bin_means)))
        ax.set_xticklabels(bin_labels_plot)
        ax.set_ylabel('Mean AUROC'); ax.set_ylim(0,1.05)
        ax.set_title('Performance by Sample Size', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    ax = axes[1,1]
    vdf = agg[agg['auroc'].notna()]
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
        ax.set_xlabel('Number of Samples (log scale)')
        ax.set_ylabel('AUROC')
        ax.set_title('Sample Size vs AUROC', fontweight='bold')
        ax.set_xscale('log'); ax.set_ylim(0,1); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / 'netmhcpan_evaluation.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Visualization saved: {fig_path}")

    # 7. 保存
    per_hla_file = output_dir / 'netmhcpan_per_hla.csv'
    agg.to_csv(per_hla_file, index=False)
    print(f"✓ Per-HLA metrics: {per_hla_file}")

    summary_file = output_dir / 'netmhcpan_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("NetMHCpan-4.2c Evaluation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total alleles:    {len(agg)}\n")
        f.write(f"Valid alleles:    {len(valid_aurocs)}\n")
        f.write(f"Skipped (MHC-II): {len(skipped)}\n\n")
        if len(valid_aurocs) > 0:
            f.write(f"Mean AUROC:    {valid_aurocs.mean():.4f} ± {valid_aurocs.std():.4f}\n")
            f.write(f"Median AUROC:  {valid_aurocs.median():.4f}\n")
            f.write(f"Min AUROC:     {valid_aurocs.min():.4f}\n")
            f.write(f"Max AUROC:     {valid_aurocs.max():.4f}\n")
        if len(valid_auprcs) > 0:
            f.write(f"Mean AUPRC:    {valid_auprcs.mean():.4f} ± {valid_auprcs.std():.4f}\n")
    print(f"✓ Summary:        {summary_file}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return agg


# ── main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='NetMHCpan-4.2c Evaluation (fair comparison with HistoNeo)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--test_file',           required=True)
    parser.add_argument('--mode1_output_dir',    required=True)
    parser.add_argument('--mode2_data_file',     required=True,
                        help='Mode 2 训练时的 --data_file，用于还原 cache key')
    parser.add_argument('--output_dir',          required=True)
    parser.add_argument('--netmhcpan_bin',       default='netMHCpan')
    parser.add_argument('--use_negative_cache',  action='store_true')
    parser.add_argument('--cache_dir',           default='data/negative_samples')
    parser.add_argument('--negative_ratio',      type=int, default=10)
    parser.add_argument('--min_samples',         type=int, default=10,
                        help='训练时的 min_samples（影响 cache key）')
    parser.add_argument('--batch_size',          type=int, default=500)
    args = parser.parse_args()

    print("="*80)
    print("NetMHCpan-4.2c Evaluation (Fair Comparison Mode)")
    print("="*80)
    print(f"\n  test_file:        {args.test_file}")
    print(f"  mode1_output_dir: {args.mode1_output_dir}")
    print(f"  mode2_data_file:  {args.mode2_data_file}")
    print(f"  netmhcpan_bin:    {args.netmhcpan_bin}")
    print(f"  cache_dir:        {args.cache_dir}")
    print(f"  negative_ratio:   {args.negative_ratio}")
    print(f"  min_samples:      {args.min_samples}")

    if not args.use_negative_cache:
        print("\n⚠ 未加 --use_negative_cache，无法保证与 HistoNeo 使用相同负样本集")

    evaluate_netmhcpan(
        test_file=args.test_file,
        mode1_output_dir=args.mode1_output_dir,
        mode2_data_file=args.mode2_data_file,
        output_dir=Path(args.output_dir),
        netmhcpan_bin=args.netmhcpan_bin,
        cache_dir=args.cache_dir,
        negative_ratio=args.negative_ratio,
        min_samples=args.min_samples,
        batch_size=args.batch_size,
    )
    print("\n✓ 评估完成")


if __name__ == "__main__":
    main()