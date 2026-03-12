#!/usr/bin/env python3
"""
Per-Task结果可视化脚本

用法:
    python scripts/visualize_per_task_results.py output/mode2_standard
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
import numpy as np

def load_per_task_results(output_dir):
    """加载per-task结果"""
    output_dir = Path(output_dir)
    
    # 加载CSV
    csv_file = output_dir / 'per_task_metrics_test.csv'
    if not csv_file.exists():
        print(f"Error: {csv_file} not found!")
        return None, None
    
    df = pd.read_csv(csv_file)
    
    # 加载summary
    summary_file = output_dir / 'per_task_summary_test.json'
    summary = None
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    
    return df, summary


def plot_auroc_distribution(df, output_dir):
    """绘制AUROC分布直方图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 过滤有效值
    auroc_valid = df['auroc'].dropna()
    
    ax.hist(auroc_valid, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(auroc_valid.mean(), color='red', linestyle='--', 
               label=f'Mean: {auroc_valid.mean():.3f}')
    ax.axvline(auroc_valid.median(), color='green', linestyle='--',
               label=f'Median: {auroc_valid.median():.3f}')
    
    ax.set_xlabel('AUROC', fontsize=12)
    ax.set_ylabel('Number of Tasks', fontsize=12)
    ax.set_title('Distribution of Per-Task AUROC', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'auroc_distribution.png', dpi=300)
    print(f"✓ Saved: {output_dir / 'auroc_distribution.png'}")
    plt.close()


def plot_performance_by_hla(df, output_dir, top_n=20):
    """绘制HLA性能对比"""
    if 'hla' not in df.columns:
        return
    
    # 按HLA统计
    hla_stats = df.groupby('hla').agg({
        'auroc': 'mean',
        'task_id': 'count',
        'n_samples': 'sum'
    }).reset_index()
    hla_stats.columns = ['hla', 'mean_auroc', 'n_tasks', 'total_samples']
    
    # 只显示有≥3个tasks的HLA
    hla_stats = hla_stats[hla_stats['n_tasks'] >= 3]
    
    # 排序并取top N
    hla_stats = hla_stats.sort_values('mean_auroc', ascending=False).head(top_n)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(hla_stats['hla'], hla_stats['mean_auroc'])
    
    # 颜色编码（根据AUROC）
    colors = plt.cm.RdYlGn(hla_stats['mean_auroc'])
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Mean AUROC', fontsize=12)
    ax.set_ylabel('HLA Allele', fontsize=12)
    ax.set_title(f'Top {top_n} HLA Alleles by Mean AUROC', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(True, axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (auroc, n_tasks) in enumerate(zip(hla_stats['mean_auroc'], hla_stats['n_tasks'])):
        ax.text(auroc + 0.01, i, f'{auroc:.3f} ({n_tasks} tasks)', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_hla.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'performance_by_hla.png'}")
    plt.close()


def plot_performance_by_tissue(df, output_dir):
    """绘制Tissue性能对比"""
    if 'tissue' not in df.columns:
        return
    
    # 按Tissue统计
    tissue_stats = df.groupby('tissue').agg({
        'auroc': 'mean',
        'task_id': 'count',
        'n_samples': 'sum'
    }).reset_index()
    tissue_stats.columns = ['tissue', 'mean_auroc', 'n_tasks', 'total_samples']
    
    # 排序
    tissue_stats = tissue_stats.sort_values('mean_auroc', ascending=False)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(tissue_stats['tissue'], tissue_stats['mean_auroc'])
    
    # 颜色编码
    colors = plt.cm.RdYlGn(tissue_stats['mean_auroc'])
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Mean AUROC', fontsize=12)
    ax.set_ylabel('Tissue Type', fontsize=12)
    ax.set_title('Performance by Tissue Type', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(True, axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (auroc, n_tasks) in enumerate(zip(tissue_stats['mean_auroc'], tissue_stats['n_tasks'])):
        ax.text(auroc + 0.01, i, f'{auroc:.3f} ({n_tasks} tasks)', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_tissue.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'performance_by_tissue.png'}")
    plt.close()


def plot_auroc_vs_sample_size(df, output_dir):
    """绘制AUROC vs 样本数散点图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 过滤有效值
    df_valid = df[~df['auroc'].isna()]
    
    scatter = ax.scatter(df_valid['n_samples'], df_valid['auroc'], 
                        alpha=0.6, c=df_valid['auroc'], 
                        cmap='RdYlGn', s=50)
    
    # 添加趋势线
    z = np.polyfit(df_valid['n_samples'], df_valid['auroc'], 1)
    p = np.poly1d(z)
    ax.plot(df_valid['n_samples'].sort_values(), 
            p(df_valid['n_samples'].sort_values()),
            "r--", alpha=0.8, label=f'Trend: y={z[0]:.6f}x+{z[1]:.3f}')
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('AUROC vs Sample Size', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='AUROC')
    plt.tight_layout()
    plt.savefig(output_dir / 'auroc_vs_sample_size.png', dpi=300)
    print(f"✓ Saved: {output_dir / 'auroc_vs_sample_size.png'}")
    plt.close()


def plot_metrics_comparison(df, output_dir):
    """绘制多指标对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['auroc', 'accuracy', 'f1', 'auprc']
    titles = ['AUROC', 'Accuracy', 'F1-Score', 'AUPRC']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        valid_data = df[metric].dropna()
        
        ax.hist(valid_data, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(valid_data.mean(), color='red', linestyle='--',
                   label=f'Mean: {valid_data.mean():.3f}')
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Number of Tasks', fontsize=11)
        ax.set_title(f'Distribution of {title}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300)
    print(f"✓ Saved: {output_dir / 'metrics_comparison.png'}")
    plt.close()


def generate_html_report(df, summary, output_dir):
    """生成HTML报告"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Per-Task Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; background: #e8f5e9; border-radius: 5px; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; }}
            .good {{ color: #4CAF50; font-weight: bold; }}
            .medium {{ color: #FF9800; font-weight: bold; }}
            .bad {{ color: #f44336; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Per-Task Performance Report</h1>
            
            <h2>Overall Statistics</h2>
            <div>
                <div class="metric">
                    <div class="metric-label">Total Tasks</div>
                    <div class="metric-value">{summary['overall']['n_tasks']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mean AUROC</div>
                    <div class="metric-value">{summary['overall']['mean_auroc']:.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Median AUROC</div>
                    <div class="metric-value">{summary['overall']['median_auroc']:.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mean Accuracy</div>
                    <div class="metric-value">{summary['overall']['mean_accuracy']:.4f}</div>
                </div>
            </div>
            
            <h2>Visualizations</h2>
            <img src="auroc_distribution.png" alt="AUROC Distribution">
            <img src="performance_by_hla.png" alt="Performance by HLA">
    """
    
    if 'tissue' in df.columns:
        html += '<img src="performance_by_tissue.png" alt="Performance by Tissue">\n'
    
    html += """
            <img src="auroc_vs_sample_size.png" alt="AUROC vs Sample Size">
            <img src="metrics_comparison.png" alt="Metrics Comparison">
            
            <h2>Top 20 Tasks</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Task ID</th>
                    <th>HLA</th>
    """
    
    if 'tissue' in df.columns:
        html += '<th>Tissue</th>\n'
    
    html += """
                    <th>AUROC</th>
                    <th>Accuracy</th>
                    <th>F1</th>
                    <th>Samples</th>
                </tr>
    """
    
    # Top 20 tasks
    top_20 = df.head(20)
    for idx, (i, row) in enumerate(top_20.iterrows(), 1):
        auroc_class = 'good' if row['auroc'] >= 0.9 else ('medium' if row['auroc'] >= 0.7 else 'bad')
        html += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{row['task_id']}</td>
                    <td>{row['hla']}</td>
        """
        if 'tissue' in row:
            html += f"<td>{row['tissue']}</td>\n"
        
        html += f"""
                    <td class="{auroc_class}">{row['auroc']:.4f}</td>
                    <td>{row['accuracy']:.4f}</td>
                    <td>{row['f1']:.4f}</td>
                    <td>{int(row['n_samples'])}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML
    html_file = output_dir / 'per_task_report.html'
    with open(html_file, 'w') as f:
        f.write(html)
    
    print(f"✓ HTML report saved: {html_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_per_task_results.py output/mode2_standard")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist!")
        sys.exit(1)
    
    print("="*80)
    print("Visualizing Per-Task Results")
    print("="*80)
    
    # 加载结果
    df, summary = load_per_task_results(output_dir)
    if df is None:
        sys.exit(1)
    
    print(f"\nLoaded {len(df)} tasks from {output_dir}")
    
    # 生成可视化
    print("\nGenerating visualizations...")
    plot_auroc_distribution(df, output_dir)
    plot_performance_by_hla(df, output_dir)
    plot_performance_by_tissue(df, output_dir)
    plot_auroc_vs_sample_size(df, output_dir)
    plot_metrics_comparison(df, output_dir)
    
    # 生成HTML报告
    if summary:
        print("\nGenerating HTML report...")
        generate_html_report(df, summary, output_dir)
    
    print("\n" + "="*80)
    print("✓ All visualizations completed!")
    print(f"✓ Check results in: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
