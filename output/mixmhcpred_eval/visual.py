import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# -------------------------- 1. 全局设置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文适配
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
plt.rcParams['figure.facecolor'] = 'white'  # 白色背景
plt.rcParams['savefig.dpi'] = 300  # 高清保存
plt.rcParams['figure.dpi'] = 100

# 双模型配置（100%匹配你的真实列名+简化路径）
model_config = {
    'baseline': {
        'name': 'MixMHCpred 2.2 (Baseline)',
        'color': '#2E86AB',  # 蓝色
        'file_path': r"mixmhcpred_per_hla.csv",  # 简化路径
        'sep': ',',  # 逗号分隔符
        'cols': {  # 匹配Baseline真实列名
            'task_id': 'task_id',
            'hla': 'hla',
            'n_samples': 'n_samples',
            'n_positive': 'n_positive',
            'auroc': 'auroc',
            'auprc': 'auprc'
        }
    },
    'histoneo': {
        'name': 'HistoNeo',
        'color': '#C73E1D',  # 红色
        'file_path': r"per_task_metrics_test.csv",  # 简化路径
        'sep': ',',  # 逗号分隔符
        'cols': {  # 匹配HistoNeo真实列名（关键修正！）
            'task_id': 'task_id',
            'hla': 'hla',
            'n_samples': 'n_samples',
            'n_positive': 'n_positive',
            'auroc': 'auroc',
            'auprc': 'auprc'
        }
    }
}


# -------------------------- 2. 数据加载（适配单列表头+分隔符错误） --------------------------
def load_fixed_data(config):
    """
    修复加载逻辑：
    1. 强制按指定分隔符重新分割单列数据
    2. 处理表头和数据中的引号/空格
    3. 100%确保列名匹配
    """
    print(f"📄 正在读取 {config['name']} 文件：{config['file_path']}")

    # 第一步：读取原始数据（可能是单列）
    try:
        # 先按字符串读取所有行
        with open(config['file_path'], 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        # 手动分割表头和数据
        header_line = lines[0].replace('"', '').strip()  # 移除引号
        header = [col.strip() for col in header_line.split(config['sep'])]  # 按分隔符分割

        # 读取数据行
        data = []
        for line in lines[1:]:
            line = line.replace('"', '').strip()  # 移除引号
            row = [val.strip() for val in line.split(config['sep'])]
            # 确保行长度与表头一致
            if len(row) == len(header):
                data.append(row)

        # 构建DataFrame
        df = pd.DataFrame(data, columns=header)
        print(f"✅ 成功解析表头：{df.columns.tolist()}")

    except Exception as e:
        raise RuntimeError(f"读取/解析文件失败：{str(e)}") from e

    # 第二步：验证核心列是否存在
    required_cols = list(config['cols'].values())
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 当前文件表头：{df.columns.tolist()}")
        raise ValueError(
            f"缺失核心列：{missing_cols}\n"
            f"💡 请修改model_config['{config['name'].split()[0].lower()}']['cols']中的值为上述表头中的真实列名"
        )

    # 第三步：只保留需要的列
    df = df[required_cols].copy()

    # 第四步：数值列转换
    num_cols = ['n_samples', 'n_positive', 'auroc', 'auprc']
    for std_col in num_cols:
        real_col = config['cols'][std_col]
        df[real_col] = pd.to_numeric(df[real_col], errors='coerce')

    # 第五步：过滤有效数据
    df = df[
        (df[config['cols']['hla']].notna()) & (df[config['cols']['hla']] != '') &
        (df[config['cols']['auroc']].notna()) & (df[config['cols']['auroc']] >= 0) & (
                    df[config['cols']['auroc']] <= 1) &
        (df[config['cols']['n_samples']].notna()) & (df[config['cols']['n_samples']] > 0)
        ].copy()

    # 第六步：重命名为标准列名
    df.rename(columns={v: k for k, v in config['cols'].items()}, inplace=True)

    # 第七步：特征工程
    df['hla_type'] = df['hla'].apply(
        lambda x: x.split('*')[0].replace('HLA-', '') if '*' in x else 'Unknown'
    )
    df = df[df['hla_type'].isin(['A', 'B', 'C'])]

    # 样本量分组
    df['sample_bin'] = pd.cut(
        df['n_samples'],
        bins=[0, 50, 100, 500, np.inf],
        labels=['<50', '50-100', '100-500', '>500'],
        right=False
    )

    print(f"✅ {config['name']} 加载完成：{len(df)} 个有效任务")
    return df


# -------------------------- 3. 加载双模型数据 --------------------------
try:
    # 加载Baseline
    df_baseline = load_fixed_data(model_config['baseline'])
    df_baseline['model'] = model_config['baseline']['name']

    # 加载HistoNeo
    df_histo = load_fixed_data(model_config['histoneo'])
    df_histo['model'] = model_config['histoneo']['name']

except ValueError as e:
    print(f"\n❌ 数据加载错误：{str(e)}")
    exit(1)
except Exception as e:
    print(f"\n❌ 未知错误：{str(e)}")
    exit(1)

# 合并数据
df_combined = pd.concat([df_baseline, df_histo], ignore_index=True)


# 计算核心指标
def calc_metrics(df, model_name):
    return {
        'model': model_name,
        'mean_auroc': df['auroc'].mean(),
        'median_auroc': df['auroc'].median(),
        'min_auroc': df['auroc'].min(),
        'max_auroc': df['auroc'].max(),
        'mean_auprc': df['auprc'].mean()
    }


df_overall = pd.DataFrame([
    calc_metrics(df_baseline, model_config['baseline']['name']),
    calc_metrics(df_histo, model_config['histoneo']['name'])
])

# -------------------------- 4. 绘制对比图 --------------------------
fig = plt.figure(figsize=(18, 12))
gs = plt.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
fig.suptitle('MixMHCpred 2.2 vs HistoNeo 模型性能对比', fontsize=22, fontweight='bold', y=0.98)

# 子图1：整体核心指标对比
ax1 = fig.add_subplot(gs[0, :])
metrics = ['mean_auroc', 'median_auroc', 'min_auroc', 'max_auroc']
labels = ['AUROC均值', 'AUROC中位数', 'AUROC最小值', 'AUROC最大值']
x = np.arange(len(metrics))
width = 0.35

# Baseline柱子
base_vals = df_overall[df_overall['model'] == model_config['baseline']['name']][metrics].values[0]
bars1 = ax1.bar(x - width / 2, base_vals, width, label=model_config['baseline']['name'],
                color=model_config['baseline']['color'], alpha=0.8, edgecolor='black')

# HistoNeo柱子
histo_vals = df_overall[df_overall['model'] == model_config['histoneo']['name']][metrics].values[0]
bars2 = ax1.bar(x + width / 2, histo_vals, width, label=model_config['histoneo']['name'],
                color=model_config['histoneo']['color'], alpha=0.8, edgecolor='black')

# 数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f'{h:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_title('整体核心评估指标对比', fontsize=18, fontweight='bold', pad=20)
ax1.set_xlabel('评估指标', fontsize=14)
ax1.set_ylabel('指标值', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12)
ax1.legend(fontsize=12, loc='lower right')
ax1.set_ylim(0.6, 1.05)
ax1.grid(axis='y', alpha=0.3)

# 子图2：样本量分组AUROC对比
ax2 = fig.add_subplot(gs[1, 0])
sample_group = df_combined.groupby(['sample_bin', 'model'], observed=True)['auroc'].mean().reset_index()
sample_group['sample_bin'] = pd.Categorical(sample_group['sample_bin'],
                                            categories=['<50', '50-100', '100-500', '>500'], ordered=True)
sample_group = sample_group.sort_values('sample_bin')

sns.barplot(x='sample_bin', y='auroc', hue='model', data=sample_group, ax=ax2,
            palette=[model_config['baseline']['color'], model_config['histoneo']['color']],
            edgecolor='black', linewidth=1)
ax2.set_title('不同样本量分组AUROC均值对比', fontsize=16, fontweight='bold')
ax2.set_xlabel('样本量分组', fontsize=12)
ax2.set_ylabel('AUROC均值', fontsize=12)
ax2.legend(fontsize=10)
ax2.set_ylim(0.9, 1.02)
ax2.grid(axis='y', alpha=0.3)

# 子图3：HLA分型AUROC分布对比
ax3 = fig.add_subplot(gs[1, 1])
sns.boxplot(x='hla_type', y='auroc', hue='model', data=df_combined, ax=ax3,
            palette=[model_config['baseline']['color'], model_config['histoneo']['color']],
            linewidth=2, fliersize=4)
ax3.set_title('HLA-A/B/C分型AUROC分布对比', fontsize=16, fontweight='bold')
ax3.set_xlabel('HLA分型', fontsize=12)
ax3.set_ylabel('AUROC值', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# 子图4：小样本量（<500）AUROC对比
ax4 = fig.add_subplot(gs[2, 0])
small_df = df_combined[df_combined['sample_bin'].isin(['<50', '50-100', '100-500'])]
sns.violinplot(x='sample_bin', y='auroc', hue='model', data=small_df, ax=ax4,
               palette=[model_config['baseline']['color'], model_config['histoneo']['color']],
               linewidth=2, inner='point')
ax4.set_title('小样本量（<500）AUROC分布对比', fontsize=16, fontweight='bold')
ax4.set_xlabel('样本量分组', fontsize=12)
ax4.set_ylabel('AUROC值', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# 子图5：AUROC与样本量相关性对比
ax5 = fig.add_subplot(gs[2, 1])
# Baseline散点
ax5.scatter(df_baseline['n_samples'], df_baseline['auroc'],
            color=model_config['baseline']['color'], alpha=0.7, s=60,
            label=model_config['baseline']['name'], edgecolor='black', linewidth=0.5)
# HistoNeo散点
ax5.scatter(df_histo['n_samples'], df_histo['auroc'],
            color=model_config['histoneo']['color'], alpha=0.7, s=60,
            label=model_config['histoneo']['name'], edgecolor='black', linewidth=0.5)

# 智能刻度
if df_combined['n_samples'].max() / df_combined['n_samples'].min() > 10:
    ax5.set_xscale('log')
else:
    ax5.set_xscale('linear')

ax5.set_title('AUROC与样本量相关性对比', fontsize=16, fontweight='bold')
ax5.set_xlabel('样本数量', fontsize=12)
ax5.set_ylabel('AUROC值', fontsize=12)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# 保存图表（简化路径，保存到当前目录）
save_path = r"MixMHCpred_vs_HistoNeo.png"
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(save_path, bbox_inches='tight')
plt.close()

# -------------------------- 5. 输出对比结果 --------------------------
print("\n" + "=" * 80)
print("📊 双模型核心指标对比结果（数值越高越好）：")
print("-" * 50)
for metric, label in zip(metrics, labels):
    base_val = df_overall[df_overall['model'] == model_config['baseline']['name']][metric].values[0]
    histo_val = df_overall[df_overall['model'] == model_config['histoneo']['name']][metric].values[0]
    diff = histo_val - base_val
    diff_pct = (diff / base_val) * 100 if base_val != 0 else 0
    trend = "📈 提升" if diff > 0 else "📉 下降" if diff < 0 else "➡️ 持平"
    print(f"🔹 {label}:")
    print(f"   - MixMHCpred 2.2: {base_val:.4f}")
    print(f"   - HistoNeo: {histo_val:.4f}")
    print(f"   - 差值: {diff:.4f} ({diff_pct:+.2f}%) {trend}")

print(f"\n✅ 对比图已保存至：{save_path}")
print("=" * 80)