# HistoNeo

**基于图神经网络与特征线性调制（FiLM）的组织特异性 HLA-I 抗原呈递预测**

> 一个面向癌症免疫治疗新抗原发现的深度学习框架，通过引入组织上下文信息以提升 MHC-I 抗原呈递预测性能。

---

## 项目简介

HistoNeo（HNeo）是西交利物浦大学（XJTLU）博士研究课题的配套代码库，核心问题是：在 HLA 等位基因信息之外，额外引入**组织上下文**是否能有效提升 MHC-I 肽段呈递预测的准确性——这是个性化癌症疫苗设计中的关键瓶颈。

核心假设是：抗原呈递不仅由 HLA 等位基因决定，也受到组织微环境的影响。HistoNeo 通过以下两种模式的对照实验加以验证：

| 模式 | 任务定义 | 描述 |
|------|---------|------|
| **Mode 1** | 仅 HLA | 基线：仅基于 HLA 等位基因进行呈递预测 |
| **Mode 2** | HLA × Tissue | 提出方法：同时基于 HLA 等位基因和组织类型进行预测 |

Mode 2 以**图神经网络（GNN）**为骨干，配合**FiLM（Feature-wise Linear Modulation）**层，使组织嵌入向量能够动态调制 HLA 特异性的任务表示。

---

## 主要结果

在 IEDB 免疫肽组学留出测试集上，Mode 2（HLA × Tissue）相比 Mode 1 基线：

- **AUROC**：提升 +4.9%
- **AUPRC**：提升 +16.5%

一个值得关注的反直觉发现是：Mode 2 在**训练样本稀少的罕见任务**上增益更为显著，说明组织条件化在数据匮乏时能提供有效的归纳偏置。

---

## 模型架构

```
输入：(肽段序列, HLA 等位基因, 组织标签)

肽段编码器 (Peptide Encoder)
  └── 氨基酸嵌入 + BiLSTM/Conv → 肽段特征

任务图神经网络 (Task GNN)
  └── 每个 HLA 等位基因作为图节点
  └── 边权重由 HLA 伪序列相似度（BLOSUM50）计算
  └── 消息传递 → HLA 任务嵌入

[仅 Mode 2] 组织 FiLM 层
  └── 组织嵌入（可学习查找表）
  └── FiLM: task_emb = γ(tissue) * task_emb + β(tissue)

任务条件化预测器 (Task-Conditioned Predictor)
  └── MLP([肽段特征; 任务嵌入]) → 呈递得分
```

三阶段流水线（肽段编码器 → 任务 GNN → 预测器）在两种模式间共享。Mode 2 额外引入轻量级的组织 FiLM 分支，参数量增加约 5–10%。

---

## 仓库结构

```
HNeo/
├── configs/
│   └── hla_sequences.json              # HLA 伪序列（34 aa，NetMHCpan 格式）
│
├── data/
│   ├── Cleaned_data.py                 # IEDB 原始数据清洗（人类，Homo sapiens）
│   ├── Cleaned_data_mouse.py           # IEDB 原始数据清洗（小鼠，Mus musculus）
│   ├── preprocess_data.py              # 清洗结果整合，输出 Mode 系统标准 TSV
│   ├── data_statistics.py              # 清洗结果统计分析
│   ├── human_proteome.fasta            # UniProt 人类蛋白质组（用于负样本采样）
│   └── negative_samples/               # 负样本缓存
│
├── scripts/
│   ├── train_mode1.py                  # 端到端训练：Mode 1（仅 HLA）
│   ├── train_mode2.py                  # 端到端训练：Mode 2（HLA×Tissue）
│   ├── train_mode1_with_mode2_splits.py  # 在 Mode 2 划分上重训 Mode 1（公平对比）
│   ├── evaluate_per_task.py            # 逐任务评估（Mode 2，含 tissue 分层）
│   ├── evaluate_mode1_simple.py        # 逐任务评估（Mode 1）
│   ├── evaluate_mode2_simple.py        # 逐任务评估（Mode 2，含 tissue 汇总）
│   ├── evaluate_mixmhcpred.py          # 基线：MixMHCpred 2.2
│   ├── evaluate_mhcflurry.py           # 基线：MHCflurry 2.0
│   └── evaluate_mhcnuggets.py          # 基线：MHCnuggets
│
├── src/
│   ├── config/
│   │   └── mode_config.py              # ModeConfig 数据类；create_mode1/2_config()
│   │
│   ├── data/
│   │   ├── task_definition.py          # Task / TaskManager 类
│   │   ├── unified_task_creator.py     # 创建 HLA-only 或 HLA×Tissue 任务集
│   │   ├── enhanced_negative_sampler.py  # 来源蛋白配对切割负样本生成
│   │   └── dataset.py                  # PyTorch Dataset（模式感知）
│   │
│   ├── models/
│   │   ├── full_model.py               # ImmuneAppModel：FiLM 融合，模式感知前向传播
│   │   ├── peptide_encoder.py          # 氨基酸编码器
│   │   ├── task_gnn.py                 # HLA 任务图 GNN
│   │   └── predictor.py               # 任务条件化 MLP 预测器
│   │
│   ├── graph/
│   │   └── task_graph.py              # HLA 相似性图构建
│   │
│   ├── training/
│   │   ├── unified_trainer.py         # 主训练循环（标准 + 任务均衡采样）
│   │   ├── per_task_evaluator.py      # 逐任务指标，tissue 分层汇总
│   │   └── maml.py                    # MAML 训练器（遗留模块，当前已禁用）
│   │
│   └── docs/
│       └── MIGRATION_GUIDE.md         # Mode 1 → Mode 2 代码迁移说明
│
└── README.md
```

---

## 安装

**环境要求**：Python 3.11，PyTorch 2.0，CUDA 11.8+

```bash
# 克隆仓库
git clone https://github.com/<your-username>/HNeo.git
cd HNeo

# 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 安装依赖
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pandas numpy scikit-learn biopython tqdm
```

### 数据准备

1. **IEDB 免疫肽组学数据** — 从 [https://www.iedb.org](https://www.iedb.org) 下载 `mhc_ligand_full.csv`，放入 `data/` 目录。
2. **人类蛋白质组 FASTA** — 从 [UniProt](https://www.uniprot.org/proteomes/UP000005640)（物种 ID 9606）下载，保存为 `data/human_proteome.fasta`。
3. **HLA 伪序列** — 已提供于 `configs/hla_sequences.json`（34 残基，NetMHCpan 格式）。

如需进行**小鼠实验**，请从 UniProt（物种 ID 10090）下载 *Mus musculus* 蛋白质组，保存为 `data/mouse_proteome.fasta`。

---

## 使用方法

### 第一步 — 原始数据清洗

IEDB 原始数据为多列两级表头的大型 CSV 文件，需先分块清洗。**人类数据**运行：

```bash
python data/Cleaned_data.py
```

默认读取当前目录下的 `mhc_ligand_full_*.csv` 分块文件，过滤 *Homo sapiens* 条目，仅保留符合 `HLA-[ABC]*dd:dd` 格式的 MHC-I 限制性肽段（长度 8–15 aa），从 `Molecule Parent IRI` 中提取 UniProt ID，并按疾病/组织信息推断 `Inferred_Tissue`，结果输出至 `cleaned_data/` 目录（分块 CSV）。

**小鼠数据**流程相同，过滤条件改为 *Mus musculus* 和鼠 MHC-I 格式（`H-?2-[A-Za-z]\w*`）：

```bash
python data/Cleaned_data_mouse.py
```

结果输出至 `cleaned_data_mouse/`。

### 第一步（续）— 统计分析（可选）

```bash
python data/data_statistics.py
```

读取 `cleaned_data/` 中的分块文件，打印 HLA 分布、肽段长度分布、组织分布等统计信息，并保存 `data_statistics.json`。

### 第二步 — 数据整合与格式转换

将清洗后的分块数据整合为 Mode 训练脚本所需的标准 TSV 格式。

**Mode 1（HLA-only）**：

```bash
python data/preprocess_data.py --positive_dir cleaned_data --output_file data/mode1_data.tsv --mode mode1
```

**Mode 2（HLA×Tissue）**：

```bash
python data/preprocess_data.py --positive_dir cleaned_data --output_file data/mode2_data.tsv --mode mode2 --tissue_source Inferred_Tissue
```

该步骤自动检测 tissue 列可用性，输出适用性诊断（tissue 类型数量、HLA×Tissue 组合数、建议的 `--min_samples` 参数），并去除重复样本。输出列为 `peptide`、`hla`、`label`（Mode 2 额外包含 `tissue`），以及用于负样本配对采样的 `UniProt_ID`、`Epitope_Start`、`Epitope_End`。

### 第三步 — 训练 Mode 1（HLA-Only 基线）

```bash
python scripts/train_mode1.py --data_file data/mode1_data.tsv --output_dir outputs/mode1 --hla_sequences configs/hla_sequences.json --n_epochs 30 --batch_size 256 --min_samples 20 --negative_ratio 20 --use_task_balanced --use_negative_cache
```

### 第四步 — 训练 Mode 2（HLA × Tissue）

```bash
python scripts/train_mode2.py --data_file data/mode2_data.tsv --output_dir outputs/mode2 --hla_sequences configs/hla_sequences.json --tissue_source Host --n_epochs 30 --batch_size 256 --min_samples 10 --negative_ratio 20 --use_task_balanced --use_negative_cache
```

### 第五步 — 评估

```bash
# 逐任务评估（Mode 2，含 tissue 分层）
python scripts/evaluate_per_task.py --output_dir outputs/mode2 --data_file data/mode2_data.tsv --tissue_source Inferred_Tissue

# 基线对比：MixMHCpred 2.2
python scripts/evaluate_mixmhcpred.py --test_file data/mode1_data.tsv --mode1_output_dir outputs/mode1 --mixmhcpred_dir /path/to/MixMHCpred --output_dir outputs/mixmhcpred_eval --use_negative_cache
```

### Mode 1 vs. Mode 2 公平对比

为确保公平对比，Mode 1 应在与 Mode 2 **完全相同的数据划分**上重训：

```bash
python scripts/train_mode1_with_mode2_splits.py --data_file data/mode2_data.tsv --mode2_output_dir outputs/mode2 --output_dir outputs/mode1_on_mode2_splits --n_epochs 30
```

---

## 实验设计说明

### 负样本生成

负样本通过**来源蛋白配对切割策略**生成（`EnhancedNegativeSampler`）：对每条正样本（肽段, HLA），从*同一来源蛋白*上随机切取等长片段作为负样本候选，并排除所有已知阳性序列。该策略的估计假阴性率约为 1–3%，与 ImmuneApp 2024 基准方法对齐。

默认负样本比例：**1:20**（阳性:阴性）。

### 数据划分与数据泄漏防护

数据集按 HLA 类型分层，以 70/15/15 的比例划分为训练/验证/测试集。在划分后执行去重步骤，移除同时出现在训练集和验证集中的（肽段, HLA）对（可能带有不同 tissue 标签），防止 HLA × Tissue 行级分层时产生的隐式数据泄漏。该方案优于直接按组划分，可与使用相同阳性样本训练的基线方法保持可比性。

### 任务定义

- **Mode 1 任务**：每个 HLA 等位基因对应一个任务（最少 20 条样本）。
- **Mode 2 任务**：每个（HLA 等位基因, 组织类型）组合对应一个任务（最少 10 条样本），未知组织的样本被排除。

### HLA 任务图

HLA 等位基因作为图的节点，边权重由 34 残基 BLOSUM50 编码伪序列的余弦相似度计算（NetMHCpan 规范）。在 GNN 消息传递前，通过相似度阈值对图进行稀疏化处理。

---

## 基线方法

HistoNeo 与以下方法进行对比：

| 方法 | 参考文献 |
|------|---------|
| MixMHCpred 2.2 | Bassani-Sternberg et al. |
| MHCflurry 2.0 | O'Donnell et al., *Cell Syst.* 2020 |
| MHCnuggets | Shao et al., *Cancer Immunol. Res.* 2020 |
| ImmuneApp | Xu et al., *Nat. Commun.* 2024 |
| MHLAPre | Xu et al., *Brief. Bioinform.* 2024 |
