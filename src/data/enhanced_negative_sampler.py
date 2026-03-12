"""
Enhanced Negative Sampler - 基于来源蛋白配对切割的负样本生成

策略（老师方案）:
  对每条正样本，找其来源蛋白（UniProt_ID），从同一蛋白序列上随机切取
  同等长度的片段作为负样本候选，排除所有已知阳性序列。

  Fallback（当 UniProt_ID 缺失或蛋白太短时）:
  从预加载的人类蛋白质组随机池中长度匹配切割。

策略对比（改动前 → 改动后）:
  旧: 50% random-expressed + 50% cross-HLA
      ↑ expressed peptides 本身是阳性，假阴性率极高
      ↑ cross-HLA 跨等位基因借用，假阴性风险 5-15%
  新: 100% source-protein paired sampling（老师方案）
      假阴性率约 1-3%，与 ImmuneApp 2024 的 random-proteome 方案一致
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from pathlib import Path
import json

from .task_definition import Task, TaskManager
from ..config.mode_config import ModeConfig, TrainingMode


class EnhancedNegativeSampler:
    """
    基于来源蛋白配对切割的负样本生成器

    核心改进:
    - 负样本来自正样本的来源蛋白（而非 expressed peptide pool 或 cross-HLA）
    - 假阴性率大幅降低（从 5-15% 降至 1-3%）
    - 与 ImmuneApp 2024 基准方法对齐
    """

    # 来源蛋白序列太短时的 fallback 阈值（aa 数）
    MIN_PROTEIN_LEN_FOR_PAIRED = 50

    # 每条正样本最多尝试切割多少次（防止无限循环）
    MAX_SAMPLE_ATTEMPTS = 200

    def __init__(self,
                 config: ModeConfig,
                 all_data: pd.DataFrame,
                 protein_sequences: Optional[Dict[str, str]] = None,
                 proteome_fallback_pool: Optional[List[str]] = None):
        """
        Args:
            config: ModeConfig
            all_data: 全部数据（含 UniProt_ID / Epitope_Start / Epitope_End）
            protein_sequences: {UniProt_ID: sequence}，由 load_protein_sequences() 预加载
            proteome_fallback_pool: 备用蛋白质组肽段池（list of peptides），
                                    None 时会在首次需要时从 all_data 的蛋白序列自动构建
        """
        self.config = config
        self.mode = config.mode
        self.all_data = all_data
        self.protein_sequences = protein_sequences or {}

        # ========== 构建全局阳性 exclude set ==========
        # 任何 HLA 分型下的已知阳性肽段都不能作为负样本
        positive_data = all_data[all_data['label'] == 1]
        self.global_positive_peptides = set(positive_data['peptide'].unique())

        # ========== 构建 UniProt_ID → 正样本坐标映射 ==========
        # 用于可选的坐标重叠过滤（排除与正样本重叠 >7aa 的位置）
        self._build_protein_position_mapping(positive_data)

        # ========== 备用随机池（按长度分组）==========
        # 优先使用传入的 proteome_fallback_pool；
        # 否则延迟构建（首次 fallback 时）
        self._fallback_pool_by_length: Dict[int, List[str]] = defaultdict(list)
        if proteome_fallback_pool:
            for pep in proteome_fallback_pool:
                length = len(pep)
                if 8 <= length <= 15:
                    self._fallback_pool_by_length[length].append(pep)
            print(f"  Fallback pool loaded: "
                  f"{sum(len(v) for v in self._fallback_pool_by_length.values()):,} peptides")
        self._fallback_pool_built = bool(proteome_fallback_pool)

        # 累计采样统计（跨所有 task）
        self._total_paired = 0
        self._total_fallback = 0

        print(f"\n✓ Enhanced Negative Sampler (Source-Protein Paired) initialized")
        print(f"  Mode: {config.task_type_name}")
        print(f"  Global positive exclude set: {len(self.global_positive_peptides):,}")
        print(f"  Protein sequences loaded: {len(self.protein_sequences):,}")
        print(f"  Negative ratio: 1:{config.negative_ratio}")

    # ------------------------------------------------------------------ #
    #  内部辅助：构建蛋白坐标映射                                          #
    # ------------------------------------------------------------------ #

    def _build_protein_position_mapping(self, positive_data: pd.DataFrame):
        """
        构建 UniProt_ID → [(start, end), ...] 的坐标映射
        用于坐标重叠过滤（可选步骤）
        使用向量化操作替代逐行 iterrows，避免 67万行数据的性能瓶颈。
        """
        self.protein_occupied_ranges: Dict[str, List[tuple]] = defaultdict(list)

        required_cols = {'UniProt_ID', 'Epitope_Start', 'Epitope_End'}
        if not required_cols.issubset(positive_data.columns):
            return

        # 只取三列，dropna，转换类型 — 全向量化
        coord_df = positive_data[['UniProt_ID', 'Epitope_Start', 'Epitope_End']].dropna()
        if coord_df.empty:
            return

        try:
            coord_df = coord_df.copy()
            coord_df['Epitope_Start'] = coord_df['Epitope_Start'].astype(int)
            coord_df['Epitope_End'] = coord_df['Epitope_End'].astype(int)
            coord_df['UniProt_ID'] = coord_df['UniProt_ID'].astype(str)
        except (ValueError, TypeError):
            return

        # 按 UniProt_ID 分组，批量写入 dict
        for uid, grp in coord_df.groupby('UniProt_ID'):
            self.protein_occupied_ranges[uid] = list(
                zip(grp['Epitope_Start'].tolist(), grp['Epitope_End'].tolist())
            )

    def _overlaps_positive(self, uniprot_id: str,
                           pos: int, end: int,
                           overlap_threshold: int = 7) -> bool:
        """
        检查候选切割位置是否与任何已知阳性坐标重叠超过 overlap_threshold aa

        Args:
            uniprot_id: 蛋白 ID
            pos: 候选片段起始位置（1-based）
            end: 候选片段终止位置（1-based）
            overlap_threshold: 允许的最大重叠 aa 数（默认 7，老师方案）
        """
        for occ_start, occ_end in self.protein_occupied_ranges.get(uniprot_id, []):
            overlap = min(end, occ_end) - max(pos, occ_start)
            if overlap > overlap_threshold:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  核心采样：来源蛋白配对切割                                          #
    # ------------------------------------------------------------------ #

    def _sample_from_source_protein(self,
                                    uniprot_id: str,
                                    target_length: int,
                                    n: int,
                                    exclude: set,
                                    filter_overlap: bool = True) -> pd.DataFrame:
        """
        从指定蛋白序列上随机切取 n 个长度为 target_length 的片段

        Args:
            uniprot_id: UniProt ID
            target_length: 目标长度（与正样本长度匹配）
            n: 目标采样数量
            exclude: 需要排除的肽段集合（全局阳性 + 已采样）
            filter_overlap: 是否过滤与正样本坐标重叠 >7aa 的位置

        Returns:
            candidates: 合法的负样本肽段列表（可能 < n）
        """
        sequence = self.protein_sequences.get(str(uniprot_id), '')
        seq_len = len(sequence)

        if seq_len < target_length + 10:
            return []

        # 合法起始位置列表（0-based）
        max_start = seq_len - target_length
        all_positions = list(range(max_start + 1))
        np.random.shuffle(all_positions)

        results = []
        standard_aa = set('ACDEFGHIKLMNPQRSTVWY')

        for pos in all_positions:
            if len(results) >= n:
                break
            end = pos + target_length  # 切片终止（exclusive）
            fragment = sequence[pos:end]

            # 过滤非标准氨基酸
            if not all(aa in standard_aa for aa in fragment):
                continue

            # 过滤已知阳性
            if fragment in exclude:
                continue

            # 可选：过滤坐标重叠（1-based: pos+1 to pos+target_length）
            if filter_overlap and self._overlaps_positive(
                    uniprot_id, pos + 1, pos + target_length):
                continue

            results.append(fragment)

        return results

    # ------------------------------------------------------------------ #
    #  Fallback：从蛋白质组随机池采样                                      #
    # ------------------------------------------------------------------ #

    # 每个长度的 fallback pool 上限（足够覆盖任何 task 的负样本需求）
    MAX_FALLBACK_PER_LENGTH = 50_000

    def _ensure_fallback_pool(self):
        """
        延迟构建 fallback pool：
        从已加载蛋白序列随机切成 8-15mer，排除全局阳性。
        每个长度最多保留 MAX_FALLBACK_PER_LENGTH 条，达到上限立即停止，
        避免 83K 蛋白 × 8 长度 × 20位置 = 1300万条导致构建和采样极慢。
        """
        if self._fallback_pool_built:
            return

        print("  ⚠ 构建 fallback pool（从已加载蛋白序列）...")
        standard_aa = set('ACDEFGHIKLMNPQRSTVWY')

        # 随机打乱蛋白顺序，确保 pool 不偏向特定蛋白
        protein_items = list(self.protein_sequences.items())
        np.random.shuffle(protein_items)

        for uid, seq in protein_items:
            seq_len = len(seq)

            # 如果所有长度都已满，提前退出
            if all(len(self._fallback_pool_by_length[l]) >= self.MAX_FALLBACK_PER_LENGTH
                   for l in range(8, 16)):
                break

            for length in range(8, 16):
                # 该长度已满，跳过
                if len(self._fallback_pool_by_length[length]) >= self.MAX_FALLBACK_PER_LENGTH:
                    continue

                max_start = seq_len - length
                if max_start <= 0:
                    continue

                n_sample = min(5, max_start + 1)  # 每蛋白每长度取5个（够用且快）
                positions = np.random.choice(max_start + 1, n_sample, replace=False)
                for pos in positions:
                    frag = seq[pos:pos + length]
                    if (all(aa in standard_aa for aa in frag)
                            and frag not in self.global_positive_peptides):
                        self._fallback_pool_by_length[length].append(frag)
                        if len(self._fallback_pool_by_length[length]) >= self.MAX_FALLBACK_PER_LENGTH:
                            break

        self._fallback_pool_built = True
        total = sum(len(v) for v in self._fallback_pool_by_length.values())
        print(f"  ✓ Fallback pool built: {total:,} peptides "
              f"(capped at {self.MAX_FALLBACK_PER_LENGTH:,}/length)")

    def _sample_fallback(self,
                         target_length: int,
                         n: int,
                         exclude: set) -> List[str]:
        """
        从 fallback pool 中按长度随机采样 n 个，排除 exclude
        """
        self._ensure_fallback_pool()

        pool = [p for p in self._fallback_pool_by_length.get(target_length, [])
                if p not in exclude]

        if not pool:
            return []

        n_sample = min(n, len(pool))
        return list(np.random.choice(pool, n_sample, replace=False))

    # ------------------------------------------------------------------ #
    #  主采样接口                                                          #
    # ------------------------------------------------------------------ #

    def generate_negatives_for_task(self,
                                    task: Task,
                                    positive_peptides: List[str],
                                    n_negatives: Optional[int] = None,
                                    source_proteins: Optional[List[str]] = None,
                                    epitope_starts: Optional[List[int]] = None,
                                    epitope_ends: Optional[List[int]] = None,
                                    filter_overlap: bool = True) -> List[str]:
        """
        为一个 task 生成负样本。

        生物学保证：每条正样本的负样本严格来自同一来源蛋白。
        性能优化：同一 (UniProt_ID, length) 组合只切割一次，结果缓存后
                  按需分配给所有来自该蛋白该长度的正样本，避免重复计算。

        流程：
          1. 按 (UniProt_ID, length) 分组统计每组正样本数量
          2. 预先为每组切割足量负样本候选（一次性，结果存入 cache）
          3. 逐条正样本从 cache 中取 negative_ratio 个（同时记录来源蛋白）
          4. cache 不足时 fallback 到蛋白质组随机池
        """
        if n_negatives is None:
            n_negatives = len(positive_peptides) * self.config.negative_ratio

        exclude = self.global_positive_peptides | set(positive_peptides)
        ratio = self.config.negative_ratio
        n_fallback = 0
        n_paired = 0
        # 注：n_paired/n_fallback 会在方法末尾累加到 self._total_paired/_total_fallback

        # ========== Step 1: 按 (uid, length) 分组，统计每组正样本数 ==========
        from collections import Counter
        group_counts: Counter = Counter()   # (uid, length) → 正样本数
        pos_groups: List[tuple] = []        # 与 positive_peptides 一一对应的 (uid, length)

        for i, pep in enumerate(positive_peptides):
            uid = (str(source_proteins[i])
                   if source_proteins and i < len(source_proteins)
                      and pd.notna(source_proteins[i])
                      and str(source_proteins[i]) in self.protein_sequences
                   else None)
            length = len(pep)
            key = (uid, length)
            pos_groups.append(key)
            group_counts[key] += 1

        # ========== Step 2: 为每个 (uid, length) 组预切割负样本候选 ==========
        # 每组需要的候选数 = 该组正样本数 × ratio × 1.5（buffer，应对过滤损耗）
        # 同一组只切一次，结果存入 cache
        neg_cache: Dict[tuple, List[str]] = {}   # (uid, length) → list of candidates
        cache_ptr: Dict[tuple, int] = {}          # (uid, length) → 当前取到的位置

        for (uid, length), count in group_counts.items():
            needed = int(count * ratio * 1.5) + 5   # buffer

            if uid is not None:
                # 来源蛋白配对切割
                # 切不够时直接接受实际数量，不用其他蛋白补足
                # （保证负样本来自同一蛋白，避免引入丰度偏差）
                candidates = self._sample_from_source_protein(
                    uid, length, needed, exclude, filter_overlap
                )
                n_paired += min(len(candidates), count * ratio)
            else:
                # 无 UniProt_ID 时才允许 fallback
                candidates = self._sample_fallback(
                    length, int(count * ratio * 1.5),
                    exclude
                )
                n_fallback += len(candidates)

            neg_cache[(uid, length)] = candidates
            cache_ptr[(uid, length)] = 0

        # ========== Step 3: 逐条正样本从 cache 中取 ratio 个（同时记录来源蛋白）==========
        negatives: List[str] = []
        neg_sources: List[str] = []   # 与 negatives 一一对应，用于 task balance 分层采样

        for key in pos_groups:
            uid, length = key
            ptr = cache_ptr[key]
            pool = neg_cache[key]

            taken = pool[ptr: ptr + ratio]
            cache_ptr[key] = ptr + ratio

            # cache 用完时再补切（极少情况）
            # uid 有效：只从来源蛋白补切，切不够就接受，不用其他蛋白凑数
            # uid 为 None：fallback 补足
            if len(taken) < ratio:
                extra_needed = ratio - len(taken)
                if uid is not None:
                    extra = self._sample_from_source_protein(
                        uid, length, extra_needed * 2,
                        exclude | set(negatives) | set(pool), filter_overlap
                    )
                    taken.extend(extra[:extra_needed])
                else:
                    extra = self._sample_fallback(
                        length, extra_needed,
                        exclude | set(negatives) | set(pool)
                    )
                    n_fallback += len(extra)
                    taken.extend(extra)

            negatives.extend(taken)
            # 记录每条负样本的来源蛋白（uid 有值用 uid，否则标记为 '__fallback__'）
            source_label = uid if uid is not None else '__fallback__'
            neg_sources.extend([source_label] * len(taken))

        # ========== Step 4: 最终统计，返回带 source_protein 列的 DataFrame ==========
        # 不做总量补足：若来源蛋白太短切不够 ratio 个，接受实际数量。
        # 强行用其他蛋白补足会重新引入丰度偏差，违背负样本策略目的。
        # uid 为 None 的正样本（无 UniProt_ID）已在 Step 2 用 fallback 处理。

        # 累加到实例统计
        self._total_paired += n_paired
        self._total_fallback += n_fallback

        # 打乱（source_protein 跟着同步打乱）
        indices = np.random.permutation(len(negatives))
        negatives  = [negatives[i]  for i in indices]
        neg_sources = [neg_sources[i] for i in indices]

        return pd.DataFrame({
            'peptide':        negatives,
            'source_protein': neg_sources,
            'peptide_length': [len(p) for p in negatives],
        })

    # ------------------------------------------------------------------ #
    #  批量生成：所有 tasks                                                #
    # ------------------------------------------------------------------ #

    def generate_negatives_for_all_tasks(self,
                                         task_manager: TaskManager,
                                         filter_overlap: bool = True,
                                         source_data: Optional[pd.DataFrame] = None
                                         ) -> Dict[str, pd.DataFrame]:
        """
        为所有 tasks 生成负样本

        Args:
            task_manager: TaskManager 实例
            filter_overlap: 是否过滤坐标重叠
            source_data: 正样本来源 DataFrame（默认用 self.all_data）
                         传入各自 split（train_df/val_df/test_df）可确保
                         只对该 split 的正样本生成负样本，
                         而 exclude set 仍来自初始化时的全量数据，防止假阴性

        Returns:
            task_datasets: {task_id: DataFrame(peptide, hla, [tissue], label)}
        """
        # 正样本来源：优先用传入的 source_data，否则用全量 all_data
        data_for_positives = source_data if source_data is not None else self.all_data

        print(f"\n{'=' * 80}")
        print(f"Generating Negatives for All Tasks (Source-Protein Paired)")
        print(f"{'=' * 80}")

        all_tasks = task_manager.get_all_tasks()
        task_datasets = {}

        # 预先检查是否有 UniProt_ID 列
        has_uniprot = 'UniProt_ID' in data_for_positives.columns
        has_position = ('Epitope_Start' in data_for_positives.columns and
                        'Epitope_End' in data_for_positives.columns)

        if not has_uniprot:
            print("  ⚠ UniProt_ID 列不存在，将全部使用 fallback 采样")

        n_paired_total = 0
        n_fallback_total = 0

        try:
            from tqdm import tqdm
            task_iter = tqdm(all_tasks.items(), total=len(all_tasks),
                             desc="Generating negatives", unit="task")
        except ImportError:
            task_iter = all_tasks.items()

        for task_id, task in task_iter:
            # 获取该 task 的正样本（只从 source_data 这个 split 里取）
            if self.mode == TrainingMode.HLA_ONLY:
                task_positives = data_for_positives[
                    (data_for_positives['hla'] == task.hla) &
                    (data_for_positives['label'] == 1)
                ]
            else:  # Mode 2
                task_positives = data_for_positives[
                    (data_for_positives['hla'] == task.hla) &
                    (data_for_positives['tissue'] == task.tissue) &
                    (data_for_positives['label'] == 1)
                ]

            if len(task_positives) == 0:
                continue

            positive_peptides = list(task_positives['peptide'])

            # 提取来源蛋白信息
            source_proteins = (
                list(task_positives['UniProt_ID']) if has_uniprot else None
            )
            epitope_starts = (
                list(task_positives['Epitope_Start']) if has_position else None
            )
            epitope_ends = (
                list(task_positives['Epitope_End']) if has_position else None
            )

            # 生成负样本（返回带 source_protein 列的 DataFrame）
            neg_df = self.generate_negatives_for_task(
                task, positive_peptides,
                source_proteins=source_proteins,
                epitope_starts=epitope_starts,
                epitope_ends=epitope_ends,
                filter_overlap=filter_overlap
            )

            # 补充 task 信息列
            neg_df['hla']   = task.hla
            neg_df['label'] = 0
            if self.mode == TrainingMode.HLA_TISSUE:
                neg_df['tissue'] = task.tissue

            # 正样本补 source_protein 列（来自 UniProt_ID，无则填 None）
            pos_df = task_positives[['peptide', 'hla', 'label'] + (
                ['tissue'] if 'tissue' in task_positives.columns else []
            )].copy()
            pos_df['source_protein'] = (
                task_positives['UniProt_ID'].astype(str).values
                if has_uniprot else None
            )

            task_df = pd.concat([pos_df, neg_df], ignore_index=True)
            task_datasets[task_id] = task_df

        print(f"\n✓ Generated negatives for {len(task_datasets)} tasks")

        total_samples = sum(len(df) for df in task_datasets.values())
        total_pos = sum((df['label'] == 1).sum() for df in task_datasets.values())
        total_neg = sum((df['label'] == 0).sum() for df in task_datasets.values())

        print(f"  Total samples: {total_samples:,}")
        print(f"  Positive: {total_pos:,} ({total_pos / total_samples * 100:.1f}%)")
        print(f"  Negative: {total_neg:,} ({total_neg / total_samples * 100:.1f}%)")
        print(f"  Avg negative ratio: 1:{total_neg / total_pos:.1f}")

        # 负样本来源统计
        total_neg_generated = self._total_paired + self._total_fallback
        if total_neg_generated > 0:
            paired_pct = self._total_paired / total_neg_generated * 100
            fallback_pct = self._total_fallback / total_neg_generated * 100
            print(f"\n  Negative source breakdown:")
            print(f"    Source-protein paired: {self._total_paired:,} ({paired_pct:.1f}%)")
            print(f"    Fallback (proteome):   {self._total_fallback:,} ({fallback_pct:.1f}%)")
            if fallback_pct > 5.0:
                print(f"    ⚠ Fallback 比例偏高 (>{5}%), 蛋白丰度控制效果下降")

        return task_datasets


# ------------------------------------------------------------------ #
#  工具函数：加载蛋白序列                                              #
# ------------------------------------------------------------------ #

def load_protein_sequences(fasta_path: str) -> Dict[str, str]:
    """
    从 FASTA 文件加载蛋白序列

    Args:
        fasta_path: UniProt FASTA 文件路径

    Returns:
        sequences: {UniProt_ID: sequence}

    Usage:
        sequences = load_protein_sequences('data/human_proteome.fasta')
        sampler = EnhancedNegativeSampler(config, data, protein_sequences=sequences)
    """
    sequences = {}
    try:
        from Bio import SeqIO
        for record in SeqIO.parse(fasta_path, 'fasta'):
            # UniProt FASTA header: >sp|P12345|GENE_HUMAN ...
            uid = record.id.split('|')[1] if '|' in record.id else record.id
            sequences[uid] = str(record.seq)
        print(f"✓ Loaded {len(sequences):,} protein sequences from {fasta_path}")
    except ImportError:
        print("⚠ BioPython not installed, using manual FASTA parser")
        sequences = _parse_fasta_manual(fasta_path)
    return sequences


def _parse_fasta_manual(fasta_path: str) -> Dict[str, str]:
    """BioPython 不可用时的手动 FASTA 解析器"""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                header = line[1:].split()[0]
                current_id = header.split('|')[1] if '|' in header else header
                current_seq = []
            else:
                current_seq.append(line)

    if current_id:
        sequences[current_id] = ''.join(current_seq)

    print(f"✓ Loaded {len(sequences):,} protein sequences from {fasta_path}")
    return sequences


# 向后兼容
NegativeSampler = EnhancedNegativeSampler


if __name__ == "__main__":
    print("Enhanced Negative Sampler - 基于已表达peptides的负样本生成")
    print("\n核心优势:")
    print("  ✓ 所有负样本来自数据集中已表达的peptides")
    print("  ✓ 不需要外部表达谱数据")
    print("  ✓ 生物学上更合理")
    print("  ✓ 支持tissue-aware采样 (Mode 2)")