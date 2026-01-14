import pandas as pd
import numpy as np
from pathlib import Path
import random
from collections import defaultdict
import re
from Bio import SeqIO
import requests
from tqdm import tqdm


class NegativeSampleGenerator:
    """
    Phase 1: HLA-specific负样本生成器

    三种策略：
    1. 随机采样（Random Sampling）
    2. 跨HLA负样本（Cross-HLA Negatives）
    3. 困难负样本（Hard Negatives）
    """

    def __init__(self,
                 positive_data_dir='cleaned_data',
                 proteome_file='human_proteome.fasta',
                 random_seed=42):
        """
        初始化

        Args:
            positive_data_dir: 正样本数据目录
            proteome_file: 人类蛋白质组FASTA文件
            random_seed: 随机种子
        """
        self.positive_data_dir = Path(positive_data_dir)
        self.proteome_file = proteome_file
        self.random_seed = random_seed

        np.random.seed(random_seed)
        random.seed(random_seed)

        # 标准氨基酸
        self.standard_aa = set('ACDEFGHIKLMNPQRSTVWY')

        # BLOSUM62替换矩阵（简化版，只列出正分值的替换）
        self.blosum62_substitutions = {
            'A': ['S', 'T'],
            'R': ['K', 'Q'],
            'N': ['D', 'S', 'T'],
            'D': ['E', 'N'],
            'C': ['S'],
            'Q': ['E', 'K', 'R'],
            'E': ['D', 'Q', 'K'],
            'G': ['A'],
            'H': ['N', 'Y'],
            'I': ['L', 'M', 'V'],
            'L': ['I', 'M', 'V'],
            'K': ['R', 'Q', 'E'],
            'M': ['I', 'L', 'V'],
            'F': ['Y', 'W'],
            'P': ['A'],
            'S': ['A', 'T', 'N'],
            'T': ['A', 'S', 'N'],
            'W': ['Y', 'F'],
            'Y': ['F', 'H', 'W'],
            'V': ['I', 'L', 'M'],
        }

        print("=" * 80)
        print("负样本生成器初始化")
        print("=" * 80)

        # 加载正样本
        self._load_positive_samples()

        # 加载蛋白质组
        self._load_proteome()

    def _load_positive_samples(self):
        """加载所有正样本"""
        print("\n加载正样本...")

        all_files = sorted(self.positive_data_dir.glob('cleaned_*_chunk_*.csv'))

        self.positive_peptides = set()
        self.hla_peptides = defaultdict(set)  # HLA -> peptides
        self.peptide_hlas = defaultdict(set)  # peptide -> HLAs
        self.length_distribution = defaultdict(int)

        for file in tqdm(all_files, desc="读取文件"):
            df = pd.read_csv(file)

            for _, row in df.iterrows():
                peptide = row['Peptide']
                hla = row['MHC_Restriction_Name']
                length = row['Peptide_Length']

                self.positive_peptides.add(peptide)
                self.hla_peptides[hla].add(peptide)
                self.peptide_hlas[peptide].add(hla)
                self.length_distribution[length] += 1

        self.all_hlas = list(self.hla_peptides.keys())

        print(f"  加载正样本: {len(self.positive_peptides):,} 独特肽段")
        print(f"  涵盖HLA: {len(self.all_hlas)} 种")
        print(f"  长度分布: {dict(self.length_distribution)}")

    def _load_proteome(self):
        """加载人类蛋白质组"""
        print("\n加载人类蛋白质组...")

        if not Path(self.proteome_file).exists():
            print(f"  ⚠ 未找到 {self.proteome_file}")
            print(f"  尝试下载...")
            self._download_proteome()

        self.proteome_sequences = []

        with open(self.proteome_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                seq = str(record.seq)
                # 只保留标准氨基酸
                if all(aa in self.standard_aa for aa in seq):
                    self.proteome_sequences.append(seq)

        print(f"  加载蛋白序列: {len(self.proteome_sequences):,} 条")

    def _download_proteome(self):
        """下载人类蛋白质组"""
        url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz"

        print(f"  从UniProt下载...")
        print(f"  URL: {url}")
        print(f"  这可能需要几分钟...")

        import gzip
        import shutil

        # 下载
        response = requests.get(url, stream=True)
        gz_file = self.proteome_file + '.gz'

        with open(gz_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # 解压
        with gzip.open(gz_file, 'rb') as f_in:
            with open(self.proteome_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # 删除压缩文件
        Path(gz_file).unlink()

        print(f"  ✓ 下载完成: {self.proteome_file}")

    # ========================================================================
    # 策略1: 随机采样
    # ========================================================================

    def generate_random_negatives(self, n_samples, target_length=None):
        """
        策略1: 从蛋白质组随机采样

        Args:
            n_samples: 需要的负样本数量
            target_length: 目标长度（None则按正样本分布采样）

        Returns:
            list of peptides
        """
        print(f"\n策略1: 随机采样（目标: {n_samples:,} 个）")

        negatives = set()
        attempts = 0
        max_attempts = n_samples * 10

        pbar = tqdm(total=n_samples, desc="生成随机负样本")

        while len(negatives) < n_samples and attempts < max_attempts:
            attempts += 1

            # 选择长度
            if target_length:
                length = target_length
            else:
                # 按正样本长度分布采样
                length = random.choices(
                    list(self.length_distribution.keys()),
                    weights=list(self.length_distribution.values())
                )[0]

            # 随机选择一个蛋白
            protein = random.choice(self.proteome_sequences)

            # 随机选择起始位置
            if len(protein) < length:
                continue

            start = random.randint(0, len(protein) - length)
            peptide = protein[start:start + int(length)]

            # 检查是否是标准氨基酸
            if not all(aa in self.standard_aa for aa in peptide):
                continue

            # 检查是否与正样本重复
            if peptide in self.positive_peptides:
                continue

            # 检查是否已生成
            if peptide in negatives:
                continue

            negatives.add(peptide)
            pbar.update(1)

        pbar.close()

        print(f"  ✓ 生成随机负样本: {len(negatives):,} 个")
        print(f"  尝试次数: {attempts:,}")

        return list(negatives)

    # ========================================================================
    # 策略2: 跨HLA负样本
    # ========================================================================

    def generate_cross_hla_negatives(self, hla_pairs=None, samples_per_pair=100):
        """
        策略2: 跨HLA负样本

        一个HLA的阳性 → 另一个HLA的阴性

        Args:
            hla_pairs: [(hla1, hla2), ...] 如果为None，自动生成
            samples_per_pair: 每对HLA生成多少负样本

        Returns:
            dict: {hla: [negatives]}
        """
        print(f"\n策略2: 跨HLA负样本")

        if hla_pairs is None:
            hla_pairs = self._generate_hla_pairs()

        print(f"  HLA配对数: {len(hla_pairs)}")

        cross_hla_negatives = defaultdict(list)

        for hla_target, hla_source in tqdm(hla_pairs, desc="生成跨HLA负样本"):
            # 从hla_source的阳性样本中采样
            source_peptides = list(self.hla_peptides[hla_source])

            if len(source_peptides) == 0:
                continue

            # 随机选择
            n_sample = min(samples_per_pair, len(source_peptides))
            sampled = random.sample(source_peptides, n_sample)

            # 只保留不是hla_target阳性的
            for peptide in sampled:
                if hla_target not in self.peptide_hlas[peptide]:
                    cross_hla_negatives[hla_target].append(peptide)

        total = sum(len(v) for v in cross_hla_negatives.values())
        print(f"  ✓ 生成跨HLA负样本: {total:,} 个")
        print(f"  覆盖HLA: {len(cross_hla_negatives)}")

        return cross_hla_negatives

    def _generate_hla_pairs(self):
        """
        生成HLA配对

        策略：
        - HLA-A vs HLA-B/C
        - HLA-B vs HLA-A/C
        - HLA-C vs HLA-A/B
        """
        pairs = []

        hla_a = [h for h in self.all_hlas if h.startswith('HLA-A')]
        hla_b = [h for h in self.all_hlas if h.startswith('HLA-B')]
        hla_c = [h for h in self.all_hlas if h.startswith('HLA-C')]

        # A vs B
        for a in hla_a:
            for b in hla_b[:3]:  # 只取前3个最常见的B
                pairs.append((a, b))

        # A vs C
        for a in hla_a:
            for c in hla_c[:3]:
                pairs.append((a, c))

        # B vs A
        for b in hla_b:
            for a in hla_a[:3]:
                pairs.append((b, a))

        print(f"  生成HLA配对: {len(pairs)} 对")

        return pairs

    # ========================================================================
    # 策略3: 困难负样本
    # ========================================================================

    def generate_hard_negatives(self, positive_peptides, n_mutations=1):
        """
        策略3: 困难负样本（保守突变）

        Args:
            positive_peptides: list of (peptide, hla)
            n_mutations: 突变数量（1或2）

        Returns:
            list of (peptide, hla)
        """
        print(f"\n策略3: 困难负样本（突变数: {n_mutations}）")

        hard_negatives = []

        for peptide, hla in tqdm(positive_peptides, desc="生成困难负样本"):
            mutants = self._generate_mutants(peptide, n_mutations)

            for mutant in mutants:
                # 确保不在正样本中
                if mutant not in self.positive_peptides:
                    hard_negatives.append((mutant, hla))

        print(f"  ✓ 生成困难负样本: {len(hard_negatives):,} 个")

        return hard_negatives

    def _generate_mutants(self, peptide, n_mutations=1):
        """
        生成肽段的突变体

        策略：
        - 避开anchor位置（位置2和C端）
        - 使用BLOSUM62保守替换
        """
        mutants = []
        length = len(peptide)

        # 确定可突变位置（避开anchor）
        mutable_positions = list(range(length))
        if length >= 9:
            # 9-mer的anchor通常是2和9位
            mutable_positions = [i for i in range(length) if i not in [1, length - 1]]

        if len(mutable_positions) == 0:
            return mutants

        # 生成突变
        for _ in range(min(5, len(mutable_positions))):  # 每个肽段生成5个突变体
            peptide_list = list(peptide)

            # 随机选择突变位置
            positions = random.sample(mutable_positions, min(n_mutations, len(mutable_positions)))

            for pos in positions:
                original_aa = peptide[pos]

                # 从BLOSUM62选择替换
                if original_aa in self.blosum62_substitutions:
                    possible = self.blosum62_substitutions[original_aa]
                    new_aa = random.choice(possible)
                    peptide_list[pos] = new_aa

            mutant = ''.join(peptide_list)
            if mutant != peptide:
                mutants.append(mutant)

        return mutants

    # ========================================================================
    # 综合生成
    # ========================================================================

    def generate_all_negatives(self,
                               ratio=5.0,
                               strategy_weights=(0.4, 0.4, 0.2),
                               output_dir='negative_samples'):
        """
        综合生成所有负样本
        """
        print("=" * 80)
        print("Phase 1: 综合负样本生成")
        print("=" * 80)

        n_positives = len(self.positive_peptides)
        n_total_negatives = int(n_positives * ratio)

        print(f"\n正样本数: {n_positives:,}")
        print(f"目标负样本数: {n_total_negatives:,} (比例: {ratio}:1)")

        # 计算各策略的数量
        n_random = int(n_total_negatives * strategy_weights[0])
        n_cross = int(n_total_negatives * strategy_weights[1])
        n_hard = int(n_total_negatives * strategy_weights[2])

        # === 1. 随机负样本（修改这里） ===
        random_negatives = self.generate_random_negatives(n_random)

        # ★ 新增：为随机负样本分配HLA
        print(f"\n为随机负样本分配HLA...")

        # 计算HLA分布（用于加权随机分配）
        hla_distribution = {}
        total_samples = 0
        for hla, peptides in self.hla_peptides.items():
            count = len(peptides)
            hla_distribution[hla] = count
            total_samples += count

        # 归一化为概率
        hla_probs = {hla: count / total_samples for hla, count in hla_distribution.items()}
        hlas = list(hla_probs.keys())
        probs = list(hla_probs.values())

        # 随机分配HLA（按正样本分布）
        assigned_hlas = np.random.choice(hlas, size=len(random_negatives), p=probs)

        print(f"  分配HLA数量: {len(set(assigned_hlas))}")

        # === 2. 跨HLA负样本 ===
        cross_hla_negatives_dict = self.generate_cross_hla_negatives()

        # === 3. 困难负样本 ===
        positive_samples_for_hard = []
        for hla in list(self.hla_peptides.keys())[:10]:
            peptides = list(self.hla_peptides[hla])
            n_sample = min(1000, len(peptides))
            sampled = random.sample(peptides, n_sample)
            positive_samples_for_hard.extend([(p, hla) for p in sampled])

        hard_negatives = self.generate_hard_negatives(positive_samples_for_hard, n_mutations=1)

        # === 4. 整合所有负样本 ===
        print(f"\n整合负样本...")

        all_negatives = []

        # 随机负样本（★ 现在有HLA了）
        for peptide, hla in zip(random_negatives, assigned_hlas):
            all_negatives.append({
                'Peptide': peptide,
                'Peptide_Length': len(peptide),
                'MHC_Restriction_Name': hla,  # ← 不再是PAN
                'Label': 0,
                'Source': 'random',
            })

        # 跨HLA负样本
        for hla, peptides in cross_hla_negatives_dict.items():
            for peptide in peptides:
                all_negatives.append({
                    'Peptide': peptide,
                    'Peptide_Length': len(peptide),
                    'MHC_Restriction_Name': hla,
                    'Label': 0,
                    'Source': 'cross_hla',
                })

        # 困难负样本
        for peptide, hla in hard_negatives:
            all_negatives.append({
                'Peptide': peptide,
                'Peptide_Length': len(peptide),
                'MHC_Restriction_Name': hla,
                'Label': 0,
                'Source': 'hard',
            })

        df_negatives = pd.DataFrame(all_negatives)

        print(f"\n总负样本: {len(df_negatives):,}")
        print(f"  Random: {(df_negatives['Source'] == 'random').sum():,}")
        print(f"  Cross-HLA: {(df_negatives['Source'] == 'cross_hla').sum():,}")
        print(f"  Hard: {(df_negatives['Source'] == 'hard').sum():,}")

        # ★ 验证没有PAN
        if 'PAN' in df_negatives['MHC_Restriction_Name'].values:
            print(f"\n⚠ 警告：仍有PAN标记")
        else:
            print(f"\n✓ 确认：所有负样本都有具体HLA")

        # === 5. 保存 ===
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / 'phase1_negatives.csv'
        df_negatives.to_csv(output_file, index=False)

        print(f"\n✓ 负样本已保存: {output_file}")

        return df_negatives

# ========================================================================
# 主程序
# ========================================================================

if __name__ == "__main__":
    # 初始化生成器
    generator = NegativeSampleGenerator(
        positive_data_dir='cleaned_data',
        proteome_file='human_proteome.fasta'
    )

    # 生成负样本
    df_negatives = generator.generate_all_negatives(
        ratio=5.0,  # 5:1的负正比例
        strategy_weights=(0.4, 0.4, 0.2),  # 40% random, 40% cross-HLA, 20% hard
        output_dir='negative_samples'
    )

    print("\n" + "=" * 80)
    print("Phase 1 负样本生成完成！")
    print("=" * 80)