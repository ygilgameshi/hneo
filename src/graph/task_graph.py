"""
Task Graph 构建器

构建任务图并计算任务相似性
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from Bio.Align.substitution_matrices import load
import networkx as nx
import pickle
from Bio import pairwise2
from tqdm import tqdm


class TaskGraphBuilder:
    """
    构建Task Graph

    Phase 1: HLA-level tasks
    - 每个HLA allele是一个task
    - Task之间的相似性通过HLA序列相似性定义
    """

    def __init__(self,
                 dataset_dir='phase1_dataset',
                 hla_sequences_file='hla_sequences.json'):
        """
        初始化

        Args:
            dataset_dir: 数据集目录
            hla_sequences_file: HLA伪序列文件
        """
        self.dataset_dir = Path(dataset_dir)
        self.hla_sequences_file = hla_sequences_file

        print("=" * 80)
        print("Task Graph构建器初始化")
        print("=" * 80)

        # 加载HLA列表
        self._load_hla_list()

        # 加载HLA序列
        self._load_hla_sequences()

        # 构建任务数据
        self._build_task_data()

    def _load_hla_list(self):
        """加载HLA列表"""
        print("\n加载HLA列表...")

        with open(self.dataset_dir / 'hla_list.json') as f:
            hla_info = json.load(f)

        self.all_hlas = hla_info['all_hlas']
        self.major_hlas = hla_info['major_hlas']

        print(f"  所有HLA: {len(self.all_hlas)}")
        print(f"  主要HLA（样本>100）: {len(self.major_hlas)}")

        # Phase 1使用主要HLA作为tasks
        self.tasks = self.major_hlas
        print(f"  Task数量: {len(self.tasks)}")

    def _load_hla_sequences(self):
        """
        加载HLA伪序列（pseudo-sequences）

        HLA伪序列：34个关键氨基酸位点，决定肽段结合特异性
        来源：NetMHCpan的定义
        """
        print("\n加载HLA伪序列...")

        if Path(self.hla_sequences_file).exists():
            print(f"  从文件加载: {self.hla_sequences_file}")
            with open(self.hla_sequences_file) as f:
                self.hla_sequences = json.load(f)
        else:
            print(f"  文件不存在，生成默认序列...")
            self.hla_sequences = self._generate_default_sequences()

            # 保存
            with open(self.hla_sequences_file, 'w') as f:
                json.dump(self.hla_sequences, f, indent=2)

            print(f"  ✓ 已保存到: {self.hla_sequences_file}")

        print(f"  HLA序列数: {len(self.hla_sequences)}")

    def _generate_default_sequences(self):
        """
        生成默认的HLA伪序列

        注意：这是简化版本
        真实的伪序列需要从IMGT/HLA数据库获取
        """
        sequences = {}

        # 34个关键位点（NetMHCpan定义）
        # 这里用随机序列模拟，实际应该用真实序列
        positions = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77,
                     80, 81, 84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152,
                     156, 158, 159, 163, 167, 171]

        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

        for hla in self.tasks:
            # 根据HLA名称生成"伪随机"序列（为了复现性）
            np.random.seed(hash(hla) % (2 ** 32))
            seq = ''.join(np.random.choice(list(amino_acids), 34))
            sequences[hla] = seq

        return sequences

    def _build_task_data(self):
        """构建每个任务的数据统计"""
        print("\n构建任务数据...")

        df_train = pd.read_csv(self.dataset_dir / 'train.csv')

        self.task_data = {}

        for task in tqdm(self.tasks, desc="统计任务数据"):
            task_df = df_train[df_train['MHC_Restriction_Name'] == task]

            n_samples = len(task_df)
            n_pos = (task_df['Label'] == 1).sum()
            n_neg = (task_df['Label'] == 0).sum()

            self.task_data[task] = {
                'n_samples': int(n_samples),
                'n_pos': int(n_pos),
                'n_neg': int(n_neg),
                'hla_type': task.split('*')[0],  # HLA-A, HLA-B, HLA-C
            }

        print(f"  ✓ 完成")

    # ========================================================================
    # Task相似性计算
    # ========================================================================

    def compute_sequence_similarity(self, seq1, seq2, method='blosum62'):
        """
        计算两个HLA序列的相似性

        Args:
            seq1, seq2: HLA伪序列
            method: 'blosum62' or 'identity'

        Returns:
            similarity score (0-1)
        """
        # 先校验两个序列长度一致
        if len(seq1) != len(seq2):
            raise ValueError("seq1 and seq2 must have the same length")

        if method == 'identity':
            # 简单的序列一致性
            matches = sum(a == b for a, b in zip(seq1, seq2))
            return matches / len(seq1)

        elif method == 'blosum62':
            # 加载BLOSUM62矩阵（关键修正点）
            matrix = load("BLOSUM62")

            # 执行全局比对
            alignments = pairwise2.align.globaldx(seq1, seq2, matrix)

            if len(alignments) == 0:
                return 0.0

            # 取最佳比对分数
            best_alignment = alignments[0]
            score = best_alignment[2]

            # 归一化到0-1：自身与自身比对的最大分数
            max_score = sum(matrix[a, a] for a in seq1)

            if max_score == 0:
                return 0.0

            normalized_score = max(0, min(1, score / max_score))

            return normalized_score
    def compute_hla_type_similarity(self, hla1, hla2):
        """
        基于HLA类型的相似性

        HLA-A vs HLA-A: 高相似性
        HLA-A vs HLA-B: 中等相似性
        HLA-A vs HLA-C: 低相似性
        """
        type1 = hla1.split('*')[0]  # HLA-A, HLA-B, HLA-C
        type2 = hla2.split('*')[0]

        if type1 == type2:
            return 1.0
        elif type1 in ['HLA-A', 'HLA-B'] and type2 in ['HLA-A', 'HLA-B']:
            return 0.5
        else:
            return 0.3

    def compute_data_similarity(self, task1, task2):
        """
        基于数据分布的相似性

        考虑：
        - 样本数量相似性（log scale）
        - 正负样本比例相似性
        """
        data1 = self.task_data[task1]
        data2 = self.task_data[task2]

        # ========== 1. 样本数量相似性（修复除以零问题） ==========
        n1 = np.log(data1['n_samples'] + 1)
        n2 = np.log(data2['n_samples'] + 1)
        max_n = max(n1, n2)
        if max_n == 0:
            # 两个任务样本数都为0，数量相似度为1
            size_sim = 1.0
        else:
            size_sim = 1 - abs(n1 - n2) / max_n

        # ========== 2. 正负比例相似性（修复除以零问题） ==========
        # 原始逻辑保留防除零（+1e-6），但仍需处理ratio1/ratio2都为0的情况
        ratio1 = data1['n_pos'] / (data1['n_neg'] + 1e-6)
        ratio2 = data2['n_pos'] / (data2['n_neg'] + 1e-6)
        max_ratio = max(ratio1, ratio2)
        if max_ratio == 0:
            # 两个任务正样本数都为0，比例相似度为1
            ratio_sim = 1.0
        else:
            ratio_sim = 1 - abs(ratio1 - ratio2) / max_ratio

        # ========== 3. 综合相似度（加权平均） ==========
        return 0.5 * size_sim + 0.5 * ratio_sim


    def compute_task_similarity(self, task1, task2, weights=(0.6, 0.2, 0.2)):
        """
        综合任务相似性

        Args:
            task1, task2: HLA names
            weights: (seq_weight, type_weight, data_weight)

        Returns:
            similarity (0-1)
        """
        seq_sim = self.compute_sequence_similarity(
            self.hla_sequences[task1],
            self.hla_sequences[task2],
            method='blosum62'
        )

        type_sim = self.compute_hla_type_similarity(task1, task2)

        data_sim = self.compute_data_similarity(task1, task2)

        # 加权组合
        total_sim = (weights[0] * seq_sim +
                     weights[1] * type_sim +
                     weights[2] * data_sim)

        return total_sim

    def build_similarity_matrix(self):
        """
        构建任务相似性矩阵

        Returns:
            numpy array: (n_tasks, n_tasks)
        """
        print("\n构建任务相似性矩阵...")

        n_tasks = len(self.tasks)
        similarity_matrix = np.zeros((n_tasks, n_tasks))

        for i in tqdm(range(n_tasks), desc="计算相似性"):
            for j in range(i, n_tasks):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.compute_task_similarity(
                        self.tasks[i],
                        self.tasks[j]
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        print(f"  ✓ 完成")
        print(f"  矩阵形状: {similarity_matrix.shape}")
        print(f"  平均相似性: {similarity_matrix[~np.eye(n_tasks, dtype=bool)].mean():.4f}")
        print(f"  最大相似性: {similarity_matrix[~np.eye(n_tasks, dtype=bool)].max():.4f}")
        print(f"  最小相似性: {similarity_matrix[~np.eye(n_tasks, dtype=bool)].min():.4f}")

        return similarity_matrix

    # ========================================================================
    # Task Graph构建
    # ========================================================================

    def build_task_graph(self, similarity_matrix, threshold=0.3):
        """
        基于相似性矩阵构建Task Graph

        Args:
            similarity_matrix: (n_tasks, n_tasks)
            threshold: 相似性阈值，低于此值不连边

        Returns:
            networkx.Graph
        """
        print(f"\n构建Task Graph（阈值: {threshold}）...")

        G = nx.Graph()

        # 添加节点
        for i, task in enumerate(self.tasks):
            G.add_node(i,
                       hla=task,
                       n_samples=self.task_data[task]['n_samples'],
                       n_pos=self.task_data[task]['n_pos'],
                       hla_type=self.task_data[task]['hla_type'])

        # 添加边
        n_edges = 0
        for i in range(len(self.tasks)):
            for j in range(i + 1, len(self.tasks)):
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    G.add_edge(i, j, weight=sim)
                    n_edges += 1

        print(f"  节点数: {G.number_of_nodes()}")
        print(f"  边数: {G.number_of_edges()}")
        print(f"  平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")

        # 检查连通性
        if nx.is_connected(G):
            print(f"  ✓ 图是连通的")
        else:
            components = list(nx.connected_components(G))
            print(f"  ⚠ 图不连通，有 {len(components)} 个连通分量")
            print(f"    最大分量大小: {len(max(components, key=len))}")

        return G

    def visualize_task_graph(self, G, output_file='task_graph.png'):
        """可视化 Task Graph"""
        print(f"\n可视化 Task Graph...")

        try:
            import matplotlib
            matplotlib.use('Agg')  # 无显示后端
            import matplotlib.pyplot as plt

            plt.figure(figsize=(15, 15))

            # 使用 spring layout
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

            # 节点颜色（按 HLA type）
            node_colors = []
            for node in G.nodes():
                hla_type = G.nodes[node]['hla_type']
                if hla_type == 'HLA-A':
                    node_colors.append('lightblue')
                elif hla_type == 'HLA-B':
                    node_colors.append('lightgreen')
                elif hla_type == 'HLA-C':
                    node_colors.append('lightyellow')
                else:
                    node_colors.append('lightgray')

            # 节点大小（按样本数）
            node_sizes = [G.nodes[node]['n_samples'] / 10 for node in G.nodes()]

            # 绘制
            nx.draw_networkx_nodes(G, pos,
                                   node_color=node_colors,
                                   node_size=node_sizes,
                                   alpha=0.7)

            nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)

            # 标签（只标注主要节点）
            labels = {}
            for node in G.nodes():
                if G.nodes[node]['n_samples'] > 1000:
                    labels[node] = G.nodes[node]['hla'].split('*')[1]

            nx.draw_networkx_labels(G, pos, labels, font_size=8)

            plt.title(f"Task Graph (Phase 1: HLA-level)\n"
                      f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}",
                      fontsize=14)
            plt.axis('off')
            plt.tight_layout()

            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  ✓ 保存到: {output_file}")

            plt.close()

        except Exception as e:
            print(f"  可视化失败: {e}")
            print(f"  提示：可能需要安装 matplotlib")

    def save_task_graph(self, G, similarity_matrix, output_dir=None):
        """保存 Task Graph 和相关数据"""
        print(f"\n保存 Task Graph...")

        if output_dir is None:
            output_dir = self.dataset_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存图结构
        nx.write_gpickle(G, output_dir / 'task_graph.pkl')

        # 保存相似性矩阵
        np.save(output_dir / 'similarity_matrix.npy', similarity_matrix)

        # 保存 task 映射
        task_mapping = {
            'tasks': self.tasks,
            'task_to_idx': {task: i for i, task in enumerate(self.tasks)},
            'idx_to_task': {i: task for i, task in enumerate(self.tasks)},
            'task_data': self.task_data,
        }

        with open(output_dir / 'task_mapping.json', 'w') as f:
            json.dump(task_mapping, f, indent=2)

        print(f"  ✓ Task Graph 已保存到: {output_dir}")

    def build_and_save_all(self, similarity_threshold=0.3, output_dir=None):
        """一键构建并保存所有内容"""
        print("=" * 80)
        print("开始构建 Task Graph")
        print("=" * 80)

        # 1. 构建相似性矩阵
        similarity_matrix = self.build_similarity_matrix()

        # 2. 构建 Task Graph
        G = self.build_task_graph(similarity_matrix, threshold=similarity_threshold)

        # 3. 可视化
        if output_dir:
            vis_file = Path(output_dir) / 'task_graph.png'
        else:
            vis_file = self.dataset_dir / 'task_graph.png'

        self.visualize_task_graph(G, output_file=vis_file)

        # 4. 保存
        self.save_task_graph(G, similarity_matrix, output_dir)

        print("\n" + "=" * 80)
        print("Task Graph 构建完成！")
        print("=" * 80)

        return G, similarity_matrix


# ========================================================================
# 主程序
# ========================================================================


# 主程序
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build Task Graph')
    parser.add_argument('--dataset_dir', type=str, default='.\data\phase1_dataset',
                        help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as dataset_dir)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Similarity threshold for graph edges')
    parser.add_argument('--hla_sequences', type=str, default='configs/hla_sequences.json',
                        help='HLA sequences file')

    args = parser.parse_args()

    builder = TaskGraphBuilder(
        dataset_dir=args.dataset_dir,
        hla_sequences_file=args.hla_sequences
    )

    G, similarity_matrix = builder.build_and_save_all(
        similarity_threshold=args.threshold,
        output_dir=args.output_dir
    )