"""
Enhanced Negative Sampler - 支持基于已表达peptide的采样

新策略:
- Strategy A: Random from expressed peptides (从数据集中已有peptides随机采样)
- Strategy B: Cross-HLA within tissue (同tissue不同HLA的阳性)
- Strategy C: Cross-tissue within HLA (同HLA不同tissue的阳性)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from .task_definition import Task, TaskManager
from ..config.mode_config import ModeConfig, TrainingMode


class EnhancedNegativeSampler:
    """
    增强版负样本生成器
    
    核心改进: 所有负样本都来自**已在数据集中表达的peptides**
    - 不需要外部表达谱数据
    - 生物学上更合理 (这些peptides必定来自表达的蛋白)
    """
    
    def __init__(self, 
                 config: ModeConfig,
                 all_data: pd.DataFrame):
        """
        Args:
            config: ModeConfig
            all_data: 全部数据 (用于构建expressed peptide pool)
        """
        self.config = config
        self.mode = config.mode
        self.all_data = all_data
        
        # ========== 构建Expressed Peptide Pool ==========
        # 核心创新: 只使用数据集中出现过的peptides
        self.expressed_peptides = list(all_data['peptide'].unique())
        
        # 按长度分组 (采样时保持长度分布)
        self.peptides_by_length = defaultdict(list)
        for pep in self.expressed_peptides:
            self.peptides_by_length[len(pep)].append(pep)
        
        # ========== 构建HLA-specific peptide映射 ==========
        self._build_hla_peptide_mapping()
        
        # ========== Mode 2: 构建tissue-aware映射 ==========
        if self.mode == TrainingMode.HLA_TISSUE:
            self._build_tissue_aware_mappings()
        
        print(f"\n✓ Enhanced Negative Sampler initialized")
        print(f"  Mode: {config.task_type_name}")
        print(f"  Expressed peptides: {len(self.expressed_peptides)}")
        print(f"  Peptide length range: {min(self.peptides_by_length.keys())}-{max(self.peptides_by_length.keys())}")
        print(f"  Negative ratio: 1:{config.negative_ratio}")
        if self.mode == TrainingMode.HLA_TISSUE:
            print(f"  Tissue-aware negatives: {config.use_tissue_aware_negatives}")
    
    def _build_hla_peptide_mapping(self):
        """构建HLA -> positive peptides映射"""
        self.hla_positive_peptides = defaultdict(set)
        
        positive_data = self.all_data[self.all_data['label'] == 1]
        for hla, group in positive_data.groupby('hla'):
            self.hla_positive_peptides[hla] = set(group['peptide'])
        
        print(f"  HLA types: {len(self.hla_positive_peptides)}")
    
    def _build_tissue_aware_mappings(self):
        """构建tissue-aware映射 (Mode 2)"""
        if 'tissue' not in self.all_data.columns:
            print("  ⚠ No tissue column found, tissue-aware sampling disabled")
            self.tissue_positive_peptides = {}
            self.hla_tissue_positive_peptides = {}
            return
        
        positive_data = self.all_data[self.all_data['label'] == 1]
        
        # Tissue -> peptides
        self.tissue_positive_peptides = defaultdict(set)
        for tissue, group in positive_data.groupby('tissue'):
            self.tissue_positive_peptides[tissue] = set(group['peptide'])
        
        # (HLA, Tissue) -> peptides
        self.hla_tissue_positive_peptides = defaultdict(set)
        for (hla, tissue), group in positive_data.groupby(['hla', 'tissue']):
            self.hla_tissue_positive_peptides[(hla, tissue)] = set(group['peptide'])
        
        print(f"  Tissues: {len(self.tissue_positive_peptides)}")
        print(f"  HLA×Tissue combinations: {len(self.hla_tissue_positive_peptides)}")
    
    def _sample_random_expressed(self, 
                                 n: int, 
                                 exclude: set,
                                 length_dist: Optional[Dict[int, int]] = None) -> List[str]:
        """
        Strategy A: 从expressed peptides随机采样
        
        Args:
            n: 采样数量
            exclude: 需要排除的peptides
            length_dist: 长度分布 {length: count} (保持与正样本相似的长度分布)
        
        Returns:
            sampled_peptides
        """
        negatives = []
        
        if length_dist is None:
            # 简单随机采样
            candidates = [p for p in self.expressed_peptides if p not in exclude]
            if len(candidates) >= n:
                negatives = list(np.random.choice(candidates, n, replace=False))
            else:
                negatives = candidates
        else:
            # 按长度分布采样
            for length, count in length_dist.items():
                if length not in self.peptides_by_length:
                    continue
                
                candidates = [p for p in self.peptides_by_length[length] 
                             if p not in exclude]
                
                if len(candidates) > 0:
                    n_sample = min(count, len(candidates))
                    sampled = list(np.random.choice(candidates, n_sample, replace=False))
                    negatives.extend(sampled)
        
        return negatives
    
    def _sample_cross_hla_within_tissue(self, 
                                        task: Task, 
                                        n: int, 
                                        exclude: set) -> List[str]:
        """
        Strategy B1: Cross-HLA within tissue
        
        从同一tissue的其他HLA的阳性样本中采样
        (这些peptides在该tissue中被其他HLA呈递,但不被当前HLA呈递)
        """
        if not hasattr(self, 'hla_tissue_positive_peptides'):
            return []
        
        tissue = task.tissue
        current_hla = task.hla
        
        # 找到同tissue其他HLA的阳性peptides
        candidates = set()
        for (hla, tis), peptides in self.hla_tissue_positive_peptides.items():
            if tis == tissue and hla != current_hla:
                candidates.update(peptides)
        
        # 排除当前task的阳性peptides
        candidates = candidates - exclude
        
        if len(candidates) == 0:
            return []
        
        n_sample = min(n, len(candidates))
        return list(np.random.choice(list(candidates), n_sample, replace=False))
    
    def _sample_cross_tissue_within_hla(self, 
                                        task: Task, 
                                        n: int, 
                                        exclude: set) -> List[str]:
        """
        Strategy B2: Cross-tissue within HLA
        
        从其他tissue的同一HLA的阳性样本中采样
        (这些peptides被该HLA在其他tissue呈递,但不在当前tissue呈递)
        """
        if not hasattr(self, 'hla_tissue_positive_peptides'):
            return []
        
        current_hla = task.hla
        current_tissue = task.tissue
        
        # 找到同HLA其他tissue的阳性peptides
        candidates = set()
        for (hla, tissue), peptides in self.hla_tissue_positive_peptides.items():
            if hla == current_hla and tissue != current_tissue:
                candidates.update(peptides)
        
        # 排除当前task的阳性peptides
        candidates = candidates - exclude
        
        if len(candidates) == 0:
            return []
        
        n_sample = min(n, len(candidates))
        return list(np.random.choice(list(candidates), n_sample, replace=False))
    
    def _sample_cross_hla(self, task: Task, n: int, exclude: set) -> List[str]:
        """
        从其他HLA的阳性peptides采样 (Mode 1)
        """
        current_hla = task.hla
        
        # 其他HLA的阳性peptides
        candidates = set()
        for hla, peptides in self.hla_positive_peptides.items():
            if hla != current_hla:
                candidates.update(peptides)
        
        candidates = candidates - exclude
        
        if len(candidates) == 0:
            return []
        
        n_sample = min(n, len(candidates))
        return list(np.random.choice(list(candidates), n_sample, replace=False))
    
    def generate_negatives_for_task(self, 
                                    task: Task,
                                    positive_peptides: List[str],
                                    n_negatives: Optional[int] = None) -> List[str]:
        """
        为task生成负样本
        
        策略分配:
        - Mode 1: 50% random expressed + 50% cross-HLA
        - Mode 2 (不用tissue-aware): 50% random + 50% cross-HLA
        - Mode 2 (用tissue-aware): 40% random + 30% cross-HLA-within-tissue + 30% cross-tissue-within-HLA
        
        Args:
            task: Task对象
            positive_peptides: 该task的阳性peptides
            n_negatives: 负样本数量 (默认用config.negative_ratio)
        
        Returns:
            negative_peptides
        """
        if n_negatives is None:
            n_negatives = len(positive_peptides) * self.config.negative_ratio
        
        exclude = set(positive_peptides)
        negatives = []
        
        # 计算长度分布 (用于保持采样的长度分布)
        length_dist = {}
        for pep in positive_peptides:
            length = len(pep)
            length_dist[length] = length_dist.get(length, 0) + 1
        
        # 按比例放大到n_negatives
        total_pos = len(positive_peptides)
        for length in length_dist:
            length_dist[length] = int(length_dist[length] / total_pos * n_negatives)
        
        # ========== Mode 1: HLA-only ==========
        if self.mode == TrainingMode.HLA_ONLY:
            n_random = n_negatives // 2
            n_cross_hla = n_negatives - n_random
            
            # 50% random expressed
            neg_random = self._sample_random_expressed(n_random, exclude, length_dist)
            negatives.extend(neg_random)
            exclude.update(neg_random)
            
            # 50% cross-HLA
            neg_cross = self._sample_cross_hla(task, n_cross_hla, exclude)
            negatives.extend(neg_cross)
        
        # ========== Mode 2: HLA×Tissue ==========
        elif self.mode == TrainingMode.HLA_TISSUE:
            
            if not self.config.use_tissue_aware_negatives:
                # 不用tissue-aware: 50% random + 50% cross-HLA
                n_random = n_negatives // 2
                n_cross = n_negatives - n_random
                
                neg_random = self._sample_random_expressed(n_random, exclude, length_dist)
                negatives.extend(neg_random)
                exclude.update(neg_random)
                
                neg_cross = self._sample_cross_hla(task, n_cross, exclude)
                negatives.extend(neg_cross)
            
            else:
                # Tissue-aware: 40% random + 30% cross-HLA-tissue + 30% cross-tissue-HLA
                n_random = int(n_negatives * 0.4)
                n_cross_hla_tissue = int(n_negatives * 0.3)
                n_cross_tissue_hla = n_negatives - n_random - n_cross_hla_tissue
                
                # 40% random expressed
                neg_random = self._sample_random_expressed(n_random, exclude, length_dist)
                negatives.extend(neg_random)
                exclude.update(neg_random)
                
                # 30% cross-HLA within tissue
                neg_cross_hla = self._sample_cross_hla_within_tissue(
                    task, n_cross_hla_tissue, exclude
                )
                negatives.extend(neg_cross_hla)
                exclude.update(neg_cross_hla)
                
                # 30% cross-tissue within HLA
                neg_cross_tissue = self._sample_cross_tissue_within_hla(
                    task, n_cross_tissue_hla, exclude
                )
                negatives.extend(neg_cross_tissue)
        
        # 如果采样不足,用random填充
        if len(negatives) < n_negatives:
            n_fill = n_negatives - len(negatives)
            exclude.update(negatives)
            fill = self._sample_random_expressed(n_fill, exclude)
            negatives.extend(fill)
        
        return negatives[:n_negatives]
    
    def generate_negatives_for_all_tasks(self, 
                                         task_manager: TaskManager) -> Dict[str, pd.DataFrame]:
        """
        为所有tasks生成负样本
        
        Returns:
            task_datasets: {task_id: DataFrame(peptide, hla, [tissue], label)}
        """
        print(f"\n{'='*80}")
        print(f"Generating Negatives for All Tasks")
        print(f"{'='*80}")
        
        all_tasks = task_manager.get_all_tasks()
        task_datasets = {}
        
        for task_id, task in all_tasks.items():
            # 获取该task的正样本
            if self.mode == TrainingMode.HLA_ONLY:
                task_positives = self.all_data[
                    (self.all_data['hla'] == task.hla) &
                    (self.all_data['label'] == 1)
                ]
            else:  # Mode 2
                task_positives = self.all_data[
                    (self.all_data['hla'] == task.hla) &
                    (self.all_data['tissue'] == task.tissue) &
                    (self.all_data['label'] == 1)
                ]
            
            if len(task_positives) == 0:
                continue
            
            positive_peptides = list(task_positives['peptide'])
            
            # 生成负样本peptides
            negative_peptides = self.generate_negatives_for_task(
                task, positive_peptides
            )
            
            # 构建负样本DataFrame
            neg_df = pd.DataFrame({
                'peptide': negative_peptides,
                'hla': task.hla,
                'label': 0
            })
            
            if self.mode == TrainingMode.HLA_TISSUE:
                neg_df['tissue'] = task.tissue
            
            # 合并正负样本
            pos_df = task_positives.copy()
            task_df = pd.concat([pos_df, neg_df], ignore_index=True)
            
            task_datasets[task_id] = task_df
        
        print(f"\n✓ Generated negatives for {len(task_datasets)} tasks")
        
        # 统计
        total_samples = sum(len(df) for df in task_datasets.values())
        total_pos = sum((df['label']==1).sum() for df in task_datasets.values())
        total_neg = sum((df['label']==0).sum() for df in task_datasets.values())
        
        print(f"  Total samples: {total_samples:,}")
        print(f"  Positive: {total_pos:,} ({total_pos/total_samples*100:.1f}%)")
        print(f"  Negative: {total_neg:,} ({total_neg/total_samples*100:.1f}%)")
        print(f"  Avg negative ratio: 1:{total_neg/total_pos:.1f}")
        
        return task_datasets


# 向后兼容
NegativeSampler = EnhancedNegativeSampler


if __name__ == "__main__":
    print("Enhanced Negative Sampler - 基于已表达peptides的负样本生成")
    print("\n核心优势:")
    print("  ✓ 所有负样本来自数据集中已表达的peptides")
    print("  ✓ 不需要外部表达谱数据")
    print("  ✓ 生物学上更合理")
    print("  ✓ 支持tissue-aware采样 (Mode 2)")
