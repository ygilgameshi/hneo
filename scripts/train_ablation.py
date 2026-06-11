#!/usr/bin/env python3
"""
消融实验训练脚本 — 复用 Mode 2 数据划分

变体:
  no_gnn      关闭 GNN 传播，用原始 node embedding
  no_film     关闭 FiLM，改用加性 tissue 融合
  no_sampling 关闭自适应任务均衡采样

用法:
  python scripts/train_ablation.py \
      --ablation no_gnn \
      --mode2_output_dir output/0227_mode2_proteome \
      --output_dir output/ablation_no_gnn
"""

import argparse
import json
from pathlib import Path
import sys
import torch
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.config.mode_config import create_mode2_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.enhanced_negative_sampler import (
    EnhancedNegativeSampler, load_protein_sequences
)
from src.data.negative_cache import NegativeSampleCache
from src.training.unified_trainer import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', required=True,
                        choices=['no_gnn', 'no_film', 'no_sampling'])
    parser.add_argument('--mode2_output_dir', required=True)
    parser.add_argument('--output_dir',        required=True)
    parser.add_argument('--hla_sequences', default='configs/hla_sequences.json')
    parser.add_argument('--proteome_file', default='data/human_proteome.fasta')
    parser.add_argument('--n_epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',    type=int,   default=256)
    parser.add_argument('--meta_lr',       type=float, default=1e-3)
    parser.add_argument('--negative_ratio',type=int,   default=10)
    parser.add_argument('--min_samples',   type=int,   default=10)
    parser.add_argument('--graph_threshold',type=float,default=0.3)
    parser.add_argument('--peptide_dim',   type=int,   default=64)
    parser.add_argument('--task_dim',      type=int,   default=64)
    parser.add_argument('--tissue_dim',    type=int,   default=32)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--save_negative_cache', action='store_true', default=True)
    args = parser.parse_args()

    use_gnn      = args.ablation != 'no_gnn'
    use_film     = args.ablation != 'no_film'
    use_sampling = args.ablation != 'no_sampling'

    print(f"\n{'='*60}")
    print(f"  Ablation : {args.ablation}")
    print(f"  use_gnn={use_gnn}  use_film={use_film}  "
          f"use_task_balanced={use_sampling}")
    print(f"{'='*60}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载 Mode 2 数据划分 ───────────────────────────────
    splits_dir = Path(args.mode2_output_dir) / 'data_splits'
    train_df = pd.read_csv(splits_dir / 'train.tsv', sep='\t')
    val_df   = pd.read_csv(splits_dir / 'val.tsv',   sep='\t')
    test_df  = pd.read_csv(splits_dir / 'test.tsv',  sep='\t')
    print(f"Splits: train={len(train_df):,}  val={len(val_df):,}  "
          f"test={len(test_df):,}")

    # ── 2. 配置 ──────────────────────────────────────────────
    config = create_mode2_config(
        min_samples=args.min_samples,
        negative_ratio=args.negative_ratio,
        meta_lr=args.meta_lr,
        use_maml=False,
    )

    # ── 3. 创建任务 ───────────────────────────────────────────
    creator = UnifiedTaskCreator(config)
    task_manager = creator.create_tasks(
        train_df,
        save_dir=output_dir / 'tasks'
    )
    print(f"Tasks: {len(task_manager.get_all_tasks())}")

    # ── 4. 负样本（优先缓存，回退到完整模型缓存，最后重新生成）
    cache = NegativeSampleCache('data/negative_samples')
    cache_config = {
        'negative_ratio': args.negative_ratio,
        'min_samples':    args.min_samples,
        'tissue_source':  'Host',
    }

    # 尝试直接读完整模型的缓存（所有消融变体负样本完全一致）
    cached = cache.load(args.mode2_output_dir, 'mode2', cache_config)

    if cached is not None:
        train_task_datasets, val_task_datasets, test_task_datasets = cached
        print(f"✓ Loaded negative samples from cache")
    else:
        print("Generating negative samples (no cache found)...")
        protein_seqs = {}
        if Path(args.proteome_file).exists():
            protein_seqs = load_protein_sequences(args.proteome_file)

        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        sampler = EnhancedNegativeSampler(
            config, all_df, protein_sequences=protein_seqs
        )
        train_task_datasets = sampler.generate_negatives_for_all_tasks(
            task_manager, source_data=train_df
        )
        val_task_datasets = sampler.generate_negatives_for_all_tasks(
            task_manager, source_data=val_df
        )
        test_task_datasets = sampler.generate_negatives_for_all_tasks(
            task_manager, source_data=test_df
        )
        if args.save_negative_cache:
            cache.save(args.mode2_output_dir, 'mode2', cache_config,
                       train_task_datasets, val_task_datasets, test_task_datasets)

    # ── 5. 训练 ───────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, history = train_model(
        task_manager=task_manager,
        task_datasets=train_task_datasets,
        mode_config=config,
        output_dir=output_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
        val_task_datasets=val_task_datasets,
        test_task_datasets=test_task_datasets,
        hla_sequences_file=args.hla_sequences,
        graph_threshold=args.graph_threshold,
        peptide_dim=args.peptide_dim,
        task_dim=args.task_dim,
        tissue_dim=args.tissue_dim,
        dropout=args.dropout,
        # ── 消融 flags ──
        use_gnn=use_gnn,
        use_film=use_film,
        use_task_balanced=use_sampling,
        sampling_strategy='adaptive',
        use_task_weighting=True,
        weight_smoothing=0.5,
        use_class_weighting=False,
    )

    print(f"\n✓ [{args.ablation}] done — "
          f"best val AUROC: {max(history['val_auroc']):.4f}")

    # 保存消融元信息
    with open(output_dir / 'ablation_info.json', 'w') as f:
        json.dump({
            'ablation': args.ablation,
            'use_gnn': use_gnn,
            'use_film': use_film,
            'use_task_balanced': use_sampling,
            'best_val_auroc': max(history['val_auroc']),
        }, f, indent=2)


if __name__ == '__main__':
    main()