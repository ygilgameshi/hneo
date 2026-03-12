"""
端到端训练脚本 - Mode 1 (HLA-only) - 改进版

完整流程:
1. 加载数据
2. 创建任务
3. 生成负样本
4. 构建Task Graph
5. 训练模型 (支持Task-Balanced + Class权重)
6. 评估和保存

改进点:
- 支持Task-Balanced采样 (adaptive/moderate/aggressive)
- 支持Class权重自动计算
- 支持Task权重 (log/sqrt/none)
- 支持负样本缓存
- 更好的数据划分策略
"""

import pandas as pd
import argparse
from pathlib import Path
import sys
import torch

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.config.mode_config import create_mode1_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.enhanced_negative_sampler import EnhancedNegativeSampler
from src.training.unified_trainer import train_model

def main(args):
    print("=" * 80)
    print("ImmuneApp Mode 1 (HLA-only) - End-to-End Training (Enhanced)")
    print("=" * 80)
    print(f"  Data file: {args.data_file}")
    print(f"  Training method: {'MAML' if args.use_maml else 'Standard'}")
    print(f"  Output dir: {args.output_dir}")

    # 显示平衡策略
    if args.use_task_balanced:
        print(f"\n  Task Balancing:")
        print(f"    Sampling strategy: {args.sampling_strategy}")
        print(f"    Task weighting: {args.use_task_weighting}")
        if args.use_task_weighting:
            print(f"    Weight smoothing: {args.weight_smoothing}")

    if args.use_class_weighting:
        print(f"\n  Class Weighting:")
        print(f"    Auto-computed from data")

    print("=" * 80)

    # ========== 1. 加载数据 ==========
    print("\n" + "=" * 80)
    print("Step 1: Loading Data")
    print("=" * 80)

    df = pd.read_csv(args.data_file, sep='\t')
    print(f"✓ Loaded {len(df):,} samples")


    # 重命名列
    df = df.rename(columns={
        'MHC_Restriction_Name': 'hla',
        'Peptide': 'peptide',
        'Label': 'label'
    })

    print(f"  Positive: {(df['label'] == 1).sum():,}")
    print(f"  Negative: {(df['label'] == 0).sum():,}")
    print(f"  HLA types: {df['hla'].nunique()}")


    # ========== 过滤小HLA组 (确保stratified split可行) ==========
    print(f"\n🧹 Filtering small HLA groups...")

    # 设置最小样本数阈值 (需要至少10个样本才能stratified split)
    min_samples_per_hla = max(10, args.min_samples)
    print(f"  Minimum samples per HLA: {min_samples_per_hla}")

    # 统计每个HLA的样本数
    hla_counts = df['hla'].value_counts()

    print(f"\n  Before filtering:")
    print(f"    Total samples: {len(df):,}")
    print(f"    HLA types: {len(hla_counts)}")
    print(f"    Sample range: {hla_counts.min()} - {hla_counts.max()}")

    # 找出样本数>=min_samples_per_hla的HLA
    valid_hlas = hla_counts[hla_counts >= min_samples_per_hla].index.tolist()

    # 过滤数据
    df_filtered = df[df['hla'].isin(valid_hlas)].copy()

    # 统计过滤效果
    removed_hlas = len(hla_counts) - len(valid_hlas)
    removed_samples = len(df) - len(df_filtered)

    print(f"\n  After filtering:")
    print(f"    Total samples: {len(df_filtered):,}")
    print(f"    HLA types: {len(valid_hlas)}")
    print(f"    Removed HLAs: {removed_hlas}")
    print(f"    Removed samples: {removed_samples:,}")

    # 更新df
    df = df_filtered

    # 验证现在所有HLA都>=min_samples_per_hla
    hla_counts_filtered = df['hla'].value_counts()
    print(f"\n  ✓ All HLAs now have ≥{min_samples_per_hla} samples")
    print(f"    Min: {hla_counts_filtered.min()}, Max: {hla_counts_filtered.max()}")

    # ========== Stratified Split (智能两阶段) ==========
    print(f"\n📊 Splitting dataset (80/10/10)...")
    from sklearn.model_selection import train_test_split

    # 第一次split: train vs temp (80% vs 20%)
    print(f"  First split strategy: HLA stratification")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['hla']
    )

    print(f"  ✓ Train/Temp split done: {len(train_df):,} / {len(temp_df):,}")

    # 第二次split: temp -> val vs test (50% vs 50%)
    # 关键: 检查temp_df中每个HLA的样本数
    temp_hla_counts = temp_df['hla'].value_counts()
    temp_min_count = temp_hla_counts.min()
    single_sample_hlas = (temp_hla_counts == 1).sum()

    print(f"\n  Second split (val/test from temp):")
    print(f"    Temp HLA types: {len(temp_hla_counts)}")
    print(f"    Min samples in temp: {temp_min_count}")
    print(f"    Single-sample HLAs in temp: {single_sample_hlas}")

    if temp_min_count >= 2:
        # 可以继续stratify
        print(f"    ✓ Using HLA stratification")
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['hla']
        )
    else:
        # 降级到random split
        print(f"    ⚠ Some HLAs have only 1 sample in temp")
        print(f"    ✓ Fallback to random split")
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42
        )

    print(f"\n✓ Data split:")
    print(f"  Train: {len(train_df):,} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"  Val:   {len(val_df):,} ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df) / len(df) * 100:.1f}%)")


    # ========== 2. 创建配置 ==========
    print("\n" + "=" * 80)
    print("Step 2: Creating Config")
    print("=" * 80)

    config = create_mode1_config(
        min_samples=args.min_samples,
        negative_ratio=args.negative_ratio,
        use_maml=args.use_maml,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        inner_steps=args.inner_steps
    )

    print(f"✓ Config created:")
    print(f"  Mode: {config.task_type_name}")
    print(f"  Min samples: {config.min_samples}")
    print(f"  Negative ratio: {config.negative_ratio}")
    print(f"  Use MAML: {config.use_maml}")
    if config.use_maml:
        print(f"    Inner LR: {config.inner_lr}")
        print(f"    Inner steps: {config.inner_steps}")
    print(f"  Meta LR: {config.meta_lr}")

    # ========== 3. 创建任务 ==========
    print("\n" + "=" * 80)
    print("Step 3: Creating Tasks")
    print("=" * 80)

    creator = UnifiedTaskCreator(config)
    task_manager = creator.create_tasks(
        train_df,
        save_dir=Path(args.output_dir) / 'tasks'
    )

    n_tasks = len(task_manager.get_all_tasks())
    print(f"✓ Created {n_tasks} tasks (HLA alleles)")

    # 任务统计
    task_sizes = []
    for task in task_manager.get_all_tasks().values():
        task_data = train_df[train_df['hla'] == task.hla]
        task_sizes.append(len(task_data[task_data['label'] == 1]))

    import numpy as np
    print(f"\n  Positive sample distribution across tasks:")
    print(f"    Min:    {min(task_sizes)}")
    print(f"    25%:    {int(np.percentile(task_sizes, 25))}")
    print(f"    Median: {int(np.median(task_sizes))}")
    print(f"    75%:    {int(np.percentile(task_sizes, 75))}")
    print(f"    Max:    {max(task_sizes)}")

    # ========== 4. 生成负样本 (支持缓存) ==========
    print("\n" + "=" * 80)
    print("Step 4: Generating Negative Samples (Enhanced Strategy)")
    print("=" * 80)

    # 导入缓存管理器
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.negative_cache import NegativeSampleCache

    # 创建缓存管理器
    cache = NegativeSampleCache('data/negative_samples')

    # 准备缓存配置
    cache_config = {
        'negative_ratio': args.negative_ratio,
        'min_samples': args.min_samples
    }

    # 获取所有tasks
    all_tasks = task_manager.get_all_tasks()

    # 尝试加载缓存 (如果启用)
    cached_data = None
    if args.use_negative_cache:
        cached_data = cache.load(args.data_file, 'mode1', cache_config)

    if cached_data is not None:
        # 使用缓存的数据
        train_task_datasets, val_task_datasets, test_task_datasets = cached_data
        print(f"\n🚀 Using cached negative samples!")
    else:
        # 生成新的负样本
        print(f"\n⚙️  Generating new negative samples...")

        # 创建Enhanced Negative Sampler

        sampler = EnhancedNegativeSampler(config, train_df)

        # === Train ===
        print("\n  Generating train negatives...")
        train_task_datasets = sampler.generate_negatives_for_all_tasks(task_manager)

        # === Val ===
        print("\n  Generating val negatives...")
        val_sampler = EnhancedNegativeSampler(config, val_df)
        val_task_datasets = {}
        for task_id, task in all_tasks.items():
            # Mode 1: 只筛选HLA（不需要tissue）
            task_val_df = val_df[val_df['hla'] == task.hla]

            if len(task_val_df) > 0:
                positive_peptides = list(task_val_df['peptide'])
                negative_peptides = val_sampler.generate_negatives_for_task(
                    task,
                    positive_peptides
                )

                # 构建负样本DataFrame
                neg_df = pd.DataFrame({
                    'peptide': negative_peptides,
                    'hla': task.hla,
                    'label': 0
                })

                # 合并正负样本
                task_val_combined = pd.concat([task_val_df, neg_df], ignore_index=True)
                val_task_datasets[task_id] = task_val_combined

        # === Test ===
        print("\n  Generating test negatives...")
        test_sampler = EnhancedNegativeSampler(config, test_df)
        test_task_datasets = {}
        for task_id, task in all_tasks.items():
            # Mode 1: 只筛选HLA（不需要tissue）
            task_test_df = test_df[test_df['hla'] == task.hla]

            if len(task_test_df) > 0:
                positive_peptides = list(task_test_df['peptide'])
                negative_peptides = test_sampler.generate_negatives_for_task(
                    task,
                    positive_peptides
                )

                # 构建负样本DataFrame
                neg_df = pd.DataFrame({
                    'peptide': negative_peptides,
                    'hla': task.hla,
                    'label': 0
                })

                # 合并正负样本
                task_test_combined = pd.concat([task_test_df, neg_df], ignore_index=True)
                test_task_datasets[task_id] = task_test_combined

        # 保存到缓存 (如果启用)
        if args.save_negative_cache:
            cache.save(
                args.data_file,
                'mode1',
                cache_config,
                train_task_datasets,
                val_task_datasets,
                test_task_datasets
            )
            print(f"\n💾 Cached negative samples for future use")

    print(f"\n✓ Negative samples ready for all splits")

    # ========== 5. 训练模型 ==========
    print("\n" + "=" * 80)
    print("Step 5: Training Model")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    model, history = train_model(
        task_manager=task_manager,
        task_datasets=train_task_datasets,
        mode_config=config,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
        val_task_datasets=val_task_datasets,
        test_task_datasets=test_task_datasets,
        hla_sequences_file=args.hla_sequences,
        graph_threshold=args.graph_threshold,
        peptide_dim=args.peptide_dim,
        task_dim=args.task_dim,
        dropout=args.dropout,
        # === Task-Balanced参数 ===
        use_task_balanced=args.use_task_balanced,
        sampling_strategy=args.sampling_strategy,
        use_task_weighting=args.use_task_weighting,
        weight_smoothing=args.weight_smoothing,
        # === Class权重参数 ===
        use_class_weighting=args.use_class_weighting
    )

    # ========== 6. 最终评估 ==========
    print("\n" + "=" * 80)
    print("Step 6: Final Evaluation")
    print("=" * 80)

    print(f"\n✓ Training completed!")
    print(f"\n  Training History:")
    print(f"    Best Val AUROC: {max(history['val_auroc']):.4f}")
    print(f"    Best Val AUPRC: {max(history['val_auprc']):.4f}")
    print(f"    Final Val AUROC: {history['val_auroc'][-1]:.4f}")
    print(f"    Final Val AUPRC: {history['val_auprc'][-1]:.4f}")

    # 如果使用Task-Balanced，显示统计
    if args.use_task_balanced:
        print(f"\n  Task Balancing Statistics:")
        print(f"    Strategy: {args.sampling_strategy}")
        print(f"    Task weighting: {'Yes' if args.use_task_weighting else 'No'}")
        if args.use_task_weighting:
            print(f"    Weight smoothing: {args.weight_smoothing}")

    if args.use_class_weighting:
        print(f"\n  Class Weighting: Enabled")
        print(f"    Automatically computed from data")

    print(f"\n✓ All results saved to: {args.output_dir}")
    print(f"  - best_model.pt             (Best model checkpoint)")
    print(f"  - training_results.json     (Training history)")
    print(f"  - task_graph/               (Task graph structure)")
    print(f"  - tasks/                    (Task definitions)")
    if args.use_negative_cache:
        print(f"  - negative_cache/           (Negative sample cache)")

    print("\n" + "=" * 80)
    print("Training finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ImmuneApp Mode 1 End-to-End Training (Enhanced)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========== 数据参数 ==========
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--data_file', type=str, required=True,
                            help='Path to data file (TSV format)')
    data_group.add_argument('--hla_sequences', type=str, default='configs/hla_sequences.json',
                            help='Path to HLA sequences file')
    data_group.add_argument('--output_dir', type=str, default='output/mode1_experiment',
                            help='Output directory')

    # ========== 任务创建参数 ==========
    task_group = parser.add_argument_group('Task Creation Parameters')
    task_group.add_argument('--min_samples', type=int, default=10,
                            help='Minimum samples per task')
    task_group.add_argument('--negative_ratio', type=int, default=25,
                            help='Negative:Positive ratio (推荐25)')

    # ========== 训练方法 ==========
    method_group = parser.add_argument_group('Training Method')
    method_group.add_argument('--use_maml', action='store_true',
                              help='Use MAML training (default: Standard)')
    method_group.add_argument('--inner_lr', type=float, default=0.01,
                              help='MAML inner learning rate')
    method_group.add_argument('--inner_steps', type=int, default=5,
                              help='MAML inner adaptation steps')
    method_group.add_argument('--meta_lr', type=float, default=0.001,
                              help='Meta/standard learning rate')

    # ========== 训练参数 ==========
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--n_epochs', type=int, default=50,
                             help='Number of epochs')
    train_group.add_argument('--batch_size', type=int, default=256,
                             help='Batch size (推荐256)')

    # ========== Task-Balanced参数 ==========
    balance_group = parser.add_argument_group('Task Balancing Parameters')
    balance_group.add_argument('--use_task_balanced', action='store_true',
                               help='Use Task-Balanced sampling')
    balance_group.add_argument('--sampling_strategy', type=str, default='adaptive',
                               choices=['adaptive', 'moderate', 'aggressive'],
                               help='Sampling strategy: adaptive(推荐), moderate(温和), aggressive(激进)')
    balance_group.add_argument('--use_task_weighting', action='store_true',
                               help='Use task-weighted loss')
    balance_group.add_argument('--weight_smoothing', type=str, default='log',
                               choices=['none', 'sqrt', 'log'],
                               help='Loss weight smoothing: log(推荐), sqrt(更激进), none(最激进)')

    # ========== Class权重参数 ==========
    class_group = parser.add_argument_group('Class Weighting Parameters')
    class_group.add_argument('--use_class_weighting', action='store_true',
                             help='Use class weights for positive/negative imbalance (推荐启用)')

    # ========== 负样本参数 ==========
    negative_group = parser.add_argument_group('Negative Sampling Parameters')
    negative_group.add_argument('--use_negative_cache', action='store_true',
                                help='Try to load cached negative samples')
    negative_group.add_argument('--save_negative_cache', action='store_true',
                                help='Save generated negative samples to cache')

    # ========== Graph参数 ==========
    graph_group = parser.add_argument_group('Task Graph Parameters')
    graph_group.add_argument('--graph_threshold', type=float, default=0.3,
                             help='Task graph similarity threshold')

    # ========== 模型参数 ==========
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--peptide_dim', type=int, default=64,
                             help='Peptide encoder output dimension')
    model_group.add_argument('--task_dim', type=int, default=64,
                             help='Task GNN output dimension')
    model_group.add_argument('--dropout', type=float, default=0.1,
                             help='Dropout rate')

    args = parser.parse_args()

    main(args)