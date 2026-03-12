"""
Mode 1训练脚本 - 使用Mode 2的数据划分

关键特性:
- 加载Mode 2保存的train/val/test数据划分
- 训练Mode 1模型（HLA-only，忽略tissue信息）
- 确保与Mode 2使用完全相同的数据
- 作为Mode 2的ablation study

用法:
python scripts/train_mode1_with_mode2_splits.py \
    --mode2_output_dir output/task_balanced_v2 \
    --output_dir output/mode1_ablation \
    --n_epochs 85 \
    --batch_size 256
"""

import pandas as pd
import argparse
from pathlib import Path
import sys
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.config.mode_config import create_mode1_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.enhanced_negative_sampler import EnhancedNegativeSampler, load_protein_sequences
from src.data.negative_cache import NegativeSampleCache
from src.training.unified_trainer import train_model


def load_mode2_splits(mode2_output_dir):
    """从Mode 2输出目录加载数据划分"""

    splits_dir = Path(mode2_output_dir) / 'data_splits'

    if not splits_dir.exists():
        raise FileNotFoundError(
            f"Data splits not found: {splits_dir}\n"
            f"Make sure Mode 2 training has saved data splits."
        )

    print(f"\n📂 Loading Mode 2 data splits from: {splits_dir}")

    # 加载三个数据集
    train_df = pd.read_csv(splits_dir / 'train.tsv', sep='\t')
    val_df = pd.read_csv(splits_dir / 'val.tsv', sep='\t')
    test_df = pd.read_csv(splits_dir / 'test.tsv', sep='\t')

    print(f"✓ Loaded splits:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")

    # 重命名列（如果需要）
    for df in [train_df, val_df, test_df]:
        if 'MHC_Restriction_Name' in df.columns:
            df.rename(columns={
                'MHC_Restriction_Name': 'hla',
                'Peptide': 'peptide',
                'Label': 'label'
            }, inplace=True)

    # 检查tissue列
    tissue_col = None
    for col in ['tissue', 'Host', 'Tissue']:
        if col in train_df.columns:
            tissue_col = col
            print(f"  Tissue column found: '{tissue_col}'")
            print(f"  Tissues: {train_df[tissue_col].nunique()}")
            break

    if tissue_col:
        print(f"\n  ⚠️  Tissue information will be IGNORED (Mode 1 is HLA-only)")

    return train_df, val_df, test_df, tissue_col


def main(args):
    print("="*80)
    print("Mode 1 Training with Mode 2 Data Splits (Ablation Study)")
    print("="*80)
    print(f"  Mode 2 output: {args.mode2_output_dir}")
    print(f"  Mode 1 output: {args.output_dir}")
    print(f"  Training method: {'MAML' if args.use_maml else 'Standard'}")

    # 显示平衡策略
    if args.use_task_balanced:
        print(f"\n  Task Balancing:")
        print(f"    Strategy: {args.sampling_strategy}")
        print(f"    Task weighting: {args.use_task_weighting}")
        if args.use_task_weighting:
            print(f"    Weight smoothing: {args.weight_smoothing}")

    if args.use_class_weighting:
        print(f"\n  Class Weighting: Enabled")

    print("="*80)

    # ========== 1. 加载Mode 2的数据划分 ==========
    print(f"\n{'='*80}")
    print("Step 1: Loading Mode 2 Data Splits")
    print(f"{'='*80}")

    train_df, val_df, test_df, tissue_col = load_mode2_splits(args.mode2_output_dir)

    # 合并所有数据（用于创建完整的task列表）
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(f"\n  Total unique HLAs: {all_df['hla'].nunique()}")
    print(f"  Total samples: {len(all_df):,}")
    print(f"  Positive: {(all_df['label'] == 1).sum():,}")
    print(f"  Negative: {(all_df['label'] == 0).sum():,}")

    # ========== 2. 创建Mode 1配置 ==========
    print(f"\n{'='*80}")
    print("Step 2: Creating Mode 1 Config")
    print(f"{'='*80}")

    config = create_mode1_config(
        min_samples=1,  # 不过滤，使用Mode 2的所有HLA
        negative_ratio=args.negative_ratio,
        use_maml=args.use_maml,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        inner_steps=args.inner_steps
    )

    print(f"✓ Mode 1 config created")
    print(f"  Mode: {config.mode}")
    print(f"  Negative ratio: {config.negative_ratio}")

    # ========== 3. 创建Tasks ==========
    print(f"\n{'='*80}")
    print("Step 3: Creating Tasks (HLA-only)")
    print(f"{'='*80}")

    creator = UnifiedTaskCreator(config)
    task_manager = creator.create_tasks(all_df)

    print(f"✓ Created {len(task_manager.get_all_tasks())} tasks")

    # ========== 4. 生成/加载负样本 ==========
    print(f"\n{'='*80}")
    print("Step 4: Generating / Loading Negative Samples")
    print(f"{'='*80}")

    cache = NegativeSampleCache('data/negative_samples')
    cache_config = {
        'negative_ratio':             args.negative_ratio,
        'use_tissue_aware_negatives': False,
        'min_samples':                1,
        'tissue_source':              'Host',
    }

    cached_data = None
    if args.use_negative_cache:
        cached_data = cache.load(args.mode2_output_dir, 'mode1', cache_config)

    if cached_data is not None:
        train_task_datasets, val_task_datasets, test_task_datasets = cached_data
        print(f"\n🚀 Using cached negative samples!")
        print(f"  Train tasks: {len(train_task_datasets)}")
        print(f"  Val tasks:   {len(val_task_datasets)}")
        print(f"  Test tasks:  {len(test_task_datasets)}")
    else:
        # ── 加载蛋白质组 ──
        protein_sequences = None
        if args.proteome_file:
            from pathlib import Path as _Path
            if _Path(args.proteome_file).exists():
                print(f"\n📂 Loading proteome sequences: {args.proteome_file}")
                protein_sequences = load_protein_sequences(args.proteome_file)
            else:
                print(f"  ⚠️  Proteome file not found: {args.proteome_file}")
                print(f"      负样本将全部使用 fallback 采样")

        # ── 统一 sampler：exclude set = 全量数据，防假阴性 ──
        all_df_for_sampler = pd.concat([train_df, val_df, test_df], ignore_index=True)
        sampler = EnhancedNegativeSampler(
            config, all_df_for_sampler,
            protein_sequences=protein_sequences
        )

        def generate_negatives_for_split(split_df, split_name):
            print(f"\n  Processing {split_name}...")

            # source_data 限定为当前 split，exclude set 来自全量
            split_datasets = sampler.generate_negatives_for_all_tasks(
                task_manager, source_data=split_df
            )

            # 保留 tissue 列（Mode 1 训练忽略但保持数据完整性）
            if tissue_col:
                for task_id, task_df in split_datasets.items():
                    if tissue_col not in task_df.columns:
                        task_df[tissue_col] = None

            print(f"  ✓ {split_name}: {len(split_datasets)} tasks, "
                  f"{sum(len(df) for df in split_datasets.values()):,} samples")
            return split_datasets

        train_task_datasets = generate_negatives_for_split(train_df, "Train")
        val_task_datasets   = generate_negatives_for_split(val_df,   "Val")
        test_task_datasets  = generate_negatives_for_split(test_df,  "Test")

        # ── 可选：保存缓存 ──
        if args.save_negative_cache:
            cache.save(args.mode2_output_dir, 'mode1', cache_config,
                       train_task_datasets, val_task_datasets, test_task_datasets)
            print(f"\n💾 Negative samples cached to: data/negative_samples/")

    # ========== 5. 训练模型 ==========
    print(f"\n{'='*80}")
    print("Step 5: Training Mode 1 Model")
    print(f"{'='*80}")

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
        # Task-Balanced参数
        use_task_balanced=args.use_task_balanced,
        sampling_strategy=args.sampling_strategy,
        use_task_weighting=args.use_task_weighting,
        weight_smoothing=args.weight_smoothing,
        # Class权重参数
        use_class_weighting=args.use_class_weighting
    )

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"  Model saved to: {args.output_dir}")
    print(f"  Best val AUROC: {max(history['val_auroc']):.4f}")

    # ========== 6. 保存数据划分 & 说明文件 ==========
    print(f"\n{'='*80}")
    print("Step 6: Saving Data Splits")
    print(f"{'='*80}")

    output_dir = Path(args.output_dir)
    splits_dir = output_dir / 'data_splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ── 6a. 保存纯正样本 TSV（与 Mode 2 data_splits 格式一致）──
    train_df.to_csv(splits_dir / 'train.tsv', sep='\t', index=False)
    val_df.to_csv(splits_dir / 'val.tsv',   sep='\t', index=False)
    test_df.to_csv(splits_dir / 'test.tsv',  sep='\t', index=False)
    print(f"✓ Positive-only splits saved to: {splits_dir}")

    # ── 6b. 保存带负样本的 TSV（供 evaluate_mixmhcpred --test_with_negatives 使用）──
    if args.save_splits_with_negatives:
        print(f"\n  Saving splits with negatives...")

        def _save_with_negatives(task_datasets, name):
            dfs = []
            for task_id, df in task_datasets.items():
                d = df.copy()
                d['task_id'] = task_id
                dfs.append(d)
            combined = pd.concat(dfs, ignore_index=True)
            out_path = splits_dir / f'{name}_with_negatives.tsv'
            combined.to_csv(out_path, sep='\t', index=False)
            pos = (combined['label'] == 1).sum()
            neg = (combined['label'] == 0).sum()
            print(f"  ✓ {name}: {len(combined):,} rows  "
                  f"(pos={pos:,}, neg={neg:,}) → {out_path}")
            return out_path

        _save_with_negatives(train_task_datasets, 'train')
        _save_with_negatives(val_task_datasets,   'val')
        test_neg_path = _save_with_negatives(test_task_datasets, 'test')

        print(f"\n  💡 评估时使用:")
        print(f"     --test_with_negatives {test_neg_path}")

    # ── 6c. 保存 split_meta.json ──
    import json
    split_meta = {
        'source':          args.mode2_output_dir,
        'mode':            'mode1_ablation',
        'n_train':         len(train_df),
        'n_val':           len(val_df),
        'n_test':          len(test_df),
        'n_hla':           int(all_df['hla'].nunique()),
        'negative_ratio':  args.negative_ratio,
        'save_with_neg':   args.save_splits_with_negatives,
    }
    with open(splits_dir / 'split_meta.json', 'w') as f:
        json.dump(split_meta, f, indent=2)
    print(f"✓ split_meta.json saved")

    # ── 6d. DATA_SOURCE.txt ──
    readme = output_dir / 'DATA_SOURCE.txt'
    with open(readme, 'w') as f:
        f.write("Mode 1 Model - Ablation Study\n")
        f.write("="*50 + "\n\n")
        f.write(f"Data Source: {args.mode2_output_dir}\n")
        f.write(f"Training samples: {len(train_df)}\n")
        f.write(f"Val samples:      {len(val_df)}\n")
        f.write(f"Test samples:     {len(test_df)}\n")
        f.write(f"\nNote: This Mode 1 model uses the EXACT same data splits\n")
        f.write(f"as Mode 2, but ignores tissue information.\n")
        f.write(f"This allows fair comparison as an ablation study.\n")
    print(f"✓ DATA_SOURCE.txt saved")

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Mode 1 model using Mode 2 data splits'
    )

    # 数据参数
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--mode2_output_dir', type=str, required=True,
                            help='Mode 2 output directory containing data_splits/')
    data_group.add_argument('--output_dir', type=str, required=True,
                            help='Output directory for Mode 1 model')
    data_group.add_argument('--hla_sequences', type=str,
                            default='configs/hla_sequences.json',
                            help='HLA sequences file')
    data_group.add_argument('--proteome_file', type=str,
                            default='data/human_proteome.fasta',
                            help='人类蛋白质组 FASTA 文件（来源蛋白配对切割所需）')
    data_group.add_argument('--use_negative_cache', action='store_true',
                            help='从缓存加载负样本（需与之前保存时参数一致）')
    data_group.add_argument('--save_negative_cache', action='store_true',
                            help='将生成的负样本保存到缓存')
    data_group.add_argument('--save_splits_with_negatives', action='store_true',
                            help='保存带负样本的 train/val/test TSV 到 output_dir/data_splits/，'
                                 '供 evaluate_mixmhcpred.py --test_with_negatives 直接使用')

    # 训练参数
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--n_epochs', type=int, default=85,
                            help='Number of epochs')
    train_group.add_argument('--batch_size', type=int, default=256,
                            help='Batch size')
    train_group.add_argument('--negative_ratio', type=int, default=10,
                            help='Negative:positive ratio')

    # MAML参数
    maml_group = parser.add_argument_group('MAML')
    maml_group.add_argument('--use_maml', action='store_true',
                            help='Use MAML meta-learning')
    maml_group.add_argument('--inner_lr', type=float, default=0.01,
                            help='Inner loop learning rate')
    maml_group.add_argument('--meta_lr', type=float, default=0.001,
                            help='Meta learning rate')
    maml_group.add_argument('--inner_steps', type=int, default=5,
                            help='Number of inner loop steps')

    # 模型参数
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--graph_threshold', type=float, default=0.8,
                            help='Graph edge threshold')
    model_group.add_argument('--peptide_dim', type=int, default=64,
                            help='Peptide embedding dimension')
    model_group.add_argument('--task_dim', type=int, default=64,
                            help='Task embedding dimension')
    model_group.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate')

    # Task-Balanced参数
    balance_group = parser.add_argument_group('Task Balancing')
    balance_group.add_argument('--use_task_balanced', action='store_true',
                              help='Enable task-balanced sampling')
    balance_group.add_argument('--sampling_strategy', type=str,
                              default='adaptive',
                              choices=['adaptive', 'moderate', 'aggressive'],
                              help='Sampling strategy')
    balance_group.add_argument('--use_task_weighting', action='store_true',
                              help='Enable task-based loss weighting')
    balance_group.add_argument('--weight_smoothing', type=str,
                              default='log',
                              choices=['log', 'sqrt', 'none'],
                              help='Task weight smoothing method')
    balance_group.add_argument('--use_class_weighting', action='store_true',
                              help='Enable class-based loss weighting')

    args = parser.parse_args()

    # 运行训练
    model, history = main(args)