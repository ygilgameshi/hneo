"""
端到端训练脚本 - Mode 2 (HLA×Tissue)

完整流程:
1. 加载数据
2. 创建任务 (HLA×Tissue组合)
3. 生成负样本 (tissue-aware)
4. 构建Task Graph (HLA + Tissue相似性)
5. 训练模型 (FiLM融合)
6. 评估和保存

支持: MAML / Standard 训练
"""

import pandas as pd
import argparse
from pathlib import Path
import sys
import torch

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.config.mode_config import create_mode2_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.enhanced_negative_sampler import EnhancedNegativeSampler
from src.training.unified_trainer import train_model


def main(args):
    print("=" * 80)
    print("ImmuneApp Mode 2 (HLA×Tissue) - End-to-End Training")
    print("=" * 80)
    print(f"  Data file: {args.data_file}")
    print(f"  Tissue source: {args.tissue_source}")
    print(f"  Training method: {'MAML' if args.use_maml else 'Standard'}")
    print(f"  Output dir: {args.output_dir}")
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
        args.tissue_source: 'tissue',
        'Label': 'label'
    })

    # 处理tissue
    df['tissue'] = df['tissue'].fillna('Unknown')

    print(f"  Positive: {(df['label'] == 1).sum():,}")
    print(f"  Negative: {(df['label'] == 0).sum():,}")
    print(f"  HLA types: {df['hla'].nunique()}")
    print(f"  Tissues: {df['tissue'].nunique()}")
    print(f"  HLA×Tissue combinations: {df[['hla', 'tissue']].drop_duplicates().shape[0]}")

    # 过滤Unknown tissue (可选)
    if args.filter_unknown_tissue:
        before = len(df)
        df = df[df['tissue'] != 'Unknown']
        print(f"\n✓ Filtered Unknown tissue: {before:,} -> {len(df):,}")

    # 数据集划分
    from sklearn.model_selection import train_test_split

    print(f"\n{'=' * 80}")
    print("Step 1.5: Data Cleaning & Splitting")
    print("=" * 80)

    # ========== 主动过滤小样本组合 ==========
    n_tissues = df['tissue'].nunique()
    min_samples_per_combo = args.min_samples_for_split  # 使用命令行参数

    if n_tissues > 1:
        print(f"\n🧹 Filtering small HLA×Tissue combinations...")
        print(f"  Minimum samples per combo: {min_samples_per_combo}")

        # 创建组合key
        df['combo_key'] = df['hla'] + '_' + df['tissue']

        # 统计每个组合的样本数
        combo_counts = df['combo_key'].value_counts()

        print(f"\n  Before filtering:")
        print(f"    Total samples: {len(df):,}")
        print(f"    HLA×Tissue combinations: {len(combo_counts)}")
        print(f"    Min samples: {combo_counts.min()}")
        print(f"    Max samples: {combo_counts.max()}")

        # 找出样本数>=min_samples_per_combo的组合
        valid_combos = combo_counts[combo_counts >= min_samples_per_combo].index.tolist()

        # 过滤数据
        df_filtered = df[df['combo_key'].isin(valid_combos)].copy()

        # 统计过滤效果
        removed_combos = len(combo_counts) - len(valid_combos)
        removed_samples = len(df) - len(df_filtered)

        print(f"\n  After filtering:")
        print(f"    Total samples: {len(df_filtered):,}")
        print(f"    HLA×Tissue combinations: {len(valid_combos)}")
        print(f"    Removed combos: {removed_combos} ({removed_combos / len(combo_counts) * 100:.1f}%)")
        print(f"    Removed samples: {removed_samples:,} ({removed_samples / len(df) * 100:.1f}%)")

        # 更新df
        df = df_filtered

        # 验证现在所有组合都>=min_samples
        combo_counts_filtered = df['combo_key'].value_counts()
        print(f"\n  ✓ All combos now have ≥{min_samples_per_combo} samples")
        print(f"    Min: {combo_counts_filtered.min()}, Max: {combo_counts_filtered.max()}")
    else:
        print(f"\n🧹 Filtering small HLA groups...")
        print(f"  Minimum samples per HLA: {min_samples_per_combo}")

        # 统计每个HLA的样本数
        hla_counts = df['hla'].value_counts()

        print(f"\n  Before filtering:")
        print(f"    Total samples: {len(df):,}")
        print(f"    HLA types: {len(hla_counts)}")
        print(f"    Min samples: {hla_counts.min()}")

        # 找出样本数>=min_samples_per_combo的HLA
        valid_hlas = hla_counts[hla_counts >= min_samples_per_combo].index.tolist()

        # 过滤
        df_filtered = df[df['hla'].isin(valid_hlas)].copy()

        removed_hlas = len(hla_counts) - len(valid_hlas)
        removed_samples = len(df) - len(df_filtered)

        print(f"\n  After filtering:")
        print(f"    Total samples: {len(df_filtered):,}")
        print(f"    HLA types: {len(valid_hlas)}")
        print(f"    Removed HLAs: {removed_hlas}")
        print(f"    Removed samples: {removed_samples:,}")

        df = df_filtered

    # ========== Stratified Split (智能两阶段) ==========
    print(f"\n📊 Splitting dataset (80/10/10)...")

    # 第一次split: train vs temp (80% vs 20%)
    if n_tissues > 1:
        stratify_key = df['combo_key']
        print(f"  First split strategy: HLA×Tissue stratification")
    else:
        stratify_key = df['hla']
        print(f"  First split strategy: HLA stratification")

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=stratify_key
    )

    print(f"  ✓ Train/Temp split done: {len(train_df):,} / {len(temp_df):,}")

    # 第二次split: temp -> val vs test (50% vs 50%)
    # 关键: 检查temp_df中每个组合的样本数
    if n_tissues > 1:
        temp_combo_counts = temp_df['combo_key'].value_counts()
        temp_min_count = temp_combo_counts.min()
        single_sample_combos = (temp_combo_counts == 1).sum()

        print(f"\n  Second split (val/test from temp):")
        print(f"    Temp combos: {len(temp_combo_counts)}")
        print(f"    Min samples in temp: {temp_min_count}")
        print(f"    Single-sample combos in temp: {single_sample_combos}")

        if temp_min_count >= 2:
            # 可以继续stratify
            print(f"    ✓ Using HLA×Tissue stratification")
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, random_state=42,
                stratify=temp_df['combo_key']
            )
        else:
            # 第二次split时降级到HLA stratify或random
            print(f"    ⚠ Some combos have only 1 sample in temp")

            # 尝试HLA stratify
            temp_hla_counts = temp_df['hla'].value_counts()
            temp_hla_min = temp_hla_counts.min()

            if temp_hla_min >= 2:
                print(f"    ✓ Fallback to HLA stratification")
                val_df, test_df = train_test_split(
                    temp_df, test_size=0.5, random_state=42,
                    stratify=temp_df['hla']
                )
            else:
                print(f"    ✓ Fallback to random split")
                val_df, test_df = train_test_split(
                    temp_df, test_size=0.5, random_state=42
                )
    else:
        # 只有1个tissue,用HLA stratify
        temp_hla_counts = temp_df['hla'].value_counts()
        temp_hla_min = temp_hla_counts.min()

        if temp_hla_min >= 2:
            print(f"  Second split: HLA stratification")
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, random_state=42,
                stratify=temp_df['hla']
            )
        else:
            print(f"  Second split: Random (HLA samples too few in temp)")
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, random_state=42
            )

    # 删除临时列
    for d in [train_df, val_df, test_df]:
        if 'combo_key' in d.columns:
            d.drop(columns=['combo_key'], inplace=True)

    print(f"\n✓ Data split:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val: {len(val_df):,}")
    print(f"  Test: {len(test_df):,}")

    # ========== 1b. 保存数据划分（正样本，不含负样本）==========
    splits_dir = Path(args.output_dir) / 'data_splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_dir / 'train.tsv', sep='\t', index=False)
    val_df.to_csv(splits_dir / 'val.tsv', sep='\t', index=False)
    test_df.to_csv(splits_dir / 'test.tsv', sep='\t', index=False)

    import json as _json
    split_meta = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_hlas': int(train_df['hla'].nunique()),
        'val_hlas': int(val_df['hla'].nunique()),
        'test_hlas': int(test_df['hla'].nunique()),
        'columns': list(train_df.columns),
        'data_file': args.data_file,
    }
    with open(splits_dir / 'split_meta.json', 'w') as _f:
        _json.dump(split_meta, _f, indent=2)

    print(f"\n✓ Data splits saved to: {splits_dir}")
    print(f"  train.tsv  → {len(train_df):,} rows")
    print(f"  val.tsv    → {len(val_df):,} rows")
    print(f"  test.tsv   → {len(test_df):,} rows")
    print(f"  split_meta.json")

    # ========== 2. 创建配置 ==========
    print("\n" + "=" * 80)
    print("Step 2: Creating Config")
    print("=" * 80)

    config = create_mode2_config(
        min_samples=args.min_samples,
        negative_ratio=args.negative_ratio,
        tissue_source=args.tissue_source,
        use_maml=args.use_maml,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        inner_steps=args.inner_steps
    )

    print(f"✓ Config created:")
    print(f"  Mode: {config.task_type_name}")
    print(f"  Min samples: {config.min_samples}")
    print(f"  Negative ratio: {config.negative_ratio}")
    print(f"  Tissue source: {config.tissue_source}")
    print(f"  Use MAML: {config.use_maml}")
    if config.use_maml:
        print(f"    Inner LR: {config.inner_lr}")
        print(f"    Inner steps: {config.inner_steps}")
    print(f"  Meta LR: {config.meta_lr}")

    # ========== 3. 创建任务 ==========
    print("\n" + "=" * 80)
    print("Step 3: Creating Tasks (HLA×Tissue)")
    print("=" * 80)

    creator = UnifiedTaskCreator(config)
    task_manager = creator.create_tasks(
        train_df,
        save_dir=Path(args.output_dir) / 'tasks'
    )

    all_tasks = task_manager.get_all_tasks()
    print(f"✓ Created {len(all_tasks)} tasks")

    # 统计
    hlas = set(t.hla for t in all_tasks.values())
    tissues = set(t.tissue for t in all_tasks.values())
    print(f"  Unique HLAs: {len(hlas)}")
    print(f"  Unique Tissues: {len(tissues)}")

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
        'min_samples': args.min_samples,
        'tissue_source': args.tissue_source
    }

    # 尝试加载缓存 (如果启用)
    cached_data = None
    if args.use_negative_cache:
        cached_data = cache.load(args.data_file, 'mode2', cache_config)

    if cached_data is not None:
        # 使用缓存的数据
        train_task_datasets, val_task_datasets, test_task_datasets = cached_data
        print(f"\n🚀 Using cached negative samples!")
    else:
        # 生成新的负样本
        print(f"\n⚙️ Generating new negative samples...")

        # 构建全量数据（用于 exclude set，避免 train/val 阳性成为 val/test 假阴性负样本）
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # 加载蛋白质组序列
        from src.data.enhanced_negative_sampler import load_protein_sequences
        proteome_path = getattr(args, 'proteome_file', 'data/human_proteome.fasta')
        protein_sequences = {}
        if Path(proteome_path).exists():
            protein_sequences = load_protein_sequences(proteome_path)
        else:
            print(f"  ⚠ 蛋白质组文件未找到: {proteome_path}")
            print(f"  → 将使用 fallback 采样（负样本质量下降）")

        # 创建 Train Sampler（exclude set = 全量阳性，采样范围 = train_df）
        sampler = EnhancedNegativeSampler(config, all_df,
                                          protein_sequences=protein_sequences)

        # === Train ===
        print("\n  Generating train negatives...")
        train_task_datasets = sampler.generate_negatives_for_all_tasks(task_manager, source_data=train_df)

        # === Val ===
        print("\n  Generating val negatives...")
        # Val Sampler：exclude set = 全量阳性（防止 train 阳性成为 val 假阴性）
        val_sampler = EnhancedNegativeSampler(config, all_df,
                                             protein_sequences=protein_sequences)
        val_task_datasets = val_sampler.generate_negatives_for_all_tasks(task_manager, source_data=val_df)

        # === Test ===
        print("\n  Generating test negatives...")
        # Test Sampler：exclude set = 全量阳性（防止 train/val 阳性成为 test 假阴性）
        test_sampler = EnhancedNegativeSampler(config, all_df,
                                              protein_sequences=protein_sequences)
        test_task_datasets = test_sampler.generate_negatives_for_all_tasks(task_manager, source_data=test_df)

        # 保存到缓存 (如果启用)
        if args.save_negative_cache:
            cache.save(
                args.data_file,
                'mode2',
                cache_config,
                train_task_datasets,
                val_task_datasets,
                test_task_datasets
            )

    print(f"\n✓ Negative samples ready")

    # ========== 5. 训练模型 ==========
    print("\n" + "=" * 80)
    print("Step 5: Training Model (Task-Balanced + FiLM)")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        tissue_dim=args.tissue_dim,
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
    print(f"  Best Val AUROC: {max(history['val_auroc']):.4f}")
    print(f"  Best Val AUPRC: {max(history['val_auprc']):.4f}")
    print(f"  Final Val AUROC: {history['val_auroc'][-1]:.4f}")
    print(f"  Final Val AUPRC: {history['val_auprc'][-1]:.4f}")

    # Mode 2特有: 打印模型参数对比
    print(f"\n✓ Model architecture:")
    params = model.get_parameters_count()
    for component, count in params.items():
        print(f"  {component}: {count:,}")

    print(f"\n✓ All results saved to: {args.output_dir}")
    print(f"  - best_model.pt")
    print(f"  - training_results.json")
    print(f"  - task_graph/ (with tissue similarity)")
    print(f"  - tasks/")

    print("\n" + "=" * 80)
    print("Training finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ImmuneApp Mode 2 End-to-End Training')

    # 数据参数
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to data file (TSV format)')
    parser.add_argument('--tissue_source', type=str, default='Host',
                        help='Tissue column name (Host/Inferred_Tissue)')
    parser.add_argument('--filter_unknown_tissue', action='store_true',
                        help='Filter samples with Unknown tissue')
    parser.add_argument('--proteome_file', type=str, default='data/human_proteome.fasta',
                        help='人类蛋白质组 FASTA 文件路径（来源蛋白配对切割策略所需）')
    parser.add_argument('--hla_sequences', type=str, default='configs/hla_sequences.json',
                        help='Path to HLA sequences file')
    parser.add_argument('--output_dir', type=str, default='output/mode2_experiment',
                        help='Output directory')

    # 任务创建参数
    parser.add_argument('--min_samples', type=int, default=10,
                        help='Minimum samples per task (lower for Mode 2)')
    parser.add_argument('--min_samples_for_split', type=int, default=10,
                        help='Minimum samples per HLA×Tissue combo for stratified split (>=10 recommended to ensure second split succeeds)')
    parser.add_argument('--negative_ratio', type=int, default=20,
                        help='Negative:Positive ratio')

    # 负样本缓存参数 (新增)
    parser.add_argument('--save_negative_cache', action='store_true',
                        help='Save generated negatives to cache for reuse')
    parser.add_argument('--use_negative_cache', action='store_true',
                        help='Load negatives from cache if available')

    # 训练方法
    parser.add_argument('--use_maml', action='store_true',
                        help='Use MAML training (default: Standard)')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='MAML inner learning rate')
    parser.add_argument('--inner_steps', type=int, default=5,
                        help='MAML inner adaptation steps')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Meta/standard learning rate')

    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    # Graph参数
    parser.add_argument('--graph_threshold', type=float, default=0.3,
                        help='Task graph similarity threshold')

    # 模型参数
    parser.add_argument('--peptide_dim', type=int, default=64,
                        help='Peptide encoder output dimension')
    parser.add_argument('--task_dim', type=int, default=64,
                        help='Task GNN output dimension')
    parser.add_argument('--tissue_dim', type=int, default=32,
                        help='Tissue embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Task-Balanced参数
    parser.add_argument('--use_task_balanced', action='store_true',
                        help='Use Task-Balanced sampling with adaptive strategy')
    parser.add_argument('--sampling_strategy', type=str, default='adaptive',
                        choices=['adaptive', 'moderate', 'aggressive'],
                        help='Sampling strategy: adaptive(推荐), moderate(温和), aggressive(激进)')
    parser.add_argument('--use_task_weighting', action='store_true',
                        help='Use task-weighted loss')
    parser.add_argument('--weight_smoothing', type=str, default='log',
                        choices=['none', 'sqrt', 'log'],
                        help='Loss weight smoothing method')

    # Class权重参数
    parser.add_argument('--use_class_weighting', action='store_true',
                        help='Use class weights for positive/negative imbalance (auto-computed from data)')

    args = parser.parse_args()

    main(args)

