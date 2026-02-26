#!/usr/bin/env python3
"""
独立运行Per-Task评估

用于在训练中断后，基于已保存的最佳模型进行per-task性能分析

用法:
    python scripts/evaluate_per_task.py \
        --output_dir output/mode2_standard_10 \
        --data_file data/mode2_data.tsv \
        --tissue_source Host
"""

import argparse
import torch
import pandas as pd
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.config.mode_config import create_mode2_config
from src.data.unified_task_creator import UnifiedTaskCreator
from src.data.enhanced_negative_sampler import EnhancedNegativeSampler
from src.models.full_model import ImmuneAppModel
from src.graph.task_graph import TaskGraphWrapper
from src.data.dataset import PeptideHLADataset
from src.training.per_task_evaluator import evaluate_per_task_performance
from torch.utils.data import DataLoader


def main(args):
    print("=" * 80)
    print("Per-Task Performance Evaluation (独立运行)")
    print("=" * 80)
    print(f"  Output dir: {args.output_dir}")
    print(f"  Data file: {args.data_file}")
    print("=" * 80)

    output_dir = Path(args.output_dir)

    # ========== 1. 检查必要文件 ==========
    print("\n" + "=" * 80)
    print("Step 1: 检查必要文件")
    print("=" * 80)

    required_files = {
        'model': output_dir / 'best_model.pt',
        'tasks': output_dir / 'tasks',
        'task_graph': output_dir / 'task_graph'
    }

    for name, path in required_files.items():
        if not path.exists():
            print(f"❌ 缺少 {name}: {path}")
            print(f"   请确保训练已经运行到保存模型的步骤")
            return
        else:
            print(f"✓ 找到 {name}: {path}")

    # ========== 2. 加载checkpoint ==========
    print("\n" + "=" * 80)
    print("Step 2: 加载模型checkpoint")
    print("=" * 80)

    checkpoint = torch.load(required_files['model'], map_location='cpu')

    print(f"  Checkpoint信息:")
    print(f"    Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"    Val AUROC: {checkpoint.get('val_auroc', 'N/A'):.4f}")
    print(f"    Val AUPRC: {checkpoint.get('val_auprc', 'N/A'):.4f}")

    # 从checkpoint恢复config
    mode_config_dict = checkpoint.get('mode_config', {})

    # ========== 3. 重建config ==========
    print("\n" + "=" * 80)
    print("Step 3: 重建配置")
    print("=" * 80)

    # 创建config (使用checkpoint中的参数)
    config = create_mode2_config(
        tissue_source=args.tissue_source,
        negative_ratio=mode_config_dict.get('negative_ratio', 10),
        min_samples=mode_config_dict.get('min_samples', 10),
        use_tissue_aware_negatives=mode_config_dict.get('use_tissue_aware_negatives', False)
    )

    print(f"  ✓ Config重建完成")
    print(f"    Mode: {config.task_type_name}")
    print(f"    Negative ratio: {config.negative_ratio}")

    # ========== 4. 加载数据 ==========
    print("\n" + "=" * 80)
    print("Step 4: 加载数据")
    print("=" * 80)

    df = pd.read_csv(args.data_file, sep='\t')
    df = df.rename(columns={
        'MHC_Restriction_Name': 'hla',
        'Peptide': 'peptide',
        args.tissue_source: 'tissue',
        'Label': 'label'
    })

    print(f"  ✓ 加载了 {len(df):,} 样本")

    # ========== 5. 数据集划分 (和训练时完全一致) ==========
    print("\n" + "=" * 80)
    print("Step 5: 数据集划分")
    print("=" * 80)

    # Split数据 (和训练时一致)
    from sklearn.model_selection import train_test_split

    pos_df = df[df['label'] == 1]

    try:
        train_df, temp_df = train_test_split(
            pos_df,
            test_size=0.3,
            random_state=42,
            stratify=pos_df['hla']
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['hla']
        )

        print(f"  ✓ 分层Split成功")

    except Exception as e:
        print(f"  ⚠️  分层split失败: {e}")
        print(f"  使用简单split...")

        train_df, temp_df = train_test_split(
            pos_df,
            test_size=0.3,
            random_state=42
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42
        )

    print(f"  Train: {len(train_df):,} 阳性样本")
    print(f"  Val: {len(val_df):,} 阳性样本")
    print(f"  Test: {len(test_df):,} 阳性样本")

    # ========== 6. 加载TaskManager ==========
    print("\n" + "=" * 80)
    print("Step 6: 加载TaskManager")
    print("=" * 80)

    creator = UnifiedTaskCreator(config)
    task_manager = creator.create_tasks(
        train_df,
        save_dir=required_files['tasks']  # 使用已保存的tasks
    )

    all_tasks = task_manager.get_all_tasks()
    print(f"  ✓ 加载了 {len(all_tasks)} tasks")

    # ========== 7. 生成/加载test set负样本 (支持缓存) ==========
    print("\n" + "=" * 80)
    print("Step 7: 生成/加载Test set负样本")
    print("=" * 80)

    # 导入缓存管理器
    from src.data.negative_cache import NegativeSampleCache

    cache = NegativeSampleCache('data/negative_samples')

    # 准备缓存配置
    cache_config = {
        'negative_ratio': mode_config_dict.get('negative_ratio', 10),
        'use_tissue_aware_negatives': mode_config_dict.get('use_tissue_aware_negatives', False),
        'min_samples': mode_config_dict.get('min_samples', 10),
        'tissue_source': args.tissue_source
    }

    print(f"  缓存配置:")
    print(f"    negative_ratio: {cache_config['negative_ratio']}")
    print(f"    use_tissue_aware: {cache_config['use_tissue_aware_negatives']}")

    # 尝试加载缓存
    print(f"  尝试加载缓存...")
    cached_data = cache.load(args.data_file, 'mode2', cache_config)

    if cached_data is not None:
        _, _, test_task_datasets = cached_data
        print(f"  🚀 使用缓存的负样本!")
        print(f"    Test tasks: {len(test_task_datasets)}")
    else:
        print(f"  ⚙️  缓存未找到，生成新的负样本...")

        test_sampler = EnhancedNegativeSampler(config, test_df)
        test_task_datasets = {}

        for task_id, task in all_tasks.items():
            task_test_df = test_df[
                (test_df['hla'] == task.hla) &
                (test_df['tissue'] == task.tissue)
                ]

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
                    'tissue': task.tissue,
                    'label': 0
                })

                # 合并正负样本
                task_test_combined = pd.concat([task_test_df, neg_df], ignore_index=True)
                test_task_datasets[task_id] = task_test_combined

        print(f"  ✓ 生成了 {len(test_task_datasets)} 个test tasks")

    # 统计
    total_test_samples = sum(len(df) for df in test_task_datasets.values())
    total_pos = sum((df['label'] == 1).sum() for df in test_task_datasets.values())
    total_neg = total_test_samples - total_pos

    print(f"  总Test样本: {total_test_samples:,}")
    print(f"    阳性: {total_pos:,}")
    print(f"    阴性: {total_neg:,}")
    print(f"    正负比: 1:{total_neg / total_pos:.1f}")

    # ========== 8. 加载Task Graph ==========
    print("\n" + "=" * 80)
    print("Step 8: 加载Task Graph")
    print("=" * 80)

    graph_wrapper = TaskGraphWrapper(required_files['task_graph'])

    # ========== 9. 创建DataLoader ==========
    print("\n" + "=" * 80)
    print("Step 9: 创建DataLoader")
    print("=" * 80)

    test_dataset = PeptideHLADataset(
        task_datasets=test_task_datasets,
        task_manager=task_manager,
        mode_config=config,
        graph_wrapper=graph_wrapper
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0
    )

    print(f"  ✓ Test loader: {len(test_dataset)} samples, {len(test_loader)} batches")

    # ========== 10. 重建模型 ==========
    print("\n" + "=" * 80)
    print("Step 10: 重建模型")
    print("=" * 80)

    # 统计tissues数量
    n_tissues = len(set(task.tissue for task in all_tasks.values()))

    model = ImmuneAppModel(
        mode_config=config,
        n_tasks=len(all_tasks),
        n_tissues=n_tissues
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    graph_wrapper = graph_wrapper.to(device)

    print(f"  ✓ 模型加载完成")
    print(f"  Device: {device}")

    # ========== 11. 运行Per-Task评估 ==========
    print("\n" + "=" * 80)
    print("Step 11: Per-Task评估")
    print("=" * 80)

    per_task_df, per_task_summary = evaluate_per_task_performance(
        model=model,
        test_loader=test_loader,
        graph_wrapper=graph_wrapper,
        task_manager=task_manager,
        mode_config=config,
        output_dir=output_dir,
        device=device
    )

    print("\n" + "=" * 80)
    print("✓ Per-Task评估完成!")
    print("=" * 80)
    print(f"\n结果保存在:")
    print(f"  - {output_dir / 'per_task_metrics_test.csv'}")
    print(f"  - {output_dir / 'per_task_summary_test.json'}")
    print(f"  - {output_dir / 'top_20_tasks_test.csv'}")
    print(f"  - {output_dir / 'bottom_20_tasks_test.csv'}")

    # 可选: 生成可视化
    if args.visualize:
        print("\n生成可视化...")
        import subprocess
        subprocess.run([
            'python', 'scripts/visualize_per_task_results.py',
            str(output_dir)
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='独立运行Per-Task评估')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='训练输出目录 (包含best_model.pt)')

    parser.add_argument('--data_file', type=str, required=True,
                        help='数据文件路径')

    parser.add_argument('--tissue_source', type=str, default='Host',
                        help='Tissue列名')

    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化')

    args = parser.parse_args()

    main(args)