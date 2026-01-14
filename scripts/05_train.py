"""
Step 5: 训练模型
"""


import os
# ========== 必须在最开头 ==========
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# =================================



import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import train_phase1


def main():
    parser = argparse.ArgumentParser(description='Train Phase 1 Model')
    parser.add_argument('--mode', type=str, default='maml',
                        choices=['standard', 'maml'],
                        help='Training mode')
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--meta_lr', type=float, default=0.001)
    parser.add_argument('--inner_lr', type=float, default=0.01)

    args = parser.parse_args()

    print("=" * 80)
    print(f"Step 5: 训练模型 ({args.mode.upper()} mode)")
    print("=" * 80)

    model, history = train_phase1(
        dataset_dir='data/phase1_dataset',
        graph_dir='data/phase1_dataset',
        output_dir='models/phase1_models',
        use_maml=(args.mode == 'maml'),
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr
    )

    print('\n✓ 训练完成！')
    print('  模型保存在: models/phase1_models/')


if __name__ == "__main__":
    main()