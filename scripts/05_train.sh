#!/bin/bash

echo "=========================================="
echo "Step 5: 训练 Phase 1 模型"
echo "=========================================="

# 选择训练模式
MODE=${1:-"standard"}  # standard 或 maml

if [ "$MODE" == "maml" ]; then
    echo "训练模式: MAML"
    USE_MAML="--use_maml"
else
    echo "训练模式: Standard"
    USE_MAML=""
fi

python -m src.training.trainer \
    --dataset_dir data/phase1_dataset \
    --output_dir models/phase1_models \
    $USE_MAML \
    --n_epochs 1 \
    --batch_size 32 \
    --meta_lr 0.001 \
    --inner_lr 0.01

echo ""
echo "✓ 训练完成！"
echo "模型保存在: models/phase1_models/"