#!/bin/bash

# Qwen2-VL Geo-Reasoning Fine-tuning Script
# Usage: bash train_geo_reasoning.sh

# =============================================================================
# Configuration
# =============================================================================

# Paths
MODEL_PATH="/nobackup/riyad/NAVIG/NaviClues/Navig/Qwen-VL/Qwen-VL-Models"  
DATA_PATH="/nobackup/riyad/NAVIG/NaviClues/Navig/Qwen-VL/Dataset/finetuneqwen.jsonl"  
OUTPUT_DIR="./output/qwen2vl-geo-reasoning"

# Multi-GPU setup
NUM_GPUS=8  # Number of GPUs to use
MASTER_PORT=29500

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=1  # Per GPU batch size (1 for 7B model on 24GB GPU)
GRAD_ACCUM=16  # Effective batch size = NUM_GPUS * BATCH_SIZE * GRAD_ACCUM
LEARNING_RATE=2e-5
MAX_GRAD_NORM=1.0

# LoRA settings
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Save settings
SAVE_STEPS=500
LOGGING_STEPS=10

# =============================================================================
# Training Command
# =============================================================================

echo "=========================================="
echo "Qwen2-VL Geo-Reasoning Fine-tuning"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Effective Batch Size: $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))"
echo "=========================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Option 1: Single GPU
if [ $NUM_GPUS -eq 1 ]; then
    echo "Training on single GPU..."
    CUDA_VISIBLE_DEVICES=0 python finetune_qwen.py \
        --model_name_or_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs $NUM_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LEARNING_RATE \
        --weight_decay 0.01 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps $LOGGING_STEPS \
        --save_steps $SAVE_STEPS \
        --save_total_limit 2 \
        --max_grad_norm $MAX_GRAD_NORM \
        --use_lora \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --gradient_checkpointing \
        --freeze_vision_tower \
        --bf16
else
    # Option 2: Multi-GPU with torchrun (recommended)
    echo "Training on $NUM_GPUS GPUs with torchrun..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        finetune_qwen.py \
        --model_name_or_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs $NUM_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LEARNING_RATE \
        --weight_decay 0.01 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps $LOGGING_STEPS \
        --save_steps $SAVE_STEPS \
        --save_total_limit 2 \
        --max_grad_norm $MAX_GRAD_NORM \
        --use_lora \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --gradient_checkpointing \
        --freeze_vision_tower \
        --bf16
fi

echo "=========================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
