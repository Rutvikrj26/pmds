#!/bin/bash
# GRPO Training Script for Scallop-Titans Agent
# Uses FSDP for distributed training across 3x H100 NVL GPUs

set -e

echo "=========================================="
echo "  Scallop-Titans GRPO Training (FSDP)"
echo "  Started at: $(date)"
echo "=========================================="

# Configuration - using merged SFT model (LoRA already baked in)
MODEL="outputs/sft_merged"  # Merged model with SFT training (88% accuracy)
SFT_CHECKPOINT=""  # Not needed - LoRA already merged into model
DATA="data/grpo_train_6k.jsonl"  # Reduced dataset (6k samples)
OUTPUT="outputs/grpo"
EPOCHS=1  # PoC: 1 epoch

# GRPO-specific settings (MVP - Stable)
BATCH_SIZE=4   # Per device (Stable at ~65GB)
GRAD_ACCUM=2   # Effective batch = 4 * 3 GPUs * 2 = 24

echo "Configuration:"
echo "  Model: $MODEL"
echo "  SFT Checkpoint: $SFT_CHECKPOINT"
echo "  Data: $DATA"
echo "  Output: $OUTPUT"
echo "  Epochs: $EPOCHS"
echo "  Per-device Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * 3 * GRAD_ACCUM))"
echo ""
echo "GRPO Settings (Optimized PoC):"
echo "  Group Size: 8"
echo "  Max Tokens: 1024"
echo "  KL Penalty: 0.01"
echo "  Learning Rate: 5e-6"
echo ""

# Check for GRPO data
if [ ! -f "$DATA" ]; then
    echo "GRPO data not found. Generating from CLUTRR..."
    poetry run python -m scallop_titans.data.grpo_data \
        --input data/clutrr_train.jsonl \
        --output "$DATA"
    echo ""
fi

SAMPLE_COUNT=$(wc -l < "$DATA")
echo "Training samples: $SAMPLE_COUNT"
echo ""

# GPU check
echo "Parallelism: DeepSpeed ZeRO-2 on 3x H100 NVL"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Disable wandb (use 'wandb login' to enable)
export WANDB_MODE=disabled
echo "Note: wandb disabled. Use 'wandb login' to enable logging."

echo "Launching GRPO training with accelerate + DeepSpeed..."
poetry run accelerate launch \
    --config_file accelerate_deepspeed.yaml \
    -m scallop_titans.training.grpo \
    --model "$MODEL" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --epochs "$EPOCHS"

echo ""
echo "=========================================="
echo "  GRPO Training Complete!"
echo "  Finished at: $(date)"
echo "  Output: $OUTPUT"
echo "=========================================="
