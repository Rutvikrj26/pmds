#!/bin/bash
# SFT Training Script - FSDP Optimized for 3x H100 NVL
# Uses accelerate + FSDP for efficient distributed training
set -e

echo "=========================================="
echo "  Scallop-Titans SFT Training (FSDP)"
echo "  Started at: $(date)"
echo "=========================================="
echo ""

cd /home/rutvik/pmds/scallop-titans-agent

# Training configuration
MODEL="Qwen/Qwen3-32B"
DATA="data/combined_sft.jsonl"
OUTPUT="outputs/sft_full"
EPOCHS=2

# FSDP-optimized settings for 3x H100 NVL
# With Flash Attention 2 + gradient checkpointing, we can use larger batches
# Pushing to maximize GPU memory (was using 65-73GB of 96GB with batch=32)
BATCH_SIZE=48  # Per device (pushing to ~85-90GB usage)
GRAD_ACCUM=1   # Effective batch = 48 * 3 GPUs * 1 = 144

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA ($(wc -l < $DATA) samples)"
echo "  Output: $OUTPUT"
echo "  Epochs: $EPOCHS"
echo "  Per-device Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * 3 * GRAD_ACCUM))"
echo ""
echo "Parallelism: FSDP (Full Shard) on 3x H100 NVL"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Disable wandb (no login configured)
export WANDB_MODE=disabled
echo "Note: wandb disabled. Use 'wandb login' to enable logging."

# Run with accelerate using FSDP config
echo "Launching training with accelerate + FSDP..."
poetry run accelerate launch \
    --config_file accelerate_config.yaml \
    -m scallop_titans.training.sft \
    --model "$MODEL" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE

echo ""
echo "=========================================="
echo "  Training Complete!"
echo "  Finished at: $(date)"
echo "  Adapter saved to: $OUTPUT"
echo "=========================================="
