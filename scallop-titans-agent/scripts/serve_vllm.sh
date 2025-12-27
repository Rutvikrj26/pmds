#!/bin/bash
# Serve Scallop-Titans Agent with vLLM
# RS-5: vLLM Tool Parser Integration

MODEL_PATH="outputs/sft_merged"
# We need to merge the GRPO adapter into the base model first for vLLM
# Or use --enable-lora if vLLM supports it (it does)

# Verify adapter existence
ADAPTER_PATH="outputs/grpo/checkpoint-2000"
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Error: Adapter not found at $ADAPTER_PATH"
    exit 1
fi

echo "Starting vLLM server..."
echo "Model: $MODEL_PATH"
echo "Adapter: $ADAPTER_PATH"

# Note: We need to register the custom tool parser.
# The tool parser is in src/scallop_titans/serving/scallop_tool_parser.py
# vllm needs it in PYTHONPATH

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Launch vLLM
# --enable-lora to support the GRPO adapter
# --tool-parser-plugin to register our custom Scallop parser

vllm serve $MODEL_PATH \
    --enable-lora \
    --lora-modules scallop-grpo=$ADAPTER_PATH \
    --tool-parser-plugin src/scallop_titans/serving/scallop_tool_parser.py \
    --tool-call-parser scallop \
    --enable-auto-tool-choice \
    --port 8000
