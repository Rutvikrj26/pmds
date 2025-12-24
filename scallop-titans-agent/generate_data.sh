#!/bin/bash
set -e

echo "[1/4] Converting CLUTRR dataset..."
poetry run convert-clutrr --csv data/clutrr/train.csv --output data/clutrr_train.jsonl

echo "[2/4] Generating 100,000 synthetic traces..."
poetry run generate-synthetic --count 100000 --output data/synthetic_train.jsonl

echo "[3/4] Mining negative examples..."
poetry run mine-negatives --input data/synthetic_train.jsonl --output data/synthetic_negatives.jsonl --ratio 0.2

echo "[4/4] Combining datasets..."
cat data/clutrr_train.jsonl data/synthetic_train.jsonl data/synthetic_negatives.jsonl > data/combined_sft.jsonl

echo "Process Complete! Data saved to data/combined_sft.jsonl"
wc -l data/combined_sft.jsonl
