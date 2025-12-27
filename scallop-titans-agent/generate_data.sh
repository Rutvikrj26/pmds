#!/bin/bash
set -e

# Scallop-Titans Agent: Master Plan 2.0 Data Generation
# Generates 200k synthetic samples across multiple domains with SFT enhancements.

echo "========================================================"
echo "Starting Multi-Domain Data Generation"
echo "========================================================"
echo "Config:"
echo "  - Total Samples: 200,000"
echo "  - Negative Ratio: 10% (Unanswerable queries)"
echo "  - Distractor Ratio: 20% (Noise/Irrelevant reasoning)"
echo "  - Output Dir: data/multi_domain"
echo "========================================================"

# Run the parallel generator
# This uses all available CPU cores by default
poetry run python -m scallop_titans.data.generators.multi_domain_generator \
    --total 200000 \
    --negative-ratio 0.1 \
    --distractor-ratio 0.2 \
    --output data/multi_domain

echo ""
echo "Generation Complete!"
echo "Combined dataset: data/multi_domain/all_domains.jsonl"
echo "Sample count:"
wc -l data/multi_domain/all_domains.jsonl
