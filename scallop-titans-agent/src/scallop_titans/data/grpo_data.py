"""
GRPO Data Generator: Convert CLUTRR data to GRPO training format.

GRPO requires:
- prompt: The user's question
- ground_truth: Expected answer (for reward computation)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def extract_ground_truth_from_messages(messages: list[dict]) -> str | None:
    """
    Extract ground truth from SFT message format.
    
    Looks for the relationship answer in the assistant's Scallop query.
    """
    import re
    
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            
            # Pattern 1: query(relation, ?, X) - extract the relation
            match = re.search(r"query\s*\(\s*(\w+)\s*,", content)
            if match:
                return match.group(1).lower()
            
            # Pattern 2: relation(X, Y) = true - extract relation
            match = re.search(r"(\w+)\s*\([^)]+\)\s*=\s*true", content, re.IGNORECASE)
            if match:
                return match.group(1).lower()
    
    # Fallback: extract from user question "Who is X's [relation]?"
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            match = re.search(r"Who is.*'s\s+(\w+)", content, re.IGNORECASE)
            if match:
                return match.group(1).lower()
    
    return None


def extract_prompt_from_messages(messages: list[dict]) -> str | None:
    """Extract the user prompt from SFT messages."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def convert_sft_to_grpo(
    input_path: Path,
    output_path: Path,
    max_samples: int | None = None,
    shuffle: bool = True,
) -> int:
    """
    Convert SFT format data to GRPO format.
    
    Args:
        input_path: Path to SFT JSONL file (with 'messages' field)
        output_path: Path to output GRPO JSONL file
        max_samples: Maximum samples to convert (None = all)
        shuffle: Whether to shuffle data
        
    Returns:
        Number of samples converted
    """
    samples = []
    
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            messages = data.get("messages", [])
            
            prompt = extract_prompt_from_messages(messages)
            ground_truth = extract_ground_truth_from_messages(messages)
            
            if prompt and ground_truth:
                samples.append({
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                })
    
    if shuffle:
        random.shuffle(samples)
    
    if max_samples:
        samples = samples[:max_samples]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    return len(samples)


def main() -> None:
    """Generate GRPO training data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate GRPO training data")
    parser.add_argument("--input", default="data/clutrr_train.jsonl", help="Input SFT data")
    parser.add_argument("--output", default="data/grpo_train.jsonl", help="Output GRPO data")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")
    
    args = parser.parse_args()
    
    count = convert_sft_to_grpo(
        Path(args.input),
        Path(args.output),
        args.max_samples,
    )
    
    print(f"✅ Converted {count} samples to GRPO format")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
