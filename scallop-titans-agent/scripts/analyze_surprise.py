#!/usr/bin/env python3
"""
Analyze Surprise Metric

Visualizes the 'surprise' values stored in the Titans Memory across a sequence of updates.
Higher surprise indicates the model found the information novel/unexpected.
"""

import argparse
import json
import numpy as np
import os

def analyze_surprise(json_path: str):
    print(f"Loading results from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if "surprises" not in data or not data["surprises"]:
        print("No surprise data found in JSON.")
        return

    # Data structure: List of Lists (one list per sample)
    all_surprises = data["surprises"]
    
    # 1. Flatten for distribution analysis
    flat_surprises = [val for sublist in all_surprises for val in sublist]
    
    if not flat_surprises:
        print("Surprise lists are empty.")
        return
        
    print(f"\n--- Surprise Stats ({len(flat_surprises)} total updates) ---")
    print(f"Mean: {np.mean(flat_surprises):.4f}")
    print(f"Max:  {np.max(flat_surprises):.4f}")
    print(f"Min:  {np.min(flat_surprises):.4f}")
    print(f"Std:  {np.std(flat_surprises):.4f}")
    
    # 2. Analyze Trend per Sample (Average surprise profile)
    # We pad to max length to compute average profile
    max_len = max(len(s) for s in all_surprises)
    profile = np.zeros(max_len)
    counts = np.zeros(max_len)
    
    for s in all_surprises:
        for i, val in enumerate(s):
            profile[i] += val
            counts[i] += 1
            
    avg_profile = profile / np.maximum(counts, 1)
    
    print("\n--- Average Surprise Profile (First 10 steps) ---")
    print(avg_profile[:10])
    
    # 3. Plot (Skipped due to missing dependency)
    print("\nTopological analysis complete (plots skipped).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to evaluation output JSON (e.g. outputs/eval_k2.json)")
    args = parser.parse_args()
    
    if os.path.exists(args.json_path):
        analyze_surprise(args.json_path)
    else:
        print(f"File not found: {args.json_path}")
