#!/usr/bin/env python3
"""
Generate Secure Network Routing Dataset.
Generates challenging multi-hop reachability problems with security constraints.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scallop_titans.data.graph_generator import NetworkRoutingGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/routing_test.jsonl")
    args = parser.parse_args()
    
    gen = NetworkRoutingGenerator()
    
    print(f"Generating {args.num_samples} samples to {args.output}...")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    count = 0
    with open(args.output, "w") as f:
        while count < args.num_samples:
            # Generate 4-10 hop problems (challenging)
            sample = gen.generate_sample(k_min=4, k_max=10)
            if sample:
                f.write(json.dumps(sample) + "\n")
                count += 1
                if count % 10 == 0:
                    print(f"Generated {count}/{args.num_samples}")
                    
    print("Done.")

if __name__ == "__main__":
    main()
