#!/usr/bin/env python3
"""
CLUTRR Evaluation Harness

Generates CLUTRR samples of specific k-hop difficulty and evaluates the 
Scallop-Titans Agent's accuracy.

Usage:
    poetry run python scripts/eval_clutrr.py --k 2 --num_samples 50
"""

import argparse
import collections
import json
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from scallop_titans.agent import ScallopTitansAgent, AgentConfig, ChatMessage
from scallop_titans.memory import TitansMemoryAdapter
from scallop_titans.reasoning import ScallopEngine
from scallop_titans.data.synthetic_generator import SyntheticTraceGenerator, SyntheticConfig

# --- Configuration ---
BASE_MODEL = "outputs/sft_merged"
ADAPTER_PATH = "outputs/grpo/checkpoint-2000"

def load_agent():
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    
    print(f"Loading adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    print("Initializing components...")
    titans = TitansMemoryAdapter()
    scallop = ScallopEngine()
    scallop.set_titans_memory(titans)
    
    config = AgentConfig(max_new_tokens=512)
    agent = ScallopTitansAgent(
        llm=model,
        tokenizer=tokenizer,
        titans_memory=titans,
        scallop_engine=scallop,
        config=config
    )
    return agent

def generate_test_data(k: int, num_samples: int):
    """Generate on-the-fly test data for specific k-hop."""
    # Force the generator to produce only k-hop samples if possible
    # SyntheticGenerator uses a probabilistic distribution.
    # We will hack the config to 100% probability for k-hop.
    hop_dist = {k: 1.0}
    config = SyntheticConfig(
        num_samples=num_samples,
        hop_distribution=hop_dist
    )
    generator = SyntheticTraceGenerator(config)
    
    print(f"Generating {num_samples} samples with k={k}...")
    
    samples = []
    # We need to capture the output. output_path is required by generate_batch.
    # We'll use a temp file.
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        tmp_path = tmp.name
        
    generator.generate_batch(tmp_path)
    
    with open(tmp_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    os.remove(tmp_path)
    return samples[:num_samples] # clamp due to generator logic potentially overshooting

class Evaluator:
    def __init__(self, agent):
        self.agent = agent
        
    def evaluate_batch(self, samples):
        metrics = collections.defaultdict(int)
        metrics["surprises"] = [] # list of lists
        latencies = []
        
        # Determine prompt format from data
        # Data format from generator:
        # {
        #   "messages": [...],
        #   "metadata": {"target": "...", "hop": k}
        # }
        # Or similar. Let's inspect what generator produces via trace_converter.
        # It's standard chat messages format:
        # messages=[{"role": "system", ...}, {"role": "user", "content": "Story... Question?"}, {"role": "assistant", ...}]
        
        for i, sample in enumerate(tqdm(samples)):
            try:
                # Extract User Prompt
                # Typically the second message is user
                user_msg = next(m["content"] for m in sample["messages"] if m["role"] == "user")
                ground_truth = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
                
                # Extract Target Answer (Last Sentence)
                # Ground truth: "<thought>...<call>...</thought> Alice is Bob's aunt."
                # We want "aunt" or the full sentence relationship.
                # Let's verify by soft containment.
                
                # Reset Agent Memory per turn if independent samples
                self.agent.reset_state()
                
                start_t = time.time()
                response = self.agent.chat(user_msg)
                latencies.append(time.time() - start_t)
                
                # Full trace includes thought process
                full_trace = self.agent.history[-1].content
                
                # Capture Surprise History
                surprise_vals = self.agent.titans.surprise_history
                # Store roughly per-sample surprise
                metrics["surprises"].append(surprise_vals)
                
                # --- Metrics ---
                
                # 1. Correctness
                # We need the target relation from the generated sample to be robust.
                # But trace_converter might not export metadata field?
                # Actually, trace_converter returns a list of messages.
                # We iterate the ground_truth assistant message to find the relationship.
                # Heuristic: the last sentence contains the relation.
                
                # Let's rely on simple keyword matching against the ground truth assistant response.
                # If ground truth says "aunt", we look for "aunt" in prediction.
                
                # Extract clean answer from ground truth (remove thoughts)
                gt_clean = ground_truth.split("<|end_thought|>")[-1].strip()
                gt_clean = gt_clean.split("<|end_scallop_result|>")[-1].strip()
                if "<think>" in gt_clean: # format variation
                     gt_clean = gt_clean.split("</think>")[-1].strip()
                
                # Extract clean answer from prediction
                pred_clean = response # agent.chat returns clean answer by default loop
                
                # Check containment of RELATION keywords (brittle but effective for PoC)
                from scallop_titans.constants import CLUTRR_RELATIONS
                
                found_rels_gt = [r for r in CLUTRR_RELATIONS if r.replace("-", " ") in gt_clean.lower()]
                found_rels_pred = [r for r in CLUTRR_RELATIONS if r.replace("-", " ") in pred_clean.lower()]
                
                # Match strict: if sets overlap
                is_correct = not set(found_rels_gt).isdisjoint(set(found_rels_pred))
                
                if is_correct:
                    metrics["correct"] += 1
                
                # 2. Scallop Usage
                if "<|call_scallop|>" in full_trace:
                    metrics["scallop_used"] += 1
                    
                    # 3. Valid Syntax (if used)
                    if "<|scallop_result|>" in full_trace and "Error" not in full_trace:
                         metrics["valid_syntax"] += 1
                         
                metrics["total"] += 1
                
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                metrics["errors"] += 1
                
        # Summary
        total = metrics["total"]
        results = {
            "accuracy": metrics["correct"] / total if total > 0 else 0,
            "scallop_usage_rate": metrics["scallop_used"] / total if total > 0 else 0,
            "valid_syntax_rate": metrics["valid_syntax"] / metrics["scallop_used"] if metrics["scallop_used"] > 0 else 0,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "total_samples": total
        }
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()
    
    # Generate Data
    samples = generate_test_data(args.k, args.num_samples)
    if not samples:
        print("Failed to generate samples.")
        return
        
    # Load Model
    agent = load_agent()
    
    # Evaluate
    evaluator = Evaluator(agent)
    results = evaluator.evaluate_batch(samples)
    
    print("\n--- RESULTS ---")
    print(json.dumps(results, indent=2))
    
    # Save to file
    with open(f"outputs/eval_k{args.k}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
