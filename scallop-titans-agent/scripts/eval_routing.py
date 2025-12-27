#!/usr/bin/env python3
"""
Evaluate Scallop-Titans Agent on Secure Network Routing Dataset.
"""
import argparse
import json
import logging
import os
import sys
import collections
from pathlib import Path
from tqdm import tqdm

# Add src to pythonpath
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scallop_titans.agent import ScallopTitansAgent, AgentConfig
from scallop_titans.memory import TitansMemoryAdapter
from scallop_titans.reasoning import ScallopEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class RoutingEvaluator:
    def __init__(self, model_path: str, adapter_path: str = "outputs/grpo/checkpoint-2000", device: str = "cuda"):
        # 1. Load Model Manually (like eval_clutrr.py)
        logger.info(f"Loading base model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            attn_implementation="flash_attention_2"
        )
        
        logger.info(f"Loading adapter from {adapter_path}")
        # Try to load adapter, fallback to base if not found (or if merged)
        try:
            self.llm = PeftModel.from_pretrained(base_model, adapter_path)
        except Exception as e:
            logger.warning(f"Could not load adapter: {e}. Using base model.")
            self.llm = base_model

        # 2. Configure Agent
        self.config = AgentConfig(
            model_name=model_path,
            max_new_tokens=512,
            temperature=0.01 
        )
        
        # Initialize Components
        logger.info("Initializing components...")
        self.titans = TitansMemoryAdapter() # Use defaults or config if needed
        # We need the routing rules for the engine
        self.scallop = ScallopEngine()
        # Import the routing rules!
        self.scallop._ctx.import_file("src/scallop_titans/reasoning/rules/routing.scl")
        
        self.agent = ScallopTitansAgent(
            config=self.config, 
            llm=self.llm, 
            tokenizer=self.tokenizer,
            titans_memory=self.titans, 
            scallop_engine=self.scallop
        )
        logger.info("Agent initialized.")

    def evaluate_file(self, data_path: str, num_samples: int = None):
        logger.info(f"Loading data from {data_path}")
        samples = []
        with open(data_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        
        if num_samples:
            samples = samples[:num_samples]
            
        metrics = collections.defaultdict(int)
        metrics["total"] = len(samples)
        
        results = []
        
        pbar = tqdm(samples, desc="Evaluating")
        for i, sample in enumerate(pbar):
            # 1. Reset State
            self.agent.reset_state()
            
            # 2. Construct Prompt
            # We want to see if the agent uses the tool.
            # System prompt is handled by agent.
            # User message: Context + Question
            user_msg = f"{sample['context']}\nQuestion: {sample['question']}"
            
            # 3. Run Inference
            try:
                response = self.agent.chat(user_msg)
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                metrics["errors"] += 1
                continue
                
            # 4. Analyze Result
            # Tool Usage: Check agent history for tool calls
            # The agent stores history as: [..., {"role": "assistant", "content": "...<|call_scallop|>..."}, {"role": "tool", ...}]
            
            tool_used = False
            full_trace = ""
            # Iterate through history for this turn (simplification: scan last few messages)
            # Actually, agent.chat() returns the final answer text string. 
            # We need to inspect the *history* to see the trace.
            
            for msg in self.agent.history:
                if msg.role == "assistant" and "<|call_scallop|>" in msg.content:
                    tool_used = True
                if msg.role == "assistant":
                    full_trace += msg.content
            
            if tool_used:
                metrics["tool_used"] += 1
            
            # Correctness
            # Answer is "Yes" or "No".
            predicted = "Yes" if "Yes" in response else "No" if "No" in response else "Unknown"
            ground_truth = sample["answer"]
            
            is_correct = (predicted.lower() == ground_truth.lower())
            if is_correct:
                metrics["correct"] += 1
                
            # Log
            result = {
                "id": i,
                "tool_used": tool_used,
                "correct": is_correct,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "trace": full_trace[:1000] # Truncate for log
            }
            results.append(result)
            
            # Update pbar
            acc = metrics["correct"] / (i + 1)
            usage = metrics["tool_used"] / (i + 1)
            pbar.set_postfix(acc=f"{acc:.2%}", tool=f"{usage:.2%}")
            
        return metrics, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/routing_test.jsonl")
    parser.add_argument("--model", type=str, default="outputs/sft_merged")
    parser.add_argument("--num_samples", type=int, default=10) # Test 10 first
    args = parser.parse_args()
    
    evaluator = RoutingEvaluator(args.model)
    metrics, results = evaluator.evaluate_file(args.data, args.num_samples)
    
    print("\n--- RESULTS ---")
    print(json.dumps(metrics, indent=2))
    
    # Save detailed results
    with open("outputs/eval_routing.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to outputs/eval_routing.json")

if __name__ == "__main__":
    main()
