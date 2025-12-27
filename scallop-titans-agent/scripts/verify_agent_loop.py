#!/usr/bin/env python3
"""
Verify Agent Loop

This script verifies that the trained Scallop-Titans Agent:
1. Loads correctly (Base Model + GRPO Adapter).
2. Generates <|call_scallop|> tokens when prompted with a reasoning task.
3. Successfully invokes the Scallop Engine.
4. Returns a valid final answer.

Output: JSON printed to stdout.
"""

import json
import time
import dataclasses
import torch
from dataclasses import dataclass, field
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import project modules
from scallop_titans.agent import ScallopTitansAgent, AgentConfig
from scallop_titans.memory import TitansMemoryAdapter
from scallop_titans.reasoning import ScallopEngine

# Configuration from previous context
BASE_MODEL = "outputs/sft_merged"
ADAPTER_PATH = "outputs/grpo/checkpoint-2000"

@dataclass
class VerificationMetrics:
    success: bool = False
    model_loaded: bool = False
    adapter_loaded: bool = False
    scallop_invoked: bool = False
    valid_syntax: bool = False
    final_answer: str = ""
    latency_seconds: float = 0.0
    trace: list[str] = field(default_factory=list)
    error: str = ""

def run_verification():
    metrics = VerificationMetrics()
    start_time = time.time()
    
    print(f"Starting verification...")
    
    try:
        # 1. Load Model
        print(f"Loading base model: {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda",
            attn_implementation="flash_attention_2"
        )
        metrics.model_loaded = True
        
        print(f"Loading adapter: {ADAPTER_PATH}")
        try:
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
            metrics.adapter_loaded = True
        except Exception as e:
            print(f"Warning: Could not load adapter: {e}")
            metrics.error = f"Adapter load failed: {e}"
            
        # 2. Initialize Components
        print("Initializing Scallop Engine and Titans Memory...")
        try:
            titans = TitansMemoryAdapter()
            scallop = ScallopEngine()
            scallop.set_titans_memory(titans)
        except Exception as e:
            metrics.error = f"Component init failed: {e}"
            print(metrics.error)
            return metrics

        # 3. Initialize Agent
        agent = ScallopTitansAgent(
            llm=model,
            tokenizer=tokenizer,
            titans_memory=titans,
            scallop_engine=scallop,
            config=AgentConfig(max_new_tokens=512)
        )
        
        # 4. Run Reasoning Task
        prompt = "Alice is Bob's mother. Bob is Carol's father. Who is Alice to Carol?"
        print(f"Prompt: {prompt}")
        
        response = agent.chat(prompt)
        metrics.final_answer = response
        
        # 5. Analyze Trace
        history = agent.history
        # Last assistant message
        full_response = history[-1].content
        metrics.trace.append(full_response)
        
        if "<|call_scallop|>" in full_response:
            metrics.scallop_invoked = True
        
        if "<|scallop_result|>" in full_response:
            # Check if result is not empty/error
            if "Error" not in full_response.split("<|scallop_result|>")[1]:
                 metrics.valid_syntax = True
        
        # Simple correctness check (Grandmother)
        if "grandmother" in response.lower():
            metrics.success = True
            
    except Exception as e:
        metrics.error = str(e)
        import traceback
        traceback.print_exc()
        
    metrics.latency_seconds = time.time() - start_time
    return metrics

if __name__ == "__main__":
    result = run_verification()
    
    # Output JSON for automation
    output = dataclasses.asdict(result)
    print("\n--- VERIFICATION RESULT ---")
    print(json.dumps(output, indent=2))
    
    if result.success:
        print("\n✅ Verification SUCCESS")
        exit(0)
    else:
        print("\n❌ Verification FAILED")
        exit(1)
