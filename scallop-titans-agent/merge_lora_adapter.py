#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and save as full checkpoint.

This creates a single, unified model with all LoRA weights merged,
eliminating dtype mismatch issues with FSDP.

Usage:
    python merge_lora_adapter.py --adapter outputs/sft_lora_adapter --output outputs/sft_merged
"""

import argparse
from pathlib import Path

import torch


def merge_lora_adapter(adapter_path: Path, output_path: Path, base_model: str = "Qwen/Qwen3-32B"):
    """
    Load LoRA adapter, merge into base model, save as full checkpoint.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"Base model: {base_model}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_path}")
    
    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load tokenizer from adapter (has special tokens)
    print("\n1. Loading tokenizer with special tokens...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    
    # Load base model
    print("\n2. Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # Resize embeddings to match tokenizer
    print("\n3. Resizing embeddings...")
    model.resize_token_embeddings(len(tokenizer))
    
    # Ensure embeddings are bf16
    model.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data.to(torch.bfloat16)
    model.lm_head.weight.data = model.lm_head.weight.data.to(torch.bfloat16)
    print(f"   Embedding dtype: {model.model.embed_tokens.weight.dtype}")
    
    # Load LoRA adapter
    print("\n4. Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.print_trainable_parameters()
    
    # Merge LoRA into base model
    print("\n5. Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    print("   Merge complete!")
    
    # Ensure all weights are bf16
    print("\n6. Converting all weights to bf16...")
    model = model.to(torch.bfloat16)
    
    # Save merged model
    print(f"\n7. Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("\n✅ Merge complete!")
    print(f"   Output: {output_path}")
    print(f"   This model can be loaded directly for GRPO (no adapter needed)")
    

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", default="outputs/sft_lora_adapter", help="Path to LoRA adapter")
    parser.add_argument("--output", default="outputs/sft_merged", help="Output path for merged model")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Base model name")
    
    args = parser.parse_args()
    
    merge_lora_adapter(
        Path(args.adapter),
        Path(args.output),
        args.model,
    )


if __name__ == "__main__":
    main()
