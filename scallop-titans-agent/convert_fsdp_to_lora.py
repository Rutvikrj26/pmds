#!/usr/bin/env python3
"""
Convert FSDP checkpoint to LoRA adapter format.

This script:
1. Loads the FSDP sharded checkpoint 
2. Extracts LoRA weights
3. Saves as standard adapter format

Usage:
    python convert_fsdp_to_lora.py --checkpoint outputs/sft_full/checkpoint-300 --output outputs/sft_lora_adapter
"""

import argparse
import os
from pathlib import Path

import torch


def convert_fsdp_to_lora(checkpoint_path: Path, output_path: Path, base_model: str = "Qwen/Qwen3-32B"):
    """
    Convert FSDP checkpoint to LoRA adapter.
    
    The FSDP checkpoint contains the full model with merged LoRA weights.
    We need to extract just the LoRA parameters.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    
    print(f"Loading base model: {base_model}")
    
    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # Add special tokens (same as SFT training)
    special_tokens = {
        "additional_special_tokens": [
            "<|call_scallop|>",
            "<|scallop_result|>",
            "<|end_scallop_result|>",
            "<|start_thought|>",
            "<|end_thought|>",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Apply same LoRA config as SFT
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Now load the FSDP state dict
    print(f"\nLoading FSDP checkpoint from: {checkpoint_path}")
    
    fsdp_model_path = checkpoint_path / "pytorch_model_fsdp_0"
    
    if not fsdp_model_path.exists():
        raise FileNotFoundError(f"FSDP model path not found: {fsdp_model_path}")
    
    # Load distributed checkpoint
    # Note: This requires running on the same number of GPUs as training
    state_dict = {}
    dcp_load(
        state_dict={"model": state_dict},
        storage_reader=torch.distributed.checkpoint.FileSystemReader(str(fsdp_model_path)),
    )
    
    # Load state dict into model
    model.load_state_dict(state_dict, strict=False)
    
    print(f"\nSaving LoRA adapter to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save just the adapter  
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("\n✅ Conversion complete!")
    print(f"   Adapter saved to: {output_path}")
    

def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to LoRA adapter")
    parser.add_argument("--checkpoint", required=True, help="Path to FSDP checkpoint")
    parser.add_argument("--output", required=True, help="Output path for LoRA adapter")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Base model name")
    
    args = parser.parse_args()
    
    convert_fsdp_to_lora(
        Path(args.checkpoint),
        Path(args.output),
        args.model,
    )


if __name__ == "__main__":
    main()
