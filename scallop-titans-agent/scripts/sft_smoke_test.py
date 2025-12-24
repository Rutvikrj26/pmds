#!/usr/bin/env python
"""
SFT Smoke Test: Validate the training pipeline end-to-end.

Uses a small model (Qwen2.5-0.5B) and minimal data (100 samples).
Runs for 1 epoch with 10 steps to verify:
1. Data loading works
2. Tokenization works
3. Model setup + LoRA works
4. Training loop runs without errors
"""

from pathlib import Path
from scallop_titans.training.sft import SFTConfig, SFTTrainer

def main():
    print("=" * 60)
    print("SFT SMOKE TEST")
    print("=" * 60)
    
    # Minimal config for quick validation
    config = SFTConfig(
        # Use a small model for fast testing
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Small, fast-loading
        use_lora=True,
        lora_rank=8,  # Lower rank for speed
        lora_alpha=16,
        
        # Minimal training
        num_epochs=1,
        batch_size=2,  # Small batch
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        max_seq_length=512,  # Shorter context for speed
        
        # Data
        data_path=Path("data/sft_smoke_test.jsonl"),
        
        # Output
        output_dir=Path("outputs/sft_smoke_test"),
        
        # Hardware
        num_gpus=1,
        bf16=True,
    )
    
    print(f"Config: {config}")
    print(f"Data: {config.data_path} (exists: {config.data_path.exists()})")
    
    # Create trainer
    trainer = SFTTrainer(config)
    
    # Run setup (loads model, tokenizer, applies LoRA)
    print("\n[1/2] Setting up model and tokenizer...")
    trainer.setup()
    
    print("\n[2/2] Starting training (1 epoch)...")
    trainer.train()  # Let trainer handle dataset loading
    
    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED!")
    print("=" * 60)
    print(f"Adapter saved to: {config.output_dir}")

if __name__ == "__main__":
    main()
