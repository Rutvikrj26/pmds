import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

# --- Configuration ---
MODEL_NAME = "Salesforce/xLAM-1B-fc-r"
DATASET_PATH = "data/multi_domain/combined_multi_domain.jsonl"
OUTPUT_DIR = "outputs/sft_full"
MAX_SEQ_LENGTH = 2048

def formatting_prompts_func(example):
    """
    Format the conversation for training:
    System + User -> Prompt
    Assistant -> Completion (What we train on)
    """
    output_texts = []
    
    # Check if we have 'messages' key
    if 'messages' not in example:
        return []
        
    msgs = example['messages']
    
    # Ensure it's a list of messages (User/Assistant etc.)
    if not isinstance(msgs, list):
         return []

    # Extract content (Assumes logic: System -> User -> Assistant)
    # We might have variable length, but for this dataset we generated fixed 3-turn
    try:
        system = msgs[0]['content']
        user = msgs[1]['content']
        assistant = msgs[2]['content']
        
        # Simple chat format
        text = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}<|endoftext|>"
        output_texts.append(text)
    except (IndexError, KeyError):
        return [] # Skip malformed
        
    return output_texts

def main():
    print(f"Loading model: {MODEL_NAME}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for fp16
    
    # Load Model (Full 1B fits easily on H100, no need for QLoRA unless desired)
    # Using bfloat16 for H100 stability
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2", # Required for packing=True
        device_map="auto" # Accelerate handles this via DDP usually, but auto is safe fallback
    )
    
    # Load Dataset
    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Dataset size: {len(dataset)}")
    
    # Training Arguments (SFTConfig in TRL 0.25+)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field=None, # We use formatting_func not raw text field
        max_length=MAX_SEQ_LENGTH,
        packing=True, # Packs multiple short samples into 2048 context for speed
        
        num_train_epochs=3,
        per_device_train_batch_size=64, # Increased for single H100
        gradient_accumulation_steps=4,  # 64*4 = 256 effective batch
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        fp16=False,
        bf16=True, # H100 native
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
