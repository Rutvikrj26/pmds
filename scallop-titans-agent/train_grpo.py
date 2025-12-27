import os
import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from rewards import format_reward_func, syntax_reward_func, execution_reward_func

# --- Configuration ---
MODEL_PATH = "outputs/sft_full/checkpoint-138"  # Start from SFT
OUTPUT_DIR = "outputs/grpo_v1"
DATASET_PATH = "data/multi_domain/combined_multi_domain.jsonl"
Wait_Duration = 0


def process_dataset(examples):
    """
    Convert chat messages to prompt/completion format for GRPO.
    Prompt: System + User + <|assistant|>\n<|start_thought|>
    Completion: The rest (Ground Truth for reference, though GRPO generates its own)
    """
    prompts = []
    # We don't necessarily need "completions" column for training generation,
    # but we might want it for the correctness reward if we implemented it.
    # For now, we rely on implicit learning via execution reward.

    for msgs in examples["messages"]:
        try:
            system = msgs[0]["content"]
            user = msgs[1]["content"]
            # Force the thought process start
            prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n<|start_thought|>"
            prompts.append(prompt)
        except (IndexError, KeyError):
            prompts.append("")

    return {"prompt": prompts}


def main():
    print(f"Loading SFT Model from: {MODEL_PATH}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Dataset
    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # Preprocess
    print("Preprocessing dataset...")
    dataset = dataset.map(process_dataset, batched=True)
    # Filter empty
    dataset = dataset.filter(lambda x: len(x["prompt"]) > 0)
    print(f"Training on {len(dataset)} samples")

    # GRPO Config
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6,  # Lower LR for RL
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        per_device_train_batch_size=4,  # Rollout BS
        gradient_accumulation_steps=4,
        num_generations=16,  # G=16 (Group size)
        max_prompt_length=512,
        max_completion_length=512,
        num_train_epochs=1,  # 1 epoch of RL is usually enough or too much
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,  # Disable vLLM due to env instability
        vllm_gpu_memory_utilization=None,
        report_to="none",
    )

    # Model (GRPOTrainer loads distinct policy/ref models automatically usually,
    # but passing model_path string lets it handle it)

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        reward_funcs=[format_reward_func, syntax_reward_func, execution_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO Training...")
    trainer.train()

    print("Saving GRPO model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
