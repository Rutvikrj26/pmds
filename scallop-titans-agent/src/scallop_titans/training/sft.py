"""
SFT Training: Supervised Fine-Tuning for tool-calling.

Implements Phase 1 SFT from master_plan.md Part C:
- Data: Pillar I (200k synthetic traces) + Pillar II (CLUTRR 10k)
- Method: LoRA (rank=16, alpha=32) via peft
- Trainer: SFTTrainer from trl
- Epochs: 1-2 (LLMs overtrain quickly on synthetic data)
- Batch Size: 8 (with gradient accumulation = 4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Model
    model_name: str = "Qwen/Qwen3-32B"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training
    num_epochs: int = 2
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 4096

    # Data
    data_path: Path = field(default_factory=lambda: Path("data/sft"))

    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs/sft"))

    # Hardware (3x H100 available per master_plan.md)
    num_gpus: int = 2
    bf16: bool = True


class ScallopTraceDataset(Dataset):
    """
    Dataset of Scallop reasoning traces for SFT.

    Format from master_plan.md Part B Section 2:
    {
        "messages": [
            {"role": "system", "content": "You are a reasoning agent..."},
            {"role": "user", "content": "Alice's mother is Betty..."},
            {"role": "assistant", "content": "<|start_thought|>...<|call_scallop|>..."}
        ]
    }
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer: "PreTrainedTokenizer",
        max_length: int = 4096,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_path: Path to the data directory or file.
            tokenizer: The tokenizer to use.
            max_length: Maximum sequence length.
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[dict[str, Any]] = []

        self._load_data()

    def _load_data(self) -> None:
        """Load data from files."""
        import json

        if self.data_path.is_file():
            with open(self.data_path) as f:
                if self.data_path.suffix == ".jsonl":
                    self.examples = [json.loads(line) for line in f]
                else:
                    self.examples = json.load(f)
        elif self.data_path.is_dir():
            for file in self.data_path.glob("*.json*"):
                with open(file) as f:
                    if file.suffix == ".jsonl":
                        self.examples.extend([json.loads(line) for line in f])
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.examples.extend(data)
                        else:
                            self.examples.append(data)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a tokenized example."""
        example = self.examples[idx]

        # Format as chat
        messages = example.get("messages", [])
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0),
        }


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for Scallop tool-calling.

    Uses trl's SFTTrainer internally with LoRA configuration.
    """

    def __init__(
        self,
        config: SFTConfig | None = None,
        model: "PreTrainedModel | None" = None,
        tokenizer: "PreTrainedTokenizer | None" = None,
    ) -> None:
        """
        Initialize the SFT trainer.

        Args:
            config: Training configuration.
            model: Pre-loaded model (will be loaded if None).
            tokenizer: Pre-loaded tokenizer (will be loaded if None).
        """
        self.config = config or SFTConfig()
        self.model = model
        self.tokenizer = tokenizer

    def setup(self) -> None:
        """Set up model, tokenizer, and LoRA configuration."""
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )

            # Add special tokens for Scallop
            special_tokens = {
                "additional_special_tokens": [
                    "<|call_scallop|>",
                    "<|scallop_result|>",
                    "<|end_scallop_result|>",
                    "<|start_thought|>",
                    "<|end_thought|>",
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens)

        # Load model with optimizations for H100
        # Note: We don't use device_map="auto" here because FSDP handles device placement
        if self.model is None:
            # Enable TF32 for faster matmuls on H100/A100
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",  # Flash Attention 2 for 2-4x speedup
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Enable gradient checkpointing to trade compute for memory
            # This allows larger batch sizes which improve throughput
            self.model.gradient_checkpointing_enable()

        # Apply LoRA
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def train(self, dataset: Dataset | None = None) -> None:
        """
        Run SFT training.

        Args:
            dataset: Training dataset. Will be loaded from config if None.
        """
        from trl import SFTConfig as TRLSFTConfig
        from trl import SFTTrainer as TRLSFTTrainer

        if self.model is None or self.tokenizer is None:
            self.setup()

        # Load dataset using HuggingFace datasets (TRL requires this format)
        from datasets import Dataset as HFDataset
        import json
        
        if dataset is None:
            # Load from JSONL file
            with open(self.config.data_path) as f:
                if str(self.config.data_path).endswith(".jsonl"):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            
            # TRL SFTTrainer expects a "text" column or "messages" column
            # Our data has "messages" already
            hf_dataset = HFDataset.from_list(data)
        else:
            # If already a HF dataset, use directly
            hf_dataset = dataset

        # Split dataset for evaluation (5% for eval)
        split_dataset = hf_dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        # Training config with speed optimizations and early stopping
        training_config = TRLSFTConfig(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            bf16=self.config.bf16,
            # Speed optimizations
            dataloader_num_workers=4,  # Parallel data loading
            dataloader_pin_memory=True,  # Faster GPU transfers
            gradient_checkpointing=True,  # Memory efficient
            # Evaluation & Early Stopping
            eval_strategy="steps",
            eval_steps=50,  # Evaluate every 50 steps
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Logging
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            # Performance
            tf32=True,  # TF32 on H100/A100
        )

        # Early stopping callback - stop if loss doesn't improve for 5 evaluations
        from transformers import EarlyStoppingCallback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=5,  # Stop after 5 evals without improvement
            early_stopping_threshold=0.0005,  # Min improvement threshold (0.05%)
        )

        # Create trainer with evaluation dataset and early stopping
        trainer = TRLSFTTrainer(
            model=self.model,
            args=training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            callbacks=[early_stopping],
        )

        # Train
        print("Starting training with early stopping (patience=3, threshold=1%)")
        trainer.train()

        # Save
        trainer.save_model()

    def save_adapter(self, path: Path | None = None) -> None:
        """Save the LoRA adapter."""
        path = path or self.config.output_dir / "adapter"
        if self.model is not None:
            self.model.save_pretrained(path)


def main() -> None:
    """Entry point for train-sft command."""
    import argparse

    parser = argparse.ArgumentParser(description="SFT Training for Scallop-Titans Agent")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Base model name")
    parser.add_argument("--data", default="data/sft", help="Training data path")
    parser.add_argument("--output", default="outputs/sft", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")

    args = parser.parse_args()

    config = SFTConfig(
        model_name=args.model,
        data_path=Path(args.data),
        output_dir=Path(args.output),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    trainer = SFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
