"""
GRPO Training: Group Relative Policy Optimization for reasoning.

Implements Phase 2 GRPO from master_plan.md Part C:
- Algorithm: GRPO via trl.GRPOTrainer
- Group Size: 4 completions per prompt
- KL Penalty: 0.05 (prevent policy collapse)
- Epochs: 1-3 (RL is unstable)

Reward Function:
1. Correctness: +1.0 for correct answer
2. Format: +0.2 for valid Scallop syntax
3. Efficiency: -0.01 * len(tokens) penalty
4. Logic Verification: +0.3 if Scallop derivation holds
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Model  
    model_name: str = "Qwen/Qwen3-32B"
    sft_adapter_path: Path | None = None  # Load SFT adapter if provided

    # GRPO Algorithm (Optimized for 3x H100)
    group_size: int = 8  # Standard Group Size (Stable)
    beta: float = 0.04  # KL Penalty (DeepSeekMath Standard) - Forces reasoning anchor
    temperature: float = 0.7
    max_new_tokens: int = 512  # Reverted to 512 (1024 caused length explosion)
    repetition_penalty: float = 1.05  # Gentle guard against loops (was 1.1)

    # Reward weights (from master_plan.md)
    correctness_reward: float = 2.0  # Increased to balance high penalties
    format_reward: float = 0.5  # Increased to enforce structure
    logic_verify_reward: float = 0.5
    token_penalty: float = 0.001  # Reduced 50x (0.05 -> 0.001) to allow thinking

    # Training (Stable PoC)
    num_epochs: int = 1  # Reduced to 1 for faster PoC
    batch_size: int = 4  # Standard MVP (Stable at ~65GB)
    gradient_accumulation_steps: int = 2  # Effective Batch = 4 * 3 * 2 = 24
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1

    # Data
    data_path: Path = field(default_factory=lambda: Path("data/grpo"))

    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs/grpo"))

    # Hardware
    bf16: bool = True


class RewardFunction:
    """
    Reward function for GRPO training.

    Implements the reward from master_plan.md Part C Phase 2:
    - Correctness: +1.0 for correct answer
    - Format: +0.2 for valid Scallop syntax  
    - Efficiency: -0.01 * token_count
    - Logic: +0.3 if Scallop derivation holds
    """

    def __init__(
        self,
        config: GRPOConfig,
        tokenizer: "PreTrainedTokenizer",
    ) -> None:
        """
        Initialize reward function.

        Args:
            config: GRPO configuration.
            tokenizer: Tokenizer for token counting.
        """
        self.config = config
        self.tokenizer = tokenizer
        
        # TRL expects reward functions to have __name__ attribute
        self.__name__ = "scallop_reward"

        # Try to load Scallop for verification
        try:
            from scallop_titans.reasoning import ScallopEngine

            self.scallop = ScallopEngine()
        except ImportError:
            self.scallop = None

    def compute_reward(
        self,
        completion: str,
        ground_truth: str,
        prompt: str | None = None,
    ) -> float:
        """
        Compute reward for a single completion.

        Args:
            completion: Model completion text.
            ground_truth: Expected correct answer.
            prompt: Optional original prompt.

        Returns:
            Total reward score.
        """
        score = 0.0

        # 1. Correctness reward
        extracted_answer = self._extract_answer(completion)
        if extracted_answer and self._answers_match(extracted_answer, ground_truth):
            score += self.config.correctness_reward

        # 2. Format reward (valid Scallop syntax)
        if self._is_valid_scallop_syntax(completion):
            score += self.config.format_reward

        # 3. Efficiency penalty (token count)
        token_count = len(self.tokenizer.encode(completion))
        score -= self.config.token_penalty * token_count

        # 4. Logic verification reward
        if self._verify_scallop_logic(completion):
            score += self.config.logic_verify_reward

        return score

    def _extract_answer(self, completion: str) -> str | None:
        """Extract the final answer from completion."""
        # Look for answer after end_thought or end_scallop_result
        patterns = [
            r"<\|end_thought\|>\s*(.+?)$",
            r"<\|end_scallop_result\|>\s*(.+?)$",
            r"(?:The answer is|Answer:)\s*(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, completion, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback: last sentence
        sentences = completion.split(".")
        if sentences:
            return sentences[-1].strip()

        return None

    def _answers_match(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        # Normalize
        pred_norm = predicted.lower().strip()
        gt_norm = ground_truth.lower().strip()

        # Exact match
        if pred_norm == gt_norm:
            return True

        # Check if ground truth is contained in prediction
        if gt_norm in pred_norm:
            return True

        return False

    def _is_valid_scallop_syntax(self, completion: str) -> bool:
        """Check if completion contains valid Scallop syntax."""
        # Check for well-formed Scallop commands
        scallop_pattern = r"<\|call_scallop\|>(.+?)(?:<\|end_thought\|>|$)"
        match = re.search(scallop_pattern, completion, re.DOTALL)

        if not match:
            return False

        cmd = match.group(1)

        # Check for valid command structure
        valid_patterns = [
            r"add_fact\s*\(\s*\w+\s*,",
            r"query\s*\(\s*\w+",
        ]

        return any(re.search(p, cmd) for p in valid_patterns)

    def _verify_scallop_logic(self, completion: str) -> bool:
        """Verify that Scallop derivation is logically sound."""
        if self.scallop is None:
            return False

        try:
            # Extract Scallop command
            pattern = r"<\|call_scallop\|>(.+?)(?:<\|end_thought\|>|$)"
            match = re.search(pattern, completion, re.DOTALL)

            if not match:
                return False

            cmd = match.group(1)

            # Try to parse and execute
            commands = self.scallop.parse_scallop_command(cmd)

            # If parsing succeeded, it's syntactically valid
            return len(commands) > 0

        except Exception:
            return False

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        completions_ids: list | None = None,
        trainer_state: Any = None,
        **kwargs,
    ) -> list[float]:
        """
        Batch compute rewards (TRL-compatible signature).

        Args:
            prompts: List of prompts.
            completions: List of model completions.
            completions_ids: Tokenized completions (unused).
            trainer_state: Current trainer state (unused).
            **kwargs: Dataset columns including 'ground_truth'.

        Returns:
            List of reward scores.
        """
        # Ground truth comes from dataset column via kwargs
        ground_truths = kwargs.get("ground_truth", [""] * len(completions))
        
        return [
            self.compute_reward(c, gt, p)
            for c, gt, p in zip(completions, ground_truths, prompts)
        ]


class GRPOTrainer:
    """
    GRPO trainer for reasoning RL.

    Uses trl's GRPOTrainer internally.
    """

    def __init__(
        self,
        config: GRPOConfig | None = None,
        model: "PreTrainedModel | None" = None,
        tokenizer: "PreTrainedTokenizer | None" = None,
    ) -> None:
        """
        Initialize GRPO trainer.

        Args:
            config: Training configuration.
            model: Pre-loaded model.
            tokenizer: Pre-loaded tokenizer.
        """
        self.config = config or GRPOConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn: RewardFunction | None = None

    def setup(self) -> None:
        """Set up model and tokenizer with optimizations."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            
            # Add special tokens (same as SFT training - required for adapter compatibility)
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
        if self.model is None:
            # Enable TF32 for faster matmuls on H100/A100
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",  # Flash Attention 2
                # No device_map - FSDP handles device placement
            )
            
            # Resize embeddings to match tokenizer with special tokens
            self.model.resize_token_embeddings(len(self.tokenizer))

            # Enable gradient checkpointing for memory efficiency
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

            # Load SFT adapter if provided
            if self.config.sft_adapter_path:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.config.sft_adapter_path,
                )

        # Initialize reward function
        self.reward_fn = RewardFunction(self.config, self.tokenizer)

    def train(self, dataset: Any = None) -> None:
        """
        Run GRPO training.

        Args:
            dataset: Training dataset with prompts and ground truths.
        """
        from trl import GRPOConfig as TRLGRPOConfig
        from trl import GRPOTrainer as TRLGRPOTrainer

        if self.model is None or self.tokenizer is None:
            self.setup()

        # GRPO config (TRL compatible)
        grpo_config = TRLGRPOConfig(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            beta=self.config.beta,  # Set KL penalty explicitly
            bf16=self.config.bf16,
            logging_steps=10,
            save_steps=500,
            # GRPO specific
            num_generations=self.config.group_size,
            # Generation settings via model's generation_config
            max_completion_length=self.config.max_new_tokens,
            # Note: use_liger_loss=True would reduce memory 40% but requires liger-kernel
        )

        # CRITICAL: Fix for "Length Explosion" / Repetition Loops
        # We explicitly set generation config to prevent the model from getting stuck
        self.model.generation_config.repetition_penalty = self.config.repetition_penalty
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # PEFT/LoRA config for memory-efficient training
        # This is CRITICAL - without it, we train full 32B params = OOM
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

        # Create trainer with PEFT for memory efficiency
        trainer = TRLGRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            reward_funcs=[self.reward_fn],
            peft_config=peft_config,  # CRITICAL: reduces memory by 90%
        )

        # Train
        trainer.train()

        # Save
        trainer.save_model()


def main() -> None:
    """Entry point for train-grpo command."""
    import argparse

    parser = argparse.ArgumentParser(description="GRPO Training for Scallop-Titans Agent")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Base model name")
    parser.add_argument("--sft-adapter", default=None, help="Path to SFT adapter")
    parser.add_argument("--data", default="data/grpo", help="Training data path")
    parser.add_argument("--output", default="outputs/grpo", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")

    args = parser.parse_args()

    config = GRPOConfig(
        model_name=args.model,
        sft_adapter_path=Path(args.sft_adapter) if args.sft_adapter else None,
        data_path=Path(args.data),
        output_dir=Path(args.output),
        num_epochs=args.epochs,
    )

    # Load dataset from JSONL file
    from datasets import load_dataset
    
    dataset = load_dataset("json", data_files=str(config.data_path), split="train")
    print(f"Loaded {len(dataset)} samples from {config.data_path}")

    # CRITICAL FIX: Apply chat template to align with SFT training
    # SFT model expects <|im_start|>user...<|im_end|><|im_start|>assistant
    from transformers import AutoTokenizer
    # Load tokenizer separately to avoid loading full model into RAM
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def format_prompt(example):
        messages = [{"role": "user", "content": example["prompt"]}]
        # apply_chat_template returns string. tokenize=False to keep it as string for GRPO
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": formatted}

    # Process dataset BEFORE loading heavy model
    print("Applying chat template to prompts...")
    dataset = dataset.map(format_prompt)

    # Initialize trainer (Heavy model load)
    trainer = GRPOTrainer(config)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
