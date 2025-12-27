import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scallop_titans.data.generators.domain_registry import (
    get_all_domains,
    get_domain_generator,
)
import random
import re

MODEL_PATH = "outputs/sft_full/checkpoint-138"  # Epoch 2 Checkpoint
BASE_MODEL = "Salesforce/xLAM-1B-fc-r"


def generate_test_samples(n=50):
    """Generate N fresh samples from random domains"""
    domains = get_all_domains()
    samples = []
    print(f"Generating {n} fresh samples for evaluation...")
    for _ in range(n):
        dom = random.choice(domains)
        gen = get_domain_generator(dom)()
        sample = gen.generate_sample()
        if sample:
            samples.append(sample)

    print(f"Generated {len(samples)} samples.")
    return samples


def evaluate(model_path):
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Failed to load SFT model: {e}")
        print("Falling back to base model for baseline check...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    samples = generate_test_samples(50)

    valid_syntax = 0
    attempted_tool = 0

    print("\nRunning Inference...")
    for i, s in enumerate(samples):
        chat = s.to_chat_format()
        # Prompt: System + User + Assistant Start
        prompt = f"<|system|>\n{chat['messages'][0]['content']}\n<|user|>\n{chat['messages'][1]['content']}\n<|assistant|>\n<|start_thought|>"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy for stability
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract assistant response (after prompt)
        # Simplistic split
        response = generated_text.split("<|assistant|>")[-1].strip()

        # Check for tool usage
        if "<|call_scallop|>" in response:
            attempted_tool += 1
            # Check syntax: <|call_scallop|> query(...) <|end_thought|>
            # Regex for simplistic check
            match = re.search(
                r"<\|call_scallop\|>(.*?)<\|end_thought\|>", response, re.DOTALL
            )
            if match:
                content = match.group(1).strip()
                # Basic Datalog check (balanced parens, ends with .)
                if "(" in content and ")" in content:
                    valid_syntax += 1

        if i < 3:
            print(f"\n[Sample {i}]")
            print(f"Prompt: {prompt[-100:]}...")  # Show tail
            print(f"Output: {response}")

    print("=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total Samples: {len(samples)}")
    print(f"Tool Attempts: {attempted_tool} ({attempted_tool/len(samples)*100:.1f}%)")
    print(f"Valid Syntax:  {valid_syntax} ({valid_syntax/len(samples)*100:.1f}%)")
    print("=" * 40)


if __name__ == "__main__":
    evaluate(MODEL_PATH)
