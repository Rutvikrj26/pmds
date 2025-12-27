import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "outputs/sft_full/checkpoint-138"

print(f"Loading from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)

prompt = """<|system|>
You are a reasoning agent. Use <|call_scallop|> to invoke logic.
<|user|>
Bob is the father of Alice. Who is the child of Bob?
<|assistant|>
<|start_thought|>"""  # Force the thought process

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print("-" * 20)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
print("-" * 20)
