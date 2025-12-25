
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Configuration
BASE_MODEL = "outputs/sft_merged"
ADAPTER_PATH = "outputs/grpo/checkpoint-1000"
DATA_PATH = "data/grpo_train_6k.jsonl"

def debug_inference():
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )

    print(f"Loading adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    # Load a prompt
    with open(DATA_PATH, "r") as f:
        line = f.readline()
        data = json.loads(line)
        prompt = data["prompt"] # TRL expects a list of dicts usuallly, but our dataset might be formatted differently?
        # Let's check the format. Earlier we saw it was just a list of messages or prompt text.
        # Wait, if we use the prompt directly from the file.
    
    # Actually, the dataset might be {"prompt": [messages], "completion": ...} or similar.
    # Let's inspect the data format first or just try to assume "prompt" key exists nicely formatted.
    # Based on previous work, "prompt" is the key.
    
    # FORMAT PROMPT (Match Training Logic)
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    print("\nXXX Generating with Repetition Penalty 1.1 XXX\n")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512, 
        temperature=0.7,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    print("-" * 40)
    print("PROMPT:")
    print(prompt)
    print("-" * 40)
    print("GENERATED:")
    print(generated_text)
    print("-" * 40)

if __name__ == "__main__":
    debug_inference()
