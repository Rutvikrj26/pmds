import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from scallop_titans.agent import ScallopTitansAgent, AgentConfig, ChatMessage
from scallop_titans.memory import TitansMemoryAdapter
from scallop_titans.reasoning import ScallopEngine
import re

def clean_response(text):
    """
    Extracts the thought trace and cleans the visible response.
    Returns: (visible_response, trace)
    """
    # 1. Extract thought block
    thought_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if thought_match:
        trace = thought_match.group(1).strip()
        # Remove thought from visible text
        visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    else:
        # If no <think> block, the whole thing is the response, 
        # unless it's just a trace? For now assume it's mixed or empty trace.
        trace = "No separate reasoning trace found."
        visible = text

    # 2. Add Scallop calls to trace if they exist outside <think>
    # (Regex for <|call_scallop|>...<|end_thought|>)
    # For now, just include the full raw text in trace if desired, 
    # but the user specific requested separation.
    # Let's stick to the thought block separation + cleaning.
    
    # 3. Clean system tokens
    visible = visible.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    
    return visible, trace

# Global state
agent = None

def load_agent():
    global agent
    if agent is not None:
        return "Agent already loaded."
    
    BASE_MODEL = "outputs/sft_merged"
    ADAPTER_PATH = "outputs/grpo/checkpoint-2000"
    
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
    
    print("Initializing components...")
    titans = TitansMemoryAdapter()
    scallop = ScallopEngine()
    scallop.set_titans_memory(titans)
    
    agent = ScallopTitansAgent(
        llm=model,
        tokenizer=tokenizer,
        titans_memory=titans,
        scallop_engine=scallop,
        config=AgentConfig(max_new_tokens=512)
    )
    return "Agent loaded successfully!"

def chat_fn(message, history):
    global agent
    if agent is None:
        # Auto-load if not loaded
        load_agent()
    
    # Process message
    response = agent.chat(message)
    
    # Extract trace from history for display
    # The agent history has the full response including <call_scallop>
    full_response = agent.history[-1].content
    
    # Format the visible response and the hidden trace
    # If the response contains reasoning tokens, we might want to separate them?
    # For now, we return the clean final answer
    
    # Update history for Gradio
    # Gradio history is list of [user, bot]
    # We return the new history
    # Or generator for streaming? Let's keep it simple first.
    
    # Return the raw full response so we can process it in respond()
    return full_response

with gr.Blocks(title="Scallop-Titans Agent Demo") as demo:
    gr.Markdown("# Scallop-Titans Agent: Neuro-Symbolic Reasoning")
    
    with gr.Row():
        load_btn = gr.Button("Reload Agent")
        status_text = gr.Textbox(label="Status", value="Ready to load", interactive=False)
    
    chatbot = gr.Chatbot(label="Conversation", height=400)
    msg = gr.Textbox(label="Message", placeholder="Alice is Bob's mother...")
    clear = gr.ClearButton([msg, chatbot])
    
    with gr.Accordion("Reasoning Trace & Scallop Calls", open=False):
        trace_box = gr.Code(label="Last Trace", language="markdown")
    
    def respond(message, chat_history):
        full_response = chat_fn(message, chat_history)
        
        # Clean the response for display
        bot_message, trace = clean_response(full_response)
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        
        return "", chat_history, trace

    msg.submit(respond, [msg, chatbot], [msg, chatbot, trace_box])
    load_btn.click(load_agent, outputs=[status_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
