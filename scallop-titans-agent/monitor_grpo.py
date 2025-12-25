#!/usr/bin/env python3
import re
import sys
import time
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("grpo_training.log")
OUTPUT_FILE = Path("training_status.txt")

def parse_logs():
    if not LOG_FILE.exists():
        return None

    # Read last 1000 lines to find latest stats
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
    except Exception:
        return None

    latest_metrics = None
    last_step = 0
    
    # Regex to find the JSON-like dict
    # TRL logs look like: {'loss': 0.123, ...}
    # But sometimes they are split. We look for lines containing 'loss' and 'reward'
    
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and "'loss':" in line and "'reward':" in line:
            try:
                # Convert single quotes to double quotes for valid JSON if needed
                # But it might be a python dict string. use ast literal_eval safely?
                # or just simple regex replace for basic types
                json_str = line.replace("'", '"')
                data = json.loads(json_str)
                latest_metrics = data
                # Try to extract step from progress bar line before it? 
                # Diffcult. We'll rely on the 'epoch' in the dict.
                break
            except Exception:
                continue

    # Also try to find the progress bar line: "15/2000 [28:31<63:03:25, 114.36s/it]"
    # Regex: (\d+)/(\d+) \[.*<.*, (.*)s/it\]
    total_steps = 2000 # hardcoded from config
    current_step_progress = 0
    speed = "N/A"
    eta = "N/A"

    for line in reversed(lines):
        match = re.search(r"(\d+)/(\d+) \[.*<([0-9:]+), (.*)s/it\]", line)
        if match:
            current_step_progress = int(match.group(1))
            total_steps = int(match.group(2)) # Update if found
            eta = match.group(3)
            speed = match.group(4)
            break
            
    return latest_metrics, current_step_progress, total_steps, eta, speed

def generate_report():
    data = parse_logs()
    if not data:
        return "Waiting for logs..."

    metrics, step, total, eta, speed = data
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"=== GRPO TRAINING STATUS ({timestamp}) ===\n"
    report += f"Progress: {step}/{total} ({(step/total)*100:.1f}%)\n"
    report += f"ETA: {eta} remaining\n"
    report += f"Speed: {speed} s/step\n\n"
    
    if metrics:
        report += f"Metrics:\n"
        report += f"  Loss:           {metrics.get('loss', 'N/A')}\n"
        report += f"  Reward (Mean):  {metrics.get('reward', 'N/A')}\n"
        report += f"  Clipped Ratio:  {metrics.get('completions/clipped_ratio', 'N/A')}\n"
        report += f"  Mean Length:    {metrics.get('completions/mean_length', 'N/A')}\n"
        report += f"  Loop Check:     {'OK' if metrics.get('completions/clipped_ratio', 1.0) < 0.8 else 'WARNING: High Clipping'}\n"
    else:
        report += "No metrics logged yet (waiting for Step 10).\n"
        
    report += "\n============================================\n"
    
    return report

def main():
    report = generate_report()
    
    # Write to file
    with open(OUTPUT_FILE, "a") as f:
        f.write(report + "\n")
    
    # Print to stdout
    print(limit_lines(report))

def limit_lines(text):
    return "\n".join(text.splitlines()[:20])

if __name__ == "__main__":
    main()
