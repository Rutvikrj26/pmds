#!/bin/bash
# Persistent GRPO training wrapper - runs in tmux for SSH disconnect survival

SESSION_NAME="grpo-train"

# Kill existing session if any
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

echo "Starting GRPO training in tmux session '$SESSION_NAME'..."

# Create new detached session running the training script
tmux new-session -d -s "$SESSION_NAME" "cd /home/rutvik/pmds/scallop-titans-agent && ./train_grpo.sh 2>&1 | tee grpo_training.log"

echo ""
echo "✅ GRPO training started in background!"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME   # View progress"
echo "  tmux kill-session -t $SESSION_NAME  # Stop"
echo ""
echo "The process will continue even if you close this terminal."
