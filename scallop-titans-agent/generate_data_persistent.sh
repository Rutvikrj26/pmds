#!/bin/bash
# Persistent Data Generation Script
# Runs in tmux so it survives terminal/session closures.
#
# Usage:
#   ./generate_data_persistent.sh        # Start generation
#   tmux attach -t datagen              # Attach to view progress
#   Ctrl+B, D                           # Detach (leave running)

set -e

SESSION_NAME="datagen"

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists!"
    echo "To attach: tmux attach -t $SESSION_NAME"
    echo "To kill:   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create new tmux session and run the data generation script
echo "Starting data generation in tmux session '$SESSION_NAME'..."
tmux new-session -d -s $SESSION_NAME './generate_data.sh'

echo ""
echo "✅ Data generation started in background!"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME   # View progress"
echo "  tmux kill-session -t $SESSION_NAME  # Stop"
echo ""
echo "The process will continue even if you close this terminal."
