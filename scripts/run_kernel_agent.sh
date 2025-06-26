#!/bin/bash

set -e

SESSION_NAME="kernel_agent_run"
WINDOW_1="kernel_training"
WINDOW_2="rollout_buffer"

# Kill existing session if it exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t $SESSION_NAME
fi

# Create new tmux session with first window for kernel training
tmux new-session -d -s $SESSION_NAME -n $WINDOW_1
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "cd $(pwd)" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "bash ./scripts/kernel-agent-example.sh" C-m

# Create second window for rollout buffer server
tmux new-window -t $SESSION_NAME -n $WINDOW_2
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "sleep 30 && cd slime_plugins/rollout_buffer && python buffer.py" C-m

# Attach to the session
tmux attach-session -t $SESSION_NAME