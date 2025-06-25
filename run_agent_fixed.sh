#!/bin/bash

# Clean up previous runs
echo "Cleaning up previous runs..."
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3

# Create log directory
mkdir -p logs
LOG_FILE="logs/agent_training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting agent training..."
echo "Log file: $LOG_FILE"

# Run the training script
./scripts/run_agent.sh 2>&1 | tee "$LOG_FILE"

echo "Training completed. Check log at: $LOG_FILE"