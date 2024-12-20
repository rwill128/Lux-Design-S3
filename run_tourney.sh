#!/bin/bash

# Generate a timestamp for folder naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="replays/run_$TIMESTAMP"

# Create the directory
mkdir -p "$OUTPUT_DIR"

# Run the Lux AI tournament and save the replay
luxai-s3 agents/baseline_agent/main.py \
         agents/better_shooting_vision_agent/main.py \
         agents/best_agent_attacker/main.py \
          --tournament --output "$OUTPUT_DIR/replay.json" --replay.no-compressed-obs --verbose 3 --tournament-cfg-max-episodes 5 --tournament-cfg-concurrent 1

echo "Replay saved to: $OUTPUT_DIR/replay.json"
