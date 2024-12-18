#!/bin/bash

# Generate a timestamp for folder naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="replays/run_$TIMESTAMP"

# Create the directory
mkdir -p "$OUTPUT_DIR"

# Run the Lux AI tournament and save the replay
luxai-s3 agents/efficient_explorer_agent/main.py \
         agents/baseline_agent/main.py \
         agents/shooting_vision_agent/main.py \
         agents/energy_conserver_agent/main.py \
          --tournament --output "$OUTPUT_DIR/replay.json" --replay.no-compressed-obs --verbose 3 --tournament-cfg-max-episodes 20 --tournament-cfg-concurrent 5

echo "Replay saved to: $OUTPUT_DIR/replay.json"
