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
         agents/better_shooting_vision_agent/main.py \
         agents/relic_hunting_shooting_vision_agent/main.py \
         agents/high_vision_agent/main.py \
         agents/roles_based_agent/main.py \
         agents/balanced_agent/main.py \
         agents/0_hunter/main.py \
         agents/1_hunter/main.py \
         agents/2_hunter/main.py \
         agents/3_hunter/main.py \
         agents/4_hunter/main.py \
         agents/5_hunter/main.py \
         agents/10_hunter/main.py \
         agents/energy_conserver_agent/main.py \
          --tournament --output "$OUTPUT_DIR/replay.json" --replay.no-compressed-obs --verbose 3 --tournament-cfg-max-episodes 500 --tournament-cfg-concurrent 10

echo "Replay saved to: $OUTPUT_DIR/replay.json"
