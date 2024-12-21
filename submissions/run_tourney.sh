#!/bin/bash

# Generate a timestamp for folder naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="replays/run_$TIMESTAMP"

# Create the directory
mkdir -p "$OUTPUT_DIR"

# Run the Lux AI tournament and save the replay
luxai-s3 best_agent_attacker_two.py \
         best_agent_attacker_different_deduce.py \
         best_agent_attacker_three.py \
         best_agent_attacker_4.py \
         best_agent_attacker_no_nebula.py \
         best_agent_attacker_no_nebula_two.py \
         best_agent_attacker_no_nebula_different_explore_no_attack.py \
         best_agent.py \
         best_agent_no_nebula.py \
         best_agent_better_shooter.py \
         newest_agent/main.py \
         best_agent/main.py \
         best_agent_attacker/main.py \
        --tournament --output "$OUTPUT_DIR/replay.json" --tournament-cfg-max-episodes 5000 --tournament-cfg-concurrent 15

echo "Replay saved to: $OUTPUT_DIR/replay.json"
