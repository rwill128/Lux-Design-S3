#!/bin/bash

# Generate a timestamp for folder naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="replays/run_$TIMESTAMP"

# Create the directory
mkdir -p "$OUTPUT_DIR"

# Run the Lux AI tournament and save the replay
luxai-s3 attacker_different_deduce.py \
         attacker_three.py \
         attacker_4.py \
         attacker_4_no_nebula.py \
         attacker_4_explore5x5.py \
         normal_4_explore5x5.py \
         attacker_4_explore5x5_no_nebula.py \
         attacker_five.py \
         attacker_no_nebula.py \
         attacker_no_nebula_two.py \
         attacker_no_nebula_different_explore_no_attack.py \
         best.py \
         best_higher_energy.py \
         best_agent_no_nebula.py \
         best_agent_better_shooter.py \
         newest_agent/main.py \
         best_agent/main.py \
         best_agent/best_agent2_explore5x5.py \
        --tournament --output "$OUTPUT_DIR/replay.json" --tournament-cfg-max-episodes 5000 --tournament-cfg-concurrent 15

echo "Replay saved to: $OUTPUT_DIR/replay.json"
