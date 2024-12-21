#!/bin/bash

# Generate a timestamp for folder naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="replays/run_$TIMESTAMP"

# Create the directory
mkdir -p "$OUTPUT_DIR"

# Run the Lux AI tournament and save the replay
luxai-s3 agents/baseline_agent/main.py \
         submissions/relic_optimizer.py \
         submissions/best_agent_attacker_two.py \
         submissions/best_agent_attacker_three.py \
         submissions/best_agent_attacker_4.py \
         submissions/best_agent_attacker_no_nebula.py \
         submissions/relic_optimizer_refactored.py \
         submissions/relic_optimizer_with_energy_pathing.py \
         submissions/relic_optimizer_with_reward_deducing_logic.py \
         submissions/relic_optimizer_without_tile_detection.py \
         submissions/best_agent.py \
         submissions/best_agent_no_nebula.py \
         submissions/best_agent_better_shooter.py \
         agents/better_shooting_vision_agent/main.py \
         agents/roles_based_agent/main.py \
         agents/relic_hunting_shooting_vision_agent/main.py \
         agents/newest_agent/main.py \
         agents/best_agent/main.py \
         agents/best_agent_attacker/main.py \
        --tournament --output "$OUTPUT_DIR/replay.json" --tournament-cfg-max-episodes 5000 --tournament-cfg-concurrent 15

echo "Replay saved to: $OUTPUT_DIR/replay.json"
