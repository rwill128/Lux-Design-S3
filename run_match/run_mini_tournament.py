"""
Mini tournament runner for quick agent verification.
Runs a smaller number of games (50) to test agent performance.
"""

import os
import json
from datetime import datetime
from run_match import BestAgentAttacker, BestAgentBetterShooter

def run_mini_tournament(num_games=50):
    """Run a mini tournament between agents."""
    results = []
    agents = [
        ("BestAgentAttacker", BestAgentAttacker),
        ("BestAgentBetterShooter", BestAgentBetterShooter)
    ]
    
    for i in range(num_games):
        print(f"\nGame {i+1}/{num_games}")
        # Alternate which agent goes first
        if i % 2 == 0:
            agent1, agent2 = agents[0], agents[1]
        else:
            agent1, agent2 = agents[1], agents[0]
            
        # Pass agent classes to evaluate_agents
        from run_match import evaluate_agents
        winner = evaluate_agents(agent1[1], agent2[1])
        
        # Record result
        results.append({
            "game": i+1,
            "player1": agent1[0],
            "player2": agent2[0],
            "winner": winner
        })
        
        print(f"Winner: {winner}")
    
    # Save results
    timestamp = int(datetime.now().timestamp())
    filename = f"tournament_results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    wins = {"BestAgentAttacker": 0, "BestAgentBetterShooter": 0}
    for game in results:
        if game["winner"] == "player_0":
            wins[game["player1"]] += 1
        else:
            wins[game["player2"]] += 1
    
    print("\nTournament Results:")
    for agent, win_count in wins.items():
        print(f"{agent}: {win_count} wins ({win_count/num_games*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    run_mini_tournament()
