import os
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

def test_env():
    print("Testing environment setup...")
    # Set up base directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("replays", exist_ok=True)
    
    # Import agents
    from submissions.best_agent_attacker import BestAgentAttacker
    from submissions.best_agent_better_shooter import BestAgentBetterShooter
    
    # Test with different seeds to capture diverse scenarios
    seeds = [42, 123, 456]  # Different seeds for diverse scenarios
    for seed in seeds:
        print(f"\nTesting with seed {seed}")
        
        # Create game-specific directories
        game_dir = f"logs/game_{seed}"
        replay_dir = f"replays/game_{seed}"
        os.makedirs(game_dir, exist_ok=True)
        os.makedirs(replay_dir, exist_ok=True)
        
        # Configure logging for this game
        import logging
        # Reset logging handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Configure new handlers
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{game_dir}/debug_log.txt"),
                logging.StreamHandler()
            ]
        )
        
        # Initialize environment
        base_env = LuxAIS3GymEnv(numpy_output=True)
        env = RecordEpisode(base_env, save_dir=replay_dir)
        print("Environment created successfully")
        
        # Create agents
        agent1 = BestAgentAttacker("player_0", base_env.env_params)
        agent2 = BestAgentBetterShooter("player_1", base_env.env_params)
        print("Agents created successfully")
        
        # Reset environment
        obs = env.reset(seed=seed)
        print("Environment reset successfully")
        print("Initial observation shape:", obs[0].shape)
        
        # Run 50 steps to generate meaningful logs
        for step in range(50):  # Run 50 steps per seed
            action1 = agent1.act(obs[0])
            action2 = agent2.act(obs[1])
            print(f"Step {step}: Agents generated actions successfully")
            obs, reward, terminated, truncated, info = env.step({"player_0": action1, "player_1": action2})
            if terminated or truncated:
                break
        
        # Close environment
        env.close()
    
    return True

if __name__ == "__main__":
    test_env()
