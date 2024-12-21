import os
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

def test_env():
    print("Testing environment setup...")
    base_env = LuxAIS3GymEnv(numpy_output=True)
    env = RecordEpisode(base_env, save_dir="replays/test_env")
    print("Environment created successfully")
    
    # Create agents
    from submissions.best_agent_attacker import BestAgentAttacker
    from submissions.best_agent_better_shooter import BestAgentBetterShooter
    
    # Create replay directory if it doesn't exist
    os.makedirs("replays/test_env", exist_ok=True)
    
    # Test with different seeds to capture diverse scenarios
    seeds = [42, 123, 456]  # Different seeds for diverse scenarios
    for seed in seeds:
        print(f"\nTesting with seed {seed}")
        agent1 = BestAgentAttacker("player_0", env.env_cfg)
        agent2 = BestAgentBetterShooter("player_1", env.env_cfg)
        print("Agents created successfully")
        
        # Reset environment
        obs = env.reset(seed=seed)
        print("Environment reset successfully")
        print("Initial observation shape:", obs[0].shape)
        
        # Run a few steps to generate logs
        for step in range(5):  # Run 5 steps per seed
            action1 = agent1.act(obs[0])
            action2 = agent2.act(obs[1])
            print(f"Step {step}: Agents generated actions successfully")
            obs, reward, terminated, truncated, info = env.step({"player_0": action1, "player_1": action2})
            if terminated or truncated:
                break
    
    return True

if __name__ == "__main__":
    test_env()
