from luxai_s3.wrappers import LuxAIS3GymEnv
def test_env():
    print("Testing environment setup...")
    env = LuxAIS3GymEnv(numpy_output=True, max_episode_length=100)
    print("Environment created successfully")
    
    # Create agents
    from submissions.best_agent_attacker import BestAgentAttacker
    from submissions.best_agent_better_shooter import BestAgentBetterShooter
    
    agent1 = BestAgentAttacker("player_0", env.env_cfg)
    agent2 = BestAgentBetterShooter("player_1", env.env_cfg)
    print("Agents created successfully")
    
    # Reset environment
    obs = env.reset(seed=42)
    print("Environment reset successfully")
    print("Initial observation shape:", obs[0].shape)
    
    # Test agent step
    action1 = agent1.act(obs[0])
    action2 = agent2.act(obs[1])
    print("Agents generated actions successfully")
    
    return True

if __name__ == "__main__":
    test_env()
