from edge_env import EdgeEnv

env = EdgeEnv()
obs, _ = env.reset()

print("Initial observation:", obs)

action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated

print("Next observation:", next_obs)
print("Reward:", reward)
print("Done:", done)
