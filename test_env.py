from edge_env import EdgeEnv

env = EdgeEnv()
obs = env.reset()
print("Initial observation:", obs)

action = env.action_space.sample()
obs, reward, done, info = env.step(action)

print("Next observation:", obs)
print("Reward:", reward)
print("Done:", done)
