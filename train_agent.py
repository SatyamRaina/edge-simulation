# train_agent.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from edge_env import EdgeEnv
from evaluate_agent import evaluate

# List of synthetic telemetry files
telemetry_files = [
    "synthetic_telemetry.csv",
    "synthetic_telemetry_lowmob.csv",
    "synthetic_telemetry_highload.csv"
]

# Ensure output folders
os.makedirs("checkpoints_50k", exist_ok=True)
os.makedirs("ppo_tensorboard_50k", exist_ok=True)

for file in telemetry_files:
    print(f"\n📊 Training for 50k steps on: {file}")

    # Create environment with Monitor + DummyVecEnv for logging
    def make_env():
        return Monitor(EdgeEnv(csv_file=file))
    
    env = DummyVecEnv([make_env])

    # Logging path
    log_dir = f"./ppo_tensorboard_50k/PPO_{file.split('.')[0]}"
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # Train the agent
    model.learn(total_timesteps=50000)
    
    # Save model
    model_path = f"checkpoints_50k/rl_mip_agent_{file.split('.')[0]}_50k"
    model.save(model_path)
    print(f"✅ Saved model to {model_path}.zip")

    # Evaluate using raw (non-vec) env
    raw_env = EdgeEnv(csv_file=file)
    evaluate(model, raw_env, file)
