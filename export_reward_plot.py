import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ✅ Set correct log directory (change to any valid run folder)
log_dir = "ppo_tensorboard_50k/PPO_synthetic_telemetry/PPO_1"

# Check if path exists
if not os.path.exists(log_dir):
    raise FileNotFoundError(f"Log directory not found: {log_dir}")

# Load TensorBoard logs
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# Print available tags for verification
print("✅ Available tags:", event_acc.Tags()["scalars"])

# Get episode reward mean
rewards = event_acc.Scalars("rollout/ep_rew_mean")

# Extract data
steps = [e.step for e in rewards]
values = [e.value for e in rewards]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(steps, values, label="Episode Reward Mean", color='blue')
plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Reward")
plt.title("RL Agent Training Curve (PPO)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ppo_training_curve.png", dpi=300)
plt.show()
