import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv("rl_vs_mip_comparison.csv")

# Plot 1: RL vs MIP Latency
plt.figure(figsize=(10, 5))
plt.plot(df["Time"], df["RL_Latency"], label="RL Latency", marker='o')
plt.plot(df["Time"], df["MIP_Latency"], label="MIP Latency", marker='x')
plt.xlabel("Time Step")
plt.ylabel("Latency (s)")
plt.title("RL vs MIP Latency Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("latency_comparison.png")

# Plot 2: RL vs MIP CPU Time
plt.figure(figsize=(10, 5))
plt.plot(df["Time"], df["RL_CPU_Time"], label="RL CPU Time", marker='o')
plt.plot(df["Time"], df["MIP_CPU_Time"], label="MIP CPU Time", marker='x')
plt.xlabel("Time Step")
plt.ylabel("CPU Time (s)")
plt.title("RL vs MIP CPU Time Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cpu_time_comparison.png")

# Plot 3: Feasibility (Bar Plot)
plt.figure(figsize=(10, 4))
plt.bar(df["Time"], df["Feasible"], color="skyblue")
plt.xlabel("Time Step")
plt.ylabel("Feasibility")
plt.title("Feasibility of MIP Solutions (1 = Feasible, 0 = Infeasible)")
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.tight_layout()
plt.savefig("feasibility_plot.png")
