# generate_telemetry.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set time range and seed for reproducibility
time_steps = 50
np.random.seed(42)

# Synthetic telemetry generation
bandwidth = np.random.randint(10, 100, time_steps)             # B_ue(t) in Mbps
cpu_load = np.random.uniform(1, 8, time_steps)                 # F_e(t) in GHz (used capacity)
task_size = np.random.randint(10, 50, time_steps)             # d_u(t) in MB
user_mobility = np.random.uniform(0, 1, time_steps)           # stochastic mobility index
cache_occupancy = np.random.uniform(0, 1, time_steps)         # fraction cache filled (0–1)
residual_energy = np.linspace(100, 50, time_steps)            # E_u^res(t) from 100J → 50J
local_cpu = np.random.uniform(1.2, 2.0, time_steps)           # f_u^local(t) in GHz
tx_power = np.full(time_steps, 0.5)  # P_u^tx(t) = 0.5 W constant

# Construct DataFrame
df = pd.DataFrame({
    'Time': range(time_steps),
    'Bandwidth': bandwidth,
    'CPU_Load': cpu_load,
    'Task_Size': task_size,
    'Mobility': user_mobility,
    'Cache_Occupancy': cache_occupancy,
    'Residual_Energy': residual_energy,
    'Local_CPU': local_cpu,
    'Tx_Power': tx_power 
})

# Save to CSV (optional)
df.to_csv('synthetic_telemetry.csv', index=False)

# Plot key telemetry trends
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(df['Time'], df['Bandwidth'], label='Bandwidth (B_ue)', color='tab:blue')
axs[0, 0].set_title('Bandwidth (Mbps)')

axs[0, 1].plot(df['Time'], df['CPU_Load'], label='CPU Load (F_e)', color='tab:orange')
axs[0, 1].set_title('Edge CPU Load (GHz)')

axs[1, 0].plot(df['Time'], df['Residual_Energy'], label='Residual Energy (E_u^res)', color='tab:green')
axs[1, 0].set_title('Residual Energy (J)')

axs[1, 1].plot(df['Time'], df['Local_CPU'], label='Local CPU (f_u^local)', color='tab:red')
axs[1, 1].set_title('Local CPU Frequency (GHz)')

for ax in axs.flat:
    ax.set(xlabel='Time Step', ylabel='Value')
    ax.grid(True)

plt.suptitle('Synthetic Telemetry Trends')
plt.tight_layout()
plt.show()
