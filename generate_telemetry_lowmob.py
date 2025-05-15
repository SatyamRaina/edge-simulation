import numpy as np
import pandas as pd

time_steps = 50

data = {
    'Bandwidth': np.random.randint(20, 100, time_steps),
    'CPU_Load': np.random.uniform(1, 8, time_steps),
    'Task_Size': np.random.randint(10, 50, time_steps),
    'Mobility': np.random.uniform(0, 0.2, time_steps),  # Low mobility
    'Cache_Occupancy': np.random.uniform(0, 1, time_steps),
    'Residual_Energy': np.linspace(100, 50, time_steps),
    'Local_CPU': np.random.uniform(1.2, 2.0, time_steps),
    'Tx_Power': np.full(time_steps, 0.5)
}
pd.DataFrame(data).to_csv("synthetic_telemetry_lowmob.csv", index=False)
