import numpy as np
import pandas as pd

time_steps = 50

data = {
    'Bandwidth': np.random.randint(10, 30, time_steps),
    'CPU_Load': np.random.uniform(6, 9.5, time_steps),  # High edge load
    'Task_Size': np.random.randint(40, 50, time_steps),  # Large tasks
    'Mobility': np.random.uniform(0, 1, time_steps),
    'Cache_Occupancy': np.random.uniform(0.5, 1.0, time_steps),
    'Residual_Energy': np.linspace(100, 40, time_steps),
    'Local_CPU': np.random.uniform(1.0, 1.4, time_steps),
    'Tx_Power': np.full(time_steps, 0.5)
}
pd.DataFrame(data).to_csv("synthetic_telemetry_highload.csv", index=False)
