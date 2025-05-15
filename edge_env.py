import gymnasium as gym
import numpy as np
import pandas as pd
import csv
import os

class EdgeEnv(gym.Env):
    def __init__(self, csv_file='synthetic_telemetry.csv', log_file='agent_action_log.csv'):
        super(EdgeEnv, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.time_step = 0
        self.max_steps = len(self.data)
        self.log_file = log_file

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.MultiBinary(2)

        # Constants from thesis
        self.kappa_u = 1e-28
        self.kappa_e = 1e-28
        self.alpha = 0.5
        self.beta = 0.5
        self.latency_deadline = 10000  # seconds
        self.energy_limit = 100      # max energy per task (J)

        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Offload', 'Cache', 'Latency', 'Energy', 'MIP_Feasible'])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        return self._get_state(), {}


    def _get_state(self):
        row = self.data.iloc[self.time_step]
        state = np.array([
            row['Bandwidth'] / 100,
            row['CPU_Load'] / 10,
            row['Task_Size'] / 50,
            row['Mobility'],
            row['Cache_Occupancy'],
            row['Residual_Energy'] / 100
        ], dtype=np.float32)
        return state

    def step(self, action):
        offload, cache = action
        row = self.data.iloc[self.time_step]

        # Parameters
        d_u = row['Task_Size']
        B_ue = row['Bandwidth']
        f_e = row['CPU_Load']
        f_u_local = row['Local_CPU']
        P_tx = row['Tx_Power']
        E_res = row['Residual_Energy']
        c_u = d_u * 1000
        cache_hit = cache

        # Latency
        t_comm = (d_u / B_ue) + ((1 - cache_hit) * d_u / 100) if offload else 0
        t_comp = (c_u / f_e) if offload else (c_u / f_u_local)
        latency = t_comm + t_comp

        # Energy
        E_tx = P_tx * (d_u / B_ue) if offload else 0
        E_comp = self.kappa_e * c_u * f_e**2 if offload else self.kappa_u * c_u * f_u_local**2
        energy = E_tx + E_comp

        # Constraint checks
        feasible = int((latency <= self.latency_deadline) and (energy <= E_res))

        # Reward
        reward = - (self.alpha * latency + self.beta * energy)

        # Log action
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.time_step, offload, cache, latency, energy, feasible])

        # Step
        self.time_step += 1
        done = self.time_step >= self.max_steps
        next_state = self._get_state() if not done else np.zeros(6, dtype=np.float32)
        return next_state, reward, done, False, {}  # ✅ gymnasium format

    def render(self, mode='human'):
        print(f"Step: {self.time_step}")
