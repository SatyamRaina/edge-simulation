import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd

def evaluate(model, env, file_name="synthetic_telemetry.csv"):
    obs = env.reset()
    if isinstance(obs, tuple): 
        obs = obs[0]

    latency_list, energy_list = [], []
    cache_hits = mip_success = total_tasks = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated


        row = env.data.iloc[env.time_step - 1]

        offload, cache = action
        d_u = row['Task_Size']
        B_ue = row['Bandwidth']
        f_e = 10 - row['CPU_Load']
        f_local = row['Local_CPU']
        c_u = d_u
        P_tx = 0.5
        k_u = 1e-28

        # Latency
        t_comm = (d_u / B_ue) if offload else 0
        t_comp = (c_u / f_e) if offload else (c_u / f_local)
        latency = t_comm + t_comp
        latency_list.append(latency)

        # Energy
        E_tx = (d_u / B_ue) * P_tx if offload else 0
        E_comp = k_u * c_u * (f_e ** 2) if offload else k_u * c_u * (f_local ** 2)
        total_energy = E_tx + E_comp
        energy_list.append(total_energy)

        if cache == 1:
            cache_hits += 1
        if reward > -1000:
            mip_success += 1

        total_tasks += 1
        obs = next_obs
        if done:
            break

    # Calculate final metrics
    throughput = total_tasks / env.max_steps

    cache_hit_ratio = cache_hits / total_tasks
    mip_success_rate = mip_success / total_tasks

    # Prepare new result
    new_result = {
        "File": file_name,
        "Avg Latency (s)": round(np.mean(latency_list), 3),
        "Avg Energy (J)": round(np.mean(energy_list), 6),
        "Cache Hit": round(cache_hit_ratio, 3),
        "MIP Success": round(mip_success_rate, 3),
        "Throughput": round(throughput, 3)
    }

    # Log summary to CSV
    summary_path = "summary_results.csv"
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df = df[df["File"] != file_name]  # Remove existing row for the same file
        df = pd.concat([df, pd.DataFrame([new_result])], ignore_index=True)
    else:
        df = pd.DataFrame([new_result])

    df.to_csv(summary_path, index=False)

    # Optional: save evaluation figure
    plt.figure(figsize=(10, 4))
    plt.plot(latency_list, label="Latency (s)")
    plt.plot(energy_list, label="Energy (J)")
    plt.legend()
    plt.title(f"RL-MIP Evaluation: {file_name}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(f"evaluation_metrics_{file_name.split('.')[0]}.png")
    plt.close()
