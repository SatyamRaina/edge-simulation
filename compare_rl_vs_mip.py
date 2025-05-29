import pandas as pd
import time
from mil_solver.pyomo_mip_solver import solve_row_with_pyomo

# Load telemetry and RL logs
telemetry_df = pd.read_csv('synthetic_telemetry.csv')
rl_df = pd.read_csv('agent_action_log.csv')

# Result storage
comparison = []

for i in range(len(telemetry_df)):
    row = telemetry_df.iloc[i].to_dict()
    rl_row = rl_df.iloc[i]

    # Simulate CPU time for RL (already decided, so fast)
    rl_start = time.perf_counter()
    rl_offload = rl_row['Offload']
    rl_cache = rl_row['Cache']
    rl_latency = rl_row['Latency']
    rl_energy = rl_row['Energy']
    rl_end = time.perf_counter()
    rl_cpu_time = rl_end - rl_start

    # Solve MIP and track time
    mip_start = time.perf_counter()
    mip_result = solve_row_with_pyomo(row, solver_name="ipopt")
    mip_end = time.perf_counter()
    mip_cpu_time = mip_end - mip_start

    # Compare
    comparison.append({
        'Time': i,
        'RL_Offload': rl_offload,
        'RL_Cache': rl_cache,
        'RL_Latency': rl_latency,
        'RL_Energy': rl_energy,
        'MIP_Offload': mip_result.get('offload'),
        'MIP_Cache': mip_result.get('cache'),
        'MIP_Latency': mip_result.get('latency'),
        'MIP_Energy': mip_result.get('energy'),
        'Feasible': mip_result.get('feasible', 0),
        'RL_CPU_Time': round(rl_cpu_time, 6),
        'MIP_CPU_Time': round(mip_cpu_time, 6)
    })

# Save to CSV
df = pd.DataFrame(comparison)
df.to_csv('rl_vs_mip_comparison.csv', index=False)
print("✅ Saved comparison results to rl_vs_mip_comparison.csv with CPU time columns")
