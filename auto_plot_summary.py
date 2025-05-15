# auto_plot_summary.py
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("summary_results.csv")
print("📄 Column Names:", df.columns.tolist())


# Load summary CSV
summary_file = "summary_results.csv"
if not os.path.exists(summary_file):
    print("❌ summary_results.csv not found. Run evaluations first.")
    exit()

# Read data
summary_df = pd.read_csv(summary_file)

# Plot bar chart for each metric
metrics = ["Avg Latency (s)", "Avg Energy (J)", "Cache Hit", "MIP Success", "Throughput"]
colors = ["skyblue", "lightgreen", "orange", "blue", "purple"]

plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), i + 1)
    plt.bar(summary_df["File"], summary_df[metric], color=colors[i])
    plt.xticks(rotation=45, ha='right')
    plt.title(metric)
    plt.ylim(0, summary_df[metric].max() * 1.1)  # Dynamic Y-limits

    # Annotate bar values
    for idx, val in enumerate(summary_df[metric]):
        plt.text(idx, val + 0.02, f"{val:.2f}", ha='center', fontsize=8)

plt.tight_layout()
plt.suptitle("RL-MIP Framework Evaluation Summary", fontsize=14, y=1.05)
plt.savefig("summary_metrics_plot.png", dpi=300)
plt.show()

print("✅ Summary plot saved as summary_metrics_plot.png")
