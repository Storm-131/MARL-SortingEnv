import matplotlib.pyplot as plt
import numpy as np

# --- Data (unchanged) ---
data_without_masking = {
    "Random": {"mean": -109.36, "std": 6.29},
    "Rule-Based": {"mean": 43.20, "std": 1.07},
    "PPO Sort-Only": {"mean": -83.52, "std": 10.14},
    "PPO Modular": {"mean": -64.98, "std": 7.92},
    "PPO Monolith": {"mean": -100.31, "std": 1.02}
}
data_with_masking = {
    "Random": {"mean": -84.28, "std": 22.29},
    "Rule-Based": {"mean": 44.03, "std": 1.10},
    "PPO Sort-Only": {"mean": -70.22, "std": 10.56},
    "PPO Modular": {"mean": 30.61, "std": 0.87},
    "PPO Monolith": {"mean": 32.77, "std": 1.12}
}
policy_keys = ["Random", "Rule-Based", "PPO Sort-Only", "PPO Modular", "PPO Monolith"]
plot_labels = ["Random", "Rule-Based", "Sort Agent", "Modular Agents", "Monolithic Agent"]


# --- Data preparation (increased spacing) ---
x_base = np.arange(len(policy_keys)) * 2.5
# OFFSET: increased spacing between the points within a pair
offset = 0.6 
means_without = [data_without_masking[key]['mean'] for key in policy_keys]
stds_without = [data_without_masking[key]['std'] for key in policy_keys]
x_without = x_base - offset / 2
means_with = [data_with_masking[key]['mean'] for key in policy_keys]
stds_with = [data_with_masking[key]['std'] for key in policy_keys]
x_with = x_base + offset / 2

# --- Create plot (publication-optimized) ---
fig, ax = plt.subplots(figsize=(12, 5))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# 1. Plot dumbbell connecting lines
for i in range(len(policy_keys)):
    ax.plot([x_without[i], x_with[i]], [means_without[i], means_with[i]],
            color='gray', linewidth=1, linestyle='-', zorder=1)

# 2. Error bars and markers (increased size)
# NOTE: increased markersize
ax.errorbar(x_without, means_without, yerr=stds_without,
            fmt='o', markerfacecolor='white', markeredgecolor='black',
            markersize=11, capsize=5, ecolor='black', elinewidth=1.5,
            label='Without Masking', zorder=2)

# NOTE: increased markersize
ax.errorbar(x_with, means_with, yerr=stds_with,
            fmt='o', markerfacecolor='black', markeredgecolor='black',
            markersize=11, capsize=5, ecolor='black', elinewidth=1.5,
            label='With Masking', zorder=2)

# --- Finalize axes and labels (larger sizes) ---
ax.set_ylabel('Cumulative Reward', fontsize=14)
# ax.set_title('Effect of Action Masking on Agent Performance', fontsize=16, fontweight='bold')

ax.set_xticks(x_base)
ax.set_xticklabels(plot_labels, fontsize=13)

ax.tick_params(axis='y', labelsize=13)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.axhline(0, color='black', linewidth=0.8)

# NOTE: increased legend font size
ax.legend(fontsize=13, frameon=True, edgecolor='black', loc='upper left')

plt.savefig("./img/benchmark_plot_summary_refactored.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()