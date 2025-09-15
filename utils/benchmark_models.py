# ---------------------------------------------------------*\
# Title: Benchmarking MARL Models (PPO only)
# ---------------------------------------------------------*/

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

from src.testing import test_env


# ---------------------------------------------------------*/
# Model Benchmarking
# ---------------------------------------------------------*/

def run_model_benchmark(env_creator, trained_agents, num_seeds=10, steps_test=50, use_action_masking=True, tag=""):
    """
    Runs a comprehensive benchmark for all scenarios and plots the results.
    """
    out_dir = make_benchmark_dir(base="./img/benchmarks", prefix=f"benchmark_{tag}")
    all_results = []

    header = "Seed\t    Random\tRule-Based\t Sort-Only\t   Modular\t  Monolith"
    print(f"\n⚙ Running benchmark sequentially across {num_seeds} seeds...\n")
    print(header)
    print("-" * (len(header) + 20))

    for seed in range(1, num_seeds + 1):
        result = benchmark_seed_all(seed, trained_agents, env_creator, steps_test, out_dir, use_action_masking)
        all_results.append(result)

    policy_keys = ["Random", "Rule-Based", "PPO Sort-Only", "PPO Modular", "PPO Monolith"]
    summary = {}
    for key in policy_keys:
        rewards = [res[key] for res in all_results if res.get(key) is not None]
        summary[key] = {"mean": np.mean(rewards), "std": np.std(rewards)} if rewards else {"mean": 0, "std": 0}

    # --- Tabulate Results ---
    print("\n" + "="*80)
    print("Summary of Benchmark Results:")
    df = pd.DataFrame(summary).T
    df.index.name = "Policy"
    print(df.to_string(float_format="%.2f"))
    print("="*80)

    # --- Plotting ---
    # Use a serif font for a more professional, publication-ready look (like Times New Roman)
    plt.rcParams['font.family'] = 'serif'

    # Professional labels for the policies
    labels = {
        "Random": "Random",
        "Rule-Based": "Rule-Based",
        "PPO Sort-Only": "Sort Agent",
        "PPO Modular": "Sort + Press Agents",
        "PPO Monolith": "Combined Agent"
    }
    plot_labels = [labels[key] for key in policy_keys]

    means = [summary[key]['mean'] for key in policy_keys]
    stds = [summary[key]['std'] for key in policy_keys]

    x_pos = np.arange(len(plot_labels))
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted for better aspect ratio

    # Use a grayscale colormap suitable for black and white printing
    cmap = plt.get_cmap('Greys')
    colors = cmap(np.linspace(0.35, 0.85, len(plot_labels)))

    bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.95, capsize=6, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_xticks(x_pos)

    # Do not rotate x labels; center them and ensure normal font style (not italic)
    ax.set_xticklabels(plot_labels, rotation=0, ha="center", fontsize=10, fontstyle='normal')

    title_suffix = "with Action Masking" if use_action_masking else "without Action Masking"
    ax.set_title(f'Agent Performance Comparison ({num_seeds} Seeds)\n{title_suffix}', fontsize=14, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)  # Keep grid lines behind bars

    # Add value labels centered inside each bar for clarity
    for idx, bar in enumerate(bars):
        yval = bar.get_height()
        # If mean is near zero, place label above baseline to avoid overlap
        if abs(yval) < 1e-6:
            y_pos = yval + 0.1
            va = 'bottom'
        else:
            y_pos = yval / 2.0
            va = 'center'

        # determine a readable text color depending on bar luminance
        facecolor = bar.get_facecolor()
        try:
            r, g, b, a = facecolor
        except Exception:
            r, g, b = facecolor[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        txt_color = 'white' if luminance < 0.6 else 'black'

        ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f'{yval:.1f}', ha='center', va=va, fontsize=9, color=txt_color, weight='bold')

    # Improve layout for LNCS style: tighter margins, larger font for readability in prints
    plt.tight_layout(pad=1.0)

    # Save high-quality files suitable for publication
    plot_filename = f"Model_Benchmark_{'Masked' if use_action_masking else 'NoMask'}"
    plt.savefig(f"{out_dir}/{plot_filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/{plot_filename}.svg", bbox_inches='tight')
    plt.savefig(f"{out_dir}/{plot_filename}.pdf", bbox_inches='tight')  # Save as PDF for LaTeX
    plt.show()
    plt.close()
    print(f"\nBenchmark plot saved to {out_dir}/{plot_filename}.png, .svg, and .pdf")

    return summary

# ---------------------------------------------------------*/
# Benchmarking function for a single seed for all 5 scenarios
# ---------------------------------------------------------*/


def benchmark_seed_all(seed, trained_agents, env_creator, steps_test, out_dir, use_action_masking):
    """
    Benchmarks all 5 scenarios for a single seed using a single, reset environment instance.
    Returns a dictionary with cumulative rewards for each scenario.
    """
    results = {"seed": seed}

    # Create a single environment instance for all scenarios in this seed run
    env = env_creator(env_type="Monolith", seed=seed, max_steps=steps_test, log=False)

    # --- Scenario 1: Random Agent ---
    env.reset(seed=seed)
    # Ensure no agents are assigned from previous runs
    env.sort_agent = None
    env.press_agent = None
    results["Random"], _ = test_env(env=env, mode="random", stats=False, show=False,
                                    steps=steps_test, dir=out_dir, tag=f"seed_{seed}_Random", use_action_masking=use_action_masking)

    # --- Scenario 2: Rule-Based Agent ---
    env.reset(seed=seed)
    env.sort_agent = None
    env.press_agent = None
    results["Rule-Based"], _ = test_env(env=env, mode="rule_based", stats=False,
                                        show=False, steps=steps_test, dir=out_dir, tag=f"seed_{seed}_RuleBased", use_action_masking=use_action_masking)

    # --- Scenario 3: PPO Sorting Agent (Pressing is Rule-Based) ---
    env.reset(seed=seed)
    env.sort_agent = trained_agents.get("PPO_Sort")
    env.press_agent = None  # Ensure pressing is rule-based
    results["PPO Sort-Only"], _ = test_env(env=env, mode="model", stats=False,
                                           show=False, steps=steps_test, dir=out_dir, tag=f"seed_{seed}_SortOnly", use_action_masking=use_action_masking)

    # --- Scenario 4: PPO Modular (Sort + Press agents) ---
    env.reset(seed=seed)
    env.sort_agent = trained_agents.get("PPO_Sort")
    env.press_agent = trained_agents.get("PPO_Press")
    results["PPO Modular"], _ = test_env(env=env, mode="model", stats=False, show=False,
                                         steps=steps_test, dir=out_dir, tag=f"seed_{seed}_Modular", use_action_masking=use_action_masking)

    # --- Scenario 5: PPO Monolithic Agent ---
    env.reset(seed=seed)
    # The monolithic agent is passed directly to test_env, no need to set it on the env
    env.sort_agent = None
    env.press_agent = None
    results["PPO Monolith"], _ = test_env(env=env, mode="model", model=trained_agents.get(
        "PPO_Mono"), stats=False, show=False, steps=steps_test, dir=out_dir, tag=f"seed_{seed}_Monolith", use_action_masking=use_action_masking)

    env.close()

    # --- Print compact results for the seed ---
    order = ["Random", "Rule-Based", "PPO Sort-Only", "PPO Modular", "PPO Monolith"]
    line = f"  {seed: >4}"
    for key in order:
        val = results.get(key)
        line += f"\t{val: >10.2f}" if val is not None else "\t       N/A"
    print(line)

    return results

# ---------------------------------------------------------*/
# Function to load a PPO model from a given path
# ---------------------------------------------------------*/


def load_ppo_model(model_path):
    """Loads a PPO model from a file."""
    return PPO.load(model_path)


# ---------------------------------------------------------*/
# Function to create a new benchmark directory
# ---------------------------------------------------------*/
def make_benchmark_dir(base="./img/benchmarks", prefix="benchmark_results"):
    """Creates a new directory for benchmark results, avoiding overwrites.

    The function detects existing directories named in the form
    "<number>_<prefix>" (e.g. "1_benchmark_results") and picks the next
    available leading number. If a collision still occurs, it increments
    until a free name is found.
    """
    import re

    os.makedirs(base, exist_ok=True)

    existing_dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]

    existing_nums = []
    for d in existing_dirs:
        # Match patterns like '123_benchmark_results'
        m = re.match(r'^([0-9]+)_' + re.escape(prefix) + r'$', d)
        if m:
            existing_nums.append(int(m.group(1)))
        elif d == prefix:
            existing_nums.append(0)

    new_dir_num = max(existing_nums, default=0) + 1
    new_dir = os.path.join(base, f"{new_dir_num}_{prefix}")

    # Ensure uniqueness in case of race conditions or unexpected names
    while os.path.exists(new_dir):
        new_dir_num += 1
        new_dir = os.path.join(base, f"{new_dir_num}_{prefix}")

    os.makedirs(new_dir, exist_ok=False)
    return new_dir


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*/
