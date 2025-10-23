import numpy as np
import matplotlib.pyplot as plt

# Global text scale for all plot labels and texts
TEXT_SCALE = 1.3

def scaled(x: float) -> int:
    """Scale an integer font size by the global TEXT_SCALE and return an int >= 1."""
    return max(1, int(round(x * TEXT_SCALE)))

# --- 1. Reward Calculation Functions (matching env_super.py) ---


def calculate_sorting_reward(avg_purity_deviation, scaling_factor, temp):
    """
    Exact implementation of the sorting reward from env_super.py.
    """
    raw_score = avg_purity_deviation * scaling_factor
    return np.tanh(raw_score / temp)


def calculate_press_reward(amount, overall_fill_ratio, S, bale_efficiency_factor, max_state_reward, target_peaks_raw):
    """
    Exact implementation of the pressing reward from env_super.py.
    """
    # Action-based component
    num_bales = amount // S
    rem = amount % S
    dist_from_multiple = min(rem, S - rem)
    efficiency_component = (1.0 - 4.0 * (dist_from_multiple / S)) * bale_efficiency_factor
    bonus_index = min(int(num_bales), len(target_peaks_raw) - 1)
    target_peak = target_peaks_raw[bonus_index]
    multi_bale_bonus = target_peak - bale_efficiency_factor
    action_reward = efficiency_component + multi_bale_bonus

    # State-based component
    state_reward = overall_fill_ratio * max_state_reward

    return np.clip(state_reward + action_reward, -1.0, 1.0)


# --- 2. Plotting Functions for Each Subplot ---

def plot_sorting_reward(ax):
    """Creates the sorting reward plot on the given axes object."""
    # Parameters
    purity_scaling_factor = 2.0
    temperature = 0.5

    # Data Generation
    avg_purity_deviations = np.linspace(-0.6, 0.6, 400)
    final_rewards = calculate_sorting_reward(avg_purity_deviations, purity_scaling_factor, temperature)

    # Main plot
    ax.plot(avg_purity_deviations, final_rewards, label=f'Sorting Reward (T={temperature}, s={purity_scaling_factor})',
            color='darkcyan', linewidth=3, zorder=2)  # Made line thicker
    ax.fill_between(avg_purity_deviations, 0, final_rewards, where=final_rewards >= 0,
                    color='darkcyan', alpha=0.15, label='Positive Reward')
    ax.fill_between(avg_purity_deviations, 0, final_rewards, where=final_rewards < 0,
                    color='firebrick', alpha=0.15, label='Negative Reward')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Purity = Threshold')

    # Annotations
    ax.annotate('High Reward\n(Saturation)', xy=(0.4, 0.93), xytext=(0.4, 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=10),
                ha='center', fontsize=scaled(12), fontweight='bold', color='black')
    ax.annotate('High Penalty\n(Saturation)', xy=(-0.4, -0.93), xytext=(-0.4, -0.6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=10),
                ha='center', fontsize=scaled(12), fontweight='bold', color='black')
    ax.annotate('Sensitive Region\n(Linear Response)', xy=(0, 0), xytext=(-0.3, 0.45),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=10),
                ha='center', fontsize=scaled(12), fontweight='bold')

    # Example calculations box
    example_deviations = [-0.3, -0.1, 0.1, 0.3]
    example_rewards = [calculate_sorting_reward(dev, purity_scaling_factor, temperature) for dev in example_deviations]

    # Formatting
    # Increase distance between title and plot
    # ax.set_title('(a) Sorting Reward: State-Based Tanh Function', fontweight='bold', fontsize=scaled(18), pad=scaled(24))
    ax.set_xlabel('Average Purity Deviation from Threshold', fontsize=scaled(18))
    ax.set_ylabel('Final Sorting Reward', fontsize=scaled(18))
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='lower right', fontsize=scaled(12))
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax.tick_params(axis='both', which='major', labelsize=scaled(14))


def plot_press_reward(ax):
    """Creates the pressing reward plot on the given axes object."""
    # Parameters
    S = 100
    params = {
        'S': S,
        'bale_efficiency_factor': 0.5,
        'max_state_reward': 0.4,
        'target_peaks_raw': np.array([0.0, 1/3, 2/3, 1.0])
    }

    # Data Generation
    num_bales_range = np.linspace(0, 3, 500)
    amount_range = num_bales_range * S
    fill_ratios_to_plot = [0.0, 0.5, 1.0]
    colors = ['#440154', '#21918c', '#fde725']
    line_styles = ['-', '--', '-.']

    # Vectorize the calculation function
    calculate_press_reward_vec = np.vectorize(calculate_press_reward, excluded=[
                                              'S', 'bale_efficiency_factor', 'max_state_reward', 'target_peaks_raw'])

    # Main plot
    for ratio, color, style in zip(fill_ratios_to_plot, colors, line_styles):
        rewards = calculate_press_reward_vec(amount_range, ratio, **params)
        ax.plot(num_bales_range, rewards, label=f'Fill Ratio: {int(ratio*100)}%',
                color=color, linewidth=2.5, linestyle=style)

    # Annotations
    ax.annotate('Optimal Pressing\n(Integer Bales)', xy=(2.0, 1.0), xytext=(2.5, 0.75),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=10),
                ha='center', fontsize=scaled(14), fontweight='bold')
    inefficient_reward = calculate_press_reward(1.5 * S, 1.0, **params)
    ax.annotate('Inefficient Pressing\n(1.5 bales)', xy=(1.5, inefficient_reward), xytext=(1.1, -0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=10),
                ha='center', fontsize=scaled(14), fontweight='bold')

    # Formatting
    # ax.set_title('(b) Pressing Reward: Two-Component Function', fontweight='bold', fontsize=scaled(18), pad=scaled(24))
    ax.set_xlabel('Number of Bales Pressed', fontsize=scaled(18))
    ax.set_ylabel('Final Press Reward', fontsize=scaled(18))
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.legend(loc='upper left', fontsize=scaled(12))
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax.tick_params(axis='both', which='major', labelsize=scaled(14))


# --- 3. Main Script Execution ---

def main():
    """Main function to setup, create, and save the plot."""
    # Setup global plot settings
    try:
        plt.rcParams['font.family'] = 'Avenir'
    except:
        plt.rcParams['font.family'] = 'sans-serif'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))  # Increased figure size
    plt.rcParams.update({'font.size': scaled(14), 'axes.titlesize': scaled(18), 'axes.labelsize': scaled(16)})  # Increased font sizes

    # Create each subplot
    plot_sorting_reward(ax1)
    plot_press_reward(ax2)

    # Finalize and save the entire figure
    # fig.suptitle('Analysis of Agent Reward Functions', fontsize=scaled(24), fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("../img/reward_functions_analysis_refactored.png", dpi=300, bbox_inches='tight')
    plt.savefig("../img/reward_functions_analysis_refactored.svg")
    plt.show()


if __name__ == "__main__":
    main()
