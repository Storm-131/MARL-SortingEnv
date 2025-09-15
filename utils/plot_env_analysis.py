# ---------------------------------------------------------*\
# Title: Plot Environment Statistics (Adjusted for New Env)
# ---------------------------------------------------------*/

import matplotlib.pyplot as plt
from src.envs_train.env_1_sort import Env_1_Sorting
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from matplotlib.colors import to_rgba

# ---------------------------------------------------------*/
# Accuracy Analysis
# ---------------------------------------------------------*/

def plot_material_accuracies(env, distribution=None, sorting_mode=0):
    """
    Plot the accuracies of different materials against occupation levels
    for a specific sorting mode, and display the input composition as a pie chart.
    """
    # Initialize data storage
    occupation_levels = list(range(0, 101, 1))  # From 0% to 100% in steps of 1%
    material_accuracies = {material: [] for material in env.material_names}

    # Default material distribution
    default_distribution = {
        'A': 0.15,
        'B': 0.25,
        'C': 0.30,
        'D': 0.15,
        'E': 0.15
    }
    
    if distribution is None:
        distribution = default_distribution
    elif not np.isclose(sum(distribution.values()), 1.0):
        raise ValueError("The sum of the distribution must be 1 (100%).")

    env.noise_accuracy = 0.05   
            
    # Loop over occupation levels
    for occ_level in occupation_levels:
        env.reset()
        env.set_multisensor_mode(sorting_mode)  # Set the sorting mode
        env.input_action_rules(occ_level)
        env.update_accuracy()
        
        for i, material in enumerate(env.material_names):
            material_accuracies[material].append(env.accuracy_belt[i] * 100)

    # Create figure
    plt.figure(figsize=(12, 8), constrained_layout=True)
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    lines, labels = [], []

    for i, material in enumerate(env.material_names):
        label = f"{material} (Mode {sorting_mode})"
        line, = plt.plot(
            occupation_levels,
            material_accuracies[material],
            label=label,
            color=colors[i],
            linestyle='-',
            linewidth=1.5
        )
        lines.append(line)
        labels.append(label)

    plt.title(f'Accuracies per Material vs. Occupation Level (Sorting Mode {sorting_mode}, Noise {env.noise_accuracy*100}%', fontsize=14)
    plt.xlabel('Occupation Level (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 100)
    plt.ylim(0, 105)
    plt.grid(True)

    plt.legend(lines, labels, title="Materials", loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # Insert pie chart
    ax_inset = inset_axes(plt.gca(), width="70%", height="70%", loc='lower left',
                          bbox_to_anchor=(0.0, 0.0, 0.5, 0.5), bbox_transform=plt.gca().transAxes)
    pie_labels = list(distribution.keys())
    sizes = [value * 100 for value in distribution.values()]
    pie_colors = [to_rgba(color, alpha=0.6) for color in colors[:len(pie_labels)]]
    ax_inset.pie(sizes, labels=[f"{label}" for label in pie_labels],
                 colors=pie_colors, autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12})

    plt.show(block=True)
    plt.close('all')

# ---------------------------------------------------------*/
# Reward Analysis
# ---------------------------------------------------------*/

def plot_sorting_rewards_vs_purity_deviation(env, num_samples=10):
    """
    Plot sorting rewards vs. purity deviations.
    
    X-axis: Sample index (1-10)
    Left Y-axis: Purity deviations for each container (5 colored lines, no markers, 50% opaque)
    Right Y-axis: Total reward (black) and cumulative reward (grey, bold solid line)
    
    - For each sample, random purity deviations in the range [-0.25, 0.25] are generated.
    - The probability of a positive deviation is 98%.
    - The sorting reward is computed using a penalty factor of 5 for negative deviations.
    """
    # Define a scale factor for fonts and tick labels
    scale = 1.2
    fontsize_xlabel = int(16 * scale)
    fontsize_ylabel = int(16 * scale)
    fontsize_title  = int(18 * scale)
    fontsize_tick   = int(14 * scale)
    fontsize_legend = int(14 * scale)
    
    samples = np.arange(1, num_samples + 1)
    
    # Retrieve container names from the environment (assumed to be 5)
    container_names = env.material_names
    
    # Initialize storage for deviations per container and total rewards per sample
    container_deviations = {material: [] for material in container_names}
    total_rewards = []
    cumulative_rewards = []
    
    # Use environment's RNG if available, otherwise create one
    rng = env.rng if hasattr(env, 'rng') else np.random.default_rng()
    
    cumulative_reward = 0.0  # Running total of rewards
    for _ in samples:
        sample_purity_deviation = {}
        for material in container_names:
            # Generate a random deviation: 98% chance for positive, else negative
            if rng.uniform(0, 1) < 0.98:
                deviation = rng.uniform(0, 0.25)
            else:
                deviation = rng.uniform(-0.25, 0)
            container_deviations[material].append(deviation)
            sample_purity_deviation[material] = deviation
        
        # For this sample, adjust deviations (multiply negative deviations by 5)
        adjusted_deviations = [
            deviation * 5 if deviation < 0 else deviation 
            for deviation in sample_purity_deviation.values()
        ]
        total_reward = sum(adjusted_deviations)
        total_rewards.append(total_reward)
        
        cumulative_reward += total_reward
        cumulative_rewards.append(cumulative_reward)
    
    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(20, 10))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    # Plot container purity deviations on left Y-axis: no markers and half opaque
    for i, material in enumerate(container_names):
        ax1.plot(
            samples, 
            container_deviations[material], 
            linestyle='-', 
            color=colors[i], 
            alpha=0.5,
            label=f'{material} Deviation'
        )
    
    ax1.axhline(0, color='gray', linestyle='--', linewidth=2)
    ax1.set_xlabel('Sample', fontsize=fontsize_xlabel)
    ax1.set_ylabel('Purity Deviation', fontsize=fontsize_ylabel)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize_tick)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    
    ax2 = ax1.twinx()
    ax2.plot(
        samples, 
        total_rewards, 
        marker='', 
        linestyle='-', 
        color='black', 
        linewidth=6, 
        label='Current Total Reward'
    )
    ax2.plot(
        samples, 
        cumulative_rewards, 
        marker='', 
        linestyle='-', 
        color='grey', 
        linewidth=6, 
        label='Cumulative Reward'
    )
    ax2.set_ylabel('Reward', fontsize=fontsize_ylabel)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize_tick)
    ax2.grid(False)
    
    # Force both y-axes to have symmetric limits so that 0 is at the middle.
    ylim1 = ax1.get_ylim()
    max_abs1 = max(abs(ylim1[0]), abs(ylim1[1]))
    ax1.set_ylim(-max_abs1, max_abs1)
    
    ylim2 = ax2.get_ylim()
    max_abs2 = max(abs(ylim2[0]), abs(ylim2[1]))
    ax2.set_ylim(-max_abs2, max_abs2)
    
    # Combine legends and place them outside on the right side.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', 
               bbox_to_anchor=(1.1, 0.5), fontsize=fontsize_legend)
    
    ax1.set_title('Sorting Reward vs. Purity Deviation (Samples 1-10)', fontsize=fontsize_title)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# ---------------------------------------------------------*/
# Run Analysis on Environment
# ---------------------------------------------------------*/

def run_env_analysis(env):

    # Plot material accuracies
    plot_material_accuracies(env)

    # Plot reward analysis
    plot_sorting_rewards_vs_purity_deviation(env, num_samples=10)

    # Plot state analysis
    # plot_state_analysis(env)

    # Plot action analysis
    # plot_action_analysis(env)

    # Plot transition analysis
    # plot_transition_analysis(env)

    # Plot observation analysis
    # plot_observation


# ---------------------------------------------------------*/
# Main
# ---------------------------------------------------------*/
if __name__ == "__main__":
    # Import the Train_Sort_Env class from your environment code
    # If it's in the same file, you can instantiate it directly
    env = Env_1_Sorting()
    plot_material_accuracies(env)


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*/
