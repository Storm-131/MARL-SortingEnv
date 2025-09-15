# ---------------------------------------------------------*\
# Title: Plotting
# ---------------------------------------------------------*/

import cv2
from matplotlib import ticker
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import re
import math


# ---------------------------------------------------------*/
# Global Plotting Settings
# ---------------------------------------------------------*/
font_increase = 0  # Adjust this value to increase or decrease the font size globally
X_LIMIT = 200   # Max length on the x-axis for reward and container plots


# ---------------------------------------------------------*/
# Plot the current state of the sorting environment
# ---------------------------------------------------------*/

def plot_env(env, save=True, show=True, log_dir="./img/log/", filename="marl_system_diagram", sorting_mode=False, title="", format="svg", checksum=True, steps_test=None):
    """Plot the environment based on the attributes of `env`."""

    # --- Unpack all required fields from `env` -------------------------------
    material_composition = env.current_material_input
    current_material_belt = env.current_material_belt
    current_material_sorting = env.current_material_sorting
    container_materials = env.container_materials
    accuracy = env.accuracy_belt
    prev_accuracy = env.accuracy_sorter
    sensor_setting = env.sensor_current_setting
    reward_data = env.reward_data
    belt_occupancy = env.belt_occupancy
    press_state = env.press_state
    bale_count = env.bale_count
    bale_standard_size = env.bale_standard_size
    quality_thresholds = env.quality_thresholds
    press_actions_per_timestep = env.press_actions_per_timestep
    container_global_max = env.container_global_max
    press_times = env.press_times
    seed = env.seed

    # -----------------------------------------------------------------------

    sns.set_theme(style="whitegrid")

    # Define colors for the bar segments (Container E represents the rest)
    colors = {
        'A': 'lightblue',
        'A_False': 'lightgray',
        'B': 'lightgreen',
        'B_False': 'lightgray',
        'C': 'lightcoral',
        'C_False': 'lightgray',
        'D': 'lightpink',
        'D_False': 'lightgray',
        'E': 'lightsalmon',       # Represented as "Rest" in the container plot
        'E_False': 'lightgray',
        'Other': 'grey'
    }

    # Initialize n_points based on the length of reward_data['Reward']
    n_points = len(reward_data['Reward']) if 'Reward' in reward_data else 0

    # Ensure X_LIMIT is applied consistently
    x_limit = min(n_points, X_LIMIT)

    # Generate the figure and axes
    fig = plt.figure(figsize=(20, 20))
    # fig.suptitle(f"MARL System Simulation - {title}", fontsize=20, fontweight='bold')

    # Row 1: four small plots
    ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((5, 4), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((5, 4), (0, 2), colspan=1) # Old ax4 position
    ax4 = plt.subplot2grid((5, 4), (0, 3), colspan=1) # Old ax3 position

    # Row 2: full-width sorting reward metrics
    ax5 = plt.subplot2grid((5, 4), (1, 0), colspan=4)

    # Row 3: full-width current + cumulative rewards
    ax6 = plt.subplot2grid((5, 4), (2, 0), colspan=4)

    # Row 4: full-width container fill levels + press actions
    ax7 = plt.subplot2grid((5, 4), (3, 0), colspan=4)

    # Row 5: four equal subplots (bales, press, etc.)
    ax8 = plt.subplot2grid((5, 4), (4, 0), colspan=1)
    ax9 = plt.subplot2grid((5, 4), (4, 1), colspan=1)
    ax10 = plt.subplot2grid((5, 4), (4, 2), colspan=1)
    ax11 = plt.subplot2grid((5, 4), (4, 3), colspan=1)

    # ---------------------------------------------------------*/
    # 1) Input Material Composition (4 materials: A, B, C, D)
    # ---------------------------------------------------------*/
    ax1.set_title('Input', fontweight='bold', fontsize=12)
    total = sum(material_composition)
    ax1.pie(material_composition, labels=['A', 'B', 'C', 'D'],
            autopct=lambda p: f'{round(p * total / 100)}',
            colors=[colors['A'], colors['B'], colors['C'], colors['D']],
            textprops={'fontsize': 10})
    ax1.text(0.02, 0.98, f'Total: {total}', transform=ax1.transAxes, ha='left',
             va='top', fontweight='bold', fontsize=10)

    # ---------------------------------------------------------*/
    # 2) Conveyor Belt Material (4 materials)
    # ---------------------------------------------------------*/
    ax2.set_title('Conveyor Belt', fontweight='bold', fontsize=12)
    bars = ax2.bar(['A', 'B', 'C', 'D'], current_material_belt,
                   color=[colors['A'], colors['B'], colors['C'], colors['D']])
    ax2.text(0.02, 0.98, f'Total: {sum(current_material_belt)}',
             transform=ax2.transAxes, ha='left', va='top', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Quantity', fontsize=10)
    ax2.set_ylim([0, 100])
    ax2.tick_params(axis='x', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)

    for bar, value in zip(bars, current_material_belt):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, str(value),
                 ha='center', va='bottom', color='black', fontsize=10)

    # ---------------------------------------------------------*/
    # 3) Sorting Machine Material (Plot 3, old Plot 4)
    # ---------------------------------------------------------*/
    ax3.set_title('Sorting Machine', fontweight='bold', fontsize=12)
    bars = ax3.bar(['A', 'B', 'C', 'D'], current_material_sorting,
                   color=[colors['A'], colors['B'], colors['C'], colors['D']])
    ax3.text(0.02, 0.98, f'Total: {sum(current_material_sorting)}',
             transform=ax3.transAxes, ha='left', va='top', fontweight='bold', fontsize=10)
    ax3.set_ylim([0, 100])
    ax3.tick_params(axis='x', labelsize=10)
    ax3.tick_params(axis='y', labelsize=10)

    for bar, value in zip(bars, current_material_sorting):
        # Place the label at the bar height (same style as Conveyor Belt plot)
        ax3.text(bar.get_x() + bar.get_width() / 2, value, str(value),
                 ha='center', va='bottom', color='black', fontsize=10)

    # ---------------------------------------------------------*/
    # 4) Current Sorting Accuracies (Plot 4, old Plot 3)
    # ---------------------------------------------------------*/
    ax4.set_title('Current Sorting Accuracies', fontweight='bold', fontsize=12)
    
    # Data is just the accuracy for the 4 materials
    accuracy_data = np.array(accuracy)
    
    bars = ax4.bar(['A', 'B', 'C', 'D'],
                   accuracy_data, color=[colors['A'], colors['B'], colors['C'], colors['D']])

    ax4.set_ylim([0, 1.1])
    ax4.set_yticks(np.arange(0, 1.1, 0.2))
    ax4.set_ylabel('Accuracy', fontsize=10)
    ax4.tick_params(axis='x', labelsize=10)
    ax4.tick_params(axis='y', labelsize=10)

    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}',
                 ha='center', va='bottom', fontsize=10)


    # ---------------------------------------------------------*/
    # 5) Relative Material Proportions and Sorting Mode
    # ---------------------------------------------------------*/
    if n_points > 0:
        # Data extraction and slicing
        belt_proportions_sliced = reward_data['Belt_Proportions'][-x_limit:]
        settings_sliced = reward_data['Setting'][-x_limit:]

        # --- Get Sorting Accuracy from reward_data ---
        sorting_accuracy_sliced = reward_data.get('Accuracy', [])[-x_limit:]

        # Unzip proportions for plotting
        prop_A, prop_B, prop_C, prop_D = ([], [], [], [])
        if belt_proportions_sliced:
            proportions_unzipped = [
                (p.get('A', 0), p.get('B', 0), p.get('C', 0), p.get('D', 0))
                for p in belt_proportions_sliced
            ]
            # Ensure the list is not empty before unzipping
            if proportions_unzipped:
                prop_A, prop_B, prop_C, prop_D = zip(*proportions_unzipped)

        x_axis = np.arange(n_points - x_limit, n_points)

        # Plotting proportions as a stackplot
        ax5.stackplot(x_axis, prop_A, prop_B, prop_C, prop_D,
                      labels=['A', 'B', 'C', 'D'],
                      colors=[colors['A'], colors['B'], colors['C'], colors['D']],
                      alpha=0.7)

        # --- Add Sorting Accuracy Line ---
        ax5.plot(x_axis, sorting_accuracy_sliced, color='black', linestyle='-', linewidth=2, label='Mean Sorting Purity')


        # --- Sorting Mode Visualization (colored boxes below x-axis) ---
        mode_colors = {0: 'blue', 1: 'green'}
        mode_labels = {0: 'Boost A/C', 1: 'Boost B/D'}
        shown_labels = set()

        for i, mode in enumerate(settings_sliced):
            label = mode_labels[mode] if mode not in shown_labels else ""
            if label: shown_labels.add(mode)
            ax5.scatter(x_axis[i], -0.05, color=mode_colors[mode], marker='s', s=50, label=label)

        # Legends and labels
        ax5.legend(loc='upper left', fontsize=12)
        ax5.set_title('Belt Material Proportions & Sorting Mode',
                      fontweight='bold', fontsize=14)
        ax5.set_xlabel('Step', fontsize=12)
        ax5.set_ylabel('Proportion', fontsize=12)
        ax5.set_xlim(n_points - x_limit, n_points - 1 if n_points > 0 else 0)
        ax5.set_ylim(-0.1, 1) # Adjusted ylim to make space for markers
        ax5.grid(True, which='both', linestyle='--', linewidth=0.5)


    # ---------------------------------------------------------*/
    # 6) Current Rewards Plot
    # ---------------------------------------------------------*/
    if n_points > X_LIMIT:
        ax6.set_title(f'Current Rewards (Showing {X_LIMIT}/{n_points})',
                    fontweight='bold', fontsize=12)
    else:
        ax6.set_title('Current Rewards', fontweight='bold', fontsize=12)

    # Full reward series
    sorting_rewards = [r[0] for r in reward_data.get('Reward', [])] if 'Reward' in reward_data else []
    pressing_rewards = [r[1] for r in reward_data.get('Reward', [])] if 'Reward' in reward_data else []

    # Fallbacks for empty lists
    if not sorting_rewards: sorting_rewards = [0]
    if not pressing_rewards: pressing_rewards = [0]

    # Plotting window: base window length on whichever series is longer, but cap at X_LIMIT
    plot_len = min(max(len(sorting_rewards), len(pressing_rewards)), X_LIMIT)
    t_plot = np.arange(plot_len)

    # Cumulatives: calculate on full data for use in plot 11
    cum_sort_full = np.cumsum(sorting_rewards)
    cum_press_full = np.cumsum(pressing_rewards)

    # Current rewards - plot only the window
    ax6.plot(t_plot, sorting_rewards[:plot_len], color='blue', label='Sorting Reward', linewidth=2)
    ax6.plot(t_plot, pressing_rewards[:plot_len], color='darkgreen', label='Pressing Reward', linewidth=2)

    ax6.set_xlabel('Timesteps', fontsize=10)
    ax6.set_ylabel('Current Reward', fontsize=10)
    ax6.tick_params(axis='x', labelsize=10)
    ax6.tick_params(axis='y', labelsize=10)
    ax6.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%d' % (x)))
    ax6.xaxis.set_major_locator(ticker.MultipleLocator(5))

    # Y-Limits based on the plotted window
    try:
        vals_left = list(sorting_rewards[:plot_len]) + list(pressing_rewards[:plot_len])
        min_val, max_val = min(vals_left), max(vals_left)
        ax6.set_ylim([min_val - 1, max_val + 1] if min_val == max_val else [min_val * 1.1, max_val * 1.1])
    except ValueError:
        ax6.set_ylim([-1, 1])

    # X-Limits
    actual_data_length = len(sorting_rewards)
    ax6.set_xlim([-1, min(actual_data_length, X_LIMIT)])

    # Legend
    ax6.legend(loc='lower left', fontsize=8)


# ---------------------------------------------------------*/
# 7) Container Fill Levels Over Time (new subplot)
# ---------------------------------------------------------*/


    title_pts = len(reward_data.get('Reward', []))
    ax7.set_title(
        f'Container Fill Levels and Press Actions (Showing {min(title_pts, X_LIMIT)}/{title_pts})'
        if title_pts > X_LIMIT else 'Container Fill Levels and Press Actions',
        fontweight='bold', fontsize=12
    )

    # Helper


    def pad_list(lst, target_len): return list(lst) + [0] * max(0, target_len - len(lst))
    def pad_press_actions(actions, target_len): return list(actions) + [(0, None)] * max(0, target_len - len(actions))


    material_labels = ['A', 'B', 'C', 'D', 'E']

    # Build full series (un-sliced) to get correct Y-axis scales
    sum_A = [a + b for a, b in zip(reward_data.get('A_True', []), reward_data.get('A_False', []))]
    sum_B = [a + b for a, b in zip(reward_data.get('B_True', []), reward_data.get('B_False', []))]
    sum_C = [a + b for a, b in zip(reward_data.get('C_True', []), reward_data.get('C_False', []))]
    sum_D = [a + b for a, b in zip(reward_data.get('D_True', []), reward_data.get('D_False', []))]
    sum_E = list(reward_data.get('E_True', []))  # E is only 'True' sorted

    press_len = len(press_actions_per_timestep)
    full_len = max(press_len, len(sum_A), len(sum_B), len(sum_C), len(sum_D), len(sum_E))

    # Pad all lists to the full length
    sum_A = pad_list(sum_A, full_len)
    sum_B = pad_list(sum_B, full_len)
    sum_C = pad_list(sum_C, full_len)
    sum_D = pad_list(sum_D, full_len)
    sum_E = pad_list(sum_E, full_len)
    press_actions_full = pad_press_actions(press_actions_per_timestep, full_len)


    plot_len = min(full_len, X_LIMIT)
    t_full = list(range(full_len))
    t = list(range(plot_len))

    # Plot lines (sliced)
    ax7.plot(t, sum_A[:plot_len], label='A', color=colors['A'], linewidth=2)
    ax7.plot(t, sum_B[:plot_len], label='B', color=colors['B'], linewidth=2)
    ax7.plot(t, sum_C[:plot_len], label='C', color=colors['C'], linewidth=2)
    ax7.plot(t, sum_D[:plot_len], label='D', color=colors['D'], linewidth=2)
    ax7.plot(t, sum_E[:plot_len], label='E', color=colors['E'], linewidth=2)

    ax7.xaxis.set_major_locator(ticker.MultipleLocator(5))
    shown_labels = set()

    # X-Axis: always show full X_LIMIT window (do not shrink if data is shorter)
    ax7.set_xlim([-1, X_LIMIT])

    # Scatter plot of press actions (sliced)
    for i, (action, mat_idx) in enumerate(press_actions_full[:plot_len]):
        if isinstance(mat_idx, str):
            mat_idx = material_labels.index(mat_idx) if mat_idx in material_labels else None

        if action == 111:
            # Invalid Press 1 action - show as no-action box but with blue "x" for material name
            label = 'Invalid Press 1' if 'Invalid Press 1' not in shown_labels else ""
            shown_labels.add('Invalid Press 1')
            ax7.scatter(i, -container_global_max * 0.05, color='white', edgecolor='black', marker='s', s=40, label=label)
        elif action == 222:
            # Invalid Press 2 action - show as no-action box but with red "x" for material name
            label = 'Invalid Press 2' if 'Invalid Press 2' not in shown_labels else ""
            shown_labels.add('Invalid Press 2')
            ax7.scatter(i, -container_global_max * 0.05, color='white', edgecolor='black', marker='s', s=40, label=label)
        elif action == 999:
            # Other invalid action - show as no-action box but with gray "x" for material name
            label = 'Invalid Action' if 'Invalid Action' not in shown_labels else ""
            shown_labels.add('Invalid Action')
            ax7.scatter(i, -container_global_max * 0.05, color='white', edgecolor='black', marker='s', s=40, label=label)
        elif action == 1:
            label = 'Press 1' if 'Press 1' not in shown_labels else ""
            shown_labels.add('Press 1')
            ax7.scatter(i, -container_global_max * 0.05, color='skyblue', marker='s', s=40, label=label)
        elif action == 2:
            label = 'Press 2' if 'Press 2' not in shown_labels else ""
            shown_labels.add('Press 2')
            ax7.scatter(i, -container_global_max * 0.05, color='lightcoral', marker='s', s=40, label=label)
        else:
            label = 'No Action' if 'No Action' not in shown_labels else ""
            shown_labels.add('No Action')
            ax7.scatter(i, -container_global_max * 0.05, color='white', edgecolor='black', marker='s', s=40, label=label)

        # Display material label, but for invalid actions show small colored "x" instead
        if action == 111:
            # Invalid Press 1 - blue "x"
            ax7.text(i, -container_global_max * 0.12, 'x',
                     ha='center', va='top', fontsize=10, fontweight='normal', color='blue')
        elif action == 222:
            # Invalid Press 2 - red "x"
            ax7.text(i, -container_global_max * 0.12, 'x',
                     ha='center', va='top', fontsize=10, fontweight='normal', color='red')
        elif action == 999:
            # Other invalid action - gray "x"
            ax7.text(i, -container_global_max * 0.12, 'x',
                     ha='center', va='top', fontsize=10, fontweight='normal', color='gray')
        elif mat_idx is not None and 0 <= mat_idx < len(material_labels):
            # Valid action - show material name in black
            ax7.text(i, -container_global_max * 0.12, material_labels[mat_idx],
                     ha='center', va='top', fontsize=8, color='black')

    ax7.set_xlabel('Timesteps', fontsize=10)
    ax7.set_ylabel('Fill Level', fontsize=10)
    ax7.legend(loc='upper left', fontsize=8)
    ax7.tick_params(axis='x', labelsize=10)
    ax7.tick_params(axis='y', labelsize=10)

    # Y-axis scaling: compute from the actually displayed (sliced) data only
    try:
        if plot_len == 0:
            ax7.set_ylim([-container_global_max * 0.2, container_global_max * 1.1])
        else:
            max_fill_level = max(
                max(sum_A[:plot_len]) if len(sum_A[:plot_len]) else 0,
                max(sum_B[:plot_len]) if len(sum_B[:plot_len]) else 0,
                max(sum_C[:plot_len]) if len(sum_C[:plot_len]) else 0,
                max(sum_D[:plot_len]) if len(sum_D[:plot_len]) else 0,
                max(sum_E[:plot_len]) if len(sum_E[:plot_len]) else 0,
            )
            y_max = max(container_global_max * 1.1, max_fill_level * 1.1) if max_fill_level else container_global_max * 1.1
            ax7.set_ylim([-container_global_max * 0.2, y_max])
    except Exception:
        ax7.set_ylim([-container_global_max * 0.2, container_global_max * 1.1])

    ax7.axhline(y=container_global_max, color='red', linestyle='-', linewidth=2.5)
    num_multiples = int(container_global_max // bale_standard_size)
    for multiple in range(1, num_multiples + 1):
        lw = 2 if multiple == num_multiples else 1  # Thicker line for the uppermost one
        ax7.axhline(y=multiple * bale_standard_size, color='black', linestyle='--', linewidth=lw)

    # ---------------------------------------------------------*/
    # 8) Container Contents (5 Containers: A, B, C, D, and Rest)
    # ---------------------------------------------------------*/

    ax8.set_title('Container Contents', fontweight='bold', fontsize=12)
    grouped_keys = [
        ['A', 'A_False'],
        ['B', 'B_False'],
        ['C', 'C_False'],
        ['D', 'D_False'],
        ['E',]
    ]

    ax8.set_xticks(np.arange(len(grouped_keys)))
    ax8.set_xticklabels(['A', 'B', 'C', 'D', 'E'], fontsize=10)
    ax8.set_ylabel('Quantity', fontsize=10)
    ax8.tick_params(axis='x', labelsize=10)
    ax8.tick_params(axis='y', labelsize=10)
    total_container_contents = sum(container_materials.values())

    for index, group in enumerate(grouped_keys):
        bottoms = 0
        total = 0
        for key in group:
            value = container_materials.get(key, 0)
            total += value
            ax8.bar(index, value, bottom=bottoms, color=colors[key], label=key if bottoms == 0 else "")

            ax8.text(index, bottoms, str(value), ha='center', va='bottom', color='black', fontsize=10)
            bottoms += value

        if group[0] != 'E':
            ratio = (container_materials.get(group[0], 0) / total * 100) if total > 0 else 0
            ax8.text(index, total / 2, f"{ratio:.0f}%", ha='center', va='center', color='white',
                     fontweight='bold', fontsize=14)

    ax8.text(0.02, 0.98, f"Total: {total_container_contents}", ha='left',
             va='top', transform=ax8.transAxes, fontweight='bold', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    for index, key in enumerate(['A', 'B', 'C', 'D', 'E']):
        if key == 'E':
            label = ""  # Exclude percentage label for E
        else:
            label = f"{quality_thresholds[key] * 100:.0f}%" if quality_thresholds is not None else ""

        ax8.text(index, -0.10, label, ha='center', va='top',
                 color='black', fontsize=10, transform=ax8.get_xaxis_transform())

    # ---------------------------------------------------------*/
    # 9) Press Waiting Time
    # ---------------------------------------------------------*/

    # 1. Material quantities in presses
    n_1 = press_state.get("n_1", 0) or 0
    n_2 = press_state.get("n_2", 0) or 0

    # 2. Timer durations from env configuration
    press_time_1 = press_times.get(1, 1)
    press_time_2 = press_times.get(2, 1)

    # 3. Remaining countdown values
    countdown_value_1 = press_state.get("press_1", 0) or 0
    countdown_value_2 = press_state.get("press_2", 0) or 0

    # Max value is the total timer duration
    max_value_1 = press_time_1
    max_value_2 = press_time_2

    # 4. Elapsed time (never negative, as countdown <= max_value is guaranteed)
    elapsed_time_1 = max(0, max_value_1 - countdown_value_1)
    elapsed_time_2 = max(0, max_value_2 - countdown_value_2)

    if elapsed_time_2 < 0:
        print(f"[WARNING] Negative elapsed_time_2: likely inconsistent state — using 0.")
        elapsed_time_2 = 0  # Fallback to a safe value

    assert elapsed_time_1 >= 0, f"elapsed_time_1 < 0: {elapsed_time_1}"
    assert elapsed_time_2 >= 0, f"elapsed_time_2 < 0: {elapsed_time_2}"

    # Ensure no NaN or None values exist in pie chart sizes
    sizes_1 = np.nan_to_num([countdown_value_1, elapsed_time_1], nan=0.0)
    colors_1 = ['lightgray', 'skyblue']
    sizes_2 = np.nan_to_num([countdown_value_2, elapsed_time_2], nan=0.0)
    colors_2 = ['lightgray', 'lightcoral']

    # Ensure pie chart has valid sizes (avoid empty values)
    if sum(sizes_1) == 0 or np.isnan(sum(sizes_1)):
        sizes_1 = [1]
        colors_1 = ['white']
    if sum(sizes_2) == 0 or np.isnan(sum(sizes_2)):
        sizes_2 = [1]
        colors_2 = ['white']

    # Draw first pie chart
    ax9.pie(sizes_1, colors=colors_1, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.3), radius=1.8, center=(-1.8, 0))

    # Display press 1 info
    if countdown_value_1 > 0:
        material_1 = press_state.get("material_1", "Unknown")
        quantity_1 = press_state.get("n_1", 0) or 0
        quality_1 = int((press_state.get("q_1", 0) or 0) * 100)
        ax9.text(-1.8, 0, f'Material: {material_1}\nQuantity: {quantity_1}\nQuality: {quality_1}%',
                 ha='center', va='center', fontsize=10, fontweight='bold')
    else:
        ax9.text(-1.8, -2.2, 'Ready!', ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw second pie chart
    ax9.pie(sizes_2, colors=colors_2, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.3), radius=1.8, center=(1.8, 0))

    # Display press 2 info
    if countdown_value_2 > 0:
        material_2 = press_state.get("material_2", "Unknown")
        quantity_2 = press_state.get("n_2", 0) or 0
        quality_2 = int((press_state.get("q_2", 0) or 0) * 100)
        ax9.text(1.8, 0, f'Material: {material_2}\nQuantity: {quantity_2}\nQuality: {quality_2}%',
                 ha='center', va='center', fontsize=10, fontweight='bold')
    else:
        ax9.text(1.8, -2.2, 'Ready!', ha='center', va='center', fontsize=10, fontweight='bold')

    # Add labels
    ax9.text(-1.8, 2.2, 'Press 1', ha='center', va='center', fontweight='bold', fontsize=12)
    ax9.text(1.8, 2.2, 'Press 2', ha='center', va='center', fontweight='bold', fontsize=12)
    ax9.axis('equal')
    ax9.axis('off')

    # ---------------------------------------------------------*/
    # 10) Bales Produced
    # ---------------------------------------------------------*/
    bales = bale_count
    categories = ['A', 'B', 'C', 'D', 'E']
    # Here, 'E' is used as the residual container; the x-axis label is adjusted accordingly.
    cmap = LinearSegmentedColormap.from_list('deviation_cmap', ['red', 'green', 'red'], N=10)
    norm = BoundaryNorm(np.linspace(0.5, 1.5, 11), cmap.N)

    def get_color(deviation):
        return cmap(norm(deviation))
    bottoms = {key: 0 for key in categories}
    for idx, key in enumerate(categories):
        if key in bales and len(bales[key]) > 0:
            for i in range(len(bales[key])):
                size, quality = bales[key][i]
                deviation = size / bale_standard_size
                bar = ax10.bar(idx, deviation, bottom=bottoms[key], color=get_color(deviation))
                ax10.text(idx, bottoms[key] + deviation / 2, f"{int(quality)}%",
                          ha='center', va='center', fontsize=8, color='black')
                bottoms[key] += deviation
    ax10.set_xticks(range(len(categories)))
    ax10.set_xticklabels(['A', 'B', 'C', 'D', 'E'])
    ax10.set_title('Bales Produced', fontweight='bold', fontsize=12)
    ax10.set_ylabel('Relative Bale Size', fontsize=10)
    if max(bottoms.values()) > 0:
        ax10.set_ylim([0, max(bottoms.values()) * 1.1])
    else:
        ax10.set_ylim([0, 1])
    ax10.tick_params(axis='x', labelsize=10)
    ax10.tick_params(axis='y', labelsize=10)
    ax10.yaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Calculate and display mean purity and bale count ---
    for idx, key in enumerate(categories):
        bales_list = bales.get(key, [])

        # Always display bale count inside the bar (or n=0 if empty)
        ax10.text(idx, 0, f"n={len(bales_list)}", ha='center', va='bottom',
                  fontsize=14, color='white')

        # For containers A-D show mean purity below the x-axis; skip E
        if key != 'E' and bales_list:
            mean_purity = np.mean([q for s, q in bales_list])
            ax10.text(idx, -0.10, f"{mean_purity:.0f}%", ha='center', va='top',
                      color='black', fontsize=10, transform=ax10.get_xaxis_transform())

    total_material = sum(size for bales_list in bale_count.values() for size, quality in bales_list)
    ax10.text(0.02, 0.98, f'Total Material = {total_material}', transform=ax10.transAxes,
              ha='left', va='top', fontweight='bold', fontsize=10)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax10, orientation='vertical')
    cbar.set_label('Deviation from Bale Size', fontsize=10)
    cbar.set_ticks(np.linspace(0.5, 1.5, 11))
    cbar.set_ticklabels([f'{t:.1f}' for t in np.linspace(0.5, 1.5, 11)])

    # ---------------------------------------------------------*/
    # 11) Cumulative Rewards Plot
    # ---------------------------------------------------------*/
    ax11.set_title('Cumulative Rewards', fontweight='bold', fontsize=12)
    
    # Calculate total cumulative reward
    total_rewards = [sum(reward) for reward in reward_data['Reward']]
    cumulative_total_reward = np.cumsum(total_rewards)
    
    ax11.set_xlabel('Timesteps', fontsize=10)
    ax11.set_ylabel('Cumulative Reward', fontsize=10)

    # Use the length of the longest cumulative series for the x-axis
    x_len = len(cumulative_total_reward)
    x_axe = range(x_len)

    # Plot all three cumulative rewards
    ax11.plot(x_axe, cumulative_total_reward, color='purple', label='Cumulative Total', linewidth=2.5)
    ax11.plot(x_axe, cum_sort_full[:x_len], color='blue', linestyle='--', label='Cumulative Sorting', linewidth=1.5)
    ax11.plot(x_axe, cum_press_full[:x_len], color='darkgreen', linestyle='--', label='Cumulative Pressing', linewidth=1.5)
    
    ax11.legend(loc='upper left', fontsize=8)

    # --- X-Axis dynamic limit ---
    ax11.set_xlim([0, x_len - 1 if x_len > 0 else 0])

    # --- Maximum 10 x-axis labels ---
    max_labels = 10
    if x_len > 1:
        step = max(1, (x_len - 1) // (max_labels - 1))
    else:
        step = 1
    
    xticks = list(range(0, x_len, step))
    if x_len > 0 and (not xticks or (x_len - 1) > xticks[-1]):
        xticks.append(x_len - 1)
    
    ax11.set_xticks(list(dict.fromkeys(xticks))) # Remove duplicates
    ax11.set_xticklabels([str(x) for x in list(dict.fromkeys(xticks))])


    # --- Y-Axis dynamic limits ---
    try:
        all_cum_rewards = np.concatenate([
            cumulative_total_reward,
            cum_sort_full[:x_len],
            cum_press_full[:x_len]
        ])
        min_reward = np.min(all_cum_rewards)
        max_reward = np.max(all_cum_rewards)
        if min_reward == max_reward:
            ax11.set_ylim([min_reward - 1, max_reward + 1])
        else:
            ax11.set_ylim([min_reward * 1.1, max_reward * 1.1])
    except ValueError:
        ax11.set_ylim([0, 1])

    final_total_reward = round(cumulative_total_reward[-1] if cumulative_total_reward.size > 0 else 0, 2)
    ax11.text(0.02, 0.8, f'Final Cumulative Total: {final_total_reward}',
            transform=ax11.transAxes, ha='left', va='bottom',
            fontweight='bold', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


    # ---------------------------------------------------------*/
    # Calculate the checksum
    # ---------------------------------------------------------*/
    if checksum:
        # Checksum for all values
        total_material_in_containers = sum(container_materials.values())
        total_material_in_presses = sum([press_state["n_1"], press_state["n_2"]])
        total_material_in_bales = sum(value[0] for values in bale_count.values() for value in values)
        checksum_val = total_material_in_containers + total_material_in_presses + total_material_in_bales

        print(
            f"🔍 Checksum (Seed={seed}): {checksum_val} = ({total_material_in_containers} Containers + {total_material_in_presses} Presses + {total_material_in_bales} Bales)")

        input_length = len(sum(env.input_history_batches, []))
        print("🔍 Length of Inputs: ", input_length)

        flat_inputs = sum(env.input_history_batches, [])
        first_10 = flat_inputs[:10]
        print(f"First 10 elements: {first_10}")

    # ---------------------------------------------------------*/
    # Generate the plot ✍🏼
    # ---------------------------------------------------------*/
    if save or show:
        plt.tight_layout()
        if save:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            plt.savefig(f"{log_dir}/{filename}.{format}", format=format, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    plt.close()
    fig.clf()


# ---------------------------------------------------------*/
# Save the plot to a file
# ---------------------------------------------------------*/

def save_plot(fig, dir, base_filename, extension, dpi=300):
    """Saves the plot to a file with a unique filename in the specified directory."""
    os.makedirs(dir, exist_ok=True)  # Ensure the directory exists

    # Generate unique filename
    i = 0
    filename = f"{base_filename}_{i}.{extension}"
    while os.path.exists(os.path.join(dir, filename)):
        i += 1
        filename = f"{base_filename}_{i}.{extension}"

    # Save the figure
    if extension == 'svg':
        fig.savefig(os.path.join(dir, filename), format=extension)
    else:
        fig.savefig(os.path.join(dir, filename), format=extension, dpi=dpi, bbox_inches='tight')

# ---------------------------------------------------------*/
# Create a Video from a folder of images
# ---------------------------------------------------------*/


def create_video(folder_path, output_path, display_duration=1):
    """ Creates a video from a folder of images.
    - folder_path: The path to the folder containing the images.
    - output_path: The path to save the output video.
    - display_duration: The duration (in seconds) to display each image.
    """
    sample_img = cv2.imread(os.path.join(folder_path, os.listdir(folder_path)[0]))
    height, width, layers = sample_img.shape
    size = (width, height)

    # Calculate frame rate based on the desired display duration of each image
    frame_rate = 1 / display_duration

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, size)

    # Function to extract numbers from the filename for correct sorting
    def sort_key(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    # Sort files numerically and write the video
    for filename in sorted(os.listdir(folder_path), key=sort_key):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path)
            img = cv2.resize(img, size)  # Resize the image to the target size
            out.write(img)

    print("Video created. 🎥🍿")
    out.release()

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*/
