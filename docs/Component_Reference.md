# Component and Parameter Reference

This document provides a detailed reference for the components, parameters, and functionalities of the simulation environment.

## 1. Environment Overview

The simulation models a multi-stage material handling facility with three main stages:
1.  **Input Generation**: A seasonal input generator creates a stream of four material types (A, B, C, D).
2.  **Sorting**: A sorting station attempts to separate the materials into their respective containers. Its efficiency is influenced by a sensor setting that can be controlled by an agent.
3.  **Pressing**: Two presses are available to compress materials from five containers (one for each material type A-D, and one for waste 'E') into bales.

The system is designed to be controlled by different reinforcement learning agent architectures: a modular setup with specialized agents for sorting and pressing, and a single monolithic agent controlling both.

## 2. Initialization Parameters

These parameters are passed to the environment's `__init__` constructor (e.g., `Env_1_Sorting(...)`).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_steps` | `int` | `50` | The maximum number of steps per episode. |
| `seed` | `int` | `None` | The master seed for all random number generators. |
| `noise_input` | `float` | `0.05` | The noise level for the input generation process. *(Note: Currently not used in `input_generator.py`)* |
| `noise_sorting` | `float` | `0.05` | The noise level affecting sorting accuracy. A value of 0.05 means accuracy can be randomly reduced by up to 5%. |
| `balesize` | `int` | `200` | The standard size (in units) of a material bale. This is a key parameter for the pressing reward. |
| `simulation` | `bool` | `False` | A flag for simulation mode, not currently affecting core logic. |

---

## 3. Core Components & Key Parameters

These parameters are loaded from `config.yml` during environment initialization.

### Input Generation

The input is managed by the `SeasonalInputGenerator` class.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Material Types** | `["A", "B", "C", "D"]` | The four types of materials processed by the facility. |
| **Batch Size Range** | `[10, 100]` | In each step, the environment generates a new batch of materials with a size randomly chosen between these values (if not specified by a rule). |
| **Pattern Probabilities** | `[0.2, 0.2, 0.2, 0.4]` | The `SeasonalInputGenerator` has 4 patterns. This is the probability of selecting each pattern when a new material sample is generated. |

### Sorting Station

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `baseline_accuracy`| `(0.8, 0.8, 0.8, 0.8)` | The base sorting accuracy for each material type before any boosts or penalties. |
| `boost` | `0.5` | The accuracy increase applied to certain materials when a sensor mode is active. |
| `sensor_all_settings`| `[0, 1]` | The two available sensor modes. Mode `0` boosts materials A & C. Mode `1` boosts materials B & D. |
| `reduction_factor`| `((total_input / 100) ** 2) * 0.2` | A penalty applied to sorting accuracy based on the total amount of material on the belt. Higher occupancy leads to lower accuracy. |

### Pressing Station & Containers

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `press_times` | `{1: 15, 2: 20}` | The number of timesteps it takes for Press 1 and Press 2 to complete a job. |
| `container_global_max`| `650` | The maximum capacity (in units) for each container. |
| `bale_standard_size`| `200` | The target size for a standard bale. This is the same as the `balesize` init parameter. |
| `quality_thresholds`| `{"A": 0.85, ...}` | The minimum purity (e.g., 85%) required for a container's contents to be considered high quality. Used in the sorting reward calculation. |
| **Waste Container** | `"E"` | A dedicated container that collects all mis-sorted materials that cannot be redistributed. |

---

## 4. Reward Functions

### Sorting Reward (`calculate_sorting_reward`)

The sorting reward is calculated based on the purity of the material in containers A, B, C, and D.

- **Shaping Reward**: A continuous reward is given based on the difference between the container's current purity and its `quality_threshold`.
  - If `purity >= threshold`, the reward is `(purity - threshold)`.
  - If `purity < threshold`, the penalty is `(purity - threshold) * 2.0` (i.e., the penalty is twice as strong as the reward).
- **Fixed Penalty**: A fixed penalty of `-0.1` is applied for each container whose purity is below its threshold.
- **Scaling**: The final reward is normalized by dividing by 4 (the number of containers) and clipped to the range `[-1.0, 1.0]`.

### Pressing Reward (`calculate_pressing_reward`)

The pressing reward is composed of a state-based penalty and an action-based reward.

- **State-Based Overflow Penalty**: This penalty is applied at every step to discourage high container levels.
  - If `fill_ratio > 1.0`, the penalty is `-1.0` (catastrophic).
  - If `fill_ratio > 0.95`, the penalty is `-0.5`.
  - If `fill_ratio > 0.90`, the penalty is `-0.2`.
- **Action-Based Pressing Reward**: This reward is given only when a press action is initiated.
  - **Bale Efficiency Reward**: A triangular wave function that rewards pressing amounts close to multiples of the `bale_standard_size` and penalizes pressing amounts that are halfway between multiples (e.g., 1.5 bales worth). The reward is `(1.0 - 2.0 * (dist_from_multiple / S)) * 0.5`.
  - **Full Bale Bonus**: A bonus of `0.1` is added for each full bale produced in the action (e.g., `amount // S`).
- **Scaling**: The final reward is clipped to the range `[-1.0, 1.0]`.

---

## 5. Observation Spaces

### Sorting Agent (`Env_1_Sorting`)

A 14-dimensional continuous space (`Box(14,)`):
- **Belt Occupancy** (1): Total material on the belt (normalized 0-1).
- **Belt Proportions** (4): Proportion of each material (A,B,C,D) on the belt.
- **Current Sort Mode** (1): The active sensor setting (0 or 1).
- **Sorting Accuracies** (4): The current sorting accuracy for each material.
- **Purity Differences** (4): The difference between current container purity and the quality threshold for each material.

### Pressing Agent (`Env_2_Pressing`)

A 16-dimensional continuous space (`Box(16,)`), all values normalized to `[0, 1]`:
- **Container Levels** (5): Fill level of containers A, B, C, D, E.
- **Fill Ratios** (5): Same as container levels (redundant).
- **Sorter Contents** (4): Amount of each material currently in the sorting machine stage.
- **Press Timers** (2): Remaining time for Press 1 and Press 2 jobs.

### Monolithic Agent (`Env_3_Monolith`)

A 30-dimensional continuous space (`Box(30,)`), which is the concatenation of the Sorting and Pressing observation spaces.

---

## 6. Action Spaces

### Sorting Agent (`Env_1_Sorting`)

A discrete space with 2 actions (`Discrete(2)`):
- **0**: Activate sensor mode 0 (boosts materials A & C).
- **1**: Activate sensor mode 1 (boosts materials B & D).

### Pressing Agent (`Env_2_Pressing`)

A discrete space with 11 actions (`Discrete(11)`), with action masking applied.
- **0**: No-Op (do nothing).
- **1-5**: Use Press 1 on container A, B, C, D, or E.
- **6-10**: Use Press 2 on container A, B, C, D, or E.

### Monolithic Agent (`Env_3_Monolith`)

A flattened discrete space with 22 actions (`Discrete(22)`), with action masking applied. The action is a combination of the sorting and pressing actions.
- **Actions 0-10**: Corresponds to `sorting_mode=0` and `pressing_action` 0-10.
- **Actions 11-21**: Corresponds to `sorting_mode=1` and `pressing_action` 0-10.

---