# ---------------------------------------------------------*\
# Title: Environment (Master) - Modified for 4 Materials and 5 Containers
# ---------------------------------------------------------*/

import gymnasium as gym
import numpy as np
import random
import yaml
from utils.input_generator import SeasonalInputGenerator
from utils.plotting import plot_env
from collections import Counter, deque


# ---------------------------------------------------------*/
# Environment Super
# ---------------------------------------------------------*/
class Env_Super(gym.Env):
    """Custom Gym environment for a simple sorting system with 4 materials (A, B, C, D)
       and 5 containers. For active stations (A-D) the true sorted material is stored under the material key
       and the false-sorted material under the key "<material>_False".
       Any leftover false material (after sequential redistribution) is collected in container E.
       Container E is also considered for pressing.
    """

    def __init__(self, max_steps=50, seed=None, noise_sorting=None, balesize=None, simulation=False, config_path="config.yml"):

        # --- Load Configuration ---
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # --- Load Simulation Parameters from Config ---
        sim_cfg = self.config['simulation']
        self.input_batch_size = sim_cfg['input_batch_size']
        self.steps_per_pattern = sim_cfg['steps_per_pattern']
        self.input_history_length = sim_cfg['input_history_length']

        # For Training: Temporarily store the first agent
        self.train_mode = 0
        self.sort_agent = None

        # Active materials: A, B, C, D
        self.material_names = ["A", "B", "C", "D"]

        # ------ Define the Action-Space and Observation-Space ------
        self._initialize_spaces()

        # Set the seed
        self.seed = seed
        self.set_seed(seed)

        # ------Sorting Machine: Agent (1) ------
        self.current_material_input = [0, 0, 0, 0]       # Materials: A, B, C, D
        self.current_material_belt = [0, 0, 0, 0]        # On the belt
        self.current_material_sorting = [0, 0, 0, 0]     # In the sorting machine

        # Load sorting parameters from config
        sorting_cfg = self.config['sorting_station']
        self.baseline_accuracy = tuple(sorting_cfg['baseline_accuracy'])
        self.boost = sorting_cfg['boost']
        self.accuracy_belt = list(self.baseline_accuracy)
        self.accuracy_sorter = list(self.baseline_accuracy)
        self.sorting_stage_capacity = sorting_cfg['stage_capacity']

        self.sensor_current_setting = 0
        self.sensor_all_settings = [0, 1, 2, 3]

        self.input_occupancy = sum(self.current_material_input) / 100
        self.belt_occupancy = sum(self.current_material_belt) / 100

        # Noise level for sorting accuracy
        self.noise_accuracy = noise_sorting if noise_sorting is not None else sorting_cfg['noise']

        # Initialize containers
        self.container_materials = {name: 0 for name in self.material_names}
        self.container_materials.update({f"{name}_False": 0 for name in self.material_names})
        self.container_materials["E"] = 0

        # ------ Presses and Bales: Agent (2) ------
        self.press_state = {
            "press_1": 0, "material_1": 0, "n_1": 0, "q_1": 0,
            "press_2": 0, "material_2": 0, "n_2": 0, "q_2": 0
        }
        self.bale_count = {material: [] for material in self.material_names + ["E"]}

        self.press_penalty_flag = 0
        pressing_cfg = self.config['pressing_station']
        self.bale_standard_size = balesize if balesize is not None else pressing_cfg['bale_standard_size']
        self.bale_remainder_threshold = pressing_cfg['bale_remainder_threshold']

        self._last_press_started = False
        self._last_press_amount = 0
        self._last_press_start_level = 0

        self.press_actions_per_timestep = []

        # Load pressing parameters from config
        self.press_times = pressing_cfg['press_times']
        self.quality_thresholds = pressing_cfg['bale_quality_thresholds']
        self.container_global_max = pressing_cfg['container_capacity']

        self.container_max = {
            "A": self.container_global_max,
            "B": self.container_global_max,
            "C": self.container_global_max,
            "D": self.container_global_max,
            "E": self.container_global_max
        }

        # ------ Input Control ------
        self.input_generator = SeasonalInputGenerator(seed=seed, steps_per_pattern=self.steps_per_pattern)
        self.input_history_batches = []

        # --- Sorting Mode Change Penalty ---
        self.last_sort_mode = None
        self.sorting_mode_change_penalty = self.config['rewards']['sorting']['sorting_mode_change_penalty']

        # --- Asymmetric Reward Shaping ---
        sorting_reward_cfg = self.config['rewards']['sorting']
        self.purity_threshold_theta = sorting_reward_cfg['purity_threshold_theta']
        self.decay_steepness_k = sorting_reward_cfg['decay_steepness_k']
        self.min_weight = sorting_reward_cfg['min_weight']
        self.reward_scaling_factor = sorting_reward_cfg['reward_scaling_factor']
        self.sorting_mode_change_penalty = sorting_reward_cfg['sorting_mode_change_penalty']
        self.sorting_reward_temperature = sorting_reward_cfg['tanh_temperature']

        self.last_purities = {mat: 0.0 for mat in self.material_names}
        self.last_sort_mode = None

        # ----- Reward ------
        self.PRESS_REWARD = 0
        pressing_reward_cfg = self.config['rewards']['pressing']
        self.press_reward_max_state_reward = pressing_reward_cfg['max_state_reward']
        self.overflow_penalty = self.config['rewards']['overflow_termination_penalty']

        self.max_steps = max_steps
        self.previous_setting = None
        self.current_step = 0

    def _reset_containers(self):
        """Resets all container-related state variables."""
        self.container_materials = {name: 0 for name in self.material_names}
        self.container_materials.update({f"{name}_False": 0 for name in self.material_names})
        self.container_materials["E"] = 0

    def _calculate_asymmetric_weight(self, p):
        """
        Calculates an asymmetric weight factor for the reward function based on purity p.
        - Constant high (1.0) below the threshold.
        - Decays like a half-Gaussian curve above the threshold.
        """
        if p < self.purity_threshold_theta:
            return 1.0
        else:
            return (1.0 - self.min_weight) * np.exp(-self.decay_steepness_k * (p - self.purity_threshold_theta)**2) + self.min_weight

    # ---------------------------------------------------------*/
    # Global Helper Functions
    # ---------------------------------------------------------*/
    def _initialize_spaces(self):
        """Initialize the action and observation spaces.
           (Must be overridden in a subclass.)
        """
        raise NotImplementedError("Subclasses must override _initialize_spaces()")

    def set_seed(self, seed):
        """Set seed for the environment and its components with separated RNG streams."""
        self.seed = seed or 0  # fallback to 0 if None

        # Separate random sources for reproducibility
        self.rng_input = np.random.default_rng(self.seed + 1)
        self.rng_sorting = np.random.default_rng(self.seed + 2)
        self.rng_pressing = np.random.default_rng(self.seed + 3)
        self.rng_noise = np.random.default_rng(self.seed + 4)
        self.rng = np.random.default_rng(self.seed + 99)  # general fallback RNG

        # Optional: synchronize global RNG as well
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Initialize Gym spaces deterministically
        if hasattr(self, 'action_space'):
            self.action_space.seed(self.seed)
        if hasattr(self, 'observation_space'):
            self.observation_space.seed(self.seed)

    def compute_input_proportions(self):
        """Compute the proportions of the current input materials (A-D)."""
        current_input = self.current_material_input
        total_input_amount = sum(current_input)
        proportions = {}
        if total_input_amount > 0:
            for material, amount in zip(self.material_names, current_input):
                proportions[material] = amount / total_input_amount
        else:
            for material in self.material_names:
                proportions[material] = 0
        return proportions

    def compute_belt_proportions(self):
        """Compute the proportions of materials on the belt (A-D)."""
        current_belt = self.current_material_belt
        total_belt_amount = sum(current_belt)
        proportions = {}
        if total_belt_amount > 0:
            for material, amount in zip(self.material_names, current_belt):
                proportions[material] = amount / total_belt_amount
        else:
            for material in self.material_names:
                proportions[material] = 0
        return proportions

    def compute_purity_differences(self, scaling_factor=1):
        """
        Compute purity deviations from quality thresholds for active containers (A-D).
        """
        container_purities, _ = self.get_container_purity()
        purity_differences = []

        for mat in self.material_names:
            purity = container_purities.get(mat, 0)
            threshold = self.quality_thresholds[mat]
            diff = purity - threshold
            if diff < 0:
                diff *= scaling_factor
            purity_differences.append(round(diff, 2))

        return purity_differences

    def render(self, mode="human", save=False, show=True, log_dir="./img/log", filename="plot", title="", format="svg",
               checksum=True, steps_test=None):
        """Render the environment."""
        plot_env(env=self, save=save, show=show, log_dir=log_dir, filename=filename, title=title, format=format,
                 checksum=checksum, steps_test=steps_test)

    def _calculate_avg_bale_deviation(self):
        deviations = []
        for material, bales in self.bale_count.items():
            for bale_size, _ in bales:
                dev = abs(bale_size - self.bale_standard_size) / self.bale_standard_size
                deviations.append(dev)
        return np.mean(deviations) if deviations else 0.0

    def check_material_conservation(self):
        """
        Checks if the total amount of material in the system matches the sum of all input batches.
        Includes:
        - containers
        - presses
        - bales
        - input, belt, sorting stages
        """

        # Total input material (sum of all previous input batches)
        total_input = sum(len(batch) for batch in self.input_history_batches)

        # Material currently in the system:
        in_containers = sum(self.container_materials.values())
        in_presses = self.press_state["n_1"] + self.press_state["n_2"]
        in_bales = sum(b[0] for bales in self.bale_count.values() for b in bales)

        in_input_stage = sum(self.current_material_input)
        in_belt_stage = sum(self.current_material_belt)
        in_sorting_stage = sum(self.current_material_sorting)

        total_system = (
            in_containers +
            in_presses +
            in_bales +
            in_input_stage +
            in_belt_stage +
            in_sorting_stage
        )

        delta = total_system - total_input

        if abs(delta) > 1:
            print(f"❌ Material balance violated: Δ = {delta}")
            print(f"  ↪ Inputs:     {total_input}")
            print(f"  ↪ Containers: {in_containers}")
            print(f"  ↪ Presses:    {in_presses}")
            print(f"  ↪ Bales:      {in_bales}")
            print(f"  ↪ InputStage: {in_input_stage}")
            print(f"  ↪ BeltStage:  {in_belt_stage}")
            print(f"  ↪ Sorting:    {in_sorting_stage}")
            raise AssertionError("Total material in the system does not match the inputs.")
        else:
            print(f"✅ Material balance OK: Δ = {delta}")

    # in Env_Super

    def sample_masked_press_action(self):
        """Samples a random press action from the set of valid actions (mask).
        Returns (press_id, mat_id) or (None, None) if no action is possible.
        """
        mask = self.press_action_masks()
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return (None, None)
        a_disc = int(self.rng_pressing.choice(valid))
        return self.press_discrete_to_action(a_disc)

    # ---------------------------------------------------------*/
    # Observation Functions
    # ---------------------------------------------------------*/

    def get_sort_obs(self):
        """
        Returns the observation for the sorting agent.
        - Belt occupancy
        - Belt material proportions (4)
        - Sorting accuracies (4)
        - Purity differences (4)
        = 13 values
        """
        belt_proportions = list(self.compute_belt_proportions().values())
        purity_diffs = self.compute_purity_differences()

        obs = np.array(
            [self.belt_occupancy] +
            belt_proportions +
            list(self.accuracy_belt) +
            purity_diffs,
            dtype=np.float32
        )
        return np.clip(obs, -1.0, 1.0)

    def get_press_obs(self):
        """
        Assemble the observation vector for the pressing agent.

        Returns a 16-dimensional vector, with all values normalized to [0, 1].
        [
            A_level, ..., E_level          (normalized by container_global_max),
            A_ratio, ..., E_ratio          (normalized by container_global_max, redundant but kept for consistency),
            sort_A, ..., sort_D            (normalized by a typical max sorting capacity, e.g., 100),
            press1_timer, press2_timer     (normalized by max press time)
        ]
        """
        # 1. Normalized container levels
        levels = []
        for mat in self.material_names:
            total_material = self.container_materials[mat] + self.container_materials.get(f"{mat}_False", 0)
            levels.append(total_material / self.container_global_max)
        levels.append(self.container_materials["E"] / self.container_global_max)

        # 2. Normalized fill ratios (identical to levels, can be simplified later if needed)
        ratios = levels.copy()

        # 3. Normalized sorting machine contents (A–D)
        # Assuming a max capacity for the sorting stage for normalization
        sorter_amounts = [x / self.sorting_stage_capacity for x in self.current_material_sorting]

        # 4. Normalized press timers
        press_timers = [
            self.press_state[f"press_{i}"] / self.press_times[i] for i in [1, 2]
        ]

        obs = np.array(levels + ratios + sorter_amounts + press_timers, dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    # ---------------------------------------------------------*/
    # Reset, Step, and Update Functions
    # ---------------------------------------------------------*/

    def reset(self, seed=None, options=None):
        # Note: do not call super().reset() here — calling super() in the base
        # class caused a TypeError when subclasses also used super().reset().
        # Initialize/reset state directly instead.
        # Reset containers and other state variables
        self._reset_containers()
        self.current_step = 0
        self.total_reward = 0
        self.sorting_ops = 0

        self.input_generator = SeasonalInputGenerator(seed=seed)

        if seed is not None:
            self.set_seed(seed)

        self.current_material_input = [0, 0, 0, 0]
        self.current_material_belt = [0, 0, 0, 0]
        self.current_material_sorting = [0, 0, 0, 0]

        self.press_state = {
            "press_1": 0, "material_1": 0, "n_1": 0, "q_1": 0,
            "press_2": 0, "material_2": 0, "n_2": 0, "q_2": 0
        }
        self._last_press_started = False
        self._last_press_amount = 0
        self._last_press_start_level = 0

        self.press_actions_per_timestep = []

        self.input_history = {material: deque(maxlen=self.input_history_length) for material in self.material_names}
        self.accuracy_belt = list(self.baseline_accuracy)
        self.accuracy_sorter = list(self.baseline_accuracy)
        self.sensor_current_setting = 0
        self.input_occupancy = sum(self.current_material_input) / 100
        self.belt_occupancy = sum(self.current_material_belt) / 100
        self.bale_count = {material: [] for material in self.material_names + ["E"]}
        self.previous_setting = None
        self.reward_data = {
            'Accuracy': [],
            'Setting': [],
            'Belt_Occupancy': [],
            'Reward': [],
            'Belt_Proportions': [],
        }
        self.press_penalty_flag = 0
        self.input_history_batches = []

        # Reset internal state variables for rewards (after containers are reset)
        initial_purities, _ = self.get_container_purity()
        self.last_purities = initial_purities
        # Initialize last_sort_mode to current setting to avoid penalty on first step
        self.last_sort_mode = self.sensor_current_setting

        observation = self._get_obs()
        info = {}  # No _get_info() method exists, return empty dict
        return observation, info

    def _get_obs(self):
        """
        Deprecated: Legacy observation function.
        To be removed in future versions.
        """
        return self.get_sort_obs()

    def step(self, action):
        """Step function to be implemented in subclasses."""
        raise NotImplementedError("Subclasses must override _initialize_spaces()")

    def update_environment(self, batchsize=None): # Keep arg for compatibility
        """
        Advance one step in the material flow:
        - Moves sorting → pressed, belt → sorting, input → belt
        - New input is generated with a constant batch size of 100.
        """
        # Advance material along the plant
        self.current_material_sorting = self.current_material_belt.copy()
        self.current_material_belt = self.current_material_input.copy()
        self.belt_occupancy = self.input_occupancy

        # Generate new input batch with constant size of 100, ignoring argument
        input_batch = self.input_generator.generate_input(batchsize=self.input_batch_size)
        self.input_history_batches.append(list(input_batch))

        material_counts = Counter(input_batch)

        # Update input material vector
        self.current_material_input = [
            material_counts.get(m, 0) for m in self.material_names
        ]

        # Update occupancy and sorting accuracy
        self.input_occupancy = round(sum(self.current_material_input) / 100, 2)
        self.accuracy_sorter = self.accuracy_belt.copy()

        # Record material history
        for i, material in enumerate(self.material_names):
            self.input_history[material].append(self.current_material_input[i])

        # self.check_material_conservation()

    # ---------------------------------------------------------*/
    # (1) Sorting Agent
    # ---------------------------------------------------------*/

    def sorting_rules(self):
        """
        Rule-based sorting: Choose the sorting mode that maximizes the purity
        of the most abundant material group on the belt.
        """
        proportions = self.compute_belt_proportions()
        if not proportions:
            return 0  # Default action

        # Decide based on dominant material group
        if proportions.get('A', 0) + proportions.get('C', 0) > proportions.get('B', 0) + proportions.get('D', 0):
            return 0  # Boost A/C
        else:
            return 1  # Boost B/D

    def set_multisensor_mode(self, new_sensor_setting):
        """
        Sets the sorting mode.
        - 0: Boost A & C
        - 1: Boost B & D
        """
        self.sensor_current_setting = new_sensor_setting

    def update_accuracy(self):
        """
        Update sorting accuracy based on the current sensor setting.
        - Mode 0 boosts A and C.
        - Mode 1 boosts B and D.
        A small amount of noise is added.
        """
        accuracies = list(self.baseline_accuracy)
        if self.sensor_current_setting == 0:
            accuracies[0] += self.boost  # Boost A
            accuracies[2] += self.boost  # Boost C
        elif self.sensor_current_setting == 1:
            accuracies[1] += self.boost  # Boost B
            accuracies[3] += self.boost  # Boost D

        # Apply noise to all accuracies
        noise = self.rng_noise.uniform(-self.noise_accuracy, self.noise_accuracy, len(accuracies))
        self.accuracy_belt = np.clip(np.array(accuracies) + noise, 0, 1).tolist()

    def sort_material(self):
        """
        Sorts the material from the sorting machine into the containers based
        on the current sorting mode (A/C or B/D).
        The function processes each active station (A-D) sequentially.
        For each station i:
        - target_amount = current input.
        - true_material = int(round(target_amount * accuracy)).
        - false_material = target_amount - true_material.
        - Remove the input from station i.
        - Redistribute false_material from other stations using an iterative probability-based approach.
        - Store computed true_material and false_material in the corresponding containers.
        After processing, any remaining discrepancy is added to container E.
        """

        # Store initial material count before sorting
        total_input = sum(self.current_material_sorting)

        # Initialize tracking arrays
        leftover = np.array(self.current_material_sorting, dtype=int)
        true_arr = np.zeros_like(leftover)
        false_arr = np.zeros_like(leftover)

        for i in range(len(self.material_names)):
            target_amount = leftover[i]
            acc = self.accuracy_sorter[i]

            # Compute sorted (true) and mis-sorted (false) materials
            true_val = int(round(target_amount * acc))
            false_val = target_amount - true_val

            true_arr[i] = true_val
            false_arr[i] = false_val

            # Remove input from the current station
            leftover[i] = false_val

            # Identify other indices for redistribution
            other_indices = list(range(len(self.material_names)))
            distributed = np.zeros(len(other_indices), dtype=int)

            # Iteratively distribute false material unit by unit
            for _ in range(false_val):
                available = leftover[other_indices]  # Current available material in each container
                total_avail = available.sum()

                if total_avail == 0:
                    # No more material available for redistribution
                    break

                # Compute probability distribution for redistribution
                pvals = available / total_avail
                selected = self.rng.choice(len(other_indices), p=pvals)

                # Ensure that we do not remove more than available
                if leftover[other_indices[selected]] > 0:
                    leftover[other_indices[selected]] -= 1
                    distributed[selected] += 1
                else:
                    raise ValueError(f"Container {other_indices[selected]} has insufficient material "
                                     f"(available: {leftover[other_indices[selected]]}).")

            # Validate that all false materials have been redistributed
            if distributed.sum() != false_val:
                diff = false_val - distributed.sum()
                print(f"Warning: Iterative redistribution failed to distribute {diff} units.")

        # Remaining leftover material is moved to container E
        e_input = sum(leftover)

        # Ensure material conservation: Adjust discrepancies if necessary
        total_output = sum(true_arr) + sum(false_arr) + e_input
        discrepancy = total_input - total_output

        if discrepancy != 0:
            if discrepancy == -1:
                e_input -= 1
            elif discrepancy == 1:
                e_input += 1
            else:
                print(f"DISCREPANCY: {discrepancy}")
                print(f"Total Input: {total_input}")
                print(f"Total Output: True: {sum(true_arr)}, False: {sum(false_arr)}, E: {e_input}")
                raise ValueError(f"Detected material loss or gain! Input: {total_input}, Output: {total_output}")

        # Store leftover material in container E
        self.container_materials["E"] += e_input

        # Update material containers with sorted and mis-sorted materials
        for i, material in enumerate(self.material_names):
            self.container_materials[material] += true_arr[i]
            self.container_materials[f"{material}_False"] += false_arr[i]

        # Compute and return mean purity
        mean_purity = round(1 - ((total_input - sum(true_arr)) / total_input), 2) if total_input > 0 else 0

        self.reward_data.setdefault('Accuracy', []).append(mean_purity)

        return mean_purity

    def get_belt_purity(self):
        """Calculate belt error rate based on true vs. false sorting."""
        materials = self.current_material_belt
        accuracies = self.accuracy_belt
        true_sorted = [materials[i] * accuracies[i] for i in range(len(materials))]
        false_sorted = [materials[i] - true_sorted[i] for i in range(len(materials))]
        total_material = sum(materials)
        total_false = sum(false_sorted)
        error_rate = total_false / total_material if total_material > 0 else 0
        return error_rate

    # ---------------------------------------------------------*/
    # (2) Pressing Agent
    # ---------------------------------------------------------*/

    def press_action_rules(self, press_job=(None, None)):
        self.check_press_status()

        if press_job is None or press_job[0] is None:
            # No-op explicit log
            self.press_actions_per_timestep.append((0, None))
            return

        press_id, material_id = press_job
        if press_id == 0:
            self.press_actions_per_timestep.append((0, None))
            return

        material = self.material_names[material_id] if material_id < len(self.material_names) else "E"
        self.use_press(press_id, material, action_tuple=(press_id, material_id))

    def check_press_status(self):
        """Update press state and check timer -> produce bales when a press finishes."""
        for i in [1, 2]:  # Check both presses independently
            if self.press_state[f"press_{i}"] > 0:
                self.press_state[f"press_{i}"] -= 1  # Decrement timer

                if self.press_state[f"press_{i}"] == 0:
                    material = self.press_state[f"material_{i}"]
                    n = self.press_state[f"n_{i}"]
                    q = self.press_state[f"q_{i}"]

                    self.press_bale(material, n, q)

                    # Fix: Reset press correctly after job is done
                    self.press_state[f"press_{i}"] = 0
                    self.press_state[f"material_{i}"] = 0
                    self.press_state[f"n_{i}"] = 0
                    self.press_state[f"q_{i}"] = 0

    def press_bale(self, material, n, q):
        """Press a bale for the given container."""
        q = int(q * 100)  # Store quality as an integer
        bales = self.bale_count[material]

        full_bales = n // self.bale_standard_size
        remaining_material = n % self.bale_standard_size

        # Fix: Differentiate between full bales and remaining material
        for _ in range(full_bales):
            bales.append((self.bale_standard_size, q))

        if remaining_material > 0:
            # If remainder is more than half a standard bale, create a new bale
            if remaining_material > self.bale_standard_size * self.bale_remainder_threshold:
                bales.append((remaining_material, q))
            else:
                # Otherwise, add the remainder to the last bale if it exists
                if bales:
                    last_bale_size, last_bale_quality = bales[-1]
                    bales[-1] = (last_bale_size + remaining_material, last_bale_quality)
                else:
                    # If no bales exist, create a new one with the remainder
                    bales.append((remaining_material, q))

        self.bale_count[material] = bales
        return self.bale_count

    def check_container_level(self):
        """
        Rule-based check to find the best container to press.
        It selects the container with the highest fill level (A–D: true+false, E: true only)
        as soon as a press is free. No threshold other than > 0.
        Returns: (press_id, material_idx) or (None, None).
        """
        # Find an available press
        free_press = None
        if self.press_state["press_1"] == 0:
            free_press = 1
        elif self.press_state["press_2"] == 0:
            free_press = 2
        if free_press is None:
            return None, None

        # Determine the best container to press
        best_idx, best_level = None, 0
        # Check materials A-D
        for i, mat in enumerate(self.material_names):
            lvl = self.container_materials[mat] + self.container_materials.get(f"{mat}_False", 0)
            if lvl > best_level:
                best_level, best_idx = lvl, i

        # Check container E (waste)
        lvl_e = self.container_materials["E"]
        if lvl_e > best_level:
            best_level, best_idx = lvl_e, 4  # Index 4 corresponds to E

        if best_level > 0:
            return free_press, best_idx
        return None, None

    def use_press(self, press, material, action_tuple):

        # 1) Check if the press is busy
        if self.press_state[f"press_{press}"] > 0:
            self.press_penalty_flag = 1  # Mark penalty for reward calculation

            # This should not happen with proper sanitization, but log as fallback
            wrong_action_code = 111 if press == 1 else 222
            self.press_actions_per_timestep.append((wrong_action_code, material))

            # Stop further processing for this action
            return

        # 2) Log the valid action
        self.press_actions_per_timestep.append(action_tuple)

        # Calculate total material (true + false)
        if material in self.material_names:
            total_material = self.container_materials[material] + self.container_materials.get(f"{material}_False", 0)
        else:
            total_material = self.container_materials.get(material, 0)

        # Store last press info for reward
        self._last_press_started = True
        self._last_press_amount = total_material
        self._last_press_start_level = total_material

        # Calculate quality
        if material in self.material_names:
            true_material = self.container_materials[material]
            false_material = self.container_materials[f"{material}_False"]
            total = true_material + false_material
            quality = round(true_material / total, 2) if total > 0 else 0
        else:
            total = total_material
            quality = 0

        # Empty the container
        self.container_materials[material] = 0
        if material in self.material_names:
            self.container_materials[f"{material}_False"] = 0

        # Set press time using predefined values
        press_time = self.press_times.get(press, 0)  # Default to 0 if press ID is invalid
        self.press_state[f"press_{press}"] = press_time
        self.press_state[f"material_{press}"] = material
        self.press_state[f"n_{press}"] = total_material
        self.press_state[f"q_{press}"] = quality

    def get_container_purity(self):
        """
        Calculates the purity for each active container (A-D) as the ratio:
        true_material / (true_material + false_material).

        If a container is empty, its purity is defined as its quality threshold.
        Returns a dictionary of purities and the global average purity.
        """
        container_purities = {}
        for mat in self.material_names:
            true_val = self.container_materials.get(mat, 0)
            false_val = self.container_materials.get(f"{mat}_False", 0)
            total = true_val + false_val
            if total > 0:
                purity = true_val / total
            else:
                # If container is empty, its purity is considered to be at the threshold
                purity = self.quality_thresholds[mat]
            container_purities[mat] = round(purity, 2)
        global_purity = sum(container_purities.values()) / len(container_purities) if container_purities else 0
        return container_purities, round(global_purity, 2)

    def set_container_purity(self, material, purity):
        """For analysis: set purity for a given container."""
        total_volume = self.bale_standard_size
        correct_amount = purity * total_volume
        self.container_materials[material] = correct_amount

    # ---------------------------------------------------------*/
    # Pressing Helper Functions
    # ---------------------------------------------------------*/

    # --- Pressing helpers (shared) ---
    def press_discrete_to_action(self, action: int):
        if action == 0:
            return [0, None]
        press_id = 1 if action <= 5 else 2
        mat_id = (action - 1) % 5  # 0..4=A..E
        return [press_id, mat_id]

    def validate_press_action(self, press_id, mat_id):
        """
        Validates if a press action is valid.
        Returns True if valid, False if invalid.
        """
        if press_id == 0 or press_id is None:
            return True  # No-op is always valid
        
        # Check if press is available
        if press_id in [1, 2] and self.press_state[f"press_{press_id}"] > 0:
            return False  # Press is busy
        
        # Check if container has enough material
        if mat_id is not None:
            if mat_id < 4:  # Materials A-D
                material = self.material_names[mat_id]
                total_material = self.container_materials[material] + self.container_materials.get(f"{material}_False", 0)
            elif mat_id == 4:  # Material E
                total_material = self.container_materials["E"]
            else:
                return False  # Invalid material index
            
            if total_material < self.bale_standard_size:
                return False  # Not enough material
        
        return True

    def sanitize_press_action(self, action_discrete):
        """
        Sanitizes a discrete press action. If the action is invalid, 
        returns action 0 (no-op) and the invalid action info for logging.
        The calling code is responsible for logging.
        """
        if action_discrete == 0:
            return 0, (0, None), None  # Valid no-op, no invalid action info
        
        press_id, mat_id = self.press_discrete_to_action(action_discrete)
        
        if not self.validate_press_action(press_id, mat_id):
            # Prepare invalid action info for logging, but don't log here
            material_name = self.material_names[mat_id] if mat_id is not None and mat_id < 4 else ("E" if mat_id == 4 else "?")
            
            if press_id == 1:
                invalid_info = (111, material_name)  # Invalid Press 1
            elif press_id == 2:
                invalid_info = (222, material_name)  # Invalid Press 2
            else:
                invalid_info = (999, material_name)  # Other invalid action
            
            return 0, (0, None), invalid_info  # Return no-op action and invalid info
        
        return action_discrete, (press_id, mat_id), None  # Valid action, no invalid info

    def press_action_to_discrete(self, press_id: int, mat_id: int):
        if press_id == 0:
            return 0
        return (press_id - 1) * 5 + mat_id + 1

    def press_action_masks(self):
        mask = [False] * 11
        mask[0] = True
        p1_ready = self.press_state["press_1"] == 0
        p2_ready = self.press_state["press_2"] == 0
        for i in range(5):  # 0=A..4=E
            if i < 4:
                m = self.material_names[i]
                lvl = self.container_materials[m] + self.container_materials.get(f"{m}_False", 0)
            else:
                lvl = self.container_materials["E"]
            if lvl >= self.bale_standard_size:
                if p1_ready:
                    mask[1 + i] = True
                if p2_ready:
                    mask[6 + i] = True
        return np.array(mask, dtype=bool)

    def monolith_action_masks(self):
        """
        Generates a combined action mask for the monolithic agent's flattened action space.
        The action space is a Discrete(22) space, where:
        - Actions 0-10 correspond to sorting_mode=0 and pressing_actions 0-10.
        - Actions 11-21 correspond to sorting_mode=1 and pressing_actions 0-10.
        The sorting action part is always valid, so the mask only depends on the pressing part.
        """
        press_mask = self.press_action_masks()  # This is a (11,) boolean array
        # The monolith mask is the press mask repeated for each of the 2 sorting modes.
        monolith_mask = np.concatenate([press_mask, press_mask])
        return monolith_mask

    def detect_overflow(self):
        for m in self.material_names + ["E"]:
            total = self.container_materials[m] + self.container_materials.get(f"{m}_False", 0)
            if total > self.container_max[m]:
                return True, m
        return False, None

    # ---------------------------------------------------------*/
    # Input Handling
    # ---------------------------------------------------------*/

    def input_action_rules(self, occ=None):
        """
        Rule-based input: generate a new batch and return
        occupancy (0–100) and distribution over materials.
        If occ is provided, use it directly; otherwise, sample from the range
        defined in the config file.
        """
        if occ is None:
            min_occ = self.config['simulation']['input_occupancy_min']
            max_occ = self.config['simulation']['input_occupancy_max']
            occ = self.rng_input.integers(min_occ, max_occ + 1)
        return occ

    # ---------------------------------------------------------*/
    # Logging
    # ---------------------------------------------------------*/

    def _log_step_data(self, r_sort=0, r_press=0):
        """
        Central logging function to be called at the end of each step.
        """
        # Logging
        self.reward_data.setdefault('Reward', []).append((r_sort, r_press))
        self.reward_data.setdefault('Total', []).append(r_sort + r_press)  # useful for plotting
        self.reward_data.setdefault('Setting', []).append(self.sensor_current_setting)
        self.reward_data.setdefault('Belt_Occupancy', []).append(self.belt_occupancy)
        self.reward_data.setdefault('Belt_Proportions', []).append(self.compute_belt_proportions())

        for mat in self.material_names + ["E"]:
            true_val = self.container_materials.get(mat, 0)
            false_val = self.container_materials.get(f"{mat}_False", 0) if mat != "E" else 0
            if f"{mat}_True" not in self.reward_data:
                self.reward_data[f"{mat}_True"] = []
                self.reward_data[f"{mat}_False"] = []
            self.reward_data[f"{mat}_True"].append(true_val)
            self.reward_data[f"{mat}_False"].append(false_val)

    # ---------------------------------------------------------*/
    # Individual Reward Functions
    # ---------------------------------------------------------*/

    def _calculate_purity_score(self):
        """
        Helper to calculate a score based on container purities.
        The score is the sum of purity differences from the target thresholds.
        """
        purity_diffs = self.compute_purity_differences()
        # We only sum up positive differences to reward improvements,
        # and scale down negative ones to not overly penalize bad states.
        score = sum(diff for diff in purity_diffs)
        return score / len(purity_diffs) if purity_diffs else 0

    def calculate_sorting_reward(self):
        """
        Calculates a state-based reward for the sorting agent.
        The reward is based on the deviation of current purity levels 
        from a target threshold, penalized by operational costs.
        """
        # --- 1. Define Reward/Cost Parameters ---
        # These can be moved to the config file for easy tuning.
        purity_scaling_factor = 2.0  # Amplifies the importance of the purity score
        
        # --- 2. Calculate the Purity Score from the Current State ---
        current_purities, _ = self.get_container_purity()
        total_purity_score = 0.0

        for mat in self.material_names:
            purity = current_purities.get(mat, 0.0)
            
            # NEW LOGIC: The score is the difference from the target threshold.
            # This creates a clear positive/negative signal.
            score = purity - self.purity_threshold_theta
            total_purity_score += score

        # The raw reward from the state is the average purity score
        state_based_reward = (total_purity_score / len(self.material_names)) * purity_scaling_factor
        
        # Update last_sort_mode for the next step's comparison
        self.last_sort_mode = self.sensor_current_setting

        # --- 4. Calculate Final Reward and Scale with Tanh ---
        # The raw reward is the score from the state plus the costs
        raw_reward = state_based_reward

        # We can still use Tanh to keep the signal smooth and bounded.
        # The temperature now needs to be adjusted to the new scale of raw_reward.
        temperature = self.sorting_reward_temperature
        final_reward = np.tanh(raw_reward / temperature)

        # No longer need to track last_purities for this reward scheme
        # self.last_purities = current_purities

        return float(final_reward)


    def calculate_press_reward(self):
        """
        Calculates an advanced, balanced reward for pressing operations.
        This version uses a linear bonus based on the total container fill level.
        """
        # --- 1. State-based penalty: Check for overflow (highest priority) ---
        max_penalty = 0
        cfg = self.config['rewards']['pressing']

        for mat in self.material_names + ["E"]:
            level = self.container_materials.get(mat, 0) + self.container_materials.get(f"{mat}_False", 0)
            container_max = self.container_max.get(mat, self.container_global_max)
            if container_max == 0: continue
            
            fill_ratio = level / container_max

            if fill_ratio > 1.0:
                return cfg['overflow_penalty_catastrophic']
            elif fill_ratio > 0.95:
                max_penalty = min(max_penalty, cfg['overflow_penalty_severe'])
            elif fill_ratio > 0.90:
                max_penalty = min(max_penalty, cfg['overflow_penalty_mild'])

        if max_penalty < 0:
            return float(max_penalty)

        # --- 2. NEW: Continuous Linear Fill Level Reward ---
        # This reward encourages keeping containers full.
        state_reward = 0.0
        max_state_reward = self.press_reward_max_state_reward  # Max reward when all containers are full

        total_current_level = 0
        total_max_capacity = 0

        for mat in self.material_names + ["E"]:
            level = self.container_materials.get(mat, 0) + self.container_materials.get(f"{mat}_False", 0)
            container_max = self.container_max.get(mat, self.container_global_max)
            
            total_current_level += level
            total_max_capacity += container_max

        if total_max_capacity > 0:
            # Linearly scale the reward based on the overall system fill ratio
            overall_fill_ratio = total_current_level / total_max_capacity
            state_reward = overall_fill_ratio * max_state_reward

        # --- 3. Action-based reward: Only if a press was just started ---
        action_reward = 0.0
        if self._last_press_started:
            amount = self._last_press_amount
            num_bales = amount // self.bale_standard_size
            
            # A) Efficiency Component: The triangular wave shape
            bale_efficiency_factor = cfg.get('bale_efficiency_factor', 0.5)
            rem = amount % self.bale_standard_size
            dist_from_multiple = min(rem, self.bale_standard_size - rem)
            efficiency_component = (1.0 - 4.0 * (dist_from_multiple / self.bale_standard_size)) * bale_efficiency_factor
            
            # B) Distributed Multi-Bale Bonus Component
            target_peaks = np.array([0.0, 1/3, 2/3, 1.0])
            efficiency_peak = bale_efficiency_factor
            bonus_index = min(int(num_bales), len(target_peaks) - 1)
            target_peak = target_peaks[bonus_index]
            multi_bale_bonus = target_peak - efficiency_peak
            
            action_reward = efficiency_component + multi_bale_bonus
            
            # Reset flags
            self._last_press_started = False
            self._last_press_amount = 0
        
        # --- 4. Combine rewards and clip ---
        final_reward = state_reward + action_reward
        
        return float(np.clip(final_reward, -1.0, 1.0))


# -------------------------Notes-----------------------------------------------*\

# -----------------------------------------------------------------------------*/
