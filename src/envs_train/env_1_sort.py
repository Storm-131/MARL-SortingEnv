# ---------------------------------------------------------*\
# Title: Environment (Sorting)
# ---------------------------------------------------------*/

# src/envs_train/env_sorting_agent.py

import numpy as np
from gymnasium import spaces
from src.envs_train.env_super import Env_Super  # your Super‐class with rule‐based defaults


class Env_1_Sorting(Env_Super):
    """
    Subclass of EnvSuper for the Sorting Agent only.
    - Holds references to trained sub-agents.
    - Defines its own obs/action spaces, step logic, get_obs & reward.
    """

    def __init__(self, max_steps: int = 50, seed: int = None,
                 noise_sorting: float = 0.05, balesize: int = 200, simulation=False):

        # 1) init Super (state, dynamics, rule‐based defaults, etc.)
        super().__init__(max_steps=max_steps, seed=seed, noise_sorting=noise_sorting,
                         balesize=balesize, simulation=simulation)

        self.name = "sort"

        # 3) define obs & action spaces
        self._initialize_spaces()

    # ---------------------------------------------------------*/
    # Agent assignment
    # ---------------------------------------------------------*/
    def set_agents(self, press_agent=None):
        """
        Store trained agents; if None, Super.rule_*() will be used.
        """
        self.press_agent = press_agent

    # ---------------------------------------------------------*/
    # Spaces
    # ---------------------------------------------------------*/
    def _initialize_spaces(self):
        """
        Defines the observation and action spaces for the sorting environment.

        Observation Space (13 continuous values):
        - 1: Belt occupancy (0-1)
        - 4: Belt material proportions for A, B, C, D (0-1)
        - 4: Sorting accuracy for A, B, C, D (0-1)
        - 4: Purity differences for containers A, B, C, D (-1 to 1)

        Action Space (Discrete(2)):
        - 0: Boost A & C
        - 1: Boost B & D
        """
        # Define low and high bounds for the 13-dimensional observation space
        low = np.concatenate([
            np.zeros(1),         # Belt occupancy
            np.zeros(4),         # Belt proportions
            np.zeros(4),         # Sorting accuracy
            np.full(4, -1.0)     # Purity differences
        ])
        high = np.concatenate([
            np.ones(1),          # Belt occupancy
            np.ones(4),          # Belt proportions
            np.ones(4),          # Sorting accuracy
            np.ones(4)           # Purity differences
        ])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def action_masks(self):
        """Simple mask for sorting environment - all actions are always valid."""
        return np.array([True, True], dtype=bool)

    # ---------------------------------------------------------*/
    # Reset
    # ---------------------------------------------------------*/
    def reset(self, seed=None):
        """Reset via Super; return initial obs."""
        super().reset(seed=seed)

        return super().get_sort_obs(), {}

    # ---------------------------------------------------------*/
    # Observation
    # ---------------------------------------------------------*/
    def get_obs(self):
        return super().get_sort_obs()

    # ---------------------------------------------------------*/
    # Step
    # ---------------------------------------------------------*/

    def step(self, action=None, use_action_masking=True, check_overflow=False):
        """
        Step function for the sorting environment with consistent structure
        to Pressing Env.
        """

        # --- Input decision (rule-based only) ---
        occ = super().input_action_rules()

        # --- Material flow ---
        super().update_environment(batchsize=occ)

        # ---------------------------------------------------------*/
        # --- 1. Sorting Agent ---
        # ---------------------------------------------------------*/

        # Determine the sorting action based on the provided parameters.
        sort_mode = action

        super().set_multisensor_mode(sort_mode)
        super().update_accuracy()
        super().sort_material()

        # ---------------------------------------------------------*/
        # --- 2. Pressing Agent ---
        # ---------------------------------------------------------*/

        # Default to random (masked) actions for the pressing part
        press_action = super().sample_masked_press_action()
        super().press_action_rules(press_job=press_action)

        # ---------------------------------------------------------*/
        # --- 3. Logging and Return ---
        # ---------------------------------------------------------*/

        # --- Overflow check ---
        if check_overflow:
            overflow, mat = super().detect_overflow()
            if overflow:
                # On overflow, the episode terminates with a large penalty.
                # The reward is assigned to the pressing component.
                reward = self.overflow_penalty
                info = {"overflow": True, "overflow_material": mat, "action": press_action}
                self.current_step += 1
                self._log_step_data(r_sort=0, r_press=reward)
                return self.get_obs(), float(reward), True, False, info

        # --- Calculate reward and check termination ---
        reward = self.calculate_sorting_reward()
        obs_next = self.get_obs()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        # --- Log and return ---
        self._log_step_data(r_sort=reward, r_press=0)    # No press reward in sort env
        info = {"action": sort_mode}

        return obs_next, float(reward), terminated, False, info


# -------------------------Notes-----------------------------------------------*\
# ...existing code...
