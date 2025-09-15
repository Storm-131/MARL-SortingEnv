# ---------------------------------------------------------*\
# Title: Environment (Pressing)
# ---------------------------------------------------------*/

# src/envs_train/env_pressing_agent.py

import numpy as np
from gymnasium import spaces
from src.envs_train.env_super import Env_Super


class Env_2_Pressing(Env_Super):
    """
    Subclass of EnvSuper for the Pressing Agent only.
    - Holds references to Sorting & Pressing agents.
    - Defines obs/action spaces, step logic, get_obs & reward.
    """

    def __init__(self, max_steps: int = 50, seed: int = None,
                 noise_sorting: float = 0.05, balesize: int = 200, simulation=False):

        # 1) init Super (state, dynamics, rule‐based defaults, etc.)
        super().__init__(max_steps=max_steps, seed=seed, noise_sorting=noise_sorting,
                         balesize=balesize, simulation=simulation)

        self.name = "press"

        # 2) placeholders for trained agents
        self.sort_agent = None

        # 3) define obs & action spaces
        self._initialize_spaces()

        self.press_action = (None, None)

    # ---------------------------------------------------------*/
    # Agent assignment
    # ---------------------------------------------------------*/
    def set_agents(self, sort_agent=None):
        self.sort_agent = sort_agent

    # ---------------------------------------------------------*/
    # Spaces
    # ---------------------------------------------------------*/
    def _initialize_spaces(self):
        """
        Defines the observation and action spaces for the pressing environment.

        Observation Space:
        - 5 container volumes (A–E)
        - 5 normalized fill ratios (A–E)
        - 4 material amounts in sorting stage (A–D)
        - 2 press timers

        Action Space (Discrete(11)):
        - 0: Do nothing
        - 1–5: Press 1 with A–E
        - 6–10: Press 2 with A–E
        """
        low = np.zeros(16, dtype=np.float32)
        high = np.ones(16, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Discrete(11)

    def action_masks(self):
        return super().press_action_masks()

    # ---------------------------------------------------------*/
    # Reset
    # ---------------------------------------------------------*/

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return super().get_press_obs(), {}

    # ---------------------------------------------------------*/
    # Observation
    # ---------------------------------------------------------*/

    def get_obs(self):
        return super().get_press_obs()

    # ---------------------------------------------------------*/
    # Step
    # ---------------------------------------------------------*/

    def step(self, action, use_action_masking=True, check_overflow=False):
        """
        Step function for the training environment. It assumes the provided
        'action' comes from the learning algorithm (e.g., PPO) and is already
        valid according to the action mask.
        """

        # --- Input decision (rule-based only) ---
        occ = super().input_action_rules()

        # --- Material flow ---
        super().update_environment(batchsize=occ)

        # ---------------------------------------------------------*/
        # --- 1. Sorting Agent ---
        # (This part is independent and uses its own logic)
        # ---------------------------------------------------------*/

        if self.sort_agent is not None:
            # Use the pre-trained sorting agent if available
            sort_obs = super().get_sort_obs()
            sort_mode, _ = self.sort_agent.predict(sort_obs, deterministic=True)
        else:
            # Fallback to rule-based sorting if no agent is provided
            sort_mode = super().sorting_rules()

        super().set_multisensor_mode(sort_mode)
        super().update_accuracy()
        super().sort_material()

        # ---------------------------------------------------------*/
        # --- 2. Pressing Agent Action ---
        # (Simplified to only execute the provided action)
        # ---------------------------------------------------------*/

        # The 'action' is provided by the RL algorithm. When action masking is disabled,
        # we need to sanitize the action to handle invalid choices.
        chosen_action = int(action)
        
        if not use_action_masking:
            sanitized_action, press_action_tuple, invalid_info = super().sanitize_press_action(chosen_action)
            # Log invalid action if detected
            if invalid_info is not None:
                self.press_actions_per_timestep.append(invalid_info)
        else:
            # With masking, action should be valid, convert directly
            press_action_tuple = super().press_discrete_to_action(chosen_action)
            press_action_tuple = (press_action_tuple[0], press_action_tuple[1])

        # Execute the press action (sanitized action will be (0, None) if invalid)
        super().press_action_rules(press_action_tuple if press_action_tuple[0] != 0 else (None, None))

        # ---------------------------------------------------------*/
        # --- 3. Logging and Return ---
        # ---------------------------------------------------------*/

        # --- Overflow check ---
        if check_overflow:
            overflow, mat = super().detect_overflow()
            if overflow:
                # On overflow, the episode terminates with a large penalty.
                reward = self.overflow_penalty
                info = {"overflow": True, "overflow_material": mat, "action": chosen_action}
                self.current_step += 1
                self._log_step_data(r_sort=0, r_press=reward)
                return self.get_obs(), float(reward), True, False, info

        # --- Calculate reward and check termination ---
        reward = self.calculate_press_reward()
        obs_next = self.get_obs()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        # --- Log and return ---
        self._log_step_data(r_sort=0, r_press=reward)
        info = {"action": chosen_action}

        return obs_next, float(reward), terminated, False, info

# -------------------------Notes-----------------------------------------------*\
# - This environment is a specialization of Env_Super for Pressing tasks.
# -----------------------------------------------------------------------------*/
