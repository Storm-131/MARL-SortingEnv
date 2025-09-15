# ---------------------------------------------------------*\
# Title: Environment (Monolith)
# ---------------------------------------------------------*/

# src/envs_train/env_monolith_agent.py

import numpy as np
from gymnasium import spaces
from src.envs_train.env_super import Env_Super


class Env_3_Monolith(Env_Super):
    """
    Single-agent environment for joint Sorting + Pressing.
    - Combined observation = Sorting(13) + Pressing(16) = 29 dims.
    - Flattened action space of Discrete(22) combines:
      * Sorting: sensor mode (0 for A/C boost, 1 for B/D boost)
      * Pressing: (0 for idle; 1–5 for press1 A–E; 6–10 for press2 A–E)
    - The reward is the sum of the sorting and pressing rewards.
    """

    def __init__(self, max_steps: int = 50, seed: int = None,
                 noise_sorting: float = 0.05, balesize: int = 200, simulation=False):

        super().__init__(max_steps=max_steps, seed=seed,
                         noise_sorting=noise_sorting, balesize=balesize, simulation=simulation)

        self.name = "mono"

        # Agents for evaluation/simulation
        self.sort_agent = None
        self.press_agent = None
        self.mono_agent = None

        self._initialize_spaces()

    # ---------------------------------------------------------*/
    # Agent assignment
    # ---------------------------------------------------------*/
    def set_agents(self, sort_agent=None, press_agent=None, mono_agent=None):
        """Assign pre-trained agents for modular evaluation."""
        self.sort_agent = sort_agent
        self.press_agent = press_agent
        self.mono_agent = mono_agent

    # ---------------------------------------------------------*/
    # Spaces
    # ---------------------------------------------------------*/
    def _initialize_spaces(self):
        """
        Defines the observation and action spaces for the monolithic environment.
        """
        # Sorting observation space (13 dims)
        sort_low = np.concatenate([
            np.zeros(1),         # Belt occupancy
            np.zeros(4),         # Belt proportions
            np.zeros(4),         # Sorting accuracy
            np.full(4, -1.0)     # Purity differences
        ])
        sort_high = np.concatenate([
            np.ones(1),          # Belt occupancy
            np.ones(4),          # Belt proportions
            np.ones(4),          # Sorting accuracy
            np.ones(4)           # Purity differences
        ])

        # Pressing observation space (16 dims, all normalized 0-1)
        press_low = np.zeros(16)
        press_high = np.ones(16)

        # Combined observation space
        self.observation_space = spaces.Box(
            low=np.concatenate([sort_low, press_low]),
            high=np.concatenate([sort_high, press_high]),
            dtype=np.float32
        )

        # Flattened action space for 2 sorting modes and 11 pressing actions
        self.action_space = spaces.Discrete(2 * 11)

    def action_masks(self):
        """
        Returns the combined action mask from the superclass.
        """
        return super().monolith_action_masks()

    # ---------------------------------------------------------*/
    # Reset
    # ---------------------------------------------------------*/

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.get_obs(), {}

    # ---------------------------------------------------------*/
    # Observation
    # ---------------------------------------------------------*/
    def get_obs(self):
        """
        Returns the combined observation vector for the monolithic agent.
        """
        sort_obs = super().get_sort_obs()
        press_obs = super().get_press_obs()
        return np.concatenate([sort_obs, press_obs])

    # ---------------------------------------------------------*/
    # Reset
    # ---------------------------------------------------------*/
    def step(self, action=None, mode=None, use_action_masking=True, check_overflow=False):
        """
        Step function with a refactored, clear action selection hierarchy.
        """
        # --- Environment dynamics ---
        occ = super().input_action_rules()
        super().update_environment(batchsize=occ)

        # ---------------------------------------------------------
        # --- 1. Determine Sorting and Pressing Actions ---
        # ---------------------------------------------------------
        sort_mode = 0
        press_action_discrete = 0
        press_action_tuple = (0, None)  # Initialize press_action_tuple
        chosen_flat_action = 0

        if action is not None:
            # --- Path 1: Action from the main Monolith Agent (e.g., during training) ---
            chosen_flat_action = int(action)
            sort_mode = chosen_flat_action // 11
            press_action_discrete = chosen_flat_action % 11
            
            # When no masking is used, we need to sanitize the pressing action
            if not use_action_masking:
                press_action_discrete, press_action_tuple, invalid_info = super().sanitize_press_action(press_action_discrete)
                # Log invalid action if detected
                if invalid_info is not None:
                    self.press_actions_per_timestep.append(invalid_info)
                    # Mark that this was an invalid action so we don't execute it later
                    press_action_tuple = None  # Signal that no execution should happen
            else:
                # With masking, the action should already be valid
                press_action_tuple = super().press_discrete_to_action(press_action_discrete)
                press_action_tuple = (press_action_tuple[0], press_action_tuple[1])

        elif self.mono_agent is not None:
            # --- Path 2: Action from an internally stored Monolith Agent (e.g., during evaluation) ---
            obs = self.get_obs()
            mask = self.action_masks()
            chosen_flat_action, _ = self.mono_agent.predict(obs, deterministic=True, action_masks=mask)
            sort_mode = int(chosen_flat_action) // 11
            press_action_discrete = int(chosen_flat_action) % 11

        elif mode == 'random':
            # --- Path 3: Fully Random Mode ---
            if use_action_masking:
                # With masking, choose only from valid actions
                mask = self.action_masks()
                valid_actions = np.flatnonzero(mask)
                chosen_flat_action = np.random.choice(valid_actions) if valid_actions.size > 0 else 0
            else:
                # Without masking, choose completely random action from entire action space
                # This can generate invalid actions that will be sanitized
                chosen_flat_action = np.random.randint(0, self.action_space.n)
            sort_mode = chosen_flat_action // 11
            press_action_discrete = chosen_flat_action % 11

        elif mode == 'rule_based':
            # --- Path 4: Rule-Based Mode ---
            sort_mode = super().sorting_rules()
            
            # Rule-based agents should ALWAYS make valid decisions
            if use_action_masking:
                # Use proper rule-based pressing logic
                press_job = super().check_container_level()
                press_action_discrete = super().press_action_to_discrete(
                    press_job[0] or 0, press_job[1] or 0
                ) if press_job != (None, None) else 0
            else:
                # Even without masking, rule-based should be smart and check conditions
                press_job = super().check_container_level()
                press_action_discrete = super().press_action_to_discrete(
                    press_job[0] or 0, press_job[1] or 0
                ) if press_job != (None, None) else 0
            
            chosen_flat_action = int(sort_mode) * 11 + int(press_action_discrete)

        elif mode == 'model':
            # --- Path 5: Modular Agents (or random fallback if an agent is missing) ---

            # --- Block A: Determine Sorting Action ---
            if self.sort_agent is not None:
                sort_obs = super().get_sort_obs()
                predicted_sort_mode, _ = self.sort_agent.predict(sort_obs, deterministic=True)
                sort_mode = int(predicted_sort_mode)
            else:  # Fallback to random logic
                sort_mode = self.rng_sorting.choice([0, 1])

            # --- Block B: Determine Pressing Action ---
            if self.press_agent is not None:
                press_obs = super().get_press_obs()
                # Check if the agent is a MaskablePPO (supports action_masks) or regular PPO
                is_maskable = hasattr(self.press_agent, 'policy') and 'Maskable' in str(type(self.press_agent))
                
                if use_action_masking and is_maskable:
                    mask = super().press_action_masks()
                    predicted_press_action, _ = self.press_agent.predict(
                        press_obs, deterministic=True, action_masks=mask
                    )
                else:
                    predicted_press_action, _ = self.press_agent.predict(
                        press_obs, deterministic=True
                    )
                press_action_discrete = int(predicted_press_action)
            else:  # Fallback to random logic
                if use_action_masking:
                    mask = super().press_action_masks()
                    valid_actions = np.flatnonzero(mask)
                    press_action_discrete = self.rng_pressing.choice(valid_actions) if valid_actions.size > 0 else 0
                else:
                    press_action_discrete = self.rng_pressing.choice(11)

            chosen_flat_action = int(sort_mode) * 11 + int(press_action_discrete)

        else:
            raise ValueError(
                "Invalid action source: Provide 'action', set 'mode' to 'random', 'rule_based', or 'model', or assign a mono_agent.")

        # ---------------------------------------------------------
        # --- 2. Apply the determined Actions ---
        # ---------------------------------------------------------

        # Apply Sorting Action
        super().set_multisensor_mode(sort_mode)
        super().update_accuracy()
        super().sort_material()

        # Apply Pressing Action
        if action is not None:
            # If action came from external source, only execute if it wasn't an invalid action
            if press_action_tuple is not None and press_action_tuple[0] != 0:  # Valid press action
                super().press_action_rules(press_action_tuple)
            elif press_action_tuple is not None and press_action_tuple[0] == 0:  # Valid no-op
                super().press_action_rules((None, None))
            # If press_action_tuple is None, it was an invalid action and already logged
        else:
            # For other modes, decide based on mode and masking
            if mode == 'random' and not use_action_masking:
                # Only random mode without masking needs sanitization
                sanitized_action, press_action_tuple, invalid_info = super().sanitize_press_action(press_action_discrete)
                # Log invalid action if detected
                if invalid_info is not None:
                    self.press_actions_per_timestep.append(invalid_info)
                    # For invalid actions, don't execute press_action_rules to avoid double logging
                else:
                    # Only execute for valid actions (including valid no-op)
                    super().press_action_rules(press_action_tuple if press_action_tuple[0] != 0 else (None, None))
            else:
                # Rule-based or masked random: actions should be valid, convert directly
                press_action_tuple = super().press_discrete_to_action(press_action_discrete)
                super().press_action_rules((press_action_tuple[0], press_action_tuple[1]) if press_action_tuple[0] != 0 else (None, None))

        # ---------------------------------------------------------
        # --- 3. Calculate Reward, Log, and Return ---
        # ---------------------------------------------------------

        if check_overflow:
            overflow, mat = super().detect_overflow()
            if overflow:
                reward = self.overflow_penalty
                info = {"overflow": True, "overflow_material": mat, "action": chosen_flat_action}
                self.current_step += 1
                self._log_step_data(r_sort=reward/2, r_press=reward/2)
                return self.get_obs(), float(reward), True, False, info

        sort_reward = self.calculate_sorting_reward()
        press_reward = self.calculate_press_reward()
        reward = sort_reward + press_reward

        obs_next = self.get_obs()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        self._log_step_data(r_sort=sort_reward, r_press=press_reward)
        info = {"action": chosen_flat_action}
        return obs_next, float(reward), terminated, False, info


# -----------------------------------------------------------------------------*/
