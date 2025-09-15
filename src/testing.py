# ---------------------------------------------------------*\
# Title: Testing MARL
# ---------------------------------------------------------*/

import numpy as np

# ---------------------------------------------------------*/
# Test the Environment with Random Actions
# ---------------------------------------------------------*/


def test_env(env=None, tag="", save=False, title="", steps=50, dir="./img/", seed=None, show=True,
             stats=True, mode="model", model=None, use_action_masking=True):
    """
    Test the environment by running a simulation for a given number of steps.

    Args:
        env: The environment instance to test.
        mode (str): The simulation mode. Can be "model", "random", or "rule_based".
                    Defaults to "model".
        model: The model to use if mode="model". For modular setups, this can be omitted
               if agents are pre-assigned to the environment.
        use_action_masking (bool): Whether to use action masks during model prediction.
    """
    if env is None:
        raise ValueError("Environment must be provided")

    obs, info = env.reset(seed=seed)
    action_sequence = []
    cumulative_reward = 0.0

    for i in range(steps):
        action = None  # Default action is None

        # Determine action based on the explicit mode parameter
        if mode == "model":
            # If a monolithic model is provided, use it to predict.
            # Otherwise, the environment's internal logic will handle modular agents.
            if model is not None:
                if use_action_masking and hasattr(env, 'action_masks'):
                    mask = env.action_masks()
                    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                else:
                    action, _ = model.predict(obs, deterministic=True)  # No mask
            # If no model is passed, we assume modular agents are set in the env,
            # and env.step() will handle it. 'action' remains None.

        # For 'random' and 'rule_based', action remains None.
        # The environment's step function will use the 'mode' to generate actions.

        # Pass both the action (if any) and the mode to the step function.
        # Also pass use_action_masking to the environment
        obs, reward, done, _, info = env.step(action=action, mode=mode, use_action_masking=use_action_masking)
        cumulative_reward += reward

        # Get chosen action from info for logging
        chosen_action = info.get("action", action)  # Fallback to the externally provided action
        action_sequence.append(chosen_action)

        if done:
            if stats:
                print(f"\n---- Testing Results - {mode} ----")
                print(f"🏁 Epoch ended after \033[1m{i + 1}\033[0m steps.")
                env.render(save=True, log_dir=dir, filename=f'{tag}_env_simulation', title=title, show=show,
                           steps_test=steps)
            else:
                env.render(save=True, log_dir=dir, filename=f'{tag}_env_simulation', title=title, show=show,
                           checksum=False, steps_test=steps)

            total_rewards = [sum(reward) for reward in env.reward_data['Reward']]
            cumulative_total_reward = np.cumsum(total_rewards)[-1]

            if stats:
                print(f"👑 Total Reward: {cumulative_total_reward:.2f}")

            break

    # Final cumulative reward calculation
    if 'Reward' in env.reward_data and env.reward_data['Reward']:
        total_rewards = [sum(reward) for reward in env.reward_data['Reward']]
        final_cumulative = np.cumsum(total_rewards)[-1]
    else:
        final_cumulative = cumulative_reward

    return final_cumulative, action_sequence

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
