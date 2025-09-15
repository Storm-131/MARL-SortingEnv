# ---------------------------------------------------------*/
# Title: Training RL Agents (PPO & DQN only)
# ---------------------------------------------------------*/

import os
import copy
import time
import shutil
import numpy as np
import glob

from stable_baselines3 import PPO, DQN

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env as sb3_check_env

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")

# ---------------------------------------------------------*/
# Helper Functions
# ---------------------------------------------------------*/

def find_latest_model(prefix):
    """
    Find the latest saved model file with the given prefix.
    Returns the path to the most recent model or None if not found.
    """
    models_dir = "./models"
    pattern = os.path.join(models_dir, f"{prefix}_*.zip")
    model_files = glob.glob(pattern)
    
    if not model_files:
        return None
    
    # Sort by modification time and return the most recent
    latest_file = max(model_files, key=os.path.getmtime)
    return latest_file

# ---------------------------------------------------------*/
# Train a single RL Agent (PPO / DQN)
# ---------------------------------------------------------*/

def Train_Agent(model_type, env, total_timesteps, use_action_masking, save_prefix=None, experiment=None, logpath=None):
    if env is None:
        raise ValueError("Environment must be provided")
    if save_prefix is None:
        save_prefix = model_type

    # --- Masking Function ---
    mask_fn = lambda e: e.unwrapped.action_masks() if hasattr(e.unwrapped, "action_masks") else None

    # --- Wrap the training environment ---
    is_maskable = False # Flag to determine if MaskablePPO should be used
    # Apply ActionMasker only if masking is enabled and the environment supports it
    if use_action_masking and hasattr(env.unwrapped, "name") and env.unwrapped.name in ["press", "mono"] and mask_fn(env) is not None:
        env = ActionMasker(env, mask_fn)
        is_maskable = True
        print(f"✅ Action masking enabled for '{env.unwrapped.name}' training environment.")
    else:
        print(f"☑️ Action masking is disabled for '{getattr(env.unwrapped, 'name', 'unknown')}' training environment.")
    env = Monitor(env)  # Monitor zuletzt

    check_env(env)

    # --- Wrap the evaluation environment ---
    # Temporarily remove agents before deepcopying to avoid serialization issues with PPO models
    sort_agent_ref = getattr(env.unwrapped, 'sort_agent', None)
    press_agent_ref = getattr(env.unwrapped, 'press_agent', None)
    if sort_agent_ref:
        env.unwrapped.sort_agent = None
    if press_agent_ref:
        env.unwrapped.press_agent = None

    eval_env = copy.deepcopy(env.unwrapped)  # clean unwrapped copy

    # Re-assign agents to both original and eval env
    if sort_agent_ref:
        env.unwrapped.sort_agent = sort_agent_ref
        eval_env.sort_agent = sort_agent_ref
    if press_agent_ref:
        env.unwrapped.press_agent = press_agent_ref
        eval_env.press_agent = press_agent_ref

    if use_action_masking and hasattr(eval_env, "name") and eval_env.name in ["press", "mono"] and mask_fn(eval_env) is not None:
        eval_env = ActionMasker(eval_env, mask_fn)
        env_name = getattr(eval_env, 'unwrapped', eval_env).name
        print(f"✅ Action masking enabled for '{env_name}' evaluation environment.")
    else:
        env_name = getattr(eval_env, 'unwrapped', eval_env).name
        print(f"☑️ Action masking is disabled for '{env_name}' evaluation environment.")
    eval_env = Monitor(eval_env)
    eval_env.reset(seed=99)

    # --- TensorBoard logging directory (single folder per run) ---
    tb_base = logpath
    folder_name = str(save_prefix)
    tensorboard_log = os.path.join(tb_base, folder_name)
    # Remove timestamp logic to ensure only one folder is used
    os.makedirs(tensorboard_log, exist_ok=True)
    # short sleep to avoid race conditions on some filesystems
    time.sleep(0.1)

    device = "cpu"
    print(f"Using device: {device}")

    # --- Policy ---
    policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))

    # --- Model ---
    if model_type == "PPO":
        # Use MaskablePPO for environments that have the ActionMasker wrapper
        if is_maskable:
            print("Using MaskablePPO.")
            model = MaskablePPO(
                MaskableActorCriticPolicy,
                env,
                policy_kwargs=policy_kwargs,
                verbose=0,
                tensorboard_log=tensorboard_log,
                ent_coef=0.05,
                seed=42,
                device=device
            )
        else:
            print("Using standard PPO.")
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=0,
                tensorboard_log=tensorboard_log,
                ent_coef=0.05,
                seed=42,
                device=device
            )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # --- Eval Callback ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./models/best_model/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=0
    )

    # --- Load Sorting Agent for Pressing Training ---
    # This is crucial when training agents that depend on others (e.g., Pressing)
    if hasattr(env.unwrapped, "name") and env.unwrapped.name == "press":
        # Always try to load the latest saved sorting model from file
        sorting_model_path = find_latest_model("PPO_Sorting") # CORRECTED PREFIX
        if sorting_model_path:
            print(f"📂 Pressing Agent training: Loading pre-trained Sorting model from: {sorting_model_path}")
            try:
                # Create a temporary environment for loading the sorting model
                from src.envs_train.env_1_sort import Env_1_Sorting
                temp_sort_env = Env_1_Sorting()
                
                # The sorting agent is never maskable, so we always load it as a standard PPO model.
                sorting_agent = PPO.load(sorting_model_path, env=temp_sort_env)
                
                # Assign the loaded agent to the pressing environment
                env.unwrapped.set_agents(sort_agent=sorting_agent)
                if hasattr(eval_env, 'unwrapped'):
                    eval_env.unwrapped.set_agents(sort_agent=sorting_agent)
                else:
                    eval_env.set_agents(sort_agent=sorting_agent)
                
                print("✅ Successfully loaded and assigned pre-trained Sorting Agent for Pressing training.")
            except Exception as e:
                print(f"❌ Failed to load Sorting model: {e}")
                print("⚠️ WARNING: Training Pressing Agent without pre-trained Sorting Agent!")
        else:
            print("⚠️ WARNING: No saved Sorting model found! Training Pressing Agent without pre-trained Sorting Agent!")


    # --- Train ---
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
    dur = time.time() - start_time
    print(f"✅ Training done in {dur//60:.0f} m {dur%60:.0f} s")

    # --- Evaluate ---
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Final Performance: {mean_r:.2f} ± {std_r:.2f}")

    # --- Optional: Best Checkpoint ---
    best_path = "./models/best_model/best_model.zip"
    if os.path.exists(best_path):
        print(f"📂 Loading best checkpoint model from: {best_path}")
        best_model = model.__class__.load(best_path, env=env)
        shutil.rmtree("./models/best_model/", ignore_errors=True)
        mean_best, std_best = evaluate_policy(best_model, env, n_eval_episodes=10)
        print(f"Best-Checkpoint: {mean_best:.2f} ± {std_best:.2f}")
        if mean_best > mean_r:
            model = best_model
            print("🏅 Using best checkpoint for saving.")

    # --- Save ---
    save_model(model, prefix=save_prefix, timesteps=total_timesteps)
    return model

# ---------------------------------------------------------*/
# High-level Trainer – loops over a list of algorithms
# ---------------------------------------------------------*/


def RL_Trainer(
    env,
    env_class,
    model_list,
    max_steps,
    total_timesteps,
    noise_sorting,
    tag,
    seed,
    use_action_masking, # Add parameter here
    test_steps=None,
    test_dir="./img/figures/",
    test_save=False,
    experiment=None,
):
    if test_steps is None:
        test_steps=max_steps

    trained={}
    # create an experiment name if none provided
    exp_name = experiment or f"{tag}_{env_class}_{int(time.time())}"
    for algo in model_list:
        if algo not in ["PPO", "DQN"]:
            print(f"⏭️  Unsupported (or removed) algo '{algo}' – skipping.")
            continue

        print("\n----------------------------------------")
        print(f"🏋🏽 Training {algo} - {env_class} ...")
        print("----------------------------------------")

        # Reset the environment with a new seed for each run
        env.reset(seed=None)

        agent=Train_Agent(
            model_type=algo,
            env=env,
            total_timesteps=total_timesteps,
            save_prefix=f"{algo}_{env_class}",
            experiment=exp_name,
            logpath=f"./log/tensorboard/{tag}",
            use_action_masking=use_action_masking, # Pass parameter
        )
        trained[algo]=agent
        shutil.rmtree("./models/best_model/", ignore_errors=True)

    return trained


# ---------------------------------------------------------*/
# Model saving helper
# ---------------------------------------------------------*/
def save_model(model, prefix, timesteps):
    models_dir="./models"
    os.makedirs(models_dir, exist_ok=True)

    fname=f"{prefix}_{timesteps}.zip"
    fpath=os.path.join(models_dir, fname)

    # move older versions to ./models/prev/
    existing=[f for f in os.listdir(models_dir) if f.startswith(prefix) and f.endswith(".zip")]
    if existing:
        prev_dir=os.path.join(models_dir, "prev")
        os.makedirs(prev_dir, exist_ok=True)
        for old in existing:
            shutil.move(os.path.join(models_dir, old), os.path.join(prev_dir, old))

    model.save(fpath)
    print(f"💾 Saved {prefix} model → {fpath}")

  
# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*/
