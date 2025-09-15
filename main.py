# ---------------------------------------------------------*/
# Title: MARL - Environment (Main)
# ---------------------------------------------------------*/
import os

# Environment & Parameters
from src.envs_train.env_1_sort import Env_1_Sorting
from src.envs_train.env_2_press import Env_2_Pressing
from src.envs_train.env_monolith import Env_3_Monolith

# RL: Trainer / Tester
from src.testing import test_env
from src.training import RL_Trainer

# Utils
from utils.plot_env_analysis import run_env_analysis

# Benchmark
from utils.benchmark_models import run_model_benchmark, load_ppo_model
from datetime import datetime

# ---------------------------------------------------------*/
# Parameters
# ---------------------------------------------------------*/

# 1. Choose Mode
# ---------------------------------------------------------*/
TAG = f"Gold_{datetime.now().strftime('%d-%m-%Y_%H-%M')}"

ENV_ANALYSIS = 1     # Analyze the Environment
TRAIN_AND_BENCHMARK = 1  # Train and Benchmark all agents sequentially WITH action masking
TRAIN_WITHOUT_MASKING = 1  # Same as above, but WITHOUT action masking

SIMULATION = 0       # For Simulation Mode (Interactive)
VIDEO = 0            # Record Video, for Test and Load Mode

BENCHMARK_MODEL = 0  # Benchmark Models
BENCHMARK_TEST = 0   # Benchmark Tests

# 2. Environment Parameters
# ---------------------------------------------------------*/
NOISE_SORTING = 0.0
BALESIZE = 200

# 3. RL-Parameter
# ---------------------------------------------------------*/
MODEL = ["PPO"]
TIMESTEPS = 100_000  # single timestep value for both modular and monolith training
STEPS_TRAIN = 200
STEPS_TEST = 200
SEED = 42
BENCH_SEEDS = 10

SAVE = 1
DIR = "./img/figures/"


# ---------------------------------------------------------*/
# Run Environment Simulation
# ---------------------------------------------------------*/


def run_sim(
    TRAIN_AND_BENCHMARK=TRAIN_AND_BENCHMARK,
    TRAIN_WITHOUT_MASKING=TRAIN_WITHOUT_MASKING,
    NOISE_SORTING=NOISE_SORTING,
    TIMESTEPS=TIMESTEPS,
    STEPS_TRAIN=STEPS_TRAIN,
    STEPS_TEST=STEPS_TEST,
    SEED=SEED,
    BENCH_SEEDS=BENCH_SEEDS,
    MODEL=MODEL,
    TAG=TAG
):

    print("\n--------------------------------")
    print("Starting Simulation... 🚀")
    print("--------------------------------")

    # ---------------------------------------------------------*/
    # Basic Functionality
    # ---------------------------------------------------------*/

    if ENV_ANALYSIS:
        ENV = "Monolith"  # Use Monolith env to see both reward components

        env = create_environment(env_type=ENV, max_steps=STEPS_TEST, seed=SEED, log=True)

        print(f"\n--- Running Environment Analysis ({ENV}) ---")
        
        print("\n--- Running Environment Analysis (Masked Actions) ---")
        test_env(env=env, tag=TAG, save=SAVE,
                 title=f"(Random Run - Masking)", steps=STEPS_TEST, dir=DIR, seed=SEED, mode="random")
        env.reset(seed=SEED)
        test_env(env=env, tag=TAG, save=SAVE, mode="rule_based",
                 title=f"(Rule Based Agent - Masking)", steps=STEPS_TEST, dir=DIR, seed=SEED)
        
        print("\n--- Running Environment Analysis (No Masking) ---")
        env.reset(seed=SEED)
        test_env(env=env, tag=TAG, save=SAVE,
                 title=f"(Random Run - No Masking)", steps=STEPS_TEST, dir=DIR, seed=SEED, mode="random",
                 use_action_masking=False)
        env.reset(seed=SEED)
        test_env(env=env, tag=TAG, save=SAVE, mode="rule_based",
                 title=f"(Rule Based Agent - No Masking)", steps=STEPS_TEST, dir=DIR, seed=SEED, use_action_masking=False)

    if TRAIN_AND_BENCHMARK:
        print("\n--- Running Training & Benchmark with Action Masking ---")
        run_training_flow(
            use_action_masking=True,
            timesteps=TIMESTEPS,
            seed=SEED,
            tag=f"{TAG}_Masked",
            bench_seeds=BENCH_SEEDS,
            steps_test=STEPS_TEST
        )

    if TRAIN_WITHOUT_MASKING:
        print("\n--- Running Training & Benchmark WITHOUT Action Masking ---")
        run_training_flow(
            use_action_masking=False,
            timesteps=TIMESTEPS,
            seed=SEED,
            tag=f"{TAG}_NoMask",
            bench_seeds=BENCH_SEEDS,
            steps_test=STEPS_TEST
        )

    print("\n--------------------------------")
    print("Simulation Completed. 🌵")
    print("--------------------------------")


# ---------------------------------------------------------*/
# Run Training and Benchmark Flow
# ---------------------------------------------------------*/
def run_training_flow(use_action_masking, timesteps, seed, tag, bench_seeds, steps_test):
    """
    Encapsulates the full training and benchmarking pipeline.
    Can be run with or without action masking.
    """
    trained_agents = {}

    # --- 1. Train Sorting Agent ---
    # Sorting agent does not use masking, so it's trained the same way in both modes.
    print("\n[1/3] Training Sorting Agent...")
    trained_agents["PPO_Sort"] = train_agent(
        env_type="Sorting",
        total_timesteps=timesteps,
        seed=seed,
        tag=tag,
        use_action_masking=use_action_masking  # Pass flag, though it won't be used here
    )

    # --- 2. Train Pressing Agent (with pre-trained Sorting Agent) ---
    print("\n[2/3] Training Pressing Agent...")
    trained_agents["PPO_Press"] = train_agent(
        env_type="Pressing",
        total_timesteps=timesteps,
        seed=seed,
        prev_agents={"sort_agent": trained_agents["PPO_Sort"]},
        tag=tag,
        use_action_masking=use_action_masking
    )

    # --- 3. Train Monolith Agent ---
    print("\n[3/3] Training Monolith Agent...")
    trained_agents["PPO_Mono"] = train_agent(
        env_type="Monolith",
        total_timesteps=timesteps,
        seed=seed,
        tag=tag,
        use_action_masking=use_action_masking
    )

    # --- 4. Run Final Model Benchmark ---
    print("\n--- Running Final Model Benchmark ---")
    run_model_benchmark(
        env_creator=create_environment,
        trained_agents=trained_agents,
        num_seeds=bench_seeds,
        steps_test=steps_test,
        use_action_masking=use_action_masking,
        tag=tag
    )


# ---------------------------------------------------------*/
# Helper Functions
# ---------------------------------------------------------*/

def create_environment(env_type, max_steps=STEPS_TEST, seed=SEED, SIMULATION=False, log=True):

    # print(f"Creating Environment: type={env_type}")

    if log:
        log_dir = "./log/"
        os.makedirs(log_dir, exist_ok=True)

    if env_type == "Sorting":
        env = Env_1_Sorting(max_steps=max_steps, seed=seed, noise_sorting=NOISE_SORTING,
                            balesize=BALESIZE, simulation=SIMULATION)
    elif env_type == "Pressing":
        env = Env_2_Pressing(max_steps=max_steps, seed=seed, noise_sorting=NOISE_SORTING,
                             balesize=BALESIZE, simulation=SIMULATION)
    elif env_type == "Monolith":
        env = Env_3_Monolith(max_steps=max_steps, seed=seed, noise_sorting=NOISE_SORTING,
                             balesize=BALESIZE, simulation=SIMULATION)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    return env


def train_agent(env_type, total_timesteps, seed, use_action_masking, prev_agents=None, tag=None):
    """
    Trains an agent for a given environment class, tests it in the full
    monolithic environment, and returns the trained agent.
    """
    # --- 1. Create the specific training environment ---
    train_env = create_environment(env_type=env_type, max_steps=STEPS_TRAIN, seed=seed, log=False)

    # Assign previously trained agents if needed for the training phase (e.g., for Pressing)
    if prev_agents:
        if "sort_agent" in prev_agents:
            train_env.sort_agent = prev_agents["sort_agent"]

    # --- 2. Train the agent ---
    # RL_Trainer returns a dictionary like {"PPO": trained_model_object}
    trainer_results = RL_Trainer(
        env=train_env,
        env_class=env_type,
        model_list=MODEL,
        total_timesteps=total_timesteps,
        max_steps=STEPS_TRAIN,
        noise_sorting=NOISE_SORTING,
        tag=tag,
        seed=seed,
        use_action_masking=use_action_masking
    )
    # Extract the single trained model from the results
    trained_agent = list(trainer_results.values())[0]

    # --- Create a test environment and test the newly trained agent ---
    test_env_instance = create_environment(env_type="Monolith", max_steps=STEPS_TEST, seed=seed, log=False)
    title_suffix = "Masked" if use_action_masking else "No-Mask"

    if env_type == "Sorting":
        test_env_instance.set_agents(sort_agent=trained_agent)
        test_env(env=test_env_instance, tag=tag,
                 title=f"(Test: PPO_Sort - {title_suffix})", steps=STEPS_TEST, dir=DIR, seed=seed)

    elif env_type == "Pressing":
        test_env_instance.set_agents(sort_agent=prev_agents["sort_agent"], press_agent=trained_agent)
        test_env(env=test_env_instance, tag=tag,
                 title=f"(Test: PPO_Press - {title_suffix})", steps=STEPS_TEST, dir=DIR, seed=seed)

    elif env_type == "Monolith":
        test_env(env=test_env_instance, model=trained_agent, tag=tag,
                 title=f"(Test: PPO_Mono - {title_suffix})", steps=STEPS_TEST, dir=DIR, seed=seed, use_action_masking=use_action_masking)

    # --- 5. Return the trained agent for the next stage ---
    return trained_agent


# ---------------------------------------------------------*/
# Main Function
# ---------------------------------------------------------*/
if __name__ == "__main__":
    run_sim(TAG=TAG)


# -------------------------Notes-----------------------------------------------*\

# -----------------------------------------------------------------------------
