# Balancing Specialization and Centralization
## A Multi-Agent Reinforcement Learning Benchmark for Sequential Industrial Control

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-390/)  
[![Code Style: Autopep](https://img.shields.io/badge/code%20style-autopep8-lightgrey)](https://pypi.org/project/autopep8/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Dependency Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)]()

In the context of advancing autonomous decision-making systems within industrial processes, this study proposes a novel platform for **multi-stage material handling** by integrating two established RL benchmarks, SortingEnv and ContainerGym, into a single, serial workflow. Materials flow through three stages:

1. **Input Control**: the rate and composition of incoming material streams  
2. **Sorting**: the allocation of items into correct bins based on sensor settings  
3. **Container Management**: the monitoring and compression of full bins into bales  

Each standalone environment reflects near–industry-standard benchmarks for RL research. By chaining them, we create a **combined environment** that supports experiments in both **decentralized multi-agent training** and **hierarchical, master-agent control**.

---

## 🔍 Research Hypotheses and Contributions

**Core Thesis**  
> *“A sequence of specialized agents, one for each subtask, outperforms a single monolithic agent that controls the entire process end-to-end.”*

We will investigate:

- **Modular Training**
  - Train two separate "expert" agents sequentially:
    1. A **Sorting Agent** is trained on the `Env_1_Sorting` environment.
    2. A **Pressing Agent** is trained on the `Env_2_Pressing` environment, which uses the policy of the pre-trained sorting agent internally.
  - This setup tests a hierarchical approach where specialized agents handle sub-tasks.

- **Monolithic Training**
  - Train one master agent on the `Env_3_Monolith` environment, which has a combined observation and action space covering both sorting and pressing tasks.

- **Comparative Evaluation**  
  - **Sample efficiency**: How many training steps to reach a performance threshold?  
  - **Final policy quality**: Throughput, purity, and compression rewards.  
  - **Robustness**: Sensitivity to noise in input or sensor accuracy.  
  - **Transfer & maintenance**: How easily can one sub-agent be retrained in case of process changes?

This setup bridges **modular/hierarchical RL** with **flat end-to-end RL**, providing insights into design trade-offs in real-world industrial control.

---

## 🤖 Environment Design

We build on [Gymnasium](https://gymnasium.farama.org/) to simulate an industrial sorting plant:

1. **Input Generator** →  
2. **Sorting Conveyor** →  
3. **Container & Pressing Station**

For details on observation and action spaces, see [docs/Component_Reference.md].

![Sorting Process Flowchart](docs/Sorting_Flowchart.svg)

---

## 🏗 Folder Structure

```
.
├── docs/
│   ├── Component_Reference.md  --> Detailed description of environment components and parameters.
│   └── ...
├── img/
│   └── ...
├── log/
│   └── ...
├── src/
│   ├── envs_train/
│   │   ├── env_super.py        --> Base class for the environments.
│   │   ├── env_1_sort.py       --> Environment for the sorting agent.
│   │   ├── env_2_press.py      --> Environment for the pressing agent.
│   │   └── env_monolith.py     --> Environment for the monolithic agent.
│   ├── testing.py              --> Script for testing and evaluating trained agents.
│   └── training.py             --> Script for training the RL agents.
├── utils/
│   ├── benchmark_models.py     --> Logic for running model benchmarks.
│   ├── input_generator.py
│   ├── plot_env_analysis.py
│   ├── plotting.py
│   └── simulation.py
├── main.py                     --> Main execution script.
├── Readme.md
└── ...
```

---

## 📚 Setup

Follow these steps to set up the environment on your local machine.

```sh
# 1) Create a new conda environment with the required packages
conda env create -f environment_full.yml

# Optionally: Manually create the environment with the following command:
conda create -n marl_env python=3.11 gymnasium numpy matplotlib tqdm stable-baselines3 tensorboard pandas scipy seaborn scikit-learn ipykernel opencv tabulate sb3-contrib -c conda-forge
pip install opencv-python

# 2) Activate the environment
conda activate marl_env
```

---

## 🚀 Quickstart

To run the environment, you can use the `main.py` file. This file allows you to set the parameters for the environment and the agent.

```bash
# Set parameters in the file and run the environment
python main.py
```

---
