# MARL-SortingEnv: A Multi-Agent RL Benchmark for Sequential Industrial Control

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code Style: Autopep](https://img.shields.io/badge/code%20style-autopep8-lightgrey)](https://pypi.org/project/autopep8/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependency Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)]()

This repository provides an industry-inspired benchmark environment for **multi-stage material handling**, designed to explore the trade-offs between modular and monolithic control architectures in Multi-Agent Reinforcement Learning (MARL). The simulation combines tasks from two existing benchmarks, **SortingEnv** and **ContainerGym**, into a sequential recycling scenario involving sorting and pressing operations. It serves as a testbed for developing practical and robust MARL solutions for industrial automation.

The core investigation compares two primary control strategies: a **modular architecture** with specialized agents for each sub-task and a **monolithic agent** governing the entire system. Our experiments highlight the critical role of action space constraints, showing that while modular agents excel in unconstrained environments, the performance gap narrows significantly when **action masking** is applied.

---

## 📖 Introduction

Developing robust and flexible strategies for automated decision-making is a central challenge in the context of Industry 4.0. While Reinforcement Learning (RL) has achieved impressive results in complex domains, its transition from simulation to real-world industrial systems remains a significant hurdle due to challenges in reward design, sample efficiency, and safe exploration.

This project introduces a benchmark environment that integrates two distinct process types—sorting and pressing—to facilitate research on these challenges with low computational overhead. The sorting task provides a dense, continuous feedback signal based on material purity, whereas the pressing task relies on a sparser, delayed reward tied to bale production efficiency. This setup allows us to study the impact of modular versus monolithic control in a minimal multi-agent system.

![Material Handling Process](docs/Sorting_Flowchart.svg)
*FIGURE 1: Illustration of the material handling process, where distinct agents can manage the "Sorting Machine" and "Presses" stages.*

---

## 🔍 Research Hypotheses

**Core Thesis**
> *“A sequence of specialized agents, one for each subtask, outperforms a single monolithic agent that controls the entire process end-to-end.”*

We investigate this by comparing two training paradigms:

- **Modular Training**
  - Train two separate "expert" agents sequentially:
    1. A **Sorting Agent** is trained on the `Env_1_Sorting` environment.
    2. A **Pressing Agent** is trained on the `Env_2_Pressing` environment, which uses the policy of the pre-trained sorting agent internally.
  - This setup tests a hierarchical approach where specialized agents handle sub-tasks.

- **Monolithic Training**
  - Train one master agent on the `Env_3_Monolith` environment, which has a combined observation and action space covering both sorting and pressing tasks.

---

## 📊 Key Results

Our experiments reveal that the choice between a modular and a monolithic architecture is heavily dependent on the complexity of the action space.

- **Without Action Masking**: Agents struggled to learn effective policies. The modular architecture significantly outperformed the monolithic agent, though both were inferior to a rule-based heuristic.
- **With Action Masking**: Performance improved dramatically for all RL agents. The performance gap between the modular and monolithic approaches narrowed considerably, with both achieving positive cumulative rewards.

These results highlight the decisive role of action space constraints and suggest that the advantages of specialization diminish as action complexity is reduced.

---

## 🤖 Environment Design

We build on **[Gymnasium](https://gymnasium.farama.org/)** to simulate the industrial sorting plant, which consists of three main stages:

1. **Input Generator** →
2. **Sorting Machine** →
3. **Container & Pressing Station**

The environment is designed to be highly observable through a comprehensive dashboard that visualizes the system's state, agent actions, and performance metrics in real-time. For details on observation spaces, action spaces, and reward functions, please see the **[Component Reference](docs/Component_Reference.md)**.

---

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

## 📄 Citation

This repository is based on the following research paper:

> Maus, T., Atamna, A., & Glasmachers, T. (2025). Balancing Specialization and Centralization: A Multi-Agent Reinforcement Learning Benchmark for Sequential Industrial Control.

---

## Contact 📬

For questions or feedback, feel free to [reach out to me](https://www.ini.rub.de/the_institute/people/tom-maus/).

---
