# Deep Reinforcement Learning: Reward Shaping & Kinematic Vision

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-PPO-orange)
![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-lightgrey)
![Pygame](https://img.shields.io/badge/Graphics-Pygame-yellow)

> An experimental thesis analyzing how Proximal Policy Optimization (PPO) learns to master 2D games of varying complexity. Through a comprehensive pipeline of 21 experiments across three environments, the project explores baseline AI behaviors and tests how modifying Neural Network Architecture, Reward Functions, and Observation Spaces ultimately dictates the agent's playstyle.

---

## 🎮 Gameplay Highlights

<!-- TODO: Replace the placeholder links below with your actual uploaded GIFs -->
| Snake (Egocentric) | Flappy Bird (Kinematics) | Shooter (The Mastermind) |
| :---: | :---: | :---: |
| <video src="https://github.com/user-attachments/assets/5fdeda9c-f0a5-4687-892c-72093df68323" width="250" autoplay loop muted></video> | <img src="assets/flappy_placeholder.gif" width="250"/> | <img src="assets/shooter_placeholder.gif" width="250"/> |
| *Learned to maximize score without distance guidance by relying on egocentric rays.* | *Learned to calculate perfect trajectory arcs by processing vertical velocity.* | *Learned flawless kiting and dodging by reading enemy and bullet velocity vectors.* |

---

## 🧠 Project Overview

The objective of this thesis is to empirically demonstrate that increasing a neural network's capacity does not solve fundamental bottlenecks caused by poor state representation. Through a rigorous **7-experiment pipeline** applied consistently across three games (Snake, Flappy Bird, Top-Down Shooter), this project investigates:
1. **Reward Hacking:** How survival rewards create "Camper" behaviors.
2. **Blind Rushing:** How time penalties cause suicidal behaviors when vision is limited.
3. **Kinematic Integration:** How adding velocity vectors ($V_x, V_y$) to the observation space unlocks high-level tactical behaviors like dodging and kiting.

---

## 🔬 The 7-Experiment Pipeline

For each environment, the agent progresses through the following structured experiments to isolate the effects of architecture, rewards, and vision:

| Exp | Name | Observation Space | Reward/Penalty Focus | Key Finding / Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Baseline PPO | Static (Coordinates/Rays) | Standard (Score only) | Basic competence, heavily reliant on luck in dynamic scenarios. |
| **2** | Network Scaling | Static | Standard | Proved that bigger networks (256x256) cause underfitting without better inputs. |
| **3** | The Camper | Static | **+ Survival Reward** | *Reward Hacking:* Agent avoids the main objective just to stay alive. |
| **4** | The Rusher | Static | **- Time Penalty** | *Panic:* Agent plays aggressively but dies instantly due to lack of trajectory foresight. |
| **5** | Kinematic Vision | **Advanced** (Velocity/Egocentric) | Standard | Drastic performance spike. Agent learns to dodge and predict the future. |
| **6** | The Lazy Observer | Advanced | **+ Survival Reward** | Agent sees perfectly but uses it to indefinitely avoid combat/risk (Kiting without shooting). |
| **7** | The Mastermind | Advanced | **- Time Penalty** | **Optimal AI:** The time penalty forces aggression, while the advanced vision ensures flawless tactical execution. |

---

## 📁 Repository Structure

```text
Thesis_RL_Games/
├── README.md
├── requirements.txt
├── assets/                       # Stores GIFs and TensorBoard graphs
├── Snake/
│   ├── train_exp1_no_penalty.py
│   ├── ...
│   └── eval_snake.py             # Interactive control room to test models
├── Flappy_Bird/
│   ├── train_exp1_basic.py
│   ├── ...
│   └── eval_flappy.py
└── Shooter/
    ├── train_exp1_basic.py
    ├── ...
    └── eval_shooter.py
