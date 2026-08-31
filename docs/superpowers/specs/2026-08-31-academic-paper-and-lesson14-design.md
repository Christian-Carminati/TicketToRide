# Design Specification: Academic LaTeX Research Paper, Lesson 14 & RL-Lab Integration

**Date**: 2026-08-31  
**Topic**: Academic Research Paper (LaTeX), Lesson 14 Meta-Analysis, Verified Bibliography, and RL-Lab Integration  
**Status**: Validated Design Specification  

---

## 1. Overview & Objectives

This specification defines the creation and integration of a full academic-grade scientific paper package, an interactive bilingual course module (**Lesson 14**), and comprehensive integration within the **Ticket to Ride RL-Lab Studio**.

The paper formalizes the imperfect-information graph-based board game *Ticket to Ride* as a Partially Observable Markov Decision Process (POMDP), documents all agent paradigms (from heuristic baselines to AlphaZero and Recurrent PPO), and evaluates them through an extensive empirical benchmark: a **5,130-game round-robin tournament across 19 agent configurations** and **1,000,000-step training runs**.

All bibliographic citations are strictly verified against authentic, published literature with valid DOIs/ArXiv identifiers.

---

## 2. Theoretical Framework & Problem Formulation

### 2.1 Graph-Constrained POMDP Specification
The game is formalized as a tuple $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$:
- **Graph Topology**: $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ representing the USA 1885 railway network ($|\mathcal{V}| = 36$ cities, $|\mathcal{E}| = 78$ route segments with lengths $w(e) \in \{1,\dots,6\}$ and colors $c(e) \in \{0,\dots,8\}$).
- **State Space $\mathcal{S}$**: Full underlying game configuration (train card deck distribution, private destination tickets of all players, claimed routes, hand compositions, remaining plastic trains).
- **Observation Space $\Omega$ & Emission $\mathcal{O}$**: Player $i$'s partial observation $o_i = \mathcal{O}(s, i) \in \mathbb{R}^{356}$, guaranteeing zero-information leakage (opponent hand card types and destination tickets are concealed).
- **Action Space $\mathcal{A}$ & Action Masking**: Discrete action set of size $|\mathcal{A}| = 85$ (drawing open cards, drawing from blind deck, claiming specific routes, drawing destination tickets). An action mask $M(o_i) \in \{0, 1\}^{85}$ enforces strict game rule legality at each step.
- **Reward Function $\mathcal{R}$**: Step rewards reflecting route claim points $R_{length}$, milestone progression $R_{dest}$, destination ticket completion bonuses (+points) and unfinished ticket penalties (-points).

### 2.2 Agent Paradigms Categorization
The paper explicitly describes and compares three macro-paradigms:
1. **Algorithmic & Heuristic Baselines**:
   - **Uniform Random**: Randomly samples legal actions $a \sim \text{Uniform}(\{a \mid M(a) = 1\})$.
   - **Greedy Score**: Myopically selects the legal action maximizing immediate point gain.
   - **Strategic Dijkstra**: Computes minimum-weight Steiner trees over secret destination tickets using Dijkstra's algorithm, prioritizes critical bottleneck tracks, and dynamically claims high-value routes.
2. **Model-Free Deep Reinforcement Learning**:
   - **Double-DQN**: Q-network with target network decoupling and prioritized experience replay.
   - **CleanRL Masked PPO**: Actor-Critic architecture with Generalized Advantage Estimation (GAE $\lambda=0.95$), clipped surrogate loss ($\epsilon = 0.2$), and logit masking.
   - **Recurrent PPO (LSTM)**: BPTT recurrent layer tracking belief states and history across variable-length POMDP trajectories.
   - **Multi-Agent Self-Play (PFSP)**: Prioritized Fictitious Self-Play maintaining a dynamic historical policy pool to prevent cyclical strategies.
3. **Tree Search & Opponent Modeling**:
   - **Information-Set MCTS (IS-MCTS)**: Determinization-based Monte Carlo tree search sampling hidden state configurations.
   - **Bayesian Detour MCTS**: Online opponent destination belief tracking $P(T_k \mid \mathcal{E}_{\text{opp}})$ via Dijkstra detour calculation, biasing determinization rollouts.
   - **AlphaZero Dual-Head**: Joint Policy ($\pi_\theta$) and Value ($v_\theta$) neural network paired with PUCT Monte Carlo Tree Search ($c_{puct} = 1.5$).

---

## 3. Verified Bibliography (`paper/references.bib`)

Every entry is verified against authentic published literature:
- `schulman2017proximal`: Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 (2017).
- `schulman2015high`: Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR (2016).
- `mnih2015human`: Mnih et al., *Human-level control through deep reinforcement learning*, Nature 518, 529–533 (2015).
- `vanhasselt2016deep`: Van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*, AAAI (2016).
- `hausknecht2015deep`: Hausknecht & Stone, *Deep Recurrent Q-Learning for Partially Observable MDPs*, AAAI Fall Symposium (2015).
- `silver2017mastering`: Silver et al., *Mastering the game of Go without human knowledge*, Nature 550, 354–359 (2017).
- `silver2018general`: Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*, Science 362, 1140–1144 (2018).
- `cowling2012information`: Cowling, Powley, Whitehouse, *Information Set Monte Carlo Tree Search*, IEEE Transactions on Computational Intelligence and AI in Games 4(2), 120–143 (2012).
- `sutton2018reinforcement`: Sutton & Barto, *Reinforcement Learning: An Introduction*, MIT Press (2018).
- `dijkstra1959note`: Dijkstra, *A note on two problems in connexion with graphs*, Numerische Mathematik 1, 269–271 (1959).
- `shannon1948mathematical`: Shannon, *A Mathematical Theory of Communication*, Bell System Technical Journal 27, 379–423 (1948).
- `elo1978rating`: Elo, *The Rating of Chessplayers, Past and Present*, Arco Publishing (1978).

---

## 4. Empirical Tournament & Training Data Integration

The paper embeds exact empirical data from:
1. **1M Step Training Regimes**:
   - CleanRL PPO: +42.21 mean reward convergence
   - AlphaZero: Validation loss 0.034, dual-head policy cross-entropy + value MSE
   - Recurrent PPO: +36.50 mean reward convergence
   - Self-Play PPO: +30.00 mean reward convergence
   - Double-DQN: +22.40 mean reward convergence
2. **5,130-Match Round-Robin Tournament (19 Agents, 30 Games/Pair)**:
   - 🥇 Strategic Heuristic (Dijkstra): Elo **1329.6** (Win Rate **80.6%**, Mean Score 69.7)
   - 🥈 AlphaZero Dual Head (1M): Elo **1275.3** (Win Rate **67.4%**, Mean Score 35.2)
   - 🥉 AlphaZero Live Checkpoint: Elo **1263.8** (Win Rate **63.9%**, Mean Score 33.3)
   - 4. CleanRL PPO Live Checkpoint: Elo **1256.1** (Win Rate **63.1%**, Mean Score 39.5)
   - 5. Pure IS-MCTS (40 Sims): Elo **1254.5** (Win Rate **63.1%**, Mean Score 49.2)
   - 6. PPO Baseline (1M): Elo **1253.7** (Win Rate **62.6%**, Mean Score 36.5)
   - 8. Bayesian MCTS: Elo **1239.5** (Win Rate **59.8%**, Mean Score 34.7)
   - 10. Recurrent PPO LSTM: Elo **1222.8** (Win Rate **54.4%**, Mean Score 36.5)
   - 12. Self-Play PPO PFSP: Elo **1200.8** (Win Rate **49.1%**, Mean Score 30.0)
   - 15. Double-DQN: Elo **1150.8** (Win Rate **38.9%**, Mean Score 21.6)
   - 18. Uniform Random: Elo **1049.4** (Win Rate **13.5%**, Mean Score -45.1)

---

## 5. Artifact Directory & File Structure

```
TicketToRide/
├── paper/
│   ├── main_paper_en.tex            # Full academic publication paper (English)
│   ├── capitolo_tesi_ita.tex         # Master's thesis chapter (Italian)
│   ├── references.bib               # Verified BibTeX citations
│   ├── tables/
│   │   ├── table1_tournament_elo.tex
│   │   ├── table2_training_hyperparameters.tex
│   │   └── table3_computational_profile.tex
│   └── figures/
│       ├── fig1_elo_tournament_matrix.png
│       ├── fig2_bayesian_entropy_convergence.png
│       ├── fig3_ablation_determinization.png
│       ├── fig4_deception_noise_robustness.png
│       └── fig5_compute_vs_elo_frontier.png
├── docs/course/
│   ├── index.html                   # Updated with Lesson 14 capstone module
│   ├── course.js                    # Updated navigation & quiz listeners
│   └── lesson_14_tournament_paper.html # New bilingual interactive lesson
└── src/api/
    └── report_service.py            # Enhanced to index paper and LaTeX reports
```

---

## 6. Implementation & Integration Plan

1. **Step 1: Standalone LaTeX Package Creation**:
   - Write `paper/references.bib` with verified references.
   - Write `paper/tables/table1_tournament_elo.tex`, `table2_training_hyperparameters.tex`, `table3_computational_profile.tex`.
   - Copy figures from `results/thesis/figures/` into `paper/figures/`.
   - Write `paper/main_paper_en.tex` (English academic paper) and `paper/capitolo_tesi_ita.tex` (Italian thesis chapter).
2. **Step 2: Interactive Lesson 14 Creation**:
   - Create `docs/course/lesson_14_tournament_paper.html` with synchronous ITA/ENG language toggling, MathJax equations, interactive results tables, LaTeX source code viewer, and interactive self-assessment quiz.
   - Update `docs/course/index.html` hub and `docs/course/course.js` navigation.
3. **Step 3: RL-Lab Studio Dispatch & Report Integration**:
   - Update `src/api/report_service.py` to index `.tex` files in `paper/` and markdown tournament reports.
   - Verify frontend `ReportsView` displays and handles LaTeX downloads and previews cleanly.
4. **Step 4: Verification & Validation**:
   - Run tests to confirm zero regressions in backend API and report service.
   - Validate HTML markup, MathJax equations, and language toggle functionality.
