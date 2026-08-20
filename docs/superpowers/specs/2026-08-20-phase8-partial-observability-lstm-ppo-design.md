# Design Specification: Phase 8 — Partial Observability & Recurrent PPO (LSTM)

**Document Version:** 1.0.0  
**Date:** 2026-08-20  
**Phase:** 8 (Partial Observability & Recurrent PPO)  
**Status:** Approved for Implementation  

---

## 1. Executive Summary & Goals

In Ticket to Ride, complete game information is not directly accessible to players (Partially Observable Markov Decision Process — **POMDP**). While public information is visible (face-up card pool, board routes claimed, train counts, public score track), key private information remains hidden:
- Opponent hand card colors
- Opponent destination tickets
- Remaining draw deck sequence

A standard stateless MLP policy observes only an instantaneous snapshot $s_t$ without temporal memory. Conversely, a recurrent neural network (**LSTM**) maintains an internal hidden state $(h_t, c_t)$ across turns, allowing the agent to remember opponent card draw selections (e.g. which face-up colors were drawn in past turns), route claiming patterns, and tempo.

### Primary Deliverables
1. **Strict POMDP Information Hiding Verification**: Invariant testing guaranteeing that `ObservationV1` leaks zero hidden information (opponent cards/tickets, hidden deck order).
2. **Recurrent Actor-Critic Architecture (`RecurrentMaskedActorCritic`)**: Shared linear feature encoder + PyTorch `nn.LSTM` layer + Masked Actor (policy) & Critic (value) linear heads with CleanRL-standard orthogonal initialization.
3. **Recurrent Rollout Buffer (`RecurrentRolloutBuffer`)**: Multi-step trajectory buffer storing transitions, masks, dones, and hidden states, supporting sequence-chunk mini-batch generation for truncated Backpropagation Through Time (BPTT).
4. **Masked Recurrent PPO Trainer (`MaskedRecurrentPPOTrainer`)**: Full PPO training loop with GAE computation, clipped surrogate policy loss, value loss clipping, entropy bonus, KL early stopping, learning rate annealing, and hidden state resetting across episodic boundaries.
5. **Recurrent PPO Agent (`RecurrentPPOAgent`)**: High-level agent implementing `BaseAgent`, maintaining internal hidden state during inference across game turns, compatible with `Evaluator`, `Tournament`, and CLI.
6. **POMDP Benchmark Runner & Scientific Comparison (`POMDPBenchmarkRunner`)**: Automated benchmark comparing MLP PPO vs LSTM PPO vs scripted baselines (Random, Greedy, Strategic), exporting metrics to JSON and Markdown (`experiments/results/phase8_report.md`).
7. **Phase 8 Acceptance Test Suite (`tests/rl/test_phase8_acceptance.py`)**: End-to-end verification of all acceptance criteria.

---

## 2. Architecture & Data Flow

```
                                  ┌────────────────────────┐
                                  │  GameState (True State) │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                           ┌──────────────────────────────────────┐
                           │ ObservationV1 (POMDP Anti-Leakage)   │
                           │ - No opponent hand colors            │
                           │ - No opponent tickets                │
                           │ - No hidden deck sequence            │
                           └──────────────────┬───────────────────┘
                                              │
                                              ▼
                                ┌───────────────────────────┐
                                │ RecurrentMaskedActorCritic│
                                │ Linear Encoder -> LSTM    │
                                │ -> Masked Actor / Critic  │
                                └─────────────┬─────────────┘
                                              │
                         ┌────────────────────┴────────────────────┐
                         ▼                                         ▼
           ┌───────────────────────────┐             ┌───────────────────────────┐
           │ MaskedRecurrentPPOTrainer │             │    RecurrentPPOAgent      │
           │ - RecurrentRolloutBuffer  │             │ - Persistent (h, c) state │
           │ - Truncated BPTT / Chunks │             │ - Tournament / Evaluator  │
           │ - GAE & Clipped Loss      │             │ - Save / Load Checkpoint  │
           └───────────────────────────┘             └───────────────────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │    POMDPBenchmarkRunner       │
                              │  MLP PPO vs LSTM PPO vs Bots  │
                              │  JSON & Markdown Report       │
                              └───────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Strict POMDP Information Hiding Verification
- **Module:** `tests/environment/test_pomdp_anti_leakage.py`
- **Invariants:**
  1. **Opponent Hand Invariance**: Modifying opponent card color distribution while preserving total card count produces an identical observation vector for player 0.
  2. **Opponent Ticket Invariance**: Adding, removing, or swapping opponent destination tickets produces an identical observation vector for player 0.
  3. **Deck Permutation Invariance**: Shuffling the unseen train card deck or destination ticket deck produces an identical observation vector for player 0.
  4. **Visible Card Sensitivity**: Modifying face-up visible cards changes only the visible card slice in the observation vector.

### 3.2 Recurrent Neural Network (`RecurrentMaskedActorCritic`)
- **Module:** `src/rl/lstm_ppo.py`
- **Specification:**
  - `input_dim`: Observation vector dimension (e.g. 100 for mini board, 323 for USA board).
  - `action_dim`: Discrete action space dimension (e.g. 56 for mini, 160 for USA).
  - `hidden_dim`: Linear encoder hidden dimension (default: 128).
  - `lstm_hidden_dim`: LSTM recurrent cell dimension (default: 128).
- **Sub-modules:**
  - `encoder`: `nn.Sequential(layer_init(nn.Linear(input_dim, hidden_dim), sqrt(2)), nn.Tanh())`
  - `lstm`: `nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)` with orthogonal initialization on `weight_ih_l0` and `weight_hh_l0`, biases initialized to 0.0.
  - `actor`: `layer_init(nn.Linear(lstm_hidden_dim, action_dim), 0.01)`
  - `critic`: `layer_init(nn.Linear(lstm_hidden_dim, 1), 1.0)`
- **Key Methods:**
  - `forward(obs_seq, hidden_state)`:
    - Input: `obs_seq` `(batch_size, seq_len, input_dim)`, `hidden_state` `(h, c)` of shape `(1, batch_size, lstm_hidden_dim)`.
    - Output: `logits (batch_size, seq_len, action_dim)`, `values (batch_size, seq_len, 1)`, `new_hidden`.
  - `get_action_and_value(obs, hidden_state, action_mask=None, action=None, deterministic=False)`:
    - Supports both single step `(batch_size=1, seq_len=1)` during inference/rollouts and batch sequence during training.
    - Masking: applies `-1e8` to invalid action logits before `Categorical` distribution calculation.
    - Returns `(action, log_prob, entropy, value, new_hidden)`.

### 3.3 Recurrent Rollout Buffer (`RecurrentRolloutBuffer`)
- **Module:** `src/rl/rollout.py` (and re-exported in `src/rl/lstm_ppo.py`)
- **Capacity:** $N$ rollout steps (e.g. 512).
- **Stored Buffers:**
  - `obs_buf`: `(N, obs_dim)` float32
  - `actions_buf`: `(N,)` int64
  - `rewards_buf`: `(N,)` float32
  - `values_buf`: `(N,)` float32
  - `log_probs_buf`: `(N,)` float32
  - `dones_buf`: `(N,)` bool
  - `masks_buf`: `(N, action_dim)` bool
  - `h_buf`: `(N, lstm_hidden_dim)` float32 (hidden state $h_t$ before transition $t$)
  - `c_buf`: `(N, lstm_hidden_dim)` float32 (cell state $c_t$ before transition $t$)
- **Mini-batch Generation (`generate_recurrent_minibatches(seq_len=8, batch_size=32, advantages, returns, device="cpu")`):**
  - Chunks $N$ steps into contiguous sequences of length `seq_len` (with padding/masking if needed).
  - Supplies the initial $(h_0, c_0)$ for each sequence chunk.
  - Yields dictionaries containing `obs`, `actions`, `old_log_probs`, `values`, `advantages`, `returns`, `action_masks`, `dones`, `initial_h`, `initial_c`.

### 3.4 Masked Recurrent PPO Trainer (`MaskedRecurrentPPOTrainer`)
- **Module:** `src/rl/lstm_ppo.py`
- **Features:**
  - On-policy rollout collection with step-by-step hidden state tracking.
  - Hidden state reset on episode termination: `(1.0 - done) * hidden`.
  - Generalized Advantage Estimation (GAE) via `compute_gae`.
  - Sequence-level PPO optimization over multiple epochs with mini-batches of sequence chunks.
  - Surrogate clipped objective, clipped/unclipped value loss, entropy bonus, KL divergence tracking with early stopping (`target_kl`).
  - Linear learning rate annealing.
  - Model checkpoint saving and loading (`save(path)`, `load(path)`).

### 3.5 Recurrent PPO Agent (`RecurrentPPOAgent`)
- **Module:** `src/agents/recurrent_ppo_agent.py`
- **Features:**
  - Inherits from `BaseAgent(name="RecurrentPPOAgent")`.
  - Encapsulates `RecurrentMaskedActorCritic`, `BaseObservationEncoder`, `DiscreteActionSpace`, and `ActionMasker`.
  - Maintains `current_hidden: tuple[torch.Tensor, torch.Tensor]` initialized to zeros.
  - `reset()`: resets `current_hidden` to zeros at start of a new game.
  - `act(state, valid_actions, board)`: encodes observation, computes mask, runs step forward pass updating `current_hidden`, decodes discrete action to domain `Action`.
  - `save(path)` / `load(path)`: full model state dictionary serialization.

### 3.6 Scientific Benchmark Runner (`POMDPBenchmarkRunner`)
- **Module:** `src/evaluation/pomdp_benchmark.py`
- **Features:**
  - Trains both `MaskedPPOTrainer` (MLP baseline) and `MaskedRecurrentPPOTrainer` (LSTM recurrent policy) under identical environment seeds and step budgets.
  - Evaluates both trained models head-to-head (alternating first player) and against `RandomAgent`, `GreedyAgent`, and `StrategicAgent`.
  - Evaluates metrics:
    - Head-to-Head Win Rate and Score Differential (LSTM vs MLP)
    - Win Rates vs Baselines
    - Average Scores & Score Differentials
    - Destination Ticket Completion Rate
    - Route Claiming Efficiency
    - Average Turns per Game
    - Training loss curves and sample efficiency
  - Exports structured JSON report and GitHub Flavored Markdown report (`experiments/results/phase8_report.md`).

---

## 4. Testing & Acceptance Criteria

### Acceptance Criterion 1: Strict POMDP Information Hiding
- Invariant tests pass, verifying that hidden opponent cards, opponent tickets, and deck order do not affect encoded observation vectors.

### Acceptance Criterion 2: Recurrent Architecture & Action Masking
- `RecurrentMaskedActorCritic` correctly handles single-step and batched sequence forward passes.
- Action masking strictly assigns infinitesimal probability (-1e8 logits) to illegal actions.
- Hidden states are correctly tracked and updated.

### Acceptance Criterion 3: Recurrent Rollout & Mini-batching
- `RecurrentRolloutBuffer` correctly chunks trajectories into sequences with corresponding initial $(h_0, c_0)$ states.

### Acceptance Criterion 4: Deterministic Reproducibility
- Two `MaskedRecurrentPPOTrainer` instances initialized with the same random seed produce bitwise/numerically identical losses and parameter weights.

### Acceptance Criterion 5: Training Convergence & Outperforming Random
- `RecurrentPPOAgent` trained on synthetic mini board achieves $\ge 65\%$ win rate against `RandomAgent`.
- Checkpoints save and load correctly, preserving evaluation performance.

### Acceptance Criterion 6: Scientific Benchmark & Comparison Report
- `POMDPBenchmarkRunner` runs automated comparison between MLP PPO and LSTM PPO, generating valid JSON and Markdown reports.

---

## 5. File Structure Changes

```
src/
├── agents/
│   ├── recurrent_ppo_agent.py          # NEW: RecurrentPPOAgent implementation
│   └── __init__.py                     # Expose RecurrentPPOAgent
├── rl/
│   ├── lstm_ppo.py                     # EXPAND: RecurrentMaskedActorCritic & MaskedRecurrentPPOTrainer
│   ├── rollout.py                      # EXPAND: RecurrentRolloutBuffer with sequence minibatches
│   └── __init__.py                     # Expose recurrent classes
└── evaluation/
    ├── pomdp_benchmark.py              # NEW: POMDPBenchmarkRunner (MLP vs LSTM study)
    └── __init__.py                     # Expose POMDPBenchmarkRunner

tests/
├── environment/
│   └── test_pomdp_anti_leakage.py      # NEW: Strict POMDP information hiding tests
└── rl/
    ├── test_recurrent_ppo.py           # NEW: Unit tests for recurrent network, buffer & trainer
    └── test_phase8_acceptance.py       # NEW: Complete Phase 8 Acceptance Test Suite
```
