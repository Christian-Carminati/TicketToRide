# TicketToRide RL Lab

## 1. Vision

TicketToRide RL Lab è un progetto da zero che ricostruisce un gioco strategico ispirato a Ticket to Ride e lo trasforma in un **laboratorio sperimentale per Reinforcement Learning**.

Il progetto non deve essere concepito come:

> "Un gioco con un bot AI."

Deve essere concepito come:

> **"Un ambiente sperimentale nel quale possiamo costruire, addestrare, osservare, confrontare e valutare agenti di Reinforcement Learning."**

L'obiettivo principale è imparare in profondità:

* Reinforcement Learning
* Deep Reinforcement Learning
* DQN
* PPO
* recurrent policies
* partial observability / POMDP
* reward engineering
* action masking
* self-play
* opponent modeling
* curriculum learning
* generalization
* MCTS
* neural MCTS / AlphaZero-style approaches
* experiment tracking
* RL evaluation
* visualization
* software architecture per sistemi ML

La componente grafica e web non è un semplice frontend del gioco.

È parte integrante del progetto e deve permettere di **osservare ciò che l'agente sta imparando**.

---

# 2. Core Philosophy

Il progetto deve seguire questi principi.

## 2.1 RL-first

Il Reinforcement Learning è il cuore del progetto.

Il game engine deve essere sufficientemente pulito da poter essere utilizzato come ambiente RL.

---

## 2.2 Learning-first

Ogni tecnologia deve essere introdotta per imparare qualcosa.

Non usare una libreria solamente perché "fa tutto automaticamente".

Esempio:

* inizialmente usare Stable-Baselines3 per validare l'ambiente;
* successivamente implementare PPO in modo autonomo;
* utilizzare CleanRL come riferimento;
* successivamente implementare componenti RL manualmente quando questo porta valore didattico.

---

## 2.3 No black boxes without a reason

Durante le prime fasi bisogna capire cosa succede.

Evitare di nascondere completamente:

* policy
* value function
* reward
* observations
* actions
* trajectories
* advantages
* entropy
* losses
* training statistics

---

## 2.4 Experiment-driven development

Ogni importante modifica al sistema deve poter essere trasformata in un esperimento riproducibile.

Ogni esperimento deve avere:

* seed
* configurazione
* algoritmo
* environment version
* reward version
* model configuration
* training parameters
* evaluation results

---

## 2.5 Deterministic core

Dato:

```text
same seed
same initial state
same action sequence
```

il Game Core deve produrre lo stesso risultato.

La determinismo deve essere separata dalla parte RL.

---

## 2.6 Separation of concerns

Il progetto deve mantenere una separazione netta tra:

```text
Game Core
Environment
Agents
Training
Evaluation
Visualization
Experiment Tracking
```

Il Game Core non deve conoscere PPO.

PPO non deve conoscere il frontend.

Il frontend non deve implementare la logica del gioco.

---

# 3. High-Level Architecture

```text
                           ┌───────────────────────┐
                           │       WEB LAB         │
                           │                       │
                           │ Game Viewer            │
                           │ Training Viewer        │
                           │ Agent Brain             │
                           │ Metrics                 │
                           │ Experiments             │
                           │ Replay                  │
                           └───────────┬───────────┘
                                       │
                                  WebSocket/API
                                       │
                           ┌───────────▼───────────┐
                           │       API LAYER       │
                           └───────────┬───────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
       ┌────────────┐           ┌─────────────┐          ┌────────────┐
       │ GAME CORE  │           │ RL ENGINE   │          │ EVALUATION │
       │            │           │             │          │            │
       │ Board      │           │ DQN         │          │ Win rate   │
       │ Cards      │           │ PPO         │          │ Elo        │
       │ Routes     │           │ LSTM PPO    │          │ Score      │
       │ Players    │           │ Self Play   │          │ Generalize │
       │ Rules      │           │ MCTS        │          │            │
       └─────┬──────┘           └──────┬──────┘          └─────┬──────┘
             │                         │                       │
             └─────────────────────────┼───────────────────────┘
                                       │
                              ┌────────▼────────┐
                              │ EXPERIMENT DATA │
                              │                 │
                              │ configs         │
                              │ checkpoints     │
                              │ trajectories     │
                              │ metrics         │
                              │ replays         │
                              └─────────────────┘
```

---

# 4. Technology Stack

## Backend

Python 3.12+.

Primary libraries:

* PyTorch
* Gymnasium
* NumPy
* FastAPI
* WebSockets
* Pydantic

Useful reference implementations:

* Stable-Baselines3
* CleanRL

These libraries are allowed as references and validation tools.

They must not prevent later implementation of algorithms from scratch.

---

# Frontend

Recommended:

* React
* TypeScript
* Vite
* Canvas or SVG
* WebSocket

The frontend must be capable of rendering:

* board
* routes
* trains
* cards
* agent actions
* observations
* action probabilities
* training metrics
* reward curves
* neural network visualization
* replay

---

# Storage

Initial implementation:

* JSON
* JSONL
* SQLite

Do not introduce distributed infrastructure prematurely.

Later experiments may use:

* PostgreSQL
* Redis
* MLflow
* Weights & Biases

but these are not required initially.

---

# 5. Repository Structure

Recommended structure:

```text
ticket-to-ride-rl/
│
├── README.md
├── DESIGN.md
├── CLAUDE.md
├── pyproject.toml
│
├── src/
│   │
│   ├── game/
│   │   ├── board.py
│   │   ├── card.py
│   │   ├── route.py
│   │   ├── ticket.py
│   │   ├── player.py
│   │   ├── state.py
│   │   ├── action.py
│   │   ├── rules.py
│   │   ├── game.py
│   │   └── random.py
│   │
│   ├── environment/
│   │   ├── env.py
│   │   ├── observation.py
│   │   ├── action_space.py
│   │   ├── action_mask.py
│   │   └── reward.py
│   │
│   ├── agents/
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   ├── dqn_agent.py
│   │   ├── ppo_agent.py
│   │   └── mcts_agent.py
│   │
│   ├── rl/
│   │   ├── networks.py
│   │   ├── replay_buffer.py
│   │   ├── rollout.py
│   │   ├── advantage.py
│   │   ├── dqn.py
│   │   ├── ppo.py
│   │   ├── lstm_ppo.py
│   │   └── self_play.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── tournament.py
│   │   ├── elo.py
│   │   └── generalization.py
│   │
│   ├── experiments/
│   │   ├── config.py
│   │   ├── runner.py
│   │   └── registry.py
│   │
│   └── api/
│       ├── main.py
│       ├── websocket.py
│       └── schemas.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── views/
│   │   ├── canvas/
│   │   ├── hooks/
│   │   └── api/
│   └── package.json
│
├── experiments/
│   ├── configs/
│   ├── results/
│   ├── checkpoints/
│   └── replays/
│
├── tests/
│   ├── game/
│   ├── environment/
│   ├── agents/
│   └── rl/
│
└── scripts/
    ├── train.py
    ├── evaluate.py
    ├── tournament.py
    └── replay.py
```

The exact structure may evolve, but the conceptual separation must remain.

---

# 6. Game Core

The Game Core must be completely independent from RL.

It represents the actual game.

Core entities:

```text
Board
City
Route
TrainCard
DestinationTicket
Player
GameState
Action
Game
```

The Game Core must support:

```python
state = game.reset(seed=42)

valid_actions = game.valid_actions()

next_state = game.step(action)
```

The Game Core must never import:

```text
torch
gymnasium
stable_baselines
fastapi
react
```

---

# 7. State Representation

The game state must be explicit.

At minimum:

```text
board
current_player
train cards
visible cards
deck
discard pile
player train counts
claimed routes
destination tickets
scores
remaining trains
turn number
```

The state must support serialization.

Example:

```python
state.to_dict()
state.to_json()
GameState.from_dict(...)
```

This is important for:

* replay
* debugging
* web visualization
* experiment reproduction

---

# 8. Action Model

Actions must be explicit objects.

Examples:

```text
DRAW_VISIBLE_CARD
DRAW_HIDDEN_CARD
CLAIM_ROUTE
DRAW_TICKETS
KEEP_TICKET
DISCARD_TICKET
```

Every action must be serializable.

Every action must be validated by the Game Core.

The environment must never allow an invalid action to reach the Game Core without being detected.

---

# 9. Gymnasium Environment

The Game Core must be wrapped by a Gymnasium-compatible environment.

Conceptually:

```python
observation, info = env.reset(seed=42)

observation, reward, terminated, truncated, info = env.step(action)
```

The environment is responsible for:

* observation construction
* action encoding
* reward calculation
* action masking
* episode termination
* RL-specific information

The Game Core remains unaware of all of this.

---

# 10. Action Space

Initially use a flattened discrete action space.

Example:

```text
0                 draw hidden card
1..5              draw visible cards
6..N              claim routes
N+1...             ticket actions
```

The exact encoding should be data-driven rather than hardcoded.

Every action should have a stable ID.

Example:

```python
ActionId(
    type="CLAIM_ROUTE",
    route_id="route_42"
)
```

Internally it can map to an integer for RL.

---

# 11. Action Masking

Action masking is a first-class concept.

At every state:

```text
action_mask[action_id] = True / False
```

Example:

```text
DRAW_CARD       true
ROUTE_12        false
ROUTE_13        true
ROUTE_14        false
ROUTE_15        true
```

The policy must never receive invalid actions as equally valid choices.

Masking must be tested independently.

---

# 12. Observation Space

Do not expose the entire GameState blindly.

Create an explicit observation encoder.

Initial observation:

```text
own cards
visible cards
own trains
claimed routes
own tickets
current score
turn
remaining deck information
public opponent information
```

The observation encoder must be versioned.

Example:

```text
ObservationV1
ObservationV2
ObservationV3
```

This is important because experiments need to know which representation was used.

---

# 13. Partial Observability

The agent must NOT receive hidden information.

The environment must distinguish:

```text
TRUE STATE
```

from:

```text
AGENT OBSERVATION
```

Conceptually:

```text
                    TRUE STATE
                         │
               ┌─────────┴─────────┐
               │                   │
          visible info        hidden info
               │
               ▼
          observation
               │
               ▼
             agent
```

Never leak:

* opponent hidden cards
* opponent hidden tickets
* hidden deck order unless observable

This should be explicitly tested.

---

# 14. Baseline Agents

Before RL, implement:

## Random Agent

Selects random valid action.

Purpose:

* validate environment
* establish baseline

---

## Greedy Agent

Simple deterministic heuristics.

Possible strategy:

```text
1. Complete immediately achievable high-value route.
2. Otherwise progress toward an active ticket.
3. Otherwise collect useful cards.
4. Otherwise perform fallback action.
```

---

## Strategic Heuristic Agent

Later introduce:

* shortest path
* ticket completion probability
* route value
* opponent blocking
* longest route potential

These agents provide non-RL baselines.

---

# 15. RL Roadmap

The project must progress through RL algorithms in this order.

## Stage 1 — DQN

Learn:

* Q-function
* replay buffer
* target network
* epsilon-greedy
* TD loss
* Bellman equation

Use a simplified environment first.

---

## Stage 2 — PPO

Learn:

* policy gradient
* actor
* critic
* rollout
* advantage
* GAE
* entropy
* clipping
* policy loss
* value loss

PPO becomes the primary baseline.

---

## Stage 3 — Recurrent PPO

Introduce:

```text
Observation
      ↓
LSTM
      ↓
Policy
```

Purpose:

handle partial observability.

Compare:

```text
MLP PPO
vs
LSTM PPO
```

---

# 16. Reward Engineering

Reward must be configurable.

Do not hardcode reward logic inside the environment.

Example:

```yaml
reward:
  version: 3

  route_points: 1.0
  ticket_completion: 10.0
  final_score: 1.0
  win_bonus: 20.0
  loss_penalty: 10.0
```

The system must support multiple reward versions.

Experiments should be able to compare reward functions.

---

# 17. Reward Experiments

Create explicit experiments:

```text
reward_v1
reward_v2
reward_v3
reward_v4
```

Measure:

* win rate
* average score
* score differential
* ticket completion
* game length
* route efficiency

The purpose is not merely finding the highest reward.

The goal is understanding how reward design changes behavior.

---

# 18. Evaluation

Training performance is NOT evaluation performance.

Always maintain separate:

```text
TRAIN
VALIDATION
TEST
```

Evaluation should include:

```text
win rate
average score
score differential
average game length
ticket completion
route efficiency
```

Every trained model must be evaluated against fixed baselines.

---

# 19. Tournament System

Implement a tournament engine.

Example:

```text
PPO_v1
PPO_v2
DQN_v3
Greedy
Random
MCTS
```

Run:

```text
100 / 1000 / 10000 games
```

and calculate:

```text
wins
losses
draws
average score
Elo rating
```

The tournament system must be deterministic given a seed.

---

# 20. Self Play

After basic PPO works, introduce self-play.

Initially:

```text
PPO vs PPO
```

Later:

```text
current_policy
vs
historical_policies
```

Maintain a pool:

```text
policy_001
policy_002
policy_003
...
policy_N
```

Sample opponents from the pool.

Purpose:

avoid overfitting to one opponent.

---

# 21. Opponent Modeling

Later introduce an opponent model.

The model receives public game history:

```text
actions
routes
cards publicly drawn
scores
```

and predicts:

```text
likely objective
likely ticket
likely next route
```

This model may initially be supervised rather than RL.

The goal is to study whether opponent modeling improves decision making.

---

# 22. Curriculum Learning

Training complexity should increase gradually.

Example:

```text
Level 1
2 players
small map
few routes

Level 2
2 players
medium map

Level 3
3 players

Level 4
4 players

Level 5
full environment
```

The curriculum must be configurable.

---

# 23. Procedural Maps

Once the standard map is mastered, introduce procedural environments.

Generate:

* cities
* routes
* route lengths
* colors
* tickets

Each generated map must be deterministic from a seed.

Example:

```text
seed=42 → map_42
seed=43 → map_43
```

Training:

```text
many generated maps
```

Evaluation:

```text
completely unseen maps
```

Purpose:

measure generalization.

---

# 24. Generalization Experiments

Important question:

> Did the agent learn Ticket to Ride strategy or memorize the map?

Experiments must compare:

```text
same map
vs
unseen map
vs
procedural map
```

Metrics:

```text
performance degradation
win rate
score
ticket completion
```

---

# 25. MCTS

After PPO/self-play is stable, implement Monte Carlo Tree Search.

MCTS must be independent from PPO.

Components:

```text
Node
├── state
├── visits
├── value
├── children
└── action
```

Implement:

```text
Selection
Expansion
Simulation
Backpropagation
```

Start with a simple heuristic rollout.

---

# 26. Neural MCTS

Final advanced stage.

Explore AlphaZero-style architecture:

```text
Game State
    │
    ▼
Neural Network
    │
    ├──── Policy
    │
    └──── Value
           │
           ▼
          MCTS
           │
           ▼
         Action
```

This is an advanced optional stage.

Do not implement it until the previous RL stages are stable.

---

# 27. Web Application

The web interface is a laboratory.

It must have at least these views.

---

## Game View

Display:

* board
* routes
* players
* cards
* tickets
* current action
* scores

---

## Agent View

Display:

```text
current observation
action probabilities
selected action
action mask
reward
```

---

## Training View

Display live:

```text
episode
reward
mean reward
policy loss
value loss
entropy
KL
win rate
```

Graphs should update in real time.

---

## Brain View

Display the neural network.

Show:

```text
inputs
hidden layers
activations
outputs
```

The visualization does not need to be a mathematically perfect explanation of the model.

It is an introspection tool.

---

## Replay View

Every completed game should be replayable.

Controls:

```text
|<  <<  <  ▶  >  >>  >|
```

At every timestep show:

```text
state
observation
action
reward
```

---

## Experiment View

Show:

```text
experiment ID
algorithm
environment version
reward version
model
seed
training steps
results
```

Allow experiments to be compared.

---

# 28. Live Training

The frontend should be able to subscribe to a training run.

Example WebSocket events:

```text
episode_started
step
action_selected
reward
episode_finished
training_update
checkpoint_saved
```

The UI must not poll excessively.

Prefer WebSockets for live visualization.

---

# 29. Experiment Configuration

Training must be configuration-driven.

Example:

```yaml
experiment:
  name: ppo_lstm_selfplay_v1
  seed: 42

environment:
  players: 3
  observation_version: 2
  reward_version: 4

algorithm:
  name: ppo
  recurrent: true

network:
  hidden_size: 256
  layers: 2

training:
  total_timesteps: 1000000
  learning_rate: 0.0003
  batch_size: 256

evaluation:
  opponents:
    - greedy
    - random
    - historical
```

Do not require code modifications for normal experiments.

---

# 30. Reproducibility

Every experiment must save:

```text
config
seed
git commit
environment version
observation version
reward version
model checkpoint
training metrics
evaluation results
```

A result should be reproducible.

---

# 31. Testing Strategy

Game Core:

```text
unit tests
property tests
deterministic replay tests
```

Environment:

```text
reset tests
step tests
action mask tests
observation tests
reward tests
```

RL:

```text
network shape tests
buffer tests
advantage tests
loss tests
```

Integration:

```text
environment → agent → training
```

The project must have tests before implementing advanced RL.

---

# 32. Important Invariants

The following must always hold.

### Game invariants

```text
no negative trains
no invalid route claims
cards are conserved
routes can only be claimed once
scores are consistent
game terminates correctly
```

### Environment invariants

```text
observation is valid
action mask is valid
masked actions cannot be executed
reward is finite
```

### RL invariants

```text
loss is finite
gradients are finite
no NaN
no exploding values
```

---

# 33. Performance

Do not prematurely optimize.

First goal:

```text
correctness
```

Then:

```text
training throughput
```

Later investigate:

* vectorized environments
* parallel games
* multiprocessing
* GPU
* batched inference

A useful metric:

```text
environment steps / second
```

must be tracked.

---

# 34. Visualization vs Training

Training must be possible without the web UI.

For example:

```bash
python scripts/train.py --config experiments/configs/ppo.yaml
```

must work headlessly.

The web UI is a visualization layer, not a training dependency.

This is critical for running large experiments.

---

# 35. Development Phases

## Phase 0 — Project Skeleton

Deliver:

* repository structure
* Python environment
* frontend skeleton
* tests
* CI
* configuration system

No RL yet.

---

## Phase 1 — Game Core

Deliver:

* board
* routes
* cards
* tickets
* players
* game rules
* deterministic seed
* serialization

Acceptance:

A complete game can be played programmatically.

---

## Phase 2 — Baseline Agents

Deliver:

* random
* greedy
* heuristic

Acceptance:

1000-game tournament can run automatically.

---

## Phase 3 — Gymnasium Environment

Deliver:

* reset
* step
* observation
* action encoding
* action masking
* reward

Acceptance:

Environment passes Gymnasium checks.

---

## Phase 4 — First RL

Deliver:

* DQN
* PPO baseline

Use Stable-Baselines3 initially if necessary.

Acceptance:

RL agent beats random consistently.

---

## Phase 5 — Web Lab

Deliver:

* game visualization
* training visualization
* live metrics
* replay

Acceptance:

A user can watch an agent train and replay games.

---

## Phase 6 — PPO From Scratch

Deliver:

* actor
* critic
* rollout
* GAE
* PPO loss
* clipping
* entropy
* optimizer

Acceptance:

Custom PPO reaches comparable performance to reference implementation.

---

## Phase 7 — Reward Research

Deliver:

multiple reward configurations.

Acceptance:

Experiments can compare reward functions automatically.

---

## Phase 8 — Partial Observability

Deliver:

* strict information hiding
* recurrent observation
* LSTM PPO

Acceptance:

Compare MLP vs LSTM on partially observable environments.

---

## Phase 9 — Self Play

Deliver:

* self-play
* historical policy pool
* tournament evaluation

Acceptance:

Self-play agents can outperform scripted baselines.

---

## Phase 10 — Generalization

Deliver:

* procedural maps
* train/test split
* unseen map evaluation

Acceptance:

Report generalization metrics.

---

## Phase 11 — MCTS

Deliver:

* MCTS
* heuristic rollout
* evaluation

Acceptance:

MCTS becomes a valid opponent.

---

## Phase 12 — Advanced Research

Optional:

* neural MCTS
* AlphaZero-style training
* opponent modeling
* population-based training
* curriculum learning
* distributed training

---

# 36. Definition of Done

The project is considered successful when it can do the following:

```text
1. Start a game.
2. Play against random.
3. Play against heuristic.
4. Expose the game as Gymnasium environment.
5. Train DQN.
6. Train PPO.
7. Visualize PPO live.
8. Replay games.
9. Compare agents.
10. Run tournaments.
11. Train self-play.
12. Evaluate unseen maps.
13. Visualize agent decisions.
```

Advanced success:

```text
14. LSTM PPO
15. Opponent modeling
16. Curriculum learning
17. MCTS
18. Neural MCTS
```

---

# 37. What NOT to do

Do NOT:

* implement the complete game and RL simultaneously;
* start with PPO before validating the environment;
* start with self-play;
* start with MCTS;
* optimize GPU performance before correctness;
* create a huge microservice architecture;
* introduce Kubernetes;
* introduce distributed training early;
* make the frontend a dependency of the RL system;
* hide all RL internals behind libraries;
* optimize only training reward;
* evaluate an agent only against itself;
* leak hidden game information into observations.

---

# 38. Claude Code Development Rules

Claude Code must follow these rules.

## Rule 1

**Never implement multiple phases at once unless explicitly requested.**

If Phase 2 is being implemented, do not start Phase 3.

---

## Rule 2

Before implementing a new subsystem, inspect the current architecture.

Do not duplicate existing functionality.

---

## Rule 3

Prefer small, testable changes.

Each meaningful change should be accompanied by tests.

---

## Rule 4

Do not add dependencies without explaining why they are necessary.

---

## Rule 5

When implementing RL algorithms, explain the mathematical concept in code comments/docstrings where useful.

The goal is learning, not merely obtaining a working model.

---

## Rule 6

When a reference implementation exists, use it to validate results but do not blindly copy architecture.

---

## Rule 7

Every experiment must be reproducible.

---

## Rule 8

If an RL agent is not learning, do not immediately change hyperparameters randomly.

Debug in this order:

```text
environment
↓
action space
↓
action masking
↓
observation
↓
reward
↓
episode termination
↓
algorithm
↓
network
↓
hyperparameters
```

---

# 39. Learning Philosophy

The project should periodically answer:

```text
What did this experiment teach us?
```

Every major milestone should have a short experiment report:

```text
Hypothesis
Experiment
Configuration
Result
Interpretation
Next step
```

Example:

```text
Hypothesis:

LSTM should outperform MLP when hidden opponent information
is important.

Experiment:

100k games
MLP PPO vs LSTM PPO

Result:

MLP: 51%
LSTM: 63%

Interpretation:

The recurrent policy appears to benefit from historical
observations.

Next:

Introduce explicit opponent modeling.
```

The goal is not merely to accumulate features.

The goal is to build **understanding**.

---

# 40. Final Project Vision

The final system should feel like:

```text
                  🎫 TICKET TO RIDE RL LAB

 ┌─────────────────────────────────────────────────────┐
 │                                                     │
 │                     GAME                            │
 │                                                     │
 │     Agent A                          Agent B        │
 │                                                     │
 │                 BOARD                               │
 │                                                     │
 └───────────────────────┬─────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │     AGENT BRAIN      │
              │                      │
              │ Observation          │
              │       ↓              │
              │ Neural Network       │
              │       ↓              │
              │ Policy               │
              │       ↓              │
              │ Action               │
              └──────────┬───────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │    REWARD     │
                 └───────┬───────┘
                         │
                         ▼
                    RL TRAINING
                         │
                         ▼
                 ┌───────────────┐
                 │ EXPERIMENTS   │
                 └───────┬───────┘
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
             PPO        DQN         MCTS
              │          │           │
              └──────────┼───────────┘
                         ▼
                    TOURNAMENT
                         │
                         ▼
                  GENERALIZATION
```

The project should ultimately allow a user to ask:

> "Which agent is strongest?"

> "Why?"

> "Against which opponents?"

> "On which maps?"

> "Does it generalize?"

> "How did its strategy evolve?"

> "What happens if we change the reward?"

> "What happens if we remove memory?"

> "What happens if we introduce self-play?"

And the system should provide experimental evidence rather than just an answer.

---

# 41. First Task for Claude Code

Do NOT start implementing the entire project.

The first task is:

1. Inspect this design document.
2. Create the repository skeleton.
3. Create the Python project configuration.
4. Create the frontend skeleton.
5. Create the test structure.
6. Create minimal documentation.
7. Do NOT implement RL.
8. Do NOT implement the complete game.
9. Do NOT add unnecessary dependencies.
10. Stop after the project skeleton is complete.

Then report:

```text
Files created
Dependencies added
Architecture decisions
Tests created
Next recommended phase
```

The project must evolve incrementally from:

```text
GAME
 ↓
ENVIRONMENT
 ↓
BASELINES
 ↓
RL
 ↓
SELF PLAY
 ↓
GENERALIZATION
 ↓
MCTS
 ↓
RESEARCH LAB
```

**Do not skip the foundations.**
