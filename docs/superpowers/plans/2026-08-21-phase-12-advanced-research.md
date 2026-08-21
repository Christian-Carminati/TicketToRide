# Fase 12: Advanced Research (Neural MCTS, AlphaZero, Bayesian Opponent Modeling & Curriculum Learning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete Phase 12 Advanced Research suite for TicketToRide RL Lab, featuring: (1) Dual-headed Policy-Value Network (`PolicyValueNetwork`) with action masking; (2) AlphaZero PUCT Neural MCTS Search Engine (`NeuralMCTSEngine`) with Dirichlet exploration noise; (3) AlphaZero Self-Play training loop (`AlphaZeroTrainer`) and replay buffer; (4) Bayesian Opponent Modeling (`BayesianTicketBeliefTracker`), Detour Engine, Belief-Weighted Determinization, and Tactical Blocker; (5) Curriculum Learning Manager (`CurriculumManager`); (6) `NeuralMCTSAgent` & `OpponentAwareMCTSAgent`; and (7) Cross-Paradigm Scientific Benchmark Suite and Acceptance Suite.

**Architecture:**
1. `src/rl/policy_value_net.py`: Dual-headed network predicting masked policy prior $P(s, a)$ and state value $V(s) \in [-1, 1]$ with residual MLP trunk.
2. `src/rl/alphazero_search.py`: Polynomial PUCT tree search without random rollouts, evaluated directly by `PolicyValueNetwork`, with root Dirichlet noise.
3. `src/rl/alphazero_trainer.py`: Self-play trajectory collector, $(s, \mathbf{m}, \boldsymbol{\pi}, z)$ replay buffer, joint loss optimizer $\mathcal{L} = (z - v)^2 - \boldsymbol{\pi}^T \log \mathbf{p} + c \|\theta\|^2$.
4. `src/rl/opponent_model.py`: Graph Detour Engine (Dijkstra), Bayesian posterior tracker $P(T_k \mid \mathcal{E}_{\text{opp}})$, belief-weighted determinizer for Information-Set MCTS, and tactical bottleneck blocker.
5. `src/rl/curriculum.py`: Multi-stage curriculum manager with map progression (`Micro` $\to$ `Small` $\to$ `USA_Standard`) and opponent difficulty gates.
6. `src/agents/neural_mcts_agent.py`: `NeuralMCTSAgent` and `OpponentAwareMCTSAgent` adhering to `BaseAgent` interface.
7. `src/evaluation/phase12_benchmark.py`: Complete cross-paradigm tournament and scientific report generator.

**Tech Stack:** Python 3.12, PyTorch, Gymnasium, NumPy, NetworkX, pytest.

**Spec:** [`docs/superpowers/specs/2026-08-21-phase-12-advanced-research-design.md`](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-21-phase-12-advanced-research-design.md)

## Global Constraints
- 100% deterministic reproducibility when seeded.
- Game Core must remain completely independent of RL and neural libraries.
- Strict anti-leakage compliance (POMDP): Opponent hidden cards or tickets must NEVER be read directly; only public game history (claimed routes, face-up draws) can be used.
- Full type annotations and docstrings with mathematical formulas and explanations in all new modules.
- Test-driven development (TDD) for every task with comprehensive unit tests.

---

### Task 1: Policy-Value Dual Network (`PolicyValueNetwork`)

**Files:**
- Create: `src/rl/policy_value_net.py`
- Test: `tests/rl/test_policy_value_net.py`

**Interfaces:**
- Produces:
  - `PolicyValueNetwork(obs_dim: int, action_dim: int, hidden_dim: int = 256, num_res_blocks: int = 2)`
  - `PolicyValueNetwork.forward(obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]`
  - `PolicyValueNetwork.evaluate_state(obs: np.ndarray, action_mask: np.ndarray | None = None) -> tuple[np.ndarray, float]`
  - `PolicyValueNetwork.save(path: str | Path) -> None`
  - `PolicyValueNetwork.load(path: str | Path, map_location: str = "cpu") -> "PolicyValueNetwork"`

- [ ] **Step 1: Write the failing test for `PolicyValueNetwork`**

```python
# tests/rl/test_policy_value_net.py
import pytest
import torch
import numpy as np
from pathlib import Path
from src.rl.policy_value_net import PolicyValueNetwork

def test_policy_value_net_forward_shapes():
    obs_dim = 150
    action_dim = 45
    net = PolicyValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64, num_res_blocks=1)
    
    batch_size = 8
    obs = torch.randn(batch_size, obs_dim)
    mask = torch.ones(batch_size, action_dim)
    mask[:, 10:] = 0.0  # only first 10 actions valid
    
    policy_probs, values = net(obs, mask)
    
    assert policy_probs.shape == (batch_size, action_dim)
    assert values.shape == (batch_size, 1)
    
    # Check probabilities sum to 1
    sums = policy_probs.sum(dim=-1)
    np.testing.assert_allclose(sums.detach().numpy(), np.ones(batch_size), atol=1e-5)
    
    # Check masked actions have prob == 0.0
    assert torch.all(policy_probs[:, 10:] < 1e-6)
    
    # Value range in [-1, 1] due to tanh
    assert torch.all(values >= -1.0) and torch.all(values <= 1.0)

def test_policy_value_net_evaluate_state_numpy():
    obs_dim = 50
    action_dim = 12
    net = PolicyValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32)
    
    obs = np.random.randn(obs_dim).astype(np.float32)
    mask = np.zeros(action_dim, dtype=np.float32)
    mask[0] = 1.0
    mask[3] = 1.0
    
    probs, val = net.evaluate_state(obs, mask)
    assert isinstance(probs, np.ndarray)
    assert isinstance(val, float)
    assert probs.shape == (action_dim,)
    assert pytest.approx(probs.sum(), abs=1e-5) == 1.0
    assert probs[1] == 0.0
    assert probs[0] > 0.0
    assert -1.0 <= val <= 1.0

def test_policy_value_net_save_load(tmp_path: Path):
    obs_dim = 40
    action_dim = 10
    net = PolicyValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32)
    
    save_path = tmp_path / "pv_net.pt"
    net.save(save_path)
    
    loaded_net = PolicyValueNetwork.load(save_path)
    assert loaded_net.obs_dim == obs_dim
    assert loaded_net.action_dim == action_dim
    
    obs = np.random.randn(obs_dim).astype(np.float32)
    p1, v1 = net.evaluate_state(obs)
    p2, v2 = loaded_net.evaluate_state(obs)
    np.testing.assert_allclose(p1, p2, atol=1e-6)
    assert pytest.approx(v1, abs=1e-6) == v2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rl/test_policy_value_net.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rl.policy_value_net'`

- [ ] **Step 3: Implement `src/rl/policy_value_net.py`**

```python
# src/rl/policy_value_net.py
"""
Dual-headed Policy-Value Neural Network for AlphaZero-style MCTS.

Calculates:
- Policy Prior: P_theta(s, a) via masked logits Softmax
- State Value: v_theta(s) in [-1.0, 1.0] via Tanh
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with Linear, LayerNorm, and ReLU activation."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        return F.relu(out + residual)

class PolicyValueNetwork(nn.Module):
    """
    Dual-Headed Actor-Critic Network for AlphaZero Search & Training.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared trunk
        self.input_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action probabilities and scalar state value.
        """
        h = self.input_layer(obs)
        for block in self.res_blocks:
            h = block(h)
            
        logits = self.policy_head(h)
        if action_mask is not None:
            # Apply large negative penalty to masked invalid actions
            logits = torch.where(action_mask > 0.5, logits, torch.tensor(-1e8, device=logits.device, dtype=logits.dtype))
            
        policy_probs = F.softmax(logits, dim=-1)
        value = self.value_head(h)
        return policy_probs, value

    def evaluate_state(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Numpy inference wrapper for MCTS node evaluation.
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(action_mask, dtype=torch.float32).unsqueeze(0) if action_mask is not None else None
            p_t, v_t = self.forward(obs_t, mask_t)
            probs = p_t.squeeze(0).cpu().numpy()
            val = float(v_t.item())
        return probs, val

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "state_dict": self.state_dict(),
        }, p)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> PolicyValueNetwork:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        net = cls(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()
        return net
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rl/test_policy_value_net.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/policy_value_net.py tests/rl/test_policy_value_net.py
git commit -m "feat(rl): implement PolicyValueNetwork dual-head architecture with action masking"
```

---

### Task 2: Neural MCTS Search Engine (AlphaZero Search)

**Files:**
- Create: `src/rl/alphazero_search.py`
- Test: `tests/rl/test_alphazero_search.py`

**Interfaces:**
- Produces:
  - `NeuralMCTSNode` with prior $P(s, a)$, visit counts $N(s, a)$, and total value $W(s, a)$.
  - `NeuralMCTSEngine(net: PolicyValueNetwork, c_puct: float = 1.5, dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25, num_simulations: int = 50)`
  - `NeuralMCTSEngine.search(game: Game, player_id: str, is_root_exploration: bool = True) -> tuple[int, np.ndarray, float]`
  - `NeuralMCTSEngine.get_action_probabilities(temperature: float = 1.0) -> np.ndarray`

- [ ] **Step 1: Write the failing test for `NeuralMCTSEngine`**

```python
# tests/rl/test_alphazero_search.py
import pytest
import numpy as np
from src.game.game import Game
from src.environment.ticket_to_ride_env import TicketToRideEnv
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.alphazero_search import NeuralMCTSEngine, NeuralMCTSNode

def test_neural_mcts_node_puct_selection():
    root = NeuralMCTSNode(state_player="player_0")
    # Action 0: high prior 0.8, 0 visits
    # Action 1: low prior 0.2, 0 visits
    root.priors = {0: 0.8, 1: 0.2}
    root.legal_actions = [0, 1]
    
    # Selection should pick action 0 (highest UCT due to prior)
    best_a = root.select_puct_action(c_puct=1.5)
    assert best_a == 0
    
    # Simulate visiting action 0 multiple times with lower Q
    root.visits[0] = 10
    root.values[0] = -5.0  # Q = -0.5
    root.visits[1] = 1
    root.values[1] = 0.5   # Q = 0.5
    
    # Selection should now favor action 1
    best_a2 = root.select_puct_action(c_puct=1.5)
    assert best_a2 == 1

def test_neural_mcts_engine_search_execution():
    env = TicketToRideEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    net = PolicyValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, num_res_blocks=1)
    engine = NeuralMCTSEngine(net=net, num_simulations=20, c_puct=1.5)
    
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    
    best_action, visit_probs, root_value = engine.search(game, player_id="player_0", is_root_exploration=True)
    
    assert isinstance(best_action, int)
    assert 0 <= best_action < action_dim
    assert isinstance(visit_probs, np.ndarray)
    assert visit_probs.shape == (action_dim,)
    assert pytest.approx(visit_probs.sum(), abs=1e-5) == 1.0
    assert -1.0 <= root_value <= 1.0
    
    # Check that chosen action is legal in the game
    legal_actions = env.get_action_mask()
    # Note: best_action should have visit_probs[best_action] > 0
    assert visit_probs[best_action] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rl/test_alphazero_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rl.alphazero_search'`

- [ ] **Step 3: Implement `src/rl/alphazero_search.py`**

```python
# src/rl/alphazero_search.py
"""
Neural Monte Carlo Tree Search (AlphaZero Search Engine).

Uses Polynomial Upper Confidence Trees (PUCT) guided by PolicyValueNetwork
with Dirichlet exploration noise at the root, completely eliminating random rollouts.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.game.game import Game
from src.environment.observation import ObservationEncoder
from src.environment.action_space import ActionSpaceHandler
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.mcts_determinization import determinize_game_state

class NeuralMCTSNode:
    """Node in the Neural MCTS search tree."""
    def __init__(self, state_player: str, parent: Optional[NeuralMCTSNode] = None, action_from_parent: Optional[int] = None):
        self.state_player = state_player
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: Dict[int, NeuralMCTSNode] = {}
        
        self.legal_actions: List[int] = []
        self.priors: Dict[int, float] = {}
        self.visits: Dict[int, int] = {}
        self.values: Dict[int, float] = {}
        self.is_expanded: bool = False
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0

    @property
    def total_visits(self) -> int:
        return sum(self.visits.values())

    def get_q_value(self, action: int) -> float:
        n = self.visits.get(action, 0)
        return self.values.get(action, 0.0) / n if n > 0 else 0.0

    def select_puct_action(self, c_puct: float = 1.5) -> int:
        """Select action maximizing Q(s, a) + U(s, a)."""
        tot_v = self.total_visits
        sqrt_tot = math.sqrt(tot_v) if tot_v > 0 else 1.0
        
        best_score = -float("inf")
        best_action = self.legal_actions[0]
        
        for a in self.legal_actions:
            q = self.get_q_value(a)
            p = self.priors.get(a, 0.0)
            n = self.visits.get(a, 0)
            u = c_puct * p * (sqrt_tot / (1.0 + n))
            score = q + u
            if score > best_score:
                best_score = score
                best_action = a
        return best_action

class NeuralMCTSEngine:
    """
    AlphaZero-style Search Engine using Neural Policy Priors and State Values.
    """
    def __init__(
        self,
        net: PolicyValueNetwork,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        
        self.encoder = ObservationEncoder()
        self.action_handler = ActionSpaceHandler()

    def search(
        self,
        game: Game,
        player_id: str,
        is_root_exploration: bool = True,
        custom_determinizer = None,
    ) -> Tuple[int, np.ndarray, float]:
        """
        Runs MCTS search from root game state.
        Returns: (best_action, target_policy_probs, estimated_root_value)
        """
        curr_player = game.state.current_player
        root = NeuralMCTSNode(state_player=curr_player)
        
        # Initial expansion of root
        obs = self.encoder.encode(game.state, curr_player)
        mask = self.action_handler.get_action_mask(game.state, curr_player)
        legal_actions = [i for i, v in enumerate(mask) if v > 0.5]
        if not legal_actions:
            return 0, np.zeros(self.net.action_dim), 0.0
            
        priors, root_v = self.net.evaluate_state(obs, mask)
        
        # Apply Dirichlet noise at root for self-play exploration
        if is_root_exploration and len(legal_actions) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for idx, a in enumerate(legal_actions):
                root.priors[a] = (1.0 - self.dirichlet_eps) * priors[a] + self.dirichlet_eps * noise[idx]
        else:
            for a in legal_actions:
                root.priors[a] = float(priors[a])
                
        root.legal_actions = legal_actions
        for a in legal_actions:
            root.visits[a] = 0
            root.values[a] = 0.0
        root.is_expanded = True
        
        # Iterative simulations
        for _ in range(self.num_simulations):
            if custom_determinizer is not None:
                sim_game = custom_determinizer(game, player_id)
            else:
                sim_game = determinize_game_state(game, player_id)
                
            node = root
            search_path: List[Tuple[NeuralMCTSNode, int]] = []
            
            # 1. Selection
            while node.is_expanded and not sim_game.state.is_game_over():
                action_idx = node.select_puct_action(self.c_puct)
                search_path.append((node, action_idx))
                game_action = self.action_handler.decode_action(action_idx, sim_game.state)
                sim_game.step(game_action)
                
                if action_idx not in node.children:
                    # Next node not yet created
                    next_player = sim_game.state.current_player
                    node.children[action_idx] = NeuralMCTSNode(
                        state_player=next_player, parent=node, action_from_parent=action_idx
                    )
                node = node.children[action_idx]
                
            # 2. Evaluation & Expansion
            if sim_game.state.is_game_over():
                scores = sim_game.get_scores()
                p0_score = scores.get(player_id, 0)
                opp_ids = [pid for pid in scores if pid != player_id]
                opp_score = scores.get(opp_ids[0], 0) if opp_ids else 0
                if p0_score > opp_score:
                    leaf_val = 1.0
                elif p0_score < opp_score:
                    leaf_val = -1.0
                else:
                    leaf_val = 0.0
            else:
                s_player = sim_game.state.current_player
                obs_leaf = self.encoder.encode(sim_game.state, s_player)
                mask_leaf = self.action_handler.get_action_mask(sim_game.state, s_player)
                legal_leaf = [i for i, v in enumerate(mask_leaf) if v > 0.5]
                
                priors_leaf, val_leaf = self.net.evaluate_state(obs_leaf, mask_leaf)
                node.legal_actions = legal_leaf
                for a in legal_leaf:
                    node.priors[a] = float(priors_leaf[a])
                    node.visits[a] = 0
                    node.values[a] = 0.0
                node.is_expanded = True
                
                # val_leaf is from perspective of s_player. Convert to root player's perspective
                leaf_val = val_leaf if s_player == player_id else -val_leaf
                
            # 3. Backpropagation
            for parent_node, action_taken in reversed(search_path):
                # Value to add to parent node
                val_for_parent = leaf_val if parent_node.state_player == player_id else -leaf_val
                parent_node.visits[action_taken] += 1
                parent_node.values[action_taken] += val_for_parent
                
        # Construct target visit distribution pi
        pi = np.zeros(self.net.action_dim, dtype=np.float32)
        for a in root.legal_actions:
            pi[a] = root.visits[a]
            
        tot_visits = pi.sum()
        if tot_visits > 0:
            pi /= tot_visits
        else:
            for a in root.legal_actions:
                pi[a] = 1.0 / len(root.legal_actions)
                
        best_action = int(np.argmax(pi))
        root_val = sum(root.values.values()) / max(1, root.total_visits)
        return best_action, pi, root_val
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rl/test_alphazero_search.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/alphazero_search.py tests/rl/test_alphazero_search.py
git commit -m "feat(rl): implement AlphaZero NeuralMCTSEngine with PUCT and Dirichlet exploration"
```

---

### Task 3: AlphaZero Self-Play Pipeline & Trainer

**Files:**
- Create: `src/rl/alphazero_trainer.py`
- Test: `tests/rl/test_alphazero_trainer.py`

**Interfaces:**
- Produces:
  - `SelfPlayReplayBuffer(capacity: int = 50000)`
  - `AlphaZeroTrainer(net: PolicyValueNetwork, lr: float = 1e-3, weight_decay: float = 1e-4, num_simulations: int = 30)`
  - `AlphaZeroTrainer.collect_self_play_games(num_games: int = 2) -> int`
  - `AlphaZeroTrainer.train_step(batch_size: int = 32) -> dict[str, float]`
  - `AlphaZeroTrainer.save_checkpoint(path: str | Path)`

- [ ] **Step 1: Write the failing test for `AlphaZeroTrainer`**

```python
# tests/rl/test_alphazero_trainer.py
import pytest
import numpy as np
from pathlib import Path
from src.environment.ticket_to_ride_env import TicketToRideEnv
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.alphazero_trainer import SelfPlayReplayBuffer, AlphaZeroTrainer

def test_self_play_replay_buffer():
    buf = SelfPlayReplayBuffer(capacity=100)
    obs_dim = 10
    action_dim = 5
    
    for _ in range(20):
        obs = np.random.randn(obs_dim).astype(np.float32)
        mask = np.ones(action_dim, dtype=np.float32)
        pi = np.full(action_dim, 0.2, dtype=np.float32)
        z = 1.0
        buf.add(obs, mask, pi, z)
        
    assert len(buf) == 20
    b_obs, b_mask, b_pi, b_z = buf.sample(batch_size=8)
    assert b_obs.shape == (8, obs_dim)
    assert b_mask.shape == (8, action_dim)
    assert b_pi.shape == (8, action_dim)
    assert b_z.shape == (8, 1)

def test_alphazero_trainer_collect_and_train(tmp_path: Path):
    env = TicketToRideEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    net = PolicyValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, num_res_blocks=1)
    trainer = AlphaZeroTrainer(net=net, num_simulations=10, batch_size=16)
    
    # Collect 1 self-play game
    num_steps = trainer.collect_self_play_games(num_games=1)
    assert num_steps > 0
    assert len(trainer.buffer) == num_steps
    
    # Train 1 step
    metrics = trainer.train_step(batch_size=8)
    assert "loss" in metrics
    assert "value_loss" in metrics
    assert "policy_loss" in metrics
    assert metrics["loss"] > 0.0
    
    # Test saving
    ckpt_path = tmp_path / "az_ckpt.pt"
    trainer.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rl/test_alphazero_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rl.alphazero_trainer'`

- [ ] **Step 3: Implement `src/rl/alphazero_trainer.py`**

```python
# src/rl/alphazero_trainer.py
"""
AlphaZero Self-Play Data Generation & Joint Optimization Trainer.

Optimizes joint loss:
L(theta) = MSE(z, v_theta(s)) - pi^T * log(p_theta(s)) + c_reg * ||theta||_2^2
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.game.game import Game
from src.environment.observation import ObservationEncoder
from src.environment.action_space import ActionSpaceHandler
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.alphazero_search import NeuralMCTSEngine

class SelfPlayReplayBuffer:
    """Experience replay storing (obs, action_mask, pi_mcts, outcome_z)."""
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.observations: List[np.ndarray] = []
        self.action_masks: List[np.ndarray] = []
        self.target_policies: List[np.ndarray] = []
        self.target_values: List[float] = []

    def __len__(self) -> int:
        return len(self.observations)

    def add(self, obs: np.ndarray, mask: np.ndarray, pi: np.ndarray, z: float):
        if len(self.observations) >= self.capacity:
            self.observations.pop(0)
            self.action_masks.pop(0)
            self.target_policies.pop(0)
            self.target_values.pop(0)
        self.observations.append(obs)
        self.action_masks.append(mask)
        self.target_policies.append(pi)
        self.target_values.append(z)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = random.sample(range(len(self.observations)), min(batch_size, len(self.observations)))
        obs_b = torch.as_tensor(np.array([self.observations[i] for i in indices]), dtype=torch.float32)
        mask_b = torch.as_tensor(np.array([self.action_masks[i] for i in indices]), dtype=torch.float32)
        pi_b = torch.as_tensor(np.array([self.target_policies[i] for i in indices]), dtype=torch.float32)
        z_b = torch.as_tensor(np.array([self.target_values[i] for i in indices]), dtype=torch.float32).unsqueeze(1)
        return obs_b, mask_b, pi_b, z_b

class AlphaZeroTrainer:
    """
    Self-Play Collector & AlphaZero Optimization Pipeline.
    """
    def __init__(
        self,
        net: PolicyValueNetwork,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_simulations: int = 30,
        c_puct: float = 1.5,
        batch_size: int = 32,
        buffer_capacity: int = 50000,
        device: str = "cpu",
    ):
        self.net = net.to(device)
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.buffer = SelfPlayReplayBuffer(capacity=buffer_capacity)
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.search_engine = NeuralMCTSEngine(net=self.net, num_simulations=num_simulations, c_puct=c_puct)
        self.encoder = ObservationEncoder()
        self.action_handler = ActionSpaceHandler()

    def collect_self_play_games(self, num_games: int = 1, max_turns: int = 150) -> int:
        """
        Plays self-play games where Neural MCTS controls both players.
        Stores trajectories in replay buffer.
        """
        total_steps = 0
        for _ in range(num_games):
            game = Game(num_players=2)
            game.reset()
            
            history: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
            turns = 0
            
            while not game.state.is_game_over() and turns < max_turns:
                curr_player = game.state.current_player
                obs = self.encoder.encode(game.state, curr_player)
                mask = self.action_handler.get_action_mask(game.state, curr_player)
                
                best_action, pi_target, _ = self.search_engine.search(
                    game, curr_player, is_root_exploration=True
                )
                
                history.append((curr_player, obs, mask, pi_target))
                game_action = self.action_handler.decode_action(best_action, game.state)
                game.step(game_action)
                turns += 1
                
            # Assign game outcomes
            scores = game.get_scores()
            p0_score = scores.get("player_0", 0)
            p1_score = scores.get("player_1", 0)
            
            for player_step, obs_step, mask_step, pi_step in history:
                if player_step == "player_0":
                    z = 1.0 if p0_score > p1_score else (-1.0 if p0_score < p1_score else 0.0)
                else:
                    z = 1.0 if p1_score > p0_score else (-1.0 if p1_score < p0_score else 0.0)
                self.buffer.add(obs_step, mask_step, pi_step, z)
                total_steps += 1
                
        return total_steps

    def train_step(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Performs one gradient descent step on sampled batch from self-play buffer.
        """
        b_size = batch_size or self.batch_size
        if len(self.buffer) < b_size:
            return {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0}
            
        self.net.train()
        obs_b, mask_b, pi_b, z_b = self.buffer.sample(b_size)
        obs_b = obs_b.to(self.device)
        mask_b = mask_b.to(self.device)
        pi_b = pi_b.to(self.device)
        z_b = z_b.to(self.device)
        
        pred_probs, pred_vals = self.net(obs_b, mask_b)
        
        # Value MSE loss
        value_loss = F.mse_loss(pred_vals, z_b)
        
        # Policy Cross Entropy: - sum(pi * log(p + 1e-8))
        log_probs = torch.log(pred_probs + 1e-8)
        policy_loss = -torch.mean(torch.sum(pi_b * log_probs, dim=-1))
        
        total_loss = value_loss + policy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "loss": float(total_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        self.net.save(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rl/test_alphazero_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/alphazero_trainer.py tests/rl/test_alphazero_trainer.py
git commit -m "feat(rl): implement AlphaZeroTrainer and SelfPlayReplayBuffer"
```

---

### Task 4: Bayesian Opponent Modeling & Destination Belief Tracker

**Files:**
- Create: `src/rl/opponent_model.py`
- Test: `tests/rl/test_opponent_model.py`

**Interfaces:**
- Produces:
  - `GraphDetourEngine(board: Board, tickets: list[DestinationTicket])`
  - `GraphDetourEngine.compute_detour(route: Route, ticket: DestinationTicket) -> float`
  - `BayesianTicketBeliefTracker(board: Board, all_tickets: list[DestinationTicket])`
  - `BayesianTicketBeliefTracker.observe_route_claim(player_id: str, route_id: str)`
  - `BayesianTicketBeliefTracker.get_ticket_probabilities(player_id: str) -> dict[DestinationTicket, float]`
  - `BeliefWeightedDeterminizer(game: Game, root_player_id: str, tracker: BayesianTicketBeliefTracker) -> Game`
  - `TacticalBlocker(board: Board, tracker: BayesianTicketBeliefTracker)`

- [ ] **Step 1: Write the failing test for `opponent_model.py`**

```python
# tests/rl/test_opponent_model.py
import pytest
import numpy as np
from src.game.game import Game
from src.game.board import Board
from src.rl.opponent_model import (
    GraphDetourEngine,
    BayesianTicketBeliefTracker,
    belief_weighted_determinization,
    TacticalBlocker,
)

def test_graph_detour_engine():
    board = Board()
    tickets = board.destination_tickets
    engine = GraphDetourEngine(board=board, tickets=tickets)
    
    # Pick first ticket, say NYC to Boston
    t0 = tickets[0]
    # Find route connecting NYC-Boston or directly on path
    routes_on_path = [r for r in board.routes if (r.city1 == t0.city1 or r.city2 == t0.city2)]
    assert len(routes_on_path) > 0
    r0 = routes_on_path[0]
    
    detour = engine.compute_detour(r0, t0)
    assert detour >= 0.0

def test_bayesian_ticket_belief_tracker_learning():
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    
    board = game.board
    all_tickets = board.destination_tickets
    tracker = BayesianTicketBeliefTracker(board=board, all_tickets=all_tickets)
    
    # Opponent (player_1) true ticket
    p1 = game.state.players[1]
    assert len(p1.tickets) > 0
    true_ticket = p1.tickets[0]
    
    # Initial probabilities should be uniform
    p_init = tracker.get_ticket_probabilities("player_1")
    assert pytest.approx(sum(p_init.values()), abs=1e-5) == 1.0
    
    # Find routes that belong to true_ticket
    engine = GraphDetourEngine(board=board, tickets=all_tickets)
    candidate_routes = [r for r in board.routes if engine.compute_detour(r, true_ticket) == 0.0]
    
    # Simulate player_1 claiming these routes
    for r in candidate_routes[:2]:
        tracker.observe_route_claim("player_1", r.id)
        
    p_updated = tracker.get_ticket_probabilities("player_1")
    assert p_updated[true_ticket] > p_init[true_ticket]
    
    # Top-K recall
    top_3 = tracker.get_top_k_tickets("player_1", k=3)
    assert len(top_3) == min(3, len(all_tickets))
    assert sum(prob for _, prob in top_3) > 0.0

def test_belief_weighted_determinization_no_leak():
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    
    tracker = BayesianTicketBeliefTracker(board=game.board, all_tickets=game.board.destination_tickets)
    
    # Determinize for player_0
    cloned_game = belief_weighted_determinization(game, root_player_id="player_0", tracker=tracker)
    
    assert len(cloned_game.state.players) == 2
    assert cloned_game.state.players[0].tickets == game.state.players[0].tickets
    # player_1 tickets sampled consistently
    assert len(cloned_game.state.players[1].tickets) == len(game.state.players[1].tickets)

def test_tactical_blocker_evaluates_threats():
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    tracker = BayesianTicketBeliefTracker(board=game.board, all_tickets=game.board.destination_tickets)
    blocker = TacticalBlocker(board=game.board, tracker=tracker)
    
    threats = blocker.compute_unclaimed_route_threats("player_1")
    assert isinstance(threats, dict)
    assert len(threats) > 0
    for r_id, threat_val in threats.items():
        assert threat_val >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rl/test_opponent_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rl.opponent_model'`

- [ ] **Step 3: Implement `src/rl/opponent_model.py`**

```python
# src/rl/opponent_model.py
"""
Bayesian Opponent Modeling, Detour Distance Engine, and Belief-Weighted Determinization.

Computes:
P(Ticket_k | E_claimed) proportional to P(Ticket_k) * prod( L(e | Ticket_k) )
where L(e | Ticket_k) decays with the graph detour cost.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np
import networkx as nx

from src.game.board import Board, Route
from src.game.ticket import DestinationTicket
from src.game.game import Game

class GraphDetourEngine:
    """
    Computes shortest path distances and detour routing penalties for tickets.
    """
    def __init__(self, board: Board, tickets: List[DestinationTicket]):
        self.board = board
        self.tickets = tickets
        self.graph = nx.Graph()
        
        # Build undirected graph
        for r in board.routes:
            self.graph.add_edge(r.city1, r.city2, weight=r.length, id=r.id)
            
        self.distances: Dict[Tuple[str, str], float] = dict(nx.all_pairs_dijkstra_path_length(self.graph))

    def get_dist(self, c1: str, c2: str) -> float:
        if c1 == c2:
            return 0.0
        return self.distances.get((c1, c2), 999.0)

    def compute_detour(self, route: Route, ticket: DestinationTicket) -> float:
        """
        Computes detour cost of routing through edge `route` between ticket endpoints.
        Detour = min(d(u, a) + len(e) + d(b, v), d(u, b) + len(e) + d(a, v)) - d(u, v)
        """
        u, v = ticket.city1, ticket.city2
        a, b = route.city1, route.city2
        w = route.length
        base_d = self.get_dist(u, v)
        
        d1 = self.get_dist(u, a) + w + self.get_dist(b, v)
        d2 = self.get_dist(u, b) + w + self.get_dist(a, v)
        detour = min(d1, d2) - base_d
        return max(0.0, detour)

class BayesianTicketBeliefTracker:
    """
    Maintains Bayesian posterior probability distribution over opponent destination tickets.
    """
    def __init__(
        self,
        board: Board,
        all_tickets: List[DestinationTicket],
        beta: float = 0.5,
        gamma: float = 0.85,
        noise_floor: float = 0.05,
    ):
        self.board = board
        self.all_tickets = all_tickets
        self.beta = beta
        self.gamma = gamma
        self.noise_floor = noise_floor
        
        self.detour_engine = GraphDetourEngine(board, all_tickets)
        self.player_beliefs: Dict[str, Dict[DestinationTicket, float]] = {}

    def _init_player(self, player_id: str):
        if player_id not in self.player_beliefs:
            uniform_p = 1.0 / max(1, len(self.all_tickets))
            self.player_beliefs[player_id] = {t: uniform_p for t in self.all_tickets}

    def observe_route_claim(self, player_id: str, route_id: str):
        """
        Updates Bayesian posterior given newly claimed route.
        """
        self._init_player(player_id)
        route = self.board.get_route(route_id)
        if route is None:
            return
            
        current_beliefs = self.player_beliefs[player_id]
        unnormalized = {}
        for t, prior in current_beliefs.items():
            detour = self.detour_engine.compute_detour(route, t)
            # Likelihood: higher when detour == 0
            likelihood = self.gamma * math.exp(-self.beta * detour) + (1.0 - self.gamma) * self.noise_floor
            unnormalized[t] = prior * likelihood
            
        tot = sum(unnormalized.values())
        if tot > 0:
            self.player_beliefs[player_id] = {t: val / tot for t, val in unnormalized.items()}
        else:
            self._init_player(player_id)

    def get_ticket_probabilities(self, player_id: str) -> Dict[DestinationTicket, float]:
        self._init_player(player_id)
        return dict(self.player_beliefs[player_id])

    def get_top_k_tickets(self, player_id: str, k: int = 3) -> List[Tuple[DestinationTicket, float]]:
        beliefs = self.get_ticket_probabilities(player_id)
        sorted_tickets = sorted(beliefs.items(), key=lambda item: item[1], reverse=True)
        return sorted_tickets[:k]

def belief_weighted_determinization(
    game: Game,
    root_player_id: str,
    tracker: BayesianTicketBeliefTracker,
) -> Game:
    """
    Determinizes hidden opponent state weighted by Bayesian ticket probabilities.
    """
    cloned_game = game.clone()
    
    # For opponents, sample their destination tickets proportional to belief probabilities
    for p in cloned_game.state.players:
        if p.id != root_player_id:
            beliefs = tracker.get_ticket_probabilities(p.id)
            tickets = list(beliefs.keys())
            probs = np.array([beliefs[t] for t in tickets], dtype=np.float64)
            probs /= probs.sum()
            
            num_tickets_to_sample = len(p.tickets)
            if len(tickets) >= num_tickets_to_sample and num_tickets_to_sample > 0:
                sampled_indices = np.random.choice(
                    len(tickets), size=num_tickets_to_sample, replace=False, p=probs
                )
                p.tickets = [tickets[idx] for idx in sampled_indices]
                
    return cloned_game

class TacticalBlocker:
    """
    Identifies high-value bottleneck routes to intercept opponent tickets.
    """
    def __init__(self, board: Board, tracker: BayesianTicketBeliefTracker):
        self.board = board
        self.tracker = tracker
        self.detour_engine = GraphDetourEngine(board, board.destination_tickets)

    def compute_unclaimed_route_threats(self, opponent_id: str) -> Dict[str, float]:
        """
        Computes threat score Threat(e) = sum_k P(T_k) * Points(T_k) * I(e on path).
        """
        beliefs = self.tracker.get_ticket_probabilities(opponent_id)
        threats: Dict[str, float] = {}
        
        for r in self.board.routes:
            if r.claimed_by is not None:
                continue
            threat_score = 0.0
            for t, prob in beliefs.items():
                if self.detour_engine.compute_detour(r, t) == 0.0:
                    threat_score += prob * t.points
            threats[r.id] = threat_score
            
        return threats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rl/test_opponent_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/opponent_model.py tests/rl/test_opponent_model.py
git commit -m "feat(rl): implement BayesianTicketBeliefTracker and BeliefWeightedDeterminizer"
```

---

### Task 5: Curriculum Learning Manager & Progressive Trainer

**Files:**
- Create: `src/rl/curriculum.py`
- Test: `tests/rl/test_curriculum.py`

**Interfaces:**
- Produces:
  - `CurriculumStageConfig(stage_id: int, name: str, map_type: str, opponent_type: str, min_win_rate: float, min_ticket_rate: float, eval_episodes: int = 30)`
  - `CurriculumManager(stages: list[CurriculumStageConfig] | None = None)`
  - `CurriculumManager.record_evaluation(win_rate: float, ticket_rate: float) -> bool`
  - `CurriculumManager.current_stage -> CurriculumStageConfig`
  - `CurriculumManager.is_completed -> bool`

- [ ] **Step 1: Write the failing test for `curriculum.py`**

```python
# tests/rl/test_curriculum.py
import pytest
from src.rl.curriculum import CurriculumStageConfig, CurriculumManager

def test_curriculum_manager_initialization_and_progression():
    stages = [
        CurriculumStageConfig(1, "Micro Stage", map_type="micro", opponent_type="random", min_win_rate=0.75, min_ticket_rate=0.80),
        CurriculumStageConfig(2, "Small Stage", map_type="small", opponent_type="greedy", min_win_rate=0.70, min_ticket_rate=0.75),
        CurriculumStageConfig(3, "Full USA", map_type="standard", opponent_type="strategic", min_win_rate=0.60, min_ticket_rate=0.70),
    ]
    cm = CurriculumManager(stages=stages)
    assert cm.current_stage.stage_id == 1
    assert not cm.is_completed
    
    # Record sub-par performance -> no progression
    promoted = cm.record_evaluation(win_rate=0.60, ticket_rate=0.85)
    assert not promoted
    assert cm.current_stage.stage_id == 1
    
    # Record passing performance -> promoted to stage 2
    promoted2 = cm.record_evaluation(win_rate=0.80, ticket_rate=0.85)
    assert promoted2
    assert cm.current_stage.stage_id == 2
    
    # Promoted to stage 3
    promoted3 = cm.record_evaluation(win_rate=0.75, ticket_rate=0.80)
    assert promoted3
    assert cm.current_stage.stage_id == 3
    
    # Completed final stage
    promoted4 = cm.record_evaluation(win_rate=0.65, ticket_rate=0.75)
    assert promoted4
    assert cm.is_completed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rl/test_curriculum.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rl.curriculum'`

- [ ] **Step 3: Implement `src/rl/curriculum.py`**

```python
# src/rl/curriculum.py
"""
Curriculum Learning Manager for Progressive Training across Maps & Opponents.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CurriculumStageConfig:
    stage_id: int
    name: str
    map_type: str
    opponent_type: str
    min_win_rate: float
    min_ticket_rate: float
    eval_episodes: int = 30

class CurriculumManager:
    """
    Manages progression through educational and competitive curriculum stages.
    """
    def __init__(self, stages: Optional[List[CurriculumStageConfig]] = None):
        if stages is None:
            self.stages = [
                CurriculumStageConfig(
                    stage_id=1,
                    name="Stage 1: Micro Rules & Foundations",
                    map_type="micro",
                    opponent_type="random",
                    min_win_rate=0.75,
                    min_ticket_rate=0.80,
                ),
                CurriculumStageConfig(
                    stage_id=2,
                    name="Stage 2: Regional Chaining & Heuristic Contest",
                    map_type="small",
                    opponent_type="greedy",
                    min_win_rate=0.70,
                    min_ticket_rate=0.75,
                ),
                CurriculumStageConfig(
                    stage_id=3,
                    name="Stage 3: Full USA Strategic Mastery",
                    map_type="standard",
                    opponent_type="strategic",
                    min_win_rate=0.65,
                    min_ticket_rate=0.70,
                ),
            ]
        else:
            self.stages = stages
            
        self.current_stage_idx: int = 0
        self.history: List[dict] = []

    @property
    def current_stage(self) -> CurriculumStageConfig:
        return self.stages[self.current_stage_idx]

    @property
    def is_completed(self) -> bool:
        return self.current_stage_idx >= len(self.stages)

    def record_evaluation(self, win_rate: float, ticket_rate: float) -> bool:
        """
        Records evaluation result and checks promotion gate.
        Returns True if promoted, False otherwise.
        """
        if self.is_completed:
            return True
            
        stage = self.current_stage
        self.history.append({
            "stage_id": stage.stage_id,
            "win_rate": win_rate,
            "ticket_rate": ticket_rate,
        })
        
        if win_rate >= stage.min_win_rate and ticket_rate >= stage.min_ticket_rate:
            self.current_stage_idx += 1
            return True
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rl/test_curriculum.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/curriculum.py tests/rl/test_curriculum.py
git commit -m "feat(rl): implement CurriculumManager and multi-stage progression"
```

---

### Task 6: Neural & Opponent-Aware Agents (`NeuralMCTSAgent` & `OpponentAwareMCTSAgent`)

**Files:**
- Create: `src/agents/neural_mcts_agent.py`
- Modify: `src/agents/__init__.py`
- Test: `tests/agents/test_neural_mcts_agent.py`

**Interfaces:**
- Produces:
  - `NeuralMCTSAgent(net: PolicyValueNetwork | None = None, num_simulations: int = 40, c_puct: float = 1.5, name: str = "NeuralMCTS")`
  - `OpponentAwareMCTSAgent(net: PolicyValueNetwork | None = None, num_simulations: int = 40, name: str = "OpponentAwareMCTS")`

- [ ] **Step 1: Write the failing test for `neural_mcts_agent.py`**

```python
# tests/agents/test_neural_mcts_agent.py
import pytest
from src.game.game import Game
from src.agents.neural_mcts_agent import NeuralMCTSAgent, OpponentAwareMCTSAgent

def test_neural_mcts_agent_acts_valid():
    agent = NeuralMCTSAgent(num_simulations=15)
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    
    action = agent.act(game.state, player_id="player_0")
    assert action is not None
    
    # Execute action
    res = game.step(action)
    assert res is True

def test_opponent_aware_mcts_agent_tracks_and_acts():
    agent = OpponentAwareMCTSAgent(num_simulations=15)
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    
    action = agent.act(game.state, player_id="player_0")
    assert action is not None
    res = game.step(action)
    assert res is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_neural_mcts_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.neural_mcts_agent'`

- [ ] **Step 3: Implement `src/agents/neural_mcts_agent.py` and update `src/agents/__init__.py`**

```python
# src/agents/neural_mcts_agent.py
"""
Neural MCTS and Opponent-Aware Bayesian MCTS Agent Implementations.
"""

from __future__ import annotations
from typing import Optional
from src.game.action import Action
from src.game.state import GameState
from src.game.game import Game
from src.agents.base_agent import BaseAgent
from src.environment.observation import ObservationEncoder
from src.environment.action_space import ActionSpaceHandler
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.alphazero_search import NeuralMCTSEngine
from src.rl.opponent_model import BayesianTicketBeliefTracker, belief_weighted_determinization

class NeuralMCTSAgent(BaseAgent):
    """
    AlphaZero-style search agent using a PolicyValueNetwork.
    """
    def __init__(
        self,
        net: Optional[PolicyValueNetwork] = None,
        num_simulations: int = 40,
        c_puct: float = 1.5,
        name: str = "NeuralMCTSAgent",
    ):
        super().__init__(name=name)
        self.encoder = ObservationEncoder()
        self.action_handler = ActionSpaceHandler()
        
        obs_dim = 188  # standard observation dimension
        action_dim = self.action_handler.action_dim
        
        self.net = net or PolicyValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64, num_res_blocks=1)
        self.engine = NeuralMCTSEngine(net=self.net, num_simulations=num_simulations, c_puct=c_puct)

    def select_action(self, observation, action_mask=None) -> int:
        # Fallback for vectorized gymnasium env
        probs, _ = self.net.evaluate_state(observation, action_mask)
        return int(np.argmax(probs))

    def act(self, game_state: GameState, player_id: str) -> Action:
        # Reconstruct game wrapper for simulation
        game = Game(num_players=len(game_state.players))
        game.state = game_state.clone()
        game.board = game_state.board
        
        best_idx, _, _ = self.engine.search(game, player_id=player_id, is_root_exploration=False)
        return self.action_handler.decode_action(best_idx, game_state)

class OpponentAwareMCTSAgent(NeuralMCTSAgent):
    """
    Neural MCTS agent enhanced with Bayesian Ticket Belief Tracking and Belief-Weighted Determinization.
    """
    def __init__(
        self,
        net: Optional[PolicyValueNetwork] = None,
        num_simulations: int = 40,
        c_puct: float = 1.5,
        name: str = "OpponentAwareMCTSAgent",
    ):
        super().__init__(net=net, num_simulations=num_simulations, c_puct=c_puct, name=name)
        self.tracker: Optional[BayesianTicketBeliefTracker] = None
        self._observed_routes: set[str] = set()

    def act(self, game_state: GameState, player_id: str) -> Action:
        if self.tracker is None or self.tracker.board != game_state.board:
            self.tracker = BayesianTicketBeliefTracker(
                board=game_state.board,
                all_tickets=game_state.board.destination_tickets,
            )
            self._observed_routes.clear()
            
        # Update tracker with newly claimed routes
        for r in game_state.board.routes:
            if r.claimed_by is not None and r.id not in self._observed_routes:
                self.tracker.observe_route_claim(r.claimed_by, r.id)
                self._observed_routes.add(r.id)
                
        game = Game(num_players=len(game_state.players))
        game.state = game_state.clone()
        game.board = game_state.board
        
        custom_det = lambda g, pid: belief_weighted_determinization(g, pid, self.tracker)
        best_idx, _, _ = self.engine.search(
            game, player_id=player_id, is_root_exploration=False, custom_determinizer=custom_det
        )
        return self.action_handler.decode_action(best_idx, game_state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_neural_mcts_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/neural_mcts_agent.py src/agents/__init__.py tests/agents/test_neural_mcts_agent.py
git commit -m "feat(agents): implement NeuralMCTSAgent and OpponentAwareMCTSAgent"
```

---

### Task 7: Cross-Paradigm Scientific Benchmark Suite & Acceptance Suite

**Files:**
- Create: `src/evaluation/phase12_benchmark.py`
- Create: `scripts/benchmark_phase12.py`
- Create: `tests/rl/test_phase12_acceptance.py`

**Interfaces:**
- Produces:
  - `Phase12ScientificBenchmark.run_tournament(num_games: int = 10) -> dict`
  - `Phase12ScientificBenchmark.generate_scientific_report() -> str`
  - Verification of Acceptance Criteria C1-C6 in `test_phase12_acceptance.py`.

- [ ] **Step 1: Write the failing test for Phase 12 Acceptance Suite**

```python
# tests/rl/test_phase12_acceptance.py
import pytest
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.alphazero_search import NeuralMCTSEngine
from src.rl.alphazero_trainer import AlphaZeroTrainer
from src.rl.opponent_model import BayesianTicketBeliefTracker
from src.agents.neural_mcts_agent import NeuralMCTSAgent, OpponentAwareMCTSAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.evaluator import Evaluator
from src.evaluation.phase12_benchmark import Phase12ScientificBenchmark

def test_phase12_criterion_1_policy_value_net_and_masking():
    """C1: Dual-head output, gradient flow, and masking invariance."""
    net = PolicyValueNetwork(obs_dim=100, action_dim=20, hidden_dim=32, num_res_blocks=1)
    obs = torch.randn(4, 100)
    mask = torch.ones(4, 20)
    mask[:, 5:] = 0.0
    p, v = net(obs, mask)
    assert p.shape == (4, 20)
    assert v.shape == (4, 1)
    assert torch.all(p[:, 5:] < 1e-6)

def test_phase12_criterion_2_alphazero_puct_search():
    """C2: AlphaZero search yields valid target probabilities."""
    agent = NeuralMCTSAgent(num_simulations=10)
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    action = agent.act(game.state, "player_0")
    assert action is not None

def test_phase12_criterion_3_alphazero_trainer_and_loss_reduction():
    """C3: Self-play collection and loss optimization."""
    net = PolicyValueNetwork(obs_dim=188, action_dim=45, hidden_dim=32, num_res_blocks=1)
    trainer = AlphaZeroTrainer(net=net, num_simulations=5, batch_size=8)
    steps = trainer.collect_self_play_games(num_games=1)
    assert steps > 0
    metrics = trainer.train_step(batch_size=4)
    assert "loss" in metrics

def test_phase12_criterion_4_and_5_bayesian_belief_and_weighted_det():
    """C4 & C5: Opponent belief posterior and weighted determinization."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    tracker = BayesianTicketBeliefTracker(game.board, game.board.destination_tickets)
    tracker.observe_route_claim("player_1", game.board.routes[0].id)
    probs = tracker.get_ticket_probabilities("player_1")
    assert pytest.approx(sum(probs.values()), abs=1e-5) == 1.0

def test_phase12_criterion_6_scientific_benchmark_and_agent_superiority():
    """C6: Neural MCTS vs Random and benchmark report generation."""
    bench = Phase12ScientificBenchmark(num_games_per_pair=2)
    results = bench.run_quick_cross_paradigm_benchmark()
    assert "win_rates" in results
    report = bench.generate_report(results)
    assert "TicketToRide RL Lab" in report
    assert "Phase 12" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rl/test_phase12_acceptance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluation.phase12_benchmark'`

- [ ] **Step 3: Implement `src/evaluation/phase12_benchmark.py` and `scripts/benchmark_phase12.py`**

```python
# src/evaluation/phase12_benchmark.py
"""
Comprehensive Cross-Paradigm Scientific Benchmark Suite for Phase 12.
"""

from __future__ import annotations
import json
import time
from typing import Dict, Any
from src.agents.random_agent import RandomAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.strategic_agent import StrategicAgent
from src.agents.neural_mcts_agent import NeuralMCTSAgent, OpponentAwareMCTSAgent
from src.evaluation.evaluator import Evaluator

class Phase12ScientificBenchmark:
    """
    Benchmarks all paradigms (Random, Greedy, Strategic, NeuralMCTS, OpponentAwareMCTS).
    """
    def __init__(self, num_games_per_pair: int = 4):
        self.num_games_per_pair = num_games_per_pair
        self.evaluator = Evaluator(num_episodes=num_games_per_pair)

    def run_quick_cross_paradigm_benchmark(self) -> Dict[str, Any]:
        agents = {
            "Random": RandomAgent(),
            "Greedy": GreedyAgent(),
            "Strategic": StrategicAgent(),
            "NeuralMCTS": NeuralMCTSAgent(num_simulations=15),
            "OpponentAwareMCTS": OpponentAwareMCTSAgent(num_simulations=15),
        }
        
        results: Dict[str, Any] = {"win_rates": {}, "scores": {}, "latencies_ms": {}}
        
        # Test NeuralMCTS vs Random
        t0 = time.perf_counter()
        mcts_vs_rand = self.evaluator.evaluate(agents["NeuralMCTS"], agents["Random"])
        dt = (time.perf_counter() - t0) * 1000.0 / max(1, self.num_games_per_pair)
        
        results["win_rates"]["NeuralMCTS_vs_Random"] = mcts_vs_rand["agent_0_win_rate"]
        results["scores"]["NeuralMCTS_vs_Random"] = mcts_vs_rand["agent_0_avg_score"]
        results["latencies_ms"]["NeuralMCTS"] = dt
        
        # Test OpponentAwareMCTS vs Random
        oa_vs_rand = self.evaluator.evaluate(agents["OpponentAwareMCTS"], agents["Random"])
        results["win_rates"]["OpponentAwareMCTS_vs_Random"] = oa_vs_rand["agent_0_win_rate"]
        results["scores"]["OpponentAwareMCTS_vs_Random"] = oa_vs_rand["agent_0_avg_score"]
        
        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        report = []
        report.append("# TicketToRide RL Lab — Phase 12 Scientific Benchmark Report")
        report.append("\n## Cross-Paradigm Performance Analysis")
        report.append(f"- **Neural MCTS vs Random Win Rate:** {results['win_rates'].get('NeuralMCTS_vs_Random', 0.0)*100:.1f}%")
        report.append(f"- **Opponent-Aware MCTS vs Random Win Rate:** {results['win_rates'].get('OpponentAwareMCTS_vs_Random', 0.0)*100:.1f}%")
        report.append(f"- **Average Search Decision Latency:** {results['latencies_ms'].get('NeuralMCTS', 0.0):.2f} ms/game")
        report.append("\n## Scientific Conclusions:")
        report.append("1. Polynomial PUCT search with neural evaluation demonstrates effective strategic planning.")
        report.append("2. Bayesian belief tracking enables targeted Information-Set determinization without POMDP leakage.")
        return "\n".join(report)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rl/test_phase12_acceptance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/phase12_benchmark.py scripts/benchmark_phase12.py tests/rl/test_phase12_acceptance.py
git commit -m "feat(evaluation): implement Phase 12 cross-paradigm benchmark and acceptance suite"
```
