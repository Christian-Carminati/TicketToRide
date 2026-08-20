# Phase 7: Reward Research & Behavioral Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement modular versioned reward calculators (V1 Sparse, V2 Dense Routes, V3 Ticket Milestones, V4 Strategic Shaped, Custom), reward component telemetry, a behavioral game profiler, and an automated multi-reward research runner with scientific report generation.

**Architecture:** Build a clean hierarchy of `BaseRewardCalculator` subclasses registered in `RewardFactory`. Integrate component-level breakdown into `TicketToRideEnv` info dictionaries and `ExperimentConfig`. Build `BehavioralEvaluator` to extract strategic game metrics and `RewardResearchRunner` to automate A/B testing of reward formulations.

**Tech Stack:** Python 3.12+, PyTorch, Gymnasium, NumPy, Pydantic, Pytest.

**Spec:** [docs/superpowers/specs/2026-08-20-phase-7-reward-research-design.md](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-20-phase-7-reward-research-design.md)

## Global Constraints
- Python 3.12+ compatibility with strict type annotations.
- Game Core (`src/game/`) remains pure and untouched.
- All experiments and evaluation must be deterministic given random seeds.
- Component breakdown `get_components(...)` must sum to the step reward calculated by `calculate(...)` within float precision.
- No regression across existing test suites (Game, Agents, Environment, PPO, DQN, Web Lab).

---

### Task 1: Modular Reward Engines & Factory

**Files:**
- Create: `tests/environment/test_reward_versions.py`
- Modify: `src/environment/reward.py`

**Interfaces:**
- Consumes: `GameState`, `Action`, `Board`, `GameRules`, `check_tickets_completed_batch`
- Produces:
  - `BaseRewardCalculator.calculate(prev_state, action, next_state, player_index) -> float`
  - `BaseRewardCalculator.get_components(prev_state, action, next_state, player_index) -> dict[str, float]`
  - `RewardV1_Sparse`, `RewardV2_DenseRoutes`, `RewardV3_TicketMilestones`, `RewardV4_StrategicShaped`, `CustomRewardCalculator`
  - `RewardFactory.create(version_or_name, board=None, weights=None) -> BaseRewardCalculator`

- [ ] **Step 1: Write the failing tests for reward versions and component breakdown**

```python
# tests/environment/test_reward_versions.py
import pytest
from src.environment.reward import (
    RewardFactory,
    RewardV1_Sparse,
    RewardV2_DenseRoutes,
    RewardV3_TicketMilestones,
    RewardV4_StrategicShaped,
    CustomRewardCalculator,
    RewardWeights,
)
from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.game import Game
from src.game.maps import create_synthetic_mini_board


def test_reward_factory_aliases():
    board, _ = create_synthetic_mini_board()
    r1 = RewardFactory.create("sparse", board=board)
    assert isinstance(r1, RewardV1_Sparse)

    r2 = RewardFactory.create("dense_routes", board=board)
    assert isinstance(r2, RewardV2_DenseRoutes)

    r3 = RewardFactory.create("ticket_milestones", board=board)
    assert isinstance(r3, RewardV3_TicketMilestones)

    r4 = RewardFactory.create("strategic", board=board)
    assert isinstance(r4, RewardV4_StrategicShaped)

    rc = RewardFactory.create("custom", board=board, weights=RewardWeights(win_bonus=50.0))
    assert isinstance(rc, CustomRewardCalculator)


def test_reward_v1_sparse_intermediate_and_terminal():
    board, tickets = create_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=42)
    calc = RewardV1_Sparse(board=board)

    prev = game.state
    # Draw card action
    action = Action(action_type=ActionType.DRAW_CARD_DECK)
    game.step(action)
    next_s = game.state

    step_reward = calc.calculate(prev, action, next_s, player_index=0)
    components = calc.get_components(prev, action, next_s, player_index=0)
    assert step_reward == 0.0
    assert components["intermediate_step"] == 0.0


def test_reward_v2_dense_routes_claim_reward():
    board, tickets = create_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=42)
    calc = RewardV2_DenseRoutes(board=board)

    # Give player 0 enough cards and claim route
    p = game.state.players[0]
    p.cards["red"] = 5
    route = board.routes[0]
    action = Action(action_type=ActionType.CLAIM_ROUTE, route_id=route.id, color_chosen="red")
    prev = game.state
    game.step(action)
    next_s = game.state

    reward = calc.calculate(prev, action, next_s, player_index=0)
    components = calc.get_components(prev, action, next_s, player_index=0)
    assert reward > 0.0
    assert components["route_points"] > 0.0
    assert pytest.approx(sum(components.values())) == reward


def test_reward_v3_ticket_milestones():
    board, tickets = create_synthetic_mini_board()
    calc = RewardV3_TicketMilestones(board=board)
    game = Game(board=board, tickets_deck=tickets, num_players=2)
    game.reset(seed=42)

    # Claim a route completing a ticket
    p = game.state.players[0]
    p.cards["red"] = 10
    p.cards["blue"] = 10
    p.cards["green"] = 10
    prev = game.state

    # Claim route
    route = board.routes[0]
    action = Action(action_type=ActionType.CLAIM_ROUTE, route_id=route.id, color_chosen="red")
    game.step(action)
    next_s = game.state

    components = calc.get_components(prev, action, next_s, player_index=0)
    assert "ticket_completion" in components
    assert "route_points" in components
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/environment/test_reward_versions.py -v`
Expected: FAIL (classes not yet defined)

- [ ] **Step 3: Implement modular reward classes and RewardFactory in `src/environment/reward.py`**

Refactor `src/environment/reward.py` to provide:
- `BaseRewardCalculator` with `calculate` and `get_components`.
- `RewardWeights` dataclass with all configurable parameters.
- `RewardV1_Sparse`: purely terminal.
- `RewardV2_DenseRoutes`: immediate route points + step penalty + terminal win/loss/tickets.
- `RewardV3_TicketMilestones`: route points + ticket completion milestones + step penalty + terminal fail penalty.
- `RewardV4_StrategicShaped`: balanced formulation with score diff scaling and route efficiency.
- `CustomRewardCalculator`: uses custom `RewardWeights`.
- `DefaultRewardCalculator`: alias/backward-compatible wrapper for V4 / default.
- `RewardFactory`: handles string, int, and config instantiations.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/environment/test_reward_versions.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add src/environment/reward.py tests/environment/test_reward_versions.py
git commit -m "feat(reward): implement modular versioned reward calculators and RewardFactory"
```

---

### Task 2: Environment & Configuration Integration

**Files:**
- Modify: `src/environment/env.py`
- Modify: `src/experiments/config.py`
- Modify: `src/experiments/runner.py`
- Modify: `tests/environment/test_environment.py`

**Interfaces:**
- Consumes: `RewardFactory`, `RewardConfig`, `EnvironmentConfig`
- Produces:
  - `TicketToRideEnv` populated with `info["reward_components"]`
  - `EnvironmentConfig.reward_config` supported in YAML and experiment orchestration

- [ ] **Step 1: Write tests for env reward components telemetry and config integration**

Add tests in `tests/environment/test_environment.py` checking that:
- `env.step(action)` returns `info["reward_components"]` containing float breakdowns.
- `TicketToRideEnv(reward_calculator="sparse")` or `"ticket_milestones"` initializes correctly using `RewardFactory`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/environment/test_environment.py -k "reward" -v`
Expected: FAIL

- [ ] **Step 3: Update `TicketToRideEnv`, `EnvironmentConfig`, and `ExperimentRunner`**

- In `src/experiments/config.py`, add `RewardConfig` and allow `reward_version: int | str = 1` or `reward_config: RewardConfig | None`.
- In `src/environment/env.py`, allow `reward_calculator: BaseRewardCalculator | str | int | None`. In `step()`, populate `info["reward_components"] = self.reward_calc.get_components(...)`.
- In `src/experiments/runner.py`, instantiate `reward_calc = RewardFactory.create(self.config.environment.reward_version, board=board, weights=...)` and pass to `TicketToRideEnv`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/environment/test_environment.py tests/experiments/test_experiments.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 2**

```bash
git add src/environment/env.py src/experiments/config.py src/experiments/runner.py tests/environment/test_environment.py
git commit -m "feat(env): integrate reward factory, telemetry components and experiment config"
```

---

### Task 3: Behavioral Profiler & Strategic Metrics

**Files:**
- Create: `src/evaluation/behavioral.py`
- Create: `tests/evaluation/test_behavioral.py`

**Interfaces:**
- Consumes: `GameState`, `Board`, `Action`, `BaseAgent`
- Produces:
  - `BehavioralProfile` dataclass containing `win_rate`, `avg_score`, `score_differential`, `avg_routes_claimed`, `avg_route_length`, `route_efficiency`, `ticket_completion_rate`, `tickets_completed_avg`, `ticket_penalty_avg`, `avg_game_turns`, `cards_drawn_ratio`
  - `BehavioralEvaluator.profile_agent(agent, opponent, num_games=20, seed=42) -> BehavioralProfile`

- [ ] **Step 1: Write tests for `BehavioralEvaluator`**

```python
# tests/evaluation/test_behavioral.py
import pytest
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.behavioral import BehavioralEvaluator, BehavioralProfile
from src.game.maps import create_synthetic_mini_board


def test_behavioral_evaluator_metrics():
    board, tickets = create_synthetic_mini_board()
    evaluator = BehavioralEvaluator(board=board, tickets_deck=tickets, seed=42)
    agent = GreedyAgent(name="GreedyTest")
    opp = RandomAgent(name="RandomTest")

    profile = evaluator.profile_agent(agent=agent, opponent=opp, num_games=10)

    assert isinstance(profile, BehavioralProfile)
    assert 0.0 <= profile.win_rate <= 1.0
    assert profile.avg_routes_claimed >= 0.0
    assert profile.avg_route_length >= 0.0
    assert profile.route_efficiency >= 0.0
    assert 0.0 <= profile.ticket_completion_rate <= 1.0
    assert profile.avg_game_turns > 0.0
    assert 0.0 <= profile.cards_drawn_ratio <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/evaluation/test_behavioral.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `BehavioralEvaluator` and `BehavioralProfile` in `src/evaluation/behavioral.py`**

- Play $N$ games between `agent` and `opponent` alternating starter.
- Track per-game and per-step actions (claiming routes, drawing cards, drawing tickets).
- Calculate:
  - `ticket_completion_rate` = $\frac{\sum \text{completed}}{\sum \text{kept}}$
  - `route_efficiency` = $\frac{\text{total route points}}{\text{total trains consumed}}$
  - `avg_route_length` = $\frac{\sum \text{length}}{\text{number of routes}}$
  - `cards_drawn_ratio` = $\frac{\text{card draw actions}}{\text{total actions}}$
  - Duration in turns and score differential.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/evaluation/test_behavioral.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 3**

```bash
git add src/evaluation/behavioral.py tests/evaluation/test_behavioral.py
git commit -m "feat(evaluation): add BehavioralEvaluator for strategic profiling"
```

---

### Task 4: Reward Research Runner & Automated Scientific Report Engine

**Files:**
- Create: `src/evaluation/reward_research.py`
- Create: `scripts/reward_research.py`
- Create: `tests/evaluation/test_reward_research.py`

**Interfaces:**
- Consumes: `RewardFactory`, `BehavioralEvaluator`, `MaskedPPOTrainer`, `PPOAgent`, `RandomAgent`, `GreedyAgent`, `StrategicAgent`
- Produces:
  - `RewardResearchRunner.run_study(...) -> dict[str, Any]`
  - JSON and Markdown report generation (`experiments/results/reward_research.json` & `.md`)
  - CLI entry point `scripts/reward_research.py`

- [ ] **Step 1: Write tests for `RewardResearchRunner` and report generation**

```python
# tests/evaluation/test_reward_research.py
from pathlib import Path
import pytest
from src.evaluation.reward_research import RewardResearchRunner


def test_reward_research_study_quick_execution(tmp_path: Path):
    json_path = tmp_path / "study.json"
    md_path = tmp_path / "study.md"

    runner = RewardResearchRunner(
        board_type="mini",
        reward_versions=["v1", "v2", "v3", "v4"],
        training_steps=300,
        eval_games=4,
        output_json=str(json_path),
        output_md=str(md_path),
        seed=42,
    )

    results = runner.run_study()

    assert "v1" in results["reward_studies"]
    assert "v2" in results["reward_studies"]
    assert "v3" in results["reward_studies"]
    assert "v4" in results["reward_studies"]
    assert json_path.exists()
    assert md_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "# Reward Research & Behavioral Benchmarking Report" in md_content
    assert "Ticket Completion" in md_content
    assert "Route Efficiency" in md_content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/evaluation/test_reward_research.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `RewardResearchRunner` and CLI script**

- `RewardResearchRunner` trains PPO policies under each reward version using identical seeds and hyperparameter baselines.
- Evaluates each policy against standard baselines with `BehavioralEvaluator`.
- Builds structured summary tables comparing:
  - Win Rate vs Baselines
  - Average Score & Differential
  - Ticket Completion Rate & Ticket Penalty
  - Route Count, Average Length & Efficiency
  - Game Length & Cards Drawn Ratio
- Formats markdown analysis discussing credit assignment, ticket prioritization, and route greed.
- Write CLI `scripts/reward_research.py` with `argparse`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/evaluation/test_reward_research.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 4**

```bash
git add src/evaluation/reward_research.py scripts/reward_research.py tests/evaluation/test_reward_research.py
git commit -m "feat(research): implement RewardResearchRunner and scientific reporting CLI"
```

---

### Task 5: Phase 7 Acceptance Test & Full System Validation

**Files:**
- Create: `tests/rl/test_phase7_acceptance.py`

**Interfaces:**
- Validates the entire Phase 7 deliverable against [DESIGN.md](file:///home/christian/Projects/Python/TicketToRide/DESIGN.md#L1502-L1513) acceptance criteria.

- [ ] **Step 1: Write Phase 7 Acceptance Test Suite**

```python
# tests/rl/test_phase7_acceptance.py
import pytest
from pathlib import Path
from src.evaluation.reward_research import RewardResearchRunner
from src.environment.reward import RewardFactory, RewardV1_Sparse, RewardV2_DenseRoutes, RewardV3_TicketMilestones, RewardV4_StrategicShaped


def test_phase7_reward_hierarchy_and_factories():
    """Acceptance Criterion 1: All reward versions exist and support component telemetry."""
    for v in ["v1", "v2", "v3", "v4"]:
        calc = RewardFactory.create(v)
        assert calc is not None


def test_phase7_automated_reward_comparison_study(tmp_path: Path):
    """Acceptance Criterion 2: Automated study compares reward functions and quantifies behavioral divergence."""
    json_path = tmp_path / "acceptance_report.json"
    md_path = tmp_path / "acceptance_report.md"

    runner = RewardResearchRunner(
        board_type="mini",
        reward_versions=["v1", "v2", "v3", "v4"],
        training_steps=500,
        eval_games=6,
        output_json=str(json_path),
        output_md=str(md_path),
        seed=123,
    )
    results = runner.run_study()

    assert len(results["reward_studies"]) == 4
    assert json_path.exists()
    assert md_path.exists()

    # Verify structured metrics exist for all versions
    for v_key, study in results["reward_studies"].items():
        assert "vs_random" in study
        assert "ticket_completion_rate" in study["vs_random"]
        assert "route_efficiency" in study["vs_random"]
        assert "avg_routes_claimed" in study["vs_random"]
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/pytest tests/rl/test_phase7_acceptance.py -v`
Expected: PASS

- [ ] **Step 3: Run entire test suite to ensure zero regressions**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS across all test modules.

- [ ] **Step 4: Commit Task 5**

```bash
git add tests/rl/test_phase7_acceptance.py
git commit -m "test(acceptance): add Phase 7 reward research acceptance test suite"
```
