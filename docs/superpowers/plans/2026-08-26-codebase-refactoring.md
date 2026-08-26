# Codebase Refactoring & Type Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a clean, comprehensive refactoring of TicketToRide RL Lab to achieve 100% type safety (0 mypy errors), 0 ruff lint violations, standardized `BaseAgent` interfaces, and a centralized `TrainerFactory` eliminating code duplication.

**Architecture:** Standardize the Gymnasium agent interface in `BaseAgent`, type guard Gymnasium spaces, correct union types, extract a centralized `TrainerFactory` in `src/rl/factory.py` for both CLI `runner.py` and API `trainer_service.py`, and clean up all linting/import issues.

**Tech Stack:** Python 3.12, PyTorch, Gymnasium, FastAPI, Pydantic, Mypy, Ruff, Pytest.

---

### Task 1: Linting & Dead Imports Cleanup with Ruff

**Files:**
- Modify: All source and test files with linting/import issues.

**Interfaces:**
- Input/Output: Formats and removes unused imports according to Ruff configuration in `pyproject.toml`.

- [ ] **Step 1: Run automated ruff fixes**
Run: `./.venv/bin/ruff check --fix src tests`
- [ ] **Step 2: Verify ruff passes cleanly or manual fixes for remaining errors**
Run: `./.venv/bin/ruff check src tests`

---

### Task 2: Dependency Stubs & Type Config in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Config: Add `types-PyYAML` to `dev` optional-dependencies and ensure Mypy configuration is properly configured.

- [ ] **Step 1: Update `pyproject.toml` with `types-PyYAML>=6.0.12`**
- [ ] **Step 2: Install stub package in `.venv`**
Run: `./.venv/bin/pip install types-PyYAML`

---

### Task 3: BaseAgent Interface Standardization & Subclass Alignment

**Files:**
- Modify: `src/agents/base_agent.py`
- Modify: `src/agents/base.py`
- Modify: `src/agents/recurrent_ppo_agent.py`
- Modify: `src/agents/ppo_agent.py`
- Modify: `src/agents/dqn_agent.py`
- Modify: `src/agents/neural_mcts_agent.py`
- Modify: `src/agents/greedy_agent.py`
- Modify: `src/agents/strategic_agent.py`
- Modify: `src/agents/random_agent.py`
- Test: `tests/agents/test_agents.py`

**Interfaces:**
- Produces: Standardized `select_action(self, observation: np.ndarray, action_mask: np.ndarray | None = None, deterministic: bool = True, info: dict[str, Any] | None = None) -> int` across all agents.

- [ ] **Step 1: Update `BaseAgent.select_action` signature and docstring**
- [ ] **Step 2: Update `RecurrentPPOAgent.select_action` to use `observation` parameter matching `BaseAgent`**
- [ ] **Step 3: Update `DQNAgent`, `PPOAgent`, and heuristic agents to respect the common signature**
- [ ] **Step 4: Run agent tests to verify compatibility**
Run: `./.venv/bin/pytest tests/agents/`

---

### Task 4: Type Safety & Mypy Error Resolution

**Files:**
- Modify: `src/rl/ppo.py`
- Modify: `src/rl/dqn.py`
- Modify: `src/rl/lstm_ppo.py`
- Modify: `src/rl/rollout.py`
- Modify: `src/rl/replay_buffer.py`
- Modify: `src/rl/self_play.py`
- Modify: `src/rl/alphazero_search.py`
- Modify: `src/evaluation/behavioral.py`
- Modify: `src/evaluation/generalization.py`
- Modify: `src/evaluation/pomdp_benchmark.py`
- Modify: `src/game/procedural.py`
- Modify: `src/game/rules.py`
- Modify: `src/environment/reward.py`
- Modify: `src/api/brain_service.py`
- Modify: `src/api/trainer_service.py`
- Modify: `src/experiments/runner.py`

**Interfaces:**
- Produces: 100% type-checked Python codebase satisfying `mypy src`.

- [ ] **Step 1: Fix Gym space type guards (Discrete action space and observation shapes)**
- [ ] **Step 2: Fix `behavioral.py` Route list narrowing and union types**
- [ ] **Step 3: Fix `rollout.py` and `replay_buffer.py` array/tensor typing and indexing**
- [ ] **Step 4: Fix `self_play.py` and `brain_service.py` model assignment typing**
- [ ] **Step 5: Run mypy to verify 0 errors**
Run: `./.venv/bin/mypy src`

---

### Task 5: Extraction of Centralized `TrainerFactory` (DRY)

**Files:**
- Create: `src/rl/factory.py`
- Modify: `src/experiments/runner.py`
- Modify: `src/api/trainer_service.py`
- Test: `tests/rl/test_factory.py`

**Interfaces:**
- Produces: `create_trainer(algorithm: str, env: Any, config: dict[str, Any], seed: int = 42) -> BaseTrainer`

- [ ] **Step 1: Write test for `TrainerFactory`**
- [ ] **Step 2: Implement `src/rl/factory.py`**
- [ ] **Step 3: Refactor `src/experiments/runner.py` to use `create_trainer`**
- [ ] **Step 4: Refactor `src/api/trainer_service.py` to use `create_trainer`**
- [ ] **Step 5: Run experiment and trainer tests**
Run: `./.venv/bin/pytest tests/experiments/ tests/api/test_trainer_service.py tests/test_pfsp_training.py`

---

### Task 6: Final Full-Stack Verification

**Files:**
- All files across the repository.

- [ ] **Step 1: Run `ruff check src tests`**
- [ ] **Step 2: Run `mypy src`**
- [ ] **Step 3: Run `pytest` full test suite**
- [ ] **Step 4: Run `npm run build` in `frontend/`**
