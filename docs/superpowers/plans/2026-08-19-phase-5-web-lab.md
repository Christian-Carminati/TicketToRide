# Phase 5: Web Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete Web Lab subsystem for TicketToRide RL Lab, featuring a FastAPI REST & WebSocket streaming backend, a modular React 18/TypeScript SPA with SVG vector board rendering, neural network brain introspection, live training telemetry curves, a frame-by-frame replay player, and experiment comparison tools.

**Architecture:** A decoupled, asynchronous architecture where the deterministic Game Core and PyTorch RL Engine remain completely headless-capable, communicating with a modern React SPA via typed REST endpoints and non-blocking WebSocket event streams (`/ws/telemetry`).

**Tech Stack:** Python 3.12+, FastAPI, WebSockets, Pydantic v2, PyTorch, React 18, TypeScript, Vite, Lucide-react.

**Spec:** [docs/superpowers/specs/2026-08-19-phase-5-web-lab-design.md](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-19-phase-5-web-lab-design.md)

## Global Constraints

- Game Core (`src/game/`) must never import FastAPI, Torch, or Gymnasium.
- Script training and evaluation (`scripts/train.py`, `scripts/evaluate.py`, `scripts/tournament.py`) must remain 100% functional in headless CLI mode without requiring the web server.
- All backend REST & WebSocket models must use Pydantic v2.
- All frontend components must use TypeScript with strict types without `any`.
- All Python tests must pass with pytest in the virtual environment `.venv/bin/pytest`.

---

### Task 1: API Schemas & DTO Models

**Files:**
- Create: `tests/api/test_schemas.py`
- Modify: `src/api/schemas.py`

**Interfaces:**
- Produces: `ActionDTO`, `PlayerStateDTO`, `GameStateDTO`, `GameSessionCreateRequest`, `GameStepRequest`, `LayerActivationDTO`, `BrainInspectionDTO`, `TrainingStartRequest`, `TrainingStatusDTO`, `TelemetryEventDTO`, `ReplayFrameDTO`, `ReplayDetailDTO`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_schemas.py
from src.api.schemas import (
    ActionDTO,
    PlayerStateDTO,
    GameStateDTO,
    GameSessionCreateRequest,
    GameStepRequest,
    LayerActivationDTO,
    BrainInspectionDTO,
    TrainingStartRequest,
    TrainingStatusDTO,
    TelemetryEventDTO,
    ReplayFrameDTO,
    ReplayDetailDTO,
)


def test_game_dto_serialization():
    action = ActionDTO(action_type="CLAIM_ROUTE", route_id="r_0_bos_ny", card_color="RED")
    assert action.action_type == "CLAIM_ROUTE"
    assert action.route_id == "r_0_bos_ny"

    player = PlayerStateDTO(
        player_id="p0",
        name="Agent 1",
        score=15,
        trains_remaining=35,
        cards_in_hand={"RED": 4, "BLUE": 2},
        tickets=[{"id": "t1", "points": 10, "completed": True}],
        claimed_route_ids=["r_0_bos_ny"],
        color="#3B82F6",
    )
    assert player.score == 15

    state = GameStateDTO(
        session_id="sess_123",
        turn_number=5,
        current_player_index=0,
        map_name="mini",
        players=[player],
        visible_cards=["RED", "BLUE", "LOCOMOTIVE", "GREEN", "YELLOW"],
        deck_size=90,
        discard_pile_size=5,
        tickets_deck_size=20,
        claimed_routes={"r_0_bos_ny": "p0"},
        valid_actions=[action],
        action_mask=[True, False, True],
        is_game_over=False,
    )
    data = state.model_dump()
    assert data["session_id"] == "sess_123"
    assert len(data["players"]) == 1


def test_brain_dto_serialization():
    layer = LayerActivationDTO(
        layer_name="fc1",
        shape=[1, 256],
        mean=0.45,
        std=0.12,
        min=0.0,
        max=1.85,
        values=[0.1, 0.4, 0.9],
    )
    brain = BrainInspectionDTO(
        model_type="ppo",
        observation_vector=[0.1] * 128,
        action_mask=[True] * 56,
        layer_activations=[layer],
        raw_logits_or_q=[1.5, 3.2, 0.1],
        masked_logits_or_q=[1.5, 3.2, 0.1],
        action_probabilities=[0.15, 0.80, 0.05],
        estimated_value=4.2,
        greedy_action_index=1,
        action_labels=["Draw Deck", "Draw Card 1", "Claim R1"],
    )
    assert brain.greedy_action_index == 1
    assert brain.estimated_value == 4.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/api/test_schemas.py -v`
Expected: FAIL due to missing schemas in `src/api/schemas.py`

- [ ] **Step 3: Implement schemas**

Implement all Pydantic v2 schemas in `src/api/schemas.py` matching the spec.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/api/test_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/schemas.py tests/api/test_schemas.py
git commit -m "feat(api): define Pydantic DTO models for Phase 5 Web Lab"
```

---

### Task 2: Game Service & Session Management

**Files:**
- Create: `tests/api/test_game_service.py`
- Create: `src/api/game_service.py`

**Interfaces:**
- Consumes: `src/game/game.py`, `src/game/maps.py`, `src/agents/base_agent.py`, `src/api/schemas.py`
- Produces: `GameService` class (`create_session`, `get_session_state`, `step_session`, `delete_session`)

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_game_service.py
from src.api.game_service import GameService
from src.api.schemas import GameSessionCreateRequest, ActionDTO


def test_game_service_lifecycle_and_step():
    service = GameService()
    req = GameSessionCreateRequest(map_name="mini", player_types=["random", "greedy"], seed=42)
    state = service.create_session(req)
    assert state.session_id is not None
    assert state.turn_number == 1
    assert len(state.players) == 2
    assert len(state.valid_actions) > 0

    # Step automatically (bot turn)
    next_state = service.step_session(state.session_id, action=None)
    assert next_state.session_id == state.session_id
    assert next_state.turn_number >= 1

    # Cleanup
    assert service.delete_session(state.session_id) is True
    assert service.get_session_state(state.session_id) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/api/test_game_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.game_service'`

- [ ] **Step 3: Implement GameService**

Create `src/api/game_service.py` implementing session creation, agent binding (Human, Random, Greedy, Strategic, DQN, PPO), state conversion to `GameStateDTO`, and step execution.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/api/test_game_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/game_service.py tests/api/test_game_service.py
git commit -m "feat(api): implement GameService for interactive and bot sessions"
```

---

### Task 3: Brain Introspection Service

**Files:**
- Create: `tests/api/test_brain_service.py`
- Create: `src/api/brain_service.py`

**Interfaces:**
- Consumes: `src/rl/networks.py`, `src/environment/observation.py`, `src/api/schemas.py`
- Produces: `BrainService` class (`inspect_q_network`, `inspect_actor_critic`, `inspect_from_game_state`)

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_brain_service.py
import torch
from src.rl.networks import MaskedQNetwork, MaskedActorCritic
from src.api.brain_service import BrainService


def test_brain_service_q_network():
    service = BrainService()
    net = MaskedQNetwork(input_dim=128, action_dim=56, hidden_dim=64)
    obs = [0.1] * 128
    mask = [True] * 56
    mask[2] = False  # Action 2 is masked

    inspection = service.inspect_q_network(net, obs, mask)
    assert inspection.model_type == "dqn"
    assert len(inspection.layer_activations) > 0
    assert inspection.action_probabilities[2] == 0.0
    assert len(inspection.raw_logits_or_q) == 56


def test_brain_service_actor_critic():
    service = BrainService()
    net = MaskedActorCritic(input_dim=128, action_dim=56, hidden_dim=64)
    obs = [0.1] * 128
    mask = [True] * 56
    mask[0] = False

    inspection = service.inspect_actor_critic(net, obs, mask)
    assert inspection.model_type == "ppo"
    assert inspection.estimated_value is not None
    assert inspection.action_probabilities[0] == 0.0
    assert abs(sum(inspection.action_probabilities) - 1.0) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/api/test_brain_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.brain_service'`

- [ ] **Step 3: Implement BrainService**

Create `src/api/brain_service.py` extracting PyTorch layer activations, applying action masks, computing Softmax distributions, and returning `BrainInspectionDTO`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/api/test_brain_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/brain_service.py tests/api/test_brain_service.py
git commit -m "feat(api): implement BrainService for neural network layer introspection"
```

---

### Task 4: Replay Service & Trajectory Storage

**Files:**
- Create: `tests/api/test_replay_service.py`
- Create: `src/api/replay_service.py`

**Interfaces:**
- Consumes: `src/api/schemas.py`, `src/game/state.py`
- Produces: `ReplayService` (`save_replay`, `list_replays`, `load_replay`, `delete_replay`)

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_replay_service.py
import tempfile
from pathlib import Path
from src.api.replay_service import ReplayService
from src.api.schemas import ReplayDetailDTO, ReplayFrameDTO, ActionDTO


def test_replay_service_save_list_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        service = ReplayService(replays_dir=Path(tmpdir))
        frame = ReplayFrameDTO(
            step_index=0,
            turn_number=1,
            player_index=0,
            action=ActionDTO(action_type="DRAW_TRAIN_CARDS"),
            reward=0.0,
            state_snapshot={"turn": 1, "scores": [0, 0]},
        )
        replay = ReplayDetailDTO(
            replay_id="rep_test_01",
            map_name="mini",
            seed=42,
            date="2026-08-19",
            player_names=["P1", "P2"],
            total_steps=1,
            winner_index=0,
            final_scores=[10, 5],
            frames=[frame],
        )
        saved_path = service.save_replay(replay)
        assert Path(saved_path).exists()

        replays = service.list_replays()
        assert len(replays) == 1
        assert replays[0]["replay_id"] == "rep_test_01"

        loaded = service.load_replay("rep_test_01")
        assert loaded is not None
        assert loaded.replay_id == "rep_test_01"
        assert len(loaded.frames) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/api/test_replay_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.replay_service'`

- [ ] **Step 3: Implement ReplayService**

Create `src/api/replay_service.py` to handle saving, loading, listing, and parsing JSON/JSONL replay files.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/api/test_replay_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/replay_service.py tests/api/test_replay_service.py
git commit -m "feat(api): implement ReplayService for saving and loading match trajectories"
```

---

### Task 5: Trainer Service & WebSocket Telemetry Streaming Hub

**Files:**
- Create: `tests/api/test_trainer_service.py`
- Create: `tests/api/test_websocket_telemetry.py`
- Modify: `src/api/websocket.py`
- Create: `src/api/trainer_service.py`

**Interfaces:**
- Consumes: `src/api/schemas.py`, `src/rl/ppo.py`, `src/rl/dqn.py`, `src/experiments/runner.py`
- Produces: `ConnectionManager` enhancements and `TrainerService` (`start_training`, `stop_training`, `get_status`)

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_trainer_service.py
import asyncio
from src.api.trainer_service import TrainerService
from src.api.schemas import TrainingStartRequest
from src.api.websocket import ConnectionManager


def test_trainer_service_start_stop():
    manager = ConnectionManager()
    service = TrainerService(connection_manager=manager)

    req = TrainingStartRequest(config_name="ppo_mini.yaml", override_timesteps=200, seed=42)
    status = service.start_training(req)
    assert status.is_training is True
    assert status.algorithm in ["ppo", "dqn"]

    service.stop_training()
    status_after = service.get_status()
    assert status_after.is_training is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/api/test_trainer_service.py -v`
Expected: FAIL with missing `TrainerService`

- [ ] **Step 3: Implement TrainerService & ConnectionManager**

Implement `TrainerService` with background execution, non-blocking telemetry queuing, and WebSocket broadcast.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/api/test_trainer_service.py tests/api/test_websocket_telemetry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/trainer_service.py src/api/websocket.py tests/api/test_trainer_service.py tests/api/test_websocket_telemetry.py
git commit -m "feat(api): implement TrainerService and WebSocket telemetry broadcaster"
```

---

### Task 6: FastAPI REST & WebSocket Endpoints Aggregation

**Files:**
- Create: `tests/api/test_api_endpoints.py`
- Modify: `src/api/main.py`

**Interfaces:**
- Consumes: `GameService`, `BrainService`, `ReplayService`, `TrainerService`, `ConnectionManager`
- Produces: Complete FastAPI app with endpoints under `/api/game/*`, `/api/training/*`, `/api/replays/*`, `/api/experiments/*`, `/api/brain/*`, and `/ws/telemetry`.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_api_endpoints.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_api_game_session_and_step():
    res = client.post("/api/game/new", json={"map_name": "mini", "player_types": ["random", "random"]})
    assert res.status_code == 200
    sess = res.json()
    session_id = sess["session_id"]

    step_res = client.post("/api/game/step", json={"session_id": session_id})
    assert step_res.status_code == 200
    assert step_res.json()["session_id"] == session_id


def test_api_replays_and_experiments():
    res = client.get("/api/replays/list")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    exp_res = client.get("/api/experiments/list")
    assert exp_res.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/api/test_api_endpoints.py -v`
Expected: FAIL due to missing routes in `src/api/main.py`

- [ ] **Step 3: Implement routes in main.py**

Wire all routers and service instances cleanly in `src/api/main.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/api/test_api_endpoints.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/main.py tests/api/test_api_endpoints.py
git commit -m "feat(api): expose complete REST endpoints and WebSocket telemetry hub"
```

---

### Task 7: Frontend Types, API Client & Custom Hooks

**Files:**
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/hooks/useWebSocket.ts`
- Create: `frontend/src/hooks/useGameSession.ts`
- Create: `frontend/src/hooks/useTrainingStream.ts`
- Create: `frontend/src/hooks/useReplayPlayer.ts`
- Modify: `frontend/src/hooks/index.ts`
- Modify: `frontend/src/api/index.ts`

**Interfaces:**
- Produces: Typed API client methods (`createGame`, `stepGame`, `startTraining`, `stopTraining`, `fetchReplays`, `fetchBrainInspection`) and reusable React state hooks.

- [ ] **Step 1: Define TypeScript Types**

Create `frontend/src/api/types.ts` with all interfaces matching Pydantic DTOs.

- [ ] **Step 2: Implement HTTP API Client**

Create `frontend/src/api/client.ts` with `fetch` wrapper and error handling.

- [ ] **Step 3: Implement React Custom Hooks**

Create `useWebSocket.ts`, `useGameSession.ts`, `useTrainingStream.ts`, `useReplayPlayer.ts`.

- [ ] **Step 4: Verify with TypeScript build check**

Run: `npm --prefix frontend run build`
Expected: Build passes or validates types cleanly.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/ frontend/src/hooks/
git commit -m "feat(frontend): create TypeScript DTO types, API client, and state hooks"
```

---

### Task 8: Vector SVG Board Component (`BoardSVG`)

**Files:**
- Create: `frontend/src/components/board/BoardSVG.tsx`
- Create: `frontend/src/components/board/CityNode.tsx`
- Create: `frontend/src/components/board/RouteEdge.tsx`
- Modify: `frontend/src/components/index.ts`

**Interfaces:**
- Produces: `BoardSVG` component accepting `mapName`, `cities`, `routes`, `claimedRoutes`, `highlightedCities`, `onRouteClick`, `onCityHover`.

- [ ] **Step 1: Implement CityNode & RouteEdge**

Create SVG subcomponents with hover tooltips, color coding, double-route parallel vector offsets, and segmented train tracks.

- [ ] **Step 2: Implement BoardSVG**

Create `BoardSVG.tsx` with responsive `viewBox="0 0 1000 650"`, normalized coordinate transformation, and interactive overlays.

- [ ] **Step 3: Verify TypeScript compilation**

Run: `npm --prefix frontend run build`
Expected: Compiles with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/board/ frontend/src/components/index.ts
git commit -m "feat(frontend): create interactive vector BoardSVG for USA and Mini maps"
```

---

### Task 9: Game Components & GameView

**Files:**
- Create: `frontend/src/components/cards/TrainCardHand.tsx`
- Create: `frontend/src/components/cards/VisibleDeck.tsx`
- Create: `frontend/src/components/cards/TicketsList.tsx`
- Create: `frontend/src/views/GameView.tsx`
- Modify: `frontend/src/views/index.ts`

**Interfaces:**
- Produces: `GameView` component allowing Human vs AI or AI vs AI games with live board, cards, turn controls, and simulation speed.

- [ ] **Step 1: Implement Cards & Tickets Components**

Create `TrainCardHand.tsx`, `VisibleDeck.tsx`, and `TicketsList.tsx`.

- [ ] **Step 2: Implement GameView**

Wire `BoardSVG`, player cards, and action buttons into `frontend/src/views/GameView.tsx`.

- [ ] **Step 3: Verify TypeScript compilation**

Run: `npm --prefix frontend run build`
Expected: Compiles with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/cards/ frontend/src/views/GameView.tsx frontend/src/views/index.ts
git commit -m "feat(frontend): implement GameView with interactive board, cards, and turn controls"
```

---

### Task 10: Agent & Brain Introspection Components (`AgentView` & `BrainView`)

**Files:**
- Create: `frontend/src/components/brain/NeuralNetworkDiagram.tsx`
- Create: `frontend/src/components/brain/ActionProbabilitiesChart.tsx`
- Create: `frontend/src/views/AgentView.tsx`
- Create: `frontend/src/views/BrainView.tsx`
- Modify: `frontend/src/views/index.ts`

**Interfaces:**
- Produces: `AgentView` and `BrainView` with visual layer graph, activation bars, action masking breakdown, and Softmax probability distributions.

- [ ] **Step 1: Implement Neural Diagram & Action Chart**

Create `NeuralNetworkDiagram.tsx` and `ActionProbabilitiesChart.tsx`.

- [ ] **Step 2: Implement AgentView and BrainView**

Create `AgentView.tsx` (real-time observation and decision breakdown) and `BrainView.tsx` (deep MLP layer introspection).

- [ ] **Step 3: Verify TypeScript compilation**

Run: `npm --prefix frontend run build`
Expected: Compiles with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/brain/ frontend/src/views/AgentView.tsx frontend/src/views/BrainView.tsx frontend/src/views/index.ts
git commit -m "feat(frontend): implement AgentView and BrainView for deep neural introspection"
```

---

### Task 11: Training Stream & Experiment Explorer (`TrainingView` & `ExperimentView`)

**Files:**
- Create: `frontend/src/components/charts/LiveTelemetryChart.tsx`
- Create: `frontend/src/views/TrainingView.tsx`
- Create: `frontend/src/views/ExperimentView.tsx`
- Modify: `frontend/src/views/index.ts`

**Interfaces:**
- Produces: `TrainingView` (live WebSocket streaming curves for reward, loss, entropy, KL) and `ExperimentView` (registry comparator and checkpoint explorer).

- [ ] **Step 1: Implement LiveTelemetryChart**

Create lightweight SVG/Canvas real-time multi-metric line chart with sliding window buffer.

- [ ] **Step 2: Implement TrainingView and ExperimentView**

Implement config launcher, pause/stop buttons, metrics dashboard, and experiment registry table.

- [ ] **Step 3: Verify TypeScript compilation**

Run: `npm --prefix frontend run build`
Expected: Compiles with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/charts/ frontend/src/views/TrainingView.tsx frontend/src/views/ExperimentView.tsx frontend/src/views/index.ts
git commit -m "feat(frontend): implement TrainingView with live WebSocket charts and ExperimentView"
```

---

### Task 12: Replay Player & Scrubber (`ReplayView`)

**Files:**
- Create: `frontend/src/components/replay/ReplayControls.tsx`
- Create: `frontend/src/views/ReplayView.tsx`
- Modify: `frontend/src/views/index.ts`

**Interfaces:**
- Produces: `ReplayView` with timeline scrubber, transport controls (`|< << < ▶ > >> >|`), and step-by-step state inspector.

- [ ] **Step 1: Implement ReplayControls**

Create scrubber timeline, play/pause toggle, speed selector (1x, 2x, 5x, 10x), and step buttons.

- [ ] **Step 2: Implement ReplayView**

Wire replay loader, board state updates, and step-by-step action logger.

- [ ] **Step 3: Verify TypeScript compilation**

Run: `npm --prefix frontend run build`
Expected: Compiles with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/replay/ frontend/src/views/ReplayView.tsx frontend/src/views/index.ts
git commit -m "feat(frontend): implement ReplayView with timeline scrubber and transport controls"
```

---

### Task 13: Full Integration, Phase 5 Acceptance Test & Impeccable Critique

**Files:**
- Create: `tests/api/test_phase5_acceptance.py`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/index.css`

**Interfaces:**
- Produces: Complete end-to-end Web Lab integration, green acceptance test, and refined UI with impeccable critique.

- [ ] **Step 1: Write Phase 5 Acceptance Test**

```python
# tests/api/test_phase5_acceptance.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.schemas import GameSessionCreateRequest, TrainingStartRequest

client = TestClient(app)


def test_phase5_acceptance_full_web_lab_pipeline():
    """Phase 5 Acceptance Test: Verifies game sessions, bot step, brain inspection, telemetry streaming, and replay listing."""
    # 1. Game Session
    res = client.post("/api/game/new", json={"map_name": "mini", "player_types": ["random", "greedy"], "seed": 42})
    assert res.status_code == 200
    sess = res.json()
    session_id = sess["session_id"]

    # 2. Step Game
    step_res = client.post("/api/game/step", json={"session_id": session_id})
    assert step_res.status_code == 200
    assert step_res.json()["turn_number"] >= 1

    # 3. Brain Inspection
    brain_res = client.post("/api/brain/inspect", json={"session_id": session_id, "model_type": "ppo"})
    assert brain_res.status_code == 200
    brain_data = brain_res.json()
    assert len(brain_data["action_probabilities"]) > 0

    # 4. Training Status & Start
    train_res = client.post("/api/training/start", json={"config_name": "ppo_mini.yaml", "override_timesteps": 200})
    assert train_res.status_code in [200, 202]

    # 5. Replays & Experiments
    assert client.get("/api/replays/list").status_code == 200
    assert client.get("/api/experiments/list").status_code == 200
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/pytest tests/api/test_phase5_acceptance.py -v`
Expected: PASS

- [ ] **Step 3: Integrate views into App.tsx and style in index.css**

Connect all 6 views into `App.tsx` navigation and polish CSS theme.

- [ ] **Step 4: Run full backend and frontend validation**

Run:
1. `.venv/bin/pytest tests/api/`
2. `npm --prefix frontend run build`

- [ ] **Step 5: Run impeccable critique to evaluate and polish frontend design**

Execute `impeccable critique` (or evaluate UI layout and usability) and apply refinements.

- [ ] **Step 6: Commit and tag Phase 5 completion**

```bash
git add .
git commit -m "feat: complete Phase 5 Web Lab with full backend API, React SPA, and tests"
```
