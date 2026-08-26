# Implementation Plan: Map Focus Mode, Game Over Modal Refinement, Multi-Checkpoint Tournaments/Duels, and PFSP Live Training Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement map track decolorization/player focus mode, fix GameOverModal tab-switching re-triggering, enable multi-checkpoint selection for tournaments and duels, and resolve the Self-Play / PFSP background training crash.

**Architecture:** 
- Frontend SVG map rendering update with focused player isolation and desaturated background tracks.
- Workbench context session-level completion tracking for GameOverModal.
- TournamentService & GameService checkpoint auto-discovery and dynamic naming for multi-checkpoint duels and round-robin matrices.
- Correct `SelfPlayPPOTrainer` lifecycle and parameter initialization with exception reporting.

**Tech Stack:** Python 3.12+, FastAPI, PyTorch, React 18, TypeScript 5, Vite, Lucide-React.

---

### Task 1: Fix Self-Play Policy Pool (PFSP / Lesson 9) Live Training Bug

**Files:**
- Modify: `src/api/trainer_service.py:270-325`
- Test: `tests/test_pfsp_training.py`

**Interfaces:**
- Consumes: `src.rl.self_play.SelfPlayPPOTrainer`, `src.rl.self_play.PolicyPool`, `src.rl.self_play.SelfPlayOpponentSampler`
- Produces: Working background training execution emitting live telemetry for `self_play_ppo` without crashing at step 0.

- [ ] **Step 1: Write a test verifying `SelfPlayPPOTrainer` execution loop and `TrainerService` instantiation**

Create `tests/test_pfsp_training.py`:
```python
import pytest
from src.environment.env import TicketToRideEnv
from src.game.maps import create_synthetic_mini_board
from src.rl.self_play import PolicyPool, SelfPlayOpponentSampler, SelfPlayPPOTrainer
from src.api.trainer_service import TrainerService
from src.api.schemas import TrainingStartRequest


def test_self_play_trainer_rollout_and_epoch():
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, num_players=2)
    pool = PolicyPool(max_size=10)
    sampler = SelfPlayOpponentSampler(strategy="pfsp", seed=42)
    config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "rollout_steps": 16,
        "minibatch_size": 8,
        "num_epochs": 2,
        "snapshot_interval": 50,
    }
    trainer = SelfPlayPPOTrainer(env=env, config=config, pool=pool, sampler=sampler, seed=42)
    
    rollout_info = trainer.collect_rollout()
    assert "mean_rollout_reward" in rollout_info
    metrics = trainer.train_epoch()
    assert "policy_loss" in metrics
    assert "value_loss" in metrics


def test_trainer_service_self_play_startup():
    service = TrainerService()
    req = TrainingStartRequest(
        config_name="self_play_mini.yaml",
        algorithm_type="self_play_ppo",
        override_timesteps=32,
        map_name="mini",
        seed=42,
    )
    status = service.start_training(req)
    assert status.is_training is True
    assert status.algorithm == "self_play_ppo"
    service.stop_training()
```

- [ ] **Step 2: Run test to verify failure before fix**

Run: `pytest tests/test_pfsp_training.py -v`

- [ ] **Step 3: Fix `_run_training` in `src/api/trainer_service.py`**

In `src/api/trainer_service.py`:
```python
        elif algo in ("self_play", "self_play_ppo"):
            # Lesson 9: Self-Play Policy Pool PPO Trainer with PFSP Matchmaking
            pool = PolicyPool(max_size=20)
            sampler = SelfPlayOpponentSampler(
                strategy="pfsp",
                baseline_mix_rate=0.2,
                pfsp_exponent=1.0,
                seed=config.seed,
            )
            sp_config = {
                "lr": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "rollout_steps": 64,
                "minibatch_size": 32,
                "num_epochs": 4,
                "snapshot_interval": 200,
            }
            sp_trainer = SelfPlayPPOTrainer(
                env=env,
                config=sp_config,
                pool=pool,
                sampler=sampler,
                seed=config.seed,
            )

            step = 0
            while step < total_steps and not self._stop_requested:
                rollout_info = sp_trainer.collect_rollout()
                step += sp_trainer.rollout_steps
                episodes_in_rollout = int(rollout_info.get("episodes", 0))
                episode_count += episodes_in_rollout
                mean_r = float(rollout_info.get("mean_rollout_reward", 0.0))
                if episodes_in_rollout > 0 or not rolling_rewards:
                    rolling_rewards.append(mean_r)

                metrics = sp_trainer.train_epoch()

                elapsed = time.time() - start_time
                fps = float(step / elapsed) if elapsed > 0 else 0.0
                smooth_reward = float(np.mean(rolling_rewards[-10:])) if rolling_rewards else mean_r

                self._status.current_step = min(step, total_steps)
                self._status.episodes = episode_count
                self._status.mean_reward = smooth_reward

                now = time.time()
                if now - last_broadcast_time >= broadcast_interval or step >= total_steps:
                    last_broadcast_time = now
                    telemetry = TelemetryEventDTO(
                        type="training_step",
                        experiment_id=exp_id,
                        step=min(step, total_steps),
                        episode=episode_count,
                        reward=mean_r,
                        mean_reward=smooth_reward,
                        policy_loss=float(metrics.get("policy_loss", 0.0)),
                        value_loss=float(metrics.get("value_loss", 0.0)),
                        entropy=float(metrics.get("entropy", 0.0)),
                        approx_kl=float(metrics.get("approx_kl", 0.0)),
                        clip_fraction=float(metrics.get("clip_fraction", 0.0)),
                        explained_var=float(metrics.get("explained_var", 0.0)),
                        win_rate=min(max(0.5 + smooth_reward * 0.05, 0.0), 1.0),
                        fps=round(fps, 1),
                    )
                    self.connection_manager.broadcast_sync(telemetry.model_dump())

            sp_trainer.save(ckpt_path)
            sp_trainer.save(os.path.join(config.training.checkpoint_dir, "self_play_live_latest.pt"))
```
Also wrap the entire `_run_training` body in a `try...except Exception as exc:` block that logs tracebacks and broadcasts an error event if something goes wrong.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pfsp_training.py -v`

---

### Task 2: Multi-Checkpoint Support & Dynamic Player Naming for Tournaments and Duels

**Files:**
- Modify: `src/api/tournament_service.py`
- Modify: `src/api/game_service.py`
- Modify: `frontend/src/components/board/MatchScoreBoardHUD.tsx`
- Modify: `frontend/src/components/tournament/EloMatrixHeatmap.tsx`
- Test: `tests/test_multi_checkpoint_duel.py`

**Interfaces:**
- Consumes: Checkpoint files from `experiments/checkpoints/*.pt`
- Produces: Distinct player names (e.g. `PPO: epoch_50` vs `PPO: epoch_100`) and accurate tournament registration.

- [ ] **Step 1: Write backend tests for checkpoint naming and tournament agent loading**

Create `tests/test_multi_checkpoint_duel.py`:
```python
import os
import torch
import pytest
from src.api.game_service import GameService
from src.api.schemas import GameSessionCreateRequest
from src.api.tournament_service import TournamentService
from src.rl.networks import MaskedActorCritic


def test_game_service_multi_checkpoint_names(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    
    net1 = MaskedActorCritic(input_dim=180, action_dim=150)
    net2 = MaskedActorCritic(input_dim=180, action_dim=150)
    p1_path = str(ckpt_dir / "ppo_gen_10.pt")
    p2_path = str(ckpt_dir / "ppo_gen_50.pt")
    torch.save({"actor_critic_state_dict": net1.state_dict(), "total_timesteps": 1000}, p1_path)
    torch.save({"actor_critic_state_dict": net2.state_dict(), "total_timesteps": 5000}, p2_path)
    
    service = GameService()
    req = GameSessionCreateRequest(
        map_name="mini",
        player_types=["ppo", "ppo"],
        player_checkpoints=[p1_path, p2_path],
        seed=42,
    )
    state = service.create_session(req)
    assert len(state.players) == 2
    assert "ppo_gen_10" in state.players[0].name.lower() or "1" in state.players[0].name
    assert "ppo_gen_50" in state.players[1].name.lower() or "2" in state.players[1].name
```

- [ ] **Step 2: Update `src/api/game_service.py` to format agent names from checkpoints**

When `ckpt_path` is supplied in `create_session`:
Format the agent name to include the checkpoint file basename (e.g., `f"PPO ({os.path.basename(ckpt_path).replace('.pt', '')})"` instead of generic `PPO Trained (1)`).

- [ ] **Step 3: Update `src/api/tournament_service.py` to correctly categorize participants**

Ensure `get_available_participants()` returns all baselines and all `.pt` checkpoints found, accurately setting `category="checkpoint"` and `category="baseline"`.

- [ ] **Step 4: Update `MatchScoreBoardHUD.tsx` and `EloMatrixHeatmap.tsx`**

In `MatchScoreBoardHUD.tsx`:
- Allow choosing from all discovered checkpoints for Player 1 and Player 2.
- Make checkpoint dropdown always accessible when selecting neural/RL algorithms (PPO, DQN, Recurrent PPO, AlphaZero, Self-Play).
- Display the selected checkpoint name clearly.

In `EloMatrixHeatmap.tsx`:
- Ensure the participant filter tabs (All, RL Models, Baselines) show dynamic counts based on `available_participants`.
- Show helpful empty state hint if RL Models count is 0 explaining that checkpoints will appear as soon as a training session finishes.

- [ ] **Step 5: Run tests and verify**

Run: `pytest tests/test_multi_checkpoint_duel.py -v`

---

### Task 3: Map Focus Mode & Track Decolorization Button

**Files:**
- Modify: `frontend/src/components/board/RouteEdge.tsx`
- Modify: `frontend/src/components/board/BoardSVG.tsx`
- Modify: `frontend/src/components/board/BoardCanvas.tsx`
- Modify: `frontend/src/components/board/MatchScoreBoardHUD.tsx`

**Interfaces:**
- Produces: `focusPlayerMode: 'all' | 'player_0' | 'player_1'` state toggle and conditional track styling in SVG.

- [ ] **Step 1: Add focus mode support to `RouteEdge.tsx`**

Add `focusPlayerId?: string | null` or `isDimmed?: boolean` and `isHighlighted?: boolean` props:
- When focus mode is active and a route does NOT belong to the focused player:
  - Render tracks with muted parchment gray tone (`#9E8F80` / `#C5B9A8`) and reduced opacity (`0.25`).
- When a route DOES belong to the focused player:
  - Render tracks with full vivid color, intensified sleeper bed, and a distinctive golden/player-colored drop-shadow glow!

- [ ] **Step 2: Add focus mode state and toggle button to `BoardCanvas.tsx` / `BoardSVG.tsx`**

In `BoardCanvas.tsx`:
- Maintain `focusMode: 'all' | 'player_0' | 'player_1'` state.
- Add an intuitive Steampunk map toolbar button:
  - 🎨 **Filtro Tratte Giocatore** / **Decolora Mappa**:
    - Mode 1: 🌐 **Tutte le Tratte** (Standard)
    - Mode 2: 🔵 **Solo Tratte Giocatore 1**
    - Mode 3: 🔴 **Solo Tratte Giocatore 2**
- Pass `focusMode` down to `BoardSVG.tsx` and each `RouteEdge.tsx`.

---

### Task 4: Fix GameOverModal Tab-Switching Pop-up Behavior

**Files:**
- Modify: `frontend/src/context/workbenchTypes.ts`
- Modify: `frontend/src/context/WorkbenchContext.tsx`
- Modify: `frontend/src/components/board/BoardCanvas.tsx`
- Modify: `frontend/src/components/board/GameOverModal.tsx`

**Interfaces:**
- Produces: Session-level `dismissedGameOverSessions: Set<string>` and trigger condition checking if game ended during active viewing.

- [ ] **Step 1: Add dismissed session tracking in `WorkbenchContext`**

Add `dismissedGameOverSessionIds: string[]` and action `dismissGameOver(sessionId: string)` to `WorkbenchContext`.

- [ ] **Step 2: Update `BoardCanvas.tsx` to respect dismissed session state**

In `BoardCanvas.tsx`:
- Show `GameOverModal` ONLY IF:
  - `gameState.is_game_over === true`
  - `!dismissedGameOverSessionIds.includes(gameState.session_id)`
  - The game is in `interactive` or active `replay_scrub` mode.
- When `onClose` or rematch is clicked, call `dismissGameOver(gameState.session_id)`.
- When switching tabs, the modal will never re-appear for an already dismissed or background-ended session!

---

### Task 5: End-to-End Verification

**Files:**
- Run all python unit and integration tests.
- Run frontend type check and build.

- [ ] **Step 1: Run pytest across the whole suite**

Run: `pytest tests/ -v`

- [ ] **Step 2: Run frontend build check**

Run: `npm --prefix frontend run build`

- [ ] **Step 3: Verify live services and functionality**
