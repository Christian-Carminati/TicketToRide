# Ticket to Ride RL Lab — Web Suite Workbench & Telemetry Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Web Lab into an Industrial Research Avionics Workbench featuring a synchronized split-pane studio (Vector Board Canvas + Neural Brain Inspector + Collapsible Telemetry & Tournament Matrix Dock) with live WebSocket streaming, partial-observability toggle, and interactive brain-to-board hover links.

**Architecture:** A centralized React Workbench Context (`useWorkbenchContext`) synchronizes high-frequency board state, neural policy/value tensors, timeline scrubber indices, and hover-linked action IDs across modular sub-panes. The backend WebSocket streaming hub supplies real-time game state, neural activations, and training loss metrics, while the client enables frame-by-frame counterfactual exploration and partial observability filtering.

**Tech Stack:** React 18, TypeScript, Vite, SVG/HTML5 Canvas, Lucide-react, FastAPI WebSocket/REST API.

**Spec:** [.impeccable/surfaces/frontend-src-app-tsx.md](file:///home/christian/Projects/Python/TicketToRide/.impeccable/surfaces/frontend-src-app-tsx.md) and [docs/superpowers/specs/2026-08-19-phase-5-web-lab-design.md](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-19-phase-5-web-lab-design.md)

## Global Constraints

- Strict partial observability: Player agents must never receive opponent hidden cards or deck order; "God-Mode" is strictly a client-side researcher toggle.
- Non-blocking 60 FPS UI rendering: Hover interactions and high-frequency WebSocket streams must not cause full-tree React re-render thrashing.
- Zero TypeScript compiler warnings or `any` types; all API models must match backend Pydantic DTOs.
- Existing CLI headless scripts (`scripts/train.py`, `scripts/evaluate.py`, `scripts/tournament.py`) and pytest suites must remain 100% operational.

---

### Task 1: Centralized Workbench State & Synchronizer Context

**Files:**
- Create: `frontend/src/context/workbenchTypes.ts`
- Create: `frontend/src/context/WorkbenchContext.tsx`
- Create: `frontend/src/context/index.ts`

**Interfaces:**
- Produces: `WorkbenchProvider`, `useWorkbench()`, `WorkbenchState`, `WorkbenchAction`, `ObservabilityMode` (`'god' | 'player_0' | 'player_1'`), `StudioMode` (`'interactive' | 'training_live' | 'replay_scrub' | 'tournament'`).
- Manages: `activeGameState`, `brainInspection`, `hoveredAction`, `hoveredRouteId`, `scrubberStep`, `isStreaming`, `playbackSpeed`, `bottomDockTab` (`'telemetry' | 'tournament' | 'logs'`), `isBottomDockOpen`.

- [ ] **Step 1: Define Workbench Types**

Create `frontend/src/context/workbenchTypes.ts` with all state, action, and filter interfaces:

```typescript
import { GameStateDTO, BrainInspectionDTO, TelemetryEventDTO, ReplayDetailDTO, ActionDTO } from '../api/types';

export type ObservabilityMode = 'god' | 'player_0' | 'player_1';
export type StudioMode = 'interactive' | 'training_live' | 'replay_scrub' | 'tournament';
export type BottomDockTab = 'telemetry' | 'tournament' | 'logs';

export interface HoveredActionMeta {
  actionIndex?: number;
  actionType: string;
  routeId?: string;
  probability?: number;
  value?: number;
  isMasked?: boolean;
}

export interface WorkbenchState {
  studioMode: StudioMode;
  observabilityMode: ObservabilityMode;
  gameState: GameStateDTO | null;
  brainData: BrainInspectionDTO | null;
  telemetryHistory: TelemetryEventDTO[];
  replayData: ReplayDetailDTO | null;
  currentStepIndex: number;
  maxStepIndex: number;
  isPlaying: boolean;
  playbackSpeed: number; // 0.25x, 0.5x, 1x, 2x, 5x
  hoveredAction: HoveredActionMeta | null;
  hoveredRouteId: string | null;
  bottomDockTab: BottomDockTab;
  isBottomDockOpen: boolean;
  selectedAgentModel: 'ppo' | 'dqn' | 'heuristic' | 'random';
}
```

- [ ] **Step 2: Implement Context Provider and Reducer**

Create `frontend/src/context/WorkbenchContext.tsx` implementing `workbenchReducer`, `WorkbenchProvider`, and `useWorkbench()` hook with actions: `setGameState`, `setBrainData`, `setHoveredAction`, `setHoveredRoute`, `setObservabilityMode`, `setStudioMode`, `togglePlayback`, `setScrubberStep`, `toggleBottomDock`, `setBottomDockTab`.

- [ ] **Step 3: Export in index.ts and verify build**

Export context in `frontend/src/context/index.ts`.
Run: `npm --prefix frontend run build`
Expected: Compiles cleanly.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/context/
git commit -m "feat(frontend): create centralized WorkbenchContext for state synchronization"
```

---

### Task 2: Synchronized Interactive Vector Board Canvas (`BoardCanvas` & Hover Links)

**Files:**
- Modify: `frontend/src/components/board/BoardSVG.tsx`
- Modify: `frontend/src/components/board/RouteEdge.tsx`
- Modify: `frontend/src/components/board/CityNode.tsx`
- Create: `frontend/src/components/board/BoardHeaderControls.tsx`
- Create: `frontend/src/components/board/BoardCanvas.tsx`
- Modify: `frontend/src/components/board/index.ts`

**Interfaces:**
- Consumes: `useWorkbench()`, `BoardSVG`
- Produces: `BoardCanvas` with God-Mode vs Partial-Observability view switcher, Route Hover Highlight with policy/value badge overlays, and responsive pan/zoom coordinate system.

- [ ] **Step 1: Update RouteEdge and CityNode for Synced Hover Link**

Enhance `RouteEdge.tsx` to listen to `hoveredRouteId` from Workbench context. When `hoveredRouteId === route.id`, render an intense pulsating highlight glow (`#38BDF8`), display the target action's $P(\text{action})$ / validity badge, and trigger bidirectional `setHoveredRoute(route.id)`.

- [ ] **Step 2: Implement Partial Observability Masking in BoardSVG**

In `BoardSVG.tsx`, apply filtering based on `observabilityMode`:
- In `god` mode: Show full ground truth (all player cards, scores, destination tickets).
- In `player_0` / `player_1` mode: Mask opponent destination tickets as secret count cards, mask opponent unrevealed cards with backface cards, highlighting what is visible vs inferred.

- [ ] **Step 3: Create BoardHeaderControls and BoardCanvas container**

Create `BoardHeaderControls.tsx` containing:
- Observability selector: `[👁️ God Mode (Observer)]`, `[🤖 Agent A View]`, `[🤖 Agent B View]`.
- Map selector: `[Mini Map (4 Cities)]` | `[USA Map (36 Cities)]`.
- Zoom/Reset controls: `[＋] [－] [⟲ Reset View]`.

Wrap in `frontend/src/components/board/BoardCanvas.tsx`.

- [ ] **Step 4: Verify with TypeScript build**

Run: `npm --prefix frontend run build`
Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/board/
git commit -m "feat(frontend): add synchronized hover links and partial-observability to BoardCanvas"
```

---

### Task 3: Neural Brain Inspector Pane (`BrainInspectorPane`)

**Files:**
- Modify: `frontend/src/components/brain/ActionProbabilitiesChart.tsx`
- Modify: `frontend/src/components/brain/NeuralNetworkDiagram.tsx`
- Create: `frontend/src/components/brain/ValueHeadGauge.tsx`
- Create: `frontend/src/components/brain/ObservationTensorViewer.tsx`
- Create: `frontend/src/components/brain/BrainInspectorPane.tsx`
- Modify: `frontend/src/components/brain/index.ts`

**Interfaces:**
- Consumes: `useWorkbench()`, `BrainInspectionDTO`
- Produces: `BrainInspectorPane` displaying:
  1. Estimated Value Gauge $V(s)$ with trajectory sparkline and baseline advantage.
  2. Ranked Policy Distribution bars with action mask badges (`VALID` green vs `MASKED` strikethrough red) and hover link triggers.
  3. Interactive Observation Tensor feature heatmap.
  4. MLP Layer Activation sparkline bars.

- [ ] **Step 1: Implement ValueHeadGauge**

Create `frontend/src/components/brain/ValueHeadGauge.tsx`:
- Render $V(s)$ scalar gauge (e.g. `+4.82 points expected return`).
- Render historical $V(s)$ timeline sparkline across recent game steps.

- [ ] **Step 2: Enhance ActionProbabilitiesChart for Interactive Hover Linking**

Update `ActionProbabilitiesChart.tsx`:
- Render ranked horizontal distribution bars for top actions.
- On row hover: invoke `setHoveredAction({ actionType, routeId, probability, isMasked })` and `setHoveredRouteId(routeId)`.
- Render distinct visual styles for masked illegal actions (opacity 0.35, strikeout, red `[ILLEGAL MASK]` tag).

- [ ] **Step 3: Implement ObservationTensorViewer and Assemble BrainInspectorPane**

Create `ObservationTensorViewer.tsx` and assemble `BrainInspectorPane.tsx` with collapsible sub-sections:
- Header: Active Model (`PPO Actor-Critic (Masked)` / `DQN Q-Head`).
- Section 1: Value Head & Policy Entropy.
- Section 2: Action Distribution & Mask Filter.
- Section 3: Feature Saliency & Hidden State Tensor.

- [ ] **Step 4: Verify with TypeScript build**

Run: `npm --prefix frontend run build`
Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/brain/
git commit -m "feat(frontend): implement BrainInspectorPane with interactive action linking and value gauge"
```

---

### Task 4: Collapsible Bottom Telemetry & Tournament Matrix Dock

**Files:**
- Modify: `frontend/src/components/charts/LiveTelemetryChart.tsx`
- Create: `frontend/src/components/tournament/EloMatrixHeatmap.tsx`
- Create: `frontend/src/components/dock/ActionLogStream.tsx`
- Create: `frontend/src/components/dock/TelemetryTournamentDock.tsx`
- Modify: `frontend/src/components/index.ts`

**Interfaces:**
- Consumes: `useWorkbench()`, `TelemetryEventDTO`, `ReplayFrameDTO`
- Produces: `TelemetryTournamentDock` with 3 tabbed drawers:
  - Tab 1: **Live Training Telemetry** (Real-time PPO policy loss, value loss, entropy, mean episode reward).
  - Tab 2: **Tournament Elo Matrix** (Head-to-head win-rate heatmap, agent leaderboard, Elo rating progression).
  - Tab 3: **Step Event Logs** (Chronological move stream with clickable step time-travel).

- [ ] **Step 1: Implement EloMatrixHeatmap Component**

Create `frontend/src/components/tournament/EloMatrixHeatmap.tsx`:
- Render an $N \times N$ round-robin matrix table (e.g. `PPO-v4`, `PPO-v1`, `Strategic`, `Greedy`, `Random`).
- Cell color: Green for win-rate $>50\%$, Red for $<50\%$, Slate for diagonal.
- Display Elo rating and win/loss records.

- [ ] **Step 2: Implement ActionLogStream Component**

Create `frontend/src/components/dock/ActionLogStream.tsx`:
- Render chronological turn-by-turn action list with player color badges, action summaries, and step indices.
- Clicking any step moves `currentStepIndex` to scrub board and brain inspectors to that moment.

- [ ] **Step 3: Assemble TelemetryTournamentDock**

Create `frontend/src/components/dock/TelemetryTournamentDock.tsx` with:
- Collapsible tray bar with badge indicators (loss value, current turn, FPS).
- Tab switcher: `[📈 Live Telemetry]`, `[🏆 Elo Tournament Matrix]`, `[📜 Event Log]`.
- Minimize/Maximize toggle button.

- [ ] **Step 4: Verify with TypeScript build**

Run: `npm --prefix frontend run build`
Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/tournament/ frontend/src/components/dock/ frontend/src/components/charts/
git commit -m "feat(frontend): create TelemetryTournamentDock with Elo matrix and live telemetry tabs"
```

---

### Task 5: Master Dockable Research Studio Shell & Navigation Integration

**Files:**
- Create: `frontend/src/components/layout/StudioHeader.tsx`
- Create: `frontend/src/components/layout/ScrubberTransportBar.tsx`
- Create: `frontend/src/components/layout/WorkbenchShell.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/index.css`

**Interfaces:**
- Consumes: `WorkbenchProvider`, `BoardCanvas`, `BrainInspectorPane`, `TelemetryTournamentDock`
- Produces: Complete unified single-window research studio with responsive split workbench, synchronized scrubber bar, and live WebSocket telemetry hooks.

- [ ] **Step 1: Implement StudioHeader & ScrubberTransportBar**

Create `StudioHeader.tsx` with:
- Branding: `🚂 Ticket to Ride RL Lab`
- Active Agent Pair: `[🤖 Agent A: PPO-v4 (1420 Elo)]` vs `[🤖 Agent B: Heuristic (1180 Elo)]`
- Studio Mode Switcher: `[🎮 Interactive Match]` `[📈 Live Training]` `[🎞️ Replay Studio]` `[🏆 Tournament]`
- Backend Connection Status pill with latency indicator.

Create `ScrubberTransportBar.tsx` with:
- Step scrubber slider (`0` to `maxSteps`).
- Transport buttons: `[⏮ First]` `[◀ Step Back]` `[▶ Play / ⏸ Pause]` `[Step Fwd ▶]` `[⏭ Last]`.
- Playback speed dropdown (`0.25x`, `0.5x`, `1x`, `2x`, `5x`).

- [ ] **Step 2: Implement WorkbenchShell**

Create `frontend/src/components/layout/WorkbenchShell.tsx`:
- Grid layout:
  - Top: `StudioHeader` + `ScrubberTransportBar`
  - Middle Left (60%): `BoardCanvas` (Map + Route Claims + Cards in hand)
  - Middle Right (40%): `BrainInspectorPane` (Value Gauge + Policy Distributions + Masks)
  - Bottom (Collapsible): `TelemetryTournamentDock`

- [ ] **Step 3: Update App.tsx and index.css**

Refactor `frontend/src/App.tsx` to wrap `WorkbenchProvider` around `WorkbenchShell` and apply dark obsidian/slate theme variables (`#090D16`, `#0F172A`, `#1E293B`, `#38BDF8`, `#818CF8`, `#34D399`, `#F43F5E`) in `index.css`.

- [ ] **Step 4: Verify with TypeScript build**

Run: `npm --prefix frontend run build`
Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/layout/ frontend/src/App.tsx frontend/src/index.css
git commit -m "feat(frontend): integrate master WorkbenchShell with dockable split-pane studio"
```

---

### Task 6: End-to-End Verification, Backend Telemetry Integration & Impeccable Quality Pass

**Files:**
- Create: `tests/api/test_workbench_telemetry_flow.py`
- Modify: `src/api/main.py`

**Interfaces:**
- Produces: Green backend tests, zero-warning frontend build, verified WebSocket streaming, and passing Impeccable quality checklist.

- [ ] **Step 1: Write and run backend workbench telemetry integration test**

Create `tests/api/test_workbench_telemetry_flow.py`:
- Test game session creation $\rightarrow$ bot step $\rightarrow$ brain inspection payload generation $\rightarrow$ replay frame recording.
- Run: `.venv/bin/pytest tests/api/test_workbench_telemetry_flow.py -v`
- Expected: PASS.

- [ ] **Step 2: Run full Python test suite**

Run: `.venv/bin/pytest tests/`
Expected: All tests pass.

- [ ] **Step 3: Run full TypeScript production build**

Run: `npm --prefix frontend run build`
Expected: Build passes with 0 errors.

- [ ] **Step 4: Run Impeccable detector check**

Run: `node /home/christian/.gemini/skills/impeccable/scripts/detect.mjs --json frontend/src/App.tsx`
Address any detected layout, contrast, or typography issues.

- [ ] **Step 5: Final commit and summary**

```bash
git add .
git commit -m "feat: complete TicketToRide RL Lab Web Suite Workbench & Telemetry Hub"
```
