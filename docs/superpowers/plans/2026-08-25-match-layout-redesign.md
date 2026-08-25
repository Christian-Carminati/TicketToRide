# Match Layout Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the primary Ticket to Ride match screen (homepage / interactive mode) into "The Master Conductor Table": a clean, full-width cartography board with prominent player score plaques, clear turn indicator, victory/game-over modal with score breakdown, 3-column lower station table, and on-demand slide-out AI telemetry drawer.

**Architecture:** Replace the cramped 2-column split (Board + Fixed BrainInspector) with a full-width board layout. Create a dedicated `MatchScoreBoardHUD` with high-contrast P1 vs P2 comparison, a `GameOverModal` for rich victory celebration, a 3-column station table (`VisibleDeck`, `TrainCardHand`, `TicketsList`), and an `AITelemetryDrawer` for optional inspection.

**Tech Stack:** React 18, TypeScript, Tailwind/Custom CSS Steampunk Design System, Lucide Icons, Vite.

**Spec:** `docs/superpowers/specs/2026-08-25-match-layout-redesign-design.md`

## Global Constraints
- Aesthetic direction: Vintage Steampunk Cartographer & Victorian Analytical Lab (warm parchment palette, brass/bronze/copper trims, high-contrast readable typography).
- Deterministic and partial observability compliance (never leak hidden cards or opponent tickets).
- Responsive grid supporting desktop (1800px max) down to tablet.
- Remove raw neural telemetry (Critic Steam Manometer, Filaments 464d, Policy Actuators π(a|s)) from the default layout; make it accessible only via the slide-out drawer.

---

### Task 1: Create `GameOverModal.tsx` Victory Component

**Files:**
- Create: `frontend/src/components/board/GameOverModal.tsx`
- Modify: `frontend/src/components/board/index.ts`

**Interfaces:**
- Props:
  ```typescript
  interface GameOverModalProps {
    gameState: GameStateDTO;
    onRematch?: () => void;
    onClose?: () => void;
  }
  ```

- [ ] **Step 1: Create GameOverModal with winner detection and score breakdown**
  - Calculate winner based on highest score.
  - Display Victorian gilded frame modal with winner banner, score margin, and score breakdown.
  - Provide buttons: "🎮 Nuova Partita / Rivincita" and "🗺️ Esplora Tabellone".

- [ ] **Step 2: Export in `frontend/src/components/board/index.ts`**

- [ ] **Step 3: Verify TypeScript compilation**

---

### Task 2: Create `MatchScoreBoardHUD.tsx` Component

**Files:**
- Create: `frontend/src/components/board/MatchScoreBoardHUD.tsx`
- Modify: `frontend/src/components/board/index.ts`

**Interfaces:**
- Props:
  ```typescript
  interface MatchScoreBoardHUDProps {
    gameState: GameStateDTO | null;
    isAutoplaying?: boolean;
    onToggleAutoplay?: () => void;
    autoplaySpeedMs?: number;
    onSpeedChange?: (speed: number) => void;
    onSelectMap?: (mapName: 'usa' | 'mini') => void;
    onNewMatch?: (p1Type: string, p2Type: string, map: 'usa' | 'mini', ckpt1?: string, ckpt2?: string) => void;
    onToggleTelemetryDrawer?: () => void;
    isTelemetryDrawerOpen?: boolean;
    selectedModel?: string;
  }
  ```

- [ ] **Step 1: Implement MatchScoreBoardHUD**
  - P1 and P2 Score plaques with large bold numbers (`48 pts`), trains remaining counter (`🚂 38/45`), cards in hand count, tickets count.
  - Active turn indicator with glowing brass border and actionable turn prompt.
  - Match Setup / Duel Configuration dropdown.
  - Map selector (USA 1885 / Mini).
  - Autoplay / Pause button + speed range slider.
  - `🧠 Telemetria AI` toggle button.

- [ ] **Step 2: Export in `frontend/src/components/board/index.ts`**

---

### Task 3: Create `AITelemetryDrawer.tsx` Component

**Files:**
- Create: `frontend/src/components/brain/AITelemetryDrawer.tsx`
- Modify: `frontend/src/components/brain/index.ts`

**Interfaces:**
- Props:
  ```typescript
  interface AITelemetryDrawerProps {
    isOpen: boolean;
    onClose: () => void;
  }
  ```

- [ ] **Step 1: Implement slide-out drawer with backdrop blur**
  - Render backdrop and sliding right panel (440px width).
  - Embed existing `BrainInspectorPane` inside.
  - Add clean header with close button (X) and model info.

- [ ] **Step 2: Export in `frontend/src/components/brain/index.ts`**

---

### Task 4: Refactor `BoardCanvas.tsx` for 3-Column Lower Station Platform

**Files:**
- Modify: `frontend/src/components/board/BoardCanvas.tsx`

- [ ] **Step 1: Replace old BoardHeaderControls with new MatchScoreBoardHUD**
- [ ] **Step 2: Render 3-column lower station platform grid**
  - Column 1: `VisibleDeck` (5 face-up cards, train deck, tickets deck).
  - Column 2: `TrainCardHand` (active player hand).
  - Column 3: `TicketsList` (destination tickets with completion status).
- [ ] **Step 3: Include `GameOverModal` when `gameState.is_game_over === true`**

---

### Task 5: Update `WorkbenchShell.tsx` Layout

**Files:**
- Modify: `frontend/src/components/layout/WorkbenchShell.tsx`

- [ ] **Step 1: Update interactive & replay modes to use full-width board**
  - Remove fixed side-by-side `BrainInspectorPane`.
  - Add state `isTelemetryDrawerOpen`.
  - Wire `AITelemetryDrawer` to the drawer toggle in the HUD.
- [ ] **Step 2: Verify all game actions (draw, claim route, bot step, autoplay, new match) work seamlessly**

---

### Task 6: Add CSS Styles & Responsive Polish

**Files:**
- Modify: `frontend/src/index.css`

- [ ] **Step 1: Add HUD score badge animations and active-turn pulse**
- [ ] **Step 2: Add 3-column lower station responsive grid styles**
- [ ] **Step 3: Add drawer slide animation (`@keyframes slideInRight`)**

---

### Task 7: Verification & Build Check

**Files:**
- Run: `npm run build` inside `frontend/`
- Run: python tests `pytest tests/ -v`

- [ ] **Step 1: Verify frontend build compiles with 0 errors**
- [ ] **Step 2: Verify Python backend tests pass**
