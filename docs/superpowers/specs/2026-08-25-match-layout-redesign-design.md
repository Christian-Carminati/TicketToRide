# Design Spec: Match Layout Redesign (The Master Conductor Table)

## 1. Overview & Goals
The goal of this redesign is to transform the interactive game screen (homepage / interactive mode) into an immersive, crystal-clear, Victorian-styled board game experience. The current raw neural telemetry panel (Critic Steam Manometer, Filaments 464d, Policy Actuators π(a|s)) will be removed from the default view and relocated to an on-demand slide-out drawer ("🧠 Telemetria AI").

### Key Priorities:
1. **Crystal Clear Scoring & Player Status:** Large, unmistakable score readouts, train counts (out of 45), cards in hand, and destination tickets for both Player 1 and Player 2.
2. **Definitive Game-Over / Victory Screen:** A prominent Victorian Victory Plaque / Modal that clearly announces the winner, shows a detailed score breakdown (routes, completed tickets, penalty for incomplete tickets), and offers one-click rematch.
3. **Full-Width Hero Map:** The USA 1885 / Mini map takes center stage with maximum visual clarity and breathing room.
4. **Organized 3-Column Lower Station Table:**
   - Left: 5 Face-up cards + Draw Train Deck + Destination Tickets Deck.
   - Center: Active Player Hand (color chips with clear quantity badges).
   - Right: Destination Tickets list with real-time route completion status (✅/📜) and bonus points.
5. **Recent Moves Ticker & Turn Indicator:** Clear feedback on who is playing and what action was just dispatched.
6. **On-Demand AI Telemetry Drawer:** Available via a top brass button for researchers/curious players, without cluttering the game table.

---

## 2. Component Architecture & Changes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. StudioHeader (Sticky Top)                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. MatchScoreBoardHUD                                                       │
│    ├── Player 1 Plaque (Name, Big Score pts, 🚂 38/45, 🃏 8, 🎫 3)          │
│    ├── Turn Indicator Banner (Active Conductor + Action hint)               │
│    ├── Controls (Map USA/Mini, Duel Setup, Autoplay, 🧠 Telemetria Drawer)  │
│    └── Player 2 Plaque (AI / Opponent, Big Score pts, 🚂 32/45, 🃏 10)      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Recent Move Ticker (Chronological last move notification)                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. BoardCanvas (Full-Width USA / Mini Cartography Map)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. Lower Station Platform (3-Column Responsive Grid)                        │
│    ├── VisibleDeck (5 Face-up train vouchers, Hidden deck, Tickets deck)    │
│    ├── TrainCardHand (Active player cards by color with count badges)       │
│    └── TicketsList (Destination telegrams with completion checkmarks)       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6. GameOverModal (Triggered when is_game_over === true)                     │
│    ├── 🏆 Winner Announcement Banner                                        │
│    ├── Score Breakdown Comparison Table (Routes, Tickets, Penalties, Total) │
│    └── [🎮 Rivincita / Nuova Partita] [🎬 Analizza Replay]                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 7. AITelemetryDrawer (Slide-out side sheet, toggleable via HUD button)     │
│    └── Contains BrainInspectorPane (ValueGauge, PolicyChart, Filaments)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed UI Specifications

### 3.1 Match Scoreboard HUD (`MatchScoreBoardHUD.tsx`)
- **P1 Plaque (Blue Conductor):**
  - Player Avatar & Name (e.g. `👤 Conductor (Tu)`).
  - Score badge: Huge font (`48 pts`) in red/gold enamel plate.
  - Resource bar: `🚂 38 / 45 Treni` (with mini visual bar), `🃏 7 Carte`, `🎫 3 Biglietti`.
  - Active Turn Glow: Bright brass border and pulsing steam indicator when it is P1's turn.
- **Center Turn & Control Hub:**
  - Turn indicator: `Turno #14: È il tuo turno! Clicca una tratta sulla mappa o pesca carte`.
  - Match Setup button (configure models / duel).
  - Autoplay / Pause button with speed slider.
  - Map switch: `USA 1885` | `Mini`.
  - `🧠 Telemetria AI` toggle button (with badge indicating model e.g. AlphaZero / PPO).
- **P2 Plaque (Red Locomotive - AI):**
  - AI Avatar & Name (e.g. `🤖 AlphaZero (PUCT MCTS)`).
  - Score badge: Huge font (`42 pts`).
  - Resource bar: `🚂 31 / 45 Treni`, `🃏 9 Carte`.
  - Active Turn Glow: Red-gold pulsing indicator when AI is computing.

### 3.2 Victory / Game Over Screen (`GameOverModal.tsx`)
When `gameState.is_game_over === true`:
- Modal overlay with blurred background and Victorian gilded frame.
- **Crown Banner:**
  - `🏆 VITTORIA TRIONFALE: PLAYER 1 HA VINTO!` (or AI name if AI won, or `PAREGGIO!`).
  - Subtitle: `Partita conclusa in 36 turni con un margine di +14 punti!`.
- **Side-by-Side Score Card:**
  | Categoria | Player 1 | Player 2 (AI) |
  | :--- | :---: | :---: |
  | Punti Tratte Reclamate | 38 pts | 42 pts |
  | Biglietti Completati | +22 pts (2) | +11 pts (1) |
  | Biglietti Falliti | -0 pts (0) | -9 pts (1) |
  | **PUNTEGGIO FINALE** | **60 pts** 👑 | **44 pts** |
- **Buttons:**
  - `🎮 Gioca di Nuovo (Rivincita)`
  - `🗺️ Esplora Tabellone Finale` (closes modal to view final claimed routes)
  - `🎬 Rivedi Partita nello Scrubber Replay`

### 3.3 Full-Width Board & 3-Column Lower Station
- **Board:** Takes full container width (max 1600px, aspect-ratio preserved), centered, with high clarity and crisp route highlights.
- **Station Platform (Lower Deck):**
  - **Left (Vetrina Carte):** 5 Face-up cards with distinct vivid train colors + Train Draw Deck (with card count) + Telegrams Draw Deck.
  - **Center (Mano del Giocatore):** Color-grouped cards (Red, Blue, Green, Yellow, Orange, Purple, Black, White, Locomotive) with large numerical counters.
  - **Right (Biglietti Destinazione):** Scrollable list of active tickets, showing origin-destination, point value, and real-time status (✅ Completato / 📜 In corso).

### 3.4 Slide-Out AI Telemetry Drawer (`AITelemetryDrawer.tsx`)
- Clicking `🧠 Telemetria AI` opens a side panel sliding smoothly from the right (420px width).
- Contains the full `BrainInspectorPane` (Critic Steam Manometer, Policy Actuators, MCTS Tree Search visualizer, Filament Array).
- Easily dismissible via an `✕ Chiudi Telemetria` button or backdrop click.

---

## 4. Implementation Steps
1. Create `MatchScoreBoardHUD.tsx` in `frontend/src/components/board/` to replace the old cramped header.
2. Create `GameOverModal.tsx` in `frontend/src/components/board/` for rich end-of-game victory celebrations and score breakdowns.
3. Create `AITelemetryDrawer.tsx` in `frontend/src/components/brain/` for optional slide-out inspection.
4. Refactor `BoardCanvas.tsx` to integrate the new 3-column lower station table and the new HUD.
5. Update `WorkbenchShell.tsx` to use full-width layout without fixed side telemetry.
6. Verify responsive behavior across desktop and tablet screen sizes.
