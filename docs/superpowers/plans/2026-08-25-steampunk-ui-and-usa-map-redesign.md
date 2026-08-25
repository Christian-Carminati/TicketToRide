# Steampunk UI & USA Vector Cartography Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the entire TicketToRide RL Lab web frontend into an authentic Victorian Steampunk Cartographer & Analytical Laboratory theme with a high-clarity geographical vector USA map background.

**Architecture:** Implement a modular Steampunk design system with parchment CSS tokens, brass/copper borders, period typography (Cinzel, Playfair Display, Crimson Pro), and SVG procedural textures. Construct a dedicated, high-resolution vector cartography component (`UsaCartographyBackground.tsx`) for geographic landmass, Great Lakes, mountain hachures, and compass rose. Restyle all RL introspection gauges as steam pressure meters, thermionic tubes, and telegraphic ticker tapes.

**Tech Stack:** React 18, TypeScript 5, Vite 5, SVG Cartography, Lucide-React, CSS3 Variables & Filters.

**Spec:** `docs/superpowers/specs/2026-08-25-steampunk-ui-and-usa-map-redesign.md`

## Global Constraints
- Target workspace: `frontend/`
- Zero regression on existing game logic, WebSocket 60Hz telemetry, or state schemas.
- Ensure high contrast ($>4.5:1$ for body/data, $>3:1$ for large headings) on parchment backgrounds.
- Pure SVG vector art for the map to ensure razor-sharp rendering at all display resolutions.

---

### Task 1: Google Fonts, Global Steampunk CSS Tokens & Palette Constants

**Files:**
- Modify: `frontend/index.html:8-12`
- Modify: `frontend/src/index.css:1-68`
- Modify: `frontend/src/components/board/mapData.ts:22-34`

**Interfaces:**
- Produces: CSS variables (`--bg-parchment-base`, `--bg-panel-brass`, `--ink-primary`, `--ink-secondary`, `--ink-muted`, `--brass-gradient`, `--copper-gradient`, etc.) and updated `COLOR_HEX` constants.

- [ ] **Step 1: Update `frontend/index.html` with Victorian Google Fonts**
Add `Cinzel Decorative`, `Playfair Display`, `Crimson Pro`, and `Courier Prime` to `index.html`.

- [ ] **Step 2: Update `frontend/src/index.css` with Steampunk Theme Tokens**
Add parchment texture base, brass borders, rivet shadows, custom scrollbars, and typography defaults.

- [ ] **Step 3: Update `COLOR_HEX` in `mapData.ts` with Steampunk Jewel Hues**
Update route and card color hex values for rich, high-saturation contrast against parchment.

- [ ] **Step 4: Verify build with type check**
Run: `npm run build --prefix frontend`
Expected: Build succeeds.

- [ ] **Step 5: Commit changes**
```bash
git add frontend/index.html frontend/src/index.css frontend/src/components/board/mapData.ts
git commit -m "style: define steampunk parchment tokens, victorian fonts and jewel palette"
```

---

### Task 2: High-Clarity USA Vector Cartography Engine & Board Components

**Files:**
- Create: `frontend/src/components/board/UsaCartographyBackground.tsx`
- Modify: `frontend/src/components/board/BoardSVG.tsx`
- Modify: `frontend/src/components/board/CityNode.tsx`
- Modify: `frontend/src/components/board/RouteEdge.tsx`

**Interfaces:**
- Consumes: `mapName`, `SVG_WIDTH`, `SVG_HEIGHT` from `BoardSVG.tsx`.
- Produces: `<UsaCartographyBackground />` rendering coastlines, Great Lakes, mountain hachures, graticule grid, compass rose, and survey cartouche.

- [ ] **Step 1: Create `UsaCartographyBackground.tsx`**
Implement detailed vector paths for North America continent outline (Pacific coast, Atlantic coast, Gulf of Mexico, Florida, Canadian border), Great Lakes (Superior, Michigan, Huron, Erie, Ontario), mountain hachures (Rockies, Appalachians), compass rose, graticule lines, and survey cartouche.

- [ ] **Step 2: Update `BoardSVG.tsx` to integrate `UsaCartographyBackground`**
Embed `UsaCartographyBackground` beneath routes and cities, apply parchment filter and brass frame border.

- [ ] **Step 3: Update `CityNode.tsx` for Steampunk Brass Seals**
Render cities as brass rivets with dark sepia engraved typography and ivory halo outlines for crisp legibility.

- [ ] **Step 4: Update `RouteEdge.tsx` for Dual-Rail Wooden Tracks**
Render dual rails with wooden ties, jewel-colored train segments, and glowing brass hover/claimable highlights.

- [ ] **Step 5: Verify build with type check**
Run: `npm run build --prefix frontend`
Expected: Build succeeds.

- [ ] **Step 6: Commit changes**
```bash
git add frontend/src/components/board/UsaCartographyBackground.tsx frontend/src/components/board/BoardSVG.tsx frontend/src/components/board/CityNode.tsx frontend/src/components/board/RouteEdge.tsx
git commit -m "feat(board): add high-clarity USA vector cartography background and steampunk tracks"
```

---

### Task 3: Header, Scrubber Chronometer & Board Header Controls

**Files:**
- Modify: `frontend/src/components/layout/StudioHeader.tsx`
- Modify: `frontend/src/components/layout/ScrubberTransportBar.tsx`
- Modify: `frontend/src/components/board/BoardHeaderControls.tsx`

**Interfaces:**
- Consumes: `useWorkbench()`, `StudioMode`, `ObservabilityMode`.
- Produces: Steampunk navigation header, conductor's chronometer scrubber, and brass perception controls.

- [ ] **Step 1: Update `StudioHeader.tsx`**
Redesign header as an engraved brass instrument panel with steam locomotive crest, gear-shaped mode selectors, and brass status pills.

- [ ] **Step 2: Update `ScrubberTransportBar.tsx`**
Redesign as a brass railway chronometer with throttle range slider, machined brass playback buttons, and turn indicators.

- [ ] **Step 3: Update `BoardHeaderControls.tsx`**
Style perception toggle (God / Agent A / Agent B) and map selector with brass button groups and parchment inlays.

- [ ] **Step 4: Verify build with type check**
Run: `npm run build --prefix frontend`
Expected: Build succeeds.

- [ ] **Step 5: Commit changes**
```bash
git add frontend/src/components/layout/StudioHeader.tsx frontend/src/components/layout/ScrubberTransportBar.tsx frontend/src/components/board/BoardHeaderControls.tsx
git commit -m "feat(layout): style studio header, chronometer scrubber and board controls"
```

---

### Task 4: Victorian Analytical Laboratory (Brain Inspector & RL Instruments)

**Files:**
- Modify: `frontend/src/components/brain/ValueHeadGauge.tsx`
- Modify: `frontend/src/components/brain/ActionProbabilitiesChart.tsx`
- Modify: `frontend/src/components/brain/ObservationTensorViewer.tsx`
- Modify: `frontend/src/components/brain/BrainInspectorPane.tsx`

**Interfaces:**
- Consumes: `BrainInspectionDTO`, `useWorkbench()`.
- Produces: Steam Manometer gauge, pneumatic action bars, thermionic vacuum tube tensor activations.

- [ ] **Step 1: Redesign `ValueHeadGauge.tsx` as a Steam Pressure Manometer**
Render circular brass pressure gauge with needle indicator, PSI / Win Probability scale, and aged dial markings.

- [ ] **Step 2: Redesign `ActionProbabilitiesChart.tsx` as Pneumatic Frequency Levers**
Render probabilities with brass level indicators, barred locked marks for masked actions, and typewriter annotations.

- [ ] **Step 3: Redesign `ObservationTensorViewer.tsx` as Thermionic Filament Matrix**
Render tensor heatmap and layer activations with warm amber vacuum tube glow and copper wireframe lines.

- [ ] **Step 4: Update `BrainInspectorPane.tsx`**
Enclose analytical instruments in a riveted brass laboratory casing with model switcher plates.

- [ ] **Step 5: Verify build with type check**
Run: `npm run build --prefix frontend`
Expected: Build succeeds.

- [ ] **Step 6: Commit changes**
```bash
git add frontend/src/components/brain/ValueHeadGauge.tsx frontend/src/components/brain/ActionProbabilitiesChart.tsx frontend/src/components/brain/ObservationTensorViewer.tsx frontend/src/components/brain/BrainInspectorPane.tsx
git commit -m "feat(brain): transform neural inspector into steampunk steam manometer and thermionic displays"
```

---

### Task 5: Cards, Decks, Telegram Tickets & Telemetry Dock

**Files:**
- Modify: `frontend/src/components/cards/VisibleDeck.tsx`
- Modify: `frontend/src/components/cards/TrainCardHand.tsx`
- Modify: `frontend/src/components/cards/TicketsList.tsx`
- Modify: `frontend/src/components/dock/ActionLogStream.tsx`
- Modify: `frontend/src/components/dock/TelemetryTournamentDock.tsx`
- Modify: `frontend/src/components/charts/LineChartSVG.tsx`

**Interfaces:**
- Consumes: `GameStateDTO`, `TelemetryEventDTO`.
- Produces: 19th-century railway shares/tickets, Western Union destination telegrams, and telegraphic ticker stream.

- [ ] **Step 1: Redesign `VisibleDeck.tsx` and `TrainCardHand.tsx`**
Style card draw slots as brass spring-loaded dispensers; style train cards with ornate Victorian ticket borders.

- [ ] **Step 2: Redesign `TicketsList.tsx` as Telegrams with Wax Seals**
Style destination tickets as urgent telegram slips with red wax seal points and station stamps.

- [ ] **Step 3: Redesign `ActionLogStream.tsx` as Telegraph Ticker Tape**
Render action events on continuous parchment ticker tape with monospace typewriter font.

- [ ] **Step 4: Update `TelemetryTournamentDock.tsx` & `LineChartSVG.tsx`**
Style dock panel with brass tabs, leather trim, and sepia chart paper gridlines.

- [ ] **Step 5: Verify build with type check**
Run: `npm run build --prefix frontend`
Expected: Build succeeds.

- [ ] **Step 6: Commit changes**
```bash
git add frontend/src/components/cards/VisibleDeck.tsx frontend/src/components/cards/TrainCardHand.tsx frontend/src/components/cards/TicketsList.tsx frontend/src/components/dock/ActionLogStream.tsx frontend/src/components/dock/TelemetryTournamentDock.tsx frontend/src/components/charts/LineChartSVG.tsx
git commit -m "feat(cards-dock): style railway ticket cards, telegram tickets and telegraph ticker dock"
```

---

### Task 6: Views & Studio Shell Unification (Training, Tournament, Reports, GameView)

**Files:**
- Modify: `frontend/src/components/layout/WorkbenchShell.tsx`
- Modify: `frontend/src/components/board/BoardCanvas.tsx`
- Modify: `frontend/src/views/GameView.tsx`
- Modify: `frontend/src/views/TrainingView.tsx`
- Modify: `frontend/src/views/TournamentArenaView.tsx`
- Modify: `frontend/src/views/ReportsView.tsx`
- Modify: `frontend/src/components/tournament/EloMatrixHeatmap.tsx`
- Modify: `frontend/src/components/checkpoints/CheckpointManagerPane.tsx`

**Interfaces:**
- Unifies all studio modes (Interactive, Replay, Training, Tournament, Reports) under the Steampunk Cartographer & Analytical Laboratory theme.

- [ ] **Step 1: Update `WorkbenchShell.tsx` and `BoardCanvas.tsx`**
Apply parchment substrate, brass dividers, and consistent spacing across workbench views.

- [ ] **Step 2: Update `GameView.tsx`**
Apply parchment and brass controls to standalone game view and color selection modals.

- [ ] **Step 3: Update `TournamentArenaView.tsx`, `EloMatrixHeatmap.tsx`, `CheckpointManagerPane.tsx`**
Style tournament matrix, heatmaps, and checkpoint files as vintage railroad ledgers.

- [ ] **Step 4: Update `TrainingView.tsx` and `ReportsView.tsx`**
Style training metrics and research experiment reports with parchment document layouts.

- [ ] **Step 5: Verify build with type check**
Run: `npm run build --prefix frontend`
Expected: Build succeeds.

- [ ] **Step 6: Commit changes**
```bash
git add frontend/src/components/layout/WorkbenchShell.tsx frontend/src/components/board/BoardCanvas.tsx frontend/src/views/GameView.tsx frontend/src/views/TrainingView.tsx frontend/src/views/TournamentArenaView.tsx frontend/src/views/ReportsView.tsx frontend/src/components/tournament/EloMatrixHeatmap.tsx frontend/src/components/checkpoints/CheckpointManagerPane.tsx
git commit -m "feat(views): unify all workbench views with steampunk laboratory styling"
```

---

### Task 7: Quality Verification & Impeccable Mechanical Audit

**Files:**
- Test all modified frontend files

- [ ] **Step 1: Run TypeScript compiler and production build**
Run: `npm run build --prefix frontend`
Expected: Complete clean build without errors.

- [ ] **Step 2: Run Impeccable mechanical detector**
Run: `node /home/christian/.gemini/skills/impeccable/scripts/detect.mjs --json frontend/src/App.tsx frontend/src/components/layout/WorkbenchShell.tsx frontend/src/components/board/BoardSVG.tsx`
Expected: Clean report or actionable findings.

- [ ] **Step 3: Commit final updates and documentation**
```bash
git add .
git commit -m "chore: complete steampunk UI overhaul and USA vector map redesign"
```
