---
target: frontend/src/App.tsx
total_score: 35
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 1
timestamp: 2026-08-25T09-43-51Z
slug: frontend-src-app-tsx
---
Method: dual-agent (A: 2978d57a-dc41-4083-a4cb-a1eb0d234d89 · B: 575cab97-f6f5-4bf0-9cb2-516a25ba7637)

## Design Health Score

| # | Heuristic | Score | Key Observation |
|---|---|:---:|---|
| 1 | Visibility of System Status | 4 | "Telegraph 60Hz" lamp, active turn badge, chronometer (Step X/Y), live FPS, and WebSocket indicator provide constant real-time feedback. |
| 2 | Match System / Real World | 4 | Victorian steampunk cartographer and steam laboratory metaphors map naturally to RL concepts (manometer for value, telegraph for telemetry). |
| 3 | User Control and Freedom | 3 | Rich transport controls (First, Prev, Play/Pause, Next, Last, Range slider, Speed dial) and New Match reset. |
| 4 | Consistency and Standards | 3 | Consistent panel framing and typography across views. Minor bilingual copy leakage (some Italian buttons mixed with English). |
| 5 | Error Prevention | 4 | Neural action masking locks invalid decisions (opacity: 0.5, locked badge); claim modal disambiguates multi-color route claims; confirm guards for purges. |
| 6 | Recognition Rather Than Recall | 4 | Synchronized hover linking from ActionProbabilities to board tracks; city highlighting on destination ticket hover; face-up card slots with stock counts. |
| 7 | Flexibility and Efficiency | 3 | Spacebar turn stepping; preset filter toggles (valid, top10, all); speed throttle controls. |
| 8 | Aesthetic and Minimalist Design | 4 | Immersive steampunk world without visual clutter; ivory halo strokes under labels ensure zero visual interference from underlying terrain engravings. |
| 9 | Error Recovery | 3 | Themed red error panels (#FEE2E2 with #B91C1C border) with clear diagnostic messages and empty states explaining missing prerequisites. |
| 10 | Help and Documentation | 3 | Built-in Research Reports reader for scientific Markdown dispatches and benchmark logs; descriptive badges on RL parameters. |
| **Total** | | **35/40** | **Good (87.5% — High Distinction)** |

---

## Design Specificity Verdict

**LLM Assessment**: The visual composition is **deeply grounded, authentic, and bespoke** to the Ticket to Ride RL Lab. Rather than defaulting to generic dark-mode cyberpunk neon or sterile flat SaaS tables, the interface establishes a cohesive 1880s Victorian Analytical Laboratory aesthetic:
- **Cartographical Authenticity**: Vector USA cartography implementing 19th-century hydrographic survey and topographical techniques: procedural parchment noise grain (`#F9F3E5` to `#E2CFAC`), subtle coastal engraving waves, 45° mountain ridge hachures for the Rockies and Appalachians, Great Lakes contours, geodetic graticule lines (latitude/longitude degree annotations), Mississippi river engraving, multi-tier Victorian nautical compass rose, and antique ornamental cartouche title plaque (*"UNITED STATES RAILWAY SURVEY — CARTOGRAPHICAL REINFORCEMENT LEARNING LAB • 1885"*).
- **Physicalized Neural Telemetry**:
  - *Critic State Value Head V(s)*: Circular brass steam manometer gauge with pivoting needle, safe/high-pressure zones, and PSI calibration dial.
  - *Policy Distribution π(a|s)*: Pneumatic copper/brass actuator progress bars with star indicators for greedy actions and padlock icons for masked invalid moves.
  - *Observation Tensor*: Thermionic vacuum tube filament array with voltage potentials (-1.0V ↔ +1.0V).
  - *Live Stream*: Telegraph 60Hz indicator lamp.

**Deterministic Scan**:
- Executed `node .gemini/skills/impeccable/scripts/detect.mjs --json` across all board, brain, cards, dock, layout, and view components.
- Result: **0 findings, 0 antipatterns triggered**.
- No generic AI slop markers (no side-tab accent borders, no decorative text gradients, no unstyled cream palettes, no overused generic system fonts).

---

## Overall Impression

A standout, museum-grade transformation that bridges rigorous reinforcement learning diagnostics with rich 19th-century Victorian cartography. The high-contrast jewel tracks and ivory text halos solve the core requirement of clear board legibility over realistic topography, while the steam manometer and telegraph ticker elevate the neural workbench from a clinical spreadsheet into an evocative laboratory instrument.

---

## What's Working

1. **High-Clarity Vector USA Cartography**: Coastlines, Great Lakes, and engraved mountain hachures provide rich geographical immersion without ever obscuring the 36 city nodes or 100 railway routes.
2. **Synchronized Neural-to-Board Hover Linkage**: Hovering over policy distribution items in the Brain Inspector pane instantly illuminates the corresponding railway track on the map with a glowing steam pulse and brass neural plaque (P: XX%).
3. **Dense, High-Signal Laboratory Architecture**: Clean spatial chunking across interactive play, counterfactual scrubbing, live training curves, Elo tournament heatmaps, and scientific report dispatches.

---

## Priority Issues

### [P1] Localization Consistency: Standardize to English Across All Views
- **Why it matters**: In StudioHeader, TournamentArenaView, GameView, and ReportsView, some strings remain in Italian ("Nuova Partita", "Matrice Torneo & Elo", "Archivio Checkpoint", "Registro Esperimenti", "Eliminare il report"), disrupting the international scientific research persona.
- **Fix**: Standardize all button labels, tab titles, and dialog messages to English ("New Match", "Tournament Matrix & Elo", "Checkpoint Archive", "Experiment Registry", "Delete dispatch").
- **Suggested command**: `/impeccable polish`

### [P2] Responsive Layout: Flexible Split-Pane Stacking on Smaller Displays
- **Why it matters**: The grid layout in WorkbenchShell assumes a wide desktop viewport (≥1050px). On smaller split screens or 13" laptops, the right inspector pane can get cramped.
- **Fix**: Add a responsive CSS breakpoint (`@media (max-width: 1100px)`) to automatically stack the board canvas and brain inspector vertically.
- **Suggested command**: `/impeccable adapt`

### [P3] Keyboard Navigation & Accessibility Enhancements
- **Why it matters**: Power users (Alex) and accessibility users (Sam) benefit from global hotkeys (e.g. arrow keys for timeline scrubber, 1-5 for mode tabs, Space for stepping) and focusable SVG nodes.
- **Fix**: Add global window keydown listener in WorkbenchShell and add tabIndex / ARIA attributes to interactive board routes.
- **Suggested command**: `/impeccable audit`

---

## Persona Red Flags

- **Alex (Power User / RL Researcher)**: Currently relies heavily on clicking the scrubber buttons; adding ArrowLeft / ArrowRight scrubbing and Spacebar step in the main workbench will drastically accelerate counterfactual analysis.
- **Jordan (First-Timer / ML Student)**: Might benefit from an optional introductory tooltip plaque explaining how the steam pressure manometer (V(s)) relates to game advantage.
- **Sam (Accessibility-Dependent / Keyboard-Only)**: SVG track routes and city nodes require mouse hovering for tooltip inspection; adding keyboard focus rings will ensure complete keyboard parity.

---

## Minor Observations

- The procedural SVG noise filter (`#carto-parchment-grain`) in `UsaCartographyBackground.tsx` renders smoothly on modern hardware; a solid gradient fallback ensures resilience on low-power devices.
- Number formatters across telemetry dock and Elo table are cleanly aligned to monospace fonts.

---

## Questions to Consider

1. **Acoustic Feedback**: What if the Victorian Telegraph status lamp was accompanied by an optional, subtle mechanical relay click or typewriter chime when injecting bot steps?
2. **Counterfactual Branching**: Could the scrubber transport bar support a "Fork Timeline" feature allowing researchers to intervene at Step N and simulate divergent game futures?
3. **Entropy Wind Rose**: Could the Victorian nautical compass rose dynamically pulse its star points to visualize policy entropy in real time?
