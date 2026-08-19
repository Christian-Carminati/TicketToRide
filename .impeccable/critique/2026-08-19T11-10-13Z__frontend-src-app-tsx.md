---
target: frontend/src/App.tsx
total_score: 36
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 0
timestamp: 2026-08-19T11-10-13Z
slug: frontend-src-app-tsx
---
#### Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 4 | Real-time WebSocket connection beacon, turn indicators, training step counters, and FPS metrics. |
| 2 | Match System / Real World | 4 | Faithful Ticket to Ride board layouts (USA & Mini), standard train cards and route colors matching board game rules. |
| 3 | User Control and Freedom | 3 | Step next turn, autoplay toggle with variable speed (100ms-1.5s), start/stop training, replay scrubbing controls. |
| 4 | Consistency and Standards | 4 | Cohesive dark cyber-lab theme (`#0F172A`), unified typography, consistent card palettes across all 6 views. |
| 5 | Error Prevention | 4 | Strict action masking on invalid moves, disabled interactive controls during bot turns, route claim confirmation modals. |
| 6 | Recognition Rather Than Recall | 4 | Visual board tracks, route highlights, city connectivity glows, destination ticket progress tracking. |
| 7 | Flexibility and Efficiency | 3 | Multi-speed autoplay, direct frame jumping via scrubber, filterable experiment registry and action distribution charts. |
| 8 | Aesthetic and Minimalist Design | 4 | Zero CSS-jank hardware-accelerated GPU transforms (`transform: scaleX/scaleY`), uncluttered 2x2 telemetry charts. |
| 9 | Error Recovery | 3 | Clear banner feedback on invalid moves or network disconnects with auto-reconnection backoff. |
| 10 | Help and Documentation | 3 | Contextual tooltips on route edges, hover explanations for POMDP observation feature channels. |
| **Total** | | **36/40** | **Excellent** |

#### Design Specificity Verdict

**LLM assessment**: The interface is deeply tailored for reinforcement learning research in board games. Rather than generic tabular views, it provides an end-to-end laboratory with interactive SVG board topologies, live POMDP observation feature maps, action masking distributions ($\pi_\theta(a|s)$), real-time loss/reward telemetry streaming via WebSockets, and frame-by-frame match replay scrubbing.

**Deterministic scan**: Automated `detect.mjs` scan returned 0 anti-patterns across all frontend components. GPU hardware-accelerated transforms (`scaleX`, `scaleY`) are used for 60fps smooth animations.

#### Overall Impression
A highly responsive, production-ready RL Web Lab combining interactive gameplay with deep neural network introspection and training telemetry.

#### What's Working
1. **Interactive SVG Board Vector Rendering**: Clean projection of normalized coordinates with parallel double-route offsets and hover/claim highlights.
2. **Real-time Telemetry & Live Streaming Hub**: Seamless WebSocket streaming for training metrics (loss, reward, entropy, win rate) with zero-dependency SVG line charts.
3. **Deep Brain Introspection**: Live decomposition of layer activations, critic value estimates, and visual policy distribution with active vs masked action badges.

#### Priority Issues
- **[P2] Keyboard Shortcuts**: Add spacebar shortcut for Step Turn / Autoplay toggle in `GameView` and `ReplayView`.
  - *Why it matters*: Enhances power-user interaction speed during rapid match evaluations.
  - *Fix*: Bind `keydown` listeners for Space (step/pause) and Arrow keys (prev/next frame).
  - *Suggested command*: `/impeccable polish`
- **[P3] Ticket Path Highlighting**: Highlight all connecting board routes for destination tickets on hover in `TicketsList`.
  - *Why it matters*: Gives immediate spatial feedback for route completion progress.
  - *Fix*: Connect ticket city pairs to active claimed route pathfinding.
  - *Suggested command*: `/impeccable delight`

#### Persona Red Flags
- **Alex (Power User)**: Keyboard navigation in replay player could be expedited with hotkeys (Space for play/pause, Left/Right for frame steps).
- **Jordan (First-Timer)**: Explanations of RL terms (Policy Loss, Value Loss, GAE) are accessible in tooltips and subtext.
- **Sam (Accessibility)**: SVG routes feature accessible `<title>` tooltips and high contrast color borders (WCAG AA compliant).
