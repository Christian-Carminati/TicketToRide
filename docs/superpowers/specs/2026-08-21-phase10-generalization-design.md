# Design Specification: Phase 10 — Generalization & Procedural Maps

**Date:** 2026-08-21  
**Phase:** 10 (Generalization, Procedural Environments, Official Europe Board, Cross-Map Evaluation)  
**Status:** Approved by User  

---

## 1. Executive Summary & Goals

Phase 10 addresses the core scientific question of TicketToRide RL Lab:
> **"Did the RL agent learn general strategic abstractions for Ticket to Ride, or did it merely overfit and memorize the standard USA map topology?"**

To answer this question systematically, Phase 10 introduces:
1. **Deterministic Procedural Map Generator (`ProceduralMapGenerator`):** Graph-theoretic procedural generation of playable, connected boards with balanced colors, lengths, and destination tickets with point values derived from shortest-path graph distances.
2. **Official Europe Board (`load_europe_board`):** Implementation of the classic Ticket to Ride Europe board (cities, routes, color assignments, double routes, and European destination tickets).
3. **Map Split & Dataset Infrastructure (`MapSplit`, `ProceduralMapDataset`):** Formal partitioning into Train, Validation, and Test (unseen) map distributions.
4. **Multi-Map Training Environment (`MultiMapTicketToRideEnv`):** Gymnasium-compatible environment that samples topologies across resets, enabling multi-environment RL training.
5. **Generalization Evaluation Framework (`GeneralizationEvaluator`, `GeneralizationBenchmarkRunner`):** Mathematical formulation of Generalization Gap ($\Delta_{\text{gen}}$), Relative Retention Rate ($R_{\text{ret}}$), Cross-Map Win Rate retention, and Ticket Completion Efficiency.
6. **CLI & Acceptance Suite:** Dedicated script `scripts/benchmark_generalization.py` and comprehensive test coverage (`test_procedural_maps.py`, `test_europe_board.py`, `test_multi_map_env.py`, `test_generalization.py`, `test_phase10_acceptance.py`).

---

## 2. Mathematical Formalization & Metrics

### 2.1 Generalization Gap ($\Delta_{\text{gen}}$)
For an agent evaluated over a set of training maps $\mathcal{M}_{\text{train}}$ and a set of unseen test maps $\mathcal{M}_{\text{test}}$:

$$\overline{S}_{\text{train}} = \frac{1}{|\mathcal{M}_{\text{train}}|} \sum_{m \in \mathcal{M}_{\text{train}}} \mathbb{E}[\text{Score}(m)]$$

$$\overline{S}_{\text{test}} = \frac{1}{|\mathcal{M}_{\text{test}}|} \sum_{m' \in \mathcal{M}_{\text{test}}} \mathbb{E}[\text{Score}(m')]$$

$$\Delta_{\text{gen}} = \overline{S}_{\text{train}} - \overline{S}_{\text{test}}$$

A lower generalization gap indicates that policy performance does not degrade when encountering novel topologies.

### 2.2 Relative Performance Retention ($R_{\text{ret}}$)
$$R_{\text{ret}} = \frac{\overline{S}_{\text{test}}}{\max(1.0, \overline{S}_{\text{train}})} \times 100\%$$

### 2.3 Cross-Map Win Rate & Ticket Completion Rate
- **Cross-Map Win Rate ($WR_{\text{test}}$):** Head-to-head win rate against baseline opponents (Random, Greedy, Strategic) across unseen test maps.
- **Ticket Completion Rate ($TCR_{\text{test}}$):** Percentage of destination tickets successfully fulfilled on novel graph layouts.

---

## 3. Architecture & Component Design

### 3.1 Procedural Map Generation (`src/game/procedural.py`)

#### Algorithm
1. **City Sampling:**
   - Sample $N$ cities with coordinates $(x, y) \in [0.05, 0.95]^2$.
   - Enforce minimum pairwise Euclidean distance $d_{\min} = 0.15$ using rejection sampling.
2. **Topology & Connectivity:**
   - Compute the Minimum Spanning Tree (MST) on the complete Euclidean distance graph to guarantee 100% graph connectivity (no disconnected cities or subgraphs).
   - Add $k$ nearest-neighbor edges to reach the desired target route count $R$, ensuring redundant pathways and strategic alternative routes.
3. **Route Construction:**
   - Assign route length $L \in \{1, 2, 3, 4, 5, 6\}$ based on normalized Euclidean distances.
   - Assign colors in a balanced cyclic or sampled manner across the 8 standard colors (`PURPLE`, `WHITE`, `BLUE`, `YELLOW`, `ORANGE`, `BLACK`, `RED`, `GREEN`) and gray (`None`).
   - Add double routes for top-density pairs if specified.
4. **Destination Ticket Generation:**
   - Compute all-pairs shortest paths using BFS/Dijkstra on the generated board graph.
   - Sample $T$ city pairs $(u, v)$ with graph distance $d(u, v) \ge 2$.
   - Assign ticket point values proportional to the shortest path train length:
     $$\text{points}(u, v) = \max(2, \min(22, \lfloor d_{\text{path}}(u, v) \times 1.2 \rfloor))$$

```python
class ProceduralMapConfig:
    num_cities: int = 8
    num_routes: int = 14
    num_tickets: int = 10
    allow_double_routes: bool = False
    min_city_distance: float = 0.15

class ProceduralMapGenerator:
    def __init__(self, config: ProceduralMapConfig | None = None) -> None: ...
    def generate(self, seed: int) -> tuple[Board, list[DestinationTicket]]: ...
```

### 3.2 Official Europe Board (`src/game/maps.py`)

Implementation of `load_europe_board() -> tuple[Board, list[DestinationTicket]]`:
- **Cities (47 cities):** Amsterdam, Athina, Barcelona, Berlin, Brest, Brindisi, Bruxelles, Bucuresti, Budapest, Cadiz, Constantinopla, Danzig, Dieppe, Edinburgh, Erzurum, Essen, Frankfurt, Kobenhavn, Kyiv, Lisboa, London, Madrid, Marseille, Moskva, Munchen, Palermo, Paris, Petrograd, Riga, Roma, Rostov, Sarajevo, Sevastopol, Smolensk, Smyrna, Sofia, Stockholm, Venezia, Wien, Wilno, Zagreb, Zurich, etc.
- **Routes (100+ routes):** Authentic lengths, standard colors, and double routes.
- **Tickets (46 tickets):** Long tickets (e.g. Brest-Petrograd 20, Cadiz-Stockholm 21, Edinburgh-Athina 21, Kobenhavn-Erzurum 21) and standard regular tickets.

### 3.3 Dataset Partitioning (`src/game/procedural.py`)

```python
class MapSplit:
    train_maps: list[tuple[Board, list[DestinationTicket]]]
    val_maps: list[tuple[Board, list[DestinationTicket]]]
    test_maps: list[tuple[Board, list[DestinationTicket]]]

class ProceduralMapDataset:
    def __init__(self, generator: ProceduralMapGenerator) -> None: ...
    def create_split(self, train_seeds: list[int], val_seeds: list[int], test_seeds: list[int]) -> MapSplit: ...
```

### 3.4 Multi-Map Gymnasium Environment (`src/environment/multi_map_env.py`)

`MultiMapTicketToRideEnv` encapsulates training across multiple procedural maps:
- Inherits from `gym.Env` (or wraps `TicketToRideEnv`).
- On `reset(seed=...)`, selects a map from the assigned map dataset (round-robin or random sampling).
- Standard canonical topology wrapper guarantees uniform observation vector shape and discrete action space dimension across all procedural maps sharing the same `ProceduralMapConfig`.

### 3.5 Generalization Evaluator & Benchmark Runner (`src/evaluation/generalization.py`)

#### `GeneralizationEvaluator`
- Evaluates any agent on a target map list or `MapSplit`.
- Runs $N$ games per map with alternating seat positions (P0 vs P1).
- Returns aggregated metrics across train, val, and test splits.

#### `GeneralizationBenchmarkRunner`
- Automates the full experimental study:
  1. Generates standard Train / Val / Test map splits.
  2. Evaluates zero-shot baseline heuristics (`StrategicAgent`, `GreedyAgent`, `RandomAgent`).
  3. Trains single-map RL agent (`PPO_SingleMap`) vs multi-map RL agent (`PPO_MultiMap`).
  4. Runs cross-map evaluation on unseen test maps and official boards (USA $\leftrightarrow$ Europe).
  5. Computes Generalization Gap, Retention Rate, and cross-map win rates.
  6. Exports structured JSON (`phase10_report.json`) and Markdown (`phase10_report.md`).

---

## 4. Acceptance Criteria & Test Plan

1. **Procedural Maps Unit Tests (`tests/game/test_procedural_maps.py`):**
   - Seed determinism: same seed produces identical boards, routes, and tickets.
   - Graph connectivity: BFS traversal visits 100% of cities on all generated maps.
   - Valid routes & tickets: lengths between 1 and 6, valid colors, tickets reference existing cities with positive points.
2. **Europe Board Unit Tests (`tests/game/test_europe_board.py`):**
   - Board loads cleanly with valid cities, routes, and destination tickets.
   - Playable with Game Core engine.
3. **Multi-Map Environment Tests (`tests/environment/test_multi_map_env.py`):**
   - Resets across diverse maps without errors.
   - Valid action masking and bounded observation vectors.
4. **Generalization Evaluation Tests (`tests/evaluation/test_generalization.py`):**
   - Accurate computation of Generalization Gap and Retention Rates.
5. **Acceptance Suite (`tests/evaluation/test_phase10_acceptance.py`):**
   - End-to-end execution of the generalization benchmark suite verifying all deliverables.
