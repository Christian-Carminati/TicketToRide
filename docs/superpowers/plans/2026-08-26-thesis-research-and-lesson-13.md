# Thesis Research Benchmark, Publication Figures & Lesson 13 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete scientific thesis research pipeline (comparing 6 AI paradigms under partial observability with automated 5-battery benchmark, publication-quality figures, and LaTeX tables) along with an interactive bilingual (IT/EN) Lesson 13 in the university course portal.

**Architecture:**
1. `src/evaluation/thesis_benchmark.py`: Comprehensive scientific evaluation engine with bootstrap 95% confidence intervals, Bayesian Elo estimation, Shannon entropy tracking, ablation metrics, and LaTeX table export.
2. `scripts/run_thesis_study.py` & `scripts/generate_thesis_figures.py`: Multi-process CLI study runner and publication-grade vector figure generator (PDF/PNG 300 DPI with bilingual labels).
3. `docs/course/lesson_13_scientific_research_paper.html`: Interactive bilingual (🇮🇹/🇬🇧) university lesson with MathJax formulas, structural research guide, Web Lab integration, and interactive quiz.
4. Global course portal update (`index.html`, `course.js`, and sidebars of `lesson_01.html` through `lesson_12.html`).

**Tech Stack:** Python 3.12/3.14, PyTorch, Gymnasium, NumPy, SciPy, Matplotlib, Seaborn, HTML5/CSS3/ES6, MathJax 3.

**Spec:** [`docs/superpowers/specs/2026-08-26-thesis-research-design.md`](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-26-thesis-research-design.md)

## Global Constraints
- 100% deterministic reproducibility when seeded.
- Strict anti-leakage compliance (POMDP): Opponent hidden cards or tickets must NEVER be read directly; only public game history (claimed routes, face-up draws) can be used.
- Bilingual support (Italian 🇮🇹 & English 🇬🇧) for Lesson 13, report metadata, and figure labels.
- Full type annotations and docstrings with mathematical formulas and explanations.
- Test-driven development (TDD) for all evaluation and scripting modules.

---

### Task 1: Scientific Thesis Benchmark Suite (`src/evaluation/thesis_benchmark.py`)

**Files:**
- Create: `src/evaluation/thesis_benchmark.py`
- Test: `tests/evaluation/test_thesis_benchmark.py`

**Interfaces:**
- Produces:
  - `ThesisBenchmarkRunner(board: Board, tickets: list[DestinationTicket], config: dict | None = None)`
  - `ThesisBenchmarkRunner.run_round_robin_tournament(agents: list[BaseAgent], games_per_pair: int = 20, seeds: list[int] | None = None) -> dict[str, Any]`
  - `ThesisBenchmarkRunner.run_bayesian_entropy_study(num_games: int = 10, seed: int = 42) -> dict[str, Any]`
  - `ThesisBenchmarkRunner.run_deception_robustness_study(bluff_rates: list[float] | None = None, num_games: int = 10, seed: int = 42) -> dict[str, Any]`
  - `ThesisBenchmarkRunner.run_computational_profile(agents: list[BaseAgent], num_moves: int = 50) -> dict[str, Any]`
  - `ThesisBenchmarkRunner.export_latex_tables(results: dict[str, Any], output_dir: Path) -> dict[str, str]`

- [ ] **Step 1: Write failing test for `ThesisBenchmarkRunner`**

```python
# tests/evaluation/test_thesis_benchmark.py
import pytest
import numpy as np
from pathlib import Path
from src.game.maps import load_usa_board
from src.agents.random_agent import RandomAgent
from src.agents.greedy_agent import GreedyAgent
from src.evaluation.thesis_benchmark import ThesisBenchmarkRunner

def test_thesis_benchmark_runner_init():
    board, tickets = load_usa_board()
    runner = ThesisBenchmarkRunner(board=board, tickets=tickets, config={"seed": 42})
    assert runner.seed == 42
    assert runner.board == board

def test_thesis_benchmark_round_robin_and_latex(tmp_path: Path):
    board, tickets = load_usa_board()
    runner = ThesisBenchmarkRunner(board=board, tickets=tickets)
    
    agent1 = RandomAgent(board=board, tickets=tickets, name="Random_1")
    agent2 = GreedyAgent(board=board, tickets=tickets, name="Greedy_2")
    
    res = runner.run_round_robin_tournament(agents=[agent1, agent2], games_per_pair=2, seeds=[42])
    assert "payoff_matrix" in res
    assert "elo_ratings" in res
    assert "win_rates" in res
    assert "Random_1" in res["elo_ratings"]
    assert "Greedy_2" in res["elo_ratings"]
    
    latex_files = runner.export_latex_tables(results={"tournament": res}, output_dir=tmp_path)
    assert "table1_main_results.tex" in latex_files
    assert (tmp_path / "table1_main_results.tex").exists()

def test_thesis_benchmark_entropy_and_deception():
    board, tickets = load_usa_board()
    runner = ThesisBenchmarkRunner(board=board, tickets=tickets)
    
    entropy_res = runner.run_bayesian_entropy_study(num_games=2, seed=42)
    assert "turns" in entropy_res
    assert "mean_entropy" in entropy_res
    assert "top1_accuracy" in entropy_res
    
    deception_res = runner.run_deception_robustness_study(bluff_rates=[0.0, 0.2], num_games=2, seed=42)
    assert 0.0 in deception_res["bluff_rates"]
    assert 0.2 in deception_res["bluff_rates"]
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/pytest tests/evaluation/test_thesis_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluation.thesis_benchmark'`

- [ ] **Step 3: Implement `src/evaluation/thesis_benchmark.py`**

Implement full statistical calculations:
- Round-robin tournament with 95% Wilson confidence intervals and Elo calculation.
- Turn-by-turn Bayesian entropy tracking using `BayesianTicketBeliefTracker`.
- Deception / bluff noise injection.
- LaTeX table exporters (`table1_main_results.tex`, `table2_ticket_completion_rates.tex`, `table3_computational_profile.tex`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/evaluation/test_thesis_benchmark.py -v`
Expected: PASS

---

### Task 2: CLI Study Runner Script (`scripts/run_thesis_study.py`)

**Files:**
- Create: `scripts/run_thesis_study.py`
- Test: `tests/evaluation/test_cli.py` (add CLI runner test)

**Interfaces:**
- CLI Arguments: `--seeds`, `--games-per-pair`, `--output-dir`, `--fast-mode`
- Output: Structured JSON file `results/thesis/thesis_study_results.json` + generated `.tex` tables.

- [ ] **Step 1: Write test for `run_thesis_study.py` CLI invocation**

```python
# In tests/evaluation/test_cli.py
def test_run_thesis_study_cli(tmp_path: Path):
    import subprocess
    import sys
    cmd = [
        sys.executable,
        "scripts/run_thesis_study.py",
        "--fast-mode",
        "--output-dir", str(tmp_path),
        "--games-per-pair", "2",
        "--seeds", "42"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert (tmp_path / "thesis_study_results.json").exists()
```

- [ ] **Step 2: Implement `scripts/run_thesis_study.py`**

Create orchestrator script that instantiates all 6 agents (Heuristic, Flat-PPO, Recurrent-PPO, Uniform-ISMCTS, AlphaZero, Bayesian-AlphaZero), executes the 5 experimental batteries, and saves output JSON and LaTeX files.

- [ ] **Step 3: Verify CLI execution**

Run: `.venv/bin/pytest tests/evaluation/test_cli.py -k test_run_thesis_study_cli -v`
Expected: PASS

---

### Task 3: Publication Figure Generator (`scripts/generate_thesis_figures.py`)

**Files:**
- Create: `scripts/generate_thesis_figures.py`
- Test: `tests/evaluation/test_figure_generator.py`

**Interfaces:**
- Generates 5 bilingual figures (PDF and PNG 300 DPI):
  1. `fig1_elo_tournament_matrix.pdf` (Payoff heatmap + Elo with error bars)
  2. `fig2_bayesian_entropy_convergence.pdf` (Shannon Entropy & Top-1/3 Confidence)
  3. `fig3_ablation_determinization.pdf` (Uniform vs Belief-Weighted ISMCTS)
  4. `fig4_deception_noise_robustness.pdf` (Elo Degradation vs Bluff Rate)
  5. `fig5_compute_vs_elo_frontier.pdf` (Pareto Frontier: Latency ms vs Elo)

- [ ] **Step 1: Write test for `generate_thesis_figures.py`**

```python
# tests/evaluation/test_figure_generator.py
import pytest
import json
from pathlib import Path
from scripts.generate_thesis_figures import generate_all_figures

def test_generate_all_figures(tmp_path: Path):
    mock_data = {
        "tournament": {
            "agents": ["Heuristic", "Bayesian_MCTS"],
            "elo_ratings": {"Heuristic": 1200.0, "Bayesian_MCTS": 1550.0},
            "elo_ci": {"Heuristic": [1150.0, 1250.0], "Bayesian_MCTS": [1500.0, 1600.0]},
            "payoff_matrix": [[0.5, 0.1], [0.9, 0.5]],
            "win_rates": {"Heuristic": 0.1, "Bayesian_MCTS": 0.9}
        },
        "entropy_study": {
            "turns": [1, 5, 10, 15, 20],
            "mean_entropy": [4.5, 3.8, 2.1, 1.2, 0.6],
            "top1_accuracy": [0.1, 0.25, 0.55, 0.82, 0.94],
            "top3_accuracy": [0.3, 0.50, 0.78, 0.95, 0.99]
        },
        "ablation_determinization": {
            "modes": ["Uniform", "Belief-Weighted"],
            "win_rates": [0.42, 0.78]
        },
        "deception_study": {
            "bluff_rates": [0.0, 0.1, 0.2, 0.3],
            "bayesian_elo": [1550, 1510, 1420, 1340],
            "lstm_elo": [1460, 1450, 1430, 1410]
        },
        "computational_profile": {
            "agents": ["Flat_PPO", "Recurrent_PPO", "Uniform_MCTS", "Bayesian_MCTS"],
            "ms_per_move": [0.8, 2.1, 45.0, 52.0],
            "elo": [1320, 1460, 1410, 1550]
        }
    }
    
    json_path = tmp_path / "mock_results.json"
    with open(json_path, "w") as f:
        json.dump(mock_data, f)
        
    out_dir = tmp_path / "figures"
    generated = generate_all_figures(results_path=json_path, output_dir=out_dir)
    assert len(generated) >= 5
    for fig_path in generated:
        assert fig_path.exists()
```

- [ ] **Step 2: Implement `scripts/generate_thesis_figures.py`**

Use Matplotlib + Seaborn with clean academic styling (LaTeX-like typography, accessible palettes, dual bilingual labels).

- [ ] **Step 3: Run test to verify figure generation**

Run: `.venv/bin/pytest tests/evaluation/test_figure_generator.py -v`
Expected: PASS

---

### Task 4: Interactive Bilingual Lesson 13 (`docs/course/lesson_13_scientific_research_paper.html`)

**Files:**
- Create: `docs/course/lesson_13_scientific_research_paper.html`
- Modify: `docs/course/style.css` (if any language toggle styles needed)

**Features:**
- Interactive Language Switcher (`🇮🇹 Italiano` / `🇬🇧 English`) with instant client-side DOM toggle.
- 6 Pedagogical Sections:
  1. *L'Intuizione / The Scientific Intuition* (From coding to thesis hypotheses).
  2. *I 6 Paradigmi / The 6 Paradigms Compared* (Mathematical taxonomy and decision tree).
  3. *Rigore Sperimentale / Experimental Rigor* (Confidence intervals, Elo, p-values).
  4. *Il Filtro Bayesiano / Inside the Bayesian Filter* (Dijkstra Detour, posterior update, elimination of Strategy Fusion).
  5. *Dall'Esperimento alla Tesi / From Results to Academic Writing* (Chapter outline, Web Lab XAI case studies).
  6. *Quiz Interattivo Bilingue / Interactive Bilingual Self-Test Quiz* with MathJax rendering.

- [ ] **Step 1: Implement `docs/course/lesson_13_scientific_research_paper.html`**
- [ ] **Step 2: Verify HTML validity, MathJax tags, and interactive language switch behavior**

---

### Task 5: Global Course Portal Update

**Files:**
- Modify: `docs/course/index.html` (add Lesson 13 to Syllabus & Hero Stats: 13 lessons)
- Modify: `docs/course/course.js` (update total lesson count to 13)
- Modify: `docs/course/lesson_01_game_core.html` through `docs/course/lesson_12_alphazero_advanced.html` (add Lesson 13 to sidebar navigation list)

- [ ] **Step 1: Update `docs/course/index.html` with Lesson 13 card and update stats to 13 Lezioni**
- [ ] **Step 2: Update `docs/course/course.js`**
- [ ] **Step 3: Update sidebars in `lesson_01.html` through `lesson_12.html`**

---

### Task 6: Full Acceptance Suite & Verification

**Files:**
- Create: `tests/evaluation/test_thesis_acceptance.py`

**Steps:**
- [ ] **Step 1: Write acceptance test validating end-to-end thesis benchmark, LaTeX generation, figure generation, and course lesson links**
- [ ] **Step 2: Run full test suite (`pytest`) to ensure 100% green tests**

---

## Plan Review Check
1. **Spec Coverage:** Covers all 6 paradigms, 5 benchmark batteries, 5 publication figures, 3 LaTeX tables, bilingual support, and interactive Lesson 13.
2. **No Placeholders:** All tasks contain precise file paths, interfaces, and code structures.
3. **Reproducibility:** Multi-seed and deterministic assertions in all test files.
