# Academic LaTeX Research Paper, Lesson 14 & RL-Lab Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author a university-grade academic research paper in LaTeX (in both English and Italian) with 100% verified authentic BibTeX citations, build an interactive bilingual Lesson 14 in the course suite, and integrate the paper and 5,130-game tournament meta-analysis into the RL-Lab Web Studio.

**Architecture:** 
- A standalone LaTeX package in `paper/` containing `main_paper_en.tex`, `capitolo_tesi_ita.tex`, verified `references.bib`, modular tables (`paper/tables/`), and visual figures (`paper/figures/`).
- An interactive bilingual web module in `docs/course/lesson_14_tournament_paper.html` featuring instant ITA-ENG toggling, MathJax mathematical rendering, interactive tournament data tables, and an academic self-assessment quiz.
- Backend indexing in `src/api/report_service.py` to automatically register the scientific paper and LaTeX dispatches in the RL-Lab Reports view.

**Tech Stack:** LaTeX (pdflatex/bibtex compatible), HTML5/CSS3/JavaScript (MathJax 3), Python 3.11+, Pytest, FastAPI.

**Spec:** `docs/superpowers/specs/2026-08-31-academic-paper-and-lesson14-design.md`

## Global Constraints
- Ground all empirical claims on the exact 5,130-game tournament data (`results/tournament_results_10x_5130games.json`) and 1M step training summaries (`results/training_summary_1m_steps.json`).
- Ensure all BibTeX bibliography entries are 100% authentic, published scientific papers with valid DOIs/ArXiv identifiers (no hallucinated citations).
- Provide complete bilingual parity (Italian and English) for Lesson 14 and web reader components.
- Maintain full backward compatibility across all existing backend APIs and frontend components.

---

### Task 1: Standalone Verified Bibliography, Modular LaTeX Tables & Visual Assets

**Files:**
- Create: `paper/references.bib`
- Create: `paper/tables/table1_tournament_elo.tex`
- Create: `paper/tables/table2_training_hyperparameters.tex`
- Create: `paper/tables/table3_computational_profile.tex`
- Copy: `results/thesis/figures/*` -> `paper/figures/`
- Create: `tests/test_paper_assets.py`

**Interfaces:**
- Produces: Verified BibTeX entries (`schulman2017proximal`, `silver2018general`, `cowling2012information`, `hausknecht2015deep`, `mnih2015human`, `vanhasselt2016deep`, `dijkstra1959note`, `shannon1948mathematical`, `sutton2018reinforcement`, `elo1978rating`) and compilation-ready LaTeX tables.

- [ ] **Step 1: Write the failing test for paper assets and BibTeX verification**

```python
import os
import re
from pathlib import Path

def test_paper_structure_and_bib():
    paper_dir = Path("paper")
    assert paper_dir.exists()
    assert (paper_dir / "references.bib").exists()
    assert (paper_dir / "tables" / "table1_tournament_elo.tex").exists()
    assert (paper_dir / "tables" / "table2_training_hyperparameters.tex").exists()
    assert (paper_dir / "tables" / "table3_computational_profile.tex").exists()
    assert (paper_dir / "figures" / "fig1_elo_tournament_matrix.png").exists()

    # Verify BibTeX keys
    bib_content = (paper_dir / "references.bib").read_text(encoding="utf-8")
    required_keys = [
        "schulman2017proximal",
        "silver2018general",
        "cowling2012information",
        "hausknecht2015deep",
        "mnih2015human",
        "dijkstra1959note",
        "shannon1948mathematical",
    ]
    for k in required_keys:
        assert f"@{{" in bib_content or f"@article{{{k}" in bib_content or f"@inproceedings{{{k}" in bib_content or f"@book{{{k}" in bib_content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paper_assets.py -v`  
Expected: FAIL with missing `paper/` directory or files.

- [ ] **Step 3: Create directory structure, write `references.bib`, tables, and copy figures**

Write `paper/references.bib` with verified references.  
Write `paper/tables/table1_tournament_elo.tex`, `paper/tables/table2_training_hyperparameters.tex`, `paper/tables/table3_computational_profile.tex`.  
Copy figures from `results/thesis/figures/` to `paper/figures/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_paper_assets.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/ tests/test_paper_assets.py
git commit -m "feat(paper): add verified bibtex, modular latex tables, and figures"
```

---

### Task 2: Academic LaTeX Research Paper in English (`paper/main_paper_en.tex`)

**Files:**
- Create: `paper/main_paper_en.tex`
- Modify: `tests/test_paper_assets.py`

**Interfaces:**
- Consumes: `paper/references.bib`, `paper/tables/*.tex`, `paper/figures/*.png`.
- Produces: Complete academic research paper covering Abstract, POMDP Formulation, Related Work, Agent Formulations (Random, Greedy, Dijkstra, DQN, PPO, LSTM-PPO, AlphaZero, Bayesian MCTS), 5,130-Game Tournament Benchmark, Ablation Studies, and Conclusions.

- [ ] **Step 1: Write test validating `main_paper_en.tex` structure, sections, citations, and table inputs**

Add to `tests/test_paper_assets.py`:
```python
def test_main_paper_en_content():
    paper_file = Path("paper/main_paper_en.tex")
    assert paper_file.exists()
    content = paper_file.read_text(encoding="utf-8")
    assert "\\documentclass" in content
    assert "\\begin{document}" in content
    assert "\\section{Introduction" in content or "\\section{Introduzione" in content
    assert "Strategic Dijkstra" in content
    assert "1329.6" in content  # Elo of Strategic Dijkstra
    assert "5,130" in content or "5130" in content
    assert "\\cite{schulman2017proximal}" in content
    assert "\\cite{silver2018general}" in content
    assert "\\cite{dijkstra1959note}" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paper_assets.py::test_main_paper_en_content -v`  
Expected: FAIL with file not found.

- [ ] **Step 3: Write comprehensive `paper/main_paper_en.tex`**

Include:
- Formal POMDP formulation: $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$, Action Masking, Observation Vector $\mathbb{R}^{356}$.
- Comprehensive descriptions of all baseline, model-free, recurrent, and tree-search agents (Uniform Random, Greedy Score, Strategic Dijkstra, Double-DQN, CleanRL PPO, Recurrent LSTM PPO, Prioritized Fictitious Self-Play, IS-MCTS, Bayesian Detour MCTS, AlphaZero).
- Complete 5,130-game tournament analysis and Elo rankings with 95% confidence intervals.
- Ablations: Shannon entropy decay curves $H(t)$, Strategy Fusion in IS-MCTS, inference latency vs Elo Pareto frontier.
- References to `paper/references.bib`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_paper_assets.py::test_main_paper_en_content -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/main_paper_en.tex tests/test_paper_assets.py
git commit -m "feat(paper): author full academic research paper in english (latex)"
```

---

### Task 3: Master's Thesis Research Chapter in Italian (`paper/capitolo_tesi_ita.tex`)

**Files:**
- Create: `paper/capitolo_tesi_ita.tex`
- Modify: `tests/test_paper_assets.py`

**Interfaces:**
- Consumes: `paper/references.bib`, `paper/tables/*.tex`, `paper/figures/*.png`.
- Produces: Complete university thesis monograph research chapter in Italian.

- [ ] **Step 1: Write test validating `capitolo_tesi_ita.tex` structure, Italian academic prose, and citations**

Add to `tests/test_paper_assets.py`:
```python
def test_capitolo_tesi_ita_content():
    thesis_file = Path("paper/capitolo_tesi_ita.tex")
    assert thesis_file.exists()
    content = thesis_file.read_text(encoding="utf-8")
    assert "\\section{Introduzione" in content
    assert "\\section{Stato dell'Arte" in content or "\\section{Letteratura" in content
    assert "Dijkstra" in content
    assert "AlphaZero" in content
    assert "1329.6" in content
    assert "\\cite{cowling2012information}" in content
    assert "\\cite{hausknecht2015deep}" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paper_assets.py::test_capitolo_tesi_ita_content -v`  
Expected: FAIL with file not found.

- [ ] **Step 3: Write comprehensive `paper/capitolo_tesi_ita.tex`**

Include:
- Rigorous Italian academic prose for university thesis submission.
- Full mathematical descriptions of Graph POMDP, PPO clipping, Shannon entropy, Action Masking, and belief tracking.
- Analysis of all agents and the 5,130-game tournament results.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_paper_assets.py::test_capitolo_tesi_ita_content -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/capitolo_tesi_ita.tex tests/test_paper_assets.py
git commit -m "feat(paper): author complete master's thesis research chapter in italian (latex)"
```

---

### Task 4: Interactive Bilingual Lesson 14 (`docs/course/lesson_14_tournament_paper.html`) & Course Hub Update

**Files:**
- Create: `docs/course/lesson_14_tournament_paper.html`
- Modify: `docs/course/index.html`
- Modify: `docs/course/course.js`
- Create: `tests/test_lesson14.py`

**Interfaces:**
- Consumes: Tournament data, LaTeX paper extracts, bilingual toggles.
- Produces: Capstone course module with full MathJax rendering, interactive tournament tables, paper viewer, quiz widget, and updated syllabus.

- [ ] **Step 1: Write test validating Lesson 14 markup, bilingual spans, quiz interactivity, and syllabus links**

```python
from pathlib import Path
from bs4 import BeautifulSoup

def test_lesson14_integrity():
    l14_path = Path("docs/course/lesson_14_tournament_paper.html")
    assert l14_path.exists()
    content = l14_path.read_text(encoding="utf-8")
    
    # Check bilingual structure
    assert 'class="lang-it"' in content
    assert 'class="lang-en"' in content
    assert 'id="langToggleBtn"' in content
    assert 'MathJax' in content
    
    # Check quiz presence
    assert 'l14_q1' in content
    assert 'l14_q2' in content
    assert 'l14_q3' in content
    
    # Check syllabus integration in index.html
    index_content = Path("docs/course/index.html").read_text(encoding="utf-8")
    assert "lesson_14_tournament_paper.html" in index_content
    assert "Lezione 14" in index_content or "Lesson 14" in index_content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lesson14.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement `lesson_14_tournament_paper.html`, update `index.html` and `course.js`**

- Create `docs/course/lesson_14_tournament_paper.html` with:
  - Header with `[🇬🇧 Switch to English / 🇮🇹 Passa a Italiano]` toggle button.
  - Complete bilingual text for all sections (Abstract, Formulation, All Agent Paradigms, Tournament 5,130 Analysis, Elo Table, Ablations, Conclusions).
  - MathJax rendered LaTeX equations.
  - Interactive Paper Source Tabs (Overview, English LaTeX, Italian LaTeX, BibTeX).
  - 3-Question Self-Assessment Research Quiz with bilingual explanations.
- Update `docs/course/index.html` with Lesson 14 card and stats.
- Update `docs/course/course.js` to support Lesson 14 active link highlighting and quiz solution buttons.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lesson14.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/course/ tests/test_lesson14.py
git commit -m "feat(course): add interactive bilingual lesson 14 and update course hub"
```

---

### Task 5: RL-Lab Dispatch & Report Service Integration

**Files:**
- Modify: `src/api/report_service.py`
- Modify: `tests/test_report_service.py` (or create if not present)

**Interfaces:**
- Consumes: `paper/*.tex`, `results/*.md`, `results/*.json`.
- Produces: API endpoints `/api/reports` and `/api/reports/{filename}` serving LaTeX papers, thesis chapters, and tournament reports with proper metadata and friendly titles.

- [ ] **Step 1: Write test for ReportService indexing LaTeX papers and tournament reports**

```python
from pathlib import Path
from src.api.report_service import ReportService

def test_report_service_indexes_paper_and_tournament():
    service = ReportService(results_dir="results")
    reports = service.list_reports()
    filenames = [r.filename for r in reports]
    
    # Should include tournament and latex papers
    assert any("tournament" in f.lower() for f in filenames)
    assert any("main_paper_en.tex" in f or "paper" in f.lower() for f in filenames)
    
    # Test report detail fetch
    detail = service.get_report("main_paper_en.tex")
    assert detail is not None
    assert "\\documentclass" in detail.raw_content
    assert detail.file_type == "text" or detail.file_type == "latex"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report_service_paper.py -v`  
Expected: FAIL if `ReportService` does not search `paper/` directory or `.tex` extension.

- [ ] **Step 3: Update `src/api/report_service.py`**

- Add `Path("paper")` and `Path("results/thesis")` to search paths.
- Allow `.tex` and `.bib` extensions alongside `.md`, `.json`, `.txt`.
- Add friendly names and phase resolution for LaTeX papers (e.g. `🔬 Scientific Paper (English LaTeX)`, `🎓 Master's Thesis Chapter (Italian LaTeX)`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report_service_paper.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/report_service.py tests/test_report_service_paper.py
git commit -m "feat(api): index latex papers and tournament reports in report service"
```

---

### Task 6: Full System Integration & Regression Testing

**Files:**
- Test all components end-to-end.

- [ ] **Step 1: Run complete test suite**

Run: `pytest -v`  
Expected: All tests PASS.

- [ ] **Step 2: Verify git status and clean working tree**

Run: `git status`  
Expected: Clean working tree.

- [ ] **Step 3: Final Commit & Tagging**

```bash
git commit --allow-empty -m "chore: complete academic paper, lesson 14, and RL-lab integration"
```
