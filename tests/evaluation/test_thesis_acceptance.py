"""
Comprehensive End-to-End Acceptance Test for Thesis Research Benchmark & Lesson 13.
"""

import json
from pathlib import Path
import pytest

from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.thesis_benchmark import ThesisBenchmarkRunner
from src.game.maps import load_usa_board
from scripts.generate_thesis_figures import generate_all_figures


def test_thesis_end_to_end_acceptance(tmp_path: Path):
    # 1. Test Benchmark Runner
    board, tickets = load_usa_board()
    runner = ThesisBenchmarkRunner(board=board, tickets=tickets, config={"seed": 42})
    
    agent1 = RandomAgent(name="Acceptance_Random")
    agent2 = GreedyAgent(name="Acceptance_Greedy")
    
    tournament = runner.run_round_robin_tournament(agents=[agent1, agent2], games_per_pair=2, seeds=[42])
    entropy_study = runner.run_bayesian_entropy_study(num_games=2, seed=42)
    deception_study = runner.run_deception_robustness_study(bluff_rates=[0.0, 0.1], num_games=2, seed=42)
    comp_profile = runner.run_computational_profile(agents=[agent1, agent2], num_moves=10)
    
    results = {
        "tournament": tournament,
        "entropy_study": entropy_study,
        "deception_study": deception_study,
        "computational_profile": comp_profile,
        "ablation_determinization": {
            "modes": ["Uniform", "Belief-Weighted"],
            "win_rates": [0.42, 0.78],
        },
    }
    
    json_path = tmp_path / "thesis_study_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    latex_files = runner.export_latex_tables(results=results, output_dir=tmp_path)
    assert "table1_main_results.tex" in latex_files
    assert "table3_computational_profile.tex" in latex_files
    
    # 2. Test Figure Generation
    fig_dir = tmp_path / "figures"
    generated_figs = generate_all_figures(results_path=json_path, output_dir=fig_dir)
    assert len(generated_figs) >= 5
    for f in generated_figs:
        assert f.exists()
        assert f.stat().st_size > 0
        
    # 3. Test Course Portal & Lesson 13 Integration
    course_dir = Path("docs/course")
    lesson13 = course_dir / "lesson_13_scientific_research_paper.html"
    assert lesson13.exists()
    
    l13_content = lesson13.read_text(encoding="utf-8")
    assert "MathJax" in l13_content
    assert "lang-it" in l13_content
    assert "lang-en" in l13_content
    assert "langToggleBtn" in l13_content
    
    # Check index.html references lesson 13
    index_content = (course_dir / "index.html").read_text(encoding="utf-8")
    assert "lesson_13_scientific_research_paper.html" in index_content
    assert "LEZIONE 13" in index_content
