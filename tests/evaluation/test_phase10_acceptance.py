"""Acceptance Test Suite for Phase 10: Generalization, Europe Board, and Procedural Maps."""

from pathlib import Path
import pytest
from src.evaluation.generalization import GeneralizationBenchmarkRunner
from src.game.maps import load_europe_board, load_usa_board
from src.game.procedural import ProceduralMapGenerator


def test_phase10_acceptance_full_pipeline(tmp_path):
    json_path = str(tmp_path / "phase10_test.json")
    md_path = str(tmp_path / "phase10_test.md")

    # 1. Verify procedural map generator determinism and connectivity
    gen = ProceduralMapGenerator()
    b1, t1 = gen.generate(seed=42)
    b2, t2 = gen.generate(seed=42)
    assert len(b1.cities) == 8
    assert [c.name for c in b1.cities.values()] == [c.name for c in b2.cities.values()]
    assert len(b1.routes) == 14
    assert len(t1) == 10

    # 2. Verify Europe board
    eur_board, eur_tickets = load_europe_board()
    assert len(eur_board.cities) >= 40
    assert len(eur_board.routes) >= 90
    assert len(eur_tickets) >= 40

    # 3. Verify USA board
    usa_board, usa_tickets = load_usa_board()
    assert len(usa_board.cities) == 36
    assert len(usa_board.routes) == 100
    assert len(usa_tickets) == 30

    # 4. Verify benchmark runner and report generation
    runner = GeneralizationBenchmarkRunner(
        config={
            "train_seeds": [1, 2],
            "test_seeds": [101, 102],
            "training_steps": 256,
            "games_per_map": 2,
            "seed": 42,
        }
    )
    results = runner.run_study()
    runner.generate_report(results, output_md_path=md_path, output_json_path=json_path)

    assert Path(json_path).exists()
    assert Path(md_path).exists()
    assert results["heuristic_generalization"]["strategic"]["unseen_win_rate"] >= 0.50
    assert "retention_rate" in results["heuristic_generalization"]["strategic"]
    assert "official_cross_map" in results
