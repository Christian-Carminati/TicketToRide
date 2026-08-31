import os
import unittest
from pathlib import Path

class TestPaperAssets(unittest.TestCase):
    def test_paper_structure_and_bib(self):
        paper_dir = Path("paper")
        self.assertTrue(paper_dir.exists(), "paper/ directory should exist")
        self.assertTrue((paper_dir / "references.bib").exists(), "paper/references.bib should exist")
        self.assertTrue((paper_dir / "tables" / "table1_tournament_elo.tex").exists(), "table1_tournament_elo.tex should exist")
        self.assertTrue((paper_dir / "tables" / "table2_training_hyperparameters.tex").exists(), "table2_training_hyperparameters.tex should exist")
        self.assertTrue((paper_dir / "tables" / "table3_computational_profile.tex").exists(), "table3_computational_profile.tex should exist")
        self.assertTrue((paper_dir / "figures" / "fig1_elo_tournament_matrix.png").exists(), "fig1_elo_tournament_matrix.png should exist")

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
            self.assertTrue(
                f"@{{" in bib_content or f"@article{{{k}" in bib_content or f"@inproceedings{{{k}" in bib_content or f"@book{{{k}" in bib_content,
                f"Missing BibTeX key: {k}"
            )

if __name__ == "__main__":
    unittest.main()
