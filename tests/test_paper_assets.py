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

    def test_main_paper_en_content(self):
        paper_file = Path("paper/main_paper_en.tex")
        self.assertTrue(paper_file.exists(), "paper/main_paper_en.tex should exist")
        content = paper_file.read_text(encoding="utf-8")
        self.assertIn(r"\documentclass", content)
        self.assertIn(r"\begin{document}", content)
        self.assertIn(r"\section{Introduction}", content)
        self.assertIn("Strategic Dijkstra", content)
        self.assertIn("1329.6", content)
        self.assertTrue("5,130" in content or "5130" in content)
        self.assertIn("schulman2017proximal", content)
        self.assertIn("silver2018general", content)
        self.assertIn("dijkstra1959note", content)
        self.assertIn(r"\input{tables/table1_tournament_elo.tex}", content)

    def test_capitolo_tesi_ita_content(self):
        thesis_file = Path("paper/capitolo_tesi_ita.tex")
        self.assertTrue(thesis_file.exists(), "paper/capitolo_tesi_ita.tex should exist")
        content = thesis_file.read_text(encoding="utf-8")
        self.assertIn(r"\documentclass", content)
        self.assertIn(r"\begin{document}", content)
        self.assertIn(r"\section{Introduzione", content)
        self.assertTrue(r"\section{Stato dell'Arte" in content or r"\section{Letteratura" in content or r"\section{Fondamenti" in content)
        self.assertIn("Dijkstra", content)
        self.assertIn("AlphaZero", content)
        self.assertIn("1329.6", content)
        self.assertTrue("5.130" in content or "5130" in content)
        self.assertIn("cowling2012information", content)
        self.assertIn("hausknecht2015deep", content)
        self.assertIn(r"\input{tables/table1_tournament_elo.tex}", content)

if __name__ == "__main__":
    unittest.main()
