import unittest
from pathlib import Path

class TestLesson14(unittest.TestCase):
    def test_lesson14_integrity(self):
        l14_path = Path("docs/course/lesson_14_tournament_paper.html")
        self.assertTrue(l14_path.exists(), "docs/course/lesson_14_tournament_paper.html must exist")
        content = l14_path.read_text(encoding="utf-8")
        
        # Check bilingual structure
        self.assertIn('class="lang-it"', content)
        self.assertIn('class="lang-en"', content)
        self.assertIn('id="langToggleBtn"', content)
        self.assertIn("MathJax", content)
        
        # Check tournament metrics & references in paper
        self.assertIn("1329.6", content)
        self.assertTrue("5.130" in content or "5,130" in content)
        self.assertIn("AlphaZero", content)
        self.assertIn("Dijkstra", content)
        
        # Check quiz presence
        self.assertIn("l14_q1", content)
        self.assertIn("l14_q2", content)
        self.assertIn("l14_q3", content)
        
        # Check syllabus integration in index.html
        index_content = Path("docs/course/index.html").read_text(encoding="utf-8")
        self.assertIn("lesson_14_tournament_paper.html", index_content)
        self.assertTrue("lezione 14" in index_content.lower() or "lesson 14" in index_content.lower())

if __name__ == "__main__":
    unittest.main()
