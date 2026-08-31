import unittest
from pathlib import Path
from src.api.report_service import ReportService

class TestReportServicePaper(unittest.TestCase):
    def test_report_service_indexes_paper_and_tournament(self):
        service = ReportService(results_dir="results")
        reports = service.list_reports()
        filenames = [r.filename for r in reports]
        
        # Should include tournament and latex papers
        self.assertTrue(any("tournament" in f.lower() for f in filenames), f"Tournament not in {filenames}")
        self.assertTrue(any("main_paper_en.tex" in f or "capitolo_tesi_ita.tex" in f for f in filenames), f"LaTeX paper not in {filenames}")
        
        # Test report detail fetch
        detail = service.get_report("main_paper_en.tex")
        self.assertIsNotNone(detail, "main_paper_en.tex detail should not be None")
        self.assertIn("\\documentclass", detail.raw_content)
        self.assertEqual(detail.file_type, "latex")
        self.assertIn("English", detail.name)

        # Test Italian thesis detail fetch
        detail_ita = service.get_report("capitolo_tesi_ita.tex")
        self.assertIsNotNone(detail_ita, "capitolo_tesi_ita.tex detail should not be None")
        self.assertIn("Italian", detail_ita.name)
        self.assertEqual(detail_ita.file_type, "latex")

if __name__ == "__main__":
    unittest.main()
