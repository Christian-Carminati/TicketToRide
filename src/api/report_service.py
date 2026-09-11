"""Service for discovering, reading, and managing scientific research and benchmark reports."""

import json
import os
import time
from pathlib import Path

from src.api.schemas import ReportDetailDTO, ReportItemDTO


class ReportService:
    """Discovers and parses Markdown, JSON, and text benchmark reports."""

    def __init__(self, results_dir: str = "experiments/results") -> None:
        self.results_dir = results_dir

    def _resolve_phase(self, filename: str) -> str | None:
        fn_lower = filename.lower()
        if "paper" in fn_lower or "tesi" in fn_lower or fn_lower.endswith(".tex") or fn_lower.endswith(".bib"):
            return "Fase 14 — Paper & Ricerca LaTeX"
        if "phase6" in fn_lower or "phase_6" in fn_lower or "ppo" in fn_lower:
            return "Fase 6 — PPO & CleanRL"
        if "reward" in fn_lower or "phase7" in fn_lower or "phase_7" in fn_lower:
            return "Fase 7 — Reward Research"
        if "tourn" in fn_lower:
            return "Tournament Arena"
        return "Benchmark Generale"

    def _resolve_friendly_name(self, filename: str) -> str:
        fn_lower = filename.lower()
        if "main_paper_en.tex" in fn_lower:
            return "🔬 Scientific Research Paper (English LaTeX)"
        if "capitolo_tesi_ita.tex" in fn_lower:
            return "📄 Monografia di Ricerca Indipendente (Italian LaTeX)"
        if "references.bib" in fn_lower:
            return "📚 Bibliografia Scientifica Verificata (BibTeX)"
        if "tournament_report" in fn_lower:
            return "🏆 Report Ufficiale Torneo 5.130 Partite (10x)"
        if "tournament_results" in fn_lower:
            return "📦 Dati Matrice Torneo 5.130 Partite (JSON)"
        if "extended_telemetry" in fn_lower:
            return "🛰️ Telemetria Strategica Avanzata & Colli di Bottiglia (JSON)"
        if "training_summary" in fn_lower:
            return "📈 Riepilogo Training 1M Step (JSON)"
        if "reward_research.md" in fn_lower:
            return "📊 Studio Comparativo Reward (Fase 7)"
        if "reward_research.json" in fn_lower:
            return "📦 Dati Strutturati Reward (Fase 7 JSON)"
        if "benchmark_phase6.md" in fn_lower:
            return "📈 Benchmark PPO & Ablation CleanRL (Fase 6)"
        if "benchmark_phase6.json" in fn_lower:
            return "📦 Dati Benchmark PPO (Fase 6 JSON)"
        if "comparison_report.txt" in fn_lower:
            return "📑 Report Comparativo Ottimizzazioni"

        base = os.path.splitext(filename)[0]
        return base.replace("_", " ").title()

    def list_reports(self) -> list[ReportItemDTO]:
        """List all available reports in paper/, results/, experiments/results and project root."""
        search_paths = [
            Path("paper"),
            Path(self.results_dir),
            Path("results"),
            Path("results/thesis"),
            Path("experiments/results"),
            Path("."),
        ]
        reports: list[ReportItemDTO] = []
        seen_filenames: set[str] = set()

        for search_dir in search_paths:
            if not search_dir.exists():
                continue

            for f in sorted(
                search_dir.iterdir(),
                key=lambda x: x.stat().st_mtime if x.is_file() else 0,
                reverse=True,
            ):
                if not f.is_file():
                    continue

                fname = f.name
                ext = f.suffix.lower()

                # Only include report-like files
                if ext not in [".md", ".json", ".txt", ".tex", ".bib"]:
                    continue

                if fname in seen_filenames or fname in [
                    "package.json",
                    "pyproject.toml",
                    "tsconfig.json",
                    "uv.lock",
                ]:
                    continue

                # Filter relevant benchmark/report files
                is_report_dir = search_dir in [
                    Path("paper"),
                    Path(self.results_dir),
                    Path("results"),
                    Path("results/thesis"),
                    Path("experiments/results"),
                ]
                is_root_report = (
                    "benchmark" in fname.lower()
                    or "report" in fname.lower()
                    or "reward" in fname.lower()
                    or "tournament" in fname.lower()
                    or "paper" in fname.lower()
                )

                if not is_report_dir and not is_root_report:
                    continue

                seen_filenames.add(fname)
                mtime = f.stat().st_mtime
                size_kb = round(f.stat().st_size / 1024, 2)
                dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                if ext in [".tex", ".bib"]:
                    file_type = "latex"
                elif ext == ".md":
                    file_type = "markdown"
                elif ext == ".json":
                    file_type = "json"
                else:
                    file_type = "text"

                reports.append(
                    ReportItemDTO(
                        id=fname,
                        name=self._resolve_friendly_name(fname),
                        filename=fname,
                        file_type=file_type,
                        size_kb=size_kb,
                        modified_at=dt_str,
                        phase=self._resolve_phase(fname),
                    )
                )

        return reports

    def get_report(self, filename: str) -> ReportDetailDTO | None:
        """Fetch content and metadata of a specific report."""
        search_dirs = [
            Path("paper"),
            Path(self.results_dir),
            Path("results"),
            Path("results/thesis"),
            Path("experiments/results"),
            Path("."),
        ]
        target_path: Path | None = None
        for d in search_dirs:
            p = d / filename
            if p.exists() and p.is_file():
                target_path = p
                break

        if not target_path or not target_path.exists():
            return None

        content = target_path.read_text(encoding="utf-8", errors="replace")
        ext = target_path.suffix.lower()
        if ext in [".tex", ".bib"]:
            file_type = "latex"
        elif ext == ".md":
            file_type = "markdown"
        elif ext == ".json":
            file_type = "json"
        else:
            file_type = "text"

        json_data = None
        if file_type == "json":
            try:
                json_data = json.loads(content)
            except Exception:
                pass

        mtime = target_path.stat().st_mtime
        size_kb = round(target_path.stat().st_size / 1024, 2)
        dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

        return ReportDetailDTO(
            id=filename,
            name=self._resolve_friendly_name(filename),
            filename=filename,
            file_type=file_type,
            raw_content=content,
            json_data=json_data,
            size_kb=size_kb,
            modified_at=dt_str,
        )

    def delete_report(self, filename: str) -> bool:
        """Delete a report file."""
        target_path = Path(self.results_dir) / filename
        if not target_path.exists():
            target_path = Path(filename)
            if not target_path.exists():
                return False

        try:
            target_path.unlink()
            return True
        except Exception:
            return False

    def get_extended_telemetry_summary(self) -> dict[str, Any] | None:
        """Loads and returns summary metrics from extended_telemetry_results.json."""
        candidates = [
            Path("results/thesis/extended_telemetry_results.json"),
            Path("results/extended_telemetry_results.json"),
            Path("extended_telemetry_results.json"),
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data.get("summary")
                except Exception:
                    continue
        return None
