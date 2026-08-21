"""
Curriculum Learning Manager for Progressive Training across Maps & Opponents.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class CurriculumStageConfig:
    stage_id: int
    name: str
    map_type: str
    opponent_type: str
    min_win_rate: float
    min_ticket_rate: float
    eval_episodes: int = 30

class CurriculumManager:
    """
    Manages progression through educational and competitive curriculum stages.
    """
    def __init__(self, stages: Optional[List[CurriculumStageConfig]] = None):
        if stages is None:
            self.stages = [
                CurriculumStageConfig(
                    stage_id=1,
                    name="Stage 1: Micro Rules & Foundations",
                    map_type="micro",
                    opponent_type="random",
                    min_win_rate=0.75,
                    min_ticket_rate=0.80,
                ),
                CurriculumStageConfig(
                    stage_id=2,
                    name="Stage 2: Regional Chaining & Heuristic Contest",
                    map_type="small",
                    opponent_type="greedy",
                    min_win_rate=0.70,
                    min_ticket_rate=0.75,
                ),
                CurriculumStageConfig(
                    stage_id=3,
                    name="Stage 3: Full USA Strategic Mastery",
                    map_type="standard",
                    opponent_type="strategic",
                    min_win_rate=0.65,
                    min_ticket_rate=0.70,
                ),
            ]
        else:
            self.stages = stages
            
        self.current_stage_idx: int = 0
        self.history: List[Dict[str, float]] = []

    @property
    def current_stage(self) -> CurriculumStageConfig:
        if self.is_completed:
            return self.stages[-1]
        return self.stages[self.current_stage_idx]

    @property
    def is_completed(self) -> bool:
        return self.current_stage_idx >= len(self.stages)

    def record_evaluation(self, win_rate: float, ticket_rate: float) -> bool:
        """
        Records evaluation result and checks promotion gate.
        Returns True if promoted or already completed, False otherwise.
        """
        if self.is_completed:
            return True
            
        stage = self.current_stage
        self.history.append({
            "stage_id": float(stage.stage_id),
            "win_rate": win_rate,
            "ticket_rate": ticket_rate,
        })
        
        if win_rate >= stage.min_win_rate and ticket_rate >= stage.min_ticket_rate:
            self.current_stage_idx += 1
            return True
        return False

    def get_summary(self) -> str:
        """Returns human-readable summary of curriculum progression."""
        lines = ["# Curriculum Progression Summary"]
        for idx, stage in enumerate(self.stages):
            status = "[COMPLETED]" if idx < self.current_stage_idx else ("[ACTIVE]" if idx == self.current_stage_idx else "[LOCKED]")
            lines.append(f"- {stage.name} ({stage.map_type} map vs {stage.opponent_type}): {status}")
        return "\n".join(lines)
