import pytest
from src.rl.curriculum import CurriculumStageConfig, CurriculumManager

def test_curriculum_manager_initialization_and_progression():
    stages = [
        CurriculumStageConfig(1, "Micro Stage", map_type="micro", opponent_type="random", min_win_rate=0.75, min_ticket_rate=0.80),
        CurriculumStageConfig(2, "Small Stage", map_type="small", opponent_type="greedy", min_win_rate=0.70, min_ticket_rate=0.75),
        CurriculumStageConfig(3, "Full USA", map_type="standard", opponent_type="strategic", min_win_rate=0.60, min_ticket_rate=0.70),
    ]
    cm = CurriculumManager(stages=stages)
    assert cm.current_stage.stage_id == 1
    assert not cm.is_completed
    
    # Record sub-par performance -> no progression
    promoted = cm.record_evaluation(win_rate=0.60, ticket_rate=0.85)
    assert not promoted
    assert cm.current_stage.stage_id == 1
    
    # Record passing performance -> promoted to stage 2
    promoted2 = cm.record_evaluation(win_rate=0.80, ticket_rate=0.85)
    assert promoted2
    assert cm.current_stage.stage_id == 2
    
    # Promoted to stage 3
    promoted3 = cm.record_evaluation(win_rate=0.75, ticket_rate=0.80)
    assert promoted3
    assert cm.current_stage.stage_id == 3
    
    # Completed final stage
    promoted4 = cm.record_evaluation(win_rate=0.65, ticket_rate=0.75)
    assert promoted4
    assert cm.is_completed

def test_default_curriculum_manager():
    cm = CurriculumManager()
    assert len(cm.stages) == 3
    assert cm.current_stage.stage_id == 1
    summary = cm.get_summary()
    assert "Stage 1" in summary
