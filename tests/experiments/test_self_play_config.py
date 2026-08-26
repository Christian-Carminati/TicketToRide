from src.experiments.config import ExperimentConfig, SelfPlayConfig


def test_selfplay_config_parsing():
    cfg = ExperimentConfig(
        name="test_selfplay",
        self_play=SelfPlayConfig(
            enabled=True,
            snapshot_interval=1000,
            strategy="pfsp",
            baseline_mix_rate=0.2,
            pfsp_exponent=2.0,
            pool_max_size=30,
        ),
    )
    assert cfg.self_play.enabled is True
    assert cfg.self_play.snapshot_interval == 1000
    assert cfg.self_play.strategy == "pfsp"
    assert cfg.self_play.baseline_mix_rate == 0.2
    assert cfg.self_play.pfsp_exponent == 2.0
    assert cfg.self_play.pool_max_size == 30
