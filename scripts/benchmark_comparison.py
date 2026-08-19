"""Scientific Benchmark Comparison Script for TicketToRide RL Lab."""

import json
from pathlib import Path


def generate_comparison_report():
    baseline_path = Path("benchmark_baseline.json")
    optimized_path = Path("benchmark_optimized.json")

    if not baseline_path.exists() or not optimized_path.exists():
        print("Missing benchmark JSON files. Run benchmark_baseline.py and benchmark_optimized.py first.")
        return

    with open(baseline_path, "r") as f:
        base = json.load(f)

    with open(optimized_path, "r") as f:
        opt = json.load(f)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("🏆 TICKETTORIDE RL LAB: SCIENTIFIC PERFORMANCE IMPROVEMENT REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"{'Subsystem / Benchmark':<35} | {'Baseline':<14} | {'Optimized':<14} | {'Delta / Speedup':<15}")
    report_lines.append("-" * 80)

    # 1. DQN FPS
    base_dqn_fps = base["dqn_training"]["fps"]
    opt_dqn_fps = opt["dqn_training"]["fps"]
    dqn_speedup = ((opt_dqn_fps - base_dqn_fps) / base_dqn_fps) * 100.0
    report_lines.append(f"{'DQN Training Throughput (FPS)':<35} | {base_dqn_fps:>8.1f} steps/s | {opt_dqn_fps:>8.1f} steps/s | {dqn_speedup:>+8.1f}% 🚀")

    # 2. DQN Latency
    base_dqn_ms = base["dqn_training"]["ms_per_step"]
    opt_dqn_ms = opt["dqn_training"]["ms_per_step"]
    dqn_lat_red = ((base_dqn_ms - opt_dqn_ms) / base_dqn_ms) * 100.0
    report_lines.append(f"{'DQN Step Latency (ms)':<35} | {base_dqn_ms:>10.3f} ms | {opt_dqn_ms:>10.3f} ms | {-dqn_lat_red:>+8.1f}% ⚡")

    # 3. PPO FPS
    base_ppo_fps = base["ppo_training"]["fps"]
    opt_ppo_fps = opt["ppo_training"]["fps"]
    ppo_speedup = ((opt_ppo_fps - base_ppo_fps) / base_ppo_fps) * 100.0
    report_lines.append(f"{'PPO Training Throughput (FPS)':<35} | {base_ppo_fps:>8.1f} steps/s | {opt_ppo_fps:>8.1f} steps/s | {ppo_speedup:>+8.1f}% 🚀")

    # 4. PPO Latency
    base_ppo_ms = base["ppo_training"]["ms_per_step"]
    opt_ppo_ms = opt["ppo_training"]["ms_per_step"]
    ppo_lat_red = ((base_ppo_ms - opt_ppo_ms) / base_ppo_ms) * 100.0
    report_lines.append(f"{'PPO Step Latency (ms)':<35} | {base_ppo_ms:>10.3f} ms | {opt_ppo_ms:>10.3f} ms | {-ppo_lat_red:>+8.1f}% ⚡")

    # 5. Observation Encoder
    base_obs_cps = base["observation_encoder"]["encodes_per_sec"]
    opt_obs_cps = opt["observation_encoder"]["encodes_per_sec"]
    obs_speedup = ((opt_obs_cps - base_obs_cps) / base_obs_cps) * 100.0
    report_lines.append(f"{'ObservationV1 Encoding (calls/s)':<35} | {base_obs_cps:>8.1f} call/s | {opt_obs_cps:>8.1f} call/s | {obs_speedup:>+8.1f}% 🚀")

    # 6. Reward Calculator
    base_rew_cps = base["reward_calculator"]["rewards_per_sec"]
    opt_rew_cps = opt["reward_calculator"]["rewards_per_sec"]
    rew_speedup = ((opt_rew_cps - base_rew_cps) / base_rew_cps) * 100.0
    report_lines.append(f"{'Reward Calculator (calls/s)':<35} | {base_rew_cps:>8.1f} call/s | {opt_rew_cps:>8.1f} call/s | {rew_speedup:>+8.1f}% 🚀")

    # 7. Action Masker
    base_mask_cps = base["action_masker"]["masks_per_sec"]
    opt_mask_cps = opt["action_masker"]["masks_per_sec"]
    mask_speedup = ((opt_mask_cps - base_mask_cps) / base_mask_cps) * 100.0
    report_lines.append(f"{'Action Masker (calls/s)':<35} | {base_mask_cps:>8.1f} call/s | {opt_mask_cps:>8.1f} call/s | {mask_speedup:>+8.1f}% 🚀")

    # 8. Game Core Simulation
    base_game_gps = base["game_core"]["games_per_sec"]
    opt_game_gps = opt["game_core"]["games_per_sec"]
    game_speedup = ((opt_game_gps - base_game_gps) / base_game_gps) * 100.0
    report_lines.append(f"{'Game Engine Simulation (games/s)':<35} | {base_game_gps:>9.2f} game/s | {opt_game_gps:>9.2f} game/s | {game_speedup:>+8.1f}% 🚀")

    # 9. Tournament
    base_tourn_gps = base["tournament"]["games_per_sec"]
    opt_tourn_gps = opt["tournament"]["games_per_sec"]
    tourn_speedup = ((opt_tourn_gps - base_tourn_gps) / base_tourn_gps) * 100.0
    report_lines.append(f"{'Tournament Benchmark (games/s)':<35} | {base_tourn_gps:>9.2f} game/s | {opt_tourn_gps:>9.2f} game/s | {tourn_speedup:>+8.1f}% 🚀")

    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)
    print(report_text)

    with open("benchmark_comparison_report.txt", "w") as f:
        f.write(report_text)


if __name__ == "__main__":
    generate_comparison_report()
