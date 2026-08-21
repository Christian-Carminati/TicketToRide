#!/usr/bin/env python3
"""CLI runner for Phase 12 Advanced Research cross-paradigm benchmark."""

import argparse
import json
from pathlib import Path
from src.evaluation.phase12_benchmark import Phase12ScientificBenchmark

def main():
    parser = argparse.ArgumentParser(description="TicketToRide Phase 12 Scientific Benchmark")
    parser.add_argument("--episodes", type=int, default=4, help="Episodes per matchup")
    parser.add_argument("--output", type=str, default="benchmark_phase12.json", help="Output JSON results path")
    args = parser.parse_args()

    print(f"[*] Running Phase 12 Benchmark with {args.episodes} episodes per matchup...")
    bench = Phase12ScientificBenchmark(num_games_per_pair=args.episodes)
    results = bench.run_quick_cross_paradigm_benchmark()
    
    report = bench.generate_report(results)
    print("\n" + report + "\n")
    
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved results to {out_path}")

if __name__ == "__main__":
    main()
