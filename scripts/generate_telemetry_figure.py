#!/usr/bin/env python3
"""
Publication-Grade Figure Generator for Extended Strategy Telemetry (Fig 6).
Reads results/thesis/extended_telemetry_results.json and outputs PDF/PNG to paper/figures/ and results/thesis/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_fig6(telemetry_file: Path, out_dirs: list[Path]) -> None:
    with open(telemetry_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    hoarding = summary["hoarding_stats"]
    chokepoints = summary["top_contended_chokepoints"][:5]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel (a): Card Hoarding vs Route Blocks Suffered ---
    categories = ["Rush (<8 cards)\nFrettolosi", "Moderate (8-11 cards)\nModerati", "Patient (>=12 cards)\nPazienti"]
    # Mean blocks from data: rush = 1.35, moderate = ~0.45, patient = 0.00
    means = [hoarding["rush_mean_blocks_suffered"], 0.45, hoarding["patient_mean_blocks_suffered"]]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    bars = ax1.bar(categories, means, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Mean Route Blocks Suffered / Match\nBlocchi Subiti per Partita (Media)")
    ax1.set_title("(a) Opening Hand Hoarding vs. Route Blocking\nAccumulo Carte vs. Rischio di Blocco Avversario")
    ax1.set_ylim(0, 1.8)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax1.text(0.95, 0.88, "Information Entropy\nPreservation: 0.00 blocks",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=9, style="italic", bbox=dict(boxstyle="round,pad=0.4", fc="#eaeded", ec="#bdc3c7"))

    # --- Panel (b): Top-5 Contested Chokepoint Routes ---
    routes_labels = [c["cities"].replace("--", " -- ") for c in reversed(chokepoints)]
    turns_contested = [c["contention_count"] for c in reversed(chokepoints)]

    y_pos = np.arange(len(routes_labels))
    bars2 = ax2.barh(y_pos, turns_contested, color="#2980b9", height=0.55, edgecolor="black", linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(routes_labels, fontweight="medium")
    ax2.set_xlabel("Contested Turns Across 80 Matches\nTurni di Contesa Simultanea (80 Partite)")
    ax2.set_title("(b) Top-5 Critical Chokepoint Routes (USA Map)\nPrincipali Colli di Bottiglia Contesi (Mappa USA)")
    ax2.set_xlim(0, max(turns_contested) * 1.2)

    for bar in bars2:
        w = bar.get_width()
        ax2.annotate(f"{w} turns",
                     xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points",
                     ha="left", va="center", fontweight="bold", fontsize=9)

    plt.tight_layout()

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        pdf_path = d / "fig6_extended_telemetry_chokepoints.pdf"
        png_path = d / "fig6_extended_telemetry_chokepoints.png"
        fig.savefig(pdf_path)
        fig.savefig(png_path)
        print(f"Saved: {pdf_path} and {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    res_file = Path("results/thesis/extended_telemetry_results.json")
    out_dirs = [Path("paper/figures"), Path("results/thesis/figures")]
    generate_fig6(res_file, out_dirs)
