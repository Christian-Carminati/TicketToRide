"""
Main script per confrontare soluzione ottima vs algoritmi euristici.

Questo script esegue un confronto completo tra:
- Algoritmi euristici (Greedy, Local Search, Simulated Annealing, Genetic Algorithm)
- Reinforcement Learning agents
- Soluzione ottima (Branch and Bound)

Genera report comparativi e visualizzazioni dei risultati.
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from best_solution.heuristic_solver import HeuristicSolver, compare_all_heuristics
from best_solution.optimal_solver import OptimalSolver
from best_solution.report_comparison import (
    generate_comparison_report,
    generate_detailed_solution_report,
    plot_rl_training_progress,
    visualize_comparison,
)
from best_solution.rl_solver import RLSolver, train_and_evaluate_rl_agent
from map import TicketToRideMap


def main() -> None:
    """
    Esegue il confronto completo tra tutti i solver disponibili.
    
    Il processo include:
    1. Caricamento della mappa e dei biglietti
    2. Esecuzione di tutti gli algoritmi euristici
    3. Training e valutazione di agenti RL
    4. Generazione di report e visualizzazioni comparativi
    """
    
    print("="*70)
    print(" TICKET TO RIDE - CONFRONTO SOLUZIONE OTTIMA VS EURISTICHE")
    print("="*70)
    print()
    
    # Carica mappa
    print("📍 Caricamento mappa...")
    ttr_map = TicketToRideMap()
    ttr_map.load_graph("src/map/city_locations.json", "src/map/routes.csv")
    graph = ttr_map.get_graph()
    tickets_file = "src/map/tickets.csv"
    print("✅ Mappa caricata\n")
    
    # PARTE 1: Algoritmi Euristici
    print("="*70)
    print(" PARTE 1: ALGORITMI EURISTICI")
    print("="*70)
    print()
    
    heuristic_results = compare_all_heuristics(graph, tickets_file)
    
    # PARTE 2: Reinforcement Learning
    print("\n" + "="*70)
    print(" PARTE 2: REINFORCEMENT LEARNING")
    print("="*70)
    print()

    rl_results = train_and_evaluate_rl_agent(graph, tickets_file, training_episodes=5000, eval_episodes=100)
    
    # PARTE 3: Confronto Risultati
    print("\n" + "="*70)
    print(" PARTE 3: CONFRONTO RISULTATI")
    print("="*70)
    print()
    
    #Report comparativo
    generate_comparison_report(heuristic_results, rl_results, 
                            output_file="my_report.md")

    # Visualizzazioni
    visualize_comparison(heuristic_results, rl_results, 
                        save_path="my_viz.png")

if __name__ == "__main__":
    main()