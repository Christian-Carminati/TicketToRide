"""
Report Generator per confronti tra algoritmi
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os


def generate_comparison_report(heuristic_results: Dict, 
                              rl_results: Dict = None,
                              output_file: str = "comparison_report.md"):
    """
    Genera un report markdown dettagliato confrontando tutti gli algoritmi.
    """
    lines = []
    
    lines.append("# Ticket to Ride - Analisi Comparativa Algoritmi\n\n")
    lines.append("## Executive Summary\n\n")
    
    # Trova il migliore tra euristiche
    best_heuristic = max(heuristic_results.items(), key=lambda x: x[1].total_score)
    
    lines.append(f"- **Miglior algoritmo euristico**: {best_heuristic[0]} "
                f"({best_heuristic[1].total_score} punti)\n")
    lines.append(f"- **Tempo miglior euristica**: {best_heuristic[1].computation_time:.3f}s\n")
    
    if rl_results and 'q_learning' in rl_results:
        rl_solution = rl_results['q_learning']
        lines.append(f"- **Punteggio RL**: {rl_solution.total_score} punti\n")
        lines.append(f"- **Training episodes**: {rl_solution.episodes}\n")
    
    lines.append("\n---\n\n")
    
    # Tabella comparativa euristiche
    lines.append("## Algoritmi Euristici\n\n")
    lines.append("| Algoritmo | Punteggio | Tempo (s) | Rotte | Vagoni | Efficienza |\n")
    lines.append("|-----------|----------:|----------:|------:|-------:|-----------:|\n")
    
    for name, solution in sorted(heuristic_results.items(), 
                                 key=lambda x: x[1].total_score, reverse=True):
        efficiency = solution.route_points / solution.trains_used if solution.trains_used > 0 else 0
        lines.append(f"| {solution.algorithm} | {solution.total_score} | "
                    f"{solution.computation_time:.3f} | {len(solution.routes)} | "
                    f"{solution.trains_used} | {efficiency:.2f} |\n")
    
    lines.append("\n---\n\n")
    
    # Dettagli per ogni algoritmo
    lines.append("## Dettagli Algoritmi\n\n")
    
    for name, solution in heuristic_results.items():
        lines.append(f"### {solution.algorithm}\n\n")
        lines.append(f"**Breakdown Punteggio:**\n")
        lines.append(f"- Punti da rotte: {solution.route_points}\n")
        lines.append(f"- Percorso più lungo: {solution.longest_path_length} vagoni\n")
        lines.append(f"- Bonus percorso lungo: +{solution.longest_bonus}\n")
        lines.append(f"- Punti da tickets: {solution.ticket_points}\n")
        lines.append(f"- **TOTALE**: {solution.total_score}\n\n")
        
        lines.append(f"**Performance:**\n")
        lines.append(f"- Tempo di esecuzione: {solution.computation_time:.3f}s\n")
        lines.append(f"- Rotte selezionate: {len(solution.routes)}\n")
        lines.append(f"- Vagoni utilizzati: {solution.trains_used}/45\n")
        
        if solution.trains_used > 0:
            efficiency = solution.route_points / solution.trains_used
            lines.append(f"- Efficienza: {efficiency:.3f} punti/vagone\n")
        
        lines.append("\n")
    
    # Se c'è RL, aggiungi sezione
    if rl_results and 'q_learning' in rl_results:
        lines.append("---\n\n")
        lines.append("## Reinforcement Learning\n\n")
        
        rl_solution = rl_results['q_learning']
        lines.append(f"### Q-Learning Agent\n\n")
        lines.append(f"**Training:**\n")
        lines.append(f"- Episodi di training: {rl_solution.episodes}\n")
        lines.append(f"- Tempo di training: (vedi output console)\n\n")
        
        lines.append(f"**Risultati:**\n")
        lines.append(f"- Punteggio: {rl_solution.total_score}\n")
        lines.append(f"- Punti da rotte: {rl_solution.route_points}\n")
        lines.append(f"- Bonus percorso lungo: +{rl_solution.longest_bonus}\n")
        lines.append(f"- Punti da tickets: {rl_solution.ticket_points}\n")
        lines.append(f"- Rotte selezionate: {len(rl_solution.routes)}\n")
        lines.append(f"- Vagoni utilizzati: {rl_solution.trains_used}/45\n\n")
        
        # Confronto con migliore euristica
        gap = best_heuristic[1].total_score - rl_solution.total_score
        if gap > 0:
            lines.append(f"**Confronto con migliore euristica ({best_heuristic[0]}):**\n")
            lines.append(f"- Gap: -{gap} punti\n")
            lines.append(f"- RL raggiunge il {rl_solution.total_score/best_heuristic[1].total_score*100:.1f}% "
                        f"del punteggio migliore\n\n")
        else:
            lines.append(f"**RL supera tutte le euristiche!** (+{abs(gap)} punti)\n\n")
    
    lines.append("---\n\n")
    
    # Raccomandazioni
    lines.append("## Raccomandazioni\n\n")
    
    # Trova l'algoritmo più veloce
    fastest = min(heuristic_results.items(), key=lambda x: x[1].computation_time)
    lines.append(f"### Velocità\n")
    lines.append(f"**Algoritmo più veloce**: {fastest[0]} ({fastest[1].computation_time:.3f}s)\n\n")
    
    # Miglior qualità
    lines.append(f"### Qualità\n")
    lines.append(f"**Miglior punteggio**: {best_heuristic[0]} ({best_heuristic[1].total_score} punti)\n\n")
    
    # Best trade-off
    lines.append(f"### Trade-off Qualità/Velocità\n")
    
    # Calcola score per secondo
    scores_per_sec = {name: sol.total_score / max(sol.computation_time, 0.001) 
                      for name, sol in heuristic_results.items()}
    best_tradeoff = max(scores_per_sec.items(), key=lambda x: x[1])
    
    lines.append(f"**Miglior rapporto qualità/tempo**: {best_tradeoff[0]} "
                f"({best_tradeoff[1]:.0f} punti/secondo)\n\n")
    
    lines.append("### Quando usare ogni algoritmo\n\n")
    lines.append("- **Greedy**: Quando serve una risposta istantanea o come baseline\n")
    lines.append("- **Local Search**: Per migliorare rapidamente una soluzione greedy\n")
    lines.append("- **Simulated Annealing**: Quando local search si blocca in ottimi locali\n")
    lines.append("- **Genetic Algorithm**: Per problemi complessi dove la qualità è prioritaria\n")
    lines.append("- **Tabu Search**: Alternativa a SA con diverso tipo di esplorazione\n")
    
    if rl_results:
        lines.append("- **Q-Learning (RL)**: Quando vuoi che l'agente impari e si adatti\n")
    
    lines.append("\n---\n\n")
    
    # Conclusioni
    lines.append("## Conclusioni\n\n")
    lines.append(f"L'analisi ha confrontato {len(heuristic_results)} algoritmi euristici")
    
    if rl_results:
        lines.append(" e 1 algoritmo di reinforcement learning")
    
    lines.append(".\n\n")
    
    lines.append(f"Il **{best_heuristic[0]}** ha ottenuto il miglior risultato con "
                f"**{best_heuristic[1].total_score} punti** in "
                f"**{best_heuristic[1].computation_time:.3f} secondi**.\n\n")
    
    # Statistiche aggregate
    all_scores = [sol.total_score for sol in heuristic_results.values()]
    lines.append(f"**Statistiche aggregate (euristiche)**:\n")
    lines.append(f"- Punteggio medio: {np.mean(all_scores):.1f}\n")
    lines.append(f"- Deviazione standard: {np.std(all_scores):.1f}\n")
    lines.append(f"- Range: {min(all_scores)} - {max(all_scores)}\n")
    
    # Scrivi file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"\n✅ Report generato: {os.path.abspath(output_file)}")


def visualize_comparison(heuristic_results: Dict, rl_results: Dict = None, 
                        save_path: str = "comparison_visualization.png"):
    """
    Crea visualizzazioni comparative tra tutti gli algoritmi.
    """
    # Prepara dati
    algorithms = [sol.algorithm for sol in heuristic_results.values()]
    scores = [sol.total_score for sol in heuristic_results.values()]
    times = [sol.computation_time for sol in heuristic_results.values()]
    
    if rl_results and 'q_learning' in rl_results:
        algorithms.append("Q-Learning")
        scores.append(rl_results['q_learning'].total_score)
        times.append(rl_results['q_learning'].computation_time)
    
    # Crea figura con 4 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Confronto Algoritmi - Ticket to Ride', fontsize=16, fontweight='bold')
    
    # 1. Punteggi totali
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(algorithms)))
    bars1 = ax1.barh(algorithms, scores, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Punteggio Totale', fontsize=12, fontweight='bold')
    ax1.set_title('Confronto Punteggi', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Aggiungi valori sulle barre
    for bar, score in zip(bars1, scores):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2., 
                f' {int(score)}', ha='left', va='center', fontweight='bold')
    
    # 2. Tempi di esecuzione (scala logaritmica)
    ax2 = axes[0, 1]
    bars2 = ax2.barh(algorithms, times, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Tempo (secondi)', fontsize=12, fontweight='bold')
    ax2.set_title('Confronto Tempi di Esecuzione', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, time in zip(bars2, times):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2., 
                f' {time:.3f}s', ha='left', va='center', fontweight='bold', fontsize=9)
    
    # 3. Breakdown punteggi (stacked bar)
    ax3 = axes[1, 0]
    
    route_points = []
    longest_bonus = []
    ticket_points = []
    
    for sol in heuristic_results.values():
        route_points.append(sol.route_points)
        longest_bonus.append(sol.longest_bonus)
        ticket_points.append(sol.ticket_points)
    
    if rl_results and 'q_learning' in rl_results:
        rl_sol = rl_results['q_learning']
        route_points.append(rl_sol.route_points)
        longest_bonus.append(rl_sol.longest_bonus)
        ticket_points.append(rl_sol.ticket_points)
    
    x = np.arange(len(algorithms))
    width = 0.6
    
    p1 = ax3.barh(x, route_points, width, label='Punti Rotte', color='steelblue', edgecolor='black')
    p2 = ax3.barh(x, longest_bonus, width, left=route_points, 
                  label='Bonus Percorso', color='orange', edgecolor='black')
    p3 = ax3.barh(x, ticket_points, width, 
                  left=[r+b for r, b in zip(route_points, longest_bonus)], 
                  label='Tickets', color='lightgreen', edgecolor='black')
    
    ax3.set_yticks(x)
    ax3.set_yticklabels(algorithms)
    ax3.set_xlabel('Punti', fontsize=12, fontweight='bold')
    ax3.set_title('Breakdown Punteggio per Componente', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Efficienza (Punti per Secondo)
    ax4 = axes[1, 1]
    
    efficiency = [s / max(t, 0.001) for s, t in zip(scores, times)]
    
    bars4 = ax4.barh(algorithms, efficiency, color=colors, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Punti / Secondo', fontsize=12, fontweight='bold')
    ax4.set_title('Efficienza (Quality/Speed Trade-off)', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    for bar, eff in zip(bars4, efficiency):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2., 
                f' {eff:.0f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualizzazione salvata: {os.path.abspath(save_path)}")
    plt.show()


def plot_rl_training_progress(rewards_history: List[float], 
                              window_size: int = 100,
                              save_path: str = "rl_training_progress.png"):
    """
    Visualizza il progresso del training RL
    """
    plt.figure(figsize=(12, 6))
    
    # Plot rewards grezzi
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, alpha=0.3, color='blue', label='Raw Rewards')
    
    # Moving average
    if len(rewards_history) >= window_size:
        moving_avg = np.convolve(rewards_history, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        plt.plot(range(window_size-1, len(rewards_history)), moving_avg, 
                color='red', linewidth=2, label=f'Moving Avg (window={window_size})')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title('RL Training Progress - Rewards', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot distribuzione rewards
    plt.subplot(1, 2, 2)
    plt.hist(rewards_history, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(rewards_history), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(rewards_history):.1f}')
    plt.axvline(np.median(rewards_history), color='orange', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(rewards_history):.1f}')
    plt.xlabel('Cumulative Reward', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Reward Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Grafico training RL salvato: {os.path.abspath(save_path)}")
    plt.show()


def generate_detailed_solution_report(solution, output_file: str = "solution_details.md"):
    """
    Genera un report dettagliato per una singola soluzione
    """
    lines = []
    
    lines.append(f"# Dettaglio Soluzione - {solution.algorithm}\n\n")
    
    lines.append("## Statistiche Generali\n\n")
    lines.append(f"- **Punteggio Totale**: {solution.total_score}\n")
    lines.append(f"- **Tempo di Calcolo**: {solution.computation_time:.3f}s\n")
    lines.append(f"- **Numero Rotte**: {len(solution.routes)}\n")
    lines.append(f"- **Vagoni Utilizzati**: {solution.trains_used}/45\n")
    lines.append(f"- **Vagoni Rimanenti**: {45 - solution.trains_used}\n\n")
    
    lines.append("## Breakdown Punteggio\n\n")
    lines.append(f"- **Punti da Rotte**: {solution.route_points}\n")
    lines.append(f"- **Percorso Più Lungo**: {solution.longest_path_length} vagoni\n")
    lines.append(f"- **Bonus Percorso**: +{solution.longest_bonus}\n")
    lines.append(f"- **Punti da Tickets**: {solution.ticket_points}\n")
    lines.append(f"- **TOTALE**: {solution.total_score}\n\n")
    
    if solution.trains_used > 0:
        efficiency = solution.route_points / solution.trains_used
        lines.append(f"**Efficienza**: {efficiency:.3f} punti/vagone\n\n")
    
    lines.append("---\n\n")
    
    lines.append("## Rotte Selezionate\n\n")
    lines.append("| # | Da | A | Lunghezza | Colore | Punti |\n")
    lines.append("|--:|---|---|----------:|--------|------:|\n")
    
    points_table = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
    
    for i, (city1, city2, length, color) in enumerate(sorted(solution.routes, 
                                                             key=lambda r: r[2], 
                                                             reverse=True), 1):
        points = points_table.get(length, 0)
        lines.append(f"| {i} | {city1} | {city2} | {length} | {color} | {points} |\n")
    
    lines.append("\n")
    
    # Distribuzione lunghezze
    lines.append("## Distribuzione Lunghezze Rotte\n\n")
    from collections import Counter
    length_counts = Counter(r[2] for r in solution.routes)
    
    for length in sorted(length_counts.keys(), reverse=True):
        count = length_counts[length]
        points = points_table.get(length, 0)
        total_points = count * points
        lines.append(f"- **Lunghezza {length}**: {count} rotte × {points} punti = {total_points} punti\n")
    
    lines.append("\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✅ Report dettagliato salvato: {os.path.abspath(output_file)}")


def create_summary_table(heuristic_results: Dict, rl_results: Dict = None):
    """
    Crea una tabella riassuntiva stampabile
    """
    print("\n" + "="*80)
    print(" " * 25 + "TABELLA RIASSUNTIVA")
    print("="*80)
    print()
    
    # Header
    print(f"{'Algoritmo':<25} {'Score':>8} {'Tempo':>10} {'Rotte':>7} {'Vagoni':>8} {'Eff.':>7}")
    print("-" * 80)
    
    # Dati euristiche
    for name, sol in sorted(heuristic_results.items(), 
                           key=lambda x: x[1].total_score, 
                           reverse=True):
        eff = sol.route_points / sol.trains_used if sol.trains_used > 0 else 0
        print(f"{sol.algorithm:<25} {sol.total_score:>8} {sol.computation_time:>9.3f}s "
              f"{len(sol.routes):>7} {sol.trains_used:>7}/45 {eff:>7.2f}")
    
    # Dati RL
    if rl_results and 'q_learning' in rl_results:
        print("-" * 80)
        rl_sol = rl_results['q_learning']
        eff = rl_sol.route_points / rl_sol.trains_used if rl_sol.trains_used > 0 else 0
        print(f"{'Q-Learning (RL)':<25} {rl_sol.total_score:>8} {rl_sol.computation_time:>9.3f}s "
              f"{len(rl_sol.routes):>7} {rl_sol.trains_used:>7}/45 {eff:>7.2f}")
    
    print("="*80)
    print()


def export_results_csv(heuristic_results: Dict, rl_results: Dict = None,
                      output_file: str = "results.csv"):
    """
    Esporta risultati in formato CSV per analisi ulteriori
    """
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Algorithm', 'Total_Score', 'Route_Points', 'Longest_Bonus', 
                        'Ticket_Points', 'Num_Routes', 'Trains_Used', 'Computation_Time',
                        'Efficiency'])
        
        # Euristiche
        for name, sol in heuristic_results.items():
            eff = sol.route_points / sol.trains_used if sol.trains_used > 0 else 0
            writer.writerow([
                sol.algorithm,
                sol.total_score,
                sol.route_points,
                sol.longest_bonus,
                sol.ticket_points,
                len(sol.routes),
                sol.trains_used,
                sol.computation_time,
                eff
            ])
        
        # RL
        if rl_results and 'q_learning' in rl_results:
            rl_sol = rl_results['q_learning']
            eff = rl_sol.route_points / rl_sol.trains_used if rl_sol.trains_used > 0 else 0
            writer.writerow([
                'Q-Learning',
                rl_sol.total_score,
                rl_sol.route_points,
                rl_sol.longest_bonus,
                rl_sol.ticket_points,
                len(rl_sol.routes),
                rl_sol.trains_used,
                rl_sol.computation_time,
                eff
            ])
    
    print(f"✅ Risultati esportati in CSV: {os.path.abspath(output_file)}")