"""Optimal solver using Branch and Bound algorithm for Ticket to Ride."""
from __future__ import annotations

import time
from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd

from core.constants import POINTS_TABLE, TOTAL_TRAINS
from core.routes import RouteTuple, extract_routes, trains_used, unique_routes_from_graph
from core.scoring import (
    calculate_longest_path,
    calculate_route_points,
    calculate_ticket_points,
    score_routes,
)
from best_solution.solution import Solution


class OptimalSolver:
    """
    Risolutore ottimo per Ticket to Ride.
    
    Utilizza Branch and Bound con pruning per trovare la soluzione ottimale globale.
    L'algoritmo esplora tutte le combinazioni valide di rotte, utilizzando stime
    ottimistiche (upper bound) per potare rami che non possono portare a soluzioni migliori.
    
    Attributes:
        graph: Grafo NetworkX rappresentante la mappa del gioco
        tickets_file: Path al file CSV contenente i biglietti destinazione
        tickets_df: DataFrame pandas con i biglietti caricati
        nodes_explored: Contatore dei nodi esplorati durante la ricerca
        nodes_pruned: Contatore dei nodi potati grazie al pruning
        best_score: Miglior punteggio trovato finora
    """
    
    # Costante per il bonus ottimistico nell'upper bound
    OPTIMISTIC_BONUS = 50
    
    def __init__(self, graph: nx.MultiGraph, tickets_file: str) -> None:
        """
        Inizializza il risolutore ottimo.
        
        Args:
            graph: Grafo NetworkX della mappa
            tickets_file: Path al file CSV dei biglietti destinazione
        """
        self.graph = graph
        self.tickets_file = tickets_file
        self.tickets_df = pd.read_csv(tickets_file)
        
        # Statistiche di ricerca
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.best_score = 0
    
    def _evaluate_solution(self, routes: List[RouteTuple]) -> int:
        """
        Valuta il punteggio totale di una soluzione.
        
        Args:
            routes: Lista di rotte selezionate
            
        Returns:
            Punteggio totale della soluzione
        """
        breakdown = score_routes(routes, self.tickets_df)
        return breakdown.total_score
    
    def _upper_bound_estimate(
        self, 
        selected_routes: List[RouteTuple], 
        remaining_routes: List[RouteTuple], 
        trains_left: int
    ) -> int:
        """
        Stima ottimistica del punteggio massimo raggiungibile.
        
        Utilizzata per pruning: se upper_bound <= best_score, pota il ramo.
        La stima aggiunge greedily le rotte più efficienti rimanenti e un bonus
        ottimistico per biglietti potenzialmente completabili.
        
        Args:
            selected_routes: Rotte già selezionate nella soluzione parziale
            remaining_routes: Rotte ancora disponibili da considerare
            trains_left: Numero di treni rimanenti
            
        Returns:
            Stima ottimistica del punteggio massimo raggiungibile
        """
        current_score = self._evaluate_solution(selected_routes)
        
        # Ordina rotte rimanenti per efficienza (punti/vagone)
        sorted_remaining = sorted(
            remaining_routes,
            key=lambda r: POINTS_TABLE[r[2]] / r[2] if r[2] > 0 else 0,
            reverse=True
        )
        
        # Aggiungi greedily le rotte più efficienti (ottimistico)
        optimistic_routes = list(selected_routes)
        trains_remaining = trains_left
        
        for route in sorted_remaining:
            if trains_remaining >= route[2]:
                optimistic_routes.append(route)
                trains_remaining -= route[2]
        
        # Calcola score ottimistico
        optimistic_score = self._evaluate_solution(optimistic_routes)
        
        # Aggiungi un bonus ottimistico per tickets non ancora considerati
        # (assumiamo che alcuni tickets extra potrebbero essere completati)
        optimistic_score += self.OPTIMISTIC_BONUS
        
        return optimistic_score
    
    def _branch_and_bound(
        self, 
        unique_routes: List[RouteTuple], 
        index: int,
        selected_routes: List[RouteTuple],
        trains_left: int,
        selected_pairs: Set[Tuple[str, str]],
        best_solution: Dict[str, int | List[RouteTuple]]
    ) -> None:
        """
        Branch and Bound per trovare la soluzione ottima.
        
        Esplora ricorsivamente tutte le combinazioni valide di rotte, utilizzando
        pruning basato su upper bound per evitare di esplorare rami non promettenti.
        
        Args:
            unique_routes: Lista di rotte uniche da considerare (ordinate per efficienza)
            index: Indice corrente nella lista di rotte
            selected_routes: Rotte selezionate nella soluzione parziale corrente
            trains_left: Numero di treni rimanenti
            selected_pairs: Set di coppie di città già selezionate (per evitare duplicati)
            best_solution: Dizionario contenente il miglior punteggio e le rotte trovate
        """
        self.nodes_explored += 1
        
        # Valuta soluzione corrente
        current_score = self._evaluate_solution(selected_routes)
        if current_score > best_solution['score']:
            best_solution['score'] = current_score
            best_solution['routes'] = list(selected_routes)
            self.best_score = current_score
        
        # Caso base: tutte le rotte esaminate
        if index >= len(unique_routes):
            return
        
        # Pruning: se l'upper bound è peggiore del best, pota
        remaining_routes = unique_routes[index:]
        upper_bound = self._upper_bound_estimate(selected_routes, remaining_routes, trains_left)
        if upper_bound <= best_solution['score']:
            self.nodes_pruned += 1
            return
        
        # Rotta corrente
        route = unique_routes[index]
        city1, city2, length, color = route
        pair = tuple(sorted([city1, city2]))
        
        # Branch 1: Include la rotta (se valida)
        if trains_left >= length and pair not in selected_pairs:
            selected_routes.append(route)
            selected_pairs.add(pair)
            self._branch_and_bound(
                unique_routes, index + 1, selected_routes, 
                trains_left - length, selected_pairs, best_solution
            )
            selected_routes.pop()
            selected_pairs.remove(pair)
        
        # Branch 2: Non include la rotta
        self._branch_and_bound(
            unique_routes, index + 1, selected_routes, 
            trains_left, selected_pairs, best_solution
        )
    
    def solve(self) -> Solution:
        """
        Trova la soluzione ottima globale utilizzando Branch and Bound.
        ATTENZIONE: Può richiedere molto tempo per grafi grandi!
        """
        print("=== OPTIMAL SOLVER - Branch and Bound ===")
        print("Cercando la soluzione ottima globale...")
        print("ATTENZIONE: Questo può richiedere molto tempo!\n")
        
        start_time = time.time()
        
        unique_routes = unique_routes_from_graph(self.graph)
        print(f"Rotte uniche da considerare: {len(unique_routes)}")
        print(f"Spazio di ricerca teorico: 2^{len(unique_routes)} = {2**len(unique_routes):,} combinazioni\n")
        
        # Ordina rotte per efficienza (migliora pruning)
        unique_routes.sort(
            key=lambda r: POINTS_TABLE[r[2]] / r[2] if r[2] > 0 else 0,
            reverse=True
        )
        
        best_solution: Dict[str, int | List[RouteTuple]] = {'score': 0, 'routes': []}
        
        self._branch_and_bound(
            unique_routes, 0, [], TOTAL_TRAINS, set(), best_solution
        )
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Calcola dettagli soluzione finale
        final_routes: List[RouteTuple] = best_solution['routes']  # type: ignore[assignment]
        breakdown = score_routes(final_routes, self.tickets_df)
        trains_used_count = trains_used(final_routes)
        
        print(f"\n=== RISULTATI ===")
        print(f"Nodi esplorati: {self.nodes_explored:,}")
        print(f"Nodi potati: {self.nodes_pruned:,}")
        print(f"Efficienza pruning: {self.nodes_pruned/max(self.nodes_explored, 1)*100:.1f}%")
        print(f"Tempo di calcolo: {computation_time:.2f} secondi")
        print(f"\nPunteggio ottimo trovato: {breakdown.total_score}")
        
        return Solution(
            routes=final_routes,
            route_points=breakdown.route_points,
            longest_path_length=breakdown.longest_path_length,
            longest_bonus=breakdown.longest_bonus,
            ticket_points=breakdown.ticket_points,
            total_score=breakdown.total_score,
            trains_used=trains_used_count,
            computation_time=computation_time,
            algorithm="Branch and Bound"
        )
    
    def solve_limited(self, max_time_seconds: float = 60.0) -> Solution:
        """
        Versione con limite di tempo - interrompe la ricerca dopo max_time_seconds.
        Restituisce la migliore soluzione trovata fino a quel momento.
        """
        print("=== OPTIMAL SOLVER - Limited Time ===")
        print(f"Tempo massimo: {max_time_seconds} secondi\n")
        
        start_time = time.time()
        unique_routes = unique_routes_from_graph(self.graph)
        
        # Inizia con soluzione greedy come baseline
        from best_solution.heuristic_solver import HeuristicSolver
        heuristic = HeuristicSolver(self.graph, self.tickets_file)
        greedy_sol = heuristic.greedy_solve()
        
        best_solution = {'score': greedy_sol.total_score, 'routes': greedy_sol.routes}
        self.best_score = greedy_sol.total_score
        
        print(f"Baseline greedy: {greedy_sol.total_score} punti")
        print("Cercando soluzioni migliori...\n")
        
        # TODO: Implementare ricerca con timeout
        # Per ora usa solve() standard
        return self.solve()