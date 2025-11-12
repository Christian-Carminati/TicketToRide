"""
Heuristic solvers for Ticket to Ride.

This module implements various heuristic algorithms for finding good (but not
necessarily optimal) solutions to the Ticket to Ride route selection problem.

Algorithms included:
- Greedy: Selects routes with best points/train ratio
- Local Search: Iteratively improves a solution using neighborhood operators
- Simulated Annealing: Probabilistic search that accepts worse solutions
- Genetic Algorithm: Evolutionary approach with crossover and mutation
- Tabu Search: Uses memory to avoid local optima
- Ticket-Centric Genetic Algorithm: Specialized GA prioritizing connectivity
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd

from best_solution.solution import Solution
from core.constants import POINTS_TABLE, TOTAL_TRAINS
from core.routes import extract_routes, unique_routes_from_graph, trains_used
from core.scoring import (
    LONGEST_PATH_BONUS,
    calculate_longest_path,
    calculate_route_points,
    calculate_ticket_points,
    score_routes,
)

class HeuristicSolver:
    """
    Risolutore euristico per Ticket to Ride.
    Implementa diversi algoritmi euristici veloci:
    - Greedy
    - Simulated Annealing
    - Genetic Algorithm
    - Local Search
    """
    
    def __init__(self, graph: nx.MultiGraph, tickets_file: str):
        self.graph = graph
        self.tickets_file = tickets_file
        self.routes = extract_routes(graph)
        self.points_table = POINTS_TABLE
        self.total_trains = TOTAL_TRAINS
        self.tickets_df = pd.read_csv(tickets_file)
        self.unique_routes = unique_routes_from_graph(graph)
    
    def _evaluate_solution(self, routes: List[Tuple[str, str, int, str]]) -> Tuple[int, Dict[str, int]]:
        breakdown = score_routes(routes, self.tickets_df)
        return breakdown.total_score, {
            'route_points': breakdown.route_points,
            'longest': breakdown.longest_path_length,
            'longest_bonus': breakdown.longest_bonus,
            'ticket_points': breakdown.ticket_points,
        }
    
    def greedy_solve(self) -> Solution:
        """
        Algoritmo Greedy: seleziona iterativamente la rotta con miglior rapporto punti/vagone.
        Veloce ma non garantisce l'ottimo.
        """
        print("=== GREEDY SOLVER ===")
        start_time = time.time()
        
        # Ordina per efficienza punti/vagone
        sorted_routes = sorted(
            self.unique_routes,
            key=lambda r: self.points_table.get(r[2], 0) / r[2],
            reverse=True
        )
        
        selected_routes = []
        selected_pairs = set()
        trains_used_total = 0

        for route in sorted_routes:
            city1, city2, length, color = route
            pair = tuple(sorted([city1, city2]))

            if trains_used_total + length <= self.total_trains and pair not in selected_pairs:
                selected_routes.append(route)
                selected_pairs.add(pair)
                trains_used_total += length
        
        end_time = time.time()
        total_score, details = self._evaluate_solution(selected_routes)
        
        print(f"Punteggio: {total_score}")
        print(f"Tempo: {end_time - start_time:.3f}s\n")
        
        return Solution(
            routes=selected_routes,
            route_points=details['route_points'],
            longest_path_length=details['longest'],
            longest_bonus=details['longest_bonus'],
            ticket_points=details['ticket_points'],
            total_score=total_score,
            trains_used=trains_used_total,
            computation_time=end_time - start_time,
            algorithm="Greedy"
        )
    
    def local_search_solve(self, initial_solution: List[Tuple[str, str, int, str]] = [],
                          max_iterations: int = 1000) -> Solution:
        """
        Local Search con operatori: Add, Remove, Swap.
        Parte da una soluzione iniziale e cerca miglioramenti locali.
        """
        print("=== LOCAL SEARCH SOLVER ===")
        print(f"Max iterazioni: {max_iterations}")
        start_time = time.time()
        
        # Inizializza con greedy se non fornita
        if initial_solution is None or len(initial_solution) == 0:
            greedy_sol = self.greedy_solve()
            current_routes = list(greedy_sol.routes)
        else:
            current_routes = list(initial_solution)
        
        current_score, _ = self._evaluate_solution(current_routes)
        best_routes = list(current_routes)
        best_score = current_score
        
        pair_to_route = {tuple(sorted([r[0], r[1]])): r for r in self.unique_routes}
        
        iterations_without_improvement = 0
        
        for iteration in range(max_iterations):
            improved = False
            
            # Operator 1: Try adding a route
            selected_pairs = {tuple(sorted([r[0], r[1]])) for r in current_routes}
            unselected_pairs = set(pair_to_route.keys()) - selected_pairs
            trains_used_current = trains_used(current_routes)

            for pair in unselected_pairs:
                route = pair_to_route[pair]
                if trains_used_current + route[2] <= self.total_trains:
                    trial_routes = current_routes + [route]
                    trial_score, _ = self._evaluate_solution(trial_routes)
                    
                    if trial_score > best_score:
                        best_score = trial_score
                        best_routes = trial_routes
                        current_routes = trial_routes
                        improved = True
                        break
            
            if improved:
                iterations_without_improvement = 0
                continue
            
            # Operator 2: Try swapping routes
            selected_pairs = {tuple(sorted([r[0], r[1]])) for r in current_routes}
            unselected_pairs = list(set(pair_to_route.keys()) - selected_pairs)
            
            for sel_pair in list(selected_pairs):
                sel_route = pair_to_route[sel_pair]
                for unsel_pair in unselected_pairs:
                    unsel_route = pair_to_route[unsel_pair]
                    new_trains = trains_used_current - sel_route[2] + unsel_route[2]
                    
                    if new_trains <= self.total_trains:
                        # Crea vicino
                        neighbor_routes = [r for r in current_routes if tuple(sorted([r[0], r[1]])) != sel_pair]
                        neighbor_routes.append(unsel_route)
                        neighbor_score, _ = self._evaluate_solution(neighbor_routes)
                        
                        if neighbor_score > best_score:
                            best_score = neighbor_score
                            best_routes = neighbor_routes
                            current_routes = neighbor_routes
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                iterations_without_improvement += 1
                if iterations_without_improvement > 50:
                    break
        
        end_time = time.time()
        total_score, details = self._evaluate_solution(best_routes)
        
        print(f"Punteggio: {total_score}")
        print(f"Tempo: {end_time - start_time:.3f}s\n")
        
        return Solution(
            routes=best_routes,
            route_points=details['route_points'],
            longest_path_length=details['longest'],
            longest_bonus=details['longest_bonus'],
            ticket_points=details['ticket_points'],
            total_score=total_score,
            trains_used=trains_used(best_routes),
            computation_time=end_time - start_time,
            algorithm="Local Search"
        )
    
    def simulated_annealing_solve(self, initial_temp: float = 100.0,
                                 cooling_rate: float = 0.95,
                                 max_iterations: int = 2000) -> Solution:
        """
        Simulated Annealing: accetta soluzioni peggiori con probabilità decrescente
        per evitare ottimi locali.
        """
        print("=== SIMULATED ANNEALING SOLVER ===")
        print(f"Temperatura iniziale: {initial_temp}")
        print(f"Cooling rate: {cooling_rate}")
        start_time = time.time()
        
        # Inizia con soluzione greedy
        greedy_sol = self.greedy_solve()
        current_routes = list(greedy_sol.routes)
        current_score, _ = self._evaluate_solution(current_routes)
        
        best_routes = list(current_routes)
        best_score = current_score
        
        temperature = initial_temp
        pair_to_route = {tuple(sorted([r[0], r[1]])): r for r in self.unique_routes}
        
        for iteration in range(max_iterations):
            # Genera vicino: swap casuale
            selected_pairs = {tuple(sorted([r[0], r[1]])) for r in current_routes}
            unselected_pairs = list(set(pair_to_route.keys()) - selected_pairs)
            
            if not unselected_pairs or not selected_pairs:
                continue
            
            # Scegli casualmente una rotta da rimuovere e una da aggiungere
            sel_pair = random.choice(list(selected_pairs))
            unsel_pair = random.choice(unselected_pairs)
            
            sel_route = pair_to_route[sel_pair]
            unsel_route = pair_to_route[unsel_pair]
            
            trains_used_current = trains_used(current_routes)
            new_trains = trains_used_current - sel_route[2] + unsel_route[2]
            
            if new_trains <= self.total_trains:
                # Crea vicino
                neighbor_routes = [r for r in current_routes if tuple(sorted([r[0], r[1]])) != sel_pair]
                neighbor_routes.append(unsel_route)
                neighbor_score, _ = self._evaluate_solution(neighbor_routes)
                
                # Accetta o rifiuta
                delta = neighbor_score - current_score
                
                if delta > 0 or random.random() < self._acceptance_probability(delta, temperature):
                    current_routes = neighbor_routes
                    current_score = neighbor_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_routes = list(current_routes)
            
            # Raffredda
            temperature *= cooling_rate
        
        end_time = time.time()
        total_score, details = self._evaluate_solution(best_routes)
        
        print(f"Iterazioni: {max_iterations}")
        print(f"Punteggio: {total_score}")
        print(f"Tempo: {end_time - start_time:.3f}s\n")
        
        return Solution(
            routes=best_routes,
            route_points=details['route_points'],
            longest_path_length=details['longest'],
            longest_bonus=details['longest_bonus'],
            ticket_points=details['ticket_points'],
            total_score=total_score,
            trains_used=trains_used(best_routes),
            computation_time=end_time - start_time,
            algorithm="Simulated Annealing"
        )
    
    def _acceptance_probability(self, delta: float, temperature: float) -> float:
        """Probabilità di accettare una soluzione peggiore in SA"""
        import math
        if temperature <= 0:
            return 0
        return math.exp(delta / temperature)
    
    def genetic_algorithm_solve(self, population_size: int = 50,
                               generations: int = 100,
                               mutation_rate: float = 0.1) -> Solution:
        """
        Genetic Algorithm: evolve una popolazione di soluzioni attraverso
        selezione, crossover e mutazione.
        """
        print("=== GENETIC ALGORITHM SOLVER ===")
        print(f"Population: {population_size}, Generations: {generations}")
        start_time = time.time()
        
        # Inizializza popolazione casuale
        population = self._initialize_population(population_size)
        
        best_individual = max(population, key=lambda ind: self._evaluate_solution(ind)[0])
        best_score, _ = self._evaluate_solution(best_individual)
        
        for gen in range(generations):
            # Valuta fitness
            fitness_scores = [(ind, self._evaluate_solution(ind)[0]) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Aggiorna best
            if fitness_scores[0][1] > best_score:
                best_score = fitness_scores[0][1]
                best_individual = list(fitness_scores[0][0])
            
            # Selezione (top 50%)
            selected = [ind for ind, _ in fitness_scores[:population_size // 2]]
            
            # Genera nuova popolazione
            new_population = list(selected)
            
            while len(new_population) < population_size:
                # Crossover
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                child = self._crossover(parent1, parent2)
                
                # Mutazione
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        end_time = time.time()
        total_score, details = self._evaluate_solution(best_individual)
        
        print(f"Generazioni: {generations}")
        print(f"Punteggio: {total_score}")
        print(f"Tempo: {end_time - start_time:.3f}s\n")
        
        return Solution(
            routes=best_individual,
            route_points=details['route_points'],
            longest_path_length=details['longest'],
            longest_bonus=details['longest_bonus'],
            ticket_points=details['ticket_points'],
            total_score=total_score,
            trains_used=trains_used(best_individual),
            computation_time=end_time - start_time,
            algorithm="Genetic Algorithm"
        )
    
    def _initialize_population(self, size: int) -> List[List[Tuple[str, str, int, str]]]:
        """Genera popolazione iniziale casuale"""
        population = []
        pair_to_route = {tuple(sorted([r[0], r[1]])): r for r in self.unique_routes}
        all_pairs = list(pair_to_route.keys())

        for _ in range(size):
            # Genera soluzione casuale valida
            individual = []
            selected_pairs = set()
            trains_used_current = 0

            # Randomizza ordine
            shuffled_pairs = random.sample(all_pairs, len(all_pairs))

            for pair in shuffled_pairs:
                route = pair_to_route[pair]
                if trains_used_current + route[2] <= self.total_trains and pair not in selected_pairs:
                    individual.append(route)
                    selected_pairs.add(pair)
                    trains_used_current += route[2]

            population.append(individual)

        return population
    
    def _crossover(self, parent1: List[Tuple[str, str, int, str]], 
                   parent2: List[Tuple[str, str, int, str]]) -> List[Tuple[str, str, int, str]]:
        """Crossover: combina due genitori"""
        pair_to_route = {tuple(sorted([r[0], r[1]])): r for r in self.unique_routes}
        
        # Unisci rotte dei genitori
        pairs1 = {tuple(sorted([r[0], r[1]])) for r in parent1}
        pairs2 = {tuple(sorted([r[0], r[1]])) for r in parent2}
        
        # Prendi casualmente da entrambi
        child_pairs = set()
        trains_used_current = trains_used(parent1)
        for pair in (pairs1 | pairs2):
            route = pair_to_route[pair]
            if trains_used_current + route[2] <= self.total_trains:
                if random.random() < 0.5:  # 50% probabilità
                    child_pairs.add(pair)
                    trains_used_current += route[2]
        
        return [pair_to_route[p] for p in child_pairs]
    
    def _mutate(self, individual: List[Tuple[str, str, int, str]]) -> List[Tuple[str, str, int, str]]:
        """Mutazione: modifica casualmente un individuo"""
        pair_to_route = {tuple(sorted([r[0], r[1]])): r for r in self.unique_routes}
        selected_pairs = {tuple(sorted([r[0], r[1]])) for r in individual}
        unselected_pairs = list(set(pair_to_route.keys()) - selected_pairs)
        
        if not unselected_pairs:
            return individual
        
        # Rimuovi casualmente una rotta e aggiungi una nuova (se possibile)
        if selected_pairs and random.random() < 0.5:
            remove_pair = random.choice(list(selected_pairs))
            mutated = [r for r in individual if tuple(sorted([r[0], r[1]])) != remove_pair]
        else:
            mutated = list(individual)
        
        # Aggiungi rotta casuale
        trains_used_current = trains_used(mutated)
        random.shuffle(unselected_pairs)

        for pair in unselected_pairs:
            route = pair_to_route[pair]
            if trains_used_current + route[2] <= self.total_trains:
                mutated.append(route)
                break
        
        return mutated
    
    
    # === TABU SEARCH IMPLEMENTATION ===
    
    def tabu_search_solve(self, initial_solution: List[Tuple[str, str, int, str]] = [],
                          max_iterations: int = 2000,
                          tabu_list_size: int = 10) -> Solution:
        """
        Tabu Search: utilizza una lista tabù per evitare ottimi locali,
        proibendo il ritorno a stati o mosse recenti.
        """
        print("=== TABU SEARCH SOLVER ===")
        print(f"Max iterazioni: {max_iterations}, Tabu size: {tabu_list_size}")
        start_time = time.time()
        
        # Inizializza con greedy
        if initial_solution is None or len(initial_solution) == 0:
            greedy_sol = self.greedy_solve()
            current_routes = list(greedy_sol.routes)
        else:
            current_routes = list(initial_solution)
        
        current_score, _ = self._evaluate_solution(current_routes)
        best_routes = list(current_routes)
        best_score = current_score
        
        pair_to_route = {tuple(sorted([r[0], r[1]])): r for r in self.unique_routes}
        all_routes_tuples = list(self.unique_routes)
        
        # Lista Tabu: memorizza le rotte coinvolte in mosse recenti
        # Formato: (rotta_tuple, iterazione_di_scadenza)
        tabu_list: Dict[Tuple[str, str, int, str], int] = {}
        
        for iteration in range(max_iterations):
            best_neighbor = None
            best_neighbor_score = -1
            best_move = None  # (route_removed, route_added)
            
            selected_routes_set = set(current_routes)
            
            # Esplora il vicinato: solo mosse di Swap
            neighbors = []
            
            # Genera tutti i possibili Swap (o Add/Remove, ma lo swap è più efficace)
            for sel_route in current_routes:
                trains_after_remove = trains_used(current_routes) - sel_route[2]
                
                for unsel_route in all_routes_tuples:
                    if unsel_route not in selected_routes_set:
                        new_trains = trains_after_remove + unsel_route[2]
                        
                        if new_trains <= self.total_trains:
                            # Creazione del vicino
                            neighbor_routes = [r for r in current_routes if r != sel_route]
                            neighbor_routes.append(unsel_route)
                            neighbors.append((neighbor_routes, sel_route, unsel_route))
            
            # Trova il miglior vicino non tabù
            for neighbor_routes, sel_route, unsel_route in neighbors:
                neighbor_score, _ = self._evaluate_solution(neighbor_routes)
                
                # Check Tabu: la rotta rimossa (sel_route) è tabù se è stata aggiunta di recente.
                # La rotta aggiunta (unsel_route) è tabù se è stata rimossa di recente.
                is_tabu = (tabu_list.get(sel_route, 0) > iteration) or \
                          (tabu_list.get(unsel_route, 0) > iteration)
                
                # Criterio di Aspirazione: se la mossa porta a un punteggio migliore del BEST, ignora il tabù
                aspiration_criterion = neighbor_score > best_score
                
                if (not is_tabu or aspiration_criterion) and neighbor_score > best_neighbor_score:
                    best_neighbor_score = neighbor_score
                    best_neighbor = neighbor_routes
                    best_move = (sel_route, unsel_route) # (rimosso, aggiunto)
            
            # Aggiorna soluzione
            if best_neighbor is None:
                break # Nessun vicino valido trovato
            
            current_routes = best_neighbor
            current_score = best_neighbor_score
            
            # Aggiorna Best Solution
            if current_score > best_score:
                best_score = current_score
                best_routes = list(current_routes)
            
            # Aggiorna Lista Tabu (proibisce l'inversione della mossa)
            if best_move is not None:
                removed_route, added_route = best_move
            
                # Proibisce di riaggiungere la rotta rimossa: rende ADD tabù
                tabu_list[removed_route] = iteration + tabu_list_size
                
                # Proibisce di rimuovere la rotta aggiunta: rende REMOVE tabù
                tabu_list[added_route] = iteration + tabu_list_size
            
            
        end_time = time.time()
        total_score, details = self._evaluate_solution(best_routes)
        
        print(f"Punteggio: {total_score}")
        print(f"Tempo: {end_time - start_time:.3f}s\n")
        
        return Solution(
            routes=best_routes,
            route_points=details['route_points'],
            longest_path_length=details['longest'],
            longest_bonus=details['longest_bonus'],
            ticket_points=details['ticket_points'],
            total_score=total_score,
            trains_used=trains_used(best_routes),
            computation_time=end_time - start_time,
            algorithm="Tabu Search"
        )

    # === TICKET-CENTRIC GENETIC ALGORITHM IMPLEMENTATION (Specialized) ===
    
    def _evaluate_weighted_fitness(self, routes: List[Tuple[str, str, int, str]]) -> float:
        """
        Calcola la fitness ponderata per privilegiare i bonus.
        
        Utilizza pesi maggiori per ticket_points e longest_bonus per guidare
        l'evoluzione verso soluzioni con migliore connettività.
        
        Args:
            routes: Lista di rotte da valutare
            
        Returns:
            Fitness ponderata (valore float)
        """
        route_points = calculate_route_points(routes)
        longest = calculate_longest_path(routes)
        longest_bonus = LONGEST_PATH_BONUS if longest > 0 else 0
        ticket_points = calculate_ticket_points(routes, self.tickets_df)
        
        # Ponderazione per i bonus
        weighted_fitness = (1.0 * route_points) + \
                           (1.5 * ticket_points) + \
                           (1.5 * longest_bonus)
        return weighted_fitness
    
    def _targeted_mutation(self, individual: List[Tuple[str, str, int, str]]) -> List[Tuple[str, str, int, str]]:
        """Mutazione mirata per estendere il percorso più lungo (prioritizza il bonus +10)."""
        
        # 1. Trova le rotte non usate
        pair_to_route = {tuple(sorted([r[0], r[1]])): r for r in self.unique_routes}
        selected_pairs = {tuple(sorted([r[0], r[1]])) for r in individual}
        unselected_routes = [r for r in self.unique_routes if tuple(sorted([r[0], r[1]])) not in selected_pairs]
        
        # 2. Identifica i nodi finali del percorso più lungo corrente (richiede logica complessa, semplifichiamo)
        # Per semplicità: proviamo ad aggiungere una rotta di lunghezza 1 o 2 che si connette
        
        if not individual:
            return self._mutate(individual) # Usa mutazione base se vuoto
        
        current_trains = trains_used(individual)
        
        # Ordina le rotte non usate per lunghezza (priorità alle corte per estendere il percorso)
        unselected_routes.sort(key=lambda r: r[2])
        
        for route in unselected_routes:
            city1, city2, length, _ = route
            
            # Se la rotta è corta (lunghezza 1 o 2) e abbiamo spazio
            if length <= 2 and current_trains + length <= self.total_trains:
                
                # Check 1: Connette due nodi già presenti (non crea isolati)
                existing_nodes = {r[0] for r in individual} | {r[1] for r in individual}
                
                # Cerca se la nuova rotta collega un nodo esistente (probabile estensione)
                if city1 in existing_nodes or city2 in existing_nodes:
                    mutated = list(individual) + [route]
                    
                    # Rimuovi una rotta a caso se i vagoni sono troppi (non dovrebbe succedere se il check è corretto)
                    if trains_used(mutated) > self.total_trains:
                        if mutated: mutated.pop(random.randrange(len(mutated)-1))
                    
                    # Ritorna la mutazione mirata
                    return mutated
        
        # Se la mutazione mirata fallisce, esegui una mutazione base
        return self._mutate(individual)

    def ticket_centric_genetic_algorithm_solve(self, population_size: int = 50,
                               generations: int = 100,
                               mutation_rate: float = 0.1) -> Solution:
        """
        Genetic Algorithm (Ticket-Centric): evolve una popolazione privilegiando
        i punti bonus (connettività) tramite fitness ponderata e mutazione mirata.
        """
        print("=== TICKET-CENTRIC GENETIC ALGORITHM SOLVER ===")
        print(f"Population: {population_size}, Generations: {generations}")
        start_time = time.time()
        
        # Inizializza popolazione casuale
        population = self._initialize_population(population_size)
        
        best_individual = max(population, key=lambda ind: self._evaluate_solution(ind)[0])
        best_score, _ = self._evaluate_solution(best_individual)
        
        for gen in range(generations):
            # Valuta fitness PONDERATA per la selezione
            fitness_scores = [(ind, self._evaluate_weighted_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Aggiorna best in base al punteggio REALE (non ponderato)
            real_score = self._evaluate_solution(fitness_scores[0][0])[0]
            if real_score > best_score:
                best_score = real_score
                best_individual = list(fitness_scores[0][0])
            
            # Selezione (top 50% in base alla fitness ponderata)
            selected = [ind for ind, _ in fitness_scores[:population_size // 2]]
            
            # Genera nuova popolazione
            new_population = list(selected)
            
            while len(new_population) < population_size:
                # Crossover
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                child = self._crossover(parent1, parent2) # Usa il crossover base per semplicità di implementazione
                
                # Mutazione
                if random.random() < mutation_rate:
                    # Usa la mutazione mirata con una probabilità
                    if random.random() < 0.5:
                        child = self._targeted_mutation(child)
                    else:
                        child = self._mutate(child) # Mutazione base di backup
                
                new_population.append(child)
            
            population = new_population
        
        end_time = time.time()
        total_score, details = self._evaluate_solution(best_individual)
        
        print(f"Generazioni: {generations}")
        print(f"Punteggio: {total_score}")
        print(f"Tempo: {end_time - start_time:.3f}s\n")
        
        return Solution(
            routes=best_individual,
            route_points=details['route_points'],
            longest_path_length=details['longest'],
            longest_bonus=details['longest_bonus'],
            ticket_points=details['ticket_points'],
            total_score=total_score,
            trains_used=trains_used(best_individual),
            computation_time=end_time - start_time,
            algorithm="Ticket-Centric Genetic Algorithm"
        )


def compare_all_heuristics(graph: nx.MultiGraph, tickets_file: str) -> Dict:
    """
    Esegue tutti gli algoritmi euristici e confronta i risultati.
    """
    solver = HeuristicSolver(graph, tickets_file)
    
    results = {}
    
    # 1. Greedy
    results['greedy'] = solver.greedy_solve()
    
    # 2. Local Search
    results['local_search'] = solver.local_search_solve(max_iterations=1000)
    
    # 3. Simulated Annealing
    results['simulated_annealing'] = solver.simulated_annealing_solve(
        initial_temp=100.0,
        cooling_rate=0.95,
        max_iterations=2000
    )
    
    # 4. Genetic Algorithm
    results['genetic'] = solver.genetic_algorithm_solve(
        population_size=50,
        generations=100,
        mutation_rate=0.1
    )

    results['ticket_centric_genetic'] = solver.ticket_centric_genetic_algorithm_solve(
        population_size=50,
        generations=100,
        mutation_rate=0.1
    )
    
    # Trova il migliore
    best_algo = max(results.items(), key=lambda x: x[1].total_score)
    
    print("\n" + "="*60)
    print("CONFRONTO ALGORITMI EURISTICI")
    print("="*60)
    for name, solution in results.items():
        print(f"\n{solution.algorithm}:")
        print(f"  Punteggio totale: {solution.total_score}")
        print(f"  Tempo: {solution.computation_time:.3f}s")
        print(f"  Rotte: {len(solution.routes)}")
        print(f"  Treni usati: {solution.trains_used}/45")
    
    print(f"\n{'='*60}")
    print(f"MIGLIOR ALGORITMO: {best_algo[0].upper()}")
    print(f"Punteggio: {best_algo[1].total_score}")
    print(f"{'='*60}\n")
    
    return results