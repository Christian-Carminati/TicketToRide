import networkx as nx
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
import pandas as pd


class MaxScoreCalculator:
    
    def __init__(self, graph: nx.MultiGraph) -> None:
        self.graph = graph
        self.routes = self._extract_routes()
        self.points_table = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
    
    def _extract_routes(self) -> List[Tuple[str, str, int, str]]:
        routes = []
        
        for u, v, data in self.graph.edges(data=True):
            length = data.get('weight', 1)
            color = data.get('color', 'X')
            routes.append((u, v, length, color))
        
        return routes
    
    def calculate_total_combinations(self) -> int:
        unique_pairs = set()
        for city1, city2, _, _ in self.routes:
            pair = tuple(sorted([city1, city2]))
            unique_pairs.add(pair)
        
        n = len(unique_pairs)
        return 2 ** n
    
    def calculate_max_route_points_greedy(self) -> Tuple[int, List[Tuple[str, str, int, str]]]:
        sorted_routes = sorted(self.routes, 
                              key=lambda r: self.points_table.get(r[2], 0) / r[2], 
                              reverse=True)
        
        total_trains = 45
        trains_used = 0
        total_score = 0
        selected_routes = []
        claimed_pairs = set()
        
        for route in sorted_routes:
            city1, city2, length, color = route
            pair = tuple(sorted([city1, city2]))
            
            if trains_used + length <= total_trains and pair not in claimed_pairs:
                trains_used += length
                total_score += self.points_table.get(length, 0)
                selected_routes.append((city1, city2, length, color))
                claimed_pairs.add(pair)
        
        return total_score, selected_routes
    
    def calculate_max_route_points_dp(self) -> Tuple[int, List[Tuple[str, str, int, str]]]:
        unique_routes = {}
        for city1, city2, length, color in self.routes:
            pair = tuple(sorted([city1, city2]))
            if pair not in unique_routes:
                unique_routes[pair] = (city1, city2, length, color)
            elif length > unique_routes[pair][2]:
                unique_routes[pair] = (city1, city2, length, color)
        
        routes_list = list(unique_routes.values())
        n = len(routes_list)
        max_trains = 45
        
        dp = [[0 for _ in range(max_trains + 1)] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            city1, city2, length, color = routes_list[i - 1]
            points = self.points_table.get(length, 0)
            
            for trains in range(max_trains + 1):
                dp[i][trains] = dp[i - 1][trains]
                
                if trains >= length:
                    dp[i][trains] = max(dp[i][trains], 
                                       dp[i - 1][trains - length] + points)
        
        max_score = dp[n][max_trains]
        
        selected_routes = []
        trains = max_trains
        for i in range(n, 0, -1):
            if dp[i][trains] != dp[i - 1][trains]:
                city1, city2, length, color = routes_list[i - 1]
                selected_routes.append((city1, city2, length, color))
                trains -= length
        
        return max_score, selected_routes
    
    def calculate_max_longest_path_bonus(self, claimed_routes: List[Tuple[str, str, int, str]]) -> int:
        if not claimed_routes:
            return 0
        
        temp_graph = nx.Graph()
        for city1, city2, length, _ in claimed_routes:
            temp_graph.add_edge(city1, city2, weight=length)
        
        max_length = 0
        for node in temp_graph.nodes():
            visited_edges = set()
            length = self._dfs_longest_path(temp_graph, node, visited_edges, 0)
            max_length = max(max_length, length)
        
        return max_length
    
    def _dfs_longest_path(
        self, 
        graph: nx.Graph, 
        node: str, 
        visitedEdges: Set[Tuple[str, str]], 
        current_length: int
    ) -> int:
        """
        Perform a depth-first search to find the longest path in the graph.

        :param graph: The graph to search.
        :type graph: nx.Graph
        :param node: The current node to search from.
        :type node: str
        :param visitedEdges: A set of edges that have already been visited.
        :type visitedEdges: Set[Tuple[str, str]]
        :param currentLength: The current length of the path.
        :type currentLength: int
        :return: The longest path found.
        :rtype: int
        """
        max_length = current_length

        for neighbor in graph.neighbors(node):
            edge: Tuple[str, str] = tuple(sorted([node, neighbor])) # type: ignore
            if edge not in visitedEdges:
                visitedEdges.add(edge)
                edge_length = graph[node][neighbor]['weight']
                length = self._dfs_longest_path(
                    graph, neighbor, visitedEdges, 
                    current_length + edge_length
                )
                max_length = max(max_length, length)
                visitedEdges.remove(edge)

        return max_length
    
    def find_completable_tickets(self, claimed_routes: List[Tuple[str, str, int, str]], 
                                tickets_file: str) -> List[Tuple[str, str, int, bool]]:
        route_graph = nx.Graph()
        for city1, city2, _, _ in claimed_routes:
            route_graph.add_edge(city1, city2)
        
        df = pd.read_csv(tickets_file)
        completable_tickets = []
        
        for _, row in df.iterrows():
            city1 = row['From']
            city2 = row['To']
            points = row['Points']
            
            if city1 in route_graph and city2 in route_graph:
                is_completable = nx.has_path(route_graph, city1, city2)
                completable_tickets.append((city1, city2, points, is_completable))
        
        return completable_tickets
    
    def visualize_optimal_routes(self, optimal_routes: List[Tuple[str, str, int, str]], 
                                figsize: Tuple[int, int] = (20, 12)) -> None:
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        plt.figure(figsize=figsize)
        
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color='lightgray',
            node_size=300,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.6
        )
        
        nx.draw_networkx_labels(self.graph, pos, font_size=7, font_weight='bold')
        
        for u, v, data in self.graph.edges(data=True):
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=[(u, v)],
                edge_color='lightgray',
                width=1,
                alpha=0.3
            )
        
        color_map = {
            'R': 'red', 'B': 'blue', 'G': 'green', 'Y': 'yellow',
            'O': 'orange', 'K': 'black', 'W': 'silver', 'P': 'purple',
            'X': 'grey'
        }
        
        for city1, city2, length, color in optimal_routes:
            color_code = str(color).upper()[0]
            edge_color = color_map.get(color_code, 'red')
            
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=[(city1, city2)],
                edge_color=edge_color,
                width=4,
                alpha=0.8
            )
        
        highlighted_cities = set()
        for city1, city2, _, _ in optimal_routes:
            highlighted_cities.add(city1)
            highlighted_cities.add(city2)
        
        highlighted_pos = {city: pos[city] for city in highlighted_cities if city in pos}
        nx.draw_networkx_nodes(
            self.graph, highlighted_pos,
            nodelist=list(highlighted_cities),
            node_color='yellow',
            node_size=500,
            edgecolors='red',
            linewidths=2
        )
        
        plt.title("STRATEGIA OTTIMALE - Rotte Selezionate per Massimizzare il Punteggio", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def find_optimal_strategy(self, tickets_file: str) -> Dict:
        route_points_greedy, best_routes_greedy = self.calculate_max_route_points_greedy()
        route_points_dp, best_routes_dp = self.calculate_max_route_points_dp()
        
        longest_path_dp = self.calculate_max_longest_path_bonus(best_routes_dp)
        longest_path_bonus = 10 if longest_path_dp > 0 else 0
        
        completable_tickets = self.find_completable_tickets(best_routes_dp, tickets_file)
        completed_tickets = [t for t in completable_tickets if t[3]]
        completed_tickets_sorted = sorted(completed_tickets, key=lambda x: x[2], reverse=True)
        
        top_tickets = completed_tickets_sorted[:10]
        realistic_dest_points = sum(t[2] for t in top_tickets)
        
        route_distribution = {}
        for _, _, length, _ in best_routes_dp:
            route_distribution[length] = route_distribution.get(length, 0) + 1
        
        total_combinations = self.calculate_total_combinations()
        
        return {
            'greedy': {
                'route_points': route_points_greedy,
                'num_routes': len(best_routes_greedy),
                'trains_used': sum(r[2] for r in best_routes_greedy),
                'routes': best_routes_greedy
            },
            'dp': {
                'route_points': route_points_dp,
                'longest_path_bonus': longest_path_bonus,
                'realistic_destination_points': realistic_dest_points,
                'total_realistic_max': route_points_dp + longest_path_bonus + realistic_dest_points,
                'trains_used': sum(r[2] for r in best_routes_dp),
                'num_routes': len(best_routes_dp),
                'longest_path_length': longest_path_dp,
                'routes': best_routes_dp
            },
            'route_distribution': route_distribution,
            'completable_tickets': completable_tickets,
            'top_tickets': top_tickets,
            'total_combinations': total_combinations
        }


def main() -> None:
    from map import TicketToRideMap
    
    print("=== CALCOLO PUNTEGGIO MASSIMO - TICKET TO RIDE ===\n")
    
    ttr_map = TicketToRideMap()
    ttr_map.load_graph("src/map/city_locations.json", "src/map/routes.csv")
    graph = ttr_map.get_graph()
    
    calculator = MaxScoreCalculator(graph)
    
    print("Calcolando strategia ottimale...")
    strategy = calculator.find_optimal_strategy("src/map/tickets.csv")
    
    print("\n" + "="*80)
    print("SPIEGAZIONE ALGORITMI")
    print("="*80)
    
    print(f"\nCOMBINAZIONI TOTALI POSSIBILI: {strategy['total_combinations']:,}")
    print(f"(2^n dove n = numero di coppie di città uniche)")
    
    print("\n1. ALGORITMO GREEDY (Goloso):")
    print("   - Ordina tutte le rotte per efficienza (punti/vagone)")
    print("   - Seleziona la rotta più efficiente disponibile")
    print("   - Ripete finché non finiscono i vagoni (45)")
    print("   - VANTAGGI: Velocissimo O(n log n)")
    print("   - SVANTAGGI: Non garantisce la soluzione ottimale assoluta")
    
    print("\n2. PROGRAMMAZIONE DINAMICA (Knapsack Problem):")
    print("   - Problema dello zaino: dato uno zaino con capacità limitata (45 vagoni)")
    print("   - e oggetti con peso (lunghezza rotta) e valore (punti)")
    print("   - trova la combinazione che massimizza il valore totale")
    print("   - Costruisce una tabella dp[rotta][vagoni] con il massimo punteggio ottenibile")
    print("   - VANTAGGI: Soluzione ottimale garantita")
    print("   - COMPLESSITÀ: O(n * m) dove n=rotte, m=45 vagoni")
    
    print("\n" + "="*80)
    print("RISULTATI GREEDY")
    print("="*80)
    print(f"Punti da rotte: {strategy['greedy']['route_points']}")
    print(f"Numero rotte: {strategy['greedy']['num_routes']}")
    print(f"Vagoni usati: {strategy['greedy']['trains_used']}/45")
    if strategy['greedy']['trains_used'] > 0:
        print(f"Efficienza: {strategy['greedy']['route_points'] / strategy['greedy']['trains_used']:.3f} punti/vagone")
    
    print("\n" + "="*80)
    print("RISULTATI PROGRAMMAZIONE DINAMICA (OTTIMALE)")
    print("="*80)
    print(f"Punti da rotte: {strategy['dp']['route_points']}")
    print(f"Bonus percorso più lungo: {strategy['dp']['longest_path_bonus']}")
    print(f"Punti destinazione (top 10): {strategy['dp']['realistic_destination_points']}")
    print(f"TOTALE REALISTICO: {strategy['dp']['total_realistic_max']}")
    print(f"Vagoni usati: {strategy['dp']['trains_used']}/45")
    print(f"Numero rotte: {strategy['dp']['num_routes']}")
    print(f"Lunghezza percorso più lungo: {strategy['dp']['longest_path_length']}")
    if strategy['dp']['trains_used'] > 0:
        print(f"Efficienza: {strategy['dp']['route_points'] / strategy['dp']['trains_used']:.3f} punti/vagone")
    
    print("\n" + "="*80)
    print("DISTRIBUZIONE LUNGHEZZA ROTTE OTTIMALE")
    print("="*80)
    for length in sorted(strategy['route_distribution'].keys(), reverse=True):
        count = strategy['route_distribution'][length]
        points = calculator.points_table.get(length, 0)
        efficiency = points / length
        print(f"Lunghezza {length}: {count} rotte ({points} punti, efficienza {efficiency:.2f})")
    
    print("\n" + "="*80)
    print("TUTTE LE ROTTE SELEZIONATE (DP)")
    print("="*80)
    for i, (city1, city2, length, color) in enumerate(strategy['dp']['routes'], 1):
        points = calculator.points_table.get(length, 0)
        print(f"{i:2d}. {city1:20s} -> {city2:20s} | Lunghezza: {length} | Punti: {points:2d} | Colore: {color}")
    
    print("\n" + "="*80)
    print("DESTINATION TICKETS COMPLETABILI")
    print("="*80)
    completable = [t for t in strategy['completable_tickets'] if t[3]]
    not_completable = [t for t in strategy['completable_tickets'] if not t[3]]
    
    print(f"\nCOMPLETABILI ({len(completable)}):")
    for city1, city2, points, _ in sorted(completable, key=lambda x: x[2], reverse=True):
        print(f"  {city1:20s} -> {city2:20s} | {points:2d} punti")
    
    print(f"\nNON COMPLETABILI ({len(not_completable)}):")
    for city1, city2, points, _ in sorted(not_completable, key=lambda x: x[2], reverse=True)[:10]:
        print(f"  {city1:20s} -> {city2:20s} | {points:2d} punti")
    
    print("\n" + "="*80)
    print("TOP 10 DESTINATION TICKETS DA TENERE")
    print("="*80)
    for i, (city1, city2, points, _) in enumerate(strategy['top_tickets'], 1):
        print(f"{i:2d}. {city1:20s} -> {city2:20s} | {points:2d} punti")
    
    print("\n" + "="*80)
    print("STRATEGIA OTTIMALE CONSIGLIATA")
    print("="*80)
    print("1. Dare priorità alle rotte di lunghezza 6 e 5 (massima efficienza)")
    print("2. Costruire un percorso continuo per ottenere il bonus da 10 punti")
    print("3. Selezionare i destination tickets mostrati sopra (già completabili)")
    print("4. Giocare le rotte nell'ordine mostrato per massimizzare i punti")
    
    print("\nVisualizzando la mappa con le rotte ottimali...")
    calculator.visualize_optimal_routes(strategy['dp']['routes'])


if __name__ == "__main__":
    main()