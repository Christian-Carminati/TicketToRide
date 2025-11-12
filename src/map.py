"""Map loading and visualization utilities for Ticket to Ride."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from core.graph import load_ticket_to_ride_graph


class TicketToRideMap:
    # Configurazione colori come costante di classe
    COLOR_MAP = {
        'R': 'red', 'B': 'blue', 'G': 'green', 'Y': 'yellow',
        'O': 'orange', 'K': 'black', 'W': 'silver', 'P': 'purple',
        'X': 'grey'
    }
    
    # Configurazione curve per archi multipli
    CURVE_RADII = [0.15, -0.15, 0.25, -0.25]
    
    def __init__(self) -> None:
        """Inizializza TicketToRideMap"""
        self.graph = nx.MultiGraph()
        
    def load_graph(self, city_locations_file: str | Path, routes_file: str | Path) -> None:
        """
        Carica città e rotte nel grafo.
        
        Utilizza la funzione condivisa da core.graph per evitare duplicazione di codice.
        
        Args:
            city_locations_file: Path del file contenente le coordinate delle città (JSON)
            routes_file: Path del file contenente le rotte (CSV)
        """
        self.graph = load_ticket_to_ride_graph(city_locations_file, routes_file)
    
    def get_graph(self) -> nx.MultiGraph:
        """Restituisce il grafo della mappa.
        
        :return: Il grafo della mappa di Ticket to Ride.
        :rtype: nx.MultiGraph
        """
        return self.graph
    
    def draw_graph(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Disegna il grafo della mappa.
        
        :param figsize: Dimensioni della finestra.
        """
        if self.graph.number_of_nodes() == 0:
            raise ValueError("Il grafo è vuoto. Carica prima i dati con load_graph().")
        
        pos = nx.get_node_attributes(self.graph, 'pos')
        plt.figure(figsize=figsize)
        
        # Disegna nodi
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color='lightblue',
            node_size=500,
            edgecolors='black',
            linewidths=1
        )
        nx.draw_networkx_labels(self.graph, pos, font_size=9)
        
        # Disegna archi
        edge_counter: Dict[Tuple[str, str], int] = {}
        drawn_labels = set()
        
        for u, v, data in self.graph.edges(data=True):
            edge_key = tuple(sorted((u, v)))
            current_index = edge_counter.get(edge_key, 0)
            edge_counter[edge_key] = current_index + 1
            
            # Calcola curvatura per archi multipli
            radius = 0.0
            if self.graph.number_of_edges(u, v) > 1:
                radius = self.CURVE_RADII[current_index] if current_index < len(self.CURVE_RADII) else 0.0
            
            # Ottieni colore
            color_code = str(data.get('color', 'X')).upper()[0]
            edge_color = self.COLOR_MAP.get(color_code, 'grey')
            
            # Disegna arco
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=[(u, v)],
                edge_color=[edge_color], # type: ignore
                width=2,
                connectionstyle=f"arc3,rad={radius}"
            )
            
            # Disegna etichetta peso solo una volta per coppia di città
            if edge_key not in drawn_labels:
                nx.draw_networkx_edge_labels(
                    self.graph, pos,
                    edge_labels={(u, v): data.get('weight', '')},
                    font_color='black',
                    font_size=8,
                    label_pos=0.5
                )
                drawn_labels.add(edge_key)
        
        plt.title("Mappa di Ticket to ride (USA) - Rappresentazione Geografica")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def main() -> None:
    ttr_map = TicketToRideMap()
  
    ttr_map.load_graph("src/map/city_locations.json", "src/map/routes.csv")
    
    
    print("Numero di nodi:", len(ttr_map.get_graph().nodes))
    print("Numero di archi:", len(ttr_map.get_graph().edges))
    print("Nodi del grafo:", list(ttr_map.get_graph().nodes))
    print("Archi del grafo:", list(ttr_map.get_graph().edges(data=True)))
    ttr_map.draw_graph()
    


if __name__ == "__main__":
    main()