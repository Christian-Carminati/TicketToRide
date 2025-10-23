import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import json

def load_graph() -> nx.MultiGraph:
    G = nx.MultiGraph()
    
    try:
        
        with open('src/map/city_locations.json', 'r') as f:
            city_coords = json.load(f)
            
      
        df_routes = pd.read_csv('src/map/routes.csv')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Errore: {e}. Controlla che i file 'city_locations.json' e 'routes.csv' siano caricati.")
    
    for city, coords in city_coords.items():
        G.add_node(city, pos=(coords[0], coords[1]))
    
    # Aggiunge gli archi
    for _, row in df_routes.iterrows():
        # Assumiamo che il nome delle colonne siano: From, To, Distance, Color
        G.add_edge(row['From'], row['To'], weight=row['Distance'], color=row['Color'])
    
    return G

def draw_graph(G: nx.MultiGraph) -> None:
    # Mappatura dei colori (usa K, O, X, ecc. come codici comuni TTR)
    color_map = {'R': 'red', 'B': 'blue', 'G': 'green', 'Y': 'yellow',
                 'O': 'orange', 'K': 'black', 'W': 'silver', 'P': 'purple',
                 'X': 'grey'} 
    
    pos = nx.get_node_attributes(G, 'pos')
    
    plt.figure(figsize=(15, 10)) 
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=9)

    # Mappa dei raggi per curvare e separare gli archi multipli
    radius_map = {0: 0.15, 1: -0.15, 2: 0.25, 3: -0.25} 
    edge_counter = {}

    for u, v, k, data in G.edges(keys=True, data=True):
        # Normalizza l'ordine delle città per l'identificazione (per grafo non orientato)
        edge_key = tuple(sorted((u, v)))
        
        if edge_key not in edge_counter:
            edge_counter[edge_key] = 0
        else:
            edge_counter[edge_key] += 1
            
        current_index = edge_counter[edge_key]
        rad = 0.0
        
        # Applica una curvatura solo se ci sono archi multipli
        if G.number_of_edges(u, v) > 1:
             rad = radius_map.get(current_index, 0.0)

        # Prende il codice colore (prima lettera o X)
        ttr_color_code = str(data.get('color', 'X')).upper()
        mpl_color = color_map.get(ttr_color_code[0], 'grey')
        
        # Disegno dell'arco singolo
        nx.draw_networkx_edges(
            G, 
            pos, 
            edgelist=[(u, v)], # Disegna un solo arco alla volta
            edge_color=[mpl_color],  # type: ignore
            width=2,
            connectionstyle=f"arc3,rad={rad}" 
        )
        
        # Etichette del peso (opzionale: disegna solo per il primo segmento per evitare sovrapposizioni)
        if current_index == 0:
            nx.draw_networkx_edge_labels(
                G, 
                pos, 
                edge_labels={ (u,v): data.get('weight', '') }, 
                font_color='black', 
                label_pos=0.5
            )

    plt.title("Mappa di Ticket to Ride (USA) - Rappresentazione Geografica")
    plt.axis('off')
    plt.show()

def main() -> None:
    G: nx.MultiGraph = load_graph()
    print(f"Grafo caricato con {G.number_of_nodes()} città e {G.number_of_edges()} tratte.")
    draw_graph(G)

if __name__ == "__main__":
    main()