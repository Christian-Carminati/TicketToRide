"""
Demo script migliorato per visualizzare due agenti che giocano a Ticket to Ride.
Include correzioni dei bug e visualizzazione grafica rallentata.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt

from game import Game, Color
from map import TicketToRideMap
from visualization.game_viewer import GameViewer


TIME_DELAY = 0.5


# ============================================================================
# AGENTI
# ============================================================================

def random_agent(game: Game, player_index: int) -> dict:
    """
    Agente casuale che pesca carte o reclama rotte a caso.
    
    Args:
        game: Istanza del gioco
        player_index: Indice del giocatore (0 o 1)
        
    Returns:
        Dizionario con l'azione da eseguire
    """
    player = game.players[player_index]
    available_routes = game.route_manager.get_available_routes(game.number_of_players)
    
    # 60% probabilità di provare a reclamare una rotta, 40% di pescare carte
    if random.random() < 0.6 and available_routes:
        # Trova rotte reclamabili
        claimable = []
        for route in available_routes:
            if route.color == Color.GREY:
                for color in [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, 
                             Color.ORANGE, Color.BLACK, Color.SILVER, Color.PURPLE]:
                    if player.hand.can_claim_route(route, color):
                        claimable.append((route, color))
                        break
            else:
                if player.hand.can_claim_route(route, route.color):
                    claimable.append((route, route.color))
        
        if claimable:
            route, color_to_use = random.choice(claimable)
            return {
                'type': 'claim_route',
                'city1': route.city1,
                'city2': route.city2,
                'color': route.color,
                'color_to_use': color_to_use,
                'route_length': route.length
            }
    
    # Pesca carte
    return {
        'type': 'draw_cards',
        'face_up_index': random.choice([None, None, 0, 1, 2, 3, 4])  # Preferisci deck
    }


def greedy_agent(game: Game, player_index: int) -> dict:
    """
    Agente greedy che reclama sempre la rotta più lunga possibile.
    
    Args:
        game: Istanza del gioco
        player_index: Indice del giocatore
        
    Returns:
        Dizionario con l'azione da eseguire
    """
    player = game.players[player_index]
    available_routes = game.route_manager.get_available_routes(game.number_of_players)
    
    if not available_routes:
        return {'type': 'draw_cards', 'face_up_index': None}
    
    # Trova tutte le rotte reclamabili
    claimable = []
    for route in available_routes:
        if route.color == Color.GREY:
            for color in [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, 
                         Color.ORANGE, Color.BLACK, Color.SILVER, Color.PURPLE]:
                if player.hand.can_claim_route(route, color):
                    claimable.append((route, color))
                    break
        else:
            if player.hand.can_claim_route(route, route.color):
                claimable.append((route, route.color))
    
    if claimable:
        # Scegli la rotta più lunga (strategia greedy)
        best_route, best_color = max(claimable, key=lambda x: x[0].length)
        return {
            'type': 'claim_route',
            'city1': best_route.city1,
            'city2': best_route.city2,
            'color': best_route.color,
            'color_to_use': best_color,
            'route_length': best_route.length
        }
    
    # Non può reclamare rotte, pesca carte
    # Preferisci carte scoperte che già abbiamo
    face_up_cards = game.game_state.face_up_cards
    for i, card in enumerate(face_up_cards):
        if card != Color.JOLLY and player.hand.cards.get(card, 0) > 0:
            return {'type': 'draw_cards', 'face_up_index': i}
    
    return {'type': 'draw_cards', 'face_up_index': None}


def smart_agent(game: Game, player_index: int) -> dict:
    """
    Agente più intelligente che bilancia lunghezza e efficienza.
    
    Args:
        game: Istanza del gioco
        player_index: Indice del giocatore
        
    Returns:
        Dizionario con l'azione da eseguire
    """
    player = game.players[player_index]
    available_routes = game.route_manager.get_available_routes(game.number_of_players)
    
    if not available_routes:
        return {'type': 'draw_cards', 'face_up_index': None}
    
    # Trova rotte reclamabili e calcola il loro valore
    claimable = []
    points_table = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
    
    for route in available_routes:
        if route.color == Color.GREY:
            for color in [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, 
                         Color.ORANGE, Color.BLACK, Color.SILVER, Color.PURPLE]:
                if player.hand.can_claim_route(route, color):
                    # Calcola efficienza: punti per vagone
                    efficiency = points_table[route.length] / route.length
                    claimable.append((route, color, efficiency))
                    break
        else:
            if player.hand.can_claim_route(route, route.color):
                efficiency = points_table[route.length] / route.length
                claimable.append((route, route.color, efficiency))
    
    if claimable:
        # Scegli basandoti su efficienza, ma con preferenza per rotte lunghe
        best_route, best_color, _ = max(
            claimable, 
            key=lambda x: (x[2], x[0].length)  # Prima efficienza, poi lunghezza
        )
        return {
            'type': 'claim_route',
            'city1': best_route.city1,
            'city2': best_route.city2,
            'color': best_route.color,
            'color_to_use': best_color,
            'route_length': best_route.length
        }
    
    # Pesca carte intelligentemente
    face_up_cards = game.game_state.face_up_cards
    
    # Preferisci jolly se disponibili
    for i, card in enumerate(face_up_cards):
        if card == Color.JOLLY:
            return {'type': 'draw_cards', 'face_up_index': i}
    
    # Altrimenti, preferisci carte che già abbiamo
    for i, card in enumerate(face_up_cards):
        if player.hand.cards.get(card, 0) > 0:
            return {'type': 'draw_cards', 'face_up_index': i}
    
    return {'type': 'draw_cards', 'face_up_index': None}


# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    """Esegui una partita demo tra due agenti con visualizzazione."""
    print("="*70)
    print("Ticket to Ride - Agent Visualization Demo")
    print("="*70)
    
    # Carica mappa
    print("\nCaricamento mappa...")
    ttr_map = TicketToRideMap()
    ttr_map.load_graph("map/city_locations.json", "map/routes.csv")
    graph = ttr_map.get_graph()
    print("✓ Mappa caricata")
    
    # Crea gioco
    print("\nImpostazione gioco...")
    game = Game(number_of_players=2, graph=graph)
    game.setup()
    print("✓ Gioco pronto")
    
    # Crea visualizzatore
    print("\nInizializzazione visualizzazione...")
    viewer = GameViewer(game, figsize=(20, 12))
    viewer.setup_figure()
    viewer.update()
    plt.ion()  # Modalità interattiva
    print("✓ Visualizzazione pronta")
    
    # Configurazione agenti
    agent1_name = "Smart"
    agent2_name = "Greedy"
    agent1_fn = smart_agent
    agent2_fn = greedy_agent
    
    print(f"\n{'='*70}")
    print(f"Partita: {agent1_name} (Player 1 - {Color.RED.value.upper()}) vs "
          f"{agent2_name} (Player 2 - {Color.BLUE.value.upper()})")
    print(f"{'='*70}")
    print("\nChiudi la finestra per terminare la visualizzazione")
    print("Premi Ctrl+C nel terminale per fermare la partita\n")
    
    # Loop di gioco
    max_turns = 100
    turn_count = 0
    turn_delay = TIME_DELAY  # Secondi tra i turni (rallentato)
    
    try:
        while not game.is_game_over and turn_count < max_turns:
            current_player = game.current_player()
            player_idx = game.current_player_index
            
            # Nome agente corrente
            agent_name = agent1_name if player_idx == 0 else agent2_name
            
            print(f"\n{'─'*70}")
            print(f"Turno {turn_count + 1} - {agent_name} "
                  f"({current_player.color.value.upper()})")
            print(f"  Carte: {sum(current_player.hand.cards.values())}, "
                  f"Vagoni: {current_player.trains_left}, "
                  f"Punteggio: {current_player.score}")
            
            # Ottieni azione dall'agente
            if player_idx == 0:
                action = agent1_fn(game, player_idx)
            else:
                action = agent2_fn(game, player_idx)
            
            # Esegui azione
            action_success = False
            if action['type'] == 'claim_route':
                success = game.claim_route_action(
                    action['city1'],
                    action['city2'],
                    action['color'],
                    action['color_to_use']
                )
                if success:
                    print(f"  → Reclama: {action['city1']} ↔ {action['city2']} "
                          f"(lunghezza {action['route_length']}, "
                          f"colore {action['color_to_use'].value})")
                    action_success = True
                else:
                    print(f"  → Tentativo fallito, pesca carte")
                    game.draw_card_action(None)
            else:
                game.draw_card_action(action['face_up_index'])
                if action['face_up_index'] is not None:
                    print(f"  → Pesca carta scoperta (pos {action['face_up_index']})")
                else:
                    print(f"  → Pesca dal mazzo")
            
            # Aggiorna visualizzazione
            viewer.update()
            plt.pause(turn_delay)
            
            turn_count += 1
            
            # Controlla se la finestra è stata chiusa
            if not plt.fignum_exists(viewer.fig.number):
                print("\n✗ Finestra chiusa, interruzione partita")
                break
        
        # Stato finale
        print(f"\n{'='*70}")
        if game.is_game_over:
            print(f"FINE PARTITA - {turn_count} turni giocati")
        else:
            print(f"PARTITA INTERROTTA - {turn_count} turni giocati")
        print(f"{'='*70}")
        
        # Calcola vincitore
        winner, scores = game.calculate_winner()
        
        print(f"\nPunteggi Finali:")
        print(f"{'='*70}")
        for i, player in enumerate(game.players):
            agent_type = agent1_name if i == 0 else agent2_name
            is_winner = "👑 VINCITORE" if player == winner else ""
            print(f"\nPlayer {i+1} ({player.color.value.upper()}) - {agent_type} {is_winner}")
            print(f"  Punteggio finale: {scores[player.color]} punti")
            print(f"  Rotte reclamate: {len(player.routes)}")
            print(f"  Vagoni usati: {45 - player.trains_left}")
            print(f"  Carte rimanenti: {sum(player.hand.cards.values())}")
            
            # Mostra biglietti completati
            completed = sum(1 for ticket in player.destination_tickets 
                           if ticket.is_completed(player.routes))
            print(f"  Biglietti destinazione: {completed}/{len(player.destination_tickets)} completati")
        
        print(f"\n{'='*70}")
        margin = scores[winner.color] - min(scores.values())
        print(f"Vincitore: Player {game.players.index(winner) + 1} "
              f"({winner.color.value.upper()}) - {agent1_name if game.players.index(winner) == 0 else agent2_name}")
        print(f"Margine di vittoria: {margin} punti")
        print(f"{'='*70}\n")
        
        # Aggiornamento finale
        viewer.update()
        
        # Mantieni la finestra aperta
        print("\nVisualizzazione finale - Chiudi la finestra per uscire")
        plt.ioff()  # Disattiva modalità interattiva
        viewer.show(block=True)
        
    except KeyboardInterrupt:
        print("\n\n✗ Partita interrotta dall'utente")
        viewer.close()
    except Exception as e:
        print(f"\n\n✗ Errore durante la partita: {e}")
        import traceback
        traceback.print_exc()
        viewer.close()


if __name__ == "__main__":
    main()