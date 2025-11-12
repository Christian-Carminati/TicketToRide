"""
Reinforcement Learning Solver per Ticket to Ride
Implementa Q-Learning, Deep Q-Network (DQN), e Policy Gradient
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set, Union
from dataclasses import dataclass
import time
import random
from collections import defaultdict, deque
import pickle


@dataclass
class Solution:
    """Rappresenta una soluzione completa del gioco"""
    routes: List[Tuple[str, str, int, str]]
    route_points: int
    longest_path_length: int
    longest_bonus: int
    ticket_points: int
    total_score: int
    trains_used: int
    computation_time: float
    algorithm: str = "RL"
    episodes: int = 0


class TicketToRideEnvironment:
    """
    Ambiente RL per Ticket to Ride
    Segue l'interfaccia standard: state, action, reward, next_state, done
    """
    
    def __init__(self, graph: nx.MultiGraph, tickets_file: str):
        self.graph = graph
        self.tickets_df = pd.read_csv(tickets_file)
        self.points_table = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
        self.total_trains = 45
        
        # Estrai rotte uniche
        self.unique_routes = self._build_unique_routes()
        self.route_to_idx = {route: idx for idx, route in enumerate(self.unique_routes)}
        self.idx_to_route = {idx: route for idx, route in enumerate(self.unique_routes)}
        
        # Stato iniziale
        self.reset()
    
    def _build_unique_routes(self) -> List[Tuple[str, str, int, str]]:
        """Costruisce lista di rotte uniche"""
        unique_routes = {}
        for u, v, data in self.graph.edges(data=True):
            length = data.get('weight', 1)
            color = data.get('color', 'X')
            pair = tuple(sorted([u, v]))
            if pair not in unique_routes or length > unique_routes[pair][2]:
                unique_routes[pair] = (u, v, length, color)
        return list(unique_routes.values())
    
    def reset(self) -> np.ndarray:
        """Resetta l'ambiente allo stato iniziale"""
        self.selected_routes = []
        self.selected_pairs = set()
        self.trains_remaining = self.total_trains
        self.done = False
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Rappresentazione dello stato come vettore numerico:
        - Route selection (binary per ogni rotta)
        - Trains remaining (normalized)
        - Current connectivity features
        """
        # Binary encoding: 1 se rotta selezionata, 0 altrimenti
        route_selection = np.zeros(len(self.unique_routes))
        for route in self.selected_routes:
            idx = self.route_to_idx[route]
            route_selection[idx] = 1
        
        # Features aggiuntivi
        trains_normalized = self.trains_remaining / self.total_trains
        num_selected = len(self.selected_routes)
        
        # Connectivity: numero di componenti connesse
        if self.selected_routes:
            temp_graph = nx.Graph()
            for city1, city2, _, _ in self.selected_routes:
                temp_graph.add_edge(city1, city2)
            num_components = nx.number_connected_components(temp_graph)
        else:
            num_components = 0
        
        # Combina features
        state = np.concatenate([
            route_selection,
            [trains_normalized, num_selected / len(self.unique_routes), num_components]
        ])
        
        return state
    
    def get_valid_actions(self) -> List[int]:
        """Restituisce lista di azioni valide (indici di rotte che possono essere aggiunte)"""
        valid_actions = []
        
        for idx, route in enumerate(self.unique_routes):
            city1, city2, length, _ = route
            pair = tuple(sorted([city1, city2]))
            
            # Azione valida se: rotta non già selezionata E abbiamo abbastanza treni
            if pair not in self.selected_pairs and self.trains_remaining >= length:
                valid_actions.append(idx)
        
        return valid_actions
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Esegue un'azione e restituisce (next_state, reward, done, info)
        
        action: indice della rotta da aggiungere
        """
        if self.done:
            raise ValueError("Episode già terminato, chiama reset()")
        
        route = self.idx_to_route[action]
        city1, city2, length, color = route
        pair = tuple(sorted([city1, city2]))
        
        # Verifica validità azione
        if pair in self.selected_pairs or self.trains_remaining < length:
            # Azione non valida: penalità
            reward = -10
            self.done = True
            return self._get_state(), reward, True, {'invalid_action': True}
        
        # Aggiungi rotta
        self.selected_routes.append(route)
        self.selected_pairs.add(pair)
        self.trains_remaining -= length
        
        # Calcola reward
        reward = self._calculate_reward(route)
        
        # Check se episodio è terminato
        valid_actions = self.get_valid_actions()
        if len(valid_actions) == 0 or self.trains_remaining < 1:
            self.done = True
            # Bonus finale per punteggio totale
            final_score = self._calculate_final_score()
            reward += final_score * 0.1  # Normalizza il bonus finale
        
        next_state = self._get_state()
        
        return next_state, reward, self.done, {}
    
    def _calculate_reward(self, route: Tuple[str, str, int, str]) -> float:
        """
        Calcola il reward immediato per aver aggiunto una rotta.
        Considera: punti rotta, efficienza, connettività, potenziale tickets.
        """
        city1, city2, length, color = route
        
        # 1. Punti base dalla rotta
        route_points = self.points_table.get(length, 0)
        reward = route_points
        
        # 2. Bonus efficienza (punti/vagone)
        efficiency = route_points / length
        reward += efficiency * 2  # Peso per efficienza
        
        # 3. Bonus connettività: se la rotta connette componenti esistenti
        if len(self.selected_routes) > 1:
            temp_graph_before = nx.Graph()
            for r in self.selected_routes[:-1]:  # Escludi la rotta appena aggiunta
                temp_graph_before.add_edge(r[0], r[1])
            
            temp_graph_after = nx.Graph()
            for r in self.selected_routes:  # Includi la rotta appena aggiunta
                temp_graph_after.add_edge(r[0], r[1])
            
            components_before = nx.number_connected_components(temp_graph_before)
            components_after = nx.number_connected_components(temp_graph_after)
            
            if components_after < components_before:
                reward += 5  # Bonus per unire componenti
        
        # 4. Potenziale tickets: controlla se la rotta aiuta a completare tickets
        ticket_potential = self._calculate_ticket_potential()
        reward += ticket_potential * 0.5
        
        return reward
    
    def _calculate_ticket_potential(self) -> float:
        """Stima quanti tickets potrebbero essere completati con le rotte correnti"""
        if not self.selected_routes:
            return 0
        
        route_graph = nx.Graph()
        for city1, city2, _, _ in self.selected_routes:
            route_graph.add_edge(city1, city2)
        
        potential = 0
        for _, row in self.tickets_df.iterrows():
            city1 = row['From']
            city2 = row['To']
            points = int(row['Points'])
            
            if city1 in route_graph and city2 in route_graph:
                if nx.has_path(route_graph, city1, city2):
                    potential += points
        
        return potential
    
    def _calculate_final_score(self) -> int:
        """Calcola il punteggio finale completo"""
        # Punti rotte
        route_points = sum(self.points_table.get(r[2], 0) for r in self.selected_routes)
        
        # Percorso più lungo
        longest = self._calculate_longest_path()
        longest_bonus = 10 if longest > 0 else 0
        
        # Tickets
        ticket_points = self._calculate_ticket_points()
        
        return route_points + longest_bonus + ticket_points
    
    def _calculate_longest_path(self) -> int:
        """Calcola lunghezza percorso continuo più lungo"""
        if not self.selected_routes:
            return 0
        
        temp_graph = nx.Graph()
        for city1, city2, length, _ in self.selected_routes:
            temp_graph.add_edge(city1, city2, weight=length)
        
        max_length = 0
        for node in temp_graph.nodes():
            visited_edges = set()
            length = self._dfs_longest_path(temp_graph, node, visited_edges, 0)
            max_length = max(max_length, length)
        
        return max_length
    
    def _dfs_longest_path(self, graph: nx.Graph, node: str,
                         visited_edges: Set[Tuple[str, str]],
                         current_length: int) -> int:
        max_length = current_length
        
        for neighbor in graph.neighbors(node):
            edge = tuple(sorted([node, neighbor]))
            if edge not in visited_edges:
                visited_edges.add(edge)
                edge_length = graph[node][neighbor]['weight']
                length = self._dfs_longest_path(
                    graph, neighbor, visited_edges,
                    current_length + edge_length
                )
                max_length = max(max_length, length)
                visited_edges.remove(edge)
        
        return max_length
    
    def _calculate_ticket_points(self) -> int:
        """Calcola punti dai tickets completabili"""
        if not self.selected_routes:
            return 0
        
        route_graph = nx.Graph()
        for city1, city2, _, _ in self.selected_routes:
            route_graph.add_edge(city1, city2)
        
        ticket_points = 0
        for _, row in self.tickets_df.iterrows():
            city1 = row['From']
            city2 = row['To']
            points = int(row['Points'])
            
            if city1 in route_graph and city2 in route_graph:
                if nx.has_path(route_graph, city1, city2):
                    ticket_points += points
        
        return ticket_points


class QLearningAgent:
    """
    Q-Learning Agent: Tabular RL con Q-table
    Adatto per spazi di stato discreti e limitati
    """
    
    def __init__(self, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dizionario {state_hash: {action: q_value}}
        self.q_table: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    
    def _hash_state(self, state: np.ndarray) -> int:
        """Crea hash dello stato per Q-table"""
        # Usa solo binary encoding delle rotte (ignora features continue per semplicità)
        route_selection = state[:-3]  # Primi n elementi sono binary
        return hash(tuple(route_selection))
    
    def choose_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Epsilon-greedy action selection"""
        if not valid_actions:
            return 0  # Fallback
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation
        state_hash = self._hash_state(state)
        q_values = {action: self.q_table[state_hash][action] for action in valid_actions}
        
        if not q_values:
            return random.choice(valid_actions)
        
        return max(q_values, key=q_values.get)
    
    def update(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool, valid_next_actions: List[int]):
        """Q-learning update rule"""
        state_hash = self._hash_state(state)
        next_state_hash = self._hash_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_hash][action]
        
        # Max Q-value per next state
        if done or not valid_next_actions:
            max_next_q = 0
        else:
            max_next_q = max([self.q_table[next_state_hash][a] for a in valid_next_actions],
                           default=0)
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_hash][action] = new_q
    
    def decay_epsilon(self):
        """Decai epsilon per ridurre exploration nel tempo"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DeepQLearningAgent:
    """
    Deep Q-Network (DQN) Agent
    Usa una rete neurale per approssimare Q-values
    NOTA: Richiede librerie di deep learning (non implementato completamente qui)
    """
    
    def __init__(self, state_size: int, n_actions: int):
        self.state_size = state_size
        self.n_actions = n_actions
        print("⚠️ DQN non completamente implementato - richiede TensorFlow/PyTorch")
        print("   Usa Q-Learning per implementazione completa")


class PolicyGradientAgent:
    """
    Policy Gradient Agent (REINFORCE)
    Impara direttamente una policy parametrizzata
    """
    
    def __init__(self, state_size: int, n_actions: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.n_actions = n_actions
        self.lr = learning_rate
        
        # Policy: dizionario di probabilità per ogni stato-azione
        self.policy: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(lambda: 1.0 / n_actions))
        
        # Memory per episodio
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def _hash_state(self, state: np.ndarray) -> int:
        route_selection = state[:-3]
        return hash(tuple(route_selection))
    
    def choose_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Campiona azione dalla policy"""
        if not valid_actions:
            return 0
        
        state_hash = self._hash_state(state)
        
        # Probabilità per valid actions
        probs = np.array([self.policy[state_hash][action] for action in valid_actions])
        probs = probs / probs.sum()  # Normalizza
        
        # Campiona
        action_idx = np.random.choice(len(valid_actions), p=probs)
        return valid_actions[action_idx]
    
    def store_transition(self, state: np.ndarray, action: int, reward: float):
        """Memorizza transizione per update policy"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def update_policy(self, gamma: float = 0.95):
        """Update policy usando REINFORCE"""
        # Calcola returns (cumulative discounted rewards)
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        # Normalizza returns
        returns = np.array(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update policy
        for state, action, G in zip(self.episode_states, self.episode_actions, returns):
            state_hash = self._hash_state(state)
            
            # Gradient ascent (aumenta probabilità di azioni con alto return)
            self.policy[state_hash][action] += self.lr * G
            
            # Assicura probabilità positive
            self.policy[state_hash][action] = max(0.01, self.policy[state_hash][action])
        
        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []


class RLSolver:
    """
    Wrapper per training e evaluation di agenti RL
    """
    
    def __init__(self, graph: nx.MultiGraph, tickets_file: str):
        self.env = TicketToRideEnvironment(graph, tickets_file)
        self.graph = graph
        self.tickets_file = tickets_file
    
    def train_qlearning(self, episodes: int = 1000, verbose: bool = True) -> Tuple[QLearningAgent, List[float]]:
        """Train Q-Learning agent"""
        print("=== TRAINING Q-LEARNING AGENT ===")
        print(f"Episodes: {episodes}\n")
        
        agent = QLearningAgent(
            n_actions=len(self.env.unique_routes),
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        rewards_history = []
        best_score = 0
        best_routes = []
        
        start_time = time.time()
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break
                
                action = agent.choose_action(state, valid_actions)
                next_state, reward, done, info = self.env.step(action)
                
                valid_next_actions = self.env.get_valid_actions()
                agent.update(state, action, reward, next_state, done, valid_next_actions)
                
                state = next_state
                episode_reward += reward
            
            agent.decay_epsilon()
            rewards_history.append(episode_reward)
            
            # Valuta soluzione finale
            final_score = self.env._calculate_final_score()
            if final_score > best_score:
                best_score = final_score
                best_routes = list(self.env.selected_routes)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.2f} - "
                      f"Best Score: {best_score} - Epsilon: {agent.epsilon:.3f}")
        
        end_time = time.time()
        
        print(f"\n✅ Training completato in {end_time - start_time:.2f}s")
        print(f"Best Score: {best_score}")
        print(f"Best Routes: {len(best_routes)}\n")
        
        return agent, rewards_history
    
    def evaluate_agent(self, agent, n_episodes: int = 100) -> Solution:
        """Valuta l'agente e restituisce la migliore soluzione"""
        print(f"=== EVALUATING AGENT ({n_episodes} episodes) ===")
        
        best_score = 0
        best_routes = []
        
        # Salva epsilon originale e impostalo a 0 (no exploration durante eval)
        original_epsilon = agent.epsilon if hasattr(agent, 'epsilon') else None
        if original_epsilon is not None:
            agent.epsilon = 0
        
        start_time = time.time()
        
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break
                
                action = agent.choose_action(state, valid_actions)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
            
            final_score = self.env._calculate_final_score()
            if final_score > best_score:
                best_score = final_score
                best_routes = list(self.env.selected_routes)
        
        # Ripristina epsilon
        if original_epsilon is not None:
            agent.epsilon = original_epsilon
        
        end_time = time.time()
        
        # Calcola dettagli soluzione
        route_points = sum(self.env.points_table.get(r[2], 0) for r in best_routes)
        longest = self.env._calculate_longest_path()
        longest_bonus = 10 if longest > 0 else 0
        ticket_points = self.env._calculate_ticket_points()
        trains_used = sum(r[2] for r in best_routes)
        
        print(f"✅ Evaluation completata")
        print(f"Best Score: {best_score}")
        print(f"Routes: {len(best_routes)}")
        print(f"Trains: {trains_used}/45\n")
        
        return Solution(
            routes=best_routes,
            route_points=route_points,
            longest_path_length=longest,
            longest_bonus=longest_bonus,
            ticket_points=ticket_points,
            total_score=best_score,
            trains_used=trains_used,
            computation_time=end_time - start_time,
            algorithm="Q-Learning",
            episodes=n_episodes
        )


def train_and_evaluate_rl_agent(
    graph: nx.MultiGraph,
    tickets_file: str,
    training_episodes: int = 1000,
    eval_episodes: int = 100
) -> Dict[str, Union[Solution, List[float], QLearningAgent]]:
    """
    Funzione principale per training e evaluation di agenti RL

    :param graph: Grafo delle stazioni
    :param tickets_file: File CSV contenente le informazioni sui biglietti
    :param training_episodes: Numero di episodi per il training
    :param eval_episodes: Numero di episodi per l'evaluazione
    :return: Dizionario contenente le informazioni sull'agente e sulla storia delle ricompense
    """
    solver = RLSolver(graph, tickets_file)
    
    # Train Q-Learning
    agent, rewards_history = solver.train_qlearning(episodes=training_episodes, verbose=True)
    
    # Evaluate
    solution = solver.evaluate_agent(agent, n_episodes=eval_episodes)
    
    return {
        'q_learning': solution,
        'rewards_history': rewards_history,
        'agent': agent
    }
