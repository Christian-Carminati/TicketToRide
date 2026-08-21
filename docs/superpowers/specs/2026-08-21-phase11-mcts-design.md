# Specifica di Design: Fase 11 — Monte Carlo Tree Search (MCTS) & Heuristic Rollout

**Data:** 2026-08-21  
**Fase:** 11 (Monte Carlo Tree Search, Heuristic Rollout, Determinizzazione POMDP, Valutazione Cross-Paradigma)  
**Stato:** Approvato dall'Utente  

---

## 1. Obiettivi e Visione Scientifica

La Fase 11 introduce per la prima volta in TicketToRide RL Lab la famiglia degli algoritmi di **Pianificazione e Ricerca ad Albero (Tree Search)** con **Monte Carlo Tree Search (MCTS)**.

Nel contesto del laboratorio, MCTS riveste un ruolo strategico fondamentale:
1. **Validazione e Benchmark Senza Addestramento (Zero-Training Strong Baseline):** A differenza delle reti neurali (DQN, PPO) che richiedono centinaia di migliaia di step di gradient descent, MCTS ragiona direttamente *online* simulando il futuro attraverso il modello del gioco.
2. **Superamento dei Limiti di Parziale Osservabilità (POMDP):** Implementazione di **Determinized Information Set MCTS (SO-MCTS)** che campiona mondi determinizzati consistenti con le carte pubbliche note, preservando le invarianti anti-leakage.
3. **Ponte Verso la Fase 12 (Neural MCTS / AlphaZero):** La separazione modulare tra albero di ricerca, funzioni di selezione/espansione e policy di rollout/valutazione foglia rende il motore direttamente estendibile all'uso di policy/value network neurali.
4. **Trasparenza Didattica e Documentazione Completa:** Ogni componente algoritmico (Selection UCT, Expansion, Rollout Euristico, Backpropagation, Determinizzazione) è documentato nel dettaglio teorico e matematico.

---

## 2. Fondamenti Teorici e Matematici

### 2.1 Le 4 Fasi di MCTS
Dato uno stato di partenza $s_0$, MCTS itera per $N_{\text{simulations}}$ ripetendo 4 fasi:

```text
               ┌─────────────┐
               │  SELECTION  │ ──► Discende l'albero via UCT finché il nodo non è foglia
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │  EXPANSION  │ ──► Aggiunge uno o più nodi figli con azioni legali inesplorate
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │ SIMULATION  │ ──► Esegue un rollout (Random, Greedy, Strategic) fino a profondità D
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │ BACKPROP    │ ──► Risale l'albero aggiornando N(s) <- N(s)+1 e W(s) <- W(s)+v
               └─────────────┘
```

### 2.2 Formula UCT (Upper Confidence Bounds applied to Trees)
Per bilanciare **Exploitation** (sfruttare rami con alto valore atteso) ed **Exploration** (esplorare rami poco visitati), la fase di selezione applica l'algoritmo UCB1 adattato agli alberi:

$$\text{UCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot \sqrt{\frac{\ln N(s)}{N(s, a) + \epsilon}}$$

dove:
- $Q(s, a) = \frac{W(s, a)}{N(s, a)}$ è il valore medio accumulato dall'azione $a$ nello stato $s$;
- $N(s)$ è il conteggio visite del nodo genitore;
- $N(s, a)$ è il conteggio visite del nodo figlio generato da $a$;
- $c_{\text{puct}} = \sqrt{2} \approx 1.4142$ è la costante di esplorazione teorica di Kocsis e Szepesvári (2006);
- $\epsilon = 10^{-6}$ impedisce divisioni per zero.

### 2.3 Gestione Multi-Player e Prospettiva dei Valori
In partite a 2 giocatori a somma costante/zero-sum, il valore $v \in [-1.0, 1.0]$ retropropagato durante la Backpropagation viene orientato dal punto di vista del giocatore alla radice:
$$v_{\text{node}} = \begin{cases} +v & \text{se il giocatore al nodo è il giocatore alla radice} \\ -v & \text{se il giocatore al nodo è l'avversario} \end{cases}$$

### 2.4 Criterio di Scelta dell'Azione alla Radice (Decision Rule)
Al termine del budget di simulazioni, l'azione selezionata dall'agente segue la regola del **Robust Child**:
$$a^* = \arg\max_{a \in \mathcal{A}(s_0)} N(s_0, a)$$
La frequenza di visita $N(s_0, a)$ è asintoticamente più stabile del valore $Q(s_0, a)$ rispetto a outlier casuali nelle simulazioni stocastiche.

### 2.5 Heuristic Leaf Evaluation Function
Quando la simulazione raggiunge il limite massimo di profondità di rollout $D$ (`max_rollout_depth`), lo stato foglia $s_{\text{leaf}}$ viene valutato tramite una funzione euristica continua normalizzata in $[-1, 1]$:

$$V(s) = w_{\text{score}} \cdot \Delta_{\text{score}} + w_{\text{tickets}} \cdot \Delta_{\text{tickets}} + w_{\text{routes}} \cdot \Delta_{\text{routes}} + w_{\text{trains}} \cdot \Delta_{\text{trains}}$$

con:
- $\Delta_{\text{score}} = \tanh\left(\frac{\text{Score}(P_{\text{root}}) - \text{Score}(P_{\text{opp}})}{30.0}\right) \in [-1, 1]$
- $\Delta_{\text{tickets}} = \frac{\text{CompletedTickets}(P_{\text{root}}) - \text{CompletedTickets}(P_{\text{opp}})}{\max(1, \text{TotalTickets}(P_{\text{root}}) + \text{TotalTickets}(P_{\text{opp}}))}$
- $\Delta_{\text{routes}} = \frac{\text{ClaimedLength}(P_{\text{root}}) - \text{ClaimedLength}(P_{\text{opp}})}{45.0}$
- $\Delta_{\text{trains}} = \frac{\text{TrainsRemaining}(P_{\text{root}}) - \text{TrainsRemaining}(P_{\text{opp}})}{45.0}$
- Pesi predefiniti: $w_{\text{score}} = 0.50, w_{\text{tickets}} = 0.30, w_{\text{routes}} = 0.15, w_{\text{trains}} = 0.05$.

Se lo stato è terminale prima di $D$:
$$V(s_{\text{terminal}}) = \begin{cases} +1.0 & \text{se } \text{Score}(P_{\text{root}}) > \text{Score}(P_{\text{opp}}) \\ -1.0 & \text{se } \text{Score}(P_{\text{root}}) < \text{Score}(P_{\text{opp}}) \\ 0.0 & \text{se pareggio} \end{cases}$$

---

## 3. Architettura dei Moduli e Componenti Software

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                               GAME CORE LAYER                                │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ Game.clone() / GameState.clone() / Player.clone() / Board.clone()      │  │
│  │ (Fast In-Memory Deep Copy senza overhead di serializzazione JSON)      │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            MCTS ENGINE LAYER                                 │
│                                                                              │
│  ┌───────────────────────────┐         ┌──────────────────────────────────┐  │
│  │    mcts_determinization   │         │            MCTSNode              │  │
│  │ (Campiona carte nascoste) │         │ (N, W, Q, children, untried)     │  │
│  └─────────────┬─────────────┘         └────────────────┬─────────────────┘  │
│                │                                        │                    │
│                ▼                                        ▼                    │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           MCTSSearchEngine                             │  │
│  │  - Selection (UCT formula)                                             │  │
│  │  - Expansion (Legal Action Branching)                                  │  │
│  │  - Simulation (Rollout Policies: Random, Greedy, Strategic)            │  │
│  │  - Backpropagation (Root-Perspective Value Update)                     │  │
│  └───────────────────────────────────┬────────────────────────────────────┘  │
└──────────────────────────────────────┼───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                               AGENT LAYER                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                               MCTSAgent                                │  │
│  │  - Implements BaseAgent interface (act() & select_action())            │  │
│  │  - Full compatibility with Evaluator, Tournament, MultiMapEnv          │  │
│  └───────────────────────────────────┬────────────────────────────────────┘  │
└──────────────────────────────────────┼───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       EVALUATION & BENCHMARK LAYER                           │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          MCTSBenchmarkRunner                           │  │
│  │  - Head-to-Head vs Random, Greedy, Strategic, PPO                      │  │
│  │  - Simulation Budget Scaling Ablation (N = 10, 25, 50, 100, 200)       │  │
│  │  - Rollout Policy Ablation (Random vs Greedy vs Strategic)             │  │
│  │  - Generates phase11_report.json & phase11_report.md                   │  │
│  └───────────────────────────────────┬────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       scripts/benchmark_mcts.py                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Dettaglio Implementativo dei Componenti

### 4.1 Fast State Cloning (`src/game/game.py`, `src/game/state.py`, `src/game/board.py`, `src/game/player.py`)
- **`Board.clone() -> Board`**: Crea una nuova istanza riutilizzando i riferimenti immutabili delle città (`City`) e clonando la lista delle `Route` (inclusi i campi `claimed_by`).
- **`Player.clone() -> Player`**: Copia le carte (`cards.copy()`), i biglietti (`tickets.copy()`), le tratte reclamate (`claimed_route_ids.copy()`) e i biglietti pendenti (`pending_tickets.copy()`).
- **`GameState.clone() -> GameState`**: Clona tutti i giocatori, la lista `visible_cards`, `train_deck`, `discard_pile`, `ticket_deck` e preserva le flag di stato.
- **`Game.clone(rng_seed: int | None = None) -> Game`**: Restituisce una nuova istanza di `Game` isolata, con `board`, `state` e `rng` clonati.

### 4.2 Determinizzazione POMDP (`src/rl/mcts_determinization.py`)
```python
def determinize_game(game: Game, root_player_id: str, rng: SeededRNG) -> Game:
    """Costruisce un mondo di gioco determinizzato compatibile con le informazioni pubbliche.
    
    1. Clona il gioco reale.
    2. Identifica tutte le carte non visibili al giocatore radice (carte in mano agli avversari + train deck).
    3. Rimescola uniformemente il pool di carte nascoste.
    4. Ridistribuisce a ciascun avversario esattamente il numero di carte che possiede, e rimette le restanti nel train deck.
    5. Ripete la procedura per i Destination Tickets degli avversari se presenti nel mazzo biglietti.
    """
```

### 4.3 Motore di Ricerca MCTS (`src/rl/mcts.py`)
```python
class RolloutPolicyType(str, Enum):
    RANDOM = "random"
    GREEDY = "greedy"
    STRATEGIC = "strategic"

@dataclass
class MCTSConfig:
    """Configurazione parametrica del motore MCTS."""
    num_simulations: int = 100
    c_puct: float = 1.41421356
    max_rollout_depth: int = 10
    rollout_policy: RolloutPolicyType = RolloutPolicyType.STRATEGIC
    use_determinization: bool = True
    seed: int = 42
    heuristic_weights: tuple[float, float, float, float] = (0.50, 0.30, 0.15, 0.05)

class MCTSNode:
    """Nodo singolo dell'albero di ricerca MCTS."""
    def __init__(
        self,
        state: GameState,
        parent: "MCTSNode | None" = None,
        action: Action | None = None,
        player_id: str = "player_0",
        untried_actions: list[Action] | None = None,
    ) -> None: ...

    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def select_best_child(self, c_puct: float) -> tuple[Action, "MCTSNode"]: ...
    def expand(self, action: Action, next_state: GameState, next_player_id: str, untried_actions: list[Action]) -> "MCTSNode": ...
    def update(self, reward: float) -> None: ...

class MCTSSearchEngine:
    """Esecutore delle simulazioni MCTS con supporto a determinizzazione e rollout euristico."""
    def __init__(self, config: MCTSConfig | None = None) -> None: ...
    def search(self, game: Game, root_player_id: str) -> Action: ...
```

### 4.4 Agente MCTS (`src/agents/mcts_agent.py`)
```python
class MCTSAgent(BaseAgent):
    """Agente MCTS conforme all'interfaccia standard BaseAgent."""
    def __init__(self, config: MCTSConfig | None = None, name: str = "MCTSAgent") -> None:
        super().__init__(name=name)
        self.config = config or MCTSConfig()
        self.engine = MCTSSearchEngine(self.config)

    def act(self, state: GameState, valid_actions: list[Action], board: Board | None = None) -> Action:
        # Costruisce l'istanza di Game coerente e avvia la ricerca MCTS
        ...

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray | None = None, info: dict[str, Any] | None = None) -> int:
        # Mappatura azione discreta per Gymnasium
        ...
```

### 4.5 Suite di Valutazione e Benchmark (`src/evaluation/mcts_benchmark.py`)
```python
class MCTSBenchmarkRunner:
    """Benchmark scientifico formale per MCTS."""
    def run_full_benchmark(self, num_games: int = 50, seed: int = 42) -> dict[str, Any]:
        """Esegue:
        1. Torneo Head-to-Head: MCTSAgent vs Random, Greedy, Strategic, PPO.
        2. Simulazione Scaling: N in [10, 25, 50, 100, 200] vs StrategicAgent.
        3. Rollout Policy Comparison: Random vs Greedy vs Strategic.
        4. Generazione automatica di report JSON e Markdown.
        """
```

---

## 5. Piano dei Test e Accettazione

1. `tests/game/test_fast_cloning.py`:
   - Verifica isolamento totale delle modifiche tra istanza originale e clone.
   - Benchmark throughput di clonazione ($\ge 10\,000$ cloni/sec).
2. `tests/rl/test_mcts_determinization.py`:
   - Invarianza conteggio carte e carte visibili.
   - Assenza di carte fantasma o perdita di carte nella ridistribuzione.
   - Rispetto dei principi POMDP anti-leakage.
3. `tests/rl/test_mcts_node.py`:
   - Calcolo UCT e selezione del miglior figlio.
   - Espansione corretta delle azioni non ancora provate.
   - Aggiornamento statistiche di visita e backpropagation.
4. `tests/rl/test_mcts_engine.py`:
   - Risoluzione deterministica di stati di gioco con mossa forzata o tattica evidente.
   - Convergenza asintotica della frequenza di visita delle azioni migliori al crescere di $N$.
5. `tests/agents/test_mcts_agent.py`:
   - Conformità interfaccia `BaseAgent` (`act()`) e Gymnasium (`select_action()`).
   - Riproducibilità deterministica data dal seed.
6. `tests/rl/test_phase11_acceptance.py`:
   - MCTS batte RandomAgent ($\ge 90\%$ win rate su 50 partite).
   - MCTS batte GreedyAgent ($\ge 70\%$ win rate su 50 partite).
   - MCTS è competitivo o superiore a StrategicAgent ($\ge 55\%$ win rate su 50 partite).
   - Generazione dei report `phase11_report.json` e `phase11_report.md`.
