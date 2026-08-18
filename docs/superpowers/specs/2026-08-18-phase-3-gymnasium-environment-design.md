# Specifiche di Design: Fase 3 — Gymnasium Environment & Formalizzazione Matematica

## 1. Panoramica & Obiettivo

La **Fase 3** introduce il sottosistema dell'**Ambiente di Reinforcement Learning (Gymnasium Environment)** per il *TicketToRide RL Lab*.

Questo modulo incapsula il **Game Core** deterministico (sviluppato nella Fase 1) e sfrutta le baseline (sviluppate nella Fase 2) per fornire un ambiente conforme agli standard **Farama Gymnasium** (`gym.Env`). 

L'ambiente è progettato per supportare:
1. Algoritmi RL discreti standard (DQN, Actor-Critic, PPO nella Fase 4 e Fase 6).
2. Algoritmi con supporto all'**Action Masking** (Maskable PPO / Invalid Action Masking).
3. Benchmark di **Reward Engineering & Shaping** modulare (Fase 7).
4. Studio formale della **Parziale Osservabilità (POMDP)** garantendo la totale assenza di leakage di informazioni nascoste (Fase 8).

---

## 2. Formalizzazione Matematica del POMDP

Il gioco *Ticket to Ride* è formalizzato come un **Partially Observable Markov Decision Process (POMDP)** a tempo discreto:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

### 2.1 Componenti della Tupla

1. **Spazio degli Stati Globali ($\mathcal{S}$)**:
   Lo stato globale (o *True State*) $s \in \mathcal{S}$ descrive la configurazione completa e onnisciente del sistema:
   $$s = \left( \mathbf{p}_0, \mathbf{p}_1, \dots, \mathbf{p}_{N-1}, \mathbf{c}_{\text{visible}}, \mathbf{d}_{\text{train}}, \mathbf{d}_{\text{discard}}, \mathbf{d}_{\text{ticket}}, \mathbf{B}, \tau, k, \text{phase} \right)$$
   dove:
   - $\mathbf{p}_i = (\mathbf{h}_i, \mathcal{K}_i, \mathcal{P}_i, t_i, u_i)$ rappresenta lo stato del giocatore $i$ (mano di carte vagone $\mathbf{h}_i \in \mathbb{N}^9$, biglietti posseduti $\mathcal{K}_i$, biglietti pendenti $\mathcal{P}_i$, vagoni residui $t_i \in [0, 45]$, punteggio $u_i \in \mathbb{N}$).
   - $\mathbf{c}_{\text{visible}} \in (\mathcal{C} \cup \{\emptyset\})^5$ sono le 5 carte scoperte sul tavolo.
   - $\mathbf{d}_{\text{train}} \in \mathcal{C}^*$ e $\mathbf{d}_{\text{ticket}} \in \mathcal{T}^*$ sono le sequenze ordinate dei mazzi coperti.
   - $\mathbf{d}_{\text{discard}} \in \mathcal{C}^*$ è la pila degli scarti.
   - $\mathbf{B} = \{ r_j \mapsto \text{owner}_j \}_{j=1}^{|\mathcal{E}|}$ è la mappa delle tratte e dei relativi proprietari.
   - $\tau \in \{0, \dots, N-1\}$ è l'indice del giocatore attivo.
   - $k \in \mathbb{N}$ è il numero del turno.
   - $\text{phase} \in \{ \text{NORMAL}, \text{DRAWING\_SECOND\_CARD}, \text{CHOOSING\_TICKETS}, \text{CHOOSING\_INITIAL\_TICKETS} \}$.

2. **Spazio delle Azioni ($\mathcal{A}$)**:
   L'insieme discreto e finito di tutte le decisioni elementari:
   $$\mathcal{A} = \{0, 1, \dots, |\mathcal{A}| - 1\}$$

3. **Funzione di Transizione Stocastica ($\mathcal{T}$)**:
   $$\mathcal{T}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$$
   La dinamica è stocastica a causa di:
   - Pesca da mazzi coperti rimescolati ($\mathbf{d}_{\text{train}}, \mathbf{d}_{\text{ticket}}$).
   - Eventuale avanzamento delle mosse degli avversari tra due turni consecutivi dell'agente attivo.

4. **Funzione di Ricompensa ($\mathcal{R}$)**:
   $$\mathcal{R} : \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$$
   Segnale scalare $r_t = \mathcal{R}(s_t, a_t, s_{t+1})$ calcolato per il giocatore target.

5. **Spazio delle Osservazioni ($\Omega$) e Funzione di Proiezione ($\mathcal{O}$)**:
   L'agente non osserva $s$, ma un vettore compatto normalizzato:
   $$o_t = \mathcal{O}(s_t, i) \in [0, 1]^D \subset \Omega = \mathbb{R}^D$$
   dove la funzione $\mathcal{O}$ proietta lo stato $s$ esclusivamente sulle componenti pubbliche e private lecite del giocatore $i$.

6. **Fattore di Sconto Temporale ($\gamma$)**:
   $$\gamma \in [0, 1)$$

---

## 3. Spazio delle Osservazioni (`ObservationV1`)

### 3.1 Architettura Vettoriale

`ObservationV1` codifica lo stato visibile in un array monodimensionale di tipo `np.float32` e dimensione fissa $D$:

$$o = \left[ \mathbf{v}_{\text{hand}} \,\|\, \mathbf{v}_{\text{visible}} \,\|\, \mathbf{v}_{\text{player}} \,\|\, \mathbf{v}_{\text{routes}} \,\|\, \mathbf{v}_{\text{tickets}} \,\|\, \mathbf{v}_{\text{opponents}} \,\|\, \mathbf{v}_{\text{decks}} \,\|\, \mathbf{v}_{\text{phase}} \right]$$

### 3.2 Dettaglio dei Sottovettori e Normalizzazioni

| Sottovettore | Simbolo | Dim. (USA) | Dim. (Mini) | Definizione Matematica / Codifica | Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mano Giocatore** | $\mathbf{v}_{\text{hand}}$ | $9$ | $9$ | $v_c = \min\left(\frac{\text{count}(c)}{12}, 1.0\right)$ per gli 8 colori + Locomotiva | $[0, 1]$ |
| **Carte Scoperte** | $\mathbf{v}_{\text{visible}}$ | $50$ | $50$ | 5 slot $\times$ 10 canali (9 colori + 1 canale slot vuoto) in One-Hot | $\{0, 1\}$ |
| **Stato Giocatore** | $\mathbf{v}_{\text{player}}$ | $2$ | $2$ | $\left[ \frac{\text{trains}}{45}, \min\left(\frac{\text{score}}{150}, 1.0\right) \right]$ | $[0, 1]$ |
| **Tratte Mappa** | $\mathbf{v}_{\text{routes}}$ | $3 \times |\mathcal{E}|$ | $3 \times |\mathcal{E}|$ | Per ogni tratta $r_j$: $[1, 0, 0]$ (libera), $[0, 1, 0]$ (propria), $[0, 0, 1]$ (avversaria) | $\{0, 1\}$ |
| **Biglietti** | $\mathbf{v}_{\text{tickets}}$ | $3 \times |\mathcal{K}|$ | $3 \times |\mathcal{K}|$ | Per ogni biglietto $k_m$: $[\mathbb{I}_{\text{owned}}, \mathbb{I}_{\text{completed}}, \frac{\text{points}}{25}]$ | $[0, 1]$ |
| **Avversari** | $\mathbf{v}_{\text{opponents}}$ | $4 \times (N-1)$ | $4 \times (N-1)$ | Per ogni avversario: $\left[ \frac{|\mathbf{h}_{\text{opp}}|}{30}, \frac{t_{\text{opp}}}{45}, \frac{|\mathcal{E}_{\text{opp}}|}{30}, \frac{u_{\text{opp}}}{150} \right]$ | $[0, 1]$ |
| **Mazzi e Gioco** | $\mathbf{v}_{\text{decks}}$ | $5$ | $5$ | $\left[ \frac{|\mathbf{d}_{\text{train}}|}{110}, \frac{|\mathbf{d}_{\text{discard}}|}{110}, \frac{|\mathbf{d}_{\text{ticket}}|}{30}, \frac{k}{100}, \mathbb{I}_{\text{last\_round}} \right]$ | $[0, 1]$ |
| **Fase Turno** | $\mathbf{v}_{\text{phase}}$ | $4$ | $4$ | One-Hot di $\text{phase}$ su 4 stati (`NORMAL`, `DRAWING_SECOND_CARD`, `CHOOSING_TICKETS`, `CHOOSING_INITIAL_TICKETS`) | $\{0, 1\}$ |

* **Dimensione Totale Mappa USA ($|\mathcal{E}|=100, |\mathcal{K}|=30, N=2$)**:
  $$D_{\text{USA}} = 9 + 50 + 2 + 300 + 90 + 4 + 5 + 4 = 464$$
* **Dimensione Totale Mappa Mini ($|\mathcal{E}|=6, |\mathcal{K}|=4, N=2$)**:
  $$D_{\text{Mini}} = 9 + 50 + 2 + 18 + 12 + 4 + 5 + 4 = 104$$

### 3.3 Teorema di Non-Leakage (Fairness dell'Osservazione)

Siano $s_1, s_2 \in \mathcal{S}$ due stati globali distinti del gioco. Definiamo la funzione di equivalenza delle informazioni pubbliche e personali per il giocatore $i$:
$$\text{Equiv}(s_1, s_2, i) \iff \begin{cases} \mathbf{p}_i^{(1)} = \mathbf{p}_i^{(2)} \\ \mathbf{c}_{\text{visible}}^{(1)} = \mathbf{c}_{\text{visible}}^{(2)} \\ \mathbf{B}^{(1)} = \mathbf{B}^{(2)} \\ |\mathbf{h}_j^{(1)}| = |\mathbf{h}_j^{(2)}|, \, t_j^{(1)} = t_j^{(2)}, \, u_j^{(1)} = u_j^{(2)} \quad \forall j \neq i \\ |\mathbf{d}_{\text{train}}^{(1)}| = |\mathbf{d}_{\text{train}}^{(2)}|, \, |\mathbf{d}_{\text{discard}}^{(1)}| = |\mathbf{d}_{\text{discard}}^{(2)}|, \, |\mathbf{d}_{\text{ticket}}^{(1)}| = |\mathbf{d}_{\text{ticket}}^{(2)}| \end{cases}$$

**Invariante Garantita**:
$$\text{Equiv}(s_1, s_2, i) \implies \mathcal{O}(s_1, i) = \mathcal{O}(s_2, i)$$
Nessuna componente del vettore dipende da:
1. Colore delle carte segrete in mano agli avversari $\mathbf{h}_j$.
2. Biglietti destinazione posseduti dagli avversari $\mathcal{K}_j$.
3. Sequenza interna e ordine futuro delle carte nei mazzi $\mathbf{d}_{\text{train}}$ e $\mathbf{d}_{\text{ticket}}$.

---

## 4. Spazio delle Azioni (`DiscreteActionSpace`) & Action Masking (`ActionMasker`)

### 4.1 Mappatura Biunivoca Discreta

`DiscreteActionSpace` mappa in modo deterministico e biunivoco ogni intero $a \in \{0, \dots, |\mathcal{A}|-1\}$ in una `Action` del dominio.

$$\phi : \mathcal{A} \longleftrightarrow \text{DomainAction}$$

#### Partizione dello Spazio delle Azioni:
1. **$a = 0$**: `Action(ActionType.DRAW_HIDDEN_CARD)` (Pesca dal mazzo coperto).
2. **$a \in [1, 5]$**: `Action(ActionType.DRAW_VISIBLE_CARD, card_index = a - 1)` (Pesca carta scoperta slot $0..4$).
3. **$a = 6$**: `Action(ActionType.DRAW_TICKETS)` (Pesca 3 nuovi biglietti destinazione).
4. **$a \in [7, 13]$**: `Action(ActionType.KEEP_TICKETS, ticket_ids = ...)` (7 sottoinsiemi non vuoti di $\{0, 1, 2\}$):
   - $a = 7 \to \{0\}$
   - $a = 8 \to \{1\}$
   - $a = 9 \to \{2\}$
   - $a = 10 \to \{0, 1\}$
   - $a = 11 \to \{0, 2\}$
   - $a = 12 \to \{1, 2\}$
   - $a = 13 \to \{0, 1, 2\}$
5. **$a \ge 14$**: `Action(ActionType.CLAIM_ROUTE, route_id = r.id, color_chosen = c)`:
   - Tratte con colore specifico: 1 azione con $c = r.\text{color}$.
   - Tratte grigie: 8 azioni distinte, una per ciascun colore $c \in \mathcal{C}_{\text{standard}}$.
   - *Allocazione Locomotive*: Risolta deterministicamente ed efficacemente a livello di esecuzione (consuma prima le carte colore possedute e completa con locomotive se necessario).

### 4.2 Mascheramento delle Azioni (`ActionMasker`)

Dato lo stato $s \in \mathcal{S}$ e la lista delle azioni valide $\mathcal{V}(s) \subseteq \text{DomainAction}$ calcolate da `Game.valid_actions()`:

$$M(s)_a = \begin{cases} 1 & \text{se } \phi(a) \in \mathcal{V}(s) \\ 0 & \text{altrimenti} \end{cases} \quad \forall a \in \{0, \dots, |\mathcal{A}| - 1\}$$

L'`ActionMasker` garantisce che per qualsiasi policy di campionamento $a \sim \pi(\cdot \mid o)$:
$$\mathbb{P}(a \mid o, M(s)_a = 0) = 0$$

---

## 5. Calcolo della Ricompensa (`RewardV1` / `DefaultRewardCalculator`)

La ricompensa scalare per il giocatore in addestramento è definita modularmente come:

$$R_t = R_t^{\text{step}} + \mathbb{I}_{\text{terminal}} \cdot R_t^{\text{terminal}}$$

### 5.1 Componente Step Intermedia ($R_t^{\text{step}}$)

$$R_t^{\text{step}} = w_{\text{route}} \cdot \Delta \text{RoutePoints}_t + w_{\text{ticket\_complete}} \cdot \Delta \text{TicketPoints}_t - w_{\text{step}}$$

* $\Delta \text{RoutePoints}_t$: Punti di tratta guadagnati istantaneamente dall'azione (secondo la tabella ufficiale: $1\to1, 2\to2, 3\to4, 4\to7, 5\to10, 6\to15$).
* $\Delta \text{TicketPoints}_t$: Punti del biglietto se la mossa ha completato la connettività per un biglietto posseduto precedentemente incompleto.
* $w_{\text{step}} \ge 0$: Penalità di passo temporale (default $0.0$) per incentivare l'efficienza.

### 5.2 Componente Terminale di Fine Episodio ($R_t^{\text{terminal}}$)

Quando `terminated == True`:

$$R_t^{\text{terminal}} = w_{\text{win}} \cdot \text{Sign}(\Delta \text{Score}) + w_{\text{score\_diff}} \cdot \Delta \text{Score} - w_{\text{ticket\_fail}} \cdot \sum_{k \in \mathcal{K}_{\text{uncompleted}}} \text{points}(k)$$

dove $\Delta \text{Score} = \text{Score}_{\text{player0}} - \text{Score}_{\text{opponent}}$.

### 5.3 Configurazione Parametrica (`RewardWeights`)

```python
@dataclass
class RewardWeights:
    route_points_weight: float = 1.0
    ticket_completion_weight: float = 1.0
    step_penalty: float = 0.0
    win_bonus: float = 20.0
    loss_penalty: float = 10.0
    score_diff_weight: float = 0.5
    ticket_failure_penalty_weight: float = 1.0
```

---

## 6. Dinamica di `TicketToRideEnv` (`gym.Env`)

### 6.1 Interfaccia e Inizializzazione

```python
class TicketToRideEnv(gym.Env):
    def __init__(
        self,
        board: Board | None = None,
        tickets_deck: list[DestinationTicket] | None = None,
        opponent: BaseAgent | None = None,
        observation_encoder: BaseObservationEncoder | None = None,
        reward_calculator: BaseRewardCalculator | None = None,
        max_turns: int = 300,
    ) -> None: ...
```

### 6.2 Flusso di `reset(seed=None, options=None)`
1. Resetta il `Game` deterministico con `seed`.
2. Se lo stato iniziale è `TurnState.CHOOSING_INITIAL_TICKETS`:
   - Se `opponent` è presente ed è il turno del Player 0, produce l'osservazione e l'`action_mask` limitata alla scelta dei biglietti iniziali ($\ge 2$ biglietti).
3. Restituisce `(observation, info)` con `info["action_mask"] = masker.compute_mask(valid_actions)`.

### 6.3 Flusso di `step(action: int)`
1. Converte l'intero $a_t$ in `domain_action = discrete_actions.to_action(action)`.
2. Salva lo stato precedente $s_t$.
3. Esegue `next_state = game.step(domain_action)` per il Player 0.
4. **Auto-Stepping Avversario**: Se `opponent` è configurato e `not next_state.is_game_over`:
   - Finchè `game.state.current_player_index != 0` e `not game.state.is_game_over`:
     - L'avversario calcola la sua azione: $a_{\text{opp}} = \text{opponent.act}(s, \text{valid\_actions}, \text{board})$.
     - L'ambiente esegue `game.step(a_{\text{opp}})`.
5. Calcola il reward scalare $R_t$ tramite `reward_calculator.calculate(...)`.
6. Calcola `terminated = game.state.is_game_over`.
7. Calcola `truncated = (game.state.turn_number >= max_turns)`.
8. Genera la nuova osservazione $o_{t+1}$ per Player 0.
9. Calcola la maschera $M(s_{t+1})$ per il prossimo turno di Player 0.
10. Restituisce `(obs, reward, terminated, truncated, info)`.

---

## 7. Verifica e Criteri di Accettazione

### 7.1 Acceptance Criteria
1. **Gymnasium Compliance**:
   - `TicketToRideEnv` supera senza avvisi né eccezioni `gymnasium.utils.env_checker.check_env(env)`.
2. **Anti-Leakage Verification**:
   - Test automatico che dimostra che alterazioni arbitrarie alle carte segrete avversarie e all'ordine dei mazzi coperti producono osservazioni numeriche identiche per Player 0.
3. **Biunivocità e Copertura dello Spazio Azioni**:
   - Ogni mossa valida generata dal Game Core corrisponde a un'azione valida decodificabile senza perdite.
4. **Simulazione Multi-Episodio ad Alto Rendimento**:
   - Esecuzione di 100 episodi completi via interfaccia Gymnasium contro `RandomAgent`, `GreedyAgent` e `StrategicHeuristicAgent` con reward finiti e assenza di crash.
