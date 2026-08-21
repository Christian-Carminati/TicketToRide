# Specifica di Design: Fase 12 — Advanced Research (Neural MCTS, AlphaZero Self-Play, Bayesian Opponent Modeling & Curriculum Learning)

**Data:** 2026-08-21  
**Fase:** 12 (Advanced Research — Neural MCTS, AlphaZero, Bayesian Opponent Modeling, Curriculum Learning, Benchmark Scientifico Cross-Paradigma)  
**Stato:** Approvato dall'Utente  

---

## 1. Obiettivi e Visione Scientifica

La **Fase 12 (Advanced Research)** rappresenta il culmine della ricerca teorica e algoritmica di **TicketToRide RL Lab**.

Nel percorso delle fasi precedenti abbiamo esplorato:
- Motori basati su euristiche tattiche e greedy (**Fase 2**);
- Reinforcement Learning Model-Free basato su Value/Q-Learning con DQN (**Fase 4**);
- Reinforcement Learning Model-Free basato su Policy Gradient con PPO from scratch (**Fase 6**);
- Reward Engineering gerarchico e A/B testing comportamentale (**Fase 7**);
- Parziale osservabilità con Recurrent PPO (LSTM) e invarianza anti-leakage (**Fase 8**);
- Addestramento competitivo tramite Self-Play e Historical Policy Pools con PFSP (**Fase 9**);
- Generalizzazione e adattamento su mappe procedurali non viste (**Fase 10**);
- Pianificazione esplicita e Ricerca ad Albero con Monte Carlo Tree Search (MCTS) determinizzato ed euristica foglia (**Fase 11**).

La **Fase 12** sintetizza questi paradigmi in un ecosistema avanzato articolato su tre pilastri scientifici:

1. **Sintesi tra Ricerca ad Albero e Deep Learning (Neural MCTS & AlphaZero):**
   Sostituzione dei rollout casuali/euristici con una **Policy-Value Dual Network** addestrata tramite Self-Play iterativo. La rete guida l'espansione e la selezione dei rami tramite la formula polinomiale PUCT ed effettua la stima del valore terminale.
2. **Modellazione Esplicita dell'Avversario (Bayesian Opponent Modeling & Belief Tracking):**
   Tracciamento in tempo reale delle tratte rivendicate e delle carte pescate pubblicamente per calcolare la distribuzione di probabilità a posteriori $P(\text{Ticket}_k \mid \mathcal{H}_{\text{public}})$ sui Destination Tickets segreti dell'avversario. Questa stima abilita la **Belief-Weighted Determinization** (campionamento realistico per Information-Set MCTS) e l'intercettazione tattica dei colli di bottiglia (**Tactical Blocker**).
3. **Curriculum Learning & Scientific Cross-Paradigm Benchmark:**
   Addestramento progressivo su gradienti di complessità spaziale (mappe da `Micro` a `USA_Standard`) e strategica (da agenti casuali a pool di self-play), accompagnato da una suite di valutazione comparativa completa di tutti i paradigmi sviluppati.

---

## 2. Fondamenti Teorici e Matematici

### 2.1 Architettura Duale AlphaZero (Policy-Value Network)
Dato il vettore di osservazione $s \in \mathbb{R}^{d_{\text{obs}}}$ e la maschera delle azioni legali $\mathbf{m}(s) \in \{0, 1\}^{|\mathcal{A}|}$, una singola rete neurale parametrizzata da $\theta$ produce contemporaneamente:
1. **Vettore di Policy a Priori $\mathbf{p}_\theta(s) \in [0, 1]^{|\mathcal{A}|}$:**
   $$\mathbf{z}_{\text{logits}} = f_{\text{policy}}(\mathbf{h}(s))$$
   $$\tilde{z}_a = \begin{cases} z_a & \text{se } m_a = 1 \\ -10^8 & \text{se } m_a = 0 \end{cases}$$
   $$p_\theta(s, a) = \frac{\exp(\tilde{z}_a)}{\sum_{b=1}^{|\mathcal{A}|} \exp(\tilde{z}_b)}$$
2. **Valore Scalare $v_\theta(s) \in [-1.0, 1.0]$:**
   $$v_\theta(s) = \tanh(f_{\text{value}}(\mathbf{h}(s)))$$
   che predice l'esito atteso della partita dal punto di vista del giocatore corrente al nodo.

```text
                           ┌─────────────────────────┐
                           │   Observation Vector s  │
                           └────────────┬────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │  Shared Residual Trunk  │
                           │   Linear + LayerNorm +  │
                           │   ReLU + Skip Connect   │
                           └─────┬─────────────┬─────┘
                                 │             │
                    ┌────────────┘             └────────────┐
                    ▼                                       ▼
       ┌─────────────────────────┐             ┌─────────────────────────┐
       │      Policy Head        │             │       Value Head        │
       │   Linear -> Action Mask │             │     Linear -> Tanh      │
       └────────────┬────────────┘             └────────────┬────────────┘
                    │                                       │
                    ▼                                       ▼
             p_θ(s, ·) ∈ Δ^|A|                        v_θ(s) ∈ [-1, 1]
         (Prior Probabilities)                         (Expected Value)
```

---

### 2.2 Algoritmo di Ricerca Neurale PUCT (Polynomial Upper Confidence Trees)
Durante la fase di selezione dell'albero di ricerca, ad ogni nodo $s$, l'azione $a^*$ viene selezionata massimizzando la combinazione tra valore medio $Q(s, a)$ e termine di esplorazione guidato dal prior neurale $P(s, a)$:

$$a^* = \arg\max_{a \in \mathcal{A}(s)} \left[ Q(s, a) + U(s, a) \right]$$

dove:
$$U(s, a) = c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_{b \in \mathcal{A}(s)} N(s, b)}}{1 + N(s, a)}$$

- $Q(s, a) = \frac{W(s, a)}{N(s, a)}$ è il valore medio memorizzato per l'arco $(s, a)$;
- $P(s, a) = p_\theta(s, a)$ è la probabilità a priori stimata dalla rete per l'azione $a$;
- $c_{\text{puct}}$ è la costante di bilanciamento (default: $c_{\text{puct}} = 1.5$).

#### Esplorazione con Rumore di Dirichlet alla Radice
Per garantire una diversità sufficiente nelle traiettorie generate in Self-Play ed evitare il collasso prematuro della ricerca su minimi locali, alla radice $s_{\text{root}}$ il prior viene perturbato con rumore di Dirichlet $\text{Dir}(\alpha)$:

$$P(s_{\text{root}}, a) = (1 - \epsilon_{\text{dir}}) \cdot p_\theta(s_{\text{root}}, a) + \epsilon_{\text{dir}} \cdot \eta_a, \quad \boldsymbol{\eta} \sim \text{Dir}(\alpha)$$
con $\alpha = 0.3$ ed $\epsilon_{\text{dir}} = 0.25$.

---

### 2.3 Generazione Dati Self-Play e Funzione di Loss AlphaZero
1. **Distribuzione Target delle Visite:**
   Completate $N_{\text{simulations}}$ iterazioni MCTS per lo stato $s_t$, la policy target di addestramento $\boldsymbol{\pi}_t$ è calcolata dalle frequenze di visita:
   $$\pi_t(a) = \frac{N(s_t, a)^{1/\tau}}{\sum_b N(s_t, b)^{1/\tau}}$$
   dove $\tau = 1.0$ nelle prime $T_{\text{temp}}$ mosse della partita (esplorazione stocastica) e $\tau \to 0$ (scelta deterministica argmax) nelle mosse successive.
2. **Esito Terminale ($z_t$):**
   Al termine della partita (step $T$), l'esito reale della partita per il giocatore di turno al tempo $t$ è:
   $$z_t = \begin{cases} +1.0 & \text{se il giocatore al turno } t \text{ vince la partita} \\ -1.0 & \text{se perde} \\ 0.0 & \text{in caso di pareggio} \end{cases}$$
3. **Loss Congiunta AlphaZero:**
   La rete neurale viene addestrata minimizzando l'errore quadratico medio sul valore e la cross-entropia sulla policy, con regolarizzazione $L_2$:
   $$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B \left[ \left( z_i - v_\theta(s_i) \right)^2 - \boldsymbol{\pi}_i^T \log \mathbf{p}_\theta(s_i) \right] + c_{\text{reg}} \|\theta\|_2^2$$

---

### 2.4 Bayesian Opponent Modeling & Destination Ticket Belief Tracking

#### Formulazione del Problema
In *Ticket to Ride*, un giocatore avversario $P_{\text{opp}}$ persegue un insieme di Destination Tickets segreti $\mathcal{T}_{\text{opp}} \subset \mathcal{T}$.
L'agente osserva la storia pubblica $\mathcal{H}_{\text{public}}$, che include l'insieme delle tratte acquisite dall'avversario $\mathcal{E}_{\text{opp}} = \{e_1, e_2, \dots, e_m\}$.

Vogliamo calcolare la distribuzione di probabilità a posteriori per ciascun ticket $T_k \in \mathcal{T}$:
$$P(T_k \mid \mathcal{E}_{\text{opp}}) \propto P(T_k) \cdot P(\mathcal{E}_{\text{opp}} \mid T_k)$$

#### Metrica di Costo di Deviazione sul Grafo (Detour Cost)
Sia $G = (V, E)$ il grafo della mappa, dove i vertici $V$ sono le città e gli archi $E$ sono le tratte, con pesi $w(e)$ pari alla lunghezza della tratta in vagoni.
Sia $T_k = (u_k, v_k, \text{punti}_k)$ un Destination Ticket che richiede di connettere $u_k$ a $v_k$, con distanza minima $d_G(u_k, v_k)$.

Per ogni tratta $e = (a, b)$ rivendicata dall'avversario, il costo di deviazione rispetto al ticket $T_k$ è:
$$\text{Detour}(e, T_k) = \min\big( d_G(u_k, a) + w(e) + d_G(b, v_k),\, d_G(u_k, b) + w(e) + d_G(a, v_k) \big) - d_G(u_k, v_k)$$

- Se la tratta $e$ appartiene al cammino minimo tra $u_k$ e $v_k$, allora $\text{Detour}(e, T_k) = 0$.
- Più la tratta $e$ è distante o irrilevante per $T_k$, maggiore è il valore di $\text{Detour}(e, T_k)$.

#### Modello di Verosimiglianza (Likelihood)
La verosimiglianza di osservare la tratta $e$ dato il target $T_k$ segue una legge esponenziale attenuata:
$$L(e \mid T_k) = \gamma \cdot \exp\left(-\beta \cdot \text{Detour}(e, T_k)\right) + (1 - \gamma) \cdot \epsilon_{\text{noise}}$$
con parametri di calibrazione: $\beta = 0.5$, $\gamma = 0.85$, $\epsilon_{\text{noise}} = 0.05$.

L'aggiornamento bayesiano cumulativo su tutte le tratte osservate $\mathcal{E}_{\text{opp}}$ produce il posterior normalizzato:
$$P(T_k \mid \mathcal{E}_{\text{opp}}) = \frac{P(T_k) \prod_{e \in \mathcal{E}_{\text{opp}}} L(e \mid T_k)}{\sum_{j=1}^{|\mathcal{T}|} P(T_j) \prod_{e \in \mathcal{E}_{\text{opp}}} L(e \mid T_j)}$$

#### Belief-Weighted Determinization
Durante la fase di Information-Set determinization di MCTS:
- Invece di assegnare all'avversario ticket casuali estratti con probabilità uniforme $\frac{1}{|\mathcal{T}_{\text{available}}|}$;
- I ticket vengono campionati in proporzione alla probabilità a posteriori $P(T_k \mid \mathcal{E}_{\text{opp}})$.
Ciò garantisce che le simulazioni MCTS esplorino mondi possibili altamente probabili e coerenti con la strategia reale dell'avversario.

#### Tactical Bottleneck & Blocker Detector
Per ogni tratta ancora libera $e_{\text{free}} \in E_{\text{unclaimed}}$, definiamo il valore di minaccia/blocco:
$$\text{Threat}(e_{\text{free}}) = \sum_{k=1}^{|\mathcal{T}|} P(T_k \mid \mathcal{E}_{\text{opp}}) \cdot \text{punti}_k \cdot \mathbb{I}(e_{\text{free}} \in \text{CriticalPath}(T_k))$$
dove $\mathbb{I}$ è l'indicatore di appartenenza della tratta libera all'unico o primario percorso di completamento per il ticket $T_k$. Quando $\text{Threat}(e) > \theta_{\text{block}}$, l'agente può considerare l'azione di blocco difensivo come prioritaria.

---

### 2.5 Curriculum Learning (Progressione Spaziale & Strategica)

Il modulo di Curriculum Learning struttura l'addestramento dell'agente su stadi crescenti di complessità:

```text
                           STADI DI CURRICULUM
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ STADIO 1: Micro-Board (6 città, 8 tratte, 4 ticket)                     │
 │   Obiettivo: Apprendimento rapido delle meccaniche di pesca, colore     │
 │              e completamento tratte senza interferenze complesse.       │
 │   Avversario: RandomAgent                                               │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │ Gate: WinRate ≥ 75%, TicketRate ≥ 80%
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ STADIO 2: Small-Board (14 città, 20 tratte, 10 ticket)                  │
 │   Obiettivo: Chaining multi-tratta, gestione vagoni e contesa risorse.  │
 │   Avversario: GreedyAgent & StrategicAgent                              │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │ Gate: WinRate ≥ 70%, TicketRate ≥ 75%
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ STADIO 3: Full USA Standard (36 città, 100+ tratte, 30 ticket)          │
 │   Obiettivo: Strategia globale, pianificazione a lungo termine,        │
 │              blocco tattico e self-play competitivo.                    │
 │   Avversario: PPO Pool & AlphaZero Self-Play                            │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Architettura del Software e Moduli

I componenti software della Fase 12 sono posizionati rispettando la rigida separazione architetturale del progetto:

```text
src/
├── rl/
│   ├── policy_value_net.py        # PolicyValueNetwork (Dual-head ResNet/MLP + Masking)
│   ├── alphazero_search.py        # NeuralMCTSEngine (PUCT, Dirichlet, No Rollout)
│   ├── alphazero_trainer.py       # AlphaZeroTrainer, SelfPlayReplayBuffer, Training Loop
│   ├── opponent_model.py          # BayesianTicketBeliefTracker, DetourEngine, BeliefWeightedDeterminizer
│   └── curriculum.py              # CurriculumManager, StageConfig, Transition Gates
│
├── agents/
│   └── neural_mcts_agent.py       # NeuralMCTSAgent, OpponentAwareMCTSAgent
│
└── evaluation/
    └── phase12_benchmark.py       # Scientific Cross-Paradigm Benchmark Suite & Report Generator
```

---

## 4. Dettaglio dei Componenti e Interfacce

### 4.1 `PolicyValueNetwork` (`src/rl/policy_value_net.py`)
- **Input:** Tensor di osservazione `[B, obs_dim]`, Mask Tensor `[B, action_dim]`.
- **Trunk Condiviso:** 2 blocchi residuali con Linear, LayerNorm, ReLU e skip connections.
- **Policy Head:** Linear `[hidden_dim, action_dim]` $\to$ applicazione maschera con logit $-10^8$ per azioni invalide $\to$ Softmax.
- **Value Head:** Linear `[hidden_dim, 64]` $\to$ ReLU $\to$ Linear `[64, 1]` $\to$ Tanh $\to v \in [-1, 1]$.
- **Metodi principali:**
  - `forward(obs, action_mask) -> (policy_probs, value)`
  - `evaluate_state(obs, action_mask) -> (policy_probs, value)` (versione inferenza senza gradiente)
  - `save(path)` / `load(path)`

### 4.2 `NeuralMCTSEngine` (`src/rl/alphazero_search.py`)
- Integra il motore di ricerca ad albero con la rete neurale.
- **Parametri:** `c_puct=1.5`, `dirichlet_alpha=0.3`, `dirichlet_eps=0.25`, `num_simulations=50`.
- **Algoritmo di Ricerca:**
  1. Determinizzazione dello stato (uniforme o belief-weighted);
  2. Selezione PUCT fino al primo nodo foglia non ancora espanso;
  3. Valutazione foglia tramite `PolicyValueNetwork` (nessun rollout random);
  4. Espansione simultanea di tutte le azioni legali con prior $P(s, a)$;
  5. Backpropagation del valore $V(s_{\text{leaf}})$ lungo il cammino percorso con inversione di segno alternata per il giocatore corrente.
- **Output:** Mappa di frequenza di visita $N(s, a)$ e distribuzione target $\boldsymbol{\pi}(s)$.

### 4.3 `AlphaZeroTrainer` (`src/rl/alphazero_trainer.py`)
- **Gestione Self-Play:** Generazione parallela o sequenziale di partite giocate da `NeuralMCTS` contro se stesso.
- **Buffer di Replay:** Memorizzazione di tuple $(s_t, \mathbf{m}_t, \boldsymbol{\pi}_t, z_t)$.
- **Ottimizzazione:** Adam optimizer con weight decay e gradient clipping su loss congiunta:
  $$\mathcal{L} = \text{MSE}(z, v) + \text{CrossEntropy}(\boldsymbol{\pi}, \mathbf{p}) + c_{\text{reg}} \|\theta\|^2$$
- **Checkpointing & Snapshots:** Salvataggio automatico dei modelli per storicizzazione nel pool di self-play.

### 4.4 `BayesianTicketBeliefTracker` & `OpponentModel` (`src/rl/opponent_model.py`)
- **`GraphDetourEngine`:** Calcola e memorizza le matrici di cammini minimi (Dijkstra) per tutte le coppie di città e per tutti i destination ticket della mappa attiva.
- **`BayesianTicketBeliefTracker`:**
  - Inizializza prior uniforme su $\mathcal{T}$;
  - All'evento pubblico `route_claimed(player_id, route)`, aggiorna la distribuzione bayesiana $P(T_k \mid \mathcal{E})$;
  - Espone `get_ticket_probabilities() -> Dict[Ticket, float]`;
  - Espone `get_top_k_tickets(k=3) -> List[Tuple[Ticket, float]]`.
- **`BeliefWeightedDeterminizer`:**
  - Sostituisce il campionatore uniforme di `mcts_determinization.py`;
  - Pesa i ticket assegnati all'avversario tramite $P(T_k \mid \mathcal{E}_{\text{opp}})$.
- **`TacticalBlocker`:**
  - Rileva tratte ad alta criticità per l'avversario e fornisce bonus di valutazione o raccomandazioni di intercettazione.

### 4.5 `CurriculumManager` (`src/rl/curriculum.py`)
- Definisce `CurriculumStage` con mappa target, tipo avversario, metriche minime di promozione e numero minimo di partite di validazione.
- Gestisce l'avanzamento automatico di stadio e genera report di progressione didattica.

### 4.6 `NeuralMCTSAgent` & `OpponentAwareMCTSAgent` (`src/agents/neural_mcts_agent.py`)
- `NeuralMCTSAgent`: Esegue la ricerca neurale AlphaZero online con determinizzazione base.
- `OpponentAwareMCTSAgent`: Combina la ricerca neurale con il `BayesianTicketBeliefTracker` e la `BeliefWeightedDeterminization`.

### 4.7 `Phase12ScientificBenchmark` (`src/evaluation/phase12_benchmark.py`)
- Esegue un torneo scientifico esteso tra tutti i paradigmi di agenti del laboratorio:
  1. `RandomAgent`
  2. `GreedyAgent`
  3. `StrategicAgent`
  4. `DQNAgent`
  5. `PPOAgent`
  6. `RecurrentPPOAgent`
  7. `MCTSAgent` (Euristico)
  8. `NeuralMCTSAgent` (AlphaZero)
  9. `OpponentAwareMCTSAgent` (Belief MCTS)
- Calcola: Elo Ratings, Win Rates, Delta Punteggi, Ticket Completion Rate, Recall@1 e Recall@3 del Belief Tracker, Latenza Decisionale media (ms).
- Genera automaticamente un report scientifico testuale e JSON con le conclusioni didattiche dell'intero laboratorio.

---

## 5. Criteri di Accettazione e Piano di Test (TDD)

Per considerare la **Fase 12** completata con successo, la suite di test deve verificare i seguenti 6 Criteri Scientifici:

| # | Criterio Scientifico | Test Module | Risultato Atteso |
|---|---|---|---|
| **C1** | **Policy-Value Network & Masking** | `tests/rl/test_policy_value_net.py` | La rete calcola correttamente shape, gradienti, maschera delle azioni illegali (probabilità 0.0) e output del valore scalare in $[-1, 1]$. |
| **C2** | **AlphaZero PUCT & Search Engine** | `tests/rl/test_alphazero_search.py` | La ricerca PUCT espande i nodi con prior neurale, applica il rumore di Dirichlet alla radice e restituisce distribuzioni di visita valide $\boldsymbol{\pi}$. |
| **C3** | **AlphaZero Self-Play Trainer** | `tests/rl/test_alphazero_trainer.py` | Il loop di training genera traiettorie self-play, memorizza tuple nel buffer e riduce la loss congiunta (policy cross-entropy + value MSE). |
| **C4** | **Bayesian Ticket Belief Tracker** | `tests/rl/test_opponent_model.py` | Il modello bayesiano assegna probabilità a posteriori significativamente più alte ai veri ticket quando l'avversario rivendica tratte su tali cammini. |
| **C5** | **Belief-Weighted Determinization** | `tests/rl/test_opponent_model.py` | La determinizzazione orientata dal belief campiona coerentemente i mondi possibili senza alcuna violazione o leakage di informazioni nascoste. |
| **C6** | **Phase 12 Acceptance & Cross-Paradigm Benchmark** | `tests/rl/test_phase12_acceptance.py` | `NeuralMCTSAgent` e `OpponentAwareMCTSAgent` sconfiggono consistentemente gli agenti baseline, il belief model raggiunge elevata accuratezza predittiva e il report scientifico viene esportato correttamente. |

---

## 6. Lezione Didattica & Sintesi Teorica del Laboratorio

### 6.1 Perché la combinazione Ricerca + Reti Neurali (AlphaZero) supera il Model-Free puro
Nel Deep RL model-free puro (come DQN o PPO standard), la rete deve memorizzare l'intera policy complessa nei suoi pesi $\theta$.
In giochi a grafo combinatorio come *Ticket to Ride*, lo spazio degli stati è nell'ordine di $10^{30}$ configurazioni possibili.
L'algoritmo AlphaZero adotta un paradigma profondamente diverso:
1. **La rete neurale funge da "intuizione" e "valutazione euristica":** suggerisce le mosse più promettenti ($P(s, a)$) e stima la qualità della posizione ($V(s)$).
2. **La ricerca ad albero MCTS funge da "calcolo e ragionamento esplicito":** simula in avanti gli effetti delle scelte, correggendo eventuali errori locali dell'intuizione neurale.
3. **Il Self-Play funge da "laboratorio di auto-miglioramento":** l'albero di ricerca produce decisioni sistematicamente più forti della rete grezza ($\boldsymbol{\pi}_{\text{mcts}} > \mathbf{p}_\theta$); addestrando la rete su $\boldsymbol{\pi}_{\text{mcts}}$, la rete diventa iterativamente più intelligente ad ogni ciclo.

### 6.2 Risolvere l'Informazione Imperfetta: Da AlphaZero a Belief-Weighted AlphaZero
AlphaZero classico eccelle in Go e Scacchi perché sono giochi a **informazione perfetta**.
In *Ticket to Ride*, l'incertezza sui ticket segreti dell'avversario introduce il rischio di *Strategy Fusion* (cercare una singola mossa ottima per tutti i mondi possibili quando in realtà il mondo è uno solo specifico).
L'introduzione del **Bayesian Ticket Belief Tracker** trasforma un problema a informazione imperfetta sconosciuta in un problema a **informazione parzialmente inferita**, concentrando il budget computazionale di MCTS sulle ipotesi strategiche più plausibili.
