# Specifica di Design & Lezione Didattica: Fase 8 — Parziale Osservabilità & Recurrent PPO (LSTM)

**Versione Documento:** 1.0.0  
**Data:** 2026-08-20  
**Fase:** 8 (Partial Observability & Recurrent PPO)  
**Stato:** Approvato per l'Implementazione  
**Lingua:** Italiano  

---

# 1. Lezione Teorica: Parziale Osservabilità e Memoria nel Reinforcement Learning

## 1.1 Fondamenti dei POMDP (Partially Observable Markov Decision Processes)

Nei classici problemi di Reinforcement Learning formulati come **MDP** (Markov Decision Process), l'agente osserva in ogni istante lo stato completo e reale del mondo $s_t \in \mathcal{S}$. In un MDP vale la proprietà fondamentale di Markov:

$$\mathbb{P}(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0) = \mathbb{P}(s_{t+1} \mid s_t, a_t)$$

Ovvero: *il futuro è condizionatamente indipendente dal passato, dato il presente*.

Tuttavia, nella maggioranza dei giochi strategici reali (come Ticket to Ride) e nei problemi di robotica e finanza, lo stato reale $s_t$ non è accessibile per intero. Il sistema è un **POMDP**, formalmente definito dalla 7-tupla:

$$\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

dove:
- $\mathcal{S}$ è lo spazio degli stati veri (*True State*): include le carte segrete in mano a ogni giocatore, i biglietti destinazione segreti e l'ordine esatto delle carte nel mazzo coperto;
- $\mathcal{A}$ è lo spazio delle azioni legali;
- $\mathcal{T}(s' \mid s, a)$ è la funzione di transizione tra stati veri;
- $\mathcal{R}(s, a)$ è la funzione scalare di ricompensa;
- $\Omega$ è lo spazio delle osservazioni (*Observation Space*);
- $\mathcal{O}(o \mid s', a)$ è la funzione di emissione dell'osservazione: proietta lo stato vero $s'$ nell'osservazione visibile $o \in \Omega$ dal punto di vista del giocatore corrente;
- $\gamma \in [0, 1)$ è il fattore di sconto temporale.

### Perché la singola osservazione $o_t$ rompe la proprietà di Markov?
Quando passiamo dallo stato vero $s_t$ all'osservazione parziale $o_t$, perdiamo la proprietà di Markov:

$$\mathbb{P}(s_{t+1} \mid o_t, a_t) \neq \mathbb{P}(s_{t+1} \mid o_t, a_t, o_{t-1}, a_{t-1}, \dots, o_0)$$

**Esempio didattico in Ticket to Ride:**
Immaginiamo che al turno $t$, l'osservazione $o_t$ mostri che l'avversario possiede 6 carte in mano.
- Un agente feed-forward stateless (MLP) vede solo il vettore istantaneo $o_t$: sa che l'avversario ha 6 carte, ma non ha alcuna idea di quali colori siano.
- Un agente con memoria (ricorrente) ha osservato la sequenza temporale $\mathcal{H}_t = (o_0, a_0, o_1, a_1, \dots, o_t)$. Se nei tre turni precedenti l'avversario ha pescato 4 carte rosse e 2 locomotive scoperte dal tavolo, l'agente ricorrente sa che l'avversario sta quasi certamente preparando una tratta rossa lunga o contesa (es. New York - Boston o Helena - Duluth).

Per prendere decisioni ottimali in un POMDP, la politica deve basarsi non sulla singola osservazione $o_t$, ma sull'intera storia $\mathcal{H}_t$, formando un **Belief State** $\mathbb{P}(s_t \mid \mathcal{H}_t)$.

---

## 1.2 Reti Neurali Ricorrenti (LSTM) come Approssimatori del Belief State

Poiché la lunghezza della storia $\mathcal{H}_t$ cresce linearmente con il numero di turni (rendendo impraticabile la concatenazione di tutti i vettori passati), utilizziamo una rete neurale ricorrente **LSTM** (Long Short-Term Memory).

L'LSTM comprime la storia passata in un vettore latente compatto $(h_t, c_t)$ aggiornato ricorsivamente ad ogni passo:

$$(h_t, c_t) = \text{LSTM}(e(o_t), (h_{t-1}, c_{t-1}))$$

dove:
- $e(o_t) = \text{Tanh}(W_e o_t + b_e)$ è l'embedding dell'osservazione estratto da un encoder lineare;
- $h_t \in \mathbb{R}^{d_{lstm}}$ è lo **hidden state**, che funge da rappresentazione latente del belief state e viene passato alle teste Actor e Critic;
- $c_t \in \mathbb{R}^{d_{lstm}}$ è il **cell state**, che funge da nastro di memoria a lungo termine per preservare informazioni su orizzonti estesi.

```text
                  o_t (Osservazione Corrente)
                              │
                              ▼
                   Linear Feature Encoder
                              │
                              ▼
   (h_{t-1}, c_{t-1}) ──► ┌──────────────┐
                 │        │  Cella LSTM  │ ──► (h_t, c_t)
                 │        └──────┬───────┘
                 │               │
                 │               ▼
                 │        h_t (Belief State)
                 │               │
                 │        ┌──────┴──────┐
                 │        ▼             ▼
                 │   Actor Head    Critic Head
                 │  (Logits + Mask)   V(h_t)
                 │        │
                 ▼        ▼
          (1 - done)   π(a_t | h_t)
```

---

## 1.3 Inizializzazione Ortogonale e Stabilità Numerica (Principi CleanRL)

Nelle reti ricorrenti in RL, l'accumulo dei gradienti nel tempo può portare rapidamente a fenomeni di *gradient explosion* o *gradient vanishing*. Per garantire massima stabilità durante l'addestramento:

1. **Inizializzazione Ortogonale dei Pesi**:
   - Per i layer lineari dell'encoder: `nn.init.orthogonal_(layer.weight, gain=sqrt(2))` e `bias = 0.0`;
   - Per i pesi ricorrenti della cella LSTM: `nn.init.orthogonal_(lstm.weight_ih_l0)` e `nn.init.orthogonal_(lstm.weight_hh_l0)`;
   - Per la testa Actor: `nn.init.orthogonal_(actor.weight, gain=0.01)` per iniziare con una distribuzione d'azione quasi uniforme;
   - Per la testa Critic: `nn.init.orthogonal_(critic.weight, gain=1.0)`.

2. **Action Masking Numerico con Distribuzione Categorica**:
   I logit non validi vengono forzati a un valore estremamente negativo (es. $-10^8$) prima della softmax:
   $$\text{logits}_{\text{masked}}[i] = \begin{cases} \text{logits}[i] & \text{se } \text{mask}[i] = \text{True} \\ -10^8 & \text{se } \text{mask}[i] = \text{False} \end{cases}$$
   $$\pi(a_i \mid h_t) = \frac{\exp(\text{logits}_{\text{masked}}[i])}{\sum_{j \in \text{valid}} \exp(\text{logits}_{\text{masked}}[j])}$$
   Questo garantisce che la probabilità per le azioni illegali sia rigorosamente zero e che il gradiente scorra esclusivamente attraverso le azioni legali.

---

## 1.4 Truncated BPTT e Gestione dei Confini Episodici

L'ottimizzazione di PPO con politiche ricorrenti differisce dalle politiche MLP in due punti fondamentali:

1. **Reset Episodico dello Stato Nascosto**:
   Durante il rollout e l'ottimizzazione, quando un episodio finisce (`done == True`), la memoria dell'episodio concluso non deve riversarsi nella nuova partita:
   $$h_t \leftarrow (1 - \text{done}_t) \cdot h_t, \quad c_t \leftarrow (1 - \text{done}_t) \cdot c_t$$

2. **Chunking dei Minibatch Sequenziali (Truncated BPTT)**:
   In PPO standard (MLP), le transizioni nel buffer vengono mescolate e campionate in minibatch in modo completamente casuale.
   In Recurrent PPO, **mescolare singoli step distruggerebbe la sequenzialità temporale dell'LSTM**.
   Il `RecurrentRolloutBuffer` memorizza quindi lo stato $(h_t, c_t)$ presente prima di ogni transizione e genera minibatch composti da **chunk di sequenze contigue** di lunghezza $T_{seq}$ (es. 8 o 16 timestep), fornendo alla cella LSTM lo stato iniziale $(h_0, c_0)$ registrato all'inizio di ciascun chunk.

---

# 2. Architettura & Flusso dei Dati

```
                                  ┌────────────────────────┐
                                  │  GameState (True State) │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                           ┌──────────────────────────────────────┐
                           │ ObservationV1 (POMDP Anti-Leakage)   │
                           │ - Zero leak carte mano avversari     │
                           │ - Zero leak biglietti avversari      │
                           │ - Zero leak mazzo coperto            │
                           └──────────────────┬───────────────────┘
                                              │
                                              ▼
                                ┌───────────────────────────┐
                                │ RecurrentMaskedActorCritic│
                                │ Linear Encoder -> LSTM    │
                                │ -> Masked Actor / Critic  │
                                └─────────────┬─────────────┘
                                              │
                         ┌────────────────────┴────────────────────┐
                         ▼                                         ▼
           ┌───────────────────────────┐             ┌───────────────────────────┐
           │ MaskedRecurrentPPOTrainer │             │    RecurrentPPOAgent      │
           │ - RecurrentRolloutBuffer  │             │ - Stato (h, c) persistente│
           │ - Truncated BPTT Chunks   │             │ - Tournament / Evaluator  │
           │ - GAE & Clipped Loss      │             │ - Salvataggio Checkpoint  │
           └───────────────────────────┘             └───────────────────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │    POMDPBenchmarkRunner       │
                              │  MLP PPO vs LSTM PPO vs Bots  │
                              │  Report JSON & Markdown       │
                              └───────────────────────────────┘
```

---

# 3. Specifiche Dettagliate dei Componenti

### 3.1 Verifica Formale di Anti-Leakage POMDP
- **File:** `tests/environment/test_pomdp_anti_leakage.py`
- **Invarianti da verificare formalmente:**
  1. **Invarianza Carte Coperte Avversario**: Modificando i colori delle carte in mano all'avversario mantenendo inalterato il totale, il vettore generato da `ObservationV1` per il Giocatore 0 rimane rigorosamente identico bit a bit.
  2. **Invarianza Biglietti Nascosti Avversario**: Aggiungendo, eliminando o scambiando i destination tickets in mano all'avversario, il vettore di osservazione del Giocatore 0 non subisce alcuna alterazione.
  3. **Invarianza Ordine Mazzo Nascosto**: Permutando/mescolando le carte del mazzo di pesca coperto, l'osservazione del Giocatore 0 rimane identica (viene osservata solo la lunghezza totale del mazzo).
  4. **Sensibilità alle Informazioni Pubbliche**: La modifica di una carta scoperta sul tavolo produce una variazione corretta e circoscritta nello slice delle 5 carte visibili.

---

### 3.2 Architettura Neurale Ricorrente (`RecurrentMaskedActorCritic`)
- **File:** `src/rl/lstm_ppo.py`
- **Specifiche Tecniche:**
  - `input_dim`: Dimensione del vettore di osservazione (calcolata dinamicamente su mappa USA standard, es. 323).
  - `action_dim`: Dimensione dello spazio delle azioni discrete (es. 160).
  - `hidden_dim`: Dimensione del layer di embedding lineare (default: 128).
  - `lstm_hidden_dim`: Dimensione della memoria ricorrente LSTM (default: 128).
- **Sottoreticolati:**
  - `encoder`: `nn.Sequential(layer_init(nn.Linear(input_dim, hidden_dim), sqrt(2)), nn.Tanh())`
  - `lstm`: `nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)` con inizializzazione ortogonale dei pesi `weight_ih_l0` e `weight_hh_l0` e bias inizializzati a zero.
  - `actor`: `layer_init(nn.Linear(lstm_hidden_dim, action_dim), 0.01)`
  - `critic`: `layer_init(nn.Linear(lstm_hidden_dim, 1), 1.0)`
- **Metodi Principali:**
  - `forward(obs_seq, hidden_state)`:
    - Input: tensore osservazioni `(batch_size, seq_len, input_dim)` e stato nascosto `hidden_state = (h, c)` con forma `(1, batch_size, lstm_hidden_dim)`.
    - Output: `logits (batch_size, seq_len, action_dim)`, `values (batch_size, seq_len, 1)`, `new_hidden`.
  - `get_action_and_value(obs, hidden_state, action_mask=None, action=None, deterministic=False)`:
    - Esegue il forward pass unificato per inferenza (`batch_size=1, seq_len=1`) o per mini-batch di sequenze di training.
    - Applica l'action masking sostituendo con $-10^8$ i logit non validi.
    - Calcola la distribuzione `Categorical(logits=masked_logits)`.
    - Restituisce `(action, log_prob, entropy, value, new_hidden)`.

---

### 3.3 Buffer di Rollout Ricorrente (`RecurrentRolloutBuffer`)
- **File:** `src/rl/rollout.py` (e re-export in `src/rl/lstm_ppo.py`)
- **Buffer Allocati:**
  - `obs_buf`: `(capacity, obs_dim)` float32
  - `actions_buf`: `(capacity,)` int64
  - `rewards_buf`: `(capacity,)` float32
  - `values_buf`: `(capacity,)` float32
  - `log_probs_buf`: `(capacity,)` float32
  - `dones_buf`: `(capacity,)` bool
  - `masks_buf`: `(capacity, action_dim)` bool
  - `h_buf`: `(capacity, lstm_hidden_dim)` float32 (stato $h$ prima del timestep $t$)
  - `c_buf`: `(capacity, lstm_hidden_dim)` float32 (stato $c$ prima del timestep $t$)
- **Generatore di Minibatch Ricorrenti (`generate_recurrent_minibatches`):**
  - Suddivide la traiettoria da $N$ step in sequenze contigue di lunghezza $T_{seq}$ (es. 8 o 16 step).
  - Estrae lo stato iniziale $(h_0, c_0)$ registrato all'inizio di ciascuna sequenza.
  - Emette dizionari contenenti:
    - `obs`: tensore `(batch_size, seq_len, obs_dim)`
    - `actions`: tensore `(batch_size, seq_len)`
    - `old_log_probs`: tensore `(batch_size, seq_len)`
    - `values`: tensore `(batch_size, seq_len)`
    - `advantages`: tensore `(batch_size, seq_len)` (normalizzati su intero rollout)
    - `returns`: tensore `(batch_size, seq_len)`
    - `action_masks`: tensore `(batch_size, seq_len, action_dim)`
    - `dones`: tensore `(batch_size, seq_len)`
    - `initial_h`: tensore `(1, batch_size, lstm_hidden_dim)`
    - `initial_c`: tensore `(1, batch_size, lstm_hidden_dim)`

---

### 3.4 Trainer Recurrent PPO (`MaskedRecurrentPPOTrainer`)
- **File:** `src/rl/lstm_ppo.py`
- **Caratteristiche di Ottimizzazione:**
  - Raccolta traiettorie con tracciamento dello stato nascosto ad ogni step e azzeramento a termine episodio: `hidden = (1.0 - done) * hidden`.
  - Calcolo Generalized Advantage Estimation (GAE) tramite `compute_gae`.
  - Ottimizzazione multi-epoch con minibatch di sequenze temporali.
  - Surrogate Clipped Objective di PPO:
    $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
  - Value Loss con clipping opzionale $L^{VF}(\theta)$ ed Entropy Bonus $S[\pi_\theta]$.
  - Gradient clipping a norma massima 0.5.
  - Early stopping su Target KL (`target_kl`) e linear learning rate annealing.
  - Metodi `save(path)` e `load(path)` per il checkpointing dei modelli.

---

### 3.5 Agente Recurrent PPO (`RecurrentPPOAgent`)
- **File:** `src/agents/recurrent_ppo_agent.py`
- **Specifiche:**
  - Eredita da `BaseAgent(name="RecurrentPPOAgent")`.
  - Mantiene internamente `self.current_hidden: tuple[torch.Tensor, torch.Tensor]` inizializzato a zeri.
  - `reset()`: azzera `self.current_hidden` all'inizio di una nuova partita.
  - `select_action(observation, action_mask, deterministic=True)`: esegue un singolo step di forward pass aggiornando `self.current_hidden` e restituendo l'azione discreta scelta.
  - `act(state, valid_actions, board)`: codifica lo stato in osservazione con `ObservationV1`, calcola la maschera d'azione, invoca `select_action` e converte l'indice nell'oggetto di dominio `Action`.
  - Compatibilità diretta con `Evaluator` e `Tournament`.

---

### 3.6 Runner di Benchmark Scientifico (`POMDPBenchmarkRunner`)
- **File:** `src/evaluation/pomdp_benchmark.py`
- **Protocollo Sperimentale:**
  - Addestra con lo stesso seed sia `MaskedPPOTrainer` (MLP stateless baseline) sia `MaskedRecurrentPPOTrainer` (LSTM recurrent policy).
  - Valuta entrambi i modelli in scontri diretti testa a testa (alternando il primo giocatore) e contro i baseline deterministici (`RandomAgent`, `GreedyAgent`, `StrategicAgent`).
  - Calcola e confronta le metriche comportamentali e di performance:
    - **Win Rate** e **Score Differential** testa a testa (LSTM vs MLP);
    - **Win Rate** contro i baseline;
    - **Tasso di Completamento Biglietti Destinazione**;
    - **Efficienza di Reclamo Tratte**;
    - **Numero Medio di Turni per Partita**;
    - **Curve di Apprendimento & Stabilità delle Perdite**.
  - Genera il report strutturato JSON e il report Markdown accademico/didattico in `experiments/results/phase8_report.md`.

---

# 4. Criteri di Accettazione & Test Suite (Fase 8)

### Criterio 1: Verifica Anti-Leakage POMDP
I test di invarianza formale certificano che carte coperte, biglietti avversari e ordine del mazzo non hanno alcun impatto sul vettore di osservazione.

### Criterio 2: Validazione dell'Architettura Ricorrente
`RecurrentMaskedActorCritic` gestisce correttamente step singoli e sequenze batch, e l'Action Masking impone rigorosamente probabilità zero per le azioni illegali.

### Criterio 3: Buffer di Rollout e Minibatch di Sequenze
`RecurrentRolloutBuffer` assembla sequenze contigue con gli stati $(h_0, c_0)$ associati senza corruzione di forma o memoria.

### Criterio 4: Riproducibilità Deterministica
Due istanze di `MaskedRecurrentPPOTrainer` inizializzate con lo stesso seed producono metriche di perdita e pesi neurali numericamente identici.

### Criterio 5: Convergenza e Superiorità sul Random Baseline
L'agente `RecurrentPPOAgent` addestrato supera il baseline casuale con una percentuale di vittoria $\ge 65\%$.

### Criterio 6: Generazione del Benchmark Comparativo MLP vs LSTM
`POMDPBenchmarkRunner` esegue lo studio comparativo automatico e genera correttamente i report JSON e Markdown con tutte le metriche strutturate.

---

# 5. File Creati e Modificati

```
src/
├── agents/
│   ├── recurrent_ppo_agent.py          # Implementazione di RecurrentPPOAgent
│   └── __init__.py                     # Export di RecurrentPPOAgent
├── rl/
│   ├── lstm_ppo.py                     # RecurrentMaskedActorCritic & MaskedRecurrentPPOTrainer
│   ├── rollout.py                      # RecurrentRolloutBuffer con minibatch di sequenze
│   └── __init__.py                     # Export classi ricorrenti
└── evaluation/
    ├── pomdp_benchmark.py              # POMDPBenchmarkRunner (studio MLP vs LSTM)
    └── __init__.py                     # Export POMDPBenchmarkRunner

tests/
├── environment/
│   └── test_pomdp_anti_leakage.py      # Test formali di invarianza anti-leakage
└── rl/
    ├── test_recurrent_ppo.py           # Test unitari per rete ricorrente, buffer e trainer
    └── test_phase8_acceptance.py       # Suite completa di test di accettazione Fase 8
```
