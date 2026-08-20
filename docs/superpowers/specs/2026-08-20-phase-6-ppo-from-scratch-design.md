# Spec Tecnica e Didattica - Fase 6: PPO From Scratch, CleanRL Details & Advanced Benchmarking

## 1. Visione, Filosofia e Obiettivi Didattici della Fase 6

La **Fase 6: PPO From Scratch** ha l'obiettivo di elevare l'implementazione di **Proximal Policy Optimization (PPO)** a uno standard di eccellenza scientifica e ingegneristica conforme alla letteratura moderna e ai principi di **CleanRL** ("The 37 Implementation Details of PPO").

```text
                        ┌────────────────────────────────────────────────────────┐
                        │        FASE 6: PPO FROM SCRATCH & BENCHMARKING         │
                        └────────────────────────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┴────────────────────────────────┐
                    ▼                                                                 ▼
    ┌───────────────────────────────┐                                 ┌───────────────────────────────┐
    │     PPO CORE ENHANCEMENTS     │                                 │     VECTORIZED ROLLOUTS       │
    │  • Inizializzazione Ortogonale│                                 │  • Multi-env batching         │
    │  • Linear LR Annealing        │                                 │  • VectorRolloutBuffer (T,N,D)│
    │  • Value Function Clipping    │                                 │  • GAE parallelo accelerato   │
    │  • Target KL Early Stopping   │                                 │  • Throughput moltiplicato    │
    │  • Advantage Normalization    │                                 │                               │
    └───────────────────────────────┘                                 └───────────────────────────────┘
                    │                                                                 │
                    └────────────────────────────────┬────────────────────────────────┘
                                                     │
                                                     ▼
                        ┌────────────────────────────────────────────────────────┐
                        │          RIGOROUS BENCHMARK & ABLATION STUDY           │
                        │    Vs Random, Greedy, Strategic (Mini & USA maps)      │
                        │    Misura empirica dell'impatto dei dettagli CleanRL   │
                        └────────────────────────────────────────────────────────┘
```

---

### 1.1 Differenza Chiave rispetto alle Fasi Precedenti

* **Fase 4 (First RL)**: Ha implementato i primi prototipi funzionanti di DQN e PPO per verificare la compatibilità con l'ambiente Gymnasium.
* **Fase 5 (Web Lab)**: Ha costruito l'infrastruttura di visualizzazione e introspezione web (Replay, Brain Viewer, WebSocket live telemetry, Tournament Arena).
* **Fase 6 (PPO From Scratch & Benchmark)**: Si concentra sul **motore algoritmico puro**:
  1. Integra **tutti i dettagli matematici fondamentali di CleanRL** per eliminare ogni instabilità numerica e massimizzare l'efficienza campionaria.
  2. Introduce **ambienti vettorizzati e buffer multidimensionali** per abbattere i tempi di addestramento su mappe grandi (USA).
  3. Costruisce una **suite di benchmark formale con Ablation Study** per quantificare sperimentalmente il contributo di ciascun elemento matematico.

---

## 2. Fondamenti Teorici e Dettagli Matematici di PPO (CleanRL Standard)

In questa sezione formalizziamo la matematica e le scelte implementative rigorose di PPO.

---

### 2.1 Inizializzazione Ortogonale e Scaling dei Logits

Nelle reti neurali profonde per RL, l'inizializzazione standard (es. Xavier o Kaiming uniforme/normale) induce un'elevata varianza nelle attivazioni e nei gradienti iniziali, provocando scelte di policy premature o collasso dell'entropia.

Adottiamo l'**inizializzazione ortogonale** (`nn.init.orthogonal_`) combinata con bias azzerati (`nn.init.constant_(bias, 0.0)`):
* **Strati lineari nascosti (con attivazione Tanh)**:
  $$\text{gain} = \sqrt{2} \approx 1.4142$$
* **Policy Output Layer (Logits delle azioni)**:
  $$\text{gain} = 0.01$$
  *Motivazione*: Un gain ridotto garantisce che all'inizio dell'addestramento i logits $z_a$ siano prossimi allo zero, rendendo la distribuzione $\pi_\theta(a|s)$ quasi perfettamente uniforme sulle sole azioni lecite (grazie al masking). Ciò preserva la massima entropia ed esplorazione nei primi episodi.
* **Value Output Layer (Critico scalare)**:
  $$\text{gain} = 1.0$$

---

### 2.2 Linear Learning Rate Annealing

Nel corso del training, mantenere un tasso di apprendimento fisso può ostacolare la convergenza fine della policy. Il decadimento lineare garantisce grandi passi esplorativi all'inizio e stabilità man mano che il budget di campionamento si esaurisce:

$$\alpha_t = 1.0 - \frac{t}{T_{\text{total}}}, \quad \text{lr}_t = \text{lr}_0 \cdot \max(\alpha_t, 0.0)$$

Ad ogni iterazione di aggiornamento, l'ottimizzatore Adam aggiorna il learning rate del gruppo di parametri.

---

### 2.3 Value Function Loss Clipping (Clipped Value Loss)

Come per la policy ratio, l'aggiornamento della funzione di valore $V_\theta(s)$ può subire oscillazioni distruttive se un singolo batch di transizioni produce gradienti sproporzionati.

Definiamo il valore predetto non clippato e clippato rispetto al valore calcolato al momento del rollout $V_{\text{old}}(s)$:

$$V^{\text{clipped}}_\theta(s) = V_{\text{old}}(s) + \text{clip}\left(V_\theta(s) - V_{\text{old}}(s), -\epsilon_{vf}, \epsilon_{vf}\right)$$

La perdita del critico clippata è il massimo tra le due discrepanze quadratiche rispetto ai ritorni empirici $R_t$:

$$\mathcal{L}^{VF}(\theta) = \frac{1}{2} \mathbb{E} \left[ \max \left( (V_\theta(s_t) - R_t)^2, (V^{\text{clipped}}_\theta(s_t) - R_t)^2 \right) \right]$$

In questo modo, penalizziamo gli aggiornamenti che allontanano eccessivamente la nuova stima del valore da quella precedente.

---

### 2.4 Target KL Divergence & Early Stopping

Durante le $K$ epoche di ottimizzazione PPO sullo stesso rollout di dati, la policy $\pi_\theta$ si discosta progressivamente dalla vecchia policy di campionamento $\pi_{\theta_{\text{old}}}$. Se la divergenza di Kullback-Leibler supera una soglia critica, la policy rischia un degrado irreversibile.

Calcoliamo l'approssimazione efficiente della KL (sviluppata da John Schulman):

$$d_{\text{KL}} \approx \mathbb{E} \left[ (r_t(\theta) - 1) - \log r_t(\theta) \right], \quad \text{dove } r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

Se durante le sotto-epoche di un update $d_{\text{KL}} > \text{target\_kl}$ (tipicamente impostato a $0.015 \sim 0.02$), il loop di ottimizzazione del batch corrente si interrompe immediatamente (**Early Stopping**), passando alla successiva fase di rollout.

---

### 2.5 Normalizzazione dei Vantaggi (Advantage Normalization)

Per stabilizzare la magnitudo dei gradienti di policy rispetto a ricompense di scale diverse:
$$\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + 10^{-8}}$$
La normalizzazione viene calcolata sui minibatch di ottimizzazione o sull'intero buffer di rollout.

---

### 2.6 Categorical Action Masking con Stabilità Numerica

Data la maschera booleana $\mathbf{M}(s) \in \{0, 1\}^{|\mathcal{A}|}$, i logits $z_a$ vengono trasformati prima della distribuzione categorica:
$$z'_a = \begin{cases} z_a & \text{se } M(s)_a = 1 \\ -10^8 & \text{se } M(s)_a = 0 \end{cases}$$
L'entropia $H(\pi)$ viene calcolata esclusivamente rispetto alle azioni lecite, evitando contributi spuri derivanti da $\log(0)$ o da azioni vietate.

---

## 3. Architettura dei Componenti della Fase 6

```text
src/
├── rl/
│   ├── networks.py          # MaskedActorCritic con orthogonal init e gain personalizzati
│   ├── rollout.py           # VectorRolloutBuffer multidimensionale (T, N, D)
│   ├── advantage.py         # compute_gae vettorizzato
│   ├── ppo.py               # MaskedPPOTrainer potenziato (CleanRL standard, LR annealing, VF clip, Target KL)
│   └── dqn.py               # Masked Double DQN esistente (mantenuto e validato)
│
├── evaluation/
│   ├── benchmark.py         # PPOBenchmarkRunner & AblationEngine
│   ├── evaluator.py         # Evaluator testa a testa
│   ├── tournament.py        # RoundRobinTournament
│   └── elo.py               # Sistema di calcolo Elo rating
│
└── api/
    ├── trainer_service.py   # Supporto alla telemetria PPO potenziata (KL, clip_frac, lr)
    └── websocket.py         # Canale live stream
```

---

### 3.1 VectorRolloutBuffer

Il nuovo buffer supporta una gestione a tensori compatti per memorizzare $T$ passi temporali su $N$ ambienti concorrenti:

* `obs_buf`: $\mathbb{R}^{T \times N \times D_{\text{obs}}}$
* `actions_buf`: $\mathbb{Z}^{T \times N}$
* `log_probs_buf`: $\mathbb{R}^{T \times N}$
* `rewards_buf`: $\mathbb{R}^{T \times N}$
* `dones_buf`: $\{0, 1\}^{T \times N}$
* `values_buf`: $\mathbb{R}^{T \times N}$
* `action_masks_buf`: $\{0, 1\}^{T \times N \times D_{\text{act}}}$

#### Minibatch Generation:
Il buffer appiattisce i tensori da $(T, N)$ a $(T \cdot N)$ campioni totali e genera permutazioni casuali per alimentare i minibatch di dimensione $B$:
$$N_{\text{minibatches}} = \frac{T \cdot N}{B}$$

---

### 3.2 PPOBenchmarkRunner & Ablation Study

Il modulo `src/evaluation/benchmark.py` esegue una suite completa di test prestazionali:

1. **Valutazione Comparativa Multi-Avversario**:
   - PPO vs `RandomAgent` ($100$ partite, seed controllati)
   - PPO vs `GreedyAgent` ($100$ partite, seed controllati)
   - PPO vs `StrategicAgent` ($100$ partite, seed controllati)
   - Calcolo di: Win Rate ($\% $), Mean Score Agente, Mean Score Avversario, Score Differential ($\Delta$), Ticket Completion ($\% $), Elo finale.
2. **Ablation Study Automatico**:
   - Allena varianti PPO su configurazioni speculari per misurare l'effetto di ogni componente:
     - `ppo_full`: PPO completo con tutte le ottimizzazioni CleanRL.
     - `ppo_no_ortho`: PPO con inizializzazione standard (non ortogonale).
     - `ppo_no_vf_clip`: PPO senza clipping della value function loss.
     - `ppo_no_lr_anneal`: PPO con learning rate costante.
3. **Report Generation**:
   - Salva i dati strutturati in `experiments/results/benchmark_phase6.json`.
   - Genera una tabella comparativa leggibile in markdown in `experiments/results/benchmark_phase6.md`.

---

## 4. Testing & Criteri di Accettazione

La suite di test per la Fase 6 si articolerà in:

1. **`tests/rl/test_networks.py`**:
   - Verifica dell'inizializzazione ortogonale (verificare che i pesi siano ortogonali $W W^T \approx I \cdot \text{gain}^2$).
   - Verifica dei gain specifici per actor ($\text{gain}=0.01$) e critic ($\text{gain}=1.0$).
2. **`tests/rl/test_vector_rollout.py`**:
   - Test di aggiunta dati e generazione minibatch su dimensioni $(T, N)$.
   - Test di calcolo GAE vettorizzato multi-ambiente.
3. **`tests/rl/test_ppo_cleanrl_details.py`**:
   - Test del Value Function Clipping (verificare che la loss clippata rispetti la bound $\epsilon_{vf}$).
   - Test del Target KL Early Stopping (simulare divergenza elevata e verificare l'interruzione anticipata delle epoche).
   - Test del Linear LR Annealing durante l'avanzamento dei timesteps.
4. **`tests/evaluation/test_benchmark_suite.py`**:
   - Test del runner di benchmark e generazione corretta dei report JSON/Markdown.
5. **`tests/rl/test_phase6_acceptance.py`**:
   - Acceptance test completo: PPO addestrato batte consistentemente `RandomAgent` con Win Rate $\ge 80\% $ e compete solidamente contro `GreedyAgent`.

---

## 5. Non-Obiettivi della Fase 6 (YAGNI & Guardrails)

* **No LSTM / Ricorrenza**: La memoria ricorrente appartiene alla Fase 8 (Partial Observability).
* **No Self-Play Pool dinamico**: Il self-play avanzato con pool storico appartiene alla Fase 9.
* **No MCTS**: L'albero Monte Carlo appartiene alla Fase 11.
* **No Modifiche Breaking al Game Engine o Gymnasium Env**: Le interfacce di gioco e osservazione rimangono stabili e invariate.
