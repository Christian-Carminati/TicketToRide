# Specifica di Design & Lezione Didattica: Fase 9 — Self-Play & Historical Policy Pool

**Versione Documento:** 1.0.0  
**Data:** 2026-08-21  
**Fase:** 9 (Self-Play & Historical Policy Pool)  
**Stato:** Approvato per l'Implementazione  
**Lingua:** Italiano  

---

# 1. Lezione Teorica: Self-Play e Matchmaking nel Reinforcement Learning

## 1.1 Il Problema del Single-Opponent Overfitting e dei Cicli Strategici

Nei giochi multi-agente e competitivi come Ticket to Ride, l'addestramento di una politica di Reinforcement Learning contro un singolo avversario fisso (ad esempio unicamente contro `RandomAgent` o una specifica euristica greedy) induce un forte fenomeno di **overfitting strategico**:
- La policy ottimizza la propria funzione valore $V(s)$ e la distribuzione d'azione $\pi(a \mid s)$ per sfruttare le specifiche debolezze o prevedibilità di quell'unico avversario.
- Quando la policy viene poi valutata contro una classe di avversari differente o non vista (es. un bot che pianifica rotte lunghe o che contende attivamente i nodi centrali), le sue prestazioni crollano drasticamente.

Inoltre, nei giochi strategici complessi lo spazio delle politiche non è puramente transitivo. Emergono dinamiche non transitive note come **Policy Cycling** o dinamiche *sasso-carta-forbice*:

$$\pi_A \succ \pi_B \succ \pi_C \succ \pi_A$$

Se l'agente gioca unicamente contro la versione immediatamente precedente di se stesso (**Pure Self-Play** ingenuo: $\pi_t$ contro $\pi_{t-1}$):
1. La politica $\pi_t$ impara a battere $\pi_{t-1}$.
2. Nel farlo, la politica $\pi_{t+1}$ muta e perde le difese contro strategie più arcaiche o basilari ($\pi_0, \pi_1$).
3. L'addestramento entra in un'oscillazione infinita (*cycling* o *policy drift*) senza alcun reale incremento di abilità assoluta.

```text
Pure Self-Play Ingenuo:
π_t  ── gioca contro ──►  π_{t-1}  (Rischio: cicli strategici, policy drift)

Historical Policy Pool (Fictitious Play):
π_t  ── gioca contro ──►  Sample( {π_0, π_1, ..., π_{t-1}}, P(matchmaking) )
```

---

## 1.2 Fondamenti di Fictitious Play e Matchmaking nel Policy Pool

Per garantire convergenza verso un equilibrio di Nash e stimolare un apprendimento monotonico della forza di gioco, utilizziamo il principio del **Fictitious Self-Play (FSP)** e del **Prioritized Fictitious Self-Play (PFSP)** (ispirato ad AlphaStar, OpenAI Five e Suphx).

Manteniamo un **Historical Policy Pool**:

$$\mathcal{M} = \{\pi_0, \pi_1, \pi_2, \dots, \pi_K\}$$

dove ogni $\pi_i$ è uno snapshot congelato (*checkpoint*) dei parametri neurali della policy catturato a intervalli regolari durante l'addestramento.

All'inizio di ciascun episodio di gioco nel simulatore Gymnasium, il motore di matchmaking campiona un avversario $\pi_{\text{opp}} \sim \mathcal{P}(\mathcal{M})$ secondo una specifica distribuzione probabilistica:

### 1. Uniform Historical Sampling
Ogni generazione storica ha identica probabilità di essere affrontata:
$$P(\pi_i) = \frac{1}{|\mathcal{M}|}$$
Garantisce che la politica corrente conservi sempre la capacità di dominare le versioni passate.

### 2. Latest-Biased Mixture Sampling
Una frazione fissata di episodi (es. 50%) affronta la versione più recente $\pi_{\text{latest}}$ per stimolare la spinta agonistica alla frontiera, mentre il restante 50% viene distribuito uniformemente tra le generazioni più vecchie:
$$P(\pi_{\text{latest}}) = p_{\text{latest}}, \quad P(\pi_i \mid i < \text{latest}) = \frac{1 - p_{\text{latest}}}{|\mathcal{M}| - 1}$$

### 3. Prioritized Fictitious Self-Play (PFSP)
Campiona con probabilità proporzionale al livello di sfida o al potenziale di apprendimento. Se stimiamo il tasso di vittoria $w_i = \text{WinRate}(\pi_t, \pi_i) \in [0, 1]$:
$$P(\pi_i) \propto (1 - w_i)^p + \epsilon$$
con $p \ge 1$ (es. $p=1$ o $p=2$). Questo meccanismo concentra l'allenamento contro gli avversari storici che risultano ancora ostici o che battono la policy corrente, evitando di sprecare troppi gradienti contro versioni deboli già dominate.

### 4. Baseline Anchoring (Mix-in)
Con probabilità $\epsilon_{\text{baseline}} \in [0.10, 0.20]$, il matchmaking assegna all'ambiente un bot euristico o random (`RandomAgent`, `GreedyAgent`, `StrategicAgent`). Questo crea un ancoraggio esterno invariante alle convenzioni di gioco umane, impedendo che la popolazione di self-play converga verso equilibri bizzarri o sub-ottimali (*degenerative equilibria*).

---

## 1.3 Matrice di Payoff e Rating Elo Generazionale

Per verificare scientificamente la bontà dell'addestramento in Self-Play, valutiamo l'insieme delle generazioni storiche $\{\pi_0, \pi_1, \dots, \pi_K\}$ e dei baseline in un **Torneo Round-Robin Completo**.

Dalla matrice dei risultati testa a testa:

$$M_{i,j} = \text{WinRate}(\pi_i, \pi_j)$$

calcoliamo il rating **Elo** di ciascuna generazione $R(\pi_i)$:

$$R_{\text{new}}(\pi_i) = R_{\text{old}}(\pi_i) + K \cdot (S_{i,j} - E_{i,j})$$
$$E_{i,j} = \frac{1}{1 + 10^{(R(\pi_j) - R(\pi_i)) / 400}}$$

Un addestramento in Self-Play efficace e sano produce una **curva Elo strettamente crescente** con le generazioni:

$$R(\pi_0) < R(\pi_1) < R(\pi_2) < \dots < R(\pi_{\text{final}})$$

---

# 2. Architettura del Sistema

```text
                               ┌────────────────────────┐
                               │   TicketToRideEnv      │
                               │  - Player 0: Trainee   │
                               │  - Player 1: Opponent  │
                               └───────────┬────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │                                     │
                        ▼                                     ▼
           ┌────────────────────────┐            ┌────────────────────────┐
           │   SelfPlayPPOTrainer   │            │   PolicyPool           │
           │  (Masked PPO / LSTM)   │            │  - Generazioni π_0..π_K│
           │  - Rollout Collection  │            │  - State dicts / Models│
           │  - Snapshot periodic   │            │  - Elo & Match Stats   │
           └────────────┬───────────┘            └───────────┬────────────┘
                        │                                    │
                        │        1. Step / Rollout           │
                        ├────────────────────────────────────┤
                        │        2. On Done / Episode Reset  │
                        │           Sample Opponent via      │
                        │           SelfPlayOpponentSampler  │
                        │        3. Set env.opponent         │
                        │        4. Reset env.opponent(seed) │
                        ▼                                    ▼
           ┌────────────────────────┐            ┌────────────────────────┐
           │ SelfPlayOpponentSampler│            │ SelfPlayBenchmarkRunner│
           │ - Uniform / Latest     │            │ - Generational Tourney │
           │ - PFSP (WinRate Based) │            │ - Elo Progression      │
           │ - Baseline Mix-in      │            │ - Markdown & JSON Rep. │
           └────────────────────────┘            └────────────────────────┘
```

---

# 3. Specifiche Dettagliate dei Componenti

### 3.1 `PolicySnapshot` & `PolicyPool`
- **File:** `src/rl/self_play.py`
- **`PolicySnapshot` DataClass:**
  - `generation: int`: Indice incrementale della generazione (0, 1, 2, ...).
  - `step: int`: Timestep di training in cui è stato catturato lo snapshot.
  - `name: str`: Identificativo leggibile (es. `"gen_003_step_15000"`).
  - `state_dict: dict[str, torch.Tensor]`: Copia profonda (CPU) dei pesi neurali.
  - `is_recurrent: bool`: Flag che indica se il modello è `RecurrentMaskedActorCritic` o `MaskedActorCritic`.
  - `hidden_dim: int`: Iperparametro della dimensione nascosta.
  - `lstm_hidden_dim: int`: Iperparametro LSTM (se ricorrente).
  - `metadata: dict[str, Any]`: Metadati aggiuntivi (loss, timestamp, win rate stimato).
- **`PolicyPool` Class:**
  - `max_size: int`: Capacità massima del pool (default 50). Quando il pool è pieno, elimina le generazioni intermedie o più vecchie preservando sempre $\pi_0$ come ancora fondamentale.
  - `add_policy(model, step, name, metadata)`: Estrae i pesi dal modello, crea il `PolicySnapshot` e lo aggiunge al pool.
  - `get_snapshot(index_or_name)`: Recupera uno snapshot specifico.
  - `create_agent(snapshot_or_idx, board, tickets, deterministic=True)`: Crea al volo un'istanza pronta all'uso di `PPOAgent` o `RecurrentPPOAgent` configurata con i pesi dello snapshot.
  - `save_pool(directory)` / `load_pool(directory)`: Serializzazione e ripristino del pool su disco.

---

### 3.2 `SelfPlayOpponentSampler`
- **File:** `src/rl/self_play.py`
- **Strategie Supportate:**
  - `UNIFORM`: Selezione casuale uniforme tra tutte le generazioni presenti nel pool.
  - `LATEST_BIASED`: Probabilità concentrata sulla generazione più recente (es. 50%), distribuendo il resto sulle generazioni passate.
  - `PFSP` (Prioritized Fictitious Self-Play): Probabilità pesata sulle performance storiche contro la policy corrente.
  - `BASELINES_MIX`: Iniezione con probabilità $\epsilon_{\text{baseline}}$ di agenti deterministici (`RandomAgent`, `GreedyAgent`, `StrategicAgent`).
- **Metodi Principali:**
  - `sample_opponent(pool, current_trainee_step)`: Restituisce un'istanza di `BaseAgent` pronta per essere assegnata all'ambiente.
  - `record_match_result(opponent_name, trainee_won, trainee_score, opp_score)`: Aggiorna le statistiche interne utilizzate dal campionamento PFSP.

---

### 3.3 Trainer di Self-Play (`SelfPlayPPOTrainer` & `SelfPlayRecurrentPPOTrainer`)
- **File:** `src/rl/self_play.py`
- **Caratteristiche Operative:**
  - Supportano sia architetture MLP feed-forward (`MaskedActorCritic`) sia ricorrenti (`RecurrentMaskedActorCritic`).
  - Parametri di configurazione:
    - `snapshot_interval`: Intervallo in timesteps tra snapshot successivi nel `PolicyPool` (es. ogni 5.000 timesteps).
    - `sampling_strategy`: Strategia di matchmaking (`"latest_biased"`, `"uniform"`, `"pfsp"`, `"hybrid"`).
    - `baseline_mix_rate`: Probabilità di affrontare baseline deterministici (default 0.15).
  - Ciclo di rollout (`collect_rollout`):
    - Al reset di ogni episodio o al termine della partita (`done == True`):
      1. Viene invocato il sampler per selezionare il prossimo avversario.
      2. L'avversario viene assegnato a `self.env.opponent`.
      3. Viene invocato `self.env.opponent.reset(seed)` per garantire lo stato pulito.
    - Se `self.total_timesteps - last_snapshot_step >= snapshot_interval`:
      Viene registrata automaticamente una nuova generazione nel pool.

---

### 3.4 Torneo Generazionale & Benchmark Suite (`SelfPlayBenchmarkRunner`)
- **File:** `src/evaluation/self_play_benchmark.py`
- **Protocollo Scientifico:**
  1. Addestra due modelli con lo stesso seed:
     - **Modello A (Self-Play)**: Addestrato con `SelfPlayPPOTrainer` / `PolicyPool`.
     - **Modello B (Single-Opponent)**: Addestrato solo contro `RandomAgent`.
  2. Costruisce una popolazione di valutazione contenente:
     - Le generazioni del Self-Play: `Gen_0` (inizio), `Gen_1` (intermedia), `Gen_Final`.
     - Il modello Single-Opponent baseline.
     - I baseline deterministici: `RandomAgent`, `GreedyAgent`, `StrategicAgent`.
  3. Esegue un **Torneo Round-Robin Completo** tramite `Tournament` (calcolando rating Elo e matrice di scontro diretto $N \times N$).
  4. Misura la monotonicità della progressione Elo:
     $$\text{Elo}(\text{Gen\_Final}) > \text{Elo}(\text{Gen\_Initial})$$
  5. Misura la superiorità contro i bot deterministici:
     $$\text{WinRate}(\text{Gen\_Final}, \text{Random}) \ge 65\%$$
  6. Genera automaticamente:
     - `experiments/results/phase9_report.json`: Dati analitici completi.
     - `experiments/results/phase9_report.md`: Relazione didattico-scientifica formattata in Markdown.

---

### 3.5 Integrazione Configurazioni e Script CLI
- **File Modificati:**
  - `src/experiments/config.py`: Aggiunta dello schema Pydantic `SelfPlayConfig` all'interno di `ExperimentConfig`.
  - `src/experiments/runner.py`: Supporto all'esecuzione di esperimenti con `self_play.enabled = True`.
  - `scripts/train_selfplay.py`: Script eseguibile da riga di comando per avviare addestramenti in Self-Play headless.
  - `scripts/tournament.py`: Supporto per caricare ed eseguire tornei tra checkpoint del pool.

---

# 4. Criteri di Accettazione & Test Suite (Fase 9)

La suite di test formale in `tests/rl/test_phase9_acceptance.py` verificherà i 6 criteri di accettazione:

### Criterio 1: Gestione del Policy Pool & Snapshotting
Il `PolicyPool` gestisce correttamente l'aggiunta di snapshot, l'estrazione per indice e nome, il limite massimo di capacità (`max_size`) e la creazione di agenti (`PPOAgent` / `RecurrentPPOAgent`) funzionanti.

### Criterio 2: Strategie di Matchmaking del Sampler
Il `SelfPlayOpponentSampler` campiona correttamente secondo le distribuzioni `UNIFORM`, `LATEST_BIASED`, `PFSP` e rispetta il `baseline_mix_rate` specificato.

### Criterio 3: Switch Dinamico dell'Avversario nell'Ambiente
L'ambiente `TicketToRideEnv` commuta correttamente l'avversario `env.opponent` tra un episodio e l'altro senza memory leak, corruzioni di memoria nascosta o eccezioni runtime.

### Criterio 4: Riproducibilità Deterministica
Due sessioni di addestramento in Self-Play avviate con il medesimo seed numerico producono pool storici, pesi delle generazioni e metriche di training identiche.

### Criterio 5: Progressione Generazionale e Superiorità sui Baseline
In un torneo generazionale, la generazione finale `Gen_Final` dimostra un Elo superiore a `Gen_0` e raggiunge un tasso di vittoria $\ge 65\%$ contro `RandomAgent`.

### Criterio 6: Report Scientifico del Benchmark Automatizzato
`SelfPlayBenchmarkRunner` esegue lo studio comparativo automatico e genera correttamente i file `phase9_report.json` e `phase9_report.md` contenenti la matrice di payoff e i punteggi Elo.

---

# 5. File da Creare e Modificare

```text
src/
├── rl/
│   ├── self_play.py                    # Implementazione completa di PolicyPool, Sampler, SelfPlay Trainer
│   └── __init__.py                     # Re-export delle classi di self-play
├── evaluation/
│   ├── self_play_benchmark.py          # SelfPlayBenchmarkRunner
│   └── __init__.py                     # Re-export SelfPlayBenchmarkRunner
├── experiments/
│   ├── config.py                       # Aggiunta di SelfPlayConfig
│   └── runner.py                       # Integrazione Self-Play in ExperimentRunner
└── scripts/
    └── train_selfplay.py               # CLI per l'addestramento Self-Play

tests/
├── rl/
│   ├── test_self_play.py               # Test unitari per PolicyPool, Sampler e Trainer
│   └── test_phase9_acceptance.py       # Suite di accettazione dei 6 criteri di Fase 9
```
