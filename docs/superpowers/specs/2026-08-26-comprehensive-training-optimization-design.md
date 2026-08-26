# Comprehensive End-to-End RL Training Acceleration & Optimization Design

**Data:** 26 Agosto 2026  
**Autore:** Antigravity AI & Pair Programmer  
**Stato:** In Approvazione / Design Spec  
**Target:** Massimizzazione del Throughput e dell'Efficienza di Calcolo in Tutti i Pipeline di Addestramento (`AlphaZero`, `PPO`, `Recurrent PPO / LSTM`, `DQN`, `Self-Play PPO`).

---

## 1. Visione e Obiettivi dell'Ottimizzazione

Il laboratorio **TicketToRide RL** implementa l'intero spettro di algoritmi di Reinforcement Learning per giochi da tavolo ad informazione imperfetta. Per consentire cicli di studio empirici rapidi e un throughput di simulazione in tempo reale superiore nei benchmark e nella Web UI, questa specifica definisce l'architettura di ottimizzazione globale del training.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               COMPREHENSIVE TRAINING ACCELERATION ARCHITECTURE                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   ALPHAZERO      │         │     CLEANRL      │         │   DQN & REPLAY   │
│  NEURAL MCTS     │         │   PPO & LSTM     │         │     BUFFERS      │
│ • Subtree Reuse  │         │ • Zero-copy roll │         │ • train_freq = 4 │
│ • Zero-Copy Ring │         │ • Vectorized env │         │ • Fast contiguous│
│ • Persistent TT  │         │ • Fast GAE calc  │         │   sample buffer  │
└──────────────────┘         └──────────────────┘         └──────────────────┘
```

### Obiettivi Prestazionali Chiave
1. **AlphaZero Throughput**: Aumentare ulteriormente il throughput del self-play tramite **Subtree Reuse** (ereditarietà delle simulazioni dei turni precedenti) e **Zero-Copy Replay Buffer**.
2. **DQN Speedup**: Decuplicare l'efficienza campionaria introducendo il disaccoppiamento tra passo di simulazione e passo di gradiente (`train_frequency=4`), portando il throughput DQN oltre 1.000 FPS.
3. **PPO & LSTM Memory Efficiency**: Eliminare le conversioni e allocazioni di tensori intermedie nei buffer di rollout, eseguendo il caricamento batch direttamente da memoria NumPy contigua.
4. **Verifica Scientifica e Non-Regressione**: Conservare la bit-exact determinism, preservare le invarianti anti-leakage POMDP e validare che tutti i 97+ test unitari passino con metriche di convergenza reward e loss identiche o superiori.

---

## 2. Benchmark Baseline Pre-Ottimizzazione (Misurato)

Di seguito sono riportate le misurazioni empiriche ufficiali registrate prima dell'implementazione:

| Algoritmo | Configurazione & Batch | Step / Turni | Tempo Simulazione | Tempo Gradiente / Epoch | Throughput Baseline | Metrica Iniziale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AlphaZero (Neural MCTS)** | 20 sim/mossa, batch 16 | 55 turni | 0.4411 s | 0.0465 s | **124.7 FPS** (2.493 sim/s) | Loss: 3.8495 |
| **CleanRL Masked PPO** | 256 rollout, minibatch 32 | 256 step | 0.2995 s | 0.1524 s | **854.6 FPS** | Policy Loss: -0.0180 |
| **Recurrent PPO (LSTM)** | 128 rollout, seq 8, chunks 4| 128 step | 0.1843 s | 0.2260 s | **694.5 FPS** | Policy Loss: -0.0029 |
| **Masked Double-DQN** | batch 32, learn_starts 50 | 200 step | N/A (1:1) | 0.5856 s (totale) | **341.5 FPS** | Q-Update: 1:1 |
| **Self-Play PPO (PFSP)** | 128 rollout, pool 20 | 128 step | 0.2295 s | 0.1095 s | **557.6 FPS** | Policy Loss: -0.0261 |

---

## 3. Specifiche di Dettaglio delle Ottimizzazioni

---

### 3.1 AlphaZero & Neural MCTS: Subtree Reuse & Zero-Copy Circular Buffer

#### 3.1.1 Subtree Reuse (Conservazione della Radice)
- **Problema**: Attualmente, ad ogni mossa del gioco `NeuralMCTSEngine.search()` distrugge l'albero e ricrea un nuovo `root = NeuralMCTSNode(...)`, scartando centinaia di nodi ed esplorazioni già calcolate nei rami figli.
- **Soluzione**: Quando il motore AlphaZero o il giocatore compie l'azione $a^*$, se $a^* \in \text{root.children}$, la nuova radice per il turno successivo diventa direttamente `root.children[a^*]`, preservando i conteggi di visita $N(s, a)$ e le stime di valore $Q(s, a)$ accumulate.
- **Formula PUCT con Radice Ereditata**:
  $$U(s, a) = c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$
- **Impatto**: Riduzione del 40% delle simulazioni richieste a parità di profondità di calcolo tattico.

#### 3.1.2 Zero-Copy Circular Replay Buffer
- **Implementazione**: Sostituzione delle liste dinamiche di array in `SelfPlayReplayBuffer` con buffer NumPy 2D pre-allocati:
  - `self.obs_buf: np.ndarray` di dimensione `(capacity, obs_dim)`
  - `self.masks_buf: np.ndarray` di dimensione `(capacity, action_dim)`
  - `self.policy_buf: np.ndarray` di dimensione `(capacity, action_dim)`
  - `self.values_buf: np.ndarray` di dimensione `(capacity, 1)`
- **Campionamento Vettoriale**:
  `obs_batch = torch.from_numpy(self.obs_buf[indices])`
  Azzera l'allocazione di liste e tuple temporanee.

---

### 3.2 Masked Double-DQN: Step-to-Train Frequency Decoupling

#### 3.2.1 Frequenza di Training (`train_frequency = 4`)
- **Problema**: In `TrainerService` per DQN, ad ogni singolo `step()` dell'ambiente veniva eseguito immediatamente `train_step()` (campionamento batch 32, 2 forward pass, 1 backward pass e optimizer step). Questo limitava il throughput a soli ~340 FPS su CPU.
- **Soluzione**: Introdurre il parametro `train_frequency: int = 4` (standard CleanRL / Nature DQN). Il gradient update viene eseguito solo quando `step_count % train_frequency == 0`.
- **Target Network Soft/Hard Update**:
  L'aggiornamento della rete target `self.target_net.load_state_dict(self.policy_net.state_dict())` avviene ogni `target_update_freq = 500` step.
- **Impatto Previsto**: Throughput DQN incrementato da 341 FPS a **1.000+ FPS** (3x speedup).

---

### 3.3 CleanRL PPO & Recurrent PPO: Zero-Overhead Memory & Vector Rollout

#### 3.3.1 Pinned / Pre-allocated Tensor Transfers in `RolloutBuffer`
- **Problema**: Durante `collect_rollout()`, la policy esegue `float(value.item())`, `float(log_prob.item())` e inserisce i valori nei buffer Python prima di riconvertirli in tensori per il calcolo delle loss.
- **Soluzione**: In `RolloutBuffer`, l'indice di scrittura `self.ptr` opera direttamente su array contigui in memoria C. Durante `train_epoch()`, i tensori PyTorch vengono generati tramite slicing con `torch.as_tensor()` senza copie ridondanti di memoria.

#### 3.3.2 Vettorizzazione e Linear LR Annealing
- Utilizzo efficiente di `compute_gae` con array contiguous `np.float32`.
- Decadimento lineare del learning rate $lr_t = lr_0 \cdot \left(1 - \frac{t}{T_{\text{total}}}\right)$ integrato in modo nativo.

---

### 3.4 Direct Boolean Bitmasking (Environment & ActionSpace)

- Mappatura $O(1)$ pre-calcolata tra tratte e indici di `DiscreteActionSpace`.
- Generazione immediata della maschera booleana senza allocazione di oggetti `Action` o iteratori `combinations` durante le simulazioni MCTS e i rollout PPO.

---

## 4. Piano di Verifica e Criteri di Accettazione

1. **Parità Funzionale**: Tutti i 97+ test unitari esistenti (`pytest`) devono passare senza errori né warning.
2. **Benchmark Side-by-Side**: Uno script di benchmark automatizzato (`scripts/benchmark_training_suite.py`) confronterà le prestazioni prima e dopo le modifiche.
3. **Report di Confronto**: Generazione del report `benchmark_comparison_report.txt` che documenta:
   - FPS prima e dopo per ciascun algoritmo.
   - Percentuale di guadagno di throughput.
   - Verifica di convergenza delle loss e delle ricompense.

---

## 5. File Coinvolti e Modifiche Previste

1. `src/rl/alphazero_search.py`: Integrazione Subtree Reuse e gestione ciclo di vita della radice.
2. `src/rl/alphazero_trainer.py`: `SelfPlayReplayBuffer` pre-allocato e indicizzazione vettoriale.
3. `src/rl/dqn.py` & `src/api/trainer_service.py`: `train_frequency=4` per DQN.
4. `src/rl/rollout.py` & `src/rl/ppo.py`: Ottimizzazione zero-copy per PPO RolloutBuffer.
5. `src/rl/replay_buffer.py`: Ottimizzazione buffer contiguo per DQN.
6. `scripts/benchmark_training_suite.py`: Suite di benchmark per misurazione prima/dopo e generazione report.
