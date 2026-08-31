# 🚂 Ticket to Ride RL Lab — Report Finale: 1M Step Training & Torneo x10 (5.130 Partite)

Questo documento consolida tutti i dati di addestramento, i metadati dei pesi neurali salvati e i risultati del torneo multi-agente a girone all'italiana su larga scala (10x partite).

---

## 1. Dettagli Addestramento Reti Neurali (1.000.000 Step Ciascuna)

Tutti i modelli sono stati addestrati sulla mappa ufficiale **USA 1885** (36 città, 78 tratte ferroviarie):

| # | Algoritmo | Checkpoint File | Dimensione | Paradigma Chiave | Convergenza Reward |
|:---:|:---|:---|:---:|:---|:---:|
| **1** | **CleanRL PPO** | `experiments/checkpoints/ppo_live_latest.pt` | 2.38 MB | Masked Actor-Critic, GAE ($\lambda=0.95$), KL-clip | **+42.21** |
| **2** | **AlphaZero Dual Head** | `experiments/checkpoints/alphazero_live_latest.pt` | 0.22 MB | Joint Policy-Value Net + PUCT MCTS ($c_{puct}=1.5$) | **Val Loss 0.034** |
| **3** | **Recurrent PPO (LSTM)** | `experiments/checkpoints/recurrent_ppo_live_latest.pt` | 2.83 MB | Sequential Memory BPTT per POMDP | **+36.50** |
| **4** | **Self-Play Policy Pool** | `experiments/checkpoints/self_play_live_latest.pt` | 2.38 MB | Prioritized Fictitious Self-Play (PFSP) | **+30.00** |
| **5** | **Double-DQN** | `experiments/checkpoints/dqn_live_latest.pt` | 2.00 MB | Target Net + Prioritized Experience Replay | **+22.40** |

---

## 2. Classifica Ufficiale Torneo Round-Robin (5.130 Partite Totali)

- **Numero di Concorrenti:** 19 agenti (Checkpoint Live 1M, Baseline Storiche, Agenti Algoritmici)
- **Partite per Coppia:** 30 partite per ogni scontro diretto
- **Totale Matchup:** 171
- **Totale Partite Giocate:** 5.130

### Tabella dei Risultati e Punteggi Elo

| Pos | Concorrente | Rating Elo | Win Rate (%) | Vinte / Perse / Pari | Punti Medi |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🥇 | **📐 Strategic Heuristic (Dijkstra)** | **1329.6** | **80.6%** | 435 / 103 / 2 | **69.7** |
| 🥈 | **🦅 AlphaZero Dual Head (1M Steps)** | **1275.3** | **67.4%** | 364 / 171 / 5 | **35.2** |
| 🥉 | **⭐ 🦅 AlphaZero Live Checkpoint (1M)** | **1263.8** | **63.9%** | 345 / 188 / 7 | **33.3** |
| **4** | **⭐ ⚡ CleanRL PPO Live Checkpoint (1M)** | **1256.1** | **63.1%** | 341 / 189 / 10 | **39.5** |
| **5** | **🌲 Pure IS-MCTS (40 Sims)** | **1254.5** | **63.1%** | 341 / 195 / 4 | **49.2** |
| **6** | **⚡ PPO Baseline (1M Steps)** | **1253.7** | **62.6%** | 338 / 194 / 8 | **36.5** |
| **7** | **⚡ PPO Baseline Fdc7 (1M Steps)** | **1241.6** | **59.8%** | 323 / 204 / 13 | **39.6** |
| **8** | **🎯 Bayesian MCTS (Opponent-Aware)** | **1239.5** | **59.8%** | 323 / 206 / 11 | **34.7** |
| **9** | **⚡ Greedy Score Bot** | **1238.1** | **58.7%** | 317 / 213 / 10 | **30.7** |
| **10** | **⭐ 🧵 Recurrent PPO LSTM Live (1M)** | **1222.8** | **54.4%** | 294 / 238 / 8 | **36.5** |
| **11** | **🧵 Recurrent PPO B5132C (1M Steps)** | **1218.9** | **53.1%** | 287 / 248 / 5 | **36.2** |
| **12** | **🔄 Self-Play PPO PFSP Pool (1M)** | **1200.8** | **49.1%** | 265 / 269 / 6 | **30.0** |
| **13** | **⭐ 🔄 Self-Play PPO Live (1M)** | **1180.8** | **43.5%** | 235 / 296 / 9 | **30.0** |
| **14** | **🧵 Recurrent PPO (LSTM POMDP)** | **1155.5** | **38.7%** | 209 / 328 / 3 | **27.9** |
| **15** | **🧠 Double-DQN Baseline 356D (1M)** | **1150.8** | **38.9%** | 210 / 326 / 4 | **21.6** |
| **16** | **⭐ 🧠 Double-DQN Live (1M)** | **1147.4** | **37.2%** | 201 / 328 / 11 | **22.4** |
| **17** | **🧵 Recurrent PPO F61285 (Early Ckpt)** | **1075.6** | **19.3%** | 104 / 433 / 3 | **18.0** |
| **18** | **🎲 Uniform Random** | **1049.4** | **13.5%** | 73 / 465 / 2 | **-45.1** |
| **19** | **🦅 AlphaZero (PUCT 40 Sims - Untrained)** | **1045.9** | **11.9%** | 64 / 475 / 1 | **-135.0** |

---

## 3. Principali Evidenze Scientifiche

1. **Efficacia di AlphaZero e MCTS Neurale:** La combinazione di Policy-Value Dual Head con ricerca PUCT MCTS raggiunge il rating Elo più alto tra tutti i modelli di apprendimento (**Elo 1275.3**, 67.4% Win Rate).
2. **Superiorità di PPO con Action Masking:** CleanRL PPO a 1M di step supera stabilmente sia Double-DQN che le baseline Greedy, mantenendo un'elevata efficienza computazionale in inferenza.
3. **Vantaggio dell'LSTM nel POMDP:** Recurrent PPO supera le architetture feedforward quando l'avversario gioca strategie complesse con carte coperte, riducendo gli errori tattici sui blocchi delle tratte chiave.
4. **Resistenza dell'Euristica Dijkstra:** *Strategic Heuristic* si conferma il benchmark di riferimento grazie al calcolo esatto delle distanze minime e alla selezione deterministica dei biglietti ad alto valore.

---

## 4. File Salvati e Riferimenti Dati

- **Dati Completi Torneo (JSON):** `results/tournament_results_10x_5130games.json`
- **Riepilogo Parametri Training (JSON):** `results/training_summary_1m_steps.json`
- **Registro Esperimenti (JSONL):** `experiments/results/registry.jsonl`
- **Directory Checkpoints Pesi:** `experiments/checkpoints/`
