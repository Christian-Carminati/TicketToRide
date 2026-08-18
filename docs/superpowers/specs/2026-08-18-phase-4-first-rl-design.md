# Spec Tecnica e Didattica - Fase 4: First RL (DQN & PPO)

## 1. Visione e Obiettivi della Fase 4

La **Fase 4: First RL** trasforma l'ambiente di gioco Gymnasium realizzato nella Fase 3 in un vero e proprio laboratorio di Reinforcement Learning.

In accordo con i principi fondanti del progetto:
* **RL-first & Learning-first**: ogni algoritmo viene implementato da zero in puro PyTorch, senza dipendere da framework esterni "a scatola chiusa", per comprendere ogni singolo dettaglio matematico e implementativo.
* **No black boxes without a reason**: esponiamo e monitoriamo in tempo reale tutte le grandezze matematiche interne (valori Q, logits di policy, stime del critic, perdite, entropia, divergenza KL, explained variance).
* **Action Masking di prima classe**: nei giochi da tavolo strategici le regole impediscono determinate mosse in certi turni. Il masking matematico diretto garantisce che l'agente impari la strategia anziché sprecare tempo a tentare mosse illegali.
* **Valutazione Multi-Avversario**: il training include una routine di validazione periodica contro tutti i tipi di avversari noti (**RandomAgent**, **GreedyAgent** e **StrategicAgent**).

---

## 2. Fondamenti Teorici ed Esempi Matematici Passo-Passo

### 2.1 Il Problema Decisionale (MDP) in Ticket to Ride
Un problema di Reinforcement Learning è descritto da una tupla $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:
* $\mathcal{S}$: spazio degli stati/osservazioni (vettori continui contenenti carte possedute, tratte occupate, obiettivi, treni rimanenti).
* $\mathcal{A}$: spazio delle azioni discrete (es. 56 azioni sulla mappa Mini, 142 sulla mappa USA).
* $\mathcal{P}(s' | s, a)$: probabilità di transizione dello stato (deterministica per le regole del tabellone, stocastica per la pesca delle carte coperte e le mosse dell'avversario).
* $\mathcal{R}(s, a, s')$: funzione di ricompensa (punti per tratte completate, bonus per completamento obiettivi, penalità di turno, bonus per vittoria finale).
* $\gamma \in [0, 1)$: fattore di sconto per i guadagni futuri (tipicamente $\gamma = 0.99$).

---

### 2.2 Action Masking nel Deep RL

#### Perché serve?
In Ticket to Ride, su 142 azioni totali, al turno $t$ solo un sottoinsieme ristretto è valido (ad esempio pescare una carta scoperta o prendere una specifica tratta per cui si hanno carte e treni a sufficienza).
Se la rete provasse a scegliere un'azione non valida, l'ambiente dovrebbe rigettarla o penalizzarla, rallentando drasticamente la convergenza.

#### La Matematica del Masking
Sia $M(s) \in \{0, 1\}^{|\mathcal{A}|}$ il vettore booleano dove $M(s)_a = 1$ indica che l'azione $a$ è legale e $M(s)_a = 0$ indica che è illegale.

1. **In DQN (Valori Q)**:
   Prima di calcolare l'azione ottimale $\arg\max_a Q(s, a)$, applichiamo la maschera:
   $$Q_{\text{masked}}(s, a) = \begin{cases} Q(s, a) & \text{se } M(s)_a = 1 \\ -\infty & \text{se } M(s)_a = 0 \end{cases}$$
   In questo modo, $\arg\max_a Q_{\text{masked}}(s, a)$ selezionerà matematicamente sempre e solo un'azione lecita.

2. **In PPO (Logits di Policy)**:
   L'Actor produce un vettore di punteggi non normalizzati chiamati *logits* $z(s) \in \mathbb{R}^{|\mathcal{A}|}$.
   Prima di applicare la funzione Softmax per ottenere la distribuzione di probabilità $\pi(a|s)$, modifichiamo i logits:
   $$z'(s)_a = \begin{cases} z(s)_a & \text{se } M(s)_a = 1 \\ -10^8 & \text{se } M(s)_a = 0 \end{cases}$$
   Poiché $e^{-10^8} \approx 0$, la probabilità associata a un'azione illegale sarà esattamente:
   $$\pi(a|s) = \frac{e^{z'(s)_a}}{\sum_{j} e^{z'(s)_j}} = 0.0$$

#### Esempio Numerico di Masking
Supponiamo che lo spazio abbia 3 azioni $\mathcal{A} = [a_0, a_1, a_2]$ e l'Actor produca i logits $z = [2.0, 5.0, 1.0]$.
Supponiamo che l'azione $a_1$ sia illegale (es. non ho abbastanza vagoni): $M = [1, 0, 1]$.

- Senza Masking:
  $$\pi = \text{softmax}([2.0, 5.0, 1.0]) = [0.047, 0.936, 0.017]$$
  (L'agente sceglierebbe l'azione illegale $a_1$ con il 93.6% di probabilità!)
- Con Masking:
  $$z' = [2.0, -10^8, 1.0]$$
  $$e^{2.0} \approx 7.389, \quad e^{-10^8} \approx 0, \quad e^{1.0} \approx 2.718, \quad \text{Somma} = 10.107$$
  $$\pi = \left[ \frac{7.389}{10.107}, 0, \frac{2.718}{10.107} \right] = [0.731, 0.000, 0.269]$$
  L'azione illegale $a_1$ ha probabilità 0.0 esatta e la probabilità totale $1.0$ è ridistribuita proporzionalmente solo tra le azioni lecite $a_0$ e $a_2$.

---

### 2.3 Deep Q-Network (DQN) & Double DQN

DQN è un algoritmo *off-policy*: impara la funzione di valore ottima $Q^*(s, a)$ da esperienze memorizzate in un buffer, anche se tali esperienze sono state generate da una policy precedente o con esplorazione casuale.

#### Componenti di DQN
1. **Q-Network principale ($Q_\theta$)**: rete neurale con parametri $\theta$ che mappa l'osservazione $s$ nei valori stimati per tutte le azioni $Q(s, a)$.
2. **Target Network ($Q_{\theta^-}$)**: copia della rete principale con parametri congelati $\theta^-$, aggiornata periodicamente (es. ogni 1.000 step) per evitare che il bersaglio di apprendimento si sposti continuamente ad ogni aggiornamento dei pesi.
3. **Replay Buffer**: memoria circolare di capacità $N$ (es. 100.000 transizioni) che memorizza $(s_t, a_t, r_t, s_{t+1}, d_t, M_{t+1})$. Rompe le correlazioni temporali tra campioni consecutivi.
4. **Esplorazione $\epsilon$-Greedy**:
   - Con probabilità $\epsilon$: sceglie casualmente un'azione tra quelle per cui $M(s)_a = 1$.
   - Con probabilità $1 - \epsilon$: sceglie l'azione lecita a valore Q massimo: $\arg\max_{a \in \text{valid}} Q_\theta(s, a)$.
   - $\epsilon$ decade linearmente da $1.0$ (esplorazione pura) a $0.05$ (sfruttamento quasi puro).

#### Perché Double DQN (DDQN)?
Nel DQN classico, il target di Bellman è:
$$y_t = r_t + \gamma (1 - d_t) \max_{a'} Q_{\theta^-}(s_{t+1}, a')$$
Poiché usa l'operatore $\max$ sulla stessa rete che stima il valore, tende a sovrastimare sistematicamente i valori $Q$ a causa del rumore stocastico.
**Double DQN** risolve questo disaccoppiando la *scelta* dell'azione dalla sua *valutazione*:
1. La rete principale $Q_\theta$ sceglie l'azione migliore: $a^* = \arg\max_{a' \in \text{valid}} Q_\theta(s_{t+1}, a')$.
2. La Target Network $Q_{\theta^-}$ stima il valore di quell'azione:
   $$y_t = r_t + \gamma (1 - d_t) Q_{\theta^-}(s_{t+1}, a^*)$$

#### Esempio Numerico di Aggiornamento Double DQN
Immaginiamo una transizione:
- Stato $s$, azione eseguita $a = 0$, ricompensa $r = 2.0$, non terminale ($d = 0$).
- Fattore di sconto $\gamma = 0.99$.
- Nello stato successivo $s'$, le azioni valide sono $\{0, 1\}$.
- Stime della rete principale $Q_\theta(s'): Q_\theta(s', 0) = 4.0, Q_\theta(s', 1) = 5.5 \implies a^* = 1$ (azione 1 è la migliore per $Q_\theta$).
- Stime della Target Network $Q_{\theta^-}(s'): Q_{\theta^-}(s', 0) = 3.8, Q_{\theta^-}(s', 1) = 5.0$.

Calcolo del Target:
$$y = r + \gamma \cdot Q_{\theta^-}(s', a^*) = 2.0 + 0.99 \cdot 5.0 = 2.0 + 4.95 = 6.95$$

Se la rete corrente stimava $Q_\theta(s, 0) = 6.0$:
- Errore di Bellman: $\delta = Q_\theta(s, 0) - y = 6.0 - 6.95 = -0.95$.
- La loss $\text{MSE} = (-0.95)^2 = 0.9025$ guiderà la discesa del gradiente ad aumentare la stima di $Q_\theta(s, 0)$ verso $6.95$.

---

### 2.4 Proximal Policy Optimization (PPO) & Actor-Critic

PPO è un algoritmo *on-policy* basato su gradienti di policy, ampiamente considerato lo standard di riferimento per affidabilità, robustezza e stabilità nel Deep RL.

#### Componenti di PPO
1. **Actor ($\pi_\theta(a|s)$)**: rete neurale che restituisce la distribuzione di probabilità sulle azioni lecite.
2. **Critic ($V_\phi(s)$)**: rete neurale che stima il valore intrinseco dello stato $V(s) \approx \mathbb{E}[\sum \gamma^k r_{t+k}]$.
3. **Rollout Buffer**: raccoglie una traiettoria completa di $T$ passi (es. 512 o 1024 step) prima di eseguire l'aggiornamento.
4. **Generalized Advantage Estimation (GAE)**:
   Calcola quanto un'azione $a_t$ è stata vantaggiosa rispetto alla media dello stato $s_t$.
   - Errore TD (Temporal Difference): $\delta_t = r_t + \gamma (1 - d_t) V(s_{t+1}) - V(s_t)$.
   - Advantage ricorsivo: $\hat{A}_t = \delta_t + (\gamma \lambda) (1 - d_t) \hat{A}_{t+1}$.
   - Ritorno target per il Critic: $R_t = \hat{A}_t + V(s_t)$.

#### Il Clipped Surrogate Objective di PPO
Nei metodi Policy Gradient classici, aggiornare troppo la policy con un passo di gradiente grande può far collassare le prestazioni senza possibilità di recupero.
PPO risolve questo limitando (clipping) il rapporto di probabilità $r_t(\theta)$:
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

La funzione obiettivo per l'Actor è:
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right) \right]$$
dove tipicamente $\epsilon = 0.2$, impedendo al rapporto di superare l'intervallo $[0.8, 1.2]$ se il gradiente spinge in modo eccessivo.

#### Esempio Numerico di PPO Clipping
Supponiamo che l'azione $a_t$ abbia prodotto un advantage positivo $\hat{A}_t = +2.0$ (un'ottima mossa!).
- La vecchia policy dava $\pi_{\text{old}}(a_t|s_t) = 0.20$.
- Durante l'ottimizzazione, la nuova policy aggiornata dà $\pi_\theta(a_t|s_t) = 0.30$.
- Il rapporto è: $r_t(\theta) = \frac{0.30}{0.20} = 1.50$.
- Poiché $\epsilon = 0.2$, il valore clippato è $\text{clip}(1.50, 0.8, 1.2) = 1.20$.
- Termine non clippato: $r_t \hat{A}_t = 1.50 \cdot 2.0 = 3.0$.
- Termine clippato: $\text{clip}(r_t) \hat{A}_t = 1.20 \cdot 2.0 = 2.4$.
- L'obiettivo finale prende il minimo: $\min(3.0, 2.4) = 2.4$.

**Cosa è successo?** Il gradiente non riceve un incentivo sproporzionato ($3.0$), ma viene limitato a $2.4$, proteggendo la stabilità dell'apprendimento.

#### Loss Complessiva di PPO
L'ottimizzatore minimizza congiuntamente:
$$L^{\text{PPO}}(\theta, \phi) = - L^{\text{CLIP}}(\theta) + c_{\text{vf}} L^{\text{VF}}(\phi) - c_{\text{ent}} S[\pi_\theta]$$
dove:
- $L^{\text{VF}}(\phi) = \frac{1}{B}\sum (V_\phi(s_t) - R_t)^2$ allena il Critic a prevedere i ritorni reali.
- $S[\pi_\theta] = -\sum_a \pi(a|s)\ln \pi(a|s)$ è l'entropia della policy (incoraggia l'esplorazione e previene il collasso prematuro).
- $c_{\text{vf}} = 0.5$ e $c_{\text{ent}} = 0.01$ sono i coefficienti di ponderazione standard.

---

## 3. Architettura del Software e Moduli (`src/rl/`)

```text
src/
├── rl/
│   ├── __init__.py
│   ├── networks.py        # MaskedQNetwork & MaskedActorCritic
│   ├── replay_buffer.py   # ReplayBuffer per DQN con next_action_masks
│   ├── rollout.py         # RolloutBuffer per PPO con mini-batch generator
│   ├── advantage.py       # Calcolo GAE vettorizzato
│   ├── dqn.py             # MaskedDQNTrainer (Double DQN)
│   └── ppo.py             # MaskedPPOTrainer (PPO con clipping & entropy)
├── agents/
│   ├── dqn_agent.py       # DQNAgent (inferenza, carica .pt)
│   └── ppo_agent.py       # PPOAgent (inferenza, carica .pt)
└── experiments/
    ├── config.py          # Schema Pydantic per configurazioni YAML
    ├── runner.py          # Orchestratore di training ed evaluation
    └── evaluator.py       # Valutatore periodico multi-avversario
```

---

### 3.1 `src/rl/networks.py`

```python
class MaskedQNetwork(nn.Module):
    """Q-Network con supporto per action masking e selezione epsilon-greedy."""
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def select_action(
        self, 
        obs: torch.Tensor, 
        action_mask: np.ndarray, 
        epsilon: float = 0.0
    ) -> int:
        """Seleziona un'azione valida rispettando epsilon-greedy e action mask."""
        ...

class MaskedActorCritic(nn.Module):
    """Architettura Actor-Critic per PPO con distribuzione categorica mascherata."""
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_action_and_value(
        self, 
        obs: torch.Tensor, 
        action_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Restituisce (action, log_prob, entropy, value)."""
        ...
```

---

### 3.2 `src/rl/replay_buffer.py` & `src/rl/rollout.py`

* `ReplayBuffer`: memorizza fino a $N$ transizioni composte da $(s_t, a_t, r_t, s_{t+1}, d_t, M_{t+1})$. Fornisce `sample(batch_size)` che restituisce dizionari di tensori PyTorch pronti per la GPU o CPU.
* `RolloutBuffer`: memorizza $T$ passi sequenziali generati dalla policy corrente. Fornisce `generate_minibatches(batch_size, advantages, returns)` per iterare casualmente sui dati raccolti durante le $E$ epoche di aggiornamento di PPO.

---

### 3.3 `src/rl/dqn.py` (`MaskedDQNTrainer`)

Parametri configurabili:
* `gamma`: 0.99
* `lr`: 0.0005 (Adam)
* `buffer_size`: 50.000
* `batch_size`: 64
* `target_update_freq`: 500 step
* `epsilon_start`: 1.0, `epsilon_end`: 0.05, `epsilon_decay_steps`: 20.000
* `learning_starts`: 1.000 step di warmup casuale

Metriche restituite ad ogni step: `{"loss": float, "q_mean": float, "epsilon": float}`.

---

### 3.4 `src/rl/ppo.py` (`MaskedPPOTrainer`)

Parametri configurabili:
* `gamma`: 0.99
* `gae_lambda`: 0.95
* `clip_eps`: 0.2
* `vf_coef`: 0.5
* `ent_coef`: 0.01
* `lr`: 0.0003 (Adam con gradient clipping a 0.5)
* `rollout_steps`: 512
* `num_epochs`: 4
* `minibatch_size`: 64

Metriche restituite ad ogni ciclo di aggiornamento:
`{"policy_loss": float, "value_loss": float, "entropy": float, "approx_kl": float, "explained_var": float}`.

---

## 4. Pipeline di Addestramento, Checkpoint e Valutazione Multi-Avversario

### 4.1 Valutatore Periodico Multi-Avversario
Durante l'addestramento (es. ogni 5.000 step), il trainer chiama il modulo di valutazione:
1. Gioca $K$ partite (es. 20) contro **`RandomAgent`** $\rightarrow$ calcola `eval/win_rate_vs_random`, `eval/score_diff_vs_random`.
2. Gioca $K$ partite contro **`GreedyAgent`** $\rightarrow$ calcola `eval/win_rate_vs_greedy`, `eval/score_diff_vs_greedy`.
3. Gioca $K$ partite contro **`StrategicAgent`** $\rightarrow$ calcola `eval/win_rate_vs_strategic`, `eval/score_diff_vs_strategic`.
4. Se `win_rate_vs_random` supera il record precedente, salva il checkpoint come `experiments/checkpoints/<exp_name>_best.pt`.
5. Salva sempre l'ultimo modello come `experiments/checkpoints/<exp_name>_latest.pt`.

---

### 4.2 File di Configurazione YAML

Creeremo file di configurazione dichiarativi pronti all'uso:
* `experiments/configs/dqn_mini.yaml`: Double DQN su mappa Mini per test rapidi.
* `experiments/configs/ppo_mini.yaml`: Masked PPO su mappa Mini per test rapidi.
* `experiments/configs/ppo_usa.yaml`: Masked PPO su mappa USA per benchmark completo.

Esempio `ppo_mini.yaml`:
```yaml
experiment:
  name: ppo_mini_baseline
  seed: 42

environment:
  board: mini
  observation_version: 1
  reward_version: default

algorithm:
  name: ppo
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  vf_coef: 0.5
  ent_coef: 0.01
  lr: 0.0003
  rollout_steps: 512
  num_epochs: 4
  minibatch_size: 64

training:
  total_timesteps: 30000
  eval_frequency: 5000
  eval_episodes_per_opponent: 20
  eval_opponents:
    - random
    - greedy
    - strategic
```

---

## 5. Piano dei Test e Criteri di Accettazione

### 5.1 Struttura dei Test Unitari e di Integrazione
1. `tests/rl/test_networks.py`:
   - Dimensioni tensori corrette per qualsiasi forma di input/output.
   - Test masking matematico: azioni mascherate hanno valore $-\infty$ (Q-net) o probabilità $0.0$ (ActorCritic).
   - Verifica determinismo (`deterministic=True` seleziona l'azione a probabilità massima).
2. `tests/rl/test_replay_buffer.py`:
   - Corretto salvataggio e campionamento dei batch con `next_action_masks`.
3. `tests/rl/test_rollout_buffer.py`:
   - Generazione dei mini-batch, calcolo corretto di GAE e normalizzazione advantages.
4. `tests/rl/test_dqn_trainer.py`:
   - Aggiornamento dei pesi durante `train_step()` (perdita finita, niente NaN).
   - Decadimento lineare di $\epsilon$.
   - Aggiornamento target network (hard e soft update).
5. `tests/rl/test_ppo_trainer.py`:
   - Ciclo completo di rollout e calcolo di policy loss, value loss, entropy, approx_kl.
6. `tests/agents/test_rl_agents.py`:
   - `DQNAgent` e `PPOAgent` eseguono correttamente `select_action(obs, action_mask)` e supportano il caricamento di checkpoint `.pt`.
7. `tests/rl/test_phase4_acceptance.py`:
   - **Test di Accettazione Criterio 1**: Addestramento rapido DQN su mappa Mini $\rightarrow$ l'agente supera `RandomAgent` con Win Rate $\ge 75\%$.
   - **Test di Accettazione Criterio 2**: Addestramento rapido PPO su mappa Mini $\rightarrow$ l'agente supera `RandomAgent` con Win Rate $\ge 75\%$.
   - **Test di Accettazione Criterio 3**: La routine di valutazione multi-avversario restituisce le metriche complete per `random`, `greedy` e `strategic`.
   - **Test di Accettazione Criterio 4**: Riproducibilità deterministica (stesso seed $\implies$ stessi pesi e identiche metriche).

---

## 6. Definition of Done (Criteri di Completamento)
- [x] Spec tecnica e didattica completata.
- [ ] Implementazione moduli `src/rl/` (reti, buffer, DQN trainer, PPO trainer).
- [ ] Implementazione agenti d'inferenza `src/agents/dqn_agent.py` e `src/agents/ppo_agent.py`.
- [ ] Integrazione del valutatore multi-avversario e dell'`ExperimentRunner`.
- [ ] Creazione dei file di configurazione YAML per mappe Mini e USA.
- [ ] CLI headless `scripts/train.py` e `scripts/evaluate.py` testate e funzionanti.
- [ ] Suite completa di test automatizzati (unit, integration, acceptance) con esito 100% PASS.
