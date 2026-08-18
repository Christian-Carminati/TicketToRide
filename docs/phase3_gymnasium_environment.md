# 🏛 Documentazione Tecnica e Matematica — Fase 3: Gymnasium Environment & Action Masking

Questo documento descrive la formalizzazione matematica, l'architettura software e le scelte implementative introdotte nella **Fase 3** del progetto **TicketToRide RL Lab** per interfacciare il core deterministico del gioco con i framework di Reinforcement Learning standard (Gymnasium).

---

## 1. Formalizzazione Matematica del Gioco come POMDP

Poiché i giocatori non hanno accesso alle carte segrete nella mano dell'avversario, né all'ordine esatto delle carte coperte nei mazzi, Ticket to Ride viene formalizzato rigorosamente come un **Partially Observable Markov Decision Process (POMDP)**:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

### Definizione dei componenti:
1. **$\mathcal{S}$ (True Global State Space)**:
   Include la configurazione completa e non troncata del gioco:
   $$s = \Big( (H_i, K_i, T_i, C_i)_{i=0}^{N-1}, \mathcal{E}_{\text{claimed}}, V, D_{\text{train}}, D_{\text{ticket}}, D_{\text{discard}}, \text{phase}, t \Big)$$
   - $H_i \in \mathbb{N}^9$: multiset esatto delle carte treno in mano al giocatore $i$.
   - $K_i \subseteq \mathcal{K}$: sottoinsieme dei Destination Tickets posseduti dal giocatore $i$.
   - $T_i \in [0, 45]$: vagoni fisici rimanenti per il giocatore $i$.
   - $C_i \in \mathbb{Z}$: punteggio corrente del giocatore $i$.
   - $\mathcal{E}_{\text{claimed}}: \mathcal{E} \to \{0, \dots, N-1\} \cup \{\emptyset\}$: mappa di occupazione delle tratte sulla plancia.
   - $V \in (\mathcal{C} \cup \{\emptyset\})^5$: 5 carte scoperte sul tavolo.
   - $D_{\text{train}}, D_{\text{ticket}}, D_{\text{discard}}$: sequenze ordinate dei mazzi coperti e della pila degli scarti.
   - $\text{phase} \in \{\text{NORMAL}, \text{DRAWING\_SECOND\_CARD}, \text{CHOOSING\_TICKETS}, \text{CHOOSING\_INITIAL\_TICKETS}\}$.
   - $t \in \mathbb{N}$: numero del turno corrente.

2. **$\mathcal{A}$ (Discrete Action Space)**:
   Spazio delle azioni discrete biunivocamente indicizzate (vedi Sezione 3).

3. **$\mathcal{T}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$**:
   Funzione di transizione stocastica determinata dal rimescolamento dei mazzi e dal comportamento stocastico o deterministico degli avversari.

4. **$\mathcal{R}(s, a, s')$**:
   Funzione di ricompensa scalare percepita dall'agente (vedi Sezione 5).

5. **$\Omega \subseteq [0, 1]^D$ (Observation Space)**:
   Vettore di osservazione a valori continui e limitati percepito dall'agente.

6. **$\mathcal{O}(o \mid s, i)$ (Observation Emission Function)**:
   Proiezione deterministica dello stato globale $s$ nel punto di vista parziale del giocatore $i$:
   $$o_t^{(i)} = \mathcal{O}(s_t, i) \in [0, 1]^D$$

---

## 2. Vettore di Osservazione (`ObservationV1`)

Per la mappa USA standard ($|\mathcal{E}|=100$ tratte, $|\mathcal{K}|=30$ biglietti, $N=2$), la dimensione totale del vettore è **$D = 464$**:

| Segmento | Formula Dimensione | Dim. USA | Normalizzazione / Codifica | Note & Proprietà Anti-Leakage |
| :--- | :--- | :--- | :--- | :--- |
| **Carte in mano proprie** | $9$ | 9 | $c / 12.0 \in [0, 1]$ | 8 colori standard + Locomotiva |
| **5 Carte scoperte** | $5 \times 10$ | 50 | One-hot per slot (9 colori + 1 empty) | Informazione pubblica globale |
| **Stato giocatore proprio** | $2$ | 2 | Trains $/ 45$, Score $/ 150$ | Vagoni rimanenti e punti attuali |
| **Stato delle tratte** | $3 \times \|\mathcal{E}\|$ | 300 | One-hot per tratta: `[libera, mia, avversario]` | Topologia completa della plancia |
| **Destination Tickets propri** | $3 \times \|\mathcal{K}\|$ | 90 | Per biglietto: `[posseduto, completato, punti/25]` | Calcolato solo sui biglietti di $P_i$ |
| **Dati pubblici avversari** | $4 \times (N-1)$ | 4 | `[tot_carte/30, treni/45, tratte/30, score/150]` | **Nessun leak** di colori o biglietti avversari |
| **Progressione mazzi & gioco**| $5$ | 5 | `[deck/110, discard/110, tickets/30, turn/100, last_round]` | Stato di avanzamento della partita |
| **Fase del turno (Turn State)**| $4$ | 4 | One-hot: `[NORMAL, DRAW_2ND, CHOOSE_TICKETS, CHOOSE_INIT]` | Guida la policy sulle sotto-fasi |
| **TOTALE** | — | **464** | $\in [0, 1]^{464}$ | Bounded Box space |

### Teorema di Non-Leakage (Fair Information POMDP):
Siano $s_1, s_2 \in \mathcal{S}$ due stati globali identici per il giocatore $i$ tranne che per la sequenza dei mazzi coperti $D_{\text{train}}$ o per la composizione specifica dei colori delle carte in mano all'avversario $H_j$ ($j \neq i$) con $|H_j^{(1)}| = |H_j^{(2)}|$:
$$\mathcal{O}(s_1, i) = \mathcal{O}(s_2, i)$$
L'agente non può in alcun modo dedurre informazioni nascoste dal vettore $o_t$.

---

## 3. Spazio delle Azioni Discrete (`DiscreteActionSpace`)

Lo spazio delle azioni è partizionato deterministicamente con una mappatura biunivoca $\phi : \{0, \dots, |\mathcal{A}|-1\} \leftrightarrow \text{DomainAction}$:

$$\begin{aligned}
a = 0 &\iff \text{DRAW\_HIDDEN\_CARD} \\
a \in [1, 5] &\iff \text{DRAW\_VISIBLE\_CARD}(\text{slot} = a - 1) \\
a = 6 &\iff \text{DRAW\_TICKETS} \\
a \in [7, 13] &\iff \text{KEEP\_TICKETS}(\text{subset\_id} \in \mathcal{P}_{\neq \emptyset}(\{0, 1, 2\})) \\
a \ge 14 &\iff \text{CLAIM\_ROUTE}(\text{route\_id}, \text{color\_chosen})
\end{aligned}$$

Per la mappa USA standard:
* Tratte colorate univoche ($1$ azione per tratta).
* Tratte grigie/jolly ($8$ azioni per tratta, corrispondenti alla scelta del colore con cui pagare).
* Dimensione totale dello spazio delle azioni USA: **$|\mathcal{A}| = 432$**.

La spesa delle locomotive avviene in modo canonico: vengono usate le carte del colore scelto e le locomotive strettamente necessarie a completare il costo $L$.

---

## 4. Invalid Action Masking (`ActionMasker`)

Nel gioco, molte azioni sono illegali in base allo stato corrente (es. rivendicare una tratta già occupata, pescare una locomotiva come seconda carta, o scegliere un colore non posseduto in quantità sufficiente).

L'`ActionMasker` produce un vettore booleano $M(s_t, i) \in \{0, 1\}^{|\mathcal{A}|}$:

$$M_a(s_t, i) = \begin{cases} 1 & \text{se } \phi(a) \in \text{ValidActions}(s_t, i) \\ 0 & \text{altrimenti} \end{cases}$$

### Integrazione con la Policy Neurale:
Nelle architetture Policy Gradient (es. PPO) o Value-based (es. DQN), i logit $z \in \mathbb{R}^{|\mathcal{A}|}$ vengono mascherati prima della Softmax:

$$\tilde{z}_a = \begin{cases} z_a & \text{se } M_a = 1 \\ -\infty \text{ (o } -10^8) & \text{se } M_a = 0 \end{cases}$$

$$\pi_\theta(a \mid o) = \frac{\exp(\tilde{z}_a)}{\sum_{a'} \exp(\tilde{z}_{a'})}$$

Questo garantisce che la probabilità di selezionare un'azione illegale sia identicamente $0$, accelerando la convergenza dell'apprendimento di diversi ordini di grandezza.

---

## 5. Funzione di Ricompensa Modulare (`RewardV1`)

La ricompensa al passo $t$ è formulata in modo modulare:

$$R_t = R_t^{\text{step}} + \mathbb{I}_{\text{terminal}} \cdot R_t^{\text{terminal}}$$

### 1. Step Reward ($R_t^{\text{step}}$):
$$R_t^{\text{step}} = w_{\text{route}} \cdot \Delta \text{Score}_{\text{route}} + w_{\text{ticket\_complete}} \cdot \Delta \text{Score}_{\text{ticket\_comp}} + w_{\text{step\_penalty}}$$

* $w_{\text{route}} = 1.0$: punti immediati da regolamento per la lunghezza della tratta ($1 \to 1, 2 \to 2, 3 \to 4, 4 \to 7, 5 \to 10, 6 \to 15$).
* $w_{\text{ticket\_complete}} = 1.0$: premio per il completamento di un Destination Ticket durante la partita.
* $w_{\text{step\_penalty}} = -0.01$: leggera penalità temporale per incentivare percorsi ottimali ed evitare cicli inerti.

### 2. Terminal Reward ($R_t^{\text{terminal}}$):
A fine partita ($t = T$):
$$R_T^{\text{terminal}} = w_{\text{win}} \cdot \text{Outcome} + w_{\text{score\_diff}} \cdot \frac{C_{\text{player}} - C_{\text{opponent}}}{50.0} + w_{\text{ticket\_penalty}} \cdot \Delta \text{Score}_{\text{ticket\_fail}}$$

* $\text{Outcome} \in \{+1.0 \text{ (Vittoria)}, -1.0 \text{ (Sconfitta)}, 0.0 \text{ (Pareggio)}\}$.
* $w_{\text{win}} = 10.0$.
* $w_{\text{score\_diff}} = 1.0$.
* $w_{\text{ticket\_penalty}} = 1.0$ (penalizzazione per i biglietti non completati a fine partita).

---

## 6. Ambiente Gymnasium (`TicketToRideEnv`)

L'ambiente `TicketToRideEnv` eredita da `gymnasium.Env` e implementa il pattern a **auto-stepping dell'avversario**:

```text
       Agent Step a_t (Player 0)
              │
              ▼
   ┌──────────────────────┐
   │   Game.step(a_t)     │
   └──────────┬───────────┘
              │
              ▼
    Is Player 0's turn?
         /        \
       Yes         No (Opponent Turn)
       /            \
      │       ┌──────────────────────────────┐
      │       │ Opponent.act(s, valid, board)│
      │       │ Game.step(opp_action)        │
      │       └──────────────┬───────────────┘
      │                      │
      │◄─────────────────────┘ (loop until Player 0's turn or Game Over)
      │
      ▼
 Return (obs, reward, terminated, truncated, info)
```

### Conformità Gymnasium:
* Supera formalmente la verifica `gymnasium.utils.env_checker.check_env(env)`.
* Include `info["action_mask"]` a ogni `reset()` e `step()`.
* Gestisce deterministicamente il troncamento al raggiungimento di `max_turns`.
* Garantisce la riproducibilità esatta data la medesima coppia di seed `(reset_seed, action_sequence)`.
