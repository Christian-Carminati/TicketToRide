# Spec Tecnica e Didattica - Fase 4: First RL (DQN & PPO)

## 1. Visione, Filosofia e Obiettivi Didattici della Fase 4

La **Fase 4: First RL** rappresenta il punto di svolta fondamentale dell'intero progetto: trasformare l'ambiente di simulazione Gymnasium (sviluppato e validato nella Fase 3) in un laboratorio didattico e sperimentale avanzato di **Deep Reinforcement Learning**.

```text
                       ┌────────────────────────────────────────────────────────┐
                       │          FASE 4: FIRST REINFORCEMENT LEARNING          │
                       └────────────────────────────────────────────────────────┘
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
   ┌───────────────────────────────┐                                 ┌───────────────────────────────┐
   │        VALUE-BASED RL         │                                 │       POLICY-BASED RL         │
   │          Double DQN           │                                 │             PPO               │
   │  "Stimo il valore Q(s, a)     │                                 │  "Ottimizzo direttamente la   │
   │   di ogni azione lecita"      │                                 │   distribuzione di policy"    │
   └───────────────────────────────┘                                 └───────────────────────────────┘
                   │                                                                 │
                   └────────────────────────────────┬────────────────────────────────┘
                                                    │
                                                    ▼
                       ┌────────────────────────────────────────────────────────┐
                       │             ACTION MASKING & ZERO-BLACK-BOX            │
                       │     Regole ferree, monitoraggio KL, entropia, GAE      │
                       └────────────────────────────────────────────────────────┘
```

---

### 1.1 Il Salto di Paradigma: Dalle Euristiche all'Auto-Apprendimento

Nelle Fasi 1 e 2 abbiamo implementato e testato tre agenti con logiche algoritmiche predeterminate ("hard-coded"):
* **`RandomAgent`**: sceglie uniformemente a caso tra le azioni lecite, fungendo da baseline minima di controllo.
* **`GreedyAgent`**: valuta solo il tornaconto immediato (punti tratta al turno corrente), ignorando completamente la strategia a lungo termine.
* **`StrategicAgent`**: applica algoritmi di cammino minimo su grafi (Dijkstra) per completare sistematicamente i biglietti di destinazione.

Sebbene efficaci, questi agenti soffrono di un limite intrinseco: **non apprendono dall'esperienza**. Le loro decisioni sono vincolate dai presupposti del programmatore umano. Se un avversario adotta una contromossa specifica (come bloccare una strozzatura nevralgica sulla mappa), un'euristica statica non può adattarsi.

Nel **Reinforcement Learning**, l'agente inizia in totale assenza di conoscenza strategica. Non sa cosa sia un "treno", né quale sia la connessione tra città o l'importanza di conservare carte dello stesso colore. L'agente interagisce con il mondo esclusivamente attraverso:
1. Un vettore continuo di osservazione $\mathbf{s}_t \in \mathcal{S}$.
2. Un vettore booleano di validità delle azioni $\mathbf{M}(s_t) \in \{0, 1\}^{|\mathcal{A}|}$.
3. Uno scalare di ricompensa $r_t \in \mathbb{R}$.

Attraverso il ciclo continuo di **tentativi, errori e feedback (Trial-and-Error)**, l'agente deve estrarre autonomamente concetti strategici complessi:
* **Ragionamento a lungo termine (Credit Assignment)**: sacrificare punti immediati per pescare carte e completare una rotta da 6 vagoni che sblocca un biglietto da 20 punti.
* **Gestione del rischio**: decidere se pescare carte scoperte (certezza) o dal mazzo coperto (possibilità di pescare locomotive Jolly).
* **Adattamento dinamico**: modificare il percorso pianificato se l'avversario occupa una tratta chiave.

---

### 1.3 Le Sfide Fondamentali di Ticket to Ride per il Deep RL

1. **Ricompense Differite e Sparse (Delayed & Sparse Credit Assignment)**:
   La maggior parte delle mosse (es. pescare carte per 5 turni consecutivi) riceve una ricompensa immediata nulla ($r=0$). I punti cruciali derivano dal completamento dei biglietti di destinazione, calcolati solo al termine della partita o alla chiusura del percorso. L'algoritmo deve propagare all'indietro il valore di queste ricompense future attraverso decine di passi temporali.
2. **Spazio di Azioni Ibrido, Ampio e Dinamicamente Vincolato**:
   Su 56 azioni (mappa Mini) o 142 azioni (mappa USA), in ogni singolo turno solo un piccolo sottoinsieme (tipicamente da 2 a 15 azioni) è lecitamente eseguibile. Senza un controllo matematico rigoroso, la rete neurale sprecherebbe la quasi totalità dell'addestramento tentando mosse vietate.
3. **Ambiente Parzialmente Stocastico e Avversario Competitivo**:
   Le carte coperte nel mazzo introducono aleatorietà, mentre l'avversario introduce dinamiche di gioco a somma non-zero con risorse condivise limitate (le tratte del tabellone possono essere reclamate da un solo giocatore).

---

### 1.4 I Quattro Pilastri Didattici del Progetto

1. **RL-First & Learning-First (Implementazione da Zero in PyTorch)**:
   Non utilizzeremo framework a scatola chiusa come *Stable-Baselines3*, *Ray/RLlib* o *CleanRL*. Ogni equazione, buffer di memoria, operazione di masking, backward pass e passo di ottimizzazione viene implementato in puro PyTorch. Questo approccio garantisce la padronanza assoluta della matematica sottostante e della gestione dei tensori.
2. **No Black Boxes Without a Reason (Osservabilità e Diagnostica Continua)**:
   Nel Reinforcement Learning, monitorare semplicemente la funzione di perdita (*loss*) è ingannevole: la loss può aumentare mentre la policy dell'agente sta migliorando drasticamente. Esporremo e tracceremo in tempo reale tutte le grandezze teoriche interne: *Mean Q-Values*, *Policy Entropy*, *Approximate KL Divergence*, *Value Explained Variance*, *Clip Fraction*.
3. **Action Masking di Prima Classe**:
   Nei giochi strategici, penalizzare le mosse illegali con ricompense negative (es. $r = -10$) è dannoso: corrompe la funzione di valore e rallenta l'esplorazione. Integreremo il masking direttamente all'interno dei logits di policy e dei valori Q.
4. **Validazione Multi-Avversario & Benchmark Accademico**:
   L'addestramento include una pipeline di valutazione automatizzata periodica contro `RandomAgent`, `GreedyAgent` e `StrategicAgent`, permettendo di quantificare oggettivamente l'avanzamento strategico della rete.

---

## 2. Lezione Teorica e Fondamenti Matematici Passo-Passo

In questa sezione esploriamo formalmente la teoria del Reinforcement Learning applicata a Ticket to Ride, accompagnando ogni concetto con la derivazione analitica e un esempio numerico passo-passo.

---

### 2.1 Il Framework Matematico: Processi Decisionali di Markov (MDP)

Il problema decisionale sequenziale di Ticket to Ride viene formulato come un **Processo Decisionale di Markov** (Markov Decision Process, MDP) formalizzato dalla tupla a 5 elementi:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

```text
    ┌─────────────────────────────────────────────────────────────┐
    │                        AMBIENTE                             │
    │  (Tabellone, Mazzo Carte, Tratte, Vagoni, Avversario)       │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
               Stato s_t           │   Ricompensa r_t
               Maschera M(s_t)     │   Terminazione d_t
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                         AGENTE                              │
    │               (Masked DQN o Masked PPO)                     │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
                                   │   Azione a_t ∈ A_valid(s_t)
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                        TRANSIZIONE                          │
    │                 s_{t+1} ~ P(· | s_t, a_t)                   │
    └─────────────────────────────────────────────────────────────┘
```

#### 1. Spazio degli Stati $\mathcal{S}$ (Continuous Observation Vector)
Lo stato $\mathbf{s}_t \in \mathbb{R}^D$ è un vettore continuo a dimensione fissa ($D = 128$ nella mappa Mini, $D = 256$ nella mappa USA) che codifica lo stato del gioco dal punto di vista dell'agente:
* **Carte in mano**: conteggio normalizzato per ciascuno degli 8 colori + locomotive Jolly (9 valori).
* **Vagoni rimanenti**: rapporto rispetto ai 45 vagoni iniziali (1 valore scalare $\in [0, 1]$).
* **Tratte occupate**: per ogni tratta della mappa, un one-hot o valore scalare che indica se la tratta è libera ($0$), occupata dall'agente ($+1$), o occupata dall'avversario ($-1$).
* **Carte scoperte sul tavolo**: distribuzione delle 5 carte visibili disponibili per la pesca.
* **Biglietti di destinazione**: codifica dei biglietti posseduti e del loro stato di connettività attuale sul grafo.

#### 2. Spazio delle Azioni $\mathcal{A}$ (Discrete Action Space)
L'insieme $\mathcal{A} = \{0, 1, \dots, |\mathcal{A}|-1\}$ contiene tutte le mosse possibili del gioco discretizzate in un unico indice intero:
* $a = 0$: Pesca 2 carte dal mazzo coperto.
* $a \in [1, 5]$: Pesca una specifica carta scoperta dal tavolo tra le 5 visibili.
* $a \in [6, |\mathcal{A}|-2]$: Reclama una specifica tratta $(u, v)$ spendendo carte di colore $c$.
* $a = |\mathcal{A}|-1$: Pesca nuovi biglietti di destinazione dal mazzo degli obiettivi.

Dimensione complessiva: $|\mathcal{A}| = 56$ per la mappa **Mini**, $|\mathcal{A}| = 142$ per la mappa **USA**.

#### 3. Probabilità di Transizione dello Stato $\mathcal{P}(s_{t+1} | s_t, a_t)$
Rappresenta la dinamica dell'ambiente. In Ticket to Ride è un processo stocastico:
* Se l'agente reclama una tratta, la modifica sul tabellone è **deterministica**.
* Se l'agente pesca dal mazzo coperto, il nuovo stato delle carte in mano è **stocastico**.
* Tra il turno $t$ e il turno $t+1$, l'avversario esegue la propria mossa, modificando ulteriormente il tabellone e le carte scoperte in modo non stazionario.

#### 4. Funzione di Ricompensa $\mathcal{R}(s_t, a_t, s_{t+1})$
Assegna un segnale scalare $r_t \in \mathbb{R}$ a ogni transizione:
* **Punti Tratta Immediati**: commisurati alla lunghezza $L$ della tratta reclamata ($L=1 \to +1, L=2 \to +2, L=3 \to +4, L=4 \to +7, L=6 \to +15$).
* **Progresso Obiettivi**: ricompensa positiva quando un percorso tra le città di un biglietto viene completato.
* **Penalità di Ritardo / Step Penalty**: piccolo costo (es. $-0.01$) per incentivare l'efficienza temporale.
* **Esito Finale**: ricompensa scalare a fine partita basata sul differenziale di punteggio totale rispetto all'avversario ($\text{Score}_{\text{agent}} - \text{Score}_{\text{opponent}}$).

#### 5. Fattore di Sconto $\gamma \in [0, 1)$
Definisce il valore presente delle ricompense future. Il **Ritorno Scontato Cumulativo** al tempo $t$ è definito da:
$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots$$

Nel nostro progetto impostiamo $\gamma = 0.99$.
* Se $\gamma = 0$, l'agente è completamente "miope" e massimizza solo la ricompensa immediata al passo successivo.
* Con $\gamma = 0.99$, l'agente possiede un **orizzonte temporale efficace** di:
  $$H_{\text{eff}} \approx \frac{1}{1 - \gamma} = \frac{1}{1 - 0.99} = 100 \text{ turni}$$
  Questo valore è ideale per una partita di Ticket to Ride (che dura tipicamente tra i 40 e gli 80 turni).

---

### 2.2 Action Masking nel Deep RL: Teoria, Derivazione e Numeri

#### 2.2.1 La Patologia delle Penalità Negative ("Negative Reward Penalty")
Un approccio ingenuo per impedire all'agente di eseguire mosse illegali consiste nel lasciare lo spazio delle azioni libero e assegnare una penalità severa (es. $r = -10$) ogni volta che l'agente seleziona un'azione non valida, terminando o reiterando il turno.

Questo approccio fallisce sistematicamente nei giochi strategici per tre ragioni matematiche:
1. **Esplosione Combinatoria dello Spazio di Esplorazione**: con 142 azioni di cui solo 5 legali, all'inizio del training un'esplorazione casuale sceglie un'azione valida solo nel $5 / 142 \approx 3.5\%$ dei casi. Il $96.5\%$ delle transizioni nel replay buffer conterrà campioni spazzatura con penalità negative.
2. **Distorsione della Funzione di Valore (Value Distortion)**: la rete neurale spenderà la maggior parte della sua capacità di rappresentazione (capacità dei pesi $\theta$) per imparare la superficie delle azioni illegali anziché distinguere tra mosse strategiche buone e mediocri.
3. **Comportamento Iper-Conservativo (Policy Paralysis)**: l'agente impara rapidamente che fare qualsiasi mossa comporta un rischio del $96.5\%$ di ricevere $-10$, portando la policy a collassare sulla prima singola azione lecita scoperta (es. continuare a pescare sempre dal mazzo coperto per tutta la partita).

#### 2.2.2 Formulazione Matematica dell'Action Masking
Sia $\mathbf{M}(s) \in \{0, 1\}^{|\mathcal{A}|}$ il vettore di maschera booleana calcolato deterministicamente dal motore di gioco al turno corrente:
$$M(s)_a = \begin{cases} 1 & \text{se l'azione } a \text{ rispetta le regole nello stato } s \\ 0 & \text{se l'azione } a \text{ è vietata (carte/vagoni insufficienti o tratta occupata)} \end{cases}$$

L'insieme delle azioni lecite è:
$$\mathcal{A}_{\text{valid}}(s) = \{ a \in \mathcal{A} \mid M(s)_a = 1 \}$$

---

#### 2.2.3 Action Masking in Value-Based RL (DQN)
La rete neurale stima i valori $Q(s, a)$ per tutte le $|\mathcal{A}|$ azioni contemporaneamente.
Prima di selezionare l'azione ottima tramite l'operatore $\arg\max$, applichiamo una maschera additiva:
$$Q_{\text{masked}}(s, a) = \begin{cases} Q(s, a) & \text{se } M(s)_a = 1 \\ -\infty \text{ (numericamente } -10^8\text{)} & \text{se } M(s)_a = 0 \end{cases}$$

La scelta dell'azione greedy diventa:
$$a^* = \arg\max_{a \in \mathcal{A}} Q_{\text{masked}}(s, a) \equiv \arg\max_{a \in \mathcal{A}_{\text{valid}}(s)} Q(s, a)$$

Poiché $-10^8 < Q(s, a)$ per qualsiasi stima finita, l'operatore $\arg\max$ selezionerà **con certezza matematica assoluta** un'azione lecita.

---

#### 2.2.4 Action Masking in Policy-Based RL (PPO)
La rete Actor produce un vettore di punteggi non normalizzati chiamati **logits** $\mathbf{z}(s) \in \mathbb{R}^{|\mathcal{A}|}$.
Prima di trasformare i logits in probabilità tramite la funzione Softmax, modifichiamo i logits delle azioni illegali ponendoli a un valore numericamente molto negativo:
$$z'(s)_a = \begin{cases} z(s)_a & \text{se } M(s)_a = 1 \\ -10^8 & \text{se } M(s)_a = 0 \end{cases}$$

La distribuzione di probabilità categorica risultante $\pi_\theta(a|s)$ è:
$$\pi_\theta(a|s) = \text{Softmax}(\mathbf{z}'(s))_a = \frac{e^{z'(s)_a}}{\sum_{j \in \mathcal{A}} e^{z'(s)_j}}$$

Poiché $e^{-10^8} = 0.0$ in precisione floating point IEEE-754:
1. Per ogni azione illegale $a \notin \mathcal{A}_{\text{valid}}(s)$:
   $$\pi_\theta(a|s) = \frac{0}{\sum_{j \in \mathcal{A}_{\text{valid}}(s)} e^{z(s)_j} + 0} = 0.0$$
2. Per ogni azione lecita $a \in \mathcal{A}_{\text{valid}}(s)$:
   $$\pi_\theta(a|s) = \frac{e^{z(s)_a}}{\sum_{j \in \mathcal{A}_{\text{valid}}(s)} e^{z(s)_j}}$$
3. La somma delle probabilità sull'intero spazio è esattamente unitaria:
   $$\sum_{a \in \mathcal{A}} \pi_\theta(a|s) = \sum_{a \in \mathcal{A}_{\text{valid}}(s)} \pi_\theta(a|s) = 1.0$$

---

#### 2.2.5 Calcolo dell'Entropia Categorica Mascherata
L'entropia di Shannon della policy $S[\pi_\theta(s)]$ misura il grado di esplorazione e casualità della distribuzione:
$$S[\pi_\theta(s)] = - \sum_{a \in \mathcal{A}_{\text{valid}}(s)} \pi_\theta(a|s) \ln \pi_\theta(a|s)$$

In PyTorch, calcolare ingenuamente $\pi(a|s) \ln \pi(a|s)$ su tutto lo spazio $\mathcal{A}$ provocherebbe un'indeterminazione numerica $0 \cdot \ln(0) = 0 \cdot (-\infty) = \text{NaN}$.
Per garantire stabilità numerica assoluta, istanziamo la classe `torch.distributions.Categorical` passando direttamente i logits mascherati $\mathbf{z}'(s)$:
```python
dist = torch.distributions.Categorical(logits=masked_logits)
entropy = dist.entropy()  # PyTorch gestisce internamente 0 * log(0) = 0 in modo stabile
```

---

#### 2.2.6 Esempio Numerico Passo-Passo di Action Masking
Consideriamo uno spazio con 4 azioni $\mathcal{A} = [a_0, a_1, a_2, a_3]$:
* $a_0$: Pesca dal mazzo coperto.
* $a_1$: Reclama Tratta A (richiede 3 vagoni blu - il giocatore ne ha 3).
* $a_2$: Reclama Tratta B (richiede 4 vagoni rossi - il giocatore ne ha 1).
* $a_3$: Pesca nuovi obiettivi.

Supponiamo che l'Actor produca i seguenti logits:
$$\mathbf{z} = [1.5, \; 3.2, \; 4.8, \; 0.5]$$
Notiamo che la rete predilige fortemente l'azione $a_2$ (logit massimo $= 4.8$). Tuttavia, l'azione $a_2$ è **illegale** perché il giocatore non possiede abbastanza carte rosse.

Vettore maschera di validità:
$$\mathbf{M} = [1, \; 1, \; 0, \; 1]$$

| Parametro | $a_0$ (Mazzo) | $a_1$ (Tratta A) | $a_2$ (Tratta B - Illegale) | $a_3$ (Obiettivi) | Somma |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logit grezzo $z$** | $1.5$ | $3.2$ | $4.8$ | $0.5$ | — |
| **Softmax senza mask** | $\frac{e^{1.5}}{154.67} = \mathbf{0.029}$ | $\frac{e^{3.2}}{154.67} = \mathbf{0.159}$ | $\frac{e^{4.8}}{154.67} = \mathbf{0.801}$ | $\frac{e^{0.5}}{154.67} = \mathbf{0.011}$ | **1.000** |
| **Logit mascherato $z'$** | $1.5$ | $3.2$ | $-10^8$ | $0.5$ | — |
| **Esponenziale $e^{z'}$** | $4.4817$ | $24.5325$ | $0.0000$ | $1.6487$ | **$30.6629$** |
| **Softmax mascherata $\pi$** | $\frac{4.4817}{30.6629} = \mathbf{0.146}$ | $\frac{24.5325}{30.6629} = \mathbf{0.800}$ | $\mathbf{0.000}$ | $\frac{1.6487}{30.6629} = \mathbf{0.054}$ | **1.000** |

**Risultato didattico**:
1. L'azione illegale $a_2$ riceve probabilità **0.000 esatta**.
2. Il gradiente associato ad $a_2$ durante il campionamento non verrà mai generato.
3. La preferenza della rete si sposta naturalmente sulla migliore azione lecita disponibile ($a_1$ con probabilità $80.0\%$).

---

### 2.3 Value-Based RL: Deep Q-Networks (DQN) & Double DQN (DDQN)

Gli algoritmi Value-Based mirano ad apprendere la **Funzione di Valore Azione Ottimale** $Q^*(s, a)$, definita come il massimo ritorno cumulativo atteso partendo dallo stato $s$, eseguendo l'azione $a$, e seguendo da quel momento in poi la strategia ottimale:
$$Q^*(s, a) = \max_\pi \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \;\middle|\; s_t = s, a_t = a \right]$$

Una volta nota $Q^*(s, a)$, la policy ottima è banalmente ottenuta scegliendo in ogni stato l'azione con il valore $Q$ più alto:
$$\pi^*(s) = \arg\max_{a \in \mathcal{A}_{\text{valid}}(s)} Q^*(s, a)$$

---

#### 2.3.1 L'Equazione di Ottimalità di Bellman
Il pilastro fondamentale su cui si regge tutto il Q-Learning è la scomposizione ricorsiva di Bellman:
$$Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{P}} \left[ \mathcal{R}(s, a, s') + \gamma \max_{a' \in \mathcal{A}_{\text{valid}}(s')} Q^*(s', a') \right]$$

Questa equazione afferma che il valore dell'azione corrente è uguale alla ricompensa immediata più il valore scontato della migliore azione possibile nello stato successivo $s'$.

Nel **Deep Q-Learning**, poiché lo spazio degli stati $\mathcal{S}$ è continuo e ad alta dimensione, non possiamo usare una tabella. Utilizziamo una rete neurale con parametri $\theta$, denotata come $Q_\theta(s, a) \approx Q^*(s, a)$.

---

#### 2.3.2 Le Tre Cause di Instabilità (The Deadly Triad)
Secondo la teoria di Sutton & Barto, l'apprendimento per rinforzo diventa instabile o divergente quando combina tre elementi contemporaneamente:
1. **Function Approximation (Approssimazione di Funzione)**: l'uso di reti neurali fa sì che l'aggiornamento del valore di uno stato modifichi inevitabilmente anche le stime degli stati vicini.
2. **Bootstrapping**: la stima corrente viene aggiornata utilizzando come bersaglio un'altra stima ($Q(s', a')$) anziché un ritorno reale osservato fino a fine partita.
3. **Off-Policy Learning**: i dati usati per l'addestramento provengono da una policy diversa (es. una policy esplorativa $\epsilon$-greedy con buffer di memoria storica) rispetto alla policy greedy che stiamo ottimizzando.

Per sconfiggere la *Deadly Triad*, DeepMind (Mnih et al., 2015) ha introdotto due soluzioni strutturali: l'**Experience Replay Buffer** e la **Target Network**.

```text
  ┌────────────────────────────────────────────────────────────────────────┐
  │                        ESPERIENZA NEL GIOCO                            │
  │     (s_t, a_t, r_t, s_{t+1}, d_t, M_{t+1})                             │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │ Salva transizione
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                      EXPERIENCE REPLAY BUFFER                          │
  │     Capacità N = 50.000 transizioni (rompe le correlazioni seriali)    │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │ Campiona mini-batch uniforme (B=64)
                                     ▼
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
  ┌───────────────────────────────┐                   ┌───────────────────────────────┐
  │     ONLINE NETWORK Q_θ        │                   │     TARGET NETWORK Q_θ⁻       │
  │  1. Calcola Q_θ(s_t, a_t)     │                   │  2. Valuta lo stato futuro:   │
  │  2. Sceglie a* in s_{t+1}     │                   │     Q_θ⁻(s_{t+1}, a*)         │
  └──────────────┬────────────────┘                   └───────────────┬───────────────┘
                 │                                                    │
                 └─────────────────────────┬──────────────────────────┘
                                           │
                                           ▼
                 ┌─────────────────────────────────────────────────────┐
                 │       TARGET DI BELLMAN (Double DQN)                │
                 │   y_t = r_t + γ (1 - d_t) Q_θ⁻(s_{t+1}, a*)         │
                 │   Loss MSE = || Q_θ(s_t, a_t) - y_t ||²             │
                 └─────────────────────────────────────────────────────┘
```

---

#### 2.3.3 Componenti Fondamentali di DQN
1. **Experience Replay Buffer ($\mathcal{D}$)**:
   Una memoria circolare FIFO di capacità $N = 50.000$. Conserva tuple:
   $$e_t = (s_t, a_t, r_t, s_{t+1}, d_t, \mathbf{M}_{t+1})$$
   dove $d_t \in \{0, 1\}$ è il flag di fine partita (*done*). Durante il training campioniamo un mini-batch di dimensione $B = 64$ uniformemente a caso da $\mathcal{D}$.
   * **Perché è vitale?** In una partita di Ticket to Ride, passi temporali consecutivi sono fortemente correlati. Allenare la rete su dati sequenziali viola l'assunzione fondamentale di campioni I.I.D. (indipendenti e identicamente distribuiti), portando all'oblio catastrofico. Il buffer distribuisce i campioni nel tempo.
2. **Target Network ($Q_{\theta^-}$)**:
   Una rete neurale identica alla rete principale ($Q_\theta$) ma con parametri congelati $\theta^-$.
   * **Perché è vitale?** Se usassimo $Q_\theta$ per calcolare sia la stima che il target $y_t = r + \gamma \max Q_\theta(s', a')$, ogni aggiornamento dei gradienti su $\theta$ sposterebbe istantaneamente anche il target di Bellman. Questo fenomeno ("inseguire un bersaglio mobile") provoca gravi oscillazioni o divergenza della perdita. Congelando $\theta^-$ e aggiornandolo solo periodicamente (es. ogni 500 step $\theta^- \leftarrow \theta$), il target rimane fisso e stabile.
3. **Esplorazione $\epsilon$-Greedy Decadente**:
   All'inizio dell'addestramento l'agente deve esplorare l'ambiente casualmente; man mano che apprende, deve sfruttare le conoscenze acquisite.
   $$a_t = \begin{cases} \text{Azione casuale uniforme in } \mathcal{A}_{\text{valid}}(s_t) & \text{con probabilità } \epsilon_t \\ \arg\max_{a \in \mathcal{A}_{\text{valid}}(s_t)} Q_\theta(s_t, a) & \text{con probabilità } 1 - \epsilon_t \end{cases}$$
   $\epsilon$ decade linearmente da $\epsilon_{\text{start}} = 1.0$ fino a $\epsilon_{\text{end}} = 0.05$ su $T_{\text{decay}} = 20.000$ step:
   $$\epsilon_t = \max\left(\epsilon_{\text{end}}, \; \epsilon_{\text{start}} - \frac{t}{T_{\text{decay}}}(\epsilon_{\text{start}} - \epsilon_{\text{end}})\right)$$

---

#### 2.3.4 La Patologia della Sovrastima (Overestimation Bias) e Double DQN
Nel DQN classico (Mnih et al., 2015), il target di Bellman è calcolato come:
$$y_t^{\text{DQN}} = r_t + \gamma (1 - d_t) \max_{a' \in \mathcal{A}_{\text{valid}}(s_{t+1})} Q_{\theta^-}(s_{t+1}, a')$$

**Dimostrazione analitica del Bias di Sovrastima**:
Supponiamo che per uno stato $s'$, i veri valori ottimi siano uguali per due azioni: $Q^*(s', a_1) = Q^*(s', a_2) = 5.0$.
A causa dell'approssimazione neurale e del rumore statistico, la rete stima questi valori con un errore stocastico $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, \sigma^2)$:
$$Q(s', a_1) = 5.0 + \epsilon_1, \quad Q(s', a_2) = 5.0 + \epsilon_2$$
Quando calcoliamo il target, applichiamo l'operatore $\max$:
$$\mathbb{E}[\max(Q(s', a_1), Q(s', a_2))] = 5.0 + \mathbb{E}[\max(\epsilon_1, \epsilon_2)]$$
Poiché il massimo di due variabili aleatorie a media nulla ha valore atteso strettamente positivo ($\mathbb{E}[\max(\epsilon_1, \epsilon_2)] > 0$ se $\sigma > 0$), il target $y_t$ risulterà **sistematicamente sovrastimato**. Questo errore positivo si accumula ad ogni iterazione di Bellman, portando a stime di Q completamente slegate dalla realtà.

**La Soluzione di Double DQN (Van Hasselt et al., 2016)**:
Double DQN risolve l'overestimation bias **disaccoppiando la selezione dell'azione dalla sua valutazione**:
1. **Selezione dell'Azione**: affidata alla rete principale $Q_\theta$:
   $$a^* = \arg\max_{a' \in \mathcal{A}_{\text{valid}}(s_{t+1})} Q_\theta(s_{t+1}, a')$$
2. **Valutazione del Valore**: affidata alla Target Network $Q_{\theta^-}$:
   $$y_t^{\text{DDQN}} = r_t + \gamma (1 - d_t) Q_{\theta^-}(s_{t+1}, a^*)$$

In questo modo, se la rete principale sovrastima casualmente un'azione a causa del rumore, la Target Network (avendo pesi diversi $\theta^-$) fornirà una stima non correlata, neutralizzando il bias di sovrastima.

---

#### 2.3.5 La Funzione di Perdita (Loss) e il Gradiente di Bellman
Dato un batch di $B$ transizioni, l'obiettivo è minimizzare l'errore quadratico medio (MSE) tra la stima corrente e il target di Bellman:
$$L^{\text{DQN}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left( Q_\theta(s_i, a_i) - y_i^{\text{DDQN}} \right)^2$$

Il gradiente rispetto ai pesi $\theta$ della rete principale è:
$$\nabla_\theta L^{\text{DQN}}(\theta) = \frac{2}{B} \sum_{i=1}^B \underbrace{\left( Q_\theta(s_i, a_i) - y_i^{\text{DDQN}} \right)}_{\text{Temporal Difference Error } \delta_i} \nabla_\theta Q_\theta(s_i, a_i)$$

*Nota fondamentale*: durante il calcolo del gradiente in PyTorch, il target $y_i^{\text{DDQN}}$ deve essere trattato come una costante fissa, applicando `.detach()` sul tensore della target network.

---

#### 2.3.6 Esempio Numerico Passo-Passo di Double DQN
Immaginiamo una singola transizione estratta dal Replay Buffer:
* Stato corrente $s_t$: l'agente ha eseguito l'azione $a_t = 1$ (pesca carta).
* Ricompensa ricevuta: $r_t = +2.0$.
* Stato non terminale: $d_t = 0$.
* Fattore di sconto: $\gamma = 0.99$.
* Nello stato successivo $s_{t+1}$, le azioni lecite sono $\mathcal{A}_{\text{valid}}(s_{t+1}) = \{0, 1, 2\}$ mentre l'azione $3$ è illegale ($\mathbf{M}_{t+1} = [1, 1, 1, 0]$).

**Passo 1: Valutazione della rete principale $Q_\theta(s_{t+1})$ con Masking**:
$$Q_\theta(s_{t+1}) = [3.5, \; 6.2, \; 5.8, \; 9.9]$$
Applicando il masking, l'azione 3 (che aveva il valore fittizio più alto $9.9$) viene azzerata:
$$Q_{\text{masked},\theta}(s_{t+1}) = [3.5, \; 6.2, \; 5.8, \; -10^8]$$
La migliore azione valida secondo la rete principale è:
$$a^* = \arg\max([3.5, \; 6.2, \; 5.8, \; -10^8]) = 1 \quad (\text{valore } 6.2)$$

**Passo 2: Valutazione tramite la Target Network $Q_{\theta^-}(s_{t+1})$**:
La target network produce le seguenti stime per $s_{t+1}$:
$$Q_{\theta^-}(s_{t+1}) = [3.2, \; 5.0, \; 5.4, \; 8.5]$$
Valutiamo l'azione selezionata $a^* = 1$ sulla Target Network:
$$Q_{\theta^-}(s_{t+1}, a^*) = Q_{\theta^-}(s_{t+1}, 1) = 5.0$$
*(Notiamo che nel DQN classico avremmo preso $\max(3.2, 5.0, 5.4) = 5.4$. Double DQN evita di prendere il massimo rumoroso e usa 5.0)*.

**Passo 3: Calcolo del Target di Bellman**:
$$y_t^{\text{DDQN}} = r_t + \gamma (1 - d_t) Q_{\theta^-}(s_{t+1}, a^*) = 2.0 + 0.99 \cdot (1 - 0) \cdot 5.0 = 2.0 + 4.95 = \mathbf{6.95}$$

**Passo 4: Calcolo dell'Errore di Bellman e della Loss**:
Supponiamo che per la coppia iniziale $(s_t, a_t = 1)$, la rete principale stimi attualmente:
$$Q_\theta(s_t, 1) = 6.00$$
* TD Error: $\delta_t = Q_\theta(s_t, 1) - y_t^{\text{DDQN}} = 6.00 - 6.95 = \mathbf{-0.95}$.
* Loss MSE: $L(\theta) = (\delta_t)^2 = (-0.95)^2 = \mathbf{0.9025}$.
* **Effetto del Gradiente**: Poiché $\delta_t < 0$, la discesa del gradiente modificherà i pesi $\theta$ per **aumentare** la stima di $Q_\theta(s_t, 1)$ da $6.00$ verso $6.95$.

---

### 2.4 Policy-Based RL: Proximal Policy Optimization (PPO) & Actor-Critic

Mentre DQN apprende indirettamente la funzione di valore $Q(s, a)$, i metodi **Policy Gradient** parametrizzano direttamente la strategia dell'agente come una distribuzione di probabilità $\pi_\theta(a|s)$ e ottimizzano i parametri $\theta$ tramite ascesa del gradiente per massimizzare il rendimento atteso:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

---

#### 2.4.1 Dal Teorema del Policy Gradient all'Actor-Critic
Il fondamentale **Policy Gradient Theorem** (Sutton et al., 1999) dimostra che il gradiente dell'obiettivo rispetto ai pesi della policy è:
$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \ln \pi_\theta(a|s) \cdot Q^\pi(s, a) \right]$$

Nell'algoritmo classico *REINFORCE*, $Q^\pi(s, a)$ viene approssimato tramite il ritorno Monte Carlo reale $G_t = \sum_{k=0}^T \gamma^k r_{t+k+1}$.
Tuttavia, il ritorno $G_t$ accumula la stocasticità di tutte le mosse future fino alla fine della partita, introducendo una **varianza enorme** che rende l'apprendimento estremamente lento e instabile.

Per abbattere la varianza senza introdurre bias, sottraiamo una funzione baseline $V(s)$ dipendente solo dallo stato. La quantità risultante è la **Advantage Function** (Funzione di Vantaggio):
$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

Il gradiente dell'Actor diventa:
$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \ln \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t) \right]$$

```text
               ┌────────────────────────────────────────────────────────┐
               │                  STATO OSSERVATO s_t                   │
               └───────────────────────────┬────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
      ┌─────────────────────────────┐             ┌─────────────────────────────┐
      │       ACTOR π_θ(a|s)        │             │       CRITIC V_φ(s)         │
      │   "Quale azione scelgo?"    │             │   "Quanto è buono lo stato?"│
      │   Produce Logits mascherati │             │   Produce scalare V(s)      │
      └──────────────┬──────────────┘             └──────────────┬──────────────┘
                     │                                           │
                     │  Campiona a_t                             │  Stima V(s_t), V(s_{t+1})
                     ▼                                           ▼
      ┌─────────────────────────────────────────────────────────────────────────┐
      │               GENERALIZED ADVANTAGE ESTIMATION (GAE)                    │
      │  1. Calcola TD error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)                 │
      │  2. Propaga Advantage: A_t = δ_t + (γ λ) A_{t+1}                        │
      │  3. Ritorno target Critic: R_t = A_t + V(s_t)                           │
      └────────────────────────────────────┬────────────────────────────────────┘
                                           │
                                           ▼
      ┌─────────────────────────────────────────────────────────────────────────┐
      │               PPO CLIPPED SURROGATE OPTIMIZATION                        │
      │  Ratio: r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)                          │
      │  L_CLIP(θ) = min( r_t A_t, clip(r_t, 1-ε, 1+ε) A_t )                    │
      └─────────────────────────────────────────────────────────────────────────┘
```

**Interpretazione intuitiva dell'Advantage**:
* Se $A(s, a) > 0$: l'azione $a$ ha prodotto un risultato **migliore della media** degli esiti possibili nello stato $s$. Il gradiente spingerà ad **aumentare** la probabilità $\pi_\theta(a|s)$.
* Se $A(s, a) < 0$: l'azione $a$ è stata **peggiore della media**. Il gradiente spingerà a **ridurre** la probabilità $\pi_\theta(a|s)$.

---

#### 2.4.2 Generalized Advantage Estimation (GAE)
Per stimare $A(s_t, a_t)$ con un controllo ottimale del bilanciamento tra bias e varianza, utilizziamo **GAE** (Schulman et al., 2015).

Definiamo prima l'**errore di Temporal Difference (TD Residual)** del Critic a 1-passo:
$$\delta_t^V = r_t + \gamma (1 - d_t) V_\phi(s_{t+1}) - V_\phi(s_t)$$

L'Advantage stimato tramite GAE($\gamma, \lambda$) è la media geometricamente ponderata di tutti gli errori TD futuri:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

In pratica, viene calcolato in modo efficiente con una **ricorsione all'indietro** su un rollout di $T$ passi:
$$\hat{A}_t^{\text{GAE}} = \delta_t^V + \gamma \lambda (1 - d_t) \hat{A}_{t+1}^{\text{GAE}}$$
dove $\hat{A}_T^{\text{GAE}} = 0$.

**Il ruolo cruciale di $\lambda \in [0, 1]$**:
* Se $\lambda = 0$: $\hat{A}_t = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ (Stima TD(0): **minima varianza**, ma bias elevato se il Critic non è ancora accurato).
* Se $\lambda = 1$: $\hat{A}_t = \sum_{k=0}^\infty \gamma^k r_{t+k+1} - V(s_t)$ (Stima Monte Carlo: **zero bias**, ma massima varianza).
* Nel nostro progetto impostiamo $\lambda = 0.95$: il perfetto compromesso empirico per giochi a medio/lungo orizzonte.

Il **Ritorno Target** per l'addestramento del Critic è ricavato direttamente sommando l'advantage alla stima iniziale:
$$\hat{R}_t = \hat{A}_t^{\text{GAE}} + V_\phi(s_t)$$

Infine, per stabilizzare l'apprendimento su mini-batch, normalizziamo gli Advantage a media nulla e varianza unitaria:
$$\hat{A}_t^{\text{norm}} = \frac{\hat{A}_t - \mu_{\hat{A}}}{\sigma_{\hat{A}} + 10^{-8}}$$

---

#### 2.4.3 Il Problema del "Policy Collapse" e la Ratio di Probabilità
Nei metodi di gradiente standard, l'aggiornamento dei parametri $\theta$ può provocare variazioni brusche nella distribuzione $\pi_\theta$. Poiché i dati di training successivi vengono raccolti dalla nuova policy, se un singolo passo di gradiente rende la policy catastrofica, l'agente non riuscirà più a raccogliere buone traiettorie, innescando un collasso irreversibile delle prestazioni (**Policy Collapse**).

Per controllare l'ampiezza dell'aggiornamento, definiamo il **Probability Ratio** $r_t(\theta)$:
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$
* Quando iniziamo ad ottimizzare sul batch raccolto, $\theta = \theta_{\text{old}}$, quindi $r_t(\theta) = 1.0$.
* Se $r_t(\theta) > 1.0$, l'azione $a_t$ è diventata più probabile sotto la nuova policy.
* Se $r_t(\theta) < 1.0$, l'azione $a_t$ è diventata meno probabile.

---

#### 2.4.4 Il Clipped Surrogate Objective di PPO
PPO (Schulman et al., 2017) impedisce a $r_t(\theta)$ di allontanarsi troppo da $1.0$ applicando un operatore di **clipping** con un iperparametro $\epsilon = 0.2$:
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right) \right]$$

dove:
$$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) = \begin{cases} 1 - \epsilon & \text{se } r_t(\theta) < 1 - \epsilon \\ 1 + \epsilon & \text{se } r_t(\theta) > 1 + \epsilon \\ r_t(\theta) & \text{altrimenti} \end{cases}$$

```text
       L^CLIP
         ▲
         │               Advantage A > 0 (Azione Buona)
         │                                   / (Non clippato)
   (1+ε)A├───────────────────────============
         │                      /
        A├─────────────────────/
         │                    /
         │                   /
         │                  /
         │                 /
─────────┼────────────────┼──────────┼───────────────► r_t(θ)
         0               1-ε        1+ε
```

**Analisi Rigorosa dei Casi**:
1. **Caso Advantage Positivo ($\hat{A}_t > 0$, Azione Buona)**:
   * Vogliamo premiare questa azione aumentando $\pi_\theta(a_t|s_t)$, quindi $r_t(\theta) > 1.0$.
   * Se $r_t(\theta) \le 1 + \epsilon$ (es. $r_t \le 1.2$), l'obiettivo è $r_t \hat{A}_t$ e il gradiente spinge regolarmente ad aumentare la probabilità.
   * Se $r_t(\theta) > 1 + \epsilon$ (es. $r_t = 1.5$), interviene il clipping: $\min(1.5 \hat{A}_t, 1.2 \hat{A}_t) = 1.2 \hat{A}_t$. La funzione diventa costante rispetto a $\theta$ e il **gradiente si azzera**, impedendo alla policy di diventare eccessivamente sicura o deterministica in un solo passo.
2. **Caso Advantage Negativo ($\hat{A}_t < 0$, Azione Cattiva)**:
   * Vogliamo disincentivare questa azione riducendo $\pi_\theta(a_t|s_t)$, quindi $r_t(\theta) < 1.0$.
   * Se $r_t(\theta) \ge 1 - \epsilon$ (es. $r_t \ge 0.8$), il gradiente riduce la probabilità.
   * Se $r_t(\theta) < 1 - \epsilon$ (es. $r_t = 0.5$), il termine clippato è $(1-\epsilon)\hat{A}_t = 0.8 \hat{A}_t$. Poiché $\hat{A}_t < 0$, si ha $0.5 \hat{A}_t > 0.8 \hat{A}_t$, quindi il $\min$ seleziona il termine non clippato, permettendo alla policy di non penalizzare oltre misura un'azione già ampiamente ridotta.

---

#### 2.4.5 La Funzione di Perdita Complessiva di PPO
L'ottimizzazione congiunta dell'Actor e del Critic minimizza la seguente loss multi-obiettivo:
$$L^{\text{PPO}}(\theta, \phi) = - L^{\text{CLIP}}(\theta) + c_{\text{vf}} L^{\text{VF}}(\phi) - c_{\text{ent}} S[\pi_\theta]$$

dove:
1. **$- L^{\text{CLIP}}(\theta)$**: Funzione di perdita dell'Actor. Il segno negativo è necessario perché gli ottimizzatori PyTorch eseguono la discesa del gradiente (minimizzazione), mentre noi vogliamo massimizzare il rendimento surrogato.
2. **$L^{\text{VF}}(\phi) = \frac{1}{B} \sum_{i=1}^B \left( V_\phi(s_i) - \hat{R}_i \right)^2$**: Errore quadratico medio del Critic per allineare la stima del valore $V_\phi(s)$ ai ritorni empirici GAE $\hat{R}_i$. Ponderato da $c_{\text{vf}} = 0.5$.
3. **$- c_{\text{ent}} S[\pi_\theta]$**: Bonus di entropia. Sottrarre l'entropia equivale a massimizzarla, impedendo alla policy di collassare prematuramente su una singola azione e preservando l'esplorazione. Ponderato da $c_{\text{ent}} = 0.01$.

---

#### 2.4.6 Metriche Diagnostiche di Monitoraggio in Tempo Reale
Durante l'addestramento di PPO non ci limitiamo a osservare la loss totale, ma monitoriamo 5 metriche diagnostiche essenziali:

1. **Approximate KL Divergence ($\hat{D}_{\text{KL}}$)**:
   Misura quanto la nuova policy $\pi_\theta$ si è allontanata dalla vecchia $\pi_{\theta_{\text{old}}}$:
   $$\hat{D}_{\text{KL}} = \frac{1}{B} \sum_{i=1}^B \left( \frac{\pi_\theta(a_i|s_i)}{\pi_{\theta_{\text{old}}}(a_i|s_i)} - 1 - \ln \frac{\pi_\theta(a_i|s_i)}{\pi_{\theta_{\text{old}}}(a_i|s_i)} \right) \approx \frac{1}{B} \sum_{i=1}^B \left( \ln \pi_{\theta_{\text{old}}}(a_i|s_i) - \ln \pi_\theta(a_i|s_i) \right)$$
   * Se $\hat{D}_{\text{KL}} \in [0.005, 0.015]$: l'aggiornamento è sano e stabile.
   * Se $\hat{D}_{\text{KL}} > 1.5 \times \text{target\_kl}$ (es. $> 0.03$): scatta l'**Early Stopping** delle epoche correnti per proteggere la policy.
2. **Explained Variance ($\text{EV}$)**:
   Misura la capacità del Critic di spiegare la variabilità dei ritorni reali:
   $$\text{EV} = 1 - \frac{\text{Var}(\hat{R} - V_\phi(s))}{\text{Var}(\hat{R})}$$
   * $\text{EV} \to 1.0$: il Critic predice i ritorni con eccellente accuratezza.
   * $\text{EV} \approx 0.0$: il Critic non fa meglio della semplice media aritmetica.
   * $\text{EV} < 0.0$: il Critic sta introducendo rumore nocivo (segnale di instabilità del learning rate).
3. **Clip Fraction**:
   Frazione percentuale di campioni del mini-batch per cui $|r_t(\theta) - 1.0| > \epsilon$. Un valore sano oscilla tipicamente tra il $5\%$ e il $20\%$.

---

#### 2.4.7 Esempio Numerico Passo-Passo di PPO e GAE
Immaginiamo una mini-traiettoria di 2 passi temporali ($t=0$ e $t=1$):
* $s_0$: l'agente esegue $a_0$, riceve $r_0 = 1.0$, $d_0 = 0$. Stima Critic: $V_\phi(s_0) = 3.0$.
* $s_1$: l'agente esegue $a_1$, riceve $r_1 = 4.0$, $d_1 = 1$ (partita terminata). Stima Critic: $V_\phi(s_1) = 5.0$.
* Parametri: $\gamma = 0.99$, $\lambda = 0.95$.

**Passo 1: Calcolo degli Errori TD ($\delta^V$)**:
* Al passo $t=1$ (terminale $d_1 = 1 \implies V(s_2) = 0$):
  $$\delta_1^V = r_1 + \gamma (1 - d_1) V(s_2) - V(s_1) = 4.0 + 0 - 5.0 = \mathbf{-1.0}$$
* Al passo $t=0$:
  $$\delta_0^V = r_0 + \gamma (1 - d_0) V(s_1) - V(s_0) = 1.0 + 0.99 \cdot 5.0 - 3.0 = 1.0 + 4.95 - 3.0 = \mathbf{+2.95}$$

**Passo 2: Calcolo Ricorsivo di GAE ($\hat{A}$)**:
* $\hat{A}_1^{\text{GAE}} = \delta_1^V = \mathbf{-1.0}$
* $\hat{A}_0^{\text{GAE}} = \delta_0^V + (\gamma \lambda)(1 - d_0) \hat{A}_1^{\text{GAE}} = 2.95 + (0.99 \cdot 0.95) \cdot 1 \cdot (-1.0) = 2.95 - 0.9405 = \mathbf{+2.0095}$

**Passo 3: Calcolo dei Target per il Critic ($\hat{R}$)**:
* $\hat{R}_0 = \hat{A}_0 + V(s_0) = 2.0095 + 3.0 = \mathbf{5.0095}$
* $\hat{R}_1 = \hat{A}_1 + V(s_1) = -1.0 + 5.0 = \mathbf{4.0000}$

**Passo 4: Calcolo del Clipped Surrogate per $t=0$**:
* Supponiamo $\hat{A}_0 = +2.0$.
* Vecchia probabilità: $\pi_{\theta_{\text{old}}}(a_0|s_0) = 0.20$.
* Durante l'aggiornamento, la nuova policy produce: $\pi_\theta(a_0|s_0) = 0.30$.
* Ratio: $r_0(\theta) = \frac{0.30}{0.20} = \mathbf{1.50}$.
* Con $\epsilon = 0.2$, l'intervallo di clipping è $[0.8, 1.2]$.
* Termine non clippato: $r_0 \hat{A}_0 = 1.50 \cdot 2.0 = \mathbf{3.0}$.
* Termine clippato: $\text{clip}(1.50, 0.8, 1.2) \cdot \hat{A}_0 = 1.20 \cdot 2.0 = \mathbf{2.4}$.
* Obiettivo finale: $L^{\text{CLIP}} = \min(3.0, 2.4) = \mathbf{2.4}$.

**Risultato didattico**: Il gradiente premia la buona mossa $a_0$, ma il clipping impedisce un aggiornamento eccessivo ($3.0 \to 2.4$), garantendo la massima stabilità e preservando l'integrità della policy.

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
