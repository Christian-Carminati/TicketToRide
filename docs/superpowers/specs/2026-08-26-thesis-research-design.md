# Specifiche di Progetto: Dalla Memoria Neurale alla Deduzione Bayesiana nei Giochi a Informazione Imperfetta su Grafo

- **Data:** 2026-08-26
- **Stato:** Approvato (Draft Validato)
- **Target:** Tesi di Laurea Magistrale / Report di Ricerca Accademica / Scientific Paper
- **Titolo Ufficiale (Italiano):** *Dalla Memoria Neurale alla Deduzione Probabilistica: Studio Comparativo di Agenti per Giochi a Informazione Imperfetta su Grafo*
- **Official Title (English):** *From Recurrent Memory to Bayesian Deduction: A Comparative Study of Decision-Making Paradigms in Imperfect-Information Network Games*
- **Lingue Supportate (Bilingual Support):** Italiano 🇮🇹 & English 🇬🇧 (Doppia lingua per la Lezione 13 interattiva, report di ricerca, figure scientifiche e template tesi/paper).

---

## 1. Abstract e Visione Scientifica / Scientific Vision (Bilingual)

### 🇮🇹 Italiano
I giochi da tavolo strategici su grafo con informazione parziale asimmetrica (come *Ticket to Ride*) presentano sfide uniche per l'Intelligenza Artificiale:
1. **Spazio degli stati e delle azioni combinatorio**: la topologia planare della rete ferroviaria impone vincoli di connettività, colli di bottiglia e contesa di risorse scarse.
2. **Informazione imperfetta asimmetrica**: le carte pescate dal mazzo cieco e i biglietti obiettivo segreti degli avversari non sono direttamente osservabili (POMDP).
3. **Orizzonte temporale lungo e reward ritardato**: il completamento o fallimento dei biglietti viene valutato a fine partita, con penalità simmetriche catastrofiche.

Questo lavoro di tesi indaga e confronta in modo sistematico due filosofie fondamentali per affrontare l'informazione parziale:
- **L'approccio implicito / end-to-end**: memorie neurali ricorrenti (Recurrent PPO con LSTM).
- **L'approccio esplicito / deduttivo**: tracciatore di credenza Bayesiana con detour su grafo unito ad AlphaZero (Belief-Weighted MCTS).

### 🇬🇧 English
Strategic graph-based board games with asymmetric partial observability (such as *Ticket to Ride*) pose distinct challenges for AI:
1. **Combinatorial state-action space**: planar railway topology enforces connectivity constraints, bottlenecks, and sparse resource contention.
2. **Asymmetric imperfect information**: cards drawn from the blind deck and secret destination tickets are unobservable (POMDP).
3. **Long horizons and delayed rewards**: terminal ticket evaluation induces high-stakes symmetric penalties.

This thesis systematically investigates and compares two core paradigms:
- **Implicit / End-to-End**: Recurrent neural memory (Recurrent PPO with LSTM).
- **Explicit / Deductive**: Bayesian belief tracking over graph detours coupled with AlphaZero (Belief-Weighted Neural MCTS).

Il lavoro è validato attraverso una pipeline sperimentale multi-seed su larga scala resa possibile dal motore di simulazione nativo ad altissime prestazioni in Rust (~1.44M step/s) sviluppato all'interno del progetto.

---

## 2. Domande di Ricerca Formali ($RQ_1 - RQ_4$)

* **$\mathbf{RQ_1}$ (Impatto della Memoria Neurale in POMDP):**  
  *In quale misura l'introduzione di una memoria ricorrente (LSTM) migliora la qualità decisionale, la gestione delle risorse e l'Elo rating rispetto a una policy Markoviana reattiva (MLP) quando l'informazione è parzialmente osservabile?*

* **$\mathbf{RQ_2}$ (Determinizzazione Guidata da Credenza in MCTS):**  
  *L'utilizzo della distribuzione a posteriori $P(T_k \mid \mathcal{E}_{\text{opp}})$ per campionare gli stati nascosti (*Belief-Weighted Determinization*) riduce l'errore di *Strategy Fusion* e migliora l'efficacia tattica rispetto alla determinizzazione uniforme casuale (Information-Set MCTS)?*

* **$\mathbf{RQ_3}$ (Dinamica e Convergenza dell'Inferenza Bayesiana):**  
  *Con quanta rapidità (misurata tramite il decadimento dell'entropia di Shannon e la Top-$K$ accuracy) il motore di detour su grafo riesce a circoscrivere gli obiettivi segreti dell'avversario durante lo svolgimento della partita?*

* **$\mathbf{RQ_4}$ (Resistenza al Bluff e Frontiera Computazionale):**  
  *Come reagiscono i due paradigmi (neurale implicito vs bayesiano esplicito) in presenza di rumore stocastico o mosse deliberate di disinformazione (bluff), e qual è il trade-off empirico tra qualità della decisione (Elo) ed efficienza temporale (millisecondi per mossa)?*

---

## 3. I 6 Paradigmi Decisionali a Confronto

La tesi adotta un protocollo di triangolazione metodologica confrontando 6 agenti progettati per isolare ogni singola innovazione algoritmica:

```
                                  [ SPAZIO DEGLI AGENTI ]
                                             │
      ┌──────────────────────┬───────────────┴───────────────┬──────────────────────┐
      ▼                      ▼                               ▼                      ▼
  [ EURISTICO ]       [ MODEL-FREE RL ]              [ TREE SEARCH ]        [ BAYESIAN ALPHA-ZERO ]
  Dijkstra Cammino    - Flat PPO (MLP)               - ISMCTS Uniforme      - AlphaZero + Detour
  Minimo Greedy       - Recurrent PPO (LSTM)         - Neural AlphaZero       Belief Tracker
```

### 3.1 Dettaglio Matematico degli Agenti

1. **`Heuristic-Dijkstra` (Baseline Deterministica)**:
   * Calcola il cammino minimo su grafo tra le città dei propri biglietti tramite Dijkstra pesato sulla lunghezza delle tratte libere.
   * Raccoglie carte necessarie e occupa prioritariamente le tratte appartenenti al cammino minimo.
2. **`Flat-PPO` (Deep RL Reattivo)**:
   * Policy $\pi_\theta(a \mid o_t)$ e Value $V_\phi(o_t)$ parametrizzate da un MLP a 3 strati con connessioni residue.
   * Valuta l'osservazione istantanea $o_t$ senza memoria degli eventi passati.
3. **`Recurrent-PPO` (Deep RL con Memoria Neurale Implicita)**:
   * Architettura Actor-Critic con cella LSTM a 256 unità:
     $$h_t, c_t = \text{LSTM}(f_{\text{enc}}(o_t), (h_{t-1}, c_{t-1}))$$
     $$\pi_\theta(a \mid h_t) = \text{Softmax}(\text{MaskedLogits}(W_\pi h_t)), \quad V_\phi(h_t) = W_v h_t$$
   * Addestrato con BPTT su sequenze temporali (`seq_len = 8`) con maschere di validità.
4. **`Uniform-ISMCTS` (Information-Set MCTS Classico)**:
   * Genera determinizzazioni $\tilde{s} \sim \mathcal{U}(\text{Compatibili}(o_t))$ campionando uniformemente carte e biglietti avversari.
   * Esegue simulazioni UCT con rollout euristici/casuali.
5. **`Neural-AlphaZero` (PUCT Search + Policy-Value Network)**:
   * Rete neurale dual-head $(\mathbf{p}_\theta, v_\theta) = f_\theta(o_t)$.
   * Ricerca ad albero senza rollout casuali, guidata dalla formula PUCT con rumore di Dirichlet alla radice:
     $$a^* = \arg\max_a \left( Q(s, a) + c_{\text{puct}} \cdot P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right)$$
6. **`Bayesian-AlphaZero` (Belief-Weighted PUCT + Tactical Blocker)**:
   * **Detour Engine su Grafo**: Per ogni tratta occupata $e=(a, b)$ e ogni biglietto $T_k=(u, v)$:
     $$\Delta(e, T_k) = \min(d(u, a) + \text{len}(e) + d(b, v), d(u, b) + \text{len}(e) + d(a, v)) - d(u, v)$$
   * **Aggiornamento Bayesiano a Posteriori**:
     $$P(T_k \mid \mathcal{E}_{1:t}) \propto P(T_k) \cdot \prod_{\tau=1}^t \exp(-\beta \cdot \Delta(e_\tau, T_k))$$
   * **Determinizzazione Pesata**: Campionamento dei mondi possibili proporzionale a $P(T_k \mid \mathcal{E})$.
   * **Tactical Chokepoint Blocker**: Boosting selettivo del prior $P(s, e)$ per tratte che costituiscono ponti di taglio (cut-edges) per i biglietti ad alta probabilità dell'avversario.

---

## 4. Protocollo Sperimentale e Pipeline dei Risultati

La pipeline scientifica è articolata in 5 batterie sperimentali automatizzate:

### 4.1 Le 5 Batterie Sperimentali

| Batteria | Obiettivo | Metriche Raccolte | Output Scientifico |
| :--- | :--- | :--- | :--- |
| **1. Torneo Round-Robin** | Misurare la gerarchia di forza globale | Win rate, Elo rating, 95% CI, Score $\Delta$ | `fig1_elo_tournament_matrix.pdf`, `table1_main_results.tex` |
| **2. Convergenza Bayesiana** | Valutare l'accuratezza del filtro probabilistico | Entropia di Shannon $H(t)$, Top-1/Top-3 Accuracy | `fig2_bayesian_entropy_convergence.pdf` |
| **3. Ablation Determinizzazione** | Isolare l'impatto della determinizzazione pesata | Win rate vs MCTS uniforme, Strategy Fusion errors | `fig3_ablation_determinization.pdf`, `table2_ticket_completion_rates.tex` |
| **4. Resistenza al Bluff/Inganno** | Testare la robustezza a mosse fuorvianti | Degradazione Elo in funzione della % di bluff | `fig4_deception_noise_robustness.pdf` |
| **5. Frontiera Computazionale** | Analizzare il trade-off qualità/tempo | Latenza (ms/mossa), throughput (sim/s), Elo | `fig5_compute_vs_elo_frontier.pdf`, `table3_computational_profile.tex` |

### 4.2 Artefatti della Pipeline Software

* `src/evaluation/thesis_benchmark.py`: Engine scientifico con calcolo statistico avanzato (Bootstrap confidence intervals, Elo Bayesiano, matrici di contingenza).
* `scripts/run_thesis_study.py`: Script CLI per esecuzione headless multi-process parametrizzabile (`--seeds`, `--matchups`, `--output-dir`).
* `scripts/generate_thesis_figures.py`: Generatore di figure vettoriali publication-grade (300 DPI, stile accademico Seaborn/Matplotlib) e tabelle LaTeX pronte per l'inclusione nella tesi.

---

## 5. Struttura Monografica della Tesi di Laurea

* **Capitolo 1: Introduzione e Formulazione del Problema**
  * Sfide dell'IA nei giochi moderni; informazione asimmetrica; formulazione formale del POMDP su grafo; $RQ_1 - RQ_4$ e contributi originali.
* **Capitolo 2: Stato dell'Arte e Fondamenti Teorici**
  * Teoria dei giochi a informazione imperfetta; Policy Gradient e PPO; Recurrent Neural Networks in RL; ISMCTS e problemi di Strategy Fusion; AlphaZero e Policy-Value Networks; Inferenza Bayesiana e Opponent Modeling.
* **Capitolo 3: Architettura del Sistema e Metodologia**
  * Core Rust deterministico e Gymnasium environment; Action masking e garanzie di anti-leakage; Formalizzazione dei 6 agenti; Motore di Detour Dijkstra e aggiornamento a posteriori.
* **Capitolo 4: Risultati Sperimentali e Discussione**
  * Analisi dei tornei multi-seed e ranking Elo; Dinamiche dell'entropia Bayesiana; Studio di ablation su determinizzazione e memoria; Esperimenti di inganno/bluff; Frontiera di Pareto efficienza-qualità.
* **Capitolo 5: Introspezione e Casi di Studio Qualitativi (Web Lab)**
  * Telemetria real-time; Analisi dettagliata di match emblematici; Heatmap delle contese sui colli di bottiglia; Mappe di attivazione neurale.
* **Capitolo 6: Conclusioni e Sviluppi Futuri**
  * Risposte alle domande di ricerca; Limiti dello studio; Direzioni future (Graph Neural Networks, generalizzazione su mappe procedurali, estensione multi-player a 3-5 giocatori).

---

## 6. Integrazione nel Portale Didattico del Corso: Lezione 13 (Bilingue IT / EN)

* **File Creato:** `docs/course/lesson_13_scientific_research_paper.html`
* **Titolo:** *Lezione 13: Metodologia di Ricerca Scientifica, Benchmark Cross-Paradigma e Redazione Accademica / Scientific Research Methodology & Academic Benchmark*
* **Supporto Bilingue Dinamico:** Switch interattivo con pulsante lingua (🇮🇹 Italiano / 🇬🇧 English) per visualizzare l'intera lezione, formule, diagrammi e spiegazioni nella lingua desiderata in tempo reale.
* **Contenuti Pedagogici:**
  1. *L'Intuizione / Intuition:* Dalla costruzione del codice alla formulazione di ipotesi scientifiche rigorose.
  2. *I 6 Paradigmi a Confronto / 6 Paradigms Compared:* Mappa concettuale interattiva e albero decisionale dei modelli.
  3. *Il Metodo Sperimentale / Experimental Rigor:* Multi-seed, intervalli di confidenza al 95%, test di significatività e metriche Elo.
  4. *Dentro il Motore Bayesiano / Inside the Bayesian Filter:* Formule di detour su grafo, probabilità a posteriori ed eliminazione dello Strategy Fusion.
  5. *Dall'Esperimento alla Tesi / From Experiments to Monograph:* Come strutturare capitoli, analizzare i casi di studio nel Web Lab e discutere i dati.
  6. *Interactive Bilingual Quiz:* Quiz di autovalutazione finale con rendering MathJax.
* **Aggiornamento Globale:**
  * Aggiornata la sidebar di navigazione di tutte le lezioni (`index.html` e `lesson_01.html` fino a `lesson_12.html`).
  * Aggiornato `docs/course/course.js` per il tracciamento del progresso e il conteggio delle 13 lezioni totali.

---

## 7. Criteri di Accettazione e Verifica

1. **Riproducibilità:** Ogni esperimento deve essere deterministicamente riproducibile fornendo il seed iniziale.
2. **Integrità Anti-Leakage:** Nessuna informazione nascosta (biglietti o mazzo avversario) deve essere accessibile agli agenti se non attraverso la storia pubblica osservabile.
3. **Rigore Statistico:** I risultati dei tornei devono riportare intervalli di confidenza al 95% e test di significatività statistica.
4. **Qualità Tipografica:** Le figure generate devono essere in formato vettoriale PDF e PNG ad alta risoluzione (300 DPI) con font chiari e palette colori accessibili.
5. **Integrità Web Lab & Course Portal:** La Lezione 13 deve essere navigabile, con MathJax funzionante, quiz interattivo e layout coerente con le precedenti 12 lezioni.
