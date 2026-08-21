# Specifica di Design: Fase 10 — Generalizzazione e Mappe Procedurali

**Data:** 2026-08-21  
**Fase:** 10 (Generalizzazione, Ambienti Procedurali, Mappa Ufficiale Europa, Valutazione Cross-Mappa)  
**Stato:** Approvato dall'Utente  

---

## 1. Obiettivi e Visione Scientifica

La Fase 10 risponde alla domanda scientifica cardine di TicketToRide RL Lab:
> **"L'agente di Reinforcement Learning ha appreso principi strategici generali e astratti di Ticket to Ride, oppure si è limitato a sovradattarsi (overfitting) memorizzando la topologia specifica della mappa USA standard?"**

Per rispondere a questa domanda con rigore metodologico, la Fase 10 introduce:
1. **Generatore Deterministico di Mappe Procedurali (`ProceduralMapGenerator`):** Generazione procedurale basata sulla teoria dei grafi di mappe giocabili, garantite al 100% connesse, con bilanciamento cromatico, lunghezze coerenti e Biglietti di Destinazione calcolati sui cammini minimi.
2. **Mappa Ufficiale Europa (`load_europe_board`):** Ricostruzione fedele del tabellone ufficiale di Ticket to Ride Europa (città europee con coordinate, tratte ufficiali con colori e doppie tratte, deck dei biglietti di destinazione classici e lunghi).
3. **Infrastruttura di Dataset e Split (`MapSplit`, `ProceduralMapDataset`):** Partizionamento formale in distribuzioni di mappe di Addestramento (Train), Validazione (Val) e Test (Unseen / mai viste).
4. **Ambiente di Addestramento Multi-Mappa (`MultiMapTicketToRideEnv`):** Ambiente compatibile con Gymnasium che campiona topologie diverse a ogni reset, consentendo l'addestramento dell'agente RL su variabilità topologica.
5. **Framework di Valutazione della Generalizzazione (`GeneralizationEvaluator`, `GeneralizationBenchmarkRunner`):** Formalizzazione matematica del Generalization Gap ($\Delta_{\text{gen}}$), del Relative Retention Rate ($R_{\text{ret}}$), del mantenimento del Win Rate e dell'efficienza nel completamento dei biglietti su topologie inedite.
6. **CLI e Suite di Accettazione:** Script dedicato `scripts/benchmark_generalization.py` e suite completa di test TDD (`test_procedural_maps.py`, `test_europe_board.py`, `test_multi_map_env.py`, `test_generalization.py`, `test_phase10_acceptance.py`).

---

## 2. Formalizzazione Matematica e Metriche

### 2.1 Generalization Gap ($\Delta_{\text{gen}}$)
Dato un agente valutato su un insieme di mappe di addestramento $\mathcal{M}_{\text{train}}$ e su un insieme di mappe di test mai viste $\mathcal{M}_{\text{test}}$:

$$\overline{S}_{\text{train}} = \frac{1}{|\mathcal{M}_{\text{train}}|} \sum_{m \in \mathcal{M}_{\text{train}}} \mathbb{E}[\text{Score}(m)]$$

$$\overline{S}_{\text{test}} = \frac{1}{|\mathcal{M}_{\text{test}}|} \sum_{m' \in \mathcal{M}_{\text{test}}} \mathbb{E}[\text{Score}(m')]$$

$$\Delta_{\text{gen}} = \overline{S}_{\text{train}} - \overline{S}_{\text{test}}$$

Un Generalization Gap prossimo a zero (o negativo) indica che l'agente generalizza efficacemente la propria capacità decisionale senza soffrire il cambio di mappa.

### 2.2 Relative Performance Retention ($R_{\text{ret}}$)
$$R_{\text{ret}} = \frac{\overline{S}_{\text{test}}}{\max(1.0, \overline{S}_{\text{train}})} \times 100\%$$

Rappresenta la percentuale di punteggio preservata dall'agente quando opera su topologie inedite rispetto al suo benchmark su mappe note.

### 2.3 Win Rate Cross-Mappa e Completamento Biglietti
- **Win Rate su Mappe Inedite ($WR_{\text{test}}$):** Tasso di vittoria testa a testa contro gli avversari di riferimento (Random, Greedy, Strategic) sulle mappe di test.
- **Ticket Completion Rate ($TCR_{\text{test}}$):** Percentuale di Destination Tickets completati con successo su grafi non noti a priori.

---

## 3. Architettura e Componenti Software

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MAP INFRASTRUCTURE                                 │
│                                                                             │
│  ┌─────────────────────────┐  ┌───────────────────────┐  ┌────────────────┐ │
│  │ ProceduralMapGenerator  │  │  load_europe_board()  │  │ load_usa_board │ │
│  │ (MST + Delaunay Edges)  │  │   (Official Europe)   │  │ (Official USA) │ │
│  └────────────┬────────────┘  └───────────┬───────────┘  └───────┬────────┘ │
│               │                           │                      │          │
│               ▼                           │                      │          │
│  ┌─────────────────────────┐              │                      │          │
│  │   ProceduralMapDataset  │              │                      │          │
│  │  (Train / Val / Test)   │              │                      │          │
│  └────────────┬────────────┘              │                      │          │
└───────────────┼───────────────────────────┼──────────────────────┼──────────┘
                │                           │                      │
                ▼                           ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GYMNASIUM / RL ENVIRONMENT                              │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                       MultiMapTicketToRideEnv                         │  │
│  │  - Resets with dynamic sampling over MapSplit                         │  │
│  │  - Canonical topology mapping for stable Observation & Action shapes  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION & BENCHMARK SUITE                             │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                       GeneralizationEvaluator                         │  │
│  │  - Computes cross-map Head-to-Head & Generalization Gap               │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    GeneralizationBenchmarkRunner                      │  │
│  │  - Single-Map PPO vs Multi-Map PPO vs StrategicAgent vs GreedyAgent   │  │
│  │  - USA <-> Europe Cross-Map Benchmarking                              │  │
│  │  - Generates phase10_report.json & phase10_report.md                  │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                  scripts/benchmark_generalization.py                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Generazione Mappe Procedurali (`src/game/procedural.py`)

#### Algoritmo Dettagliato
1. **Campionamento Città Spaziali:**
   - Campiona $N$ città con coordinate $(x, y) \in [0.05, 0.95]^2$.
   - Applica un rejection sampling per garantire una distanza minima euclidea $d_{\min} = 0.15$ tra tutte le coppie di città, impedendo addensamenti innaturali o sovrapposizioni grafiche.
2. **Topologia e Garanzia di Connettività:**
   - Costruisce il grafo completo delle distanze euclidee e ne calcola il **Minimum Spanning Tree (MST)**: questo assicura matematicamente che l'intero grafo sia connesso in un'unica componente (nessuna città isolata o irraggiungibile).
   - Aggiunge ulteriori $k$ archi tra vicini prossimi (k-nearest neighbors) fino al raggiungimento del target $R$ di tratte desiderato, creando percorsi alternativi e anelli strategici.
3. **Assegnazione Attributi Tratte:**
   - **Lunghezza:** Quantizzata da $1$ a $6$ in funzione della distanza euclidea normalizzata tra le città collegate.
   - **Colori:** Assegnati in modo bilanciato ed equo tra gli 8 colori standard (`PURPLE`, `WHITE`, `BLUE`, `YELLOW`, `ORANGE`, `BLACK`, `RED`, `GREEN`) e tratte neutre/grigie (`None`).
   - **Doppie Tratte:** Generate tra le coppie di città a più alta densità se specificato dalla configurazione.
4. **Generazione Biglietti di Destinazione:**
   - Calcola i cammini minimi (tramite BFS / Dijkstra pesato) tra tutte le coppie di vertici.
   - Seleziona $T$ coppie con distanza di grafo $d(u, v) \ge 2$.
   - Calcola il valore in punti proporzionale alla lunghezza del percorso minimo:
     $$\text{points}(u, v) = \max(2, \min(22, \lfloor d_{\text{path}}(u, v) \times 1.2 \rfloor))$$

```python
class ProceduralMapConfig:
    num_cities: int = 8
    num_routes: int = 14
    num_tickets: int = 10
    allow_double_routes: bool = False
    min_city_distance: float = 0.15

class ProceduralMapGenerator:
    def __init__(self, config: ProceduralMapConfig | None = None) -> None: ...
    def generate(self, seed: int) -> tuple[Board, list[DestinationTicket]]: ...
```

### 3.2 Mappa Ufficiale Europa (`src/game/maps.py`)

Funzione `load_europe_board() -> tuple[Board, list[DestinationTicket]]`:
- **47 Città Europee:** Amsterdam, Athina, Barcelona, Berlin, Brest, Brindisi, Bruxelles, Bucuresti, Budapest, Cadiz, Constantinopla, Danzig, Dieppe, Edinburgh, Erzurum, Essen, Frankfurt, Kobenhavn, Kyiv, Lisboa, London, Madrid, Marseille, Moskva, Munchen, Palermo, Paris, Petrograd, Riga, Roma, Rostov, Sarajevo, Sevastopol, Smolensk, Smyrna, Sofia, Stockholm, Venezia, Wien, Wilno, Zagreb, Zurich, ecc. con coordinate spaziali $(x, y)$ normalizzate.
- **Rete Ufficiale delle Tratte:** Oltre 100 tratte fedeli al regolamento con relative colorazioni e doppie tratte.
- **Deck dei Biglietti Europei:** 46 biglietti ufficiali, inclusi i Biglietti Lunghi (Long Tickets come Brest-Petrograd 20, Cadiz-Stockholm 21, Edinburgh-Athina 21, Kobenhavn-Erzurum 21) e i Biglietti Standard.

### 3.3 Partizionamento del Dataset (`src/game/procedural.py`)

```python
@dataclass
class MapSplit:
    train_maps: list[tuple[Board, list[DestinationTicket]]]
    val_maps: list[tuple[Board, list[DestinationTicket]]]
    test_maps: list[tuple[Board, list[DestinationTicket]]]

class ProceduralMapDataset:
    def __init__(self, generator: ProceduralMapGenerator) -> None: ...
    def create_split(self, train_seeds: list[int], val_seeds: list[int], test_seeds: list[int]) -> MapSplit: ...
```

### 3.4 Ambiente di Addestramento Multi-Mappa (`src/environment/multi_map_env.py`)

`MultiMapTicketToRideEnv` permette di addestrare policy RL su insiemi di mappe:
- Estende o incapsula `TicketToRideEnv`.
- A ogni invocazione di `reset(seed=...)`: seleziona una mappa dallo split di training (campionamento casuale o sequenziale) e reinizializza il motore di gioco e i relativi encoder/masker.
- La rappresentazione canonica assicura che le dimensioni dello spazio delle osservazioni e dello spazio delle azioni rimangano stabili e compatibili con i tensori PyTorch di `MaskedActorCritic` e `RecurrentMaskedActorCritic`.

### 3.5 Framework di Valutazione e Benchmark (`src/evaluation/generalization.py`)

#### `GeneralizationEvaluator`
- Esegue la valutazione di uno o più agenti su insiemi arbitrari di mappe (`MapSplit` o coppie di mappe come USA vs Europa).
- Esegue $N$ partite per mappa alternando il primo giocatore per eliminare ogni bias di posizione.
- Calcola metriche aggregate per ogni split: punteggio medio, scarto di punteggio, tasso di vittoria, completamento biglietti e turni medi.

#### `GeneralizationBenchmarkRunner`
- Orchestra lo studio sperimentale completo della Fase 10:
  1. Genera lo split procedurale standard Train / Val / Test.
  2. Valuta le euristiche zero-shot (`StrategicAgent`, `GreedyAgent`, `RandomAgent`).
  3. Addestra e confronta l'agente RL a singola mappa (`PPO_SingleMap`) vs l'agente RL multi-mappa (`PPO_MultiMap`).
  4. Valuta le prestazioni su mappe di test inedite e il trasferimento cross-mappa (USA $\leftrightarrow$ Europa).
  5. Calcola Generalization Gap, Retention Rate ed Elo rating generalizzato.
  6. Genera e salva automaticamente i report `experiments/results/phase10_report.json` e `experiments/results/phase10_report.md`.

---

## 4. Piano di Test TDD e Criteri di Accettazione

1. **Test Unitari Mappe Procedurali (`tests/game/test_procedural_maps.py`):**
   - Determinismo: a parità di seed, la funzione restituisce esattamente la stessa mappa, le stesse tratte e gli stessi biglietti.
   - Connettività: visita BFS/DFS tocca il 100% delle città in tutte le mappe generate.
   - Validità tratte e biglietti: lunghezze $1 \le L \le 6$, colori validi, biglietti con città esistenti e punti strettamente positivi.
2. **Test Unitari Mappa Europa (`tests/game/test_europe_board.py`):**
   - Caricamento corretto di città, tratte e biglietti.
   - Giocabilità completa e assenza di riferimenti a città inesistenti.
3. **Test Ambiente Multi-Mappa (`tests/environment/test_multi_map_env.py`):**
   - Reset fluido attraverso mappe diverse.
   - Maschere d'azione valide e osservazioni bounded in $[0, 1]$.
4. **Test Valutatore di Generalizzazione (`tests/evaluation/test_generalization.py`):**
   - Calcolo esatto di Generalization Gap, Retention Rate e metriche statistiche.
5. **Suite di Accettazione Fase 10 (`tests/evaluation/test_phase10_acceptance.py`):**
   - Esecuzione end-to-end dello studio di generalizzazione e validazione di tutti i deliverable richiesti in `DESIGN.md`.
