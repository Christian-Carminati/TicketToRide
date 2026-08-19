# Spec Tecnica e Didattica - Fase 5: Web Lab (Visualizzazione, Introspezione & Replay)

## 1. Visione, Filosofia e Obiettivi della Fase 5

La **Fase 5: Web Lab** costituisce il ponte tra i modelli matematici di Reinforcement Learning (sviluppati e validati nelle Fasi 1-4) e l'interfaccia interattiva di introspezione ed esperimento.

Come espresso nei principi cardine del [DESIGN.md](file:///home/christian/Projects/Python/TicketToRide/DESIGN.md):

> *"La componente grafica e web non è un semplice frontend del gioco. È parte integrante del progetto e deve permettere di osservare ciò che l'agente sta imparando."*

```text
                            ┌────────────────────────────────────────────────────────┐
                            │                    WEB LAB FRONTEND                    │
                            │  React 18 + TypeScript + Vite + Lucide Icons           │
                            └───────────────────────────┬────────────────────────────┘
                                                        │
                         ┌──────────────────────────────┴──────────────────────────────┐
                         │   REST API + Real-Time WebSocket Telemetry Streaming Hub    │
                         │   FastAPI (Asyncio Non-Blocking Workers & Broadcast)       │
                         └──────────────────────────────┬──────────────────────────────┘
                                                        │
        ┌───────────────────────────────┬───────────────┴───────────────┬───────────────────────────────┐
        ▼                               ▼                               ▼                               ▼
 ┌──────────────┐               ┌──────────────┐                ┌──────────────┐                ┌──────────────┐
 │  GAME CORE   │               │  RL ENGINE   │                │ EXPERIMENTS  │                │    REPLAY    │
 │ - Board SVG  │               │ - Brain View │                │ - Registry   │                │ - Scrubber   │
 │ - Human/Bot  │               │ - Live Plots │                │ - Checkpoint │                │ - Timeline   │
 └──────────────┘               └──────────────┘                └──────────────┘                └──────────────┘
```

---

### 1.1 Obiettivi Chiave della Fase 5
1. **Visualizzazione Interattiva del Tabellone (`BoardSVG`)**:
   Rendering vettoriale reattivo in SVG per le mappe USA e Mini con evidenziazione tratte singole e doppie, colori proprietari, indicatori città, hover tooltip e gestione click per mosse umane.
2. **Dashboard di Introspezione Neurale (`BrainView` & `AgentView`)**:
   Scomposizione visiva dell'osservazione $\mathbf{s}_t$, visualizzazione a blocchi dei layer neurali (Input $\to$ Hidden $\to$ Output Heads) con ampiezza delle attivazioni, confronto dei logits prima e dopo l'applicazione dell'Action Masking, stima del valore $V(s)$ del Critic e distribuzione di probabilità $\pi_\theta(a|s)$.
3. **Monitoraggio Live dell'Addestramento (`TrainingView`)**:
   Streaming asincrono via WebSocket `/ws/telemetry` di eventi e metriche in tempo reale (*Mean Reward*, *Policy Loss*, *Value Loss*, *Entropy*, *Approximate KL*, *Win Rate*) con grafici SVG leggeri e comandi di controllo (*Start*, *Pause*, *Stop*).
4. **Player di Replay Frame-by-Frame (`ReplayView`)**:
   Navigazione temporale completa di partite registrate (`experiments/replays/*.jsonl`), con controlli di trasporto standard (`|< << < ▶ > >> >|`), cursore temporale (scrubber), slider di velocità e ispezione dettagliata delle decisioni prese da ciascun agente.
5. **Registro ed Esploratore Esperimenti (`ExperimentView`)**:
   Interrogazione visiva del registro esperimenti (`registry.jsonl`), comparazione di iperparametri, metriche finali e download/ispezione dei checkpoint `.pt`.
6. **Invariante Headless (Regola 34)**:
   Mantenimento del disaccoppiamento totale: l'intero motore di gioco e gli algoritmi RL funzionano e si addestrano al 100% da CLI in modalità headless (`scripts/train.py`, `scripts/evaluate.py`, `scripts/tournament.py`) senza dipendere dal server web o dal frontend.

---

## 2. Architettura del Backend (`src/api/`)

Il backend espone servizi REST ad alte prestazioni e canali WebSocket asincroni basati su FastAPI e Pydantic v2.

```text
src/api/
├── __init__.py
├── main.py                  # Entrypoint applicativo FastAPI, CORS middleware, router registration
├── schemas.py               # Schemi Pydantic v2 per DTO, richieste, risposte e telemetria
├── websocket.py             # ConnectionManager avanzato per broadcast e gestione client WebSocket
├── game_service.py          # Gestione sessioni di gioco attive in memoria (Human vs Bot, Bot vs Bot)
├── trainer_service.py       # Worker asincrono per avvio/stop job di training e streaming metriche
├── replay_service.py        # Lettura, scrittura e validazione file replay JSON/JSONL
└── brain_service.py         # Introspezione modelli PyTorch: estrazione attivazioni layer e logits
```

---

### 2.1 Schemi DTO e Modelli Dati (`src/api/schemas.py`)

Gli schemi definiscono in modo tipizzato e rigoroso i contratti di interfaccia:

```python
from typing import Any, Literal
from pydantic import BaseModel, Field


# --- Game Schemas ---
class ActionDTO(BaseModel):
    action_type: str  # DRAW_TRAIN_CARDS, DRAW_VISIBLE_CARD, CLAIM_ROUTE, DRAW_DESTINATION_TICKETS, etc.
    card_index: int | None = None
    route_id: str | None = None
    card_color: str | None = None
    ticket_ids: list[str] | None = None


class PlayerStateDTO(BaseModel):
    player_id: str
    name: str
    score: int
    trains_remaining: int
    cards_in_hand: dict[str, int]
    tickets: list[dict[str, Any]]
    claimed_route_ids: list[str]
    color: str


class GameSessionCreateRequest(BaseModel):
    map_name: Literal["mini", "usa"] = "mini"
    player_types: list[str] = ["human", "random"]  # "human", "random", "greedy", "strategic", "dqn", "ppo"
    seed: int = 42
    model_checkpoint: str | None = None


class GameStateDTO(BaseModel):
    session_id: str
    turn_number: int
    current_player_index: int
    map_name: str
    players: list[PlayerStateDTO]
    visible_cards: list[str]
    deck_size: int
    discard_pile_size: int
    tickets_deck_size: int
    claimed_routes: dict[str, str]  # route_id -> player_id
    valid_actions: list[ActionDTO]
    action_mask: list[bool]
    is_game_over: bool
    winner_id: str | None = None
    last_action: ActionDTO | None = None
    last_reward: float | None = None


class GameStepRequest(BaseModel):
    session_id: str
    action: ActionDTO | None = None  # Se None, esegue il passo automatico del bot di turno


# --- Brain & Introspection Schemas ---
class LayerActivationDTO(BaseModel):
    layer_name: str
    shape: list[int]
    mean: float
    std: float
    min: float
    max: float
    values: list[float]  # Campione o aggregato per visualizzazione grafica


class BrainInspectionDTO(BaseModel):
    model_type: Literal["dqn", "ppo"]
    observation_vector: list[float]
    action_mask: list[bool]
    layer_activations: list[LayerActivationDTO]
    raw_logits_or_q: list[float]
    masked_logits_or_q: list[float]
    action_probabilities: list[float]
    estimated_value: float | None = None  # Per PPO Critic
    greedy_action_index: int
    action_labels: list[str]


# --- Training Telemetry Schemas ---
class TrainingStartRequest(BaseModel):
    config_name: str  # es. "ppo_mini.yaml", "dqn_mini.yaml"
    override_timesteps: int | None = None
    seed: int = 42


class TrainingStatusDTO(BaseModel):
    is_training: bool
    experiment_id: str | None = None
    algorithm: str | None = None
    current_step: int = 0
    total_timesteps: int = 0
    episodes: int = 0
    mean_reward: float = 0.0


class TelemetryEventDTO(BaseModel):
    type: Literal["training_started", "training_step", "checkpoint_saved", "training_finished", "error"]
    experiment_id: str
    step: int
    episode: int
    reward: float
    mean_reward: float
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
    approx_kl: float | None = None
    win_rate: float | None = None
    fps: float | None = None


# --- Replay Schemas ---
class ReplayFrameDTO(BaseModel):
    step_index: int
    turn_number: int
    player_index: int
    action: ActionDTO
    reward: float
    state_snapshot: dict[str, Any]
    observation: list[float] | None = None
    action_mask: list[bool] | None = None
    action_probabilities: list[float] | None = None


class ReplayDetailDTO(BaseModel):
    replay_id: str
    map_name: str
    seed: int
    date: str
    player_names: list[str]
    total_steps: int
    winner_index: int
    final_scores: list[int]
    frames: list[ReplayFrameDTO]
```

---

### 2.2 Servizi Backend e Logica Operativa

#### 1. `GameService` (`src/api/game_service.py`)
Mantiene un dizionario thread-safe di sessioni di gioco in memoria. Quando una sessione viene creata:
* Istanzia la mappa richiesta (`load_usa_board()` o `create_synthetic_mini_board()`).
* Inizializza l'istanza `Game` con il seed stabilito.
* Assegna i controllori di ciascun giocatore:
  * `HumanAgent`: attende le chiamate REST con azioni esplicite.
  * `RandomAgent`, `GreedyAgent`, `StrategicAgent`: agenti algoritmici.
  * `DQNAgent`, `PPOAgent`: caricati opzionalmente da checkpoint `.pt`.
* Permette di eseguire un singolo step (`step(action=None)` per i bot o `step(action=user_action)` per gli umani) e restituisce il `GameStateDTO` completo di maschera e azioni lecite.

#### 2. `TrainerService` (`src/api/trainer_service.py`)
Gestisce l'esecuzione di esperimenti di addestramento in background senza bloccare il server FastAPI:
* Avvia un thread o task asincrono dedicato che esegue `MaskedDQNTrainer` o `MaskedPPOTrainer`.
* Utilizza una coda asincrona `asyncio.Queue` su cui il trainer deposita eventi di telemetria ad intervalli regolari (es. ogni 100 timesteps o a fine episodio).
* Un task di broadcast legge dalla coda e distribuisce i messaggi ai client connessi su `/ws/telemetry`.
* Supporta comandi di terminazione controllata (`stop_training()`).

#### 3. `ReplayService` (`src/api/replay_service.py`)
Fornisce funzioni per:
* Elencare tutti i replay registrati nella cartella `experiments/replays/`.
* Caricare un replay completo in formato JSON/JSONL, con validazione strutturale dei frame.
* Registrare automaticamente partite complete giocate durante tornei o sessioni di visualizzazione.

#### 4. `BrainService` (`src/api/brain_service.py`)
Ispezione avanzata di reti neurali PyTorch:
* Registra hook PyTorch (`register_forward_hook`) o esegue passaggi analitici per estrarre le attivazioni intermedie di ogni layer lineare (`fc1`, `fc2`, `actor_fc`, `critic_fc`).
* Applica l'Action Masking vettoriale sui logits: $z_{\text{masked}} = z - 10^8 (1 - M)$.
* Calcola la distribuzione di probabilità Softmax $\pi_\theta(a|s) = \text{Softmax}(z_{\text{masked}})$.
* Ritorna statistiche e valori per il rendering del diagramma neurale.

---

## 3. Architettura Frontend (`frontend/src/`)

Il frontend React 18 / TypeScript è organizzato con una separazione netta tra componenti di presentazione, viste applicative e custom hooks per la gestione dello stato e delle comunicazioni.

```text
frontend/src/
├── App.tsx                  # Layout principale, navigazione a schede e stato globale
├── main.tsx                 # Bootstrapping React DOM
├── index.css                # Stili globali CSS con variabili di tema scuro/RL Lab
├── api/
│   ├── client.ts            # Client HTTP tipizzato (fetch con gestione errori)
│   └── types.ts             # Interfacce TypeScript mirror degli schemi Pydantic
├── hooks/
│   ├── useWebSocket.ts      # Hook di connessione e sottoscrizione eventi WebSocket
│   ├── useGameSession.ts    # Hook per la gestione della partita interattiva
│   ├── useTrainingStream.ts # Hook con buffer circolare per telemetria live
│   └── useReplayPlayer.ts   # Hook per il controllo della riproduzione frame-by-frame
├── components/
│   ├── board/
│   │   ├── BoardSVG.tsx     # Renderer vettoriale interattivo mappa USA e Mini
│   │   ├── CityNode.tsx     # Nodo città SVG con etichetta e tooltip
│   │   └── RouteEdge.tsx    # Tratta SVG singola/doppia con vagoni stilizzati
│   ├── cards/
│   │   ├── TrainCardHand.tsx# Carte del giocatore e conteggio locomotive Jolly
│   │   ├── VisibleDeck.tsx  # Mazzo 5 carte scoperte e mazzo coperto
│   │   └── TicketsList.tsx  # Biglietti destinazione e stato completamento
│   ├── brain/
│   │   ├── NeuralNetworkDiagram.tsx # Diagramma grafico del grafo della rete neurale
│   │   └── ActionProbabilitiesChart.tsx # Istogramma probabilità e maschere
│   ├── charts/
│   │   └── LiveTelemetryChart.tsx   # Grafico SVG in tempo reale per le curve RL
│   └── replay/
│       └── ReplayControls.tsx       # Barra di controllo temporale e velocità
└── views/
    ├── GameView.tsx         # Vista partita (Human vs AI / AI vs AI)
    ├── AgentView.tsx        # Vista agente (Osservazione, Maschera, Decisione)
    ├── TrainingView.tsx     # Vista addestramento (Controlli & Grafici Live)
    ├── BrainView.tsx        # Vista introspezione rete neurale
    ├── ReplayView.tsx       # Vista lettore di replay
    └── ExperimentView.tsx   # Vista registro esperimenti e confronto
```

---

### 3.1 Il Componente `BoardSVG` (Rendering Vettoriale Mappa)

Il componente `BoardSVG` garantisce massima nitidezza visiva a qualsiasi risoluzione grazie all'uso di SVG scalabile:
1. **Coordinate Normalizzate**:
   Le città sono definite con coordinate $(x, y) \in [0, 1]$.
   All'interno dell'`<svg viewBox="0 0 1000 650">`, le coordinate vengono proiettate:
   $$X_{\text{screen}} = 60 + x \cdot (1000 - 120), \quad Y_{\text{screen}} = 50 + (1 - y) \cdot (650 - 100)$$
2. **Geometria delle Tratte Singole e Doppie**:
   * Tratta singola tra $(X_1, Y_1)$ e $(X_2, Y_2)$: tracciata con linea spessa con colore della carta (o grigio se neutra).
   * Doppie tratte parallele: calcolo del vettore direzione unitario $\mathbf{u} = \frac{\mathbf{p}_2 - \mathbf{p}_1}{\|\mathbf{p}_2 - \mathbf{p}_1\|}$ e del vettore normale $\mathbf{n} = (-u_y, u_x)$. Le due tratte vengono spostate rispettivamente di $+8\mathbf{n}$ e $-8\mathbf{n}$ pixel, evitando qualsiasi sovrapposizione visiva.
   * Suddivisione in segmenti per ciascun vagone con bordo marcato.
3. **Stato di Occupazione**:
   * Se una tratta è libera: visualizza il colore della risorsa richiesta con opacità standard.
   * Se una tratta è occupata: assume il colore distintivo del giocatore (es. Blu = Player 0, Rosso = Player 1) con vagoni stilizzati evidenziati.
4. **Interazione**:
   * Hover su una città evidenzia tutte le tratte incidenti e mostra un tooltip con il nome.
   * Click su una tratta lecita (quando è il turno del giocatore umano) attiva la finestra di selezione per reclamarla.

---

### 3.2 Introspezione Brain & Visualizzazione Rete Neurale

Il componente `NeuralNetworkDiagram` visualizza visivamente l'architettura:
* **Colonna Input**: $D$ neuroni di osservazione (carte in mano, vagoni, stato tratte, carte visibili, biglietti).
* **Colonne Hidden Layers**: Livelli densi (es. 256 neuroni ciascuno), raggruppati in barre verticali con gradiente di colore proporzionale al valore medio di attivazione post-ReLU.
* **Colonne Output**:
  * **Head di Policy (Logits)**: $|\mathcal{A}|$ barre orizzontali. Le azioni mascherate ($M(s)_a = 0$) sono colorate in grigio scuro opaco con icona di blocco e probabilità $0.0\%$; le azioni lecite ($M(s)_a = 1$) sono colorate con tonalità accesa proporzionale a $\pi_\theta(a|s)$.
  * **Head di Valore (Critic)**: Box scalare che mostra la stima $V_\phi(s)$ attesa.

---

### 3.3 Streaming Telemetria Live & Grafici in Tempo Reale

Il componente `LiveTelemetryChart` implementa grafici SVG ultra-leggeri:
* Riceve gli eventi `training_step` dal WebSocket hook.
* Mantiene una finestra scorrevole degli ultimi $N=200$ punti per garantire rendering a 60 fps senza accumulo di memoria nel browser.
* Traccia simultaneamente su assi dedicati:
  1. **Mean Episode Reward** (curva verde smeraldo).
  2. **Policy Loss & Value Loss** (curve arancione e blu).
  3. **Entropy & Approximate KL Divergence** (curve viola e ciano).
  4. **Win Rate vs Baselines** (curva dorata percentuale).

---

### 3.4 Replay Player & Navigazione Temporale

Il componente `ReplayControls` fornisce l'esperienza di un player video per le partite di RL:
* **Controlli**:
  * `|<` Primo turno / Inizio partita.
  * `<<` Indietro di 5 turni.
  * `<` Passo precedente.
  * `▶` / `⏸` Play / Pausa automatica con timer configurabile (100ms - 2000ms).
  * `>` Passo successivo.
  * `>>` Avanti di 5 turni.
  * `>|` Fine partita / Risultati finali.
* **Scrubber**: Barra di avanzamento interattiva con indicatore del turno corrente rispetto al totale.

---

## 4. Strategia di Test e Criteri di Accettazione

Per garantire la massima affidabilità prima del rilascio, la Fase 5 comprende una suite completa di test:

### 4.1 Test Backend (`tests/api/`)
* **`test_api_endpoints.py`**: Verifica che tutti gli endpoint REST (`/health`, `/api/game/new`, `/api/game/step`, `/api/game/state`, `/api/replays/list`, `/api/experiments/list`, `/api/brain/inspect`) rispondano con codici di stato 200 e payload validati secondo gli schemi Pydantic.
* **`test_websocket_telemetry.py`**: Test di connessione WebSocket, sottoscrizione e ricezione eventi di broadcast con `TestClient`.
* **`test_replay_service.py`**: Test di serializzazione, salvataggio e caricamento di una partita completa in formato JSONL.
* **`test_brain_service.py`**: Test di estrazione delle attivazioni dei layer nascosti e dei logits mascherati da modelli `MaskedQNetwork` e `MaskedActorCritic`.
* **`test_phase5_acceptance.py`**: Test di accettazione end-to-end che valida l'intero ciclo: creazione sessione, esecuzione turni, introspezione neurale, emissione eventi telemetria e replay.

### 4.2 Verifica Frontend
* Compilazione TypeScript rigorosa (`tsc --noEmit`) senza errori di tipo.
* Build di produzione con Vite (`npm run build`) completata con successo nella cartella `frontend/dist`.

---

## 5. Definition of Done della Fase 5

La Fase 5 è considerata completata con successo quando:
1. ✅ Il backend FastAPI fornisce gli endpoint REST e i canali WebSocket necessari per partite, telemetria, replay e introspezione neurale.
2. ✅ Il frontend React visualizza il tabellone vettoriale `BoardSVG` in modo reattivo e interattivo per le mappe Mini e USA.
3. ✅ Tutte le 6 viste del Web Lab (`Game`, `Agent`, `Training`, `Brain`, `Replay`, `Experiments`) sono implementate, funzionanti e collegate al backend.
4. ✅ Un utente può guardare l'agente addestrarsi in tempo reale con grafici aggiornati via WebSocket e riprodurre frame-by-frame le partite registrate.
5. ✅ L'esecuzione headless (CLI) da terminale rimane totalmente indipendente dal server web.
6. ✅ Tutti i test unitari, di integrazione e di tipo (Python e TypeScript) superano la validazione al 100%.
