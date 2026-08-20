# Spec Tecnica e Didattica - Fase 7: Reward Research & Behavioral Benchmarking

## 1. Visione, Filosofia e Obiettivi Didattici della Fase 7

In Reinforcement Learning, la funzione di ricompensa definisce implicitamente l'obiettivo ottimale dell'agente. Nella maggior parte dei giochi complessi a lungo orizzonte temporale come Ticket to Ride, la ricompensa naturale (punteggio finale e vittoria a fine partita) è estremamente **rada (sparse)** e presenta un severo problema di **credit assignment**.

Tuttavia, l'aggiunta di ricompense intermedie (**reward shaping**) altera profondamente lo spazio dei gradienti e la policy risultante. Se non calibrato con precisione, il reward shaping induce comportamenti miopi o degeneri (es. rivendicare tratte corte per raccogliere reward immediato ignorando la connettività dei Destination Ticket a lungo termine).

```text
                        ┌────────────────────────────────────────────────────────┐
                        │      FASE 7: REWARD RESEARCH & BEHAVIORAL ANALYSIS     │
                        └────────────────────────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┴────────────────────────────────┐
                    ▼                                                                 ▼
    ┌───────────────────────────────┐                                 ┌───────────────────────────────┐
    │     MODULAR REWARD ENGINES    │                                 │     BEHAVIORAL PROFILING      │
    │  • RewardV1: Sparse / Outcome │                                 │  • Ticket Completion Rate     │
    │  • RewardV2: Dense Routes     │                                 │  • Route Efficiency & Length  │
    │  • RewardV3: Ticket Milestones│                                 │  • Game Flow & Card Turnover  │
    │  • RewardV4: Strategic Shaped │                                 │  • Score Differentials        │
    │  • Custom Configurable Engine │                                 │                               │
    └───────────────────────────────┘                                 └───────────────────────────────┘
                    │                                                                 │
                    └────────────────────────────────┬────────────────────────────────┘
                                                     │
                                                     ▼
                        ┌────────────────────────────────────────────────────────┐
                        │              REWARD RESEARCH STUDY RUNNER              │
                        │    Multi-Reward Training & Standardized Benchmarking   │
                        │    Report Scientifico Comparativo (JSON & Markdown)    │
                        └────────────────────────────────────────────────────────┘
```

### 1.1 Obiettivi Scientifici e Didattici
1. **Isolare l'impatto del Reward Shaping**: Dimostrare empiricamente come diverse formulazioni matematiche della ricompensa modificano lo stile di gioco (aggressività, pianificazione di rete, gestione del tempo).
2. **Trasparenza e Decomposizione dei Componenti**: Fornire una scomposizione chiara (`get_components`) di ogni frazione di reward accumulata per consentire l'analisi dettagliata nel Web Lab e nei log.
3. **Automazione della Ricerca Sperimentale**: Fornire un motore `RewardResearchRunner` in grado di allenare e valutare le 4 varianti di reward a parità di condizioni iniziali, producendo tabelle comparative formali.

---

## 2. Architettura dei Calcolatori di Reward e Formulazione Matematica

Tutti i calcolatori ereditano dall'interfaccia astratta `BaseRewardCalculator`.

```python
class BaseRewardCalculator(ABC):
    """Abstract base class for modular reward functions."""

    @abstractmethod
    def calculate(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> float:
        """Compute the scalar step reward for player_index."""

    @abstractmethod
    def get_components(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> dict[str, float]:
        """Compute the breakdown of reward components for telemetry and introspection."""
```

---

### 2.1 Le 4 Versioni Formali di Reward

#### Versione 1: `RewardV1_Sparse` (Pure Terminal Outcome)
* **Filosofia**: Nessun feedback intermedio durante il gioco. L'agente riceve una ricompensa solo al termine dell'episodio.
* **Formulazione**:
  $$r_t = 0.0 \quad \forall t < T$$
  $$r_T = \text{win\_bonus} \cdot \mathbf{1}_{\text{win}} - \text{loss\_penalty} \cdot \mathbf{1}_{\text{loss}} + w_{\Delta} \cdot (\text{score}_{\text{agent}} - \text{score}_{\text{opp}}) - \sum_{t \in \text{uncompleted}} \text{points}(t)$$
* **Ipotesi Sperimentale**: Rappresenta il problema RL più puro e non distorto, ma soffre di alta varianza e convergenza lenta su orizzonti lunghi.

#### Versione 2: `RewardV2_DenseRoutes` (Route-Centric Shaping)
* **Filosofia**: Incentivo forte e immediato a rivendicare tratte e accumulare punti percorso, unito a una piccola penalità per ogni turno trascorso.
* **Formulazione**:
  $$r_t = w_{\text{route}} \cdot \Delta \text{score}_{\text{route}} - \text{step\_penalty}$$
  Al termine dell'episodio: aggiunta di bonus/malus vittoria e penalità per ticket non completati.
* **Ipotesi Sperimentale**: L'agente imparerà a rivendicare percorsi con avidità (alto numero di tratte e punteggio grezzo), ma potrebbe trascurare la pianificazione dei Destination Ticket a lungo raggio.

#### Versione 3: `RewardV3_TicketMilestones` (Ticket-Focused Shaping)
* **Filosofia**: Feedback orientato alla connessione topologica: premia in tempo reale il completamento di ciascun Destination Ticket e penalizza severamente il fallimento dei ticket a fine partita.
* **Formulazione**:
  $$r_t = w_{\text{route}} \cdot \Delta \text{score}_{\text{route}} + w_{\text{ticket}} \cdot \sum_{t \in \text{newly\_completed}} \text{points}(t) - \text{step\_penalty}$$
  $$r_T = \text{win\_bonus} \cdot \mathbf{1}_{\text{win}} - w_{\text{fail}} \cdot \sum_{t \in \text{uncompleted}} \text{points}(t)$$
* **Ipotesi Sperimentale**: L'agente svilupperà un'elevata efficienza nel completamento dei ticket (`ticket_completion_rate`), selezionando solo le tratte strettamente necessarie a unire le città bersaglio.

#### Versione 4: `RewardV4_StrategicShaped` (Balanced & Strategic)
* **Filosofia**: Sintesi equilibrata di tutti i segnali di gioco con calibrazione accurata delle scale (tratte, ticket, penalità turno per incentivare la rapidità, differenziale punteggio finale relativo).
* **Formulazione**: Punti tratta intermedi + bonus progressivo ticket + step penalty calibrata + bonus vittoria relativo con score differential normalizzato.

#### Versione Custom: `CustomRewardCalculator`
* Riceve un'istanza di `RewardWeights` o dizionario arbitrario di pesi, consentendo la sperimentazione di qualsiasi combinazione parametrica da file YAML.

---

### 2.2 `RewardFactory` e Registry

Il modulo `src/environment/reward.py` espone una `RewardFactory`:
```python
class RewardFactory:
    @staticmethod
    def create(
        version_or_name: int | str,
        board: Board | None = None,
        weights: RewardWeights | None = None,
    ) -> BaseRewardCalculator:
        ...
```
Alias supportati:
* `1`, `"1"`, `"v1"`, `"sparse"` $\to$ `RewardV1_Sparse`
* `2`, `"2"`, `"v2"`, `"dense_routes"`, `"dense"` $\to$ `RewardV2_DenseRoutes`
* `3`, `"3"`, `"v3"`, `"ticket_milestones"`, `"tickets"` $\to$ `RewardV3_TicketMilestones`
* `4`, `"4"`, `"v4"`, `"strategic"`, `"balanced"` $\to$ `RewardV4_StrategicShaped`
* `"custom"` $\to$ `CustomRewardCalculator`

---

## 3. Profiling Comportamentale & Metriche di Gioco

Per valutare qualitativamente e quantitativamente l'effetto del reward, definiamo una serie di **metriche comportamentali**:

| Categoria | Metrica | Descrizione |
|---|---|---|
| **Risultato** | `win_rate` | Percentuale di partite vinte contro avversari standard |
| | `avg_score` | Punteggio finale medio conseguito |
| | `score_differential` | $\mathbb{E}[\text{score}_{\text{agent}} - \text{score}_{\text{opp}}]$ |
| **Pianificazione Tratte** | `avg_routes_claimed` | Numero medio di tratte rivendicate per partita |
| | `avg_route_length` | Lunghezza media delle tratte occupate ($\sum \text{len} / N$) |
| | `route_efficiency` | Punti percorso ottenuti per vagone speso |
| **Topologia Ticket** | `ticket_completion_rate` | Rapporto tra ticket completati con successo e ticket tenuti |
| | `tickets_completed_avg` | Numero medio di ticket completati per partita |
| | `ticket_penalty_avg` | Punti medi persi a causa di ticket falliti |
| **Tempo e Risorse** | `avg_game_turns` | Durata media della partita in turni |
| | `cards_drawn_ratio` | Frazione di azioni dedicate a pescare carte |

Queste metriche vengono calcolate da `BehavioralEvaluator` (`src/evaluation/behavioral.py`) estendendo le capacità dell'`Evaluator` standard.

---

## 4. `RewardResearchRunner` & Studio Comparativo

Il modulo `src/evaluation/reward_research.py` orchestra lo studio comparativo automatico:

1. **Setup Sperimentale**:
   - Definisce le 4 configurazioni di reward (`v1`, `v2`, `v3`, `v4`).
   - Allena un agente PPO (o carica checkpoint) per ciascuna configurazione con lo stesso seed e budget di addestramento.
2. **Valutazione Comportamentale Incrociata**:
   - Esegue $N$ partite contro `RandomAgent`, `GreedyAgent` e `StrategicAgent`.
   - Aggrega le metriche di risultato e comportamentali in un dizionario strutturato.
3. **Generazione Report Scientifico**:
   - Esporta i risultati in formato JSON (`experiments/results/reward_research.json`).
   - Genera un report leggibile in Markdown (`experiments/results/reward_research.md`) con tabelle comparative e analisi interpretativa delle divergenze strategiche osservate.

---

## 5. File Modificati e Nuovi File

```text
src/
├── environment/
│   └── reward.py             # Nuova gerarchia modulare (V1-V4, Custom, RewardFactory, get_components)
├── evaluation/
│   ├── behavioral.py         # BehavioralEvaluator (estrazione metriche strategiche)
│   └── reward_research.py    # RewardResearchRunner (motore dello studio comparativo)
├── experiments/
│   └── config.py             # Aggiornamento EnvironmentConfig e RewardConfig
└── scripts/
    └── reward_research.py    # CLI per lanciare lo studio sperimentale sui reward

tests/
├── environment/
│   └── test_reward_versions.py # Test per V1, V2, V3, V4, Custom, get_components e Factory
├── evaluation/
│   ├── test_behavioral.py      # Test del profiler comportamentale
│   └── test_reward_research.py # Test del runner di ricerca e generazione report
└── rl/
    └── test_phase7_acceptance.py # Test di accettazione scientifica comparativa
```

---

## 6. Testing e Criteri di Accettazione

1. **Isolamento e Correttezza delle Componenti (`test_reward_versions.py`)**:
   - Verifica che `RewardV1_Sparse` restituisca 0.0 durante i turni intermedi e applichi il reward finale corretto a `is_game_over`.
   - Verifica che `RewardV2_DenseRoutes` assegni reward immediato al claiming di tratte e applichi lo step penalty.
   - Verifica che `RewardV3_TicketMilestones` riconosca il momento esatto in cui un ticket passa da incompleto a completato e assegni il bonus corrispondente.
   - Verifica che `RewardFactory.create` supporti tutti gli alias e la configurazione custom.
2. **Metriche Comportamentali (`test_behavioral.py`)**:
   - Verifica che `BehavioralEvaluator` calcoli accuratamente `ticket_completion_rate`, `avg_route_length`, `route_efficiency` e `cards_drawn_ratio`.
3. **Runner e Report (`test_reward_research.py`)**:
   - Verifica che `RewardResearchRunner` esegua lo studio sui 4 reward e salvi correttamente sia il file JSON che il file Markdown formattato.
4. **Acceptance Test di Fase 7 (`test_phase7_acceptance.py`)**:
   - Dimostrazione sperimentale che la scelta del reward altera misurabilmente le metriche comportamentali (es. `RewardV3` produce un `ticket_completion_rate` superiore rispetto a `RewardV1`).
5. **Non-regressione globale**:
   - L'intera suite di test del progetto (Game Engine, Baselines, Gymnasium Env, PPO, DQN, Web Lab) continua a superare i test.
