# Ticket to Ride AI Toolkit

Toolkit per studiare e automatizzare partite a Ticket to Ride (mappa USA) con
solver classici e agenti di Reinforcement Learning.

## Contenuti principali

- **Simulazione del gioco**: modellazione del tabellone, delle carte e delle
  regole base in `src/game.py` e `src/map/`.
- **Solver tradizionali**: implementazioni greedy, local search, simulated
  annealing, tabu e algoritmi genetici in `src/best_solution/`.
- **Solver ottimo**: branch & bound con pruning per soluzioni esatte.
- **Modulo RL** (`src/rl/`):
  - Funzioni di scoring condivise (`scoring.py`).
  - Ambienti Gymnasium (single-agent) e PettingZoo (self-play) (`envs.py`).
  - Politiche avversarie scriptate (`opponents.py`).
  - Script di training per Stable Baselines3 (`train_sb3.py`).
  - Script di training self-play con RLlib (`train_rllib.py`).

## Requisiti e installazione

1. Creare un ambiente virtuale (opzionale ma consigliato):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # su Windows: .venv\Scripts\activate
   pip install --upgrade pip
   ```

2. Installare le dipendenze principali:

   ```bash
   pip install networkx pandas matplotlib gymnasium pettingzoo \
     stable-baselines3[extra] ray[rllib] tensorboard
   ```

   > Se preferisci un file dei requisiti, crea `requirements.txt` con i pacchetti
   > sopra e usa `pip install -r requirements.txt`.

## Dataset della mappa

Nella cartella `src/map/` trovi:

- `city_locations.json`: coordinate approssimate delle città sulla mappa USA.
- `routes.csv`: rotte disponibili (città, colore, lunghezza).
- `tickets.csv`: biglietti destinazione e punteggi.
- `USA_map.jpg`: immagine di riferimento.

La classe `TicketToRideMap` in `src/map.py` carica questi file in un
`networkx.MultiGraph` pronto per gli algoritmi.

## Solver classici

Per eseguire uno dei solver euristici:

```bash
python -m best_solution.main --solver greedy
python -m best_solution.main --solver tabu
python -m best_solution.main --solver simulated_annealing
```

Il modulo stampa punteggio, rotte selezionate e statistiche di calcolo.

## Training con Stable Baselines3

Lo script `rl/train_sb3.py` addestra un agente contro un avversario scriptato
utilizzando PPO (default) o DQN.

```bash
python -m rl.train_sb3 \
  --algorithm ppo \
  --timesteps 200000 \
  --opponent greedy \
  --log-dir runs/sb3
```

Parametri utili:

- `--opponent`: politica avversaria (`random`, `greedy`, `heuristic`, `tabu`, `genetic`).
- `--reward-*`: pesi per efficienza, biglietti, bonus connettività e punteggio finale.
- `--checkpoint-freq`: frequenza (in step) per salvataggio modelli.

TensorBoard può essere lanciato su `runs/sb3` per monitorare reward e metrica di
valutazione.

## Self-play con RLlib

Per il training multi-agente su PettingZoo:

```bash
python -m rl.train_rllib \
  --iterations 300 \
  --num-workers 4 \
  --log-dir runs/rllib
```

Lo script imposta PPO con una singola policy condivisa dai due giocatori.
Checkpoint ed evaluation vengono salvati in `runs/rllib/`.

## Struttura del progetto

```text
src/
  best_solution/    # Solver euristici e ottimo
  map/              # Dataset e utilities legate alla mappa
  rl/               # Ambienti, avversari e script di training RL
  game.py           # Modello generale del gioco Ticket to Ride
```

## Prossimi passi suggeriti

- Ampliare il modello del gioco con carte vagone e pesca biglietti dettagliata.
- Aggiungere test automatici per ambiente e avversari.
- Esplorare policy distillation o population-based training per agenti RL.
