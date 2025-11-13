# Guida al fine-tuning degli agenti RL

Questa guida raccoglie esperienze pratiche per migliorare l'agente PPO/DQN che gioca a Ticket to Ride utilizzando gli ambienti in `src/rl/`.

## 1. Preparazione dell'ambiente

- Lavora dalla cartella `src` per avere i path dei file mappa corretti: `cd src && PYTHONPATH=.`
- Assicurati di avere TensorBoard attivo (`tensorboard --logdir runs/full_game/tensorboard`) per monitorare reward medi, valore stimato e entropia.
- Usa `rl.eval_full_game` dopo ogni sessione di training per validare il modello: `PYTHONPATH=. python -m rl.eval_full_game --model-path <checkpoint>`.

## 2. Configurare l'ambiente di training

### Reward shaping (`RewardConfig`)

| Parametro | Effetto | Suggerimento |
|-----------|---------|--------------|
| `invalid_action_penalty` | Penalità quando l'agente forza un'azione impossibile. | 5–15. Valori alti stabilizzano la policy ma rallentano l'esplorazione. |
| `efficiency_weight` | Bonus in base a punti/vagoni per ogni rotta presa. | 0.3–0.7. Aumentalo per spingere il giocatore verso rotte corte ma redditizie. |
| `connectivity_bonus` | Favorisce catene lunghe di rotte connesse. | 3–8. Utile per completare biglietti e ottenere il bonus longest path. |
| `ticket_weight` | Reward proporzionale ai ticket completati. | 0.02–0.1. Valori alti incentivano la pianificazione di lungo periodo. |
| `final_score_scale` | Punteggio finale (differenza vs avversario). | 0.05–0.2. Incrementalo quando l'agente è già stabile. |
| `card_draw_bonus` | Bonus quando nuove carte aumentano le rotte reclamabili. | 0.05–0.15. Troppo alto porta a pescare in loop. |
| `card_draw_penalty` | Penalità per pescate inutili. | 0.03–0.1. Aumenta se l'agente accumula troppe carte senza usarle. |

Puoi passare un JSON personalizzato allo script di training oppure modificare direttamente `RewardConfig` in `train_full_game.py`.

### Selezione dell'avversario

- `random`: ideale per il bootstrap iniziale.
- `greedy`: buona via di mezzo, concentra le rotte ad alto valore.
- `heuristic`: prende decisioni pianificate (uso consigliato per valutare miglioramenti reali).
- Allenamento curriculum: parti da `random`, passa a `greedy` dopo ~50k step, poi `heuristic`.

## 3. Hyper-parameter consigliati (PPO SB3)

- `learning_rate`: 1e-4–3e-4. Riduci se il reward oscilla molto.
- `n_steps`: 2048 è un buon compromesso; aumenta a 4096 per training più stabili (serve più RAM).
- `batch_size`: 64–128. Più grande = gradienti più stabili.
- `gae_lambda`: 0.9–0.95. Valori bassi riducono la varianza ma limitano il vantaggio di lungo termine.
- `clip_range`: 0.1–0.2. Riduci se vedi exploit aggressivi.
- `ent_coef`: 0.003–0.01. Più alto = maggiore esplorazione.
- `vf_coef`: 0.5–1.0. Alza se il valore stimato diverge dal reward.

Per DQN:

- `learning_rate`: 5e-5–1e-4.
- `buffer_size`: ≥ 200k è consigliato (gioco lungo).
- `target_update_interval`: 500–2000.
- `exploration_fraction`: 0.2 per esplorare più a lungo.

## 4. Pianificare gli esperimenti

1. **Baseline breve**: 50k step contro `random`, valida che l'agente completi almeno un biglietto.
2. **Run principale**: 200k–500k step contro `greedy`, salva checkpoint ogni 50k.
3. **Valutazione**: 100 partite con `rl.eval_full_game` su 3 modelli (miglior, ultimo, media).
4. **Analisi TensorBoard**:
   - *Episode Reward*: deve crescere e stabilizzarsi.
  - *Value Loss*: decrescente ma non zero.
  - *Entropy*: usare come indicatore di esplorazione (non scendere <0.1 troppo presto).

Annota tutto in un file `runs/<data>/notes.md` per replicare l'esperimento.

## 5. Debug di comportamenti strani

- **Pesca infinita di carte**: aumenta `card_draw_penalty`, riduci `card_draw_bonus`, o accorcia le partite con `--max-episode-steps`.
- **Non reclama rotte anche con carte disponibili**: abbassa `invalid_action_penalty` (permette tentativi) e controlla che l'agente abbia abbastanza vagoni (`render()` dell'ambiente aiuta).
- **Blocchi per azioni invalide**: stampa `env.get_valid_actions(for_agent=True)` durante il debug per verificare la maschera.
- **Policy troppo deterministica**: alza `ent_coef` e riavvia l'allenamento con checkpoint precedente.

## 6. Strumenti di supporto

- `rl/test_environment.py`: unit test di base sugli ambienti RL.
- `visualization/game_viewer.py`: replay grafico di partite salvate.
- `docs/RL_FULL_GAME.md`: dettagli su osservazioni e spazio delle azioni.

---

Hai trovato un settaggio efficace? Condividilo aprendo una issue o aggiungendo una nota in `docs/RL_CARDS_ANALYSIS.md`. Con un po' di iterazione l'agente smette di pescare a vuoto e inizia a pianificare rotte competitive!

