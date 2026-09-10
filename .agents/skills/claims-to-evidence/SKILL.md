---
name: claims-to-evidence
description: Use when verifying empirical claims, statistical assertions, or benchmark tables in scientific papers against raw experimental logs, JSON datasets, and code artifacts.
---

# Claims-to-Evidence Auditor

Audits every assertion made in a research paper or monograph to ensure it is backed by concrete empirical data, verified JSON artifacts, and reproducible code.

## When to Use

- When verifying that numbers quoted in the Abstract, Introduction, and Conclusion match the data files in `results/`.
- When auditing win rates, Elo rankings, match counts, and confidence intervals.
- When cross-checking hyperparameters listed in tables against configuration scripts.
- When confirming that ablation charts reflect actual experimental runs rather than theoretical sketches.

---

## The Audit Protocol

For each claim in the text:

1. **Extract Claim**: Exact quote and location (section, paragraph).
2. **Locate Ground Truth**: Find corresponding raw artifact (e.g. `results/tournament_results_10x_5130games.json`).
3. **Recompute / Verify**: Compute the metric from raw data or verify exact string match.
4. **Determine Status**: `[VERIFIED]`, `[DISCREPANCY]`, or `[UNSUPPORTED]`.

```
Paper Text: "Dijkstra achieves an Elo of 1329.6 and an 80.6% win rate across 5,130 games."
   |
   v
Lookup `results/tournament_results_10x_5130games.json`
   |
   +--> Total Matches: 5,130 (171 matchups * 30 games) -> MATCH
   +--> Dijkstra Wins: 435 / 540 = 80.555% -> Rounds to 80.6% -> MATCH
   +--> Dijkstra Elo: 1329.61 -> Rounds to 1329.6 -> MATCH
   |
   v
Result: [VERIFIED]
```

---

## Checklist

- [ ] **Match Counts:** Does the total game count equal $\frac{N(N-1)}{2} \times \text{matches\_per\_pair}$? (e.g. $19 \times 18 / 2 = 171 \times 30 = 5,130$).
- [ ] **Symmetric Matchups:** Did each agent play equal games as Player 1 and Player 2?
- [ ] **Elo Formulas:** Is the $K$-factor and update rule stated and consistent?
- [ ] **Score Means vs Penalties:** Do mean scores accurately reflect negative ticket penalties for failing agents (e.g., random agent negative mean score)?
- [ ] **Inference Latency:** Does the latency table report both mean decision time and throughput (moves/second) measured on consistent hardware?

---

## Output Audit Matrix Format

```markdown
| ID | Section | Quoted Claim | Artifact Source | Data Value | Status |
|:---|:---|:---|:---|:---|:---|
| C1 | Abstract | "Elo 1329.6, 80.6% win rate" | `results/tournament_results_*.json` | Elo 1329.6, 80.56% | VERIFIED |
| C2 | Sec 6.2 | "AlphaZero Elo 1275.3 (67.4% win rate)" | `paper/tables/table1_*.tex` | Elo 1275.3, 67.41% | VERIFIED |
| C3 | Sec 7.1 | "Shannon entropy decay: 3.84 to 0.42 bits" | `results/thesis/thesis_study_results.json` | 3.84 -> 0.42 | VERIFIED |
```

---

## Common Rationalizations to Reject

| Rationalization | Reality |
| :--- | :--- |
| *"The author probably rounded up or down, no need to check raw JSON."* | Unchecked rounding errors propagate into false rank orderings. |
| *"The chart looks plausible so the underlying numbers must match."* | Charts are often static images. Always verify against the underlying numerical source. |
