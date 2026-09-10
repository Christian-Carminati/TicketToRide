---
name: paper-peer-reviewer
description: Use when conducting a formal peer review or critical audit of an academic or technical research paper, evaluating theoretical soundness, empirical validation, ablation rigor, and overclaiming before submission or publication.
---

# Paper Peer Reviewer

Conducts a rigorous, adversarial, top-tier conference-grade (NeurIPS, ICML, ICLR, IEEE Transactions) peer review of research papers, manuscripts, and technical monographs.

## When to Use

- When auditing a draft paper before publication or public dissemination.
- When stress-testing mathematical problem formulations against implicit assumptions.
- When checking whether empirical benchmarks, ablations, and baselines are scientifically fair.
- When hunting for overclaiming, unjustified causal statements, or missing baselines.

Do NOT use for:
- Simple spelling/grammar proofreading (use `academic-writing-and-clarity`).
- Pure LaTeX typesetting or BibTeX checks (use `latex-typesetting-auditor`).

---

## The Area Chair Review Framework

Every review must evaluate the manuscript across six orthogonal axes:

```
+-----------------------------------------------------------------------------+
|                      PEER REVIEW EVALUATION MATRIX                          |
+-----------------------------------------------------------------------------+
| 1. Soundness & Formalism     -> Mathematical proofs, POMDP tuple consistency |
| 2. Novelty & Positioning     -> Genuine delta vs. prior literature          |
| 3. Baseline Fairness         -> No strawman baselines, identical compute    |
| 4. Ablation Quality          -> Isolated factor changes, entropy decay      |
| 5. Claims vs. Evidence       -> Every claim mapped to Table/Figure          |
| 6. Reproducibility & Rigor   -> Seeds, hyperparameters, compute latency     |
+-----------------------------------------------------------------------------+
```

---

## Review Checklist & Workflow

### 1. Soundness & Mathematical Formulation
- [ ] **State & Action Representation:** Are the state $s \in \mathcal{S}$, observation $o \in \Omega$, and action spaces $\mathcal{A}$ rigorously defined?
- [ ] **Information Leakage Check:** Does any observation component inadvertently leak private hidden opponent state?
- [ ] **Action Masking Formalism:** Is logit masking mathematically grounded ($-\infty$ addition before softmax)?
- [ ] **Bayesian Update Derivation:** Is the likelihood and prior explicitly formulated without hand-waving?

### 2. Experimental Rigor & Baselines
- [ ] **Non-Trivial Baselines:** Are baselines representative of state-of-the-art heuristics and learning paradigms (e.g. Dijkstra, CleanRL PPO, AlphaZero PUCT)?
- [ ] **Sample Size & Match Count:** Is the tournament scale statistically meaningful (e.g. thousands of matches, symmetric alternating starting player)?
- [ ] **Statistical Significance:** Are mean scores accompanied by standard deviations, confidence intervals, or Elo ratings?

### 3. Ablation Studies
- [ ] Does the ablation isolate **one variable at a time**?
  - E.g., Recurrent memory (LSTM) vs. feedforward memory.
  - E.g., Bayesian belief determinization vs. uniform determinization (quantified via Shannon entropy decay).
- [ ] Is there a Pareto efficiency analysis comparing inference latency (ms/move) against task performance (Elo)?

### 4. Overclaiming Scan (Red Flags)
- [ ] "Demonstrates that X is universally superior" $\rightarrow$ Flag: Is it limited to specific map topologies or 2-player regimes?
- [ ] "Proves optimal play" $\rightarrow$ Flag: Dijkstra is optimal for single-pair shortest path, but only an approximation for Steiner trees.
- [ ] Check if all Research Questions ($RQ_1-RQ_4$) posed in the Introduction are answered explicitly with quantitative data.

---

## Output Review Template

When reviewing a paper, produce a structured markdown report following this format:

```markdown
# Critical Peer Review: [Paper Title]

## Meta-Review Summary & Verdict
- **Verdict:** [Strong Accept / Accept / Weak Accept / Borderline / Revise]
- **Confidence:** [5: Absolute / 4: High / 3: Medium]
- **Core Contribution:** (2-3 sentences summarizing the exact scientific contribution)

## Strengths
1. **[Strength 1 Title]:** Concrete evidence from text/data.
2. **[Strength 2 Title]:** Concrete evidence from text/data.

## Weaknesses & Critical Questions
1. **[Weakness 1 Title]:** Precise critique with line numbers/section references.
2. **[Potential Blindspot]:** Unaddressed edge cases or assumptions.

## Detailed Section-by-Section Audit
- **Section 1-2 (Intro & Formulation):** ...
- **Section 3-4 (Related Work & Paradigms):** ...
- **Section 5-6 (Experiments & Benchmark):** ...
- **Section 7-8 (Ablations & Discussion):** ...

## Actionable Recommendations for Final Version
- [ ] Fix 1...
- [ ] Fix 2...
```

---

## Common Rationalizations to Reject

| Rationalization | Reality |
| :--- | :--- |
| *"The paper looks complete so I will just praise it."* | Superficial praise does not prepare the work for hostile readers. Probe edge cases. |
| *"The numbers are impressive so the theory must be correct."* | High Elo can mask flawed POMDP leakage or overfit heuristics. Verify math independently. |
| *"Small discrepancies in dimensions don't matter."* | Dimensionality mismatches (e.g. 356D vs 464D) destroy reproducibility. Flag them. |
