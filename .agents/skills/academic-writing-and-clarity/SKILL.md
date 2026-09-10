---
name: academic-writing-and-clarity
description: Use when editing or refining scientific papers, research monographs, or technical manuscripts for precision, concise academic register, mathematical consistency, and eliminating verbosity.
---

# Academic Writing & Scientific Clarity

Refines scientific prose, abstracts, methodology sections, and mathematical notation to meet top-tier publication standards (IEEE, ACM, NeurIPS).

## When to Use

- When transforming informal project notes into publication-grade academic prose.
- When polishing bilingual manuscripts (English IEEE style and Italian academic/monograph style).
- When eliminating wordiness, passive voice overuse, and vague buzzwords.
- When ensuring mathematical typography and notation consistency across all sections.

---

## Core Principles

### 1. Ruthless Economy of Language
Scientific writing is measured by density of insight, not word count. Prune filler phrases immediately:
- *Instead of:* "Due to the fact that the agent is lacking information..." $\rightarrow$ *Use:* "Because the agent lacks information..."
- *Instead of:* "It is important to notice that Dijkstra's algorithm achieves..." $\rightarrow$ *Use:* "Dijkstra's algorithm achieves..."
- *Instead of:* "In order to demonstrate the efficacy..." $\rightarrow$ *Use:* "To demonstrate..."

### 2. Active, Precise Verbs
Replace weak verbs paired with heavy nominalizations:
- *Instead of:* "conducted an evaluation of" $\rightarrow$ "evaluated"
- *Instead of:* "performed the implementation of" $\rightarrow$ "implemented"
- *Instead of:* "achieved the suppression of" $\rightarrow$ "suppressed"

### 3. Mathematical Notation Consistency
- Sets: Calligraphic or uppercase ($\mathcal{S}, \mathcal{A}, \mathcal{V}, \mathcal{E}$).
- Vectors: Bold lowercase ($\mathbf{h}_i, \mathbf{c}_i$).
- Tensors/Matrices: Bold uppercase ($\mathbf{M}, \mathbf{W}$).
- Functions and Operators: Standard roman math operator ($\arg\max$, $\mathbb{E}_{s \sim \mathcal{D}}$, $\log_2$, $\min$).
- Never use raw text in math mode: use `\text{dist}` instead of `$dist$` (which renders as $d \cdot i \cdot s \cdot t$).

---

## Language-Specific Guidelines

### English (IEEE / ACM / NeurIPS Style)
- Maintain present tense for general truths and established algorithm behaviors: *"AlphaZero leverages PUCT..."*
- Use simple past for specific experimental actions taken: *"We evaluated 19 checkpoints across 5,130 games..."*
- Use Oxford commas consistently.
- Ensure all acronyms are expanded upon first mention (e.g. Partially Observable Markov Decision Process (POMDP)).

### Italian (Monografia / Report Tecnico)
- Maintain an elevated, rigorous technical register: *"Nel presente studio...", "Si dimostra...", "L'indagine empirica evidenzia..."*
- Avoid awkward literal translations of English terms when standard Italian equivalents exist, but preserve established RL terms in italics (e.g. *Action Masking*, *Self-Play*, *Information-Set MCTS*).
- Never use bureaucratic student phrasing (e.g. avoid *"il candidato"*, *"lavoro di tesi"*; prefer *"il presente studio"*, *"l'analisi condotta"*).

---

## Quick Reference: Fluff vs. Precision

| Fluff / Unclear | Scientific & Direct |
| :--- | :--- |
| "A massive amount of games were played" | "A round-robin tournament of 5,130 matches" |
| "The bot plays amazingly well" | "Strategic Dijkstra achieves an Elo of 1329.6" |
| "Basically it reduces the problem" | "This formulation reduces the search space" |
| "Clearly, AlphaZero is the best neural model" | "AlphaZero outperforms all learning architectures by +19.2 Elo" |
