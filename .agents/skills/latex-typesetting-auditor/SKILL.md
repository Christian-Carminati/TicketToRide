---
name: latex-typesetting-auditor
description: Use when auditing, validating, or debugging LaTeX manuscripts, table environments, math notation, BibTeX references, or compilation issues in scientific papers.
---

# LaTeX Typesetting & BibTeX Auditor

Performs automated syntax checks, micro-typography auditing, floating table/figure layout verification, and BibTeX citation integrity inspection.

## When to Use

- When preparing LaTeX papers (`.tex`) for compilation or review.
- When formatting mathematical equations, tables (`booktabs`), and cross-references.
- When auditing `references.bib` for missing fields, inconsistent author formats, or duplicate keys.
- When resolving LaTeX warnings like `Overfull \hbox`, broken `\ref`, or unresolved citations `[?]`.

---

## The LaTeX Micro-Typography Checklist

### 1. Cross-References and Non-Breaking Spaces
- [ ] **Always use a tilde `~` before `\cite{...}` and `\ref{...}`:**
  - ❌ `Algorithm 1 in [5]` $\rightarrow$ Can break line as "Algorithm 1 in" / "[5]".
  - ✅ `Algorithm~1 in~\cite{silver2018general}`
  - ✅ `Table~\ref{tab:tournament_elo}`
  - ✅ `Figure~\ref{fig:tournament_matrix}`
  - ✅ `Section~\ref{sec:intro}`

### 2. Math Mode Quality
- [ ] **Multi-letter variables vs. operators:**
  - ❌ `$dist(u, v)$` $\rightarrow$ renders as product $d \cdot i \cdot s \cdot t$.
  - ✅ `\text{dist}(u, v)` or `\operatorname{dist}(u, v)`.
- [ ] **Proper punctuation after display math:** Display math `\begin{equation} ... \end{equation}` is part of a sentence. It must end with a comma or period if grammatically required.
- [ ] **Percent signs:** In text, always escape percent signs `\%` to avoid accidental comment truncation.

### 3. Professional Tables (`booktabs`)
- [ ] Avoid ugly vertical rules `|`.
- [ ] Use `\toprule`, `\midrule`, and `\bottomrule` from the `booktabs` package.
- [ ] For two-column layouts (like IEEEtran), use `\begin{table*}[t]` for full-width spanning tables to prevent horizontal overflow.
- [ ] Center tables with `\centering` (not `\begin{center}` which introduces unwanted vertical margin).

### 4. BibTeX Integrity (`references.bib`)
- [ ] Check that every key cited via `\cite{key}` exists in `references.bib`.
- [ ] Capitalize proper nouns in titles using double braces:
  - ❌ `title = {Monte carlo tree search in ticket to ride}`
  - ✅ `title = {{Monte Carlo} Tree Search in {Ticket to Ride}}`
- [ ] Ensure all authors use `Lastname, Firstname and Lastname, Firstname` syntax.
- [ ] Verify standard publication fields: `author`, `title`, `booktitle` or `journal`, `year`, `pages`.

---

## Quick Fix Recipes

### Broken Reference Scan
Run a regex search for `\ref{` without a preceding tilde `~`:
```regex
(?<!~)\\ref\{
```
Replace with `~\ref{`.

### Missing Cite Tilde Scan
Run a regex search for `\cite{` without a preceding tilde `~`:
```regex
(?<!~)\\cite\{
```
Replace with `~\cite{`.
