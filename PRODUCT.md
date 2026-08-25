# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

RL researchers, machine learning engineers, and algorithmic game theorists investigating reinforcement learning on board games, partial observability (POMDP), neural network introspection, and multi-agent tournament dynamics.

## Product Purpose

TicketToRide RL Lab is an experimental Reinforcement Learning laboratory built from scratch to investigate and benchmark DQN, PPO, Recurrent PPO (LSTM), Partial Observability, Reward Shaping, Action Masking, Historical Policy Pools, Elo Tracking, and MCTS/AlphaZero-style tree search on Ticket to Ride mechanics.

## Positioning

A deterministic, high-performance RL laboratory combining zero-lag vector board interaction with real-time neural brain telemetry (masked logits, policy distribution, critic value head gauges, and observation tensor activations).

## Operating Context

Multi-Agent training runs, interactive human-vs-agent validation games, counterfactual replay scrubber analysis, and automated Elo tournament matrices.

## Capabilities and Constraints

- Pure Python deterministic core (`src/game/`) with zero frontend or heavy ML coupling.
- Real-time WebSocket streaming telemetry for 60Hz training and inference.
- Strict partial observability: Agent views never leak hidden deck order or opponent tickets.
- Vector Board Canvas for USA map (36 cities, 100 routes) and Mini map (5 cities).
- Synced Brain-to-Board hover links and neural action masking.

## Brand Commitments

- Aesthetic direction: Vintage Steampunk Cartographer & Victorian Analytical Lab (warm parchment paper palette, brass/bronze/copper mechanical instruments, riveted bezels, antique typography, high-clarity geographical vector USA map background with coastlines, lakes, relief hatches, and compass rose).
- No toy gamification or artificial slowness; high precision analytical instruments.

## Product Principles

1. RL-First & Learning-First: Clear, pedagogical, deterministic architecture.
2. High-Clarity Cartography: Clear, legible geographical board background that enhances rather than clutters game perception.
3. Synchronized Neural Introspection: Direct visual linkage between agent brain activations and board decisions.
4. Strict Reproducibility: Seeded deterministic game execution and replayability.
