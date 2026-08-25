import React, { useState, useEffect } from 'react';
import { GameStateDTO, CheckpointDTO } from '../../api/types';
import { api } from '../../api/client';
import {
  Swords,
  Play,
  Pause,
  FastForward,
  MapPin,
  Bot,
  User,
  Brain,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
} from 'lucide-react';


interface MatchScoreBoardHUDProps {
  gameState: GameStateDTO | null;
  isAutoplaying?: boolean;
  onToggleAutoplay?: () => void;
  autoplaySpeedMs?: number;
  onSpeedChange?: (speed: number) => void;
  onSelectMap?: (mapName: 'usa' | 'mini') => void;
  onNewMatch?: (p1Type: string, p2Type: string, map: 'usa' | 'mini', ckpt1?: string, ckpt2?: string) => void;
  onToggleTelemetryDrawer?: () => void;
  isTelemetryDrawerOpen?: boolean;
  selectedModel?: string;
}

export const MatchScoreBoardHUD: React.FC<MatchScoreBoardHUDProps> = ({
  gameState,
  isAutoplaying = false,
  onToggleAutoplay,
  autoplaySpeedMs = 600,
  onSpeedChange,
  onSelectMap,
  onNewMatch,
  onToggleTelemetryDrawer,
  isTelemetryDrawerOpen = false,
  selectedModel = 'alphazero',
}) => {
  const currentMap = gameState?.map_name?.toLowerCase() || 'usa';
  const players = gameState?.players || [];
  const p1 = players[0];
  const p2 = players[1];
  const currentTurnIdx = gameState?.current_player_index ?? 0;
  const isGameOver = gameState?.is_game_over ?? false;
  const activePlayer = players[currentTurnIdx];

  const [isDuelConfigOpen, setIsDuelConfigOpen] = useState(false);
  const [p1Type, setP1Type] = useState<string>('human');
  const [p2Type, setP2Type] = useState<string>('alphazero');
  const [p1Ckpt, setP1Ckpt] = useState<string>('');
  const [p2Ckpt, setP2Ckpt] = useState<string>('');
  const [selectedMap, setSelectedMap] = useState<'usa' | 'mini'>((currentMap as 'usa' | 'mini') || 'usa');
  const [checkpoints, setCheckpoints] = useState<CheckpointDTO[]>([]);

  useEffect(() => {
    api.listCheckpoints()
      .then((data) => {
        setCheckpoints(data);
        if (data.length > 0 && !p2Ckpt) {
          setP2Ckpt(data[0].path);
        }
      })
      .catch((e) => console.error('Failed to load checkpoints:', e));
  }, [p2Ckpt]);

  const handleApplyDuel = () => {
    onNewMatch?.(p1Type, p2Type, selectedMap, p1Ckpt || undefined, p2Ckpt || undefined);
    setIsDuelConfigOpen(false);
  };

  const AGENT_OPTIONS = [
    { value: 'human', label: '👤 Conduttore Umano (Interattivo)' },
    { value: 'alphazero', label: '🦅 AlphaZero (PUCT MCTS)' },
    { value: 'bayesian_mcts', label: '🎯 Bayesian MCTS (Opponent-Aware)' },
    { value: 'mcts', label: '🌲 Pure IS-MCTS (40 Sims)' },
    { value: 'recurrent_ppo', label: '🧵 Recurrent PPO (LSTM POMDP)' },
    { value: 'ppo', label: '⚡ CleanRL PPO (Feedforward)' },
    { value: 'dqn', label: '🧠 Double-DQN' },
    { value: 'strategic', label: '📐 Strategico Heuristic (Dijkstra)' },
    { value: 'greedy', label: '⚡ Greedy Score Bot' },
    { value: 'random', label: '🎲 Random Bot' },
  ];

  const isHumanTurn = activePlayer?.name.toLowerCase().includes('human') || p1Type === 'human' && currentTurnIdx === 0;

  return (
    <div className="match-scoreboard-hud" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      {/* Main Scoreboard Bar */}
      <div
        className="steampunk-panel"
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(220px, 1fr) minmax(280px, 1.4fr) minmax(220px, 1fr)',
          alignItems: 'center',
          gap: '0.75rem',
          padding: '0.65rem 1rem',
          background: 'linear-gradient(180deg, #FAF4E6 0%, #E8D7BC 100%)',
          border: '2px solid #C59B27',
          borderRadius: '12px',
          boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
        }}
      >
        {/* Left: Player 1 Plaque */}
        {p1 ? (
          <div
            className="player-hud-card"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0.5rem 0.75rem',
              borderRadius: '8px',
              backgroundColor: currentTurnIdx === 0 && !isGameOver ? '#FFFDF7' : '#FAF5EB',
              border: currentTurnIdx === 0 && !isGameOver ? '2px solid #2563EB' : '1.5px solid rgba(184, 134, 11, 0.4)',
              boxShadow: currentTurnIdx === 0 && !isGameOver
                ? '0 0 14px rgba(37, 99, 235, 0.4), inset 0 1px 0 rgba(255,255,255,0.9)'
                : '0 2px 4px rgba(0,0,0,0.08)',
              transition: 'all 0.25s ease',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: '50%',
                  backgroundColor: p1.color || '#2563EB',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#FAF5EB',
                  border: '1.5px solid #1E3A8A',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.25)',
                }}
              >
                <User size={16} />
              </div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <span style={{ fontSize: '0.88rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif", color: '#1E3A8A' }}>
                    {p1.name}
                  </span>
                  {currentTurnIdx === 0 && !isGameOver && (
                    <span
                      style={{
                        fontSize: '0.62rem',
                        background: '#2563EB',
                        color: '#FFFFFF',
                        padding: '0.05rem 0.35rem',
                        borderRadius: '3px',
                        fontWeight: 900,
                        fontFamily: "'Courier Prime', monospace",
                        letterSpacing: '0.04em',
                      }}
                    >
                      TURNO
                    </span>
                  )}
                </div>
                <div style={{ fontSize: '0.72rem', color: '#5A3822', fontFamily: "'Courier Prime', monospace" }}>
                  🚂 {p1.trains_remaining}/45 · 🃏 {Object.values(p1.cards_in_hand || {}).reduce((a, b) => a + b, 0)} · 🎫 {p1.tickets?.length || 0}
                </div>
              </div>
            </div>

            {/* Giant Score */}
            <div
              style={{
                textAlign: 'right',
                background: '#FAF5EB',
                padding: '0.2rem 0.6rem',
                borderRadius: '6px',
                border: '1.5px solid #C59B27',
              }}
            >
              <div style={{ fontSize: '0.6rem', color: '#785A42', fontFamily: "'Courier Prime', monospace", textTransform: 'uppercase' }}>Punti</div>
              <div style={{ fontSize: '1.45rem', fontWeight: 900, fontFamily: "'Courier Prime', monospace", color: '#B91C1C', lineHeight: 1 }}>
                {p1.score}
              </div>
            </div>
          </div>
        ) : (
          <div style={{ fontSize: '0.8rem', color: '#785A42' }}>Caricamento Giocatore 1...</div>
        )}

        {/* Center: Turn Status Banner & Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.35rem' }}>
          {/* Turn Prompt Badge */}
          <div
            style={{
              width: '100%',
              textAlign: 'center',
              background: isGameOver
                ? 'linear-gradient(180deg, #FDE68A 0%, #D97706 100%)'
                : currentTurnIdx === 0
                ? 'linear-gradient(180deg, #DBEAFE 0%, #BFDBFE 100%)'
                : 'linear-gradient(180deg, #FEE2E2 0%, #FECACA 100%)',
              border: `1.5px solid ${isGameOver ? '#B45309' : currentTurnIdx === 0 ? '#3B82F6' : '#EF4444'}`,
              borderRadius: '6px',
              padding: '0.3rem 0.65rem',
              boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
            }}
          >
            <div style={{ fontSize: '0.78rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
              {isGameOver
                ? '🏆 Partita Terminata — Consulta i Punteggi'
                : isHumanTurn
                ? `👉 Turno #${gameState?.turn_number || 1}: È il tuo turno! Clicca una tratta o pesca carte.`
                : `⏳ Turno #${gameState?.turn_number || 1}: ${activePlayer?.name || 'Avversario'} sta pianificando la mossa...`}
            </div>
          </div>

          {/* Quick Action Controls Cluster */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap', justifyContent: 'center' }}>
            {/* Match Setup Button */}
            <button
              onClick={() => setIsDuelConfigOpen(!isDuelConfigOpen)}
              className="steampunk-btn"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.3rem',
                fontSize: '0.72rem',
                padding: '0.2rem 0.55rem',
                background: isDuelConfigOpen
                  ? 'linear-gradient(180deg, #FDE68A 0%, #D97706 100%)'
                  : 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)',
              }}
              title="Configura Giocatori e Modelli AI"
            >
              <Swords size={12} />
              <span>Nuova Sfida</span>
              {isDuelConfigOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
            </button>

            {/* Map Switcher */}
            <div
              style={{
                display: 'flex',
                background: '#D8C3A0',
                borderRadius: '5px',
                padding: '2px',
                border: '1px solid #A88D75',
              }}
            >
              <button
                onClick={() => onSelectMap?.('usa')}
                style={{
                  background: currentMap === 'usa' ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 100%)' : 'transparent',
                  color: currentMap === 'usa' ? '#23140C' : '#5A3822',
                  border: currentMap === 'usa' ? '1px solid #6E4E04' : '1px solid transparent',
                  borderRadius: '3px',
                  padding: '0.15rem 0.45rem',
                  fontSize: '0.68rem',
                  fontFamily: "'Playfair Display', Georgia, serif",
                  fontWeight: currentMap === 'usa' ? 800 : 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.2rem',
                }}
              >
                <MapPin size={10} /> USA (1885)
              </button>
              <button
                onClick={() => onSelectMap?.('mini')}
                style={{
                  background: currentMap === 'mini' ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 100%)' : 'transparent',
                  color: currentMap === 'mini' ? '#23140C' : '#5A3822',
                  border: currentMap === 'mini' ? '1px solid #6E4E04' : '1px solid transparent',
                  borderRadius: '3px',
                  padding: '0.15rem 0.45rem',
                  fontSize: '0.68rem',
                  fontFamily: "'Playfair Display', Georgia, serif",
                  fontWeight: currentMap === 'mini' ? 800 : 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.2rem',
                }}
              >
                <MapPin size={10} /> Mini (5)
              </button>
            </div>

            {/* Autoplay Duel Controls */}
            {onToggleAutoplay && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <button
                  onClick={onToggleAutoplay}
                  disabled={isGameOver}
                  className="steampunk-btn"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.25rem',
                    padding: '0.2rem 0.55rem',
                    fontSize: '0.7rem',
                    background: isAutoplaying
                      ? 'linear-gradient(180deg, #FCA5A5 0%, #DC2626 100%)'
                      : 'linear-gradient(180deg, #86EFAC 0%, #16A34A 100%)',
                    color: '#FFFFFF',
                    border: isAutoplaying ? '1px solid #7F1D1D' : '1px solid #14532D',
                  }}
                  title="Autoplay Partita Continuo"
                >
                  {isAutoplaying ? <Pause size={11} /> : <Play size={11} />}
                  <span>{isAutoplaying ? 'Pausa' : 'Autoplay'}</span>
                </button>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.2rem', fontSize: '0.65rem', color: '#5A3822', fontFamily: "'Courier Prime', monospace" }}>
                  <FastForward size={10} color="#8C6305" />
                  <input
                    type="range"
                    min="150"
                    max="1500"
                    step="50"
                    value={autoplaySpeedMs}
                    onChange={(e) => onSpeedChange?.(Number(e.target.value))}
                    style={{ width: '50px', accentColor: '#B8860B' }}
                    title={`Velocità: ${autoplaySpeedMs}ms / turno`}
                  />
                  <span>{autoplaySpeedMs}ms</span>
                </div>
              </div>
            )}

            {/* AI Telemetry Slide-Out Drawer Toggle Button */}
            {onToggleTelemetryDrawer && (
              <button
                onClick={onToggleTelemetryDrawer}
                className="steampunk-btn"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  fontSize: '0.72rem',
                  padding: '0.2rem 0.55rem',
                  background: isTelemetryDrawerOpen
                    ? 'linear-gradient(180deg, #FDE68A 0%, #D97706 100%)'
                    : 'linear-gradient(180deg, #3A261A 0%, #23140C 100%)',
                  color: isTelemetryDrawerOpen ? '#23140C' : '#F6DC88',
                  border: '1.5px solid #C59B27',
                }}
                title="Apri Telemetria Reti Neurali & Tree Search"
              >
                <Brain size={12} color={isTelemetryDrawerOpen ? '#23140C' : '#F6DC88'} />
                <span>Telemetria AI ({selectedModel.toUpperCase()})</span>
                {isTelemetryDrawerOpen && <span style={{ fontSize: '0.6rem' }}>●</span>}
              </button>
            )}
          </div>
        </div>

        {/* Right: Player 2 Plaque */}
        {p2 ? (
          <div
            className="player-hud-card"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0.5rem 0.75rem',
              borderRadius: '8px',
              backgroundColor: currentTurnIdx === 1 && !isGameOver ? '#FFFDF7' : '#FAF5EB',
              border: currentTurnIdx === 1 && !isGameOver ? '2px solid #DC2626' : '1.5px solid rgba(184, 134, 11, 0.4)',
              boxShadow: currentTurnIdx === 1 && !isGameOver
                ? '0 0 14px rgba(220, 38, 38, 0.4), inset 0 1px 0 rgba(255,255,255,0.9)'
                : '0 2px 4px rgba(0,0,0,0.08)',
              transition: 'all 0.25s ease',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: '50%',
                  backgroundColor: p2.color || '#DC2626',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#FAF5EB',
                  border: '1.5px solid #991B1B',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.25)',
                }}
              >
                <Bot size={16} />
              </div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <span style={{ fontSize: '0.88rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif", color: '#991B1B' }}>
                    {p2.name}
                  </span>
                  {currentTurnIdx === 1 && !isGameOver && (
                    <span
                      style={{
                        fontSize: '0.62rem',
                        background: '#DC2626',
                        color: '#FFFFFF',
                        padding: '0.05rem 0.35rem',
                        borderRadius: '3px',
                        fontWeight: 900,
                        fontFamily: "'Courier Prime', monospace",
                        letterSpacing: '0.04em',
                      }}
                    >
                      TURNO
                    </span>
                  )}
                </div>
                <div style={{ fontSize: '0.72rem', color: '#5A3822', fontFamily: "'Courier Prime', monospace" }}>
                  🚂 {p2.trains_remaining}/45 · 🃏 {Object.values(p2.cards_in_hand || {}).reduce((a, b) => a + b, 0)} · 🎫 {p2.tickets?.length || 0}
                </div>
              </div>
            </div>

            {/* Giant Score */}
            <div
              style={{
                textAlign: 'right',
                background: '#FAF5EB',
                padding: '0.2rem 0.6rem',
                borderRadius: '6px',
                border: '1.5px solid #C59B27',
              }}
            >
              <div style={{ fontSize: '0.6rem', color: '#785A42', fontFamily: "'Courier Prime', monospace", textTransform: 'uppercase' }}>Punti</div>
              <div style={{ fontSize: '1.45rem', fontWeight: 900, fontFamily: "'Courier Prime', monospace", color: '#B91C1C', lineHeight: 1 }}>
                {p2.score}
              </div>
            </div>
          </div>
        ) : (
          <div style={{ fontSize: '0.8rem', color: '#785A42', textAlign: 'right' }}>Caricamento Giocatore 2...</div>
        )}
      </div>

      {/* Collapsible Match Setup & Duel Configuration Plate */}
      {isDuelConfigOpen && (
        <div
          className="steampunk-panel"
          style={{
            background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
            border: '2px solid #C59B27',
            borderRadius: '10px',
            padding: '0.85rem 1.15rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
            boxShadow: '0 6px 20px rgba(0,0,0,0.25)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.9rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
              ⚔️ Configura Sfida: Scegli Conduttore 1 e Conduttore 2
            </span>
            <span style={{ fontSize: '0.72rem', color: '#785A42', fontStyle: 'italic' }}>
              Gioca contro qualsiasi modello AI o osserva duelli tra algoritmi
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
            {/* Player 1 (Blue) */}
            <div style={{ background: '#FAF5EB', padding: '0.65rem 0.85rem', borderRadius: '6px', border: '1.5px solid #3B82F6' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', marginBottom: '0.4rem' }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: '#3B82F6' }} />
                <strong style={{ fontSize: '0.82rem', color: '#1E3A8A', fontFamily: "'Playfair Display', Georgia, serif" }}>
                  Giocatore 1 (Blu)
                </strong>
              </div>

              <select
                value={p1Type}
                onChange={(e) => setP1Type(e.target.value)}
                style={{
                  width: '100%',
                  backgroundColor: '#FFFFFF',
                  color: '#23140C',
                  border: '1.5px solid #8C6305',
                  borderRadius: '5px',
                  padding: '0.3rem 0.5rem',
                  fontSize: '0.78rem',
                  fontWeight: 700,
                  fontFamily: "'Playfair Display', Georgia, serif",
                  marginBottom: '0.4rem',
                }}
              >
                {AGENT_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>

              {['alphazero', 'ppo', 'dqn', 'recurrent_ppo'].includes(p1Type) && checkpoints.length > 0 && (
                <div>
                  <label style={{ fontSize: '0.7rem', color: '#5A3822', fontWeight: 700 }}>Checkpoint (.pt):</label>
                  <select
                    value={p1Ckpt}
                    onChange={(e) => setP1Ckpt(e.target.value)}
                    style={{
                      width: '100%',
                      backgroundColor: '#FFFFFF',
                      color: '#23140C',
                      border: '1px solid #C59B27',
                      borderRadius: '4px',
                      padding: '0.2rem 0.4rem',
                      fontSize: '0.72rem',
                      fontFamily: "'Courier Prime', monospace",
                    }}
                  >
                    <option value="">-- Pesi Rete Predefiniti --</option>
                    {checkpoints.map((c) => (
                      <option key={c.checkpoint_id} value={c.path}>{c.name} ({c.algorithm.toUpperCase()})</option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {/* Player 2 (Red) */}
            <div style={{ background: '#FAF5EB', padding: '0.65rem 0.85rem', borderRadius: '6px', border: '1.5px solid #EF4444' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', marginBottom: '0.4rem' }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: '#EF4444' }} />
                <strong style={{ fontSize: '0.82rem', color: '#991B1B', fontFamily: "'Playfair Display', Georgia, serif" }}>
                  Giocatore 2 (Rosso)
                </strong>
              </div>

              <select
                value={p2Type}
                onChange={(e) => setP2Type(e.target.value)}
                style={{
                  width: '100%',
                  backgroundColor: '#FFFFFF',
                  color: '#23140C',
                  border: '1.5px solid #8C6305',
                  borderRadius: '5px',
                  padding: '0.3rem 0.5rem',
                  fontSize: '0.78rem',
                  fontWeight: 700,
                  fontFamily: "'Playfair Display', Georgia, serif",
                  marginBottom: '0.4rem',
                }}
              >
                {AGENT_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>

              {['alphazero', 'ppo', 'dqn', 'recurrent_ppo'].includes(p2Type) && checkpoints.length > 0 && (
                <div>
                  <label style={{ fontSize: '0.7rem', color: '#5A3822', fontWeight: 700 }}>Checkpoint (.pt):</label>
                  <select
                    value={p2Ckpt}
                    onChange={(e) => setP2Ckpt(e.target.value)}
                    style={{
                      width: '100%',
                      backgroundColor: '#FFFFFF',
                      color: '#23140C',
                      border: '1px solid #C59B27',
                      borderRadius: '4px',
                      padding: '0.2rem 0.4rem',
                      fontSize: '0.72rem',
                      fontFamily: "'Courier Prime', monospace",
                    }}
                  >
                    <option value="">-- Pesi Rete Predefiniti --</option>
                    {checkpoints.map((c) => (
                      <option key={c.checkpoint_id} value={c.path}>{c.name} ({c.algorithm.toUpperCase()})</option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          </div>

          {/* Map & Action Row */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <label style={{ fontSize: '0.78rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>
                Mappa Partita:
              </label>
              <select
                value={selectedMap}
                onChange={(e) => setSelectedMap(e.target.value as 'usa' | 'mini')}
                style={{
                  backgroundColor: '#FFFFFF',
                  color: '#23140C',
                  border: '1.5px solid #8C6305',
                  borderRadius: '5px',
                  padding: '0.25rem 0.5rem',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  fontFamily: "'Playfair Display', Georgia, serif",
                }}
              >
                <option value="usa">USA Completa (1885 - 36 Città)</option>
                <option value="mini">Mini Sintetica (5 Città)</option>
              </select>
            </div>

            <div style={{ display: 'flex', gap: '0.6rem' }}>
              <button
                onClick={() => setIsDuelConfigOpen(false)}
                className="steampunk-btn"
                style={{
                  background: 'linear-gradient(180deg, #D4C09D 0%, #A88D75 100%)',
                  color: '#23140C',
                  border: '1px solid #785A42',
                  padding: '0.35rem 0.85rem',
                  fontSize: '0.78rem',
                }}
              >
                Annulla
              </button>
              <button
                onClick={handleApplyDuel}
                className="steampunk-btn"
                style={{
                  background: 'linear-gradient(180deg, #86EFAC 0%, #16A34A 50%, #14532D 100%)',
                  color: '#FFFFFF',
                  border: '1px solid #14532D',
                  padding: '0.35rem 1.25rem',
                  fontSize: '0.82rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                }}
              >
                <CheckCircle2 size={14} /> Avvia Nuova Partita
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
