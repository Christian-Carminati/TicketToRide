import React, { useState, useEffect } from 'react';
import { useWorkbench } from '../../context';
import { ObservabilityMode } from '../../context/workbenchTypes';
import { MapPin, RefreshCw, Swords, Play, Pause, FastForward, CheckCircle2 } from 'lucide-react';
import { api } from '../../api/client';
import { CheckpointDTO } from '../../api/types';

interface BoardHeaderControlsProps {
  onResetZoom?: () => void;
  onSelectMap?: (mapName: 'usa' | 'mini') => void;
  onNewMatch?: (p1Type: string, p2Type: string, map: 'usa' | 'mini', ckpt1?: string, ckpt2?: string) => void;
  isAutoplaying?: boolean;
  onToggleAutoplay?: () => void;
  autoplaySpeedMs?: number;
  onSpeedChange?: (speed: number) => void;
}

export const BoardHeaderControls: React.FC<BoardHeaderControlsProps> = ({
  onResetZoom,
  onSelectMap,
  onNewMatch,
  isAutoplaying = false,
  onToggleAutoplay,
  autoplaySpeedMs = 600,
  onSpeedChange,
}) => {
  const { state, setObservabilityMode } = useWorkbench();
  const currentMap = state.gameState?.map_name?.toLowerCase() || 'usa';

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
        if (data.length > 0) {
          setP2Ckpt(data[0].path);
        }
      })
      .catch((e) => console.error('Failed to load checkpoints for duel plate:', e));
  }, []);

  const handleObservabilityChange = (mode: ObservabilityMode) => {
    setObservabilityMode(mode);
  };

  const handleApplyDuel = () => {
    onNewMatch?.(p1Type, p2Type, selectedMap, p1Ckpt || undefined, p2Ckpt || undefined);
    setIsDuelConfigOpen(false);
  };

  const currentPlayerIdx = state.gameState?.current_player_index ?? 0;
  const activePlayer = state.gameState?.players[currentPlayerIdx];
  const isGameOver = state.gameState?.is_game_over ?? false;

  const AGENT_OPTIONS = [
    { value: 'human', label: '👤 Human Conductor (Interactive)' },
    { value: 'alphazero', label: '🦅 AlphaZero (PUCT MCTS)' },
    { value: 'bayesian_mcts', label: '🎯 Bayesian MCTS (Opponent-Aware)' },
    { value: 'mcts', label: '🌲 Pure IS-MCTS (40 Sims)' },
    { value: 'recurrent_ppo', label: '🧵 Recurrent PPO (LSTM POMDP)' },
    { value: 'ppo', label: '⚡ CleanRL PPO (Feedforward)' },
    { value: 'dqn', label: '🧠 Double-DQN' },
    { value: 'strategic', label: '📐 Strategic Heuristic (Dijkstra)' },
    { value: 'greedy', label: '⚡ Greedy Score Bot' },
    { value: 'random', label: '🎲 Uniform Random' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', marginBottom: '0.4rem' }}>
      {/* 1. Main Header Controls Bar */}
      <div
        className="board-header-controls steampunk-panel"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'linear-gradient(180deg, #FAF4E6 0%, #E8D7BC 100%)',
          border: '1.5px solid #C59B27',
          borderRadius: '8px',
          padding: '0.4rem 0.85rem',
          gap: '0.5rem',
          flexWrap: 'wrap',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
        }}
      >
        {/* Left: Match Duel Setup Toggle & Active Turn Readout */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
          <button
            onClick={() => setIsDuelConfigOpen(!isDuelConfigOpen)}
            className="steampunk-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              fontSize: '0.78rem',
              background: isDuelConfigOpen
                ? 'linear-gradient(180deg, #FDE68A 0%, #D97706 100%)'
                : 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)',
            }}
            title="Configure model vs model duel"
          >
            <Swords size={13} color="#23140C" />
            <span>Match Setup & Models</span>
          </button>

          {/* Turn Indicator Banner */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: '#EADBBE',
              border: '1px solid #8C6305',
              padding: '0.2rem 0.6rem',
              borderRadius: '5px',
              fontSize: '0.75rem',
              fontFamily: "'Courier Prime', monospace",
              fontWeight: 700,
              color: '#23140C',
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: isGameOver ? '#785A42' : activePlayer?.color || '#3B82F6',
              }}
            />
            <span>
              {isGameOver
                ? '🏆 Match Finished'
                : `Turn ${state.gameState?.turn_number || 1}: ${activePlayer?.name || 'Player'} to act`}
            </span>
          </div>

          {/* Autoplay / Live Continuous Duel Controls */}
          {onToggleAutoplay && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <button
                onClick={onToggleAutoplay}
                disabled={isGameOver}
                className="steampunk-btn"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  padding: '0.25rem 0.6rem',
                  fontSize: '0.74rem',
                  background: isAutoplaying
                    ? 'linear-gradient(180deg, #FCA5A5 0%, #DC2626 100%)'
                    : 'linear-gradient(180deg, #86EFAC 0%, #16A34A 100%)',
                  color: '#FFFFFF',
                  border: isAutoplaying ? '1px solid #7F1D1D' : '1px solid #14532D',
                }}
                title="Continuous Model vs Model Execution"
              >
                {isAutoplaying ? <Pause size={12} /> : <Play size={12} />}
                <span>{isAutoplaying ? 'Pause Duel' : 'Autoplay Duel'}</span>
              </button>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.7rem', color: '#5A3822', fontFamily: "'Courier Prime', monospace" }}>
                <FastForward size={11} color="#8C6305" />
                <input
                  type="range"
                  min="150"
                  max="1500"
                  step="50"
                  value={autoplaySpeedMs}
                  onChange={(e) => onSpeedChange?.(Number(e.target.value))}
                  style={{ width: '60px', accentColor: '#B8860B' }}
                  title={`Speed: ${autoplaySpeedMs}ms / turn`}
                />
                <span>{autoplaySpeedMs}ms</span>
              </div>
            </div>
          )}
        </div>

        {/* Right: Observability Mode & Map Cartography Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          {/* Perception Toggle */}
          <div
            style={{
              display: 'flex',
              background: '#D8C3A0',
              borderRadius: '6px',
              padding: '2px',
              border: '1px solid #A88D75',
            }}
          >
            <button
              onClick={() => handleObservabilityChange('god')}
              style={{
                background: state.observabilityMode === 'god'
                  ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                  : 'transparent',
                color: state.observabilityMode === 'god' ? '#23140C' : '#5A3822',
                border: state.observabilityMode === 'god' ? '1px solid #6E4E04' : '1px solid transparent',
                borderRadius: '4px',
                padding: '0.2rem 0.5rem',
                fontSize: '0.72rem',
                fontFamily: "'Playfair Display', Georgia, serif",
                fontWeight: state.observabilityMode === 'god' ? 800 : 600,
                cursor: 'pointer',
              }}
              title="Omniscient Observer Mode"
            >
              God Mode
            </button>
            <button
              onClick={() => handleObservabilityChange('player_0')}
              style={{
                background: state.observabilityMode === 'player_0'
                  ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                  : 'transparent',
                color: state.observabilityMode === 'player_0' ? '#23140C' : '#5A3822',
                border: state.observabilityMode === 'player_0' ? '1px solid #6E4E04' : '1px solid transparent',
                borderRadius: '4px',
                padding: '0.2rem 0.5rem',
                fontSize: '0.72rem',
                fontFamily: "'Playfair Display', Georgia, serif",
                fontWeight: state.observabilityMode === 'player_0' ? 800 : 600,
                cursor: 'pointer',
              }}
              title="Player 1 Observation"
            >
              P1 View
            </button>
            <button
              onClick={() => handleObservabilityChange('player_1')}
              style={{
                background: state.observabilityMode === 'player_1'
                  ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                  : 'transparent',
                color: state.observabilityMode === 'player_1' ? '#23140C' : '#5A3822',
                border: state.observabilityMode === 'player_1' ? '1px solid #6E4E04' : '1px solid transparent',
                borderRadius: '4px',
                padding: '0.2rem 0.5rem',
                fontSize: '0.72rem',
                fontFamily: "'Playfair Display', Georgia, serif",
                fontWeight: state.observabilityMode === 'player_1' ? 800 : 600,
                cursor: 'pointer',
              }}
              title="Player 2 Observation"
            >
              P2 View
            </button>
          </div>

          {/* Map Cartography Selector */}
          <div
            style={{
              display: 'flex',
              background: '#D8C3A0',
              borderRadius: '6px',
              padding: '2px',
              border: '1px solid #A88D75',
            }}
          >
            <button
              onClick={() => onSelectMap?.('usa')}
              style={{
                background: currentMap === 'usa'
                  ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                  : 'transparent',
                color: currentMap === 'usa' ? '#23140C' : '#5A3822',
                border: currentMap === 'usa' ? '1px solid #6E4E04' : '1px solid transparent',
                borderRadius: '4px',
                padding: '0.2rem 0.55rem',
                fontSize: '0.72rem',
                fontFamily: "'Playfair Display', Georgia, serif",
                fontWeight: currentMap === 'usa' ? 800 : 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
              }}
            >
              <MapPin size={11} /> USA (1885)
            </button>
            <button
              onClick={() => onSelectMap?.('mini')}
              style={{
                background: currentMap === 'mini'
                  ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                  : 'transparent',
                color: currentMap === 'mini' ? '#23140C' : '#5A3822',
                border: currentMap === 'mini' ? '1px solid #6E4E04' : '1px solid transparent',
                borderRadius: '4px',
                padding: '0.2rem 0.55rem',
                fontSize: '0.72rem',
                fontFamily: "'Playfair Display', Georgia, serif",
                fontWeight: currentMap === 'mini' ? 800 : 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
              }}
            >
              <MapPin size={11} /> Mini (5 Cities)
            </button>
          </div>

          {onResetZoom && (
            <button
              onClick={onResetZoom}
              className="steampunk-btn"
              style={{ padding: '0.2rem 0.45rem', fontSize: '0.7rem' }}
              title="Reset Viewport"
            >
              <RefreshCw size={11} />
            </button>
          )}
        </div>
      </div>

      {/* 2. Collapsible Match Setup & Duel Configuration Drawer */}
      {isDuelConfigOpen && (
        <div
          className="steampunk-panel"
          style={{
            background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
            border: '2px solid #C59B27',
            borderRadius: '8px',
            padding: '0.85rem 1.15rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
            boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.9rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
              ⚔️ Configure Matchup: Choose Locomotive A & B
            </span>
            <span style={{ fontSize: '0.72rem', color: '#785A42', fontStyle: 'italic' }}>
              Pit any two algorithms against each other or play manually
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
            {/* Player 1 (Blue) */}
            <div style={{ background: '#FAF5EB', padding: '0.65rem 0.85rem', borderRadius: '6px', border: '1.5px solid #3B82F6' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', marginBottom: '0.4rem' }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: '#3B82F6' }} />
                <strong style={{ fontSize: '0.82rem', color: '#1E3A8A', fontFamily: "'Playfair Display', Georgia, serif" }}>
                  Player 1 (Blue Locomotive)
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
                    <option value="">-- Fresh Network Weights --</option>
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
                  Player 2 (Red Locomotive)
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
                    <option value="">-- Fresh Network Weights --</option>
                    {checkpoints.map((c) => (
                      <option key={c.checkpoint_id} value={c.path}>{c.name} ({c.algorithm.toUpperCase()})</option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          </div>

          {/* Map Selector & Action Row */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <label style={{ fontSize: '0.78rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>
                Survey Map:
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
                <option value="usa">USA Full (1885 - 36 Cities)</option>
                <option value="mini">Mini Synthetic (5 Cities)</option>
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
                Cancel
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
                <CheckCircle2 size={14} /> Start New Duel Match
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
