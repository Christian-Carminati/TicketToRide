import React from 'react';
import { useWorkbench } from '../../context';
import { StudioMode } from '../../context/workbenchTypes';
import {
  Gamepad2,
  TrendingUp,
  Film,
  Trophy,
  Wifi,
  WifiOff,
  RotateCcw,
  Bot,
  BookOpen,
} from 'lucide-react';

interface StudioHeaderProps {
  onNewGame?: () => void;
}

export const StudioHeader: React.FC<StudioHeaderProps> = React.memo(({ onNewGame }) => {
  const { state, setStudioMode, resetSession } = useWorkbench();
  const { studioMode, isConnected } = state;

  const MODES: Array<{ id: StudioMode; label: string; icon: React.ReactNode }> = [
    { id: 'interactive', label: 'Interactive Game', icon: <Gamepad2 size={14} /> },
    { id: 'training_live', label: 'Live Training', icon: <TrendingUp size={14} /> },
    { id: 'replay_scrub', label: 'Replay Studio', icon: <Film size={14} /> },
    { id: 'tournament', label: 'Tournament Arena', icon: <Trophy size={14} /> },
    { id: 'reports', label: 'Research Reports', icon: <BookOpen size={14} /> },
  ];

  return (
    <header
      className="studio-header"
      style={{
        background: 'linear-gradient(180deg, #3A261A 0%, #26180F 100%)',
        borderBottom: '3px solid #C59B27',
        boxShadow: '0 4px 20px rgba(0,0,0,0.5), inset 0 -1px 0 rgba(246, 220, 136, 0.2)',
        padding: '0.6rem 1.25rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '0.75rem',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}
    >
      {/* Brand Plate */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: '8px',
            background: 'linear-gradient(135deg, #E6C665 0%, #B8860B 50%, #6E4E04 100%)',
            border: '1.5px solid #F6DC88',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.3rem',
            boxShadow: '0 2px 8px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.6)',
          }}
        >
          🚂
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <h1
              style={{
                margin: 0,
                fontSize: '1.25rem',
                fontFamily: "'Cinzel Decorative', Georgia, serif",
                fontWeight: 900,
                color: '#FAF5EB',
                letterSpacing: '0.02em',
                textShadow: '0 2px 4px rgba(0,0,0,0.6)',
              }}
            >
              Ticket to Ride <span style={{ color: '#F6DC88' }}>RL Lab</span>
            </h1>
            <span
              style={{
                fontSize: '0.65rem',
                padding: '0.15rem 0.45rem',
                borderRadius: '4px',
                background: 'rgba(197, 155, 39, 0.25)',
                color: '#F6DC88',
                fontFamily: "'Courier Prime', monospace",
                fontWeight: 700,
                border: '1px solid rgba(246, 220, 136, 0.4)',
                letterSpacing: '0.05em',
              }}
            >
              VICTORIAN COCKPIT
            </span>
          </div>
          <div
            style={{
              fontSize: '0.72rem',
              color: '#D4C09D',
              fontFamily: "'Crimson Pro', Georgia, serif",
              fontStyle: 'italic',
            }}
          >
            Multi-Agent Neural Introspection & Cartographical Analytical Hub
          </div>
        </div>
      </div>

      {/* Mode Switcher Navigation */}
      <nav
        style={{
          display: 'flex',
          background: 'rgba(26, 16, 10, 0.8)',
          padding: '0.25rem',
          borderRadius: '8px',
          border: '1.5px solid #8C6305',
          gap: '0.25rem',
          boxShadow: 'inset 0 2px 6px rgba(0,0,0,0.4)',
        }}
      >
        {MODES.map((m) => {
          const isActive = studioMode === m.id;
          return (
            <button
              key={m.id}
              onClick={() => setStudioMode(m.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                padding: '0.35rem 0.8rem',
                borderRadius: '6px',
                border: isActive ? '1px solid #6E4E04' : '1px solid transparent',
                background: isActive
                  ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                  : 'transparent',
                color: isActive ? '#23140C' : '#D4C09D',
                fontFamily: "'Playfair Display', Georgia, serif",
                fontSize: '0.82rem',
                fontWeight: isActive ? 800 : 500,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                boxShadow: isActive
                  ? '0 2px 6px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.6)'
                  : 'none',
                textShadow: isActive ? '0 1px 0 rgba(255,255,255,0.4)' : 'none',
              }}
            >
              {m.icon}
              <span>{m.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Status & Control Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        {/* Agent Matchup Badge */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: 'rgba(26, 16, 10, 0.7)',
            border: '1px solid rgba(197, 155, 39, 0.35)',
            borderRadius: '6px',
            padding: '0.25rem 0.65rem',
            fontSize: '0.75rem',
            fontFamily: "'Courier Prime', monospace",
          }}
        >
          <Bot size={13} color="#F6DC88" />
          <span style={{ color: '#F6DC88', fontWeight: 700 }}>Agent A (PPO)</span>
          <span style={{ color: '#A88D75' }}>vs</span>
          <span style={{ color: '#CD7F32', fontWeight: 700 }}>Agent B (Heuristic)</span>
        </div>

        {/* WebSocket Telemetry Lamp */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.35rem',
            background: isConnected ? 'rgba(21, 128, 61, 0.25)' : 'rgba(120, 90, 66, 0.25)',
            border: `1px solid ${isConnected ? '#15803D' : '#785A42'}`,
            borderRadius: '6px',
            padding: '0.25rem 0.6rem',
            fontSize: '0.75rem',
            fontFamily: "'Courier Prime', monospace",
            color: isConnected ? '#86EFAC' : '#D4C09D',
            fontWeight: 700,
          }}
        >
          {isConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
          <span>{isConnected ? 'Telegraph 60Hz' : 'Standby'}</span>
        </div>

        {/* Reset / New Match Button */}
        <button
          onClick={() => {
            resetSession();
            onNewGame?.();
          }}
          className="steampunk-btn"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.35rem',
            padding: '0.35rem 0.75rem',
            fontSize: '0.78rem',
          }}
          title="Start fresh session"
        >
          <RotateCcw size={13} /> New Match
        </button>
      </div>
    </header>
  );
});
