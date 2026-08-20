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
        background: 'rgba(15, 23, 42, 0.95)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
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
      {/* Brand & Subtitle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: '9px',
            background: 'linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.2rem',
            boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)',
          }}
        >
          🚂
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <h1 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 800, color: '#F8FAFC', letterSpacing: '-0.02em' }}>
              Ticket to Ride <span style={{ color: '#38BDF8' }}>RL Lab</span>
            </h1>
            <span
              style={{
                fontSize: '0.65rem',
                padding: '0.1rem 0.35rem',
                borderRadius: '4px',
                background: 'rgba(56, 189, 248, 0.15)',
                color: '#38BDF8',
                fontWeight: 700,
                border: '1px solid rgba(56, 189, 248, 0.3)',
              }}
            >
              RESEARCH COCKPIT
            </span>
          </div>
          <div style={{ fontSize: '0.68rem', color: '#94A3B8' }}>
            Multi-Agent Neural Introspection & Deterministic Experiment Hub
          </div>
        </div>
      </div>

      {/* Mode Switcher Buttons */}
      <nav
        style={{
          display: 'flex',
          background: 'rgba(30, 41, 59, 0.7)',
          padding: '0.2rem',
          borderRadius: '8px',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          gap: '0.2rem',
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
                padding: '0.35rem 0.75rem',
                borderRadius: '6px',
                border: 'none',
                background: isActive ? '#3B82F6' : 'transparent',
                color: isActive ? '#FFFFFF' : '#94A3B8',
                fontSize: '0.8rem',
                fontWeight: isActive ? 700 : 500,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                boxShadow: isActive ? '0 2px 6px rgba(59, 130, 246, 0.35)' : 'none',
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
            background: 'rgba(30, 41, 59, 0.5)',
            border: '1px solid rgba(255, 255, 255, 0.06)',
            borderRadius: '6px',
            padding: '0.25rem 0.6rem',
            fontSize: '0.72rem',
          }}
        >
          <Bot size={13} color="#38BDF8" />
          <span style={{ color: '#38BDF8', fontWeight: 600 }}>Agent A (PPO)</span>
          <span style={{ color: '#64748B' }}>vs</span>
          <span style={{ color: '#818CF8', fontWeight: 600 }}>Agent B (Heuristic)</span>
        </div>

        {/* WebSocket Connection Pill */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.3rem',
            background: isConnected ? 'rgba(16, 185, 129, 0.12)' : 'rgba(100, 116, 139, 0.12)',
            border: `1px solid ${isConnected ? 'rgba(16, 185, 129, 0.3)' : 'rgba(100, 116, 139, 0.3)'}`,
            borderRadius: '6px',
            padding: '0.25rem 0.55rem',
            fontSize: '0.72rem',
            color: isConnected ? '#34D399' : '#94A3B8',
            fontWeight: 600,
          }}
        >
          {isConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
          <span>{isConnected ? 'Telemetry 60Hz' : 'Standby'}</span>
        </div>

        {/* Reset / New Session */}
        <button
          onClick={() => {
            resetSession();
            onNewGame?.();
          }}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.3rem',
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '6px',
            padding: '0.3rem 0.6rem',
            fontSize: '0.75rem',
            color: '#F1F5F9',
            fontWeight: 600,
            cursor: 'pointer',
          }}
          title="Start fresh session"
        >
          <RotateCcw size={12} /> New Match
        </button>
      </div>
    </header>
  );
});
