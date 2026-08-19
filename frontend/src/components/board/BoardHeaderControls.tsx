import React from 'react';
import { useWorkbench } from '../../context';
import { ObservabilityMode } from '../../context/workbenchTypes';
import { Eye, Bot, MapPin, RefreshCw } from 'lucide-react';

interface BoardHeaderControlsProps {
  onResetZoom?: () => void;
}

export const BoardHeaderControls: React.FC<BoardHeaderControlsProps> = ({ onResetZoom }) => {
  const { state, setObservabilityMode } = useWorkbench();

  const handleObservabilityChange = (mode: ObservabilityMode) => {
    setObservabilityMode(mode);
  };

  return (
    <div
      className="board-header-controls"
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'rgba(15, 23, 42, 0.85)',
        backdropFilter: 'blur(8px)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '8px',
        padding: '0.4rem 0.75rem',
        marginBottom: '0.5rem',
        gap: '0.5rem',
        flexWrap: 'wrap',
      }}
    >
      {/* Left: Observability Mode selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
        <span style={{ fontSize: '0.75rem', color: '#94A3B8', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          <Eye size={13} color="#38BDF8" /> Perception:
        </span>
        <div style={{ display: 'flex', background: 'rgba(30, 41, 59, 0.6)', borderRadius: '6px', padding: '2px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
          <button
            onClick={() => handleObservabilityChange('god')}
            style={{
              background: state.observabilityMode === 'god' ? '#3B82F6' : 'transparent',
              color: state.observabilityMode === 'god' ? '#FFFFFF' : '#94A3B8',
              border: 'none',
              borderRadius: '4px',
              padding: '0.2rem 0.5rem',
              fontSize: '0.72rem',
              fontWeight: state.observabilityMode === 'god' ? 700 : 500,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem',
              transition: 'all 0.15s ease',
            }}
            title="Full Omniscient View (Observer Mode)"
          >
            <Eye size={12} /> God Mode
          </button>
          <button
            onClick={() => handleObservabilityChange('player_0')}
            style={{
              background: state.observabilityMode === 'player_0' ? '#3B82F6' : 'transparent',
              color: state.observabilityMode === 'player_0' ? '#FFFFFF' : '#94A3B8',
              border: 'none',
              borderRadius: '4px',
              padding: '0.2rem 0.5rem',
              fontSize: '0.72rem',
              fontWeight: state.observabilityMode === 'player_0' ? 700 : 500,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem',
              transition: 'all 0.15s ease',
            }}
            title="Agent A (Player 0) Partial Observability Observation"
          >
            <Bot size={12} color="#38BDF8" /> Agent A
          </button>
          <button
            onClick={() => handleObservabilityChange('player_1')}
            style={{
              background: state.observabilityMode === 'player_1' ? '#3B82F6' : 'transparent',
              color: state.observabilityMode === 'player_1' ? '#FFFFFF' : '#94A3B8',
              border: 'none',
              borderRadius: '4px',
              padding: '0.2rem 0.5rem',
              fontSize: '0.72rem',
              fontWeight: state.observabilityMode === 'player_1' ? 700 : 500,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem',
              transition: 'all 0.15s ease',
            }}
            title="Agent B (Player 1) Partial Observability Observation"
          >
            <Bot size={12} color="#818CF8" /> Agent B
          </button>
        </div>
      </div>

      {/* Right: Map information & reset */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ fontSize: '0.72rem', color: '#64748B', display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
          <MapPin size={12} /> {state.gameState?.map_name?.toUpperCase() || 'USA BOARD'} (Deterministic)
        </span>
        {onResetZoom && (
          <button
            onClick={onResetZoom}
            style={{
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '4px',
              color: '#94A3B8',
              padding: '0.2rem 0.4rem',
              fontSize: '0.7rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.2rem',
            }}
            title="Reset Board Viewport"
          >
            <RefreshCw size={11} /> Reset
          </button>
        )}
      </div>
    </div>
  );
};
