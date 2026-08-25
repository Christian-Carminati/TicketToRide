import React from 'react';
import { useWorkbench } from '../../context';
import { ObservabilityMode } from '../../context/workbenchTypes';
import { Eye, Bot, MapPin, RefreshCw } from 'lucide-react';

interface BoardHeaderControlsProps {
  onResetZoom?: () => void;
  onSelectMap?: (mapName: 'usa' | 'mini') => void;
}

export const BoardHeaderControls: React.FC<BoardHeaderControlsProps> = ({
  onResetZoom,
  onSelectMap,
}) => {
  const { state, setObservabilityMode } = useWorkbench();
  const currentMap = state.gameState?.map_name?.toLowerCase() || 'usa';

  const handleObservabilityChange = (mode: ObservabilityMode) => {
    setObservabilityMode(mode);
  };

  return (
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
        marginBottom: '0.4rem',
        gap: '0.5rem',
        flexWrap: 'wrap',
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
      }}
    >
      {/* Left: Observability / Perception Mode Selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span
          style={{
            fontSize: '0.8rem',
            color: '#4A2F1D',
            fontFamily: "'Playfair Display', Georgia, serif",
            fontWeight: 800,
            display: 'flex',
            alignItems: 'center',
            gap: '0.3rem',
          }}
        >
          <Eye size={14} color="#9E6B00" /> Perception:
        </span>
        <div
          style={{
            display: 'flex',
            background: '#D8C3A0',
            borderRadius: '6px',
            padding: '2px',
            border: '1px solid #A88D75',
            boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.2)',
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
              padding: '0.2rem 0.6rem',
              fontSize: '0.75rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: state.observabilityMode === 'god' ? 800 : 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              transition: 'all 0.15s ease',
              boxShadow: state.observabilityMode === 'god' ? '0 1px 4px rgba(0,0,0,0.2)' : 'none',
            }}
            title="Full Omniscient View (Observer Mode)"
          >
            <Eye size={12} /> God Mode
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
              padding: '0.2rem 0.6rem',
              fontSize: '0.75rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: state.observabilityMode === 'player_0' ? 800 : 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              transition: 'all 0.15s ease',
              boxShadow: state.observabilityMode === 'player_0' ? '0 1px 4px rgba(0,0,0,0.2)' : 'none',
            }}
            title="Agent A (Player 0) Partial Observability Observation"
          >
            <Bot size={12} color="#B8860B" /> Agent A
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
              padding: '0.2rem 0.6rem',
              fontSize: '0.75rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: state.observabilityMode === 'player_1' ? 800 : 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              transition: 'all 0.15s ease',
              boxShadow: state.observabilityMode === 'player_1' ? '0 1px 4px rgba(0,0,0,0.2)' : 'none',
            }}
            title="Agent B (Player 1) Partial Observability Observation"
          >
            <Bot size={12} color="#CD7F32" /> Agent B
          </button>
        </div>
      </div>

      {/* Right: Map Cartography Selector & Reset */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <div
          style={{
            display: 'flex',
            background: '#D8C3A0',
            borderRadius: '6px',
            padding: '2px',
            border: '1px solid #A88D75',
            boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.2)',
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
              padding: '0.2rem 0.6rem',
              fontSize: '0.75rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: currentMap === 'usa' ? 800 : 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
            }}
          >
            <MapPin size={12} /> USA (36 Cities, 100 Routes)
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
              padding: '0.2rem 0.6rem',
              fontSize: '0.75rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: currentMap === 'mini' ? 800 : 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
            }}
          >
            <MapPin size={12} /> Mini (5 Cities)
          </button>
        </div>

        {onResetZoom && (
          <button
            onClick={onResetZoom}
            className="steampunk-btn"
            style={{
              padding: '0.25rem 0.5rem',
              fontSize: '0.72rem',
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
