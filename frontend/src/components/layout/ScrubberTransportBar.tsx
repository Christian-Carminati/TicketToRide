import React from 'react';
import { useWorkbench } from '../../context';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ChevronLeft,
  ChevronRight,
  Zap,
} from 'lucide-react';

interface ScrubberTransportBarProps {
  onBotStep?: () => void;
  isStepping?: boolean;
}

export const ScrubberTransportBar: React.FC<ScrubberTransportBarProps> = ({
  onBotStep,
  isStepping = false,
}) => {
  const {
    state,
    setStepIndex,
    setIsPlaying,
    setPlaybackSpeed,
  } = useWorkbench();

  const {
    currentStepIndex,
    maxStepIndex,
    isPlaying,
    playbackSpeed,
    gameState,
  } = state;

  const currentTurn = gameState?.turn_number ?? 1;

  return (
    <div
      className="scrubber-transport-bar"
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(15, 23, 42, 0.9)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '10px',
        padding: '0.45rem 1rem',
        gap: '1rem',
        flexWrap: 'wrap',
      }}
    >
      {/* Left: Turn & Step metadata */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
        <div
          style={{
            background: 'rgba(59, 130, 246, 0.15)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '6px',
            padding: '0.2rem 0.5rem',
            fontSize: '0.75rem',
            color: '#38BDF8',
            fontWeight: 700,
            fontFamily: 'monospace',
          }}
        >
          TURN {currentTurn}
        </div>
        <div style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
          Step: <strong style={{ color: '#F1F5F9', fontFamily: 'monospace' }}>{currentStepIndex + 1}</strong> / {Math.max(1, maxStepIndex + 1)}
        </div>
      </div>

      {/* Center: Transport Controls & Scrubber Slider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: 1, maxWidth: '650px' }}>
        {/* Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          <button
            onClick={() => setStepIndex(0)}
            disabled={currentStepIndex === 0}
            style={{
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '4px',
              color: currentStepIndex === 0 ? '#475569' : '#94A3B8',
              padding: '0.25rem 0.4rem',
              cursor: currentStepIndex === 0 ? 'default' : 'pointer',
              display: 'flex',
              alignItems: 'center',
            }}
            title="First Step"
          >
            <SkipBack size={13} />
          </button>

          <button
            onClick={() => setStepIndex(Math.max(0, currentStepIndex - 1))}
            disabled={currentStepIndex === 0}
            style={{
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '4px',
              color: currentStepIndex === 0 ? '#475569' : '#94A3B8',
              padding: '0.25rem 0.4rem',
              cursor: currentStepIndex === 0 ? 'default' : 'pointer',
              display: 'flex',
              alignItems: 'center',
            }}
            title="Step Backward"
          >
            <ChevronLeft size={13} />
          </button>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            style={{
              background: isPlaying ? '#EF4444' : '#3B82F6',
              border: 'none',
              borderRadius: '6px',
              color: '#FFFFFF',
              padding: '0.3rem 0.75rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              fontSize: '0.75rem',
              fontWeight: 700,
              boxShadow: isPlaying ? '0 2px 8px rgba(239, 68, 68, 0.4)' : '0 2px 8px rgba(59, 130, 246, 0.4)',
            }}
          >
            {isPlaying ? <Pause size={13} /> : <Play size={13} />}
            <span>{isPlaying ? 'Pause' : 'Play'}</span>
          </button>

          <button
            onClick={() => setStepIndex(Math.min(maxStepIndex, currentStepIndex + 1))}
            disabled={currentStepIndex >= maxStepIndex}
            style={{
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '4px',
              color: currentStepIndex >= maxStepIndex ? '#475569' : '#94A3B8',
              padding: '0.25rem 0.4rem',
              cursor: currentStepIndex >= maxStepIndex ? 'default' : 'pointer',
              display: 'flex',
              alignItems: 'center',
            }}
            title="Step Forward"
          >
            <ChevronRight size={13} />
          </button>

          <button
            onClick={() => setStepIndex(maxStepIndex)}
            disabled={currentStepIndex >= maxStepIndex}
            style={{
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '4px',
              color: currentStepIndex >= maxStepIndex ? '#475569' : '#94A3B8',
              padding: '0.25rem 0.4rem',
              cursor: currentStepIndex >= maxStepIndex ? 'default' : 'pointer',
              display: 'flex',
              alignItems: 'center',
            }}
            title="Last Step"
          >
            <SkipForward size={13} />
          </button>
        </div>

        {/* Timeline Slider */}
        <input
          type="range"
          min={0}
          max={Math.max(1, maxStepIndex)}
          value={currentStepIndex}
          onChange={(e) => setStepIndex(Number(e.target.value))}
          style={{
            flex: 1,
            accentColor: '#38BDF8',
            cursor: 'pointer',
          }}
        />
      </div>

      {/* Right: Bot Step Trigger & Speed selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        {onBotStep && (
          <button
            onClick={onBotStep}
            disabled={isStepping || isPlaying}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.3rem 0.75rem',
              fontSize: '0.75rem',
              fontWeight: 700,
              cursor: isStepping || isPlaying ? 'not-allowed' : 'pointer',
              opacity: isStepping || isPlaying ? 0.6 : 1,
              boxShadow: '0 2px 8px rgba(16, 185, 129, 0.3)',
            }}
          >
            <Zap size={13} />
            <span>{isStepping ? 'Stepping...' : 'Bot Action'}</span>
          </button>
        )}

        {/* Speed Dropdown */}
        <select
          value={playbackSpeed}
          onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
          style={{
            background: 'rgba(30, 41, 59, 0.8)',
            color: '#F1F5F9',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '6px',
            padding: '0.25rem 0.5rem',
            fontSize: '0.72rem',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          <option value={0.25}>0.25x Speed</option>
          <option value={0.5}>0.5x Speed</option>
          <option value={1}>1.0x Speed</option>
          <option value={2}>2.0x Speed</option>
          <option value={5}>5.0x Speed</option>
        </select>
      </div>
    </div>
  );
};
