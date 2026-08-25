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

export const ScrubberTransportBar: React.FC<ScrubberTransportBarProps> = React.memo(({
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
      className="scrubber-transport-bar steampunk-panel"
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.5rem 1.25rem',
        gap: '1rem',
        flexWrap: 'wrap',
        border: '2px solid #C59B27',
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        boxShadow: '0 4px 16px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.7)',
      }}
    >
      {/* Left: Turn & Step Chronometer Badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div
          style={{
            background: 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)',
            border: '1.5px solid #6E4E04',
            borderRadius: '6px',
            padding: '0.25rem 0.65rem',
            fontSize: '0.8rem',
            color: '#23140C',
            fontWeight: 800,
            fontFamily: "'Cinzel Decorative', Georgia, serif",
            letterSpacing: '0.05em',
            boxShadow: '0 2px 4px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.6)',
            textShadow: '0 1px 0 rgba(255,255,255,0.4)',
          }}
        >
          TURN {currentTurn}
        </div>
        <div style={{ fontSize: '0.82rem', color: '#4A2F1D', fontFamily: "'Crimson Pro', Georgia, serif" }}>
          Step: <strong style={{ color: '#23140C', fontFamily: "'Courier Prime', monospace" }}>{currentStepIndex + 1}</strong> / {Math.max(1, maxStepIndex + 1)}
        </div>
      </div>

      {/* Center: Machined Brass Playback Buttons & Throttle Slider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: 1, maxWidth: '650px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <button
            onClick={() => setStepIndex(0)}
            disabled={currentStepIndex === 0}
            className="steampunk-btn"
            style={{ padding: '0.3rem 0.5rem', display: 'flex', alignItems: 'center' }}
            title="First Step"
          >
            <SkipBack size={13} />
          </button>

          <button
            onClick={() => setStepIndex(Math.max(0, currentStepIndex - 1))}
            disabled={currentStepIndex === 0}
            className="steampunk-btn"
            style={{ padding: '0.3rem 0.5rem', display: 'flex', alignItems: 'center' }}
            title="Step Backward"
          >
            <ChevronLeft size={13} />
          </button>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="steampunk-btn"
            style={{
              padding: '0.35rem 0.9rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              fontSize: '0.82rem',
              background: isPlaying
                ? 'linear-gradient(180deg, #F87171 0%, #DC2626 50%, #991B1B 100%)'
                : 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)',
              color: isPlaying ? '#FFFFFF' : '#23140C',
              border: isPlaying ? '1px solid #7F1D1D' : '1px solid #6E4E04',
            }}
          >
            {isPlaying ? <Pause size={13} /> : <Play size={13} />}
            <span>{isPlaying ? 'Pause' : 'Play'}</span>
          </button>

          <button
            onClick={() => setStepIndex(Math.min(maxStepIndex, currentStepIndex + 1))}
            disabled={currentStepIndex >= maxStepIndex}
            className="steampunk-btn"
            style={{ padding: '0.3rem 0.5rem', display: 'flex', alignItems: 'center' }}
            title="Step Forward"
          >
            <ChevronRight size={13} />
          </button>

          <button
            onClick={() => setStepIndex(maxStepIndex)}
            disabled={currentStepIndex >= maxStepIndex}
            className="steampunk-btn"
            style={{ padding: '0.3rem 0.5rem', display: 'flex', alignItems: 'center' }}
            title="Last Step"
          >
            <SkipForward size={13} />
          </button>
        </div>

        {/* Brass Throttle Range Slider */}
        <input
          type="range"
          min={0}
          max={Math.max(1, maxStepIndex)}
          value={currentStepIndex}
          onChange={(e) => setStepIndex(Number(e.target.value))}
          style={{
            flex: 1,
            accentColor: '#B8860B',
            cursor: 'pointer',
            height: '8px',
          }}
        />
      </div>

      {/* Right: Steam Injection Bot Step & Tachometer Speed */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
        {onBotStep && (
          <button
            onClick={onBotStep}
            disabled={isStepping || isPlaying}
            className="steampunk-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              padding: '0.35rem 0.85rem',
              background: 'linear-gradient(180deg, #86EFAC 0%, #16A34A 50%, #14532D 100%)',
              color: '#FFFFFF',
              border: '1px solid #14532D',
              textShadow: '0 1px 2px rgba(0,0,0,0.5)',
            }}
          >
            <Zap size={13} color="#FEF08A" />
            <span>{isStepping ? 'Injecting...' : 'Bot Step'}</span>
          </button>
        )}

        {/* Speed Dial Dropdown */}
        <select
          value={playbackSpeed}
          onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
          style={{
            background: 'linear-gradient(180deg, #FAF4E6 0%, #E8D7BC 100%)',
            color: '#23140C',
            border: '1.5px solid #8C6305',
            borderRadius: '6px',
            padding: '0.3rem 0.6rem',
            fontSize: '0.78rem',
            fontFamily: "'Courier Prime', monospace",
            fontWeight: 700,
            cursor: 'pointer',
            boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.1)',
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
});
