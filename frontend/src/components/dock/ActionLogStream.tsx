import React from 'react';
import { useWorkbench } from '../../context';
import { ListFilter, PlayCircle } from 'lucide-react';

export const ActionLogStream: React.FC = () => {
  const { state, setStepIndex } = useWorkbench();
  const frames = state.replayData?.frames || [];

  return (
    <div
      className="action-log-stream"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem',
        padding: '0.5rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <ListFilter size={16} color="#9E6B00" />
          <span
            style={{
              fontSize: '0.82rem',
              fontWeight: 800,
              color: '#23140C',
              fontFamily: "'Cinzel Decorative', Georgia, serif",
              letterSpacing: '0.04em',
            }}
          >
            Telegraphic Action Ticker & Log
          </span>
        </div>
        <div style={{ fontSize: '0.75rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
          {frames.length > 0 ? `${frames.length} steps recorded in telegraph buffer` : 'Live turn stream active'}
        </div>
      </div>

      {/* Frame List - Telegraph Ticker Tape */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.35rem',
          maxHeight: '160px',
          overflowY: 'auto',
          paddingRight: '0.4rem',
        }}
      >
        {frames.length > 0 ? (
          frames.map((frame) => {
            const isSelected = frame.step_index === state.currentStepIndex;
            const playerColor = frame.player_index === 0 ? '#B8860B' : '#CD7F32';

            return (
              <div
                key={frame.step_index}
                onClick={() => setStepIndex(frame.step_index)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.35rem 0.65rem',
                  borderRadius: '6px',
                  background: isSelected
                    ? 'linear-gradient(180deg, #FBF6ED 0%, #F5E5C9 100%)'
                    : '#FAF5EB',
                  border: isSelected ? '1.5px solid #B8860B' : '1px solid rgba(184, 134, 11, 0.25)',
                  cursor: 'pointer',
                  fontSize: '0.78rem',
                  transition: 'all 0.15s ease',
                  boxShadow: isSelected ? '0 2px 6px rgba(184, 134, 11, 0.25)' : '0 1px 2px rgba(0,0,0,0.05)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ fontFamily: "'Courier Prime', monospace", color: '#785A42', width: 32, fontWeight: 700 }}>
                    #{frame.step_index + 1}
                  </span>
                  <span
                    style={{
                      background: playerColor,
                      color: '#FAF5EB',
                      padding: '0.1rem 0.4rem',
                      borderRadius: '4px',
                      fontWeight: 800,
                      fontSize: '0.72rem',
                      fontFamily: "'Courier Prime', monospace",
                    }}
                  >
                    P{frame.player_index}
                  </span>
                  <span style={{ color: '#23140C', fontWeight: 700, fontFamily: "'Playfair Display', Georgia, serif" }}>
                    {frame.action.action_type}
                    {frame.action.route_id ? ` [${frame.action.route_id}]` : ''}
                    {frame.action.card_color ? ` (${frame.action.card_color})` : ''}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: "'Courier Prime', monospace", fontWeight: 700 }}>
                  <span style={{ color: frame.reward >= 0 ? '#15803D' : '#B91C1C' }}>
                    r: {frame.reward >= 0 ? `+${frame.reward}` : frame.reward}
                  </span>
                  {isSelected && <PlayCircle size={14} color="#9E6B00" />}
                </div>
              </div>
            );
          })
        ) : (
          <div style={{ textAlign: 'center', padding: '1.5rem', color: '#785A42', fontSize: '0.78rem', fontFamily: "'Crimson Pro', serif", fontStyle: 'italic' }}>
            No recorded actions in ticker buffer. Actions dispatched in live games or replays will print here in chronological telegram sequence.
          </div>
        )}
      </div>
    </div>
  );
};
