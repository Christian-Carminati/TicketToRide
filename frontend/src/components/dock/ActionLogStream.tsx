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
          <ListFilter size={16} color="#38BDF8" />
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#F1F5F9', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Action Trajectory & Event Log
          </span>
        </div>
        <div style={{ fontSize: '0.72rem', color: '#94A3B8' }}>
          {frames.length > 0 ? `${frames.length} total steps recorded` : 'Live turn stream'}
        </div>
      </div>

      {/* Frame List */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.3rem',
          maxHeight: '160px',
          overflowY: 'auto',
          paddingRight: '0.4rem',
        }}
      >
        {frames.length > 0 ? (
          frames.map((frame) => {
            const isSelected = frame.step_index === state.currentStepIndex;
            const playerColor = frame.player_index === 0 ? '#38BDF8' : '#818CF8';

            return (
              <div
                key={frame.step_index}
                onClick={() => setStepIndex(frame.step_index)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.35rem 0.6rem',
                  borderRadius: '6px',
                  background: isSelected ? 'rgba(59, 130, 246, 0.2)' : 'rgba(30, 41, 59, 0.4)',
                  border: isSelected ? '1px solid #38BDF8' : '1px solid rgba(255,255,255,0.04)',
                  cursor: 'pointer',
                  fontSize: '0.75rem',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ fontFamily: 'monospace', color: '#64748B', width: 28 }}>
                    #{frame.step_index + 1}
                  </span>
                  <span
                    style={{
                      background: playerColor,
                      color: '#0F172A',
                      padding: '0.1rem 0.35rem',
                      borderRadius: '4px',
                      fontWeight: 700,
                      fontSize: '0.7rem',
                    }}
                  >
                    P{frame.player_index}
                  </span>
                  <span style={{ color: '#F1F5F9', fontWeight: 600 }}>
                    {frame.action.action_type}
                    {frame.action.route_id ? ` [${frame.action.route_id}]` : ''}
                    {frame.action.card_color ? ` (${frame.action.card_color})` : ''}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'monospace' }}>
                  <span style={{ color: frame.reward >= 0 ? '#34D399' : '#F43F5E' }}>
                    r: {frame.reward >= 0 ? `+${frame.reward}` : frame.reward}
                  </span>
                  {isSelected && <PlayCircle size={12} color="#38BDF8" />}
                </div>
              </div>
            );
          })
        ) : (
          <div style={{ textAlign: 'center', padding: '1.5rem', color: '#64748B', fontSize: '0.75rem' }}>
            No recorded actions in buffer. Actions taken in live games or replays will appear here in chronological sequence.
          </div>
        )}
      </div>
    </div>
  );
};
