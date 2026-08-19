import React from 'react';

interface TicketItem {
  id: string;
  city_a: string;
  city_b: string;
  points: number;
  completed?: boolean;
}

interface TicketsListProps {
  tickets: TicketItem[];
  playerName?: string;
  onCityHighlight?: (cities: string[]) => void;
}

export const TicketsList: React.FC<TicketsListProps> = ({
  tickets,
  playerName = 'Player',
  onCityHighlight,
}) => {
  return (
    <div
      style={{
        background: 'rgba(15, 23, 42, 0.85)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '10px',
        padding: '1rem',
      }}
    >
      <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.95rem', fontWeight: 600, color: '#F1F5F9' }}>
        🎫 {playerName}'s Destination Tickets ({tickets.length})
      </h4>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '180px', overflowY: 'auto' }}>
        {tickets.map((t) => (
          <div
            key={t.id}
            onMouseEnter={() => onCityHighlight?.([t.city_a, t.city_b])}
            onMouseLeave={() => onCityHighlight?.([])}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '0.4rem 0.6rem',
              borderRadius: '6px',
              backgroundColor: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(255,255,255,0.05)',
              fontSize: '0.8rem',
              cursor: 'pointer',
              transition: 'background 0.2s',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span>{t.completed ? '✅' : '📍'}</span>
              <span style={{ color: '#F1F5F9', fontWeight: 500 }}>
                {t.city_a} ⟷ {t.city_b}
              </span>
            </div>
            <div
              style={{
                fontWeight: 700,
                color: '#F59E0B',
                background: 'rgba(245, 158, 11, 0.15)',
                padding: '0.1rem 0.4rem',
                borderRadius: '4px',
              }}
            >
              +{t.points}
            </div>
          </div>
        ))}
        {tickets.length === 0 && (
          <div style={{ fontSize: '0.8rem', color: '#64748B', fontStyle: 'italic', padding: '0.5rem 0' }}>
            No destination tickets held.
          </div>
        )}
      </div>
    </div>
  );
};
