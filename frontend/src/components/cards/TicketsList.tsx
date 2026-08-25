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
      className="tickets-list steampunk-panel"
      style={{
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: '2px solid #C59B27',
        borderRadius: '10px',
        padding: '0.85rem',
        boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
      }}
    >
      <h4
        style={{
          margin: '0 0 0.65rem 0',
          fontSize: '0.9rem',
          fontFamily: "'Cinzel Decorative', Georgia, serif",
          fontWeight: 800,
          color: '#23140C',
          letterSpacing: '0.03em',
        }}
      >
        🎫 {playerName}'s Destination Telegrams ({tickets.length})
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
              padding: '0.45rem 0.65rem',
              borderRadius: '6px',
              backgroundColor: '#FAF5EB',
              border: '1px solid rgba(184, 134, 11, 0.35)',
              fontSize: '0.82rem',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
              boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span>{t.completed ? '✅' : '📜'}</span>
              <span
                style={{
                  color: '#23140C',
                  fontFamily: "'Playfair Display', Georgia, serif",
                  fontWeight: 700,
                }}
              >
                {t.city_a} ⟷ {t.city_b}
              </span>
            </div>
            <div
              style={{
                fontWeight: 900,
                color: '#FAF5EB',
                background: 'linear-gradient(135deg, #B91C1C 0%, #7F1D1D 100%)',
                padding: '0.15rem 0.5rem',
                borderRadius: '4px',
                fontFamily: "'Courier Prime', monospace",
                border: '1px solid #7F1D1D',
                boxShadow: '0 1px 3px rgba(185, 28, 28, 0.3)',
              }}
            >
              +{t.points}
            </div>
          </div>
        ))}
        {tickets.length === 0 && (
          <div style={{ fontSize: '0.78rem', color: '#785A42', fontStyle: 'italic', padding: '0.4rem 0', fontFamily: "'Crimson Pro', serif" }}>
            No destination vouchers held.
          </div>
        )}
      </div>
    </div>
  );
};
