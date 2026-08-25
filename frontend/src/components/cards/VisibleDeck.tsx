import React from 'react';
import { COLOR_HEX } from '../board/mapData';

interface VisibleDeckProps {
  visibleCards: string[];
  deckSize: number;
  discardSize: number;
  ticketsDeckSize: number;
  isHumanTurn?: boolean;
  onDrawVisible?: (slotIndex: number) => void;
  onDrawHidden?: () => void;
  onDrawTickets?: () => void;
}

export const VisibleDeck: React.FC<VisibleDeckProps> = ({
  visibleCards,
  deckSize,
  discardSize,
  ticketsDeckSize,
  isHumanTurn = false,
  onDrawVisible,
  onDrawHidden,
  onDrawTickets,
}) => {
  return (
    <div
      className="visible-deck steampunk-panel"
      style={{
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: '2px solid #C59B27',
        borderRadius: '10px',
        padding: '0.85rem',
        boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
        <h4
          style={{
            margin: 0,
            fontSize: '0.9rem',
            fontFamily: "'Cinzel Decorative', Georgia, serif",
            fontWeight: 800,
            color: '#23140C',
            letterSpacing: '0.03em',
          }}
        >
          🚂 Railway Shares & Decks
        </h4>
        <div
          style={{
            fontSize: '0.75rem',
            color: '#5A3822',
            fontFamily: "'Courier Prime', monospace",
            background: '#EADBBE',
            padding: '0.15rem 0.5rem',
            borderRadius: '4px',
            border: '1px solid #C59B27',
          }}
        >
          Discard: <strong style={{ color: '#23140C' }}>{discardSize}</strong>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(85px, 1fr))', gap: '0.5rem' }}>
        {/* Hidden Train Draw Deck */}
        <button
          className="deck-btn"
          disabled={!isHumanTurn || deckSize === 0}
          onClick={() => onDrawHidden?.()}
          style={{
            background: 'linear-gradient(135deg, #3D2617 0%, #23140C 100%)',
            border: '2px solid #C59B27',
            borderRadius: '8px',
            padding: '0.6rem 0.4rem',
            textAlign: 'center',
            cursor: isHumanTurn && deckSize > 0 ? 'pointer' : 'default',
            color: '#FAF5EB',
            boxShadow: '0 3px 6px rgba(0,0,0,0.3), inset 0 0 10px rgba(0,0,0,0.5)',
            transition: 'all 0.15s ease',
            opacity: deckSize === 0 ? 0.4 : 1,
          }}
        >
          <div style={{ fontSize: '1.2rem', marginBottom: '0.1rem' }}>🂠</div>
          <div style={{ fontSize: '0.75rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Train Stock</div>
          <div style={{ fontSize: '0.7rem', color: '#F6DC88', fontFamily: "'Courier Prime', monospace" }}>{deckSize} left</div>
        </button>

        {/* 5 Visible Face-Up Railway Share Vouchers */}
        {visibleCards.map((color, idx) => {
          const hex = COLOR_HEX[color] || '#7D6A5A';
          const isLight = color === 'WHITE' || color === 'YELLOW';
          return (
            <button
              key={idx}
              disabled={!isHumanTurn}
              onClick={() => onDrawVisible?.(idx)}
              style={{
                backgroundColor: hex,
                color: isLight ? '#23140C' : '#FAF5EB',
                border: '2px solid #FAF5EB',
                outline: '1px solid #4D311E',
                borderRadius: '8px',
                padding: '0.6rem 0.4rem',
                textAlign: 'center',
                cursor: isHumanTurn ? 'pointer' : 'default',
                fontWeight: 800,
                boxShadow: '0 4px 8px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.4)',
                transition: 'all 0.15s ease',
                position: 'relative',
              }}
            >
              <div style={{ fontSize: '0.7rem', opacity: 0.85, fontFamily: "'Courier Prime', monospace" }}>Slot {idx + 1}</div>
              <div style={{ fontSize: '0.82rem', fontFamily: "'Playfair Display', Georgia, serif", letterSpacing: '0.02em' }}>{color}</div>
            </button>
          );
        })}

        {/* Destination Tickets Deck */}
        <button
          className="deck-btn"
          disabled={!isHumanTurn || ticketsDeckSize === 0}
          onClick={() => onDrawTickets?.()}
          style={{
            background: 'linear-gradient(135deg, #7A5028 0%, #4A2E1B 100%)',
            border: '2px dashed #F6DC88',
            borderRadius: '8px',
            padding: '0.6rem 0.4rem',
            textAlign: 'center',
            cursor: isHumanTurn && ticketsDeckSize > 0 ? 'pointer' : 'default',
            color: '#FAF5EB',
            boxShadow: '0 3px 6px rgba(0,0,0,0.3)',
            transition: 'all 0.15s ease',
            opacity: ticketsDeckSize === 0 ? 0.4 : 1,
          }}
        >
          <div style={{ fontSize: '1.2rem', marginBottom: '0.1rem' }}>🎫</div>
          <div style={{ fontSize: '0.75rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Telegrams</div>
          <div style={{ fontSize: '0.7rem', color: '#F6DC88', fontFamily: "'Courier Prime', monospace" }}>{ticketsDeckSize} left</div>
        </button>
      </div>
    </div>
  );
};
