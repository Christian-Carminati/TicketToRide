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
      style={{
        background: 'rgba(15, 23, 42, 0.85)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '10px',
        padding: '1rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: '#F1F5F9' }}>
          🃏 Cards & Decks
        </h4>
        <div style={{ fontSize: '0.8rem', color: '#94A3B8' }}>
          Discard: <strong style={{ color: '#F1F5F9' }}>{discardSize}</strong>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(80px, 1fr))', gap: '0.5rem' }}>
        {/* Hidden Draw Deck */}
        <button
          className="deck-btn"
          disabled={!isHumanTurn || deckSize === 0}
          onClick={() => onDrawHidden?.()}
          style={{
            background: 'linear-gradient(135deg, #1E293B 0%, #0F172A 100%)',
            border: '2px dashed rgba(255,255,255,0.2)',
            borderRadius: '8px',
            padding: '0.6rem 0.4rem',
            textAlign: 'center',
            cursor: isHumanTurn ? 'pointer' : 'default',
            color: '#F8FAFC',
            transition: 'all 0.2s',
          }}
        >
          <div style={{ fontSize: '1.2rem' }}>🂠</div>
          <div style={{ fontSize: '0.75rem', fontWeight: 600 }}>Train Deck</div>
          <div style={{ fontSize: '0.7rem', color: '#38BDF8' }}>{deckSize} left</div>
        </button>

        {/* 5 Visible Cards */}
        {visibleCards.map((color, idx) => {
          const hex = COLOR_HEX[color] || '#64748B';
          const isLight = color === 'WHITE' || color === 'YELLOW';
          return (
            <button
              key={idx}
              disabled={!isHumanTurn}
              onClick={() => onDrawVisible?.(idx)}
              style={{
                backgroundColor: hex,
                color: isLight ? '#0F172A' : '#FFFFFF',
                border: '2px solid rgba(255,255,255,0.2)',
                borderRadius: '8px',
                padding: '0.6rem 0.4rem',
                textAlign: 'center',
                cursor: isHumanTurn ? 'pointer' : 'default',
                fontWeight: 700,
                boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
                transition: 'all 0.2s',
              }}
            >
              <div style={{ fontSize: '0.8rem' }}>Slot {idx + 1}</div>
              <div style={{ fontSize: '0.85rem' }}>{color}</div>
            </button>
          );
        })}

        {/* Tickets Deck */}
        <button
          className="deck-btn"
          disabled={!isHumanTurn || ticketsDeckSize === 0}
          onClick={() => onDrawTickets?.()}
          style={{
            background: 'linear-gradient(135deg, #312E81 0%, #1E1B4B 100%)',
            border: '2px dashed rgba(165,180,252,0.4)',
            borderRadius: '8px',
            padding: '0.6rem 0.4rem',
            textAlign: 'center',
            cursor: isHumanTurn ? 'pointer' : 'default',
            color: '#F8FAFC',
            transition: 'all 0.2s',
          }}
        >
          <div style={{ fontSize: '1.2rem' }}>🎫</div>
          <div style={{ fontSize: '0.75rem', fontWeight: 600 }}>Tickets</div>
          <div style={{ fontSize: '0.7rem', color: '#A5B4FC' }}>{ticketsDeckSize} left</div>
        </button>
      </div>
    </div>
  );
};
