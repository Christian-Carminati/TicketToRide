import React from 'react';
import { PlayerStateDTO } from '../../api/types';
import { COLOR_HEX } from '../board/mapData';

interface TrainCardHandProps {
  player: PlayerStateDTO;
  isCurrentTurn?: boolean;
}

export const TrainCardHand: React.FC<TrainCardHandProps> = ({
  player,
  isCurrentTurn = false,
}) => {
  const totalCards = Object.values(player.cards_in_hand).reduce((a, b) => a + b, 0);

  return (
    <div
      className={`player-card-panel steampunk-panel ${isCurrentTurn ? 'active-turn' : ''}`}
      style={{
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: `2.5px solid ${isCurrentTurn ? '#B8860B' : '#C59B27'}`,
        borderRadius: '10px',
        padding: '0.85rem',
        boxShadow: isCurrentTurn
          ? '0 0 16px rgba(184, 134, 11, 0.4), 0 4px 12px rgba(0,0,0,0.25)'
          : '0 4px 12px rgba(0,0,0,0.15)',
        transition: 'all 0.3s ease',
      }}
    >
      {/* Player Identity Strip */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div
            style={{
              width: 14,
              height: 14,
              borderRadius: '50%',
              backgroundColor: player.color,
              border: '1.5px solid #23140C',
              boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
            }}
          />
          <h4
            style={{
              margin: 0,
              fontSize: '0.95rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: 800,
              color: '#23140C',
            }}
          >
            {player.name} {isCurrentTurn && <span style={{ fontSize: '0.75rem', color: '#9E6B00', fontStyle: 'italic' }}>(Current Turn)</span>}
          </h4>
        </div>
        <div
          style={{
            fontSize: '1.15rem',
            fontWeight: 800,
            fontFamily: "'Courier Prime', monospace",
            color: '#B91C1C',
            background: '#FAF5EB',
            padding: '0.1rem 0.5rem',
            borderRadius: '4px',
            border: '1px solid #C59B27',
          }}
        >
          {player.score} <span style={{ fontSize: '0.75rem', color: '#785A42', fontFamily: "'Crimson Pro', serif" }}>pts</span>
        </div>
      </div>

      {/* Inventory Counters */}
      <div
        style={{
          display: 'flex',
          gap: '1rem',
          marginBottom: '0.65rem',
          fontSize: '0.8rem',
          color: '#5A3822',
          fontFamily: "'Crimson Pro', Georgia, serif",
          borderBottom: '1px solid rgba(184, 134, 11, 0.25)',
          paddingBottom: '0.4rem',
        }}
      >
        <div>🚂 Trains: <strong style={{ color: '#23140C', fontFamily: "'Courier Prime', monospace" }}>{player.trains_remaining}</strong></div>
        <div>🃏 Shares: <strong style={{ color: '#23140C', fontFamily: "'Courier Prime', monospace" }}>{totalCards}</strong></div>
        <div>🎫 Telegrams: <strong style={{ color: '#23140C', fontFamily: "'Courier Prime', monospace" }}>{player.tickets.length}</strong></div>
      </div>

      {/* Cards Share Badges Grid */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
        {Object.entries(player.cards_in_hand).map(([color, count]) => {
          if (count === 0) return null;
          const hex = COLOR_HEX[color] || '#7D6A5A';
          const isLight = color === 'WHITE' || color === 'YELLOW';
          return (
            <div
              key={color}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                padding: '0.2rem 0.55rem',
                borderRadius: '6px',
                backgroundColor: hex,
                color: isLight ? '#23140C' : '#FAF5EB',
                fontSize: '0.75rem',
                fontWeight: 800,
                fontFamily: "'Playfair Display', Georgia, serif",
                border: '1.5px solid #FAF5EB',
                boxShadow: '0 2px 4px rgba(0,0,0,0.25)',
              }}
            >
              <span>{color}</span>
              <span
                style={{
                  background: isLight ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.25)',
                  padding: '0.1rem 0.35rem',
                  borderRadius: '4px',
                  fontSize: '0.72rem',
                  fontFamily: "'Courier Prime', monospace",
                  fontWeight: 900,
                }}
              >
                {count}
              </span>
            </div>
          );
        })}
        {totalCards === 0 && (
          <span style={{ fontSize: '0.78rem', color: '#785A42', fontStyle: 'italic', fontFamily: "'Crimson Pro', serif" }}>
            No stock shares held in hand.
          </span>
        )}
      </div>
    </div>
  );
};
