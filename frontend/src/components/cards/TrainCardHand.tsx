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
      className={`player-card-panel ${isCurrentTurn ? 'active-turn' : ''}`}
      style={{
        background: 'rgba(15, 23, 42, 0.85)',
        border: `2px solid ${isCurrentTurn ? player.color : 'rgba(255,255,255,0.1)'}`,
        borderRadius: '10px',
        padding: '1rem',
        boxShadow: isCurrentTurn ? `0 0 16px ${player.color}40` : 'none',
        transition: 'all 0.3s ease',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div
            style={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              backgroundColor: player.color,
            }}
          />
          <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#F1F5F9' }}>
            {player.name} {isCurrentTurn && <span style={{ fontSize: '0.75rem', color: player.color }}>(Current Turn)</span>}
          </h4>
        </div>
        <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#F59E0B' }}>
          {player.score} <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>pts</span>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.75rem', fontSize: '0.85rem', color: '#94A3B8' }}>
        <div>🚂 Trains: <strong style={{ color: '#F1F5F9' }}>{player.trains_remaining}</strong></div>
        <div>🃏 Cards: <strong style={{ color: '#F1F5F9' }}>{totalCards}</strong></div>
        <div>🎫 Tickets: <strong style={{ color: '#F1F5F9' }}>{player.tickets.length}</strong></div>
      </div>

      {/* Cards Badges Grid */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
        {Object.entries(player.cards_in_hand).map(([color, count]) => {
          if (count === 0) return null;
          const hex = COLOR_HEX[color] || '#64748B';
          const isLight = color === 'WHITE' || color === 'YELLOW';
          return (
            <div
              key={color}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.3rem',
                padding: '0.2rem 0.5rem',
                borderRadius: '6px',
                backgroundColor: hex,
                color: isLight ? '#0F172A' : '#FFFFFF',
                fontSize: '0.75rem',
                fontWeight: 700,
                boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
              }}
            >
              <span>{color}</span>
              <span
                style={{
                  background: isLight ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.25)',
                  padding: '0.1rem 0.3rem',
                  borderRadius: '4px',
                  fontSize: '0.7rem',
                }}
              >
                {count}
              </span>
            </div>
          );
        })}
        {totalCards === 0 && (
          <span style={{ fontSize: '0.75rem', color: '#64748B', fontStyle: 'italic' }}>No cards in hand</span>
        )}
      </div>
    </div>
  );
};
