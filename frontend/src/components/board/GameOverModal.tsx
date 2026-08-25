import React from 'react';
import { GameStateDTO } from '../../api/types';
import { Trophy, Award, RotateCcw, Eye, Sparkles } from 'lucide-react';

interface GameOverModalProps {
  gameState: GameStateDTO;
  onRematch?: () => void;
  onClose?: () => void;
  onSwitchToReplay?: () => void;
}

export const GameOverModal: React.FC<GameOverModalProps> = ({
  gameState,
  onRematch,
  onClose,
  onSwitchToReplay,
}) => {
  const players = gameState.players || [];
  if (players.length === 0) return null;

  // Determine ranking
  const sortedPlayers = [...players].sort((a, b) => b.score - a.score);
  const topScore = sortedPlayers[0]?.score ?? 0;
  const isTie = sortedPlayers.length > 1 && sortedPlayers[0]?.score === sortedPlayers[1]?.score;
  const winner = sortedPlayers[0];
  const runnerUp = sortedPlayers[1];
  const scoreMargin = runnerUp ? topScore - runnerUp.score : 0;

  return (
    <div
      className="game-over-modal-backdrop"
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(26, 16, 10, 0.82)',
        backdropFilter: 'blur(6px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 120,
        padding: '1rem',
      }}
    >
      <div
        className="game-over-modal steampunk-panel"
        style={{
          maxWidth: '680px',
          width: '100%',
          background: 'linear-gradient(180deg, #FDF8EE 0%, #F5E8D0 100%)',
          border: '3px solid #C59B27',
          borderRadius: '16px',
          padding: '2rem',
          boxShadow: '0 16px 48px rgba(0, 0, 0, 0.6), inset 0 0 30px rgba(184, 134, 11, 0.15)',
          position: 'relative',
          animation: 'modalPop 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {/* Close / Dismiss Button (Top Right) */}
        {onClose && (
          <button
            onClick={onClose}
            aria-label="Chiudi e visualizza mappa"
            title="Chiudi e visualizza mappa finale"
            style={{
              position: 'absolute',
              top: '1rem',
              right: '1rem',
              background: 'transparent',
              border: 'none',
              fontSize: '1.25rem',
              color: '#785A42',
              cursor: 'pointer',
              padding: '0.25rem 0.5rem',
              borderRadius: '4px',
            }}
          >
            ✕
          </button>
        )}

        {/* Victory Header Banner */}
        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 64,
              height: 64,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #FDE68A 0%, #D97706 50%, #78350F 100%)',
              border: '2.5px solid #FAF5EB',
              boxShadow: '0 4px 16px rgba(184, 134, 11, 0.4), inset 0 2px 4px rgba(255,255,255,0.6)',
              marginBottom: '0.75rem',
            }}
          >
            <Trophy size={32} color="#FFFFFF" />
          </div>

          <div
            style={{
              fontSize: '0.8rem',
              textTransform: 'uppercase',
              letterSpacing: '0.15em',
              fontWeight: 800,
              color: '#9E6B00',
              fontFamily: "'Courier Prime', monospace",
              marginBottom: '0.25rem',
            }}
          >
            ✦ Gran Finale Ferroviario ✦
          </div>

          <h2
            style={{
              margin: '0 0 0.4rem 0',
              fontSize: '1.75rem',
              fontFamily: "'Cinzel Decorative', Georgia, serif",
              fontWeight: 900,
              color: '#23140C',
              textShadow: '0 1px 2px rgba(255,255,255,0.8)',
            }}
          >
            {isTie ? 'Pareggio Trionfale!' : `Vittoria di ${winner.name}!`}
          </h2>

          <p
            style={{
              margin: 0,
              fontSize: '0.95rem',
              color: '#5A3822',
              fontFamily: "'Crimson Pro', Georgia, serif",
              fontStyle: 'italic',
            }}
          >
            {isTie
              ? `Entrambi i conduttori hanno concluso la tratta con ${topScore} punti!`
              : `Partita conclusa al turno ${gameState.turn_number} con un margine di +${scoreMargin} punti!`}
          </p>
        </div>

        {/* Players Final Score Cards Grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${Math.min(players.length, 2)}, 1fr)`,
            gap: '1rem',
            marginBottom: '1.5rem',
          }}
        >
          {players.map((p) => {
            const isWinner = !isTie && p.player_id === winner.player_id;
            const completedTickets = p.tickets.filter((t) => t.completed);
            const incompleteTickets = p.tickets.filter((t) => !t.completed);
            const ticketPointsGained = completedTickets.reduce((acc, t) => acc + t.points, 0);
            const ticketPointsLost = incompleteTickets.reduce((acc, t) => acc + t.points, 0);

            return (
              <div
                key={p.player_id}
                className="steampunk-panel"
                style={{
                  background: isWinner
                    ? 'linear-gradient(180deg, #FFFDF8 0%, #FDF1D6 100%)'
                    : 'linear-gradient(180deg, #FAF5EB 0%, #EFE1C7 100%)',
                  border: isWinner ? '2.5px solid #D97706' : '1.5px solid #C59B27',
                  borderRadius: '12px',
                  padding: '1.15rem',
                  boxShadow: isWinner
                    ? '0 6px 20px rgba(217, 119, 6, 0.25), inset 0 1px 0 rgba(255,255,255,0.9)'
                    : '0 4px 12px rgba(0,0,0,0.1)',
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                {isWinner && (
                  <div
                    style={{
                      position: 'absolute',
                      top: '0.5rem',
                      right: '0.5rem',
                      background: 'linear-gradient(135deg, #FDE68A 0%, #D97706 100%)',
                      color: '#23140C',
                      padding: '0.15rem 0.45rem',
                      borderRadius: '4px',
                      fontSize: '0.65rem',
                      fontFamily: "'Courier Prime', monospace",
                      fontWeight: 900,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.2rem',
                      border: '1px solid #B45309',
                    }}
                  >
                    <Sparkles size={11} /> 1° POSTO
                  </div>
                )}

                {/* Player Identity */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                  <div
                    style={{
                      width: 14,
                      height: 14,
                      borderRadius: '50%',
                      backgroundColor: p.color || '#3B82F6',
                      border: '1.5px solid #23140C',
                    }}
                  />
                  <span
                    style={{
                      fontSize: '1.05rem',
                      fontWeight: 800,
                      fontFamily: "'Playfair Display', Georgia, serif",
                      color: '#23140C',
                    }}
                  >
                    {p.name}
                  </span>
                </div>

                {/* Big Score Readout */}
                <div
                  style={{
                    background: '#FAF5EB',
                    border: '1.5px solid #C59B27',
                    borderRadius: '8px',
                    padding: '0.6rem',
                    textAlign: 'center',
                    marginBottom: '0.85rem',
                  }}
                >
                  <div style={{ fontSize: '0.72rem', color: '#785A42', fontFamily: "'Courier Prime', monospace", textTransform: 'uppercase' }}>
                    Punteggio Finale
                  </div>
                  <div
                    style={{
                      fontSize: '2rem',
                      fontWeight: 900,
                      fontFamily: "'Courier Prime', monospace",
                      color: isWinner ? '#B91C1C' : '#4A2F1D',
                      lineHeight: 1.1,
                    }}
                  >
                    {p.score} <span style={{ fontSize: '1rem', color: '#785A42', fontFamily: "'Crimson Pro', serif" }}>pts</span>
                  </div>
                </div>

                {/* Breakdown List */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.78rem', color: '#5A3822', fontFamily: "'Crimson Pro', Georgia, serif" }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed rgba(184, 134, 11, 0.3)', paddingBottom: '0.2rem' }}>
                    <span>🚂 Tratte Reclamate:</span>
                    <strong style={{ fontFamily: "'Courier Prime', monospace", color: '#23140C' }}>{p.claimed_route_ids.length} tratte</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed rgba(184, 134, 11, 0.3)', paddingBottom: '0.2rem' }}>
                    <span>✅ Biglietti Completati ({completedTickets.length}):</span>
                    <strong style={{ fontFamily: "'Courier Prime', monospace", color: '#15803D' }}>+{ticketPointsGained} pts</strong>
                  </div>
                  {incompleteTickets.length > 0 && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed rgba(184, 134, 11, 0.3)', paddingBottom: '0.2rem' }}>
                      <span>❌ Biglietti Incompleti ({incompleteTickets.length}):</span>
                      <strong style={{ fontFamily: "'Courier Prime', monospace", color: '#B91C1C' }}>-{ticketPointsLost} pts</strong>
                    </div>
                  )}
                  <div style={{ display: 'flex', justifyContent: 'space-between', paddingTop: '0.1rem' }}>
                    <span>🚃 Treni Rimasti:</span>
                    <strong style={{ fontFamily: "'Courier Prime', monospace", color: '#23140C' }}>{p.trains_remaining} / 45</strong>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Action Buttons Row */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          {onRematch && (
            <button
              onClick={onRematch}
              className="steampunk-btn"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                padding: '0.6rem 1.4rem',
                fontSize: '0.9rem',
                background: 'linear-gradient(180deg, #86EFAC 0%, #16A34A 50%, #14532D 100%)',
                color: '#FFFFFF',
                border: '1px solid #14532D',
                boxShadow: '0 4px 12px rgba(22, 163, 74, 0.35)',
              }}
            >
              <RotateCcw size={15} /> Rivincita / Nuova Partita
            </button>
          )}

          {onClose && (
            <button
              onClick={onClose}
              className="steampunk-btn"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                padding: '0.6rem 1.2rem',
                fontSize: '0.9rem',
              }}
            >
              <Eye size={15} /> Esplora Mappa Finale
            </button>
          )}

          {onSwitchToReplay && (
            <button
              onClick={onSwitchToReplay}
              className="steampunk-btn"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                padding: '0.6rem 1.2rem',
                fontSize: '0.9rem',
                background: 'linear-gradient(180deg, #FDE68A 0%, #D97706 100%)',
              }}
            >
              <Award size={15} /> Rivedi Replay
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
