import React from 'react';
import { Target, ShieldAlert } from 'lucide-react';
import { BayesianBeliefDTO } from '../../api/types';

interface BayesianBeliefRadarProps {
  beliefs?: BayesianBeliefDTO[];
  opponentName?: string;
}

export const BayesianBeliefRadar: React.FC<BayesianBeliefRadarProps> = ({
  beliefs = [],
  opponentName = 'Opponent',
}) => {
  if (beliefs.length === 0) {
    return null;
  }

  const getThreatColor = (level: string) => {
    switch (level) {
      case 'critical':
        return { bg: 'rgba(185, 28, 28, 0.15)', border: '#B91C1C', text: '#991B1B', badge: '#B91C1C' };
      case 'high':
        return { bg: 'rgba(217, 119, 6, 0.15)', border: '#D97706', text: '#92400E', badge: '#D97706' };
      case 'moderate':
        return { bg: 'rgba(197, 155, 39, 0.15)', border: '#C59B27', text: '#785A42', badge: '#C59B27' };
      default:
        return { bg: 'rgba(120, 90, 66, 0.1)', border: '#A88D75', text: '#5A3822', badge: '#785A42' };
    }
  };

  return (
    <div
      className="bayesian-belief-radar steampunk-panel"
      style={{
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: '2px solid #C59B27',
        borderRadius: '10px',
        padding: '0.75rem 0.9rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.6rem',
        boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Target size={16} color="#B91C1C" />
          <span
            style={{
              fontSize: '0.82rem',
              fontWeight: 800,
              color: '#23140C',
              fontFamily: "'Playfair Display', Georgia, serif",
              letterSpacing: '0.02em',
            }}
          >
            Bayesian Opponent Belief Radar
          </span>
        </div>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.3rem',
            background: 'rgba(185, 28, 28, 0.15)',
            border: '1px solid #B91C1C',
            padding: '0.15rem 0.45rem',
            borderRadius: '4px',
            fontSize: '0.7rem',
            fontFamily: "'Courier Prime', monospace",
            fontWeight: 700,
            color: '#B91C1C',
          }}
        >
          <ShieldAlert size={12} />
          <span>Tracking: {opponentName}</span>
        </div>
      </div>

      <div style={{ fontSize: '0.74rem', color: '#5A3822', fontFamily: "'Crimson Pro', Georgia, serif", fontStyle: 'italic' }}>
        Calculates posterior destination probabilities via Bayesian detour graph analysis.
      </div>

      {/* Belief Table / Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
        {beliefs.slice(0, 5).map((b) => {
          const colors = getThreatColor(b.threat_level);
          const pct = (b.probability * 100).toFixed(0);

          return (
            <div
              key={b.ticket_id}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: colors.bg,
                border: `1px solid ${colors.border}`,
                borderRadius: '6px',
                padding: '0.35rem 0.6rem',
              }}
            >
              {/* City Pair */}
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span
                  style={{
                    fontSize: '0.78rem',
                    fontWeight: 800,
                    color: '#23140C',
                    fontFamily: "'Playfair Display', Georgia, serif",
                  }}
                >
                  {b.city_a} ⟷ {b.city_b}
                </span>
                <span style={{ fontSize: '0.68rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
                  Value: {b.points} pts
                </span>
              </div>

              {/* Probability & Threat Badge */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div
                  style={{
                    fontSize: '0.85rem',
                    fontWeight: 900,
                    fontFamily: "'Courier Prime', monospace",
                    color: colors.text,
                  }}
                >
                  {pct}%
                </div>
                <span
                  style={{
                    fontSize: '0.65rem',
                    fontWeight: 800,
                    textTransform: 'uppercase',
                    padding: '0.1rem 0.35rem',
                    borderRadius: '3px',
                    background: colors.badge,
                    color: '#FFFFFF',
                    letterSpacing: '0.05em',
                  }}
                >
                  {b.threat_level}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
