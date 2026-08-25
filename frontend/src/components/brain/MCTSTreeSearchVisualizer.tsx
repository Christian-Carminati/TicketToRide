import React from 'react';
import { GitFork, Network, Zap } from 'lucide-react';

interface MCTSTreeSearchVisualizerProps {
  totalSimulations?: number;
  priors?: number[];
  visits?: number[];
  qValues?: number[];
  actionLabels?: string[];
  actionMask?: boolean[];
  greedyActionIndex?: number;
}

export const MCTSTreeSearchVisualizer: React.FC<MCTSTreeSearchVisualizerProps> = ({
  totalSimulations = 40,
  priors = [],
  visits = [],
  actionLabels = [],
  actionMask = [],
  greedyActionIndex,
}) => {
  const maxVisits = Math.max(1, ...visits);

  // Filter to legal actions with non-zero visits or top priors
  const topActions = actionLabels
    .map((label, idx) => ({
      index: idx,
      label,
      prior: priors[idx] ?? 0,
      visits: visits[idx] ?? 0,
      isValid: actionMask[idx] ?? true,
      isGreedy: idx === greedyActionIndex,
    }))
    .filter((a) => a.isValid && (a.visits > 0 || a.prior > 0.02))
    .sort((a, b) => b.visits - a.visits || b.prior - a.prior)
    .slice(0, 7);

  return (
    <div
      className="mcts-tree-visualizer steampunk-panel"
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
      {/* Header Plaque */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Network size={16} color="#9E6B00" />
          <span
            style={{
              fontSize: '0.82rem',
              fontWeight: 800,
              color: '#23140C',
              fontFamily: "'Playfair Display', Georgia, serif",
              letterSpacing: '0.02em',
            }}
          >
            MCTS Tree Search & PUCT Radar
          </span>
        </div>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.3rem',
            background: '#FAF4E6',
            border: '1px solid #C59B27',
            padding: '0.2rem 0.5rem',
            borderRadius: '4px',
            fontSize: '0.72rem',
            fontFamily: "'Courier Prime', monospace",
            fontWeight: 700,
            color: '#785A42',
          }}
        >
          <GitFork size={12} color="#C59B27" />
          <span>{totalSimulations} Simulations / Turn</span>
        </div>
      </div>

      <div style={{ fontSize: '0.74rem', color: '#5A3822', fontFamily: "'Crimson Pro', Georgia, serif", fontStyle: 'italic' }}>
        Polynomial Upper Confidence Trees (PUCT) balances Neural Prior $P(s,a)$ against Monte Carlo visit counts $N(s,a)$.
      </div>

      {/* Action Simulation Rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
        {topActions.map((item) => {
          const priorPct = (item.prior * 100).toFixed(1);

          return (
            <div
              key={item.index}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                background: item.isGreedy ? 'rgba(197, 155, 39, 0.15)' : 'rgba(240, 230, 210, 0.5)',
                border: item.isGreedy ? '1px solid #C59B27' : '1px solid rgba(140, 99, 5, 0.2)',
                borderRadius: '6px',
                padding: '0.3rem 0.55rem',
              }}
            >
              {/* Action Name */}
              <div
                style={{
                  minWidth: '110px',
                  fontSize: '0.75rem',
                  fontWeight: item.isGreedy ? 800 : 600,
                  color: item.isGreedy ? '#8C6305' : '#23140C',
                  fontFamily: "'Playfair Display', Georgia, serif",
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.25rem',
                }}
              >
                {item.isGreedy && <Zap size={11} color="#C59B27" />}
                <span title={item.label}>
                  {item.label.length > 18 ? `${item.label.slice(0, 18)}...` : item.label}
                </span>
              </div>

              {/* Dual Bar: Amber (Prior) + Green/Brass (Visits) */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {/* Visit Count Bar */}
                <div
                  style={{
                    height: '8px',
                    background: '#D8C3A0',
                    borderRadius: '3px',
                    overflow: 'hidden',
                    position: 'relative',
                    border: '1px solid rgba(74, 47, 29, 0.25)',
                    boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.15)',
                  }}
                >
                  <div
                    style={{
                      height: '100%',
                      width: '100%',
                      transform: `scaleX(${item.visits / maxVisits})`,
                      transformOrigin: 'left',
                      background: 'linear-gradient(90deg, #15803D 0%, #C59B27 100%)',
                      borderRadius: '2px',
                      transition: 'transform 0.2s ease',
                    }}
                  />
                </div>
              </div>

              {/* Numerical Metrics */}
              <div
                style={{
                  display: 'flex',
                  gap: '0.45rem',
                  fontSize: '0.74rem',
                  fontFamily: "'Courier Prime', monospace",
                  fontWeight: 700,
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                <span style={{ color: '#15803D' }} title="MCTS Visit Count">
                  N={item.visits}
                </span>
                <span style={{ color: '#CD7F32' }} title="Neural Policy Prior">
                  P={priorPct}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

