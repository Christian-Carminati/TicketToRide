import React, { useState, useMemo } from 'react';
import { HoveredActionMeta } from '../../context/workbenchTypes';

interface ActionProbabilitiesChartProps {
  probabilities: number[];
  actionMask: boolean[];
  actionLabels: string[];
  rawLogitsOrQ?: number[];
  greedyActionIndex?: number;
  onActionHover?: (meta: HoveredActionMeta | null) => void;
}

export const ActionProbabilitiesChart: React.FC<ActionProbabilitiesChartProps> = ({
  probabilities,
  actionMask,
  actionLabels,
  rawLogitsOrQ,
  greedyActionIndex,
  onActionHover,
}) => {
  const [filterMode, setFilterMode] = useState<'all' | 'valid' | 'top10'>('valid');

  const items = useMemo(() => {
    const list = probabilities.map((prob, idx) => {
      const label = actionLabels[idx] || `Action ${idx}`;

      // Extract routeId if present in label
      let routeId: string | undefined = undefined;
      const match = label.match(/r_\d+_[a-z]+_[a-z]+/i) || label.match(/route_([a-z0-9_]+)/i);
      if (match) {
        routeId = match[0];
      }

      return {
        index: idx,
        label,
        routeId,
        prob: prob,
        isValid: actionMask[idx] ?? true,
        rawScore: rawLogitsOrQ ? rawLogitsOrQ[idx] : undefined,
        isGreedy: idx === greedyActionIndex,
      };
    });

    if (filterMode === 'valid') {
      return list.filter((i) => i.isValid);
    }
    if (filterMode === 'top10') {
      return [...list].sort((a, b) => b.prob - a.prob).slice(0, 10);
    }
    return list;
  }, [probabilities, actionMask, actionLabels, rawLogitsOrQ, greedyActionIndex, filterMode]);

  return (
    <div
      className="action-probabilities-chart steampunk-panel"
      style={{
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: '2px solid #C59B27',
        borderRadius: '10px',
        padding: '0.85rem',
        boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.4rem',
          marginBottom: '0.6rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <span
            style={{
              fontSize: '0.82rem',
              fontWeight: 800,
              color: '#23140C',
              fontFamily: "'Cinzel Decorative', Georgia, serif",
              letterSpacing: '0.04em',
            }}
          >
            Policy Actuators: π(a|s)
          </span>
          <span style={{ fontSize: '0.72rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
            ({items.length} positions)
          </span>
        </div>

        {/* Filter Buttons */}
        <div
          style={{
            display: 'flex',
            gap: '0.2rem',
            background: '#D8C3A0',
            borderRadius: '6px',
            padding: '2px',
            border: '1px solid #A88D75',
          }}
        >
          {(['valid', 'top10', 'all'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setFilterMode(mode)}
              style={{
                background: filterMode === mode
                  ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                  : 'transparent',
                color: filterMode === mode ? '#23140C' : '#5A3822',
                border: filterMode === mode ? '1px solid #6E4E04' : '1px solid transparent',
                borderRadius: '4px',
                padding: '0.15rem 0.5rem',
                fontSize: '0.7rem',
                cursor: 'pointer',
                fontFamily: "'Playfair Display', Georgia, serif",
                fontWeight: filterMode === mode ? 800 : 600,
                textTransform: 'capitalize',
                boxShadow: filterMode === mode ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
              }}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.35rem',
          maxHeight: '260px',
          overflowY: 'auto',
          paddingRight: '0.3rem',
        }}
      >
        {items.map((item) => {
          const pct = (item.prob * 100).toFixed(1);
          return (
            <div
              key={item.index}
              onMouseEnter={() => {
                onActionHover?.({
                  actionIndex: item.index,
                  actionType: item.label,
                  routeId: item.routeId,
                  probability: item.prob,
                  value: item.rawScore,
                  isMasked: !item.isValid,
                });
              }}
              onMouseLeave={() => onActionHover?.(null)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.3rem 0.55rem',
                borderRadius: '6px',
                background: item.isGreedy
                  ? 'linear-gradient(180deg, #FAF3E6 0%, #F5E5C9 100%)'
                  : item.isValid
                  ? '#FAF5EB'
                  : 'rgba(232, 219, 190, 0.4)',
                border: item.isGreedy
                  ? '1.5px solid #B8860B'
                  : '1px solid rgba(184, 134, 11, 0.25)',
                opacity: item.isValid ? 1 : 0.5,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                boxShadow: item.isGreedy ? '0 2px 6px rgba(184, 134, 11, 0.25)' : '0 1px 2px rgba(0,0,0,0.05)',
              }}
            >
              {/* Action Label */}
              <div
                style={{
                  minWidth: '140px',
                  fontSize: '0.78rem',
                  color: '#23140C',
                  fontFamily: "'Crimson Pro', Georgia, serif",
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                }}
              >
                <span style={{ fontSize: '0.75rem' }}>{item.isValid ? (item.isGreedy ? '⭐' : '⚙️') : '🔒'}</span>
                <span style={{ fontWeight: item.isGreedy ? 800 : 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {item.label}
                </span>
              </div>

              {/* Progress Pneumatic Actuator Bar */}
              <div
                style={{
                  flex: 1,
                  height: '10px',
                  backgroundColor: '#D8C3A0',
                  borderRadius: '3px',
                  overflow: 'hidden',
                  position: 'relative',
                  border: '1px solid rgba(74, 47, 29, 0.3)',
                  boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.2)',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: '100%',
                    transform: `scaleX(${item.prob})`,
                    transformOrigin: 'left',
                    background: item.isGreedy
                      ? 'linear-gradient(to right, #F6DC88, #C59B27, #8C6305)'
                      : item.isValid
                      ? 'linear-gradient(to right, #CD7F32, #B85D38)'
                      : '#991B1B',
                    borderRadius: '2px',
                    transition: 'transform 0.2s ease',
                  }}
                />
              </div>

              {/* Numerical Readout */}
              <div
                style={{
                  minWidth: '55px',
                  textAlign: 'right',
                  fontSize: '0.78rem',
                  fontWeight: 800,
                  fontFamily: "'Courier Prime', 'JetBrains Mono', monospace",
                  color: item.isGreedy ? '#9E6B00' : item.isValid ? '#23140C' : '#991B1B',
                }}
              >
                {item.isValid ? `${pct}%` : 'LOCKED'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
