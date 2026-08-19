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

      // Extract routeId if present in label (e.g., "Claim Route r_0_bos_ny" or "Claim r_0_bos_ny")
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
      className="action-probabilities-chart"
      style={{
        background: 'rgba(15, 23, 42, 0.85)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '10px',
        padding: '0.85rem',
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
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#F1F5F9', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Policy Head: π(a|s)
          </span>
          <span style={{ fontSize: '0.7rem', color: '#64748B' }}>
            ({items.length} actions)
          </span>
        </div>

        {/* Filter Buttons */}
        <div style={{ display: 'flex', gap: '0.25rem', background: 'rgba(30, 41, 59, 0.6)', borderRadius: '6px', padding: '2px' }}>
          {(['valid', 'top10', 'all'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setFilterMode(mode)}
              style={{
                background: filterMode === mode ? '#3B82F6' : 'transparent',
                color: filterMode === mode ? '#FFFFFF' : '#94A3B8',
                border: 'none',
                borderRadius: '4px',
                padding: '0.15rem 0.45rem',
                fontSize: '0.7rem',
                cursor: 'pointer',
                fontWeight: filterMode === mode ? 700 : 500,
                textTransform: 'capitalize',
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
                padding: '0.3rem 0.5rem',
                borderRadius: '6px',
                background: item.isGreedy
                  ? 'rgba(16, 185, 129, 0.15)'
                  : item.isValid
                  ? 'rgba(30, 41, 59, 0.5)'
                  : 'rgba(15, 23, 42, 0.4)',
                border: item.isGreedy
                  ? '1px solid #10B981'
                  : '1px solid rgba(255,255,255,0.05)',
                opacity: item.isValid ? 1 : 0.45,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {/* Action Label */}
              <div style={{ minWidth: '140px', fontSize: '0.75rem', color: '#F1F5F9', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <span style={{ fontSize: '0.7rem' }}>{item.isValid ? (item.isGreedy ? '⭐' : '✓') : '🔒'}</span>
                <span style={{ fontWeight: item.isGreedy ? 700 : 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {item.label}
                </span>
              </div>

              {/* Progress Probability Bar */}
              <div
                style={{
                  flex: 1,
                  height: '10px',
                  backgroundColor: 'rgba(15, 23, 42, 0.8)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                  position: 'relative',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: '100%',
                    transform: `scaleX(${item.prob})`,
                    transformOrigin: 'left',
                    background: item.isGreedy
                      ? 'linear-gradient(to right, #10B981, #34D399)'
                      : item.isValid
                      ? 'linear-gradient(to right, #3B82F6, #38BDF8)'
                      : '#F43F5E',
                    borderRadius: '3px',
                    transition: 'transform 0.2s ease',
                  }}
                />
              </div>

              {/* Numerical Value / Mask Badge */}
              <div
                style={{
                  minWidth: '50px',
                  textAlign: 'right',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  fontFamily: 'monospace',
                  color: item.isGreedy ? '#34D399' : item.isValid ? '#38BDF8' : '#F43F5E',
                }}
              >
                {item.isValid ? `${pct}%` : 'MASKED'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
