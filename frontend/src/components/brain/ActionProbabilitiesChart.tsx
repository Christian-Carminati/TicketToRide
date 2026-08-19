import React, { useState, useMemo } from 'react';

interface ActionProbabilitiesChartProps {
  probabilities: number[];
  actionMask: boolean[];
  actionLabels: string[];
  rawLogitsOrQ?: number[];
  greedyActionIndex?: number;
}

export const ActionProbabilitiesChart: React.FC<ActionProbabilitiesChartProps> = ({
  probabilities,
  actionMask,
  actionLabels,
  rawLogitsOrQ,
  greedyActionIndex,
}) => {
  const [filterMode, setFilterMode] = useState<'all' | 'valid' | 'top10'>('valid');

  const items = useMemo(() => {
    const list = probabilities.map((prob, idx) => ({
      index: idx,
      label: actionLabels[idx] || `Action ${idx}`,
      prob: prob,
      isValid: actionMask[idx] ?? true,
      rawScore: rawLogitsOrQ ? rawLogitsOrQ[idx] : undefined,
      isGreedy: idx === greedyActionIndex,
    }));

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
      style={{
        background: 'rgba(15, 23, 42, 0.9)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '12px',
        padding: '1.25rem',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.5rem',
          marginBottom: '1rem',
        }}
      >
        <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#F1F5F9' }}>
          📊 Action Distribution & Masking (π(a|s))
        </h4>

        {/* Filter Buttons */}
        <div style={{ display: 'flex', gap: '0.3rem' }}>
          {(['valid', 'top10', 'all'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setFilterMode(mode)}
              style={{
                background: filterMode === mode ? '#3B82F6' : 'rgba(30, 41, 59, 0.8)',
                color: filterMode === mode ? '#FFFFFF' : '#94A3B8',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '4px',
                padding: '0.2rem 0.5rem',
                fontSize: '0.75rem',
                cursor: 'pointer',
                fontWeight: 600,
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
          gap: '0.4rem',
          maxHeight: '340px',
          overflowY: 'auto',
          paddingRight: '0.4rem',
        }}
      >
        {items.map((item) => {
          const pct = (item.prob * 100).toFixed(1);
          return (
            <div
              key={item.index}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                padding: '0.35rem 0.6rem',
                borderRadius: '6px',
                background: item.isGreedy
                  ? 'rgba(16, 185, 129, 0.15)'
                  : item.isValid
                  ? 'rgba(30, 41, 59, 0.6)'
                  : 'rgba(15, 23, 42, 0.4)',
                border: item.isGreedy
                  ? '1px solid #10B981'
                  : '1px solid rgba(255,255,255,0.05)',
                opacity: item.isValid ? 1 : 0.4,
              }}
            >
              {/* Action Label */}
              <div style={{ minWidth: '160px', fontSize: '0.8rem', color: '#F1F5F9', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span>{item.isValid ? (item.isGreedy ? '⭐' : '✓') : '🔒'}</span>
                <span style={{ fontWeight: item.isGreedy ? 700 : 500 }}>{item.label}</span>
              </div>

              {/* Progress Probability Bar */}
              <div
                style={{
                  flex: 1,
                  height: '14px',
                  backgroundColor: 'rgba(15, 23, 42, 0.8)',
                  borderRadius: '4px',
                  overflow: 'hidden',
                  position: 'relative',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${item.prob * 100}%`,
                    background: item.isGreedy
                      ? 'linear-gradient(to right, #10B981, #34D399)'
                      : item.isValid
                      ? 'linear-gradient(to right, #3B82F6, #38BDF8)'
                      : '#475569',
                    borderRadius: '4px',
                    transition: 'width 0.3s ease',
                  }}
                />
              </div>

              {/* Numerical Value */}
              <div
                style={{
                  minWidth: '55px',
                  textAlign: 'right',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  color: item.isGreedy ? '#34D399' : item.isValid ? '#38BDF8' : '#64748B',
                }}
              >
                {item.isValid ? `${pct}%` : 'Masked'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
