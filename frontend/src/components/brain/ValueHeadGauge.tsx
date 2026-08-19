import React from 'react';
import { Gauge, TrendingUp } from 'lucide-react';

interface ValueHeadGaugeProps {
  estimatedValue?: number | null;
  modelType?: string;
  historicalValues?: number[];
}

export const ValueHeadGauge: React.FC<ValueHeadGaugeProps> = ({
  estimatedValue = 0,
  modelType = 'ppo',
  historicalValues = [],
}) => {
  const val = estimatedValue ?? 0;
  const isPositive = val >= 0;

  // Normalized progress bar from -20 to +50 points
  const minVal = -20;
  const maxVal = 50;
  const clamped = Math.max(minVal, Math.min(maxVal, val));
  const normPct = ((clamped - minVal) / (maxVal - minVal)) * 100;

  return (
    <div
      className="value-head-gauge"
      style={{
        background: 'rgba(15, 23, 42, 0.85)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '10px',
        padding: '0.85rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Gauge size={16} color="#818CF8" />
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#F1F5F9', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Critic Value Head: V(s)
          </span>
        </div>
        <span
          style={{
            fontSize: '0.7rem',
            background: 'rgba(99, 102, 241, 0.15)',
            border: '1px solid rgba(99, 102, 241, 0.3)',
            color: '#A5B4FC',
            padding: '0.15rem 0.45rem',
            borderRadius: '4px',
            fontWeight: 600,
            textTransform: 'uppercase',
          }}
        >
          {modelType.toUpperCase()} Baseline
        </span>
      </div>

      {/* Main Scalar Readout */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', margin: '0.2rem 0' }}>
        <div
          style={{
            fontSize: '1.75rem',
            fontWeight: 800,
            fontFamily: 'monospace',
            color: isPositive ? '#34D399' : '#F43F5E',
          }}
        >
          {val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}
        </div>
        <div style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
          expected game return (score advantage)
        </div>
      </div>

      {/* Gauge Bar */}
      <div
        style={{
          width: '100%',
          height: '8px',
          background: 'rgba(30, 41, 59, 0.8)',
          borderRadius: '4px',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Center line */}
        <div
          style={{
            position: 'absolute',
            left: `${((-minVal) / (maxVal - minVal)) * 100}%`,
            top: 0,
            bottom: 0,
            width: '2px',
            background: 'rgba(255, 255, 255, 0.3)',
            zIndex: 2,
          }}
        />
        <div
          style={{
            height: '100%',
            width: `${normPct}%`,
            background: isPositive
              ? 'linear-gradient(to right, #3B82F6, #34D399)'
              : 'linear-gradient(to right, #F43F5E, #FB7185)',
            borderRadius: '4px',
            transition: 'width 0.3s ease',
          }}
        />
      </div>

      {/* Historical sparkline if present */}
      {historicalValues.length > 1 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.2rem' }}>
          <TrendingUp size={12} color="#94A3B8" />
          <span style={{ fontSize: '0.7rem', color: '#64748B' }}>
            Trajectory (last {historicalValues.length} turns): [
            {historicalValues.slice(-5).map((v) => (v >= 0 ? `+${v.toFixed(1)}` : v.toFixed(1))).join(', ')}]
          </span>
        </div>
      )}
    </div>
  );
};
