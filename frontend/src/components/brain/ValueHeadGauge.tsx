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

  // Normalized angle from -20 to +50 points -> -120deg to +120deg
  const minVal = -20;
  const maxVal = 50;
  const clamped = Math.max(minVal, Math.min(maxVal, val));
  const normPct = (clamped - minVal) / (maxVal - minVal);
  const angleDeg = -120 + normPct * 240;

  return (
    <div
      className="value-head-gauge steampunk-panel"
      style={{
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: '2px solid #C59B27',
        borderRadius: '10px',
        padding: '0.85rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.6rem',
        boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Gauge size={16} color="#9E6B00" />
          <span
            style={{
              fontSize: '0.82rem',
              fontWeight: 800,
              color: '#23140C',
              fontFamily: "'Cinzel Decorative', Georgia, serif",
              letterSpacing: '0.04em',
            }}
          >
            Critic Steam Manometer: V(s)
          </span>
        </div>
        <span
          style={{
            fontSize: '0.68rem',
            background: 'linear-gradient(180deg, #F7E099 0%, #CBA232 100%)',
            border: '1px solid #6E4E04',
            color: '#23140C',
            padding: '0.15rem 0.5rem',
            borderRadius: '4px',
            fontFamily: "'Courier Prime', monospace",
            fontWeight: 800,
            textTransform: 'uppercase',
            boxShadow: '0 1px 2px rgba(0,0,0,0.15)',
          }}
        >
          {modelType.toUpperCase()} CALIBRATION
        </span>
      </div>

      {/* Steam Pressure Dial & Value Readout */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', margin: '0.2rem 0' }}>
        {/* Circular SVG Brass Pressure Gauge */}
        <div style={{ width: 90, height: 90, position: 'relative', flexShrink: 0 }}>
          <svg viewBox="0 0 100 100" style={{ width: '100%', height: '100%', display: 'block' }}>
            <defs>
              <radialGradient id="manometer-bezel" cx="40%" cy="40%" r="60%">
                <stop offset="0%" stopColor="#F6DC88" />
                <stop offset="50%" stopColor="#C59B27" />
                <stop offset="100%" stopColor="#6E4E04" />
              </radialGradient>
              <radialGradient id="manometer-face" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#FAF5EB" />
                <stop offset="85%" stopColor="#EADBBE" />
                <stop offset="100%" stopColor="#D4C09D" />
              </radialGradient>
            </defs>

            {/* Outer Brass Ring */}
            <circle cx="50" cy="50" r="48" fill="url(#manometer-bezel)" stroke="#382C26" strokeWidth="1" />
            <circle cx="50" cy="50" r="43" fill="none" stroke="#23140C" strokeWidth="0.8" strokeDasharray="3 1.5" />

            {/* Parchment Dial Face */}
            <circle cx="50" cy="50" r="41" fill="url(#manometer-face)" stroke="#4A2F1D" strokeWidth="1.2" />

            {/* Pressure Arc Scales */}
            <path
              d="M 22 75 A 35 35 0 1 1 78 75"
              fill="none"
              stroke="#A88D75"
              strokeWidth="2.5"
            />
            {/* Safe / High Pressure zones */}
            <path
              d="M 50 15 A 35 35 0 0 1 78 75"
              fill="none"
              stroke="#15803D"
              strokeWidth="2.5"
              strokeDasharray="4 2"
            />
            <path
              d="M 22 75 A 35 35 0 0 1 50 15"
              fill="none"
              stroke="#B91C1C"
              strokeWidth="2.5"
              strokeDasharray="4 2"
            />

            {/* Dial Tick Labels */}
            <text x="24" y="72" fontSize="6" fontFamily="serif" fontWeight="bold" fill="#B91C1C">-20</text>
            <text x="47" y="24" fontSize="6" fontFamily="serif" fontWeight="bold" fill="#785A42">0</text>
            <text x="68" y="72" fontSize="6" fontFamily="serif" fontWeight="bold" fill="#15803D">+50</text>
            <text x="50" y="38" fontSize="5.5" fontFamily="'Cinzel Decorative', serif" textAnchor="middle" fill="#785A42">PSI</text>

            {/* Swinging Brass Needle */}
            <g transform={`rotate(${angleDeg}, 50, 50)`} style={{ transition: 'transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)' }}>
              <polygon points="48,50 50,16 52,50" fill="#B91C1C" stroke="#23140C" strokeWidth="0.5" />
              <line x1="50" y1="50" x2="50" y2="18" stroke="#F6DC88" strokeWidth="0.8" />
            </g>

            {/* Center Pivot Rivet */}
            <circle cx="50" cy="50" r="5.5" fill="#382C26" stroke="#C59B27" strokeWidth="1.5" />
            <circle cx="50" cy="50" r="2" fill="#FAF5EB" />
          </svg>
        </div>

        {/* Digital / Monospace Score Readout */}
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.4rem' }}>
            <div
              style={{
                fontSize: '1.85rem',
                fontWeight: 900,
                fontFamily: "'Courier Prime', 'JetBrains Mono', monospace",
                color: isPositive ? '#15803D' : '#B91C1C',
                textShadow: '0 1px 0 rgba(255,255,255,0.7)',
              }}
            >
              {val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}
            </div>
            <span style={{ fontSize: '0.78rem', color: '#785A42', fontFamily: "'Crimson Pro', Georgia, serif" }}>
              expected score advantage
            </span>
          </div>

          <div style={{ fontSize: '0.72rem', color: '#5A3822', fontStyle: 'italic', fontFamily: "'Crimson Pro', Georgia, serif" }}>
            Estimated Value State & Advantage Margin
          </div>
        </div>
      </div>

      {/* Historical Trajectory Readout */}
      {historicalValues.length > 1 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', borderTop: '1px solid rgba(184, 134, 11, 0.25)', paddingTop: '0.4rem' }}>
          <TrendingUp size={13} color="#9E6B00" />
          <span style={{ fontSize: '0.72rem', color: '#4A2F1D', fontFamily: "'Courier Prime', monospace" }}>
            Trajectory (last {historicalValues.length} turns): [
            {historicalValues.slice(-5).map((v) => (v >= 0 ? `+${v.toFixed(1)}` : v.toFixed(1))).join(', ')}]
          </span>
        </div>
      )}
    </div>
  );
};
