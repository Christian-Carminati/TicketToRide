import React from 'react';

interface TelemetryCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  color?: string;
  icon?: string;
}

export const TelemetryCard: React.FC<TelemetryCardProps> = ({
  label,
  value,
  subtext,
  color = '#9E6B00',
  icon = '📈',
}) => {
  return (
    <div
      className="telemetry-card steampunk-panel"
      style={{
        background: 'linear-gradient(180deg, #FAF4E6 0%, #EADBBE 100%)',
        border: '1.5px solid #C59B27',
        borderRadius: '8px',
        padding: '0.85rem 1rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.25rem',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.8rem', color: '#4A2F1D', fontFamily: "'Playfair Display', Georgia, serif", fontWeight: 800 }}>{label}</span>
        <span style={{ fontSize: '1.1rem' }}>{icon}</span>
      </div>

      <div
        style={{
          fontSize: '1.4rem',
          fontWeight: 800,
          fontFamily: "'Courier Prime', monospace",
          color: color,
          textShadow: '0 1px 0 rgba(255,255,255,0.7)',
        }}
      >
        {value}
      </div>

      {subtext && <div style={{ fontSize: '0.75rem', color: '#785A42', fontFamily: "'Crimson Pro', Georgia, serif" }}>{subtext}</div>}
    </div>
  );
};
