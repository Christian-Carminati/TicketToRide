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
  color = '#38BDF8',
  icon = '📈',
}) => {
  return (
    <div
      style={{
        background: 'rgba(15, 23, 42, 0.85)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '10px',
        padding: '1rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.25rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.8rem', color: '#94A3B8', fontWeight: 600 }}>{label}</span>
        <span style={{ fontSize: '1rem' }}>{icon}</span>
      </div>

      <div style={{ fontSize: '1.4rem', fontWeight: 700, color: color }}>
        {value}
      </div>

      {subtext && <div style={{ fontSize: '0.75rem', color: '#64748B' }}>{subtext}</div>}
    </div>
  );
};
