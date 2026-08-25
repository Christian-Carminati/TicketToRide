import React, { useMemo } from 'react';

export interface ChartSeries {
  id: string;
  name: string;
  color: string;
  data: Array<{ x: number; y: number }>;
}

interface LineChartSVGProps {
  title: string;
  series: ChartSeries[];
  width?: number;
  height?: number;
  yLabel?: string;
  xLabel?: string;
}

export const LineChartSVG: React.FC<LineChartSVGProps> = React.memo(({
  title,
  series,
  width = 500,
  height = 240,
  yLabel = '',
  xLabel = 'Steps',
}) => {
  const padding = { top: 30, right: 20, bottom: 40, left: 50 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const { minX, maxX, minY, maxY } = useMemo(() => {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    series.forEach((s) => {
      s.data.forEach((p) => {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      });
    });

    if (!isFinite(minX)) minX = 0;
    if (!isFinite(maxX) || maxX === minX) maxX = minX + 100;
    if (!isFinite(minY)) minY = 0;
    if (!isFinite(maxY) || maxY === minY) maxY = minY + 1;

    const yMargin = (maxY - minY) * 0.1 || 0.5;
    return { minX, maxX, minY: minY - yMargin, maxY: maxY + yMargin };
  }, [series]);

  const scaleX = (x: number) => padding.left + ((x - minX) / (maxX - minX)) * plotWidth;
  const scaleY = (y: number) => padding.top + (1 - (y - minY) / (maxY - minY)) * plotHeight;

  return (
    <div
      className="steampunk-chart-panel"
      style={{
        background: 'linear-gradient(180deg, #FAF4E6 0%, #EADBBE 100%)',
        border: '1.5px solid #C59B27',
        borderRadius: '8px',
        padding: '0.85rem',
        width: '100%',
        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
        <h5
          style={{
            margin: 0,
            fontSize: '0.85rem',
            fontFamily: "'Cinzel Decorative', Georgia, serif",
            fontWeight: 800,
            color: '#23140C',
            letterSpacing: '0.02em',
          }}
        >
          {title}
        </h5>
        {/* Legend */}
        <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.75rem' }}>
          {series.map((s) => (
            <div key={s.id} style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: s.color, border: '1px solid #23140C' }} />
              <span style={{ color: '#4A2F1D', fontFamily: "'Playfair Display', Georgia, serif", fontWeight: 700 }}>{s.name}</span>
            </div>
          ))}
        </div>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {/* Horizontal Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((frac, i) => {
          const yVal = minY + (maxY - minY) * frac;
          const py = scaleY(yVal);
          return (
            <g key={i}>
              <line
                x1={padding.left}
                y1={py}
                x2={width - padding.right}
                y2={py}
                stroke="rgba(110, 75, 45, 0.2)"
                strokeDasharray="4 3"
              />
              <text
                x={padding.left - 8}
                y={py + 3}
                textAnchor="end"
                fontSize={10}
                fontFamily="'Courier Prime', monospace"
                fill="#785A42"
              >
                {yVal.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* X Axis Labels */}
        {[0, 0.5, 1.0].map((frac, i) => {
          const xVal = minX + (maxX - minX) * frac;
          const px = scaleX(xVal);
          return (
            <text
              key={i}
              x={px}
              y={height - 10}
              textAnchor={i === 0 ? 'start' : i === 2 ? 'end' : 'middle'}
              fontSize={10}
              fontFamily="'Courier Prime', monospace"
              fill="#785A42"
            >
              {Math.round(xVal)}
            </text>
          );
        })}

        {/* Axis Titles */}
        {yLabel && (
          <text
            x={-height / 2}
            y={14}
            transform="rotate(-90)"
            textAnchor="middle"
            fontSize={10}
            fontFamily="'Playfair Display', Georgia, serif"
            fontStyle="italic"
            fill="#5A3822"
          >
            {yLabel}
          </text>
        )}
        <text
          x={width / 2}
          y={height - 2}
          textAnchor="middle"
          fontSize={10}
          fontFamily="'Playfair Display', Georgia, serif"
          fontStyle="italic"
          fill="#5A3822"
        >
          {xLabel}
        </text>

        {/* Lines */}
        {series.map((s) => {
          if (s.data.length === 0) return null;
          const dataPoints = s.data.length > 100 
            ? s.data.filter((_, idx) => idx % Math.ceil(s.data.length / 100) === 0 || idx === s.data.length - 1)
            : s.data;

          const pointsStr = dataPoints
            .map((p) => `${scaleX(p.x).toFixed(1)},${scaleY(p.y).toFixed(1)}`)
            .join(' ');

          return (
            <polyline
              key={s.id}
              fill="none"
              stroke={s.color}
              strokeWidth={2.5}
              strokeLinecap="round"
              strokeLinejoin="round"
              points={pointsStr}
            />
          );
        })}
      </svg>
    </div>
  );
});
