import React from 'react';
import { BoardCity } from './mapData';

interface CityNodeProps {
  city: BoardCity;
  x: number;
  y: number;
  isHighlighted?: boolean;
  isHovered?: boolean;
  onMouseEnter?: (city: BoardCity) => void;
  onMouseLeave?: (city: BoardCity) => void;
  onClick?: (city: BoardCity) => void;
}

export const CityNode: React.FC<CityNodeProps> = ({
  city,
  x,
  y,
  isHighlighted = false,
  isHovered = false,
  onMouseEnter,
  onMouseLeave,
  onClick,
}) => {
  return (
    <g
      className="city-node"
      style={{ cursor: 'pointer', transition: 'all 0.2s ease' }}
      onMouseEnter={() => onMouseEnter?.(city)}
      onMouseLeave={() => onMouseLeave?.(city)}
      onClick={() => onClick?.(city)}
    >
      {/* Outer Pulse/Glow on highlight */}
      {(isHighlighted || isHovered) && (
        <circle
          cx={x}
          cy={y}
          r={16}
          fill="none"
          stroke={isHighlighted ? '#F59E0B' : '#38BDF8'}
          strokeWidth={3}
          strokeDasharray="4 2"
          opacity={0.8}
        >
          <animateTransform
            attributeName="transform"
            type="rotate"
            from={`0 ${x} ${y}`}
            to={`360 ${x} ${y}`}
            dur="4s"
            repeatCount="indefinite"
          />
        </circle>
      )}

      {/* Main City Circle */}
      <circle
        cx={x}
        cy={y}
        r={isHovered ? 9 : 7}
        fill={isHighlighted ? '#F59E0B' : isHovered ? '#38BDF8' : '#F8FAFC'}
        stroke="#0F172A"
        strokeWidth={2.5}
      />

      {/* City Label with stroke outline */}
      <text
        x={x}
        y={y - 12}
        textAnchor="middle"
        fontSize={11}
        fontWeight="600"
        fill="#0F172A"
        stroke="#0F172A"
        strokeWidth={3}
        strokeLinejoin="round"
        paintOrder="stroke"
        opacity={0.95}
      >
        {city.name}
      </text>
      <text
        x={x}
        y={y - 12}
        textAnchor="middle"
        fontSize={11}
        fontWeight="600"
        fill={isHighlighted ? '#FBBF24' : '#F1F5F9'}
      >
        {city.name}
      </text>
    </g>
  );
};
