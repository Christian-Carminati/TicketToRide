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
      {/* Outer Rotating Brass Cogwheel on Highlight / Hover */}
      {(isHighlighted || isHovered) && (
        <circle
          cx={x}
          cy={y}
          r={17}
          fill="none"
          stroke={isHighlighted ? '#B91C1C' : '#C59B27'}
          strokeWidth={2.5}
          strokeDasharray="4 2"
          opacity={0.9}
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

      {/* Shadow Base */}
      <circle
        cx={x + 1}
        cy={y + 2}
        r={isHovered ? 10 : 8}
        fill="rgba(56, 44, 38, 0.45)"
      />

      {/* Outer Brass Ring */}
      <circle
        cx={x}
        cy={y}
        r={isHovered ? 9.5 : 7.5}
        fill={isHighlighted ? '#D97706' : isHovered ? '#F6DC88' : '#C59B27'}
        stroke="#4A2F1D"
        strokeWidth={1.5}
      />

      {/* Inner Rivet Core */}
      <circle
        cx={x}
        cy={y}
        r={isHovered ? 5.5 : 4}
        fill={isHighlighted ? '#B91C1C' : isHovered ? '#23140C' : '#FAF5EB'}
        stroke="#23140C"
        strokeWidth={1}
      />

      {/* City Label with Crisp Ivory Halo Underlay for 100% Cartographic Readability */}
      <text
        x={x}
        y={y - 12}
        textAnchor="middle"
        fontSize={11.5}
        fontFamily="'Playfair Display', Georgia, serif"
        fontWeight="800"
        fill="#FAF5EB"
        stroke="#FAF5EB"
        strokeWidth={4.5}
        strokeLinejoin="round"
        paintOrder="stroke"
        opacity={0.95}
      >
        {city.name}
      </text>

      {/* Foreground Sepia Ink City Label */}
      <text
        x={x}
        y={y - 12}
        textAnchor="middle"
        fontSize={11.5}
        fontFamily="'Playfair Display', Georgia, serif"
        fontWeight="800"
        fill={isHighlighted ? '#B91C1C' : isHovered ? '#996515' : '#23140C'}
        letterSpacing="-0.01em"
      >
        {city.name}
      </text>
    </g>
  );
};
