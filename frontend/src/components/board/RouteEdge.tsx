import React from 'react';
import { BoardRoute, COLOR_HEX } from './mapData';
import { HoveredActionMeta } from '../../context/workbenchTypes';

interface RouteEdgeProps {
  route: BoardRoute;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  claimedByPlayerColor?: string | null;
  claimedByPlayerName?: string | null;
  isClaimable?: boolean;
  isHovered?: boolean;
  hoveredMeta?: HoveredActionMeta | null;
  onMouseEnter?: (route: BoardRoute) => void;
  onMouseLeave?: (route: BoardRoute) => void;
  onClick?: (route: BoardRoute) => void;
}

export const RouteEdge: React.FC<RouteEdgeProps> = ({
  route,
  x1,
  y1,
  x2,
  y2,
  claimedByPlayerColor,
  claimedByPlayerName,
  isClaimable = false,
  isHovered = false,
  hoveredMeta = null,
  onMouseEnter,
  onMouseLeave,
  onClick,
}) => {
  // Vector calculations
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len === 0) return null;

  const ux = dx / len;
  const uy = dy / len;
  const nx = -uy;
  const ny = ux;

  // Apply parallel offset if double route
  const offsetDist = (route.offset_index || 0) * 8.5;
  const ox1 = x1 + nx * offsetDist;
  const oy1 = y1 + ny * offsetDist;
  const ox2 = x2 + nx * offsetDist;
  const oy2 = y2 + ny * offsetDist;

  const midX = (ox1 + ox2) / 2;
  const midY = (oy1 + oy2) / 2;

  const baseColor = route.color ? COLOR_HEX[route.color] || '#7D6A5A' : '#7D6A5A';
  const isClaimed = Boolean(claimedByPlayerColor);
  const trackColor = isClaimed ? claimedByPlayerColor! : baseColor;

  return (
    <g
      className={`route-edge ${isClaimable ? 'claimable' : ''} ${isClaimed ? 'claimed' : ''} ${isHovered ? 'hover-linked' : ''}`}
      style={{ cursor: isClaimable || isClaimed ? 'pointer' : 'default' }}
      tabIndex={isClaimable ? 0 : -1}
      role="button"
      aria-label={`${route.city_a} to ${route.city_b}, ${route.length} cars, ${route.color || 'Gray'}${isClaimed ? `, Claimed by ${claimedByPlayerName || 'Player'}` : isClaimable ? ', Claimable' : ''}`}
      onKeyDown={(e) => {
        if ((e.key === 'Enter' || e.key === ' ') && isClaimable) {
          e.preventDefault();
          onClick?.(route);
        }
      }}
      onMouseEnter={() => onMouseEnter?.(route)}
      onMouseLeave={() => onMouseLeave?.(route)}
      onClick={() => onClick?.(route)}
    >
      {/* 1. Track Shadow */}
      <line
        x1={ox1 + 1}
        y1={oy1 + 2}
        x2={ox2 + 1}
        y2={oy2 + 2}
        stroke="rgba(56, 44, 38, 0.45)"
        strokeWidth={isHovered ? 14 : 10}
        strokeLinecap="round"
      />

      {/* 2. Wooden Sleepers / Track Bed */}
      <line
        x1={ox1}
        y1={oy1}
        x2={ox2}
        y2={oy2}
        stroke={isHovered ? '#C59B27' : '#3D281A'}
        strokeWidth={isHovered ? 13 : 9.5}
        strokeLinecap="round"
        style={{
          transition: 'all 0.15s ease',
          filter: isHovered ? 'drop-shadow(0 0 6px rgba(197, 155, 39, 0.7))' : undefined,
        }}
      />

      {/* 3. Dual Steel Rails Layer */}
      <line
        x1={ox1}
        y1={oy1}
        x2={ox2}
        y2={oy2}
        stroke="#E8DBBE"
        strokeWidth={7.5}
        strokeLinecap="round"
      />

      {/* 4. Jewel Colored Train Cars Segments */}
      <line
        x1={ox1}
        y1={oy1}
        x2={ox2}
        y2={oy2}
        stroke={trackColor}
        strokeWidth={isHovered ? 7 : 5.5}
        strokeDasharray={`${Math.max(8, len / route.length - 4)} 4`}
        strokeLinecap="round"
        style={{
          transition: 'all 0.15s ease',
          filter: isClaimed ? 'drop-shadow(0 1px 3px rgba(0,0,0,0.4))' : undefined,
        }}
      />

      {/* 5. Claimable Golden Brass Pulse */}
      {isClaimable && !isClaimed && !isHovered && (
        <line
          x1={ox1}
          y1={oy1}
          x2={ox2}
          y2={oy2}
          stroke="#F6DC88"
          strokeWidth={7.5}
          strokeDasharray="5 5"
          strokeOpacity={0.85}
          strokeLinecap="round"
        >
          <animate
            attributeName="stroke-dashoffset"
            from="0"
            to="20"
            dur="1.2s"
            repeatCount="indefinite"
          />
        </line>
      )}

      {/* 6. Victorian Brass Neural Plaque on Hover */}
      {isHovered && hoveredMeta && hoveredMeta.probability !== undefined && (
        <g transform={`translate(${midX}, ${midY})`} style={{ pointerEvents: 'none' }}>
          <rect
            x={-38}
            y={-15}
            width={76}
            height={30}
            rx={5}
            fill="#FAF3E6"
            stroke={hoveredMeta.isMasked ? '#B91C1C' : '#C59B27'}
            strokeWidth={2}
            filter="drop-shadow(0 2px 8px rgba(0,0,0,0.4))"
          />
          <rect
            x={-35}
            y={-12}
            width={70}
            height={24}
            rx={3}
            fill="none"
            stroke="#4D311E"
            strokeWidth={0.75}
            strokeDasharray="2 1"
          />
          <text
            x={0}
            y={3}
            textAnchor="middle"
            fill={hoveredMeta.isMasked ? '#B91C1C' : '#23140C'}
            fontSize={11}
            fontWeight="bold"
            fontFamily="'Courier Prime', 'JetBrains Mono', monospace"
          >
            {hoveredMeta.isMasked ? 'LOCKED' : `P: ${(hoveredMeta.probability * 100).toFixed(0)}%`}
          </text>
        </g>
      )}

      {/* Tooltip on hover */}
      {isHovered && (
        <title>
          {`${route.city_a} ⟷ ${route.city_b} (${route.length} trains, ${route.color || 'Gray'}) ${
            isClaimed ? `[Claimed by ${claimedByPlayerName || 'Player'}]` : isClaimable ? '[Click to Claim]' : ''
          }`}
        </title>
      )}
    </g>
  );
};
