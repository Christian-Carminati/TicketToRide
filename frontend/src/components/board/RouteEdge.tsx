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
  const offsetDist = (route.offset_index || 0) * 8.0;
  const ox1 = x1 + nx * offsetDist;
  const oy1 = y1 + ny * offsetDist;
  const ox2 = x2 + nx * offsetDist;
  const oy2 = y2 + ny * offsetDist;

  const midX = (ox1 + ox2) / 2;
  const midY = (oy1 + oy2) / 2;

  const baseColor = route.color ? COLOR_HEX[route.color] || '#64748B' : '#64748B';
  const isClaimed = Boolean(claimedByPlayerColor);
  const trackColor = isClaimed ? claimedByPlayerColor! : baseColor;

  return (
    <g
      className={`route-edge ${isClaimable ? 'claimable' : ''} ${isClaimed ? 'claimed' : ''} ${isHovered ? 'hover-linked' : ''}`}
      style={{ cursor: isClaimable || isClaimed ? 'pointer' : 'default' }}
      onMouseEnter={() => onMouseEnter?.(route)}
      onMouseLeave={() => onMouseLeave?.(route)}
      onClick={() => onClick?.(route)}
    >
      {/* Background wider track bed */}
      <line
        x1={ox1}
        y1={oy1}
        x2={ox2}
        y2={oy2}
        stroke={isHovered ? '#38BDF8' : '#0F172A'}
        strokeWidth={isHovered ? 14 : 10}
        strokeLinecap="round"
        opacity={isHovered ? 0.9 : 0.8}
        style={{
          transition: 'all 0.15s ease',
          filter: isHovered ? 'drop-shadow(0 0 8px rgba(56, 189, 248, 0.8))' : undefined,
        }}
      />

      {/* Segmented dashed track for train pieces */}
      <line
        x1={ox1}
        y1={oy1}
        x2={ox2}
        y2={oy2}
        stroke={trackColor}
        strokeWidth={isHovered ? 8 : 6}
        strokeDasharray={`${Math.max(8, len / route.length - 4)} 4`}
        strokeLinecap="round"
        style={{
          transition: 'all 0.15s ease',
          filter: isClaimed ? 'drop-shadow(0 0 4px rgba(255,255,255,0.4))' : undefined,
        }}
      />

      {/* Claimable highlight pulse */}
      {isClaimable && !isClaimed && !isHovered && (
        <line
          x1={ox1}
          y1={oy1}
          x2={ox2}
          y2={oy2}
          stroke="#38BDF8"
          strokeWidth={8}
          strokeDasharray="6 6"
          strokeOpacity={0.7}
          strokeLinecap="round"
        >
          <animate
            attributeName="stroke-dashoffset"
            from="0"
            to="24"
            dur="1.5s"
            repeatCount="indefinite"
          />
        </line>
      )}

      {/* Neural Hover Overlay Badge on Track */}
      {isHovered && hoveredMeta && hoveredMeta.probability !== undefined && (
        <g transform={`translate(${midX}, ${midY})`} style={{ pointerEvents: 'none' }}>
          <rect
            x={-35}
            y={-14}
            width={70}
            height={28}
            rx={6}
            fill="rgba(15, 23, 42, 0.95)"
            stroke={hoveredMeta.isMasked ? '#F43F5E' : '#38BDF8'}
            strokeWidth={1.5}
            filter="drop-shadow(0 2px 8px rgba(0,0,0,0.6))"
          />
          <text
            x={0}
            y={4}
            textAnchor="middle"
            fill={hoveredMeta.isMasked ? '#F43F5E' : '#38BDF8'}
            fontSize={11}
            fontWeight="bold"
            fontFamily="monospace"
          >
            {hoveredMeta.isMasked ? 'MASKED' : `P: ${(hoveredMeta.probability * 100).toFixed(0)}%`}
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
