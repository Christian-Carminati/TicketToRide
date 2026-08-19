import React from 'react';
import { BoardRoute, COLOR_HEX } from './mapData';

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

  const baseColor = route.color ? COLOR_HEX[route.color] || '#64748B' : '#64748B';
  const isClaimed = Boolean(claimedByPlayerColor);
  const trackColor = isClaimed ? claimedByPlayerColor! : baseColor;

  return (
    <g
      className={`route-edge ${isClaimable ? 'claimable' : ''} ${isClaimed ? 'claimed' : ''}`}
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
        stroke="#0F172A"
        strokeWidth={isHovered ? 12 : 10}
        strokeLinecap="round"
        opacity={0.8}
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
          transition: 'all 0.2s ease',
          filter: isClaimed ? 'drop-shadow(0 0 4px rgba(255,255,255,0.4))' : undefined,
        }}
      />

      {/* Claimable highlight pulse */}
      {isClaimable && !isClaimed && (
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
