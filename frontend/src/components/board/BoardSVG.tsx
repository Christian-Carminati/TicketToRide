import React, { useState, useMemo } from 'react';
import { ActionDTO, PlayerStateDTO } from '../../api/types';
import { CityNode } from './CityNode';
import {
  BoardCity,
  BoardRoute,
  BOARD_CITIES,
  buildBoardRoutes,
} from './mapData';
import { RouteEdge } from './RouteEdge';

interface BoardSVGProps {
  mapName?: string;
  claimedRoutes?: Record<string, string>; // route_id -> player_id
  players?: PlayerStateDTO[];
  validActions?: ActionDTO[];
  highlightedCities?: string[];
  onRouteClick?: (route: BoardRoute) => void;
  onCityClick?: (city: BoardCity) => void;
}

const SVG_WIDTH = 1000;
const SVG_HEIGHT = 650;
const PADDING_X = 70;
const PADDING_Y = 60;

export const BoardSVG: React.FC<BoardSVGProps> = ({
  claimedRoutes = {},
  players = [],
  validActions = [],
  highlightedCities = [],
  onRouteClick,
  onCityClick,
}) => {
  const [hoveredCity, setHoveredCity] = useState<string | null>(null);
  const [hoveredRoute, setHoveredRoute] = useState<string | null>(null);

  const { cities, routes, cityCoordsMap } = useMemo(() => {
    const cityList = BOARD_CITIES;
    const routeList = buildBoardRoutes();

    const coords: Record<string, { x: number; y: number }> = {};
    cityList.forEach((c) => {
      // Scale from [0, 1] to viewBox coordinates
      const sx = PADDING_X + c.x * (SVG_WIDTH - 2 * PADDING_X);
      const sy = PADDING_Y + (1.0 - c.y) * (SVG_HEIGHT - 2 * PADDING_Y);
      coords[c.name] = { x: sx, y: sy };
      coords[c.id] = { x: sx, y: sy };
    });

    return { cities: cityList, routes: routeList, cityCoordsMap: coords };
  }, []);

  // Set of claimable route IDs from valid actions
  const claimableRouteIds = useMemo(() => {
    const set = new Set<string>();
    validActions.forEach((act) => {
      if (act.action_type === 'CLAIM_ROUTE' && act.route_id) {
        set.add(act.route_id);
      }
    });
    return set;
  }, [validActions]);

  // Map player id -> player object for colors
  const playerMap = useMemo(() => {
    const map: Record<string, PlayerStateDTO> = {};
    players.forEach((p) => {
      map[p.player_id] = p;
    });
    return map;
  }, [players]);

  return (
    <div className="board-svg-container" style={{ width: '100%', position: 'relative', overflow: 'hidden' }}>
      <svg
        viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
        style={{
          width: '100%',
          height: 'auto',
          display: 'block',
          background: 'linear-gradient(135deg, #090D16 0%, #0F172A 100%)',
          borderRadius: '12px',
          border: '1px solid rgba(255,255,255,0.08)',
          boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
        }}
      >
        {/* Subtle grid pattern background */}
        <defs>
          <pattern id="board-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255, 255, 255, 0.03)" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width={SVG_WIDTH} height={SVG_HEIGHT} fill="url(#board-grid)" />

        {/* Routes Layer (drawn behind cities) */}
        <g className="routes-layer">
          {routes.map((r) => {
            const p1 = cityCoordsMap[r.city_a];
            const p2 = cityCoordsMap[r.city_b];
            if (!p1 || !p2) return null;

            const claimedPlayerId = claimedRoutes[r.id];
            const claimedPlayer = claimedPlayerId ? playerMap[claimedPlayerId] : null;
            const isClaimable = claimableRouteIds.has(r.id);
            const isHovered = hoveredRoute === r.id;

            return (
              <RouteEdge
                key={r.id}
                route={r}
                x1={p1.x}
                y1={p1.y}
                x2={p2.x}
                y2={p2.y}
                claimedByPlayerColor={claimedPlayer ? claimedPlayer.color : null}
                claimedByPlayerName={claimedPlayer ? claimedPlayer.name : null}
                isClaimable={isClaimable}
                isHovered={isHovered}
                onMouseEnter={() => setHoveredRoute(r.id)}
                onMouseLeave={() => setHoveredRoute(null)}
                onClick={() => onRouteClick?.(r)}
              />
            );
          })}
        </g>

        {/* Cities Layer */}
        <g className="cities-layer">
          {cities.map((c) => {
            const pos = cityCoordsMap[c.name] || cityCoordsMap[c.id];
            if (!pos) return null;

            const isHighlighted = highlightedCities.includes(c.name) || highlightedCities.includes(c.id);
            const isHovered = hoveredCity === c.name || hoveredCity === c.id;

            return (
              <CityNode
                key={c.id}
                city={c}
                x={pos.x}
                y={pos.y}
                isHighlighted={isHighlighted}
                isHovered={isHovered}
                onMouseEnter={() => setHoveredCity(c.name)}
                onMouseLeave={() => setHoveredCity(null)}
                onClick={() => onCityClick?.(c)}
              />
            );
          })}
        </g>
      </svg>
    </div>
  );
};
