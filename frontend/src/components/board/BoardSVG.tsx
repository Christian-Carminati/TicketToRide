import React, { useState, useMemo } from 'react';
import { ActionDTO, PlayerStateDTO } from '../../api/types';
import { ObservabilityMode, HoveredActionMeta } from '../../context/workbenchTypes';
import { CityNode } from './CityNode';
import {
  BoardCity,
  BoardRoute,
  getMapData,
} from './mapData';
import { RouteEdge } from './RouteEdge';
import { UsaCartographyBackground } from './UsaCartographyBackground';

interface BoardSVGProps {
  mapName?: string;
  claimedRoutes?: Record<string, string>; // route_id -> player_id
  players?: PlayerStateDTO[];
  validActions?: ActionDTO[];
  highlightedCities?: string[];
  observabilityMode?: ObservabilityMode;
  hoveredRouteId?: string | null;
  hoveredMeta?: HoveredActionMeta | null;
  focusPlayerMode?: 'all' | 'player_0' | 'player_1' | null;
  onToggleFocusMode?: () => void;
  onRouteClick?: (route: BoardRoute) => void;
  onCityClick?: (city: BoardCity) => void;
  onRouteHover?: (routeId: string | null) => void;
}

const SVG_WIDTH = 1000;
const SVG_HEIGHT = 650;
const PADDING_X = 70;
const PADDING_Y = 60;

export const BoardSVG: React.FC<BoardSVGProps> = ({
  mapName = 'usa',
  claimedRoutes = {},
  players = [],
  validActions = [],
  highlightedCities = [],
  observabilityMode = 'god',
  hoveredRouteId = null,
  hoveredMeta = null,
  focusPlayerMode = 'all',
  onToggleFocusMode,
  onRouteClick,
  onCityClick,
  onRouteHover,
}) => {
  const [internalHoveredCity, setInternalHoveredCity] = useState<string | null>(null);
  const [internalHoveredRoute, setInternalHoveredRoute] = useState<string | null>(null);

  const activeHoveredRoute = hoveredRouteId !== undefined && hoveredRouteId !== null ? hoveredRouteId : internalHoveredRoute;

  const { cities, routes, cityCoordsMap } = useMemo(() => {
    const { cities: cityList, routes: routeList } = getMapData(mapName);

    const coords: Record<string, { x: number; y: number }> = {};
    cityList.forEach((c) => {
      // Scale from [0, 1] to viewBox coordinates
      const sx = PADDING_X + c.x * (SVG_WIDTH - 2 * PADDING_X);
      const sy = PADDING_Y + (1.0 - c.y) * (SVG_HEIGHT - 2 * PADDING_Y);
      coords[c.name] = { x: sx, y: sy };
      coords[c.id] = { x: sx, y: sy };
    });

    return { cities: cityList, routes: routeList, cityCoordsMap: coords };
  }, [mapName]);

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

  const isUsaMap = (mapName || 'usa').toLowerCase() === 'usa';

  return (
    <div
      className="board-svg-container steampunk-panel"
      style={{
        width: '100%',
        position: 'relative',
        overflow: 'hidden',
        padding: '6px',
        backgroundColor: '#26180F',
        border: '3px solid #C59B27',
        borderRadius: '12px',
        boxShadow: '0 8px 30px rgba(0, 0, 0, 0.45), inset 0 0 15px rgba(0,0,0,0.5)',
      }}
    >
      <svg
        viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
        style={{
          width: '100%',
          height: 'auto',
          display: 'block',
          borderRadius: '8px',
          background: '#F4ECDC',
        }}
      >
        {/* Background Cartography Layer */}
        {isUsaMap ? (
          <UsaCartographyBackground width={SVG_WIDTH} height={SVG_HEIGHT} />
        ) : (
          <g>
            <rect width={SVG_WIDTH} height={SVG_HEIGHT} fill="#F4ECDC" />
            <defs>
              <pattern id="mini-board-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(110, 75, 45, 0.1)" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width={SVG_WIDTH} height={SVG_HEIGHT} fill="url(#mini-board-grid)" />
            {/* Border frame */}
            <rect x={10} y={10} width={SVG_WIDTH - 20} height={SVG_HEIGHT - 20} rx={6} fill="none" stroke="#B8860B" strokeWidth={2} />
          </g>
        )}

        {/* Observability Mode & Cartographer Banner (Top Left) */}
        <g
          transform="translate(24, 30)"
          opacity={0.92}
          style={{ cursor: onToggleFocusMode ? 'pointer' : 'default' }}
          onClick={onToggleFocusMode}
        >
          <rect
            x={-6}
            y={-14}
            width={focusPlayerMode && focusPlayerMode !== 'all' ? 440 : 340}
            height={22}
            rx={4}
            fill="rgba(250, 245, 235, 0.92)"
            stroke="rgba(184, 134, 11, 0.7)"
            strokeWidth={1.5}
          />
          <text fill="#4A2F1D" fontSize={10} fontFamily="'Courier Prime', 'JetBrains Mono', monospace" fontWeight="700" letterSpacing="0.06em">
            {focusPlayerMode === 'player_0'
              ? `● FILTRO: RETE ISOLATA GIOCATORE 1 (BLU) [Clicca per Giocatore 2]`
              : focusPlayerMode === 'player_1'
              ? `● FILTRO: RETE ISOLATA GIOCATORE 2 (ROSSO) [Clicca per Tutte]`
              : observabilityMode === 'god'
              ? `● PERCEPTION: OMNISCIENT OBSERVER [${mapName.toUpperCase()}]`
              : observabilityMode === 'player_0'
              ? `● AGENT A: PARTIAL OBSERVABILITY [${mapName.toUpperCase()}]`
              : `● AGENT B: PARTIAL OBSERVABILITY [${mapName.toUpperCase()}]`}
          </text>
        </g>

        {/* Routes Layer (drawn behind cities) */}
        <g className="routes-layer">
          {routes.map((r) => {
            const p1 = cityCoordsMap[r.city_a];
            const p2 = cityCoordsMap[r.city_b];
            if (!p1 || !p2) return null;

            const claimedPlayerId = claimedRoutes[r.id];
            const claimedPlayer = claimedPlayerId ? playerMap[claimedPlayerId] : null;
            const isClaimable = claimableRouteIds.has(r.id);
            const isHovered = activeHoveredRoute === r.id;

            let isDimmed = false;
            let isHighlighted = false;

            if (focusPlayerMode === 'player_0') {
              if (claimedPlayerId === 'player_0') {
                isHighlighted = true;
              } else {
                isDimmed = true;
              }
            } else if (focusPlayerMode === 'player_1') {
              if (claimedPlayerId === 'player_1') {
                isHighlighted = true;
              } else {
                isDimmed = true;
              }
            }

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
                isDimmed={isDimmed}
                isHighlighted={isHighlighted}
                hoveredMeta={isHovered ? hoveredMeta : null}
                onMouseEnter={() => {
                  setInternalHoveredRoute(r.id);
                  onRouteHover?.(r.id);
                }}
                onMouseLeave={() => {
                  setInternalHoveredRoute(null);
                  onRouteHover?.(null);
                }}
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
            const isHovered = internalHoveredCity === c.name || internalHoveredCity === c.id;

            return (
              <CityNode
                key={c.id}
                city={c}
                x={pos.x}
                y={pos.y}
                isHighlighted={isHighlighted}
                isHovered={isHovered}
                onMouseEnter={() => setInternalHoveredCity(c.name)}
                onMouseLeave={() => setInternalHoveredCity(null)}
                onClick={() => onCityClick?.(c)}
              />
            );
          })}
        </g>
      </svg>
    </div>
  );
};
