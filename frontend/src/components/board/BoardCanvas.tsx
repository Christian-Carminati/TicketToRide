import React from 'react';
import { useWorkbench } from '../../context';
import { BoardSVG } from './BoardSVG';
import { BoardHeaderControls } from './BoardHeaderControls';
import { BoardRoute, BoardCity } from './mapData';
import { TrainCardHand, VisibleDeck } from '../cards';

interface BoardCanvasProps {
  onRouteClick?: (route: BoardRoute) => void;
  onCityClick?: (city: BoardCity) => void;
  onDrawDeckCard?: () => void;
  onDrawVisibleCard?: (index: number) => void;
  onDrawTickets?: () => void;
  onSelectMap?: (mapName: 'usa' | 'mini') => void;
}

export const BoardCanvas: React.FC<BoardCanvasProps> = React.memo(({
  onRouteClick,
  onCityClick,
  onDrawDeckCard,
  onDrawVisibleCard,
  onDrawTickets,
  onSelectMap,
}) => {
  const { state, setHoveredRouteId } = useWorkbench();
  const gameState = state.gameState;

  const activePlayer = gameState && gameState.players && gameState.players[gameState.current_player_index];

  return (
    <div
      className="board-canvas-workbench"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem',
        background: 'rgba(15, 23, 42, 0.6)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '12px',
        padding: '0.75rem',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)',
      }}
    >
      {/* Header controls with partial-observability & map selector */}
      <BoardHeaderControls onSelectMap={onSelectMap} />

      {/* Interactive Vector Board */}
      <div style={{ position: 'relative', width: '100%' }}>
        <BoardSVG
          mapName={gameState?.map_name || 'usa'}
          claimedRoutes={gameState?.claimed_routes}
          players={gameState?.players}
          validActions={gameState?.valid_actions}
          observabilityMode={state.observabilityMode}
          hoveredRouteId={state.hoveredRouteId}
          hoveredMeta={state.hoveredAction}
          onRouteClick={onRouteClick}
          onCityClick={onCityClick}
          onRouteHover={(routeId) => setHoveredRouteId(routeId)}
        />
      </div>

      {/* Card Table & Deck Track */}
      {gameState && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(280px, 1fr) minmax(280px, 1fr)',
            gap: '0.75rem',
            marginTop: '0.25rem',
          }}
        >
          {/* Visible Face-Up Cards & Draw Options */}
          <div>
            <VisibleDeck
              visibleCards={gameState.visible_cards}
              deckSize={gameState.deck_size}
              discardSize={gameState.discard_pile_size}
              ticketsDeckSize={gameState.tickets_deck_size}
              onDrawHidden={onDrawDeckCard}
              onDrawVisible={onDrawVisibleCard}
              onDrawTickets={onDrawTickets}
            />
          </div>

          {/* Active Player Hand / Agent Inventory */}
          <div>
            {activePlayer && (
              <TrainCardHand
                player={activePlayer}
                isCurrentTurn={true}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
});
