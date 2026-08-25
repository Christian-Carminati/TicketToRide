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
      className="board-canvas-workbench steampunk-panel"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.6rem',
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: '2px solid #C59B27',
        borderRadius: '12px',
        padding: '0.85rem',
        boxShadow: '0 6px 24px rgba(0, 0, 0, 0.35)',
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
