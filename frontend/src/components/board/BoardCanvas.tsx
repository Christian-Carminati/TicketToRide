import React, { useState } from 'react';
import { useWorkbench } from '../../context';
import { BoardSVG } from './BoardSVG';
import { MatchScoreBoardHUD } from './MatchScoreBoardHUD';
import { GameOverModal } from './GameOverModal';
import { BoardRoute, BoardCity } from './mapData';
import { TrainCardHand, VisibleDeck, TicketsList } from '../cards';

interface BoardCanvasProps {
  onRouteClick?: (route: BoardRoute) => void;
  onCityClick?: (city: BoardCity) => void;
  onDrawDeckCard?: () => void;
  onDrawVisibleCard?: (index: number) => void;
  onDrawTickets?: () => void;
  onSelectMap?: (mapName: 'usa' | 'mini') => void;
  onNewMatch?: (p1Type: string, p2Type: string, map: 'usa' | 'mini', ckpt1?: string, ckpt2?: string) => void;
  isAutoplaying?: boolean;
  onToggleAutoplay?: () => void;
  autoplaySpeedMs?: number;
  onSpeedChange?: (speed: number) => void;
  onToggleTelemetryDrawer?: () => void;
  isTelemetryDrawerOpen?: boolean;
  onSwitchToReplay?: () => void;
}

export const BoardCanvas: React.FC<BoardCanvasProps> = React.memo(({
  onRouteClick,
  onCityClick,
  onDrawDeckCard,
  onDrawVisibleCard,
  onDrawTickets,
  onSelectMap,
  onNewMatch,
  isAutoplaying,
  onToggleAutoplay,
  autoplaySpeedMs,
  onSpeedChange,
  onToggleTelemetryDrawer,
  isTelemetryDrawerOpen,
  onSwitchToReplay,
}) => {
  const { state, setHoveredRouteId } = useWorkbench();
  const gameState = state.gameState;
  const [isGameOverModalDismissed, setIsGameOverModalDismissed] = useState(false);

  const activePlayer = gameState && gameState.players && gameState.players[gameState.current_player_index];
  const isHumanTurn =
    activePlayer &&
    !gameState?.is_game_over &&
    (activePlayer.name.toLowerCase().includes('human') || gameState?.current_player_index === 0);

  const handleRematch = () => {
    setIsGameOverModalDismissed(false);
    onNewMatch?.('human', 'alphazero', (gameState?.map_name as any) || 'usa');
  };

  return (
    <div
      className="board-canvas-workbench steampunk-panel"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.75rem',
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: '2px solid #C59B27',
        borderRadius: '14px',
        padding: '1rem',
        boxShadow: '0 6px 24px rgba(0, 0, 0, 0.35)',
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      {/* 1. Master Match Scoreboard HUD */}
      <MatchScoreBoardHUD
        gameState={gameState}
        isAutoplaying={isAutoplaying}
        onToggleAutoplay={onToggleAutoplay}
        autoplaySpeedMs={autoplaySpeedMs}
        onSpeedChange={onSpeedChange}
        onSelectMap={onSelectMap}
        onNewMatch={onNewMatch}
        onToggleTelemetryDrawer={onToggleTelemetryDrawer}
        isTelemetryDrawerOpen={isTelemetryDrawerOpen}
        selectedModel={state.selectedAgentModel}
      />

      {/* 2. Interactive Full-Width Vector Board */}
      <div
        className="board-svg-container"
        style={{
          position: 'relative',
          width: '100%',
          borderRadius: '10px',
          overflow: 'hidden',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2), inset 0 0 20px rgba(110, 70, 30, 0.1)',
        }}
      >
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

      {/* 3. Lower Station Platform: 3-Column Balanced Game Table */}
      {gameState && (
        <div
          className="lower-station-platform"
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(280px, 1.1fr) minmax(280px, 1.2fr) minmax(240px, 0.9fr)',
            gap: '0.75rem',
            alignItems: 'stretch',
          }}
        >
          {/* Column 1: Visible Face-Up Vouchers & Draw Decks */}
          <VisibleDeck
            visibleCards={gameState.visible_cards}
            deckSize={gameState.deck_size}
            discardSize={gameState.discard_pile_size}
            ticketsDeckSize={gameState.tickets_deck_size}
            isHumanTurn={Boolean(isHumanTurn)}
            onDrawHidden={onDrawDeckCard}
            onDrawVisible={onDrawVisibleCard}
            onDrawTickets={onDrawTickets}
          />

          {/* Column 2: Active Conductor Hand */}
          {activePlayer && (
            <TrainCardHand
              player={activePlayer}
              isCurrentTurn={true}
            />
          )}

          {/* Column 3: Active Destination Tickets */}
          {activePlayer && (
            <TicketsList
              tickets={activePlayer.tickets}
              playerName={activePlayer.name}
            />
          )}
        </div>
      )}

      {/* 4. Game Over Celebration Modal */}
      {gameState?.is_game_over && !isGameOverModalDismissed && (
        <GameOverModal
          gameState={gameState}
          onRematch={handleRematch}
          onClose={() => setIsGameOverModalDismissed(true)}
          onSwitchToReplay={onSwitchToReplay}
        />
      )}
    </div>
  );
});
