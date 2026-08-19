import React, { useState, useEffect } from 'react';
import { useGameSession } from '../hooks/useGameSession';
import { BoardSVG } from '../components/board/BoardSVG';
import { BoardRoute } from '../components/board/mapData';
import { TrainCardHand } from '../components/cards/TrainCardHand';
import { VisibleDeck } from '../components/cards/VisibleDeck';
import { TicketsList } from '../components/cards/TicketsList';
import { ActionDTO } from '../api/types';

export const GameView: React.FC = () => {
  const {
    gameState,
    isLoading,
    error,
    isAutoPlaying,
    playSpeedMs,
    setPlaySpeedMs,
    createGame,
    stepGame,
    toggleAutoPlay,
  } = useGameSession();

  const [mapName, setMapName] = useState<'mini' | 'usa'>('mini');
  const [player1Type, setPlayer1Type] = useState<string>('human');
  const [player2Type, setPlayer2Type] = useState<string>('greedy');
  const [highlightedCities, setHighlightedCities] = useState<string[]>([]);
  const [claimModalRoute, setClaimModalRoute] = useState<BoardRoute | null>(null);

  // Auto-initialize a mini game session on first mount if none exists
  useEffect(() => {
    if (!gameState && !isLoading) {
      createGame({ map_name: 'mini', player_types: ['human', 'greedy'], seed: 42 });
    }
  }, [gameState, isLoading, createGame]);

  // Keyboard shortcut: Spacebar steps turn
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.code === 'Space') {
        e.preventDefault();
        if (!gameState?.is_game_over && !isLoading) {
          stepGame(null);
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [gameState, isLoading, stepGame]);

  const handleStartNewGame = () => {
    createGame({
      map_name: mapName,
      player_types: [player1Type, player2Type],
      seed: Math.floor(Math.random() * 10000),
    });
  };

  const isHumanTurn =
    gameState &&
    !gameState.is_game_over &&
    gameState.players[gameState.current_player_index]?.name.toLowerCase().includes('human');

  const handleRouteClick = (route: BoardRoute) => {
    if (!isHumanTurn || !gameState) return;
    // Check if there is a valid CLAIM_ROUTE action for this route
    const matchingActions = gameState.valid_actions.filter(
      (a) => a.action_type === 'CLAIM_ROUTE' && a.route_id === route.id
    );

    if (matchingActions.length === 1) {
      stepGame(matchingActions[0]);
    } else if (matchingActions.length > 1) {
      setClaimModalRoute(route);
    }
  };

  const handleDrawVisible = (slotIndex: number) => {
    if (!isHumanTurn) return;
    const act: ActionDTO = {
      action_type: 'DRAW_VISIBLE_CARD',
      card_index: slotIndex,
    };
    stepGame(act);
  };

  const handleDrawHidden = () => {
    if (!isHumanTurn) return;
    const act: ActionDTO = {
      action_type: 'DRAW_HIDDEN_CARD',
    };
    stepGame(act);
  };

  const handleDrawTickets = () => {
    if (!isHumanTurn) return;
    const act: ActionDTO = {
      action_type: 'DRAW_TICKETS',
    };
    stepGame(act);
  };

  const currentPlayer = gameState?.players[gameState.current_player_index];

  return (
    <div className="game-view" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Top Configuration & Control Bar */}
      <div
        className="control-bar"
        style={{
          background: 'rgba(15, 23, 42, 0.9)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '12px',
          padding: '1rem 1.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Map:</label>
          <select
            value={mapName}
            onChange={(e) => setMapName(e.target.value as 'mini' | 'usa')}
            style={{
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.6rem',
              fontSize: '0.85rem',
            }}
          >
            <option value="mini">Mini (5 cities, 6 routes)</option>
            <option value="usa">USA Official (36 cities, 100 routes)</option>
          </select>

          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600, marginLeft: '0.5rem' }}>P1:</label>
          <select
            value={player1Type}
            onChange={(e) => setPlayer1Type(e.target.value)}
            style={{
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.6rem',
              fontSize: '0.85rem',
            }}
          >
            <option value="human">Human Player</option>
            <option value="ppo">PPO Agent</option>
            <option value="dqn">DQN Agent</option>
            <option value="strategic">StrategicBot</option>
            <option value="greedy">GreedyBot</option>
            <option value="random">RandomBot</option>
          </select>

          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>vs P2:</label>
          <select
            value={player2Type}
            onChange={(e) => setPlayer2Type(e.target.value)}
            style={{
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.6rem',
              fontSize: '0.85rem',
            }}
          >
            <option value="greedy">GreedyBot</option>
            <option value="strategic">StrategicBot</option>
            <option value="ppo">PPO Agent</option>
            <option value="dqn">DQN Agent</option>
            <option value="random">RandomBot</option>
            <option value="human">Human Player</option>
          </select>

          <button
            onClick={handleStartNewGame}
            disabled={isLoading}
            style={{
              backgroundColor: '#3B82F6',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.4rem 0.8rem',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
            }}
          >
            New Game
          </button>
        </div>

        {/* Step & Autoplay Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            onClick={() => stepGame(null)}
            disabled={isLoading || isAutoPlaying || Boolean(gameState?.is_game_over)}
            style={{
              backgroundColor: '#10B981',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.4rem 0.8rem',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
              opacity: isAutoPlaying ? 0.5 : 1,
            }}
          >
            ▶ Step Next Turn
          </button>

          <button
            onClick={toggleAutoPlay}
            disabled={Boolean(gameState?.is_game_over)}
            style={{
              backgroundColor: isAutoPlaying ? '#EF4444' : '#6366F1',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.4rem 0.8rem',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
            }}
          >
            {isAutoPlaying ? '⏸ Pause' : '▶▶ Autoplay'}
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem', color: '#94A3B8' }}>
            <span>Speed:</span>
            <input
              type="range"
              min="100"
              max="1500"
              step="100"
              value={playSpeedMs}
              onChange={(e) => setPlaySpeedMs(Number(e.target.value))}
              style={{ width: '80px' }}
            />
            <span>{playSpeedMs}ms</span>
          </div>
        </div>
      </div>

      {error && (
        <div style={{ padding: '0.75rem', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.2)', border: '1px solid #EF4444', color: '#FCA5A5', fontSize: '0.85rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Game Over Banner */}
      {gameState?.is_game_over && (
        <div
          style={{
            padding: '1.25rem',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(16, 185, 129, 0.2) 100%)',
            border: '2px solid #F59E0B',
            textAlign: 'center',
          }}
        >
          <h3 style={{ margin: '0 0 0.5rem 0', color: '#FCD34D' }}>🏆 Game Over!</h3>
          <p style={{ margin: 0, color: '#F1F5F9', fontSize: '1rem' }}>
            Final Scores: {gameState.players.map((p) => `${p.name}: ${p.score} pts`).join(' | ')}
          </p>
        </div>
      )}

      {/* Main Game Grid Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 340px', gap: '1.25rem' }}>
        {/* Left Column: Board & Cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <BoardSVG
            mapName={gameState?.map_name || 'mini'}
            claimedRoutes={gameState?.claimed_routes || {}}
            players={gameState?.players || []}
            validActions={gameState?.valid_actions || []}
            highlightedCities={highlightedCities}
            onRouteClick={handleRouteClick}
          />

          {gameState && (
            <VisibleDeck
              visibleCards={gameState.visible_cards}
              deckSize={gameState.deck_size}
              discardSize={gameState.discard_pile_size}
              ticketsDeckSize={gameState.tickets_deck_size}
              isHumanTurn={Boolean(isHumanTurn)}
              onDrawVisible={handleDrawVisible}
              onDrawHidden={handleDrawHidden}
              onDrawTickets={handleDrawTickets}
            />
          )}
        </div>

        {/* Right Column: Player Hands & Ticket Objectives */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Turn Status Badge */}
          <div
            style={{
              padding: '0.75rem 1rem',
              borderRadius: '10px',
              background: 'rgba(15, 23, 42, 0.85)',
              border: '1px solid rgba(255,255,255,0.08)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div>
              <span style={{ fontSize: '0.8rem', color: '#94A3B8' }}>Turn #{gameState?.turn_number || 1}</span>
              <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#F1F5F9' }}>
                Active: <span style={{ color: currentPlayer?.color || '#38BDF8' }}>{currentPlayer?.name}</span>
              </div>
            </div>
            {gameState?.last_reward !== null && gameState?.last_reward !== undefined && (
              <div style={{ textAlign: 'right' }}>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Last Reward</span>
                <div style={{ fontSize: '0.95rem', fontWeight: 700, color: (gameState.last_reward || 0) >= 0 ? '#10B981' : '#EF4444' }}>
                  {(gameState.last_reward || 0) > 0 ? `+${gameState.last_reward}` : gameState.last_reward}
                </div>
              </div>
            )}
          </div>

          {/* Players Hand Panels */}
          {gameState?.players.map((p, idx) => (
            <TrainCardHand
              key={p.player_id}
              player={p}
              isCurrentTurn={idx === gameState.current_player_index}
            />
          ))}

          {/* Current Player's Destination Tickets */}
          {currentPlayer && (
            <TicketsList
              tickets={currentPlayer.tickets}
              playerName={currentPlayer.name}
              onCityHighlight={setHighlightedCities}
            />
          )}
        </div>
      </div>

      {/* Claim Route Color Selection Modal (if multiple colors valid) */}
      {claimModalRoute && gameState && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0,0,0,0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
          }}
        >
          <div
            style={{
              background: '#1E293B',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: '12px',
              padding: '1.5rem',
              maxWidth: '400px',
              width: '100%',
            }}
          >
            <h3 style={{ margin: '0 0 0.5rem 0', color: '#F1F5F9' }}>Claim Route</h3>
            <p style={{ color: '#94A3B8', fontSize: '0.9rem', marginBottom: '1rem' }}>
              Choose which color cards to spend for {claimModalRoute.city_a} ⟷ {claimModalRoute.city_b}:
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1rem' }}>
              {gameState.valid_actions
                .filter((a) => a.action_type === 'CLAIM_ROUTE' && a.route_id === claimModalRoute.id)
                .map((a, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      stepGame(a);
                      setClaimModalRoute(null);
                    }}
                    style={{
                      padding: '0.6rem 1rem',
                      borderRadius: '8px',
                      background: '#334155',
                      color: '#F8FAFC',
                      border: '1px solid rgba(255,255,255,0.1)',
                      cursor: 'pointer',
                      fontWeight: 600,
                      textAlign: 'left',
                    }}
                  >
                    Spend {claimModalRoute.length} {a.card_color || 'Cards'}
                  </button>
                ))}
            </div>

            <button
              onClick={() => setClaimModalRoute(null)}
              style={{
                width: '100%',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'transparent',
                border: '1px solid rgba(255,255,255,0.2)',
                color: '#94A3B8',
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
