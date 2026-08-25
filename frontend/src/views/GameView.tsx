import React, { useState, useEffect } from 'react';
import { useGameSession } from '../hooks/useGameSession';
import { BoardSVG } from '../components/board/BoardSVG';
import { BoardRoute } from '../components/board/mapData';
import { TrainCardHand } from '../components/cards/TrainCardHand';
import { VisibleDeck } from '../components/cards/VisibleDeck';
import { TicketsList } from '../components/cards/TicketsList';
import { ActionDTO, CheckpointDTO } from '../api/types';
import { api } from '../api/client';

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

  const [player1Type, setPlayer1Type] = useState<string>('human');
  const [player2Type, setPlayer2Type] = useState<string>('ppo');
  const [checkpoints, setCheckpoints] = useState<CheckpointDTO[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('');
  const [highlightedCities, setHighlightedCities] = useState<string[]>([]);
  const [claimModalRoute, setClaimModalRoute] = useState<BoardRoute | null>(null);

  // Fetch available checkpoints
  useEffect(() => {
    api.listCheckpoints()
      .then((data) => {
        setCheckpoints(data);
        if (data.length > 0) {
          setSelectedCheckpoint(data[0].path);
        }
      })
      .catch((err) => console.error('Failed to list checkpoints:', err));
  }, []);

  // Auto-initialize a USA game session on first mount if none exists
  useEffect(() => {
    if (!gameState && !isLoading) {
      createGame({
        map_name: 'usa',
        player_types: ['human', 'ppo'],
        seed: 42,
        model_checkpoint: selectedCheckpoint || undefined,
      });
    }
  }, [gameState, isLoading, createGame, selectedCheckpoint]);

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
      map_name: 'usa',
      player_types: [player1Type, player2Type],
      seed: Math.floor(Math.random() * 10000),
      model_checkpoint: selectedCheckpoint || undefined,
    });
  };

  const isHumanTurn =
    gameState &&
    !gameState.is_game_over &&
    gameState.players[gameState.current_player_index]?.name.toLowerCase().includes('human');

  const handleRouteClick = (route: BoardRoute) => {
    if (!isHumanTurn || !gameState) return;
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
    <div className="game-view" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Top Configuration & Control Bar */}
      <div
        className="control-bar steampunk-panel"
        style={{
          padding: '0.85rem 1.25rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: '#FAF0DA',
              border: '1.5px solid #C59B27',
              padding: '0.3rem 0.65rem',
              borderRadius: '6px',
              fontSize: '0.82rem',
              color: '#23140C',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: 800,
            }}
          >
            <span>🗺️ USA Cartographical Survey (36 Cities, 100 Routes)</span>
          </div>

          <label style={{ fontSize: '0.82rem', color: '#4A2F1D', fontWeight: 700, fontFamily: "'Crimson Pro', serif", marginLeft: '0.3rem' }}>P1:</label>
          <select
            value={player1Type}
            onChange={(e) => setPlayer1Type(e.target.value)}
            style={{
              backgroundColor: '#FAF5EB',
              color: '#23140C',
              border: '1.5px solid #8C6305',
              borderRadius: '6px',
              padding: '0.3rem 0.6rem',
              fontSize: '0.82rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: 700,
            }}
          >
            <option value="human">Human Player</option>
            <option value="ppo">PPO Agent</option>
            <option value="dqn">DQN Agent</option>
            <option value="strategic">StrategicBot</option>
            <option value="greedy">GreedyBot</option>
            <option value="random">RandomBot</option>
          </select>

          <label style={{ fontSize: '0.82rem', color: '#4A2F1D', fontWeight: 700, fontFamily: "'Crimson Pro', serif" }}>vs P2:</label>
          <select
            value={player2Type}
            onChange={(e) => setPlayer2Type(e.target.value)}
            style={{
              backgroundColor: '#FAF5EB',
              color: '#23140C',
              border: '1.5px solid #8C6305',
              borderRadius: '6px',
              padding: '0.3rem 0.6rem',
              fontSize: '0.82rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: 700,
            }}
          >
            <option value="ppo">PPO Agent</option>
            <option value="dqn">DQN Agent</option>
            <option value="greedy">GreedyBot</option>
            <option value="strategic">StrategicBot</option>
            <option value="random">RandomBot</option>
            <option value="human">Human Player</option>
          </select>

          {(player1Type === 'ppo' || player1Type === 'dqn' || player2Type === 'ppo' || player2Type === 'dqn') && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <label style={{ fontSize: '0.82rem', color: '#9E6B00', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>🤖 Checkpoint:</label>
              <select
                value={selectedCheckpoint}
                onChange={(e) => setSelectedCheckpoint(e.target.value)}
                style={{
                  backgroundColor: '#FAF5EB',
                  color: '#23140C',
                  border: '1.5px solid #8C6305',
                  borderRadius: '6px',
                  padding: '0.3rem 0.6rem',
                  fontSize: '0.82rem',
                  fontWeight: 700,
                  maxWidth: '260px',
                  fontFamily: "'Courier Prime', monospace",
                }}
              >
                {checkpoints.map((ckpt) => (
                  <option key={ckpt.checkpoint_id} value={ckpt.path}>
                    {ckpt.name} ({ckpt.size_mb} MB)
                  </option>
                ))}
              </select>
            </div>
          )}

          <button
            onClick={handleStartNewGame}
            disabled={isLoading}
            className="steampunk-btn"
            style={{ padding: '0.35rem 0.9rem', fontSize: '0.82rem' }}
          >
            🎮 New Match
          </button>
        </div>

        {/* Step & Autoplay Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            onClick={() => stepGame(null)}
            disabled={isLoading || isAutoPlaying || Boolean(gameState?.is_game_over)}
            className="steampunk-btn"
            style={{
              background: 'linear-gradient(180deg, #86EFAC 0%, #16A34A 50%, #14532D 100%)',
              color: '#FFFFFF',
              border: '1px solid #14532D',
              padding: '0.35rem 0.8rem',
              fontSize: '0.82rem',
              opacity: isAutoPlaying ? 0.5 : 1,
            }}
          >
            ▶ Step Turn
          </button>

          <button
            onClick={toggleAutoPlay}
            disabled={Boolean(gameState?.is_game_over)}
            className="steampunk-btn"
            style={{
              background: isAutoPlaying
                ? 'linear-gradient(180deg, #F87171 0%, #DC2626 50%, #991B1B 100%)'
                : 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)',
              color: isAutoPlaying ? '#FFFFFF' : '#23140C',
              padding: '0.35rem 0.8rem',
              fontSize: '0.82rem',
            }}
          >
            {isAutoPlaying ? '⏸ Pause' : '▶▶ Autoplay'}
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.78rem', color: '#5A3822', fontFamily: "'Courier Prime', monospace" }}>
            <span>Rate:</span>
            <input
              type="range"
              min="100"
              max="1500"
              step="100"
              value={playSpeedMs}
              onChange={(e) => setPlaySpeedMs(Number(e.target.value))}
              style={{ width: '70px', accentColor: '#B8860B' }}
            />
            <span>{playSpeedMs}ms</span>
          </div>
        </div>
      </div>

      {error && (
        <div style={{ padding: '0.75rem', borderRadius: '8px', background: '#FEE2E2', border: '1.5px solid #DC2626', color: '#991B1B', fontSize: '0.85rem', fontFamily: "'Playfair Display', serif" }}>
          ⚠️ {error}
        </div>
      )}

      {/* Game Over Banner */}
      {gameState?.is_game_over && (
        <div
          className="steampunk-panel"
          style={{
            padding: '1.25rem',
            border: '3px solid #B8860B',
            textAlign: 'center',
            background: 'linear-gradient(180deg, #FAF3E6 0%, #EADBBE 100%)',
          }}
        >
          <h3 style={{ margin: '0 0 0.5rem 0', color: '#23140C', fontFamily: "'Cinzel Decorative', Georgia, serif", fontSize: '1.3rem' }}>
            🏆 Match Concluded — Final Tally
          </h3>
          <p style={{ margin: 0, color: '#4A2F1D', fontSize: '1rem', fontFamily: "'Playfair Display', Georgia, serif", fontWeight: 700 }}>
            Final Scores: {gameState.players.map((p) => `${p.name}: ${p.score} pts`).join(' | ')}
          </p>
        </div>
      )}

      {/* Main Game Grid Layout */}
      <div className="game-view-split">
        {/* Left Column: Board & Decks */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <BoardSVG
            mapName={gameState?.map_name || 'usa'}
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

        {/* Right Column: Player Hands & Telegrams */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {/* Turn Status Badge */}
          <div
            className="steampunk-panel"
            style={{
              padding: '0.75rem 1rem',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div>
              <span style={{ fontSize: '0.78rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>Turn #{gameState?.turn_number || 1}</span>
              <div style={{ fontSize: '0.95rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                Active Turn: <span style={{ color: currentPlayer?.color || '#9E6B00' }}>{currentPlayer?.name}</span>
              </div>
            </div>
            {gameState?.last_reward !== null && gameState?.last_reward !== undefined && (
              <div style={{ textAlign: 'right' }}>
                <span style={{ fontSize: '0.72rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>Last Reward</span>
                <div style={{ fontSize: '1rem', fontWeight: 800, fontFamily: "'Courier Prime', monospace", color: (gameState.last_reward || 0) >= 0 ? '#15803D' : '#B91C1C' }}>
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

      {/* Claim Route Color Selection Modal */}
      {claimModalRoute && gameState && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(38, 24, 15, 0.75)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
            backdropFilter: 'blur(4px)',
          }}
        >
          <div
            className="steampunk-panel"
            style={{
              padding: '1.5rem',
              maxWidth: '420px',
              width: '100%',
              border: '3px solid #C59B27',
            }}
          >
            <h3 style={{ margin: '0 0 0.5rem 0', color: '#23140C', fontFamily: "'Cinzel Decorative', Georgia, serif" }}>
              Claim Railway Track
            </h3>
            <p style={{ color: '#5A3822', fontSize: '0.9rem', marginBottom: '1rem', fontFamily: "'Crimson Pro', Georgia, serif" }}>
              Choose which stock color to spend for {claimModalRoute.city_a} ⟷ {claimModalRoute.city_b}:
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1.25rem' }}>
              {gameState.valid_actions
                .filter((a) => a.action_type === 'CLAIM_ROUTE' && a.route_id === claimModalRoute.id)
                .map((a, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      stepGame(a);
                      setClaimModalRoute(null);
                    }}
                    className="steampunk-btn"
                    style={{
                      padding: '0.6rem 1rem',
                      textAlign: 'left',
                      fontSize: '0.85rem',
                    }}
                  >
                    Spend {claimModalRoute.length} {a.card_color || 'Cards'}
                  </button>
                ))}
            </div>

            <button
              onClick={() => setClaimModalRoute(null)}
              className="steampunk-btn"
              style={{
                width: '100%',
                padding: '0.45rem',
                background: 'linear-gradient(180deg, #D4C09D 0%, #A88D75 100%)',
                color: '#23140C',
                border: '1px solid #785A42',
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
