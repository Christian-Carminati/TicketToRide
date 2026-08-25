import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useWorkbench } from '../../context';
import { StudioHeader } from './StudioHeader';
import { ScrubberTransportBar } from './ScrubberTransportBar';
import { BoardCanvas } from '../board';
import { AITelemetryDrawer } from '../brain';
import { TelemetryTournamentDock } from '../dock';
import { TrainingView, TournamentArenaView, ReportsView } from '../../views';
import { api } from '../../api';
import { TelemetryEventDTO } from '../../api/types';
import { useWebSocket } from '../../hooks';
import { BoardRoute } from '../board/mapData';

export const WorkbenchShell: React.FC = () => {
  const {
    state,
    setGameState,
    setBrainData,
    addTelemetryEvent,
    setConnected,
    setStepIndex,
    setIsPlaying,
    setStudioMode,
  } = useWorkbench();

  const { studioMode, currentStepIndex, maxStepIndex, isPlaying } = state;
  const [isStepping, setIsStepping] = useState(false);
  const [selectedMap, setSelectedMap] = useState<'usa' | 'mini'>('usa');
  const [isAutoplaying, setIsAutoplaying] = useState(false);
  const [autoplaySpeedMs, setAutoplaySpeedMs] = useState(600);
  const [claimModalRoute, setClaimModalRoute] = useState<BoardRoute | null>(null);
  const [isTelemetryDrawerOpen, setIsTelemetryDrawerOpen] = useState(false);
  const isInitializingRef = useRef(false);
  const lastDispatchRef = useRef(0);

  // Initialize interactive game session with concurrency guard
  const initGame = useCallback(async (mapName: 'usa' | 'mini') => {
    if (isInitializingRef.current) return;
    isInitializingRef.current = true;
    try {
      const res = await api.createGame({
        map_name: mapName,
        player_types: ['human', 'alphazero'],
        seed: 42,
      });
      setGameState(res);

      // Fetch initial brain inspection
      const brain = await api.inspectBrain({
        session_id: res.session_id,
        model_type: state.selectedAgentModel,
      });
      setBrainData(brain);
    } catch (err) {
      console.error('Failed to init game session:', err);
    } finally {
      isInitializingRef.current = false;
    }
  }, [setGameState, setBrainData, state.selectedAgentModel]);

  // Only run when selectedMap explicitly changes
  useEffect(() => {
    initGame(selectedMap);
  }, [selectedMap, initGame]);

  const handleMapChange = (mapName: 'usa' | 'mini') => {
    setSelectedMap(mapName);
  };

  // Shared WebSocket for Live Telemetry (Throttled global context dispatch)
  const { isConnected, lastMessage } = useWebSocket<TelemetryEventDTO>('ws://localhost:8000/ws/telemetry');

  useEffect(() => {
    setConnected(isConnected);
  }, [isConnected, setConnected]);

  useEffect(() => {
    if (lastMessage && lastMessage.type) {
      const now = performance.now();
      if (lastMessage.type !== 'training_step' || now - lastDispatchRef.current > 100) {
        lastDispatchRef.current = now;
        addTelemetryEvent(lastMessage);
      }
    }
  }, [lastMessage, addTelemetryEvent]);

  // Step Bot
  const handleBotStep = useCallback(async () => {
    if (!state.gameState || isStepping || state.gameState.is_game_over) return;
    setIsStepping(true);
    try {
      const nextState = await api.stepGame(state.gameState.session_id);
      setGameState(nextState);

      // Update brain inspection for new state
      const brain = await api.inspectBrain({
        session_id: nextState.session_id,
        model_type: state.selectedAgentModel,
      });
      setBrainData(brain);
    } catch (err) {
      console.error('Bot step failed:', err);
    } finally {
      setIsStepping(false);
    }
  }, [state.gameState, isStepping, state.selectedAgentModel, setGameState, setBrainData]);

  // Draw Face-Up Card
  const handleDrawVisibleCard = useCallback(async (index: number) => {
    if (!state.gameState || isStepping || state.gameState.is_game_over) return;
    setIsStepping(true);
    try {
      const nextState = await api.stepGame(state.gameState.session_id, {
        action_type: 'DRAW_VISIBLE_CARD',
        card_index: index,
      });
      setGameState(nextState);

      const brain = await api.inspectBrain({
        session_id: nextState.session_id,
        model_type: state.selectedAgentModel,
      });
      setBrainData(brain);
    } catch (err) {
      console.error('Draw visible card failed:', err);
    } finally {
      setIsStepping(false);
    }
  }, [state.gameState, isStepping, state.selectedAgentModel, setGameState, setBrainData]);

  // Draw Hidden Card
  const handleDrawDeckCard = useCallback(async () => {
    if (!state.gameState || isStepping || state.gameState.is_game_over) return;
    setIsStepping(true);
    try {
      const nextState = await api.stepGame(state.gameState.session_id, {
        action_type: 'DRAW_HIDDEN_CARD',
      });
      setGameState(nextState);

      const brain = await api.inspectBrain({
        session_id: nextState.session_id,
        model_type: state.selectedAgentModel,
      });
      setBrainData(brain);
    } catch (err) {
      console.error('Draw hidden card failed:', err);
    } finally {
      setIsStepping(false);
    }
  }, [state.gameState, isStepping, state.selectedAgentModel, setGameState, setBrainData]);

  // Draw Tickets
  const handleDrawTickets = useCallback(async () => {
    if (!state.gameState || isStepping || state.gameState.is_game_over) return;
    setIsStepping(true);
    try {
      const nextState = await api.stepGame(state.gameState.session_id, {
        action_type: 'DRAW_TICKETS',
      });
      setGameState(nextState);

      const brain = await api.inspectBrain({
        session_id: nextState.session_id,
        model_type: state.selectedAgentModel,
      });
      setBrainData(brain);
    } catch (err) {
      console.error('Draw tickets failed:', err);
    } finally {
      setIsStepping(false);
    }
  }, [state.gameState, isStepping, state.selectedAgentModel, setGameState, setBrainData]);

  // Autoplay Continuous Duel loop
  useEffect(() => {
    if (!isAutoplaying || !state.gameState || state.gameState.is_game_over || isStepping) {
      if (state.gameState?.is_game_over && isAutoplaying) {
        setIsAutoplaying(false);
      }
      return;
    }

    const timer = setTimeout(() => {
      handleBotStep();
    }, autoplaySpeedMs);

    return () => clearTimeout(timer);
  }, [isAutoplaying, state.gameState, isStepping, autoplaySpeedMs, handleBotStep]);

  // New Match Setup Handler
  const handleNewMatch = useCallback(async (
    p1Type: string,
    p2Type: string,
    mapName: 'usa' | 'mini',
    ckpt1?: string,
    ckpt2?: string
  ) => {
    setIsAutoplaying(false);
    try {
      const res = await api.createGame({
        map_name: mapName,
        player_types: [p1Type, p2Type],
        player_checkpoints: [ckpt1 || null, ckpt2 || null],
        seed: Math.floor(Math.random() * 10000),
      });
      setGameState(res);
      const brain = await api.inspectBrain({
        session_id: res.session_id,
        model_type: state.selectedAgentModel,
      });
      setBrainData(brain);
    } catch (err) {
      console.error('Failed to create new duel match:', err);
    }
  }, [state.selectedAgentModel, setGameState, setBrainData]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // Mode Selection (1-5)
      if (e.key === '1') setStudioMode('interactive');
      else if (e.key === '2') setStudioMode('training_live');
      else if (e.key === '3') setStudioMode('replay_scrub');
      else if (e.key === '4') setStudioMode('tournament');
      else if (e.key === '5') setStudioMode('reports');

      // Timeline Scrubber & Step Shortcuts
      if (studioMode === 'interactive' || studioMode === 'replay_scrub') {
        if (e.key === 'ArrowLeft') {
          e.preventDefault();
          setStepIndex(Math.max(0, currentStepIndex - 1));
        } else if (e.key === 'ArrowRight') {
          e.preventDefault();
          setStepIndex(Math.min(maxStepIndex, currentStepIndex + 1));
        } else if (e.code === 'Space') {
          e.preventDefault();
          if (studioMode === 'interactive') {
            handleBotStep();
          } else {
            setIsPlaying(!isPlaying);
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [studioMode, currentStepIndex, maxStepIndex, isPlaying, setStudioMode, setStepIndex, setIsPlaying, handleBotStep]);

  // Claim Route Handler (Human interactive move)
  const handleRouteClick = useCallback(async (route: BoardRoute) => {
    if (!state.gameState || isStepping || state.gameState.is_game_over) return;
    const matchingActions = (state.gameState.valid_actions || []).filter(
      (a) => a.action_type === 'CLAIM_ROUTE' && a.route_id === route.id
    );

    if (matchingActions.length === 1) {
      setIsStepping(true);
      try {
        const nextState = await api.stepGame(state.gameState.session_id, matchingActions[0]);
        setGameState(nextState);

        const brain = await api.inspectBrain({
          session_id: nextState.session_id,
          model_type: state.selectedAgentModel,
        });
        setBrainData(brain);
      } catch (err) {
        console.error('Route claim failed:', err);
      } finally {
        setIsStepping(false);
      }
    } else if (matchingActions.length > 1) {
      setClaimModalRoute(route);
    }
  }, [state.gameState, isStepping, state.selectedAgentModel, setGameState, setBrainData]);

  const handleExecuteModalAction = useCallback(async (action: import('../../api/types').ActionDTO) => {
    if (!state.gameState || isStepping || state.gameState.is_game_over) return;
    setIsStepping(true);
    setClaimModalRoute(null);
    try {
      const nextState = await api.stepGame(state.gameState.session_id, action);
      setGameState(nextState);

      const brain = await api.inspectBrain({
        session_id: nextState.session_id,
        model_type: state.selectedAgentModel,
      });
      setBrainData(brain);
    } catch (err) {
      console.error('Modal route claim failed:', err);
    } finally {
      setIsStepping(false);
    }
  }, [state.gameState, isStepping, state.selectedAgentModel, setGameState, setBrainData]);

  return (
    <div
      className="workbench-shell"
      style={{
        minHeight: '100vh',
        background: 'radial-gradient(circle at 50% 50%, #3D281A 0%, #1F140E 100%)',
        color: '#23140C',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: "'Crimson Pro', Georgia, serif",
      }}
    >
      {/* Studio Master Header */}
      <StudioHeader onNewGame={() => initGame(selectedMap)} />

      {/* Main Studio Body */}
      <div
        style={{
          flex: 1,
          padding: '0.75rem 1.25rem',
          maxWidth: '1800px',
          width: '100%',
          margin: '0 auto',
          boxSizing: 'border-box',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem',
        }}
      >
        {/* Scrubber Transport Bar (Only in Interactive and Replay modes) */}
        {(studioMode === 'interactive' || studioMode === 'replay_scrub') && (
          <ScrubberTransportBar
            onBotStep={handleBotStep}
            isStepping={isStepping}
          />
        )}

        {/* Studio Mode Views */}
        {studioMode === 'interactive' && (
          <div className="workbench-single-pane" style={{ width: '100%' }}>
            <BoardCanvas
              onRouteClick={handleRouteClick}
              onDrawDeckCard={handleDrawDeckCard}
              onDrawVisibleCard={handleDrawVisibleCard}
              onDrawTickets={handleDrawTickets}
              onSelectMap={handleMapChange}
              onNewMatch={handleNewMatch}
              isAutoplaying={isAutoplaying}
              onToggleAutoplay={() => setIsAutoplaying(!isAutoplaying)}
              autoplaySpeedMs={autoplaySpeedMs}
              onSpeedChange={(s) => setAutoplaySpeedMs(s)}
              onToggleTelemetryDrawer={() => setIsTelemetryDrawerOpen(!isTelemetryDrawerOpen)}
              isTelemetryDrawerOpen={isTelemetryDrawerOpen}
              onSwitchToReplay={() => setStudioMode('replay_scrub')}
            />
          </div>
        )}

        {studioMode === 'replay_scrub' && (
          <div className="workbench-single-pane" style={{ width: '100%' }}>
            <BoardCanvas
              onSelectMap={handleMapChange}
              onNewMatch={handleNewMatch}
              isAutoplaying={isAutoplaying}
              onToggleAutoplay={() => setIsAutoplaying(!isAutoplaying)}
              autoplaySpeedMs={autoplaySpeedMs}
              onSpeedChange={(s) => setAutoplaySpeedMs(s)}
              onToggleTelemetryDrawer={() => setIsTelemetryDrawerOpen(!isTelemetryDrawerOpen)}
              isTelemetryDrawerOpen={isTelemetryDrawerOpen}
              onSwitchToReplay={() => setStudioMode('replay_scrub')}
            />
          </div>
        )}

        {studioMode === 'training_live' && (
          <div className="steampunk-panel" style={{ padding: '1rem' }}>
            <TrainingView />
          </div>
        )}

        {studioMode === 'tournament' && (
          <TournamentArenaView />
        )}

        {studioMode === 'reports' && (
          <ReportsView />
        )}

        {/* Collapsible Telemetry Dock */}
        {studioMode !== 'tournament' && studioMode !== 'reports' && (
          <TelemetryTournamentDock />
        )}
      </div>

      {/* Slide-Out AI Telemetry Inspection Drawer */}
      <AITelemetryDrawer
        isOpen={isTelemetryDrawerOpen}
        onClose={() => setIsTelemetryDrawerOpen(false)}
      />

      {/* Claim Route Color Selection Modal */}
      {claimModalRoute && state.gameState && (
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
              Reclama Tratta Ferroviaria
            </h3>
            <p style={{ color: '#5A3822', fontSize: '0.9rem', marginBottom: '1rem', fontFamily: "'Crimson Pro', Georgia, serif" }}>
              Scegli quali carte treno spendere per {claimModalRoute.city_a} ⟷ {claimModalRoute.city_b}:
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1.25rem' }}>
              {(state.gameState.valid_actions || [])
                .filter((a) => a.action_type === 'CLAIM_ROUTE' && a.route_id === claimModalRoute.id)
                .map((a, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleExecuteModalAction(a)}
                    className="steampunk-btn"
                    style={{
                      padding: '0.6rem 1rem',
                      textAlign: 'left',
                      fontSize: '0.85rem',
                    }}
                  >
                    Spendi {claimModalRoute.length} {a.card_color?.toUpperCase() || 'CARTE'}
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
              Annulla
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
