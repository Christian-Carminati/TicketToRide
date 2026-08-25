import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useWorkbench } from '../../context';
import { StudioHeader } from './StudioHeader';
import { ScrubberTransportBar } from './ScrubberTransportBar';
import { BoardCanvas } from '../board';
import { BrainInspectorPane } from '../brain';
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
  const isInitializingRef = useRef(false);
  const lastDispatchRef = useRef(0);

  // Initialize interactive game session with concurrency guard
  const initGame = useCallback(async (mapName: 'usa' | 'mini') => {
    if (isInitializingRef.current) return;
    isInitializingRef.current = true;
    try {
      const res = await api.createGame({
        map_name: mapName,
        player_types: ['ppo', 'greedy'],
        seed: 42,
      });
      setGameState(res);

      // Fetch initial brain inspection
      const brain = await api.inspectBrain({
        session_id: res.session_id,
        model_type: 'ppo',
      });
      setBrainData(brain);
    } catch (err) {
      console.error('Failed to init game session:', err);
    } finally {
      isInitializingRef.current = false;
    }
  }, [setGameState, setBrainData]);

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
    if (!state.gameState || isStepping) return;
    setIsStepping(true);
    try {
      const nextState = await api.stepGame(state.gameState.session_id);
      setGameState(nextState);

      // Update brain inspection for new state
      const brain = await api.inspectBrain({
        session_id: nextState.session_id,
        model_type: state.selectedAgentModel === 'dqn' ? 'dqn' : 'ppo',
      });
      setBrainData(brain);
    } catch (err) {
      console.error('Bot step failed:', err);
    } finally {
      setIsStepping(false);
    }
  }, [state.gameState, isStepping, state.selectedAgentModel, setGameState, setBrainData]);

  // Global Keyboard Shortcuts (Alex Power User & Accessibility)
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
  const handleRouteClick = async (route: BoardRoute) => {
    if (!state.gameState || isStepping) return;
    setIsStepping(true);
    try {
      const nextState = await api.stepGame(state.gameState.session_id, {
        action_type: 'CLAIM_ROUTE',
        route_id: route.id,
        card_color: route.color || undefined,
      });
      setGameState(nextState);

      const brain = await api.inspectBrain({
        session_id: nextState.session_id,
        model_type: state.selectedAgentModel === 'dqn' ? 'dqn' : 'ppo',
      });
      setBrainData(brain);
    } catch (err) {
      console.error('Route claim failed:', err);
    } finally {
      setIsStepping(false);
    }
  };

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
          <div className="workbench-main-split">
            {/* Left: Vector Board Canvas & Inventory */}
            <BoardCanvas
              onRouteClick={handleRouteClick}
              onSelectMap={handleMapChange}
            />

            {/* Right: Synced Neural Brain Inspector */}
            <BrainInspectorPane />
          </div>
        )}

        {studioMode === 'replay_scrub' && (
          <div className="workbench-main-split">
            <BoardCanvas onSelectMap={handleMapChange} />
            <BrainInspectorPane />
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

        {/* Collapsible Telemetry Dock (Available in Interactive, Replay, and Training views) */}
        {studioMode !== 'tournament' && studioMode !== 'reports' && (
          <TelemetryTournamentDock />
        )}
      </div>
    </div>
  );
};
