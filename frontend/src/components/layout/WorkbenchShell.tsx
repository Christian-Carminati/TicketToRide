import React, { useEffect, useState, useCallback } from 'react';
import { useWorkbench } from '../../context';
import { StudioHeader } from './StudioHeader';
import { ScrubberTransportBar } from './ScrubberTransportBar';
import { BoardCanvas } from '../board';
import { BrainInspectorPane } from '../brain';
import { TelemetryTournamentDock } from '../dock';
import { TrainingView, ExperimentView } from '../../views';
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
  } = useWorkbench();

  const { studioMode } = state;
  const [isStepping, setIsStepping] = useState(false);

  // Initialize interactive game session
  const initGame = useCallback(async () => {
    try {
      const res = await api.createGame({
        map_name: 'mini',
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
    }
  }, [setGameState, setBrainData]);

  useEffect(() => {
    initGame();
  }, [initGame]);

  // WebSocket for Live Telemetry
  const { isConnected, lastMessage } = useWebSocket<TelemetryEventDTO>('ws://localhost:8000/ws/telemetry');

  useEffect(() => {
    setConnected(isConnected);
  }, [isConnected, setConnected]);

  useEffect(() => {
    if (lastMessage && lastMessage.type) {
      addTelemetryEvent(lastMessage);
    }
  }, [lastMessage, addTelemetryEvent]);

  // Step Bot
  const handleBotStep = async () => {
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
  };

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
        background: '#090D16',
        color: '#F8FAFC',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      }}
    >
      {/* Studio Master Header */}
      <StudioHeader onNewGame={initGame} />

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
        {/* Scrubber Transport Bar */}
        <ScrubberTransportBar
          onBotStep={handleBotStep}
          isStepping={isStepping}
        />

        {/* Studio Mode Views */}
        {studioMode === 'interactive' && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(600px, 1.4fr) minmax(400px, 1fr)',
              gap: '0.75rem',
              alignItems: 'start',
            }}
          >
            {/* Left: Vector Board Canvas & Inventory */}
            <BoardCanvas
              onRouteClick={handleRouteClick}
            />

            {/* Right: Synced Neural Brain Inspector */}
            <BrainInspectorPane />
          </div>
        )}

        {studioMode === 'replay_scrub' && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(600px, 1.4fr) minmax(400px, 1fr)',
              gap: '0.75rem',
              alignItems: 'start',
            }}
          >
            <BoardCanvas />
            <BrainInspectorPane />
          </div>
        )}

        {studioMode === 'training_live' && (
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.08)', padding: '1rem' }}>
            <TrainingView />
          </div>
        )}

        {studioMode === 'tournament' && (
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.08)', padding: '1rem' }}>
            <ExperimentView />
          </div>
        )}

        {/* Collapsible Telemetry & Tournament Dock */}
        <TelemetryTournamentDock />
      </div>
    </div>
  );
};
