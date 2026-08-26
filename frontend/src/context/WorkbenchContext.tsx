import React, { createContext, useContext, useReducer, useMemo, useCallback, ReactNode } from 'react';
import {
  WorkbenchState,
  WorkbenchAction,
  StudioMode,
  ObservabilityMode,
  BottomDockTab,
  HoveredActionMeta,
} from './workbenchTypes';
import { GameStateDTO, BrainInspectionDTO, TelemetryEventDTO, ReplayDetailDTO } from '../api/types';

const initialState: WorkbenchState = {
  studioMode: 'interactive',
  observabilityMode: 'god',
  gameState: null,
  brainData: null,
  telemetryHistory: [],
  replayData: null,
  currentStepIndex: 0,
  maxStepIndex: 0,
  isPlaying: false,
  playbackSpeed: 1,
  hoveredAction: null,
  hoveredRouteId: null,
  bottomDockTab: 'telemetry',
  isBottomDockOpen: true,
  selectedAgentModel: 'ppo',
  isConnected: false,
  dismissedGameOverSessions: [],
};

function workbenchReducer(state: WorkbenchState, action: WorkbenchAction): WorkbenchState {
  switch (action.type) {
    case 'SET_STUDIO_MODE':
      return { ...state, studioMode: action.payload };
    case 'SET_OBSERVABILITY_MODE':
      return { ...state, observabilityMode: action.payload };
    case 'SET_GAME_STATE':
      return { ...state, gameState: action.payload };
    case 'SET_BRAIN_DATA':
      return { ...state, brainData: action.payload };
    case 'ADD_TELEMETRY_EVENT':
      return {
        ...state,
        telemetryHistory: [...state.telemetryHistory.slice(-200), action.payload],
      };
    case 'SET_REPLAY_DATA':
      return {
        ...state,
        replayData: action.payload,
        maxStepIndex: action.payload ? action.payload.frames.length - 1 : 0,
        currentStepIndex: 0,
      };
    case 'SET_STEP_INDEX':
      return { ...state, currentStepIndex: Math.max(0, Math.min(action.payload, state.maxStepIndex)) };
    case 'SET_MAX_STEPS':
      return { ...state, maxStepIndex: Math.max(0, action.payload) };
    case 'SET_IS_PLAYING':
      return { ...state, isPlaying: action.payload };
    case 'SET_PLAYBACK_SPEED':
      return { ...state, playbackSpeed: action.payload };
    case 'SET_HOVERED_ACTION':
      return {
        ...state,
        hoveredAction: action.payload,
        hoveredRouteId: action.payload?.routeId || null,
      };
    case 'SET_HOVERED_ROUTE_ID':
      return { ...state, hoveredRouteId: action.payload };
    case 'SET_BOTTOM_DOCK_TAB':
      return { ...state, bottomDockTab: action.payload, isBottomDockOpen: true };
    case 'TOGGLE_BOTTOM_DOCK':
      return { ...state, isBottomDockOpen: !state.isBottomDockOpen };
    case 'SET_BOTTOM_DOCK_OPEN':
      return { ...state, isBottomDockOpen: action.payload };
    case 'SET_SELECTED_AGENT_MODEL':
      return { ...state, selectedAgentModel: action.payload };
    case 'SET_CONNECTED':
      return { ...state, isConnected: action.payload };
    case 'DISMISS_GAME_OVER_SESSION':
      return {
        ...state,
        dismissedGameOverSessions: state.dismissedGameOverSessions.includes(action.payload)
          ? state.dismissedGameOverSessions
          : [...state.dismissedGameOverSessions, action.payload],
      };
    case 'RESET_SESSION':
      return {
        ...initialState,
        studioMode: state.studioMode,
        observabilityMode: state.observabilityMode,
      };
    default:
      return state;
  }
}

interface WorkbenchContextValue {
  state: WorkbenchState;
  dispatch: React.Dispatch<WorkbenchAction>;
  setStudioMode: (mode: StudioMode) => void;
  setObservabilityMode: (mode: ObservabilityMode) => void;
  setGameState: (game: GameStateDTO | null) => void;
  setBrainData: (brain: BrainInspectionDTO | null) => void;
  addTelemetryEvent: (event: TelemetryEventDTO) => void;
  setReplayData: (replay: ReplayDetailDTO | null) => void;
  setStepIndex: (index: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;
  setHoveredAction: (meta: HoveredActionMeta | null) => void;
  setHoveredRouteId: (routeId: string | null) => void;
  setBottomDockTab: (tab: BottomDockTab) => void;
  toggleBottomDock: () => void;
  setBottomDockOpen: (open: boolean) => void;
  setSelectedAgentModel: (model: 'ppo' | 'dqn' | 'heuristic' | 'random') => void;
  setConnected: (connected: boolean) => void;
  dismissGameOverSession: (sessionId: string) => void;
  resetSession: () => void;
}

const WorkbenchContext = createContext<WorkbenchContextValue | undefined>(undefined);

export const WorkbenchProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(workbenchReducer, initialState);

  // Stable action dispatchers that NEVER change reference across renders
  const setStudioMode = useCallback((mode: StudioMode) => dispatch({ type: 'SET_STUDIO_MODE', payload: mode }), []);
  const setObservabilityMode = useCallback((mode: ObservabilityMode) => dispatch({ type: 'SET_OBSERVABILITY_MODE', payload: mode }), []);
  const setGameState = useCallback((game: GameStateDTO | null) => dispatch({ type: 'SET_GAME_STATE', payload: game }), []);
  const setBrainData = useCallback((brain: BrainInspectionDTO | null) => dispatch({ type: 'SET_BRAIN_DATA', payload: brain }), []);
  const addTelemetryEvent = useCallback((event: TelemetryEventDTO) => dispatch({ type: 'ADD_TELEMETRY_EVENT', payload: event }), []);
  const setReplayData = useCallback((replay: ReplayDetailDTO | null) => dispatch({ type: 'SET_REPLAY_DATA', payload: replay }), []);
  const setStepIndex = useCallback((index: number) => dispatch({ type: 'SET_STEP_INDEX', payload: index }), []);
  const setIsPlaying = useCallback((playing: boolean) => dispatch({ type: 'SET_IS_PLAYING', payload: playing }), []);
  const setPlaybackSpeed = useCallback((speed: number) => dispatch({ type: 'SET_PLAYBACK_SPEED', payload: speed }), []);
  const setHoveredAction = useCallback((meta: HoveredActionMeta | null) => dispatch({ type: 'SET_HOVERED_ACTION', payload: meta }), []);
  const setHoveredRouteId = useCallback((routeId: string | null) => dispatch({ type: 'SET_HOVERED_ROUTE_ID', payload: routeId }), []);
  const setBottomDockTab = useCallback((tab: BottomDockTab) => dispatch({ type: 'SET_BOTTOM_DOCK_TAB', payload: tab }), []);
  const toggleBottomDock = useCallback(() => dispatch({ type: 'TOGGLE_BOTTOM_DOCK' }), []);
  const setBottomDockOpen = useCallback((open: boolean) => dispatch({ type: 'SET_BOTTOM_DOCK_OPEN', payload: open }), []);
  const setSelectedAgentModel = useCallback((model: 'ppo' | 'dqn' | 'heuristic' | 'random') => dispatch({ type: 'SET_SELECTED_AGENT_MODEL', payload: model }), []);
  const setConnected = useCallback((connected: boolean) => dispatch({ type: 'SET_CONNECTED', payload: connected }), []);
  const dismissGameOverSession = useCallback((sessionId: string) => dispatch({ type: 'DISMISS_GAME_OVER_SESSION', payload: sessionId }), []);
  const resetSession = useCallback(() => dispatch({ type: 'RESET_SESSION' }), []);

  const value = useMemo<WorkbenchContextValue>(() => ({
    state,
    dispatch,
    setStudioMode,
    setObservabilityMode,
    setGameState,
    setBrainData,
    addTelemetryEvent,
    setReplayData,
    setStepIndex,
    setIsPlaying,
    setPlaybackSpeed,
    setHoveredAction,
    setHoveredRouteId,
    setBottomDockTab,
    toggleBottomDock,
    setBottomDockOpen,
    setSelectedAgentModel,
    setConnected,
    dismissGameOverSession,
    resetSession,
  }), [
    state,
    setStudioMode,
    setObservabilityMode,
    setGameState,
    setBrainData,
    addTelemetryEvent,
    setReplayData,
    setStepIndex,
    setIsPlaying,
    setPlaybackSpeed,
    setHoveredAction,
    setHoveredRouteId,
    setBottomDockTab,
    toggleBottomDock,
    setBottomDockOpen,
    setSelectedAgentModel,
    setConnected,
    dismissGameOverSession,
    resetSession,
  ]);

  return <WorkbenchContext.Provider value={value}>{children}</WorkbenchContext.Provider>;
};

export function useWorkbench(): WorkbenchContextValue {
  const context = useContext(WorkbenchContext);
  if (!context) {
    throw new Error('useWorkbench must be used within a WorkbenchProvider');
  }
  return context;
}
