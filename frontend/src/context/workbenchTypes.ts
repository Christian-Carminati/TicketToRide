import { GameStateDTO, BrainInspectionDTO, TelemetryEventDTO, ReplayDetailDTO } from '../api/types';

export type ObservabilityMode = 'god' | 'player_0' | 'player_1';
export type StudioMode = 'interactive' | 'training_live' | 'replay_scrub' | 'tournament';
export type BottomDockTab = 'telemetry' | 'tournament' | 'logs';

export interface HoveredActionMeta {
  actionIndex?: number;
  actionType: string;
  routeId?: string;
  probability?: number;
  value?: number;
  isMasked?: boolean;
}

export interface WorkbenchState {
  studioMode: StudioMode;
  observabilityMode: ObservabilityMode;
  gameState: GameStateDTO | null;
  brainData: BrainInspectionDTO | null;
  telemetryHistory: TelemetryEventDTO[];
  replayData: ReplayDetailDTO | null;
  currentStepIndex: number;
  maxStepIndex: number;
  isPlaying: boolean;
  playbackSpeed: number; // 0.25x, 0.5x, 1x, 2x, 5x
  hoveredAction: HoveredActionMeta | null;
  hoveredRouteId: string | null;
  bottomDockTab: BottomDockTab;
  isBottomDockOpen: boolean;
  selectedAgentModel: 'ppo' | 'dqn' | 'heuristic' | 'random';
  isConnected: boolean;
}

export type WorkbenchAction =
  | { type: 'SET_STUDIO_MODE'; payload: StudioMode }
  | { type: 'SET_OBSERVABILITY_MODE'; payload: ObservabilityMode }
  | { type: 'SET_GAME_STATE'; payload: GameStateDTO | null }
  | { type: 'SET_BRAIN_DATA'; payload: BrainInspectionDTO | null }
  | { type: 'ADD_TELEMETRY_EVENT'; payload: TelemetryEventDTO }
  | { type: 'SET_REPLAY_DATA'; payload: ReplayDetailDTO | null }
  | { type: 'SET_STEP_INDEX'; payload: number }
  | { type: 'SET_MAX_STEPS'; payload: number }
  | { type: 'SET_IS_PLAYING'; payload: boolean }
  | { type: 'SET_PLAYBACK_SPEED'; payload: number }
  | { type: 'SET_HOVERED_ACTION'; payload: HoveredActionMeta | null }
  | { type: 'SET_HOVERED_ROUTE_ID'; payload: string | null }
  | { type: 'SET_BOTTOM_DOCK_TAB'; payload: BottomDockTab }
  | { type: 'TOGGLE_BOTTOM_DOCK' }
  | { type: 'SET_BOTTOM_DOCK_OPEN'; payload: boolean }
  | { type: 'SET_SELECTED_AGENT_MODEL'; payload: 'ppo' | 'dqn' | 'heuristic' | 'random' }
  | { type: 'SET_CONNECTED'; payload: boolean }
  | { type: 'RESET_SESSION' };
