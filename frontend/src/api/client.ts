/**
 * Typed HTTP Client for TicketToRide RL Lab API.
 */

import {
  ActionDTO,
  BrainInspectionDTO,
  ExperimentRecordDTO,
  GameSessionCreateRequest,
  GameStateDTO,
  ReplayDetailDTO,
  TrainingStartRequest,
  TrainingStatusDTO,
} from './types';

const API_BASE = 'http://localhost:8000';

async function fetchJSON<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    let errorDetail = res.statusText;
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || JSON.stringify(errJson);
    } catch {
      // Ignored
    }
    throw new Error(`API Error [${res.status}] ${endpoint}: ${errorDetail}`);
  }

  return res.json() as Promise<T>;
}

export const api = {
  // Health
  checkHealth: () => fetchJSON<{ status: string; app: string }>('/health'),

  // Game Sessions
  createGame: (req: GameSessionCreateRequest) =>
    fetchJSON<GameStateDTO>('/api/game/new', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  getGameState: (sessionId: string) => fetchJSON<GameStateDTO>(`/api/game/${sessionId}`),

  stepGame: (sessionId: string, action?: ActionDTO | null) =>
    fetchJSON<GameStateDTO>('/api/game/step', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, action: action || null }),
    }),

  deleteGame: (sessionId: string) =>
    fetchJSON<{ success: boolean }>(`/api/game/${sessionId}`, {
      method: 'DELETE',
    }),

  // Training
  getTrainingStatus: () => fetchJSON<TrainingStatusDTO>('/api/training/status'),

  startTraining: (req: TrainingStartRequest) =>
    fetchJSON<TrainingStatusDTO>('/api/training/start', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  stopTraining: () =>
    fetchJSON<TrainingStatusDTO>('/api/training/stop', {
      method: 'POST',
    }),

  // Replays & Experiments
  listReplays: () => fetchJSON<Array<{ replay_id: string; map_name: string; total_steps: number; date: string; winner_index: number }>>('/api/replays/list'),

  getReplay: (replayId: string) => fetchJSON<ReplayDetailDTO>(`/api/replays/${replayId}`),

  listExperiments: () => fetchJSON<ExperimentRecordDTO[]>('/api/experiments/list'),

  deleteExperiment: (experimentId: string) =>
    fetchJSON<{ success: boolean; deleted_id: string }>(`/api/experiments/${experimentId}`, {
      method: 'DELETE',
    }),

  deleteAllExperiments: () =>
    fetchJSON<{ success: boolean; deleted_count: number }>('/api/experiments', {
      method: 'DELETE',
    }),

  // Checkpoints
  listCheckpoints: () => fetchJSON<import('./types').CheckpointDTO[]>('/api/checkpoints/list'),

  // Brain Introspection
  inspectBrain: (payload: { session_id?: string; model_type?: 'dqn' | 'ppo'; observation?: number[]; action_mask?: boolean[] }) =>
    fetchJSON<BrainInspectionDTO>('/api/brain/inspect', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
};
