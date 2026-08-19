import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import { TelemetryEventDTO, TrainingStartRequest, TrainingStatusDTO } from '../api/types';
import { useWebSocket } from './useWebSocket';

export function useTrainingStream() {
  const [status, setStatus] = useState<TrainingStatusDTO | null>(null);
  const [telemetryHistory, setTelemetryHistory] = useState<TelemetryEventDTO[]>([]);
  const [error, setError] = useState<string | null>(null);

  const { isConnected, lastMessage } = useWebSocket<TelemetryEventDTO>();

  // Fetch initial status on mount
  useEffect(() => {
    api.getTrainingStatus()
      .then(setStatus)
      .catch((err) => setError(err.message));
  }, []);

  // Process incoming WebSocket telemetry messages
  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === 'training_started') {
      setStatus({
        is_training: true,
        experiment_id: lastMessage.experiment_id,
        algorithm: (lastMessage as any).algorithm || 'ppo',
        current_step: 0,
        total_timesteps: (lastMessage as any).total_timesteps || 0,
        episodes: 0,
        mean_reward: 0.0,
      });
      setTelemetryHistory([]);
    } else if (lastMessage.type === 'training_step') {
      setTelemetryHistory((prev) => {
        const next = [...prev, lastMessage];
        // Keep max 200 items in buffer for smooth rendering
        return next.length > 200 ? next.slice(next.length - 200) : next;
      });
      setStatus((prev) =>
        prev
          ? {
              ...prev,
              current_step: lastMessage.step,
              episodes: lastMessage.episode,
              mean_reward: lastMessage.mean_reward,
            }
          : null
      );
    } else if (lastMessage.type === 'training_finished') {
      setStatus((prev) => (prev ? { ...prev, is_training: false } : null));
    }
  }, [lastMessage]);

  const startTraining = useCallback(async (req: TrainingStartRequest) => {
    setError(null);
    try {
      const res = await api.startTraining(req);
      setStatus(res);
      setTelemetryHistory([]);
    } catch (err: any) {
      setError(err.message || 'Failed to start training');
    }
  }, []);

  const stopTraining = useCallback(async () => {
    setError(null);
    try {
      const res = await api.stopTraining();
      setStatus(res);
    } catch (err: any) {
      setError(err.message || 'Failed to stop training');
    }
  }, []);

  return {
    status,
    telemetryHistory,
    isConnected,
    error,
    startTraining,
    stopTraining,
  };
}
