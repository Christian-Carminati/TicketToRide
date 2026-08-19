import { useState, useCallback, useRef, useEffect } from 'react';
import { api } from '../api/client';
import { ActionDTO, GameSessionCreateRequest, GameStateDTO } from '../api/types';

export function useGameSession() {
  const [gameState, setGameState] = useState<GameStateDTO | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [playSpeedMs, setPlaySpeedMs] = useState(500);

  const autoPlayTimerRef = useRef<number | null>(null);

  const createGame = useCallback(async (req: GameSessionCreateRequest) => {
    setIsLoading(true);
    setError(null);
    try {
      const state = await api.createGame(req);
      setGameState(state);
      setIsAutoPlaying(false);
    } catch (err: any) {
      setError(err.message || 'Failed to create game session');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const stepGame = useCallback(async (action?: ActionDTO | null) => {
    if (!gameState) return;
    setIsLoading(true);
    setError(null);
    try {
      const nextState = await api.stepGame(gameState.session_id, action);
      setGameState(nextState);
    } catch (err: any) {
      setError(err.message || 'Failed to step game');
      setIsAutoPlaying(false);
    } finally {
      setIsLoading(false);
    }
  }, [gameState]);

  // Handle autoplay loop for Bot vs Bot
  useEffect(() => {
    if (isAutoPlaying && gameState && !gameState.is_game_over) {
      autoPlayTimerRef.current = window.setTimeout(() => {
        // Only step automatically if current player is a bot or in autoplay
        stepGame(null);
      }, playSpeedMs);
    } else {
      setIsAutoPlaying(false);
    }

    return () => {
      if (autoPlayTimerRef.current) {
        clearTimeout(autoPlayTimerRef.current);
      }
    };
  }, [isAutoPlaying, gameState, stepGame, playSpeedMs]);

  const toggleAutoPlay = useCallback(() => {
    setIsAutoPlaying((prev) => !prev);
  }, []);

  return {
    gameState,
    isLoading,
    error,
    isAutoPlaying,
    playSpeedMs,
    setPlaySpeedMs,
    createGame,
    stepGame,
    toggleAutoPlay,
    setGameState,
  };
}
