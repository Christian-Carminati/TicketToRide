import { useState, useCallback, useEffect, useRef } from 'react';
import { api } from '../api/client';
import { ReplayDetailDTO, ReplayFrameDTO } from '../api/types';

export function useReplayPlayer() {
  const [replayList, setReplayList] = useState<Array<{ replay_id: string; map_name: string; total_steps: number; date: string; winner_index: number }>>([]);
  const [selectedReplay, setSelectedReplay] = useState<ReplayDetailDTO | null>(null);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeedMs, setPlaybackSpeedMs] = useState(500);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const timerRef = useRef<number | null>(null);

  const fetchReplayList = useCallback(async () => {
    try {
      const list = await api.listReplays();
      setReplayList(list);
    } catch (err: any) {
      setError(err.message || 'Failed to list replays');
    }
  }, []);

  const loadReplay = useCallback(async (replayId: string) => {
    setIsLoading(true);
    setError(null);
    setIsPlaying(false);
    try {
      const detail = await api.getReplay(replayId);
      setSelectedReplay(detail);
      setCurrentFrameIndex(0);
    } catch (err: any) {
      setError(err.message || 'Failed to load replay');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchReplayList();
  }, [fetchReplayList]);

  // Frame stepper controls
  const nextFrame = useCallback(() => {
    if (!selectedReplay) return;
    setCurrentFrameIndex((prev) => Math.min(prev + 1, selectedReplay.frames.length - 1));
  }, [selectedReplay]);

  const prevFrame = useCallback(() => {
    setCurrentFrameIndex((prev) => Math.max(prev - 1, 0));
  }, []);

  const jumpToFrame = useCallback((index: number) => {
    if (!selectedReplay) return;
    const clamped = Math.max(0, Math.min(index, selectedReplay.frames.length - 1));
    setCurrentFrameIndex(clamped);
  }, [selectedReplay]);

  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  useEffect(() => {
    if (isPlaying && selectedReplay) {
      if (currentFrameIndex >= selectedReplay.frames.length - 1) {
        setIsPlaying(false);
      } else {
        timerRef.current = window.setTimeout(() => {
          setCurrentFrameIndex((prev) => prev + 1);
        }, playbackSpeedMs);
      }
    }

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [isPlaying, currentFrameIndex, selectedReplay, playbackSpeedMs]);

  const currentFrame: ReplayFrameDTO | null =
    selectedReplay && selectedReplay.frames[currentFrameIndex] ? selectedReplay.frames[currentFrameIndex] : null;

  return {
    replayList,
    selectedReplay,
    currentFrame,
    currentFrameIndex,
    totalFrames: selectedReplay ? selectedReplay.frames.length : 0,
    isPlaying,
    playbackSpeedMs,
    isLoading,
    error,
    setPlaybackSpeedMs,
    fetchReplayList,
    loadReplay,
    nextFrame,
    prevFrame,
    jumpToFrame,
    togglePlay,
  };
}
