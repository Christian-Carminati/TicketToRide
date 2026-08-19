import React from 'react';
import { useReplayPlayer } from '../hooks/useReplayPlayer';
import { BoardSVG } from '../components/board/BoardSVG';
import { ActionProbabilitiesChart } from '../components/brain/ActionProbabilitiesChart';

export const ReplayView: React.FC = () => {
  const {
    replayList,
    selectedReplay,
    currentFrame,
    currentFrameIndex,
    totalFrames,
    isPlaying,
    playbackSpeedMs,
    isLoading,
    error,
    setPlaybackSpeedMs,
    loadReplay,
    nextFrame,
    prevFrame,
    jumpToFrame,
    togglePlay,
  } = useReplayPlayer();

  const handleSelectReplay = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (e.target.value) {
      loadReplay(e.target.value);
    }
  };

  // Keyboard navigation: Space toggles play/pause, Left/Right arrows step frames
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        nextFrame();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        prevFrame();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePlay, nextFrame, prevFrame]);

  const snapshot = currentFrame?.state_snapshot || {};
  const claimedRoutes: Record<string, string> = snapshot.claimed_routes || {};
  const players = snapshot.players || [];
  const mapName: 'mini' | 'usa' = (selectedReplay?.map_name as 'mini' | 'usa') || 'mini';

  return (
    <div className="replay-view" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Replay Selection & Transport Controls Bar */}
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.9)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '12px',
          padding: '1rem 1.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Select Replay:</label>
          <select
            value={selectedReplay?.replay_id || ''}
            onChange={handleSelectReplay}
            style={{
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.6rem',
              fontSize: '0.85rem',
              minWidth: '220px',
            }}
          >
            <option value="">-- Choose Recorded Match --</option>
            {replayList.map((r) => (
              <option key={r.replay_id} value={r.replay_id}>
                {r.replay_id} ({r.map_name.toUpperCase()} - {r.total_steps} steps)
              </option>
            ))}
          </select>

          {/* Transport Buttons */}
          <button
            onClick={prevFrame}
            disabled={!selectedReplay || currentFrameIndex <= 0 || isPlaying}
            style={{
              backgroundColor: '#334155',
              color: '#F8FAFC',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.4rem 0.8rem',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
            }}
          >
            ⏮ Prev
          </button>

          <button
            onClick={togglePlay}
            disabled={!selectedReplay || totalFrames === 0}
            style={{
              backgroundColor: isPlaying ? '#EF4444' : '#3B82F6',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.4rem 1.2rem',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
            }}
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>

          <button
            onClick={nextFrame}
            disabled={!selectedReplay || currentFrameIndex >= totalFrames - 1 || isPlaying}
            style={{
              backgroundColor: '#334155',
              color: '#F8FAFC',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.4rem 0.8rem',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
            }}
          >
            Next ⏭
          </button>
        </div>

        {/* Speed & Scrubber */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flex: '1', maxWidth: '400px' }}>
          <span style={{ fontSize: '0.8rem', color: '#94A3B8', whiteSpace: 'nowrap' }}>
            Frame {totalFrames > 0 ? currentFrameIndex + 1 : 0} / {totalFrames}
          </span>

          <input
            type="range"
            min="0"
            max={Math.max(0, totalFrames - 1)}
            value={currentFrameIndex}
            onChange={(e) => jumpToFrame(Number(e.target.value))}
            disabled={!selectedReplay || totalFrames === 0}
            style={{ flex: 1 }}
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.8rem', color: '#94A3B8' }}>
            <select
              value={playbackSpeedMs}
              onChange={(e) => setPlaybackSpeedMs(Number(e.target.value))}
              style={{
                backgroundColor: '#1E293B',
                color: '#F1F5F9',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '4px',
                padding: '0.2rem 0.4rem',
                fontSize: '0.75rem',
              }}
            >
              <option value="1000">1x (1s)</option>
              <option value="500">2x (0.5s)</option>
              <option value="250">4x (0.25s)</option>
              <option value="100">10x (0.1s)</option>
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div style={{ padding: '0.75rem', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.2)', border: '1px solid #EF4444', color: '#FCA5A5', fontSize: '0.85rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Main Replay Content Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.2fr) minmax(0, 0.8fr)', gap: '1.25rem' }}>
        {/* Board View */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <BoardSVG
            mapName={mapName}
            claimedRoutes={claimedRoutes}
            players={players}
          />
        </div>

        {/* Frame Action Details & Introspection */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Step Meta Card */}
          <div
            style={{
              background: 'rgba(15, 23, 42, 0.85)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '10px',
              padding: '1.25rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#F1F5F9' }}>
                🎞️ Step #{currentFrame?.step_index ?? 0} Details
              </h4>
              {currentFrame?.reward !== undefined && (
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: currentFrame.reward >= 0 ? '#10B981' : '#EF4444' }}>
                  Reward: {currentFrame.reward > 0 ? `+${currentFrame.reward}` : currentFrame.reward}
                </span>
              )}
            </div>

            {currentFrame ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem' }}>
                <div>
                  <span style={{ color: '#94A3B8' }}>Active Player:</span>{' '}
                  <strong style={{ color: '#38BDF8' }}>
                    {selectedReplay?.player_names[currentFrame.player_index] || `Player ${currentFrame.player_index + 1}`}
                  </strong>
                </div>
                <div>
                  <span style={{ color: '#94A3B8' }}>Action Type:</span>{' '}
                  <span
                    style={{
                      padding: '0.2rem 0.5rem',
                      borderRadius: '4px',
                      backgroundColor: '#3B82F6',
                      color: '#FFFFFF',
                      fontWeight: 700,
                      fontSize: '0.75rem',
                    }}
                  >
                    {currentFrame.action.action_type}
                  </span>
                </div>
                {currentFrame.action.route_id && (
                  <div>
                    <span style={{ color: '#94A3B8' }}>Claimed Route:</span>{' '}
                    <strong style={{ color: '#F1F5F9' }}>{currentFrame.action.route_id}</strong>
                  </div>
                )}
                {currentFrame.action.card_color && (
                  <div>
                    <span style={{ color: '#94A3B8' }}>Color Spent:</span>{' '}
                    <strong style={{ color: '#F59E0B' }}>{currentFrame.action.card_color}</strong>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: '0.85rem', color: '#64748B', fontStyle: 'italic' }}>
                {isLoading ? 'Loading replay frame...' : 'No replay loaded. Select a replay from the dropdown.'}
              </div>
            )}
          </div>

          {/* Action Probabilities if available in frame */}
          {currentFrame?.action_probabilities && currentFrame?.action_mask && (
            <ActionProbabilitiesChart
              probabilities={currentFrame.action_probabilities}
              actionMask={currentFrame.action_mask}
              actionLabels={currentFrame.action_probabilities.map((_, i) => `Action ${i}`)}
            />
          )}
        </div>
      </div>
    </div>
  );
};
