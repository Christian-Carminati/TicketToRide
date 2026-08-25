import React from 'react';
import { useReplayPlayer } from '../hooks/useReplayPlayer';
import { BoardSVG } from '../components/board/BoardSVG';
import { ActionProbabilitiesChart } from '../components/brain/ActionProbabilitiesChart';
import { Film, Play, Pause, SkipBack, SkipForward, Clock } from 'lucide-react';

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

  const snapshot = (currentFrame?.state_snapshot || {}) as Record<string, any>;
  const claimedRoutes: Record<string, string> = (snapshot.claimed_routes as Record<string, string>) || {};
  const players = (snapshot.players as any[]) || [];
  const mapName: string = selectedReplay?.map_name || 'usa';

  return (
    <div
      className="replay-view"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem',
        color: '#23140C',
        fontFamily: "'Crimson Pro', Georgia, serif",
      }}
    >
      {/* Replay Selection & Transport Controls Bar */}
      <div
        className="steampunk-panel"
        style={{
          padding: '0.85rem 1.25rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
          background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Film size={16} color="#9E6B00" />
            <label style={{ fontSize: '0.82rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>
              Match Chronicle:
            </label>
          </div>

          <select
            value={selectedReplay?.replay_id || ''}
            onChange={handleSelectReplay}
            style={{
              backgroundColor: '#FAF5EB',
              color: '#23140C',
              border: '1.5px solid #8C6305',
              borderRadius: '6px',
              padding: '0.3rem 0.6rem',
              fontSize: '0.82rem',
              fontWeight: 700,
              minWidth: '240px',
              fontFamily: "'Courier Prime', monospace",
            }}
          >
            <option value="">-- Select Recorded Match --</option>
            {replayList.map((r) => (
              <option key={r.replay_id} value={r.replay_id}>
                {r.replay_id} ({r.map_name.toUpperCase()} • {r.total_steps} steps)
              </option>
            ))}
          </select>

          {/* Transport Buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <button
              onClick={prevFrame}
              disabled={!selectedReplay || currentFrameIndex <= 0 || isPlaying}
              className="steampunk-btn"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
                padding: '0.3rem 0.65rem',
                fontSize: '0.78rem',
              }}
              title="Previous Frame (Left Arrow)"
            >
              <SkipBack size={12} /> Prev
            </button>

            <button
              onClick={togglePlay}
              disabled={!selectedReplay || totalFrames === 0}
              className="steampunk-btn"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                padding: '0.3rem 0.85rem',
                fontSize: '0.82rem',
                background: isPlaying
                  ? 'linear-gradient(180deg, #F87171 0%, #DC2626 50%, #991B1B 100%)'
                  : 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)',
                color: isPlaying ? '#FFFFFF' : '#23140C',
              }}
              title="Toggle Play / Pause (Spacebar)"
            >
              {isPlaying ? <Pause size={13} /> : <Play size={13} />}
              <span>{isPlaying ? 'Pause' : 'Play'}</span>
            </button>

            <button
              onClick={nextFrame}
              disabled={!selectedReplay || currentFrameIndex >= totalFrames - 1 || isPlaying}
              className="steampunk-btn"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
                padding: '0.3rem 0.65rem',
                fontSize: '0.78rem',
              }}
              title="Next Frame (Right Arrow)"
            >
              Next <SkipForward size={12} />
            </button>
          </div>
        </div>

        {/* Speed & Scrubber */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: '1', maxWidth: '420px' }}>
          <span style={{ fontSize: '0.76rem', color: '#5A3822', whiteSpace: 'nowrap', fontFamily: "'Courier Prime', monospace", fontWeight: 700 }}>
            Frame {totalFrames > 0 ? currentFrameIndex + 1 : 0} / {totalFrames}
          </span>

          <input
            type="range"
            min="0"
            max={Math.max(0, totalFrames - 1)}
            value={currentFrameIndex}
            onChange={(e) => jumpToFrame(Number(e.target.value))}
            disabled={!selectedReplay || totalFrames === 0}
            style={{ flex: 1, accentColor: '#B8860B' }}
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.76rem', color: '#5A3822' }}>
            <Clock size={12} color="#785A42" />
            <select
              value={playbackSpeedMs}
              onChange={(e) => setPlaybackSpeedMs(Number(e.target.value))}
              style={{
                backgroundColor: '#FAF5EB',
                color: '#23140C',
                border: '1.5px solid #8C6305',
                borderRadius: '4px',
                padding: '0.2rem 0.4rem',
                fontSize: '0.72rem',
                fontFamily: "'Courier Prime', monospace",
                fontWeight: 700,
              }}
            >
              <option value="1000">1x (1.0s)</option>
              <option value="500">2x (0.5s)</option>
              <option value="250">4x (0.25s)</option>
              <option value="100">10x (0.1s)</option>
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div style={{ padding: '0.75rem', borderRadius: '8px', background: '#FEE2E2', border: '1.5px solid #DC2626', color: '#991B1B', fontSize: '0.85rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Main Replay Content Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.25fr) minmax(320px, 0.75fr)', gap: '1.25rem' }}>
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
            className="steampunk-panel"
            style={{
              padding: '1.1rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(184, 134, 11, 0.3)', paddingBottom: '0.5rem' }}>
              <h4 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                🎞️ Step #{currentFrame?.step_index ?? 0} Chronicle Details
              </h4>
              {currentFrame?.reward !== undefined && (
                <span style={{ fontSize: '0.82rem', fontWeight: 800, color: currentFrame.reward >= 0 ? '#15803D' : '#B91C1C', fontFamily: "'Courier Prime', monospace" }}>
                  Reward: {currentFrame.reward > 0 ? `+${currentFrame.reward}` : currentFrame.reward}
                </span>
              )}
            </div>

            {currentFrame ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', fontSize: '0.82rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#785A42', fontFamily: "'Playfair Display', serif", fontWeight: 700 }}>Active Player:</span>
                  <strong style={{ color: '#9E6B00', fontFamily: "'Playfair Display', Georgia, serif" }}>
                    {selectedReplay?.player_names[currentFrame.player_index] || `Player ${currentFrame.player_index + 1}`}
                  </strong>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#785A42', fontFamily: "'Playfair Display', serif", fontWeight: 700 }}>Action Executed:</span>
                  <span
                    style={{
                      padding: '0.15rem 0.5rem',
                      borderRadius: '4px',
                      backgroundColor: '#FAF0DA',
                      border: '1px solid #C59B27',
                      color: '#23140C',
                      fontWeight: 800,
                      fontSize: '0.75rem',
                      fontFamily: "'Courier Prime', monospace",
                    }}
                  >
                    {currentFrame.action.action_type}
                  </span>
                </div>

                {currentFrame.action.route_id && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#785A42', fontFamily: "'Playfair Display', serif", fontWeight: 700 }}>Claimed Route:</span>
                    <strong style={{ color: '#23140C', fontFamily: "'Courier Prime', monospace" }}>{currentFrame.action.route_id}</strong>
                  </div>
                )}

                {currentFrame.action.card_color && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#785A42', fontFamily: "'Playfair Display', serif", fontWeight: 700 }}>Color Spent:</span>
                    <strong style={{ color: '#CD7F32', fontFamily: "'Courier Prime', monospace" }}>{currentFrame.action.card_color}</strong>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: '0.82rem', color: '#785A42', fontStyle: 'italic', padding: '1rem 0', textAlign: 'center' }}>
                {isLoading ? 'Loading replay frame...' : 'No replay loaded. Select a recorded match from the dropdown above.'}
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
