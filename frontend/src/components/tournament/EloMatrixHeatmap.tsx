import React, { useState, useEffect } from 'react';
import { Trophy, Award, RefreshCw, Play, Settings, CheckSquare, Square, Bot, BrainCircuit, MapPin } from 'lucide-react';
import { api } from '../../api/client';
import { TournamentAgentDTO, TournamentMatchupDTO, TournamentParticipantOptionDTO, TournamentProgressDTO } from '../../api/types';

export const EloMatrixHeatmap: React.FC = () => {
  const [agents, setAgents] = useState<TournamentAgentDTO[]>([]);
  const [matchups, setMatchups] = useState<TournamentMatchupDTO[]>([]);
  const [totalGames, setTotalGames] = useState<number>(0);
  const [updatedAt, setUpdatedAt] = useState<string>('');
  const [mapName, setMapName] = useState<'usa' | 'mini'>('usa');
  const [availableOptions, setAvailableOptions] = useState<TournamentParticipantOptionDTO[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [gamesPerPair, setGamesPerPair] = useState<number>(3);
  const [isConfigOpen, setIsConfigOpen] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isRunningLive, setIsRunningLive] = useState<boolean>(false);
  const [progress, setProgress] = useState<TournamentProgressDTO | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const fetchLeaderboard = () => {
    setIsLoading(true);
    api.getTournamentLeaderboard()
      .then((data) => {
        setAgents(data.leaderboard);
        setMatchups(data.matchups);
        setTotalGames(data.total_games);
        setUpdatedAt(data.updated_at);
        if (data.map_name) setMapName(data.map_name as 'usa' | 'mini');

        if (data.available_participants && data.available_participants.length > 0) {
          setAvailableOptions(data.available_participants);
          if (selectedIds.length === 0) {
            setSelectedIds(data.available_participants.map((p) => p.id));
          }
        }
      })
      .catch((err) => {
        console.error('Failed to load tournament leaderboard:', err);
      })
      .finally(() => setIsLoading(false));
  };

  useEffect(() => {
    fetchLeaderboard();
  }, []);

  // Poll progress during live tournament execution
  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;
    if (isRunningLive) {
      timer = setInterval(() => {
        api.getTournamentProgress()
          .then((prog) => {
            setProgress(prog);
          })
          .catch(() => {});
      }, 600);
    } else {
      setProgress(null);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isRunningLive]);

  const handleToggleParticipant = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const handleSelectAll = () => {
    setSelectedIds(availableOptions.map((o) => o.id));
  };

  const handleSelectBaselinesOnly = () => {
    setSelectedIds(availableOptions.filter((o) => o.category === 'baseline').map((o) => o.id));
  };

  const handleSelectModelsOnly = () => {
    setSelectedIds(availableOptions.filter((o) => o.category === 'checkpoint').map((o) => o.id));
  };

  const handleRunCustomTournament = async () => {
    if (selectedIds.length < 2) {
      alert('Please select at least 2 contestants to execute a round-robin tournament.');
      return;
    }

    setIsRunningLive(true);
    setStatusMessage('Executing round-robin tournament simulation...');
    try {
      const res = await api.runTournament({
        participant_ids: selectedIds,
        games_per_pair: gamesPerPair,
        map_name: mapName,
        seed: Date.now() % 10000,
      });
      setAgents(res.leaderboard);
      setMatchups(res.matchups);
      setTotalGames(res.total_games);
      setUpdatedAt(res.updated_at);
      setStatusMessage(`Tournament completed successfully (${res.total_games} total matches).`);
      setTimeout(() => setStatusMessage(null), 4000);
    } catch (err) {
      console.error('Tournament run error:', err);
      setStatusMessage('Error executing tournament simulation.');
    } finally {
      setIsRunningLive(false);
    }
  };

  const getMatchupWinRate = (agentAName: string, agentBName: string): number | null => {
    const direct = matchups.find((m) => m.agent_a === agentAName && m.agent_b === agentBName);
    if (direct) return direct.win_rate_a;

    const reverse = matchups.find((m) => m.agent_a === agentBName && m.agent_b === agentAName);
    if (reverse) return 1.0 - reverse.win_rate_a;

    return null;
  };

  const totalPairings =
    selectedIds.length >= 2 ? (selectedIds.length * (selectedIds.length - 1)) / 2 : 0;
  const totalCalculatedGames = totalPairings * gamesPerPair;

  const formatEstTime = (gamesCount: number, map: string) => {
    const secPerGame = map === 'mini' ? 0.15 : 1.6;
    const totalSec = Math.round(gamesCount * secPerGame);
    if (totalSec < 60) return `~${Math.max(1, totalSec)}s`;
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `~${mins}m ${secs > 0 ? secs + 's' : ''}`;
  };

  const baselines = availableOptions.filter((o) => o.category === 'baseline');
  const checkpoints = availableOptions.filter((o) => o.category === 'checkpoint');

  return (
    <div
      className="elo-matrix-heatmap"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.75rem',
      }}
    >
      {/* Top Action Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Trophy size={18} color="#9E6B00" />
          <span
            style={{
              fontSize: '0.9rem',
              fontWeight: 800,
              color: '#23140C',
              fontFamily: "'Cinzel Decorative', Georgia, serif",
              letterSpacing: '0.03em',
            }}
          >
            Round-Robin Tournament Ledger & Elo Matrix
          </span>
          <span
            style={{
              fontSize: '0.72rem',
              padding: '0.15rem 0.5rem',
              borderRadius: '4px',
              background: '#EADBBE',
              border: '1px solid #C59B27',
              color: '#4A2F1D',
              fontFamily: "'Courier Prime', monospace",
              fontWeight: 700,
            }}
          >
            SURVEY: {mapName.toUpperCase()}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <div style={{ fontSize: '0.75rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
            Updated: {updatedAt || 'Loading...'} ({totalGames} matches)
          </div>

          {/* Toggle Configuration Panel */}
          <button
            onClick={() => setIsConfigOpen(!isConfigOpen)}
            className="steampunk-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              fontSize: '0.75rem',
            }}
            title="Configure contestants and tournament parameters"
          >
            <Settings size={13} />
            <span>Participants ({selectedIds.length}/{availableOptions.length})</span>
          </button>

          {/* Run Live Tournament Button */}
          <button
            onClick={handleRunCustomTournament}
            disabled={isRunningLive || isLoading || selectedIds.length < 2}
            className="steampunk-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: isRunningLive
                ? 'linear-gradient(180deg, #D4C09D 0%, #A88D75 100%)'
                : 'linear-gradient(180deg, #86EFAC 0%, #16A34A 50%, #14532D 100%)',
              color: '#FFFFFF',
              border: '1px solid #14532D',
              fontSize: '0.75rem',
            }}
            title="Launch round-robin tournament across selected contestants"
          >
            <Play size={13} />
            <span>{isRunningLive ? 'Simulating...' : 'Run Tournament'}</span>
          </button>

          <button
            onClick={fetchLeaderboard}
            disabled={isLoading}
            className="steampunk-btn"
            style={{ padding: '0.3rem 0.55rem' }}
            title="Refresh leaderboard"
          >
            <RefreshCw size={13} />
          </button>
        </div>
      </div>

      {/* Live Simulation Progress Panel */}
      {isRunningLive && (
        <div
          style={{
            background: 'linear-gradient(180deg, #3A261A 0%, #22150D 100%)',
            border: '2px solid #C59B27',
            borderRadius: '8px',
            padding: '0.75rem 1rem',
            color: '#F5E6C8',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.5rem',
            boxShadow: '0 4px 12px rgba(0,0,0,0.35)',
            animation: 'fadeIn 0.3s ease',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <RefreshCw size={15} color="#F7E099" style={{ animation: 'spin 2s linear infinite' }} />
              <span style={{ fontSize: '0.85rem', fontWeight: 800, fontFamily: "'Cinzel Decorative', Georgia, serif", color: '#F7E099' }}>
                Match {progress?.current_match || 1} / {progress?.total_matches || totalPairings}
              </span>
              <span style={{ fontSize: '0.78rem', color: '#D4C09D', fontFamily: "'Playfair Display', serif" }}>
                {progress?.current_agent_a && progress?.current_agent_b ? `${progress.current_agent_a} vs ${progress.current_agent_b}` : 'Initializing match pairings...'}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.75rem', fontFamily: "'Courier Prime', monospace", color: '#D4C09D' }}>
              <span>⏱️ Elapsed: {progress?.elapsed_seconds ? `${progress.elapsed_seconds}s` : '0s'}</span>
              <span>⏳ Remaining: ~{progress?.estimated_remaining_seconds ? `${progress.estimated_remaining_seconds}s` : formatEstTime(totalCalculatedGames, mapName)}</span>
              <span style={{ fontWeight: 800, color: '#86EFAC', fontSize: '0.82rem' }}>
                {progress?.percentage ? `${progress.percentage.toFixed(1)}%` : '0.0%'}
              </span>
            </div>
          </div>

          {/* Progress Bar Track */}
          <div
            style={{
              width: '100%',
              height: '10px',
              backgroundColor: '#1E120A',
              borderRadius: '5px',
              overflow: 'hidden',
              border: '1px solid #6E4E04',
              position: 'relative',
            }}
          >
            <div
              style={{
                width: `${Math.min(100, Math.max(progress?.percentage || 0, 2))}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #B8860B 0%, #F7E099 50%, #16A34A 100%)',
                borderRadius: '4px',
                transition: 'width 0.4s ease',
              }}
            />
          </div>

          {progress?.recent_matchup && (
            <div style={{ fontSize: '0.72rem', color: '#C8B28B', fontFamily: "'Courier Prime', monospace" }}>
              ✓ Completed: {progress.recent_matchup.agent_a} ({progress.recent_matchup.wins_a}) vs ({progress.recent_matchup.wins_b}) {progress.recent_matchup.agent_b} {progress.recent_matchup.draws > 0 ? `(${progress.recent_matchup.draws} draws)` : ''}
            </div>
          )}
        </div>
      )}

      {/* Notification Toast */}
      {statusMessage && !isRunningLive && (
        <div
          style={{
            background: 'linear-gradient(180deg, #FAF3E6 0%, #E8D7BC 100%)',
            border: '1.5px solid #15803D',
            borderRadius: '6px',
            padding: '0.4rem 0.85rem',
            color: '#15803D',
            fontSize: '0.78rem',
            fontFamily: "'Playfair Display', Georgia, serif",
            fontWeight: 700,
          }}
        >
          {statusMessage}
        </div>
      )}

      {/* Config Drawer Panel */}
      {isConfigOpen && (
        <div
          style={{
            background: '#FAF4E6',
            border: '2px solid #C59B27',
            borderRadius: '8px',
            padding: '0.85rem 1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
            boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.1)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 800, color: '#23140C', fontFamily: "'Cinzel Decorative', serif" }}>
              ⚙️ Contestant Selection & Parameters
            </div>

            {/* Quick Selection Presets */}
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              <button
                onClick={handleSelectAll}
                className="steampunk-btn"
                style={{ padding: '0.15rem 0.5rem', fontSize: '0.7rem' }}
              >
                All ({availableOptions.length})
              </button>
              <button
                onClick={handleSelectModelsOnly}
                className="steampunk-btn"
                style={{ padding: '0.15rem 0.5rem', fontSize: '0.7rem' }}
                title={checkpoints.length === 0 ? "Nessun checkpoint trovato in experiments/checkpoints/" : ""}
              >
                RL Checkpoints ({checkpoints.length})
              </button>
              <button
                onClick={handleSelectBaselinesOnly}
                className="steampunk-btn"
                style={{ padding: '0.15rem 0.5rem', fontSize: '0.7rem' }}
              >
                Baselines Algoritmiche ({baselines.length})
              </button>
            </div>
          </div>

          {checkpoints.length === 0 && (
            <div
              style={{
                backgroundColor: 'rgba(197, 155, 39, 0.12)',
                border: '1px dashed #C59B27',
                borderRadius: '6px',
                padding: '0.5rem 0.75rem',
                fontSize: '0.75rem',
                color: '#5A3822',
                fontFamily: "'Crimson Pro', Georgia, serif",
                marginBottom: '0.25rem',
              }}
            >
              💡 <strong>Nota Checkpoint:</strong> La cartella <code>experiments/checkpoints/</code> non contiene file <code>.pt</code> salvati.
              I 7 concorrenti elencati sotto sono le <strong>Baseline Algoritmiche</strong> integrate nel motore (AlphaZero Neural MCTS, Bayesian MCTS, Pure IS-MCTS, Recurrent PPO, Heuristic Dijkstra, Greedy e Random).
              Per aggiungere modelli personalizzati al torneo, avvia una sessione in <strong>Live Training</strong> o <strong>Self-Play PFSP</strong>!
            </div>
          )}

          {/* Participant Checkbox Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '0.6rem' }}>
            {availableOptions.map((opt) => {
              const isSelected = selectedIds.includes(opt.id);
              const isCkpt = opt.category === 'checkpoint';

              return (
                <div
                  key={opt.id}
                  onClick={() => handleToggleParticipant(opt.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '0.5rem',
                    background: isSelected ? '#FAF0DA' : '#F4EADC',
                    border: `1.5px solid ${isSelected ? '#B8860B' : 'rgba(110, 70, 30, 0.2)'}`,
                    borderRadius: '6px',
                    padding: '0.5rem 0.65rem',
                    cursor: 'pointer',
                    transition: 'all 0.12s ease',
                    boxShadow: isSelected ? '0 2px 4px rgba(184, 134, 11, 0.2)' : 'none',
                  }}
                >
                  <div style={{ marginTop: '0.1rem', color: isSelected ? '#9E6B00' : '#785A42' }}>
                    {isSelected ? <CheckSquare size={15} /> : <Square size={15} />}
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      {isCkpt ? <BrainCircuit size={13} color="#7E22CE" /> : <Bot size={13} color="#9E6B00" />}
                      <span style={{ fontSize: '0.8rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                        {opt.name}
                      </span>
                    </div>
                    {opt.description && (
                      <span style={{ fontSize: '0.72rem', color: '#5A3822', fontFamily: "'Crimson Pro', Georgia, serif" }}>
                        {opt.description}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Options Row (Games per pair & Map) */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', paddingTop: '0.5rem', borderTop: '1px solid rgba(184, 134, 11, 0.25)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.78rem', color: '#4A2F1D', fontFamily: "'Crimson Pro', serif", fontWeight: 700 }}>Matches per pair:</span>
                <select
                  value={gamesPerPair}
                  onChange={(e) => setGamesPerPair(Number(e.target.value))}
                  style={{
                    backgroundColor: '#FAF5EB',
                    color: '#23140C',
                    border: '1.5px solid #8C6305',
                    borderRadius: '4px',
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    fontFamily: "'Courier Prime', monospace",
                  }}
                >
                  <option value={1}>1 match ({formatEstTime(totalPairings * 1, mapName)})</option>
                  <option value={2}>2 matches ({formatEstTime(totalPairings * 2, mapName)})</option>
                  <option value={3}>3 matches ({formatEstTime(totalPairings * 3, mapName)})</option>
                  <option value={5}>5 matches ({formatEstTime(totalPairings * 5, mapName)})</option>
                  <option value={10}>10 matches ({formatEstTime(totalPairings * 10, mapName)})</option>
                  <option value={30}>30 matches ({formatEstTime(totalPairings * 30, mapName)})</option>
                  <option value={50}>50 matches ({formatEstTime(totalPairings * 50, mapName)})</option>
                </select>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <MapPin size={13} color="#9E6B00" />
                <span style={{ fontSize: '0.78rem', color: '#4A2F1D', fontFamily: "'Crimson Pro', serif", fontWeight: 700 }}>Survey Map:</span>
                <select
                  value={mapName}
                  onChange={(e) => setMapName(e.target.value as 'usa' | 'mini')}
                  style={{
                    backgroundColor: '#FAF5EB',
                    color: '#23140C',
                    border: '1.5px solid #8C6305',
                    borderRadius: '4px',
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    fontFamily: "'Courier Prime', monospace",
                  }}
                >
                  <option value="usa">USA Full (36 Cities)</option>
                  <option value="mini">Mini Synthetic (5 Cities)</option>
                </select>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span style={{ fontSize: '0.75rem', color: '#5A3822', fontFamily: "'Courier Prime', monospace" }}>
                {selectedIds.length} contestants = {totalCalculatedGames} matches ({formatEstTime(totalCalculatedGames, mapName)})
              </span>
              <button
                onClick={handleRunCustomTournament}
                disabled={isRunningLive || selectedIds.length < 2}
                className="steampunk-btn"
                style={{
                  padding: '0.35rem 0.85rem',
                  fontSize: '0.78rem',
                }}
              >
                {isRunningLive ? 'Simulating...' : 'Launch Simulation'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Grid Table */}
      <div style={{ overflowX: 'auto', background: '#FAF5EB', borderRadius: '8px', border: '1.5px solid #C59B27' }}>
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '0.78rem',
            textAlign: 'center',
          }}
        >
          <thead>
            <tr style={{ background: 'linear-gradient(180deg, #EFE1C7 0%, #E2CFAC 100%)', color: '#23140C', borderBottom: '2px solid #C59B27' }}>
              <th style={{ textAlign: 'left', padding: '0.5rem 0.75rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Rank / Contestant</th>
              <th style={{ padding: '0.5rem 0.75rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Elo Rating</th>
              <th style={{ padding: '0.5rem 0.75rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Win Rate</th>
              <th style={{ padding: '0.5rem 0.75rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>W / L / D</th>
              <th style={{ padding: '0.5rem 0.75rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Avg Score</th>
              {agents.map((ag) => (
                <th key={ag.agent_id} style={{ padding: '0.5rem 0.75rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>
                  vs {ag.name.replace(' Trained', '').replace(' Bot', '').replace('⭐ ', '').replace('🏆 ', '').replace('🧠 ', '')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {agents.map((rowAgent, rIdx) => (
              <tr
                key={rowAgent.agent_id}
                style={{
                  borderBottom: '1px solid rgba(184, 134, 11, 0.2)',
                  background: rIdx % 2 === 0 ? 'rgba(246, 238, 223, 0.5)' : 'transparent',
                }}
              >
                {/* Agent Name & Rank */}
                <td style={{ textAlign: 'left', padding: '0.5rem 0.75rem', fontWeight: 700, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    {rIdx === 0 ? <Award size={14} color="#B8860B" /> : <span style={{ color: '#785A42', width: 14, fontFamily: "'Courier Prime', monospace" }}>#{rIdx + 1}</span>}
                    <span>{rowAgent.name}</span>
                  </div>
                </td>

                {/* Elo */}
                <td style={{ padding: '0.5rem 0.75rem', fontFamily: "'Courier Prime', monospace", fontWeight: 800, color: '#9E6B00' }}>
                  {rowAgent.elo.toFixed(0)}
                </td>

                {/* Overall Win% */}
                <td style={{ padding: '0.5rem 0.75rem', fontFamily: "'Courier Prime', monospace", fontWeight: 800, color: rowAgent.win_rate >= 0.5 ? '#15803D' : '#B91C1C' }}>
                  {(rowAgent.win_rate * 100).toFixed(1)}%
                </td>

                {/* W / L / D */}
                <td style={{ padding: '0.5rem 0.75rem', fontFamily: "'Courier Prime', monospace", color: '#5A3822', fontSize: '0.72rem' }}>
                  {rowAgent.wins} / {rowAgent.losses} / {rowAgent.draws}
                </td>

                {/* Avg Score */}
                <td style={{ padding: '0.5rem 0.75rem', fontFamily: "'Courier Prime', monospace", color: '#23140C', fontWeight: 700 }}>
                  {rowAgent.avg_score.toFixed(1)}
                </td>

                {/* Head-to-Head Win% Cells */}
                {agents.map((colAgent) => {
                  if (rowAgent.agent_id === colAgent.agent_id) {
                    return (
                      <td key={colAgent.agent_id} style={{ background: '#EFE1C7', color: '#785A42', fontSize: '0.75rem' }}>
                        —
                      </td>
                    );
                  }

                  const empRate = getMatchupWinRate(rowAgent.name, colAgent.name);
                  const pct = empRate !== null ? Math.round(empRate * 100) : 50;

                  const isHigh = pct >= 60;
                  const isLow = pct <= 40;

                  return (
                    <td
                      key={colAgent.agent_id}
                      style={{
                        padding: '0.4rem 0.6rem',
                        fontFamily: "'Courier Prime', monospace",
                        fontWeight: 800,
                        backgroundColor: isHigh
                          ? 'rgba(21, 128, 61, 0.15)'
                          : isLow
                          ? 'rgba(185, 28, 28, 0.15)'
                          : 'rgba(184, 134, 11, 0.1)',
                        color: isHigh ? '#15803D' : isLow ? '#B91C1C' : '#5A3822',
                      }}
                    >
                      {pct}%
                    </td>
                  );
                })}
              </tr>
            ))}
            {agents.length === 0 && (
              <tr>
                <td colSpan={5 + agents.length} style={{ padding: '1.5rem', color: '#785A42', textAlign: 'center', fontStyle: 'italic', fontFamily: "'Crimson Pro', serif" }}>
                  Loading tournament records...
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
