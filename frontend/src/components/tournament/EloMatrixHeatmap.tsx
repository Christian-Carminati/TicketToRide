import React, { useState, useEffect } from 'react';
import { Trophy, Award, RefreshCw, Play } from 'lucide-react';
import { api } from '../../api/client';
import { TournamentAgentDTO, TournamentMatchupDTO } from '../../api/types';

export const EloMatrixHeatmap: React.FC = () => {
  const [agents, setAgents] = useState<TournamentAgentDTO[]>([]);
  const [matchups, setMatchups] = useState<TournamentMatchupDTO[]>([]);
  const [totalGames, setTotalGames] = useState<number>(0);
  const [updatedAt, setUpdatedAt] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isRunningLive, setIsRunningLive] = useState<boolean>(false);

  const fetchLeaderboard = () => {
    setIsLoading(true);
    api.getTournamentLeaderboard()
      .then((data) => {
        setAgents(data.leaderboard);
        setMatchups(data.matchups);
        setTotalGames(data.total_games);
        setUpdatedAt(data.updated_at);
      })
      .catch((err) => {
        console.error('Failed to load tournament leaderboard:', err);
      })
      .finally(() => setIsLoading(false));
  };

  const handleRunLiveTournament = async () => {
    setIsRunningLive(true);
    try {
      const res = await api.runTournament({ games_per_pair: 15, seed: Date.now() % 10000 });
      setAgents(res.leaderboard);
      setMatchups(res.matchups);
      setTotalGames(res.total_games);
      setUpdatedAt(res.updated_at);
    } catch (err) {
      console.error('Tournament run error:', err);
    } finally {
      setIsRunningLive(false);
    }
  };

  useEffect(() => {
    fetchLeaderboard();
  }, []);

  // Helper to find matchup win rate between agentA and agentB
  const getMatchupWinRate = (agentAName: string, agentBName: string): number | null => {
    const direct = matchups.find((m) => m.agent_a === agentAName && m.agent_b === agentBName);
    if (direct) return direct.win_rate_a;

    const reverse = matchups.find((m) => m.agent_a === agentBName && m.agent_b === agentAName);
    if (reverse) return 1.0 - reverse.win_rate_a;

    return null;
  };

  return (
    <div
      className="elo-matrix-heatmap"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.75rem',
        padding: '0.5rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Trophy size={16} color="#F59E0B" />
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#F1F5F9', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Live Round-Robin Tournament Elo & Empirical Win-Rate Matrix
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ fontSize: '0.72rem', color: '#94A3B8' }}>
            Aggiornato: {updatedAt || 'Inizializzazione...'} ({totalGames} partite totali)
          </div>

          <button
            onClick={handleRunLiveTournament}
            disabled={isRunningLive || isLoading}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.25rem 0.6rem',
              fontSize: '0.72rem',
              fontWeight: 700,
              cursor: isRunningLive || isLoading ? 'not-allowed' : 'pointer',
              opacity: isRunningLive || isLoading ? 0.6 : 1,
              boxShadow: '0 2px 6px rgba(16, 185, 129, 0.3)',
            }}
            title="Lancia un torneo live su mappa USA con tutti gli agenti"
          >
            {isRunningLive ? <RefreshCw size={11} className="spin" /> : <Play size={11} />}
            <span>{isRunningLive ? 'Calcolo Torneo...' : 'Lancia Torneo Live'}</span>
          </button>

          <button
            onClick={fetchLeaderboard}
            disabled={isLoading}
            style={{
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '6px',
              padding: '0.25rem 0.5rem',
              color: '#94A3B8',
              cursor: 'pointer',
            }}
            title="Ricarica classifica"
          >
            <RefreshCw size={12} />
          </button>
        </div>
      </div>

      {/* Grid Table */}
      <div style={{ overflowX: 'auto' }}>
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '0.75rem',
            textAlign: 'center',
          }}
        >
          <thead>
            <tr style={{ background: 'rgba(30, 41, 59, 0.6)', color: '#94A3B8', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
              <th style={{ textAlign: 'left', padding: '0.4rem 0.6rem', fontWeight: 600 }}>Rank / Agente</th>
              <th style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>Elo Rating</th>
              <th style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>Win Rate Globale</th>
              <th style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>W / L / D</th>
              <th style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>Score Medio</th>
              {agents.map((ag) => (
                <th key={ag.agent_id} style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>
                  vs {ag.name.replace(' Trained', '').replace(' Bot', '')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {agents.map((rowAgent, rIdx) => (
              <tr
                key={rowAgent.agent_id}
                style={{
                  borderBottom: '1px solid rgba(255,255,255,0.04)',
                  background: rIdx % 2 === 0 ? 'rgba(15, 23, 42, 0.3)' : 'transparent',
                }}
              >
                {/* Agent Name & Rank */}
                <td style={{ textAlign: 'left', padding: '0.45rem 0.6rem', fontWeight: 600, color: '#F1F5F9' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    {rIdx === 0 ? <Award size={13} color="#F59E0B" /> : <span style={{ color: '#64748B', width: 13 }}>#{rIdx + 1}</span>}
                    <span>{rowAgent.name}</span>
                  </div>
                </td>

                {/* Elo */}
                <td style={{ padding: '0.45rem 0.6rem', fontFamily: 'monospace', fontWeight: 700, color: '#38BDF8' }}>
                  {rowAgent.elo.toFixed(0)}
                </td>

                {/* Overall Win% */}
                <td style={{ padding: '0.45rem 0.6rem', fontFamily: 'monospace', fontWeight: 700, color: rowAgent.win_rate >= 0.5 ? '#34D399' : '#F43F5E' }}>
                  {(rowAgent.win_rate * 100).toFixed(1)}%
                </td>

                {/* W / L / D */}
                <td style={{ padding: '0.45rem 0.6rem', fontFamily: 'monospace', color: '#94A3B8', fontSize: '0.7rem' }}>
                  {rowAgent.wins} / {rowAgent.losses} / {rowAgent.draws}
                </td>

                {/* Avg Score */}
                <td style={{ padding: '0.45rem 0.6rem', fontFamily: 'monospace', color: '#CBD5E1', fontWeight: 600 }}>
                  {rowAgent.avg_score.toFixed(1)}
                </td>

                {/* Head-to-Head Empirical Win% Cells */}
                {agents.map((colAgent) => {
                  if (rowAgent.agent_id === colAgent.agent_id) {
                    return (
                      <td key={colAgent.agent_id} style={{ background: 'rgba(30, 41, 59, 0.3)', color: '#475569', fontSize: '0.7rem' }}>
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
                        padding: '0.35rem 0.5rem',
                        fontFamily: 'monospace',
                        fontWeight: 600,
                        backgroundColor: isHigh
                          ? 'rgba(16, 185, 129, 0.15)'
                          : isLow
                          ? 'rgba(244, 63, 94, 0.15)'
                          : 'rgba(59, 130, 246, 0.08)',
                        color: isHigh ? '#34D399' : isLow ? '#F43F5E' : '#94A3B8',
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
                <td colSpan={5 + agents.length} style={{ padding: '1.5rem', color: '#64748B', textAlign: 'center' }}>
                  Caricamento dati torneo in corso...
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
