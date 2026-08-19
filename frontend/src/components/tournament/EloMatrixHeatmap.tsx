import React from 'react';
import { Trophy, Award } from 'lucide-react';

export interface AgentTournamentEntry {
  agentId: string;
  name: string;
  elo: number;
  winRate: number; // 0.0 to 1.0
  gamesPlayed: number;
}

export interface MatchupStat {
  agentA: string;
  agentB: string;
  winRateA: number; // 0.0 to 1.0
  gamesPlayed: number;
}

interface EloMatrixHeatmapProps {
  agents?: AgentTournamentEntry[];
  matchups?: MatchupStat[];
}

const DEFAULT_AGENTS: AgentTournamentEntry[] = [
  { agentId: 'ppo_v4', name: 'PPO Recurrent v4', elo: 1485, winRate: 0.78, gamesPlayed: 120 },
  { agentId: 'ppo_v1', name: 'PPO Baseline v1', elo: 1320, winRate: 0.62, gamesPlayed: 120 },
  { agentId: 'dqn_v2', name: 'DQN Masked v2', elo: 1240, winRate: 0.54, gamesPlayed: 100 },
  { agentId: 'strategic', name: 'Heuristic Strategic', elo: 1180, winRate: 0.48, gamesPlayed: 140 },
  { agentId: 'greedy', name: 'Greedy Score Bot', elo: 1020, winRate: 0.35, gamesPlayed: 140 },
  { agentId: 'random', name: 'Uniform Random', elo: 800, winRate: 0.05, gamesPlayed: 150 },
];

export const EloMatrixHeatmap: React.FC<EloMatrixHeatmapProps> = ({
  agents = DEFAULT_AGENTS,
}) => {
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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Trophy size={16} color="#F59E0B" />
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#F1F5F9', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Tournament Round-Robin Elo & Win-Rate Matrix
          </span>
        </div>
        <div style={{ fontSize: '0.72rem', color: '#94A3B8' }}>
          Updated across 770 benchmark games
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
              <th style={{ textAlign: 'left', padding: '0.4rem 0.6rem', fontWeight: 600 }}>Rank / Agent</th>
              <th style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>Elo</th>
              <th style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>Overall Win%</th>
              {agents.map((ag) => (
                <th key={ag.agentId} style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>
                  vs {ag.name.split(' ')[0]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {agents.map((rowAgent, rIdx) => (
              <tr
                key={rowAgent.agentId}
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
                  {rowAgent.elo}
                </td>

                {/* Overall Win% */}
                <td style={{ padding: '0.45rem 0.6rem', fontFamily: 'monospace', fontWeight: 700, color: rowAgent.winRate >= 0.5 ? '#34D399' : '#F43F5E' }}>
                  {(rowAgent.winRate * 100).toFixed(0)}%
                </td>

                {/* Head-to-Head Cells */}
                {agents.map((colAgent) => {
                  if (rowAgent.agentId === colAgent.agentId) {
                    return (
                      <td key={colAgent.agentId} style={{ background: 'rgba(30, 41, 59, 0.3)', color: '#475569', fontSize: '0.7rem' }}>
                        —
                      </td>
                    );
                  }

                  // Simulated / calculated Elo win probability: P = 1 / (1 + 10^((EloB - EloA) / 400))
                  const eloDiff = colAgent.elo - rowAgent.elo;
                  const winProb = 1 / (1 + Math.pow(10, eloDiff / 400));
                  const pct = Math.round(winProb * 100);

                  const isHigh = pct >= 60;
                  const isLow = pct <= 40;

                  return (
                    <td
                      key={colAgent.agentId}
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
          </tbody>
        </table>
      </div>
    </div>
  );
};
