import React, { useState, useEffect } from 'react';
import { Trophy, Award, RefreshCw, Play, Settings, CheckSquare, Square, Bot, BrainCircuit, MapPin } from 'lucide-react';
import { api } from '../../api/client';
import { TournamentAgentDTO, TournamentMatchupDTO, TournamentParticipantOptionDTO } from '../../api/types';

export const EloMatrixHeatmap: React.FC = () => {
  const [agents, setAgents] = useState<TournamentAgentDTO[]>([]);
  const [matchups, setMatchups] = useState<TournamentMatchupDTO[]>([]);
  const [totalGames, setTotalGames] = useState<number>(0);
  const [updatedAt, setUpdatedAt] = useState<string>('');
  const [mapName, setMapName] = useState<'usa' | 'mini'>('usa');
  const [availableOptions, setAvailableOptions] = useState<TournamentParticipantOptionDTO[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [gamesPerPair, setGamesPerPair] = useState<number>(10);
  const [isConfigOpen, setIsConfigOpen] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isRunningLive, setIsRunningLive] = useState<boolean>(false);
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
          // Default selection if none selected yet
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
      alert('Seleziona almeno 2 partecipanti per poter eseguire un torneo round-robin.');
      return;
    }

    setIsRunningLive(true);
    setStatusMessage('Esecuzione torneo round-robin in corso...');
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
      setStatusMessage(`Torneo completato con successo (${res.total_games} partite).`);
      setTimeout(() => setStatusMessage(null), 4000);
    } catch (err) {
      console.error('Tournament run error:', err);
      setStatusMessage('Errore durante l\'esecuzione del torneo.');
    } finally {
      setIsRunningLive(false);
    }
  };

  // Helper to find matchup win rate between agentA and agentB
  const getMatchupWinRate = (agentAName: string, agentBName: string): number | null => {
    const direct = matchups.find((m) => m.agent_a === agentAName && m.agent_b === agentBName);
    if (direct) return direct.win_rate_a;

    const reverse = matchups.find((m) => m.agent_a === agentBName && m.agent_b === agentAName);
    if (reverse) return 1.0 - reverse.win_rate_a;

    return null;
  };

  const totalCalculatedGames =
    selectedIds.length >= 2 ? (selectedIds.length * (selectedIds.length - 1)) / 2 * gamesPerPair : 0;

  const baselines = availableOptions.filter((o) => o.category === 'baseline');
  const checkpoints = availableOptions.filter((o) => o.category === 'checkpoint');

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
      {/* Top Action Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Trophy size={16} color="#F59E0B" />
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#F1F5F9', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Torneo Round-Robin Elo & Win-Rate Matrix
          </span>
          <span style={{ fontSize: '0.7rem', padding: '0.15rem 0.4rem', borderRadius: '4px', background: 'rgba(59, 130, 246, 0.2)', color: '#60A5FA', fontWeight: 600 }}>
            Mappa: {mapName.toUpperCase()}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <div style={{ fontSize: '0.72rem', color: '#94A3B8' }}>
            Aggiornato: {updatedAt || 'Caricamento...'} ({totalGames} partite totali)
          </div>

          {/* Toggle Configuration Panel */}
          <button
            onClick={() => setIsConfigOpen(!isConfigOpen)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              background: isConfigOpen ? '#3B82F6' : 'rgba(255, 255, 255, 0.08)',
              color: '#F8FAFC',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              borderRadius: '6px',
              padding: '0.3rem 0.65rem',
              fontSize: '0.74rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            title="Personalizza partecipanti e opzioni del torneo"
          >
            <Settings size={12} />
            <span>Scegli Partecipanti ({selectedIds.length}/{availableOptions.length})</span>
          </button>

          {/* Run Live Tournament Button */}
          <button
            onClick={handleRunCustomTournament}
            disabled={isRunningLive || isLoading || selectedIds.length < 2}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              background: isRunningLive
                ? 'rgba(16, 185, 129, 0.4)'
                : 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.3rem 0.75rem',
              fontSize: '0.74rem',
              fontWeight: 700,
              cursor: isRunningLive || selectedIds.length < 2 ? 'not-allowed' : 'pointer',
              opacity: isRunningLive || selectedIds.length < 2 ? 0.6 : 1,
              boxShadow: '0 2px 8px rgba(16, 185, 129, 0.3)',
            }}
            title="Lancia il torneo tra tutti i partecipanti selezionati"
          >
            {isRunningLive ? <RefreshCw size={12} className="spin" /> : <Play size={12} />}
            <span>{isRunningLive ? 'Simulazione Partite...' : 'Lancia Torneo'}</span>
          </button>

          <button
            onClick={fetchLeaderboard}
            disabled={isLoading}
            style={{
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '6px',
              padding: '0.3rem 0.5rem',
              color: '#94A3B8',
              cursor: 'pointer',
            }}
            title="Ricarica classifica"
          >
            <RefreshCw size={12} />
          </button>
        </div>
      </div>

      {/* Notification Toast */}
      {statusMessage && (
        <div
          style={{
            background: 'rgba(16, 185, 129, 0.15)',
            border: '1px solid #10B981',
            borderRadius: '6px',
            padding: '0.4rem 0.8rem',
            color: '#34D399',
            fontSize: '0.75rem',
            fontWeight: 600,
          }}
        >
          {statusMessage}
        </div>
      )}

      {/* Config Drawer / Drawer Panel */}
      {isConfigOpen && (
        <div
          style={{
            background: '#0B1120',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '8px',
            padding: '0.85rem 1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#38BDF8' }}>
              ⚙️ Configurazione Concorrenti & Parametri Torneo
            </div>

            {/* Quick Selection Presets */}
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              <button
                onClick={handleSelectAll}
                style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', color: '#CBD5E1', padding: '0.2rem 0.5rem', fontSize: '0.7rem', cursor: 'pointer' }}
              >
                Tutti ({availableOptions.length})
              </button>
              <button
                onClick={handleSelectModelsOnly}
                style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', color: '#CBD5E1', padding: '0.2rem 0.5rem', fontSize: '0.7rem', cursor: 'pointer' }}
              >
                Solo Modelli RL ({checkpoints.length})
              </button>
              <button
                onClick={handleSelectBaselinesOnly}
                style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', color: '#CBD5E1', padding: '0.2rem 0.5rem', fontSize: '0.7rem', cursor: 'pointer' }}
              >
                Solo Baseline ({baselines.length})
              </button>
            </div>
          </div>

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
                    background: isSelected ? 'rgba(59, 130, 246, 0.15)' : 'rgba(30, 41, 59, 0.4)',
                    border: `1px solid ${isSelected ? '#3B82F6' : 'rgba(255, 255, 255, 0.06)'}`,
                    borderRadius: '6px',
                    padding: '0.5rem 0.65rem',
                    cursor: 'pointer',
                    transition: 'all 0.12s ease',
                  }}
                >
                  <div style={{ marginTop: '0.1rem', color: isSelected ? '#38BDF8' : '#64748B' }}>
                    {isSelected ? <CheckSquare size={15} /> : <Square size={15} />}
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      {isCkpt ? <BrainCircuit size={13} color="#A855F7" /> : <Bot size={13} color="#38BDF8" />}
                      <span style={{ fontSize: '0.78rem', fontWeight: 700, color: isSelected ? '#FFFFFF' : '#94A3B8' }}>
                        {opt.name}
                      </span>
                    </div>
                    {opt.description && (
                      <span style={{ fontSize: '0.68rem', color: '#64748B', lineHeight: '1.2' }}>
                        {opt.description}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Options Row (Games per pair & Map) */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', paddingTop: '0.5rem', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
              {/* Games Per Pair */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.74rem', color: '#94A3B8', fontWeight: 600 }}>Partite per scontro:</span>
                <select
                  value={gamesPerPair}
                  onChange={(e) => setGamesPerPair(Number(e.target.value))}
                  style={{
                    backgroundColor: '#1E293B',
                    color: '#F8FAFC',
                    border: '1px solid rgba(255,255,255,0.15)',
                    borderRadius: '4px',
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                  }}
                >
                  <option value={5}>5 partite (Veloce)</option>
                  <option value={10}>10 partite (Bilanciato)</option>
                  <option value={15}>15 partite (Accurato)</option>
                  <option value={25}>25 partite (Deep Benchmark)</option>
                </select>
              </div>

              {/* Map Selection */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <MapPin size={13} color="#94A3B8" />
                <span style={{ fontSize: '0.74rem', color: '#94A3B8', fontWeight: 600 }}>Mappa:</span>
                <select
                  value={mapName}
                  onChange={(e) => setMapName(e.target.value as 'usa' | 'mini')}
                  style={{
                    backgroundColor: '#1E293B',
                    color: '#F8FAFC',
                    border: '1px solid rgba(255,255,255,0.15)',
                    borderRadius: '4px',
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                  }}
                >
                  <option value="usa">USA Full (Ufficiale)</option>
                  <option value="mini">Mini Synthetic (Ultra Rapido)</option>
                </select>
              </div>
            </div>

            {/* Total Calculation Stats & Confirmation */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span style={{ fontSize: '0.72rem', color: '#38BDF8', fontFamily: 'monospace' }}>
                {selectedIds.length} concorrenti = {totalCalculatedGames} partite stimate
              </span>
              <button
                onClick={handleRunCustomTournament}
                disabled={isRunningLive || selectedIds.length < 2}
                style={{
                  background: '#3B82F6',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '0.35rem 0.85rem',
                  fontSize: '0.76rem',
                  fontWeight: 700,
                  cursor: isRunningLive || selectedIds.length < 2 ? 'not-allowed' : 'pointer',
                }}
              >
                {isRunningLive ? 'Calcolo in corso...' : 'Avvia Simulazione'}
              </button>
            </div>
          </div>
        </div>
      )}

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
              <th style={{ textAlign: 'left', padding: '0.4rem 0.6rem', fontWeight: 600 }}>Rank / Concorrente</th>
              <th style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>Elo Rating</th>
              <th style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>Win Rate Globale</th>
              <th style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>W / L / D</th>
              <th style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>Score Medio</th>
              {agents.map((ag) => (
                <th key={ag.agent_id} style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>
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
