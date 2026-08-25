import React, { useState, useEffect, useMemo } from 'react';
import { useTrainingStream } from '../hooks/useTrainingStream';
import { LineChartSVG } from '../components/charts/LineChartSVG';
import { TelemetryCard } from '../components/charts/TelemetryCard';
import { api } from '../api/client';
import { CheckpointDTO } from '../api/types';
import { Activity, Flame, Award, Cpu, ShieldCheck, Zap, RefreshCw } from 'lucide-react';

export const TrainingView: React.FC = () => {
  const {
    status,
    telemetryHistory,
    isConnected,
    error,
    startTraining,
    stopTraining,
  } = useTrainingStream();

  const [selectedAlgo, setSelectedAlgo] = useState<'ppo' | 'dqn' | 'recurrent_ppo' | 'self_play_ppo' | 'alphazero'>('ppo');
  const [opponentType, setOpponentType] = useState<string>('greedy');
  const [overrideTimesteps, setOverrideTimesteps] = useState<number>(15000);
  const [numSimulations, setNumSimulations] = useState<number>(30);
  const [mapName, setMapName] = useState<'usa' | 'mini'>('usa');
  const [seed, setSeed] = useState<number>(42);
  const [checkpoints, setCheckpoints] = useState<CheckpointDTO[]>([]);
  const [isLoadingCheckpoints, setIsLoadingCheckpoints] = useState(false);

  const loadCheckpoints = () => {
    setIsLoadingCheckpoints(true);
    api.listCheckpoints()
      .then((data) => setCheckpoints(data))
      .catch((err) => console.error('Failed to load checkpoints:', err))
      .finally(() => setIsLoadingCheckpoints(false));
  };

  useEffect(() => {
    loadCheckpoints();
  }, []);

  useEffect(() => {
    if (!status?.is_training) {
      loadCheckpoints();
    }
  }, [status?.is_training]);

  const handleStart = () => {
    startTraining({
      config_name: `${selectedAlgo}_${mapName}.yaml`,
      algorithm_type: selectedAlgo,
      override_timesteps: overrideTimesteps,
      num_simulations: numSimulations,
      seed: seed,
      opponent_type: opponentType,
      map_name: mapName,
    });
  };

  const isTraining = Boolean(status?.is_training);
  const latest = telemetryHistory.length > 0 ? telemetryHistory[telemetryHistory.length - 1] : null;

  const rewardSeries = useMemo(() => [
    {
      id: 'mean_reward',
      name: 'Mean Reward (Rolling)',
      color: '#15803D',
      data: telemetryHistory.map((t) => ({ x: t.step, y: t.mean_reward })),
    },
    {
      id: 'step_reward',
      name: 'Step Reward',
      color: '#1D4ED8',
      data: telemetryHistory.map((t) => ({ x: t.step, y: t.reward })),
    },
  ], [telemetryHistory]);

  const lossSeries = useMemo(() => [
    {
      id: 'policy_loss',
      name: 'Policy Loss',
      color: '#B8860B',
      data: telemetryHistory
        .filter((t) => t.policy_loss !== null && t.policy_loss !== undefined)
        .map((t) => ({ x: t.step, y: t.policy_loss! })),
    },
    {
      id: 'value_loss',
      name: 'Value Loss',
      color: '#B91C1C',
      data: telemetryHistory
        .filter((t) => t.value_loss !== null && t.value_loss !== undefined)
        .map((t) => ({ x: t.step, y: t.value_loss! })),
    },
  ], [telemetryHistory]);

  const entropySeries = useMemo(() => [
    {
      id: 'entropy',
      name: 'Policy Entropy / Exploration',
      color: '#7E22CE',
      data: telemetryHistory
        .filter((t) => t.entropy !== null && t.entropy !== undefined)
        .map((t) => ({ x: t.step, y: t.entropy! })),
    },
  ], [telemetryHistory]);

  const ALGO_CARDS: Array<{
    id: 'ppo' | 'dqn' | 'recurrent_ppo' | 'self_play_ppo' | 'alphazero';
    name: string;
    lesson: string;
    badge: string;
    description: string;
    icon: React.ReactNode;
  }> = [
    {
      id: 'ppo',
      name: 'CleanRL PPO',
      lesson: 'Lessons 4 & 6',
      badge: 'Feedforward',
      description: 'Orthogonal init, GAE, KL early stopping, and advantage normalization.',
      icon: <Zap size={18} color="#C59B27" />,
    },
    {
      id: 'alphazero',
      name: 'AlphaZero Dual Head',
      lesson: 'Lesson 12',
      badge: 'PUCT MCTS',
      description: 'Joint Policy-Value Network guided by self-play Monte Carlo Tree Search.',
      icon: <Award size={18} color="#B91C1C" />,
    },
    {
      id: 'recurrent_ppo',
      name: 'Recurrent PPO (LSTM)',
      lesson: 'Lesson 8',
      badge: 'POMDP Memory',
      description: 'Sequential LSTM state memory for tracking hidden opponent cards.',
      icon: <Cpu size={18} color="#1D4ED8" />,
    },
    {
      id: 'self_play_ppo',
      name: 'Self-Play Policy Pool',
      lesson: 'Lesson 9',
      badge: 'PFSP Pool',
      description: 'Prioritized Fictitious Self-Play against historical checkpoint generations.',
      icon: <Flame size={18} color="#D97706" />,
    },
    {
      id: 'dqn',
      name: 'Double-DQN',
      lesson: 'Lesson 4',
      badge: 'Q-Learning',
      description: 'Target network stabilization and prioritized experience replay.',
      icon: <Activity size={18} color="#15803D" />,
    },
  ];

  return (
    <div
      className="training-view"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem',
        color: '#23140C',
        fontFamily: "'Crimson Pro', Georgia, serif",
      }}
    >
      {/* Header Banner */}
      <div
        className="steampunk-panel"
        style={{
          padding: '1rem 1.25rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        <div>
          <h2 style={{ margin: 0, fontSize: '1.3rem', fontWeight: 900, color: '#23140C', fontFamily: "'Cinzel Decorative', Georgia, serif" }}>
            🧪 Steam Locomotive Neural Training Furnace
          </h2>
          <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#5A3822', fontFamily: "'Crimson Pro', Georgia, serif", fontStyle: 'italic' }}>
            Train CleanRL PPO, AlphaZero PUCT, Recurrent LSTM, and Self-Play models in real time directly from the workbench.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: isConnected ? 'rgba(21, 128, 61, 0.2)' : 'rgba(185, 28, 28, 0.2)',
              border: `1px solid ${isConnected ? '#15803D' : '#B91C1C'}`,
              borderRadius: '6px',
              padding: '0.35rem 0.75rem',
              fontSize: '0.78rem',
              fontFamily: "'Courier Prime', monospace",
              fontWeight: 700,
              color: isConnected ? '#15803D' : '#B91C1C',
            }}
          >
            <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: isConnected ? '#15803D' : '#B91C1C' }} />
            {isConnected ? 'Telegraph 60Hz Active' : 'Offline'}
          </div>

          <button
            onClick={loadCheckpoints}
            className="steampunk-btn"
            style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', padding: '0.35rem 0.75rem', fontSize: '0.78rem' }}
          >
            <RefreshCw size={13} /> Checkpoint Archive
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: '0.75rem', borderRadius: '8px', background: '#FEE2E2', border: '1.5px solid #DC2626', color: '#991B1B', fontSize: '0.85rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* 1. Algorithm Selection Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <span style={{ fontSize: '0.9rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
          Select Neural Reinforcement Learning Algorithm:
        </span>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem' }}>
          {ALGO_CARDS.map((card) => {
            const isSelected = selectedAlgo === card.id;
            return (
              <div
                key={card.id}
                onClick={() => !isTraining && setSelectedAlgo(card.id)}
                style={{
                  background: isSelected
                    ? 'linear-gradient(180deg, #FBF4E4 0%, #EAD7B8 100%)'
                    : '#FAF5EB',
                  border: isSelected ? '2px solid #C59B27' : '1.5px solid #D4C09D',
                  borderRadius: '8px',
                  padding: '0.75rem 0.9rem',
                  cursor: isTraining ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.35rem',
                  boxShadow: isSelected ? '0 3px 12px rgba(197, 155, 39, 0.35)' : '0 1px 3px rgba(0,0,0,0.05)',
                  transition: 'all 0.15s ease',
                  opacity: isTraining && !isSelected ? 0.6 : 1,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    {card.icon}
                    <span style={{ fontSize: '0.85rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                      {card.name}
                    </span>
                  </div>
                  <span
                    style={{
                      fontSize: '0.65rem',
                      fontWeight: 800,
                      background: isSelected ? '#C59B27' : '#E8D8C0',
                      color: isSelected ? '#FFFFFF' : '#785A42',
                      padding: '0.1rem 0.4rem',
                      borderRadius: '3px',
                      fontFamily: "'Courier Prime', monospace",
                    }}
                  >
                    {card.badge}
                  </span>
                </div>

                <div style={{ fontSize: '0.72rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
                  {card.lesson}
                </div>

                <div style={{ fontSize: '0.74rem', color: '#5A3822', fontFamily: "'Crimson Pro', Georgia, serif", lineHeight: 1.3 }}>
                  {card.description}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 2. Configuration & Parameter Bar */}
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
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          {/* Timesteps */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <label style={{ fontSize: '0.8rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>
              Timesteps:
            </label>
            <select
              value={overrideTimesteps}
              disabled={isTraining}
              onChange={(e) => setOverrideTimesteps(Number(e.target.value))}
              style={{
                backgroundColor: '#FAF5EB',
                color: '#23140C',
                border: '1.5px solid #8C6305',
                borderRadius: '6px',
                padding: '0.3rem 0.6rem',
                fontSize: '0.8rem',
                fontWeight: 700,
                fontFamily: "'Courier Prime', monospace",
              }}
            >
              <option value={5000}>5,000 Steps (Fast Test)</option>
              <option value={15000}>15,000 Steps (Standard)</option>
              <option value={40000}>40,000 Steps (Convergence)</option>
              <option value={100000}>100,000 Steps (Mastery)</option>
            </select>
          </div>

          {/* AlphaZero MCTS Sims Selector if AlphaZero is active */}
          {selectedAlgo === 'alphazero' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <label style={{ fontSize: '0.8rem', color: '#B91C1C', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>
                PUCT Sims:
              </label>
              <select
                value={numSimulations}
                disabled={isTraining}
                onChange={(e) => setNumSimulations(Number(e.target.value))}
                style={{
                  backgroundColor: '#FAF5EB',
                  color: '#B91C1C',
                  border: '1.5px solid #B91C1C',
                  borderRadius: '6px',
                  padding: '0.3rem 0.6rem',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  fontFamily: "'Courier Prime', monospace",
                }}
              >
                <option value={15}>15 Sims / Move (Rapid)</option>
                <option value={30}>30 Sims / Move (Balanced)</option>
                <option value={60}>60 Sims / Move (Deep Search)</option>
              </select>
            </div>
          )}

          {/* Opponent Type (Only if not Self-Play or AlphaZero) */}
          {selectedAlgo !== 'self_play_ppo' && selectedAlgo !== 'alphazero' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <label style={{ fontSize: '0.8rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>
                Sparring Opponent:
              </label>
              <select
                value={opponentType}
                disabled={isTraining}
                onChange={(e) => setOpponentType(e.target.value)}
                style={{
                  backgroundColor: '#FAF5EB',
                  color: '#23140C',
                  border: '1.5px solid #8C6305',
                  borderRadius: '6px',
                  padding: '0.3rem 0.6rem',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  fontFamily: "'Playfair Display', Georgia, serif",
                }}
              >
                <option value="greedy">Greedy Score Bot</option>
                <option value="strategic">Strategic Heuristic</option>
                <option value="random">Uniform Random</option>
              </select>
            </div>
          )}

          {/* Map Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <label style={{ fontSize: '0.8rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>
              Survey Map:
            </label>
            <select
              value={mapName}
              disabled={isTraining}
              onChange={(e) => setMapName(e.target.value as 'usa' | 'mini')}
              style={{
                backgroundColor: '#FAF5EB',
                color: '#23140C',
                border: '1.5px solid #8C6305',
                borderRadius: '6px',
                padding: '0.3rem 0.6rem',
                fontSize: '0.8rem',
                fontWeight: 700,
                fontFamily: "'Playfair Display', Georgia, serif",
              }}
            >
              <option value="usa">USA Full (Official 1885)</option>
              <option value="mini">Mini Synthetic (Rapid)</option>
            </select>
          </div>

          {/* Seed Input */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <label style={{ fontSize: '0.8rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>
              Seed:
            </label>
            <input
              type="number"
              value={seed}
              disabled={isTraining}
              onChange={(e) => setSeed(Number(e.target.value))}
              style={{
                width: '55px',
                backgroundColor: '#FAF5EB',
                color: '#23140C',
                border: '1.5px solid #8C6305',
                borderRadius: '6px',
                padding: '0.25rem 0.4rem',
                fontSize: '0.8rem',
                fontWeight: 700,
                fontFamily: "'Courier Prime', monospace",
              }}
            />
          </div>
        </div>

        {/* Start / Stop Button */}
        <div>
          {isTraining ? (
            <button
              onClick={stopTraining}
              className="steampunk-btn"
              style={{
                background: 'linear-gradient(180deg, #FCA5A5 0%, #DC2626 50%, #7F1D1D 100%)',
                color: '#FFFFFF',
                border: '1px solid #7F1D1D',
                padding: '0.45rem 1.25rem',
                fontSize: '0.85rem',
              }}
            >
              🛑 Abort Training Session
            </button>
          ) : (
            <button
              onClick={handleStart}
              className="steampunk-btn"
              style={{
                background: 'linear-gradient(180deg, #86EFAC 0%, #16A34A 50%, #14532D 100%)',
                color: '#FFFFFF',
                border: '1px solid #14532D',
                padding: '0.45rem 1.4rem',
                fontSize: '0.88rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
              }}
            >
              <Flame size={15} color="#FEF08A" />
              <span>Ignite {selectedAlgo.toUpperCase()} Training</span>
            </button>
          )}
        </div>
      </div>

      {/* 3. Live Telemetry KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
        <TelemetryCard
          label="Training Progress"
          value={`${(status?.current_step || 0).toLocaleString()} / ${(status?.total_timesteps || overrideTimesteps).toLocaleString()}`}
          subtext={`Completed Episodes: ${status?.episodes || 0}`}
        />
        <TelemetryCard
          label="Convergence Reward"
          value={latest?.mean_reward !== undefined ? latest.mean_reward.toFixed(2) : (status?.mean_reward || 0).toFixed(2)}
          subtext={latest ? `Step Reward: ${latest.reward.toFixed(2)}` : 'Awaiting telemetry...'}
          color={latest && latest.mean_reward > 0 ? '#15803D' : '#9E6B00'}
        />
        <TelemetryCard
          label="Policy Loss"
          value={latest?.policy_loss !== undefined && latest.policy_loss !== null ? latest.policy_loss.toFixed(4) : '--'}
          subtext={latest?.value_loss !== undefined && latest.value_loss !== null ? `Value Loss: ${latest.value_loss.toFixed(4)}` : 'Awaiting loss values...'}
        />
        <TelemetryCard
          label="Sim Speed / Throughput"
          value={latest?.fps ? `${latest.fps.toFixed(1)} FPS` : '--'}
          subtext={isTraining ? 'Training in progress...' : 'Ready for ignition'}
        />
      </div>

      {/* 4. Telemetry Multi-Charts */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1rem' }}>
        <LineChartSVG
          series={rewardSeries}
          title="📈 Mean Episodic Reward Convergence"
          yLabel="Reward"
          height={240}
        />
        <LineChartSVG
          series={lossSeries}
          title="⚡ Optimization Losses (Policy & Value)"
          yLabel="Loss"
          height={240}
        />
        <LineChartSVG
          series={entropySeries}
          title="🔮 Policy Entropy & Exploration Degree"
          yLabel="Entropy"
          height={240}
        />
      </div>

      {/* 5. Checkpoint Auto-Save Ledger */}
      <div className="steampunk-panel" style={{ padding: '1rem 1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <ShieldCheck size={16} color="#15803D" />
            <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
              Trained Model Archive & Ready Checkpoints (.pt)
            </h3>
          </div>
          <span style={{ fontSize: '0.75rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
            {checkpoints.length} Available Weights
          </span>
        </div>

        {checkpoints.length === 0 ? (
          <div style={{ padding: '1rem', textAlign: 'center', color: '#785A42', fontSize: '0.85rem', fontStyle: 'italic' }}>
            {isLoadingCheckpoints ? 'Loading archive...' : 'No checkpoints found. Start a training session above to generate weights.'}
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.65rem' }}>
            {checkpoints.map((ckpt) => (
              <div
                key={ckpt.checkpoint_id}
                style={{
                  background: '#FAF5EB',
                  padding: '0.6rem 0.85rem',
                  borderRadius: '6px',
                  border: '1.5px solid #C59B27',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div>
                  <div style={{ fontSize: '0.82rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                    {ckpt.name}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
                    {ckpt.algorithm.toUpperCase()} • {ckpt.size_mb} MB • {ckpt.modified_at}
                  </div>
                </div>
                <span
                  style={{
                    fontSize: '0.65rem',
                    fontWeight: 800,
                    padding: '0.15rem 0.4rem',
                    borderRadius: '3px',
                    background: '#15803D',
                    color: '#FFFFFF',
                    fontFamily: "'Courier Prime', monospace",
                  }}
                >
                  READY
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
