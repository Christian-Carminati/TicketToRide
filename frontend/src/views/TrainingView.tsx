import React, { useState, useEffect, useMemo } from 'react';
import { useTrainingStream } from '../hooks/useTrainingStream';
import { LineChartSVG } from '../components/charts/LineChartSVG';
import { TelemetryCard } from '../components/charts/TelemetryCard';
import { api } from '../api/client';
import { ExperimentRecordDTO } from '../api/types';

export const TrainingView: React.FC = () => {
  const {
    status,
    telemetryHistory,
    isConnected,
    error,
    startTraining,
    stopTraining,
  } = useTrainingStream();

  const [configName, setConfigName] = useState('ppo_usa.yaml');
  const [opponentType, setOpponentType] = useState<'random' | 'greedy' | 'strategic'>('greedy');
  const [overrideTimesteps, setOverrideTimesteps] = useState<number>(15000);
  const [seed, setSeed] = useState<number>(42);
  const [experiments, setExperiments] = useState<ExperimentRecordDTO[]>([]);
  const [isLoadingExperiments, setIsLoadingExperiments] = useState(false);

  const loadExperiments = () => {
    setIsLoadingExperiments(true);
    api.listExperiments()
      .then((data) => setExperiments(data))
      .catch((err) => console.error('Failed to load experiments:', err))
      .finally(() => setIsLoadingExperiments(false));
  };

  useEffect(() => {
    loadExperiments();
  }, []);

  useEffect(() => {
    if (!status?.is_training) {
      loadExperiments();
    }
  }, [status?.is_training]);

  const handleStart = () => {
    startTraining({
      config_name: configName,
      override_timesteps: overrideTimesteps,
      seed: seed,
      opponent_type: opponentType,
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
      name: 'Policy Entropy',
      color: '#7E22CE',
      data: telemetryHistory
        .filter((t) => t.entropy !== null && t.entropy !== undefined)
        .map((t) => ({ x: t.step, y: t.entropy! })),
    },
  ], [telemetryHistory]);

  const winRateSeries = useMemo(() => [
    {
      id: 'win_rate',
      name: 'Estimated Win Rate %',
      color: '#0D9488',
      data: telemetryHistory
        .filter((t) => t.win_rate !== null && t.win_rate !== undefined)
        .map((t) => ({ x: t.step, y: t.win_rate! * 100 })),
    },
  ], [telemetryHistory]);

  return (
    <div className="training-view" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Dynamic Experiments History & Benchmark Card */}
      <div
        className="steampunk-panel"
        style={{
          padding: '1.25rem',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 800, color: '#23140C', fontFamily: "'Cinzel Decorative', Georgia, serif" }}>
              📊 Empirical Benchmark Ledger & Saved Models
            </h3>
            <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.8rem', color: '#5A3822', fontFamily: "'Crimson Pro', serif" }}>
              Live telemetry metrics loaded dynamically from experiment registry
            </p>
          </div>

          <button
            onClick={loadExperiments}
            className="steampunk-btn"
            style={{ padding: '0.3rem 0.75rem', fontSize: '0.78rem' }}
          >
            🔄 Refresh Registry
          </button>
        </div>

        {experiments.length === 0 ? (
          <div style={{ padding: '1rem', textAlign: 'center', color: '#785A42', fontSize: '0.85rem', fontStyle: 'italic', fontFamily: "'Crimson Pro', serif" }}>
            {isLoadingExperiments ? 'Loading experiments...' : 'No logged experiments found. Launch training below to log real convergence metrics.'}
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.75rem' }}>
            {experiments.slice(-4).reverse().map((exp) => (
              <div
                key={exp.experiment_id}
                style={{
                  background: '#FAF5EB',
                  padding: '0.75rem 1rem',
                  borderRadius: '8px',
                  border: '1.5px solid #C59B27',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.3rem',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                    {exp.name}
                  </span>
                  <span style={{ background: '#EADBBE', border: '1px solid #C59B27', padding: '0.1rem 0.4rem', borderRadius: '4px', fontSize: '0.7rem', color: '#23140C', fontWeight: 800, fontFamily: "'Courier Prime', monospace" }}>
                    {exp.algorithm.toUpperCase()}
                  </span>
                </div>

                <div style={{ fontSize: '0.75rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
                  Seed: {exp.seed} | ID: {exp.experiment_id}
                </div>

                <div style={{ marginTop: '0.2rem', fontSize: '0.75rem', color: '#23140C', fontFamily: "'Courier Prime', monospace", lineHeight: '1.4' }}>
                  {Object.entries(exp.metrics).map(([key, val]) => (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#5A3822' }}>{key.replace(/_/g, ' ')}:</span>
                      <strong style={{ color: typeof val === 'number' && val > 0.5 ? '#15803D' : '#23140C' }}>
                        {typeof val === 'number' ? (val <= 1.0 && val > 0 && !Number.isInteger(val) ? `${(val * 100).toFixed(1)}%` : val.toFixed(2)) : String(val)}
                      </strong>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Configuration & Control Bar with Opponent Selection */}
      <div
        className="steampunk-panel"
        style={{
          padding: '0.85rem 1.25rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
          {/* Algorithm Config */}
          <label style={{ fontSize: '0.82rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>Algorithm:</label>
          <select
            value={configName}
            disabled={isTraining}
            onChange={(e) => setConfigName(e.target.value)}
            style={{
              backgroundColor: '#FAF5EB',
              color: '#23140C',
              border: '1.5px solid #8C6305',
              borderRadius: '6px',
              padding: '0.3rem 0.6rem',
              fontSize: '0.82rem',
              fontWeight: 700,
              fontFamily: "'Playfair Display', Georgia, serif",
            }}
          >
            <option value="ppo_usa.yaml">⭐ Masked PPO (Recommended - Actor Critic)</option>
            <option value="dqn_usa.yaml">Double-DQN (Deep Q-Network)</option>
          </select>

          {/* Opponent Selector */}
          <label style={{ fontSize: '0.82rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>Opponent:</label>
          <select
            value={opponentType}
            disabled={isTraining}
            onChange={(e) => setOpponentType(e.target.value as 'random' | 'greedy' | 'strategic')}
            style={{
              backgroundColor: '#FAF5EB',
              color: '#23140C',
              border: '1.5px solid #8C6305',
              borderRadius: '6px',
              padding: '0.3rem 0.6rem',
              fontSize: '0.82rem',
              fontWeight: 700,
              fontFamily: "'Playfair Display', Georgia, serif",
            }}
          >
            <option value="greedy">⭐ GreedyBot (Route Competitive Pressure)</option>
            <option value="strategic">StrategicBot (Ticket Optimization)</option>
            <option value="random">RandomBot (Exploration)</option>
          </select>

          {/* Timesteps */}
          <label style={{ fontSize: '0.82rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>Steps:</label>
          <input
            type="number"
            disabled={isTraining}
            value={overrideTimesteps}
            onChange={(e) => setOverrideTimesteps(Number(e.target.value))}
            style={{
              width: '85px',
              backgroundColor: '#FAF5EB',
              color: '#23140C',
              border: '1.5px solid #8C6305',
              borderRadius: '6px',
              padding: '0.3rem 0.5rem',
              fontSize: '0.82rem',
              fontFamily: "'Courier Prime', monospace",
              fontWeight: 700,
            }}
          />

          {/* Seed */}
          <label style={{ fontSize: '0.82rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>Seed:</label>
          <input
            type="number"
            disabled={isTraining}
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            style={{
              width: '65px',
              backgroundColor: '#FAF5EB',
              color: '#23140C',
              border: '1.5px solid #8C6305',
              borderRadius: '6px',
              padding: '0.3rem 0.5rem',
              fontSize: '0.82rem',
              fontFamily: "'Courier Prime', monospace",
              fontWeight: 700,
            }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {/* WebSocket Status Indicator */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem', fontFamily: "'Courier Prime', monospace" }}>
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: isConnected ? '#15803D' : '#B91C1C',
                boxShadow: isConnected ? '0 0 6px #15803D' : 'none',
              }}
            />
            <span style={{ color: '#4A2F1D' }}>{isConnected ? 'Telemetry Online' : 'Offline'}</span>
          </div>

          {isTraining ? (
            <button
              onClick={stopTraining}
              className="steampunk-btn"
              style={{
                background: 'linear-gradient(180deg, #F87171 0%, #DC2626 50%, #991B1B 100%)',
                color: '#FFFFFF',
                border: '1px solid #7F1D1D',
                padding: '0.4rem 1rem',
                fontSize: '0.82rem',
              }}
            >
              ⏹️ Halt Training
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={!isConnected}
              className="steampunk-btn"
              style={{
                background: 'linear-gradient(180deg, #86EFAC 0%, #16A34A 50%, #14532D 100%)',
                color: '#FFFFFF',
                border: '1px solid #14532D',
                padding: '0.4rem 1.1rem',
                fontSize: '0.82rem',
                opacity: isConnected ? 1 : 0.5,
              }}
            >
              ▶️ Launch Training Session
            </button>
          )}
        </div>
      </div>

      {/* Live Training Status Bar */}
      <div
        className="steampunk-panel"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0.6rem 1rem',
          fontSize: '0.82rem',
          background: isTraining ? '#FAF0DA' : 'linear-gradient(180deg, #FAF4E6 0%, #EADBBE 100%)',
        }}
      >
        <span style={{ color: '#23140C', fontWeight: 700, fontFamily: "'Playfair Display', Georgia, serif" }}>
          {isTraining
            ? `⚡ Training Active: Step ${status?.current_step} / ${status?.total_timesteps} (${status?.algorithm?.toUpperCase()}) vs ${opponentType.toUpperCase()}`
            : telemetryHistory.length > 0
            ? `✅ Telemetry Acquired: ${telemetryHistory.length} steps sampled in buffer`
            : '💡 Set parameters and click "Launch Training Session" to record live policy gradient curves.'}
        </span>
        <span style={{ color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
          {telemetryHistory.length} live samples
        </span>
      </div>

      {error && (
        <div style={{ padding: '0.75rem', backgroundColor: '#FEE2E2', border: '1.5px solid #DC2626', borderRadius: '8px', color: '#991B1B', fontSize: '0.85rem', fontFamily: "'Playfair Display', serif" }}>
          ⚠️ {error}
        </div>
      )}

      {/* KPI Cards Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
        <TelemetryCard
          label="Current Step"
          value={latest ? `${latest.step.toLocaleString()} / ${(status?.total_timesteps || overrideTimesteps).toLocaleString()}` : '--'}
          subtext={latest ? `Episodes: ${latest.episode}` : 'Waiting for training'}
          icon="⏱️"
        />
        <TelemetryCard
          label="Mean Reward (20-ep)"
          value={latest ? `${latest.mean_reward.toFixed(2)}` : '--'}
          subtext={latest ? `Instant: ${latest.reward.toFixed(2)}` : undefined}
          color="#15803D"
          icon="🎯"
        />
        <TelemetryCard
          label="Policy / Value Loss"
          value={latest && latest.policy_loss !== null ? `${latest.policy_loss?.toFixed(3)} / ${latest.value_loss?.toFixed(3)}` : '--'}
          subtext="Optimization Gradient"
          color="#9E6B00"
          icon="📉"
        />
        <TelemetryCard
          label="Throughput (FPS)"
          value={latest && latest.fps ? `${latest.fps.toFixed(0)} FPS` : '--'}
          subtext="Sampling Rate"
          color="#7E22CE"
          icon="⚡"
        />
      </div>

      {/* 2x2 Telemetry Charts Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '1.25rem' }}>
        <LineChartSVG
          title="Reward Curves (Mean vs Instant)"
          series={rewardSeries}
          yLabel="Reward"
          xLabel="Training Step"
          height={240}
        />
        <LineChartSVG
          title="Loss Convergence (Policy vs Value Loss)"
          series={lossSeries}
          yLabel="Loss"
          xLabel="Training Step"
          height={240}
        />
        <LineChartSVG
          title="Policy Entropy Exploration"
          series={entropySeries}
          yLabel="Entropy"
          xLabel="Training Step"
          height={240}
        />
        <LineChartSVG
          title="Estimated Win Rate (%)"
          series={winRateSeries}
          yLabel="Win Rate %"
          xLabel="Training Step"
          height={240}
        />
      </div>
    </div>
  );
};
