import React, { useState } from 'react';
import { useTrainingStream } from '../hooks/useTrainingStream';
import { LineChartSVG } from '../components/charts/LineChartSVG';
import { TelemetryCard } from '../components/charts/TelemetryCard';

export const TrainingView: React.FC = () => {
  const {
    status,
    telemetryHistory,
    isConnected,
    error,
    startTraining,
    stopTraining,
  } = useTrainingStream();

  const [configName, setConfigName] = useState('ppo_mini.yaml');
  const [overrideTimesteps, setOverrideTimesteps] = useState<number>(5000);
  const [seed, setSeed] = useState<number>(42);

  const handleStart = () => {
    startTraining({
      config_name: configName,
      override_timesteps: overrideTimesteps,
      seed: seed,
    });
  };

  const isTraining = Boolean(status?.is_training);
  const latest = telemetryHistory[telemetryHistory.length - 1];

  // Chart data series preparations
  const rewardSeries = [
    {
      id: 'mean_reward',
      name: 'Mean Reward (20-ep)',
      color: '#10B981',
      data: telemetryHistory.map((t) => ({ x: t.step, y: t.mean_reward })),
    },
    {
      id: 'step_reward',
      name: 'Instant Reward',
      color: '#38BDF8',
      data: telemetryHistory.map((t) => ({ x: t.step, y: t.reward })),
    },
  ];

  const lossSeries = [
    {
      id: 'policy_loss',
      name: 'Policy Loss',
      color: '#F59E0B',
      data: telemetryHistory
        .filter((t) => t.policy_loss !== null && t.policy_loss !== undefined)
        .map((t) => ({ x: t.step, y: t.policy_loss! })),
    },
    {
      id: 'value_loss',
      name: 'Value Loss',
      color: '#EF4444',
      data: telemetryHistory
        .filter((t) => t.value_loss !== null && t.value_loss !== undefined)
        .map((t) => ({ x: t.step, y: t.value_loss! })),
    },
  ];

  const entropySeries = [
    {
      id: 'entropy',
      name: 'Policy Entropy',
      color: '#8B5CF6',
      data: telemetryHistory
        .filter((t) => t.entropy !== null && t.entropy !== undefined)
        .map((t) => ({ x: t.step, y: t.entropy! })),
    },
  ];

  const winRateSeries = [
    {
      id: 'win_rate',
      name: 'Estimated Win Rate',
      color: '#06B6D4',
      data: telemetryHistory
        .filter((t) => t.win_rate !== null && t.win_rate !== undefined)
        .map((t) => ({ x: t.step, y: t.win_rate! * 100 })),
    },
  ];

  return (
    <div className="training-view" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Top Configuration & Control Bar */}
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
          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Config YAML:</label>
          <select
            value={configName}
            disabled={isTraining}
            onChange={(e) => setConfigName(e.target.value)}
            style={{
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.6rem',
              fontSize: '0.85rem',
            }}
          >
            <option value="ppo_mini.yaml">ppo_mini.yaml</option>
            <option value="dqn_mini.yaml">dqn_mini.yaml</option>
            <option value="ppo_usa.yaml">ppo_usa.yaml</option>
            <option value="dqn_usa.yaml">dqn_usa.yaml</option>
          </select>

          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Timesteps:</label>
          <input
            type="number"
            disabled={isTraining}
            value={overrideTimesteps}
            onChange={(e) => setOverrideTimesteps(Number(e.target.value))}
            style={{
              width: '90px',
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.5rem',
              fontSize: '0.85rem',
            }}
          />

          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Seed:</label>
          <input
            type="number"
            disabled={isTraining}
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            style={{
              width: '70px',
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.5rem',
              fontSize: '0.85rem',
            }}
          />

          {!isTraining ? (
            <button
              onClick={handleStart}
              style={{
                backgroundColor: '#10B981',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '6px',
                padding: '0.4rem 1rem',
                fontWeight: 600,
                fontSize: '0.85rem',
                cursor: 'pointer',
              }}
            >
              ▶ Start Training
            </button>
          ) : (
            <button
              onClick={stopTraining}
              style={{
                backgroundColor: '#EF4444',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '6px',
                padding: '0.4rem 1rem',
                fontWeight: 600,
                fontSize: '0.85rem',
                cursor: 'pointer',
              }}
            >
              ⏹ Stop Training
            </button>
          )}
        </div>

        {/* Live WebSocket Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              backgroundColor: isConnected ? '#10B981' : '#EF4444',
            }}
          />
          <span style={{ color: isConnected ? '#10B981' : '#EF4444', fontWeight: 600 }}>
            {isConnected ? 'Telemetry Hub Online' : 'Telemetry Disconnected'}
          </span>
        </div>
      </div>

      {error && (
        <div style={{ padding: '0.75rem', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.2)', border: '1px solid #EF4444', color: '#FCA5A5', fontSize: '0.85rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* KPI Cards Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
        <TelemetryCard
          label="Progress"
          value={`${status?.current_step || 0} / ${status?.total_timesteps || overrideTimesteps}`}
          subtext={`Algorithm: ${(status?.algorithm || 'PPO').toUpperCase()}`}
          color="#38BDF8"
          icon="⚡"
        />
        <TelemetryCard
          label="Episodes Completed"
          value={status?.episodes || 0}
          subtext={isTraining ? 'Training Active' : 'Idle'}
          color="#A78BFA"
          icon="🎮"
        />
        <TelemetryCard
          label="Mean Reward"
          value={(status?.mean_reward || 0).toFixed(2)}
          subtext={`Instant: ${(latest?.reward || 0).toFixed(2)}`}
          color="#34D399"
          icon="🎯"
        />
        <TelemetryCard
          label="Win Rate Est."
          value={`${((latest?.win_rate || 0) * 100).toFixed(1)}%`}
          subtext="vs Random Baseline"
          color="#FBBF24"
          icon="🏆"
        />
        <TelemetryCard
          label="Training Speed"
          value={`${latest?.fps || 0} fps`}
          subtext="Environment transitions/s"
          color="#60A5FA"
          icon="🚀"
        />
      </div>

      {/* 2x2 Telemetry Charts Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
        <LineChartSVG
          title="📈 Learning Curve: Mean & Instant Rewards"
          series={rewardSeries}
          yLabel="Reward"
        />

        <LineChartSVG
          title="📉 Loss Metrics (Policy & Value Loss)"
          series={lossSeries}
          yLabel="Loss"
        />

        <LineChartSVG
          title="🎲 Policy Exploration (Entropy)"
          series={entropySeries}
          yLabel="Entropy"
        />

        <LineChartSVG
          title="🏅 Estimated Win Rate (%)"
          series={winRateSeries}
          yLabel="Win Rate %"
        />
      </div>
    </div>
  );
};
