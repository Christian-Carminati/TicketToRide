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

  // Dynamically load all real logged experiment benchmarks
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

  // Reload experiments when a training run completes
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

  // Chart data series dynamically computed from telemetry history
  const rewardSeries = useMemo(() => [
    {
      id: 'mean_reward',
      name: 'Mean Reward (Rolling)',
      color: '#10B981',
      data: telemetryHistory.map((t) => ({ x: t.step, y: t.mean_reward })),
    },
    {
      id: 'step_reward',
      name: 'Step Reward',
      color: '#38BDF8',
      data: telemetryHistory.map((t) => ({ x: t.step, y: t.reward })),
    },
  ], [telemetryHistory]);

  const lossSeries = useMemo(() => [
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
  ], [telemetryHistory]);

  const entropySeries = useMemo(() => [
    {
      id: 'entropy',
      name: 'Policy Entropy',
      color: '#8B5CF6',
      data: telemetryHistory
        .filter((t) => t.entropy !== null && t.entropy !== undefined)
        .map((t) => ({ x: t.step, y: t.entropy! })),
    },
  ], [telemetryHistory]);

  const winRateSeries = useMemo(() => [
    {
      id: 'win_rate',
      name: 'Estimated Win Rate %',
      color: '#06B6D4',
      data: telemetryHistory
        .filter((t) => t.win_rate !== null && t.win_rate !== undefined)
        .map((t) => ({ x: t.step, y: t.win_rate! * 100 })),
    },
  ], [telemetryHistory]);

  return (
    <div className="training-view" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Dynamic Experiments History & Benchmark Card */}
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%)',
          border: '1px solid rgba(56, 189, 248, 0.25)',
          borderRadius: '12px',
          padding: '1.25rem',
          boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 800, color: '#F8FAFC' }}>
              📊 Storico Benchmark & Modelli Addestrati (Calcolato Dinamicamente)
            </h3>
            <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.8rem', color: '#94A3B8' }}>
              Metrice effettive caricate in tempo reale dal registro esperimenti
            </p>
          </div>

          <button
            onClick={loadExperiments}
            style={{
              background: 'rgba(56, 189, 248, 0.1)',
              border: '1px solid rgba(56, 189, 248, 0.3)',
              borderRadius: '6px',
              padding: '0.3rem 0.75rem',
              color: '#38BDF8',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            🔄 Aggiorna Registro
          </button>
        </div>

        {experiments.length === 0 ? (
          <div style={{ padding: '1rem', textAlign: 'center', color: '#64748B', fontSize: '0.85rem' }}>
            {isLoadingExperiments ? 'Caricamento esperimenti...' : 'Nessun esperimento registrato. Avvia un training qui sotto per generare le prime metriche!'}
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.75rem' }}>
            {experiments.slice(-4).reverse().map((exp) => (
              <div
                key={exp.experiment_id}
                style={{
                  background: 'rgba(15, 23, 42, 0.7)',
                  padding: '0.75rem 1rem',
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,0.06)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.3rem',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#38BDF8' }}>
                    {exp.name}
                  </span>
                  <span style={{ background: 'rgba(59, 130, 246, 0.2)', padding: '0.1rem 0.4rem', borderRadius: '4px', fontSize: '0.7rem', color: '#93C5FD', fontWeight: 700 }}>
                    {exp.algorithm.toUpperCase()}
                  </span>
                </div>

                <div style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
                  Seed: {exp.seed} | ID: {exp.experiment_id}
                </div>

                <div style={{ marginTop: '0.2rem', fontSize: '0.75rem', color: '#CBD5E1', lineHeight: '1.4' }}>
                  {Object.entries(exp.metrics).map(([key, val]) => (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#94A3B8' }}>{key.replace(/_/g, ' ')}:</span>
                      <strong style={{ color: typeof val === 'number' && val > 0.5 ? '#10B981' : '#F1F5F9' }}>
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

      {/* Top Configuration & Control Bar with Opponent Selection */}
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
          {/* Algorithm Config */}
          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Algoritmo:</label>
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
              fontWeight: 600,
            }}
          >
            <option value="ppo_usa.yaml">⭐ Masked PPO (Consigliato - Policy Gradient)</option>
            <option value="dqn_usa.yaml">Double-DQN (Deep Q-Network)</option>
          </select>

          {/* Opponent Selector */}
          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Avversario di Training:</label>
          <select
            value={opponentType}
            disabled={isTraining}
            onChange={(e) => setOpponentType(e.target.value as 'random' | 'greedy' | 'strategic')}
            style={{
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.6rem',
              fontSize: '0.85rem',
              fontWeight: 600,
            }}
          >
            <option value="greedy">⭐ ⚡ GreedyBot (Consigliato - Pressione Competitiva Tratte)</option>
            <option value="strategic">🧠 StrategicBot (Avanzato - Ottimizzazione Biglietti)</option>
            <option value="random">🎲 RandomBot (Base - Esplorazione Iniziale)</option>
          </select>

          {/* Timesteps */}
          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Timesteps:</label>
          <input
            type="number"
            disabled={isTraining}
            value={overrideTimesteps}
            onChange={(e) => setOverrideTimesteps(Number(e.target.value))}
            style={{
              width: '85px',
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.5rem',
              fontSize: '0.85rem',
            }}
          />

          {/* Seed */}
          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Seed:</label>
          <input
            type="number"
            disabled={isTraining}
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            style={{
              width: '65px',
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.5rem',
              fontSize: '0.85rem',
            }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {/* WebSocket Status Indicator */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem' }}>
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: isConnected ? '#10B981' : '#EF4444',
                boxShadow: isConnected ? '0 0 6px #10B981' : 'none',
              }}
            />
            <span style={{ color: '#94A3B8' }}>{isConnected ? 'WebSocket Online' : 'Offline'}</span>
          </div>

          {isTraining ? (
            <button
              onClick={stopTraining}
              style={{
                backgroundColor: '#EF4444',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '6px',
                padding: '0.45rem 1rem',
                fontWeight: 700,
                fontSize: '0.85rem',
                cursor: 'pointer',
              }}
            >
              ⏹️ Ferma Training
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={!isConnected}
              style={{
                backgroundColor: '#10B981',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '6px',
                padding: '0.45rem 1.1rem',
                fontWeight: 700,
                fontSize: '0.85rem',
                cursor: isConnected ? 'pointer' : 'not-allowed',
                opacity: isConnected ? 1 : 0.5,
                boxShadow: '0 2px 10px rgba(16, 185, 129, 0.3)',
              }}
            >
              ▶️ Avvia Training Live
            </button>
          )}
        </div>
      </div>

      {/* Live Training Status Bar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: isTraining ? 'rgba(16, 185, 129, 0.1)' : 'rgba(30, 41, 59, 0.5)',
          border: `1px solid ${isTraining ? '#10B981' : 'rgba(255, 255, 255, 0.08)'}`,
          borderRadius: '8px',
          padding: '0.6rem 1rem',
          fontSize: '0.8rem',
        }}
      >
        <span style={{ color: '#F1F5F9', fontWeight: 600 }}>
          {isTraining
            ? `⚡ Training in corso: Step ${status?.current_step} / ${status?.total_timesteps} (${status?.algorithm?.toUpperCase()}) contro ${opponentType.toUpperCase()}`
            : telemetryHistory.length > 0
            ? `✅ Telemetria acquisita: ${telemetryHistory.length} step campionati in questa sessione`
            : '💡 Seleziona i parametri e clicca "Avvia Training Live" per generare curve e metriche in tempo reale.'}
        </span>
        <span style={{ color: '#94A3B8' }}>
          {telemetryHistory.length} campioni live
        </span>
      </div>

      {error && (
        <div style={{ padding: '0.75rem', backgroundColor: 'rgba(239, 68, 68, 0.2)', border: '1px solid #EF4444', borderRadius: '8px', color: '#FCA5A5', fontSize: '0.85rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* KPI Cards Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
        <TelemetryCard
          label="Current Step"
          value={latest ? `${latest.step.toLocaleString()} / ${(status?.total_timesteps || overrideTimesteps).toLocaleString()}` : '--'}
          subtext={latest ? `Episodi: ${latest.episode}` : 'In attesa di training'}
          icon="⏱️"
        />
        <TelemetryCard
          label="Mean Reward (20-ep)"
          value={latest ? `${latest.mean_reward.toFixed(2)}` : '--'}
          subtext={latest ? `Istantaneo: ${latest.reward.toFixed(2)}` : undefined}
          color="#10B981"
          icon="🎯"
        />
        <TelemetryCard
          label="Policy / Value Loss"
          value={latest && latest.policy_loss !== null ? `${latest.policy_loss?.toFixed(3)} / ${latest.value_loss?.toFixed(3)}` : '--'}
          subtext="Gradiente di Ottimizzazione"
          color="#F59E0B"
          icon="📉"
        />
        <TelemetryCard
          label="Throughput (FPS)"
          value={latest && latest.fps ? `${latest.fps.toFixed(0)} FPS` : '--'}
          subtext="Velocità Campionamento"
          color="#8B5CF6"
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
