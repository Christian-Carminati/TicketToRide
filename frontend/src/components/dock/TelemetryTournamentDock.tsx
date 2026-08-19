import React, { useMemo } from 'react';
import { useWorkbench } from '../../context';
import { LineChartSVG, ChartSeries } from '../charts/LineChartSVG';
import { EloMatrixHeatmap } from '../tournament/EloMatrixHeatmap';
import { ActionLogStream } from './ActionLogStream';
import { Activity, Trophy, ListFilter, ChevronDown, ChevronUp } from 'lucide-react';

export const TelemetryTournamentDock: React.FC = () => {
  const { state, setBottomDockTab, toggleBottomDock } = useWorkbench();
  const { bottomDockTab, isBottomDockOpen, telemetryHistory } = state;

  // Prepare chart series from telemetry history
  const { rewardSeries, lossSeries } = useMemo(() => {
    const rewards: Array<{ x: number; y: number }> = [];
    const policyLoss: Array<{ x: number; y: number }> = [];
    const valueLoss: Array<{ x: number; y: number }> = [];

    telemetryHistory.forEach((ev) => {
      const step = ev.step || 0;
      if (ev.mean_reward !== undefined && ev.mean_reward !== null) {
        rewards.push({ x: step, y: ev.mean_reward });
      }
      if (ev.policy_loss !== undefined && ev.policy_loss !== null) {
        policyLoss.push({ x: step, y: ev.policy_loss });
      }
      if (ev.value_loss !== undefined && ev.value_loss !== null) {
        valueLoss.push({ x: step, y: ev.value_loss });
      }
    });

    const rewardChartSeries: ChartSeries[] = [
      { id: 'mean_reward', name: 'Mean Reward', color: '#34D399', data: rewards },
    ];

    const lossChartSeries: ChartSeries[] = [
      { id: 'policy_loss', name: 'Policy Loss', color: '#38BDF8', data: policyLoss },
      { id: 'value_loss', name: 'Value Loss', color: '#F43F5E', data: valueLoss },
    ];

    return { rewardSeries: rewardChartSeries, lossSeries: lossChartSeries };
  }, [telemetryHistory]);

  const latestEvent = telemetryHistory[telemetryHistory.length - 1];

  return (
    <div
      className="telemetry-tournament-dock"
      style={{
        background: 'rgba(15, 23, 42, 0.95)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '12px',
        overflow: 'hidden',
        boxShadow: '0 -4px 24px rgba(0, 0, 0, 0.4)',
        transition: 'all 0.3s ease',
      }}
    >
      {/* Dock Bar / Tab Strip Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0.4rem 1rem',
          background: 'rgba(30, 41, 59, 0.7)',
          borderBottom: isBottomDockOpen ? '1px solid rgba(255, 255, 255, 0.08)' : 'none',
          cursor: 'pointer',
        }}
      >
        {/* Tab Switchers */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setBottomDockTab('telemetry');
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: bottomDockTab === 'telemetry' && isBottomDockOpen ? '#3B82F6' : 'transparent',
              color: bottomDockTab === 'telemetry' && isBottomDockOpen ? '#FFFFFF' : '#94A3B8',
              border: 'none',
              borderRadius: '6px',
              padding: '0.3rem 0.65rem',
              fontSize: '0.78rem',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <Activity size={13} /> Live Telemetry
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation();
              setBottomDockTab('tournament');
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: bottomDockTab === 'tournament' && isBottomDockOpen ? '#3B82F6' : 'transparent',
              color: bottomDockTab === 'tournament' && isBottomDockOpen ? '#FFFFFF' : '#94A3B8',
              border: 'none',
              borderRadius: '6px',
              padding: '0.3rem 0.65rem',
              fontSize: '0.78rem',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <Trophy size={13} /> Tournament Elo Matrix
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation();
              setBottomDockTab('logs');
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: bottomDockTab === 'logs' && isBottomDockOpen ? '#3B82F6' : 'transparent',
              color: bottomDockTab === 'logs' && isBottomDockOpen ? '#FFFFFF' : '#94A3B8',
              border: 'none',
              borderRadius: '6px',
              padding: '0.3rem 0.65rem',
              fontSize: '0.78rem',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <ListFilter size={13} /> Action Log
          </button>
        </div>

        {/* Right Status Summary & Toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {latestEvent && (
            <span style={{ fontSize: '0.72rem', color: '#64748B', fontFamily: 'monospace' }}>
              Step: {latestEvent.step} | FPS: {latestEvent.fps?.toFixed(0) || '60'} | Episode: {latestEvent.episode || 1}
            </span>
          )}

          <button
            onClick={toggleBottomDock}
            style={{
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '4px',
              color: '#94A3B8',
              padding: '0.2rem 0.4rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            title={isBottomDockOpen ? 'Collapse Dock' : 'Expand Dock'}
          >
            {isBottomDockOpen ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
          </button>
        </div>
      </div>

      {/* Dock Content Body */}
      {isBottomDockOpen && (
        <div style={{ padding: '0.75rem 1rem' }}>
          {bottomDockTab === 'telemetry' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <LineChartSVG
                title="Reward Convergence Trajectory"
                series={rewardSeries}
                yLabel="Mean Reward"
                xLabel="Training Steps"
                height={170}
              />
              <LineChartSVG
                title="PPO Policy & Value Loss"
                series={lossSeries}
                yLabel="Loss Magnitude"
                xLabel="Training Steps"
                height={170}
              />
            </div>
          )}

          {bottomDockTab === 'tournament' && (
            <EloMatrixHeatmap />
          )}

          {bottomDockTab === 'logs' && (
            <ActionLogStream />
          )}
        </div>
      )}
    </div>
  );
};
