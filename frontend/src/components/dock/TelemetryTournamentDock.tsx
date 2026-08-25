import React, { useMemo } from 'react';
import { useWorkbench } from '../../context';
import { LineChartSVG, ChartSeries } from '../charts/LineChartSVG';
import { ActionLogStream } from './ActionLogStream';
import { Activity, ListFilter, ChevronDown, ChevronUp } from 'lucide-react';

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
      { id: 'mean_reward', name: 'Mean Reward', color: '#15803D', data: rewards },
    ];

    const lossChartSeries: ChartSeries[] = [
      { id: 'policy_loss', name: 'Policy Loss', color: '#1D4ED8', data: policyLoss },
      { id: 'value_loss', name: 'Value Loss', color: '#B91C1C', data: valueLoss },
    ];

    return { rewardSeries: rewardChartSeries, lossSeries: lossChartSeries };
  }, [telemetryHistory]);

  const latestEvent = telemetryHistory[telemetryHistory.length - 1];

  return (
    <div
      className="telemetry-tournament-dock steampunk-panel"
      style={{
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: '2px solid #C59B27',
        borderRadius: '12px',
        overflow: 'hidden',
        boxShadow: '0 -4px 24px rgba(0, 0, 0, 0.35)',
        transition: 'all 0.3s ease',
      }}
    >
      {/* Dock Bar / Tab Strip Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0.45rem 1.25rem',
          background: 'linear-gradient(180deg, #3A261A 0%, #26180F 100%)',
          borderBottom: isBottomDockOpen ? '2px solid #C59B27' : 'none',
          cursor: 'pointer',
        }}
      >
        {/* Tab Switchers */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setBottomDockTab('telemetry');
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: bottomDockTab === 'telemetry' && isBottomDockOpen
                ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                : 'transparent',
              color: bottomDockTab === 'telemetry' && isBottomDockOpen ? '#23140C' : '#D4C09D',
              border: bottomDockTab === 'telemetry' && isBottomDockOpen ? '1px solid #6E4E04' : '1px solid transparent',
              borderRadius: '6px',
              padding: '0.3rem 0.75rem',
              fontSize: '0.8rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: bottomDockTab === 'telemetry' && isBottomDockOpen ? 800 : 600,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
              boxShadow: bottomDockTab === 'telemetry' && isBottomDockOpen ? '0 1px 4px rgba(0,0,0,0.25)' : 'none',
            }}
          >
            <Activity size={13} /> Live Telemetry
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
              background: bottomDockTab === 'logs' && isBottomDockOpen
                ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                : 'transparent',
              color: bottomDockTab === 'logs' && isBottomDockOpen ? '#23140C' : '#D4C09D',
              border: bottomDockTab === 'logs' && isBottomDockOpen ? '1px solid #6E4E04' : '1px solid transparent',
              borderRadius: '6px',
              padding: '0.3rem 0.75rem',
              fontSize: '0.8rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: bottomDockTab === 'logs' && isBottomDockOpen ? 800 : 600,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
              boxShadow: bottomDockTab === 'logs' && isBottomDockOpen ? '0 1px 4px rgba(0,0,0,0.25)' : 'none',
            }}
          >
            <ListFilter size={13} /> Telegraph Log
          </button>
        </div>

        {/* Right Status Summary & Toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {latestEvent && (
            <span style={{ fontSize: '0.72rem', color: '#D4C09D', fontFamily: "'Courier Prime', monospace" }}>
              Step: {latestEvent.step} | Rate: {latestEvent.fps?.toFixed(0) || '60'} Hz | Episode: {latestEvent.episode || 1}
            </span>
          )}

          <button
            onClick={toggleBottomDock}
            className="steampunk-btn"
            style={{
              padding: '0.2rem 0.45rem',
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

          {bottomDockTab === 'logs' && (
            <ActionLogStream />
          )}
        </div>
      )}
    </div>
  );
};
