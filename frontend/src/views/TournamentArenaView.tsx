import React from 'react';
import { EloMatrixHeatmap } from '../components/tournament/EloMatrixHeatmap';
import { ExperimentView } from './ExperimentView';

export const TournamentArenaView: React.FC = () => {
  return (
    <div
      className="tournament-arena-view"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem',
      }}
    >
      {/* 1. Primary Tournament Arena: Elo Matrix & Custom Participant Selector */}
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.9)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '12px',
          padding: '0.75rem',
        }}
      >
        <EloMatrixHeatmap />
      </div>

      {/* 2. Experiment & Benchmark Run Registry */}
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.7)',
          border: '1px solid rgba(255, 255, 255, 0.06)',
          borderRadius: '12px',
          padding: '1rem',
        }}
      >
        <ExperimentView />
      </div>
    </div>
  );
};
