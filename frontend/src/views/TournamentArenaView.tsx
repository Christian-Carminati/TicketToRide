import React, { useState } from 'react';
import { EloMatrixHeatmap } from '../components/tournament/EloMatrixHeatmap';
import { CheckpointManagerPane } from '../components/checkpoints/CheckpointManagerPane';
import { ExperimentView } from './ExperimentView';
import { Trophy, HardDrive, FlaskConical } from 'lucide-react';

export const TournamentArenaView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'tournament' | 'checkpoints' | 'experiments'>('tournament');

  return (
    <div
      className="tournament-arena-view"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
      }}
    >
      {/* Tab Navigation Strip */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          background: 'rgba(30, 41, 59, 0.7)',
          padding: '0.35rem 0.5rem',
          borderRadius: '10px',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          width: 'fit-content',
        }}
      >
        <button
          onClick={() => setActiveTab('tournament')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: activeTab === 'tournament' ? '#3B82F6' : 'transparent',
            color: activeTab === 'tournament' ? '#FFFFFF' : '#94A3B8',
            border: 'none',
            borderRadius: '6px',
            padding: '0.4rem 0.9rem',
            fontSize: '0.82rem',
            fontWeight: 700,
            cursor: 'pointer',
            transition: 'all 0.15s ease',
            boxShadow: activeTab === 'tournament' ? '0 2px 8px rgba(59, 130, 246, 0.4)' : 'none',
          }}
        >
          <Trophy size={14} />
          <span>Matrice Torneo & Elo</span>
        </button>

        <button
          onClick={() => setActiveTab('checkpoints')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: activeTab === 'checkpoints' ? '#3B82F6' : 'transparent',
            color: activeTab === 'checkpoints' ? '#FFFFFF' : '#94A3B8',
            border: 'none',
            borderRadius: '6px',
            padding: '0.4rem 0.9rem',
            fontSize: '0.82rem',
            fontWeight: 700,
            cursor: 'pointer',
            transition: 'all 0.15s ease',
            boxShadow: activeTab === 'checkpoints' ? '0 2px 8px rgba(59, 130, 246, 0.4)' : 'none',
          }}
        >
          <HardDrive size={14} />
          <span>Gestione Checkpoint (.pt)</span>
        </button>

        <button
          onClick={() => setActiveTab('experiments')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: activeTab === 'experiments' ? '#3B82F6' : 'transparent',
            color: activeTab === 'experiments' ? '#FFFFFF' : '#94A3B8',
            border: 'none',
            borderRadius: '6px',
            padding: '0.4rem 0.9rem',
            fontSize: '0.82rem',
            fontWeight: 700,
            cursor: 'pointer',
            transition: 'all 0.15s ease',
            boxShadow: activeTab === 'experiments' ? '0 2px 8px rgba(59, 130, 246, 0.4)' : 'none',
          }}
        >
          <FlaskConical size={14} />
          <span>Registro Esperimenti</span>
        </button>
      </div>

      {/* Main Tab Views */}
      {activeTab === 'tournament' && (
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
      )}

      {activeTab === 'checkpoints' && (
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.9)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '12px',
            padding: '1rem',
          }}
        >
          <CheckpointManagerPane />
        </div>
      )}

      {activeTab === 'experiments' && (
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
      )}
    </div>
  );
};
