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
          gap: '0.4rem',
          background: 'linear-gradient(180deg, #3A261A 0%, #26180F 100%)',
          padding: '0.35rem 0.6rem',
          borderRadius: '8px',
          border: '1.5px solid #C59B27',
          width: 'fit-content',
          boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
        }}
      >
        <button
          onClick={() => setActiveTab('tournament')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: activeTab === 'tournament'
              ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
              : 'transparent',
            color: activeTab === 'tournament' ? '#23140C' : '#D4C09D',
            border: activeTab === 'tournament' ? '1px solid #6E4E04' : '1px solid transparent',
            borderRadius: '6px',
            padding: '0.4rem 0.9rem',
            fontSize: '0.82rem',
            fontFamily: "'Playfair Display', Georgia, serif",
            fontWeight: activeTab === 'tournament' ? 800 : 600,
            cursor: 'pointer',
            transition: 'all 0.15s ease',
            boxShadow: activeTab === 'tournament' ? '0 2px 6px rgba(0,0,0,0.25)' : 'none',
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
            background: activeTab === 'checkpoints'
              ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
              : 'transparent',
            color: activeTab === 'checkpoints' ? '#23140C' : '#D4C09D',
            border: activeTab === 'checkpoints' ? '1px solid #6E4E04' : '1px solid transparent',
            borderRadius: '6px',
            padding: '0.4rem 0.9rem',
            fontSize: '0.82rem',
            fontFamily: "'Playfair Display', Georgia, serif",
            fontWeight: activeTab === 'checkpoints' ? 800 : 600,
            cursor: 'pointer',
            transition: 'all 0.15s ease',
            boxShadow: activeTab === 'checkpoints' ? '0 2px 6px rgba(0,0,0,0.25)' : 'none',
          }}
        >
          <HardDrive size={14} />
          <span>Archivio Checkpoint (.pt)</span>
        </button>

        <button
          onClick={() => setActiveTab('experiments')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: activeTab === 'experiments'
              ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
              : 'transparent',
            color: activeTab === 'experiments' ? '#23140C' : '#D4C09D',
            border: activeTab === 'experiments' ? '1px solid #6E4E04' : '1px solid transparent',
            borderRadius: '6px',
            padding: '0.4rem 0.9rem',
            fontSize: '0.82rem',
            fontFamily: "'Playfair Display', Georgia, serif",
            fontWeight: activeTab === 'experiments' ? 800 : 600,
            cursor: 'pointer',
            transition: 'all 0.15s ease',
            boxShadow: activeTab === 'experiments' ? '0 2px 6px rgba(0,0,0,0.25)' : 'none',
          }}
        >
          <FlaskConical size={14} />
          <span>Registro Esperimenti</span>
        </button>
      </div>

      {/* Main Tab Views */}
      {activeTab === 'tournament' && (
        <div className="steampunk-panel" style={{ padding: '0.85rem' }}>
          <EloMatrixHeatmap />
        </div>
      )}

      {activeTab === 'checkpoints' && (
        <div className="steampunk-panel" style={{ padding: '1rem' }}>
          <CheckpointManagerPane />
        </div>
      )}

      {activeTab === 'experiments' && (
        <div className="steampunk-panel" style={{ padding: '1rem' }}>
          <ExperimentView />
        </div>
      )}
    </div>
  );
};
