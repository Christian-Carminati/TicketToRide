import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { ExperimentRecordDTO } from '../api/types';

export const ExperimentView: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentRecordDTO[]>([]);
  const [selectedExp, setSelectedExp] = useState<ExperimentRecordDTO | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    setIsLoading(true);
    api.listExperiments()
      .then((list) => {
        setExperiments(list);
        if (list.length > 0) setSelectedExp(list[0]);
      })
      .catch((err) => console.error(err))
      .finally(() => setIsLoading(false));
  }, []);

  const filtered = experiments.filter(
    (e) =>
      e.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      e.algorithm.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="experiment-view" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Header Bar */}
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
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#F1F5F9' }}>
            🧪 Experiment Registry & Benchmarks ({experiments.length})
          </h4>
          <input
            type="text"
            placeholder="Search experiments..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.6rem',
              fontSize: '0.85rem',
              minWidth: '200px',
            }}
          />
        </div>

        <button
          onClick={() => {
            setIsLoading(true);
            api.listExperiments()
              .then(setExperiments)
              .finally(() => setIsLoading(false));
          }}
          disabled={isLoading}
          style={{
            backgroundColor: '#3B82F6',
            color: '#FFFFFF',
            border: 'none',
            borderRadius: '6px',
            padding: '0.4rem 0.8rem',
            fontWeight: 600,
            fontSize: '0.85rem',
            cursor: 'pointer',
          }}
        >
          🔄 Refresh
        </button>
      </div>

      {/* Grid: List on Left, Detail & JSON on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.2fr) minmax(0, 0.8fr)', gap: '1.25rem' }}>
        {/* Experiment Table */}
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.9)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '12px',
            padding: '1.25rem',
          }}
        >
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: '#94A3B8' }}>
                  <th style={{ padding: '0.6rem' }}>Name</th>
                  <th style={{ padding: '0.6rem' }}>Algo</th>
                  <th style={{ padding: '0.6rem' }}>Seed</th>
                  <th style={{ padding: '0.6rem' }}>Win Rate</th>
                  <th style={{ padding: '0.6rem' }}>Date</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((exp) => {
                  const isSelected = selectedExp?.experiment_id === exp.experiment_id;
                  const winRate = exp.metrics?.win_rate !== undefined ? `${(exp.metrics.win_rate * 100).toFixed(1)}%` : 'N/A';

                  return (
                    <tr
                      key={exp.experiment_id}
                      onClick={() => setSelectedExp(exp)}
                      style={{
                        borderBottom: '1px solid rgba(255,255,255,0.05)',
                        backgroundColor: isSelected ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                        cursor: 'pointer',
                      }}
                    >
                      <td style={{ padding: '0.6rem', fontWeight: 600, color: isSelected ? '#38BDF8' : '#F1F5F9' }}>
                        {exp.name}
                      </td>
                      <td style={{ padding: '0.6rem' }}>
                        <span
                          style={{
                            padding: '0.2rem 0.4rem',
                            borderRadius: '4px',
                            backgroundColor: exp.algorithm === 'ppo' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                            color: exp.algorithm === 'ppo' ? '#34D399' : '#FBBF24',
                            fontSize: '0.75rem',
                            fontWeight: 700,
                          }}
                        >
                          {exp.algorithm.toUpperCase()}
                        </span>
                      </td>
                      <td style={{ padding: '0.6rem', color: '#94A3B8' }}>{exp.seed}</td>
                      <td style={{ padding: '0.6rem', fontWeight: 700, color: '#10B981' }}>{winRate}</td>
                      <td style={{ padding: '0.6rem', color: '#64748B', fontSize: '0.75rem' }}>
                        {exp.timestamp ? new Date(exp.timestamp).toLocaleDateString() : 'N/A'}
                      </td>
                    </tr>
                  );
                })}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={5} style={{ padding: '1.5rem', textAlign: 'center', color: '#64748B' }}>
                      No experiment records found in experiments/registry.json. Run a training session to create benchmark records.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Selected Experiment Detail Panel */}
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.9)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '12px',
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem',
          }}
        >
          <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#F1F5F9' }}>
            📝 Experiment Details & Hyperparameters
          </h4>

          {selectedExp ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Experiment ID:</span>
                <div style={{ fontSize: '0.85rem', color: '#38BDF8', fontWeight: 600 }}>{selectedExp.experiment_id}</div>
              </div>

              {/* Metrics Grid */}
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Evaluated Metrics:</span>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.25rem' }}>
                  {Object.entries(selectedExp.metrics || {}).map(([k, v]) => (
                    <div key={k} style={{ background: 'rgba(30, 41, 59, 0.6)', padding: '0.4rem 0.6rem', borderRadius: '6px' }}>
                      <div style={{ fontSize: '0.7rem', color: '#94A3B8' }}>{k}</div>
                      <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#F1F5F9' }}>
                        {typeof v === 'number' ? v.toFixed(3) : String(v)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Config JSON viewer */}
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Configuration:</span>
                <pre
                  style={{
                    background: '#0B1120',
                    padding: '0.75rem',
                    borderRadius: '6px',
                    fontSize: '0.75rem',
                    color: '#CBD5E1',
                    maxHeight: '220px',
                    overflowY: 'auto',
                    marginTop: '0.25rem',
                  }}
                >
                  {JSON.stringify(selectedExp.config || {}, null, 2)}
                </pre>
              </div>
            </div>
          ) : (
            <div style={{ fontSize: '0.85rem', color: '#64748B', fontStyle: 'italic', padding: '1rem 0' }}>
              Select an experiment from the left table to inspect details.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
