import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { ExperimentRecordDTO } from '../api/types';
import { Trash2, RefreshCw, Search, CheckCircle, AlertTriangle } from 'lucide-react';

export const ExperimentView: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentRecordDTO[]>([]);
  const [selectedExp, setSelectedExp] = useState<ExperimentRecordDTO | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [actionMessage, setActionMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const fetchExperiments = () => {
    setIsLoading(true);
    api.listExperiments()
      .then((list) => {
        setExperiments(list);
        if (list.length > 0) {
          setSelectedExp((prev: ExperimentRecordDTO | null) => (prev ? list.find((e) => e.experiment_id === prev.experiment_id) || list[0] : list[0]));
        } else {
          setSelectedExp(null);
        }
      })
      .catch((err) => {
        console.error(err);
        setActionMessage({ type: 'error', text: 'Unable to load experiments catalog.' });
      })
      .finally(() => setIsLoading(false));
  };

  useEffect(() => {
    fetchExperiments();
  }, []);

  const handleDeleteSingle = async (expId: string, name: string) => {
    try {
      setIsLoading(true);
      await api.deleteExperiment(expId);
      setActionMessage({ type: 'success', text: `Experiment "${name}" successfully deleted.` });
      fetchExperiments();
    } catch (err) {
      console.error(err);
      setActionMessage({ type: 'error', text: 'Error deleting experiment record.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAll = async () => {
    if (experiments.length === 0) return;
    try {
      setIsLoading(true);
      const res = await api.deleteAllExperiments();
      setActionMessage({ type: 'success', text: `All experiments (${res.deleted_count}) have been purged.` });
      setExperiments([]);
      setSelectedExp(null);
    } catch (err) {
      console.error(err);
      setActionMessage({ type: 'error', text: 'Error purging all experiment records.' });
    } finally {
      setIsLoading(false);
    }
  };

  const filtered = experiments.filter(
    (e) =>
      e.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      e.algorithm.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="experiment-view" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Action Notification Banner */}
      {actionMessage && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.6rem 1rem',
            borderRadius: '8px',
            background: actionMessage.type === 'success' ? '#FAF3E6' : '#FEE2E2',
            border: `1.5px solid ${actionMessage.type === 'success' ? '#15803D' : '#B91C1C'}`,
            color: actionMessage.type === 'success' ? '#15803D' : '#B91C1C',
            fontSize: '0.85rem',
            fontFamily: "'Playfair Display', Georgia, serif",
            fontWeight: 700,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {actionMessage.type === 'success' ? <CheckCircle size={16} /> : <AlertTriangle size={16} />}
            <span>{actionMessage.text}</span>
          </div>
          <button
            onClick={() => setActionMessage(null)}
            style={{ background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer', fontWeight: 800 }}
          >
            ✕
          </button>
        </div>
      )}

      {/* Header Bar */}
      <div
        className="steampunk-panel"
        style={{
          padding: '1rem 1.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <h4 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 800, color: '#23140C', fontFamily: "'Cinzel Decorative', Georgia, serif" }}>
            🧪 Analytical Experiment Ledger ({experiments.length})
          </h4>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Search size={14} color="#785A42" style={{ position: 'absolute', left: '0.6rem' }} />
            <input
              type="text"
              placeholder="Search experiments..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                backgroundColor: '#FAF5EB',
                color: '#23140C',
                border: '1.5px solid #8C6305',
                borderRadius: '6px',
                padding: '0.35rem 0.6rem 0.35rem 2rem',
                fontSize: '0.82rem',
                minWidth: '200px',
                fontFamily: "'Courier Prime', monospace",
              }}
            />
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {/* Delete All Button */}
          <button
            onClick={handleDeleteAll}
            disabled={isLoading || experiments.length === 0}
            className="steampunk-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: 'linear-gradient(180deg, #F87171 0%, #DC2626 50%, #991B1B 100%)',
              color: '#FFFFFF',
              border: '1px solid #7F1D1D',
              padding: '0.4rem 0.85rem',
              fontSize: '0.8rem',
            }}
            title="Elimina tutti gli esperimenti dal registro"
          >
            <Trash2 size={14} />
            <span>Purge All ({experiments.length})</span>
          </button>

          {/* Refresh Button */}
          <button
            onClick={fetchExperiments}
            disabled={isLoading}
            className="steampunk-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              padding: '0.4rem 0.85rem',
              fontSize: '0.8rem',
            }}
          >
            <RefreshCw size={14} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Grid: List on Left, Detail & JSON on Right */}
      <div className="experiment-main-split">
        {/* Experiment Table */}
        <div
          className="steampunk-panel"
          style={{
            padding: '1.25rem',
          }}
        >
          <div style={{ overflowX: 'auto', background: '#FAF5EB', borderRadius: '8px', border: '1.5px solid #C59B27' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ background: 'linear-gradient(180deg, #EFE1C7 0%, #E2CFAC 100%)', color: '#23140C', borderBottom: '2px solid #C59B27' }}>
                  <th style={{ padding: '0.6rem 0.8rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Experiment Identifier</th>
                  <th style={{ padding: '0.6rem 0.8rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Algorithm</th>
                  <th style={{ padding: '0.6rem 0.8rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Seed</th>
                  <th style={{ padding: '0.6rem 0.8rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Win Rate</th>
                  <th style={{ padding: '0.6rem 0.8rem', textAlign: 'center', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((exp) => {
                  const isSelected = selectedExp?.experiment_id === exp.experiment_id;
                  const rawWr = exp.metrics?.win_rate;
                  const winRate = rawWr !== undefined ? `${(Number(rawWr) * 100).toFixed(1)}%` : 'N/A';

                  return (
                    <tr
                      key={exp.experiment_id}
                      onClick={() => setSelectedExp(exp)}
                      style={{
                        borderBottom: '1px solid rgba(184, 134, 11, 0.2)',
                        backgroundColor: isSelected ? '#FAF0DA' : 'transparent',
                        cursor: 'pointer',
                      }}
                    >
                      <td style={{ padding: '0.6rem 0.8rem', fontWeight: 700, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                        {exp.name}
                      </td>
                      <td style={{ padding: '0.6rem 0.8rem' }}>
                        <span
                          style={{
                            padding: '0.15rem 0.45rem',
                            borderRadius: '4px',
                            backgroundColor: '#EADBBE',
                            color: '#23140C',
                            border: '1px solid #C59B27',
                            fontSize: '0.72rem',
                            fontWeight: 800,
                            fontFamily: "'Courier Prime', monospace",
                          }}
                        >
                          {exp.algorithm.toUpperCase()}
                        </span>
                      </td>
                      <td style={{ padding: '0.6rem 0.8rem', color: '#5A3822', fontFamily: "'Courier Prime', monospace" }}>{exp.seed}</td>
                      <td style={{ padding: '0.6rem 0.8rem', fontWeight: 800, color: '#15803D', fontFamily: "'Courier Prime', monospace" }}>{winRate}</td>
                      <td style={{ padding: '0.6rem 0.8rem', textAlign: 'center' }}>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteSingle(exp.experiment_id, exp.name);
                          }}
                          className="steampunk-btn"
                          style={{
                            padding: '0.2rem 0.45rem',
                            fontSize: '0.72rem',
                            background: 'linear-gradient(180deg, #F87171 0%, #DC2626 50%, #991B1B 100%)',
                            color: '#FFFFFF',
                            border: '1px solid #7F1D1D',
                          }}
                          title="Elimina questo esperimento"
                        >
                          <Trash2 size={12} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={5} style={{ padding: '2rem 1rem', textAlign: 'center', color: '#785A42', fontStyle: 'italic', fontFamily: "'Crimson Pro', serif" }}>
                      No experiment records found in registry.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Selected Experiment Detail Panel */}
        <div
          className="steampunk-panel"
          style={{
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 800, color: '#23140C', fontFamily: "'Cinzel Decorative', Georgia, serif" }}>
              📝 Experiment Metrics & Parameters
            </h4>
            {selectedExp && (
              <button
                onClick={() => handleDeleteSingle(selectedExp.experiment_id, selectedExp.name)}
                className="steampunk-btn"
                style={{
                  background: 'linear-gradient(180deg, #F87171 0%, #DC2626 50%, #991B1B 100%)',
                  color: '#FFFFFF',
                  border: '1px solid #7F1D1D',
                  padding: '0.25rem 0.6rem',
                  fontSize: '0.75rem',
                }}
              >
                <Trash2 size={13} /> Delete
              </button>
            )}
          </div>

          {selectedExp ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#785A42', fontFamily: "'Crimson Pro', serif" }}>Experiment ID:</span>
                <div style={{ fontSize: '0.85rem', color: '#9E6B00', fontWeight: 800, fontFamily: "'Courier Prime', monospace" }}>
                  {selectedExp.experiment_id}
                </div>
              </div>

              {/* Metrics Grid */}
              <div>
                <span style={{ fontSize: '0.75rem', color: '#785A42', fontFamily: "'Crimson Pro', serif" }}>Evaluated Metrics:</span>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.25rem' }}>
                  {Object.entries(selectedExp.metrics || {}).map(([k, v]) => (
                    <div key={k} style={{ background: '#FAF5EB', border: '1px solid rgba(184, 134, 11, 0.3)', padding: '0.4rem 0.6rem', borderRadius: '6px' }}>
                      <div style={{ fontSize: '0.7rem', color: '#5A3822', fontFamily: "'Crimson Pro', serif" }}>{k}</div>
                      <div style={{ fontSize: '0.85rem', fontWeight: 800, color: '#23140C', fontFamily: "'Courier Prime', monospace" }}>
                        {typeof v === 'number' ? v.toFixed(3) : String(v)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Config JSON viewer */}
              <div>
                <span style={{ fontSize: '0.75rem', color: '#785A42', fontFamily: "'Crimson Pro', serif" }}>Configuration:</span>
                <pre
                  style={{
                    background: '#2B1D14',
                    border: '1.5px solid #8C6305',
                    padding: '0.75rem',
                    borderRadius: '6px',
                    fontSize: '0.75rem',
                    color: '#FAF5EB',
                    maxHeight: '220px',
                    overflowY: 'auto',
                    marginTop: '0.25rem',
                    fontFamily: "'Courier Prime', monospace",
                  }}
                >
                  {JSON.stringify(selectedExp.metrics || {}, null, 2)}
                </pre>
              </div>
            </div>
          ) : (
            <div style={{ fontSize: '0.85rem', color: '#785A42', fontStyle: 'italic', padding: '1rem 0', fontFamily: "'Crimson Pro', serif" }}>
              Select an experiment from the left table to inspect hyperparameters and recorded telemetry.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
