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
          setSelectedExp((prev) => (prev ? list.find((e) => e.experiment_id === prev.experiment_id) || list[0] : list[0]));
        } else {
          setSelectedExp(null);
        }
      })
      .catch((err) => {
        console.error(err);
        setActionMessage({ type: 'error', text: 'Impossibile caricare gli esperimenti.' });
      })
      .finally(() => setIsLoading(false));
  };

  useEffect(() => {
    fetchExperiments();
  }, []);

  const handleDeleteSingle = async (expId: string, name: string) => {
    if (!window.confirm(`Sei sicuro di voler eliminare l'esperimento "${name}" (${expId})?`)) {
      return;
    }
    try {
      setIsLoading(true);
      await api.deleteExperiment(expId);
      setActionMessage({ type: 'success', text: `Esperimento "${name}" eliminato con successo.` });
      fetchExperiments();
    } catch (err) {
      console.error(err);
      setActionMessage({ type: 'error', text: "Errore durante l'eliminazione dell'esperimento." });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAll = async () => {
    if (experiments.length === 0) return;
    if (!window.confirm(`Sei sicuro di voler cancellare TUTTI i ${experiments.length} esperimenti registrati? L'operazione è irreversibile.`)) {
      return;
    }
    try {
      setIsLoading(true);
      const res = await api.deleteAllExperiments();
      setActionMessage({ type: 'success', text: `Tutti gli esperimenti (${res.deleted_count}) sono stati cancellati.` });
      setExperiments([]);
      setSelectedExp(null);
    } catch (err) {
      console.error(err);
      setActionMessage({ type: 'error', text: "Errore durante l'eliminazione di tutti gli esperimenti." });
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
            background: actionMessage.type === 'success' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
            border: `1px solid ${actionMessage.type === 'success' ? '#10B981' : '#F43F5E'}`,
            color: actionMessage.type === 'success' ? '#34D399' : '#F43F5E',
            fontSize: '0.85rem',
            fontWeight: 600,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {actionMessage.type === 'success' ? <CheckCircle size={16} /> : <AlertTriangle size={16} />}
            <span>{actionMessage.text}</span>
          </div>
          <button
            onClick={() => setActionMessage(null)}
            style={{ background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer', fontWeight: 700 }}
          >
            ✕
          </button>
        </div>
      )}

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
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#F1F5F9' }}>
            🧪 Registro Esperimenti & Benchmark ({experiments.length})
          </h4>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Search size={14} color="#64748B" style={{ position: 'absolute', left: '0.6rem' }} />
            <input
              type="text"
              placeholder="Filtra esperimenti..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                backgroundColor: '#1E293B',
                color: '#F1F5F9',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '6px',
                padding: '0.35rem 0.6rem 0.35rem 2rem',
                fontSize: '0.85rem',
                minWidth: '200px',
              }}
            />
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {/* Delete All Button */}
          <button
            onClick={handleDeleteAll}
            disabled={isLoading || experiments.length === 0}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              backgroundColor: experiments.length === 0 ? 'rgba(239, 68, 68, 0.2)' : '#EF4444',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.4rem 0.85rem',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: experiments.length === 0 ? 'not-allowed' : 'pointer',
              opacity: experiments.length === 0 ? 0.5 : 1,
              transition: 'all 0.15s ease',
            }}
            title="Elimina tutti gli esperimenti dal registro"
          >
            <Trash2 size={14} />
            <span>Elimina Tutti ({experiments.length})</span>
          </button>

          {/* Refresh Button */}
          <button
            onClick={fetchExperiments}
            disabled={isLoading}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              backgroundColor: '#3B82F6',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.4rem 0.85rem',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
            }}
          >
            <RefreshCw size={14} />
            <span>Aggiorna</span>
          </button>
        </div>
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
                  <th style={{ padding: '0.6rem' }}>Nome Esperimento</th>
                  <th style={{ padding: '0.6rem' }}>Algoritmo</th>
                  <th style={{ padding: '0.6rem' }}>Seed</th>
                  <th style={{ padding: '0.6rem' }}>Win Rate</th>
                  <th style={{ padding: '0.6rem', textAlign: 'center' }}>Azioni</th>
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
                      <td style={{ padding: '0.6rem', color: '#94A3B8', fontFamily: 'monospace' }}>{exp.seed}</td>
                      <td style={{ padding: '0.6rem', fontWeight: 700, color: '#10B981', fontFamily: 'monospace' }}>{winRate}</td>
                      <td style={{ padding: '0.6rem', textAlign: 'center' }}>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteSingle(exp.experiment_id, exp.name);
                          }}
                          style={{
                            background: 'rgba(239, 68, 68, 0.15)',
                            border: '1px solid rgba(239, 68, 68, 0.3)',
                            borderRadius: '4px',
                            color: '#F87171',
                            padding: '0.2rem 0.45rem',
                            cursor: 'pointer',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.2rem',
                            fontSize: '0.72rem',
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
                    <td colSpan={5} style={{ padding: '2rem 1rem', textAlign: 'center', color: '#64748B' }}>
                      Nessun record di esperimento presente. Avvia una sessione di training per registrare nuovi benchmark.
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
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#F1F5F9' }}>
              📝 Dettagli Esperimento & Metriche
            </h4>
            {selectedExp && (
              <button
                onClick={() => handleDeleteSingle(selectedExp.experiment_id, selectedExp.name)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  background: 'rgba(239, 68, 68, 0.15)',
                  border: '1px solid rgba(239, 68, 68, 0.4)',
                  borderRadius: '6px',
                  color: '#F87171',
                  padding: '0.25rem 0.6rem',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                <Trash2 size={13} /> Elimina Selezionato
              </button>
            )}
          </div>

          {selectedExp ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Experiment ID:</span>
                <div style={{ fontSize: '0.85rem', color: '#38BDF8', fontWeight: 600, fontFamily: 'monospace' }}>
                  {selectedExp.experiment_id}
                </div>
              </div>

              {/* Metrics Grid */}
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Metriche Valutate:</span>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.25rem' }}>
                  {Object.entries(selectedExp.metrics || {}).map(([k, v]) => (
                    <div key={k} style={{ background: 'rgba(30, 41, 59, 0.6)', padding: '0.4rem 0.6rem', borderRadius: '6px' }}>
                      <div style={{ fontSize: '0.7rem', color: '#94A3B8' }}>{k}</div>
                      <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#F1F5F9', fontFamily: 'monospace' }}>
                        {typeof v === 'number' ? v.toFixed(3) : String(v)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Config JSON viewer */}
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Configurazione:</span>
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
                    fontFamily: 'monospace',
                  }}
                >
                  {JSON.stringify(selectedExp.config || {}, null, 2)}
                </pre>
              </div>
            </div>
          ) : (
            <div style={{ fontSize: '0.85rem', color: '#64748B', fontStyle: 'italic', padding: '1rem 0' }}>
              Seleziona un esperimento dalla tabella a sinistra per visualizzare dettagli e iperparametri.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
