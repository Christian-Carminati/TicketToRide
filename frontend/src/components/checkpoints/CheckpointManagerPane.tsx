import React, { useState, useEffect } from 'react';
import { api } from '../../api/client';
import { CheckpointDTO } from '../../api/types';
import { Trash2, RefreshCw, Search, HardDrive, BrainCircuit, CheckCircle, AlertTriangle } from 'lucide-react';

export const CheckpointManagerPane: React.FC = () => {
  const [checkpoints, setCheckpoints] = useState<CheckpointDTO[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [actionMessage, setActionMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const fetchCheckpoints = () => {
    setIsLoading(true);
    api.listCheckpoints()
      .then((list) => {
        setCheckpoints(list);
      })
      .catch((err) => {
        console.error(err);
        setActionMessage({ type: 'error', text: 'Impossibile caricare l\'elenco dei checkpoint.' });
      })
      .finally(() => setIsLoading(false));
  };

  useEffect(() => {
    fetchCheckpoints();
  }, []);

  const handleDeleteSingle = async (ckptId: string, name: string) => {
    try {
      setIsLoading(true);
      await api.deleteCheckpoint(ckptId);
      setActionMessage({ type: 'success', text: `Checkpoint "${name}" eliminato con successo.` });
      fetchCheckpoints();
    } catch (err) {
      console.error(err);
      setActionMessage({ type: 'error', text: 'Errore durante l\'eliminazione del checkpoint.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAll = async () => {
    if (checkpoints.length === 0) return;
    try {
      setIsLoading(true);
      const res = await api.deleteAllCheckpoints();
      setActionMessage({ type: 'success', text: `Tutti i file checkpoint (${res.deleted_count}) sono stati eliminati.` });
      setCheckpoints([]);
    } catch (err) {
      console.error(err);
      setActionMessage({ type: 'error', text: 'Errore durante la cancellazione massiva dei checkpoint.' });
    } finally {
      setIsLoading(false);
    }
  };

  const filtered = checkpoints.filter(
    (c) =>
      c.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      c.checkpoint_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      c.algorithm.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const totalSizeMb = checkpoints.reduce((acc, c) => acc + (c.size_mb || 0), 0);

  return (
    <div className="checkpoint-manager-pane" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Toast Notification Banner */}
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

      {/* Header Bar with Stats and Search */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
          background: 'rgba(30, 41, 59, 0.6)',
          borderRadius: '8px',
          padding: '0.75rem 1rem',
          border: '1px solid rgba(255, 255, 255, 0.08)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <HardDrive size={16} color="#A855F7" />
            <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#F1F5F9' }}>
              File Checkpoint Neurali ({checkpoints.length})
            </span>
            <span style={{ fontSize: '0.72rem', color: '#94A3B8', fontFamily: 'monospace' }}>
              • {totalSizeMb.toFixed(2)} MB totali
            </span>
          </div>

          {/* Search filter */}
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Search size={13} color="#64748B" style={{ position: 'absolute', left: '0.6rem' }} />
            <input
              type="text"
              placeholder="Filtra checkpoint (.pt)..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                backgroundColor: '#0F172A',
                color: '#F1F5F9',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '6px',
                padding: '0.3rem 0.6rem 0.3rem 1.8rem',
                fontSize: '0.8rem',
                minWidth: '220px',
              }}
            />
          </div>
        </div>

        {/* Action buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button
            onClick={handleDeleteAll}
            disabled={isLoading || checkpoints.length === 0}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              backgroundColor: checkpoints.length === 0 ? 'rgba(239, 68, 68, 0.2)' : '#EF4444',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.35rem 0.75rem',
              fontWeight: 600,
              fontSize: '0.8rem',
              cursor: checkpoints.length === 0 ? 'not-allowed' : 'pointer',
              opacity: checkpoints.length === 0 ? 0.5 : 1,
              transition: 'all 0.15s ease',
            }}
            title="Elimina tutti i file checkpoint dal disco"
          >
            <Trash2 size={13} />
            <span>Elimina Tutti ({checkpoints.length})</span>
          </button>

          <button
            onClick={fetchCheckpoints}
            disabled={isLoading}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              backgroundColor: '#3B82F6',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.35rem 0.75rem',
              fontWeight: 600,
              fontSize: '0.8rem',
              cursor: 'pointer',
            }}
          >
            <RefreshCw size={13} />
            <span>Aggiorna</span>
          </button>
        </div>
      </div>

      {/* Checkpoint Table */}
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.8)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '8px',
          overflow: 'hidden',
        }}
      >
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'rgba(30, 41, 59, 0.7)', color: '#94A3B8', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                <th style={{ padding: '0.6rem 0.8rem' }}>Nome Modello / Checkpoint</th>
                <th style={{ padding: '0.6rem 0.8rem' }}>Algoritmo</th>
                <th style={{ padding: '0.6rem 0.8rem' }}>Nome File</th>
                <th style={{ padding: '0.6rem 0.8rem' }}>Dimensione</th>
                <th style={{ padding: '0.6rem 0.8rem' }}>Data Ultima Modifica</th>
                <th style={{ padding: '0.6rem 0.8rem', textAlign: 'center' }}>Azioni</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((ckpt, idx) => (
                <tr
                  key={ckpt.checkpoint_id}
                  style={{
                    borderBottom: '1px solid rgba(255,255,255,0.04)',
                    background: idx % 2 === 0 ? 'rgba(30, 41, 59, 0.2)' : 'transparent',
                  }}
                >
                  <td style={{ padding: '0.6rem 0.8rem', fontWeight: 600, color: '#F1F5F9' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <BrainCircuit size={14} color="#A855F7" />
                      <span>{ckpt.name}</span>
                    </div>
                  </td>

                  <td style={{ padding: '0.6rem 0.8rem' }}>
                    <span
                      style={{
                        padding: '0.15rem 0.4rem',
                        borderRadius: '4px',
                        backgroundColor: ckpt.algorithm === 'ppo' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                        color: ckpt.algorithm === 'ppo' ? '#34D399' : '#FBBF24',
                        fontSize: '0.72rem',
                        fontWeight: 700,
                      }}
                    >
                      {ckpt.algorithm.toUpperCase()}
                    </span>
                  </td>

                  <td style={{ padding: '0.6rem 0.8rem', color: '#94A3B8', fontFamily: 'monospace', fontSize: '0.75rem' }}>
                    {ckpt.checkpoint_id}
                  </td>

                  <td style={{ padding: '0.6rem 0.8rem', color: '#38BDF8', fontFamily: 'monospace', fontWeight: 600 }}>
                    {ckpt.size_mb} MB
                  </td>

                  <td style={{ padding: '0.6rem 0.8rem', color: '#CBD5E1', fontFamily: 'monospace', fontSize: '0.75rem' }}>
                    {ckpt.modified_at}
                  </td>

                  <td style={{ padding: '0.6rem 0.8rem', textAlign: 'center' }}>
                    <button
                      onClick={() => handleDeleteSingle(ckpt.checkpoint_id, ckpt.name)}
                      style={{
                        background: 'rgba(239, 68, 68, 0.15)',
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        borderRadius: '4px',
                        color: '#F87171',
                        padding: '0.25rem 0.55rem',
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                        fontSize: '0.74rem',
                        fontWeight: 600,
                      }}
                      title="Elimina questo file .pt"
                    >
                      <Trash2 size={12} />
                      <span>Elimina</span>
                    </button>
                  </td>
                </tr>
              ))}

              {filtered.length === 0 && (
                <tr>
                  <td colSpan={6} style={{ padding: '2rem 1rem', textAlign: 'center', color: '#64748B' }}>
                    Nessun file di checkpoint presente in experiments/checkpoints/. Esegui un addestramento per salvare nuovi modelli.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
