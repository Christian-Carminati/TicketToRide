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
        setActionMessage({ type: 'error', text: 'Unable to load checkpoint archive.' });
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
      setActionMessage({ type: 'success', text: `Checkpoint "${name}" successfully deleted.` });
      fetchCheckpoints();
    } catch (err) {
      console.error(err);
      setActionMessage({ type: 'error', text: 'Error deleting checkpoint file.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAll = async () => {
    if (checkpoints.length === 0) return;
    try {
      setIsLoading(true);
      const res = await api.deleteAllCheckpoints();
      setActionMessage({ type: 'success', text: `All checkpoint files (${res.deleted_count}) have been deleted.` });
      setCheckpoints([]);
    } catch (err) {
      console.error(err);
      setActionMessage({ type: 'error', text: 'Error purging all checkpoint files.' });
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
            borderRadius: '6px',
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

      {/* Header Bar with Stats and Search */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
          background: 'linear-gradient(180deg, #FAF4E6 0%, #EADBBE 100%)',
          borderRadius: '8px',
          padding: '0.75rem 1rem',
          border: '1.5px solid #C59B27',
          boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <HardDrive size={16} color="#9E6B00" />
            <span style={{ fontSize: '0.9rem', fontWeight: 800, color: '#23140C', fontFamily: "'Cinzel Decorative', Georgia, serif" }}>
              Neural Model Archive (.pt) ({checkpoints.length})
            </span>
            <span style={{ fontSize: '0.75rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
              • {totalSizeMb.toFixed(2)} MB stored
            </span>
          </div>

          {/* Search filter */}
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Search size={13} color="#785A42" style={{ position: 'absolute', left: '0.6rem' }} />
            <input
              type="text"
              placeholder="Search checkpoints..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                backgroundColor: '#FAF5EB',
                color: '#23140C',
                border: '1.5px solid #8C6305',
                borderRadius: '6px',
                padding: '0.3rem 0.6rem 0.3rem 1.8rem',
                fontSize: '0.8rem',
                minWidth: '220px',
                fontFamily: "'Courier Prime', monospace",
              }}
            />
          </div>
        </div>

        {/* Action buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button
            onClick={handleDeleteAll}
            disabled={isLoading || checkpoints.length === 0}
            className="steampunk-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: 'linear-gradient(180deg, #F87171 0%, #DC2626 50%, #991B1B 100%)',
              color: '#FFFFFF',
              border: '1px solid #7F1D1D',
              fontSize: '0.78rem',
            }}
            title="Elimina tutti i file checkpoint dal disco"
          >
            <Trash2 size={13} />
            <span>Purge All ({checkpoints.length})</span>
          </button>

          <button
            onClick={fetchCheckpoints}
            disabled={isLoading}
            className="steampunk-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              fontSize: '0.78rem',
            }}
          >
            <RefreshCw size={13} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Checkpoint Table */}
      <div
        style={{
          background: '#FAF5EB',
          border: '1.5px solid #C59B27',
          borderRadius: '8px',
          overflow: 'hidden',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        }}
      >
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'linear-gradient(180deg, #EFE1C7 0%, #E2CFAC 100%)', color: '#23140C', borderBottom: '2px solid #C59B27' }}>
                <th style={{ padding: '0.6rem 0.8rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Model Name / Tag</th>
                <th style={{ padding: '0.6rem 0.8rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Algorithm</th>
                <th style={{ padding: '0.6rem 0.8rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>File Identifier</th>
                <th style={{ padding: '0.6rem 0.8rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Size</th>
                <th style={{ padding: '0.6rem 0.8rem', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Timestamp</th>
                <th style={{ padding: '0.6rem 0.8rem', textAlign: 'center', fontWeight: 800, fontFamily: "'Playfair Display', Georgia, serif" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((ckpt, idx) => (
                <tr
                  key={ckpt.checkpoint_id}
                  style={{
                    borderBottom: '1px solid rgba(184, 134, 11, 0.2)',
                    background: idx % 2 === 0 ? 'rgba(246, 238, 223, 0.5)' : 'transparent',
                  }}
                >
                  <td style={{ padding: '0.6rem 0.8rem', fontWeight: 700, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <BrainCircuit size={14} color="#9E6B00" />
                      <span>{ckpt.name}</span>
                    </div>
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
                      {ckpt.algorithm.toUpperCase()}
                    </span>
                  </td>

                  <td style={{ padding: '0.6rem 0.8rem', color: '#5A3822', fontFamily: "'Courier Prime', monospace", fontSize: '0.75rem' }}>
                    {ckpt.checkpoint_id}
                  </td>

                  <td style={{ padding: '0.6rem 0.8rem', color: '#9E6B00', fontFamily: "'Courier Prime', monospace", fontWeight: 800 }}>
                    {ckpt.size_mb} MB
                  </td>

                  <td style={{ padding: '0.6rem 0.8rem', color: '#785A42', fontFamily: "'Courier Prime', monospace", fontSize: '0.75rem' }}>
                    {ckpt.modified_at}
                  </td>

                  <td style={{ padding: '0.6rem 0.8rem', textAlign: 'center' }}>
                    <button
                      onClick={() => handleDeleteSingle(ckpt.checkpoint_id, ckpt.name)}
                      className="steampunk-btn"
                      style={{
                        padding: '0.2rem 0.5rem',
                        fontSize: '0.72rem',
                        background: 'linear-gradient(180deg, #F87171 0%, #DC2626 50%, #991B1B 100%)',
                        color: '#FFFFFF',
                        border: '1px solid #7F1D1D',
                      }}
                      title="Elimina questo file .pt"
                    >
                      <Trash2 size={12} />
                      <span>Delete</span>
                    </button>
                  </td>
                </tr>
              ))}

              {filtered.length === 0 && (
                <tr>
                  <td colSpan={6} style={{ padding: '2rem 1rem', textAlign: 'center', color: '#785A42', fontStyle: 'italic', fontFamily: "'Crimson Pro', serif" }}>
                    No checkpoint files found in experiments/checkpoints/. Run training to serialize trained automaton policies.
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
