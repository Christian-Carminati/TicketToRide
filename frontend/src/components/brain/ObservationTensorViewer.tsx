import React, { useState } from 'react';
import { Layers } from 'lucide-react';
import { LayerActivationDTO } from '../../api/types';

interface ObservationTensorViewerProps {
  observationVector?: number[];
  layerActivations?: LayerActivationDTO[];
}

export const ObservationTensorViewer: React.FC<ObservationTensorViewerProps> = ({
  observationVector = [],
  layerActivations = [],
}) => {
  const [activeTab, setActiveTab] = useState<'obs' | 'layers'>('obs');

  return (
    <div
      className="observation-tensor-viewer"
      style={{
        background: 'rgba(15, 23, 42, 0.85)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '10px',
        padding: '0.85rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.6rem',
      }}
    >
      {/* Header & Tabs */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Layers size={16} color="#38BDF8" />
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#F1F5F9', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {activeTab === 'obs' ? 'Input Observation Tensor' : 'Hidden Layer Activations'}
          </span>
        </div>

        <div style={{ display: 'flex', background: 'rgba(30, 41, 59, 0.6)', borderRadius: '6px', padding: '2px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
          <button
            onClick={() => setActiveTab('obs')}
            style={{
              background: activeTab === 'obs' ? '#3B82F6' : 'transparent',
              color: activeTab === 'obs' ? '#FFFFFF' : '#94A3B8',
              border: 'none',
              borderRadius: '4px',
              padding: '0.15rem 0.45rem',
              fontSize: '0.7rem',
              fontWeight: activeTab === 'obs' ? 700 : 500,
              cursor: 'pointer',
            }}
          >
            Input ({observationVector.length}d)
          </button>
          <button
            onClick={() => setActiveTab('layers')}
            style={{
              background: activeTab === 'layers' ? '#3B82F6' : 'transparent',
              color: activeTab === 'layers' ? '#FFFFFF' : '#94A3B8',
              border: 'none',
              borderRadius: '4px',
              padding: '0.15rem 0.45rem',
              fontSize: '0.7rem',
              fontWeight: activeTab === 'layers' ? 700 : 500,
              cursor: 'pointer',
            }}
          >
            MLP Layers ({layerActivations.length})
          </button>
        </div>
      </div>

      {/* Observation Tensor Heatmap */}
      {activeTab === 'obs' && (
        <div>
          <div style={{ fontSize: '0.7rem', color: '#94A3B8', marginBottom: '0.35rem', display: 'flex', justifyContent: 'space-between' }}>
            <span>Feature Vector Activations</span>
            <span style={{ fontFamily: 'monospace' }}>[-1.0 ⟷ +1.0]</span>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(8px, 1fr))',
              gap: '2px',
              maxHeight: '120px',
              overflowY: 'auto',
              padding: '4px',
              background: 'rgba(10, 15, 30, 0.6)',
              borderRadius: '6px',
              border: '1px solid rgba(255,255,255,0.03)',
            }}
          >
            {observationVector.slice(0, 128).map((val, idx) => {
              const intensity = Math.min(1, Math.abs(val));
              const bg = val > 0
                ? `rgba(56, 189, 248, ${0.15 + intensity * 0.85})`
                : val < 0
                ? `rgba(244, 63, 94, ${0.15 + intensity * 0.85})`
                : 'rgba(30, 41, 59, 0.4)';

              return (
                <div
                  key={idx}
                  title={`Feature [${idx}]: ${val.toFixed(3)}`}
                  style={{
                    width: '100%',
                    height: '10px',
                    backgroundColor: bg,
                    borderRadius: '1px',
                    transition: 'background-color 0.2s ease',
                  }}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* Layer Activations View */}
      {activeTab === 'layers' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '140px', overflowY: 'auto' }}>
          {layerActivations.map((layer) => (
            <div
              key={layer.layer_name}
              style={{
                background: 'rgba(30, 41, 59, 0.5)',
                borderRadius: '6px',
                padding: '0.4rem 0.6rem',
                border: '1px solid rgba(255, 255, 255, 0.04)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#F1F5F9', marginBottom: '0.2rem' }}>
                <span style={{ fontWeight: 700, color: '#38BDF8' }}>{layer.layer_name}</span>
                <span style={{ color: '#94A3B8', fontFamily: 'monospace' }}>
                  shape: [{layer.shape.join('×')}] | mean: {layer.mean.toFixed(2)}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '2px', height: '6px', background: 'rgba(15, 23, 42, 0.8)', borderRadius: '2px', overflow: 'hidden' }}>
                {layer.values.slice(0, 32).map((v, i) => (
                  <div
                    key={i}
                    style={{
                      flex: 1,
                      backgroundColor: v > 0 ? '#38BDF8' : '#64748B',
                      opacity: Math.min(1, Math.abs(v) / (layer.max || 1)),
                    }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
