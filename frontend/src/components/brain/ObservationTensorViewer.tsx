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
      className="observation-tensor-viewer steampunk-panel"
      style={{
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: '2px solid #C59B27',
        borderRadius: '10px',
        padding: '0.85rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.6rem',
        boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
      }}
    >
      {/* Header & Tabs */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Layers size={16} color="#9E6B00" />
          <span
            style={{
              fontSize: '0.82rem',
              fontWeight: 800,
              color: '#23140C',
              fontFamily: "'Cinzel Decorative', Georgia, serif",
              letterSpacing: '0.04em',
            }}
          >
            {activeTab === 'obs' ? 'Thermionic Input Array' : 'Vacuum Tube Stages'}
          </span>
        </div>

        <div
          style={{
            display: 'flex',
            gap: '0.2rem',
            background: '#D8C3A0',
            borderRadius: '6px',
            padding: '2px',
            border: '1px solid #A88D75',
          }}
        >
          <button
            onClick={() => setActiveTab('obs')}
            style={{
              background: activeTab === 'obs'
                ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                : 'transparent',
              color: activeTab === 'obs' ? '#23140C' : '#5A3822',
              border: activeTab === 'obs' ? '1px solid #6E4E04' : '1px solid transparent',
              borderRadius: '4px',
              padding: '0.15rem 0.5rem',
              fontSize: '0.7rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: activeTab === 'obs' ? 800 : 600,
              cursor: 'pointer',
              boxShadow: activeTab === 'obs' ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
            }}
          >
            Input ({observationVector.length}d)
          </button>
          <button
            onClick={() => setActiveTab('layers')}
            style={{
              background: activeTab === 'layers'
                ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                : 'transparent',
              color: activeTab === 'layers' ? '#23140C' : '#5A3822',
              border: activeTab === 'layers' ? '1px solid #6E4E04' : '1px solid transparent',
              borderRadius: '4px',
              padding: '0.15rem 0.5rem',
              fontSize: '0.7rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: activeTab === 'layers' ? 800 : 600,
              cursor: 'pointer',
              boxShadow: activeTab === 'layers' ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
            }}
          >
            Filaments ({layerActivations.length})
          </button>
        </div>
      </div>

      {/* Observation Tensor Heatmap (Thermionic Vacuum Tubes) */}
      {activeTab === 'obs' && (
        <div>
          <div
            style={{
              fontSize: '0.72rem',
              color: '#5A3822',
              marginBottom: '0.35rem',
              display: 'flex',
              justifyContent: 'space-between',
              fontFamily: "'Crimson Pro', Georgia, serif",
            }}
          >
            <span>Filament Charge Potential</span>
            <span style={{ fontFamily: "'Courier Prime', monospace" }}>[-1.0V ⟷ +1.0V]</span>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(8px, 1fr))',
              gap: '2px',
              maxHeight: '120px',
              overflowY: 'auto',
              padding: '5px',
              background: '#2B1D14',
              borderRadius: '6px',
              border: '1.5px solid #8C6305',
              boxShadow: 'inset 0 2px 6px rgba(0,0,0,0.5)',
            }}
          >
            {observationVector.slice(0, 128).map((val, idx) => {
              const intensity = Math.min(1, Math.abs(val));
              const bg = val > 0
                ? `rgba(245, 158, 11, ${0.25 + intensity * 0.75})`
                : val < 0
                ? `rgba(185, 28, 28, ${0.25 + intensity * 0.75})`
                : 'rgba(74, 47, 29, 0.4)';

              return (
                <div
                  key={idx}
                  title={`Sensor [${idx}]: ${val.toFixed(3)}`}
                  style={{
                    width: '100%',
                    height: '11px',
                    backgroundColor: bg,
                    borderRadius: '1.5px',
                    boxShadow: val > 0 ? `0 0 3px rgba(245, 158, 11, ${intensity})` : undefined,
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
                background: '#FAF5EB',
                borderRadius: '6px',
                padding: '0.45rem 0.65rem',
                border: '1px solid rgba(184, 134, 11, 0.3)',
                boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#23140C', marginBottom: '0.25rem' }}>
                <span style={{ fontWeight: 800, color: '#9E6B00', fontFamily: "'Playfair Display', Georgia, serif" }}>{layer.layer_name}</span>
                <span style={{ color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
                  grid: [{layer.shape.join('×')}] | mean: {layer.mean.toFixed(2)}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '2px', height: '8px', background: '#2B1D14', borderRadius: '3px', overflow: 'hidden', padding: '1px', border: '1px solid #8C6305' }}>
                {layer.values.slice(0, 32).map((v, i) => (
                  <div
                    key={i}
                    style={{
                      flex: 1,
                      backgroundColor: v > 0 ? '#F59E0B' : '#785A42',
                      opacity: Math.min(1, Math.abs(v) / (layer.max || 1)),
                      borderRadius: '1px',
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
