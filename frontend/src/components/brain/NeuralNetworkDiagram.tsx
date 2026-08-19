import React from 'react';
import { LayerActivationDTO } from '../../api/types';

interface NeuralNetworkDiagramProps {
  layers: LayerActivationDTO[];
  modelType?: 'dqn' | 'ppo';
  estimatedValue?: number | null;
}

export const NeuralNetworkDiagram: React.FC<NeuralNetworkDiagramProps> = ({
  layers,
  modelType = 'ppo',
  estimatedValue,
}) => {
  return (
    <div
      className="neural-diagram-container"
      style={{
        background: 'rgba(15, 23, 42, 0.9)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '12px',
        padding: '1.25rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#F1F5F9' }}>
          🧠 Neural Network Architecture & Layer Activations ({modelType.toUpperCase()})
        </h4>
        {estimatedValue !== null && estimatedValue !== undefined && (
          <div
            style={{
              padding: '0.3rem 0.75rem',
              borderRadius: '6px',
              backgroundColor: 'rgba(56, 189, 248, 0.15)',
              border: '1px solid #38BDF8',
              color: '#38BDF8',
              fontSize: '0.85rem',
              fontWeight: 700,
            }}
          >
            Critic Value V(s): {estimatedValue.toFixed(2)}
          </div>
        )}
      </div>

      {/* Layer Flow Diagram */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          overflowX: 'auto',
          padding: '1rem 0',
        }}
      >
        {/* Input Observation Column */}
        <div
          style={{
            minWidth: '120px',
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
            padding: '0.75rem',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#A5B4FC', marginBottom: '0.4rem' }}>
            Input State
          </div>
          <div style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Obs Vector s_t</div>
          <div
            style={{
              height: '80px',
              margin: '0.5rem 0',
              borderRadius: '4px',
              background: 'linear-gradient(to top, #4F46E5, #06B6D4)',
              opacity: 0.8,
            }}
          />
          <div style={{ fontSize: '0.7rem', color: '#CBD5E1' }}>POMDP Bounded</div>
        </div>

        <div style={{ color: '#64748B', fontSize: '1.2rem' }}>➔</div>

        {/* Hidden Layers */}
        {layers.map((layer, idx) => {
          const intensity = Math.min(Math.max(layer.mean, 0.1), 1.0);
          return (
            <React.Fragment key={idx}>
              <div
                style={{
                  minWidth: '140px',
                  background: 'rgba(30, 41, 59, 0.6)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                  padding: '0.75rem',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#38BDF8', marginBottom: '0.2rem' }}>
                  {layer.layer_name.split('_').slice(0, 3).join(' ')}
                </div>
                <div style={{ fontSize: '0.7rem', color: '#64748B', marginBottom: '0.4rem' }}>
                  Shape: [{layer.shape.join('×')}]
                </div>

                {/* Activation Heat Bar */}
                <div
                  style={{
                    height: '80px',
                    margin: '0.5rem 0',
                    borderRadius: '4px',
                    backgroundColor: '#0F172A',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'flex-end',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      height: '100%',
                      width: '100%',
                      transform: `scaleY(${intensity})`,
                      transformOrigin: 'bottom',
                      background: 'linear-gradient(to top, #10B981, #F59E0B)',
                      transition: 'transform 0.3s ease',
                    }}
                  />
                </div>

                <div style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
                  Mean: <strong style={{ color: '#F1F5F9' }}>{layer.mean.toFixed(2)}</strong>
                </div>
                <div style={{ fontSize: '0.7rem', color: '#64748B' }}>
                  Std: {layer.std.toFixed(2)} | Max: {layer.max.toFixed(2)}
                </div>
              </div>

              {idx < layers.length - 1 && <div style={{ color: '#64748B', fontSize: '1.2rem' }}>➔</div>}
            </React.Fragment>
          );
        })}

        <div style={{ color: '#64748B', fontSize: '1.2rem' }}>➔</div>

        {/* Output Heads Column */}
        <div
          style={{
            minWidth: '130px',
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
            padding: '0.75rem',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#F59E0B', marginBottom: '0.4rem' }}>
            Output Heads
          </div>
          <div style={{ fontSize: '0.75rem', color: '#10B981', fontWeight: 600 }}>
            Policy π(a|s)
          </div>
          <div style={{ fontSize: '0.75rem', color: '#38BDF8', fontWeight: 600, marginTop: '0.2rem' }}>
            Value V(s)
          </div>
          <div
            style={{
              height: '80px',
              margin: '0.5rem 0',
              borderRadius: '4px',
              background: 'linear-gradient(to top, #10B981, #FBBF24)',
              opacity: 0.85,
            }}
          />
          <div style={{ fontSize: '0.7rem', color: '#CBD5E1' }}>Masked Action Logits</div>
        </div>
      </div>
    </div>
  );
};
