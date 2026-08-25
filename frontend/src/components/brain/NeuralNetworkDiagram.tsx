import React from 'react';
import { LayerActivationDTO } from '../../api/types';
import { Brain } from 'lucide-react';

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
      className="neural-diagram-container steampunk-panel"
      style={{
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: '2px solid #C59B27',
        borderRadius: '10px',
        padding: '1rem',
        boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Brain size={16} color="#9E6B00" />
          <h4 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
            Neural Network Architecture & Vacuum Stages ({modelType.toUpperCase()})
          </h4>
        </div>

        {estimatedValue !== null && estimatedValue !== undefined && (
          <div
            style={{
              padding: '0.25rem 0.65rem',
              borderRadius: '6px',
              backgroundColor: '#FAF5EB',
              border: '1.5px solid #15803D',
              color: '#15803D',
              fontSize: '0.82rem',
              fontWeight: 800,
              fontFamily: "'Courier Prime', monospace",
            }}
          >
            Critic Value V(s): {estimatedValue >= 0 ? `+${estimatedValue.toFixed(2)}` : estimatedValue.toFixed(2)}
          </div>
        )}
      </div>

      {/* Layer Flow Diagram */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          overflowX: 'auto',
          padding: '0.5rem 0',
        }}
      >
        {/* Input Observation Column */}
        <div
          style={{
            minWidth: '110px',
            background: '#FAF5EB',
            border: '1px solid #C59B27',
            borderRadius: '8px',
            padding: '0.65rem',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: '0.78rem', fontWeight: 800, color: '#9E6B00', marginBottom: '0.2rem', fontFamily: "'Playfair Display', serif" }}>
            Input State
          </div>
          <div style={{ fontSize: '0.7rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>Obs Vector s_t</div>
          <div
            style={{
              height: '70px',
              margin: '0.4rem 0',
              borderRadius: '4px',
              background: 'linear-gradient(to top, #1D4ED8, #0D9488)',
              opacity: 0.85,
            }}
          />
          <div style={{ fontSize: '0.68rem', color: '#5A3822', fontWeight: 700 }}>POMDP Bounded</div>
        </div>

        <div style={{ color: '#8C6305', fontSize: '1.1rem', fontWeight: 800 }}>➔</div>

        {/* Hidden Layers */}
        {layers.map((layer, idx) => {
          const intensity = Math.min(Math.max(layer.mean, 0.1), 1.0);
          return (
            <React.Fragment key={idx}>
              <div
                style={{
                  minWidth: '120px',
                  background: '#FAF5EB',
                  border: '1px solid #C59B27',
                  borderRadius: '8px',
                  padding: '0.65rem',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '0.78rem', fontWeight: 800, color: '#23140C', marginBottom: '0.2rem', fontFamily: "'Playfair Display', serif" }}>
                  {layer.layer_name.split('_').slice(0, 3).join(' ')}
                </div>
                <div style={{ fontSize: '0.68rem', color: '#785A42', marginBottom: '0.3rem', fontFamily: "'Courier Prime', monospace" }}>
                  [{layer.shape.join('×')}]
                </div>

                {/* Activation Heat Bar */}
                <div
                  style={{
                    height: '70px',
                    margin: '0.4rem 0',
                    borderRadius: '4px',
                    backgroundColor: '#2B1D14',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'flex-end',
                    overflow: 'hidden',
                    border: '1px solid #8C6305',
                  }}
                >
                  <div
                    style={{
                      height: '100%',
                      width: '100%',
                      transform: `scaleY(${intensity})`,
                      transformOrigin: 'bottom',
                      background: 'linear-gradient(to top, #15803D, #D97706)',
                      transition: 'transform 0.3s ease',
                    }}
                  />
                </div>

                <div style={{ fontSize: '0.72rem', color: '#23140C', fontFamily: "'Courier Prime', monospace", fontWeight: 700 }}>
                  μ = {layer.mean.toFixed(2)}
                </div>
                <div style={{ fontSize: '0.65rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
                  σ={layer.std.toFixed(2)} | max={layer.max.toFixed(2)}
                </div>
              </div>

              {idx < layers.length - 1 && <div style={{ color: '#8C6305', fontSize: '1.1rem', fontWeight: 800 }}>➔</div>}
            </React.Fragment>
          );
        })}

        <div style={{ color: '#8C6305', fontSize: '1.1rem', fontWeight: 800 }}>➔</div>

        {/* Output Heads Column */}
        <div
          style={{
            minWidth: '120px',
            background: '#FAF5EB',
            border: '1px solid #C59B27',
            borderRadius: '8px',
            padding: '0.65rem',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: '0.78rem', fontWeight: 800, color: '#9E6B00', marginBottom: '0.2rem', fontFamily: "'Playfair Display', serif" }}>
            Output Heads
          </div>
          <div style={{ fontSize: '0.72rem', color: '#15803D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>
            Policy π(a|s)
          </div>
          <div style={{ fontSize: '0.72rem', color: '#1D4ED8', fontWeight: 800, marginTop: '0.1rem', fontFamily: "'Playfair Display', serif" }}>
            Value V(s)
          </div>
          <div
            style={{
              height: '70px',
              margin: '0.4rem 0',
              borderRadius: '4px',
              background: 'linear-gradient(to top, #15803D, #D97706)',
              opacity: 0.85,
            }}
          />
          <div style={{ fontSize: '0.68rem', color: '#5A3822', fontWeight: 700 }}>Masked Logits</div>
        </div>
      </div>
    </div>
  );
};
