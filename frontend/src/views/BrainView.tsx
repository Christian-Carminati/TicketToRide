import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { BrainInspectionDTO } from '../api/types';
import { NeuralNetworkDiagram } from '../components/brain/NeuralNetworkDiagram';
import { ActionProbabilitiesChart } from '../components/brain/ActionProbabilitiesChart';

export const BrainView: React.FC = () => {
  const [inspection, setInspection] = useState<BrainInspectionDTO | null>(null);
  const [modelType, setModelType] = useState<'ppo' | 'dqn'>('ppo');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    setIsLoading(true);
    api.inspectBrain({ model_type: modelType })
      .then(setInspection)
      .catch((err) => console.error(err))
      .finally(() => setIsLoading(false));
  }, [modelType]);

  return (
    <div className="brain-view" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Model Selector Bar */}
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
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Architecture:</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value as 'ppo' | 'dqn')}
            style={{
              backgroundColor: '#1E293B',
              color: '#F1F5F9',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '0.35rem 0.6rem',
              fontSize: '0.85rem',
            }}
          >
            <option value="ppo">PPO Actor-Critic (Separated Heads)</option>
            <option value="dqn">Double-DQN (Action-Value Network)</option>
          </select>

          <button
            onClick={() => {
              setIsLoading(true);
              api.inspectBrain({ model_type: modelType })
                .then(setInspection)
                .finally(() => setIsLoading(false));
            }}
            disabled={isLoading}
            style={{
              backgroundColor: '#3B82F6',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '6px',
              padding: '0.35rem 0.75rem',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
            }}
          >
            🔄 Refresh
          </button>
        </div>

        {inspection && (
          <div style={{ fontSize: '0.85rem', color: '#94A3B8' }}>
            Identified Layers: <strong style={{ color: '#F1F5F9' }}>{inspection.layer_activations.length}</strong> |
            Activation Range:{' '}
            <strong style={{ color: '#38BDF8' }}>
              [
              {Math.min(...inspection.layer_activations.map((l) => l.min)).toFixed(2)},{' '}
              {Math.max(...inspection.layer_activations.map((l) => l.max)).toFixed(2)}
              ]
            </strong>
          </div>
        )}
      </div>

      {/* Network Architecture Diagram */}
      {inspection && (
        <NeuralNetworkDiagram
          layers={inspection.layer_activations}
          modelType={inspection.model_type}
          estimatedValue={inspection.estimated_value}
        />
      )}

      {/* Grid: Layer Activations Table & Action Probabilities */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.1fr) minmax(0, 0.9fr)', gap: '1.25rem' }}>
        {/* Layer Statistics Table */}
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.9)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '12px',
            padding: '1.25rem',
          }}
        >
          <h4 style={{ margin: '0 0 1rem 0', fontSize: '1rem', fontWeight: 600, color: '#F1F5F9' }}>
            📋 Layer-by-Layer Activation Tensor Statistics
          </h4>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: '#94A3B8' }}>
                  <th style={{ padding: '0.5rem' }}>Layer Name</th>
                  <th style={{ padding: '0.5rem' }}>Tensor Shape</th>
                  <th style={{ padding: '0.5rem' }}>Mean</th>
                  <th style={{ padding: '0.5rem' }}>Std</th>
                  <th style={{ padding: '0.5rem' }}>Min</th>
                  <th style={{ padding: '0.5rem' }}>Max</th>
                </tr>
              </thead>
              <tbody>
                {inspection?.layer_activations.map((layer, idx) => (
                  <tr
                    key={idx}
                    style={{
                      borderBottom: '1px solid rgba(255,255,255,0.05)',
                      backgroundColor: idx % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent',
                    }}
                  >
                    <td style={{ padding: '0.5rem', fontWeight: 600, color: '#38BDF8' }}>
                      {layer.layer_name}
                    </td>
                    <td style={{ padding: '0.5rem', color: '#94A3B8' }}>[{layer.shape.join(', ')}]</td>
                    <td style={{ padding: '0.5rem', color: '#F1F5F9' }}>{layer.mean.toFixed(3)}</td>
                    <td style={{ padding: '0.5rem', color: '#94A3B8' }}>{layer.std.toFixed(3)}</td>
                    <td style={{ padding: '0.5rem', color: '#94A3B8' }}>{layer.min.toFixed(3)}</td>
                    <td style={{ padding: '0.5rem', color: '#F59E0B' }}>{layer.max.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Action Probabilities Chart */}
        {inspection && (
          <ActionProbabilitiesChart
            probabilities={inspection.action_probabilities}
            actionMask={inspection.action_mask}
            actionLabels={inspection.action_labels}
            rawLogitsOrQ={inspection.raw_logits_or_q}
            greedyActionIndex={inspection.greedy_action_index}
          />
        )}
      </div>
    </div>
  );
};
