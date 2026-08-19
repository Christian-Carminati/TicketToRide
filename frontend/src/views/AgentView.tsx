import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { BrainInspectionDTO } from '../api/types';
import { ActionProbabilitiesChart } from '../components/brain/ActionProbabilitiesChart';

export const AgentView: React.FC = () => {
  const [inspection, setInspection] = useState<BrainInspectionDTO | null>(null);
  const [selectedSessionId] = useState<string>('');
  const [modelType, setModelType] = useState<'ppo' | 'dqn'>('ppo');
  const [isLoading, setIsLoading] = useState(false);

  // Trigger inspection on mount or when session/model changes
  useEffect(() => {
    setIsLoading(true);
    api.inspectBrain({ session_id: selectedSessionId || undefined, model_type: modelType })
      .then(setInspection)
      .catch((err) => console.error(err))
      .finally(() => setIsLoading(false));
  }, [selectedSessionId, modelType]);

  return (
    <div className="agent-view" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Control Header */}
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
          <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Inspect Model Type:</label>
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
            <option value="ppo">PPO Policy & Value Head</option>
            <option value="dqn">Double-DQN Q-Network</option>
          </select>

          <button
            onClick={() => {
              setIsLoading(true);
              api.inspectBrain({ session_id: selectedSessionId || undefined, model_type: modelType })
                .then(setInspection)
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
            🔄 Refresh Inspection
          </button>
        </div>

        {inspection && (
          <div style={{ fontSize: '0.85rem', color: '#94A3B8' }}>
            Action Space Dim: <strong style={{ color: '#F1F5F9' }}>{inspection.action_probabilities.length}</strong> |
            Valid Actions: <strong style={{ color: '#10B981' }}>{inspection.action_mask.filter(Boolean).length}</strong> |
            Greedy Selection:{' '}
            <strong style={{ color: '#F59E0B' }}>
              {inspection.action_labels[inspection.greedy_action_index] || `Action ${inspection.greedy_action_index}`}
            </strong>
          </div>
        )}
      </div>

      {/* Grid: Observation Features on Left, Action Probabilities on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '1.25rem' }}>
        {/* Observation Vector Breakdown */}
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
              🔍 POMDP State Observation (s_t)
            </h4>
            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
              Vector Dim: {inspection?.observation_vector.length || 0}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.8rem' }}>
            <div style={{ background: 'rgba(30, 41, 59, 0.6)', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ color: '#38BDF8', fontWeight: 600, marginBottom: '0.25rem' }}>🂠 Hand Feature Channels</div>
              <div style={{ color: '#94A3B8' }}>Normalized [0, 1] per color (8 standard + Locomotive)</div>
            </div>
            <div style={{ background: 'rgba(30, 41, 59, 0.6)', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ color: '#38BDF8', fontWeight: 600, marginBottom: '0.25rem' }}>🃏 Visible 5 Card Slots</div>
              <div style={{ color: '#94A3B8' }}>5 × 10 One-hot encoded table visible slots</div>
            </div>
            <div style={{ background: 'rgba(30, 41, 59, 0.6)', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ color: '#38BDF8', fontWeight: 600, marginBottom: '0.25rem' }}>🛤️ Route Ownership Graph</div>
              <div style={{ color: '#94A3B8' }}>3 channels per route: [unclaimed, own, opponent]</div>
            </div>
            <div style={{ background: 'rgba(30, 41, 59, 0.6)', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ color: '#38BDF8', fontWeight: 600, marginBottom: '0.25rem' }}>🎫 Destination Tickets</div>
              <div style={{ color: '#94A3B8' }}>Connectivity status & point values</div>
            </div>
          </div>

          {/* Raw Values Mini-Heatmap */}
          <div>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#CBD5E1', marginBottom: '0.5rem' }}>
              Observation Tensor Sample Heatmap:
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(14px, 1fr))',
                gap: '2px',
                maxHeight: '160px',
                overflowY: 'auto',
                background: '#0B1120',
                padding: '0.5rem',
                borderRadius: '6px',
              }}
            >
              {inspection?.observation_vector.map((val, idx) => {
                const alpha = Math.min(Math.max(val, 0.05), 1.0);
                return (
                  <div
                    key={idx}
                    title={`obs[${idx}] = ${val.toFixed(3)}`}
                    style={{
                      height: '14px',
                      backgroundColor: val > 0 ? `rgba(56, 189, 248, ${alpha})` : 'rgba(255, 255, 255, 0.03)',
                      borderRadius: '2px',
                    }}
                  />
                );
              })}
            </div>
          </div>
        </div>

        {/* Action Probabilities and Masking */}
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
