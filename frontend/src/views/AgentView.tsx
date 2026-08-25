import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { BrainInspectionDTO } from '../api/types';
import { ActionProbabilitiesChart } from '../components/brain/ActionProbabilitiesChart';
import { Bot, RefreshCw } from 'lucide-react';

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
    <div
      className="agent-view"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem',
        color: '#23140C',
        fontFamily: "'Crimson Pro', Georgia, serif",
      }}
    >
      {/* Control Header */}
      <div
        className="steampunk-panel"
        style={{
          padding: '0.85rem 1.25rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
          background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
          <Bot size={18} color="#9E6B00" />
          <label style={{ fontSize: '0.82rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>
            Inspect Automaton Engine:
          </label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value as 'ppo' | 'dqn')}
            style={{
              backgroundColor: '#FAF5EB',
              color: '#23140C',
              border: '1.5px solid #8C6305',
              borderRadius: '6px',
              padding: '0.3rem 0.6rem',
              fontSize: '0.82rem',
              fontWeight: 700,
              fontFamily: "'Playfair Display', Georgia, serif",
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
            className="steampunk-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              padding: '0.35rem 0.75rem',
              fontSize: '0.78rem',
            }}
          >
            <RefreshCw size={13} />
            <span>Refresh Introspection</span>
          </button>
        </div>

        {inspection && (
          <div style={{ fontSize: '0.78rem', color: '#5A3822', fontFamily: "'Courier Prime', monospace" }}>
            Action Space: <strong style={{ color: '#23140C' }}>{inspection.action_probabilities.length}</strong> |
            Valid: <strong style={{ color: '#15803D' }}>{inspection.action_mask.filter(Boolean).length}</strong> |
            Greedy Selection:{' '}
            <strong style={{ color: '#9E6B00' }}>
              {inspection.action_labels[inspection.greedy_action_index] || `Action ${inspection.greedy_action_index}`}
            </strong>
          </div>
        )}
      </div>

      {/* Grid: Observation Features on Left, Action Probabilities on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '1.25rem' }}>
        {/* Observation Vector Breakdown */}
        <div
          className="steampunk-panel"
          style={{
            padding: '1.1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.85rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
              🔍 POMDP State Observation (s_t)
            </h4>
            <span style={{ fontSize: '0.74rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
              Vector Dim: {inspection?.observation_vector.length || 0}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.65rem', fontSize: '0.78rem' }}>
            <div style={{ background: '#FAF5EB', border: '1px solid #C59B27', padding: '0.65rem', borderRadius: '6px' }}>
              <div style={{ color: '#9E6B00', fontWeight: 800, marginBottom: '0.2rem', fontFamily: "'Playfair Display', serif" }}>🂠 Hand Feature Channels</div>
              <div style={{ color: '#5A3822', fontSize: '0.74rem' }}>Normalized [0, 1] per color (8 standard + Locomotive)</div>
            </div>
            <div style={{ background: '#FAF5EB', border: '1px solid #C59B27', padding: '0.65rem', borderRadius: '6px' }}>
              <div style={{ color: '#9E6B00', fontWeight: 800, marginBottom: '0.2rem', fontFamily: "'Playfair Display', serif" }}>🃏 Visible 5 Card Slots</div>
              <div style={{ color: '#5A3822', fontSize: '0.74rem' }}>5 × 10 One-hot encoded table visible slots</div>
            </div>
            <div style={{ background: '#FAF5EB', border: '1px solid #C59B27', padding: '0.65rem', borderRadius: '6px' }}>
              <div style={{ color: '#9E6B00', fontWeight: 800, marginBottom: '0.2rem', fontFamily: "'Playfair Display', serif" }}>🛤️ Route Ownership Graph</div>
              <div style={{ color: '#5A3822', fontSize: '0.74rem' }}>3 channels per route: [unclaimed, own, opponent]</div>
            </div>
            <div style={{ background: '#FAF5EB', border: '1px solid #C59B27', padding: '0.65rem', borderRadius: '6px' }}>
              <div style={{ color: '#9E6B00', fontWeight: 800, marginBottom: '0.2rem', fontFamily: "'Playfair Display', serif" }}>🎫 Destination Tickets</div>
              <div style={{ color: '#5A3822', fontSize: '0.74rem' }}>Connectivity status & point values</div>
            </div>
          </div>

          {/* Raw Values Mini-Heatmap */}
          <div>
            <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#4A2F1D', marginBottom: '0.4rem', fontFamily: "'Playfair Display', serif" }}>
              Thermionic Observation Array Sample:
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(14px, 1fr))',
                gap: '2px',
                maxHeight: '160px',
                overflowY: 'auto',
                background: '#2B1D14',
                padding: '6px',
                borderRadius: '6px',
                border: '1.5px solid #8C6305',
                boxShadow: 'inset 0 2px 6px rgba(0,0,0,0.5)',
              }}
            >
              {inspection?.observation_vector.map((val, idx) => {
                const intensity = Math.min(1, Math.abs(val));
                const bg = val > 0
                  ? `rgba(245, 158, 11, ${0.25 + intensity * 0.75})`
                  : val < 0
                  ? `rgba(185, 28, 28, ${0.25 + intensity * 0.75})`
                  : 'rgba(74, 47, 29, 0.3)';

                return (
                  <div
                    key={idx}
                    title={`obs[${idx}] = ${val.toFixed(3)}`}
                    style={{
                      height: '14px',
                      backgroundColor: bg,
                      borderRadius: '2px',
                      boxShadow: val > 0 ? `0 0 3px rgba(245, 158, 11, ${intensity})` : undefined,
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
