import React from 'react';
import { useWorkbench } from '../../context';
import { Brain, Cpu } from 'lucide-react';
import { ValueHeadGauge } from './ValueHeadGauge';
import { ActionProbabilitiesChart } from './ActionProbabilitiesChart';
import { ObservationTensorViewer } from './ObservationTensorViewer';

export const BrainInspectorPane: React.FC = React.memo(() => {
  const { state, setHoveredAction, setSelectedAgentModel } = useWorkbench();
  const brainData = state.brainData;

  return (
    <div
      className="brain-inspector-pane"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.65rem',
        background: 'rgba(15, 23, 42, 0.6)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '12px',
        padding: '0.75rem',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)',
      }}
    >
      {/* Inspector Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'rgba(15, 23, 42, 0.85)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '8px',
          padding: '0.4rem 0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Brain size={16} color="#818CF8" />
          <span style={{ fontSize: '0.85rem', fontWeight: 800, color: '#F8FAFC', letterSpacing: '-0.01em' }}>
            Neural Brain Inspector
          </span>
        </div>

        {/* Model Architecture Switcher */}
        <div style={{ display: 'flex', gap: '0.25rem', background: 'rgba(30, 41, 59, 0.6)', borderRadius: '6px', padding: '2px' }}>
          <button
            onClick={() => setSelectedAgentModel('ppo')}
            style={{
              background: state.selectedAgentModel === 'ppo' ? '#6366F1' : 'transparent',
              color: state.selectedAgentModel === 'ppo' ? '#FFFFFF' : '#94A3B8',
              border: 'none',
              borderRadius: '4px',
              padding: '0.2rem 0.5rem',
              fontSize: '0.7rem',
              fontWeight: state.selectedAgentModel === 'ppo' ? 700 : 500,
              cursor: 'pointer',
            }}
          >
            PPO Masked AC
          </button>
          <button
            onClick={() => setSelectedAgentModel('dqn')}
            style={{
              background: state.selectedAgentModel === 'dqn' ? '#6366F1' : 'transparent',
              color: state.selectedAgentModel === 'dqn' ? '#FFFFFF' : '#94A3B8',
              border: 'none',
              borderRadius: '4px',
              padding: '0.2rem 0.5rem',
              fontSize: '0.7rem',
              fontWeight: state.selectedAgentModel === 'dqn' ? 700 : 500,
              cursor: 'pointer',
            }}
          >
            DQN Q-Net
          </button>
        </div>
      </div>

      {brainData ? (
        <>
          {/* Critic Value Head Gauge */}
          <ValueHeadGauge
            estimatedValue={brainData.estimated_value}
            modelType={brainData.model_type}
          />

          {/* Policy Action Distribution Chart with Synchronized Hover */}
          <ActionProbabilitiesChart
            probabilities={brainData.action_probabilities}
            actionMask={brainData.action_mask}
            actionLabels={brainData.action_labels}
            rawLogitsOrQ={brainData.masked_logits_or_q}
            greedyActionIndex={brainData.greedy_action_index}
            onActionHover={(meta) => setHoveredAction(meta)}
          />

          {/* Observation Tensor & Layer Activations Viewer */}
          <ObservationTensorViewer
            observationVector={brainData.observation_vector}
            layerActivations={brainData.layer_activations}
          />
        </>
      ) : (
        <div
          style={{
            padding: '2.5rem 1.5rem',
            textAlign: 'center',
            background: 'rgba(15, 23, 42, 0.4)',
            borderRadius: '8px',
            border: '1px dashed rgba(255, 255, 255, 0.1)',
          }}
        >
          <Cpu size={32} color="#64748B" style={{ margin: '0 auto 0.75rem auto', display: 'block' }} />
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#94A3B8', marginBottom: '0.25rem' }}>
            No Active Neural Agent Attached
          </div>
          <div style={{ fontSize: '0.72rem', color: '#64748B' }}>
            Start a game with a PPO or DQN agent, or load a training checkpoint to inspect real-time logits and value states.
          </div>
        </div>
      )}
    </div>
  );
});
