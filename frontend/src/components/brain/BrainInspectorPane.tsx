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
      className="brain-inspector-pane steampunk-panel"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.65rem',
        background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
        border: '2px solid #C59B27',
        borderRadius: '12px',
        padding: '0.85rem',
        boxShadow: '0 6px 24px rgba(0, 0, 0, 0.35)',
      }}
    >
      {/* Inspector Analytical Engine Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'linear-gradient(180deg, #3A261A 0%, #26180F 100%)',
          border: '1.5px solid #C59B27',
          borderRadius: '8px',
          padding: '0.45rem 0.85rem',
          boxShadow: '0 2px 6px rgba(0,0,0,0.35)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Brain size={16} color="#F6DC88" />
          <span
            style={{
              fontSize: '0.9rem',
              fontWeight: 800,
              color: '#FAF5EB',
              fontFamily: "'Cinzel Decorative', Georgia, serif",
              letterSpacing: '0.03em',
            }}
          >
            Neural Analytical Engine
          </span>
        </div>

        {/* Model Architecture Switcher Plates */}
        <div
          style={{
            display: 'flex',
            gap: '0.25rem',
            background: 'rgba(0, 0, 0, 0.4)',
            borderRadius: '6px',
            padding: '2px',
            border: '1px solid rgba(197, 155, 39, 0.3)',
          }}
        >
          <button
            onClick={() => setSelectedAgentModel('ppo')}
            style={{
              background: state.selectedAgentModel === 'ppo'
                ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                : 'transparent',
              color: state.selectedAgentModel === 'ppo' ? '#23140C' : '#D4C09D',
              border: state.selectedAgentModel === 'ppo' ? '1px solid #6E4E04' : '1px solid transparent',
              borderRadius: '4px',
              padding: '0.2rem 0.6rem',
              fontSize: '0.72rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: state.selectedAgentModel === 'ppo' ? 800 : 600,
              cursor: 'pointer',
              boxShadow: state.selectedAgentModel === 'ppo' ? '0 1px 4px rgba(0,0,0,0.2)' : 'none',
            }}
          >
            PPO Masked AC
          </button>
          <button
            onClick={() => setSelectedAgentModel('dqn')}
            style={{
              background: state.selectedAgentModel === 'dqn'
                ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)'
                : 'transparent',
              color: state.selectedAgentModel === 'dqn' ? '#23140C' : '#D4C09D',
              border: state.selectedAgentModel === 'dqn' ? '1px solid #6E4E04' : '1px solid transparent',
              borderRadius: '4px',
              padding: '0.2rem 0.6rem',
              fontSize: '0.72rem',
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: state.selectedAgentModel === 'dqn' ? 800 : 600,
              cursor: 'pointer',
              boxShadow: state.selectedAgentModel === 'dqn' ? '0 1px 4px rgba(0,0,0,0.2)' : 'none',
            }}
          >
            DQN Q-Net
          </button>
        </div>
      </div>

      {brainData ? (
        <>
          {/* Critic Steam Pressure Manometer */}
          <ValueHeadGauge
            estimatedValue={brainData.estimated_value}
            modelType={brainData.model_type}
          />

          {/* Policy Actuator Distribution Chart with Synchronized Hover */}
          <ActionProbabilitiesChart
            probabilities={brainData.action_probabilities}
            actionMask={brainData.action_mask}
            actionLabels={brainData.action_labels}
            rawLogitsOrQ={brainData.masked_logits_or_q}
            greedyActionIndex={brainData.greedy_action_index}
            onActionHover={(meta) => setHoveredAction(meta)}
          />

          {/* Observation Tensor & Thermionic Filament Matrix */}
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
            background: 'rgba(232, 219, 190, 0.4)',
            borderRadius: '8px',
            border: '2px dashed #B8860B',
          }}
        >
          <Cpu size={36} color="#8C6305" style={{ margin: '0 auto 0.75rem auto', display: 'block' }} />
          <div
            style={{
              fontSize: '0.9rem',
              fontWeight: 800,
              color: '#23140C',
              fontFamily: "'Playfair Display', Georgia, serif",
              marginBottom: '0.25rem',
            }}
          >
            No Analytical Engine Connected
          </div>
          <div style={{ fontSize: '0.75rem', color: '#5A3822', fontFamily: "'Crimson Pro', Georgia, serif" }}>
            Initialize a match with an active PPO or DQN automaton to inspect real-time steam pressure logits, policy actuators, and thermionic tensor states.
          </div>
        </div>
      )}
    </div>
  );
});
