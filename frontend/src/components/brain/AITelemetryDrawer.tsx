import React from 'react';
import { BrainInspectorPane } from './BrainInspectorPane';
import { Brain, X } from 'lucide-react';

interface AITelemetryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AITelemetryDrawer: React.FC<AITelemetryDrawerProps> = ({
  isOpen,
  onClose,
}) => {
  if (!isOpen) return null;

  return (
    <div
      className="ai-telemetry-drawer-backdrop"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(26, 16, 10, 0.65)',
        backdropFilter: 'blur(3px)',
        zIndex: 110,
        display: 'flex',
        justifyContent: 'flex-end',
      }}
    >
      <div
        className="ai-telemetry-drawer steampunk-panel"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '460px',
          maxWidth: '92vw',
          height: '100vh',
          background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
          borderLeft: '3px solid #C59B27',
          boxShadow: '-8px 0 32px rgba(0, 0, 0, 0.5)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          animation: 'slideInRight 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {/* Drawer Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.85rem 1.25rem',
            background: 'linear-gradient(180deg, #3A261A 0%, #26180F 100%)',
            borderBottom: '2px solid #C59B27',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Brain size={18} color="#F6DC88" />
            <h3
              style={{
                margin: 0,
                fontSize: '1rem',
                color: '#FAF5EB',
                fontFamily: "'Cinzel Decorative', Georgia, serif",
                letterSpacing: '0.03em',
              }}
            >
              Telemetria AI
            </h3>
          </div>

          <button
            onClick={onClose}
            aria-label="Chiudi Telemetria"
            title="Chiudi Drawer"
            className="steampunk-btn"
            style={{
              padding: '0.25rem 0.6rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '0.8rem',
            }}
          >
            <X size={14} />
          </button>
        </div>

        {/* Scrollable Brain Content */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
          }}
        >
          <BrainInspectorPane />
        </div>
      </div>
    </div>
  );
};
