import { useState } from 'react';
import {
  GameView,
  BrainView,
  TrainingView,
  ExperimentView,
  ReplayView,
} from './views';

type ViewMode = 'game' | 'brain' | 'training' | 'experiments' | 'replay';

interface TabItem {
  id: ViewMode;
  label: string;
  icon: string;
}

const TABS: TabItem[] = [
  { id: 'game', label: 'Interactive Game', icon: '🎮' },
  { id: 'brain', label: 'Neural Brain & Agent', icon: '🧠' },
  { id: 'training', label: 'Training Live', icon: '📈' },
  { id: 'experiments', label: 'Experiments', icon: '🧪' },
  { id: 'replay', label: 'Replay Player', icon: '🎞️' },
];

export default function App() {
  const [activeView, setActiveView] = useState<ViewMode>('game');

  return (
    <div className="app-container" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header
        style={{
          background: 'rgba(15, 23, 42, 0.95)',
          backdropFilter: 'blur(12px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          padding: '0.75rem 1.5rem',
          position: 'sticky',
          top: 0,
          zIndex: 50,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div
            style={{
              width: 38,
              height: 38,
              borderRadius: '10px',
              background: 'linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '1.25rem',
              boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)',
            }}
          >
            🚂
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 800, color: '#F8FAFC', letterSpacing: '-0.02em' }}>
              Ticket to Ride <span style={{ color: '#38BDF8' }}>RL Lab</span>
            </h1>
            <span style={{ fontSize: '0.7rem', color: '#94A3B8', fontWeight: 500 }}>
              Autonomous Multi-Agent RL & Neural Introspection Hub
            </span>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav
          style={{
            display: 'flex',
            background: 'rgba(30, 41, 59, 0.7)',
            padding: '0.25rem',
            borderRadius: '10px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            gap: '0.25rem',
          }}
        >
          {TABS.map((tab) => {
            const isActive = activeView === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveView(tab.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  padding: '0.4rem 0.85rem',
                  borderRadius: '8px',
                  border: 'none',
                  background: isActive ? '#3B82F6' : 'transparent',
                  color: isActive ? '#FFFFFF' : '#94A3B8',
                  fontSize: '0.85rem',
                  fontWeight: isActive ? 700 : 500,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: isActive ? '0 2px 8px rgba(59, 130, 246, 0.4)' : 'none',
                }}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </header>

      {/* Main View Area */}
      <main style={{ flex: 1, padding: '1.5rem', maxWidth: '1600px', width: '100%', margin: '0 auto', boxSizing: 'border-box' }}>
        {activeView === 'game' && <GameView />}
        {activeView === 'brain' && <BrainView />}
        {activeView === 'training' && <TrainingView />}
        {activeView === 'experiments' && <ExperimentView />}
        {activeView === 'replay' && <ReplayView />}
      </main>
    </div>
  );
}
