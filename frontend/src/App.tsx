import React, { useState } from 'react';

type ViewMode = 'game' | 'agent' | 'training' | 'brain' | 'replay' | 'experiments';

export default function App() {
  const [activeView, setActiveView] = useState<ViewMode>('game');

  return (
    <div className="app-container">
      <header>
        <div className="brand">
          <span>🎫</span>
          <span>TicketToRide RL Lab</span>
        </div>
        <nav className="nav-tabs">
          {(['game', 'agent', 'training', 'brain', 'replay', 'experiments'] as ViewMode[]).map((tab) => (
            <button
              key={tab}
              className={`tab-btn ${activeView === tab ? 'active' : ''}`}
              onClick={() => setActiveView(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>
      </header>

      <main>
        <div className="card">
          <h2>Laboratory View: {activeView.toUpperCase()}</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
            Phase 0 Skeleton active. Connects to backend API and WebSocket telemetry in Phase 5.
          </p>
        </div>

        <div className="grid-2">
          <div className="card">
            <h3>Environment Status</h3>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem', marginTop: '0.5rem', color: 'var(--text-secondary)' }}>
              Deterministic Engine: Ready (Phase 0)<br />
              Backend: FastAPI + WebSockets<br />
              Action Space: Discrete Masked<br />
              Seed: 42
            </p>
          </div>

          <div className="card">
            <h3>RL Research Metrics</h3>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem', marginTop: '0.5rem', color: 'var(--text-secondary)' }}>
              Episodes: 0<br />
              Win Rate: --<br />
              Policy Loss: --<br />
              Value Loss: --
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
