import React, { useState, useEffect, useMemo } from 'react';
import { api } from '../api/client';
import { BrainInspectionDTO } from '../api/types';
import { BoardSVG } from '../components/board/BoardSVG';
import { COLOR_HEX } from '../components/board/mapData';
import { Brain, RefreshCw } from 'lucide-react';

interface ScenarioPreset {
  id: string;
  name: string;
  description: string;
  icon: string;
  cards: Record<string, number>;
  trains: number;
  tickets: Array<{ city_a: string; city_b: string; points: number }>;
}

const PRESET_SCENARIOS: ScenarioPreset[] = [
  {
    id: 'early_setup',
    name: 'Fase Iniziale (Raccolta Risorse)',
    description: 'Il bot possiede 45 carrozze e sta accumulando locomotive e vagoni per impostare la strategia di lungo raggio.',
    icon: '🌱',
    cards: { RED: 2, BLUE: 1, LOCOMOTIVE: 1 },
    trains: 45,
    tickets: [{ city_a: 'New York', city_b: 'Boston', points: 4 }],
  },
  {
    id: 'route_claim',
    name: 'Opportunità di Conquista Tratta',
    description: 'Il bot ha accumulato carte sufficienti per occupare la tratta strategica New York ⟷ Boston.',
    icon: '🛤️',
    cards: { RED: 4, BLUE: 2, LOCOMOTIVE: 2 },
    trains: 38,
    tickets: [{ city_a: 'New York', city_b: 'Atlanta', points: 6 }],
  },
  {
    id: 'endgame_pressure',
    name: 'Finale di Partita (Pochi Treni)',
    description: 'Rimangono meno di 6 carrozze; il bot deve ottimizzare i punti finali per massimizzare la resa dei biglietti.',
    icon: '🏁',
    cards: { GREEN: 3, YELLOW: 2, LOCOMOTIVE: 1 },
    trains: 4,
    tickets: [{ city_a: 'Los Angeles', city_b: 'Seattle', points: 9 }],
  },
];

export const BrainView: React.FC = () => {
  const [inspection, setInspection] = useState<BrainInspectionDTO | null>(null);
  const [modelType, setModelType] = useState<'ppo' | 'dqn'>('ppo');
  const [selectedScenario, setSelectedScenario] = useState<string>('route_claim');
  const [showGuide, setShowGuide] = useState<boolean>(true);
  const [filterMode, setFilterMode] = useState<'valid' | 'top5' | 'all'>('valid');
  const [isLoading, setIsLoading] = useState(false);

  // Load inspection data
  useEffect(() => {
    setIsLoading(true);
    api.inspectBrain({ model_type: modelType })
      .then((data) => {
        setInspection(data);
      })
      .catch((err) => console.error('Brain inspection error:', err))
      .finally(() => setIsLoading(false));
  }, [modelType, selectedScenario]);

  const currentScenario = PRESET_SCENARIOS.find((s) => s.id === selectedScenario) || PRESET_SCENARIOS[0];

  // Prepared actions list with human readable interpretations
  const actionItems = useMemo(() => {
    if (!inspection) return [];
    return inspection.action_probabilities.map((prob, idx) => {
      const isValid = inspection.action_mask[idx] ?? true;
      const label = inspection.action_labels[idx] || `Azione ${idx}`;
      const isGreedy = idx === inspection.greedy_action_index;
      return {
        index: idx,
        label,
        prob,
        isValid,
        isGreedy,
      };
    });
  }, [inspection]);

  const filteredActions = useMemo(() => {
    if (filterMode === 'valid') return actionItems.filter((a) => a.isValid);
    if (filterMode === 'top5') return [...actionItems].sort((a, b) => b.prob - a.prob).slice(0, 5);
    return actionItems;
  }, [actionItems, filterMode]);

  const greedyAction = actionItems.find((a) => a.isGreedy);

  return (
    <div
      className="brain-agent-unified-view"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem',
        color: '#23140C',
        fontFamily: "'Crimson Pro', Georgia, serif",
      }}
    >
      {/* Interactive Guide Banner (Accordo e Spiegazione dell'Intelligenza Artificiale) */}
      <div
        className="steampunk-panel"
        style={{
          padding: '1.1rem 1.35rem',
          background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
          border: '2px solid #C59B27',
          boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: showGuide ? '1rem' : 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: '8px',
                background: 'linear-gradient(180deg, #F7E099 0%, #CBA232 50%, #996E08 100%)',
                border: '1.5px solid #6E4E04',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#23140C',
                boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
              }}
            >
              <Brain size={20} />
            </div>
            <div>
              <h3
                style={{
                  margin: 0,
                  fontSize: '1.1rem',
                  fontWeight: 900,
                  color: '#23140C',
                  fontFamily: "'Cinzel Decorative', Georgia, serif",
                }}
              >
                Come Ragiona l'Agente AI: Dallo Stato alla Decisione
              </h3>
              <p style={{ margin: '0.15rem 0 0 0', fontSize: '0.82rem', color: '#5A3822', fontStyle: 'italic' }}>
                Pipeline neurale end-to-end: Percezione (POMDP) ➔ Strati Convoluzionali/MLP ➔ Filtro Regole ➔ Decisione Finale
              </p>
            </div>
          </div>

          <button
            onClick={() => setShowGuide(!showGuide)}
            className="steampunk-btn"
            style={{
              padding: '0.3rem 0.75rem',
              fontSize: '0.78rem',
            }}
          >
            {showGuide ? 'Nascondi Guida ▲' : 'Mostra Guida ▼'}
          </button>
        </div>

        {showGuide && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem', marginTop: '0.75rem' }}>
            <div style={{ background: '#FAF5EB', padding: '0.75rem', borderRadius: '8px', border: '1.5px solid #C59B27', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
              <div style={{ fontSize: '0.82rem', fontWeight: 800, color: '#9E6B00', fontFamily: "'Playfair Display', serif", marginBottom: '0.25rem' }}>
                1. Percezione (POMDP)
              </div>
              <div style={{ fontSize: '0.76rem', color: '#4A2F1D', lineHeight: '1.4' }}>
                Il bot osserva la propria mano di carte, i treni rimasti, le 5 carte scoperte e i binari occupati sulla mappa.
              </div>
            </div>

            <div style={{ background: '#FAF5EB', padding: '0.75rem', borderRadius: '8px', border: '1.5px solid #C59B27', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
              <div style={{ fontSize: '0.82rem', fontWeight: 800, color: '#7E22CE', fontFamily: "'Playfair Display', serif", marginBottom: '0.25rem' }}>
                2. Rete Neurale
              </div>
              <div style={{ fontSize: '0.76rem', color: '#4A2F1D', lineHeight: '1.4' }}>
                I layer nascosti estraggono pattern strategici, calcolando correlazioni tra obiettivi e risorse disponibili.
              </div>
            </div>

            <div style={{ background: '#FAF5EB', padding: '0.75rem', borderRadius: '8px', border: '1.5px solid #C59B27', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
              <div style={{ fontSize: '0.82rem', fontWeight: 800, color: '#C2410C', fontFamily: "'Playfair Display', serif", marginBottom: '0.25rem' }}>
                3. Action Masking
              </div>
              <div style={{ fontSize: '0.76rem', color: '#4A2F1D', lineHeight: '1.4' }}>
                Le mosse non permesse dalle regole del gioco vengono bloccate matematicamente con probabilità zero (-∞ logit).
              </div>
            </div>

            <div style={{ background: '#FAF5EB', padding: '0.75rem', borderRadius: '8px', border: '1.5px solid #C59B27', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
              <div style={{ fontSize: '0.82rem', fontWeight: 800, color: '#15803D', fontFamily: "'Playfair Display', serif", marginBottom: '0.25rem' }}>
                4. Decisione & Valore
              </div>
              <div style={{ fontSize: '0.76rem', color: '#4A2F1D', lineHeight: '1.4' }}>
                La Policy π(a|s) stima la mossa migliore, mentre il Critic V(s) valuta le probabilità di vittoria futura.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Control Bar: Scenario & Model Selector */}
      <div
        className="steampunk-panel"
        style={{
          padding: '0.85rem 1.25rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
          background: 'linear-gradient(180deg, #FAF3E6 0%, #EADBBE 100%)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <label style={{ fontSize: '0.82rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>
              Architettura AI:
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
              <option value="ppo">PPO (Policy Gradient + Critic Value Head)</option>
              <option value="dqn">Double-DQN (Deep Q-Network)</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <label style={{ fontSize: '0.82rem', color: '#4A2F1D', fontWeight: 800, fontFamily: "'Playfair Display', serif" }}>
              Scenario da Testare:
            </label>
            <select
              value={selectedScenario}
              onChange={(e) => setSelectedScenario(e.target.value)}
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
              {PRESET_SCENARIOS.map((sc) => (
                <option key={sc.id} value={sc.id}>
                  {sc.icon} {sc.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <button
            onClick={() => {
              setIsLoading(true);
              api.inspectBrain({ model_type: modelType })
                .then(setInspection)
                .finally(() => setIsLoading(false));
            }}
            disabled={isLoading}
            className="steampunk-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              padding: '0.35rem 0.85rem',
              fontSize: '0.8rem',
            }}
          >
            <RefreshCw size={13} />
            <span>Aggiorna Ispezione</span>
          </button>
        </div>
      </div>

      {/* Main 3-Column Interactive Cockpit */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 320px) minmax(0, 1.3fr) minmax(300px, 340px)', gap: '1.25rem', alignItems: 'start' }}>
        {/* LEFT COLUMN: WHAT THE AI SEES (Percezione) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Board Topology Miniature */}
          <div
            className="steampunk-panel"
            style={{
              padding: '0.85rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h4 style={{ margin: 0, fontSize: '0.85rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                🗺️ Topologia Mappa USA
              </h4>
              <span style={{ fontSize: '0.72rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>36 Città</span>
            </div>
            <BoardSVG mapName="usa" />
          </div>

          {/* Hand Inventory */}
          <div
            className="steampunk-panel"
            style={{
              padding: '0.85rem',
            }}
          >
            <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.85rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
              🂠 Risorse in Mano al Bot
            </h4>
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.5rem', fontSize: '0.8rem', color: '#5A3822', fontFamily: "'Courier Prime', monospace" }}>
              <div>🚂 Treni: <strong style={{ color: '#23140C' }}>{currentScenario.trains}</strong></div>
              <div>🎫 Biglietti: <strong style={{ color: '#23140C' }}>{currentScenario.tickets.length}</strong></div>
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
              {Object.entries(currentScenario.cards).map(([color, count]) => {
                const hex = COLOR_HEX[color] || '#64748B';
                const isLight = color === 'WHITE' || color === 'YELLOW';
                return (
                  <div
                    key={color}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                      padding: '0.2rem 0.5rem',
                      borderRadius: '6px',
                      backgroundColor: hex,
                      color: isLight ? '#23140C' : '#FFFFFF',
                      fontSize: '0.75rem',
                      fontWeight: 800,
                      fontFamily: "'Courier Prime', monospace",
                      boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                      border: '1px solid rgba(0,0,0,0.2)',
                    }}
                  >
                    <span>{color}</span>
                    <span style={{ background: isLight ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.3)', padding: '0.1rem 0.3rem', borderRadius: '4px' }}>
                      {count}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Destination Tickets */}
          <div
            className="steampunk-panel"
            style={{
              padding: '0.85rem',
            }}
          >
            <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.85rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
              🎯 Obiettivi Biglietti
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              {currentScenario.tickets.map((t, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    background: '#FAF5EB',
                    border: '1px solid rgba(184, 134, 11, 0.3)',
                    padding: '0.4rem 0.6rem',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                  }}
                >
                  <span style={{ fontWeight: 600, color: '#23140C' }}>📍 {t.city_a} ⟷ {t.city_b}</span>
                  <strong style={{ color: '#B8860B', fontFamily: "'Courier Prime', monospace" }}>+{t.points} pts</strong>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* CENTER COLUMN: NEURAL PROCESSING & LAYER ACTIVATIONS (Elaborazione) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Critic Evaluation Banner */}
          <div
            className="steampunk-panel"
            style={{
              padding: '0.85rem 1.25rem',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              background: 'linear-gradient(180deg, #FBF6ED 0%, #EFE1C7 100%)',
              border: '2px solid #C59B27',
            }}
          >
            <div>
              <div style={{ fontSize: '0.78rem', color: '#785A42', fontWeight: 700, fontFamily: "'Playfair Display', serif" }}>
                Stima Valutazione Posizione (Critic Head)
              </div>
              <div style={{ fontSize: '1.4rem', fontWeight: 900, color: (inspection?.estimated_value || 0) >= 0 ? '#15803D' : '#B91C1C', fontFamily: "'Courier Prime', monospace" }}>
                V(s) = {inspection?.estimated_value !== undefined && inspection?.estimated_value !== null ? (inspection.estimated_value >= 0 ? `+${inspection.estimated_value.toFixed(2)}` : inspection.estimated_value.toFixed(2)) : '0.00'}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span style={{ fontSize: '0.74rem', color: '#785A42', fontFamily: "'Playfair Display', serif" }}>Giudizio Strategico:</span>
              <div style={{ fontSize: '0.88rem', fontWeight: 800, color: (inspection?.estimated_value || 0) >= 0 ? '#15803D' : '#B91C1C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                {(inspection?.estimated_value || 0) >= 0 ? '🟢 Posizione Vantaggiosa' : '🔴 Situazione Complessa'}
              </div>
            </div>
          </div>

          {/* Neural Layers Flow Graph */}
          <div
            className="steampunk-panel"
            style={{
              padding: '1rem',
            }}
          >
            <h4 style={{ margin: '0 0 0.85rem 0', fontSize: '0.9rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
              🧠 Flusso di Attivazione dei Neuroni
            </h4>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', overflowX: 'auto', paddingBottom: '0.5rem' }}>
              {/* Input State Column */}
              <div style={{ minWidth: '100px', background: '#FAF5EB', border: '1px solid #C59B27', borderRadius: '8px', padding: '0.6rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 800, color: '#9E6B00', fontFamily: "'Playfair Display', serif" }}>Stato Input</div>
                <div style={{ fontSize: '0.7rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>{inspection?.observation_vector.length || 100} dim</div>
                <div style={{ height: '60px', background: 'linear-gradient(to top, #1D4ED8, #0D9488)', borderRadius: '4px', margin: '0.4rem 0' }} />
                <div style={{ fontSize: '0.65rem', color: '#785A42', fontWeight: 700 }}>POMDP</div>
              </div>

              <div style={{ color: '#8C6305', fontWeight: 800 }}>➔</div>

              {/* Hidden Layers */}
              {(inspection?.layer_activations || []).map((layer, idx) => {
                const norm = Math.min(Math.max(layer.mean, 0.1), 1.0);
                return (
                  <React.Fragment key={idx}>
                    <div style={{ minWidth: '110px', background: '#FAF5EB', border: '1px solid #C59B27', borderRadius: '8px', padding: '0.6rem', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.75rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', serif" }}>
                        Layer {idx + 1}
                      </div>
                      <div style={{ fontSize: '0.65rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
                        [{layer.shape.join('×')}]
                      </div>

                      <div style={{ height: '60px', backgroundColor: '#2B1D14', borderRadius: '4px', margin: '0.4rem 0', position: 'relative', overflow: 'hidden', border: '1px solid #8C6305' }}>
                        <div
                          style={{
                            height: '100%',
                            width: '100%',
                            transform: `scaleY(${norm})`,
                            transformOrigin: 'bottom',
                            background: 'linear-gradient(to top, #15803D, #D97706)',
                            transition: 'transform 0.3s ease',
                          }}
                        />
                      </div>
                      <div style={{ fontSize: '0.7rem', color: '#9E6B00', fontWeight: 700, fontFamily: "'Courier Prime', monospace" }}>μ = {layer.mean.toFixed(2)}</div>
                    </div>
                    {idx < (inspection?.layer_activations?.length || 0) - 1 && <div style={{ color: '#8C6305', fontWeight: 800 }}>➔</div>}
                  </React.Fragment>
                );
              })}

              <div style={{ color: '#8C6305', fontWeight: 800 }}>➔</div>

              {/* Policy & Masking Output */}
              <div style={{ minWidth: '110px', background: '#FAF5EB', border: '1px solid #C59B27', borderRadius: '8px', padding: '0.6rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 800, color: '#15803D', fontFamily: "'Playfair Display', serif" }}>Policy Head</div>
                <div style={{ fontSize: '0.7rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>{inspection?.action_probabilities.length || 56} azioni</div>
                <div style={{ height: '60px', background: 'linear-gradient(to top, #15803D, #22C55E)', borderRadius: '4px', margin: '0.4rem 0' }} />
                <div style={{ fontSize: '0.65rem', color: '#785A42', fontWeight: 700 }}>Softmax Masked</div>
              </div>
            </div>
          </div>

          {/* Layer Activation Tensor Table */}
          <div
            className="steampunk-panel"
            style={{
              padding: '1rem',
            }}
          >
            <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.88rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
              📊 Statistiche Dettagliate dei Tensori di Attivazione
            </h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #C59B27', color: '#4A2F1D', fontFamily: "'Playfair Display', serif", fontWeight: 800 }}>
                    <th style={{ padding: '0.4rem' }}>Nome Layer</th>
                    <th style={{ padding: '0.4rem' }}>Forma Tensore</th>
                    <th style={{ padding: '0.4rem' }}>Media (μ)</th>
                    <th style={{ padding: '0.4rem' }}>Dev. Std (σ)</th>
                    <th style={{ padding: '0.4rem' }}>Min/Max</th>
                  </tr>
                </thead>
                <tbody>
                  {(inspection?.layer_activations || []).map((l, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(184, 134, 11, 0.2)', background: i % 2 === 1 ? 'rgba(246, 238, 223, 0.5)' : 'transparent' }}>
                      <td style={{ padding: '0.4rem', fontWeight: 700, color: '#23140C' }}>{l.layer_name}</td>
                      <td style={{ padding: '0.4rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>[{l.shape.join(', ')}]</td>
                      <td style={{ padding: '0.4rem', color: '#15803D', fontWeight: 700, fontFamily: "'Courier Prime', monospace" }}>{l.mean.toFixed(3)}</td>
                      <td style={{ padding: '0.4rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>{l.std.toFixed(3)}</td>
                      <td style={{ padding: '0.4rem', color: '#9E6B00', fontWeight: 700, fontFamily: "'Courier Prime', monospace" }}>[{l.min.toFixed(2)}, {l.max.toFixed(2)}]</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: WHAT THE AI DECIDES (Decisione Finale & Policy) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Greedy Choice Highlight Card */}
          <div
            className="steampunk-panel"
            style={{
              padding: '1rem',
              border: '2px solid #15803D',
              background: 'linear-gradient(180deg, #F5FBF6 0%, #E2F5E8 100%)',
              boxShadow: '0 4px 16px rgba(21, 128, 61, 0.2)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 800, color: '#15803D', textTransform: 'uppercase', letterSpacing: '0.05em', fontFamily: "'Playfair Display', serif" }}>
                🎯 Mossa Selezionata (Greedy)
              </span>
              <span style={{ fontSize: '0.95rem', fontWeight: 900, color: '#15803D', fontFamily: "'Courier Prime', monospace" }}>
                {greedyAction ? `${(greedyAction.prob * 100).toFixed(1)}%` : '--'}
              </span>
            </div>

            <div style={{ fontSize: '0.95rem', fontWeight: 800, color: '#23140C', marginBottom: '0.4rem', fontFamily: "'Playfair Display', Georgia, serif" }}>
              {greedyAction ? greedyAction.label : 'Calcolo in corso...'}
            </div>

            <p style={{ margin: 0, fontSize: '0.78rem', color: '#4A2F1D', lineHeight: '1.4' }}>
              {greedyAction?.label.includes('CLAIM_ROUTE')
                ? 'Il bot ha identificato un vantaggio strategico nell\'occupare questa tratta per connettere le città obiettivo del suo biglietto.'
                : greedyAction?.label.includes('DRAW_VISIBLE')
                ? 'Il bot ha scelto una carta visibile utile per completare la combinazione di colori richiesta dai suoi biglietti.'
                : 'Il bot ha scelto di pescare dal mazzo per accumulare risorse e ampliare le sue opzioni.'}
            </p>
          </div>

          {/* Action Distribution Bar List */}
          <div
            className="steampunk-panel"
            style={{
              padding: '1rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.3rem' }}>
              <h4 style={{ margin: 0, fontSize: '0.85rem', fontWeight: 800, color: '#23140C', fontFamily: "'Playfair Display', Georgia, serif" }}>
                📊 Distribuzione Policy π(a|s)
              </h4>
              <div style={{ display: 'flex', gap: '0.2rem' }}>
                {(['valid', 'top5', 'all'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setFilterMode(m)}
                    className="steampunk-btn"
                    style={{
                      background: filterMode === m ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 100%)' : '#FAF5EB',
                      color: '#23140C',
                      padding: '0.15rem 0.4rem',
                      fontSize: '0.7rem',
                    }}
                  >
                    {m === 'valid' ? 'Valide' : m === 'top5' ? 'Top 5' : 'Tutte'}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', maxHeight: '360px', overflowY: 'auto' }}>
              {filteredActions.map((item) => (
                <div
                  key={item.index}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.2rem',
                    background: item.isGreedy
                      ? 'linear-gradient(180deg, #FAF3E6 0%, #F5E5C9 100%)'
                      : item.isValid
                      ? '#FAF5EB'
                      : 'rgba(232, 219, 190, 0.4)',
                    border: item.isGreedy
                      ? '1.5px solid #B8860B'
                      : '1px solid rgba(184, 134, 11, 0.25)',
                    borderRadius: '6px',
                    padding: '0.35rem 0.55rem',
                    opacity: item.isValid ? 1 : 0.45,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem' }}>
                    <span style={{ color: '#23140C', fontWeight: item.isGreedy ? 800 : 600 }}>
                      {item.isValid ? (item.isGreedy ? '⭐ ' : '✓ ') : '🔒 '}
                      {item.label}
                    </span>
                    <strong style={{ color: item.isGreedy ? '#9E6B00' : item.isValid ? '#15803D' : '#991B1B', fontFamily: "'Courier Prime', monospace" }}>
                      {item.isValid ? `${(item.prob * 100).toFixed(1)}%` : 'Mascherata'}
                    </strong>
                  </div>

                  <div style={{ height: '7px', background: '#D8C3A0', borderRadius: '3px', overflow: 'hidden', border: '1px solid rgba(74, 47, 29, 0.25)' }}>
                    <div
                      style={{
                        height: '100%',
                        width: '100%',
                        transform: `scaleX(${item.prob})`,
                        transformOrigin: 'left',
                        background: item.isGreedy
                          ? 'linear-gradient(to right, #F6DC88, #C59B27, #8C6305)'
                          : item.isValid
                          ? 'linear-gradient(to right, #15803D, #22C55E)'
                          : '#991B1B',
                        transition: 'transform 0.2s ease',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
