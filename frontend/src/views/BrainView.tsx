import React, { useState, useEffect, useMemo } from 'react';
import { api } from '../api/client';
import { BrainInspectionDTO } from '../api/types';
import { BoardSVG } from '../components/board/BoardSVG';
import { COLOR_HEX } from '../components/board/mapData';

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
    name: 'Fase Iniziale (Raccolta Carte)',
    description: 'Il bot ha 45 treni e sta raccogliendo carte e valutando i biglietti.',
    icon: '🌱',
    cards: { RED: 2, BLUE: 1, LOCOMOTIVE: 1 },
    trains: 45,
    tickets: [{ city_a: 'New York', city_b: 'Boston', points: 4 }],
  },
  {
    id: 'route_claim',
    name: 'Opportunità di Conquista Tratta',
    description: 'Il bot ha accumulato carte sufficienti per occupare la tratta New York ⟷ Boston.',
    icon: '🛤️',
    cards: { RED: 4, BLUE: 2, LOCOMOTIVE: 2 },
    trains: 38,
    tickets: [{ city_a: 'New York', city_b: 'Atlanta', points: 6 }],
  },
  {
    id: 'endgame_pressure',
    name: 'Finale di Partita (Pochi Treni)',
    description: 'Rimangono meno di 6 treni; il bot deve ottimizzare i punti finali.',
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
    <div className="brain-agent-unified-view" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Interactive Guide Banner (Accordo e Spiegazione dell'Intelligenza Artificiale) */}
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%)',
          border: '1px solid rgba(56, 189, 248, 0.25)',
          borderRadius: '12px',
          padding: '1.25rem',
          boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: showGuide ? '1rem' : 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <span style={{ fontSize: '1.4rem' }}>🧠</span>
            <div>
              <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700, color: '#F8FAFC' }}>
                Come Ragiona l'Agente AI: Dallo Stato alla Decisione
              </h3>
              <p style={{ margin: 0, fontSize: '0.8rem', color: '#94A3B8' }}>
                Processo decisionale neurale end-to-end: Percezione (POMDP) ➔ Strati Convoluzionali/MLP ➔ Filtro Regole ➔ Decisione Finale
              </p>
            </div>
          </div>

          <button
            onClick={() => setShowGuide(!showGuide)}
            style={{
              background: 'rgba(56, 189, 248, 0.15)',
              border: '1px solid rgba(56, 189, 248, 0.4)',
              borderRadius: '6px',
              padding: '0.3rem 0.75rem',
              color: '#38BDF8',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            {showGuide ? 'Nascondi Guida ▲' : 'Mostra Guida ▼'}
          </button>
        </div>

        {showGuide && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem', marginTop: '0.75rem' }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '0.75rem', borderRadius: '8px', border: '1px solid rgba(56, 189, 248, 0.2)' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#38BDF8', marginBottom: '0.25rem' }}>
                1. Percezione (POMDP)
              </div>
              <div style={{ fontSize: '0.75rem', color: '#CBD5E1', lineHeight: '1.3' }}>
                Il bot osserva la propria mano di carte, i treni rimasti, le 5 carte scoperte e i binari occupati sulla mappa.
              </div>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '0.75rem', borderRadius: '8px', border: '1px solid rgba(139, 92, 246, 0.2)' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#A78BFA', marginBottom: '0.25rem' }}>
                2. Rete Neurale
              </div>
              <div style={{ fontSize: '0.75rem', color: '#CBD5E1', lineHeight: '1.3' }}>
                I layer nascosti estraggono pattern strategici, calcolando correlazioni tra obiettivi e risorse disponibili.
              </div>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '0.75rem', borderRadius: '8px', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#FBBF24', marginBottom: '0.25rem' }}>
                3. Action Masking
              </div>
              <div style={{ fontSize: '0.75rem', color: '#CBD5E1', lineHeight: '1.3' }}>
                Le mosse non permesse dalle regole del gioco vengono bloccate matematicamente con probabilità zero ($-\infty$).
              </div>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '0.75rem', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#34D399', marginBottom: '0.25rem' }}>
                4. Decisione & Valore
              </div>
              <div style={{ fontSize: '0.75rem', color: '#CBD5E1', lineHeight: '1.3' }}>
                La Policy $\pi(a|s)$ stima la mossa migliore, mentre il Critic $V(s)$ valuta le probabilità di vittoria futura.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Control Bar: Scenario & Model Selector */}
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.9)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '12px',
          padding: '0.9rem 1.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Architettura AI:</label>
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
                fontWeight: 600,
              }}
            >
              <option value="ppo">PPO (Policy Gradient + Critic Value Head)</option>
              <option value="dqn">Double-DQN (Deep Q-Network)</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label style={{ fontSize: '0.85rem', color: '#94A3B8', fontWeight: 600 }}>Scenario da Testare:</label>
            <select
              value={selectedScenario}
              onChange={(e) => setSelectedScenario(e.target.value)}
              style={{
                backgroundColor: '#1E293B',
                color: '#F1F5F9',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '6px',
                padding: '0.35rem 0.6rem',
                fontSize: '0.85rem',
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

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
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
              padding: '0.4rem 0.9rem',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
            }}
          >
            🔄 Aggiorna Ispezione
          </button>
        </div>
      </div>

      {/* Main 3-Column Interactive Cockpit */}
      <div style={{ display: 'grid', gridTemplateColumns: '320px minmax(0, 1fr) 340px', gap: '1.25rem' }}>
        {/* LEFT COLUMN: WHAT THE AI SEES (Percezione) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Board Topology Miniature */}
          <div
            style={{
              background: 'rgba(15, 23, 42, 0.9)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '12px',
              padding: '1rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h4 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700, color: '#38BDF8' }}>
                🗺️ Topologia Mappa USA
              </h4>
              <span style={{ fontSize: '0.7rem', color: '#94A3B8' }}>36 Città</span>
            </div>
            <BoardSVG mapName="usa" />
          </div>

          {/* Hand Inventory */}
          <div
            style={{
              background: 'rgba(15, 23, 42, 0.9)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '12px',
              padding: '1rem',
            }}
          >
            <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem', fontWeight: 700, color: '#F1F5F9' }}>
              🂠 Risorse in Mano al Bot
            </h4>
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.5rem', fontSize: '0.8rem', color: '#94A3B8' }}>
              <div>🚂 Treni: <strong style={{ color: '#F1F5F9' }}>{currentScenario.trains}</strong></div>
              <div>🎫 Biglietti: <strong style={{ color: '#F1F5F9' }}>{currentScenario.tickets.length}</strong></div>
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
                      color: isLight ? '#0F172A' : '#FFFFFF',
                      fontSize: '0.75rem',
                      fontWeight: 700,
                    }}
                  >
                    <span>{color}</span>
                    <span style={{ background: isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.3)', padding: '0.1rem 0.3rem', borderRadius: '4px' }}>
                      {count}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Destination Tickets */}
          <div
            style={{
              background: 'rgba(15, 23, 42, 0.9)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '12px',
              padding: '1rem',
            }}
          >
            <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem', fontWeight: 700, color: '#F1F5F9' }}>
              🎯 Obiettivi Biglietti
            </h4>
            {currentScenario.tickets.map((t, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  background: 'rgba(30, 41, 59, 0.6)',
                  padding: '0.4rem 0.6rem',
                  borderRadius: '6px',
                  fontSize: '0.8rem',
                }}
              >
                <span>📍 {t.city_a} ⟷ {t.city_b}</span>
                <strong style={{ color: '#F59E0B' }}>+{t.points} pts</strong>
              </div>
            ))}
          </div>
        </div>

        {/* CENTER COLUMN: NEURAL PROCESSING & LAYER ACTIVATIONS (Elaborazione) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Critic Evaluation Banner */}
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%)',
              border: '1px solid #38BDF8',
              borderRadius: '12px',
              padding: '1rem 1.25rem',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div>
              <div style={{ fontSize: '0.8rem', color: '#94A3B8', fontWeight: 600 }}>Stima Valutazione Posizione (Critic Head)</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#38BDF8' }}>
                V(s) = {inspection?.estimated_value !== undefined && inspection?.estimated_value !== null ? inspection.estimated_value.toFixed(2) : '0.00'}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Giudizio Strategico:</span>
              <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#10B981' }}>
                {(inspection?.estimated_value || 0) >= 0 ? '🟢 Posizione Vantaggiosa' : '🔴 Situazione Complessa'}
              </div>
            </div>
          </div>

          {/* Neural Layers Flow Graph */}
          <div
            style={{
              background: 'rgba(15, 23, 42, 0.9)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '12px',
              padding: '1.25rem',
            }}
          >
            <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.95rem', fontWeight: 700, color: '#F8FAFC' }}>
              🧠 Flusso di Attivazione dei Neuroni
            </h4>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', overflowX: 'auto', paddingBottom: '0.5rem' }}>
              {/* Input State Column */}
              <div style={{ minWidth: '100px', background: 'rgba(30, 41, 59, 0.6)', borderRadius: '8px', padding: '0.6rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#38BDF8' }}>Stato Input</div>
                <div style={{ fontSize: '0.7rem', color: '#94A3B8' }}>{inspection?.observation_vector.length || 100} dim</div>
                <div style={{ height: '60px', background: 'linear-gradient(to top, #3B82F6, #06B6D4)', borderRadius: '4px', margin: '0.4rem 0' }} />
                <div style={{ fontSize: '0.65rem', color: '#64748B' }}>POMDP</div>
              </div>

              <div style={{ color: '#64748B' }}>➔</div>

              {/* Hidden Layers */}
              {inspection?.layer_activations.map((layer, idx) => {
                const norm = Math.min(Math.max(layer.mean, 0.1), 1.0);
                return (
                  <React.Fragment key={idx}>
                    <div style={{ minWidth: '110px', background: 'rgba(30, 41, 59, 0.6)', borderRadius: '8px', padding: '0.6rem', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#F1F5F9' }}>
                        Layer {idx + 1}
                      </div>
                      <div style={{ fontSize: '0.65rem', color: '#64748B' }}>
                        [{layer.shape.join('×')}]
                      </div>

                      <div style={{ height: '60px', backgroundColor: '#0B1120', borderRadius: '4px', margin: '0.4rem 0', position: 'relative', overflow: 'hidden' }}>
                        <div
                          style={{
                            height: '100%',
                            width: '100%',
                            transform: `scaleY(${norm})`,
                            transformOrigin: 'bottom',
                            background: 'linear-gradient(to top, #10B981, #F59E0B)',
                            transition: 'transform 0.3s ease',
                          }}
                        />
                      </div>
                      <div style={{ fontSize: '0.7rem', color: '#38BDF8', fontWeight: 600 }}>μ = {layer.mean.toFixed(2)}</div>
                    </div>
                    {idx < inspection.layer_activations.length - 1 && <div style={{ color: '#64748B' }}>➔</div>}
                  </React.Fragment>
                );
              })}

              <div style={{ color: '#64748B' }}>➔</div>

              {/* Policy & Masking Output */}
              <div style={{ minWidth: '110px', background: 'rgba(30, 41, 59, 0.6)', borderRadius: '8px', padding: '0.6rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#10B981' }}>Policy Head</div>
                <div style={{ fontSize: '0.7rem', color: '#94A3B8' }}>{inspection?.action_probabilities.length || 56} azioni</div>
                <div style={{ height: '60px', background: 'linear-gradient(to top, #10B981, #34D399)', borderRadius: '4px', margin: '0.4rem 0' }} />
                <div style={{ fontSize: '0.65rem', color: '#64748B' }}>Softmax Masked</div>
              </div>
            </div>
          </div>

          {/* Layer Activation Tensor Table */}
          <div
            style={{
              background: 'rgba(15, 23, 42, 0.9)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '12px',
              padding: '1rem',
            }}
          >
            <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.9rem', fontWeight: 700, color: '#F1F5F9' }}>
              📊 Statistiche Dettagliate dei Tensori di Attivazione
            </h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: '#94A3B8' }}>
                    <th style={{ padding: '0.4rem' }}>Nome Layer</th>
                    <th style={{ padding: '0.4rem' }}>Forma Tensore</th>
                    <th style={{ padding: '0.4rem' }}>Media (μ)</th>
                    <th style={{ padding: '0.4rem' }}>Dev. Std (σ)</th>
                    <th style={{ padding: '0.4rem' }}>Min/Max</th>
                  </tr>
                </thead>
                <tbody>
                  {inspection?.layer_activations.map((l, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '0.4rem', fontWeight: 600, color: '#38BDF8' }}>{l.layer_name}</td>
                      <td style={{ padding: '0.4rem', color: '#94A3B8' }}>[{l.shape.join(', ')}]</td>
                      <td style={{ padding: '0.4rem', color: '#F1F5F9' }}>{l.mean.toFixed(3)}</td>
                      <td style={{ padding: '0.4rem', color: '#94A3B8' }}>{l.std.toFixed(3)}</td>
                      <td style={{ padding: '0.4rem', color: '#F59E0B' }}>[{l.min.toFixed(2)}, {l.max.toFixed(2)}]</td>
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
            style={{
              background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(15, 23, 42, 0.9) 100%)',
              border: '2px solid #10B981',
              borderRadius: '12px',
              padding: '1.25rem',
              boxShadow: '0 4px 16px rgba(16, 185, 129, 0.2)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#34D399', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                🎯 Mossa Selezionata (Greedy)
              </span>
              <span style={{ fontSize: '0.9rem', fontWeight: 800, color: '#34D399' }}>
                {greedyAction ? `${(greedyAction.prob * 100).toFixed(1)}%` : '--'}
              </span>
            </div>

            <div style={{ fontSize: '1rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '0.4rem' }}>
              {greedyAction ? greedyAction.label : 'Calcolo in corso...'}
            </div>

            <p style={{ margin: 0, fontSize: '0.75rem', color: '#94A3B8', lineHeight: '1.4' }}>
              {greedyAction?.label.includes('CLAIM_ROUTE')
                ? 'Il bot ha identificato un vantaggio strategico nell\'occupare questa tratta per connettere le città obiettivo del suo biglietto.'
                : greedyAction?.label.includes('DRAW_VISIBLE')
                ? 'Il bot ha scelto una carta visibile utile per completare la combinazione di colori richiesta dai suoi biglietti.'
                : 'Il bot ha scelto di pescare dal mazzo per accumulare risorse e ampliare le sue opzioni.'}
            </p>
          </div>

          {/* Action Distribution Bar List */}
          <div
            style={{
              background: 'rgba(15, 23, 42, 0.9)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '12px',
              padding: '1.25rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h4 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700, color: '#F1F5F9' }}>
                📊 Distribuzione Policy π(a|s)
              </h4>
              <div style={{ display: 'flex', gap: '0.2rem' }}>
                {(['valid', 'top5', 'all'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setFilterMode(m)}
                    style={{
                      background: filterMode === m ? '#3B82F6' : 'transparent',
                      color: filterMode === m ? '#FFFFFF' : '#94A3B8',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '4px',
                      padding: '0.15rem 0.4rem',
                      fontSize: '0.7rem',
                      cursor: 'pointer',
                    }}
                  >
                    {m === 'valid' ? 'Valide' : m === 'top5' ? 'Top 5' : 'Tutte'}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '360px', overflowY: 'auto' }}>
              {filteredActions.map((item) => (
                <div
                  key={item.index}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.2rem',
                    background: item.isGreedy
                      ? 'rgba(16, 185, 129, 0.15)'
                      : item.isValid
                      ? 'rgba(30, 41, 59, 0.5)'
                      : 'rgba(15, 23, 42, 0.3)',
                    border: item.isGreedy
                      ? '1px solid #10B981'
                      : '1px solid rgba(255,255,255,0.05)',
                    borderRadius: '6px',
                    padding: '0.4rem 0.6rem',
                    opacity: item.isValid ? 1 : 0.4,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                    <span style={{ color: item.isGreedy ? '#34D399' : '#F1F5F9', fontWeight: item.isGreedy ? 700 : 500 }}>
                      {item.isValid ? (item.isGreedy ? '⭐ ' : '✓ ') : '🔒 '}
                      {item.label}
                    </span>
                    <strong style={{ color: item.isGreedy ? '#34D399' : item.isValid ? '#38BDF8' : '#64748B' }}>
                      {item.isValid ? `${(item.prob * 100).toFixed(1)}%` : 'Mascherata'}
                    </strong>
                  </div>

                  <div style={{ height: '6px', background: 'rgba(15, 23, 42, 0.8)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div
                      style={{
                        height: '100%',
                        width: '100%',
                        transform: `scaleX(${item.prob})`,
                        transformOrigin: 'left',
                        background: item.isGreedy ? '#10B981' : item.isValid ? '#3B82F6' : '#475569',
                        transition: 'transform 0.3s ease',
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
