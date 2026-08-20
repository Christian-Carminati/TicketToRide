import React, { useState, useEffect, useMemo } from 'react';
import { api } from '../api/client';
import { ReportItemDTO, ReportDetailDTO } from '../api/types';
import {
  FileText,
  FileCode,
  RefreshCw,
  Trash2,
  Copy,
  Check,
  Search,
  Download,
  BookOpen,
} from 'lucide-react';

export const ReportsView: React.FC = () => {
  const [reports, setReports] = useState<ReportItemDTO[]>([]);
  const [selectedFilename, setSelectedFilename] = useState<string | null>(null);
  const [reportDetail, setReportDetail] = useState<ReportDetailDTO | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDetailLoading, setIsDetailLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [phaseFilter, setPhaseFilter] = useState<string>('all');
  const [viewMode, setViewMode] = useState<'rendered' | 'raw'>('rendered');
  const [copied, setCopied] = useState(false);
  const [actionMessage, setActionMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const fetchReports = async () => {
    setIsLoading(true);
    try {
      const list = await api.listReports();
      setReports(list);
      if (list.length > 0) {
        if (!selectedFilename || !list.some((r) => r.filename === selectedFilename)) {
          setSelectedFilename(list[0].filename);
        }
      } else {
        setSelectedFilename(null);
        setReportDetail(null);
      }
    } catch (err) {
      console.error(err);
      setActionMessage({ type: 'error', text: 'Impossibile caricare la lista dei report.' });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, []);

  useEffect(() => {
    if (!selectedFilename) return;
    setIsDetailLoading(true);
    api.getReport(selectedFilename)
      .then((detail) => {
        setReportDetail(detail);
      })
      .catch((err) => {
        console.error(err);
        setActionMessage({ type: 'error', text: `Errore durante il caricamento del report ${selectedFilename}.` });
      })
      .finally(() => setIsDetailLoading(false));
  }, [selectedFilename]);

  const handleDelete = async (filename: string) => {
    if (!window.confirm(`Eliminare il report "${filename}"?`)) return;
    try {
      setIsLoading(true);
      await api.deleteReport(filename);
      setActionMessage({ type: 'success', text: `Report "${filename}" eliminato.` });
      const nextList = reports.filter((r) => r.filename !== filename);
      setReports(nextList);
      if (selectedFilename === filename) {
        setSelectedFilename(nextList.length > 0 ? nextList[0].filename : null);
      }
    } catch (err) {
      console.error(err);
      setActionMessage({ type: 'error', text: `Impossibile eliminare "${filename}".` });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    if (!reportDetail?.raw_content) return;
    navigator.clipboard.writeText(reportDetail.raw_content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!reportDetail) return;
    const blob = new Blob([reportDetail.raw_content], {
      type: reportDetail.file_type === 'json' ? 'application/json' : 'text/markdown',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = reportDetail.filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filtered reports
  const filteredReports = useMemo(() => {
    return reports.filter((r) => {
      const matchSearch =
        r.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        r.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (r.phase && r.phase.toLowerCase().includes(searchTerm.toLowerCase()));
      const matchPhase = phaseFilter === 'all' || r.phase === phaseFilter;
      return matchSearch && matchPhase;
    });
  }, [reports, searchTerm, phaseFilter]);

  const uniquePhases = useMemo(() => {
    const set = new Set<string>();
    reports.forEach((r) => {
      if (r.phase) set.add(r.phase);
    });
    return Array.from(set);
  }, [reports]);

  // Zero-dependency Markdown to styled JSX parser
  const renderMarkdown = (text: string) => {
    const lines = text.split('\n');
    const elements: React.ReactNode[] = [];
    let inTable = false;
    let tableRows: string[][] = [];
    let inCodeBlock = false;
    let codeContent: string[] = [];

    const flushTable = (key: number) => {
      if (tableRows.length === 0) return null;
      const headers = tableRows[0];
      const dataRows = tableRows.slice(1).filter((r) => !r.every((c) => c.trim().match(/^:?-+:?$/)));

      const el = (
        <div key={`table-${key}`} style={{ overflowX: 'auto', margin: '1rem 0' }}>
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '0.85rem',
              backgroundColor: 'rgba(15, 23, 42, 0.6)',
              borderRadius: '8px',
              overflow: 'hidden',
              border: '1px solid rgba(255, 255, 255, 0.1)',
            }}
          >
            <thead>
              <tr style={{ background: 'rgba(30, 41, 59, 0.9)', borderBottom: '2px solid rgba(56, 189, 248, 0.3)' }}>
                {headers.map((h, i) => (
                  <th
                    key={i}
                    style={{
                      padding: '0.65rem 0.85rem',
                      color: '#38BDF8',
                      fontWeight: 700,
                      textAlign: i === 0 ? 'left' : 'center',
                    }}
                  >
                    {renderInlineMarkdown(h.trim())}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dataRows.map((row, ri) => (
                <tr
                  key={ri}
                  style={{
                    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                    background: ri % 2 === 1 ? 'rgba(30, 41, 59, 0.25)' : 'transparent',
                  }}
                >
                  {row.map((cell, ci) => {
                    const cleanCell = cell.trim();
                    const isWinRate = cleanCell.endsWith('%') && !isNaN(parseFloat(cleanCell));
                    const isPositiveDiff = cleanCell.startsWith('+');
                    const isNegativeDiff = cleanCell.startsWith('-');

                    let cellColor = '#E2E8F0';
                    if (isWinRate && parseFloat(cleanCell) >= 50) cellColor = '#34D399';
                    else if (isPositiveDiff) cellColor = '#38BDF8';
                    else if (isNegativeDiff) cellColor = '#F87171';

                    return (
                      <td
                        key={ci}
                        style={{
                          padding: '0.55rem 0.85rem',
                          textAlign: ci === 0 ? 'left' : 'center',
                          color: cellColor,
                          fontWeight: ci === 0 ? 600 : 500,
                          fontFamily: ci > 0 ? 'monospace' : 'inherit',
                        }}
                      >
                        {renderInlineMarkdown(cleanCell)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      tableRows = [];
      inTable = false;
      return el;
    };

    lines.forEach((line, idx) => {
      const trimmed = line.trim();

      // Code blocks
      if (trimmed.startsWith('```')) {
        if (inCodeBlock) {
          elements.push(
            <pre
              key={`code-${idx}`}
              style={{
                background: '#0B1120',
                padding: '0.85rem',
                borderRadius: '8px',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                color: '#CBD5E1',
                fontSize: '0.8rem',
                overflowX: 'auto',
                margin: '0.75rem 0',
                fontFamily: 'monospace',
              }}
            >
              <code>{codeContent.join('\n')}</code>
            </pre>
          );
          codeContent = [];
          inCodeBlock = false;
        } else {
          if (inTable) {
            const tbl = flushTable(idx);
            if (tbl) elements.push(tbl);
          }
          inCodeBlock = true;
        }
        return;
      }

      if (inCodeBlock) {
        codeContent.push(line);
        return;
      }

      // Tables
      if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
        inTable = true;
        const cols = trimmed.slice(1, -1).split('|');
        tableRows.push(cols);
        return;
      } else if (inTable) {
        const tbl = flushTable(idx);
        if (tbl) elements.push(tbl);
      }

      // Headings
      if (trimmed.startsWith('# ')) {
        elements.push(
          <h2
            key={idx}
            style={{
              fontSize: '1.4rem',
              fontWeight: 800,
              color: '#F8FAFC',
              margin: '1.25rem 0 0.5rem 0',
              borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
              paddingBottom: '0.4rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <BookOpen size={20} color="#38BDF8" />
            {trimmed.slice(2)}
          </h2>
        );
        return;
      }
      if (trimmed.startsWith('## ')) {
        elements.push(
          <h3
            key={idx}
            style={{
              fontSize: '1.15rem',
              fontWeight: 700,
              color: '#38BDF8',
              margin: '1.2rem 0 0.4rem 0',
            }}
          >
            {trimmed.slice(3)}
          </h3>
        );
        return;
      }
      if (trimmed.startsWith('### ')) {
        elements.push(
          <h4
            key={idx}
            style={{
              fontSize: '0.98rem',
              fontWeight: 600,
              color: '#A78BFA',
              margin: '1rem 0 0.3rem 0',
            }}
          >
            {trimmed.slice(4)}
          </h4>
        );
        return;
      }

      // Blockquotes
      if (trimmed.startsWith('> ')) {
        elements.push(
          <div
            key={idx}
            style={{
              background: 'rgba(56, 189, 248, 0.08)',
              borderLeft: '4px solid #38BDF8',
              padding: '0.5rem 0.85rem',
              borderRadius: '0 6px 6px 0',
              margin: '0.6rem 0',
              color: '#CBD5E1',
              fontSize: '0.85rem',
            }}
          >
            {renderInlineMarkdown(trimmed.slice(2))}
          </div>
        );
        return;
      }

      // Unordered Lists
      if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
        elements.push(
          <div
            key={idx}
            style={{
              display: 'flex',
              alignItems: 'start',
              gap: '0.5rem',
              margin: '0.25rem 0',
              fontSize: '0.85rem',
              color: '#E2E8F0',
            }}
          >
            <span style={{ color: '#38BDF8', marginTop: '0.1rem' }}>•</span>
            <span>{renderInlineMarkdown(trimmed.slice(2))}</span>
          </div>
        );
        return;
      }

      // Horizontal Rule
      if (trimmed === '---' || trimmed === '***') {
        elements.push(
          <hr
            key={idx}
            style={{
              border: 'none',
              borderTop: '1px solid rgba(255, 255, 255, 0.08)',
              margin: '1.25rem 0',
            }}
          />
        );
        return;
      }

      // Paragraph
      if (trimmed.length > 0) {
        elements.push(
          <p
            key={idx}
            style={{
              margin: '0.4rem 0',
              fontSize: '0.85rem',
              lineHeight: 1.55,
              color: '#CBD5E1',
            }}
          >
            {renderInlineMarkdown(trimmed)}
          </p>
        );
      }
    });

    if (inTable) {
      const tbl = flushTable(lines.length);
      if (tbl) elements.push(tbl);
    }

    return elements;
  };

  // Inline formatting helper: **bold**, `code`, *italic*
  const renderInlineMarkdown = (text: string): React.ReactNode => {
    const parts = text.split(/(\*\*.*?\*\*|`.*?`|\*.*?\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={i} style={{ color: '#F1F5F9', fontWeight: 700 }}>
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code
            key={i}
            style={{
              backgroundColor: 'rgba(30, 41, 59, 0.8)',
              color: '#38BDF8',
              padding: '0.1rem 0.35rem',
              borderRadius: '4px',
              fontSize: '0.8rem',
              fontFamily: 'monospace',
              border: '1px solid rgba(255, 255, 255, 0.06)',
            }}
          >
            {part.slice(1, -1)}
          </code>
        );
      }
      if (part.startsWith('*') && part.endsWith('*')) {
        return (
          <em key={i} style={{ color: '#94A3B8' }}>
            {part.slice(1, -1)}
          </em>
        );
      }
      return part;
    });
  };

  return (
    <div className="reports-view" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Toast Alert */}
      {actionMessage && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.6rem 1rem',
            borderRadius: '8px',
            background: actionMessage.type === 'success' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
            border: `1px solid ${actionMessage.type === 'success' ? '#10B981' : '#F43F5E'}`,
            color: actionMessage.type === 'success' ? '#34D399' : '#F43F5E',
            fontSize: '0.85rem',
            fontWeight: 600,
          }}
        >
          <span>{actionMessage.text}</span>
          <button
            onClick={() => setActionMessage(null)}
            style={{ background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer', fontWeight: 700 }}
          >
            ✕
          </button>
        </div>
      )}

      {/* Top Header Card */}
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
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', flexWrap: 'wrap' }}>
          <div
            style={{
              width: 38,
              height: 38,
              borderRadius: '8px',
              background: 'linear-gradient(135deg, #0284C7 0%, #6366F1 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#FFFFFF',
            }}
          >
            <BookOpen size={20} />
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700, color: '#F1F5F9' }}>
              Scientific Reports & Behavioral Benchmarks
            </h3>
            <div style={{ fontSize: '0.72rem', color: '#94A3B8' }}>
              Esplora i report scientifici Markdown e i dataset di telemetria generati dagli studi di RL (Fase 6 e 7)
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <button
            onClick={fetchReports}
            disabled={isLoading}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              backgroundColor: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              color: '#E2E8F0',
              borderRadius: '6px',
              padding: '0.4rem 0.75rem',
              fontWeight: 600,
              fontSize: '0.8rem',
              cursor: 'pointer',
            }}
          >
            <RefreshCw size={14} />
            <span>Ricarica</span>
          </button>
        </div>
      </div>

      {/* Main Split: Left Files Sidebar, Right Content Viewer */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(280px, 340px) minmax(0, 1fr)',
          gap: '1.25rem',
          alignItems: 'start',
        }}
      >
        {/* Left: Reports Catalog Sidebar */}
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.9)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '12px',
            padding: '1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.85rem',
          }}
        >
          {/* Search & Phase Filters */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <Search size={13} color="#64748B" style={{ position: 'absolute', left: '0.6rem' }} />
              <input
                type="text"
                placeholder="Cerca report..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{
                  width: '100%',
                  backgroundColor: '#1E293B',
                  color: '#F1F5F9',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '6px',
                  padding: '0.35rem 0.6rem 0.35rem 1.8rem',
                  fontSize: '0.8rem',
                  boxSizing: 'border-box',
                }}
              />
            </div>

            {/* Phase Selector Tabs */}
            <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
              <button
                onClick={() => setPhaseFilter('all')}
                style={{
                  padding: '0.2rem 0.5rem',
                  borderRadius: '4px',
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  border: 'none',
                  background: phaseFilter === 'all' ? '#3B82F6' : 'rgba(30, 41, 59, 0.6)',
                  color: phaseFilter === 'all' ? '#FFFFFF' : '#94A3B8',
                  cursor: 'pointer',
                }}
              >
                Tutti ({reports.length})
              </button>
              {uniquePhases.map((phase) => (
                <button
                  key={phase}
                  onClick={() => setPhaseFilter(phase)}
                  style={{
                    padding: '0.2rem 0.5rem',
                    borderRadius: '4px',
                    fontSize: '0.7rem',
                    fontWeight: 600,
                    border: 'none',
                    background: phaseFilter === phase ? '#3B82F6' : 'rgba(30, 41, 59, 0.6)',
                    color: phaseFilter === phase ? '#FFFFFF' : '#94A3B8',
                    cursor: 'pointer',
                  }}
                >
                  {phase}
                </button>
              ))}
            </div>
          </div>

          {/* Report Items List */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.4rem',
              maxHeight: '600px',
              overflowY: 'auto',
            }}
          >
            {filteredReports.map((r) => {
              const isSelected = selectedFilename === r.filename;
              return (
                <div
                  key={r.filename}
                  onClick={() => setSelectedFilename(r.filename)}
                  style={{
                    padding: '0.65rem 0.75rem',
                    borderRadius: '8px',
                    background: isSelected ? 'rgba(59, 130, 246, 0.15)' : 'rgba(30, 41, 59, 0.4)',
                    border: `1px solid ${isSelected ? '#3B82F6' : 'rgba(255, 255, 255, 0.05)'}`,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.25rem',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', minWidth: 0 }}>
                      {r.file_type === 'markdown' ? (
                        <FileText size={15} color="#38BDF8" style={{ flexShrink: 0 }} />
                      ) : (
                        <FileCode size={15} color="#FBBF24" style={{ flexShrink: 0 }} />
                      )}
                      <span
                        style={{
                          fontSize: '0.82rem',
                          fontWeight: 700,
                          color: isSelected ? '#38BDF8' : '#F1F5F9',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                        }}
                      >
                        {r.name}
                      </span>
                    </div>
                    <span
                      style={{
                        fontSize: '0.65rem',
                        fontWeight: 700,
                        padding: '0.1rem 0.35rem',
                        borderRadius: '4px',
                        background: r.file_type === 'markdown' ? 'rgba(56, 189, 248, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                        color: r.file_type === 'markdown' ? '#38BDF8' : '#FBBF24',
                        textTransform: 'uppercase',
                      }}
                    >
                      {r.file_type}
                    </span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.68rem', color: '#64748B' }}>
                    <span style={{ fontFamily: 'monospace' }}>{r.filename}</span>
                    <span>{r.size_kb} KB</span>
                  </div>
                </div>
              );
            })}

            {filteredReports.length === 0 && (
              <div style={{ textAlign: 'center', padding: '2rem 1rem', color: '#64748B', fontSize: '0.8rem' }}>
                Nessun report corrisponde ai criteri di ricerca.
              </div>
            )}
          </div>
        </div>

        {/* Right: Active Report Viewer */}
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.9)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '12px',
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem',
            minHeight: '600px',
          }}
        >
          {reportDetail ? (
            <>
              {/* Document Action Toolbar */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
                  paddingBottom: '0.75rem',
                  flexWrap: 'wrap',
                  gap: '0.75rem',
                }}
              >
                <div>
                  <h4 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: '#F1F5F9' }}>
                    {reportDetail.name}
                  </h4>
                  <div style={{ fontSize: '0.72rem', color: '#64748B', marginTop: '0.15rem' }}>
                    File: <code style={{ color: '#38BDF8' }}>{reportDetail.filename}</code> • Ultima Modifica: {reportDetail.modified_at} • Dimensione: {reportDetail.size_kb} KB
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {/* Mode switch */}
                  <div
                    style={{
                      display: 'flex',
                      background: 'rgba(30, 41, 59, 0.8)',
                      borderRadius: '6px',
                      padding: '0.15rem',
                      border: '1px solid rgba(255, 255, 255, 0.08)',
                    }}
                  >
                    <button
                      onClick={() => setViewMode('rendered')}
                      style={{
                        border: 'none',
                        background: viewMode === 'rendered' ? '#3B82F6' : 'transparent',
                        color: viewMode === 'rendered' ? '#FFFFFF' : '#94A3B8',
                        padding: '0.25rem 0.55rem',
                        borderRadius: '4px',
                        fontSize: '0.75rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      Rendered
                    </button>
                    <button
                      onClick={() => setViewMode('raw')}
                      style={{
                        border: 'none',
                        background: viewMode === 'raw' ? '#3B82F6' : 'transparent',
                        color: viewMode === 'raw' ? '#FFFFFF' : '#94A3B8',
                        padding: '0.25rem 0.55rem',
                        borderRadius: '4px',
                        fontSize: '0.75rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      Raw
                    </button>
                  </div>

                  {/* Copy Button */}
                  <button
                    onClick={handleCopy}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                      background: 'rgba(255, 255, 255, 0.06)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '6px',
                      color: '#CBD5E1',
                      padding: '0.3rem 0.6rem',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                    title="Copia contenuto negli appunti"
                  >
                    {copied ? <Check size={13} color="#34D399" /> : <Copy size={13} />}
                    <span>{copied ? 'Copiato!' : 'Copia'}</span>
                  </button>

                  {/* Download Button */}
                  <button
                    onClick={handleDownload}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                      background: 'rgba(255, 255, 255, 0.06)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '6px',
                      color: '#CBD5E1',
                      padding: '0.3rem 0.6rem',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                    title="Scarica file su disco"
                  >
                    <Download size={13} />
                    <span>Download</span>
                  </button>

                  {/* Delete Button */}
                  <button
                    onClick={() => handleDelete(reportDetail.filename)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                      background: 'rgba(239, 68, 68, 0.15)',
                      border: '1px solid rgba(239, 68, 68, 0.3)',
                      borderRadius: '6px',
                      color: '#F87171',
                      padding: '0.3rem 0.6rem',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                    title="Elimina report"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>

              {/* Rendered View or Raw View */}
              {isDetailLoading ? (
                <div style={{ padding: '3rem 0', textAlign: 'center', color: '#94A3B8' }}>
                  Caricamento contenuto report in corso...
                </div>
              ) : viewMode === 'raw' ? (
                <pre
                  style={{
                    background: '#0B1120',
                    padding: '1rem',
                    borderRadius: '8px',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    color: '#CBD5E1',
                    fontSize: '0.8rem',
                    lineHeight: 1.5,
                    maxHeight: '650px',
                    overflowY: 'auto',
                    fontFamily: 'monospace',
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {reportDetail.raw_content}
                </pre>
              ) : reportDetail.file_type === 'json' ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {/* Summary Metric Cards for JSON Benchmarks */}
                  {reportDetail.json_data?.reward_studies && (
                    <div>
                      <h5 style={{ margin: '0 0 0.5rem 0', color: '#38BDF8', fontSize: '0.9rem' }}>
                        🎯 Riepilogo Multi-Reward (Fase 7)
                      </h5>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem' }}>
                        {Object.entries(reportDetail.json_data.reward_studies).map(([k, v]: [string, any]) => {
                          const vsRand = v?.vs_random || {};
                          return (
                            <div
                              key={k}
                              style={{
                                background: 'rgba(30, 41, 59, 0.7)',
                                border: '1px solid rgba(56, 189, 248, 0.2)',
                                borderRadius: '8px',
                                padding: '0.75rem',
                              }}
                            >
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ fontWeight: 800, color: '#38BDF8', fontSize: '0.85rem' }}>
                                  {k.toUpperCase()}
                                </span>
                                <span style={{ fontSize: '0.7rem', color: '#94A3B8' }}>vs Random</span>
                              </div>
                              <div style={{ marginTop: '0.4rem', display: 'flex', flexDirection: 'column', gap: '0.2rem', fontSize: '0.75rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                  <span style={{ color: '#94A3B8' }}>Win Rate:</span>
                                  <span style={{ fontWeight: 700, color: '#34D399' }}>
                                    {((vsRand.win_rate || 0) * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                  <span style={{ color: '#94A3B8' }}>Avg Score:</span>
                                  <span style={{ fontWeight: 700, color: '#F1F5F9' }}>
                                    {(vsRand.avg_score || 0).toFixed(1)}
                                  </span>
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                  <span style={{ color: '#94A3B8' }}>Ticket Comp:</span>
                                  <span style={{ fontWeight: 700, color: '#FBBF24' }}>
                                    {((vsRand.ticket_completion_rate || 0) * 100).toFixed(1)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Pretty printed JSON */}
                  <pre
                    style={{
                      background: '#0B1120',
                      padding: '1rem',
                      borderRadius: '8px',
                      border: '1px solid rgba(255, 255, 255, 0.08)',
                      color: '#CBD5E1',
                      fontSize: '0.8rem',
                      maxHeight: '550px',
                      overflowY: 'auto',
                      fontFamily: 'monospace',
                    }}
                  >
                    {JSON.stringify(reportDetail.json_data || JSON.parse(reportDetail.raw_content), null, 2)}
                  </pre>
                </div>
              ) : (
                <div
                  style={{
                    maxHeight: '650px',
                    overflowY: 'auto',
                    paddingRight: '0.5rem',
                  }}
                >
                  {renderMarkdown(reportDetail.raw_content)}
                </div>
              )}
            </>
          ) : (
            <div style={{ textAlign: 'center', padding: '4rem 1rem', color: '#64748B', fontStyle: 'italic' }}>
              Seleziona un report dalla lista a sinistra per visualizzarlo.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
