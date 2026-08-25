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
        <div key={`table-${key}`} style={{ overflowX: 'auto', margin: '1rem 0', borderRadius: '8px', border: '1.5px solid #C59B27' }}>
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '0.82rem',
              backgroundColor: '#FAF5EB',
              textAlign: 'left',
            }}
          >
            <thead>
              <tr style={{ background: 'linear-gradient(180deg, #EFE1C7 0%, #E2CFAC 100%)', borderBottom: '2px solid #C59B27' }}>
                {headers.map((h, i) => (
                  <th
                    key={i}
                    style={{
                      padding: '0.65rem 0.85rem',
                      color: '#23140C',
                      fontWeight: 800,
                      fontFamily: "'Playfair Display', Georgia, serif",
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
                    borderBottom: '1px solid rgba(184, 134, 11, 0.2)',
                    background: ri % 2 === 1 ? 'rgba(246, 238, 223, 0.5)' : 'transparent',
                  }}
                >
                  {row.map((cell, ci) => {
                    const cleanCell = cell.trim();
                    const isWinRate = cleanCell.endsWith('%') && !isNaN(parseFloat(cleanCell));
                    const isPositiveDiff = cleanCell.startsWith('+');
                    const isNegativeDiff = cleanCell.startsWith('-');

                    let cellColor = '#23140C';
                    if (isWinRate && parseFloat(cleanCell) >= 50) cellColor = '#15803D';
                    else if (isPositiveDiff) cellColor = '#1D4ED8';
                    else if (isNegativeDiff) cellColor = '#B91C1C';

                    return (
                      <td
                        key={ci}
                        style={{
                          padding: '0.55rem 0.85rem',
                          textAlign: ci === 0 ? 'left' : 'center',
                          color: cellColor,
                          fontWeight: ci === 0 ? 700 : 600,
                          fontFamily: ci > 0 ? "'Courier Prime', monospace" : "'Playfair Display', serif",
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
                background: '#2B1D14',
                color: '#FAF5EB',
                border: '1.5px solid #8C6305',
                borderRadius: '8px',
                padding: '0.85rem 1rem',
                overflowX: 'auto',
                fontSize: '0.8rem',
                fontFamily: "'Courier Prime', monospace",
                margin: '0.85rem 0',
                boxShadow: 'inset 0 2px 6px rgba(0,0,0,0.5)',
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
        const cells = trimmed
          .slice(1, -1)
          .split('|')
          .map((c) => c.trim());
        tableRows.push(cells);
        return;
      } else if (inTable) {
        const tbl = flushTable(idx);
        if (tbl) elements.push(tbl);
      }

      // Headings
      if (trimmed.startsWith('# ')) {
        elements.push(
          <h1
            key={idx}
            style={{
              fontSize: '1.4rem',
              fontWeight: 900,
              color: '#23140C',
              fontFamily: "'Cinzel Decorative', Georgia, serif",
              margin: '1.5rem 0 0.6rem 0',
              borderBottom: '2px solid #C59B27',
              paddingBottom: '0.4rem',
            }}
          >
            {trimmed.slice(2)}
          </h1>
        );
        return;
      }
      if (trimmed.startsWith('## ')) {
        elements.push(
          <h2
            key={idx}
            style={{
              fontSize: '1.15rem',
              fontWeight: 800,
              color: '#23140C',
              fontFamily: "'Cinzel Decorative', Georgia, serif",
              margin: '1.25rem 0 0.4rem 0',
              borderBottom: '1px solid rgba(184, 134, 11, 0.3)',
              paddingBottom: '0.25rem',
            }}
          >
            {trimmed.slice(3)}
          </h2>
        );
        return;
      }
      if (trimmed.startsWith('### ')) {
        elements.push(
          <h3
            key={idx}
            style={{
              fontSize: '0.98rem',
              fontWeight: 800,
              color: '#9E6B00',
              fontFamily: "'Playfair Display', Georgia, serif",
              margin: '1rem 0 0.3rem 0',
            }}
          >
            {trimmed.slice(4)}
          </h3>
        );
        return;
      }

      // Blockquotes
      if (trimmed.startsWith('> ')) {
        elements.push(
          <div
            key={idx}
            style={{
              background: '#FAF0DA',
              borderLeft: '4px solid #C59B27',
              padding: '0.5rem 0.85rem',
              borderRadius: '0 6px 6px 0',
              margin: '0.6rem 0',
              color: '#4A2F1D',
              fontSize: '0.85rem',
              fontFamily: "'Crimson Pro', Georgia, serif",
              fontStyle: 'italic',
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
              color: '#23140C',
              fontFamily: "'Crimson Pro', Georgia, serif",
            }}
          >
            <span style={{ color: '#9E6B00', marginTop: '0.1rem' }}>•</span>
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
              borderTop: '1px solid rgba(184, 134, 11, 0.3)',
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
              fontSize: '0.88rem',
              lineHeight: 1.6,
              color: '#23140C',
              fontFamily: "'Crimson Pro', Georgia, serif",
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

  const renderInlineMarkdown = (text: string): React.ReactNode => {
    const parts = text.split(/(\*\*.*?\*\*|`.*?`|\*.*?\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={i} style={{ color: '#23140C', fontWeight: 800 }}>
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code
            key={i}
            style={{
              background: '#EADBBE',
              color: '#23140C',
              border: '1px solid #C59B27',
              borderRadius: '4px',
              padding: '0.1rem 0.35rem',
              fontSize: '0.8rem',
              fontFamily: "'Courier Prime', monospace",
            }}
          >
            {part.slice(1, -1)}
          </code>
        );
      }
      if (part.startsWith('*') && part.endsWith('*')) {
        return (
          <em key={i} style={{ fontStyle: 'italic' }}>
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
            background: actionMessage.type === 'success' ? '#FAF3E6' : '#FEE2E2',
            border: `1.5px solid ${actionMessage.type === 'success' ? '#15803D' : '#B91C1C'}`,
            color: actionMessage.type === 'success' ? '#15803D' : '#B91C1C',
            fontSize: '0.85rem',
            fontFamily: "'Playfair Display', Georgia, serif",
            fontWeight: 700,
          }}
        >
          <span>{actionMessage.text}</span>
          <button
            onClick={() => setActionMessage(null)}
            style={{ background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer', fontWeight: 800 }}
          >
            ✕
          </button>
        </div>
      )}

      {/* Top Header Card */}
      <div
        className="steampunk-panel"
        style={{
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
              width: 40,
              height: 40,
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
            <BookOpen size={22} />
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 900, color: '#23140C', fontFamily: "'Cinzel Decorative', Georgia, serif" }}>
              Scientific Dispatch Archive & Benchmark Reports
            </h3>
            <div style={{ fontSize: '0.8rem', color: '#5A3822', fontFamily: "'Crimson Pro', Georgia, serif" }}>
              Explore official Markdown dispatches and empirical tournament logs produced during RL laboratory runs
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <button
            onClick={fetchReports}
            disabled={isLoading}
            className="steampunk-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              fontSize: '0.8rem',
            }}
          >
            <RefreshCw size={14} />
            <span>Refresh</span>
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
          className="steampunk-panel"
          style={{
            padding: '1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.85rem',
          }}
        >
          {/* Search & Phase Filters */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <Search size={13} color="#785A42" style={{ position: 'absolute', left: '0.6rem' }} />
              <input
                type="text"
                placeholder="Search reports..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{
                  width: '100%',
                  backgroundColor: '#FAF5EB',
                  color: '#23140C',
                  border: '1.5px solid #8C6305',
                  borderRadius: '6px',
                  padding: '0.35rem 0.6rem 0.35rem 1.8rem',
                  fontSize: '0.8rem',
                  boxSizing: 'border-box',
                  fontFamily: "'Courier Prime', monospace",
                }}
              />
            </div>

            {/* Phase Selector */}
            {uniquePhases.length > 0 && (
              <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                <button
                  onClick={() => setPhaseFilter('all')}
                  className="steampunk-btn"
                  style={{
                    padding: '0.15rem 0.5rem',
                    fontSize: '0.7rem',
                    background: phaseFilter === 'all' ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 100%)' : '#FAF5EB',
                  }}
                >
                  All
                </button>
                {uniquePhases.map((phase) => (
                  <button
                    key={phase}
                    onClick={() => setPhaseFilter(phase)}
                    className="steampunk-btn"
                    style={{
                      padding: '0.15rem 0.5rem',
                      fontSize: '0.7rem',
                      background: phaseFilter === phase ? 'linear-gradient(180deg, #F7E099 0%, #CBA232 100%)' : '#FAF5EB',
                    }}
                  >
                    {phase}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Files List */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '550px', overflowY: 'auto' }}>
            {filteredReports.map((r) => {
              const isSelected = selectedFilename === r.filename;
              return (
                <div
                  key={r.filename}
                  onClick={() => setSelectedFilename(r.filename)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0.5rem 0.75rem',
                    borderRadius: '6px',
                    backgroundColor: isSelected ? '#FAF0DA' : '#FAF5EB',
                    border: `1.5px solid ${isSelected ? '#B8860B' : 'rgba(184, 134, 11, 0.25)'}`,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    boxShadow: isSelected ? '0 2px 6px rgba(184, 134, 11, 0.25)' : 'none',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', overflow: 'hidden' }}>
                    {r.file_type === 'markdown' ? <FileText size={15} color="#9E6B00" /> : <FileCode size={15} color="#7E22CE" />}
                    <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                      <span style={{ fontSize: '0.8rem', fontWeight: isSelected ? 800 : 600, color: '#23140C', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontFamily: "'Playfair Display', Georgia, serif" }}>
                        {r.name}
                      </span>
                      <span style={{ fontSize: '0.7rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
                        {r.size_kb} KB • {r.modified_at.split(' ')[0]}
                      </span>
                    </div>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(r.filename);
                    }}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#B91C1C',
                      cursor: 'pointer',
                      padding: '0.2rem',
                    }}
                    title="Delete dispatch"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Selected Report Reader */}
        <div
          className="steampunk-panel"
          style={{
            padding: '1.25rem',
            minHeight: '600px',
          }}
        >
          {isDetailLoading ? (
            <div style={{ padding: '3rem', textAlign: 'center', color: '#785A42', fontStyle: 'italic', fontFamily: "'Crimson Pro', serif" }}>
              Loading dispatch document...
            </div>
          ) : reportDetail ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {/* Document Action Bar */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', borderBottom: '2px solid #C59B27', paddingBottom: '0.75rem' }}>
                <div>
                  <h2 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 900, color: '#23140C', fontFamily: "'Cinzel Decorative', Georgia, serif" }}>
                    {reportDetail.name}
                  </h2>
                  <div style={{ fontSize: '0.75rem', color: '#785A42', fontFamily: "'Courier Prime', monospace" }}>
                    {reportDetail.filename} • {reportDetail.size_kb} KB
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <button
                    onClick={() => setViewMode(viewMode === 'rendered' ? 'raw' : 'rendered')}
                    className="steampunk-btn"
                    style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
                  >
                    {viewMode === 'rendered' ? 'View Raw' : 'View Rendered'}
                  </button>
                  <button
                    onClick={handleCopy}
                    className="steampunk-btn"
                    style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
                  >
                    {copied ? <Check size={13} color="#15803D" /> : <Copy size={13} />}
                    <span>{copied ? 'Copied' : 'Copy'}</span>
                  </button>
                  <button
                    onClick={handleDownload}
                    className="steampunk-btn"
                    style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
                  >
                    <Download size={13} />
                    <span>Download</span>
                  </button>
                </div>
              </div>

              {/* Document Body */}
              <div style={{ color: '#23140C' }}>
                {viewMode === 'rendered' ? (
                  reportDetail.file_type === 'markdown' ? (
                    renderMarkdown(reportDetail.raw_content)
                  ) : (
                    <pre
                      style={{
                        background: '#2B1D14',
                        color: '#FAF5EB',
                        border: '1.5px solid #8C6305',
                        borderRadius: '8px',
                        padding: '1rem',
                        overflowX: 'auto',
                        fontSize: '0.8rem',
                        fontFamily: "'Courier Prime', monospace",
                      }}
                    >
                      <code>{reportDetail.raw_content}</code>
                    </pre>
                  )
                ) : (
                  <pre
                    style={{
                      background: '#2B1D14',
                      color: '#FAF5EB',
                      border: '1.5px solid #8C6305',
                      borderRadius: '8px',
                      padding: '1rem',
                      overflowX: 'auto',
                      fontSize: '0.8rem',
                      fontFamily: "'Courier Prime', monospace",
                    }}
                  >
                    <code>{reportDetail.raw_content}</code>
                  </pre>
                )}
              </div>
            </div>
          ) : (
            <div style={{ padding: '3rem', textAlign: 'center', color: '#785A42', fontStyle: 'italic', fontFamily: "'Crimson Pro', serif" }}>
              Select a dispatch from the left catalog to read.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
