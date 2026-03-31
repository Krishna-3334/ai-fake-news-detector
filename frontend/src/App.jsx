import { useState, useEffect } from 'react';
import './App.css';

const TrustMeter = ({ score, color }) => {
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score * circumference);

  return (
    <div className="trust-meter">
      <svg className="meter-svg" width="160" height="160">
        <circle className="meter-bg" cx="80" cy="80" r={radius} />
        <circle
          className="meter-progress"
          cx="80" cy="80" r={radius}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: isNaN(offset) ? circumference : offset,
            stroke: color
          }}
        />
      </svg>
      <div className="meter-center">
        <span className="trust-percent">{(score * 100).toFixed(0)}%</span>
        <span className="trust-label">Trust Index</span>
      </div>
    </div>
  );
};

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [engineStatus, setEngineStatus] = useState(null);

  useEffect(() => {
    const saved = localStorage.getItem('v_intel_history');
    if (saved) setHistory(JSON.parse(saved));

    const checkStatus = async () => {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        setEngineStatus(data.engines);
      } catch (e) {
        console.error("Status check failed", e);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const saveToHistory = (res, originalText) => {
    const entry = {
      id: Date.now(),
      preview: originalText.slice(0, 80) + (originalText.length > 80 ? '...' : ''),
      prediction: res.prediction,
      score: res.overall_trust,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    const updated = [entry, ...history.slice(0, 9)];
    setHistory(updated);
    localStorage.setItem('v_intel_history', JSON.stringify(updated));
  };

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
        saveToHistory(data, text);
      }
    } catch (err) {
      setError('Neural network connectivity failed. Ensure the verification engine is active.');
    } finally {
      setLoading(false);
    }
  };

  const getVerdictLabel = (t) => {
    if (t >= 0.7) return { text: 'AUTHENTIC', color: '#10b981', class: 'status-real' };
    if (t >= 0.45) return { text: 'UNCERTAIN', color: '#f59e0b', class: 'status-uncertain' };
    return { text: 'UNRELIABLE', color: '#ef4444', class: 'status-fake' };
  };

  const getStatusClass = (status) => {
    if (!status) return 'status-offline';
    if (status === 'ONLINE') return 'status-online';
    if (status.includes('DOWNLOADING') || status.includes('SYNCING')) return 'status-syncing';
    return 'status-offline';
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="title-group">
          <p>Intelligence Platform</p>
          <h1>Vérifié</h1>
        </div>
        <div className="system-status">
          <div className="engine-badges">
            <div className={`engine-badge ${getStatusClass(engineStatus?.engine_1)}`}>
              E1<span>•</span>
            </div>
            <div className={`engine-badge ${getStatusClass(engineStatus?.engine_2)}`}>
              E2<span>•</span>
            </div>
            <div className={`engine-badge ${getStatusClass(engineStatus?.engine_3)}`}>
              E3<span>•</span>
            </div>
          </div>
          <div className="status-pulse"></div>
          <span>V.2.0 ACTIVE CORE</span>
        </div>
      </header>

      <div className="main-layout">
        <div className="intelligence-desk">
          <div className="glass-panel">
            <div className="engine-sync-bar">
              {engineStatus?.engine_3 && engineStatus.engine_3 !== 'ONLINE' && (
                <div className="sync-notice">
                  <span className="sync-icon">🔄</span>
                  Engine 3 [AI-Content]: {engineStatus.engine_3}...
                  <span className="sync-detail">Downloading weights from Neural Hub</span>
                </div>
              )}
            </div>
            <div className="news-input-wrapper">
              <textarea
                className="news-input"
                placeholder="Submit text, report excerpt, or article URL for deep analysis..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>
            <div className="input-footer">
              <div className="input-meta">
                {text.length > 0 && <span>{text.split(/\s+/).length} Words Detectable</span>}
              </div>
              <button
                className="analyze-btn"
                onClick={handleAnalyze}
                disabled={loading || !text.trim()}
              >
                {loading ? <div className="spinner"></div> : "RUN VERIFICATION"}
              </button>
            </div>
          </div>

          {error && (
            <div className="glass-panel" style={{ marginTop: '2rem', borderColor: 'rgba(239, 68, 68, 0.3)' }}>
              <p style={{ color: '#ef4444', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span>⚠️</span> SYSTEM_ERROR: {error}
              </p>
            </div>
          )}

          {result && (
            <div className="results-grid">
              <div className="verdict-status">
                <TrustMeter
                  score={result.overall_trust}
                  color={getVerdictLabel(result.overall_trust).color}
                />
                <div className={`verdict-label ${getVerdictLabel(result.overall_trust).class}`}>
                  {result.prediction}
                </div>
                <div style={{ textAlign: 'center', color: '#64748b', fontSize: '0.8rem', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span>Network Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                  <span className={`correlation-badge ${result.intelligence.correlation.toLowerCase()}`}>
                    {result.intelligence.correlation} CONSISTENCY
                  </span>
                </div>
              </div>

              <div className="intelligence-content">
                <div className="metrics-box">
                  <div className="metric-card">
                    <span className="m-label">Bias Level</span>
                    <span className="m-value">{(result.metrics.bias_intensity * 100).toFixed(0)}%</span>
                  </div>
                  <div className="metric-card">
                    <span className="m-label">AI Origin</span>
                    <span className={`m-value ${result.intelligence.is_ai ? 'status-fake' : 'status-real'}`}>
                      {(result.intelligence.ai_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="metric-card">
                    <span className="m-label">Readability</span>
                    <span className="m-value">{result.metrics.readability.toFixed(0)}</span>
                  </div>
                </div>

                <div className="signals-container">
                  <h4 className="intel-section-title">Verification Signals</h4>
                  {result.signals.map((sig, idx) => (
                    <div key={idx} className={`signal ${sig.type}`}>
                      <span className="signal-icon">
                        {sig.type === 'positive' ? '🛡️' : sig.type === 'negative' ? '🚩' : '📍'}
                      </span>
                      <span className="signal-text">{sig.text}</span>
                    </div>
                  ))}
                </div>

                <div className="entity-map">
                  <h4 className="intel-section-title">Key Intelligence Entities</h4>
                  <div className="entity-tags">
                    {result.intelligence.entities.length > 0 ? result.intelligence.entities.map((ent, idx) => (
                      <span key={idx} className={`ent-tag ${ent.type.toLowerCase()}`}>
                        <strong>{ent.type}:</strong> {ent.text}
                      </span>
                    )) : <span style={{ color: '#475569', fontSize: '0.85rem' }}>No significant entities detected.</span>}
                  </div>
                </div>

                <div className="summary-box">
                  <h4 className="intel-section-title">Content Intelligence Extract</h4>
                  <p>{result.intelligence.summary}</p>
                </div>

                <div className="report-actions">
                  <button className="copy-btn" onClick={() => {
                    navigator.clipboard.writeText(`Verification Report: ${result.prediction} (${(result.overall_trust * 100).toFixed(0)}% Trust Index)`);
                    alert('Summary report copied to clipboard');
                  }}>
                    <span>📄</span> Copy Brief Report
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        <aside className="sidebar">
          <div className="glass-panel history-section">
            <h3>LOG_HISTORY</h3>
            <div className="history-list">
              {history.length > 0 ? history.map(item => (
                <div key={item.id} className="history-item">
                  <div className="history-meta">
                    <span className={`h-verdict status-${item.prediction.toLowerCase()}`}>
                      {item.prediction}
                    </span>
                    <span style={{ fontSize: '0.7rem', color: '#475569' }}>{item.time}</span>
                  </div>
                  <div className="h-preview">{item.preview}</div>
                </div>
              )) : (
                <p style={{ textAlign: 'center', color: '#475569', fontSize: '0.8rem', padding: '1rem' }}>
                  No records found
                </p>
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
