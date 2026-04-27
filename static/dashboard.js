/* ══════════════════════════════════════════════════════════
   AI Fairness Auditor — dashboard.js
   Exactly matches dashboard.html element IDs & app.py API shapes
   ══════════════════════════════════════════════════════════ */
'use strict';

// ── Application State ─────────────────────────────────────
const state = {
  currentTab: 'pretrain',
  activeDomain: 'hiring',
  activeFilter: 'all',
  preFile: null,
  postFile: null,
  postFileData: null,
  postFilePred: null,
  preMetrics: null,
  postMetrics: null,
  allReports: [],
  chartInstances: {},
};

// ══════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNav();
  initPreTraining();
  initPostTraining();
  initAppeal();
  initReports();
  initSliders();
});

// ══════════════════════════════════════════════════════════
// THEME
// ══════════════════════════════════════════════════════════
function initTheme() {
  const saved = localStorage.getItem('fairness-theme') || 'dark';
  applyTheme(saved);
  const btn = document.getElementById('themeToggle');
  if (btn) btn.addEventListener('click', () => {
    applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  });
}
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('fairness-theme', theme);
  const btn = document.getElementById('themeToggle');
  if (btn) btn.textContent = theme === 'dark' ? '☀️' : '🌙';
}

// ══════════════════════════════════════════════════════════
// NAVIGATION
// ══════════════════════════════════════════════════════════
function initNav() {
  document.querySelectorAll('.nav-item[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
  const hamburger = document.getElementById('hamburger');
  if (hamburger) hamburger.addEventListener('click', () => {
    document.getElementById('sidebar').classList.toggle('sidebar-open');
  });
}

function switchTab(tab) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const panel = document.getElementById(`tab-${tab}`);
  if (panel) panel.classList.add('active');
  const navBtn = document.querySelector(`.nav-item[data-tab="${tab}"]`);
  if (navBtn) navBtn.classList.add('active');
  state.currentTab = tab;
  const titles = { pretrain: 'Pre-Training Audit', posttrain: 'Post-Training Audit', appeal: 'Appeal Engine', reports: 'Audit Reports' };
  const bc = document.getElementById('breadcrumbCurrent');
  if (bc) bc.textContent = titles[tab] || tab;
  if (tab === 'reports') loadReports();
  document.getElementById('sidebar')?.classList.remove('sidebar-open');
}

// ══════════════════════════════════════════════════════════
// SLIDERS
// ══════════════════════════════════════════════════════════
function initSliders() {
  const preSlider = document.getElementById('preBalanceSlider');
  if (preSlider) preSlider.addEventListener('input', () => {
    document.getElementById('preBalanceVal').textContent = preSlider.value;
  });
  const postThresh = document.getElementById('postThreshSlider');
  if (postThresh) postThresh.addEventListener('input', () => {
    document.getElementById('postThreshVal').textContent = (postThresh.value / 100).toFixed(2);
  });
  const postFair = document.getElementById('postFairSlider');
  if (postFair) postFair.addEventListener('input', () => {
    document.getElementById('postFairVal').textContent = (postFair.value / 100).toFixed(2);
  });
}

// ══════════════════════════════════════════════════════════
// PRE-TRAINING
// ══════════════════════════════════════════════════════════
function initPreTraining() {
  const fileInput = document.getElementById('preFile');
  const analyzeBtn = document.getElementById('preAnalyzeBtn');
  const simBtn = document.getElementById('preSimulateBtn');

  if (fileInput) fileInput.addEventListener('change', () => {
    const f = fileInput.files[0];
    if (!f) return;
    state.preFile = f;
    document.getElementById('preFileName').textContent = f.name;
    if (analyzeBtn) { analyzeBtn.disabled = false; analyzeBtn.classList.add('ready'); }
  });
  if (analyzeBtn) analyzeBtn.addEventListener('click', runPreTraining);
  if (simBtn) simBtn.addEventListener('click', runWhatIfPre);
}

async function runPreTraining() {
  if (!state.preFile) { showToast('Please select a file first.', 'error'); return; }
  showEl('preLoading'); hideEl('preResults');

  const fd = new FormData();
  fd.append('file', state.preFile);

  try {
    const res = await fetch('/api/pretrain', { method: 'POST', body: fd, credentials: 'include' });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); }
    catch { throw new Error('Server returned non-JSON. You may be logged out or Flask crashed.'); }
    hideEl('preLoading');

    if (data.error) { showErrorInEl('preResults', data.error); return; }

    // text_analysis = TXT/PDF/XML/HTML that was text-extracted — runs full pipeline
    if (data.mode === 'text_analysis') {
      state.preMetrics = data.metrics;
      // Show info banner above metrics
      const grid = document.getElementById('preMetricsGrid');
      if (grid) {
        grid.innerHTML = `<div class="info-banner" style="grid-column:1/-1">
          📄 <strong>${escapeHtml(data.filename || 'Document')}</strong> — 
          ${escapeHtml(data.message || 'Analysed as text. Demographic co-occurrence metrics extracted.')}
        </div>`;
      }
      renderPreMetrics(data.metrics, data.alert_explanations || [], true);
      renderCharts(graphsToObj(data.graphs), 'preChartsGrid', 'pre');
      appendGroqCard('preChartsGrid', data.groq_explanations);
      setProviderTag('preProviderTag', data.provider);
      renderAIReport(data.report, 'preReportContainer', data.report_id, data.report_error);
      populateSelect('preWhatIfAttr', (data.metrics && data.metrics.protected_attributes) || []);
      showEl('preResults');
      showToast('Text analysis complete!', 'success');
      return;
    }

    state.preMetrics = data.metrics;
    renderPreMetrics(data.metrics, data.alert_explanations || []);
    renderCharts(graphsToObj(data.graphs), 'preChartsGrid', 'pre');
    appendGroqCard('preChartsGrid', data.groq_explanations);
    setProviderTag('preProviderTag', data.provider);
    renderAIReport(data.report, 'preReportContainer', data.report_id, data.report_error);
    populateSelect('preWhatIfAttr', data.metrics.protected_attributes || []);
    showEl('preResults');
    showToast('Pre-training audit complete!', 'success');

  } catch (e) {
    hideEl('preLoading');
    showErrorInEl('preResults', 'Request failed: ' + e.message);
  }
}

// ══════════════════════════════════════════════════════════
// FIX #1: RENDER PRE METRICS WITH RICH ALERT CARDS
// ══════════════════════════════════════════════════════════
function renderPreMetrics(m, alertExplanations, isTextMode) {
  const grid = document.getElementById('preMetricsGrid');
  if (!grid || !m) return;

  const totalCols = m.total_columns ?? m.total_cols ?? 0;
  const protectedCount = m.protected_attributes ? m.protected_attributes.length : (m.protected_count ?? 0);
  const alerts = m.alerts || m.imbalance_alerts || [];
  const alertCount = m.alert_count ?? alerts.length;
  const biasScore = m.bias_score ?? 0;
  const ts = m.text_stats || null;

  if (isTextMode && ts) {
    // Text-mode metric cards use word/sentence/mention counts
    grid.innerHTML += `
      ${metricCard('Bias Risk Score', biasScore, '/100', 'accent-blue', riskBadge(biasScore))}
      ${metricCard('Total Words', (ts.total_words ?? 0).toLocaleString(), '', 'accent-green')}
      ${metricCard('Sentences', (ts.total_sentences ?? 0).toLocaleString(), '', 'accent-purple')}
      ${metricCard('Demo Mentions', (ts.demographic_mentions ?? 0).toLocaleString(), '', 'accent-orange')}
      ${metricCard('Protected Attrs', protectedCount, '', 'accent-orange')}
      ${metricCard('Alerts', alertCount, '', alertCount > 0 ? 'accent-blue' : 'accent-green', alertBadge(alertCount, biasScore))}
      ${metricCard('Confidence', m.audit_confidence ?? '—', '', 'accent-purple', confidenceBadge(m.audit_confidence))}
    `;
  } else {
    grid.innerHTML = `
      ${metricCard('Bias Risk Score', biasScore, '/100', 'accent-blue', riskBadge(biasScore))}
      ${metricCard('Total Rows', (m.total_rows ?? 0).toLocaleString(), '', 'accent-green')}
      ${metricCard('Columns', totalCols, '', 'accent-purple')}
      ${metricCard('Protected Attrs', protectedCount, '', 'accent-orange')}
      ${metricCard('Alerts', alertCount, '', alertCount > 0 ? 'accent-blue' : 'accent-green', alertBadge(alertCount, biasScore))}
      ${metricCard('Confidence', m.audit_confidence ?? '—', '', 'accent-purple', confidenceBadge(m.audit_confidence))}
    `;
  }

  // FIX #1: Render rich alert explanation cards
  if (alertExplanations && alertExplanations.length > 0) {
    renderAlertExplanationCards(grid, alertExplanations);
  } else if (alerts.length > 0) {
    // Fallback: show basic alerts if no explanations yet
    let html = '<div class="section-heading" style="grid-column:1/-1">Detected Alerts</div>';
    alerts.forEach(a => {
      const sev = a.severity || (a.pct > 90 ? 'HIGH' : 'MEDIUM');
      const detail = a.detail || `Dominant group: ${a.dominant_group || '?'} (${a.pct || 0}%)`;
      html += `<div class="alert-item ${sev.toLowerCase()}" style="grid-column:1/-1">
        ⚠️ <strong>${escapeHtml(a.column || '')}</strong> — ${escapeHtml(detail)}
        <span class="badge badge-${sev.toLowerCase()}" style="margin-left:auto">${sev}</span>
      </div>`;
    });
    grid.innerHTML += html;
  }
}

// ══════════════════════════════════════════════════════════
// FIX #1: RICH ALERT EXPLANATION CARDS (Groq-powered)
// ══════════════════════════════════════════════════════════
function renderAlertExplanationCards(container, alertExplanations) {
  if (!alertExplanations || !alertExplanations.length) return;

  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'grid-column: 1 / -1; display: flex; flex-direction: column; gap: 1rem; margin-top: 0.5rem;';

  const heading = document.createElement('div');
  heading.className = 'bias-alerts-heading';
  heading.innerHTML = `
    <span class="bias-alerts-icon">🚨</span>
    <span>Bias Detection Alerts</span>
    <span class="bias-alerts-count">${alertExplanations.length} Alert${alertExplanations.length > 1 ? 's' : ''} Found</span>
  `;
  wrapper.appendChild(heading);

  alertExplanations.forEach(alert => {
    const card = document.createElement('div');
    const sev = (alert.severity || 'HIGH').toUpperCase();
    const sevCls = sev.toLowerCase();
    card.className = `bias-alert-card bias-alert-${sevCls} animate-in`;

    // Parse the Groq explanation into sections
    const parsed = parseAlertExplanation(alert.explanation || '');

    // Header info
    const attrLabel = escapeHtml(alert.column || 'Unknown Attribute');
    const typeLabel = alert.type === 'outcome_bias' ? 'Outcome Bias' : 'Representation Imbalance';
    const typeIcon = alert.type === 'outcome_bias' ? '⚖️' : '📊';

    // Build metrics row
    let metricsHtml = '';
    if (alert.type === 'outcome_bias' && alert.spd !== undefined) {
      const spd = typeof alert.spd === 'number' ? alert.spd.toFixed(4) : alert.spd;
      const di = typeof alert.di === 'number' ? alert.di.toFixed(4) : alert.di;
      const diNum = typeof alert.di === 'number' ? alert.di : parseFloat(alert.di);
      const rule80 = diNum >= 0.8;
      metricsHtml = `
        <div class="ba-metric-row">
          <div class="ba-metric-item">
            <span class="ba-metric-label">SPD</span>
            <span class="ba-metric-value ${parseFloat(spd) > 0.2 ? 'bad' : parseFloat(spd) > 0.1 ? 'warn' : 'good'}">${spd}</span>
          </div>
          <div class="ba-metric-item">
            <span class="ba-metric-label">Disparate Impact</span>
            <span class="ba-metric-value ${!rule80 ? 'bad' : 'good'}">${di}</span>
          </div>
          <div class="ba-metric-item">
            <span class="ba-metric-label">80% Rule</span>
            <span class="ba-metric-value ${rule80 ? 'good' : 'bad'}">${rule80 ? '✓ PASS' : '✗ FAIL'}</span>
          </div>
          ${alert.target ? `<div class="ba-metric-item">
            <span class="ba-metric-label">Affected Outcome</span>
            <span class="ba-metric-value neutral">${escapeHtml(alert.target)}</span>
          </div>` : ''}
        </div>`;
    } else if (alert.type === 'imbalance' && alert.pct !== undefined) {
      const dom = escapeHtml(alert.dominant_group || 'Unknown');
      metricsHtml = `
        <div class="ba-metric-row">
          <div class="ba-metric-item">
            <span class="ba-metric-label">Dominant Group</span>
            <span class="ba-metric-value bad">${dom}</span>
          </div>
          <div class="ba-metric-item">
            <span class="ba-metric-label">Dominance</span>
            <span class="ba-metric-value bad">${alert.pct}%</span>
          </div>
          <div class="ba-metric-item">
            <span class="ba-metric-label">Others Share</span>
            <span class="ba-metric-value warn">${(100 - parseFloat(alert.pct)).toFixed(1)}%</span>
          </div>
        </div>`;
    }

    // Build explanation sections
    const whereHtml = parsed.where
      ? `<div class="ba-section ba-where">
          <div class="ba-section-label">📍 Where Bias Exists</div>
          <div class="ba-section-text">${escapeHtml(parsed.where)}</div>
        </div>`
      : '';
    const whyHtml = parsed.why
      ? `<div class="ba-section ba-why">
          <div class="ba-section-label">🔍 Why Bias Exists</div>
          <div class="ba-section-text">${escapeHtml(parsed.why)}</div>
        </div>`
      : '';
    const todoHtml = parsed.todo
      ? `<div class="ba-section ba-todo">
          <div class="ba-section-label">💡 What To Do</div>
          <div class="ba-section-text">${escapeHtml(parsed.todo)}</div>
        </div>`
      : '';

    // Fallback: show raw text if parsing failed
    const fallbackHtml = (!parsed.where && !parsed.why && !parsed.todo && alert.explanation)
      ? `<div class="ba-section ba-why">
          <div class="ba-section-label">🔍 AI Analysis</div>
          <div class="ba-section-text">${escapeHtml(alert.explanation)}</div>
        </div>`
      : '';

    card.innerHTML = `
      <div class="ba-card-header">
        <div class="ba-card-left">
          <div class="ba-attr-pill">${typeIcon} ${attrLabel}</div>
          <div class="ba-type-label">${typeLabel}</div>
        </div>
        <div class="ba-card-right">
          <span class="ba-severity-badge ba-sev-${sevCls}">${sev} SEVERITY</span>
        </div>
      </div>
      ${metricsHtml}
      <div class="ba-explanation-grid">
        ${whereHtml}
        ${whyHtml}
        ${todoHtml}
        ${fallbackHtml}
      </div>
    `;

    wrapper.appendChild(card);
  });

  container.appendChild(wrapper);
}

function parseAlertExplanation(text) {
  if (!text) return { where: '', why: '', todo: '' };
  const result = { where: '', why: '', todo: '' };
  const lines = text.split('\n');
  let current = null;

  lines.forEach(line => {
    const t = line.trim();
    if (!t) return;
    const lo = t.toLowerCase();
    if (lo.startsWith('where bias exists:') || lo === 'where bias exists:') {
      current = 'where';
      const rest = t.replace(/^where bias exists:/i, '').trim();
      if (rest) result.where = rest;
    } else if (lo.startsWith('why bias exists:') || lo === 'why bias exists:') {
      current = 'why';
      const rest = t.replace(/^why bias exists:/i, '').trim();
      if (rest) result.why = rest;
    } else if (lo.startsWith('what to do:') || lo === 'what to do:') {
      current = 'todo';
      const rest = t.replace(/^what to do:/i, '').trim();
      if (rest) result.todo = rest;
    } else if (current) {
      if (current === 'where' && !result.where) result.where = t;
      else if (current === 'where') result.where += ' ' + t;
      else if (current === 'why' && !result.why) result.why = t;
      else if (current === 'why') result.why += ' ' + t;
      else if (current === 'todo' && !result.todo) result.todo = t;
      else if (current === 'todo') result.todo += ' ' + t;
    }
  });

  return result;
}

// ══════════════════════════════════════════════════════════
// POST-TRAINING
// ══════════════════════════════════════════════════════════
function initPostTraining() {
  document.querySelectorAll('.mode-tab[data-mode]').forEach(tab => {
    tab.addEventListener('click', () => switchPostMode(tab.dataset.mode, tab));
  });

  const postFile = document.getElementById('postFile');
  if (postFile) postFile.addEventListener('change', () => {
    const f = postFile.files[0];
    if (!f) return;
    state.postFile = f;
    document.getElementById('postFileName').textContent = f.name;
    const analyzeBtn = document.getElementById('postAnalyzeBtn');
    if (analyzeBtn) { analyzeBtn.disabled = false; analyzeBtn.classList.add('ready'); }
    parseColumnsFromFile(f);
  });
  const postAnalyzeBtn = document.getElementById('postAnalyzeBtn');
  if (postAnalyzeBtn) postAnalyzeBtn.addEventListener('click', () => runPostTraining('single'));

  const postFileData = document.getElementById('postFileData');
  if (postFileData) postFileData.addEventListener('change', () => {
    const f = postFileData.files[0];
    if (f) { state.postFileData = f; document.getElementById('postDataFileName').textContent = f.name; }
  });
  const postFilePred = document.getElementById('postFilePred');
  if (postFilePred) postFilePred.addEventListener('change', () => {
    const f = postFilePred.files[0];
    if (f) { state.postFilePred = f; document.getElementById('postPredFileName').textContent = f.name; }
  });
  const splitBtn = document.getElementById('postSplitAnalyzeBtn');
  if (splitBtn) splitBtn.addEventListener('click', () => runPostTraining('split'));

  const apiBtn = document.getElementById('apiTestBtn');
  if (apiBtn) apiBtn.addEventListener('click', runAPIBlackboxTest);

  const generateBtn = document.getElementById('generateSandboxBtn');
  if (generateBtn) generateBtn.addEventListener('click', runGenerateSandbox);

  const stressSandboxBtn = document.getElementById('stressSandboxRunBtn');
  if (stressSandboxBtn) stressSandboxBtn.addEventListener('click', runStressSandbox);

  const optimizeBtn = document.getElementById('postOptimizeBtn');
  if (optimizeBtn) optimizeBtn.addEventListener('click', runWhatIfPost);
}

function switchPostMode(mode, clickedTab) {
  document.querySelectorAll('.mode-tab[data-mode]').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.mode-panel').forEach(p => p.classList.remove('active'));
  clickedTab.classList.add('active');
  const panel = document.getElementById(`mode-${mode}`);
  if (panel) panel.classList.add('active');
}

async function parseColumnsFromFile(file) {
  try {
    const text = await file.text();
    const lines = text.trim().split('\n');
    if (!lines.length) return;
    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));

    const predSel = document.getElementById('postPredCol');
    const labelSel = document.getElementById('postLabelCol');
    if (!predSel || !labelSel) return;

    predSel.innerHTML = headers.map(h => `<option value="${escapeHtml(h)}">${escapeHtml(h)}</option>`).join('');
    labelSel.innerHTML = '<option value="">— None —</option>' +
      headers.map(h => `<option value="${escapeHtml(h)}">${escapeHtml(h)}</option>`).join('');

    const likely = headers.find(h => ['prediction', 'decision', 'outcome', 'result', 'approved', 'predicted'].includes(h.toLowerCase()));
    if (likely) predSel.value = likely;

    showEl('postColSelector');
  } catch (_) { /* non-CSV, skip */ }
}

async function runPostTraining(mode) {
  showEl('postLoading'); hideEl('postResults');

  const fd = new FormData();

  if (mode === 'single') {
    if (!state.postFile) { hideEl('postLoading'); showToast('Please upload a file.', 'error'); return; }
    fd.append('file', state.postFile);
    fd.append('prediction_col', document.getElementById('postPredCol')?.value || '');
    fd.append('label_col', document.getElementById('postLabelCol')?.value || '');
  } else if (mode === 'split') {
    if (!state.postFileData || !state.postFilePred) {
      hideEl('postLoading'); showToast('Please upload both files.', 'error'); return;
    }
    fd.append('file_data', state.postFileData);
    fd.append('file_pred', state.postFilePred);
    fd.append('prediction_col', document.getElementById('postSplitPredCol')?.value || '');
  }

  try {
    const res = await fetch('/api/posttrain', { method: 'POST', body: fd, credentials: 'include' });
    const data = await res.json();
    hideEl('postLoading');

    if (data.error) { showErrorInEl('postResults', data.error); return; }

    state.postMetrics = data.metrics;
    renderPostMetrics(data.metrics, data.alert_explanations || []);
    renderCharts(graphsToObj(data.graphs), 'postChartsGrid', 'post');
    appendGroqCard('postChartsGrid', data.groq_explanations);
    setProviderTag('postProviderTag', data.provider);
    renderAIReport(data.report, 'postReportContainer', data.report_id, data.report_error);
    populateSelect('postWhatIfAttr', data.metrics.protected_attributes || []);
    showEl('postResults');
    showToast('Post-training audit complete!', 'success');

  } catch (e) {
    hideEl('postLoading');
    showErrorInEl('postResults', 'Request failed: ' + e.message);
  }
}

function renderPostMetrics(m, alertExplanations) {
  const grid = document.getElementById('postMetricsGrid');
  if (!grid || !m) return;

  const spdObj = m.statistical_parity_difference || m.statistical_parity_diff || {};
  const diObj = m.disparate_impact || {};
  const spd = typeof spdObj === 'number' ? spdObj : (Object.values(spdObj)[0] ?? 0);
  const di = typeof diObj === 'number' ? diObj : (Object.values(diObj)[0] ?? 1);
  const rule80 = di >= 0.8;
  const bias = m.bias_score ?? 0;

  grid.innerHTML = `
    ${metricCard('Bias Score', bias, '/100', 'accent-blue', riskBadge(bias))}
    ${metricCard('Total Predictions', (m.total_predictions ?? 0).toLocaleString(), '', 'accent-green')}
    ${metricCard('Positive Rate', ((m.positive_rate ?? 0) * 100).toFixed(1), '%', 'accent-purple')}
    ${metricCard('Stat. Parity Diff', spd.toFixed(3), '', Math.abs(spd) > 0.1 ? 'accent-blue' : 'accent-green')}
    ${metricCard('Disparate Impact', di.toFixed(3), '', rule80 ? 'accent-green' : 'accent-blue',
    `<span class="badge badge-${rule80 ? 'low' : 'high'}">${rule80 ? '80% PASS' : '80% FAIL'}</span>`)}
    ${metricCard('Confidence', m.audit_confidence ?? '—', '', 'accent-orange', confidenceBadge(m.audit_confidence))}
  `;

  if (m.accuracy !== undefined) {
    grid.innerHTML += `
      ${metricCard('Accuracy', ((m.accuracy ?? 0) * 100).toFixed(1), '%', 'accent-green')}
      ${metricCard('F1 Score', (m.f1 ?? 0).toFixed(3), '', 'accent-purple')}
    `;
  }

  // FIX #1: Render rich alert explanation cards for post-training too
  if (alertExplanations && alertExplanations.length > 0) {
    renderAlertExplanationCards(grid, alertExplanations);
  }
}

// ── API Black-Box Test ──────────────────────────────────────
async function runAPIBlackboxTest() {
  const apiUrl = document.getElementById('apiUrl')?.value.trim();
  if (!apiUrl) { showToast('Please enter an API URL.', 'error'); return; }
  showEl('postLoading'); hideEl('postResults');

  try {
    const res = await fetch('/api/stress/api', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_url: apiUrl,
        token: document.getElementById('apiToken')?.value.trim() || '',
        sample_json: document.getElementById('apiSampleJson')?.value.trim() || '{}',
        protected_attr: document.getElementById('apiProtectedAttr')?.value.trim() || 'gender',
      }),
    });
    const data = await res.json();
    hideEl('postLoading');
    document.getElementById('postMetricsGrid').innerHTML = '';
    document.getElementById('postChartsGrid').innerHTML = '';
    document.getElementById('postReportContainer').innerHTML = '';
    renderStressResults(data, 'stressResults');
    showEl('postResults');
  } catch (e) {
    hideEl('postLoading');
    showErrorInEl('postResults', 'API test failed: ' + e.message);
  }
}

// ── Generate Sandbox with Groq ─────────────────────────────
async function runGenerateSandbox() {
  const attr = document.getElementById('stressSandboxAttr')?.value.trim() || 'gender';
  const btn = document.getElementById('generateSandboxBtn');

  if (btn) btn.disabled = true;
  showEl('sandboxGenerating');

  try {
    const res = await fetch('/api/stress/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ protected_attr: attr, domain: 'hiring' }),
    });
    const data = await res.json();

    if (data.code) {
      const codeEl = document.getElementById('stressSandboxCode');
      const jsonEl = document.getElementById('stressSandboxJson');
      if (codeEl) codeEl.value = data.code;
      if (jsonEl) jsonEl.value = typeof data.sample_json === 'string'
        ? data.sample_json
        : JSON.stringify(data.sample_json, null, 2);
      showToast('Groq generated test code!', 'success');
    } else {
      showToast('Groq unavailable — using fallback code', 'info');
    }
  } catch (e) {
    showToast('Generate failed: ' + e.message, 'error');
  } finally {
    if (btn) btn.disabled = false;
    hideEl('sandboxGenerating');
  }
}

async function runStressSandbox() {
  const code = document.getElementById('stressSandboxCode')?.value.trim();
  if (!code) { showToast('Please enter a predict() function.', 'error'); return; }
  showEl('stressLoading'); hideEl('stressResults');

  try {
    const res = await fetch('/api/stress/sandbox', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code,
        sample_json: document.getElementById('stressSandboxJson')?.value.trim() || '{}',
        protected_attr: document.getElementById('stressSandboxAttr')?.value.trim() || 'gender',
      }),
    });
    const data = await res.json();
    hideEl('stressLoading');
    renderStressResults(data, 'stressResults');
    showEl('stressResults');
  } catch (e) {
    hideEl('stressLoading');
    showErrorInEl('stressResults', 'Sandbox test failed: ' + e.message);
    showEl('stressResults');
  }
}

function renderStressResults(data, containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (data.error) {
    el.innerHTML = `<div class="error-banner">❌ ${escapeHtml(data.error)}</div>`;
    showEl(containerId); return;
  }
  const cb = data.counterfactual_bias_rate ?? data.counterfactual_rate ?? data.bias_score ?? 0;
  const risk = data.risk_level || 'UNKNOWN';
  const rates = data.approval_by_group || {};
  const cbPct = cb <= 1 ? (cb * 100).toFixed(1) : cb.toFixed(1);
  const cbColor = cb >= 40 ? 'var(--red)' : cb >= 20 ? 'var(--amber)' : 'var(--green)';

  const canvasId = `sandbox-chart-${Date.now()}`;

  let explanationHtml = '';
  if (data.groq_explanation) {
    explanationHtml = `<div style="margin-top:1rem">${formatAIExplanation(data.groq_explanation)}</div>`;
  }

  el.innerHTML = `<div class="sandbox-results-card animate-in">
    <div class="counterfactual-score">
      <div>
        <div class="score-label">Counterfactual Bias Score</div>
        <div class="score-value" style="color:${cbColor}">${cbPct}%</div>
        <div class="score-sub">Only protected attribute varied</div>
      </div>
      <div style="margin-left:auto;display:flex;flex-direction:column;align-items:flex-end;gap:.4rem">
        <span class="badge badge-${risk.toLowerCase()}">${risk} RISK</span>
        <span style="font-size:.75rem;color:var(--text-dim)">${Object.keys(rates).length} groups tested</span>
      </div>
    </div>
    <div class="sandbox-chart-wrap">
      <canvas id="${canvasId}"></canvas>
    </div>
    ${explanationHtml}
  </div>`;
  showEl(containerId);

  requestAnimationFrame(() => {
    const ctx = document.getElementById(canvasId);
    if (!ctx || !Object.keys(rates).length) return;
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const labels = Object.keys(rates);
    const values = Object.values(rates).map(v => parseFloat((v * 100).toFixed(1)));
    const palette = ['rgba(0,229,255,.75)','rgba(139,92,246,.75)','rgba(34,197,94,.75)','rgba(245,158,11,.75)'];
    const borders = ['#00e5ff','#8b5cf6','#22c55e','#f59e0b'];
    const gridColor = isDark ? 'rgba(255,255,255,.05)' : 'rgba(0,0,0,.06)';
    const tickColor = isDark ? '#718096' : '#4a5568';

    if (state.chartInstances['sandbox']) {
      state.chartInstances['sandbox'].destroy();
    }
    state.chartInstances['sandbox'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Approval Rate (%)',
          data: values,
          backgroundColor: palette,
          borderColor: borders,
          borderWidth: 1.5,
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: isDark ? '#0f172a' : '#fff',
            titleColor: isDark ? '#e2e8f0' : '#1e293b',
            bodyColor: isDark ? '#94a3b8' : '#475569',
            callbacks: { label: ctx => `${ctx.parsed.y}% approval rate` }
          }
        },
        scales: {
          x: { ticks: { color: tickColor }, grid: { color: gridColor } },
          y: { ticks: { color: tickColor, callback: v => v + '%' }, grid: { color: gridColor }, beginAtZero: true, max: 100 }
        }
      }
    });
  });
}

// ══════════════════════════════════════════════════════════
// BIAS OPTIMIZATION / WHAT-IF
// ══════════════════════════════════════════════════════════
async function runWhatIfPre() {
  if (!state.preFile || !state.preMetrics) {
    showToast('Run a Pre-Training audit first.', 'error'); return;
  }
  const attr = document.getElementById('preWhatIfAttr')?.value || '';
  const balance = parseFloat(document.getElementById('preBalanceSlider')?.value || 50);
  const resDiv = document.getElementById('preWhatIfResults');
  resDiv.innerHTML = inlineLoader('Simulating rebalancing…');
  showEl('preWhatIfResults');

  try {
    const df_data = await csvFileToJson(state.preFile);
    const res = await fetch('/api/whatif/pre', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ df_data, protected_attr: attr, desired_balance: balance }),
    });
    const data = await res.json();
    if (data.error) { resDiv.innerHTML = `<div class="error-banner">${escapeHtml(data.error)}</div>`; return; }
    renderWhatIfResult(data, 'preWhatIfResults');
  } catch (e) {
    resDiv.innerHTML = `<div class="error-banner">Error: ${escapeHtml(e.message)}</div>`;
  }
}

async function runWhatIfPost() {
  if (!state.postFile || !state.postMetrics) {
    showToast('Run a Post-Training audit (single file) first.', 'error'); return;
  }
  const attr = document.getElementById('postWhatIfAttr')?.value || '';
  const threshold = parseFloat(document.getElementById('postThreshSlider')?.value || 50) / 100;
  const fw = parseFloat(document.getElementById('postFairSlider')?.value || 50) / 100;
  const resDiv = document.getElementById('postWhatIfResults');
  resDiv.innerHTML = inlineLoader('Optimizing fairness…');
  showEl('postWhatIfResults');

  try {
    const df_data = await csvFileToJson(state.postFile);
    const res = await fetch('/api/whatif/post', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ df_data, protected_attr: attr, threshold, fairness_weight: fw }),
    });
    const data = await res.json();
    if (data.error) { resDiv.innerHTML = `<div class="error-banner">${escapeHtml(data.error)}</div>`; return; }
    renderWhatIfResult(data, 'postWhatIfResults');
  } catch (e) {
    resDiv.innerHTML = `<div class="error-banner">Error: ${escapeHtml(e.message)}</div>`;
  }
}

function renderWhatIfResult(data, containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;

  const before = data.before_score ?? 0;
  const after = data.after_score ?? 0;
  const imp = data.improvement_pct ?? data.improvement ?? Math.max(0, before - after);
  const improved = imp >= 0;
  const impColor = improved ? 'var(--green)' : 'var(--red)';

  const beforeRates = data.before_dist || data.before_rates || {};
  const afterRates = data.after_dist || data.after_rates || {};
  const groups = [...new Set([...Object.keys(beforeRates), ...Object.keys(afterRates)])];

  let ratesHtml = '';
  if (groups.length > 0) {
    ratesHtml = '<div class="group-rate-grid" style="margin-top:16px">';
    groups.forEach(g => {
      const b = ((beforeRates[g] ?? 0) * 100).toFixed(1);
      const a = ((afterRates[g] ?? 0) * 100).toFixed(1);
      ratesHtml += `<div class="group-rate-item">
        <div class="group-name">${escapeHtml(g)}</div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px">Before → After</div>
        <div class="group-rate">${b}% → ${a}%</div>
      </div>`;
    });
    ratesHtml += '</div>';
  }

  const groqHtml = data.groq_explanation ? formatOptimizerExplanation(data.groq_explanation) : '';

  el.innerHTML = `
    <div class="whatif-result-grid">
      <div class="before-after-card">
        <div class="ba-label">Before</div>
        <div class="ba-score baseline">${before}</div>
        <div class="metric-sub">Bias Score</div>
      </div>
      <div class="before-after-card">
        <div class="ba-label">After Optimization</div>
        <div class="ba-score improved">${after}</div>
        <div class="metric-sub">Bias Score</div>
      </div>
    </div>
    <div style="text-align:center;margin:12px 0">
      <span class="improvement-pill" style="color:${impColor}">
        ${improved ? '▼' : '▲'} ${Math.abs(imp).toFixed(1)}% Fairness ${improved ? 'Improvement' : 'Change'}
      </span>
    </div>
    ${ratesHtml}
    ${groqHtml}
  `;
  showEl(containerId);
}

// ══════════════════════════════════════════════════════════
// APPEAL ENGINE
// ══════════════════════════════════════════════════════════
function initAppeal() {
  document.querySelectorAll('.domain-btn[data-domain]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.domain-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeDomain = btn.dataset.domain;
    });
  });
  const runBtn = document.getElementById('appealRunBtn');
  if (runBtn) runBtn.addEventListener('click', runAppeal);
}

async function runAppeal() {
  showEl('appealLoading'); hideEl('appealResults');

  const fd = new FormData();
  fd.append('domain', state.activeDomain);
  fd.append('doc_text', document.getElementById('appealDocText')?.value.trim() || '');
  fd.append('policy_text', document.getElementById('appealPolicyText')?.value.trim() || '');
  fd.append('policy_url', document.getElementById('appealPolicyUrl')?.value.trim() || '');

  const docFile = document.getElementById('appealDocFile')?.files[0];
  const policyFile = document.getElementById('appealPolicyFile')?.files[0];
  if (docFile) fd.append('file_doc', docFile);
  if (policyFile) fd.append('file_policy', policyFile);

  try {
    const res = await fetch('/api/appeal', { method: 'POST', body: fd, credentials: 'include' });
    const data = await res.json();
    hideEl('appealLoading');

    if (data.error) { showErrorInEl('appealResults', data.error); return; }
    renderAppealResults(data);
    showEl('appealResults');
    showToast('Appeal analysis complete!', 'success');

  } catch (e) {
    hideEl('appealLoading');
    showErrorInEl('appealResults', 'Error: ' + e.message);
  }
}

function renderAppealResults(data) {
  const el = document.getElementById('appealOutput');
  if (!el) return;

  const fitScore = data.fit_score ?? 50;
  const reportText = data.report || '';
  const parsed = parseAppealReport(reportText);

  const rec = parsed.recommendation || 'Analysis Complete';
  const recCls = rec.toLowerCase().includes('strong') ? 'recommend'
    : rec.toLowerCase().includes('no appeal') ? 'no-appeal'
      : 'insufficient';

  const scoreColor = fitScore >= 70 ? 'var(--accent3, #22c55e)' : fitScore >= 40 ? 'var(--warn, #f59e0b)' : 'var(--danger, #ef4444)';
  const badgeCls = fitScore >= 70 ? 'low' : fitScore >= 40 ? 'medium' : 'high';

  const matchedHtml = parsed.matched.length
    ? parsed.matched.map(i => `<div class="appeal-list-item">✓ ${escapeHtml(i)}</div>`).join('')
    : '<div class="appeal-list-item" style="color:var(--muted)">None identified</div>';
  const missingHtml = parsed.missing.length
    ? parsed.missing.map(i => `<div class="appeal-list-item">✗ ${escapeHtml(i)}</div>`).join('')
    : '<div class="appeal-list-item" style="color:var(--muted)">None identified</div>';
  const reasonsHtml = parsed.reasons.length
    ? parsed.reasons.map(i => `<div class="appeal-list-item">⚠ ${escapeHtml(i)}</div>`).join('')
    : '<div class="appeal-list-item" style="color:var(--muted)">Not determined</div>';
  const planHtml = parsed.plan.length
    ? parsed.plan.map(i => `<div class="appeal-list-item">→ ${escapeHtml(i)}</div>`).join('')
    : '<div class="appeal-list-item" style="color:var(--muted)">No suggestions</div>';

  el.innerHTML = `
    <div class="fit-score-card animate-in">
      <div class="fit-score-label">Fit Score</div>
      <div class="fit-score-number" style="color:${scoreColor}">${fitScore}</div>
      <div class="fit-score-bar"><div class="fit-score-fill" style="width:${fitScore}%"></div></div>
      <span class="badge badge-${badgeCls}" style="margin-top:8px">${rec}</span>
    </div>

    <div class="appeal-result-grid animate-in">
      <div class="appeal-list-card">
        <div class="appeal-list-title" style="color:var(--accent3)">✅ Matched Requirements</div>
        ${matchedHtml}
      </div>
      <div class="appeal-list-card">
        <div class="appeal-list-title" style="color:var(--danger)">❌ Missing Requirements</div>
        ${missingHtml}
      </div>
      <div class="appeal-list-card">
        <div class="appeal-list-title" style="color:var(--warn)">⚠️ Likely Rejection Reasons</div>
        ${reasonsHtml}
      </div>
      <div class="appeal-list-card">
        <div class="appeal-list-title" style="color:var(--accent)">💡 Improvement Plan</div>
        ${planHtml}
      </div>
    </div>

    <div class="appeal-report-card animate-in">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
        <h3 style="font-size:16px;font-weight:700">Appeal Analysis</h3>
        <span class="appeal-rec ${recCls}">${escapeHtml(rec)}</span>
      </div>
      <div style="font-size:14px;color:var(--text-2);line-height:1.8">
        ${escapeHtml(parsed.summary || reportText).replace(/\n/g, '<br/>')}
      </div>
    </div>
  `;
}

function parseAppealReport(text) {
  const result = { matched: [], missing: [], reasons: [], plan: [], recommendation: '', summary: '' };
  if (!text) return result;

  let section = '';
  text.split('\n').forEach(line => {
    const l = line.trim();
    const lo = l.toLowerCase();
    if (!l) return;

    if (lo.startsWith('fit score')) { return; }
    if (lo.includes('matched requirement')) { section = 'matched'; return; }
    if (lo.includes('missing requirement')) { section = 'missing'; return; }
    if (lo.includes('rejection reason')) { section = 'reasons'; return; }
    if (lo.includes('improvement plan')) { section = 'plan'; return; }
    if (lo.includes('appeal recommendation')) { section = 'rec'; return; }
    if (lo.includes('appeal summary') || (lo.includes('summary') && !lo.includes('executive'))) { section = 'summary'; return; }

    const item = l.replace(/^[-•*\d.)\]]+\s*/, '').trim();
    if (!item || lo === 'none identified' || lo === 'none') return;

    switch (section) {
      case 'matched': result.matched.push(item); break;
      case 'missing': result.missing.push(item); break;
      case 'reasons': result.reasons.push(item); break;
      case 'plan': result.plan.push(item); break;
      case 'rec': if (!result.recommendation) result.recommendation = item; break;
      case 'summary': result.summary += (result.summary ? ' ' : '') + item; break;
    }
  });
  return result;
}

// ══════════════════════════════════════════════════════════
// REPORTS
// ══════════════════════════════════════════════════════════
function initReports() {
  document.getElementById('refreshReports')?.addEventListener('click', loadReports);
  document.getElementById('reportSearch')?.addEventListener('input', filterAndRenderReports);

  document.querySelectorAll('.filter-tab[data-filter]').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.activeFilter = tab.dataset.filter;
      filterAndRenderReports();
    });
  });

  document.getElementById('closeReportDetail')?.addEventListener('click', closeReportDetail);
  document.getElementById('reportDetailOverlay')?.addEventListener('click', e => {
    if (e.target.id === 'reportDetailOverlay') closeReportDetail();
  });

  document.getElementById('deleteConfirmBtn')?.addEventListener('click', executeDelete);
  document.getElementById('deleteCancelBtn')?.addEventListener('click', closeDeleteModal);
  document.getElementById('deleteConfirmModal')?.addEventListener('click', e => {
    if (e.target.id === 'deleteConfirmModal') closeDeleteModal();
  });
}

async function loadReports() {
  showEl('reportsLoading');
  document.getElementById('reportsGrid').innerHTML = '';

  try {
    const res = await fetch('/api/reports', { credentials: 'include' });
    const data = await res.json();
    hideEl('reportsLoading');
    state.allReports = data.reports || [];

    const countEl = document.getElementById('reportsCount');
    if (countEl) countEl.textContent = state.allReports.length;

    filterAndRenderReports();
  } catch (e) {
    hideEl('reportsLoading');
    document.getElementById('reportsGrid').innerHTML =
      `<div class="error-banner">Failed to load reports: ${escapeHtml(e.message)}</div>`;
  }
}

function filterAndRenderReports() {
  const search = (document.getElementById('reportSearch')?.value || '').toLowerCase();
  const modeF = state.activeFilter;

  const filtered = state.allReports.filter(r => {
    const name = (r.filename || r.file_name || '').toLowerCase();
    const nameOk = !search || name.includes(search);
    const modeOk = modeF === 'all' || r.mode === modeF;
    return nameOk && modeOk;
  });

  renderReportsList(filtered);
}

function renderReportsList(reports) {
  const grid = document.getElementById('reportsGrid');
  if (!grid) return;

  if (!reports || reports.length === 0) {
    grid.innerHTML = `<div class="no-reports">
      <div class="no-reports-icon">📭</div>
      <div class="no-reports-text">No reports found</div>
      <div class="no-reports-sub">Run a Pre-Training or Post-Training audit to generate your first report.</div>
    </div>`;
    return;
  }

  grid.innerHTML = reports.map(r => {
    const id = r._id || r.id || '';
    const filename = r.filename || r.file_name || 'Unknown';
    const mode = r.mode || 'pre-training';
    const modeLbl = mode === 'post-training' ? 'Post-Training' : mode === 'appeal' ? 'Appeal' : 'Pre-Training';
    const modeCls = mode === 'post-training' ? 'post' : mode === 'appeal' ? 'appeal' : '';
    const risk = r.risk_level || 'UNKNOWN';
    const riskCls = risk.toLowerCase();
    const score = r.bias_score ?? '—';
    const date = r.created_at
      ? new Date(r.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      : '—';

    return `<div class="report-card" id="report-card-${id}" data-id="${id}">
      <div class="report-card-header">
        <div class="report-card-title-row">
          <span class="report-mode-badge ${modeCls}">${modeLbl}</span>
          <span class="report-card-name" id="rname-${id}">${escapeHtml(filename)}</span>
        </div>
        <div class="report-card-actions">
          <button class="rc-icon-btn rc-rename" title="Rename" onclick="event.stopPropagation();startRename('${id}')">✏️</button>
          <button class="rc-icon-btn rc-delete" title="Delete" onclick="event.stopPropagation();confirmDelete('${id}', '${escapeHtml(filename).replace(/'/g,"\\'")}')">🗑</button>
        </div>
      </div>
      <div class="report-card-meta">
        <span class="rc-meta-item">📅 ${date}</span>
        <span class="rc-meta-sep">·</span>
        <span class="rc-meta-item">Score: <strong>${score}</strong></span>
        <span class="rc-meta-sep">·</span>
        <span class="badge badge-${riskCls}" style="font-size:.7rem">${risk}</span>
      </div>
      <div class="report-card-footer">
        <button class="rc-btn rc-btn-view" onclick="event.stopPropagation();viewReport('${id}')">👁 View</button>
        <button class="rc-btn rc-btn-dl" onclick="event.stopPropagation();downloadReportById('${id}')">⬇️ Download</button>
      </div>
    </div>`;
  }).join('');
}

// ── Delete Report ─────────────────────────────────────────
function confirmDelete(id, name) {
  const modal = document.getElementById('deleteConfirmModal');
  const nameEl = document.getElementById('deleteReportName');
  if (!modal) return;
  if (nameEl) nameEl.textContent = name;
  modal.dataset.targetId = id;
  modal.classList.remove('hidden');
  modal.classList.add('modal-visible');
}

function closeDeleteModal() {
  const modal = document.getElementById('deleteConfirmModal');
  if (modal) { modal.classList.add('hidden'); modal.classList.remove('modal-visible'); }
}

async function executeDelete() {
  const modal = document.getElementById('deleteConfirmModal');
  const id = modal?.dataset.targetId;
  if (!id) return;
  closeDeleteModal();

  try {
    const res = await fetch(`/api/reports/${id}`, { method: 'DELETE', credentials: 'include' });
    const data = await res.json();
    if (!res.ok || data.error) { showToast(data.error || 'Delete failed', 'error'); return; }

    const card = document.getElementById(`report-card-${id}`);
    if (card) {
      card.classList.add('report-card-deleting');
      setTimeout(() => {
        card.remove();
        state.allReports = state.allReports.filter(r => (r._id || r.id) !== id);
        const countEl = document.getElementById('reportsCount');
        if (countEl) countEl.textContent = state.allReports.length;
        if (!document.querySelector('.report-card')) {
          document.getElementById('reportsGrid').innerHTML = `<div class="no-reports">
            <div class="no-reports-icon">📭</div>
            <div class="no-reports-text">No reports found</div>
            <div class="no-reports-sub">Run an audit to generate your first report.</div>
          </div>`;
        }
      }, 350);
    }
    showToast('Report deleted', 'success');
  } catch (e) {
    showToast('Delete failed: ' + e.message, 'error');
  }
}

// ── Rename Report ─────────────────────────────────────────
function startRename(id) {
  const nameEl = document.getElementById(`rname-${id}`);
  if (!nameEl || nameEl.querySelector('input')) return;

  const current = nameEl.textContent.trim();
  nameEl.innerHTML = `
    <input class="rename-input" id="rename-input-${id}" type="text" value="${escapeHtml(current)}" maxlength="120"
      onclick="event.stopPropagation()"
      onkeydown="handleRenameKey(event,'${id}')"
    />
    <button class="rename-save-btn" title="Save" onclick="event.stopPropagation();saveRename('${id}')">✓</button>
    <button class="rename-cancel-btn" title="Cancel" onclick="event.stopPropagation();cancelRename('${id}','${escapeHtml(current).replace(/'/g,"\\'")}')">✕</button>
  `;
  const inp = document.getElementById(`rename-input-${id}`);
  if (inp) { inp.focus(); inp.select(); }
}

function handleRenameKey(e, id) {
  if (e.key === 'Enter') { e.preventDefault(); saveRename(id); }
  if (e.key === 'Escape') { const inp = document.getElementById(`rename-input-${id}`); cancelRename(id, inp ? inp.defaultValue : ''); }
}

function cancelRename(id, original) {
  const nameEl = document.getElementById(`rname-${id}`);
  if (nameEl) nameEl.innerHTML = escapeHtml(original);
}

async function saveRename(id) {
  const inp = document.getElementById(`rename-input-${id}`);
  const nameEl = document.getElementById(`rname-${id}`);
  const newName = inp ? inp.value.trim() : '';
  if (!newName) { showToast('Name cannot be empty', 'error'); return; }

  try {
    const res = await fetch(`/api/reports/${id}/rename`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ name: newName }),
    });
    const data = await res.json();
    if (!res.ok || data.error) { showToast(data.error || 'Rename failed', 'error'); cancelRename(id, inp?.defaultValue || ''); return; }

    if (nameEl) {
      nameEl.textContent = newName;
      nameEl.classList.add('rename-flash');
      setTimeout(() => nameEl.classList.remove('rename-flash'), 600);
    }
    const report = state.allReports.find(r => (r._id || r.id) === id);
    if (report) report.filename = newName;
    showToast('Report renamed', 'success');
  } catch (e) {
    showToast('Rename failed: ' + e.message, 'error');
    if (inp) cancelRename(id, inp.defaultValue);
  }
}

async function viewReport(id) {
  if (!id) return;
  showEl('reportsLoading');

  try {
    const res = await fetch(`/api/reports/${id}`, { credentials: 'include' });
    const data = await res.json();
    hideEl('reportsLoading');

    if (data.error) { showToast(data.error, 'error'); return; }

    const filename = data.filename || data.file_name || 'Report';
    document.getElementById('reportDetailTitle').textContent = `${filename} — ${data.mode || 'Audit'}`;
    renderDetailReport(data);
    showEl('reportDetailOverlay');

  } catch (e) {
    hideEl('reportsLoading');
    showToast('Failed to load report: ' + e.message, 'error');
  }
}

function closeReportDetail() {
  hideEl('reportDetailOverlay');
  Object.keys(state.chartInstances).filter(k => k.startsWith('modal-')).forEach(k => {
    state.chartInstances[k].destroy();
    delete state.chartInstances[k];
  });
}

function renderDetailReport(data) {
  const el = document.getElementById('reportDetailBody');
  if (!el) return;

  const metrics = data.metrics || {};
  const mode = data.mode || 'pre-training';
  const date = data.created_at ? new Date(data.created_at).toLocaleString() : '—';
  const filename = data.filename || data.file_name || 'Unknown';

  let metricsHtml = '';
  if (Object.keys(metrics).length > 0) {
    metricsHtml = '<div class="metrics-grid" style="margin:.75rem 0">';
    if (metrics.bias_score !== undefined) metricsHtml += metricCard('Bias Score', metrics.bias_score, '/100', 'accent-blue');
    if (metrics.total_rows !== undefined) metricsHtml += metricCard('Rows', metrics.total_rows, '', 'accent-green');
    if (metrics.total_predictions !== undefined) metricsHtml += metricCard('Predictions', metrics.total_predictions, '', 'accent-green');
    if (metrics.audit_confidence) metricsHtml += metricCard('Confidence', metrics.audit_confidence, '', 'accent-purple');
    metricsHtml += '</div>';
  }

  const modalChartsId = `modal-charts-${Date.now()}`;
  const savedGraphs = data.graph_data || (Array.isArray(data.graphs) ? graphsToObj(data.graphs) : (data.graphs || null));
  const hasCharts = savedGraphs && Object.keys(savedGraphs).length > 0;

  const reportContainerId = `modal-report-${Date.now()}`;

  // FIX #3: Show stored groq explanations and alert explanations in report detail
  let alertsHtml = '';
  if (data.alert_explanations && data.alert_explanations.length > 0) {
    alertsHtml = `<div style="margin: 1rem 0">
      <div class="results-section-header"><span class="section-icon">🚨</span> Bias Detection Alerts</div>
    </div>`;
  }

  let groqHtml = '';
  if (data.groq_explanations && data.groq_explanations.length > 0) {
    groqHtml = `<div style="margin: 1rem 0">
      <div class="results-section-header"><span class="section-icon">🤖</span> Graph Insights (Groq)</div>
      ${data.groq_explanations.map(e =>
        `<div class="groq-card" style="margin-bottom:.5rem">
          <div class="groq-header"><span class="groq-icon">📊</span>${escapeHtml(e.chart || '')}</div>
          <div class="groq-body"><p>${escapeHtml(e.explanation || '')}</p></div>
        </div>`
      ).join('')}
    </div>`;
  }

  el.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;flex-wrap:wrap">
      <span class="report-mode-badge">${escapeHtml(mode)}</span>
      <span class="report-filename">📄 ${escapeHtml(filename)}</span>
      <span style="margin-left:auto;font-size:.8rem;color:var(--muted)">📅 ${date}</span>
    </div>
    ${metricsHtml}
    ${hasCharts ? `<div class="charts-grid" id="${modalChartsId}" style="margin:1rem 0"></div>` : ''}
    ${alertsHtml}
    ${groqHtml}
    <div id="${reportContainerId}"></div>
  `;

  if (hasCharts) {
    setTimeout(() => renderCharts(savedGraphs, modalChartsId, 'modal-'), 80);
  }

  // Render stored alert explanations
  if (data.alert_explanations && data.alert_explanations.length > 0) {
    const alertContainer = el.querySelector('[style*="margin: 1rem 0"]');
    if (alertContainer) {
      renderAlertExplanationCards(alertContainer, data.alert_explanations);
    }
  }

  const reportText = data.report || data.ai_report || '';
  if (reportText) renderAIReport(reportText, reportContainerId);
}

async function downloadReportById(id) {
  try {
    const res = await fetch(`/api/reports/${id}`, { credentials: 'include' });
    const data = await res.json();
    const reportText = data.report || data.ai_report || 'No report text.';

    // FIX #3: Include groq and alert data in download
    let alertSection = '';
    if (data.alert_explanations && data.alert_explanations.length > 0) {
      alertSection = '\n\nBIAS DETECTION ALERTS\n' + '='.repeat(40) + '\n';
      data.alert_explanations.forEach((a, i) => {
        alertSection += `\nAlert ${i+1}: ${a.column || 'Unknown'} (${a.severity || 'HIGH'})\n`;
        alertSection += (a.explanation || '') + '\n';
      });
    }

    let groqSection = '';
    if (data.groq_explanations && data.groq_explanations.length > 0) {
      groqSection = '\n\nGRAPH INSIGHTS (GROQ)\n' + '='.repeat(40) + '\n';
      data.groq_explanations.forEach(e => {
        groqSection += `\n${e.chart || ''}:\n${e.explanation || ''}\n`;
      });
    }

    const content = [
      'AI FAIRNESS AUDIT REPORT',
      '='.repeat(40),
      `File:       ${data.filename || data.file_name || '—'}`,
      `Mode:       ${data.mode || '—'}`,
      `Date:       ${data.created_at || '—'}`,
      `Bias Score: ${data.bias_score ?? '—'}`,
      `Risk Level: ${data.risk_level || '—'}`,
      '',
      reportText,
      alertSection,
      groqSection,
    ].join('\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `fairness-report-${id.slice(-8)}.txt`;
    a.click();
  } catch (e) { showToast('Download failed: ' + e.message, 'error'); }
}

// ══════════════════════════════════════════════════════════
// CHART RENDERING
// ══════════════════════════════════════════════════════════
function graphsToObj(arr) {
  if (!arr) return {};
  if (!Array.isArray(arr)) return arr;
  const obj = {};
  arr.forEach((g, i) => {
    const norm = Object.assign({}, g);
    if (norm.data && !norm.values) norm.values = norm.data;
    obj[g.attr || g.title || String(i)] = norm;
  });
  return obj;
}

function renderCharts(graphData, containerId, instancePrefix) {
  const container = document.getElementById(containerId);
  if (!container || !graphData || !Object.keys(graphData).length) return;
  container.innerHTML = '';

  const prefix = instancePrefix || containerId;
  Object.keys(state.chartInstances).filter(k => k.startsWith(prefix)).forEach(k => {
    state.chartInstances[k].destroy();
    delete state.chartInstances[k];
  });

  Object.entries(graphData).forEach(([key, graph], idx) => {
    const canvasId = `cv-${containerId}-${idx}`;
    const card = document.createElement('div');
    card.className = 'chart-card animate-in';
    card.innerHTML = `
      <div class="chart-title">📊 ${escapeHtml(graph.title || key)}</div>
      <div class="chart-canvas-wrap"><canvas id="${canvasId}"></canvas></div>
    `;
    container.appendChild(card);

    requestAnimationFrame(() => {
      const ctx = document.getElementById(canvasId);
      if (!ctx) return;

      const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
      const values = graph.values || graph.data || [];
      const isPie = graph.type === 'pie' || graph.type === 'doughnut';

      const palette = [
        { bg: 'rgba(99,179,237,.75)', bd: '#63b3ed' },
        { bg: 'rgba(183,148,244,.75)', bd: '#b794f4' },
        { bg: 'rgba(104,211,145,.75)', bd: '#68d391' },
        { bg: 'rgba(246,173,85,.75)', bd: '#f6ad55' },
        { bg: 'rgba(252,129,129,.75)', bd: '#fc8181' },
        { bg: 'rgba(118,169,250,.75)', bd: '#76a9fa' },
      ];

      const bgColors = isPie ? palette.slice(0, values.length).map(p => p.bg) : palette[0].bg;
      const borderColors = isPie ? palette.slice(0, values.length).map(p => p.bd) : palette[0].bd;
      const gridColor = isDark ? 'rgba(255,255,255,.05)' : 'rgba(0,0,0,.06)';
      const tickColor = isDark ? '#718096' : '#4a5568';
      const legendColor = isDark ? '#a0aec0' : '#4a5568';

      const instance = new Chart(ctx, {
        type: isPie ? 'doughnut' : 'bar',
        data: {
          labels: graph.labels || [],
          datasets: [{
            label: graph.title || key,
            data: values,
            backgroundColor: bgColors,
            borderColor: borderColors,
            borderWidth: 1.5,
            borderRadius: isPie ? 0 : 6,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 500, easing: 'easeOutQuart' },
          plugins: {
            legend: { labels: { color: legendColor, font: { family: 'DM Sans, sans-serif', size: 12 } } },
            tooltip: {
              backgroundColor: isDark ? '#0f172a' : '#fff',
              titleColor: isDark ? '#e2e8f0' : '#1e293b',
              bodyColor: isDark ? '#94a3b8' : '#475569',
              borderColor: isDark ? 'rgba(255,255,255,.1)' : 'rgba(0,0,0,.1)',
              borderWidth: 1,
            },
          },
          scales: isPie ? {} : {
            x: { ticks: { color: tickColor }, grid: { color: gridColor } },
            y: { ticks: { color: tickColor }, grid: { color: gridColor }, beginAtZero: true },
          },
        },
      });
      state.chartInstances[`${prefix}-${idx}`] = instance;
    });
  });
}

function appendGroqCard(containerId, explanations) {
  const container = document.getElementById(containerId);
  if (!container || !explanations || !explanations.length) return;

  const expMap = {};
  explanations.forEach(e => { expMap[e.chart] = e.explanation || ''; });

  const chartCards = container.querySelectorAll('.chart-card');
  chartCards.forEach(card => {
    const titleEl = card.querySelector('.chart-title');
    if (!titleEl) return;
    const title = titleEl.textContent.trim().replace(/^📊\s*/, '');
    const expText = expMap[title] || Object.entries(expMap).find(([k]) => title.includes(k.split(' ')[0]))?.[1] || '';
    if (expText) {
      const expEl = document.createElement('div');
      expEl.innerHTML = formatAIExplanation(expText);
      card.appendChild(expEl);
    }
  });

  const allTexts = explanations.map(e => e.explanation || '').filter(Boolean);
  if (allTexts.length === 0) return;

  const matched = container.querySelectorAll('.graph-explanation-card').length;
  if (matched === 0) {
    const card = document.createElement('div');
    card.className = 'groq-card animate-in';
    card.innerHTML = `
      <div class="groq-header">
        <span class="groq-icon">🤖</span>
        Graph Insights
        <span class="badge badge-info" style="margin-left:8px">Groq llama-3.3-70b</span>
      </div>
      <div class="groq-body">${allTexts.map(t => formatAIExplanation(t)).join('')}</div>
    `;
    container.appendChild(card);
  }
}

// ══════════════════════════════════════════════════════════
// AI REPORT RENDERING
// ══════════════════════════════════════════════════════════
function renderAIReport(reportText, containerId, reportId, reportError) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!reportText) {
    const errMsg = reportError
      ? `AI report generation failed: ${escapeHtml(reportError)}`
      : 'AI report unavailable. Check your API key configuration and Flask terminal for the real error.';
    container.innerHTML = `<div class="error-banner">❌ ${errMsg}</div>`;
    return;
  }

  const sections = parseReportSections(reportText);
  let html = `<div class="report-container animate-in">
    <div class="report-toolbar">
      <span>AI-Generated Fairness Report${reportId ? ` · ID: ${reportId.slice(-6)}` : ''}</span>
      <div>
        <button class="btn btn-sm btn-secondary" onclick="copyReport('${containerId}')">📋 Copy</button>
        <button class="btn btn-sm btn-secondary" onclick="downloadReport('${containerId}')" style="margin-left:6px">⬇️ Download</button>
      </div>
    </div>`;

  const icons = {
    'executive': '📌', 'summary': '📌',
    'dataset': '📦', 'composition': '📦',
    'pre-training': '⚠️', 'post-training': '📊',
    'group': '👥', 'fairness': '👥',
    'liability': '🎯', 'bias': '🎯',
    'legal': '⚖️', 'ethical': '⚖️',
    'future': '🔭', 'risk': '🔭',
    'recommendation': '💡', 'limitation': '📋',
    'audit': '🔍', 'verdict': '🏁', 'default': '📄',
  };

  sections.forEach(sec => {
    const lower = sec.title.toLowerCase();

    if (lower.includes('verdict')) {
      const verdict = extractVerdict(sec.body);
      const vCls = verdict === 'PASS' ? 'PASS' : verdict === 'FAIL' || verdict === 'CRITICAL' ? 'FAIL' : 'INCONCLUSIVE';
      const just = sec.body.replace(/\b(PASS|FAIL|INCONCLUSIVE|CRITICAL)\b\s*[—–\-]?\s*/i, '').trim();
      html += `<div class="verdict-card animate-in">
        <div class="verdict-label">Final Verdict</div>
        <div class="verdict-badge ${vCls}">${verdict}</div>
        ${just ? `<div class="verdict-justification">${escapeHtml(just)}</div>` : ''}
      </div>`;
      return;
    }

    const icon = Object.entries(icons).find(([k]) => lower.includes(k))?.[1] || icons.default;
    const isRec = lower.includes('recommendation');
    const bodyHtml = isRec ? formatRecommendations(sec.body) : formatReportBody(sec.body);

    html += `<div class="report-section-card animate-in">
      <div class="report-section-header">
        <span class="report-section-icon">${icon}</span>
        <span class="report-section-title">${escapeHtml(sec.title)}</span>
      </div>
      <div class="report-section-body">${bodyHtml}</div>
    </div>`;
  });

  html += '</div>';
  container.innerHTML = html;
}

function parseReportSections(text) {
  if (!text) return [];
  const lines = text.split('\n');
  const sections = [];
  let current = null;

  lines.forEach(line => {
    const numbered = line.match(/^(\d+)\.\s+(.+)$/);
    const hashed = line.match(/^#{1,3}\s+(.+)$/);
    const allCaps = !numbered && !hashed && line.trim().length > 5
      && line.trim() === line.trim().toUpperCase()
      && /[A-Z]/.test(line)
      && line.trim().length < 70;

    let title = null;
    if (numbered) title = numbered[2].trim();
    else if (hashed) title = hashed[1].trim();
    else if (allCaps) title = line.trim();

    if (title) {
      if (current && current.body.trim()) sections.push(current);
      current = { title, body: '' };
    } else if (current) {
      current.body += (current.body ? '\n' : '') + line;
    } else if (line.trim()) {
      current = { title: 'Executive Summary', body: line };
    }
  });

  if (current && current.body.trim()) sections.push(current);
  return sections.filter(s => s.body.trim());
}

function formatReportBody(text) {
  if (!text) return '';
  let html = '';
  let inList = false;
  text.trim().split('\n').forEach(line => {
    const t = line.trim();
    if (!t) { if (inList) { html += '</ul>'; inList = false; } html += '<br/>'; return; }
    if (/^[-•*]\s/.test(t)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${escapeHtml(t.slice(2).trim())}</li>`;
    } else {
      if (inList) { html += '</ul>'; inList = false; }
      html += `<p>${escapeHtml(t)}</p>`;
    }
  });
  if (inList) html += '</ul>';
  return html;
}

function formatRecommendations(text) {
  const items = text.split('\n').map(l => l.trim()).filter(l => /^[-•*]\s/.test(l));
  if (!items.length) return formatReportBody(text);
  return '<ul class="rec-list">' +
    items.map(i => `<li class="rec-item"><span class="rec-check">✓</span>${escapeHtml(i.slice(2).trim())}</li>`).join('') +
    '</ul>';
}

function extractVerdict(body) {
  const m = body.match(/\b(PASS|FAIL|INCONCLUSIVE|CRITICAL)\b/i);
  return m ? m[1].toUpperCase() : 'INCONCLUSIVE';
}

function copyReport(containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;
  navigator.clipboard.writeText(el.innerText)
    .then(() => showToast('Report copied!', 'success'))
    .catch(() => showToast('Copy failed.', 'error'));
}

function downloadReport(containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const blob = new Blob([el.innerText], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `fairness-report-${Date.now()}.txt`;
  a.click();
}

// ══════════════════════════════════════════════════════════
// UI COMPONENT HELPERS
// ══════════════════════════════════════════════════════════
function metricCard(label, value, suffix, accentClass, extra = '') {
  return `<div class="metric-card ${accentClass} animate-in">
    <div class="metric-label">${escapeHtml(String(label))}</div>
    <div class="metric-value">${escapeHtml(String(value))}${suffix ? `<span style="font-size:14px;color:var(--muted)">${escapeHtml(suffix)}</span>` : ''}</div>
    ${extra ? `<div style="margin-top:8px">${extra}</div>` : ''}
  </div>`;
}

function riskBadge(score) {
  const [label, cls] = score >= 60 ? ['CRITICAL', 'critical'] : score >= 35 ? ['HIGH', 'high'] : score >= 15 ? ['MEDIUM', 'medium'] : ['LOW', 'low'];
  return `<span class="badge badge-${cls}">${label}</span>`;
}

function alertBadge(count, score) {
  const cls = count > 0 ? getRiskClass(score) : 'low';
  return count > 0
    ? `<span class="badge badge-${cls}">${count} found</span>`
    : `<span class="badge badge-low">None</span>`;
}

function confidenceBadge(conf) {
  const map = { HIGH: 'low', MEDIUM: 'medium', LOW: 'high' };
  const cls = map[conf] || 'medium';
  return conf ? `<span class="badge badge-${cls}">${escapeHtml(conf)}</span>` : '';
}

function getRiskClass(score) {
  return score >= 60 ? 'critical' : score >= 35 ? 'high' : score >= 15 ? 'medium' : 'low';
}

function setProviderTag(elId, provider) {
  const el = document.getElementById(elId);
  if (el && provider) el.textContent = provider.toUpperCase();
}

function populateSelect(selectId, attrs) {
  const sel = document.getElementById(selectId);
  if (!sel) return;
  sel.innerHTML = attrs.length
    ? attrs.map(a => `<option value="${escapeHtml(a)}">${escapeHtml(a)}</option>`).join('')
    : '<option value="">—</option>';
}

function showEl(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('hidden');
}

function hideEl(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('hidden');
}

function showErrorInEl(containerId, message) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = `<div class="error-banner">❌ ${escapeHtml(message)}</div>`;
  showEl(containerId);
}

function inlineLoader(text) {
  return `<div class="loading-card"><div class="spinner"></div><div class="loading-text">${escapeHtml(text)}</div></div>`;
}

function escapeHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatAIExplanation(text) {
  if (!text || !text.trim()) return '';

  const sections = {
    'Key Finding': { cls: 'key', label: '🔍 Key Finding' },
    'Evidence': { cls: 'evidence', label: '📊 Evidence' },
    'Risk': { cls: 'risk', label: '⚠️ Risk' },
    'Recommendation': { cls: 'rec', label: '💡 Recommendation' },
    'Bias Found': { cls: 'found', label: '🔴 Bias Found' },
    'Optimization Effect': { cls: 'effect', label: '⚡ Optimization Effect' },
    'Reliability Check': { cls: 'reliability', label: '🔎 Reliability' },
  };

  const lines = text.split('\n');
  let blocks = [];
  let currentKey = null;
  let currentText = '';

  lines.forEach(line => {
    const trimmed = line.trim();
    if (!trimmed) return;

    let matched = false;
    for (const [key] of Object.entries(sections)) {
      if (trimmed.toLowerCase().startsWith(key.toLowerCase() + ':') ||
          trimmed.toLowerCase() === key.toLowerCase() + ':') {
        if (currentKey && currentText.trim()) {
          blocks.push({ key: currentKey, text: currentText.trim() });
        }
        currentKey = key;
        currentText = trimmed.replace(new RegExp('^' + key + ':\\s*', 'i'), '');
        matched = true;
        break;
      }
    }
    if (!matched && currentKey) {
      currentText += (currentText ? ' ' : '') + trimmed;
    }
  });
  if (currentKey && currentText.trim()) {
    blocks.push({ key: currentKey, text: currentText.trim() });
  }

  if (!blocks.length) {
    return `<div class="graph-explanation-card"><div class="explanation-block" style="grid-column:1/-1">
      <div class="explanation-label key">🤖 AI Insight</div>
      <div class="explanation-text">${escapeHtml(text.trim())}</div>
    </div></div>`;
  }

  let html = '<div class="graph-explanation-card"><div class="explanation-row">';
  blocks.forEach((b, i) => {
    const sec = sections[b.key] || { cls: 'key', label: b.key };
    html += `<div class="explanation-block">
      <div class="explanation-label ${sec.cls}">${sec.label}</div>
      <div class="explanation-text">${escapeHtml(b.text)}</div>
    </div>`;
  });
  html += '</div></div>';
  return html;
}

function formatOptimizerExplanation(text) {
  if (!text || !text.trim()) return '';

  const sections = {
    'Bias Found': { cls: 'found', label: '🔴 Bias Found' },
    'Optimization Effect': { cls: 'effect', label: '⚡ Optimization Effect' },
    'Reliability Check': { cls: 'reliability', label: '🔎 Reliability Check' },
    'Recommendation': { cls: 'action', label: '💡 Recommendation' },
  };

  const lines = text.split('\n');
  let blocks = [];
  let currentKey = null;
  let currentText = '';

  lines.forEach(line => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let matched = false;
    for (const [key] of Object.entries(sections)) {
      if (trimmed.toLowerCase().startsWith(key.toLowerCase() + ':') ||
          trimmed.toLowerCase() === key.toLowerCase() + ':') {
        if (currentKey && currentText.trim()) blocks.push({ key: currentKey, text: currentText.trim() });
        currentKey = key;
        currentText = trimmed.replace(new RegExp('^' + key + ':\\s*', 'i'), '');
        matched = true;
        break;
      }
    }
    if (!matched && currentKey) currentText += (currentText ? ' ' : '') + trimmed;
  });
  if (currentKey && currentText.trim()) blocks.push({ key: currentKey, text: currentText.trim() });

  if (!blocks.length) return `<div class="optimizer-ai-card"><div class="optimizer-ai-header">🤖 AI Explanation</div>
    <div style="padding:.9rem 1.1rem;font-size:.82rem;color:var(--text-muted);line-height:1.6">${escapeHtml(text.trim())}</div></div>`;

  let html = `<div class="optimizer-ai-card"><div class="optimizer-ai-header">🤖 AI Analysis (Groq)</div><div class="optimizer-ai-grid">`;
  blocks.forEach(b => {
    const sec = sections[b.key] || { cls: 'found', label: b.key };
    html += `<div class="optimizer-ai-block">
      <div class="optimizer-ai-label ${sec.cls}">${sec.label}</div>
      <div class="optimizer-ai-text">${escapeHtml(b.text)}</div>
    </div>`;
  });
  html += '</div></div>';
  return html;
}

async function csvFileToJson(file) {
  try {
    const text = await file.text();
    const rows = text.trim().split('\n');
    if (rows.length < 2) return [];
    const headers = rows[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
    return rows.slice(1).map(row => {
      const vals = row.split(',');
      const obj = {};
      headers.forEach((h, i) => { obj[h] = (vals[i] || '').trim().replace(/^"|"$/g, ''); });
      return obj;
    });
  } catch (_) { return []; }
}

// ══════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ══════════════════════════════════════════════════════════
let _toastTimer = null;
function showToast(message, type = 'info') {
  const toast = document.getElementById('toast');
  const msg = document.getElementById('toastMsg');
  if (!toast || !msg) return;
  msg.textContent = message;
  toast.className = `toast toast-${type}`;
  toast.classList.remove('hidden');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => toast.classList.add('hidden'), 3500);
}