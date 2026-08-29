/* ============================================================
   ImageGuard AI — Frontend Application
   MobileNetV3-Small + Hybrid CV Pipeline
   ============================================================ */

const elements = {
  fileInput:      document.querySelector('#fileInput'),
  dropZone:       document.querySelector('#dropZone'),
  emptyUpload:    document.querySelector('#emptyUpload'),
  previewWrap:    document.querySelector('#previewWrap'),
  previewImage:   document.querySelector('#previewImage'),
  previewName:    document.querySelector('#previewName'),
  previewSize:    document.querySelector('#previewSize'),
  removeFile:     document.querySelector('#removeFile'),
  analyzeButton:  document.querySelector('#analyzeButton'),
  buttonLabel:    document.querySelector('.button-label'),
  buttonLoader:   document.querySelector('.button-loader'),
  uploadError:    document.querySelector('#uploadError'),
  healthBadge:    document.querySelector('#healthBadge'),
  modelCaption:   document.querySelector('#modelCaption'),
  resultSection:  document.querySelector('#resultSection'),
  scoreRing:      document.querySelector('#scoreRing'),
  qualityScore:   document.querySelector('#qualityScore'),
  qualityLabel:   document.querySelector('#qualityLabel'),
  resultHeadline: document.querySelector('#resultHeadline'),
  resultSummary:  document.querySelector('#resultSummary'),
  analysisTime:   document.querySelector('#analysisTime'),
  resultImage:    document.querySelector('#resultImage'),
  issueCount:     document.querySelector('#issueCount'),
  issuesList:     document.querySelector('#issuesList'),
  statisticsGrid: document.querySelector('#statisticsGrid'),
  newAnalysis:    document.querySelector('#newAnalysis'),
  refreshHistory: document.querySelector('#refreshHistory'),
  historyList:    document.querySelector('#historyList'),
  historyEmpty:   document.querySelector('#historyEmpty'),
  historyPager:   document.querySelector('#historyPager'),
  loadMoreBtn:    document.querySelector('#loadMoreBtn'),
  historyMeta:    document.querySelector('#historyMeta'),
};

let selectedFile = null;
let previewUrl   = null;
let historyOffset = 0;
const HISTORY_LIMIT = 20;

const issueNames = {
  blur:               'Insufficient sharpness',
  underexposure:      'Underexposure',
  overexposure:       'Overexposure',
  noise:              'Image noise',
  severe_degradation: 'Severe degradation',
  potential_defect:   'Potential visual defect',
};

const issueIcons = {
  blur:               '◐',
  underexposure:      '☽',
  overexposure:       '☀',
  noise:              '▒',
  severe_degradation: '⚠',
  potential_defect:   '⬡',
};

// ── Utilities ──────────────────────────────────────────────

function formatBytes(bytes) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(0)} KB`
    : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function timeAgo(isoString) {
  const diff = Math.floor((Date.now() - new Date(isoString)) / 1000);
  if (diff < 60)   return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} hr ago`;
  return new Date(isoString).toLocaleDateString();
}

function showError(message) {
  elements.uploadError.textContent = message;
  elements.uploadError.hidden = false;
}

function clearError() {
  elements.uploadError.hidden = true;
  elements.uploadError.textContent = '';
}

function resetFile() {
  selectedFile = null;
  elements.fileInput.value = '';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  elements.emptyUpload.hidden = false;
  elements.previewWrap.hidden = true;
  elements.analyzeButton.disabled = true;
  clearError();
}

function selectFile(file) {
  clearError();
  const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
  if (!validTypes.includes(file.type)) {
    resetFile();
    showError('Choose a JPEG, PNG, or WebP image.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    resetFile();
    showError('The selected image exceeds the 10 MB limit.');
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  elements.previewImage.src = previewUrl;
  elements.previewName.textContent = file.name;
  elements.previewSize.textContent = formatBytes(file.size);
  elements.emptyUpload.hidden = true;
  elements.previewWrap.hidden = false;
  elements.analyzeButton.disabled = false;
}

// ── File input / drag-drop ─────────────────────────────────

elements.dropZone.addEventListener('click', () => elements.fileInput.click());
elements.dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); elements.fileInput.click(); }
});
elements.fileInput.addEventListener('change', () => {
  if (elements.fileInput.files[0]) selectFile(elements.fileInput.files[0]);
});
['dragenter', 'dragover'].forEach((n) =>
  elements.dropZone.addEventListener(n, (e) => { e.preventDefault(); elements.dropZone.classList.add('dragover'); })
);
['dragleave', 'drop'].forEach((n) =>
  elements.dropZone.addEventListener(n, (e) => { e.preventDefault(); elements.dropZone.classList.remove('dragover'); })
);
elements.dropZone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
});
elements.removeFile.addEventListener('click', (e) => { e.stopPropagation(); resetFile(); });

// ── Loading state ──────────────────────────────────────────

function setLoading(isLoading) {
  elements.analyzeButton.disabled = isLoading || !selectedFile;
  elements.buttonLoader.hidden = !isLoading;
  elements.buttonLabel.textContent = isLoading ? 'Analyzing…' : 'Analyze image';
}

// ── API helper ─────────────────────────────────────────────

async function apiFetch(url, options) {
  const response = await fetch(url, options);
  const payload  = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload?.error?.message || `Request failed (${response.status})`);
  return payload;
}

// ── Result rendering ───────────────────────────────────────

function resultTone(label) {
  if (label === 'ACCEPTABLE')         return { cls: '',          ring: '#087d73', headline: 'Ready to use' };
  if (label === 'DEGRADED')           return { cls: 'degraded',  ring: '#c77b13', headline: 'Quality needs attention' };
  return                                     { cls: 'defective', ring: '#bb3e4a', headline: 'Visual anomaly detected' };
}

function confidenceColor(c) {
  if (c >= 0.80) return '#bb3e4a';
  if (c >= 0.60) return '#c77b13';
  return '#087d73';
}

function renderIssues(issues) {
  elements.issueCount.textContent = `${issues.length} ${issues.length === 1 ? 'issue' : 'issues'}`;
  if (!issues.length) {
    elements.issuesList.innerHTML = '<div class="no-issues"><strong>No significant quality issue detected.</strong></div>';
    return;
  }
  elements.issuesList.innerHTML = issues.map((issue, idx) => {
    const pct        = Math.round(issue.confidence * 100);
    const rawPct     = Math.round((issue.model_probability ?? issue.confidence) * 100);
    const gateActive = issue.confidence > (issue.model_probability ?? issue.confidence) + 0.01;
    const evidenceText = issue.evidence?.[0] || 'Pattern detected by the learned model.';
    const color      = confidenceColor(issue.confidence);
    return `
      <div class="issue-item" style="--conf-color:${color}">
        <span class="issue-icon" aria-hidden="true">${issueIcons[issue.type] || '●'}</span>
        <div class="issue-main">
          <strong>${issueNames[issue.type] || issue.type}</strong>
          <p class="issue-evidence">${evidenceText}</p>
          ${gateActive ? `<span class="gate-badge">Heuristic gate active — model raw: ${rawPct}%</span>` : ''}
        </div>
        <div class="issue-right">
          <strong class="conf-pct" style="color:${color}">${pct}%</strong>
          <span class="severity-tag">${issue.severity}</span>
          <div class="conf-bar-wrap" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100">
            <div class="conf-bar" style="width:${pct}%;background:${color}"></div>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

const statMeta = {
  brightness_mean:     { label: 'Brightness', fmt: v => v.toFixed(3), lo: 0, hi: 1 },
  dark_pixel_ratio:    { label: 'Dark pixels', fmt: v => `${(v*100).toFixed(1)}%`, lo: 0, hi: 0.5 },
  highlight_clip_ratio:{ label: 'Highlights clipped', fmt: v => `${(v*100).toFixed(1)}%`, lo: 0, hi: 0.5 },
  laplacian_variance:  { label: 'Sharpness', fmt: v => v.toFixed(4), lo: 0, hi: 5000 },
  noise_estimate:      { label: 'Noise estimate', fmt: v => v.toFixed(4), lo: 0, hi: 0.15 },
  contrast_rms:        { label: 'Contrast', fmt: v => v.toFixed(3), lo: 0, hi: 1 },
  saturation_mean:     { label: 'Saturation', fmt: v => v.toFixed(3), lo: 0, hi: 1 },
  entropy:             { label: 'Entropy', fmt: v => v.toFixed(3), lo: 0, hi: 8 },
  blockiness:          { label: 'Blockiness', fmt: v => v.toFixed(4), lo: 0, hi: 0.1 },
};

function renderStatistics(stats) {
  const dimRow = `<div class="stat stat-full"><span>Dimensions</span><strong>${stats.width} × ${stats.height}</strong></div>`;
  const rows = Object.entries(statMeta).map(([key, meta]) => {
    if (stats[key] === undefined) return '';
    const val = stats[key];
    const pct = Math.min(100, Math.max(0, ((val - meta.lo) / (meta.hi - meta.lo)) * 100));
    return `
      <div class="stat">
        <span>${meta.label}</span>
        <strong>${meta.fmt(val)}</strong>
        <div class="stat-bar-wrap"><div class="stat-bar" style="width:${pct.toFixed(1)}%"></div></div>
      </div>
    `;
  }).join('');
  elements.statisticsGrid.innerHTML = dimRow + rows;
}

function animateRing(targetScore, ringColor) {
  elements.scoreRing.style.setProperty('--score', 0);
  elements.scoreRing.style.setProperty('--ring-color', ringColor);
  const start = performance.now();
  const duration = 900;
  function tick(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
    elements.scoreRing.style.setProperty('--score', (targetScore * eased).toFixed(2));
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function renderResult(result, scroll = true) {
  const tone = resultTone(result.quality_label);
  elements.resultSection.hidden = false;
  elements.qualityScore.textContent = Math.round(result.quality_score);
  animateRing(result.quality_score, tone.ring);
  elements.qualityLabel.textContent  = result.quality_label.replaceAll('_', ' ');
  elements.qualityLabel.className    = `quality-label ${tone.cls}`;
  elements.resultHeadline.textContent = tone.headline;
  elements.resultSummary.textContent  = result.issues.length
    ? `Primary finding: ${issueNames[result.issues[0].type].toLowerCase()} at ${Math.round(result.issues[0].confidence * 100)}% confidence.`
    : 'The model and measured image statistics found no significant quality concern.';
  elements.analysisTime.textContent = `${result.timing_ms.total.toFixed(0)} ms · ${result.model_name} v${result.model_version}`;
  elements.resultImage.src = `${result.image_url}?v=${Date.now()}`;
  renderIssues(result.issues);
  renderStatistics(result.statistics);
  if (scroll) elements.resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Upload & analyse ───────────────────────────────────────

elements.analyzeButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  clearError();
  setLoading(true);
  const form = new FormData();
  form.append('file', selectedFile);
  try {
    const result = await apiFetch('/api/v1/analyses', { method: 'POST', body: form });
    renderResult(result);
    historyOffset = 0;
    await loadHistory(false);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

elements.newAnalysis.addEventListener('click', () => {
  resetFile();
  document.querySelector('#analyze').scrollIntoView({ behavior: 'smooth' });
});

// ── History ────────────────────────────────────────────────

function historyLabelClass(label) {
  return label === 'ACCEPTABLE' ? '' : label === 'DEGRADED' ? 'degraded' : 'defective';
}

async function openHistoryItem(id) {
  try {
    const result = await apiFetch(`/api/v1/analyses/${id}`);
    renderResult(result);
  } catch (err) {
    showError(err.message);
  }
}

async function loadHistory(append = false) {
  try {
    const data = await apiFetch(`/api/v1/analyses?limit=${HISTORY_LIMIT}&offset=${historyOffset}`);
    const total = data.total || 0;
    const showing = historyOffset + data.items.length;

    if (!append) elements.historyList.innerHTML = '';
    elements.historyEmpty.hidden = total > 0;

    const fragment = data.items.map((item) => `
      <button class="history-item" type="button" data-id="${item.id}"
              aria-label="Open analysis for ${item.original_filename}">
        <img class="history-thumb" src="${item.image_url}" alt="" loading="lazy"
             onerror="this.style.background='#e8eee9'" />
        <span class="history-name">
          <strong>${item.original_filename}</strong>
          <span>${item.issues[0] ? issueNames[item.issues[0].type] : 'No issue detected'}</span>
        </span>
        <span class="quality-label ${historyLabelClass(item.quality_label)}">
          ${item.quality_label.replaceAll('_', ' ')}
        </span>
        <span class="history-score">
          <strong>${Math.round(item.quality_score)}/100</strong>
          <span>Quality</span>
        </span>
        <time class="history-date">${timeAgo(item.created_at)}</time>
        <span class="history-arrow" aria-hidden="true">›</span>
      </button>
    `).join('');
    elements.historyList.insertAdjacentHTML('beforeend', fragment);

    elements.historyList.querySelectorAll('[data-id]:not([data-bound])').forEach((btn) => {
      btn.dataset.bound = '1';
      btn.addEventListener('click', () => openHistoryItem(btn.dataset.id));
    });

    const hasMore = showing < total;
    elements.historyPager.hidden = !hasMore;
    if (hasMore) {
      elements.historyMeta.textContent = `Showing ${showing} of ${total}`;
    }
  } catch {
    elements.historyEmpty.hidden = false;
    elements.historyEmpty.querySelector('p').textContent = 'History could not be loaded. Try again.';
  }
}

elements.loadMoreBtn.addEventListener('click', () => {
  historyOffset += HISTORY_LIMIT;
  loadHistory(true);
});
elements.refreshHistory.addEventListener('click', () => {
  historyOffset = 0;
  loadHistory(false);
});

// ── Health check + model caption ──────────────────────────

async function checkHealth() {
  try {
    const health = await apiFetch('/health');
    elements.healthBadge.className = 'health-badge online';
    elements.healthBadge.innerHTML = '<span class="status-dot"></span>Model ready';
    if (elements.modelCaption && health.model_name) {
      elements.modelCaption.textContent =
        `${health.model_name} · v${health.model_version}`;
    }
  } catch {
    elements.healthBadge.className = 'health-badge offline';
    elements.healthBadge.innerHTML = '<span class="status-dot"></span>Model unavailable';
  }
}

checkHealth();
loadHistory(false);
