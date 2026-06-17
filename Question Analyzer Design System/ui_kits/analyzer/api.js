// Shared API client for the analyzer backend.
// Uses relative URLs when served by the Flask server; falls back to
// localhost:5000 when the page is opened directly from disk (file://).
(function () {
  const API_BASE = location.protocol === 'file:' ? 'http://localhost:5000' : '';
  const POLL_INTERVAL_MS = 600;

  async function getJSON(path) {
    const response = await fetch(`${API_BASE}${path}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.success === false) {
      throw new Error(data.error || `Request failed (${response.status})`);
    }
    return data;
  }

  // Health check — resolves with the health payload, including provider status.
  // Pass the provider that will be used so the right backend gets verified.
  function health(provider) {
    return getJSON(`/api/health${provider ? `?provider=${encodeURIComponent(provider)}` : ''}`);
  }

  // Most recent saved analysis, or null when none exist yet.
  // Also records its id (window.ANALYSIS_ID) so exports target what's shown.
  async function latestAnalysis() {
    try {
      const data = await getJSON('/api/analyses/latest');
      window.ANALYSIS_ID = data.id;
      return data.data;
    } catch (err) {
      return null;
    }
  }

  // Summaries of all saved analyses, newest first.
  async function listAnalyses() {
    const data = await getJSON('/api/analyses');
    return data.analyses;
  }

  // Full results of one saved analysis.
  async function getAnalysis(id) {
    const data = await getJSON(`/api/analyses/${encodeURIComponent(id)}`);
    window.ANALYSIS_ID = id;
    return data.data;
  }

  // Delete a saved analysis.
  async function deleteAnalysis(id) {
    const response = await fetch(`${API_BASE}/api/analyses/${encodeURIComponent(id)}`,
      { method: 'DELETE' });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.success === false) {
      throw new Error(data.error || `Delete failed (${response.status})`);
    }
  }

  // Download URL for a saved analysis (format: 'md' | 'csv' | 'json').
  function exportUrl(id, format) {
    return `${API_BASE}/api/analyses/${encodeURIComponent(id)}/export?format=${format}`;
  }

  // Week-in-Review stats for the latest analysis, or null when unavailable.
  async function latestWeekly() {
    try {
      const data = await getJSON('/api/analyses/latest/weekly');
      return data.data;
    } catch (err) {
      return null;
    }
  }

  // Start an analysis job and poll until it finishes.
  // `input` is either a string of transcript text or a File (.json/.txt/.csv/.zip).
  // onProgress receives {stage, completed, total}; onStarted receives the job id
  // (use it with cancelJob).
  async function analyze(input, { provider = 'ollama', threshold = 'auto' } = {}, onProgress, onStarted) {
    let response;
    if (input instanceof File) {
      const form = new FormData();
      form.append('files', input);
      form.append('provider', provider);
      form.append('threshold', threshold);
      response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: form });
    } else {
      response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: input, provider, threshold }),
      });
    }
    const started = await response.json().catch(() => ({}));
    if (!response.ok || !started.success) {
      throw new Error(started.error || `Could not start analysis (${response.status})`);
    }
    if (onStarted) onStarted(started.job_id);

    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      const job = await getJSON(`/api/jobs/${started.job_id}`);
      if (job.progress && onProgress) onProgress(job.progress);
      if (job.status === 'done') {
        window.ANALYSIS_ID = job.analysis_id;
        return job.data;
      }
      if (job.status === 'cancelled') {
        const err = new Error('Analysis cancelled');
        err.cancelled = true;
        throw err;
      }
      if (job.status === 'error') throw new Error(job.error || 'Analysis failed');
    }
  }

  // Backend version + configuration ({version, config}).
  function getConfig() {
    return getJSON('/api/config');
  }

  // The learned topic bank (known topics across analyses).
  async function listTopics() {
    const data = await getJSON('/api/topics');
    return data.topics;
  }

  // Remove a junk topic from the bank.
  async function deleteTopic(topicId) {
    const response = await fetch(`${API_BASE}/api/topics/${encodeURIComponent(topicId)}`,
      { method: 'DELETE' });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.success === false) throw new Error(data.error || 'Delete failed');
  }

  // Merge one topic into another (target keeps its name).
  async function mergeTopics(sourceId, targetId) {
    const response = await fetch(`${API_BASE}/api/topics/${encodeURIComponent(sourceId)}/merge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ into: targetId }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.success === false) throw new Error(data.error || 'Merge failed');
  }

  // Rename a learned topic in the bank (fixes a bad name for good).
  async function renameTopic(topicId, newName) {
    const response = await fetch(`${API_BASE}/api/topics/${encodeURIComponent(topicId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic: newName }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.success === false) {
      throw new Error(data.error || `Rename failed (${response.status})`);
    }
  }

  // Start downloading a missing Ollama model server-side.
  async function pullModel(model) {
    const response = await fetch(`${API_BASE}/api/models/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.success === false) {
      throw new Error(data.error || `Could not start download (${response.status})`);
    }
  }

  // Progress of a model download: {status, completed, total, detail}.
  function pullStatus(model) {
    return getJSON(`/api/models/pull/${encodeURIComponent(model)}`);
  }

  // Request cancellation of a queued or running job.
  async function cancelJob(jobId) {
    try {
      await fetch(`${API_BASE}/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
    } catch (err) { /* job may already be finishing; polling will resolve it */ }
  }

  window.QA_API = { health, getConfig, latestAnalysis, listAnalyses, getAnalysis,
    deleteAnalysis, exportUrl, latestWeekly, analyze, cancelJob, pullModel, pullStatus,
    listTopics, deleteTopic, mergeTopics, renameTopic };

  // ---- Analysis settings (provider + threshold), persisted locally ----
  const SETTINGS_KEY = 'qa-analysis-settings';
  // 'auto' threshold: model-aware default that self-adjusts when nothing groups
  const DEFAULT_SETTINGS = { provider: 'ollama', threshold: 'auto' };

  function getSettings() {
    try {
      const stored = JSON.parse(localStorage.getItem(SETTINGS_KEY));
      // Migration: older builds seeded a numeric threshold (0.65) here without
      // the user ever choosing one, and it silently overrode 'auto' forever.
      // A numeric threshold now only survives if the user set it themselves.
      if (stored && typeof stored.threshold === 'number' && !stored.userSetThreshold) {
        stored.threshold = 'auto';
      }
      return { ...DEFAULT_SETTINGS, ...(stored || {}) };
    } catch (err) {
      return { ...DEFAULT_SETTINGS };
    }
  }

  function setSettings(settings) {
    const merged = { ...getSettings(), ...settings };
    if (typeof merged.threshold === 'number') merged.userSetThreshold = true;
    else delete merged.userSetThreshold;
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(merged)); } catch (err) { /* private mode */ }
    return merged;
  }

  // Seed defaults from the server's configuration the first time.
  async function loadServerDefaults() {
    try {
      const data = await getJSON('/api/config');
      if (!localStorage.getItem(SETTINGS_KEY) && data.config) {
        setSettings({ provider: data.config.provider, threshold: data.config.threshold });
      }
    } catch (err) { /* backend offline; local defaults apply */ }
    return getSettings();
  }

  window.QA_SETTINGS = { get: getSettings, set: setSettings, loadServerDefaults };
})();
