// Shared API client for the analyzer backend.
// Uses relative URLs when served by the Flask server. Opening the page
// straight from disk (file://) is unsupported: the server rejects the
// forgeable 'null' origin, so use http://localhost:5000 instead — the
// server hosts this exact page.
(function () {
  const API_BASE = location.protocol === 'file:' ? 'http://localhost:5000' : '';
  const POLL_INTERVAL_MS = 600;

  async function getJSON(path) {
    const response = await fetch(`${API_BASE}${path}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.success === false) {
      const err = new Error(data.error || `Request failed (${response.status})`);
      err.status = response.status; // callers distinguish permanent 4xx from outages
      throw err;
    }
    return data;
  }

  // Health check — resolves with the health payload (local Ollama status).
  function health() {
    return getJSON('/api/health');
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
  // Pass a week (YYYY-MM-DD, any date inside it) to review that calendar
  // week instead of the newest one.
  async function latestWeekly(week) {
    try {
      const query = week ? `?week=${encodeURIComponent(week)}` : '';
      const data = await getJSON(`/api/analyses/latest/weekly${query}`);
      return data.data;
    } catch (err) {
      return null;
    }
  }

  // A topic's volume over time across ALL saved analyses (overlap-safe).
  function topicHistory(topicId) {
    return getJSON(`/api/topics/${encodeURIComponent(topicId)}/history`);
  }

  // Save (or clear with '') the curated FAQ answer for a topic. Once a
  // human approves wording, every FAQ export uses it instead of a draft.
  async function setTopicAnswer(topicId, answer) {
    const response = await fetch(`${API_BASE}/api/topics/${encodeURIComponent(topicId)}/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answer }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.success) throw new Error(data.error || 'Update failed');
    return data;
  }

  // Mark a topic's FAQ as published (the date becomes a chart marker).
  async function setTopicPublished(topicId, published) {
    const response = await fetch(`${API_BASE}/api/topics/${encodeURIComponent(topicId)}/published`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ published }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.success) throw new Error(data.error || 'Update failed');
    return data;
  }

  // Start an analysis job and poll until it finishes.
  // `input` is a string of transcript text, a File, or an ARRAY of Files —
  // multiple files are merged server-side into one corpus (same as a zip).
  // onProgress receives {stage, completed, total}; onStarted receives the job id
  // (use it with cancelJob).
  async function analyze(input, { threshold = 'auto' } = {}, onProgress, onStarted) {
    let response;
    const files = input instanceof File ? [input]
      : (Array.isArray(input) && input.every((f) => f instanceof File) ? input : null);
    if (files) {
      const form = new FormData();
      files.forEach((f) => form.append('files', f));
      form.append('threshold', threshold);
      response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: form });
    } else {
      response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: input, threshold }),
      });
    }
    const started = await response.json().catch(() => ({}));
    if (!response.ok || !started.success) {
      throw new Error(started.error || `Could not start analysis (${response.status})`);
    }
    if (onStarted) onStarted(started.job_id);

    // A single failed poll must not report a running analysis as failed:
    // the job lives server-side (and survives restarts), so tolerate brief
    // outages (laptop sleep, server restart) with backoff before giving up.
    // Permanent 4xx answers (unknown job id) are NOT outages — fail fast
    // with the server's actual message instead of retrying into a generic one.
    const MAX_POLL_FAILURES = 6;
    let pollFailures = 0;
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS * (1 + pollFailures)));
      let job;
      try {
        job = await getJSON(`/api/jobs/${started.job_id}`);
      } catch (pollErr) {
        if (pollErr.status >= 400 && pollErr.status < 500) throw pollErr;
        pollFailures += 1;
        if (pollFailures >= MAX_POLL_FAILURES) {
          throw new Error('Lost contact with the analysis server ('
            + pollErr.message + '). The job may still be running — check '
            + 'History in a moment, or restart the app.');
        }
        continue;
      }
      pollFailures = 0;
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
    listTopics, deleteTopic, mergeTopics, renameTopic, topicHistory,
    setTopicPublished, setTopicAnswer };

  // ---- Analysis settings (threshold), persisted locally ----
  const SETTINGS_KEY = 'qa-analysis-settings';
  // 'auto' threshold: model-aware default that self-adjusts when nothing groups
  const DEFAULT_SETTINGS = { threshold: 'auto' };

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
        setSettings({ threshold: data.config.threshold });
      }
    } catch (err) { /* backend offline; local defaults apply */ }
    return getSettings();
  }

  window.QA_SETTINGS = { get: getSettings, set: setSettings, loadServerDefaults };
})();
