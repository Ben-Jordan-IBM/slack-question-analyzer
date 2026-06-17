// Modal shell + upload, history, and settings flows.
function Modal({ open, onClose, children, width = 480 }) {
  const [render, setRender] = React.useState(open);
  const [vis, setVis] = React.useState(false);
  React.useEffect(() => {
    if (open) { setRender(true); const r = requestAnimationFrame(() => setVis(true)); return () => cancelAnimationFrame(r); }
    setVis(false); const id = setTimeout(() => setRender(false), 240); return () => clearTimeout(id);
  }, [open]);
  React.useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    if (open) window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);
  if (!render) return null;
  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
      <div onClick={onClose} style={{ position: 'absolute', inset: 0, background: 'rgba(22,22,22,.55)', opacity: vis ? 1 : 0, transition: 'opacity 220ms var(--ease-entrance)' }} />
      <div style={{ position: 'relative', width, maxWidth: '92vw', background: '#fff', boxShadow: 'var(--shadow-overlay)', opacity: vis ? 1 : 0, transform: vis ? 'none' : 'translateY(14px) scale(.97)', transition: 'opacity 240ms var(--ease-entrance), transform 240ms var(--ease-entrance)' }}>
        {children}
      </div>
    </div>
  );
}

function ModalHead({ title, sub, onClose }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', padding: '22px 24px 16px' }}>
      <div>
        <div style={{ fontSize: 20, fontWeight: 400, letterSpacing: '-.01em' }}>{title}</div>
        {sub ? <div style={{ fontSize: 13.5, color: 'var(--text-secondary)', marginTop: 6, lineHeight: 1.45, maxWidth: 380 }}>{sub}</div> : null}
      </div>
      <button onClick={onClose} aria-label="Close" style={{ width: 32, height: 32, border: 'none', background: 'transparent', cursor: 'pointer', color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flex: '0 0 auto' }}>
        <Icon name="x" size={18} />
      </button>
    </div>
  );
}

// ---- Upload transcript ----
function UploadModal({ open, onClose, onImported }) {
  const { Button, FileDropzone } = window.QuestionAnalyzerDesignSystem_03a921;
  const [file, setFile] = React.useState(null);
  const [phase, setPhase] = React.useState('pick'); // pick | running | done | error
  const [progress, setProgress] = React.useState(0);
  const [results, setResults] = React.useState(null);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    if (!open) {
      setFile(null);
      setPhase('pick');
      setProgress(0);
      setResults(null);
      setError(null);
    }
  }, [open]);

  const steps = ['Parsing transcript', 'Extracting questions', 'Embedding & grouping', 'Ranking by frequency'];
  const [activeStep, setActiveStep] = React.useState(0);

  // Map backend progress events onto the step list and percent bar.
  const onProgress = ({ stage, completed, total }) => {
    const share = total > 0 ? completed / total : 1;
    if (stage === 'queued') { setActiveStep(0); setProgress(2); }
    else if (stage === 'starting') { setActiveStep(0); setProgress(3); }
    else if (stage === 'extracting') { setActiveStep(1); setProgress(completed ? 6 : 4); }
    else if (stage === 'detecting') { setActiveStep(1); setProgress(Math.round(6 + share * 4)); }
    else if (stage === 'embedding') { setActiveStep(2); setProgress(Math.round(12 + share * 70)); }
    else if (stage === 'grouping') { setActiveStep(3); setProgress(84); }
    else if (stage === 'keywords') { setActiveStep(3); setProgress(86); }
    else if (stage === 'labeling') { setActiveStep(3); setProgress(Math.round(86 + share * 8)); }
    else if (stage === 'answers') { setActiveStep(3); setProgress(Math.round(94 + share * 4)); }
    else if (stage === 'summarizing') { setActiveStep(3); setProgress(99); }
    else if (stage === 'complete') { setActiveStep(3); setProgress(100); }
  };

  const jobIdRef = React.useRef(null);

  const run = async () => {
    if (!file) return;

    setPhase('running');
    setProgress(0);
    setActiveStep(0);
    setError(null);
    jobIdRef.current = null;

    try {
      const settings = window.QA_SETTINGS.get();

      // Fail fast with a clear message if the backend/provider isn't ready
      const status = await window.QA_API.health(settings.provider).catch(() => {
        throw new Error('Cannot reach the analyzer backend. Start it with: python api_server.py');
      });
      if (status.status !== 'ok') {
        throw new Error(status.message || 'The analysis backend is not ready.');
      }

      // The File goes to the backend as-is (zips are unpacked server-side)
      const data = await window.QA_API.analyze(file, settings, onProgress,
        (jobId) => { jobIdRef.current = jobId; });

      setProgress(100);
      setResults(data);

      // Store results globally for dashboard to use
      window.ANALYSIS_RESULTS = data;

      setTimeout(() => setPhase('done'), 300);

    } catch (err) {
      jobIdRef.current = null;
      if (err.cancelled) {
        setCancelling(false);
        setPhase('pick');  // back to file selection, not an error state
        return;
      }
      setCancelling(false);
      setError(err.message);
      setPhase('error');
      console.error('Analysis error:', err);
    }
  };

  const [cancelling, setCancelling] = React.useState(false);
  const cancel = () => {
    if (jobIdRef.current) {
      setCancelling(true);  // instant feedback: the backend stops at the next LLM call
      window.QA_API.cancelJob(jobIdRef.current);
    }
  };

  return (
    <Modal open={open} onClose={onClose} width={520}>
      <ModalHead title="Upload transcript" sub="Drop the JSON export your Slack bot produces. Questions are extracted, grouped, and merged into your dashboard." onClose={onClose} />
      <div style={{ padding: '0 24px 24px' }}>
        {phase === 'pick' ? (
          <React.Fragment>
            <FileDropzone fileName={file ? file.name : null} accept=".json,.txt,.csv,.zip"
              title="Drop a transcript export here or click to browse" hint="JSON, TXT, CSV — or a zipped Slack export"
              onFile={(f) => setFile(f)} onClear={() => setFile(null)} />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 20 }}>
              <Button variant="ghost" onClick={onClose}>Cancel</Button>
              <Button variant="primary" disabled={!file} icon={<Icon name="sparkles" size={16} />} onClick={run}>Analyze</Button>
            </div>
          </React.Fragment>
        ) : null}

        {phase === 'running' ? (
          <div style={{ padding: '8px 0 4px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 8 }}>
              <span>{steps[activeStep]}…</span><span style={{ fontFamily: 'var(--font-mono)' }}>{progress}%</span>
            </div>
            <div style={{ height: 4, background: 'var(--gray-20)', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${progress}%`, background: 'var(--blue-60)', transition: 'width 40ms linear' }} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 18 }}>
              {steps.map((s, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13, color: i <= activeStep ? 'var(--text-primary)' : 'var(--text-placeholder)', transition: 'color 200ms' }}>
                  <span style={{ width: 16, height: 16, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', color: i < activeStep ? 'var(--green-60)' : 'var(--blue-60)' }}>
                    {i < activeStep ? <Icon name="check" size={14} /> : i === activeStep ? <Icon name="loader" size={14} /> : <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--gray-30)' }} />}
                  </span>
                  {s}
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginTop: 18 }}>
              <span style={{ fontSize: 11.5, color: 'var(--text-helper)', lineHeight: 1.4 }}>
                First analysis warms up the AI models and can take a few minutes; repeat runs are much faster.
              </span>
              <Button variant="ghost" icon={<Icon name="x" size={15} />} onClick={cancel} disabled={cancelling}>{cancelling ? 'Cancelling — stops at the next AI call…' : 'Cancel analysis'}</Button>
            </div>
          </div>
        ) : null}

        {phase === 'done' && results ? (
          <div style={{ textAlign: 'center', padding: '12px 0 4px' }}>
            <div className="qa-pop" style={{ width: 56, height: 56, borderRadius: '50%', background: 'var(--green-60)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
              <Icon name="check" size={28} color="#fff" />
            </div>
            <div style={{ fontSize: 18, fontWeight: 400, marginBottom: 6 }}>Transcript analyzed</div>
            <div style={{ fontSize: 13.5, color: 'var(--text-secondary)', marginBottom: 22 }}>
              Found <b>{results.total_questions} questions</b> across <b>{results.total_groups} groups</b>.
              {results.groups && results.groups[0] ? ` ${results.groups[0].representative_question.split(' ').slice(0, 4).join(' ')}... is your most-asked.` : ''}
            </div>
            <Button variant="primary" fullWidth icon={<Icon name="arrow-right" size={16} />} onClick={() => onImported && onImported()}>View dashboard</Button>
          </div>
        ) : null}

        {phase === 'error' ? (
          <div style={{ textAlign: 'center', padding: '12px 0 4px' }}>
            <div className="qa-pop" style={{ width: 56, height: 56, borderRadius: '50%', background: 'var(--red-60)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
              <Icon name="x" size={28} color="#fff" />
            </div>
            <div style={{ fontSize: 18, fontWeight: 400, marginBottom: 6 }}>Analysis failed</div>
            <div style={{ fontSize: 13.5, color: 'var(--text-secondary)', marginBottom: 22, maxHeight: 120, overflow: 'auto', textAlign: 'left', background: 'var(--gray-10)', padding: 12, fontFamily: 'var(--font-mono)', fontSize: 12 }}>
              {error || 'Unknown error occurred'}
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <Button variant="ghost" fullWidth onClick={onClose}>Close</Button>
              <Button variant="primary" fullWidth onClick={() => { setPhase('pick'); setError(null); }}>Try again</Button>
            </div>
          </div>
        ) : null}
      </div>
    </Modal>
  );
}

// ---- Analysis history ----
function HistoryModal({ open, onClose, onLoad }) {
  const { Button } = window.QuestionAnalyzerDesignSystem_03a921;
  const [items, setItems] = React.useState(null); // null = loading
  const [error, setError] = React.useState(null);
  const [loadingId, setLoadingId] = React.useState(null);

  React.useEffect(() => {
    if (!open) return;
    setItems(null); setError(null); setLoadingId(null);
    window.QA_API.listAnalyses()
      .then(setItems)
      .catch((err) => setError(err.message));
  }, [open]);

  const pick = async (id) => {
    setLoadingId(id);
    try {
      const data = await window.QA_API.getAnalysis(id);
      onLoad(data);
    } catch (err) {
      setError(err.message);
      setLoadingId(null);
    }
  };

  const remove = async (e, id) => {
    e.stopPropagation();
    try {
      await window.QA_API.deleteAnalysis(id);
      setItems((current) => (current || []).filter((item) => item.id !== id));
    } catch (err) {
      setError(err.message);
    }
  };

  const when = (iso) => iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—';

  return (
    <Modal open={open} onClose={onClose} width={560}>
      <ModalHead title="Analysis history" sub="Every analysis is saved automatically. Load a past one to revisit it in the dashboard." onClose={onClose} />
      <div style={{ padding: '0 24px 24px' }}>
        {error ? (
          <div style={{ fontSize: 13, color: 'var(--red-60)', padding: '14px 16px', background: 'var(--gray-10)', borderLeft: '3px solid var(--red-60)', marginBottom: 16 }}>{error}</div>
        ) : null}

        {items === null && !error ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-secondary)', fontSize: 13.5, padding: '18px 0' }}>
            <Icon name="loader" size={16} /> Loading history…
          </div>
        ) : null}

        {items && items.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--text-helper)', fontSize: 13.5, padding: '26px 0 18px' }}>
            No saved analyses yet. Upload a transcript to create your first one.
          </div>
        ) : null}

        {items && items.length > 0 ? (
          <div style={{ border: '1px solid var(--border-subtle)', borderBottom: 'none', maxHeight: 360, overflowY: 'auto' }}>
            {items.map((item) => (
              <div key={item.id} role="button" tabIndex={0}
                onClick={() => loadingId === null && pick(item.id)}
                onKeyDown={(e) => { if (e.key === 'Enter' && loadingId === null) pick(item.id); }}
                style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%', padding: '14px 16px', background: '#fff', borderBottom: '1px solid var(--border-subtle)', cursor: loadingId ? 'wait' : 'pointer', fontFamily: 'var(--font-sans)', boxSizing: 'border-box' }}
                onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--gray-10)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = '#fff'; }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12 }}>
                    <span style={{ fontSize: 13.5, fontWeight: 500, color: 'var(--text-primary)' }}>{when(item.analyzed_at)}</span>
                    <span style={{ fontSize: 12, color: 'var(--text-helper)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
                      {loadingId === item.id ? 'Loading…' : `${item.total_questions} questions · ${item.total_groups} groups`}
                    </span>
                  </div>
                  {item.top_question ? (
                    <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      Top: {item.top_question}
                    </div>
                  ) : null}
                </div>
                <button onClick={(e) => remove(e, item.id)} title="Delete this analysis" aria-label="Delete this analysis"
                  style={{ width: 28, height: 28, flex: '0 0 auto', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-helper)' }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--red-60)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-helper)'; }}>
                  <Icon name="trash-2" size={15} />
                </button>
              </div>
            ))}
          </div>
        ) : null}

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 18 }}>
          <Button variant="ghost" onClick={onClose}>Close</Button>
        </div>
      </div>
    </Modal>
  );
}

// ---- Learned topics (the bank): rename, merge, delete ----
function TopicsModal({ open, onClose }) {
  const { Button } = window.QuestionAnalyzerDesignSystem_03a921;
  const [topics, setTopics] = React.useState(null);
  const [error, setError] = React.useState(null);

  const load = React.useCallback(() => {
    window.QA_API.listTopics().then(setTopics).catch((err) => setError(err.message));
  }, []);
  React.useEffect(() => { if (open) { setTopics(null); setError(null); load(); } }, [open, load]);

  const act = async (fn) => {
    try { await fn(); load(); } catch (err) { setError(err.message); }
  };
  const rename = (t) => {
    const name = window.prompt('Rename this topic:', t.topic);
    if (name && name.trim() && name.trim() !== t.topic) act(() => window.QA_API.renameTopic(t.id, name.trim()));
  };
  const remove = (t) => {
    if (window.confirm(`Delete the topic "${t.topic}" from the learned bank?`)) act(() => window.QA_API.deleteTopic(t.id));
  };
  const merge = (t) => {
    const name = window.prompt(`Merge "${t.topic}" into which topic? Type its exact name:`);
    if (!name || !name.trim()) return;
    const target = (topics || []).find((x) => x.topic && x.topic.toLowerCase() === name.trim().toLowerCase() && x.id !== t.id);
    if (!target) { setError(`No other topic named "${name.trim()}"`); return; }
    act(() => window.QA_API.mergeTopics(t.id, target.id));
  };

  const iconBtn = (title, name, onClick, color) => (
    <button title={title} aria-label={title} onClick={onClick}
      style={{ width: 26, height: 26, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-helper)' }}
      onMouseEnter={(e) => { e.currentTarget.style.color = color; }}
      onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-helper)'; }}>
      <Icon name={name} size={14} />
    </button>
  );

  return (
    <Modal open={open} onClose={onClose} width={620}>
      <ModalHead title="Learned topics" sub="Everything the analyzer knows from seeds and past analyses. Rename bad names, merge duplicates, delete junk — changes apply to all future analyses." onClose={onClose} />
      <div style={{ padding: '0 24px 24px' }}>
        {error ? <div style={{ fontSize: 13, color: 'var(--red-60)', padding: '12px 16px', background: 'var(--gray-10)', borderLeft: '3px solid var(--red-60)', marginBottom: 14 }}>{error}</div> : null}
        {topics === null && !error ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-secondary)', fontSize: 13.5, padding: '18px 0' }}><Icon name="loader" size={16} /> Loading…</div>
        ) : null}
        {topics && topics.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--text-helper)', fontSize: 13.5, padding: '26px 0 18px' }}>No learned topics yet — run an analysis first.</div>
        ) : null}
        {topics && topics.length > 0 ? (
          <div style={{ border: '1px solid var(--border-subtle)', borderBottom: 'none', maxHeight: 400, overflowY: 'auto' }}>
            {topics.map((t) => (
              <div key={t.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px', borderBottom: '1px solid var(--border-subtle)', background: '#fff' }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13.5, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.topic || '(unnamed)'}</div>
                  <div style={{ fontSize: 11.5, color: 'var(--text-helper)', fontFamily: 'var(--font-mono)' }}>
                    {t.question_count || 0} questions · {t.analysis_count || 0} analyses{t.last_seen ? ` · last ${t.last_seen}` : ''}
                  </div>
                </div>
                {iconBtn('Rename', 'pencil', () => rename(t), 'var(--blue-60)')}
                {iconBtn('Merge into another topic', 'git-merge', () => merge(t), 'var(--purple-60)')}
                {iconBtn('Delete', 'trash-2', () => remove(t), 'var(--red-60)')}
              </div>
            ))}
          </div>
        ) : null}
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
          <Button variant="ghost" onClick={onClose}>Close</Button>
        </div>
      </div>
    </Modal>
  );
}

// ---- Analysis settings (provider + similarity threshold) ----
function SettingsModal({ open, onClose }) {
  const { Button, Slider } = window.QuestionAnalyzerDesignSystem_03a921;
  const [settings, setSettings] = React.useState(window.QA_SETTINGS.get());
  const providers = [
    { key: 'ollama', label: 'Ollama', hint: 'Local & free' },
    { key: 'azure', label: 'Azure OpenAI', hint: 'Needs API key' },
    { key: 'openai', label: 'OpenAI', hint: 'Needs API key' },
  ];

  React.useEffect(() => {
    if (open) window.QA_SETTINGS.loadServerDefaults().then(setSettings);
  }, [open]);

  const save = () => { window.QA_SETTINGS.set(settings); onClose(); };

  return (
    <Modal open={open} onClose={onClose} width={480}>
      <ModalHead title="Analysis settings" sub="Applied to every new transcript analysis. Provider credentials are configured in the backend's .env file." onClose={onClose} />
      <div style={{ padding: '0 24px 24px' }}>
        <label style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'block', marginBottom: 8 }}>AI provider</label>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 24 }}>
          {providers.map((p) => {
            const active = settings.provider === p.key;
            return (
              <button key={p.key} onClick={() => setSettings({ ...settings, provider: p.key })}
                style={{ padding: '12px 10px', textAlign: 'left', cursor: 'pointer', fontFamily: 'var(--font-sans)', background: active ? 'var(--blue-60)' : 'var(--field)', color: active ? '#fff' : 'var(--text-primary)', border: `1px solid ${active ? 'var(--blue-60)' : 'var(--border-strong)'}`, transition: 'background var(--duration-base), border-color var(--duration-base)' }}>
                <div style={{ fontSize: 13.5, fontWeight: 500 }}>{p.label}</div>
                <div style={{ fontSize: 11.5, marginTop: 3, color: active ? 'rgba(255,255,255,.75)' : 'var(--text-helper)' }}>{p.hint}</div>
              </button>
            );
          })}
        </div>

        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13.5, color: 'var(--text-primary)', cursor: 'pointer', marginBottom: 14 }}>
          <input type="checkbox" checked={settings.threshold === 'auto'}
            onChange={(e) => setSettings({ ...settings, threshold: e.target.checked ? 'auto' : 0.75 })} />
          Auto similarity threshold <span style={{ fontSize: 11.5, color: 'var(--text-helper)' }}>(recommended — adapts to your embedding model)</span>
        </label>

        {settings.threshold !== 'auto' ? (
          <Slider label="Similarity threshold" value={Math.round(settings.threshold * 100)}
            min={50} max={100} step={1} format={(v) => `${v}%`}
            onChange={(v) => setSettings({ ...settings, threshold: v / 100 })} />
        ) : null}
        <div style={{ fontSize: 12, color: 'var(--text-helper)', margin: '10px 0 22px', lineHeight: 1.5 }}>
          Higher = stricter grouping (questions must be nearly identical). Lower = broader topics.
          Auto starts at a model-appropriate value and relaxes itself if nothing groups.
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" icon={<Icon name="check" size={16} />} onClick={save}>Save settings</Button>
        </div>
      </div>
    </Modal>
  );
}

Object.assign(window, { UploadModal, HistoryModal, TopicsModal, SettingsModal });
