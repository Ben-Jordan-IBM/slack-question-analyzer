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
      <div style={{ position: 'relative', width, maxWidth: '92vw', background: 'var(--layer-02)', boxShadow: 'var(--shadow-overlay)', opacity: vis ? 1 : 0, transform: vis ? 'none' : 'translateY(14px) scale(.97)', transition: 'opacity 240ms var(--ease-entrance), transform 240ms var(--ease-entrance)' }}>
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
  // A LIST: drop several weekly exports and they merge into one corpus
  // server-side (the backend already accepted multiple files; only this
  // modal was single-file)
  const [files, setFiles] = React.useState([]);
  // Paste mode is the DEFAULT: copying a thread out of Slack is the
  // lowest-friction path (no export step); file upload sits one click
  // below. The backend accepts raw text either way.
  const [pasteMode, setPasteMode] = React.useState(true);
  const [pasteText, setPasteText] = React.useState('');
  const [phase, setPhase] = React.useState('pick'); // pick | running | done | error
  const [progress, setProgress] = React.useState(0);
  const [results, setResults] = React.useState(null);
  const [error, setError] = React.useState(null);

  // Bumped every time the modal closes: an in-flight run() from a previous
  // open must not set state (a finished job would repaint the "done" screen
  // onto the NEXT open of the modal)
  const runGeneration = React.useRef(0);
  React.useEffect(() => {
    if (!open) {
      runGeneration.current += 1;
      setFiles([]);
      setPasteMode(true);
      setPasteText('');
      setPhase('pick');
      setProgress(0);
      setResults(null);
      setError(null);
      setActiveStep(0);
      setCancelling(false);
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
    else if (stage === 'embedding') { setActiveStep(2); setProgress(Math.round(12 + share * 58)); }
    else if (stage === 'verifying') { setActiveStep(2); setProgress(Math.round(70 + share * 8)); }
    else if (stage === 'routing') { setActiveStep(2); setProgress(Math.round(78 + share * 6)); }
    else if (stage === 'grouping') { setActiveStep(3); setProgress(84); }
    else if (stage === 'keywords') { setActiveStep(3); setProgress(86); }
    else if (stage === 'labeling') { setActiveStep(3); setProgress(Math.round(86 + share * 8)); }
    else if (stage === 'answers') { setActiveStep(3); setProgress(Math.round(94 + share * 3)); }
    else if (stage === 'drafting') { setActiveStep(3); setProgress(Math.round(97 + share * 2)); }
    else if (stage === 'summarizing') { setActiveStep(3); setProgress(99); }
    else if (stage === 'complete') { setActiveStep(3); setProgress(100); }
  };

  const jobIdRef = React.useRef(null);

  const run = async () => {
    const input = pasteMode ? pasteText.trim() : files;
    if (pasteMode ? !input : !files.length) return;

    const generation = runGeneration.current;
    setPhase('running');
    setProgress(0);
    setActiveStep(0);
    setError(null);
    jobIdRef.current = null;

    try {
      const settings = window.QA_SETTINGS.get();

      // Fail fast with a clear message if the backend/provider isn't ready
      const status = await window.QA_API.health().catch(() => {
        throw new Error('Cannot reach the analyzer backend. Start it with: python api_server.py');
      });
      if (status.status !== 'ok') {
        throw new Error(status.message || 'The analysis backend is not ready.');
      }

      // Files go to the backend as-is (zips are unpacked server-side;
      // multiple files merge into one corpus); pasted text goes as the
      // raw transcript content
      const data = await window.QA_API.analyze(input, settings, onProgress,
        (jobId) => { jobIdRef.current = jobId; });

      // Results still land globally (the dashboard shows them on reopen),
      // but a run whose modal was closed must not repaint this one
      window.ANALYSIS_RESULTS = data;
      if (generation !== runGeneration.current) return;

      setProgress(100);
      setResults(data);

      setTimeout(() => { if (generation === runGeneration.current) setPhase('done'); }, 300);

    } catch (err) {
      jobIdRef.current = null;
      if (generation !== runGeneration.current) return;
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
      <ModalHead title="Add a transcript" sub="Paste a Slack thread straight from your channel, or upload an export file. Questions are extracted, grouped, and merged into your dashboard." onClose={onClose} />
      <div style={{ padding: '0 24px 24px' }}>
        {phase === 'pick' ? (
          <React.Fragment>
            {pasteMode ? (
              <React.Fragment>
                <textarea value={pasteText} onChange={(e) => setPasteText(e.target.value)}
                  rows={9} autoFocus
                  placeholder={'Paste a Slack thread or transcript here.\n\nCopy a whole thread straight out of Slack (names and timestamps included) and the first message becomes the question, the rest its replies.'}
                  style={{ width: '100%', boxSizing: 'border-box', resize: 'vertical', fontFamily: 'var(--font-sans)', fontSize: 13, lineHeight: 1.5, color: 'var(--text-primary)', background: 'var(--field)', border: '1px dashed var(--border-strong)', padding: '12px 14px' }} />
                <div style={{ fontSize: 11.5, color: 'var(--text-helper)', marginTop: 6 }}>
                  Also accepts the same text formats as file upload (dashed transcripts, Slack JSON, CSV).
                </div>
              </React.Fragment>
            ) : (
            <FileDropzone fileName={null} accept=".json,.txt,.csv,.zip" multiple
              title={files.length ? 'Drop another file to add it' : 'Drop a transcript export here or click to browse'}
              hint="JSON, TXT, CSV, a zipped Slack export — or several files, analyzed together"
              onFile={(f) => setFiles((prev) =>
                prev.some((p) => p.name === f.name && p.size === f.size
                  && p.lastModified === f.lastModified)
                  ? prev : [...prev, f])}
              onClear={() => {}} />
            )}
            <button onClick={() => setPasteMode(!pasteMode)}
              style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: 0, marginTop: 10, fontFamily: 'var(--font-sans)', fontSize: 12.5, color: 'var(--link)' }}>
              {pasteMode ? 'Or upload export files instead' : 'Or paste a Slack thread / text instead'}
            </button>
            {files.length && !pasteMode ? (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 12 }}>
                {files.map((f, i) => (
                  <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-secondary)', background: 'var(--field)', padding: '3px 8px' }}>
                    {f.name}
                    <button aria-label={`Remove ${f.name}`}
                      onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                      style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-helper)', padding: 0, display: 'inline-flex' }}>
                      <Icon name="x" size={11} />
                    </button>
                  </span>
                ))}
              </div>
            ) : null}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 20 }}>
              <Button variant="ghost" onClick={onClose}>Cancel</Button>
              <Button variant="primary" disabled={pasteMode ? !pasteText.trim() : !files.length}
                icon={<Icon name="sparkles" size={16} />} onClick={run}>
                {!pasteMode && files.length > 1 ? `Analyze ${files.length} files together` : 'Analyze'}
              </Button>
            </div>
          </React.Fragment>
        ) : null}

        {phase === 'running' ? (
          <div style={{ padding: '8px 0 4px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 8 }}>
              <span>{steps[activeStep]}…</span><span style={{ fontFamily: 'var(--font-mono)' }}>{progress}%</span>
            </div>
            <div style={{ height: 4, background: 'var(--layer-hover)', overflow: 'hidden' }}>
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
            <div style={{ fontSize: 13.5, color: 'var(--text-secondary)', marginBottom: 22, maxHeight: 120, overflow: 'auto', textAlign: 'left', background: 'var(--field)', padding: 12, fontFamily: 'var(--font-mono)', fontSize: 12 }}>
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
          <div style={{ fontSize: 13, color: 'var(--red-60)', padding: '14px 16px', background: 'var(--field)', borderLeft: '3px solid var(--red-60)', marginBottom: 16 }}>{error}</div>
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
                style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%', padding: '14px 16px', background: 'var(--layer-02)', borderBottom: '1px solid var(--border-subtle)', cursor: loadingId ? 'wait' : 'pointer', fontFamily: 'var(--font-sans)', boxSizing: 'border-box' }}
                onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--layer-hover)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'var(--layer-02)'; }}>
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
function TopicsModal({ open, onClose, onMutated }) {
  const { Button } = window.QuestionAnalyzerDesignSystem_03a921;
  const [topics, setTopics] = React.useState(null);
  const [error, setError] = React.useState(null);
  // Bank edits must reach an already-loaded dashboard: renames patch the
  // in-memory analysis directly, and any mutation remounts the views on
  // close (via onMutated) so labels/badges refresh without a re-upload
  const mutated = React.useRef(false);
  const close = () => {
    if (mutated.current && onMutated) onMutated();
    mutated.current = false;
    onClose();
  };

  const load = React.useCallback(() => {
    window.QA_API.listTopics().then(setTopics).catch((err) => setError(err.message));
  }, []);
  React.useEffect(() => { if (open) { setTopics(null); setError(null); load(); } }, [open, load]);

  // Topic-over-time: volume per week across EVERY saved analysis
  // (occurrences fingerprinted server-side, so overlapping uploads never
  // double the curve)
  const [historyFor, setHistoryFor] = React.useState(null); // topic id
  const [history, setHistory] = React.useState(null);
  const fetchHistory = (topicId) => {
    window.QA_API.topicHistory(topicId)
      .then(setHistory)
      .catch((err) => setHistory({ error: err.message }));
  };

  const act = async (fn) => {
    try {
      await fn();
      mutated.current = true;
      load();
      // An open history panel must reflect the change too (e.g. the
      // publish marker appearing/disappearing right after the toggle)
      if (historyFor) fetchHistory(historyFor);
    } catch (err) { setError(err.message); }
  };
  const rename = (t) => {
    const name = window.prompt('Rename this topic:', t.topic);
    if (!name || !name.trim() || name.trim() === t.topic) return;
    const clean = name.trim();
    act(async () => {
      await window.QA_API.renameTopic(t.id, clean);
      // Patch the loaded analysis in place (same as the dashboard's
      // inline pencil) so the open views show the new name immediately
      const results = window.ANALYSIS_RESULTS;
      if (results && results.groups) {
        results.groups.forEach((g) => { if (g.topic_id === t.id) g.topic = clean; });
      }
    });
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

  const showHistory = (t) => {
    if (historyFor === t.id) { setHistoryFor(null); setHistory(null); return; }
    setHistoryFor(t.id);
    setHistory(undefined); // loading
    fetchHistory(t.id);
  };

  // Curated answer editor: approved wording saved on the bank entry —
  // every FAQ export then uses it instead of re-drafting from replies
  const [answerFor, setAnswerFor] = React.useState(null); // topic id
  const [answerText, setAnswerText] = React.useState('');
  const editAnswer = (t) => {
    if (answerFor === t.id) { setAnswerFor(null); return; }
    setAnswerFor(t.id);
    setAnswerText(t.curated_answer || '');
  };
  const saveAnswer = (t, text) => {
    act(() => window.QA_API.setTopicAnswer(t.id, text));
    setAnswerFor(null);
  };

  // One plain-language line answering "did my FAQ work?"
  const effectLine = (effect) => {
    if (!effect) return null;
    const fmt = () => `${effect.before_per_week}/wk before to ${effect.after_per_week}/wk after`;
    if (effect.verdict === 'working') {
      return { color: 'var(--green-60, #198038)', text:
        `FAQ working: asks fell ${Math.abs(effect.change_pct)}% since publish (${fmt()})` };
    }
    if (effect.verdict === 'helping') {
      return { color: 'var(--teal-60)', text:
        `FAQ helping: asks down ${Math.abs(effect.change_pct)}% since publish (${fmt()})` };
    }
    if (effect.verdict === 'not_working') {
      return { color: 'var(--tag-err-fg)', text:
        `FAQ not moving the needle: ${fmt()}. The doc may be hard to find, or the answer incomplete.` };
    }
    if (effect.verdict === 'too_early') {
      return { color: 'var(--text-helper)', text:
        `Only ${effect.days_of_data_after} day(s) of data since the FAQ was published. Too early to tell if it is working.` };
    }
    return { color: 'var(--text-helper)', text:
      'Not enough pre-publish data to judge whether the FAQ is working.' };
  };

  return (
    <Modal open={open} onClose={close} width={620}>
      <ModalHead title="Learned topics" sub="Everything the analyzer knows from seeds and past analyses. Rename bad names, merge duplicates, delete junk — changes apply to all future analyses." onClose={close} />
      <div style={{ padding: '0 24px 24px' }}>
        {error ? <div style={{ fontSize: 13, color: 'var(--red-60)', padding: '12px 16px', background: 'var(--field)', borderLeft: '3px solid var(--red-60)', marginBottom: 14 }}>{error}</div> : null}
        {topics === null && !error ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-secondary)', fontSize: 13.5, padding: '18px 0' }}><Icon name="loader" size={16} /> Loading…</div>
        ) : null}
        {topics && topics.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--text-helper)', fontSize: 13.5, padding: '26px 0 18px' }}>No learned topics yet — run an analysis first.</div>
        ) : null}
        {topics && topics.length > 0 ? (
          <div style={{ border: '1px solid var(--border-subtle)', borderBottom: 'none', maxHeight: 400, overflowY: 'auto' }}>
            {topics.map((t) => (
              <div key={t.id} style={{ borderBottom: '1px solid var(--border-subtle)', background: 'var(--layer-02)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px' }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13.5, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.topic || '(unnamed)'}</div>
                    <div style={{ fontSize: 11.5, color: 'var(--text-helper)', fontFamily: 'var(--font-mono)' }}>
                      {t.question_count || 0} questions · {t.analysis_count || 0} analyses{t.last_seen ? ` · last ${t.last_seen}` : ''}
                    </div>
                  </div>
                  {t.curated_answer ? (
                    <span title={`Curated answer saved${t.answer_updated ? ` ${t.answer_updated}` : ''} — FAQ exports use this wording`}
                      style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--link)', background: 'var(--chip-active-bg)', padding: '1px 7px', whiteSpace: 'nowrap' }}>
                      answer saved
                    </span>
                  ) : null}
                  {t.faq_published ? (
                    <span title={`FAQ published ${t.faq_published} — click the chart to see whether asks fell`}
                      style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--tag-ok-fg)', background: 'var(--tag-ok-bg)', padding: '1px 7px', whiteSpace: 'nowrap' }}>
                      FAQ live
                    </span>
                  ) : null}
                  {iconBtn(t.faq_published
                    ? `Unmark as published (was ${t.faq_published})`
                    : 'Mark FAQ as published: stamps today as a marker on the history chart, so you can watch ask-volume fall after your doc went live',
                    'check', () => act(() => window.QA_API.setTopicPublished(t.id, !t.faq_published)),
                    'var(--green-60, #198038)')}
                  {iconBtn('Volume over time (across all analyses)', 'chart-line', () => showHistory(t), 'var(--teal-60)')}
                  {iconBtn(t.curated_answer
                    ? `Edit the curated answer (saved ${t.answer_updated || ''})`
                    : 'Save a curated answer: approved wording that every FAQ export uses instead of a fresh draft',
                    'file-text', () => editAnswer(t), 'var(--blue-60)')}
                  {iconBtn('Rename', 'pencil', () => rename(t), 'var(--blue-60)')}
                  {iconBtn('Merge into another topic', 'git-merge', () => merge(t), 'var(--purple-60)')}
                  {iconBtn('Delete', 'trash-2', () => remove(t), 'var(--red-60)')}
                </div>
                {answerFor === t.id ? (
                  <div style={{ padding: '0 14px 12px' }}>
                    <div style={{ fontSize: 11.5, color: 'var(--text-helper)', marginBottom: 6 }}>
                      Curated answer — the FAQ export uses this wording instead of drafting one. Leave empty and save to remove it.
                    </div>
                    <textarea value={answerText} onChange={(e) => setAnswerText(e.target.value)}
                      rows={4} placeholder="Paste or write the approved answer for this topic"
                      style={{ width: '100%', boxSizing: 'border-box', resize: 'vertical', fontFamily: 'var(--font-sans)', fontSize: 13, lineHeight: 1.5, color: 'var(--text-primary)', background: 'var(--field)', border: '1px solid var(--border-subtle)', padding: '8px 10px' }} />
                    <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 8 }}>
                      <Button variant="ghost" onClick={() => setAnswerFor(null)}>Cancel</Button>
                      <Button variant="primary" onClick={() => saveAnswer(t, answerText.trim())}>
                        {answerText.trim() ? 'Save answer' : (t.curated_answer ? 'Remove answer' : 'Save answer')}
                      </Button>
                    </div>
                  </div>
                ) : null}
                {historyFor === t.id ? (
                  <div style={{ padding: '0 14px 12px' }}>
                    {history === undefined ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-helper)', fontSize: 12.5 }}><Icon name="loader" size={13} /> Loading history…</div>
                    ) : history && history.error ? (
                      <div style={{ color: 'var(--text-helper)', fontSize: 12.5 }}>{history.error}</div>
                    ) : history && history.series && history.series.length ? (
                      <React.Fragment>
                        <div style={{ fontSize: 11.5, color: 'var(--text-helper)', fontFamily: 'var(--font-mono)', marginBottom: 4 }}>
                          {history.total_occurrences} unique occurrence(s) across {history.analyses_with_topic} analysis(es) · {history.first_asked} → {history.last_asked}{history.undated ? ` · ${history.undated} undated` : ''}
                        </div>
                        {(() => {
                          const line = effectLine(history.faq_effect);
                          return line ? (
                            <div style={{ fontSize: 12, fontWeight: 500, color: line.color, marginBottom: 6 }}>{line.text}</div>
                          ) : null;
                        })()}
                        {history.answer_stale ? (
                          <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--banner-warn-fg)', marginBottom: 6 }}>
                            {history.answer_stale.new_answered} newly answered ask(s) since your curated answer was saved ({history.answer_stale.answer_updated}). Worth a re-read for updates.
                          </div>
                        ) : null}
                        <AreaChart data={history.series.map((p) => p.count)}
                          labels={history.series.map((p) => p.week.slice(5))}
                          width={540} height={120}
                          marker={(() => {
                            if (!history.faq_published) return null;
                            const idx = history.series.findIndex(
                              (p) => p.week >= history.faq_published);
                            return idx === -1 ? null
                              : { index: idx, label: 'FAQ published' };
                          })()} />
                      </React.Fragment>
                    ) : (
                      <div style={{ color: 'var(--text-helper)', fontSize: 12.5 }}>No dated occurrences recorded for this topic yet.</div>
                    )}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        ) : null}
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
          <Button variant="ghost" onClick={close}>Close</Button>
        </div>
      </div>
    </Modal>
  );
}

// ---- Analysis settings (similarity threshold) ----
function SettingsModal({ open, onClose }) {
  const { Button, Slider } = window.QuestionAnalyzerDesignSystem_03a921;
  const [settings, setSettings] = React.useState(window.QA_SETTINGS.get());

  React.useEffect(() => {
    if (open) window.QA_SETTINGS.loadServerDefaults().then(setSettings);
  }, [open]);

  const save = () => { window.QA_SETTINGS.set(settings); onClose(); };

  return (
    <Modal open={open} onClose={onClose} width={480}>
      <ModalHead title="Analysis settings" sub="Applied to every new transcript analysis. Everything runs on your machine via local Ollama — no data ever leaves it." onClose={onClose} />
      <div style={{ padding: '0 24px 24px' }}>
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
