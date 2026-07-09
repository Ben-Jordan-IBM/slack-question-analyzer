// Overall dashboard — common questions, ranked by occurrences (all-time).
// Entirely backend-driven: real saved analyses or an empty state, never mocks.
function DashboardView({ onUpload, initialQuery, onTopics }) {
  // Use real analysis results when available; on first load, pull the most
  // recent saved analysis from the backend so results survive page refreshes.
  const [analysisResults, setAnalysisResults] = React.useState(window.ANALYSIS_RESULTS || null);
  const [loadingLatest, setLoadingLatest] = React.useState(!window.ANALYSIS_RESULTS);
  React.useEffect(() => {
    if (analysisResults || !window.QA_API) { setLoadingLatest(false); return; }
    let cancelled = false;
    window.QA_API.latestAnalysis().then((latest) => {
      if (cancelled) return;
      if (latest) {
        window.ANALYSIS_RESULTS = latest;
        setAnalysisResults(latest);
      }
      setLoadingLatest(false);
    }).catch(() => { if (!cancelled) setLoadingLatest(false); });
    return () => { cancelled = true; };
  }, []);

  const PAGE_SIZE = 50;
  // First-run onboarding: surface backend/model setup problems before the
  // user wastes an upload finding out. Missing Ollama models get a one-click
  // download button instead of terminal instructions.
  const [setupHealth, setSetupHealth] = React.useState(null); // latest health payload
  const [pulls, setPulls] = React.useState({}); // model -> {pct, detail, error}
  const checkHealth = React.useCallback(() => {
    if (!window.QA_API) return;
    window.QA_API.health()
      .then(setSetupHealth)
      .catch(() => setSetupHealth({ status: 'unavailable',
        message: 'Cannot reach the analyzer backend. Start it with: python api_server.py' }));
  }, []);
  React.useEffect(() => { checkHealth(); }, [checkHealth]);

  // The dashboard remounts on every analysis change (keyed in App) — poll
  // timers must die with the component or they poll a dead one forever
  const pullTimers = React.useRef({});
  React.useEffect(() => () => {
    Object.values(pullTimers.current).forEach(clearInterval);
  }, []);
  const downloadModel = async (model) => {
    try {
      setPulls((p) => ({ ...p, [model]: { pct: 0, detail: 'starting…' } }));
      await window.QA_API.pullModel(model);
      const timer = setInterval(async () => {
        try {
          const s = await window.QA_API.pullStatus(model);
          if (s.status === 'done') { clearInterval(timer); delete pullTimers.current[model]; setPulls((p) => ({ ...p, [model]: null })); checkHealth(); }
          else if (s.status === 'error') { clearInterval(timer); delete pullTimers.current[model]; setPulls((p) => ({ ...p, [model]: { error: s.detail } })); }
          else setPulls((p) => ({ ...p, [model]: { pct: s.total ? Math.round(s.completed / s.total * 100) : 0, detail: s.detail } }));
        } catch (err) { /* keep polling */ }
      }, 1000);
      pullTimers.current[model] = timer;
    } catch (err) {
      setPulls((p) => ({ ...p, [model]: { error: err.message } }));
    }
  };

  // Renaming updates the learned bank (future analyses) and the current view
  const renameTopic = async (topicId, newName) => {
    try {
      await window.QA_API.renameTopic(topicId, newName);
    } catch (err) {
      alert('Rename failed: ' + err.message);
      return;
    }
    if (window.ANALYSIS_RESULTS && window.ANALYSIS_RESULTS.groups) {
      window.ANALYSIS_RESULTS = {
        ...window.ANALYSIS_RESULTS,
        groups: window.ANALYSIS_RESULTS.groups.map((g) =>
          g.topic_id === topicId ? { ...g, topic: newName } : g),
      };
      setAnalysisResults(window.ANALYSIS_RESULTS);
    }
  };

  // Models the banner should offer to download (Ollama reachable but missing).
  // A missing LABEL model doesn't fail health (analysis still works), but it
  // silently downgrades topic names to keywords — so surface it here too.
  const missingModels = [];
  if (setupHealth && setupHealth.ollama && setupHealth.ollama.reachable) {
    if (!setupHealth.ollama.model_available) missingModels.push(setupHealth.ollama.model);
    if (!setupHealth.ollama.label_model_available && setupHealth.ollama.label_model) {
      missingModels.push(setupHealth.ollama.label_model);
    }
  }
  const healthBroken = setupHealth && setupHealth.status !== 'ok';
  // The full red banner belongs on an EMPTY dashboard (you can't analyze
  // until it's fixed). While browsing saved results it demotes to a quiet
  // one-line notice — Ollama being asleep is no reason to alarm someone
  // reading last week's report.
  const hasResults = !!(analysisResults
    && ((analysisResults.groups && analysisResults.groups.length)
        || (analysisResults.ungrouped_questions
            && analysisResults.ungrouped_questions.length)));
  const showSetupBanner = (healthBroken || missingModels.length > 0)
    && !hasResults;
  const showSetupHint = (healthBroken || missingModels.length > 0)
    && hasResults;

  // Memoized: re-transforming every group on every keystroke is wasted work
  const d = React.useMemo(
    () => analysisResults ? transformAnalysisResults(analysisResults) : null,
    [analysisResults]);
  const [query, setQuery] = React.useState(initialQuery || '');
  // Presentation-only controls: sorting and date scoping re-read fields the
  // pipeline already produced (per-question dates); no counts are invented.
  const [sortMode, setSortMode] = React.useState('count'); // count | trending | newest
  const [rangeDays, setRangeDays] = React.useState(null);  // null = all time
  const [themeFilter, setThemeFilter] = React.useState(null);
  const [openUnique, setOpenUnique] = React.useState(null); // expanded original message
  const [visibleCount, setVisibleCount] = React.useState(PAGE_SIZE);
  const [showUnique, setShowUnique] = React.useState(true);
  const [showUnanswered, setShowUnanswered] = React.useState(true);
  const [showDropped, setShowDropped] = React.useState(false);
  const [showFeedback, setShowFeedback] = React.useState(true);
  const [notesOpen, setNotesOpen] = React.useState(false);
  const [openDropped, setOpenDropped] = React.useState(null); // expanded source message

  const exportBtn = {
    height: 32, padding: '0 12px', display: 'inline-flex', alignItems: 'center', gap: 6,
    background: 'var(--layer-02)', color: 'var(--text-secondary)', border: '1px solid var(--border-strong)',
    fontFamily: 'var(--font-sans)', fontSize: 12.5, cursor: 'pointer', whiteSpace: 'nowrap',
  };
  const download = (format) => {
    if (window.ANALYSIS_ID) window.open(window.QA_API.exportUrl(window.ANALYSIS_ID, format), '_blank');
  };
  const parseTs = (s) => {
    if (!s || s === 'Unknown') return null;
    const t = Date.parse(s);
    return isNaN(t) ? null : t;
  };
  const lastTs = (g) => {
    let m = null;
    (g.questions || []).forEach((q) => {
      const t = parseTs(q.date);
      if (t && (!m || t > m)) m = t;
    });
    return m;
  };
  // Windows anchor to the NEWEST question date in the corpus, not "today" —
  // a historical transcript must not scope to empty
  let anchorTs = null;
  if (d) {
    (d.groups || []).forEach((g) => { const t = lastTs(g); if (t && (!anchorTs || t > anchorTs)) anchorTs = t; });
    (d.ungrouped || []).forEach((q) => { const t = parseTs(q.date); if (t && (!anchorTs || t > anchorTs)) anchorTs = t; });
  }
  const DAY = 86400000;
  const cutoffTs = rangeDays && anchorTs ? anchorTs - rangeDays * DAY : null;
  const inRange = (q) => !cutoffTs || ((parseTs(q.date) || anchorTs) >= cutoffTs);

  const scopedGroups = (d && d.groups ? d.groups : []).map((g) => {
    if (!cutoffTs) return g;
    const qs = (g.questions || []).filter(inRange);
    return { ...g, questions: qs, count: qs.length,
             answered: qs.filter((q) => q.answered === true).length };
  }).filter((g) => g.count > 0);

  // Trending = occurrences in the newest 30 days of data. Undated questions
  // count as the anchor date — the same fallback inRange uses, so a group of
  // undated questions doesn't show a full count yet sort dead-last
  const trendWindow = anchorTs ? anchorTs - 30 * DAY : null;
  const trendScore = (g) => trendWindow == null ? g.count
    : (g.questions || []).reduce((n, q) => n + (((parseTs(q.date) || anchorTs) >= trendWindow) ? 1 : 0), 0);
  const sorters = {
    count: (a, b) => b.count - a.count || (lastTs(b) || 0) - (lastTs(a) || 0),
    newest: (a, b) => (lastTs(b) || 0) - (lastTs(a) || 0) || b.count - a.count,
    trending: (a, b) => trendScore(b) - trendScore(a) || b.count - a.count
      || (lastTs(b) || 0) - (lastTs(a) || 0),
  };

  const groups = scopedGroups.filter((g) =>
    (!themeFilter || g.theme === themeFilter) &&
    (g.question.toLowerCase().includes(query.toLowerCase()) ||
    g.keywords.join(' ').toLowerCase().includes(query.toLowerCase()) ||
    (g.topic && g.topic.toLowerCase().includes(query.toLowerCase()))))
    .sort(sorters[sortMode] || sorters.count)
    .map((g, i) => ({ ...g, rank: i + 1 }));
  const max = groups.length > 0 ? Math.max(...groups.map((g) => g.count)) : 0;
  const uniqueQuestions = ((d && d.ungrouped) || []).filter(inRange).filter((q) =>
    (!themeFilter || q.theme === themeFilter) &&
    q.text.toLowerCase().includes(query.toLowerCase()));

  // Uniques grouped by theme (funnel sections), newest first within each;
  // themes ordered by the strip's counts, un-themed and review items last
  const parseDate = (s) => { const t = Date.parse(s); return isNaN(t) ? 0 : t; };
  const themeOrder = (d && d.themes ? d.themes.map((t) => t.name) : []);
  const uniqueSections = [];
  {
    const byTheme = new Map();
    uniqueQuestions.forEach((q) => {
      const key = q.needs_review ? 'Needs review' : (q.theme || 'Other');
      if (!byTheme.has(key)) byTheme.set(key, []);
      byTheme.get(key).push(q);
    });
    const keys = [...byTheme.keys()].sort((a, b) => {
      const rank = (k) => k === 'Needs review' ? 1e9 : (k === 'Other' ? 1e8
        : (themeOrder.indexOf(k) === -1 ? 1e7 : themeOrder.indexOf(k)));
      return rank(a) - rank(b);
    });
    keys.forEach((key) => uniqueSections.push({
      theme: key,
      questions: byTheme.get(key).sort((a, b) => parseDate(b.date) - parseDate(a.date)),
    }));
  }

  const stat = (label, value, accent) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <span style={{ fontSize: 11, color: 'var(--text-helper)', fontWeight: 500 }}>{label}</span>
      <span style={{ fontSize: 34, fontWeight: 300, letterSpacing: '-.02em', lineHeight: 1, color: accent || 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
        <CountUp to={value} />
      </span>
    </div>
  );

  const search = {
    display: 'flex', alignItems: 'center', gap: 8, height: 40, padding: '0 14px',
    background: 'var(--field)', borderBottom: '1px solid var(--border-strong)', width: 300, maxWidth: '40vw',
  };

  const setupHint = showSetupHint ? (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 14px', background: 'var(--tag-warn-bg)', border: '1px solid var(--banner-warn-border)', fontSize: 12.5, color: 'var(--tag-warn-fg)', marginBottom: 20 }}>
      <Icon name="alert-triangle" size={13} />
      <span>{healthBroken
        ? 'Ollama is not reachable right now — saved results work fine; start Ollama before analyzing a new transcript.'
        : 'A chat model is missing — new analyses will use keyword labels until it is downloaded (see Settings).'}</span>
    </div>
  ) : null;

  const setupBanner = showSetupBanner ? (
    <Reveal delay={100}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', padding: '16px 24px', background: 'var(--layer-02)', border: '1px solid var(--border-subtle)', borderLeft: `3px solid ${healthBroken ? 'var(--red-60)' : 'var(--blue-60)'}`, marginBottom: 32 }}>
        <span style={{ color: healthBroken ? 'var(--red-60)' : 'var(--blue-60)', flex: '0 0 auto', marginTop: 1 }}><Icon name={healthBroken ? 'alert-triangle' : 'download'} size={16} /></span>
        <div style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.55, minWidth: 0, flex: 1 }}>
          <b>{healthBroken ? 'Setup needed before analyzing.' : 'Topic labels are running in keyword-fallback mode.'}</b>
          {missingModels.length === 0 ? <span> {setupHealth.message}</span> : null}
          {missingModels.map((model) => (
            <div key={model} style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 10 }}>
              {pulls[model] && pulls[model].error ? (
                <span style={{ color: 'var(--red-60)' }}>Download of {model} failed: {pulls[model].error}</span>
              ) : pulls[model] ? (
                <React.Fragment>
                  <span style={{ whiteSpace: 'nowrap' }}>Downloading <b>{model}</b>… {pulls[model].pct}%</span>
                  <div style={{ flex: 1, height: 4, background: 'var(--layer-hover)', overflow: 'hidden', minWidth: 80 }}>
                    <div style={{ height: '100%', width: `${pulls[model].pct}%`, background: 'var(--blue-60)', transition: 'width 600ms linear' }} />
                  </div>
                </React.Fragment>
              ) : (
                <React.Fragment>
                  <span>Model <b>{model}</b> isn't downloaded yet.</span>
                  <button onClick={() => downloadModel(model)}
                    style={{ height: 28, padding: '0 12px', display: 'inline-flex', alignItems: 'center', gap: 6, background: 'var(--blue-60)', color: '#fff', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 12.5, whiteSpace: 'nowrap' }}>
                    <Icon name="download" size={13} /> Download now
                  </button>
                </React.Fragment>
              )}
            </div>
          ))}
        </div>
      </div>
    </Reveal>
  ) : null;

  if (loadingLatest) {
    return (
      <div className="qa-page" style={{ maxWidth: 1000, margin: '0 auto', padding: '60px 40px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, color: 'var(--text-secondary)', fontSize: 14 }}>
        <Icon name="loader" size={16} /> Loading…
      </div>
    );
  }

  if (!d) {
    // Backend has no saved analyses yet: honest empty state, no mock data
    return (
      <div className="qa-page" style={{ maxWidth: 1000, margin: '0 auto', padding: '44px 40px 80px', width: '100%' }}>
        {setupBanner}
        <Reveal>
          <div style={{ textAlign: 'center', padding: '70px 24px', border: '1px dashed var(--border-strong)', background: 'var(--layer-02)' }}>
            <span style={{ display: 'inline-flex', width: 56, height: 56, borderRadius: '50%', background: 'var(--field)', alignItems: 'center', justifyContent: 'center', marginBottom: 18, color: 'var(--blue-60)' }}>
              <Icon name="upload" size={24} />
            </span>
            <h1 style={{ fontSize: 26, fontWeight: 300, letterSpacing: '-.02em', margin: '0 0 8px' }}>No analyses yet</h1>
            <div style={{ fontSize: 14, color: 'var(--text-secondary)', maxWidth: 440, margin: '0 auto 24px', lineHeight: 1.55 }}>
              Paste a Slack thread straight from your channel, or upload an export
              (TXT, JSON, CSV, or a zipped export). Questions are extracted,
              grouped by topic, and ranked right here.
            </div>
            <button onClick={onUpload}
              style={{ height: 40, padding: '0 22px', display: 'inline-flex', alignItems: 'center', gap: 8, background: 'var(--blue-60)', color: '#fff', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 14 }}>
              <Icon name="plus" size={16} /> Add a transcript
            </button>
          </div>
        </Reveal>
      </div>
    );
  }

  return (
    <div className="qa-page" style={{ maxWidth: 1000, margin: '0 auto', padding: '44px 40px 80px', width: '100%' }}>
      <Reveal>
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: 24, flexWrap: 'wrap', marginBottom: 30 }}>
          <div>
            <h1 style={{ fontSize: 32, fontWeight: 300, letterSpacing: '-.02em', margin: '0 0 6px' }}>Most-asked questions</h1>
            <div style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
              {rangeDays ? `Last ${rangeDays} days of your monitored Slack channels` : 'All-time across your monitored Slack channels'} · ranked by {sortMode === 'count' ? 'occurrences' : sortMode === 'trending' ? 'recent activity' : 'newest first'}
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            {window.ANALYSIS_ID ? (
              <React.Fragment>
                <button style={exportBtn} title="Download Markdown report" onClick={() => download('md')}>
                  <Icon name="download" size={14} /> Report
                </button>
                <button style={exportBtn} title="Draft FAQ: top topics with real phrasings and the thread replies that answered them, ready to edit into documentation" onClick={() => download('faq')}>
                  <Icon name="file-text" size={14} /> Draft FAQ
                </button>
                <button style={exportBtn} title="Download CSV" onClick={() => download('csv')}>
                  <Icon name="download" size={14} /> CSV
                </button>
              </React.Fragment>
            ) : null}
            <div className="qa-search" style={search}>
              <Icon name="search" size={16} color="var(--text-helper)" />
              <input value={query} onChange={(e) => { setQuery(e.target.value); setVisibleCount(PAGE_SIZE); }} placeholder="Filter questions or topics"
                style={{ border: 'none', outline: 'none', background: 'transparent', fontFamily: 'var(--font-sans)', fontSize: 14, width: '100%', color: 'var(--text-primary)' }} />
            </div>
          </div>
        </div>
      </Reveal>

      <Reveal delay={90}>
        <div className="qa-stat-row" style={{ display: 'flex', flexWrap: 'wrap', gap: 56, padding: '22px 24px', background: 'var(--field)', borderLeft: '3px solid var(--blue-60)', marginBottom: d.executiveSummary ? 16 : 32 }}>
          {/* The tiles stay all-time (they summarize the whole analysis);
              when a date scope is active, say so instead of looking wrong
              next to the scoped list below */}
          {stat(rangeDays ? 'Questions logged · all time' : 'Questions logged', d.totalQuestions)}
          {stat(rangeDays ? 'Recurring topics · all time' : 'Recurring topics', d.totalGroups, 'var(--purple-60)')}
          {d.threadsPresent ? stat(rangeDays ? 'Answered · all time' : 'Answered', d.resolved, 'var(--teal-60)') : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}
              title="The export contains no thread replies, so answer status can't be measured — re-export with threads to enable this metric">
              <span style={{ fontSize: 11, color: 'var(--text-helper)', fontWeight: 500 }}>Answered</span>
              <span style={{ fontSize: 34, fontWeight: 300, lineHeight: 1, color: 'var(--text-placeholder)' }}>—</span>
            </div>
          )}
          {d.featureRequests && d.featureRequests.length ? stat('Product feedback', d.featureRequests.length, 'var(--purple-60, #8a3ffc)') : null}
          <div className="qa-topstat" style={{ display: 'flex', flexDirection: 'column', gap: 6, justifyContent: 'center', marginLeft: 'auto', textAlign: 'right' }}>
            <span style={{ fontSize: 11, color: 'var(--text-helper)', fontWeight: 500 }}>{d.topTheme ? 'Top theme' : 'Top recurring topic'}</span>
            {d.topTheme ? (
              <button onClick={() => { setThemeFilter(themeFilter === d.topTheme ? null : d.topTheme); setVisibleCount(PAGE_SIZE); }}
                title={themeFilter === d.topTheme ? 'Click to clear this filter' : 'Click to filter the list to this theme'}
                style={{ fontSize: 18, fontWeight: 600, color: 'var(--link)', background: 'transparent', border: 'none', padding: 0, cursor: 'pointer', fontFamily: 'var(--font-sans)', textAlign: 'inherit', textDecoration: themeFilter === d.topTheme ? 'underline' : 'none' }}>
                {d.topTheme}
              </button>
            ) : (
              <span style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>{d.topTopic}</span>
            )}
          </div>
        </div>
      </Reveal>

      {d.themes && d.themes.length ? (
        <Reveal delay={100}>
          <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, marginBottom: d.executiveSummary ? 16 : 32 }}>
            <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '.04em', textTransform: 'uppercase', color: 'var(--text-helper)', marginRight: 4 }}>Themes{rangeDays ? ' · all time' : ''}</span>
            {d.themes.map((t, i) => {
              const active = themeFilter === t.name;
              return (
                <button key={i} className="qa-chip"
                  title={active ? 'Click to clear this filter'
                    : `Show only the ${t.count} question${t.count === 1 ? '' : 's'} under this theme`}
                  onClick={() => { setThemeFilter(active ? null : t.name); setVisibleCount(PAGE_SIZE); }}
                  style={{ fontFamily: 'var(--font-mono)', fontSize: 12.5, cursor: 'pointer',
                    border: active ? '1px solid var(--blue-60)' : '1px solid var(--border-subtle)',
                    color: active ? 'var(--link)' : 'var(--text-secondary)',
                    background: active ? 'var(--chip-active-bg)' : 'var(--field)', padding: '6px 12px' }}>
                  {t.name} <b style={{ color: 'var(--link)' }}>{t.count}</b>
                </button>
              );
            })}
            {themeFilter ? (
              <button onClick={() => setThemeFilter(null)}
                style={{ fontFamily: 'var(--font-sans)', fontSize: 12, cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--text-helper)', textDecoration: 'underline' }}>
                clear filter
              </button>
            ) : null}
          </div>
        </Reveal>
      ) : null}

      {setupBanner}
      {setupHint}

      {(() => {
        // Informational notices collapse into ONE quiet line: stacked
        // full-width yellow banners buried the ranked list (the page's
        // whole point) and made routine notes look like problems
        const notes = [];
        if (d.analyzedWith && window.QA_BACKEND_VERSION && d.analyzedWith !== window.QA_BACKEND_VERSION) {
          notes.push({ icon: 'history', body: (
            <span><b>These results were produced by v{d.analyzedWith}</b> — the server is now running v{window.QA_BACKEND_VERSION}. Restarting the server does not re-analyze: <b>re-upload the transcript</b> to apply the latest fixes.</span>) });
        }
        if (d.extractionAlerts) {
          notes.push({ icon: 'filter', body: (
            <span><b>Extraction notes:</b> {d.extractionAlerts.join(' · ')}. Each one is named in the server console — if a question you expected is missing, that is where it went.</span>) });
        }
        if (d.autoAdjusted) {
          notes.push({ icon: 'sliders-horizontal', body: (
            <span>Nothing grouped at the default threshold, so it was automatically relaxed to <b>{Math.round(d.autoAdjusted * 100)}%</b> for this analysis. Pin a value in Settings if you prefer.</span>) });
        }
        if (!notes.length) return null;
        return (
          <Reveal delay={101}>
            <div style={{ border: '1px solid var(--border-subtle)', borderLeft: '3px solid var(--blue-60)', background: 'var(--layer-02)', marginBottom: 32 }}>
              <button onClick={() => setNotesOpen(!notesOpen)} aria-expanded={notesOpen}
                style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%', padding: '12px 24px', background: 'transparent', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)', textAlign: 'left' }}>
                <span style={{ display: 'inline-flex', transform: notesOpen ? 'rotate(90deg)' : 'none', transition: 'transform var(--duration-base) var(--ease-productive)' }}>
                  <Icon name="chevron-right" size={14} />
                </span>
                Analysis notes ({notes.length}) — informational, nothing needs fixing
              </button>
              {notesOpen ? notes.map((n, i) => (
                <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'flex-start', padding: '10px 24px 12px 46px', fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.55, borderTop: '1px solid var(--border-subtle)' }}>
                  <span style={{ color: 'var(--text-helper)', flex: '0 0 auto', marginTop: 1 }}><Icon name={n.icon} size={15} /></span>
                  {n.body}
                </div>
              )) : null}
            </div>
          </Reveal>
        );
      })()}

      {d.thresholdHint ? (
        <Reveal delay={110}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', padding: '16px 24px', background: 'var(--layer-02)', border: '1px solid var(--border-subtle)', borderLeft: '3px solid var(--yellow-30, #f1c21b)', marginBottom: 32 }}>
            <span style={{ color: 'var(--text-secondary)', flex: '0 0 auto', marginTop: 1 }}><Icon name="sliders-horizontal" size={16} /></span>
            <span style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.55 }}>
              No questions grouped at threshold <b>{Math.round(d.thresholdHint.threshold * 100)}%</b> — your most similar pair scored <b>{Math.round(d.thresholdHint.max * 100)}%</b> (similarity scales vary by embedding model).
              Open <b>Settings</b> (gear icon), set the threshold to about <b>{Math.round(d.thresholdHint.suggestion * 100)}%</b>, and re-upload — cached embeddings make the re-run fast.
            </span>
          </div>
        </Reveal>
      ) : null}

      {d.executiveSummary ? (
        <Reveal delay={120}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', padding: '16px 24px', background: 'var(--layer-02)', border: '1px solid var(--border-subtle)', borderLeft: '3px solid var(--purple-60)', marginBottom: 32 }}>
            <span style={{ color: 'var(--purple-60)', flex: '0 0 auto', marginTop: 1 }}><Icon name="sparkles" size={16} /></span>
            <span style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.55 }}>{d.executiveSummary}</span>
          </div>
        </Reveal>
      ) : null}

      <Reveal delay={160}>
        <div className="qa-controls" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ fontSize: 13, color: 'var(--text-helper)', fontWeight: 500, display: 'inline-flex', alignItems: 'center', gap: 6 }}>Ranked · {groups.length} recurring topic{groups.length === 1 ? '' : 's'}
            <span tabIndex={0} role="note" style={{ display: 'inline-flex', color: 'var(--text-helper)', cursor: 'help' }}
              title={`Questions asked 2+ times that merged into a group. Themes (above) count every question; this counts only the repeats.${d.groupingBar ? ` Questions group together when their wording matches at least ${Math.round(d.groupingBar * 100)}%; borderline pairs are AI-checked.` : ''}`}>
              <Icon name="info" size={13} />
            </span>
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 14 }}>
            <span style={{ display: 'inline-flex' }}>
              {[['count', 'Most asked'], ['trending', 'Trending'], ['newest', 'Newest']].map(([k, l]) => (
                <button key={k} onClick={() => { setSortMode(k); setVisibleCount(PAGE_SIZE); }}
                  title={k === 'trending' ? 'Occurrences in the newest 30 days of data' : undefined}
                  style={{ height: 26, padding: '0 10px', fontSize: 12, cursor: 'pointer', fontFamily: 'var(--font-sans)', border: '1px solid var(--border-subtle)', marginLeft: -1, background: sortMode === k ? 'var(--blue-60)' : 'var(--layer-02)', color: sortMode === k ? '#fff' : 'var(--text-secondary)' }}>{l}</button>
              ))}
            </span>
            <span style={{ display: 'inline-flex' }}>
              {[[null, 'All time'], [90, '90 days'], [30, '30 days']].map(([k, l]) => (
                <button key={String(k)} onClick={() => { setRangeDays(k); setVisibleCount(PAGE_SIZE); }}
                  title="Counts the most recent days of your uploaded data, not today's calendar date"
                  style={{ height: 26, padding: '0 10px', fontSize: 12, cursor: 'pointer', fontFamily: 'var(--font-sans)', border: '1px solid var(--border-subtle)', marginLeft: -1, background: rangeDays === k ? 'var(--blue-60)' : 'var(--layer-02)', color: rangeDays === k ? '#fff' : 'var(--text-secondary)' }}>{l}</button>
              ))}
            </span>
          </span>
        </div>
      </Reveal>

      <div style={{ border: '1px solid var(--border-subtle)', borderBottom: 'none', background: 'var(--layer-02)' }}>
        {groups.slice(0, visibleCount).map((g, i) => (
          // Keyed by identity, not rank: rank is the post-sort position, so
          // switching sort modes would hand row #1's expand state to
          // whichever group lands there
          <RankedRow key={g.topicId || g.question} rank={g.rank} index={i} question={g.question} count={g.count}
            maxCount={max} keywords={g.keywords} similarity={g.similarity} questions={g.questions}
            topic={g.topic} summary={g.summary} seenIn={g.seenIn} aiConfirmed={g.aiConfirmed} theme={g.theme}
            answered={g.answered} needsReview={g.needsReview} dateRange={g.dateRange}
            onRenameTopic={g.topicId ? (name) => renameTopic(g.topicId, name) : null}
            onFaqTools={g.topicId && onTopics ? onTopics : null}
            defaultOpen={i === 0 && !query} />
        ))}
        {groups.length === 0 ? (
          <div style={{ padding: 48, textAlign: 'center', color: 'var(--text-helper)', borderBottom: '1px solid var(--border-subtle)' }}>
            {(themeFilter || query)
              ? <React.Fragment>No topics match “{themeFilter || query}”.</React.Fragment>
              : rangeDays
                ? `No recurring topics in the newest ${rangeDays} days of data.`
                : 'No recurring topics in this analysis.'}
          </div>
        ) : null}
      </div>
      {groups.length > visibleCount ? (
        <button onClick={() => setVisibleCount((v) => v + PAGE_SIZE)}
          style={{ display: 'block', width: '100%', padding: '13px 0', marginTop: -1, background: 'var(--layer-02)', border: '1px solid var(--border-subtle)', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 13, color: 'var(--blue-60)' }}>
          Show {Math.min(PAGE_SIZE, groups.length - visibleCount)} more · {groups.length - visibleCount} remaining
        </button>
      ) : null}

      {(() => {
        // The backlog: questions a thread exists for but nobody actually
        // answered. For a support engineer this is the to-do list — the
        // data was always computed, it just had no lane of its own.
        if (!d.threadsPresent) return null;
        // Built from the FILTERED views (date scope, theme chip, search),
        // so the backlog always matches the lists above it
        const unanswered = [];
        groups.forEach((g) => (g.questions || []).forEach((q) => {
          if (q.answered === false) unanswered.push({ q, topic: g.topic });
        }));
        uniqueQuestions.forEach((q) => {
          if (q.answered === false) unanswered.push({ q, topic: null });
        });
        if (!unanswered.length) return null;
        return (
          <div style={{ marginTop: 28 }}>
            <button onClick={() => setShowUnanswered(!showUnanswered)}
              title="Questions with a thread where no reply actually answered them — the follow-up backlog"
              style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'transparent', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 13, fontWeight: 500, color: 'var(--tag-err-fg)', padding: 0 }}>
              <span style={{ display: 'inline-flex', transform: showUnanswered ? 'rotate(90deg)' : 'none', transition: 'transform var(--duration-base) var(--ease-productive)' }}>
                <Icon name="chevron-right" size={14} />
              </span>
              Still unanswered ({unanswered.length}) — threads where nobody gave an answer
            </button>
            {showUnanswered ? (
              <ul style={{ listStyle: 'none', margin: '12px 0 0', padding: 0, borderLeft: '2px solid var(--tag-err-fg)' }}>
                {unanswered.map(({ q, topic }, i) => (
                  <li key={i} style={{ display: 'flex', justifyContent: 'space-between', gap: 16, padding: '8px 16px' }}>
                    <span style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                      {topic ? <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-helper)', background: 'var(--field)', padding: '1px 7px', marginRight: 8, whiteSpace: 'nowrap' }}>{topic}</span> : null}
                      {q.text}
                    </span>
                    {q.date && q.date !== 'Unknown' ? <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-placeholder)', whiteSpace: 'nowrap' }}>{q.date}</span> : null}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        );
      })()}

      {uniqueQuestions.length ? (
        <div style={{ marginTop: 28 }}>
          <button onClick={() => setShowUnique(!showUnique)}
            style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'transparent', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)', padding: 0 }}>
            <span style={{ display: 'inline-flex', transform: showUnique ? 'rotate(90deg)' : 'none', transition: 'transform var(--duration-base) var(--ease-productive)' }}>
              <Icon name="chevron-right" size={14} />
            </span>
            Unique questions ({uniqueQuestions.length}) — asked only once, kept in every export
          </button>
          {showUnique ? uniqueSections.map((section) => (
            <div key={section.theme} style={{ margin: '14px 0 0' }}>
              <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '.04em', textTransform: 'uppercase', padding: '0 16px 6px', color: section.theme === 'Needs review' ? 'var(--banner-warn-fg)' : 'var(--text-helper)' }}
                title={section.theme === 'Needs review' ? "Didn't fit any known category — a recurring pile of these means a new category is being born" : undefined}>
                {section.theme} ({section.questions.length})
              </div>
              <ul style={{ listStyle: 'none', margin: 0, padding: 0, borderLeft: `2px solid ${section.theme === 'Needs review' ? 'var(--banner-warn-border)' : 'var(--border-subtle)'}` }}>
                {section.questions.map((q, i) => {
                  const key = `${section.theme}:${i}`;
                  const open = openUnique === key;
                  const hasOriginal = !!q.original_message;
                  return (
                    <li key={i} onClick={() => hasOriginal && setOpenUnique(open ? null : key)}
                      title={hasOriginal && !open ? 'Click to see the original Slack message' : undefined}
                      style={{ padding: '8px 16px', cursor: hasOriginal ? 'pointer' : 'default' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16 }}>
                        <span style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                          {q.qtype ? <StatusChip title="What kind of question this is (classified during extraction)" style={{ marginRight: 8 }}>{q.qtype}</StatusChip> : null}
                          {q.answered === true ? <StatusChip kind="ok" title="A thread reply answered this question" style={{ marginRight: 8 }}>answered</StatusChip> : null}
                          {q.answered === false ? <StatusChip kind="err" title="This question has a thread but no reply actually answered it" style={{ marginRight: 8 }}>unanswered</StatusChip> : null}
                          {q.text}
                        </span>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, whiteSpace: 'nowrap' }}>
                          {q.date && q.date !== 'Unknown' ? <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-placeholder)' }}>{q.date}</span> : null}
                          {hasOriginal ? <span style={{ display: 'inline-flex', color: 'var(--text-placeholder)', transform: open ? 'rotate(180deg)' : 'none', transition: 'transform var(--duration-base) var(--ease-productive)' }}><Icon name="chevron-down" size={12} /></span> : null}
                        </span>
                      </div>
                      {open ? (
                        <div style={{ margin: '7px 0 2px', padding: '8px 12px', background: 'var(--field)', fontSize: 12.5, color: 'var(--text-helper)', lineHeight: 1.5 }}>
                          <span style={{ fontWeight: 600, letterSpacing: '.03em', textTransform: 'uppercase', fontSize: 10.5, marginRight: 8 }}>Original message</span>
                          {q.original_message}
                        </div>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            </div>
          )) : null}
        </div>
      ) : null}

      {d.featureRequests && d.featureRequests.length ? (
        <div style={{ marginTop: 28 }}>
          <button onClick={() => setShowFeedback(!showFeedback)}
            title="Questions classified as feature requests are product feedback, not support questions a doc page resolves — they're routed out of the support funnel"
            style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'transparent', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)', padding: 0 }}>
            <span style={{ display: 'inline-flex', transform: showFeedback ? 'rotate(90deg)' : 'none', transition: 'transform var(--duration-base) var(--ease-productive)' }}>
              <Icon name="chevron-right" size={14} />
            </span>
            Product feedback ({d.featureRequests.length}) — feature requests, routed out of the support funnel
          </button>
          {showFeedback ? (
          <ul style={{ listStyle: 'none', margin: '12px 0 0', padding: 0, borderLeft: '2px solid var(--purple-60, #8a3ffc)' }}>
            {d.featureRequests.map((q, i) => (
              <li key={i} style={{ display: 'flex', justifyContent: 'space-between', gap: 16, padding: '8px 16px' }}>
                <span style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.45 }}>{q.text}</span>
                {q.date && q.date !== 'Unknown' ? <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-placeholder)', whiteSpace: 'nowrap' }}>{q.date}</span> : null}
              </li>
            ))}
          </ul>
          ) : null}
        </div>
      ) : null}


      {d.droppedQuestions && d.droppedQuestions.length ? (
        <div style={{ marginTop: 28 }}>
          <button onClick={() => setShowDropped(!showDropped)}
            title="Provenance trail: nothing is ever silently consumed — every question any stage removed is recorded here with its reason"
            style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'transparent', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)', padding: 0 }}>
            <span style={{ display: 'inline-flex', transform: showDropped ? 'rotate(90deg)' : 'none', transition: 'transform var(--duration-base) var(--ease-productive)' }}>
              <Icon name="chevron-right" size={14} />
            </span>
            Removed during analysis ({d.droppedQuestions.length}) — duplicates and phantoms, each with its reason
          </button>
          {showDropped ? (
            <ul style={{ listStyle: 'none', margin: '12px 0 0', padding: 0, borderLeft: '2px solid var(--border-subtle)' }}>
              {d.droppedQuestions.map((q, i) => {
                const open = openDropped === i;
                const hasSource = !!q.source;
                return (
                  <li key={i} onClick={() => hasSource && setOpenDropped(open ? null : i)}
                    title={hasSource && !open ? 'Click to see the source Slack message' : undefined}
                    style={{ padding: '8px 16px', cursor: hasSource ? 'pointer' : 'default' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16 }}>
                      <span style={{ fontSize: 13.5, color: 'var(--text-helper)', lineHeight: 1.45 }}>
                        <span title="Why this row was removed" style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--banner-warn-fg)', background: 'var(--banner-warn-bg)', padding: '1px 7px', marginRight: 8, whiteSpace: 'nowrap' }}>{q.reason}</span>
                        <span style={{ textDecoration: 'line-through', textDecorationColor: 'var(--text-placeholder)' }}>{q.text}</span>
                      </span>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, whiteSpace: 'nowrap' }}>
                        {q.date && q.date !== 'Unknown' ? <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-placeholder)' }}>{q.date}</span> : null}
                        {hasSource ? <span style={{ display: 'inline-flex', color: 'var(--text-placeholder)', transform: open ? 'rotate(180deg)' : 'none', transition: 'transform var(--duration-base) var(--ease-productive)' }}><Icon name="chevron-down" size={12} /></span> : null}
                      </span>
                    </div>
                    {open ? (
                      <div style={{ margin: '7px 0 2px', padding: '8px 12px', background: 'var(--field)', fontSize: 12.5, color: 'var(--text-helper)', lineHeight: 1.5 }}>
                        <span style={{ fontWeight: 600, letterSpacing: '.03em', textTransform: 'uppercase', fontSize: 10.5, marginRight: 8 }}>Source message</span>
                        {q.source}
                      </div>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
window.DashboardView = DashboardView;

// Suggest a workable threshold when nothing grouped (mirrors the backend's
// QuestionAnalyzer.suggested_threshold logic).
function thresholdHint(results) {
  const stats = results.metadata && results.metadata.similarity_stats;
  if (!stats || (results.total_groups || 0) > 0) return null;
  const threshold = results.metadata.effective_threshold || results.metadata.similarity_threshold;
  if (stats.max >= threshold) return null;
  const suggestion = Math.round((stats.max - 0.02) * 100) / 100;
  if (suggestion <= 0 || suggestion >= threshold) return null;
  return { threshold, max: stats.max, suggestion };
}

// Transform API results to dashboard format
function transformAnalysisResults(results) {
  if (!results || !results.groups) return null;
  
  const fallbackTopic = (g) => g.representative_question.split(' ').slice(0, 3).join(' ') + '...';
  return {
    totalQuestions: results.total_questions || 0,
    totalGroups: results.total_groups || 0,
    resolved: results.answered_questions || 0, // LLM answer detection (threads only)
    executiveSummary: results.executive_summary || null,
    ungrouped: results.ungrouped_questions || [],
    themes: results.themes || null,
    groupingBar: (results.metadata && results.metadata.effective_threshold) || null,
    thresholdHint: thresholdHint(results),
    autoAdjusted: (results.metadata && results.metadata.threshold_auto_adjusted)
      ? results.metadata.similarity_threshold : null,
    topTopic: results.groups[0] ? (results.groups[0].topic || fallbackTopic(results.groups[0])) : 'N/A',
    // The headline is the largest THEME (most representative number on the
    // page) — a 2x merged group must not be crowned over a 6-question theme
    topTheme: (results.themes && results.themes[0]) ? results.themes[0].name : null,
    featureRequests: results.feature_requests || [],
    // Provenance trail: every question any stage removed, with its reason —
    // auditable on the page instead of buried in the server console
    droppedQuestions: results.dropped_questions || [],
    threadsPresent: !!results.threads_present,
    analyzedWith: (results.metadata && results.metadata.app_version) || null,
    // Extraction anomalies must be LOUD: silent drops were the worst bug class
    extractionAlerts: (() => {
      const s = (results.metadata && results.metadata.llm_stats) || {};
      const alerts = [];
      if (s.messages_without_questions) alerts.push(`${s.messages_without_questions} message(s) produced no questions`);
      if (s.announcements_skipped) alerts.push(`${s.announcements_skipped} announcement(s)/notice(s) treated as context-only`);
      if (s.extract_dropped_unsupported) alerts.push(`${s.extract_dropped_unsupported} extraction(s) dropped (no source contains them)`);
      if (s.date_collisions_dropped) alerts.push(`${s.date_collisions_dropped} date-collision phantom(s) dropped`);
      if (s.extract_reassigned) alerts.push(`${s.extract_reassigned} extraction(s) reassigned to their true source`);
      if (s.same_ask_collapsed) alerts.push(`${s.same_ask_collapsed} same-ask rewrite(s) consolidated`);
      if (s.extract_example_leaks) alerts.push(`${s.extract_example_leaks} AI-contamination leak(s) dropped`);
      if (s.integrity_repairs) alerts.push(`${s.integrity_repairs} group(s) repaired (count couldn't prove its rows)`);
      return alerts.length ? alerts : null;
    })(),
    groups: results.groups.map((g, i) => ({
      rank: i + 1,
      count: g.count,
      similarity: `${Math.round(g.avg_similarity * 100)}%`,
      theme: g.theme || null,
      topic: g.topic || null,
      summary: g.summary || null,
      seenIn: g.seen_in_analyses || 0,
      answered: g.answered || 0,
      dateRange: g.date_range || null,
      needsReview: !!(g.questions && g.questions.some((q) => q.needs_review)),
      topicId: g.topic_id || null,
      // Below the numeric bar means the AI verifier approved this merge
      aiConfirmed: !!(results.metadata && results.metadata.effective_threshold
        && g.avg_similarity < results.metadata.effective_threshold),
      question: g.representative_question,
      keywords: g.keywords || [],
      questions: g.questions || []
    }))
  };
}
