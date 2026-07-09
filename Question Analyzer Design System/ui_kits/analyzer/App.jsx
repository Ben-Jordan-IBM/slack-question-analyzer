// Question Analyzer — consolidated app.
// Lightweight toast host: every mutation (rename, merge, publish, answer
// save, settings) confirms itself instead of succeeding silently.
function ToastHost() {
  const [toasts, setToasts] = React.useState([]);
  React.useEffect(() => {
    window.QA_TOAST = (message) => {
      const id = Date.now() + Math.random();
      setToasts((t) => [...t, { id, message }]);
      setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2600);
    };
    return () => { delete window.QA_TOAST; };
  }, []);
  if (!toasts.length) return null;
  return (
    <div style={{ position: 'fixed', bottom: 22, left: '50%', transform: 'translateX(-50%)', zIndex: 300, display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center', pointerEvents: 'none' }}>
      {toasts.map((t) => (
        <div key={t.id} className="qa-toast" style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 18px', fontFamily: 'var(--font-sans)', fontSize: 13, background: 'var(--text-primary)', color: 'var(--background)', boxShadow: 'var(--shadow-md, 0 2px 6px rgba(0,0,0,.3))' }}>
          <Icon name="check" size={14} /> {t.message}
        </div>
      ))}
    </div>
  );
}

function App() {
  const [view, setView] = React.useState('dashboard');
  const [analysisVersion, setAnalysisVersion] = React.useState(0);
  // Dark mode: one data attribute drives the CSS-variable overrides;
  // light stays the default and the choice persists locally
  const [theme, setTheme] = React.useState(
    () => localStorage.getItem('qa-theme') || 'light');
  React.useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('qa-theme', theme);
  }, [theme]);
  // Week -> Dashboard click-through: land with the topic pre-filtered
  const [dashQuery, setDashQuery] = React.useState('');
  const inspectTopic = (q) => {
    setDashQuery(q || '');
    setView('dashboard');
    setAnalysisVersion((v) => v + 1); // remount so the seeded query applies
  };
  const [uploadOpen, setUploadOpen] = React.useState(false);
  const [historyOpen, setHistoryOpen] = React.useState(false);
  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const [topicsOpen, setTopicsOpen] = React.useState(false);

  // Backend version shown in the header: a one-glance check that the browser
  // is talking to the build you think it is
  const [backendVersion, setBackendVersion] = React.useState(null);
  React.useEffect(() => {
    if (!window.QA_API) return;
    window.QA_API.getConfig()
      .then((c) => { setBackendVersion(c.version || null); window.QA_BACKEND_VERSION = c.version || null; })
      .catch(() => setBackendVersion(null));
  }, []);

  const showAnalysis = (data) => {
    window.ANALYSIS_RESULTS = data;
    setHistoryOpen(false);
    setView('dashboard');
    setAnalysisVersion((v) => v + 1);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--background)' }}>
      <AppHeader view={view} setView={(v) => { setDashQuery(''); setView(v); }} onUpload={() => setUploadOpen(true)}
        onHistory={() => setHistoryOpen(true)} onTopics={() => setTopicsOpen(true)}
        onSettings={() => setSettingsOpen(true)} version={backendVersion}
        theme={theme} onToggleTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')} />
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', background: 'var(--background)' }}>
        <div key={`${view}:${analysisVersion}`} className="qa-view">
          {view === 'dashboard'
            ? <DashboardView onUpload={() => setUploadOpen(true)} initialQuery={dashQuery}
                onTopics={() => setTopicsOpen(true)} />
            : <WeekView onInspect={inspectTopic} />}
        </div>
      </div>
      <UploadModal open={uploadOpen} onClose={() => setUploadOpen(false)}
        onImported={() => { setUploadOpen(false); setView('dashboard'); setAnalysisVersion((v) => v + 1); }} />
      <HistoryModal open={historyOpen} onClose={() => setHistoryOpen(false)} onLoad={showAnalysis} />
      <TopicsModal open={topicsOpen} onClose={() => setTopicsOpen(false)}
        onMutated={() => setAnalysisVersion((v) => v + 1)} />
      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <ToastHost />
    </div>
  );
}
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
