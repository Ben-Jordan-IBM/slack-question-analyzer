// Question Analyzer — consolidated app.
function App() {
  const [view, setView] = React.useState('dashboard');
  const [analysisVersion, setAnalysisVersion] = React.useState(0);
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
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#fff' }}>
      <AppHeader view={view} setView={setView} onUpload={() => setUploadOpen(true)}
        onHistory={() => setHistoryOpen(true)} onTopics={() => setTopicsOpen(true)}
        onSettings={() => setSettingsOpen(true)} version={backendVersion} />
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', background: '#fff' }}>
        <div key={`${view}:${analysisVersion}`} className="qa-view">
          {view === 'dashboard' ? <DashboardView onUpload={() => setUploadOpen(true)} /> : <WeekView />}
        </div>
      </div>
      <UploadModal open={uploadOpen} onClose={() => setUploadOpen(false)}
        onImported={() => { setUploadOpen(false); setView('dashboard'); setAnalysisVersion((v) => v + 1); }} />
      <HistoryModal open={historyOpen} onClose={() => setHistoryOpen(false)} onLoad={showAnalysis} />
      <TopicsModal open={topicsOpen} onClose={() => setTopicsOpen(false)} />
      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
